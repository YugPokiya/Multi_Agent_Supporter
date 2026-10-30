"""Runs the decision graph for one or many instruments, trains models, and backtests decisions."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from app.config import Settings, settings
from app.data.history import DOLLAR_INDEX, INSTRUMENTS, HistoryProvider, make_provider
from app.decision.agents import DIRECTION, HOLD, RiskConfig, Snapshot
from app.decision.graph import VOTERS, build_decision_graph
from app.decision.store import DecisionStore, to_jsonable
from app.ml.features import compute_features
from app.ml.model import ModelBundle, load_bundle, save_bundle, train_model

log = logging.getLogger(__name__)
DISCLAIMER = "Educational decision support only; not financial advice."
RETRAIN_AFTER_DAYS = 7


def dxy_ret_20(dxy: pd.DataFrame | None) -> pd.Series | None:
    if dxy is None or dxy.empty:
        return None
    return np.log(dxy["close"] / dxy["close"].shift(20))


def make_snapshot(instrument: str, df: pd.DataFrame, feats: pd.DataFrame, i: int,dxy20: pd.Series | None = None, orderflow: dict | None = None) -> Snapshot:
    as_of = feats.index[i]
    dxy_val = None
    if dxy20 is not None:
        v = dxy20.asof(as_of)
        dxy_val = None if pd.isna(v) else float(v)
    return Snapshot(instrument=instrument, as_of=as_of, close=float(df["close"].iloc[i]),
                    features=feats.iloc[i], feature_history=feats.iloc[: i + 1],
                    dxy_ret_20=dxy_val, orderflow=orderflow)


def explain(instrument: str, state: dict) -> str:
    agg, ev, ml = state["aggregate"], state["evaluation"], state.get("ml") or {}
    parts = [f"{ev['final_action']} {instrument}."]
    if agg["action"] != HOLD:
        parts.append(f"{len(agg['agreeing_agents'])} of {agg['active_agents']} active agents back {agg['action']} "
                     f"(score {agg['score']:+.2f}, threshold {agg['threshold']:.2f}).")
    else:
        parts.append(f"Weighted score {agg['score']:+.2f} is inside +/-{agg['threshold']:.2f}.")
    parts.append(f"Regime: {state['regime'].replace('_', ' ')}.")
    if ml:
        used = "used" if ml.get("has_edge") else "ignored: no proven edge"
        parts.append(f"ML says {ml['predicted']} p={ml['confidence']:.0%} ({used}).")
    if ev["verdict"] == "rejected":
        parts.append(ev["feedback"])
    if state["risk"].get("vetoes"):
        parts.append("Risk vetoes: " + "; ".join(state["risk"]["vetoes"]) + ".")
    return " ".join(parts)


class DecisionService:
    def __init__(self, cfg: Settings = settings, provider: HistoryProvider | None = None,
                 store: DecisionStore | None = None) -> None:
        self.cfg = cfg
        self.provider = provider or make_provider(cfg.data_source, cfg.data_dir)
        self.store = store if store is not None else DecisionStore(cfg.database_url)
        self.risk_cfg = RiskConfig(capital=cfg.risk_capital, risk_pct=cfg.risk_pct,
                                   max_data_age_days=cfg.max_data_age_days)
        self.graph = build_decision_graph(self.risk_cfg)

    # ------------------------------------------------------------- models
    def train(self, instrument: str, df: pd.DataFrame | None = None) -> ModelBundle:
        df = self.provider.get(instrument) if df is None else df
        bundle = train_model(df, instrument, horizon=self.cfg.horizon_days)
        save_bundle(bundle, self.cfg.model_dir)
        return bundle

    def get_bundle(self, instrument: str, df: pd.DataFrame) -> tuple[ModelBundle, bool]:
        bundle = load_bundle(instrument, self.cfg.model_dir)
        stale = bundle is None or bundle.horizon != self.cfg.horizon_days or (
            df.index[-1] - pd.Timestamp(bundle.last_bar, tz="UTC")).days > RETRAIN_AFTER_DAYS
        if stale:
            return self.train(instrument, df), True
        return bundle, False

    def _dxy(self) -> pd.DataFrame | None:
        try:
            return self.provider.get(DOLLAR_INDEX)
        except Exception as exc:  # macro agent will abstain
            log.warning("dollar index unavailable: %s", exc)
            return None

    # ----------------------------------------------------------- decisions
    def decide(self, instrument: str, orderflow: dict | None = None, dxy: pd.DataFrame | None = None,
               save: bool = True) -> dict[str, Any]:
        if instrument not in INSTRUMENTS:
            raise ValueError(f"unknown instrument {instrument}; choose from {sorted(INSTRUMENTS)}")
        df = self.provider.get(instrument)
        feats = compute_features(df)
        valid = np.flatnonzero(feats.notna().all(axis=1).to_numpy())
        if len(valid) == 0:
            raise ValueError(f"{instrument}: not enough history")
        bundle, retrained = self.get_bundle(instrument, df)
        snap = make_snapshot(instrument, df, feats, int(valid[-1]), dxy_ret_20(dxy if dxy is not None else self._dxy()), orderflow)
        state = self.graph.invoke({"snapshot": snap, "bundle": bundle, "now": pd.Timestamp.now(tz="UTC")})
        decision = {
            "instrument": instrument,
            "as_of": str(snap.as_of.date()),
            "last_close": round(snap.close, 4),
            "final_action": state["evaluation"]["final_action"],
            "regime": state["regime"],
            "explanation": explain(instrument, state),
            "aggregate": state["aggregate"],
            "votes": [v.as_dict() for v in sorted(state["votes"], key=lambda v: VOTERS.index(v.agent))],
            "ml": state.get("ml", {}),
            "model": {**bundle.summary(), "retrained_now": retrained},
            "risk": state["risk"],
            "evaluation": state["evaluation"],
            "data_source": self.cfg.data_source,
            "disclaimer": DISCLAIMER,
        }
        decision = to_jsonable(decision)
        if save:
            decision["id"] = self.store.save(decision)
        return decision

    def decide_many(self, instruments: list[str], orderflow: dict[str, dict] | None = None) -> list[dict]:
        dxy = self._dxy()
        orderflow = orderflow or {}

        def one(inst: str) -> dict:
            try:
                return self.decide(inst, orderflow.get(inst), dxy)
            except Exception as exc:
                log.exception("decision failed for %s", inst)
                return {"instrument": inst, "error": str(exc)}

        with ThreadPoolExecutor(max_workers=min(8, max(1, len(instruments)))) as pool:
            return list(pool.map(one, instruments))

    # ------------------------------------------------------------ backtest
    def evaluate(self, instrument: str) -> dict[str, Any]:
        """Replay the full agent graph over the model's hold-out period.

        Uses the model trained only on pre-hold-out data. One decision every `horizon`
        days (non-overlapping), held for `horizon` days, round-trip costs deducted.
        Stops and targets are not simulated.
        """
        df = self.provider.get(instrument)
        bundle = train_model(df, instrument, horizon=self.cfg.horizon_days)
        h = bundle.horizon
        feats = compute_features(df)
        dxy20 = dxy_ret_20(self._dxy())
        closes = df["close"].to_numpy()
        start = int(np.searchsorted(feats.index, pd.Timestamp(bundle.metrics.test_start, tz="UTC")))
        cost = 2 * self.cfg.cost_bps / 1e4

        rows, scorecard = [], {}
        for i in range(start, len(df) - h, h):
            if feats.iloc[i].isna().any():
                continue
            snap = make_snapshot(instrument, df, feats, i, dxy20)
            st = self.graph.invoke({"snapshot": snap, "bundle": bundle, "use_holdout": True, "now": snap.as_of})
            fwd = float(np.log(closes[i + h] / closes[i]))
            final = st["evaluation"]["final_action"]
            pnl = DIRECTION[final] * fwd - (cost if final != HOLD else 0.0)
            rows.append({"as_of": str(snap.as_of.date()), "final": final, "raw": st["aggregate"]["action"],
                         "regime": st["regime"], "fwd_return": fwd, "pnl": pnl})
            for v in st["votes"]:
                if v.abstained or v.action == HOLD:
                    continue
                s = scorecard.setdefault(v.agent, {"votes": 0, "hits": 0, "signed_return": 0.0})
                s["votes"] += 1
                s["hits"] += int(DIRECTION[v.action] * fwd > 0)
                s["signed_return"] += DIRECTION[v.action] * fwd

        if not rows:
            raise ValueError(f"{instrument}: hold-out period too short to evaluate")
        r = pd.DataFrame(rows)
        trades = r[r["final"] != HOLD]
        years = max(len(r) * h / 252, 1e-9)
        per_period = r["pnl"]
        sharpe = float(per_period.mean() / per_period.std() * np.sqrt(252 / h)) if per_period.std() > 0 else 0.0
        return to_jsonable({
            "instrument": instrument,
            "period": f"{r['as_of'].iloc[0]} to {r['as_of'].iloc[-1]}",
            "horizon_days": h,
            "decisions": len(r),
            "trades": len(trades),
            "actions": r["final"].value_counts().to_dict(),
            "vetoed_or_rejected": int(((r["raw"] != HOLD) & (r["final"] == HOLD)).sum()),
            "hit_rate": round(float((trades["pnl"] > 0).mean()), 4) if len(trades) else None,
            "avg_trade_bps": round(float(trades["pnl"].mean() * 1e4), 1) if len(trades) else None,
            "total_return_pct": round(float(np.expm1(r["pnl"].sum()) * 100), 2),
            "buy_and_hold_pct": round(float(np.expm1(r["fwd_return"].sum()) * 100), 2),
            "annualised_sharpe": round(sharpe, 2),
            "years": round(years, 2),
            "model_holdout": bundle.summary()["metrics"],
            "agent_scorecard": {
                a: {"directional_votes": s["votes"], "hit_rate": round(s["hits"] / s["votes"], 4),
                    "avg_signed_return_bps": round(s["signed_return"] / s["votes"] * 1e4, 1)}
                for a, s in sorted(scorecard.items())
            },
            "note": "Each decision held for the horizon; stops/targets not simulated; costs "
                    f"{self.cfg.cost_bps} bps per side.",
        })
