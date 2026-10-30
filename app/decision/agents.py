"""Decision agents. Each one looks at the same market snapshot and votes BUY / SELL / HOLD.

A vote is either
  * active  - a BUY, SELL or a deliberate HOLD ("conditions say stay out"), with a confidence, or
  * abstained - the agent has no opinion (no signal, not applicable, model without edge).
Abstentions are ignored by the aggregator; active HOLD votes pull the score towards zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from app.ml.model import ModelBundle, predict_row
from app.orderflow import OrderFlowInput, build_orderflow_plan

BUY, SELL, HOLD = "BUY", "SELL", "HOLD"
DIRECTION = {BUY: 1, SELL: -1, HOLD: 0}
METALS = {"XAU", "XAG"}


@dataclass
class Vote:
    agent: str
    action: str
    confidence: float
    reasons: list[str] = field(default_factory=list)
    abstained: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["confidence"] = round(float(self.confidence), 3)
        return d


def abstain(agent: str, reason: str, **details: Any) -> Vote:
    return Vote(agent, HOLD, 0.0, [reason], abstained=True, details=details)


@dataclass
class Snapshot:
    """Everything the agents may look at, as of one closed daily bar."""

    instrument: str
    as_of: pd.Timestamp
    close: float
    features: pd.Series                 # latest feature row
    feature_history: pd.DataFrame       # feature rows up to and including as_of
    dxy_ret_20: float | None = None     # 20-day log change of the dollar index
    orderflow: dict | None = None       # optional 5-minute order-flow input for this instrument


# --------------------------------------------------------------------------- ML
class MLSignalAgent:
    name = "ml_signal"

    def __init__(self, min_confidence: float = 0.45, require_edge: bool = True) -> None:
        self.min_confidence = min_confidence
        self.require_edge = require_edge

    def vote(self, snap: Snapshot, bundle: ModelBundle | None, use_holdout: bool = False) -> tuple[Vote, dict]:
        if bundle is None:
            return abstain(self.name, "no trained model for this instrument"), {}
        probs = predict_row(bundle, snap.features, use_holdout=use_holdout)
        predicted = max(probs, key=probs.get)
        m = bundle.metrics
        ml = {
            "probabilities": {k: round(v, 4) for k, v in probs.items()},
            "predicted": predicted,
            "confidence": round(probs[predicted], 4),
            "has_edge": m.has_edge,
            "holdout_accuracy": m.accuracy,
            "majority_baseline": m.majority_baseline,
        }
        quality = f"hold-out accuracy {m.accuracy:.1%} vs baseline {m.majority_baseline:.1%}, log-loss {m.log_loss} vs {m.prior_log_loss}"
        if self.require_edge and not m.has_edge:
            return abstain(self.name, f"model shows no out-of-sample edge ({quality}); ignoring its {predicted} call", **ml), ml
        if probs[predicted] < self.min_confidence:
            return abstain(self.name, f"model unsure: top class {predicted} at {probs[predicted]:.0%}", **ml), ml
        return Vote(self.name, predicted, probs[predicted],
                    [f"model predicts {predicted} over {bundle.horizon} days with p={probs[predicted]:.0%}", quality],
                    details=ml), ml


# ------------------------------------------------------------------- regime
class RegimeAgent:
    name = "regime"

    def __init__(self, vol_percentile_high: float = 0.85, slope_min: float = 0.005) -> None:
        self.vol_high = vol_percentile_high
        self.slope_min = slope_min

    def classify(self, snap: Snapshot) -> tuple[str, dict]:
        f = snap.features
        vol_hist = snap.feature_history["vol_20"].dropna().iloc[-252:]
        vol_pct = float((vol_hist <= f["vol_20"]).mean()) if len(vol_hist) else 0.5
        info = {"vol_percentile_1y": round(vol_pct, 3), "dist_sma50": round(float(f["dist_sma50"]), 4),
                "sma50_slope_10d": round(float(f["sma50_slope"]), 4), "vol_20_annual": round(float(f["vol_20"]), 4)}
        if vol_pct >= self.vol_high:
            return "high_volatility", info
        if f["dist_sma50"] > 0 and f["sma50_slope"] > self.slope_min:
            return "trending_up", info
        if f["dist_sma50"] < 0 and f["sma50_slope"] < -self.slope_min:
            return "trending_down", info
        return "range_bound", info

    def vote(self, snap: Snapshot) -> tuple[Vote, str]:
        regime, info = self.classify(snap)
        strength = min(0.8, 0.5 + abs(info["sma50_slope_10d"]) * 10)
        if regime == "trending_up":
            v = Vote(self.name, BUY, strength, ["price above a rising 50-day average"], details=info)
        elif regime == "trending_down":
            v = Vote(self.name, SELL, strength, ["price below a falling 50-day average"], details=info)
        elif regime == "high_volatility":
            v = Vote(self.name, HOLD, 0.6, [f"volatility in the top {1 - info['vol_percentile_1y']:.0%} of the past year"], details=info)
        else:
            v = abstain(self.name, "no clear trend; range-bound", **info)
        v.details["regime"] = regime
        return v, regime


# ----------------------------------------------------------------- momentum
class MomentumAgent:
    name = "momentum"

    def vote(self, snap: Snapshot) -> Vote:
        f = snap.features
        horizon_vol = f["vol_20"] * np.sqrt(20 / 252)
        z = float(f["ret_20"] / horizon_vol) if horizon_vol > 0 else 0.0
        details = {"ret_20": round(float(f["ret_20"]), 4), "ret_20_z": round(z, 2), "macd_hist": float(f["macd_hist"])}
        if np.sign(f["macd_hist"]) == np.sign(f["ret_20"]) and abs(z) >= 0.5:
            action = BUY if z > 0 else SELL
            conf = float(np.clip(0.4 + 0.15 * abs(z), 0, 0.85))
            return Vote(self.name, action, conf,
                        [f"20-day move is {z:+.1f} sigma and MACD histogram agrees"], details=details)
        return abstain(self.name, "MACD and 20-day return disagree or move is small", **details)


# ----------------------------------------------------------- mean reversion
class MeanReversionAgent:
    name = "mean_reversion"

    def __init__(self, oversold: float = 0.30, overbought: float = 0.70) -> None:
        self.oversold, self.overbought = oversold, overbought

    def vote(self, snap: Snapshot) -> Vote:
        rsi = float(snap.features["rsi_14"])
        details = {"rsi_14": round(rsi * 100, 1)}
        if rsi <= self.oversold:
            return Vote(self.name, BUY, min(0.8, 0.45 + (self.oversold - rsi) * 2), [f"RSI {rsi * 100:.0f} oversold"], details=details)
        if rsi >= self.overbought:
            return Vote(self.name, SELL, min(0.8, 0.45 + (rsi - self.overbought) * 2), [f"RSI {rsi * 100:.0f} overbought"], details=details)
        return abstain(self.name, f"RSI {rsi * 100:.0f} neutral", **details)


# -------------------------------------------------------------------- macro
class MacroAgent:
    """Dollar-index filter for gold and silver (they usually move against the dollar)."""

    name = "macro_dxy"

    def __init__(self, threshold: float = 0.01) -> None:
        self.threshold = threshold

    def vote(self, snap: Snapshot) -> Vote:
        if snap.instrument not in METALS:
            return abstain(self.name, "dollar-index filter only applies to gold and silver")
        if snap.dxy_ret_20 is None or np.isnan(snap.dxy_ret_20):
            return abstain(self.name, "dollar-index data unavailable")
        d = {"dxy_ret_20": round(float(snap.dxy_ret_20), 4)}
        if snap.dxy_ret_20 <= -self.threshold:
            return Vote(self.name, BUY, 0.5, [f"dollar index down {abs(snap.dxy_ret_20):.1%} over 20 days"], details=d)
        if snap.dxy_ret_20 >= self.threshold:
            return Vote(self.name, SELL, 0.5, [f"dollar index up {abs(snap.dxy_ret_20):.1%} over 20 days"], details=d)
        return abstain(self.name, "dollar index flat", **d)


# --------------------------------------------------------------- order flow
class OrderFlowAgent:
    """Wraps the existing 5-minute order-flow planner when the caller supplies order-flow input."""

    name = "order_flow"

    def vote(self, snap: Snapshot) -> Vote:
        if not snap.orderflow:
            return abstain(self.name, "no order-flow input supplied")
        try:
            plan = build_orderflow_plan(OrderFlowInput(instrument=snap.instrument, **snap.orderflow))
        except Exception as exc:  # unsupported instrument or bad payload
            return abstain(self.name, f"order-flow input not usable: {exc}")
        action = {"long": BUY, "short": SELL}.get(plan.bias, HOLD)
        d = {"bias": plan.bias, "setup": plan.setup_name, "scores": plan.score_breakdown}
        if action == HOLD:
            return Vote(self.name, HOLD, 0.5, ["order flow is mixed: no-trade state"], details=d)
        return Vote(self.name, action, plan.confidence, [plan.setup_name], details=d)


# ---------------------------------------------------------------- aggregate
BASE_WEIGHTS = {"ml_signal": 2.0, "regime": 1.0, "momentum": 1.0, "mean_reversion": 1.0, "macro_dxy": 0.5, "order_flow": 1.0}
REGIME_MULT = {
    "trending_up": {"momentum": 1.5, "mean_reversion": 0.5},
    "trending_down": {"momentum": 1.5, "mean_reversion": 0.5},
    "range_bound": {"momentum": 0.5, "mean_reversion": 1.5},
    "high_volatility": {},
}


def aggregate(votes: list[Vote], regime: str, threshold: float = 0.20) -> dict[str, Any]:
    active = [v for v in votes if not v.abstained]
    weights = {v.agent: BASE_WEIGHTS.get(v.agent, 1.0) * REGIME_MULT.get(regime, {}).get(v.agent, 1.0) for v in active}
    total_w = sum(weights.values())
    score = sum(weights[v.agent] * v.confidence * DIRECTION[v.action] for v in active) / total_w if total_w else 0.0
    thr = threshold * (1.5 if regime == "high_volatility" else 1.0)
    action = BUY if score > thr else SELL if score < -thr else HOLD
    directional = [v for v in active if v.action != HOLD]
    agree = [v.agent for v in directional if v.action == action]
    oppose = [v.agent for v in directional if action != HOLD and v.action != action]
    return {
        "action": action,
        "score": round(score, 4),
        "threshold": thr,
        "confidence": round(min(1.0, abs(score)), 3),
        "weights": {k: round(w, 2) for k, w in weights.items()},
        "agreeing_agents": agree,
        "opposing_agents": oppose,
        "active_agents": len(active),
    }


# --------------------------------------------------------------------- risk
@dataclass
class RiskConfig:
    capital: float = 100_000
    risk_pct: float = 0.01
    stop_atr: float = 2.0
    target_atr: float = 3.0
    max_data_age_days: int = 5
    high_vol_min_confidence: float = 0.5
    ml_veto_probability: float = 0.5


class RiskAgent:
    name = "risk"

    def __init__(self, cfg: RiskConfig) -> None:
        self.cfg = cfg

    def review(self, snap: Snapshot, agg: dict, regime: str, ml: dict, now: pd.Timestamp) -> dict[str, Any]:
        action = agg["action"]
        atr = float(snap.features["atr_pct"]) * snap.close
        out: dict[str, Any] = {"atr": round(atr, 4), "vetoes": [], "approved": action != HOLD}
        age = (now.normalize() - snap.as_of.normalize()).days
        if age > self.cfg.max_data_age_days:
            out["vetoes"].append(f"data is {age} days old")
        if action == HOLD:
            out["approved"] = False
            return out
        if regime == "high_volatility" and agg["confidence"] < self.cfg.high_vol_min_confidence:
            out["vetoes"].append(f"high-volatility regime and confidence {agg['confidence']:.2f} < {self.cfg.high_vol_min_confidence}")
        opposite = SELL if action == BUY else BUY
        if ml.get("has_edge") and ml["probabilities"].get(opposite, 0) >= self.cfg.ml_veto_probability:
            out["vetoes"].append(f"ML model (with edge) gives {opposite} p={ml['probabilities'][opposite]:.0%}")

        d = DIRECTION[action]
        stop = snap.close - d * self.cfg.stop_atr * atr
        target = snap.close + d * self.cfg.target_atr * atr
        risk_amount = self.cfg.capital * self.cfg.risk_pct
        units = risk_amount / (self.cfg.stop_atr * atr) if atr > 0 else 0.0
        out.update(
            entry_reference=round(snap.close, 4),
            stop_loss=round(stop, 4),
            target=round(target, 4),
            reward_to_risk=round(self.cfg.target_atr / self.cfg.stop_atr, 2),
            risk_amount=round(risk_amount, 2),
            position_units=round(units, 4),
            notional=round(units * snap.close, 2),
        )
        out["approved"] = not out["vetoes"]
        return out


# ---------------------------------------------------------------- evaluator
class DecisionEvaluator:
    """Final quality gate. Only an approved BUY/SELL becomes the final action."""

    name = "evaluator"

    def evaluate(self, votes: list[Vote], agg: dict, risk: dict) -> dict[str, Any]:
        action = agg["action"]
        if action == HOLD:
            return {"verdict": "no_trade", "final_action": HOLD, "checks": {},
                    "feedback": "Agents do not agree strongly enough on a direction."}
        strong_conflict = any(
            not v.abstained and v.action not in (action, HOLD) and v.confidence >= 0.6 for v in votes
        )
        checks = {
            "two_or_more_agents_agree": len(agg["agreeing_agents"]) >= 2,
            "no_strong_opposing_vote": not strong_conflict,
            "reward_to_risk_at_least_1_5": risk.get("reward_to_risk", 0) >= 1.5,
            "risk_agent_approved": bool(risk.get("approved")),
        }
        passed = all(checks.values())
        return {
            "verdict": "approved" if passed else "rejected",
            "final_action": action if passed else HOLD,
            "checks": checks,
            "feedback": "All checks passed." if passed
            else "Failed: " + ", ".join(k.replace("_", " ") for k, ok in checks.items() if not ok) + ". Staying flat.",
        }
