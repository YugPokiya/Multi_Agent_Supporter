import dataclasses

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.data.history import SyntheticHistoryProvider, clean_history, synthetic_history
from app.decision.agents import (
    BUY, HOLD, SELL, DecisionEvaluator, MacroAgent, MeanReversionAgent, MLSignalAgent, MomentumAgent,
    OrderFlowAgent, RegimeAgent, RiskAgent, RiskConfig, Snapshot, Vote, aggregate,
)
from app.decision.service import DecisionService
from app.decision.store import DecisionStore
from app.ml.features import FEATURES, compute_features, make_labels
from app.ml.model import ModelMetrics, train_model


# ------------------------------------------------------------------ helpers
def snap(instrument="XAU", **feat) -> Snapshot:
    base = {"ret_1": 0.0, "ret_5": 0.0, "ret_20": 0.0, "vol_20": 0.2, "rsi_14": 0.5, "macd_hist": 0.0,
            "dist_sma50": 0.0, "sma50_slope": 0.0, "atr_pct": 0.01, "range_pct": 0.01, "vol_z20": 0.0}
    base.update(feat)
    row = pd.Series(base)[FEATURES]
    hist = pd.DataFrame([row] * 30)
    hist["vol_20"] = np.linspace(0.1, 0.3, 30)
    hist.iloc[-1] = row
    return Snapshot(instrument, pd.Timestamp("2026-10-28", tz="UTC"), 100.0, row, hist)


class StubModel:
    def __init__(self, p):
        self.p = np.array([p])

    def predict_proba(self, x):
        return self.p


def stub_bundle(probs, has_edge=True):
    m = ModelMetrics(800, 300, "2025-01-01", 0.45, 0.36, 1.05, 1.09, has_edge, {})
    return type("B", (), {"model": StubModel(probs), "holdout_model": None, "metrics": m, "horizon": 5,
                          "features": FEATURES})()


@pytest.fixture
def service(tmp_path):
    cfg = dataclasses.replace(Settings(), data_source="synthetic", model_dir=str(tmp_path / "models"),
                              database_url=f"sqlite:///{tmp_path / 'd.db'}")
    return DecisionService(cfg, SyntheticHistoryProvider(days=900))


# ----------------------------------------------------------------- features
def test_features_have_no_lookahead():
    df = synthetic_history(400, seed=3)
    full = compute_features(df)
    for k in (120, 250, 399):
        part = compute_features(df.iloc[: k + 1])
        pd.testing.assert_series_equal(part.iloc[-1], full.iloc[k], check_names=False, rtol=1e-9)


def test_labels_use_only_the_forward_window():
    df = synthetic_history(200, seed=4)
    lab = make_labels(df, 5)
    assert lab.iloc[-5:].isna().all() and lab.iloc[25:-5].notna().all()
    assert set(lab.dropna().unique()) <= {0.0, 1.0, 2.0}


def test_clean_history_drops_bad_rows_and_duplicates():
    raw = pd.DataFrame({"timestamp": ["2026-01-02", "2026-01-01", "2026-01-02", "2026-01-03"],
                        "open": [10, 9, 11, 10], "high": [12, 10, 12, 9], "low": [9, 8, 10, 8], "close": [11, 9.5, 11.5, 10]})
    df = clean_history(raw)
    assert list(df.index.strftime("%Y-%m-%d")) == ["2026-01-01", "2026-01-02"]  # 01-03 has high < open
    assert df.loc["2026-01-02", "close"].item() == 11.5 and (df["volume"] == 0).all()


# -------------------------------------------------------------------- model
def test_model_purges_overlap_and_reports_baselines():
    df = synthetic_history(1200, seed=5)
    b = train_model(df, "XAU", horizon=5)
    m = b.metrics
    assert m.train_rows + m.test_rows + 5 == int((compute_features(df).notna().all(axis=1) & make_labels(df, 5).notna()).sum())
    assert 0 < m.majority_baseline < 1 and m.prior_log_loss > 0


def test_pure_noise_gets_no_edge():
    rng = np.random.default_rng(11)
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, 1500)))
    idx = pd.bdate_range("2020-01-01", periods=1500, tz="UTC")
    df = pd.DataFrame({"open": c, "high": c * 1.005, "low": c * 0.995, "close": c, "volume": 1000.0}, index=idx)
    assert train_model(df, "XAU").metrics.has_edge is False


# ------------------------------------------------------------------- agents
def test_ml_agent_abstains_without_edge_and_votes_with_edge():
    s = snap()
    v, ml = MLSignalAgent().vote(s, stub_bundle([0.1, 0.2, 0.7], has_edge=False))
    assert v.abstained and "no out-of-sample edge" in v.reasons[0] and ml["predicted"] == BUY
    v, _ = MLSignalAgent().vote(s, stub_bundle([0.1, 0.2, 0.7], has_edge=True))
    assert not v.abstained and v.action == BUY and v.confidence == pytest.approx(0.7)
    v, _ = MLSignalAgent().vote(s, stub_bundle([0.3, 0.3, 0.4], has_edge=True))
    assert v.abstained  # below min confidence
    v, _ = MLSignalAgent().vote(s, None)
    assert v.abstained


def test_regime_classification():
    assert RegimeAgent().vote(snap(dist_sma50=0.03, sma50_slope=0.02, vol_20=0.15))[1] == "trending_up"
    assert RegimeAgent().vote(snap(dist_sma50=-0.03, sma50_slope=-0.02, vol_20=0.15))[1] == "trending_down"
    assert RegimeAgent().vote(snap(vol_20=0.5))[1] == "high_volatility"
    v, regime = RegimeAgent().vote(snap(dist_sma50=0.01, sma50_slope=0.001, vol_20=0.15))
    assert regime == "range_bound" and v.abstained


def test_momentum_mean_reversion_macro_orderflow():
    assert MomentumAgent().vote(snap(ret_20=0.06, macd_hist=0.001, vol_20=0.2)).action == BUY
    assert MomentumAgent().vote(snap(ret_20=0.06, macd_hist=-0.001)).abstained
    assert MeanReversionAgent().vote(snap(rsi_14=0.22)).action == BUY
    assert MeanReversionAgent().vote(snap(rsi_14=0.80)).action == SELL
    assert MeanReversionAgent().vote(snap(rsi_14=0.50)).abstained
    s = snap("XAU"); s.dxy_ret_20 = -0.02
    assert MacroAgent().vote(s).action == BUY
    s = snap("BTC"); s.dxy_ret_20 = -0.02
    assert MacroAgent().vote(s).abstained
    s = snap("XAU")
    s.orderflow = dict(last_price=2350.5, session="new_york", cumulative_delta=1200, delta_change=450, volume_ratio=1.4,
                       bid_ask_imbalance=0.22, absorption_score=6, trapped_trader_score=8, structure_break="long",
                       liquidity_sweep="long", atr_5m=3.2)
    assert OrderFlowAgent().vote(s).action == BUY
    s = snap("CRUDE"); s.orderflow = {"last_price": 1}
    assert OrderFlowAgent().vote(s).abstained  # unusable payload never crashes the graph


def test_aggregate_weights_follow_regime():
    votes = [Vote("momentum", BUY, 0.6), Vote("mean_reversion", SELL, 0.6), Vote("ml_signal", HOLD, 0, abstained=True)]
    trend = aggregate(votes, "trending_up")
    rng = aggregate(votes, "range_bound")
    assert trend["score"] > 0 > rng["score"]
    assert trend["weights"]["momentum"] == 1.5 and rng["weights"]["mean_reversion"] == 1.5
    assert aggregate([Vote("regime", BUY, 0.3)], "high_volatility")["action"] == HOLD  # higher bar in high vol
    assert aggregate([], "range_bound")["action"] == HOLD


def test_risk_sizing_and_vetoes():
    s = snap(atr_pct=0.02)  # ATR = 2.0 on a close of 100
    agg = {"action": BUY, "confidence": 0.6}
    r = RiskAgent(RiskConfig(capital=100_000, risk_pct=0.01)).review(s, agg, "trending_up", {}, s.as_of)
    assert r["approved"] and r["stop_loss"] == 96.0 and r["target"] == 106.0
    assert r["position_units"] == pytest.approx(1000 / 4.0)
    stale = RiskAgent(RiskConfig()).review(s, agg, "trending_up", {}, s.as_of + pd.Timedelta(days=9))
    assert not stale["approved"] and "days old" in stale["vetoes"][0]
    ml = {"has_edge": True, "probabilities": {"SELL": 0.6, "HOLD": 0.2, "BUY": 0.2}}
    assert not RiskAgent(RiskConfig()).review(s, agg, "trending_up", ml, s.as_of)["approved"]
    weak = RiskAgent(RiskConfig()).review(s, {"action": BUY, "confidence": 0.3}, "high_volatility", {}, s.as_of)
    assert not weak["approved"]


def test_evaluator_requires_agreement():
    risk = {"approved": True, "reward_to_risk": 1.5}
    one = aggregate([Vote("regime", BUY, 0.7)], "trending_up")
    assert DecisionEvaluator().evaluate([Vote("regime", BUY, 0.7)], one, risk)["final_action"] == HOLD
    votes = [Vote("regime", BUY, 0.7), Vote("momentum", BUY, 0.6)]
    two = aggregate(votes, "trending_up")
    assert DecisionEvaluator().evaluate(votes, two, risk)["final_action"] == BUY
    votes.append(Vote("mean_reversion", SELL, 0.65))
    assert DecisionEvaluator().evaluate(votes, aggregate(votes, "trending_up"), risk)["verdict"] == "rejected"


# ------------------------------------------------------------ end to end
def test_decide_many_runs_graph_saves_history_and_isolates_failures(service):
    class Flaky(SyntheticHistoryProvider):
        def get(self, instrument):
            if instrument == "NIFTY":
                raise ConnectionError("feed down")
            return super().get(instrument)

    service.provider = Flaky(days=900)
    out = service.decide_many(["XAU", "BTC", "NIFTY"])
    by = {d["instrument"]: d for d in out}
    assert by["NIFTY"]["error"] == "feed down"
    for inst in ("XAU", "BTC"):
        d = by[inst]
        assert d["final_action"] in (BUY, SELL, HOLD)
        assert [v["agent"] for v in d["votes"]] == ["ml_signal", "regime", "momentum", "mean_reversion", "macro_dxy", "order_flow"]
        assert d["model"]["retrained_now"] is True and set(d["ml"]["probabilities"]) == {BUY, SELL, HOLD}
        assert d["explanation"].startswith(d["final_action"])
    assert {r["instrument"] for r in service.store.history()} == {"XAU", "BTC"}
    again = service.decide("XAU")
    assert again["model"]["retrained_now"] is False  # cached model reused


def test_evaluate_reports_scorecard(service):
    r = service.evaluate("XAG")
    assert r["decisions"] > 10 and r["horizon_days"] == 5
    assert sum(r["actions"].values()) == r["decisions"]
    assert all(0 <= a["hit_rate"] <= 1 for a in r["agent_scorecard"].values())


def test_api_decisions_endpoint(service, monkeypatch):
    import app.api.decisions as api
    from app.main import app

    monkeypatch.setattr(api, "get_service", lambda: service)
    client = TestClient(app)
    res = client.post("/api/decisions", json={"instruments": ["XAU", "BTC"]})
    assert res.status_code == 200
    body = res.json()
    assert set(body["summary"]) == {"XAU", "BTC"} and len(body["decisions"]) == 2
    assert client.post("/api/decisions", json={"instruments": ["DOGE"]}).status_code == 422
    assert len(client.get("/api/decisions/history").json()) == 2
    assert client.get("/").status_code == 200


def test_store_roundtrip(tmp_path):
    st = DecisionStore(f"sqlite:///{tmp_path / 's.db'}")
    st.save({"instrument": "XAU", "as_of": "2026-10-28", "final_action": "BUY", "regime": "trending_up",
             "aggregate": {"action": "BUY", "score": 0.5, "confidence": 0.5}, "np": np.float64(1.5)})
    assert st.history("XAU")[0]["final_action"] == "BUY" and st.history("BTC") == []
