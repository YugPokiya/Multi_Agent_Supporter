"""LangGraph workflow that turns one market snapshot into one decision.

        START
     ┌────┼─────┬──────────┬────────────┬──────────┐
  ml_signal  regime  momentum  mean_reversion  macro_dxy  order_flow     (run in parallel)
     └────┴─────┴──────────┴────────────┴──────────┘
                       aggregate      weighted vote, weights depend on the regime
                          │
                        risk          stops, size, vetoes (stale data, high vol, ML disagreement)
                          │
                       evaluator      approves or downgrades to HOLD
                          │
                         END
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

import pandas as pd
from langgraph.graph import END, START, StateGraph

from app.decision.agents import (
    DecisionEvaluator,
    MacroAgent,
    MeanReversionAgent,
    MLSignalAgent,
    MomentumAgent,
    OrderFlowAgent,
    RegimeAgent,
    RiskAgent,
    RiskConfig,
    Snapshot,
    Vote,
    aggregate,
)
from app.ml.model import ModelBundle

VOTERS = ["ml_signal", "regime", "momentum", "mean_reversion", "macro_dxy", "order_flow"]


class DecisionState(TypedDict, total=False):
    snapshot: Snapshot
    bundle: ModelBundle | None
    use_holdout: bool
    now: pd.Timestamp
    votes: Annotated[list[Vote], operator.add]
    regime: str
    ml: dict[str, Any]
    aggregate: dict[str, Any]
    risk: dict[str, Any]
    evaluation: dict[str, Any]


def build_decision_graph(risk_cfg: RiskConfig | None = None, ml_agent: MLSignalAgent | None = None):
    ml_agent = ml_agent or MLSignalAgent()
    regime_agent, momentum, meanrev = RegimeAgent(), MomentumAgent(), MeanReversionAgent()
    macro, orderflow = MacroAgent(), OrderFlowAgent()
    risk_agent, evaluator = RiskAgent(risk_cfg or RiskConfig()), DecisionEvaluator()

    def ml_node(s: DecisionState) -> dict:
        vote, ml = ml_agent.vote(s["snapshot"], s.get("bundle"), s.get("use_holdout", False))
        return {"votes": [vote], "ml": ml}

    def regime_node(s: DecisionState) -> dict:
        vote, regime = regime_agent.vote(s["snapshot"])
        return {"votes": [vote], "regime": regime}

    def aggregate_node(s: DecisionState) -> dict:
        return {"aggregate": aggregate(s["votes"], s["regime"])}

    def risk_node(s: DecisionState) -> dict:
        now = s.get("now") or pd.Timestamp.now(tz="UTC")
        return {"risk": risk_agent.review(s["snapshot"], s["aggregate"], s["regime"], s.get("ml", {}), now)}

    def evaluate_node(s: DecisionState) -> dict:
        return {"evaluation": evaluator.evaluate(s["votes"], s["aggregate"], s["risk"])}

    g = StateGraph(DecisionState)
    g.add_node("ml_signal", ml_node)
    g.add_node("regime", regime_node)
    g.add_node("momentum", lambda s: {"votes": [momentum.vote(s["snapshot"])]})
    g.add_node("mean_reversion", lambda s: {"votes": [meanrev.vote(s["snapshot"])]})
    g.add_node("macro_dxy", lambda s: {"votes": [macro.vote(s["snapshot"])]})
    g.add_node("order_flow", lambda s: {"votes": [orderflow.vote(s["snapshot"])]})
    g.add_node("aggregate", aggregate_node)
    g.add_node("risk", risk_node)
    g.add_node("evaluator", evaluate_node)
    for v in VOTERS:
        g.add_edge(START, v)
    g.add_edge(VOTERS, "aggregate")  # fan-in: waits for every voter
    g.add_edge("aggregate", "risk")
    g.add_edge("risk", "evaluator")
    g.add_edge("evaluator", END)
    return g.compile()
