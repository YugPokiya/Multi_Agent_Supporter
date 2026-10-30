from dataclasses import dataclass
from typing import Literal

Instrument = Literal["XAU", "XAG", "BTC", "NIFTY"]
Bias = Literal["long", "short", "neutral"]
Session = Literal["asia", "london", "new_york", "india", "crypto_24x7"]


@dataclass(frozen=True)
class OrderFlowInput:
    instrument: Instrument
    last_price: float
    session: Session
    cumulative_delta: float
    delta_change: float
    volume_ratio: float
    bid_ask_imbalance: float
    absorption_score: int
    trapped_trader_score: int
    structure_break: Bias
    liquidity_sweep: Bias
    atr_5m: float


@dataclass(frozen=True)
class OrderFlowPlan:
    instrument: Instrument
    timeframe: str
    bias: Bias
    confidence: float
    setup_name: str
    entry_trigger: str
    invalidation: str
    risk_plan: str
    targets: list[str]
    management_rules: list[str]
    score_breakdown: dict[str, float]
    disclaimer: str


INSTRUMENT_RISK = {
    "XAU": {
        "label": "Gold spot/futures proxy",
        "risk": "Risk a small fixed fraction per trade; gold can spike during USD news and COMEX opens.",
    },
    "XAG": {
        "label": "Silver spot/futures proxy",
        "risk": "Use smaller size than gold because silver frequently overshoots liquidity zones.",
    },
    "BTC": {
        "label": "Bitcoin perpetual/spot proxy",
        "risk": "Avoid over-leverage; crypto order books can thin out around funding and weekend liquidity.",
    },
    "NIFTY": {
        "label": "NIFTY index/futures proxy",
        "risk": "Respect India session timing, opening range volatility, and event-day gap risk.",
    },
}


def build_orderflow_plan(flow: OrderFlowInput) -> OrderFlowPlan:
    """Create a rules-based 5-minute order-flow trading plan.

    The model scores liquidity sweeps, delta confirmation, bid/ask imbalance,
    absorption, trapped traders, and market-structure breaks. It intentionally
    returns a plan rather than a direct trade command so users can train on the
    checklist before risking capital.
    """
    long_score = 0.0
    short_score = 0.0
    breakdown: dict[str, float] = {}

    if flow.structure_break == "long":
        long_score += 2.0
        breakdown["bullish_structure_break"] = 2.0
    elif flow.structure_break == "short":
        short_score += 2.0
        breakdown["bearish_structure_break"] = 2.0

    if flow.liquidity_sweep == "long":
        long_score += 1.5
        breakdown["sell-side_liquidity_sweep_reclaim"] = 1.5
    elif flow.liquidity_sweep == "short":
        short_score += 1.5
        breakdown["buy-side_liquidity_sweep_rejection"] = 1.5

    if flow.delta_change > 0 and flow.cumulative_delta > 0:
        long_score += 1.5
        breakdown["positive_delta_confirmation"] = 1.5
    elif flow.delta_change < 0 and flow.cumulative_delta < 0:
        short_score += 1.5
        breakdown["negative_delta_confirmation"] = 1.5

    if flow.bid_ask_imbalance > 0.15:
        long_score += 1.0
        breakdown["ask_lift_imbalance"] = 1.0
    elif flow.bid_ask_imbalance < -0.15:
        short_score += 1.0
        breakdown["bid_hit_imbalance"] = 1.0

    if flow.absorption_score >= 7:
        if flow.delta_change < 0 and flow.liquidity_sweep == "long":
            long_score += 1.0
            breakdown["sell_absorption_at_lows"] = 1.0
        elif flow.delta_change > 0 and flow.liquidity_sweep == "short":
            short_score += 1.0
            breakdown["buy_absorption_at_highs"] = 1.0

    if flow.trapped_trader_score >= 7:
        if flow.liquidity_sweep == "long":
            long_score += 1.0
            breakdown["trapped_shorts"] = 1.0
        elif flow.liquidity_sweep == "short":
            short_score += 1.0
            breakdown["trapped_longs"] = 1.0

    if flow.volume_ratio >= 1.25:
        long_score += 0.5 if long_score > short_score else 0
        short_score += 0.5 if short_score > long_score else 0
        breakdown["above_average_volume"] = 0.5

    edge = abs(long_score - short_score)
    if edge < 1.0 or max(long_score, short_score) < 3.0:
        bias: Bias = "neutral"
    else:
        bias = "long" if long_score > short_score else "short"

    confidence = min(0.9, round(0.45 + max(long_score, short_score) * 0.06 + edge * 0.03, 2))
    atr = max(flow.atr_5m, flow.last_price * 0.0005)
    label = INSTRUMENT_RISK[flow.instrument]["label"]

    if bias == "long":
        setup = "5m sell-side sweep reclaim with positive order-flow shift"
        entry = "Wait for a 5-minute close back above the swept low/VWAP micro-zone, then enter only if delta turns positive and offers keep lifting."
        invalidation = f"Stop below the sweep low or about {round(atr * 0.8, 2)} points below entry, whichever is tighter but outside noise."
        targets = [
            f"First target: +{round(atr, 2)} from entry or nearest internal liquidity.",
            f"Second target: +{round(atr * 2, 2)} or prior buy-side liquidity pool.",
        ]
    elif bias == "short":
        setup = "5m buy-side sweep rejection with negative order-flow shift"
        entry = "Wait for a 5-minute close back below the swept high/VWAP micro-zone, then enter only if delta turns negative and bids keep hitting."
        invalidation = f"Stop above the sweep high or about {round(atr * 0.8, 2)} points above entry, whichever is tighter but outside noise."
        targets = [
            f"First target: -{round(atr, 2)} from entry or nearest internal liquidity.",
            f"Second target: -{round(atr * 2, 2)} or prior sell-side liquidity pool.",
        ]
    else:
        setup = "No-trade order-flow training state"
        entry = "Do not enter until liquidity sweep, structure break, delta, imbalance, and volume align on the 5-minute chart."
        invalidation = "No trade means no stop; define invalidation only after a valid directional setup appears."
        targets = ["Journal the next liquidity pool above and below price.", "Wait for a clearer asymmetric setup."]

    return OrderFlowPlan(
        instrument=flow.instrument,
        timeframe="5m",
        bias=bias,
        confidence=confidence,
        setup_name=f"{label}: {setup}",
        entry_trigger=entry,
        invalidation=invalidation,
        risk_plan=INSTRUMENT_RISK[flow.instrument]["risk"],
        targets=targets,
        management_rules=[
            "Move stop to breakeven only after target 1 is reached or clear opposing absorption appears.",
            "Skip entries during major scheduled news unless this is a replay/backtest drill.",
            "Record screenshot, delta behavior, liquidity level, and result for order-flow training.",
        ],
        score_breakdown={"long_score": long_score, "short_score": short_score, **breakdown},
        disclaimer="Educational order-flow training only; not financial advice or a live trade signal.",
    )
