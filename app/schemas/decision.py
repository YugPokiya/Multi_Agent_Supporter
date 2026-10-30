from typing import Any, Literal

from pydantic import BaseModel, Field

Instrument = Literal["XAU", "XAG", "BTC", "NIFTY", "CRUDE"]


class OrderFlowSnapshot(BaseModel):
    """Optional 5-minute order-flow input for one instrument (same fields as /api/orderflow-plan)."""

    last_price: float = Field(gt=0)
    session: Literal["asia", "london", "new_york", "india", "crypto_24x7"]
    cumulative_delta: float
    delta_change: float
    volume_ratio: float = Field(ge=0)
    bid_ask_imbalance: float = Field(ge=-1, le=1)
    absorption_score: int = Field(ge=1, le=10)
    trapped_trader_score: int = Field(ge=1, le=10)
    structure_break: Literal["long", "short", "neutral"]
    liquidity_sweep: Literal["long", "short", "neutral"]
    atr_5m: float = Field(gt=0)


class DecisionRequest(BaseModel):
    instruments: list[Instrument] = Field(default_factory=lambda: ["XAU", "XAG", "BTC", "NIFTY"], min_length=1)
    orderflow: dict[Instrument, OrderFlowSnapshot] = Field(default_factory=dict)


class DecisionBatchResponse(BaseModel):
    decisions: list[dict[str, Any]]
    summary: dict[str, str]


class TrainRequest(BaseModel):
    instruments: list[Instrument] = Field(default_factory=lambda: ["XAU", "XAG", "BTC", "NIFTY"], min_length=1)
