from typing import Literal

from pydantic import BaseModel, Field


class MarketInputResponse(BaseModel):
    gold_price: float
    gold_change_pct: float
    silver_price: float
    silver_change_pct: float
    dollar_index_change_pct: float
    volatility_score: int


class OrderFlowInputRequest(BaseModel):
    instrument: Literal["XAU", "XAG", "BTC", "NIFTY"]
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


class OrderFlowPlanResponse(BaseModel):
    instrument: str
    timeframe: str
    bias: str
    confidence: float
    setup_name: str
    entry_trigger: str
    invalidation: str
    risk_plan: str
    targets: list[str]
    management_rules: list[str]
    score_breakdown: dict[str, float]
    disclaimer: str
