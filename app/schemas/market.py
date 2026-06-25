from pydantic import BaseModel


class MarketInputResponse(BaseModel):
    gold_price: float
    gold_change_pct: float
    silver_price: float
    silver_change_pct: float
    dollar_index_change_pct: float
    volatility_score: int
