from app.agents import MarketInput


DEFAULT_MARKET_SNAPSHOT = {
    "gold_price": 3149.0,
    "gold_change_pct": 5.0,
    "silver_price": 61.0,
    "silver_change_pct": 7.0,
    "dollar_index_change_pct": 3.0,
    "volatility_score": 6,
}


def fetch_market_inputs() -> MarketInput:
    """Fetch automated gold/silver inputs for the Phase 1 prototype.

    This beginner implementation returns a deterministic local snapshot so the
    UI can run without paid market-data credentials. In later phases, replace
    this function with a real provider call while keeping the same return type.
    """
    return MarketInput(**DEFAULT_MARKET_SNAPSHOT)
