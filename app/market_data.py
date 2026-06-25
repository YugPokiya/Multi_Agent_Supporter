import json
from dataclasses import dataclass
from typing import Callable
from urllib.parse import quote
from urllib.request import Request, urlopen

from app.agents import MarketInput


@dataclass(frozen=True)
class MarketQuote:
    symbol: str
    price: float
    previous_close: float


YAHOO_SYMBOLS = {
    "gold": "GC=F",
    "silver": "SI=F",
    "dollar_index": "DX-Y.NYB",
}


def fetch_market_inputs(
    quote_fetcher: Callable[[str], MarketQuote] | None = None,
) -> MarketInput:
    """Fetch latest real-world gold, silver, and dollar-index inputs.

    The default provider uses Yahoo Finance's public chart endpoint and requires
    internet access at runtime. Tests can pass a custom ``quote_fetcher`` to keep
    the behavior deterministic without calling external services.
    """
    fetcher = quote_fetcher or fetch_yahoo_quote
    gold = fetcher(YAHOO_SYMBOLS["gold"])
    silver = fetcher(YAHOO_SYMBOLS["silver"])
    dollar_index = fetcher(YAHOO_SYMBOLS["dollar_index"])

    gold_change_pct = calculate_change_pct(gold.price, gold.previous_close)
    silver_change_pct = calculate_change_pct(silver.price, silver.previous_close)
    dollar_change_pct = calculate_change_pct(
        dollar_index.price,
        dollar_index.previous_close,
    )

    return MarketInput(
        gold_price=round(gold.price, 2),
        gold_change_pct=gold_change_pct,
        silver_price=round(silver.price, 2),
        silver_change_pct=silver_change_pct,
        dollar_index_change_pct=dollar_change_pct,
        volatility_score=calculate_volatility_score(
            gold_change_pct,
            silver_change_pct,
            dollar_change_pct,
        ),
    )


def fetch_yahoo_quote(symbol: str) -> MarketQuote:
    """Return the latest quote and previous close for a Yahoo Finance symbol."""
    encoded_symbol = quote(symbol, safe="")
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{encoded_symbol}?range=5d&interval=1d"
    )
    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 MACE-Market-Regime-Analyzer/1.0",
        },
    )
    with urlopen(request, timeout=15) as response:
        payload = json.loads(response.read().decode("utf-8"))

    result = payload["chart"]["result"][0]
    meta = result["meta"]
    price = meta.get("regularMarketPrice")
    previous_close = meta.get("previousClose")

    if price is None or previous_close is None:
        closes = [
            close
            for close in result["indicators"]["quote"][0].get("close", [])
            if close is not None
        ]
        if len(closes) < 2:
            raise ValueError(f"Not enough quote data returned for {symbol}")
        previous_close, price = closes[-2], closes[-1]

    return MarketQuote(
        symbol=symbol,
        price=float(price),
        previous_close=float(previous_close),
    )


def calculate_change_pct(price: float, previous_close: float) -> float:
    if previous_close == 0:
        raise ValueError("previous_close cannot be zero")
    return round(((price - previous_close) / previous_close) * 100, 3)


def calculate_volatility_score(
    gold_change_pct: float,
    silver_change_pct: float,
    dollar_index_change_pct: float,
) -> int:
    """Convert market movement into a beginner-friendly 1-10 volatility score."""
    average_metals_move = (abs(gold_change_pct) + abs(silver_change_pct)) / 2
    metals_divergence = abs(gold_change_pct - silver_change_pct)
    raw_score = (
        1
        + average_metals_move * 1.2
        + metals_divergence * 0.8
        + abs(dollar_index_change_pct) * 0.5
    )
    return max(1, min(10, round(raw_score)))
