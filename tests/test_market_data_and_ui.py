from app.market_data import MarketQuote, fetch_market_inputs
from app.memory import InMemoryStore
from app.orchestrator import SimpleOrchestrator
from pathlib import Path


HTML = Path("static/index.html").read_text(encoding="utf-8")


def fake_quote_fetcher(symbol: str) -> MarketQuote:
    quotes = {
        "GC=F": MarketQuote(symbol="GC=F", price=3149.0, previous_close=2999.0476),
        "SI=F": MarketQuote(symbol="SI=F", price=61.0, previous_close=57.0093),
        "DX-Y.NYB": MarketQuote(symbol="DX-Y.NYB", price=103.0, previous_close=100.0),
    }
    return quotes[symbol]


def test_fetch_market_inputs_returns_latest_provider_snapshot():
    market = fetch_market_inputs(fake_quote_fetcher)
    assert market.gold_price == 3149
    assert market.gold_change_pct == 5
    assert market.silver_price == 61
    assert market.silver_change_pct == 7
    assert market.dollar_index_change_pct == 3
    assert market.volatility_score == 10


def test_automated_snapshot_runs_through_orchestrator(tmp_path):
    store = InMemoryStore(str(tmp_path / "store.json"))
    context = SimpleOrchestrator(store).run(fetch_market_inputs(fake_quote_fetcher))
    assert context["market_input"]["gold_price"] == 3149
    assert context["assessment"]["regime"] in {
        "bullish_precious_metals",
        "bearish_precious_metals",
        "high_volatility",
        "divergence",
        "range_bound",
    }
    assert context["evaluation"]["score"] >= 0


def test_ui_has_required_buttons():
    assert "Generate result" in HTML
    assert "Retrieve" in HTML
    assert "news" in HTML
