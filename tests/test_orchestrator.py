import json

from app.agents import AssessorAgent, MarketInput
from app.memory import InMemoryStore
from app.orchestrator import SimpleOrchestrator


def test_assessor_detects_bullish_precious_metals():
    market = MarketInput(2350, 1.2, 31, 1.0, -0.4, 4)
    result = AssessorAgent().assess(market)
    assert result["regime"] == "bullish_precious_metals"


def test_orchestrator_stores_context_to_json(tmp_path):
    store_path = tmp_path / "store.json"
    store = InMemoryStore(str(store_path))
    market = MarketInput(2300, -1.1, 28, -0.9, 0.5, 5)

    context = SimpleOrchestrator(store).run(market)

    assert context["assessment"]["regime"] == "bearish_precious_metals"
    assert context["evaluation"]["verdict"] in {"approved", "needs_revision", "rejected"}
    saved = json.loads(store_path.read_text(encoding="utf-8"))
    assert saved["latest_context"]["assessment"]["regime"] == "bearish_precious_metals"
    assert len(saved["sessions"]) == 1
