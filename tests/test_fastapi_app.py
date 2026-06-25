import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from app.agents import MarketInput
from app.main import app


@pytest.fixture
def client(monkeypatch, tmp_path):
    from app.api import routes

    monkeypatch.setattr(routes.settings, "store_path", str(tmp_path / "store.json"))
    monkeypatch.setattr(
        routes,
        "fetch_market_inputs",
        lambda: MarketInput(
            gold_price=3149,
            gold_change_pct=5,
            silver_price=61,
            silver_change_pct=7,
            dollar_index_change_pct=3,
            volatility_score=6,
        ),
    )
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_retrieve_endpoint_returns_news(client):
    response = client.get("/api/retrieve")
    assert response.status_code == 200
    assert response.json() == {"message": "news"}


def test_latest_market_inputs_endpoint(client):
    response = client.get("/api/latest-market-inputs")
    assert response.status_code == 200
    assert response.json()["gold_price"] == 3149


def test_generate_endpoint_runs_orchestrator(client):
    response = client.post("/api/generate")
    assert response.status_code == 200
    payload = response.json()
    assert payload["market_input"]["silver_price"] == 61
    assert "assessment" in payload
    assert "generated_task" in payload
    assert "evaluation" in payload
