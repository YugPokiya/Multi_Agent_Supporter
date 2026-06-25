from fastapi import APIRouter, HTTPException

from app.config import settings
from app.market_data import fetch_market_inputs
from app.memory import InMemoryStore
from app.orchestrator import SimpleOrchestrator
from app.schemas.market import MarketInputResponse
from app.schemas.result import GenerateResultResponse, HealthResponse, RetrieveResponse

router = APIRouter()


def build_orchestrator() -> SimpleOrchestrator:
    return SimpleOrchestrator(InMemoryStore(settings.store_path))


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
    )


@router.get("/api/retrieve", response_model=RetrieveResponse)
def retrieve() -> RetrieveResponse:
    return RetrieveResponse(message="news")


@router.get("/api/latest-market-inputs", response_model=MarketInputResponse)
def latest_market_inputs() -> MarketInputResponse:
    try:
        market_input = fetch_market_inputs()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return MarketInputResponse(**market_input.__dict__)


@router.post("/api/generate", response_model=GenerateResultResponse)
def generate_result() -> GenerateResultResponse:
    try:
        market_input = fetch_market_inputs()
        context = build_orchestrator().run(market_input)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return GenerateResultResponse(**context)
