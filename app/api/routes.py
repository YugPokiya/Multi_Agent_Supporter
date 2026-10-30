from fastapi import APIRouter, HTTPException

from app.config import settings
from app.market_data import fetch_market_inputs
from app.orderflow import OrderFlowInput, build_orderflow_plan
from app.memory import InMemoryStore
from app.orchestrator import SimpleOrchestrator
from app.schemas.market import (
    MarketInputResponse,
    OrderFlowInputRequest,
    OrderFlowPlanResponse,
)
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


@router.post("/api/orderflow-plan", response_model=OrderFlowPlanResponse)
def orderflow_plan(payload: OrderFlowInputRequest) -> OrderFlowPlanResponse:
    try:
        flow = OrderFlowInput(
            instrument=payload.instrument.upper(),
            last_price=payload.last_price,
            session=payload.session,
            cumulative_delta=payload.cumulative_delta,
            delta_change=payload.delta_change,
            volume_ratio=payload.volume_ratio,
            bid_ask_imbalance=payload.bid_ask_imbalance,
            absorption_score=payload.absorption_score,
            trapped_trader_score=payload.trapped_trader_score,
            structure_break=payload.structure_break.lower(),
            liquidity_sweep=payload.liquidity_sweep.lower(),
            atr_5m=payload.atr_5m,
        )
        plan = build_orderflow_plan(flow)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return OrderFlowPlanResponse(**plan.__dict__)
