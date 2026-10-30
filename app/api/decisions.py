from functools import lru_cache

from fastapi import APIRouter, HTTPException, Query

from app.decision.service import DecisionService
from app.ml.model import load_bundle
from app.schemas.decision import DecisionBatchResponse, DecisionRequest, Instrument, TrainRequest

router = APIRouter(prefix="/api", tags=["decisions"])


@lru_cache
def get_service() -> DecisionService:
    return DecisionService()


@router.post("/decisions", response_model=DecisionBatchResponse)
def generate_decisions(req: DecisionRequest) -> DecisionBatchResponse:
    """Run the multi-agent decision graph for every requested instrument."""
    orderflow = {k: v.model_dump() for k, v in req.orderflow.items()}
    decisions = get_service().decide_many(list(dict.fromkeys(req.instruments)), orderflow)
    summary = {d["instrument"]: d.get("final_action", "ERROR") for d in decisions}
    return DecisionBatchResponse(decisions=decisions, summary=summary)


@router.get("/decisions/history")
def decision_history(instrument: Instrument | None = None, limit: int = Query(50, ge=1, le=500)) -> list[dict]:
    return get_service().store.history(instrument, limit)


@router.post("/models/train")
def train_models(req: TrainRequest) -> list[dict]:
    svc, out = get_service(), []
    for inst in dict.fromkeys(req.instruments):
        try:
            out.append(svc.train(inst).summary())
        except Exception as exc:
            out.append({"instrument": inst, "error": str(exc)})
    return out


@router.get("/models")
def list_models() -> list[dict]:
    svc = get_service()
    return [b.summary() for inst in ("XAU", "XAG", "BTC", "NIFTY", "CRUDE")
            if (b := load_bundle(inst, svc.cfg.model_dir)) is not None]


@router.get("/decisions/evaluate/{instrument}")
def evaluate(instrument: Instrument) -> dict:
    """Backtest the whole agent graph over the model's hold-out period."""
    try:
        return get_service().evaluate(instrument)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
