from typing import Any

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class RetrieveResponse(BaseModel):
    message: str


class GenerateResultResponse(BaseModel):
    market_input: dict[str, Any]
    assessment: dict[str, Any]
    generated_task: dict[str, Any]
    evaluation: dict[str, Any]
    stored_at: str | None = None
