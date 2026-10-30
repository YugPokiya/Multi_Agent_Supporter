"""Decision history in SQL (SQLite by default; set DATABASE_URL for PostgreSQL)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Integer, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class DecisionRow(Base):
    __tablename__ = "decisions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    instrument: Mapped[str] = mapped_column(String(16), index=True)
    as_of: Mapped[str] = mapped_column(String(10))
    final_action: Mapped[str] = mapped_column(String(4))
    raw_action: Mapped[str] = mapped_column(String(4))
    score: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    regime: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict] = mapped_column(JSON)


def to_jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


class DecisionStore:
    def __init__(self, url: str) -> None:
        self.engine = create_engine(url, future=True)
        Base.metadata.create_all(self.engine)

    def save(self, decision: dict) -> int:
        with Session(self.engine) as s:
            row = DecisionRow(
                created_at=datetime.now(timezone.utc),
                instrument=decision["instrument"],
                as_of=decision["as_of"],
                final_action=decision["final_action"],
                raw_action=decision["aggregate"]["action"],
                score=decision["aggregate"]["score"],
                confidence=decision["aggregate"]["confidence"],
                regime=decision["regime"],
                payload=to_jsonable(decision),
            )
            s.add(row)
            s.commit()
            return row.id

    def history(self, instrument: str | None = None, limit: int = 50) -> list[dict]:
        with Session(self.engine) as s:
            q = select(DecisionRow).order_by(DecisionRow.id.desc()).limit(limit)
            if instrument:
                q = q.where(DecisionRow.instrument == instrument)
            return [
                {"id": r.id, "created_at": r.created_at.isoformat(), "instrument": r.instrument, "as_of": r.as_of,
                 "final_action": r.final_action, "raw_action": r.raw_action, "score": r.score,
                 "confidence": r.confidence, "regime": r.regime}
                for r in s.scalars(q)
            ]
