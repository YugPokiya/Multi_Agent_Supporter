import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class InMemoryStore:
    """Simple runtime memory store with JSON persistence."""

    def __init__(self, persist_path: str = "store.json") -> None:
        self.persist_path = Path(persist_path)
        self.data: dict[str, Any] = {
            "sessions": [],
            "latest_context": None,
        }

    def save_context(self, context: dict[str, Any]) -> None:
        """Save the latest context in memory and append it to session history."""
        context = {
            **context,
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }
        self.data["latest_context"] = context
        self.data["sessions"].append(context)
        self.persist()

    def persist(self) -> None:
        """Write the full in-memory state to disk."""
        self.persist_path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
