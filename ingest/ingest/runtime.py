"""Settings the console changes at runtime, shared with the Go gateway via a JSON file."""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Runtime:
    embed_model: str
    top_k: int = 5
    score_threshold: float = 0.35

    def validate(self) -> None:
        if not self.embed_model.strip():
            raise ValueError("embed_model must not be empty")
        if not 1 <= self.top_k <= 50:
            raise ValueError("top_k must be between 1 and 50")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score_threshold must be between 0 and 1")


def collection_name(base: str, model: str) -> str:
    """Each embedding model gets its own collection, e.g. documents_bge_m3.

    Must match CollectionName in gateway/internal/settings/settings.go.
    """
    m = model.strip().lower()
    if m.endswith(":latest"):
        m = m[: -len(":latest")]
    return f"{base}_{re.sub(r'[^a-z0-9]+', '_', m).strip('_')}"


class RuntimeStore:
    """Reads/writes the runtime settings file; values missing from it fall back to env defaults."""

    def __init__(self, path: Path | None, defaults: Runtime):
        self.path = Path(path) if path else None
        self.defaults = defaults
        self._lock = threading.Lock()
        self._memory: dict = {}

    def _read(self) -> dict:
        if self.path is None:
            return dict(self._memory)
        try:
            return json.loads(self.path.read_text("utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            log.warning("ignoring unreadable runtime settings %s: %s", self.path, exc)
            return {}

    def get(self) -> Runtime:
        data, d = self._read(), self.defaults
        try:
            return Runtime(
                embed_model=str(data.get("embed_model") or d.embed_model).strip(),
                top_k=int(data.get("top_k") or d.top_k),
                score_threshold=float(d.score_threshold if data.get("score_threshold") is None
                                      else data["score_threshold"]),
            )
        except (TypeError, ValueError):
            return d

    def update(self, **changes) -> Runtime:
        with self._lock:
            merged = asdict(self.get()) | {k: v for k, v in changes.items() if v is not None}
            new = Runtime(**merged)
            new.validate()
            payload = asdict(new) | {"updated_at": datetime.now(timezone.utc).isoformat()}
            if self.path is None:
                self._memory = payload
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                tmp = self.path.with_name(self.path.name + ".tmp")
                tmp.write_text(json.dumps(payload, indent=2), "utf-8")
                os.replace(tmp, self.path)  # atomic: the gateway never sees a half-written file
            return new
