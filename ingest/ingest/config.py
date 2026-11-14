"""Settings loaded from environment variables (shared names with the gateway)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


@dataclass(frozen=True)
class Settings:
    ollama_url: str = "http://localhost:11434"
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    collection: str = "documents"
    embed_model: str = "bge-m3"
    embed_batch_size: int = 32
    chunk_size: int = 1000
    chunk_overlap: int = 150
    docs_dir: Path = Path("./data/docs")
    scan_interval: int = 30
    host: str = "0.0.0.0"
    port: int = 8001
    # Defaults for settings the console can change at runtime (see runtime.py).
    top_k: int = 5
    score_threshold: float = 0.35
    runtime_path: Path | None = Path("data/config/settings.json")
    # Display-only values for the console.
    host_docs_path: str = ""
    webui_url: str = "http://localhost:3000"
    # Optional bearer token for the API (empty = open; fine on localhost only).
    api_key: str = ""
    # SQLite file for the board and conversations (client workspace).
    workspace_db: Path | None = Path("data/workspace/workspace.db")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            top_k=int(_env("RAG_TOP_K", str(cls.top_k))),
            score_threshold=float(_env("RAG_SCORE_THRESHOLD", str(cls.score_threshold))),
            runtime_path=Path(_env("RUNTIME_SETTINGS_PATH", str(cls.runtime_path))),
            host_docs_path=_env("HOST_DOCS_PATH", ""),
            webui_url=_env("WEBUI_URL", cls.webui_url),
            ollama_url=_env("OLLAMA_BASE_URL", cls.ollama_url).rstrip("/"),
            qdrant_url=_env("QDRANT_URL", cls.qdrant_url).rstrip("/"),
            qdrant_api_key=os.environ.get("QDRANT_API_KEY") or None,
            collection=_env("QDRANT_COLLECTION", cls.collection),
            embed_model=_env("EMBED_MODEL", cls.embed_model),
            embed_batch_size=int(_env("EMBED_BATCH_SIZE", str(cls.embed_batch_size))),
            chunk_size=int(_env("CHUNK_SIZE", str(cls.chunk_size))),
            chunk_overlap=int(_env("CHUNK_OVERLAP", str(cls.chunk_overlap))),
            docs_dir=Path(_env("DOCS_DIR", str(cls.docs_dir))),
            scan_interval=int(_env("SCAN_INTERVAL", str(cls.scan_interval))),
            host=_env("INGEST_HOST", cls.host),
            port=int(_env("INGEST_PORT", str(cls.port))),
            api_key=_env("INGEST_API_KEY", ""),
            workspace_db=Path(_env("WORKSPACE_DB", str(cls.workspace_db))),
        )
