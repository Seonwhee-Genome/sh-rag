"""Embeddings via the Ollama HTTP API."""

from __future__ import annotations

import logging
from typing import Callable

import httpx

from ingest.ollama_api import OllamaAPI

log = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, batch_size: int = 32, timeout: float = 300.0):
        self.model = model
        self.batch_size = max(1, batch_size)
        self.api = OllamaAPI(base_url)
        self._client = httpx.Client(base_url=base_url, timeout=timeout)

    def ensure_model(self, on_progress: Callable[[dict], None] | None = None) -> None:
        """Pull the embedding model if it is not installed yet."""
        if self.api.has_model(self.model):
            return
        log.info("pulling embedding model %s (first run only, may take a while)", self.model)
        self.api.pull(self.model, on_progress)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            resp = self._client.post("/api/embed", json={"model": self.model, "input": batch})
            if resp.status_code == 404:
                raise EmbeddingError(
                    f"embedding model '{self.model}' not found in Ollama; run: ollama pull {self.model}"
                )
            if resp.status_code >= 400:
                raise EmbeddingError(f"'{self.model}' cannot embed: {resp.text.strip()[:300]}")
            got = resp.json().get("embeddings") or []
            if len(got) != len(batch):
                raise EmbeddingError(f"expected {len(batch)} embeddings, got {len(got)}")
            vectors.extend(got)
        return vectors

    def dimension(self) -> int:
        return len(self.embed(["dimension probe"])[0])
