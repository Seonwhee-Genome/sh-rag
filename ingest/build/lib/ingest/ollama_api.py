"""Ollama model management: list, inspect, pull (with progress), delete."""

from __future__ import annotations

import json
from typing import Callable

import httpx

EMBED_NAME_HINTS = ("embed", "bge", "minilm", "e5-", "gte-", "granite-embedding")
EMBED_FAMILIES = {"bert", "nomic-bert", "xlm-roberta", "xlmroberta"}


class OllamaError(RuntimeError):
    pass


def model_kind(model: dict, capabilities: list[str] | None) -> str:
    """'embedding' or 'chat'. Uses Ollama's reported capabilities when available."""
    if capabilities:
        return "embedding" if "embedding" in capabilities and "completion" not in capabilities else "chat"
    name = (model.get("name") or "").lower()
    details = model.get("details") or {}
    families = {str(f).lower() for f in [details.get("family"), *(details.get("families") or [])] if f}
    if any(h in name for h in EMBED_NAME_HINTS) or families & EMBED_FAMILIES:
        return "embedding"
    return "chat"


def same_model(a: str, b: str) -> bool:
    norm = lambda s: s.strip().lower().removesuffix(":latest")
    return norm(a) == norm(b)


class PullTracker:
    """Turns Ollama's per-layer pull events into one overall fraction."""

    def __init__(self):
        self.total: dict[str, int] = {}
        self.done: dict[str, int] = {}

    def update(self, event: dict) -> tuple[str, float | None]:
        digest = event.get("digest")
        if digest and event.get("total"):
            self.total[digest] = event["total"]
            self.done[digest] = event.get("completed", 0)
        total = sum(self.total.values())
        return event.get("status", ""), (sum(self.done.values()) / total if total else None)

    @property
    def completed_bytes(self) -> int:
        return sum(self.done.values())

    @property
    def total_bytes(self) -> int:
        return sum(self.total.values())


class OllamaAPI:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self._client = httpx.Client(base_url=base_url, timeout=timeout)
        self._caps: dict[str, list[str] | None] = {}

    def version(self) -> str:
        resp = self._client.get("/api/version", timeout=3)
        resp.raise_for_status()
        return resp.json().get("version", "")

    def installed(self) -> list[dict]:
        resp = self._client.get("/api/tags")
        resp.raise_for_status()
        return resp.json().get("models", [])

    def has_model(self, model: str) -> bool:
        return any(same_model(m.get("name", ""), model) for m in self.installed())

    def capabilities(self, name: str, digest: str | None = None) -> list[str] | None:
        key = digest or name
        if key not in self._caps:
            try:
                resp = self._client.post("/api/show", json={"model": name, "name": name})
                resp.raise_for_status()
                self._caps[key] = resp.json().get("capabilities")
            except httpx.HTTPError:
                return None  # don't cache transient failures
        return self._caps[key]

    def list_models(self) -> list[dict]:
        out = []
        for m in self.installed():
            caps = self.capabilities(m["name"], m.get("digest"))
            details = m.get("details") or {}
            out.append({
                "name": m["name"],
                "size": m.get("size", 0),
                "modified_at": m.get("modified_at"),
                "family": details.get("family"),
                "parameter_size": details.get("parameter_size"),
                "quantization": details.get("quantization_level"),
                "capabilities": caps or [],
                "kind": model_kind(m, caps),
            })
        return sorted(out, key=lambda x: x["name"])

    def pull(self, model: str, on_progress: Callable[[dict], None] | None = None) -> None:
        body = {"model": model, "name": model, "stream": True}
        with self._client.stream("POST", "/api/pull", json=body, timeout=httpx.Timeout(30.0, read=None)) as resp:
            if resp.status_code >= 400:
                resp.read()
                raise OllamaError(f"pull {model} failed: {resp.text.strip()}")
            for line in resp.iter_lines():
                if not line.strip():
                    continue
                event = json.loads(line)
                if event.get("error"):
                    raise OllamaError(f"pull {model} failed: {event['error']}")
                if on_progress:
                    on_progress(event)

    def delete(self, model: str) -> None:
        resp = self._client.request("DELETE", "/api/delete", json={"model": model, "name": model})
        if resp.status_code == 404:
            raise OllamaError(f"model '{model}' is not installed")
        resp.raise_for_status()
