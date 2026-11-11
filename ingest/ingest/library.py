"""Model suggestions from the public Ollama library (ollama.com/search).

ollama.com has no JSON search API, so this parses the server-rendered search
page: one <li> per model with its name, capability badges (embedding, vision,
tools, ...) and size badges (0.6b, 8b, ...). Results are cached, and any
failure (offline, page layout changed) degrades to an empty list so the
console can fall back to its built-in suggestions.
"""

from __future__ import annotations

import html
import logging
import re
import threading
import time

import httpx

from ingest.ollama_api import model_kind

log = logging.getLogger(__name__)

SEARCH_URL = "https://ollama.com/search"
CAPABILITIES = {"embedding", "vision", "tools", "thinking", "cloud"}
MAX_RESULTS = 15

_ENTRY = re.compile(r'<a href="/library/([A-Za-z0-9][A-Za-z0-9._-]*)"[^>]*>(.*?)</a>\s*</li>', re.S)
# Note the \s+: the live page emits `<span  class=` with two spaces.
_BADGE = re.compile(r'<span\s+class="[^"]*\brounded-(?:md|full)\b[^"]*"[^>]*>\s*([^<]+?)\s*</span>')
_DESC = re.compile(r"<p[^>]*>(.*?)</p>", re.S)
_SIZE = re.compile(r"^\d+(\.\d+)?[bm]$", re.I)
_TAG = re.compile(r"<[^>]+>")


def parse_search(page: str) -> list[dict]:
    """Extract models from an ollama.com search page, in the page's order."""
    out = []
    for name, body in _ENTRY.findall(page):
        badges = [html.unescape(b).strip() for b in _BADGE.findall(body)]
        caps = sorted({b.lower() for b in badges if b.lower() in CAPABILITIES})
        sizes = [b.lower() for b in badges if _SIZE.match(b)]
        if "cloud" in caps and not sizes:
            continue  # cloud-only models can't be pulled for local use
        m = _DESC.search(body)
        desc = html.unescape(_TAG.sub("", m.group(1))).strip() if m else ""
        out.append({
            "name": name,
            "kind": model_kind({"name": name}, caps or None),
            "capabilities": caps,
            "sizes": sizes,
            "description": re.sub(r"\s+", " ", desc)[:200],
        })
    return out


class Library:
    def __init__(self, ttl: float = 600.0, timeout: float = 6.0):
        self.ttl = ttl
        self._cache: dict[tuple[str, str | None], tuple[float, dict]] = {}
        self._lock = threading.Lock()
        self._client = httpx.Client(
            timeout=timeout, follow_redirects=True, headers={"User-Agent": "sh-rag-console (+https://github.com)"}
        )

    def search(self, query: str, kind: str | None = None) -> dict:
        """Models matching `query`; `kind` ('chat' | 'embedding') narrows the results.
        Never raises: returns {"models": [], "error": "..."} when ollama.com is unreachable."""
        q = query.strip().lower()
        key = (q, kind)
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(key)
            if hit and hit[0] > now:
                return hit[1]
        params = {"q": q}
        if kind == "embedding":
            params["c"] = "embedding"  # ollama.com's own category filter
        try:
            resp = self._client.get(SEARCH_URL, params=params)
            resp.raise_for_status()
            models = parse_search(resp.text)
            if kind:
                models = [m for m in models if m["kind"] == kind]
            result, ttl = {"models": models[:MAX_RESULTS], "error": None}, self.ttl
        except httpx.HTTPError as exc:
            log.info("ollama.com search failed: %s", exc)
            result, ttl = {"models": [], "error": f"ollama.com is not reachable: {exc}"}, 60.0
        with self._lock:
            if len(self._cache) > 300:
                self._cache.clear()
            self._cache[key] = (now + ttl, result)
        return result
