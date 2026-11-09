"""HTTP API for uploads and index management, plus the background folder watcher."""

from __future__ import annotations

import logging
import threading
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, File, HTTPException, UploadFile

from ingest.config import Settings
from ingest.pipeline import Ingestor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # per-request logs drown the useful ones
log = logging.getLogger("ingest")

settings = Settings.from_env()
ingestor = Ingestor(settings)


def _summary(results) -> dict:
    return {"summary": dict(Counter(r.status for r in results)), "results": [asdict(r) for r in results]}


def _watch(stop: threading.Event) -> None:
    """Poll the docs folder. Polling (not filesystem events) works reliably on
    Docker bind mounts from both Windows and macOS hosts."""
    while not stop.is_set():
        try:
            changed = [r for r in ingestor.scan() if r.status != "unchanged"]
            if changed:
                log.info("scan: %s", dict(Counter(r.status for r in changed)))
        except Exception as exc:
            log.warning("scan failed (will retry in %ss): %s", settings.scan_interval, exc)
        stop.wait(settings.scan_interval)


@asynccontextmanager
async def lifespan(_: FastAPI):
    stop = threading.Event()
    if settings.scan_interval > 0:
        threading.Thread(target=_watch, args=(stop,), name="docs-watcher", daemon=True).start()
        log.info("watching %s every %ss", settings.docs_dir, settings.scan_interval)
    yield
    stop.set()


app = FastAPI(title="sh-rag ingest", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "collection": settings.collection, "embed_model": settings.embed_model}


@app.get("/documents")
def list_documents() -> dict:
    return {"documents": ingestor.list_docs()}


@app.post("/documents")
def upload_documents(files: list[UploadFile] = File(...)) -> dict:
    results = []
    for f in files:
        try:
            results.append(ingestor.save_upload(f.filename or "", f.file.read()))
        finally:
            f.file.close()
    return _summary(results)


@app.delete("/documents/{key:path}")
def delete_document(key: str) -> dict:
    result = ingestor.remove(key)
    if result is None:
        raise HTTPException(404, f"no document with doc_id or source '{key}'")
    return asdict(result)


@app.post("/reindex")
def reindex(force: bool = False) -> dict:
    return _summary(ingestor.scan(force=force))
