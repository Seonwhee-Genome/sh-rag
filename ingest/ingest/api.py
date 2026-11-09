"""HTTP API + web console for documents, models and retrieval settings,
plus the background folder watcher."""

from __future__ import annotations

import logging
import threading
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ingest.config import Settings
from ingest.jobs import JobManager
from ingest.ollama_api import OllamaAPI, OllamaError, PullTracker, same_model
from ingest.pipeline import Ingestor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # per-request logs drown the useful ones
log = logging.getLogger("ingest")

settings = Settings.from_env()
ingestor = Ingestor(settings)
ollama = OllamaAPI(settings.ollama_url)
jobs = JobManager()
STATIC_DIR = Path(__file__).parent / "static"


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
    log.info("console: http://localhost:%s/", settings.port)
    yield
    stop.set()


app = FastAPI(title="sh-rag ingest", version="0.2.0", lifespan=lifespan)


class ModelBody(BaseModel):
    model: str = Field(min_length=1)


class SettingsBody(BaseModel):
    top_k: int | None = None
    score_threshold: float | None = None


class SearchBody(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = None
    score_threshold: float | None = None


# --- console & status -------------------------------------------------------

@app.get("/", include_in_schema=False)
def console() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/status")
def status() -> dict:
    out: dict = {
        "embed_model": ingestor.index.embed_model,
        "collection": ingestor.index.collection,
        "docs_path": settings.host_docs_path or str(settings.docs_dir),
        "scan_interval": settings.scan_interval,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "webui_url": settings.webui_url,
    }
    try:
        out["ollama"] = {"ok": True, "version": ollama.version(), "url": settings.ollama_url}
    except Exception as exc:
        out["ollama"] = {"ok": False, "error": str(exc), "url": settings.ollama_url}
    try:
        ingestor.index.store.ping()
        out["qdrant"] = {"ok": True}
    except Exception as exc:
        out["qdrant"] = {"ok": False, "error": str(exc)}
    return out


# --- documents --------------------------------------------------------------

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
def reindex(force: bool = False, background: bool = False) -> dict:
    if not background:
        return _summary(ingestor.scan(force=force))

    def run(job):
        results = ingestor.scan(
            force=force, on_file=lambda i, n, src: job.report(f"{i}/{n}: {src}", (i - 1) / n if n else None)
        )
        return _summary(results)["summary"]

    title = "Re-embed all documents" if force else "Sync documents folder"
    return asdict(jobs.start("reindex", "reindex", title, run))


# --- models -----------------------------------------------------------------

@app.get("/models")
def list_models() -> dict:
    try:
        models = ollama.list_models()
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Ollama is not reachable at {settings.ollama_url}: {exc}")
    return {"models": models, "active_embed_model": ingestor.index.embed_model}


@app.post("/models/pull")
def pull_model(body: ModelBody) -> dict:
    model = body.model.strip()

    def run(job):
        tracker = PullTracker()
        ollama.pull(model, lambda ev: job.report(*tracker.update(ev)))
        return {"model": model}

    return asdict(jobs.start("pull", f"pull:{model}", f"Pull {model}", run))


@app.delete("/models/{name:path}")
def delete_model(name: str) -> dict:
    if same_model(name, ingestor.index.embed_model):
        raise HTTPException(409, "this is the active embedding model; switch to another one first")
    try:
        ollama.delete(name)
    except OllamaError as exc:
        raise HTTPException(404, str(exc))
    except httpx.HTTPError as exc:
        raise HTTPException(502, str(exc))
    return {"deleted": name}


@app.post("/embedding")
def switch_embedding(body: ModelBody) -> dict:
    model = body.model.strip()
    if same_model(model, ingestor.index.embed_model):
        raise HTTPException(400, f"{model} is already the active embedding model")
    running = jobs.running("embedding")
    if running and running.key != f"embedding:{model}":
        raise HTTPException(409, f"another switch is in progress: {running.title}")

    def run(job):
        results = ingestor.switch_embedding(model, on_status=job.report)
        return dict(Counter(r.status for r in results))

    return asdict(jobs.start("embedding", f"embedding:{model}", f"Switch embedding to {model}", run))


# --- retrieval settings -----------------------------------------------------

@app.get("/settings")
def get_settings() -> dict:
    return asdict(ingestor.runtime.get())


@app.put("/settings")
def put_settings(body: SettingsBody) -> dict:
    try:
        return asdict(ingestor.runtime.update(top_k=body.top_k, score_threshold=body.score_threshold))
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.post("/search")
def search(body: SearchBody) -> dict:
    try:
        hits = ingestor.search(body.query, body.top_k, body.score_threshold)
    except Exception as exc:
        raise HTTPException(502, str(exc))
    return {"results": hits}


# --- jobs -------------------------------------------------------------------

@app.get("/jobs")
def list_jobs() -> dict:
    return {"jobs": jobs.list()}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return asdict(job)
