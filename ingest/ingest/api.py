"""HTTP API + web console for documents, models and retrieval settings,
plus the background folder watcher."""

from __future__ import annotations

import hmac
import logging
import mimetypes
import threading
from collections import Counter
from contextlib import asynccontextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ingest.config import Settings
from ingest.events import ChangeFeed, event_stream, parse_topics
from ingest.jobs import JobManager
from ingest.library import Library
from ingest.ollama_api import OllamaAPI, OllamaError, PullTracker, same_model
from ingest.pipeline import Ingestor
from ingest.workspace import Workspace
from ingest.workspace_api import make_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # per-request logs drown the useful ones
log = logging.getLogger("ingest")

settings = Settings.from_env()
ingestor = Ingestor(settings)
ollama = OllamaAPI(settings.ollama_url)
jobs = JobManager()
library = Library()
feed = ChangeFeed()
workspace = Workspace(settings.workspace_db, on_change=feed.bump)
STATIC_DIR = Path(__file__).parent / "static"

VERSION = "0.3.0"
# Contract version and optional features, for client discovery (docs/client-api.md).
API_VERSION = 2
FEATURES = ["stats", "documents_detail", "files", "events", "job_cancel", "model_suggest", "board", "conversations"]
# Routes that never require the API key: the console shell, liveness, and the schema docs.
OPEN_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json", "/favicon.ico"}


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


app = FastAPI(title="sh-rag ingest", version=VERSION, lifespan=lifespan)
app.include_router(make_router(workspace))


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    """Optional bearer auth (INGEST_API_KEY). Also accepts X-API-Key or ?api_key= so
    plain links (/files/...) and EventSource (/events) can authenticate."""
    key = settings.api_key
    if key and request.url.path not in OPEN_PATHS and request.method != "OPTIONS":
        auth = request.headers.get("authorization", "")
        supplied = auth[7:] if auth.lower().startswith("bearer ") else (
            request.headers.get("x-api-key") or request.query_params.get("api_key") or ""
        )
        if not hmac.compare_digest(supplied.strip(), key):
            return JSONResponse({"detail": "invalid or missing API key"}, status_code=401,
                                headers={"WWW-Authenticate": "Bearer"})
    return await call_next(request)


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
        "version": VERSION,
        "auth_required": bool(settings.api_key),
        "api": {"version": API_VERSION, "features": FEATURES},
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


@app.get("/stats")
def stats() -> dict:
    """Dashboard aggregates: document and chunk totals, by type, indexes, jobs, workspace."""
    try:
        out = ingestor.stats()
    except Exception as exc:
        raise HTTPException(502, f"could not read the index: {exc}")
    out["jobs_running"] = sum(1 for j in jobs.list() if j["status"] == "running")
    out["workspace"] = workspace.stats()
    out["updated_at"] = datetime.now(timezone.utc).isoformat()
    return out


@app.get("/events")
async def events(topics: str | None = None) -> StreamingResponse:
    """Server-Sent Events: job changes, board and conversation versions, pings."""
    try:
        wanted = parse_topics(topics)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return StreamingResponse(
        event_stream(jobs, workspace, feed, wanted),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- documents --------------------------------------------------------------

@app.get("/documents")
def list_documents() -> dict:
    return {"documents": ingestor.list_docs()}


@app.get("/documents/{key:path}")
def get_document(key: str, include_text: bool = True) -> dict:
    """One document with its passages. `key` is a doc_id or a source path."""
    try:
        doc = ingestor.get_doc(key, include_text=include_text)
    except Exception as exc:
        raise HTTPException(502, str(exc))
    if doc is None:
        raise HTTPException(404, f"no document with doc_id or source '{key}'")
    return doc


@app.get("/files/{source:path}")
def get_file(source: str) -> FileResponse:
    """The original file behind a source path, served inline for previews."""
    try:
        path = ingestor.file_for(source)
    except PermissionError as exc:
        raise HTTPException(403, str(exc))
    if path is None:
        raise HTTPException(404, f"no file '{source}' in the documents folder")
    media, _ = mimetypes.guess_type(path.name)
    return FileResponse(path, media_type=media or "application/octet-stream", filename=path.name,
                        content_disposition_type="inline")


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


@app.get("/models/suggest")
def suggest_models(q: str = "", kind: str | None = None) -> dict:
    """Autocomplete candidates from the public Ollama library. An empty query
    returns the most popular models. Degrades to an empty list when offline."""
    if kind not in (None, "chat", "embedding"):
        raise HTTPException(422, "kind must be 'chat' or 'embedding'")
    return library.search(q, kind)


@app.post("/models/pull")
def pull_model(body: ModelBody) -> dict:
    model = body.model.strip()

    def run(job):
        tracker = PullTracker()

        def on_event(ev):
            msg, frac = tracker.update(ev)
            job.report(msg, frac, completed=tracker.completed_bytes, total=tracker.total_bytes)

        # Ollama keeps partially downloaded layers, so a cancelled pull resumes
        # where it stopped when the same model is pulled again.
        ollama.pull(model, on_event)
        return {"model": model}

    return asdict(jobs.start("pull", f"pull:{model}", f"Pull {model}", run, detail={"model": model}))


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

    return asdict(jobs.start("embedding", f"embedding:{model}", f"Switch embedding to {model}", run,
                             detail={"model": model}))


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


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    """Ask a running job to stop. It ends at its next progress event; a cancelled
    pull keeps what was downloaded and resumes when started again."""
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    if job.status != "running":
        raise HTTPException(409, f"job is already {job.status}")
    job.cancel()
    return asdict(job)
