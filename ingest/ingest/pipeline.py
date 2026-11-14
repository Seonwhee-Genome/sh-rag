"""Load → chunk → embed → upsert, with change detection, deletion sync and
switchable embedding models (one Qdrant collection per model)."""

from __future__ import annotations

import hashlib
import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from qdrant_client.http import models as qm

from ingest.chunker import split_text
from ingest.config import Settings
from ingest.embedder import OllamaEmbedder
from ingest.loaders import is_supported, load
from ingest.ollama_api import PullTracker
from ingest.runtime import Runtime, RuntimeStore, collection_name
from ingest.store import QdrantStore

log = logging.getLogger(__name__)

# Stable namespace so a given source path always maps to the same doc_id.
_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://sh-rag.local/doc")
UPLOADS_DIR = "uploads"

StatusFn = Callable[[str, "float | None"], None]


@dataclass
class IngestResult:
    source: str
    status: str  # indexed | unchanged | empty | unsupported | removed | error
    doc_id: str = ""
    chunks: int = 0
    error: str | None = None


def doc_id_for(source: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, source))


def _doc_type(source: str) -> str:
    """Lower-cased extension without the dot, or 'other' (matches the API contract)."""
    ext = Path(source).suffix.lower().lstrip(".")
    return ext or "other"


class Index:
    """One embedding model and the Qdrant collection holding its vectors."""

    def __init__(self, settings: Settings, embed_model: str, embedder=None, store=None):
        self.embed_model = embed_model
        self.collection = collection_name(settings.collection, embed_model)
        self.embedder = embedder or OllamaEmbedder(settings.ollama_url, embed_model, settings.embed_batch_size)
        self.store = store or QdrantStore(settings.qdrant_url, self.collection, settings.qdrant_api_key)
        self.lock = threading.RLock()
        self.ready = False
        # source -> (mtime_ns, size) of the last successful ingest; lets the
        # periodic scan skip unchanged files without hashing or querying Qdrant.
        self.stat_cache: dict[str, tuple[int, int]] = {}

    def setup(self, recreate: bool = False, on_progress: Callable[[dict], None] | None = None) -> None:
        with self.lock:
            self.embedder.ensure_model(on_progress=on_progress)
            self.store.ensure_collection(self.embedder.dimension(), recreate=recreate)
            if recreate:
                self.stat_cache.clear()
            self.ready = True

    def ensure_ready(self) -> None:
        if not self.ready:
            self.setup()


class Ingestor:
    def __init__(self, settings: Settings, embedder=None, store=None,
                 runtime: RuntimeStore | None = None,
                 index_factory: Callable[[str], Index] | None = None):
        self.settings = settings
        self.runtime = runtime or RuntimeStore(
            settings.runtime_path, Runtime(settings.embed_model, settings.top_k, settings.score_threshold)
        )
        self._factory = index_factory or (lambda model: Index(settings, model))
        model = self.runtime.get().embed_model
        first = Index(settings, model, embedder, store) if (embedder or store) else self._factory(model)
        self._indexes: dict[str, Index] = {model: first}
        self._active = first
        self._active_lock = threading.Lock()
        self._switch_lock = threading.Lock()

    @property
    def docs_dir(self) -> Path:
        return self.settings.docs_dir

    @property
    def index(self) -> Index:
        """The active index. Follows the runtime settings file, so a switch made
        by another process (CLI) or a hand edit is picked up too."""
        with self._active_lock:
            model = self.runtime.get().embed_model
            if self._active.embed_model != model:
                log.info("embedding model is now %s", model)
                self._active = self._get_index(model)
            return self._active

    def _get_index(self, model: str) -> Index:
        if model not in self._indexes:
            self._indexes[model] = self._factory(model)
        return self._indexes[model]

    def setup(self, recreate: bool = False) -> None:
        self.index.setup(recreate=recreate)

    def source_for(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.docs_dir.resolve()).as_posix()
        except ValueError:
            return path.name

    def ingest_file(self, path: Path, force: bool = False, idx: Index | None = None) -> IngestResult:
        idx = idx or self.index
        source = self.source_for(path)
        doc_id = doc_id_for(source)
        if not is_supported(path):
            return IngestResult(source, "unsupported", doc_id)

        with idx.lock:
            idx.ensure_ready()
            stat = path.stat()
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if not force and idx.store.doc_hash(doc_id) == content_hash:
                idx.stat_cache[source] = (stat.st_mtime_ns, stat.st_size)
                return IngestResult(source, "unchanged", doc_id)

            records: list[tuple[str, int | None]] = []
            for section in load(path):
                for chunk in split_text(section.text, self.settings.chunk_size, self.settings.chunk_overlap):
                    records.append((chunk, section.page))

            if not records:
                idx.store.delete_doc(doc_id)
                idx.stat_cache[source] = (stat.st_mtime_ns, stat.st_size)
                return IngestResult(source, "empty", doc_id)

            title = path.stem
            # Prefixing the title gives each chunk document-level context in its embedding.
            vectors = idx.embedder.embed([f"{title}\n\n{text}" for text, _ in records])
            now = datetime.now(timezone.utc).isoformat()
            base = uuid.UUID(doc_id)
            points = [
                qm.PointStruct(
                    id=str(uuid.uuid5(base, str(i))),
                    vector=vector,
                    payload={
                        "doc_id": doc_id,
                        "source": source,
                        "title": title,
                        "page": page,
                        "chunk_index": i,
                        "chunks": len(records),
                        "text": text,
                        "content_hash": content_hash,
                        "ingested_at": now,
                    },
                )
                for i, ((text, page), vector) in enumerate(zip(records, vectors))
            ]
            # Embed first, then swap: the old version stays searchable until the new one is ready.
            idx.store.delete_doc(doc_id)
            idx.store.upsert(points)
            idx.stat_cache[source] = (stat.st_mtime_ns, stat.st_size)
            log.info("indexed %s (%d chunks) into %s", source, len(points), idx.collection)
            return IngestResult(source, "indexed", doc_id, len(points))

    def iter_files(self):
        root = self.docs_dir
        if not root.exists():
            root.mkdir(parents=True, exist_ok=True)
        for path in sorted(root.rglob("*")):
            rel_parts = path.relative_to(root).parts
            if path.is_file() and is_supported(path) and not any(p.startswith(".") for p in rel_parts):
                yield path

    def scan(self, force: bool = False, prune: bool = True, idx: Index | None = None,
             on_file: Callable[[int, int, str], None] | None = None) -> list[IngestResult]:
        """Sync the docs folder into an index: add/update changed files, drop deleted ones."""
        idx = idx or self.index
        with idx.lock:
            idx.ensure_ready()
            files = list(self.iter_files())
            results: list[IngestResult] = []
            seen: set[str] = set()
            for n, path in enumerate(files, 1):
                source = self.source_for(path)
                seen.add(source)
                try:
                    st = path.stat()
                    if not force and idx.stat_cache.get(source) == (st.st_mtime_ns, st.st_size):
                        continue  # unchanged since last scan
                    if on_file:
                        on_file(n, len(files), source)
                    results.append(self.ingest_file(path, force=force, idx=idx))
                except Exception as exc:  # keep going: one bad file must not stop the scan
                    log.exception("failed to ingest %s", source)
                    results.append(IngestResult(source, "error", doc_id_for(source), error=str(exc)))
            if prune:
                for doc in idx.store.list_docs():
                    if doc.get("source") not in seen:
                        idx.store.delete_doc(doc["doc_id"])
                        idx.stat_cache.pop(doc.get("source"), None)
                        results.append(IngestResult(doc.get("source", ""), "removed", doc["doc_id"]))
            return results

    def switch_embedding(self, model: str, on_status: StatusFn | None = None) -> list[IngestResult]:
        """Build (or refresh) the index for `model`, then make it active.

        Chat keeps using the current index until the new one is complete, and
        the old index is kept, so switching back later only re-checks files.
        """
        model = model.strip()
        if not model:
            raise ValueError("model name is required")
        report = on_status or (lambda msg, frac=None: None)
        with self._switch_lock:
            if model == self.index.embed_model:
                return []
            with self._active_lock:
                idx = self._get_index(model)
            tracker = PullTracker()
            report(f"Preparing {model}…", None)
            idx.setup(on_progress=lambda ev: report(*tracker.update(ev)))
            results = self.scan(
                idx=idx, on_file=lambda i, n, src: report(f"Embedding {i}/{n}: {src}", (i - 1) / n if n else None)
            )
            with self._active_lock:
                self.runtime.update(embed_model=model)
                self._active = idx
            report(f"Now using {model}", 1.0)
            return results

    def search(self, query: str, top_k: int | None = None, threshold: float | None = None) -> list[dict]:
        rt = self.runtime.get()
        idx = self.index
        idx.ensure_ready()
        vector = idx.embedder.embed([query])[0]
        return idx.store.search(vector, top_k or rt.top_k, rt.score_threshold if threshold is None else threshold)

    def save_upload(self, filename: str, data: bytes) -> IngestResult:
        name = Path(filename.replace("\\", "/")).name.strip()
        if not name or name.startswith("."):
            return IngestResult(filename, "unsupported", error="invalid file name")
        dest = self.docs_dir / UPLOADS_DIR / name
        if not is_supported(dest):
            return IngestResult(name, "unsupported", error=f"unsupported file type: {dest.suffix or 'none'}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return self.ingest_file(dest, force=True)

    def list_docs(self) -> list[dict]:
        idx = self.index
        with idx.lock:
            return sorted(idx.store.list_docs(), key=lambda d: d.get("source", ""))

    def find_doc(self, key: str) -> dict | None:
        """One document by doc_id or source path, or None."""
        return next((d for d in self.list_docs() if key in (d.get("doc_id"), d.get("source"))), None)

    def file_for(self, source: str) -> Path | None:
        """The original file for a source path, if it is still inside the docs folder."""
        if not source:
            return None
        path = (self.docs_dir / source).resolve()
        if not path.is_relative_to(self.docs_dir.resolve()):
            raise PermissionError(f"'{source}' is outside the documents folder")
        return path if path.is_file() else None

    def get_doc(self, key: str, include_text: bool = True) -> dict | None:
        """Document detail for the client API: metadata, file info and ordered passages."""
        doc = self.find_doc(key)
        if doc is None:
            return None
        idx = self.index
        with idx.lock:
            chunks = idx.store.list_chunks(doc["doc_id"], with_text=include_text)
        source = doc.get("source", "")
        try:
            path = self.file_for(source)
        except PermissionError:
            path = None
        pages = [c["page"] for c in chunks if c.get("page")]
        passages = []
        for c in chunks:
            passage = {"index": c.get("chunk_index", len(passages)), "page": c.get("page")}
            if include_text:
                passage["text"] = c.get("text", "")
            passages.append(passage)
        return {
            **doc,
            "type": _doc_type(source),
            "size_bytes": path.stat().st_size if path else None,
            "file_url": f"/files/{source}" if path else None,
            "pages": max(pages) if pages else None,
            "passages": passages,
        }

    def stats(self) -> dict:
        """Aggregates for dashboards: totals, by file type, and the known indexes."""
        docs = self.list_docs()
        by_type: dict[str, dict] = {}
        for d in docs:
            t = _doc_type(d.get("source", ""))
            entry = by_type.setdefault(t, {"type": t, "documents": 0, "chunks": 0})
            entry["documents"] += 1
            entry["chunks"] += int(d.get("chunks") or 0)
        idx = self.index
        try:
            indexes = idx.store.list_collections(self.settings.collection + "_")
        except Exception as exc:  # Qdrant down: the rest of the stats are still useful
            log.warning("could not list collections: %s", exc)
            indexes = []
        for entry in indexes:
            entry["active"] = entry["collection"] == idx.collection
        return {
            "documents": len(docs),
            "chunks": sum(int(d.get("chunks") or 0) for d in docs),
            "by_type": sorted(by_type.values(), key=lambda e: (-e["documents"], e["type"])),
            "indexes": indexes,
            "embed_model": idx.embed_model,
        }

    def remove(self, key: str, delete_file: bool = True) -> IngestResult | None:
        """Remove a document by doc_id or source path. Returns None if unknown.

        Inactive indexes drop it on their next scan, since the file is gone."""
        idx = self.index
        with idx.lock:
            doc = next((d for d in idx.store.list_docs() if key in (d.get("doc_id"), d.get("source"))), None)
            if doc is None:
                return None
            idx.store.delete_doc(doc["doc_id"])
            source = doc.get("source", "")
            idx.stat_cache.pop(source, None)
            if delete_file and source:
                path = (self.docs_dir / source).resolve()
                if path.is_relative_to(self.docs_dir.resolve()) and path.is_file():
                    path.unlink()
            return IngestResult(source, "removed", doc["doc_id"])
