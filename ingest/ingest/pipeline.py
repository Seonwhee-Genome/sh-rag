"""Load → chunk → embed → upsert, with change detection and deletion sync."""

from __future__ import annotations

import hashlib
import logging
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client.http import models as qm

from ingest.chunker import split_text
from ingest.config import Settings
from ingest.embedder import OllamaEmbedder
from ingest.loaders import is_supported, load
from ingest.store import QdrantStore

log = logging.getLogger(__name__)

# Stable namespace so a given source path always maps to the same doc_id.
_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, "https://sh-rag.local/doc")
UPLOADS_DIR = "uploads"


@dataclass
class IngestResult:
    source: str
    status: str  # indexed | unchanged | empty | unsupported | removed | error
    doc_id: str = ""
    chunks: int = 0
    error: str | None = None


def doc_id_for(source: str) -> str:
    return str(uuid.uuid5(_NAMESPACE, source))


class Ingestor:
    def __init__(self, settings: Settings, embedder=None, store=None):
        self.settings = settings
        self.embedder = embedder or OllamaEmbedder(
            settings.ollama_url, settings.embed_model, settings.embed_batch_size
        )
        self.store = store or QdrantStore(settings.qdrant_url, settings.collection, settings.qdrant_api_key)
        self._lock = threading.RLock()
        self._ready = False
        # source -> (mtime_ns, size) of the last successful ingest; lets the
        # periodic scan skip unchanged files without hashing or querying Qdrant.
        self._stat_cache: dict[str, tuple[int, int]] = {}

    @property
    def docs_dir(self) -> Path:
        return self.settings.docs_dir

    def setup(self, recreate: bool = False) -> None:
        """Make sure the embedding model and the collection exist."""
        with self._lock:
            self.embedder.ensure_model()
            self.store.ensure_collection(self.embedder.dimension(), recreate=recreate)
            if recreate:
                self._stat_cache.clear()
            self._ready = True

    def _ensure_ready(self) -> None:
        if not self._ready:
            self.setup()

    def source_for(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.docs_dir.resolve()).as_posix()
        except ValueError:
            return path.name

    def ingest_file(self, path: Path, force: bool = False) -> IngestResult:
        source = self.source_for(path)
        doc_id = doc_id_for(source)
        if not is_supported(path):
            return IngestResult(source, "unsupported", doc_id)

        with self._lock:
            self._ensure_ready()
            stat = path.stat()
            data = path.read_bytes()
            content_hash = hashlib.sha256(data).hexdigest()
            if not force and self.store.doc_hash(doc_id) == content_hash:
                self._stat_cache[source] = (stat.st_mtime_ns, stat.st_size)
                return IngestResult(source, "unchanged", doc_id)

            records: list[tuple[str, int | None]] = []
            for section in load(path):
                for chunk in split_text(section.text, self.settings.chunk_size, self.settings.chunk_overlap):
                    records.append((chunk, section.page))

            if not records:
                self.store.delete_doc(doc_id)
                self._stat_cache[source] = (stat.st_mtime_ns, stat.st_size)
                return IngestResult(source, "empty", doc_id)

            title = path.stem
            # Prefixing the title gives each chunk document-level context in its embedding.
            vectors = self.embedder.embed([f"{title}\n\n{text}" for text, _ in records])
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
            self.store.delete_doc(doc_id)
            self.store.upsert(points)
            self._stat_cache[source] = (stat.st_mtime_ns, stat.st_size)
            log.info("indexed %s (%d chunks)", source, len(points))
            return IngestResult(source, "indexed", doc_id, len(points))

    def iter_files(self):
        root = self.docs_dir
        if not root.exists():
            root.mkdir(parents=True, exist_ok=True)
        for path in sorted(root.rglob("*")):
            rel_parts = path.relative_to(root).parts
            if path.is_file() and is_supported(path) and not any(p.startswith(".") for p in rel_parts):
                yield path

    def scan(self, force: bool = False, prune: bool = True) -> list[IngestResult]:
        """Sync the docs folder into the index: add/update changed files, drop deleted ones."""
        with self._lock:
            self._ensure_ready()
            results: list[IngestResult] = []
            seen: set[str] = set()
            for path in self.iter_files():
                source = self.source_for(path)
                seen.add(source)
                try:
                    st = path.stat()
                    if not force and self._stat_cache.get(source) == (st.st_mtime_ns, st.st_size):
                        continue  # unchanged since last scan
                    results.append(self.ingest_file(path, force=force))
                except Exception as exc:  # keep going: one bad file must not stop the scan
                    log.exception("failed to ingest %s", source)
                    results.append(IngestResult(source, "error", doc_id_for(source), error=str(exc)))
            if prune:
                for doc in self.store.list_docs():
                    if doc.get("source") not in seen:
                        self.store.delete_doc(doc["doc_id"])
                        self._stat_cache.pop(doc.get("source"), None)
                        results.append(IngestResult(doc.get("source", ""), "removed", doc["doc_id"]))
            return results

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
        with self._lock:
            return sorted(self.store.list_docs(), key=lambda d: d.get("source", ""))

    def remove(self, key: str, delete_file: bool = True) -> IngestResult | None:
        """Remove a document by doc_id or source path. Returns None if unknown."""
        with self._lock:
            doc = next((d for d in self.store.list_docs() if key in (d.get("doc_id"), d.get("source"))), None)
            if doc is None:
                return None
            self.store.delete_doc(doc["doc_id"])
            source = doc.get("source", "")
            self._stat_cache.pop(source, None)
            if delete_file and source:
                path = (self.docs_dir / source).resolve()
                if path.is_relative_to(self.docs_dir.resolve()) and path.is_file():
                    path.unlink()
            return IngestResult(source, "removed", doc["doc_id"])
