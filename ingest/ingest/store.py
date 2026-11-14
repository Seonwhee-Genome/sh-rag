"""Qdrant vector store operations."""

from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


class DimensionMismatch(RuntimeError):
    pass


class QdrantStore:
    def __init__(self, url: str, collection: str, api_key: str | None = None):
        self.collection = collection
        self.client = QdrantClient(url=url, api_key=api_key, timeout=60)

    def ensure_collection(self, dim: int, recreate: bool = False) -> None:
        exists = self.client.collection_exists(self.collection)
        if exists and recreate:
            self.client.delete_collection(self.collection)
            exists = False
        if exists:
            vectors = self.client.get_collection(self.collection).config.params.vectors
            size = getattr(vectors, "size", None)
            if size is not None and size != dim:
                raise DimensionMismatch(
                    f"collection '{self.collection}' stores {size}-dim vectors but the embedding "
                    f"model produces {dim}. Re-index with `python -m ingest scan --recreate` "
                    f"or set a different QDRANT_COLLECTION."
                )
            return
        self.client.create_collection(
            self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        for field in ("doc_id", "source"):
            self.client.create_payload_index(self.collection, field, qm.PayloadSchemaType.KEYWORD)
        self.client.create_payload_index(self.collection, "chunk_index", qm.PayloadSchemaType.INTEGER)

    def ping(self) -> None:
        self.client.get_collections()

    def search(self, vector: list[float], limit: int, threshold: float) -> list[dict[str, Any]]:
        if not self.client.collection_exists(self.collection):
            return []
        res = self.client.query_points(
            self.collection,
            query=vector,
            limit=limit,
            score_threshold=threshold or None,
            with_payload=["source", "page", "text"],
        )
        return [{"score": p.score, **(p.payload or {})} for p in res.points]

    def upsert(self, points: list[qm.PointStruct], batch: int = 256) -> None:
        for i in range(0, len(points), batch):
            self.client.upsert(self.collection, points=points[i : i + batch], wait=True)

    def delete_doc(self, doc_id: str) -> None:
        self.client.delete(
            self.collection,
            points_selector=qm.FilterSelector(filter=_match("doc_id", doc_id)),
            wait=True,
        )

    def doc_hash(self, doc_id: str) -> str | None:
        points, _ = self.client.scroll(
            self.collection, scroll_filter=_match("doc_id", doc_id), limit=1, with_payload=["content_hash"]
        )
        return points[0].payload.get("content_hash") if points else None

    def list_chunks(self, doc_id: str, with_text: bool = True) -> list[dict[str, Any]]:
        """Every chunk of one document, ordered by chunk_index."""
        if not self.client.collection_exists(self.collection):
            return []
        fields = ["chunk_index", "page"] + (["text"] if with_text else [])
        chunks: list[dict[str, Any]] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection, scroll_filter=_match("doc_id", doc_id), limit=256, offset=offset, with_payload=fields
            )
            chunks.extend(p.payload for p in points)
            if offset is None:
                break
        return sorted(chunks, key=lambda c: c.get("chunk_index", 0))

    def count(self) -> int:
        """Number of stored vectors (chunks) in this collection, 0 if it does not exist."""
        if not self.client.collection_exists(self.collection):
            return 0
        return self.client.count(self.collection, exact=True).count

    def list_collections(self, prefix: str) -> list[dict[str, Any]]:
        """All collections whose name starts with `prefix` (one per embedding model)."""
        out = []
        for c in self.client.get_collections().collections:
            if c.name.startswith(prefix):
                info = self.client.get_collection(c.name)
                out.append({"collection": c.name, "points": info.points_count or 0})
        return sorted(out, key=lambda c: c["collection"])

    def list_docs(self) -> list[dict[str, Any]]:
        """One entry per document (read from each document's first chunk)."""
        if not self.client.collection_exists(self.collection):
            return []
        docs: list[dict[str, Any]] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                self.collection,
                scroll_filter=_match("chunk_index", 0),
                limit=256,
                offset=offset,
                with_payload=["doc_id", "source", "title", "chunks", "content_hash", "ingested_at"],
            )
            docs.extend(p.payload for p in points)
            if offset is None:
                return docs


def _match(key: str, value: Any) -> qm.Filter:
    return qm.Filter(must=[qm.FieldCondition(key=key, match=qm.MatchValue(value=value))])
