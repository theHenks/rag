"""The Qdrant collection: one point per chunk with a dense and a BM25 vector.

The collection's metadata records the settings it was built with; every
process compares them with its configuration before using the collection.
"""
from __future__ import annotations

import uuid

import httpx
from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from rag.chunk import CHUNKER_VERSION, Chunk
from rag.config import Config
from rag.embed import Embedder

DENSE = "dense"
SPARSE = "bm25"
_NAMESPACE = uuid.UUID("7d3c5b0e-6a2f-4f43-9c1e-2b8f6f0a9d41")

_KEYWORD_FIELDS = ("document_id", "source_type", "tags", "event_id", "kind")


class QdrantUnavailable(RuntimeError):
    pass


class SettingsMismatch(RuntimeError):
    pass


def index_settings(config: Config, dense_size: int) -> dict:
    return {"dense_model": config.dense_model, "dense_size": dense_size,
            "sparse_model": config.sparse_model, "chunker_version": CHUNKER_VERSION}


def point_id(document_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_NAMESPACE, f"{document_id}:{chunk_index}:{CHUNKER_VERSION}"))


def build_points(document_id: str, chunks: list[Chunk], texts: list[str], payload_base: dict,
                 embedder: Embedder) -> list[models.PointStruct]:
    if not chunks:
        return []
    dense = embedder.dense_documents(texts)
    sparse = embedder.sparse_documents(texts)
    return [
        models.PointStruct(
            id=point_id(document_id, chunk.index),
            vector={DENSE: d, SPARSE: s},
            payload={**payload_base, "chunk_index": chunk.index, "kind": chunk.kind, "text": chunk.text,
                     "heading": chunk.heading, "page_start": chunk.page_start, "page_end": chunk.page_end,
                     "bboxes": chunk.bboxes},
        )
        for chunk, d, s in zip(chunks, dense, sparse, strict=True)
    ]


def _document_filter(document_id: str) -> models.Filter:
    return models.Filter(must=[models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id))])


class Index:
    def __init__(self, config: Config, dense_size: int):
        self.url = config.qdrant_url
        self.name = config.collection
        self.settings = index_settings(config, dense_size)
        self.client = QdrantClient(url=config.qdrant_url, timeout=60)

    def _call(self, fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (ResponseHandlingException, httpx.TransportError) as exc:
            raise QdrantUnavailable(
                f"Qdrant is not reachable at {self.url} ({exc}); start it with "
                "`scripts/qdrant start` in the rag repository") from exc

    def ensure_collection(self) -> None:
        if not self._call(self.client.collection_exists, self.name):
            self._create()
            return
        stored = self._call(self.client.get_collection, self.name).config.metadata or {}
        differences = [f"{key}: collection has {stored.get(key)}, configuration has {value}"
                       for key, value in self.settings.items() if stored.get(key) != value]
        if differences:
            raise SettingsMismatch(
                f"collection {self.name!r} was built with different settings ("
                + "; ".join(differences) + "); run `rag rebuild` to re-index with the current configuration")

    def _create(self) -> None:
        self._call(
            self.client.create_collection, self.name,
            vectors_config={DENSE: models.VectorParams(size=self.settings["dense_size"],
                                                       distance=models.Distance.COSINE)},
            sparse_vectors_config={SPARSE: models.SparseVectorParams(modifier=models.Modifier.IDF)},
            metadata=self.settings,
        )
        for key in _KEYWORD_FIELDS:
            self._call(self.client.create_payload_index, self.name, key, models.PayloadSchemaType.KEYWORD)
        self._call(self.client.create_payload_index, self.name, "chunk_index", models.PayloadSchemaType.INTEGER)
        self._call(self.client.create_payload_index, self.name, "date", models.PayloadSchemaType.DATETIME)
        self._call(self.client.create_payload_index, self.name, "authors",
                   models.TextIndexParams(type=models.TextIndexType.TEXT, tokenizer=models.TokenizerType.WORD,
                                          lowercase=True))

    def delete_document(self, document_id: str) -> None:
        self._call(self.client.delete, self.name,
                   points_selector=models.FilterSelector(filter=_document_filter(document_id)), wait=True)

    def add_points(self, points: list[models.PointStruct]) -> None:
        for start in range(0, len(points), 64):
            self._call(self.client.upsert, self.name, points=points[start:start + 64], wait=True)

    def replace_document(self, document_id: str, points: list[models.PointStruct]) -> None:
        self.delete_document(document_id)
        self.add_points(points)

    def count(self, document_id: str | None = None) -> int:
        flt = _document_filter(document_id) if document_id else None
        return self._call(self.client.count, self.name, count_filter=flt, exact=True).count

    def chunk(self, chunk_id: str) -> dict | None:
        try:
            records = self._call(self.client.retrieve, self.name, ids=[chunk_id], with_payload=True)
        except UnexpectedResponse:
            return None
        return {"id": str(records[0].id), **records[0].payload} if records else None

    def neighbors(self, document_id: str, chunk_index: int, before: int, after: int) -> list[dict]:
        flt = models.Filter(must=[
            models.FieldCondition(key="document_id", match=models.MatchValue(value=document_id)),
            models.FieldCondition(key="chunk_index", range=models.Range(gte=chunk_index - before,
                                                                        lte=chunk_index + after)),
        ])
        records, _ = self._call(self.client.scroll, self.name, scroll_filter=flt, limit=before + after + 1,
                                with_payload=True)
        return sorted(({"id": str(r.id), **r.payload} for r in records), key=lambda p: p["chunk_index"])

    def query(self, dense: list[float], sparse: models.SparseVector, query_filter: models.Filter | None,
              limit: int) -> list[models.ScoredPoint]:
        response = self._call(
            self.client.query_points, self.name,
            prefetch=[models.Prefetch(query=dense, using=DENSE, filter=query_filter, limit=50),
                      models.Prefetch(query=sparse, using=SPARSE, filter=query_filter, limit=50)],
            query=models.FusionQuery(fusion=models.Fusion.RRF), limit=limit, with_payload=True,
        )
        return response.points

    def drop(self) -> None:
        if self._call(self.client.collection_exists, self.name):
            self._call(self.client.delete_collection, self.name)
