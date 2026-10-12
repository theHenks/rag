"""Hybrid search: dense and BM25 candidates under the same filter, fused by RRF."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import quote

from qdrant_client import models

from rag.ingest import Services

_CANDIDATES = 100


@dataclass
class Filters:
    source_type: str | None = None
    tags: list[str] = field(default_factory=list)
    event: str | None = None
    authors: list[str] = field(default_factory=list)
    date_from: str | None = None
    date_to: str | None = None
    document_id: str | None = None


@dataclass
class Hit:
    chunk_id: str
    document_id: str
    score: float
    kind: str
    text: str
    heading: str
    title: str
    source_type: str
    authors: list[str]
    date: str | None
    event_title: str | None
    page_start: int
    page_end: int
    path: str | None
    link: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def build_filter(filters: Filters) -> models.Filter | None:
    must: list[models.Condition] = []
    if filters.source_type:
        must.append(models.FieldCondition(key="source_type", match=models.MatchValue(value=filters.source_type)))
    if filters.tags:
        must.append(models.FieldCondition(key="tags", match=models.MatchAny(any=filters.tags)))
    if filters.event:
        must.append(models.FieldCondition(key="event_id", match=models.MatchValue(value=filters.event)))
    if filters.document_id:
        must.append(models.FieldCondition(key="document_id", match=models.MatchValue(value=filters.document_id)))
    if filters.date_from or filters.date_to:
        must.append(models.FieldCondition(key="date", range=models.DatetimeRange(
            gte=filters.date_from, lte=filters.date_to)))
    if filters.authors:
        must.append(models.Filter(should=[
            models.FieldCondition(key="authors", match=models.MatchText(text=name)) for name in filters.authors]))
    return models.Filter(must=must) if must else None


def _existing_path(services: Services, document_id: str) -> str | None:
    doc = services.catalog.get_document(document_id)
    if doc is None:
        return None
    return next((p for p in doc.paths if Path(p).exists()), None)


def _hit(services: Services, chunk_id: str, score: float, payload: dict, paths: dict[str, str | None]) -> Hit:
    document_id = payload["document_id"]
    if document_id not in paths:
        paths[document_id] = _existing_path(services, document_id)
    path = paths[document_id]
    link = payload.get("indico_url") or (f"file://{quote(path)}#page={payload['page_start']}" if path else None)
    return Hit(chunk_id=chunk_id, document_id=document_id, score=score, kind=payload["kind"],
               text=payload["text"], heading=payload.get("heading", ""), title=payload["title"],
               source_type=payload["source_type"], authors=payload.get("authors", []), date=payload.get("date"),
               event_title=payload.get("event_title"), page_start=payload["page_start"],
               page_end=payload["page_end"], path=path, link=link)


def search(services: Services, query: str, filters: Filters | None = None, *, limit: int = 8,
           max_per_document: int = 3) -> list[Hit]:
    embedder = services.embedder
    points = services.index.query(embedder.dense_query(query), embedder.sparse_query(query),
                                  build_filter(filters or Filters()), limit=_CANDIDATES)
    per_document: dict[str, int] = {}
    paths: dict[str, str | None] = {}
    hits = []
    for point in points:
        document_id = point.payload["document_id"]
        if per_document.get(document_id, 0) >= max_per_document:
            continue
        per_document[document_id] = per_document.get(document_id, 0) + 1
        hits.append(_hit(services, str(point.id), point.score, point.payload, paths))
        if len(hits) == limit:
            break
    return hits


def context(services: Services, chunk_id: str, before: int = 1, after: int = 1) -> list[Hit]:
    center = services.index.chunk(chunk_id)
    if center is None:
        raise KeyError(f"no chunk {chunk_id}")
    paths: dict[str, str | None] = {}
    return [_hit(services, p["id"], 0.0, p, paths)
            for p in services.index.neighbors(center["document_id"], center["chunk_index"], before, after)]
