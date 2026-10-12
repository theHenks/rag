"""The per-document pipeline: hash → parse → metadata → chunk → [enrich] → embed → write."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from functools import cached_property
from pathlib import Path

from rag.catalog import Catalog, Document
from rag.chunk import chunk_document, embedding_text, figure_chunks
from rag.config import Config
from rag.embed import Embedder
from rag.enrich import VisionUnavailable, check_vision, describe_figures
from rag.index import Index, QdrantUnavailable, SettingsMismatch, build_points, index_settings
from rag.metadata import Metadata, Overrides, find_manifest_talk, resolve_metadata
from rag.parse import MinerUMissing, Parsed, load_parsed, mineru_version, parse_directory, parse_pdf

ABORTING = (QdrantUnavailable, SettingsMismatch, VisionUnavailable, MinerUMissing)

Stage = Callable[[str], None]


class Services:
    """Configuration plus the catalog, embedder, index and parser one process uses.

    The embedder and the index are created on first use, so commands that only
    read the catalog never load a model or contact Qdrant. `embedder` and
    `parser` can be supplied to replace the model and MinerU.
    """

    def __init__(self, config: Config, *, embedder: Embedder | None = None,
                 parser: Callable[[Path, str], Parsed] | None = None):
        self.config = config
        self.catalog = Catalog(config.catalog_path)
        self._embedder = embedder
        self.parser = parser

    @cached_property
    def embedder(self) -> Embedder:
        if self._embedder is not None:
            return self._embedder
        from rag.embed import ModelEmbedder

        return ModelEmbedder(self.config)

    @cached_property
    def index(self) -> Index:
        index = Index(self.config, self.embedder.dense_size)
        index.ensure_collection()
        return index

    def parse(self, pdf: Path, document_id: str) -> Parsed:
        if self.parser is not None:
            return self.parser(pdf, document_id)
        return parse_pdf(pdf, document_id, cache_dir=self.config.cache_dir, tier=self.config.mineru_tier)

    def load_cached(self, document_id: str) -> Parsed:
        """The cached MinerU output of an already-parsed document; MinerU is not run."""
        if self.parser is not None:
            doc = self.catalog.get_document(document_id)
            return self.parser(Path(doc.paths[0]), document_id)
        return load_parsed(parse_directory(self.config.cache_dir, document_id, mineru_version(),
                                           self.config.mineru_tier))

    def settings(self) -> str:
        """Everything that, when changed, makes an indexed document out of date."""
        return json.dumps({"mineru": mineru_version(), "tier": self.config.mineru_tier,
                           **index_settings(self.config, self.embedder.dense_size)}, sort_keys=True)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def discover(path: Path) -> tuple[list[Path], list[Path]]:
    """PDFs under `path` and the other files there (hidden files and manifests excluded)."""
    if not path.exists():
        raise FileNotFoundError(f"no such file or directory: {path}")
    if path.is_file():
        return ([path], []) if path.suffix.lower() == ".pdf" else ([], [path])
    files = sorted(p for p in path.rglob("*") if p.is_file() and not p.name.startswith("."))
    pdfs = [p for p in files if p.suffix.lower() == ".pdf"]
    others = [p for p in files if p.suffix.lower() != ".pdf" and p.name != "_manifest.json"]
    return pdfs, others


def _document(document_id: str, meta: Metadata, parsed: Parsed, enrich: bool) -> Document:
    return Document(id=document_id, title=meta.title, source_type=meta.source_type, authors=meta.authors,
                    date=meta.date, tags=meta.tags, event_id=meta.event_id, event_title=meta.event_title,
                    session=meta.session, indico_url=meta.indico_url, enrich=enrich, status="pending",
                    settings=None, page_count=len(parsed.pages), error=None)


def _payload_base(doc: Document) -> dict:
    return {"document_id": doc.id, "title": doc.title, "source_type": doc.source_type,
            "authors": doc.authors, "date": doc.date, "tags": doc.tags, "event_id": doc.event_id,
            "event_title": doc.event_title, "session": doc.session, "indico_url": doc.indico_url}


def _slides(doc: Document, parsed: Parsed) -> bool:
    return doc.source_type == "talk" or parsed.landscape


def _index_document(doc: Document, parsed: Parsed, slides: bool, services: Services, stage: Stage) -> None:
    stage("chunk")
    chunks = chunk_document(parsed, slides=slides, max_tokens=services.config.chunk_tokens,
                            count_tokens=services.embedder.count_tokens)
    if doc.enrich:
        stage("enrich")
        check_vision(services.config)
        chunks += figure_chunks(parsed, describe_figures(parsed, services.config), start=len(chunks))
    stage("embed")
    texts = [embedding_text(doc.title, c) for c in chunks]
    points = build_points(doc.id, chunks, texts, _payload_base(doc), services.embedder)
    stage("write")
    services.index.replace_document(doc.id, points)
    services.catalog.set_status(doc.id, "indexed", settings=services.settings())


def ingest_file(pdf: Path, *, root: Path, overrides: Overrides, enrich: bool, services: Services,
                stage: Stage) -> tuple[str, bool]:
    """Index one PDF; returns its document id and whether it was already up to date.

    A file whose path previously belonged to another checksum replaces that
    document once no other path refers to it.
    """
    pdf = pdf.resolve()
    stage("hash")
    document_id = sha256_of(pdf)
    catalog = services.catalog
    previous = catalog.documents_for_path(str(pdf))
    existing = catalog.get_document(document_id)
    if (existing and existing.status == "indexed" and existing.settings == services.settings()
            and (existing.enrich or not enrich)):
        catalog.add_path(str(pdf), document_id)
        return document_id, True

    stage("parse")
    parsed = services.parse(pdf, document_id)
    stage("metadata")
    talk = find_manifest_talk(pdf, document_id, root=root)
    meta = resolve_metadata(pdf, parsed, talk, overrides)
    catalog.put_document(_document(document_id, meta, parsed, enrich or bool(existing and existing.enrich)))
    catalog.add_path(str(pdf), document_id)
    if previous and previous != document_id:
        old = catalog.get_document(previous)
        if old is not None and not old.paths:
            remove_document(previous, services)
    _index_document(catalog.get_document(document_id), parsed, meta.slides, services, stage)
    return document_id, False


def enrich_document(document_id: str, services: Services, stage: Stage) -> None:
    doc = services.catalog.get_document(document_id)
    if doc is None:
        raise KeyError(f"no document {document_id}")
    doc.enrich = True
    services.catalog.put_document(doc)
    parsed = services.load_cached(document_id)
    _index_document(doc, parsed, _slides(doc, parsed), services, stage)


def remove_document(document_id: str, services: Services) -> None:
    services.index.delete_document(document_id)
    services.catalog.delete_document(document_id)


def rebuild(services: Services, report: Callable[[str], None]) -> list[tuple[str, str]]:
    """Drop the collection and re-index every catalogued document from the MinerU cache."""
    services.index.drop()
    del services.index
    failures = []
    for doc in services.catalog.list_documents():
        report(f"{doc.title} ({doc.id[:12]})")
        try:
            parsed = services.load_cached(doc.id)
            _index_document(doc, parsed, _slides(doc, parsed), services, stage=lambda s: None)
        except ABORTING:
            raise
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            services.catalog.set_status(doc.id, "failed", error=error)
            failures.append((doc.id, error))
    return failures
