from pathlib import Path

import pytest

from rag.chunk import Chunk
from rag.catalog import Document
from rag.index import build_points, point_id
from rag.ingest import Services
from rag.search import Filters, context, search


def _add(services, doc_id, texts, *, source_type="paper", tags=(), authors=("A. Author",), date="2025-05-01T00:00:00",
         path=None, indico_url=None):
    services.catalog.put_document(Document(
        id=doc_id, title=f"Doc {doc_id[:1]}", source_type=source_type, authors=list(authors), date=date,
        tags=list(tags), event_id=None, event_title=None, session=None, indico_url=indico_url, enrich=False,
        status="indexed", settings=None, page_count=len(texts), error=None))
    if path:
        services.catalog.add_path(str(path), doc_id)
    chunks = [Chunk(i, "text", t, "", i + 1, i + 1, []) for i, t in enumerate(texts)]
    base = {"document_id": doc_id, "title": f"Doc {doc_id[:1]}", "source_type": source_type,
            "authors": list(authors), "date": date, "tags": list(tags), "event_id": None,
            "event_title": None, "session": None, "indico_url": indico_url}
    services.index.replace_document(doc_id, build_points(doc_id, chunks, texts, base, services.embedder))


@pytest.fixture
def services(config, fake_embedder, tmp_path):
    s = Services(config, embedder=fake_embedder)
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(b"%PDF")
    _add(s, "a" * 64, ["thallium 208 background", "germanium detectors", "thallium again", "thallium thrice"],
         tags=["legend"], path=pdf)
    _add(s, "b" * 64, ["thallium in a talk"], source_type="talk", authors=["Maria Müller"],
         date="2026-03-03T09:00:00", indico_url="https://h/event/1/contributions/2/")
    return s


def test_exact_term_is_found_and_linked(services, tmp_path):
    hits = search(services, "thallium", limit=8)
    assert hits[0].text.startswith("thallium")
    paper = next(h for h in hits if h.document_id == "a" * 64)
    assert paper.path == str(tmp_path / "a.pdf")
    assert paper.link == f"file://{tmp_path / 'a.pdf'}#page={paper.page_start}"
    talk = next(h for h in hits if h.document_id == "b" * 64)
    assert talk.link == "https://h/event/1/contributions/2/"


def test_at_most_max_per_document(services):
    hits = search(services, "thallium", limit=8, max_per_document=2)
    assert sum(h.document_id == "a" * 64 for h in hits) == 2


@pytest.mark.parametrize("filters, expected", [
    (Filters(source_type="talk"), {"b" * 64}),
    (Filters(tags=["legend"]), {"a" * 64}),
    (Filters(authors=["müller"]), {"b" * 64}),
    (Filters(date_from="2026-01-01"), {"b" * 64}),
    (Filters(date_to="2025-12-31"), {"a" * 64}),
    (Filters(document_id="a" * 64), {"a" * 64}),
])
def test_filters(services, filters, expected):
    assert {h.document_id for h in search(services, "thallium", filters)} == expected


def test_context_returns_neighbours_in_order(services):
    hits = context(services, point_id("a" * 64, 2), before=1, after=1)
    assert [h.text for h in hits] == ["germanium detectors", "thallium again", "thallium thrice"]


def test_context_of_an_unknown_chunk_is_an_error(services):
    with pytest.raises(KeyError, match="no chunk"):
        context(services, "00000000-0000-0000-0000-000000000000")
