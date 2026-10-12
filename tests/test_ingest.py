import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest

from rag.enrich import VisionUnavailable
from rag.index import point_id
from rag.ingest import (Services, discover, enrich_document, ingest_file, rebuild, remove_document, remove_orphans,
                        sha256_of)
from rag.metadata import Overrides
from rag.parse import load_parsed

FIX = Path(__file__).parent / "fixtures" / "mineru"


class FixtureParser:
    """Stands in for MinerU: returns the paper fixture for every PDF and counts calls."""
    def __init__(self):
        self.calls = 0

    def __call__(self, pdf, document_id):
        self.calls += 1
        if pdf.read_bytes().startswith(b"BROKEN"):
            raise RuntimeError("MinerU could not open the file")
        return load_parsed(FIX / "paper")


@pytest.fixture
def services(config, fake_embedder):
    return Services(config, embedder=fake_embedder, parser=FixtureParser())


def _pdf(directory: Path, name: str, content: bytes = b"%PDF-1.4 a") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(content)
    return path


def test_discover_lists_pdfs_and_others(tmp_path):
    _pdf(tmp_path, "a.pdf"); _pdf(tmp_path / "sub", "b.PDF"); _pdf(tmp_path, "notes.txt")
    pdfs, others = discover(tmp_path)
    assert [p.name for p in pdfs] == ["a.pdf", "b.PDF"] and [p.name for p in others] == ["notes.txt"]
    with pytest.raises(FileNotFoundError):
        discover(tmp_path / "missing")


def test_ingest_indexes_and_a_second_run_skips(tmp_path, services):
    pdf = _pdf(tmp_path, "paper.pdf")
    stages = []
    doc_id, skipped = ingest_file(pdf, root=tmp_path, overrides=Overrides(source_type="paper"),
                                  enrich=False, services=services, stage=stages.append)
    assert not skipped and stages == ["hash", "parse", "metadata", "chunk", "embed", "write"]
    doc = services.catalog.get_document(doc_id)
    assert (doc.status, doc.source_type, doc.paths) == ("indexed", "paper", [str(pdf.resolve())])
    assert services.index.count(doc_id) > 0
    _, skipped = ingest_file(pdf, root=tmp_path, overrides=Overrides(), enrich=False,
                             services=services, stage=lambda s: None)
    assert skipped and services.parser.calls == 1


def test_changed_file_replaces_old_version(tmp_path, services):
    pdf = _pdf(tmp_path, "paper.pdf")
    old, _ = ingest_file(pdf, root=tmp_path, overrides=Overrides(), enrich=False, services=services, stage=lambda s: None)
    pdf.write_bytes(b"%PDF-1.4 edited")
    new, _ = ingest_file(pdf, root=tmp_path, overrides=Overrides(), enrich=False, services=services, stage=lambda s: None)
    assert old != new
    assert services.catalog.get_document(old) is None
    assert services.index.count(old) == 0 and services.index.count(new) > 0


def test_a_copy_at_a_second_path_is_one_document(tmp_path, services):
    a = _pdf(tmp_path, "a.pdf")
    b = _pdf(tmp_path / "copy", "a.pdf")
    first, _ = ingest_file(a, root=tmp_path, overrides=Overrides(), enrich=False, services=services, stage=lambda s: None)
    second, skipped = ingest_file(b, root=tmp_path, overrides=Overrides(), enrich=False, services=services, stage=lambda s: None)
    assert first == second and skipped
    assert len(services.catalog.get_document(first).paths) == 2


def test_unreadable_file_fails_only_that_file(tmp_path, services):
    pdf = _pdf(tmp_path, "locked.pdf")
    pdf.chmod(0)
    try:
        with pytest.raises(PermissionError):
            ingest_file(pdf, root=tmp_path, overrides=Overrides(), enrich=False, services=services, stage=lambda s: None)
    finally:
        pdf.chmod(0o644)


def test_enrich_adds_figure_chunks(tmp_path, services, monkeypatch):
    pdf = _pdf(tmp_path, "paper.pdf")
    doc_id, _ = ingest_file(pdf, root=tmp_path, overrides=Overrides(), enrich=False, services=services, stage=lambda s: None)
    before = services.index.count(doc_id)
    monkeypatch.setattr("rag.ingest.check_vision", lambda config: None)
    monkeypatch.setattr("rag.ingest.describe_figures",
                        lambda parsed, config: {b: "A plot." for b in parsed.blocks if b.kind == "figure"})
    enrich_document(doc_id, services, stage=lambda s: None)
    assert services.index.count(doc_id) > before
    assert services.catalog.get_document(doc_id).enrich


def test_remove_document_clears_catalog_and_index(tmp_path, services):
    doc_id, _ = ingest_file(_pdf(tmp_path, "p.pdf"), root=tmp_path, overrides=Overrides(), enrich=False,
                            services=services, stage=lambda s: None)
    remove_document(doc_id, services)
    assert services.catalog.get_document(doc_id) is None and services.index.count(doc_id) == 0


def test_rebuild_recreates_the_index_from_the_cache(tmp_path, services):
    doc_id, _ = ingest_file(_pdf(tmp_path, "p.pdf"), root=tmp_path, overrides=Overrides(), enrich=False,
                            services=services, stage=lambda s: None)
    count = services.index.count(doc_id)
    services.index.drop()
    assert rebuild(services, report=lambda m: None) == []
    assert services.index.count(doc_id) == count


def test_a_path_taking_already_indexed_content_drops_the_orphaned_document(tmp_path, services):
    first = _pdf(tmp_path, "a.pdf", b"%PDF-1.4 one")
    other = _pdf(tmp_path / "x", "b.pdf", b"%PDF-1.4 two")
    old, _ = ingest_file(first, root=tmp_path, overrides=Overrides(), enrich=False, services=services, stage=lambda s: None)
    two, _ = ingest_file(other, root=tmp_path, overrides=Overrides(), enrich=False, services=services, stage=lambda s: None)
    first.write_bytes(b"%PDF-1.4 two")
    doc_id, skipped = ingest_file(first, root=tmp_path, overrides=Overrides(), enrich=False, services=services, stage=lambda s: None)
    assert (doc_id, skipped) == (two, True)
    assert services.catalog.get_document(old) is None and services.index.count(old) == 0


def test_documents_without_paths_are_swept(tmp_path, services):
    doc_id, _ = ingest_file(_pdf(tmp_path, "p.pdf"), root=tmp_path, overrides=Overrides(), enrich=False,
                            services=services, stage=lambda s: None)
    services.catalog.db.execute("DELETE FROM paths")
    assert remove_orphans(services) == [doc_id]
    assert services.catalog.get_document(doc_id) is None and services.index.count(doc_id) == 0


def test_failed_enrichment_leaves_the_flag_unset(tmp_path, services, monkeypatch):
    doc_id, _ = ingest_file(_pdf(tmp_path, "p.pdf"), root=tmp_path, overrides=Overrides(), enrich=False,
                            services=services, stage=lambda s: None)

    def down(config):
        raise VisionUnavailable("Ollama is not reachable")
    monkeypatch.setattr("rag.ingest.check_vision", down)
    with pytest.raises(VisionUnavailable):
        enrich_document(doc_id, services, stage=lambda s: None)
    assert not services.catalog.get_document(doc_id).enrich


def test_overrides_on_an_unchanged_file_are_applied(tmp_path, services):
    pdf = _pdf(tmp_path, "p.pdf")
    doc_id, _ = ingest_file(pdf, root=tmp_path, overrides=Overrides(), enrich=False, services=services, stage=lambda s: None)
    _, skipped = ingest_file(pdf, root=tmp_path, overrides=Overrides(source_type="paper", tags=("legend",)),
                             enrich=False, services=services, stage=lambda s: None)
    doc = services.catalog.get_document(doc_id)
    assert skipped and (doc.source_type, doc.tags) == ("paper", ["legend"])
    payload = services.index.chunk(point_id(doc_id, 0))
    assert (payload["source_type"], payload["tags"]) == ("paper", ["legend"])


def test_changing_chunk_size_makes_documents_out_of_date(tmp_path, config, fake_embedder):
    pdf = _pdf(tmp_path, "p.pdf")
    first = Services(config, embedder=fake_embedder, parser=FixtureParser())
    ingest_file(pdf, root=tmp_path, overrides=Overrides(), enrich=False, services=first, stage=lambda s: None)
    second = Services(replace(config, chunk_tokens=200), embedder=fake_embedder, parser=FixtureParser())
    _, skipped = ingest_file(pdf, root=tmp_path, overrides=Overrides(), enrich=False, services=second, stage=lambda s: None)
    assert not skipped


def test_concurrent_first_use_builds_one_embedder(config, monkeypatch):
    built = []

    class Slow:
        dense_size = 8

        def __init__(self, config):
            time.sleep(0.2)
            built.append(self)
    monkeypatch.setattr("rag.embed.ModelEmbedder", Slow)
    services = Services(config)
    threads = [threading.Thread(target=lambda: services.embedder) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(built) == 1
