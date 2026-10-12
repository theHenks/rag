import os
from pathlib import Path

import pytest

from rag.index import QdrantUnavailable
from rag.ingest import Services
from rag.jobs import WorkerBusy, run_foreground, run_job, run_worker, submit_add
from rag.metadata import Overrides
from rag.parse import load_parsed

FIX = Path(__file__).parent / "fixtures" / "mineru"


class Parser:
    def __init__(self):
        self.calls: list[str] = []

    def __call__(self, pdf, document_id):
        self.calls.append(pdf.name)
        if pdf.read_bytes().startswith(b"BROKEN"):
            raise RuntimeError("MinerU could not open the file")
        return load_parsed(FIX / "paper")


@pytest.fixture
def services(config, fake_embedder):
    return Services(config, embedder=fake_embedder, parser=Parser())


def _folder(tmp_path):
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "good.pdf").write_bytes(b"%PDF good")
    (folder / "bad.pdf").write_bytes(b"BROKEN")
    (folder / "notes.txt").write_text("x")
    return folder


def test_a_failing_file_does_not_stop_the_others(tmp_path, services):
    job = submit_add(services.catalog, _folder(tmp_path), Overrides(), enrich=False)
    assert run_worker(services) == 1
    assert services.catalog.get_job(job).status == "completed_with_failures"
    files = {Path(f.path).name: f for f in services.catalog.job_files(job)}
    assert files["good.pdf"].status == "done"
    assert (files["bad.pdf"].status, files["bad.pdf"].stage) == ("failed", "parse")
    assert "RuntimeError: MinerU could not open the file" in files["bad.pdf"].error
    assert files["notes.txt"].status == "skipped"


def test_a_job_wide_error_stops_the_job(tmp_path, services, monkeypatch):
    def down(*args, **kwargs):
        raise QdrantUnavailable("Qdrant is not reachable at http://x")
    monkeypatch.setattr("rag.jobs.ingest_file", down)
    job = submit_add(services.catalog, _folder(tmp_path), Overrides(), enrich=False)
    run_worker(services)
    finished = services.catalog.get_job(job)
    assert finished.status == "failed" and "not reachable" in finished.error


def test_a_missing_path_is_rejected_at_submission(tmp_path, services):
    with pytest.raises(FileNotFoundError):
        submit_add(services.catalog, tmp_path / "nope", Overrides(), enrich=False)


def test_a_resumed_job_skips_completed_files(tmp_path, services):
    folder = _folder(tmp_path)
    job = submit_add(services.catalog, folder, Overrides(), enrich=False)
    run_job(services.catalog.get_job(job), services)
    services.catalog.db.execute("UPDATE jobs SET status = 'queued' WHERE id = ?", (job,))
    run_worker(services)
    assert services.parser.calls == ["bad.pdf", "good.pdf", "bad.pdf"]
    good = next(f for f in services.catalog.job_files(job) if f.path.endswith("good.pdf"))
    assert good.status == "done"


def test_foreground_refuses_while_a_worker_runs(tmp_path, services):
    job = submit_add(services.catalog, _folder(tmp_path), Overrides(), enrich=False)
    services.catalog.acquire_worker(os.getppid())
    with pytest.raises(WorkerBusy, match=str(os.getppid())):
        run_foreground(job, services, progress=lambda m: None)
