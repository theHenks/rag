import os
import subprocess
import sys

from rag.catalog import Catalog, Document, pid_alive


def _doc(id="a" * 64, **kw):
    base = dict(id=id, title="T", source_type="paper", authors=["A. B"], date=None, tags=["x"],
                event_id=None, event_title=None, session=None, indico_url=None, enrich=False,
                status="pending", settings=None, page_count=3, error=None, paths=[])
    base.update(kw)
    return Document(**base)


def test_document_round_trip_with_paths(tmp_path):
    cat = Catalog(tmp_path / "c.sqlite3")
    cat.put_document(_doc())
    assert cat.add_path("/p/a.pdf", "a" * 64) is None
    doc = cat.get_document("a" * 64)
    assert doc.authors == ["A. B"] and doc.tags == ["x"] and doc.paths == ["/p/a.pdf"]


def test_updating_a_document_keeps_its_paths(tmp_path):
    cat = Catalog(tmp_path / "c.sqlite3")
    cat.put_document(_doc())
    cat.add_path("/p/a.pdf", "a" * 64)
    cat.put_document(_doc(title="New"))
    assert cat.get_document("a" * 64).paths == ["/p/a.pdf"]


def test_moving_a_path_to_another_document_reports_the_old_one(tmp_path):
    cat = Catalog(tmp_path / "c.sqlite3")
    cat.put_document(_doc("a" * 64))
    cat.put_document(_doc("b" * 64))
    cat.add_path("/p/a.pdf", "a" * 64)
    assert cat.add_path("/p/a.pdf", "b" * 64) == "a" * 64
    assert cat.get_document("a" * 64).paths == []
    assert cat.documents_for_path("/p/a.pdf") == "b" * 64


def test_list_documents_filters(tmp_path):
    cat = Catalog(tmp_path / "c.sqlite3")
    cat.put_document(_doc("a" * 64, source_type="talk", event_id="50058", tags=["legend"]))
    cat.put_document(_doc("b" * 64, source_type="paper", tags=[]))
    assert [d.id for d in cat.list_documents(source_type="talk")] == ["a" * 64]
    assert [d.id for d in cat.list_documents(tag="legend")] == ["a" * 64]
    assert [d.id for d in cat.list_documents(event_id="50058")] == ["a" * 64]


def test_jobs_are_taken_in_order(tmp_path):
    cat = Catalog(tmp_path / "c.sqlite3")
    first = cat.create_job("add", {"path": "/x"})
    second = cat.create_job("add", {"path": "/y"})
    assert cat.next_queued_job().id == first
    cat.start_job(first, os.getpid())
    assert cat.next_queued_job().id == second
    cat.finish_job(first, "completed")
    assert cat.get_job(first).status == "completed"


def test_only_one_worker_holds_the_lock(tmp_path):
    cat = Catalog(tmp_path / "c.sqlite3")
    assert cat.acquire_worker(os.getpid())
    assert not Catalog(tmp_path / "c.sqlite3").acquire_worker(os.getppid())
    cat.release_worker(os.getpid())
    assert cat.worker_pid() is None


def _dead_pid() -> int:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def test_a_lock_held_by_a_dead_process_is_taken_over(tmp_path):
    cat = Catalog(tmp_path / "c.sqlite3")
    dead = _dead_pid()
    assert not pid_alive(dead)
    assert cat.acquire_worker(dead)
    assert cat.acquire_worker(os.getpid())


def test_running_jobs_of_dead_workers_are_requeued(tmp_path):
    cat = Catalog(tmp_path / "c.sqlite3")
    job = cat.create_job("add", {"path": "/x"})
    cat.start_job(job, _dead_pid())
    assert cat.requeue_dead_jobs() == [job]
    requeued = cat.get_job(job)
    assert requeued.status == "queued" and requeued.interruptions == 1


def test_a_running_job_whose_worker_died_reads_as_interrupted(tmp_path):
    cat = Catalog(tmp_path / "c.sqlite3")
    dead = cat.create_job("add", {"path": "/x"})
    cat.start_job(dead, _dead_pid())
    alive = cat.create_job("add", {"path": "/y"})
    cat.start_job(alive, os.getpid())
    assert cat.get_job(dead).status == "interrupted"
    assert cat.get_job(alive).status == "running"
    assert [j.status for j in cat.recent_jobs()] == ["running", "interrupted"]


def test_job_files_track_stage_and_error(tmp_path):
    cat = Catalog(tmp_path / "c.sqlite3")
    job = cat.create_job("add", {"path": "/x"})
    cat.add_job_file(job, "/x/a.pdf")
    cat.update_job_file(job, "/x/a.pdf", stage="parse", status="failed", error="boom")
    [f] = cat.job_files(job)
    assert (f.stage, f.status, f.error) == ("parse", "failed", "boom")
