"""Ingest jobs: queued in the catalog, run one at a time by a single worker.

The worker is a detached process so a job outlives the MCP server or the
terminal that started it. Jobs run serially because MinerU, the embedding
model and the vision model share one GPU.
"""
from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

from rag.catalog import Catalog, Job
from rag.config import Config, config_path
from rag.ingest import ABORTING, Services, discover, enrich_document, ingest_file
from rag.metadata import Overrides

Progress = Callable[[str], None]


class WorkerBusy(RuntimeError):
    pass


def submit_add(catalog: Catalog, path: Path, overrides: Overrides, enrich: bool) -> int:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"no such file or directory: {path}")
    return catalog.create_job("add", {"path": str(path), "enrich": enrich, "overrides": overrides.to_args()})


def submit_enrich(catalog: Catalog, document_ids: list[str]) -> int:
    missing = [d for d in document_ids if catalog.get_document(d) is None]
    if missing:
        raise KeyError(f"no documents with ids {', '.join(missing)}")
    return catalog.create_job("enrich", {"document_ids": document_ids})


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _run_add(job: Job, services: Services, progress: Progress) -> bool:
    catalog = services.catalog
    root = Path(job.args["path"])
    overrides = Overrides.from_args(job.args["overrides"])
    pdfs, others = discover(root)
    for path in pdfs + others:
        catalog.add_job_file(job.id, str(path))
    for path in others:
        catalog.update_job_file(job.id, str(path), stage="discover", status="skipped",
                                error="not a PDF")
    done = {f.path for f in catalog.job_files(job.id) if f.status == "done"}
    failed = False
    for number, pdf in enumerate(pdfs, start=1):
        if str(pdf) in done:
            continue
        current = {"stage": "hash"}

        def stage(name: str) -> None:
            current["stage"] = name
            catalog.update_job_file(job.id, str(pdf), stage=name, status="pending")
            progress(f"[{number}/{len(pdfs)}] {pdf.name}: {name}")

        try:
            document_id, skipped = ingest_file(pdf, root=root if root.is_dir() else root.parent,
                                               overrides=overrides, enrich=job.args["enrich"],
                                               services=services, stage=stage)
        except ABORTING:
            raise
        except Exception as exc:
            failed = True
            catalog.update_job_file(job.id, str(pdf), stage=current["stage"], status="failed", error=_error(exc))
            progress(f"[{number}/{len(pdfs)}] {pdf.name}: failed at {current['stage']}: {_error(exc)}")
            continue
        catalog.update_job_file(job.id, str(pdf), stage="skipped-unchanged" if skipped else "write",
                                status="done", document_id=document_id)
    return failed


def _run_enrich(job: Job, services: Services, progress: Progress) -> bool:
    failed = False
    for document_id in job.args["document_ids"]:
        services.catalog.add_job_file(job.id, document_id)
        try:
            enrich_document(document_id, services,
                            stage=lambda name: progress(f"{document_id[:12]}: {name}"))
        except ABORTING:
            raise
        except Exception as exc:
            failed = True
            services.catalog.update_job_file(job.id, document_id, stage="enrich", status="failed",
                                             error=_error(exc))
            continue
        services.catalog.update_job_file(job.id, document_id, stage="write", status="done",
                                         document_id=document_id)
    return failed


def run_job(job: Job, services: Services, progress: Progress = lambda m: None) -> str:
    catalog = services.catalog
    catalog.start_job(job.id, os.getpid())
    try:
        runner = {"add": _run_add, "enrich": _run_enrich}[job.kind]
        failed = runner(job, services, progress)
    except ABORTING as exc:
        catalog.finish_job(job.id, "failed", _error(exc))
        progress(f"job {job.id} stopped: {_error(exc)}")
        return "failed"
    status = "completed_with_failures" if failed else "completed"
    catalog.finish_job(job.id, status)
    return status


def run_worker(services: Services, progress: Progress = lambda m: None) -> int:
    catalog = services.catalog
    pid = os.getpid()
    if not catalog.acquire_worker(pid):
        return 0
    ran = 0
    try:
        catalog.requeue_dead_jobs()
        while (job := catalog.next_queued_job()) is not None:
            run_job(job, services, progress)
            ran += 1
    finally:
        catalog.release_worker(pid)
    return ran


def run_foreground(job_id: int, services: Services, progress: Progress) -> str:
    catalog = services.catalog
    pid = os.getpid()
    if not catalog.acquire_worker(pid):
        raise WorkerBusy(
            f"a background worker (pid {catalog.worker_pid()}) is running; "
            f"job {job_id} is queued and will run after it, or use --background")
    try:
        return run_job(catalog.get_job(job_id), services, progress)
    finally:
        catalog.release_worker(pid)


def spawn_worker(config: Config) -> None:
    config.data_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "RAG_CONFIG": str(config_path())}
    with config.log_path.open("ab") as log:
        subprocess.Popen([sys.executable, "-m", "rag", "worker"], stdin=subprocess.DEVNULL, stdout=log,
                         stderr=subprocess.STDOUT, start_new_session=True, env=env)
