"""The `rag` command line."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rag.config import load_config
from rag.ingest import ABORTING, Services, rebuild, remove_document
from rag.jobs import WorkerBusy, run_foreground, run_worker, spawn_worker, submit_add, submit_enrich
from rag.metadata import SOURCE_TYPES, Overrides
from rag.search import Filters, search


def make_services() -> Services:
    return Services(load_config())


def _resolve_document(services: Services, prefix: str) -> str:
    matches = [d.id for d in services.catalog.list_documents() if d.id.startswith(prefix)]
    if len(matches) != 1:
        raise KeyError(f"no document with id {prefix!r}" if not matches
                       else f"id prefix {prefix!r} is ambiguous ({len(matches)} documents)")
    return matches[0]


def _cmd_add(args, services: Services) -> int:
    overrides = Overrides(source_type=args.type, tags=tuple(args.tag), title=args.title,
                          authors=tuple(args.author), date=args.date)
    job = submit_add(services.catalog, Path(args.path), overrides, enrich=args.enrich)
    if args.background:
        spawn_worker(services.config)
        print(f"queued as job {job}; follow it with `rag status {job}`")
        return 0
    status = run_foreground(job, services, progress=lambda m: print(m, file=sys.stderr))
    _print_job(services, job)
    return 1 if status == "failed" else 0


def _print_job(services: Services, job_id: int) -> int:
    job = services.catalog.get_job(job_id)
    files = services.catalog.job_files(job_id)
    indexed = sum(f.status == "done" and f.stage != "skipped-unchanged" for f in files)
    unchanged = sum(f.stage == "skipped-unchanged" for f in files)
    failed = [f for f in files if f.status == "failed"]
    others = [f for f in files if f.status == "skipped"]
    print(f"job {job.id} ({job.kind}) {job.status}: {indexed} indexed, {unchanged} skipped, {len(failed)} failed")
    if job.interruptions:
        print(f"  resumed after {job.interruptions} interruption(s)")
    if job.error:
        print(f"  error: {job.error}")
    for f in failed:
        print(f"  FAILED {f.path} at {f.stage}: {f.error}")
    for f in others:
        print(f"  not indexed: {f.path} ({f.error})")
    return 0


def _cmd_search(args, services: Services) -> int:
    filters = Filters(source_type=args.type, tags=args.tag, event=args.event, authors=args.author,
                      date_from=args.since, date_to=args.until)
    for n, hit in enumerate(search(services, args.query, filters, limit=args.limit), start=1):
        pages = f"p. {hit.page_start}" if hit.page_start == hit.page_end else f"pp. {hit.page_start}–{hit.page_end}"
        print(f"{n}. {hit.title} — {pages} [{hit.source_type}] {hit.link or ''}")
        print("   " + hit.text[:300].replace("\n", " "))
    return 0


def _cmd_status(args, services: Services) -> int:
    if args.job is not None:
        return _print_job(services, args.job)
    worker = services.catalog.worker_pid()
    print(f"worker: {'running (pid %d)' % worker if worker else 'idle'}")
    for job in services.catalog.recent_jobs():
        print(f"job {job.id} {job.kind} {job.status} {job.args.get('path', '')}")
    docs = services.catalog.list_documents()
    print(f"{len(docs)} documents, {sum(d.status == 'indexed' for d in docs)} indexed")
    for doc in docs:
        if not any(Path(p).exists() for p in doc.paths):
            print(f"  missing on disk: {doc.title} ({doc.id[:12]}) {doc.paths}")
        if doc.status == "failed":
            print(f"  failed: {doc.title} ({doc.id[:12]}): {doc.error}")
    return 0


def _cmd_enrich(args, services: Services) -> int:
    ids = [_resolve_document(services, p) for p in args.documents]
    job = submit_enrich(services.catalog, ids)
    spawn_worker(services.config)
    print(f"queued as job {job}; follow it with `rag status {job}`")
    return 0


def _cmd_rm(args, services: Services) -> int:
    document_id = _resolve_document(services, args.document)
    title = services.catalog.get_document(document_id).title
    remove_document(document_id, services)
    print(f"removed {title} ({document_id[:12]})")
    return 0


def _cmd_rebuild(args, services: Services) -> int:
    failures = rebuild(services, report=lambda m: print(m, file=sys.stderr))
    for document_id, error in failures:
        print(f"FAILED {document_id[:12]}: {error}")
    return 1 if failures else 0


def _cmd_doctor(args, services: Services) -> int:
    from rag.doctor import run_checks
    results = run_checks(services.config)
    for name, ok, detail in results:
        print(f"{'ok  ' if ok else 'FAIL'} {name}: {detail}")
    return 0 if all(ok for name, ok, _ in results if not name.startswith("ollama")) else 1


def _cmd_eval(args, services: Services) -> int:
    from rag.evaluate import evaluate, load_questions
    path = Path(args.questions) if args.questions else services.config.data_dir / "eval.jsonl"
    report = evaluate(services, load_questions(path))
    print("  ".join(f"recall@{k} {v:.2f}" for k, v in report.recall.items()) + f"  MRR {report.mrr:.2f}")
    for question in report.misses:
        print(f"  miss: {question}")
    return 0


def _cmd_mcp(args, services: Services) -> int:
    from rag.mcp_server import serve
    serve()
    return 0


def _cmd_worker(args, services: Services) -> int:
    run_worker(services, progress=lambda m: print(m, flush=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag", description="Hybrid search over a PDF library.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("add", help="index a PDF or a folder of PDFs")
    p.add_argument("path")
    p.add_argument("--type", choices=SOURCE_TYPES)
    p.add_argument("--tag", action="append", default=[])
    p.add_argument("--title")
    p.add_argument("--author", action="append", default=[])
    p.add_argument("--date", help="ISO 8601 date")
    p.add_argument("--enrich", action="store_true", help="describe figures with the vision model")
    p.add_argument("--background", action="store_true", help="queue the job and return")
    p.set_defaults(fn=_cmd_add)

    p = sub.add_parser("search", help="search the index")
    p.add_argument("query")
    p.add_argument("--type", choices=SOURCE_TYPES)
    p.add_argument("--tag", action="append", default=[])
    p.add_argument("--event", help="Indico event id")
    p.add_argument("--author", action="append", default=[])
    p.add_argument("--since", help="ISO 8601 date")
    p.add_argument("--until", help="ISO 8601 date")
    p.add_argument("--limit", type=int, default=8)
    p.set_defaults(fn=_cmd_search)

    p = sub.add_parser("status", help="jobs, failures, and documents missing on disk")
    p.add_argument("job", nargs="?", type=int)
    p.set_defaults(fn=_cmd_status)

    p = sub.add_parser("enrich", help="add figure descriptions to indexed documents")
    p.add_argument("documents", nargs="+", help="document ids or unique prefixes")
    p.set_defaults(fn=_cmd_enrich)

    p = sub.add_parser("rm", help="remove a document from the catalog and the index")
    p.add_argument("document", help="document id or unique prefix")
    p.set_defaults(fn=_cmd_rm)

    sub.add_parser("rebuild", help="re-index everything from the MinerU cache").set_defaults(fn=_cmd_rebuild)
    sub.add_parser("doctor", help="check Qdrant, MinerU, Ollama and disk").set_defaults(fn=_cmd_doctor)
    p = sub.add_parser("eval", help="measure retrieval on a question set")
    p.add_argument("questions", nargs="?", help="JSON Lines file (default: eval.jsonl in the data directory)")
    p.set_defaults(fn=_cmd_eval)

    sub.add_parser("mcp", help="run the MCP server on stdio").set_defaults(fn=_cmd_mcp)
    sub.add_parser("worker", help="run queued jobs (started by add and enrich)").set_defaults(fn=_cmd_worker)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    services = make_services()
    try:
        return args.fn(args, services)
    except (FileNotFoundError, KeyError, ValueError, WorkerBusy) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"error: {message}", file=sys.stderr)
        return 2
    except ABORTING as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
