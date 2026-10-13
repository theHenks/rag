"""The `rag` command line."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.table import Table
from rich.text import Text

from rag.config import load_config
from rag.ingest import ABORTING, Services, rebuild, remove_document
from rag.jobs import WorkerBusy, run_foreground, run_worker, spawn_worker, submit_add, submit_enrich
from rag.metadata import SOURCE_TYPES, Overrides
from rag.search import Filters, search

console = Console(highlight=False)

_STATUS_STYLE = {"completed": "green", "indexed": "green", "running": "cyan", "queued": "cyan",
                 "pending": "cyan", "interrupted": "yellow", "completed_with_failures": "yellow",
                 "failed": "red"}


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
    return 0 if status == "completed" else 1


def _home(path: str) -> str:
    home = str(Path.home())
    return "~" + path[len(home):] if path.startswith(home + "/") else path


def _styled(status: str) -> Text:
    return Text(status.replace("_", " "), style=_STATUS_STYLE.get(status, ""))


def _counts(files) -> dict[str, int]:
    return {
        "indexed": sum(f.status == "done" and f.stage != "skipped-unchanged" for f in files),
        "unchanged": sum(f.stage == "skipped-unchanged" for f in files),
        "failed": sum(f.status == "failed" for f in files),
        "not PDF": sum(f.status == "skipped" for f in files),
        "waiting": sum(f.status == "pending" for f in files),
    }


def _summary(counts: dict[str, int]) -> str:
    return " · ".join(f"{n} {label}" for label, n in counts.items() if n or label == "indexed")


def _print_job(services: Services, job_id: int) -> int:
    job = services.catalog.get_job(job_id)
    if job is None:
        raise KeyError(f"no job {job_id}")
    files = services.catalog.job_files(job_id)
    folder = job.args.get("path")

    console.print(Text.assemble((f"Job {job.id}", "bold"), f" ({job.kind})  ", _styled(job.status)))
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim")
    grid.add_column()
    if folder:
        grid.add_row("folder", _home(folder))
    grid.add_row("files", _summary(_counts(files)))
    grid.add_row("started", (job.started_at or "not yet").replace("T", " ").replace("+00:00", " UTC"))
    if job.finished_at:
        grid.add_row("finished", job.finished_at.replace("T", " ").replace("+00:00", " UTC"))
    if job.interruptions:
        grid.add_row("resumed", f"after {job.interruptions} interruption(s)")
    if job.error:
        grid.add_row("error", Text(job.error, style="red"))
    console.print(grid)
    if job.status == "interrupted":
        console.print(Text("Its process stopped; the next worker resumes it "
                           "(started by any `rag add` or `rag enrich`).", style="yellow"))

    def shown(path: str) -> str:
        return path[len(folder) + 1:] if folder and path.startswith(folder + "/") else _home(path)

    problems = [f for f in files if f.status in ("failed", "skipped")]
    if problems:
        table = Table(box=box.SIMPLE_HEAD, show_edge=False)
        table.add_column("file", overflow="fold")
        table.add_column("outcome")
        table.add_column("reason", overflow="fold")
        for f in problems:
            outcome = Text(f"failed at {f.stage}", style="red") if f.status == "failed" else Text("not indexed")
            table.add_row(shown(f.path), outcome, f.error or "")
        console.print(table)
    return 0


def _cmd_search(args, services: Services) -> int:
    filters = Filters(source_type=args.type, tags=args.tag, event=args.event, authors=args.author,
                      date_from=args.since, date_to=args.until)
    hits = search(services, args.query, filters, limit=args.limit)
    if not hits:
        console.print("No results.")
    for n, hit in enumerate(hits, start=1):
        pages = f"p. {hit.page_start}" if hit.page_start == hit.page_end else f"pp. {hit.page_start}–{hit.page_end}"
        details = " · ".join(x for x in (hit.source_type, (hit.date or "")[:10], pages, hit.event_title) if x)
        snippet = " ".join(hit.text.split())
        if len(snippet) > 300:
            snippet = snippet[:300].rsplit(" ", 1)[0] + " …"
        console.print(Text.assemble((f"{n:>2}  ", "bold"), (hit.title, "bold")))
        grid = Table.grid(padding=(0, 0))
        grid.add_column(width=4)
        grid.add_column(overflow="fold")
        grid.add_row("", Text(details, style="dim"))
        if hit.link:
            grid.add_row("", Text(hit.link, style="blue"))
        grid.add_row("", snippet)
        console.print(grid)
        console.print()
    return 0


def _cmd_list(args, services: Services) -> int:
    docs = services.catalog.list_documents(source_type=args.type, tag=args.tag, event_id=args.event)
    groups: dict[str, list] = {}
    for doc in docs:
        groups.setdefault(doc.event_title or "", []).append(doc)
    show_status = any(d.status != "indexed" for d in docs)
    for event, members in sorted(groups.items(), key=lambda g: g[0] == ""):
        heading = event or ("Other documents" if len(groups) > 1 else "")
        if heading:
            console.print(Text.assemble((heading, "bold"), (f"  {len(members)}", "dim")))
        table = Table(box=box.SIMPLE_HEAD, show_edge=False)
        table.add_column("id", style="dim", no_wrap=True)
        table.add_column("date", no_wrap=True)
        table.add_column("type", no_wrap=True)
        table.add_column("title", ratio=3)
        table.add_column("authors", ratio=1, style="dim")
        if show_status:
            table.add_column("status", no_wrap=True)
        for doc in members:
            authors = doc.authors[0] + (" et al." if len(doc.authors) > 1 else "") if doc.authors else ""
            row = [doc.id[:12], (doc.date or "")[:10], doc.source_type, doc.title, authors]
            if show_status:
                row.append(_styled(doc.status))
            table.add_row(*row)
        console.print(table)
        console.print()
    console.print(f"{len(docs)} document{'' if len(docs) == 1 else 's'}")
    return 0


def _cmd_status(args, services: Services) -> int:
    if args.job is not None:
        return _print_job(services, args.job)
    catalog = services.catalog
    worker = catalog.worker_pid()
    console.print(Text.assemble(("Worker  ", "bold"),
                                (f"running (pid {worker})", "cyan") if worker else ("idle", "dim")))

    jobs = catalog.recent_jobs()
    if jobs:
        table = Table(box=box.SIMPLE_HEAD, show_edge=False, title="Recent jobs", title_justify="left",
                      title_style="bold")
        table.add_column("job", justify="right")
        table.add_column("kind")
        table.add_column("status", no_wrap=True)
        table.add_column("files")
        table.add_column("folder", overflow="fold")
        for job in jobs:
            path = job.args.get("path")
            table.add_row(str(job.id), job.kind, _styled(job.status), _summary(_counts(catalog.job_files(job.id))),
                          _home(path) if path else f"{len(job.args.get('document_ids', []))} document(s)")
        console.print(table)

    docs = catalog.list_documents()
    console.print(Text.assemble(("Documents  ", "bold"),
                                f"{len(docs)} catalogued, {sum(d.status == 'indexed' for d in docs)} indexed"))
    problems = Table(box=box.SIMPLE_HEAD, show_edge=False)
    problems.add_column("id", style="dim", no_wrap=True)
    problems.add_column("title")
    problems.add_column("problem", overflow="fold")
    for doc in docs:
        if not any(Path(p).exists() for p in doc.paths):
            problems.add_row(doc.id[:12], doc.title, Text("missing on disk: " + ", ".join(map(_home, doc.paths)),
                                                          style="yellow"))
        if doc.status == "failed":
            problems.add_row(doc.id[:12], doc.title, Text(doc.error or "failed", style="red"))
    if problems.row_count:
        console.print(problems)
    if any(j.status == "interrupted" for j in jobs):
        console.print(Text("Interrupted jobs resume when the next worker starts "
                           "(any `rag add` or `rag enrich`).", style="yellow"))
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

    p = sub.add_parser("list", help="catalogued documents, oldest first")
    p.add_argument("--type", choices=SOURCE_TYPES)
    p.add_argument("--tag")
    p.add_argument("--event", help="Indico event id")
    p.set_defaults(fn=_cmd_list)

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
