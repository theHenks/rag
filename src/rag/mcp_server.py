"""MCP tools over the index, the catalog and the job queue (stdio transport)."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict
from functools import cache, wraps
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from rag.ingest import ABORTING, Services, remove_document as _remove
from rag.jobs import spawn_worker, submit_add, submit_enrich
from rag.metadata import Overrides
from rag.search import Filters, context, search as _search

_INSTRUCTIONS = (
    "Search a personal library of physics PDFs: conference talks from Indico, papers, theses and notes. "
    "Use `search` first; cite results by title and page and give the link. Use `read_context` when a "
    "result is cut off. Adding documents runs in the background: `add_documents` returns a job id to "
    "pass to `ingest_status`."
)


_EXPECTED = (FileNotFoundError, KeyError, ValueError, *ABORTING)


def _reported(fn):
    """Turn errors a caller can act on into tool errors that carry their message;
    anything else reaches the client as an unexpected error."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except _EXPECTED as exc:
            message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
            raise ToolError(message) from exc
    return wrapper


def build_server(make_services: Callable[[], Services]) -> MCPServer:
    services = cache(make_services)
    server = MCPServer("rag", instructions=_INSTRUCTIONS)

    @server.tool()
    @_reported
    def search(query: str, limit: int = 8, source_type: str | None = None, tags: list[str] | None = None,
               event: str | None = None, authors: list[str] | None = None, date_from: str | None = None,
               date_to: str | None = None, document_id: str | None = None,
               max_per_document: int = 3) -> list[dict[str, Any]]:
        """Hybrid (semantic + keyword) search. source_type is one of talk, paper, thesis, note, other;
        event is an Indico event id; dates are ISO 8601. Returns chunks with title, pages and a link."""
        filters = Filters(source_type=source_type, tags=tags or [], event=event, authors=authors or [],
                          date_from=date_from, date_to=date_to, document_id=document_id)
        return [h.to_dict() for h in _search(services(), query, filters, limit=limit,
                                              max_per_document=max_per_document)]

    @server.tool()
    @_reported
    def read_context(chunk_id: str, before: int = 1, after: int = 1) -> list[dict[str, Any]]:
        """The chunks before and after a search result, in document order."""
        return [h.to_dict() for h in context(services(), chunk_id, before, after)]

    @server.tool()
    @_reported
    def get_document(document_id: str) -> dict[str, Any]:
        """A document's metadata, status and file paths."""
        doc = services().catalog.get_document(document_id)
        if doc is None:
            raise ValueError(f"no document {document_id}")
        return asdict(doc)

    @server.tool()
    @_reported
    def list_documents(source_type: str | None = None, tag: str | None = None,
                       event: str | None = None) -> list[dict[str, Any]]:
        """Catalogued documents, optionally filtered; each with id, title, type, date, event and status."""
        docs = services().catalog.list_documents(source_type=source_type, tag=tag, event_id=event)
        return [{"id": d.id, "title": d.title, "source_type": d.source_type, "date": d.date,
                 "event_title": d.event_title, "tags": d.tags, "status": d.status} for d in docs]

    @server.tool()
    @_reported
    def add_documents(path: str, source_type: str | None = None, tags: list[str] | None = None,
                      enrich: bool = False) -> dict[str, Any]:
        """Queue a PDF or a folder of PDFs for indexing in the background. enrich=true also describes
        figures with a vision model (about 30 s per figure). Returns a job id for ingest_status."""
        overrides = Overrides(source_type=source_type, tags=tuple(tags or ()))
        job_id = submit_add(services().catalog, Path(path), overrides, enrich=enrich)
        spawn_worker(services().config)
        return {"job_id": job_id}

    @server.tool()
    @_reported
    def ingest_status(job_id: int | None = None) -> dict[str, Any]:
        """Progress of one job (counts, the file being processed, every failure with its error),
        or the ten most recent jobs when job_id is omitted."""
        catalog = services().catalog
        if job_id is None:
            return {"jobs": [{"id": j.id, "kind": j.kind, "status": j.status, "path": j.args.get("path")}
                             for j in catalog.recent_jobs()]}
        job = catalog.get_job(job_id)
        if job is None:
            raise ValueError(f"no job {job_id}")
        files = catalog.job_files(job_id)
        return {
            "id": job.id, "status": job.status, "error": job.error, "interruptions": job.interruptions,
            "total": len(files), "done": sum(f.status == "done" for f in files),
            "in_progress": [{"path": f.path, "stage": f.stage} for f in files
                            if f.status == "pending" and f.stage != "queued"],
            "failed": [{"path": f.path, "stage": f.stage, "error": f.error} for f in files if f.status == "failed"],
            "not_indexed": [{"path": f.path, "reason": f.error} for f in files if f.status == "skipped"],
        }

    @server.tool()
    @_reported
    def enrich(document_ids: list[str]) -> dict[str, Any]:
        """Queue figure descriptions for already indexed documents. Returns a job id."""
        job_id = submit_enrich(services().catalog, document_ids)
        spawn_worker(services().config)
        return {"job_id": job_id}

    @server.tool()
    @_reported
    def remove_document(document_id: str) -> dict[str, Any]:
        """Remove a document from the catalog and the index. The PDF on disk is not touched."""
        if services().catalog.get_document(document_id) is None:
            raise ValueError(f"no document {document_id}")
        _remove(document_id, services())
        return {"removed": document_id}

    return server


def serve() -> None:
    from rag.config import load_config
    build_server(lambda: Services(load_config())).run(transport="stdio")
