"""The authoritative record of documents and ingest jobs, in SQLite.

Every process (CLI, MCP server, worker) opens its own connection; WAL mode
lets readers proceed while the worker writes.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, source_type TEXT NOT NULL,
    authors TEXT NOT NULL, date TEXT, tags TEXT NOT NULL,
    event_id TEXT, event_title TEXT, session TEXT, indico_url TEXT,
    enrich INTEGER NOT NULL, status TEXT NOT NULL, settings TEXT,
    page_count INTEGER, error TEXT, updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS paths (
    path TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, args TEXT NOT NULL,
    status TEXT NOT NULL, worker_pid INTEGER, error TEXT,
    interruptions INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT
);
CREATE TABLE IF NOT EXISTS job_files (
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    path TEXT NOT NULL, document_id TEXT, stage TEXT NOT NULL,
    status TEXT NOT NULL, error TEXT,
    PRIMARY KEY (job_id, path)
);
CREATE TABLE IF NOT EXISTS worker (
    slot INTEGER PRIMARY KEY CHECK (slot = 1), pid INTEGER NOT NULL, started_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@dataclass
class Document:
    id: str
    title: str
    source_type: str
    authors: list[str]
    date: str | None
    tags: list[str]
    event_id: str | None
    event_title: str | None
    session: str | None
    indico_url: str | None
    enrich: bool
    status: str
    settings: str | None
    page_count: int | None
    error: str | None
    paths: list[str] = field(default_factory=list)


@dataclass
class Job:
    id: int
    kind: str
    args: dict
    status: str
    worker_pid: int | None
    error: str | None
    interruptions: int
    created_at: str
    started_at: str | None
    finished_at: str | None


@dataclass
class JobFile:
    job_id: int
    path: str
    document_id: str | None
    stage: str
    status: str
    error: str | None


class Catalog:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # The MCP server calls tools from worker threads; sqlite3 serializes
        # access to one connection, so sharing it across threads is safe.
        self.db = sqlite3.connect(path, timeout=30, isolation_level=None, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(_SCHEMA)

    # documents

    def put_document(self, doc: Document) -> None:
        """Insert or update a document. An upsert rather than INSERT OR REPLACE,
        which would delete the row and cascade to its paths."""
        self.db.execute(
            """INSERT INTO documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET title=excluded.title, source_type=excluded.source_type,
                 authors=excluded.authors, date=excluded.date, tags=excluded.tags,
                 event_id=excluded.event_id, event_title=excluded.event_title, session=excluded.session,
                 indico_url=excluded.indico_url, enrich=excluded.enrich, status=excluded.status,
                 settings=excluded.settings, page_count=excluded.page_count, error=excluded.error,
                 updated_at=excluded.updated_at""",
            (doc.id, doc.title, doc.source_type, json.dumps(doc.authors), doc.date,
             json.dumps(doc.tags), doc.event_id, doc.event_title, doc.session, doc.indico_url,
             int(doc.enrich), doc.status, doc.settings, doc.page_count, doc.error, _now()),
        )

    def _document(self, row: sqlite3.Row) -> Document:
        paths = [r["path"] for r in self.db.execute(
            "SELECT path FROM paths WHERE document_id = ? ORDER BY path", (row["id"],))]
        return Document(
            id=row["id"], title=row["title"], source_type=row["source_type"],
            authors=json.loads(row["authors"]), date=row["date"], tags=json.loads(row["tags"]),
            event_id=row["event_id"], event_title=row["event_title"], session=row["session"],
            indico_url=row["indico_url"], enrich=bool(row["enrich"]), status=row["status"],
            settings=row["settings"], page_count=row["page_count"], error=row["error"], paths=paths,
        )

    def get_document(self, id: str) -> Document | None:
        row = self.db.execute("SELECT * FROM documents WHERE id = ?", (id,)).fetchone()
        return self._document(row) if row else None

    def list_documents(self, *, source_type: str | None = None, tag: str | None = None,
                       event_id: str | None = None) -> list[Document]:
        query, args = "SELECT * FROM documents WHERE 1=1", []
        if source_type:
            query += " AND source_type = ?"
            args.append(source_type)
        if event_id:
            query += " AND event_id = ?"
            args.append(event_id)
        if tag:
            query += " AND EXISTS (SELECT 1 FROM json_each(documents.tags) WHERE value = ?)"
            args.append(tag)
        query += " ORDER BY date IS NULL, date, title"
        return [self._document(row) for row in self.db.execute(query, args)]

    def delete_document(self, id: str) -> None:
        self.db.execute("DELETE FROM documents WHERE id = ?", (id,))

    def set_status(self, id: str, status: str, *, settings: str | None = None,
                   error: str | None = None) -> None:
        self.db.execute(
            "UPDATE documents SET status = ?, settings = COALESCE(?, settings), error = ?, updated_at = ? WHERE id = ?",
            (status, settings, error, _now(), id),
        )

    def add_path(self, path: str, document_id: str) -> str | None:
        previous = self.documents_for_path(path)
        self.db.execute("INSERT OR REPLACE INTO paths VALUES (?, ?)", (path, document_id))
        return previous if previous != document_id else None

    def documents_for_path(self, path: str) -> str | None:
        row = self.db.execute("SELECT document_id FROM paths WHERE path = ?", (path,)).fetchone()
        return row["document_id"] if row else None

    # jobs

    def _job(self, row: sqlite3.Row) -> Job:
        return Job(id=row["id"], kind=row["kind"], args=json.loads(row["args"]), status=row["status"],
                   worker_pid=row["worker_pid"], error=row["error"], interruptions=row["interruptions"],
                   created_at=row["created_at"], started_at=row["started_at"], finished_at=row["finished_at"])

    def create_job(self, kind: str, args: dict) -> int:
        cur = self.db.execute(
            "INSERT INTO jobs (kind, args, status, created_at) VALUES (?, ?, 'queued', ?)",
            (kind, json.dumps(args), _now()),
        )
        return cur.lastrowid

    def get_job(self, id: int) -> Job | None:
        row = self.db.execute("SELECT * FROM jobs WHERE id = ?", (id,)).fetchone()
        return self._job(row) if row else None

    def recent_jobs(self, limit: int = 10) -> list[Job]:
        rows = self.db.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,))
        return [self._job(row) for row in rows]

    def next_queued_job(self) -> Job | None:
        row = self.db.execute("SELECT * FROM jobs WHERE status = 'queued' ORDER BY id LIMIT 1").fetchone()
        return self._job(row) if row else None

    def start_job(self, id: int, pid: int) -> None:
        self.db.execute(
            "UPDATE jobs SET status = 'running', worker_pid = ?, started_at = COALESCE(started_at, ?) WHERE id = ?",
            (pid, _now(), id),
        )

    def finish_job(self, id: int, status: str, error: str | None = None) -> None:
        self.db.execute("UPDATE jobs SET status = ?, error = ?, finished_at = ? WHERE id = ?",
                        (status, error, _now(), id))

    def requeue_dead_jobs(self) -> list[int]:
        """Put back running jobs whose worker process no longer exists."""
        dead = [row["id"] for row in self.db.execute("SELECT id, worker_pid FROM jobs WHERE status = 'running'")
                if row["worker_pid"] is None or not pid_alive(row["worker_pid"])]
        for id in dead:
            self.db.execute(
                "UPDATE jobs SET status = 'queued', worker_pid = NULL, interruptions = interruptions + 1 WHERE id = ?",
                (id,),
            )
        return dead

    def add_job_file(self, job_id: int, path: str) -> None:
        self.db.execute(
            "INSERT OR IGNORE INTO job_files (job_id, path, stage, status) VALUES (?, ?, 'queued', 'pending')",
            (job_id, path),
        )

    def update_job_file(self, job_id: int, path: str, *, stage: str, status: str,
                        document_id: str | None = None, error: str | None = None) -> None:
        self.db.execute(
            """UPDATE job_files SET stage = ?, status = ?, document_id = COALESCE(?, document_id), error = ?
               WHERE job_id = ? AND path = ?""",
            (stage, status, document_id, error, job_id, path),
        )

    def job_files(self, job_id: int) -> list[JobFile]:
        rows = self.db.execute("SELECT * FROM job_files WHERE job_id = ? ORDER BY path", (job_id,))
        return [JobFile(**dict(row)) for row in rows]

    # worker lock

    def acquire_worker(self, pid: int) -> bool:
        """Take the single worker slot; a slot held by a dead process is taken over."""
        self.db.execute("BEGIN IMMEDIATE")
        try:
            row = self.db.execute("SELECT pid FROM worker WHERE slot = 1").fetchone()
            if row and row["pid"] != pid and pid_alive(row["pid"]):
                self.db.execute("ROLLBACK")
                return False
            self.db.execute("INSERT OR REPLACE INTO worker VALUES (1, ?, ?)", (pid, _now()))
            self.db.execute("COMMIT")
            return True
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def release_worker(self, pid: int) -> None:
        self.db.execute("DELETE FROM worker WHERE slot = 1 AND pid = ?", (pid,))

    def worker_pid(self) -> int | None:
        row = self.db.execute("SELECT pid FROM worker WHERE slot = 1").fetchone()
        if row is None or not pid_alive(row["pid"]):
            return None
        return row["pid"]
