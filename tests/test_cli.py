import re
from pathlib import Path

import pytest
from rich.console import Console

from rag import cli
from rag.catalog import pid_alive
from rag.ingest import Services
from rag.parse import load_parsed

FIX = Path(__file__).parent / "fixtures" / "mineru"


@pytest.fixture
def env(config, fake_embedder, monkeypatch, tmp_path):
    services = Services(config, embedder=fake_embedder, parser=lambda pdf, doc_id: load_parsed(FIX / "paper"))
    monkeypatch.setattr(cli, "make_services", lambda: services)
    monkeypatch.setattr(cli, "spawn_worker", lambda config: None)
    monkeypatch.setattr(cli, "console", Console(width=200, highlight=False))
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "paper.pdf").write_bytes(b"%PDF x")
    return services, folder


def test_add_search_status_rm(env, capsys):
    services, folder = env
    assert cli.main(["add", str(folder), "--type", "paper", "--tag", "legend"]) == 0
    out = capsys.readouterr().out
    assert "completed" in out and re.search(r"^files\s+1 indexed\s*$", out, re.M)

    assert cli.main(["search", "Majorana half-life", "--type", "paper"]) == 0
    out = capsys.readouterr().out
    assert "Final Result of the Majorana" in out and "p." in out

    assert cli.main(["status"]) == 0
    assert "completed" in capsys.readouterr().out

    [doc] = services.catalog.list_documents()
    assert cli.main(["rm", doc.id[:12]]) == 0
    assert services.catalog.list_documents() == []


def test_background_add_only_queues(env, capsys):
    services, folder = env
    assert cli.main(["add", str(folder), "--background"]) == 0
    assert "queued as job" in capsys.readouterr().out
    assert services.catalog.next_queued_job() is not None


def test_errors_exit_nonzero_with_a_message(env, capsys):
    services, folder = env
    assert cli.main(["add", str(folder / "missing")]) == 2
    assert "no such file or directory" in capsys.readouterr().err
    assert cli.main(["rm", "ffff"]) == 2
    assert "no document" in capsys.readouterr().err


def test_a_job_with_failed_files_exits_nonzero(env, capsys, monkeypatch):
    services, folder = env
    (folder / "bad.pdf").write_bytes(b"BROKEN")
    good = services.parser

    def parser(pdf, doc_id):
        if pdf.read_bytes().startswith(b"BROKEN"):
            raise RuntimeError("cannot parse")
        return good(pdf, doc_id)
    services.parser = parser
    assert cli.main(["add", str(folder)]) == 1
    assert "1 failed" in capsys.readouterr().out


def test_status_of_an_unknown_job_is_an_error(env, capsys):
    assert cli.main(["status", "999"]) == 2
    assert "no job 999" in capsys.readouterr().err


def test_list_shows_documents_and_filters(env, capsys):
    services, folder = env
    cli.main(["add", str(folder), "--type", "paper", "--tag", "legend"])
    capsys.readouterr()
    [doc] = services.catalog.list_documents()

    assert cli.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "Final Result of the Majorana" in out and doc.id[:12] in out and "paper" in out
    assert "1 document" in out

    assert cli.main(["list", "--tag", "legend"]) == 0
    assert doc.id[:12] in capsys.readouterr().out
    assert cli.main(["list", "--type", "talk"]) == 0
    assert "0 documents" in capsys.readouterr().out


def test_status_reports_a_job_whose_process_died_as_interrupted(env, capsys):
    services, folder = env
    job = services.catalog.create_job("add", {"path": str(folder)})
    dead = 2 ** 22 + 12345
    assert not pid_alive(dead)
    services.catalog.start_job(job, dead)

    assert cli.main(["status"]) == 0
    assert re.search(rf"^\s*{job}\s+add\s+interrupted\s", capsys.readouterr().out, re.M)
    cli.main(["status", str(job)])
    out = capsys.readouterr().out
    assert f"Job {job} (add)  interrupted" in out and "resumes" in out
