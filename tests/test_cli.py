from pathlib import Path

import pytest

from rag import cli
from rag.ingest import Services
from rag.parse import load_parsed

FIX = Path(__file__).parent / "fixtures" / "mineru"


@pytest.fixture
def env(config, fake_embedder, monkeypatch, tmp_path):
    services = Services(config, embedder=fake_embedder, parser=lambda pdf, doc_id: load_parsed(FIX / "paper"))
    monkeypatch.setattr(cli, "make_services", lambda: services)
    monkeypatch.setattr(cli, "spawn_worker", lambda config: None)
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "paper.pdf").write_bytes(b"%PDF x")
    return services, folder


def test_add_search_status_rm(env, capsys):
    services, folder = env
    assert cli.main(["add", str(folder), "--type", "paper", "--tag", "legend"]) == 0
    out = capsys.readouterr().out
    assert "1 indexed, 0 skipped, 0 failed" in out

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
