from pathlib import Path

import pytest
from mcp import Client

from rag.ingest import Services
from rag.jobs import run_worker
from rag.mcp_server import build_server
from rag.parse import load_parsed

FIX = Path(__file__).parent / "fixtures" / "mineru"


@pytest.fixture
def setup(config, fake_embedder, monkeypatch, tmp_path):
    services = Services(config, embedder=fake_embedder, parser=lambda pdf, d: load_parsed(FIX / "paper"))
    monkeypatch.setattr("rag.mcp_server.spawn_worker", lambda config: None)
    folder = tmp_path / "docs"
    folder.mkdir()
    (folder / "paper.pdf").write_bytes(b"%PDF x")
    return services, folder, build_server(lambda: services)


async def test_tool_list(setup):
    _, _, server = setup
    async with Client(server) as client:
        names = {t.name for t in (await client.list_tools()).tools}
    assert names == {"search", "read_context", "get_document", "list_documents", "add_documents",
                     "ingest_status", "enrich", "remove_document"}


async def test_add_then_search_then_read_context(setup):
    services, folder, server = setup
    async with Client(server) as client:
        added = await client.call_tool("add_documents", {"path": str(folder), "source_type": "paper"})
        job_id = added.structured_content["job_id"]
        run_worker(services)
        status = (await client.call_tool("ingest_status", {"job_id": job_id})).structured_content
        assert status["status"] == "completed" and status["failed"] == []

        found = (await client.call_tool("search", {"query": "Majorana half-life limit"})).structured_content
        first = found["result"][0]
        assert first["title"].startswith("Final Result of the Majorana")
        assert first["page_start"] >= 1 and first["link"].startswith("file://")

        around = (await client.call_tool("read_context", {"chunk_id": first["chunk_id"]})).structured_content
        assert first["chunk_id"] in [h["chunk_id"] for h in around["result"]]

        doc = (await client.call_tool("get_document", {"document_id": first["document_id"]})).structured_content
        assert doc["page_count"] == 8 and doc["status"] == "indexed"


async def test_bad_input_is_a_tool_error_with_a_message(setup):
    _, folder, server = setup
    async with Client(server) as client:
        result = await client.call_tool("add_documents", {"path": str(folder / "missing")})
    assert result.is_error and "no such file or directory" in result.content[0].text
