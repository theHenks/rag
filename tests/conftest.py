import os
import uuid

import httpx
import pytest
from qdrant_client import QdrantClient

from rag.config import Config

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture(scope="session")
def qdrant_url() -> str:
    url = os.environ.get("RAG_TEST_QDRANT_URL", "http://localhost:6333")
    try:
        httpx.get(url, timeout=3).raise_for_status()
    except httpx.HTTPError as exc:
        pytest.fail(f"Qdrant is not reachable at {url} ({exc}); run `scripts/qdrant start` in the rag repository")
    return url


@pytest.fixture
def config(tmp_path, qdrant_url) -> Config:
    collection = f"rag-test-{uuid.uuid4().hex[:8]}"
    yield Config(data_dir=tmp_path / "data", qdrant_url=qdrant_url, collection=collection)
    client = QdrantClient(url=qdrant_url)
    if client.collection_exists(collection):
        client.delete_collection(collection)
