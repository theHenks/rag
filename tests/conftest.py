import hashlib
import os
import uuid

import httpx
import pytest
from qdrant_client import QdrantClient

from rag.config import Config
from rag.embed import Embedder, bm25

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


class FakeEmbedder(Embedder):
    """Deterministic bag-of-words dense vectors; BM25 is the real one."""
    dense_size = 8

    def __init__(self, config: Config):
        self._bm25 = bm25(config)

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _vector(self, text: str) -> list[float]:
        v = [0.0] * 8
        for word in text.lower().split():
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 8] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def dense_documents(self, texts):
        return [self._vector(t) for t in texts]

    def dense_query(self, text):
        return self._vector(text)


@pytest.fixture
def fake_embedder(config) -> FakeEmbedder:
    return FakeEmbedder(config)
