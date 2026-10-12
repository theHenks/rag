import pytest

from rag.chunk import Chunk
from rag.config import Config
from rag.index import Index, QdrantUnavailable, SettingsMismatch, build_points, point_id


def _points(embedder, doc="d" * 64, texts=("alpha beta", "gamma delta")):
    chunks = [Chunk(i, "text", t, "", 1, 1, [[1, 0, 0, 1, 1]]) for i, t in enumerate(texts)]
    return build_points(doc, chunks, list(texts), {"document_id": doc, "title": "T"}, embedder)


def test_collection_is_created_and_settings_checked(config, fake_embedder):
    index = Index(config, fake_embedder.dense_size)
    index.ensure_collection()
    index.ensure_collection()
    with pytest.raises(SettingsMismatch, match="dense_size: collection has 8, configuration has 16"):
        Index(config, 16).ensure_collection()


def test_replace_document_overwrites_previous_points(config, fake_embedder):
    index = Index(config, fake_embedder.dense_size)
    index.ensure_collection()
    index.replace_document("d" * 64, _points(fake_embedder, texts=("a", "b", "c")))
    index.replace_document("d" * 64, _points(fake_embedder, texts=("x",)))
    assert index.count("d" * 64) == 1
    assert index.chunk(point_id("d" * 64, 0))["text"] == "x"


def test_neighbors_are_ordered(config, fake_embedder):
    index = Index(config, fake_embedder.dense_size)
    index.ensure_collection()
    index.replace_document("d" * 64, _points(fake_embedder, texts=("a", "b", "c", "d")))
    assert [p["text"] for p in index.neighbors("d" * 64, 2, before=1, after=1)] == ["b", "c", "d"]


def test_unreachable_qdrant_names_the_url(tmp_path):
    with pytest.raises(QdrantUnavailable, match="http://localhost:1"):
        Index(Config(data_dir=tmp_path, qdrant_url="http://localhost:1"), 8).ensure_collection()
