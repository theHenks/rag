import pytest

from rag.config import Config
from rag.embed import ModelEmbedder


def test_bm25_vectors_are_sparse_and_share_term_ids(fake_embedder):
    [doc] = fake_embedder.sparse_documents(["thallium 208 background in germanium"])
    query = fake_embedder.sparse_query("thallium")
    assert set(query.indices) <= set(doc.indices)


@pytest.mark.slow
def test_real_model_dimensions_and_query_prompt(tmp_path):
    embedder = ModelEmbedder(Config(data_dir=tmp_path))
    assert embedder.dense_size == 1024
    [doc] = embedder.dense_documents(["Neutrinoless double beta decay of germanium-76"])
    query = embedder.dense_query("0vbb search with Ge-76")
    other = embedder.dense_query("recipe for tiramisu")
    dot = lambda a, b: sum(x * y for x, y in zip(a, b))
    assert dot(doc, query) > dot(doc, other)
    assert embedder.count_tokens("hello world") >= 2
