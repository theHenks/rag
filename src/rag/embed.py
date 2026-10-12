"""Dense and sparse (BM25) embeddings for chunks and queries."""
from __future__ import annotations

from qdrant_client import models

from rag.config import Config


class Embedder:
    dense_size: int

    def count_tokens(self, text: str) -> int:
        raise NotImplementedError

    def dense_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def dense_query(self, text: str) -> list[float]:
        raise NotImplementedError

    def sparse_documents(self, texts: list[str]) -> list[models.SparseVector]:
        return [_sparse(e) for e in self._bm25.embed(texts)]

    def sparse_query(self, text: str) -> models.SparseVector:
        return _sparse(next(iter(self._bm25.query_embed(text))))


def _sparse(embedding) -> models.SparseVector:
    return models.SparseVector(indices=embedding.indices.tolist(), values=embedding.values.tolist())


def bm25(config: Config):
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(config.sparse_model, cache_dir=str(config.models_dir))


class ModelEmbedder(Embedder):
    def __init__(self, config: Config):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(config.dense_model, device=config.device)
        self.dense_size = self._model.get_embedding_dimension()
        self._bm25 = bm25(config)

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer(text)["input_ids"])

    def dense_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True, batch_size=16).tolist()

    def dense_query(self, text: str) -> list[float]:
        return self._model.encode([text], prompt_name="query", normalize_embeddings=True)[0].tolist()
