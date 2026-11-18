from typing import Any

import pytest

from app.rag.embeddings import (
    EmbeddingDimensionMismatchError,
    EmbeddingError,
    EmbeddingService,
)


class FakeEmbeddingsClient:
    def __init__(self, dimensions: int = 768) -> None:
        self.dimensions = dimensions
        self.last_batch_size: int | None = None

    def embed_documents(
        self,
        texts: list[str],
        *,
        batch_size: int = 100,
        output_dimensionality: int | None = None,
        **_kwargs: Any,
    ) -> list[list[float]]:
        self.last_batch_size = batch_size
        dim = output_dimensionality or self.dimensions
        return [[float(i + 1) / 100.0] * dim for i in range(len(texts))]

    def embed_query(
        self,
        text: str,
        *,
        output_dimensionality: int | None = None,
        **_kwargs: Any,
    ) -> list[float]:
        dim = output_dimensionality or self.dimensions
        return [float(len(text)) / 100.0] * dim


def test_embed_documents_and_query_success() -> None:
    fake_client = FakeEmbeddingsClient(dimensions=768)
    service = EmbeddingService(dimensions=768, batch_size=50, client=fake_client)

    assert service.embed_documents([]) == []

    doc_vecs = service.embed_documents(["First chunk", "Second chunk"])
    assert len(doc_vecs) == 2
    assert len(doc_vecs[0]) == 768
    assert len(doc_vecs[1]) == 768
    assert fake_client.last_batch_size == 50

    query_vec = service.embed_query("What is the revenue?")
    assert len(query_vec) == 768


def test_embedding_dimension_validation() -> None:
    class WrongDimClient:
        def embed_documents(self, texts: list[str], **_kwargs: Any) -> list[list[float]]:
            return [[0.1] * 512 for _ in texts]

        def embed_query(self, _text: str, **_kwargs: Any) -> list[float]:
            return [0.1] * 512

    service = EmbeddingService(dimensions=768, client=WrongDimClient())

    with pytest.raises(EmbeddingDimensionMismatchError, match="Expected embedding dimension 768"):
        service.embed_documents(["chunk"])

    with pytest.raises(EmbeddingDimensionMismatchError, match="Expected embedding dimension 768"):
        service.embed_query("question")


def test_missing_api_key_and_empty_query_errors() -> None:
    service = EmbeddingService(api_key="", client=None)

    with pytest.raises(EmbeddingError, match="GEMINI_API_KEY is not configured"):
        service.embed_documents(["hello"])

    with pytest.raises(EmbeddingError, match="Query text cannot be empty"):
        service.embed_query("   ")
