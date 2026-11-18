from typing import Any

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings


class EmbeddingError(Exception):
    """Base exception for embedding generation errors."""


class EmbeddingDimensionMismatchError(EmbeddingError):
    """Raised when returned vector dimensions do not match configured dimensions."""


class EmbeddingService:
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        batch_size: int = 100,
        client: Any | None = None,
    ) -> None:
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.gemini_api_key
        self.model = model or settings.embedding_model
        self.dimensions = dimensions if dimensions is not None else settings.embedding_dimensions
        self.batch_size = batch_size
        self._client = client

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client

        if not self.api_key:
            raise EmbeddingError("GEMINI_API_KEY is not configured.")

        self._client = GoogleGenerativeAIEmbeddings(
            model=self.model,
            google_api_key=self.api_key,
            output_dimensionality=self.dimensions,
        )
        return self._client

    def _validate_vector(self, vector: list[float]) -> list[float]:
        if len(vector) != self.dimensions:
            raise EmbeddingDimensionMismatchError(
                f"Expected embedding dimension {self.dimensions}, got {len(vector)}."
            )
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        client = self._get_client()
        try:
            vectors = client.embed_documents(
                texts,
                batch_size=self.batch_size,
                output_dimensionality=self.dimensions,
            )
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Failed to generate document embeddings: {exc}") from exc

        if len(vectors) != len(texts):
            raise EmbeddingError(
                f"Expected {len(texts)} embeddings, but provider returned {len(vectors)}."
            )

        return [self._validate_vector(vec) for vec in vectors]

    def embed_query(self, query: str) -> list[float]:
        cleaned = (query or "").strip()
        if not cleaned:
            raise EmbeddingError("Query text cannot be empty.")

        client = self._get_client()
        try:
            vector = client.embed_query(
                cleaned,
                output_dimensionality=self.dimensions,
            )
        except EmbeddingError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"Failed to generate query embedding: {exc}") from exc

        return self._validate_vector(vector)
