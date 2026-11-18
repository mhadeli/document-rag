import math
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db.models import Chunk, Document
from app.rag.embeddings import EmbeddingService


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    source_type: str
    content: str
    score: float
    chunk_index: int
    page_number: int | None = None
    source_url: str | None = None


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class VectorRetriever:
    def __init__(
        self,
        db: Session,
        *,
        embedding_service: EmbeddingService | None = None,
        default_top_k: int | None = None,
    ) -> None:
        settings = get_settings()
        self.db = db
        self.embedding_service = embedding_service or EmbeddingService()
        self.default_top_k = (
            default_top_k if default_top_k is not None else settings.retrieval_top_k
        )

    def build_postgres_query(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        document_ids: list[uuid.UUID] | None = None,
    ) -> Select[Any]:
        distance_col = Chunk.embedding.cosine_distance(query_vector).label("distance")
        stmt = (
            select(Chunk, Document, distance_col)
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.status == "completed")
        )
        if document_ids:
            stmt = stmt.where(Chunk.document_id.in_(document_ids))

        return stmt.order_by(distance_col.asc(), Chunk.chunk_index.asc()).limit(top_k)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        document_ids: list[uuid.UUID] | None = None,
    ) -> list[RetrievedChunk]:
        limit = top_k if top_k is not None else self.default_top_k
        if limit <= 0:
            return []

        if document_ids is not None and len(document_ids) == 0:
            return []

        query_vector = self.embedding_service.embed_query(query)
        dialect_name = self.db.bind.dialect.name if self.db.bind is not None else "postgresql"

        if dialect_name == "postgresql":
            stmt = self.build_postgres_query(
                query_vector,
                top_k=limit,
                document_ids=document_ids,
            )
            rows = self.db.execute(stmt).all()
            return [
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=doc.id,
                    filename=doc.filename,
                    source_type=doc.source_type,
                    content=chunk.content,
                    score=round(max(0.0, 1.0 - float(distance)), 4),
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    source_url=chunk.source_url or doc.source_url,
                )
                for chunk, doc, distance in rows
            ]

        # Fallback for SQLite in-memory tests where pgvector's <=> operator is unavailable
        base_stmt = (
            select(Chunk, Document)
            .join(Document, Chunk.document_id == Document.id)
            .where(Document.status == "completed")
        )
        if document_ids:
            base_stmt = base_stmt.where(Chunk.document_id.in_(document_ids))

        candidates = self.db.execute(base_stmt).all()
        scored: list[RetrievedChunk] = []
        for chunk, doc in candidates:
            emb = [float(x) for x in chunk.embedding]
            similarity = _cosine_similarity(query_vector, emb)
            scored.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=doc.id,
                    filename=doc.filename,
                    source_type=doc.source_type,
                    content=chunk.content,
                    score=round(max(0.0, similarity), 4),
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    source_url=chunk.source_url or doc.source_url,
                )
            )

        scored.sort(key=lambda item: (-item.score, item.chunk_index))
        return scored[:limit]
