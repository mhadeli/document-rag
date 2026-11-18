from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base
from app.db.repositories import ChunkRepository, DocumentRepository
from app.rag.embeddings import EmbeddingService
from app.rag.retriever import VectorRetriever


def _create_sqlite_session() -> Session:
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def _enable_sqlite_fk(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    return session_factory()


def _unit_vec(index: int, dim: int = 768) -> list[float]:
    vec = [0.0] * dim
    vec[index] = 1.0
    return vec


class KeywordEmbeddingClient:
    def embed_query(self, text: str, **_kwargs: Any) -> list[float]:
        lower = text.lower()
        if "revenue" in lower:
            return _unit_vec(0)
        if "security" in lower:
            return _unit_vec(1)
        return _unit_vec(2)


def test_postgres_query_uses_pgvector_cosine_distance() -> None:
    db = _create_sqlite_session()
    retriever = VectorRetriever(
        db,
        embedding_service=EmbeddingService(dimensions=768, client=KeywordEmbeddingClient()),
    )
    stmt = retriever.build_postgres_query(_unit_vec(0), top_k=5)
    compiled_sql = str(stmt.compile(dialect=postgresql.dialect()))

    assert "<=>" in compiled_sql
    assert "documents.status" in compiled_sql
    assert "LIMIT" in compiled_sql
    db.close()


def test_retrieve_ranks_chunks_and_filters_incomplete_or_unselected_docs() -> None:
    db = _create_sqlite_session()
    doc_repo = DocumentRepository(db)
    chunk_repo = ChunkRepository(db)

    doc_finance = doc_repo.create(
        filename="finance_2026.pdf",
        source_type="pdf",
        content_hash="hash_fin",
        status="completed",
    )
    doc_security = doc_repo.create(
        filename="Security Overview",
        source_type="website",
        source_url="https://example.com/security",
        content_hash="hash_sec",
        status="completed",
    )
    doc_failed = doc_repo.create(
        filename="failed_report.pdf",
        source_type="pdf",
        content_hash="hash_fail",
        status="failed",
    )

    chunk_repo.bulk_create(
        doc_finance.id,
        [
            {
                "chunk_index": 0,
                "content": "Annual revenue increased by 22% year-over-year.",
                "embedding": _unit_vec(0),
                "page_number": 5,
            },
            {
                "chunk_index": 1,
                "content": "General office locations and contact info.",
                "embedding": _unit_vec(2),
                "page_number": 12,
            },
        ],
    )
    chunk_repo.bulk_create(
        doc_security.id,
        [
            {
                "chunk_index": 0,
                "content": "SSRF protection blocks private and loopback IPs.",
                "embedding": _unit_vec(1),
                "source_url": "https://example.com/security",
            }
        ],
    )
    chunk_repo.bulk_create(
        doc_failed.id,
        [
            {
                "chunk_index": 0,
                "content": "Corrupted revenue chunk that must be excluded.",
                "embedding": _unit_vec(0),
                "page_number": 1,
            }
        ],
    )

    retriever = VectorRetriever(
        db,
        embedding_service=EmbeddingService(dimensions=768, client=KeywordEmbeddingClient()),
        default_top_k=5,
    )

    # 1. Revenue query ranks finance page 5 first and excludes failed_report.pdf
    results = retriever.retrieve("What happened to annual revenue?")
    assert len(results) == 3
    assert results[0].document_id == doc_finance.id
    assert results[0].page_number == 5
    assert results[0].score == 1.0
    assert all(r.document_id != doc_failed.id for r in results)

    # 2. Filtering by document_ids scopes results strictly to doc_security
    scoped = retriever.retrieve(
        "What happened to annual revenue?",
        document_ids=[doc_security.id],
    )
    assert len(scoped) == 1
    assert scoped[0].document_id == doc_security.id
    assert scoped[0].source_url == "https://example.com/security"

    # 3. Top-K limit is respected
    top_one = retriever.retrieve("Tell me about security", top_k=1)
    assert len(top_one) == 1
    assert top_one[0].document_id == doc_security.id
    assert top_one[0].score == 1.0

    db.close()
