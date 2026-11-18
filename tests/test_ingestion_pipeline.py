from pathlib import Path
from typing import Any

import pymupdf
import pytest
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base, Chunk, Document
from app.ingestion.chunker import DocumentChunker
from app.ingestion.pdf_loader import EmptyPDFContentError
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.web_loader import ExtractedWebsite
from app.rag.embeddings import EmbeddingError, EmbeddingService
from app.storage.local import LocalFileStorage


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


def _make_pdf(pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


class FakeEmbeddingClient:
    def embed_documents(self, texts: list[str], **_kwargs: Any) -> list[list[float]]:
        return [[0.05] * 768 for _ in texts]

    def embed_query(self, _text: str, **_kwargs: Any) -> list[float]:
        return [0.05] * 768


def test_ingest_pdf_end_to_end_and_deduplication(tmp_path: Path) -> None:
    db = _create_sqlite_session()
    storage = LocalFileStorage(base_dir=tmp_path)
    embedding_service = EmbeddingService(dimensions=768, client=FakeEmbeddingClient())
    pipeline = IngestionPipeline(
        db=db,
        storage=storage,
        chunker=DocumentChunker(chunk_size=500, chunk_overlap=50),
        embedding_service=embedding_service,
    )

    pdf_bytes = _make_pdf(["Page 1 executive summary.", "Page 2 financial metrics."])
    result1 = pipeline.ingest_pdf(pdf_bytes, filename="summary.pdf")

    assert result1.deduplicated is False
    assert result1.chunk_count == 2
    assert result1.document.status == "completed"
    assert result1.document.storage_key is not None
    assert (tmp_path / result1.document.storage_key).is_file()

    # Re-uploading identical PDF deduplicates without creating extra chunks
    result2 = pipeline.ingest_pdf(pdf_bytes, filename="summary_copy.pdf")
    assert result2.deduplicated is True
    assert result2.document.id == result1.document.id
    assert db.query(Document).count() == 1
    assert db.query(Chunk).count() == 2

    # Deleting removes both DB records and stored file
    storage_key = result1.document.storage_key
    assert pipeline.delete_document(result1.document.id) is True
    assert not (tmp_path / storage_key).exists()
    assert db.query(Document).count() == 0
    assert db.query(Chunk).count() == 0
    db.close()


def test_ingest_url_end_to_end_and_deduplication(tmp_path: Path) -> None:
    db = _create_sqlite_session()
    storage = LocalFileStorage(base_dir=tmp_path)
    embedding_service = EmbeddingService(dimensions=768, client=FakeEmbeddingClient())

    def fake_web_extractor(url: str) -> ExtractedWebsite:
        return ExtractedWebsite(
            source_url=url,
            title="Pgvector Guide",
            content="PostgreSQL with pgvector stores embeddings alongside relational data.",
        )

    pipeline = IngestionPipeline(
        db=db,
        storage=storage,
        embedding_service=embedding_service,
        web_extractor=fake_web_extractor,
    )

    res1 = pipeline.ingest_url("https://example.com/pgvector")
    assert res1.deduplicated is False
    assert res1.chunk_count == 1
    assert res1.document.status == "completed"
    assert res1.document.filename == "Pgvector Guide"

    res2 = pipeline.ingest_url("https://example.com/pgvector")
    assert res2.deduplicated is True
    assert res2.document.id == res1.document.id
    db.close()


def test_failed_ingestion_marks_document_failed_without_partial_chunks(tmp_path: Path) -> None:
    db = _create_sqlite_session()
    storage = LocalFileStorage(base_dir=tmp_path)

    class FailingEmbeddingClient:
        def embed_documents(self, _texts: list[str], **_kwargs: Any) -> list[list[float]]:
            raise RuntimeError("Rate limit exceeded")

    pipeline = IngestionPipeline(
        db=db,
        storage=storage,
        embedding_service=EmbeddingService(dimensions=768, client=FailingEmbeddingClient()),
    )

    pdf_bytes = _make_pdf(["Valid text on page 1."])
    with pytest.raises(EmbeddingError, match="Rate limit exceeded"):
        pipeline.ingest_pdf(pdf_bytes, filename="will_fail.pdf")

    docs = db.query(Document).all()
    assert len(docs) == 1
    assert docs[0].status == "failed"
    assert "Rate limit exceeded" in (docs[0].error_message or "")
    assert db.query(Chunk).count() == 0

    # Also test scanned/empty PDF failure updates status
    blank_pdf = _make_pdf([""])
    with pytest.raises(EmptyPDFContentError):
        pipeline.ingest_pdf(blank_pdf, filename="scanned.pdf")

    db.close()
