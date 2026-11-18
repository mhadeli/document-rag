from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base, Chunk, Document
from app.db.repositories import ChunkRepository, DocumentRepository


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


def test_models_registered_in_metadata() -> None:
    assert "documents" in Base.metadata.tables
    assert "chunks" in Base.metadata.tables
    chunk_cols = set(Base.metadata.tables["chunks"].columns.keys())
    assert {
        "id",
        "document_id",
        "chunk_index",
        "content",
        "embedding",
        "page_number",
        "source_url",
        "chunk_metadata",
    }.issubset(chunk_cols)


def test_document_and_chunk_repositories_crud_and_cascade() -> None:
    db = _create_sqlite_session()
    doc_repo = DocumentRepository(db)
    chunk_repo = ChunkRepository(db)

    doc = doc_repo.create(
        filename="sample.pdf",
        source_type="pdf",
        content_hash="abc123hash",
        storage_key="sample.pdf",
        status="processing",
    )
    assert doc.id is not None
    assert doc.status == "processing"

    chunk_repo.bulk_create(
        document_id=doc.id,
        chunk_items=[
            {
                "chunk_index": 0,
                "content": "First page content",
                "embedding": [0.1] * 768,
                "page_number": 1,
                "chunk_metadata": {"filename": "sample.pdf"},
            },
            {
                "chunk_index": 1,
                "content": "Second page content",
                "embedding": [0.2] * 768,
                "page_number": 2,
                "chunk_metadata": {"filename": "sample.pdf"},
            },
        ],
    )

    doc_repo.update_status(doc, status="completed")
    assert doc_repo.get_completed_by_hash("abc123hash") is not None

    stored_chunks = chunk_repo.list_by_document(doc.id)
    assert len(stored_chunks) == 2
    assert stored_chunks[0].page_number == 1
    assert stored_chunks[1].page_number == 2

    # Deleting document cascades to chunks
    doc_repo.delete(doc)
    assert doc_repo.get_by_id(doc.id) is None
    assert db.query(Chunk).count() == 0
    assert db.query(Document).count() == 0
    db.close()
