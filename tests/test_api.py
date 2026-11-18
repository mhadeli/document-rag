import uuid
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pymupdf
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.chat import get_rag_pipeline
from app.api.documents import get_ingestion_pipeline
from app.config import get_settings
from app.db.models import Base
from app.db.session import get_db
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.web_loader import ExtractedWebsite
from app.main import app
from app.rag.embeddings import EmbeddingService
from app.rag.generator import AnswerGenerator, StructuredLLMResponse
from app.rag.pipeline import RAGPipeline
from app.rag.retriever import VectorRetriever
from app.storage.local import LocalFileStorage


def _build_pdf_bytes(text: str) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class FakeEmbeddingClient:
    def embed_documents(self, texts: list[str], **_kwargs: Any) -> list[list[float]]:
        return [[0.25] * 768 for _ in texts]

    def embed_query(self, _text: str, **_kwargs: Any) -> list[float]:
        return [0.25] * 768


@dataclass
class FakeMessage:
    content: str


class FakeChatLLM:
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self.session_factory = session_factory

    def invoke(self, _messages: list[Any]) -> FakeMessage:
        return FakeMessage(content="Rewritten standalone query about Q4 revenue")

    def with_structured_output(self, _schema: Any) -> Any:
        from app.db.models import Chunk

        session_factory = self.session_factory

        class _Runner:
            def invoke(self, _messages: list[Any]) -> StructuredLLMResponse:
                with session_factory() as s:
                    first_chunk = s.query(Chunk).first()
                    cited = [str(first_chunk.id)] if first_chunk else []
                return StructuredLLMResponse(
                    answer="Grounded answer: Q4 revenue increased by 30%.",
                    cited_chunk_ids=cited,
                )

        return _Runner()


def test_settings_load() -> None:
    settings = get_settings()
    assert settings.embedding_dimensions == 768
    assert settings.chunk_size > settings.chunk_overlap
    assert settings.retrieval_top_k > 0


def test_health_endpoint() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "chat_model" in data
    assert "embedding_model" in data


def test_full_api_workflow_pdf_url_chat_and_delete(tmp_path: Path) -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    @event.listens_for(engine, "connect")
    def _enable_sqlite_fk(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    testing_session_local = sessionmaker(bind=engine, autocommit=False, autoflush=False)

    storage = LocalFileStorage(base_dir=tmp_path)
    fake_embeddings = EmbeddingService(dimensions=768, client=FakeEmbeddingClient())

    def override_get_db() -> Generator[Session, None, None]:
        db = testing_session_local()
        try:
            yield db
        finally:
            db.close()

    def override_get_ingestion_pipeline() -> Generator[IngestionPipeline, None, None]:
        db = testing_session_local()
        try:
            yield IngestionPipeline(
                db=db,
                storage=storage,
                embedding_service=fake_embeddings,
                web_extractor=lambda url: ExtractedWebsite(
                    source_url=url,
                    title="Example Docs",
                    content="Web article describing enterprise RAG architecture.",
                ),
            )
        finally:
            db.close()

    def override_get_rag_pipeline() -> Generator[RAGPipeline, None, None]:
        db = testing_session_local()
        try:
            yield RAGPipeline(
                db=db,
                retriever=VectorRetriever(db=db, embedding_service=fake_embeddings),
                generator=AnswerGenerator(llm=FakeChatLLM(testing_session_local)),
            )
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_ingestion_pipeline] = override_get_ingestion_pipeline
    app.dependency_overrides[get_rag_pipeline] = override_get_rag_pipeline

    try:
        client = TestClient(app)

        # 1. Upload PDF
        pdf_bytes = _build_pdf_bytes("Q4 revenue increased by 30% in 2026.")
        upload_res = client.post(
            "/documents/upload",
            files={"file": ("q4_report.pdf", pdf_bytes, "application/pdf")},
        )
        assert upload_res.status_code == 201
        pdf_doc = upload_res.json()
        assert pdf_doc["filename"] == "q4_report.pdf"
        assert pdf_doc["source_type"] == "pdf"
        assert pdf_doc["status"] == "completed"
        assert pdf_doc["chunk_count"] == 1
        pdf_doc_id = pdf_doc["id"]

        # 2. Invalid non-PDF upload returns 400
        bad_upload = client.post(
            "/documents/upload",
            files={"file": ("fake.pdf", b"not a pdf", "application/pdf")},
        )
        assert bad_upload.status_code == 400

        # 3. Ingest Website URL
        url_res = client.post(
            "/documents/url",
            json={"url": "https://example.com/rag"},
        )
        assert url_res.status_code == 201
        web_doc = url_res.json()
        assert web_doc["filename"] == "Example Docs"
        assert web_doc["source_type"] == "website"
        assert web_doc["chunk_count"] == 1

        # 4. List documents
        list_res = client.get("/documents")
        assert list_res.status_code == 200
        docs_list = list_res.json()["documents"]
        assert len(docs_list) == 2

        # 5. Get single document details
        detail_res = client.get(f"/documents/{pdf_doc_id}")
        assert detail_res.status_code == 200
        assert detail_res.json()["id"] == pdf_doc_id

        # 6. Ask question via POST /chat with history
        chat_res = client.post(
            "/chat",
            json={
                "question": "How much did it increase?",
                "document_ids": [pdf_doc_id],
                "history": [
                    {"role": "user", "content": "Tell me about Q4 revenue."},
                    {"role": "assistant", "content": "It covers Q4 2026 results."},
                ],
            },
        )
        assert chat_res.status_code == 200
        chat_data = chat_res.json()
        assert "30%" in chat_data["answer"]
        assert len(chat_data["sources"]) == 1
        assert chat_data["sources"][0]["document_id"] == pdf_doc_id
        assert chat_data["sources"][0]["page_number"] == 1

        # 7. Delete PDF document and confirm 404 afterwards
        del_res = client.delete(f"/documents/{pdf_doc_id}")
        assert del_res.status_code == 200
        assert del_res.json()["deleted"] is True

        missing_res = client.get(f"/documents/{pdf_doc_id}")
        assert missing_res.status_code == 404
        assert client.delete(f"/documents/{uuid.uuid4()}").status_code == 404
    finally:
        app.dependency_overrides.clear()
