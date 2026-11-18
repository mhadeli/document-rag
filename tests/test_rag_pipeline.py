import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base
from app.db.repositories import ChunkRepository, DocumentRepository
from app.rag.embeddings import EmbeddingService
from app.rag.generator import (
    AnswerGenerator,
    ChatTurn,
    StructuredLLMResponse,
)
from app.rag.pipeline import RAGPipeline
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


class FakeEmbeddingClient:
    def embed_query(self, text: str, **_kwargs: Any) -> list[float]:
        if "revenue" in text.lower() or "2026" in text.lower():
            return _unit_vec(0)
        return _unit_vec(1)


@dataclass
class FakeAIMessage:
    content: str


class FakeLLM:
    def __init__(self, structured_response: StructuredLLMResponse, rewritten: str = "") -> None:
        self.structured_response = structured_response
        self.rewritten = rewritten
        self.rewrite_calls: int = 0
        self.structured_calls: int = 0

    def invoke(self, _messages: Any) -> FakeAIMessage:
        self.rewrite_calls += 1
        return FakeAIMessage(content=self.rewritten)

    def with_structured_output(self, _schema: Any) -> "FakeLLM":
        return self

    def _invoke_structured(self, _messages: Any) -> StructuredLLMResponse:
        self.structured_calls += 1
        return self.structured_response


class DualModeFakeLLM:
    def __init__(
        self,
        *,
        answer: str,
        cited_ids: list[str],
        rewritten_query: str = "What was the Q1 2026 revenue growth?",
    ) -> None:
        self.answer = answer
        self.cited_ids = cited_ids
        self.rewritten_query = rewritten_query
        self.last_prompt: str = ""

    def invoke(self, messages: list[Any]) -> FakeAIMessage:
        return FakeAIMessage(content=self.rewritten_query)

    def with_structured_output(self, _schema: Any) -> Any:
        parent = self

        class _StructuredRunner:
            def invoke(self, messages: list[Any]) -> StructuredLLMResponse:
                parent.last_prompt = str(messages[-1].content)
                return StructuredLLMResponse(
                    answer=parent.answer,
                    cited_chunk_ids=parent.cited_ids,
                )

        return _StructuredRunner()


def test_rag_pipeline_returns_grounded_answer_and_maps_citations() -> None:
    db = _create_sqlite_session()
    doc_repo = DocumentRepository(db)
    chunk_repo = ChunkRepository(db)

    doc = doc_repo.create(
        filename="q1_2026_report.pdf",
        source_type="pdf",
        content_hash="hash_q1",
        status="completed",
    )
    created_chunks = chunk_repo.bulk_create(
        doc.id,
        [
            {
                "chunk_index": 0,
                "content": "In Q1 2026, revenue grew by 18% to $42M.",
                "embedding": _unit_vec(0),
                "page_number": 3,
            },
            {
                "chunk_index": 1,
                "content": "Employee headcount remained steady at 150.",
                "embedding": _unit_vec(1),
                "page_number": 9,
            },
        ],
    )
    target_chunk_id = str(created_chunks[0].id)

    fake_llm = DualModeFakeLLM(
        answer="In Q1 2026, revenue grew by 18% to $42M.",
        cited_ids=[target_chunk_id, "non-existent-chunk-uuid"],
    )
    embedding_service = EmbeddingService(dimensions=768, client=FakeEmbeddingClient())
    retriever = VectorRetriever(db, embedding_service=embedding_service)
    generator = AnswerGenerator(llm=fake_llm)
    pipeline = RAGPipeline(db, retriever=retriever, generator=generator)

    result = pipeline.ask(
        "What about in Q1?",
        history=[
            ChatTurn(role="user", content="Tell me about the 2026 financial report."),
            ChatTurn(role="assistant", content="The report covers 2026 revenue and headcount."),
        ],
    )

    assert result.answer == "In Q1 2026, revenue grew by 18% to $42M."
    assert result.rewritten_query == "What was the Q1 2026 revenue growth?"
    # Only the valid cited chunk ID is mapped; hallucinated IDs are filtered out
    assert len(result.sources) == 1
    assert result.sources[0].chunk_id == created_chunks[0].id
    assert result.sources[0].document_id == doc.id
    assert result.sources[0].filename == "q1_2026_report.pdf"
    assert result.sources[0].page_number == 3
    assert "18%" in result.sources[0].snippet
    db.close()


def test_rag_pipeline_handles_empty_retrieval_gracefully() -> None:
    db = _create_sqlite_session()
    fake_llm = DualModeFakeLLM(answer="Should not be called", cited_ids=[])
    embedding_service = EmbeddingService(dimensions=768, client=FakeEmbeddingClient())
    pipeline = RAGPipeline(
        db,
        retriever=VectorRetriever(db, embedding_service=embedding_service),
        generator=AnswerGenerator(llm=fake_llm),
    )

    # Filtering to a random UUID yields 0 chunks
    result = pipeline.ask("What is the revenue?", document_ids=[uuid.uuid4()])
    assert "could not find any relevant information" in result.answer.lower()
    assert result.sources == []
    db.close()
