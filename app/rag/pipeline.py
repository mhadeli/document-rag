import uuid

from sqlalchemy.orm import Session

from app.rag.embeddings import EmbeddingService
from app.rag.generator import AnswerGenerator, ChatTurn, GeneratedAnswer
from app.rag.retriever import VectorRetriever


class RAGPipeline:
    def __init__(
        self,
        db: Session,
        *,
        retriever: VectorRetriever | None = None,
        generator: AnswerGenerator | None = None,
        embedding_service: EmbeddingService | None = None,
    ) -> None:
        shared_embeddings = embedding_service or EmbeddingService()
        self.db = db
        self.retriever = retriever or VectorRetriever(
            db,
            embedding_service=shared_embeddings,
        )
        self.generator = generator or AnswerGenerator()

    def ask(
        self,
        question: str,
        *,
        document_ids: list[uuid.UUID] | None = None,
        history: list[ChatTurn] | None = None,
        top_k: int | None = None,
    ) -> GeneratedAnswer:
        rewritten_query = self.generator.rewrite_query(question, history=history)
        retrieved_chunks = self.retriever.retrieve(
            rewritten_query,
            top_k=top_k,
            document_ids=document_ids,
        )
        return self.generator.generate_answer(
            question,
            retrieved_chunks,
            history=history,
            rewritten_query=rewritten_query,
        )
