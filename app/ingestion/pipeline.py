import hashlib
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.db.models import Document
from app.db.repositories import ChunkRepository, DocumentRepository
from app.ingestion.chunker import DocumentChunker, TextChunk
from app.ingestion.pdf_loader import extract_pdf_from_bytes
from app.ingestion.web_loader import ExtractedWebsite, extract_website
from app.rag.embeddings import EmbeddingService
from app.storage.local import FileStorage, LocalFileStorage

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestionResult:
    document: Document
    chunk_count: int
    deduplicated: bool = False


def compute_sha256(data: bytes | str) -> str:
    payload = data.encode("utf-8") if isinstance(data, str) else data
    return hashlib.sha256(payload).hexdigest()


class IngestionPipeline:
    def __init__(
        self,
        db: Session,
        *,
        storage: FileStorage | None = None,
        chunker: DocumentChunker | None = None,
        embedding_service: EmbeddingService | None = None,
        web_extractor: Callable[[str], ExtractedWebsite] | None = None,
    ) -> None:
        self.db = db
        self.doc_repo = DocumentRepository(db)
        self.chunk_repo = ChunkRepository(db)
        self.storage = storage or LocalFileStorage()
        self.chunker = chunker or DocumentChunker()
        self.embedding_service = embedding_service or EmbeddingService()
        self._web_extractor = web_extractor or extract_website

    def _embed_and_persist_chunks(
        self,
        document: Document,
        text_chunks: list[TextChunk],
    ) -> int:
        if not text_chunks:
            raise ValueError(f"No text chunks generated for document '{document.filename}'.")

        embeddings = self.embedding_service.embed_documents([c.content for c in text_chunks])

        chunk_items = [
            {
                "chunk_index": chunk.chunk_index,
                "content": chunk.content,
                "embedding": embedding,
                "page_number": chunk.page_number,
                "source_url": chunk.source_url,
                "chunk_metadata": chunk.chunk_metadata,
            }
            for chunk, embedding in zip(text_chunks, embeddings, strict=True)
        ]

        self.chunk_repo.bulk_create(document.id, chunk_items)
        self.doc_repo.update_status(document, status="completed", error_message=None)
        return len(chunk_items)

    def ingest_pdf(self, pdf_bytes: bytes, filename: str = "document.pdf") -> IngestionResult:
        start_time = time.perf_counter()

        if isinstance(self.storage, LocalFileStorage):
            self.storage.validate_pdf(pdf_bytes)

        content_hash = compute_sha256(pdf_bytes)
        existing = self.doc_repo.get_completed_by_hash(content_hash)
        if existing is not None:
            existing_chunks = self.chunk_repo.list_by_document(existing.id)
            logger.info(
                "Deduplicated PDF upload '%s' matching existing document %s",
                filename,
                existing.id,
            )
            return IngestionResult(
                document=existing,
                chunk_count=len(existing_chunks),
                deduplicated=True,
            )

        storage_key = self.storage.save_pdf(pdf_bytes)
        document = self.doc_repo.create(
            filename=filename,
            source_type="pdf",
            content_hash=content_hash,
            storage_key=storage_key,
            status="processing",
        )

        try:
            extracted_pdf = extract_pdf_from_bytes(pdf_bytes, filename=filename)
            text_chunks = self.chunker.chunk_pdf(extracted_pdf)
            chunk_count = self._embed_and_persist_chunks(document, text_chunks)
        except Exception as exc:
            self.db.rollback()
            self.doc_repo.update_status(document, status="failed", error_message=str(exc))
            logger.error("Failed PDF ingestion for '%s': %s", filename, exc)
            raise

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Completed PDF ingestion '%s' (%d chunks) in %.1f ms",
            filename,
            chunk_count,
            elapsed_ms,
        )
        return IngestionResult(document=document, chunk_count=chunk_count, deduplicated=False)

    def ingest_url(self, url: str) -> IngestionResult:
        start_time = time.perf_counter()
        extracted_web = self._web_extractor(url)

        content_hash = compute_sha256(f"{extracted_web.source_url}\n{extracted_web.content}")
        existing = self.doc_repo.get_completed_by_hash(content_hash)
        if existing is not None:
            existing_chunks = self.chunk_repo.list_by_document(existing.id)
            logger.info(
                "Deduplicated website ingestion '%s' matching existing document %s",
                url,
                existing.id,
            )
            return IngestionResult(
                document=existing,
                chunk_count=len(existing_chunks),
                deduplicated=True,
            )

        document = self.doc_repo.create(
            filename=extracted_web.title,
            source_type="website",
            source_url=extracted_web.source_url,
            content_hash=content_hash,
            status="processing",
        )

        try:
            text_chunks = self.chunker.chunk_website(extracted_web)
            chunk_count = self._embed_and_persist_chunks(document, text_chunks)
        except Exception as exc:
            self.db.rollback()
            self.doc_repo.update_status(document, status="failed", error_message=str(exc))
            logger.error("Failed website ingestion for '%s': %s", url, exc)
            raise

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "Completed website ingestion '%s' (%d chunks) in %.1f ms",
            extracted_web.source_url,
            chunk_count,
            elapsed_ms,
        )
        return IngestionResult(document=document, chunk_count=chunk_count, deduplicated=False)

    def delete_document(self, document_id: uuid.UUID) -> bool:
        document = self.doc_repo.get_by_id(document_id)
        if document is None:
            return False

        if document.storage_key:
            self.storage.delete(document.storage_key)

        self.doc_repo.delete(document)
        return True
