import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import Chunk, Document


class DocumentRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create(
        self,
        *,
        filename: str,
        source_type: str,
        content_hash: str,
        source_url: str | None = None,
        storage_key: str | None = None,
        status: str = "pending",
    ) -> Document:
        doc = Document(
            filename=filename,
            source_type=source_type,
            content_hash=content_hash,
            source_url=source_url,
            storage_key=storage_key,
            status=status,
        )
        self.db.add(doc)
        self.db.commit()
        self.db.refresh(doc)
        return doc

    def get_by_id(self, document_id: uuid.UUID) -> Document | None:
        return self.db.get(Document, document_id)

    def get_completed_by_hash(self, content_hash: str) -> Document | None:
        stmt = select(Document).where(
            Document.content_hash == content_hash,
            Document.status == "completed",
        )
        return self.db.scalars(stmt).first()

    def list_all(self) -> list[Document]:
        stmt = select(Document).order_by(Document.created_at.desc())
        return list(self.db.scalars(stmt).all())

    def update_status(
        self,
        document: Document,
        *,
        status: str,
        error_message: str | None = None,
        filename: str | None = None,
    ) -> Document:
        document.status = status
        document.error_message = error_message
        if filename is not None:
            document.filename = filename
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document

    def delete(self, document: Document) -> None:
        self.db.delete(document)
        self.db.commit()


class ChunkRepository:
    def __init__(self, db: Session) -> None:
        self.db = db

    def bulk_create(
        self,
        document_id: uuid.UUID,
        chunk_items: list[dict[str, Any]],
    ) -> list[Chunk]:
        chunks = [
            Chunk(
                document_id=document_id,
                chunk_index=item["chunk_index"],
                content=item["content"],
                embedding=item["embedding"],
                page_number=item.get("page_number"),
                source_url=item.get("source_url"),
                chunk_metadata=item.get("chunk_metadata", {}),
            )
            for item in chunk_items
        ]
        self.db.add_all(chunks)
        self.db.commit()
        return chunks

    def list_by_document(self, document_id: uuid.UUID) -> list[Chunk]:
        stmt = (
            select(Chunk)
            .where(Chunk.document_id == document_id)
            .order_by(Chunk.chunk_index.asc())
        )
        return list(self.db.scalars(stmt).all())
