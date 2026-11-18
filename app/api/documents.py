import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.api.schemas import (
    DeleteDocumentResponse,
    DocumentListResponse,
    DocumentResponse,
    URLIngestRequest,
)
from app.db.models import Document
from app.db.repositories import DocumentRepository
from app.db.session import get_db
from app.ingestion.pdf_loader import PDFExtractionError
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.web_loader import (
    EmptyWebContentError,
    InvalidURLError,
    SSRFBlockedError,
    WebFetchError,
)
from app.rag.embeddings import EmbeddingError
from app.storage.local import FileTooLargeError, InvalidPDFError, StorageError

router = APIRouter(prefix="/documents", tags=["documents"])


def get_ingestion_pipeline(db: Session = Depends(get_db)) -> IngestionPipeline:  # noqa: B008
    return IngestionPipeline(db=db)


def _to_document_response(
    doc: Document,
    *,
    chunk_count: int | None = None,
    deduplicated: bool = False,
) -> DocumentResponse:
    resolved_count = chunk_count if chunk_count is not None else len(doc.chunks)
    return DocumentResponse(
        id=doc.id,
        filename=doc.filename,
        source_type=doc.source_type,
        source_url=doc.source_url,
        status=doc.status,
        error_message=doc.error_message,
        chunk_count=resolved_count,
        deduplicated=deduplicated,
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_pdf_document(
    file: UploadFile = File(...),  # noqa: B008
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),  # noqa: B008
) -> DocumentResponse:
    filename = file.filename or "document.pdf"
    content = file.file.read()

    try:
        result = pipeline.ingest_pdf(content, filename=filename)
    except FileTooLargeError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=str(exc),
        ) from exc
    except (InvalidPDFError, StorageError, PDFExtractionError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except EmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return _to_document_response(
        result.document,
        chunk_count=result.chunk_count,
        deduplicated=result.deduplicated,
    )


@router.post(
    "/url",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
def ingest_website_url(
    payload: URLIngestRequest,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),  # noqa: B008
) -> DocumentResponse:
    try:
        result = pipeline.ingest_url(payload.url)
    except (InvalidURLError, SSRFBlockedError, EmptyWebContentError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except (WebFetchError, EmbeddingError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return _to_document_response(
        result.document,
        chunk_count=result.chunk_count,
        deduplicated=result.deduplicated,
    )


@router.get("", response_model=DocumentListResponse)
def list_documents(db: Session = Depends(get_db)) -> DocumentListResponse:  # noqa: B008
    repo = DocumentRepository(db)
    docs = repo.list_all()
    return DocumentListResponse(documents=[_to_document_response(doc) for doc in docs])


@router.get("/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: uuid.UUID,
    db: Session = Depends(get_db),  # noqa: B008
) -> DocumentResponse:
    repo = DocumentRepository(db)
    doc = repo.get_by_id(document_id)
    if doc is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    return _to_document_response(doc)


@router.delete("/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(
    document_id: uuid.UUID,
    pipeline: IngestionPipeline = Depends(get_ingestion_pipeline),  # noqa: B008
) -> DeleteDocumentResponse:
    deleted = pipeline.delete_document(document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document '{document_id}' not found.",
        )
    return DeleteDocumentResponse(id=document_id, deleted=True)
