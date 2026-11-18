import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class URLIngestRequest(BaseModel):
    url: str = Field(..., min_length=1, description="Public http/https webpage URL to ingest.")


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    filename: str
    source_type: str
    source_url: str | None = None
    status: str
    error_message: str | None = None
    chunk_count: int = 0
    deduplicated: bool = False
    created_at: datetime
    updated_at: datetime


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]


class DeleteDocumentResponse(BaseModel):
    id: uuid.UUID
    deleted: bool


class ChatMessageSchema(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Natural-language question to ask.")
    document_ids: list[uuid.UUID] | None = Field(
        default=None,
        description="Optional list of document UUIDs to scope retrieval.",
    )
    history: list[ChatMessageSchema] = Field(
        default_factory=list,
        description="Optional prior conversation turns for follow-up context.",
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=20,
        description="Optional override for number of chunks to retrieve.",
    )


class SourceCitationSchema(BaseModel):
    document_id: uuid.UUID
    chunk_id: uuid.UUID
    filename: str
    source_type: str
    snippet: str
    score: float
    page_number: int | None = None
    source_url: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceCitationSchema]
    rewritten_query: str
