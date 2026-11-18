from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.api.schemas import ChatRequest, ChatResponse, SourceCitationSchema
from app.db.session import get_db
from app.rag.embeddings import EmbeddingError
from app.rag.generator import ChatTurn, GenerationError
from app.rag.pipeline import RAGPipeline

router = APIRouter(prefix="/chat", tags=["chat"])


def get_rag_pipeline(db: Session = Depends(get_db)) -> RAGPipeline:  # noqa: B008
    return RAGPipeline(db=db)


@router.post("", response_model=ChatResponse)
def ask_question(
    payload: ChatRequest,
    pipeline: RAGPipeline = Depends(get_rag_pipeline),  # noqa: B008
) -> ChatResponse:
    cleaned_question = payload.question.strip()
    if not cleaned_question:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question cannot be empty or whitespace.",
        )

    history = [ChatTurn(role=item.role, content=item.content) for item in payload.history]

    try:
        result = pipeline.ask(
            cleaned_question,
            document_ids=payload.document_ids,
            history=history,
            top_k=payload.top_k,
        )
    except (EmbeddingError, GenerationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return ChatResponse(
        answer=result.answer,
        rewritten_query=result.rewritten_query,
        sources=[
            SourceCitationSchema(
                document_id=src.document_id,
                chunk_id=src.chunk_id,
                filename=src.filename,
                source_type=src.source_type,
                snippet=src.snippet,
                score=src.score,
                page_number=src.page_number,
                source_url=src.source_url,
            )
            for src in result.sources
        ],
    )
