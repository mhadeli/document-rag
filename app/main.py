from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings

settings = get_settings()

app = FastAPI(
    title="Document Intelligence RAG API",
    description="Upload PDFs, ingest websites, and ask grounded questions with Google Gemini.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["system"])
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
    }
