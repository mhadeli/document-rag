from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    gemini_api_key: str = ""
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/ragdb"

    embedding_model: str = "models/gemini-embedding-001"
    embedding_dimensions: int = 768
    chat_model: str = "gemini-2.5-flash"

    chunk_size: int = 2500
    chunk_overlap: int = 300
    retrieval_top_k: int = 5

    upload_dir: Path = Path("./data/uploads")
    max_upload_mb: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
