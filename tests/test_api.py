from fastapi.testclient import TestClient

from app.config import get_settings
from app.main import app

client = TestClient(app)


def test_settings_load() -> None:
    settings = get_settings()
    assert settings.embedding_dimensions == 768
    assert settings.chunk_size > settings.chunk_overlap
    assert settings.retrieval_top_k > 0


def test_health_endpoint() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "chat_model" in data
    assert "embedding_model" in data
