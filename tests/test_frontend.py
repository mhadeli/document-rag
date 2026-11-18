from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests

from frontend.streamlit_app import (
    BackendAPIError,
    ask_question_to_api,
    delete_document_from_api,
    fetch_documents,
    format_source_label,
    ingest_url_to_api,
    upload_pdf_to_api,
)


def test_format_source_label_for_pdf_and_web() -> None:
    pdf_label = format_source_label(
        {"filename": "annual_report.pdf", "page_number": 4, "score": 0.912}
    )
    assert pdf_label == "annual_report.pdf — Page 4 (score: 0.91)"

    web_label = format_source_label(
        {
            "filename": "RAG Architecture Guide",
            "page_number": None,
            "source_url": "https://example.com/rag",
            "score": 0.85,
        }
    )
    assert web_label == "RAG Architecture Guide — https://example.com/rag (score: 0.85)"


@patch("frontend.streamlit_app.requests.get")
def test_fetch_documents_success(mock_get: MagicMock) -> None:
    mock_res = MagicMock()
    mock_res.json.return_value = {"documents": [{"id": "123", "filename": "doc.pdf"}]}
    mock_get.return_value = mock_res

    docs = fetch_documents("http://localhost:8000")
    assert len(docs) == 1
    assert docs[0]["filename"] == "doc.pdf"


@patch("frontend.streamlit_app.requests.post")
def test_upload_url_and_chat_helpers(mock_post: MagicMock) -> None:
    def side_effect(url: str, **_kwargs: Any) -> MagicMock:
        res = MagicMock()
        res.status_code = 200
        if url.endswith("/documents/upload"):
            res.json.return_value = {"id": "u1", "filename": "test.pdf", "chunk_count": 3}
        elif url.endswith("/documents/url"):
            res.json.return_value = {"id": "w1", "filename": "Example", "chunk_count": 2}
        elif url.endswith("/chat"):
            res.json.return_value = {"answer": "Grounded reply", "sources": []}
        return res

    mock_post.side_effect = side_effect

    pdf_res = upload_pdf_to_api("test.pdf", b"%PDF-1.7...", "http://localhost:8000")
    assert pdf_res["chunk_count"] == 3

    web_res = ingest_url_to_api("https://example.com", "http://localhost:8000")
    assert web_res["filename"] == "Example"

    chat_res = ask_question_to_api(
        "What is this?",
        document_ids=["u1"],
        history=[{"role": "user", "content": "Hi"}],
        api_base_url="http://localhost:8000",
    )
    assert chat_res["answer"] == "Grounded reply"


@patch("frontend.streamlit_app.requests.delete")
def test_delete_document_and_error_handling(mock_delete: MagicMock) -> None:
    ok_res = MagicMock()
    ok_res.status_code = 200
    ok_res.json.return_value = {"deleted": True}
    mock_delete.return_value = ok_res
    assert delete_document_from_api("doc-1", "http://localhost:8000") is True

    err_res = MagicMock()
    err_res.status_code = 404
    err_res.json.return_value = {"detail": "Document not found"}
    mock_delete.return_value = err_res
    with pytest.raises(BackendAPIError, match="Document not found"):
        delete_document_from_api("missing-id", "http://localhost:8000")


def test_upload_non_json_error_reports_server_error() -> None:
    err_res = MagicMock()
    err_res.status_code = 500
    err_res.reason = "Internal Server Error"
    err_res.text = "Internal Server Error"
    err_res.json.side_effect = requests.exceptions.JSONDecodeError("Expecting value", "", 0)

    with patch("frontend.streamlit_app.requests.post", return_value=err_res):
        with pytest.raises(BackendAPIError, match=r"Upload failed: HTTP 500"):
            upload_pdf_to_api("doc.pdf", b"%PDF-1.4")
