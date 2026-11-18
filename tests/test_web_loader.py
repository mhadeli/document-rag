import httpx
import pytest

from app.ingestion.web_loader import (
    EmptyWebContentError,
    InvalidURLError,
    SSRFBlockedError,
    WebFetchError,
    extract_website,
    extract_website_from_html,
    validate_url_and_check_ssrf,
)

SAMPLE_ARTICLE_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Understanding Retrieval-Augmented Generation</title>
</head>
<body>
    <nav>Home | About | Contact | Subscribe</nav>
    <article>
        <h1>Understanding Retrieval-Augmented Generation</h1>
        <p>Retrieval-Augmented Generation (RAG) combines information retrieval with text generation
        models so that responses stay grounded in authoritative enterprise documents.</p>
        <p>Using PostgreSQL and pgvector simplifies operational overhead because relational metadata
        and vector embeddings live inside the same ACID-compliant database.</p>
    </article>
    <footer>Copyright 2026 Example Corp</footer>
</body>
</html>
"""


def test_extract_website_from_valid_html() -> None:
    result = extract_website_from_html(
        SAMPLE_ARTICLE_HTML,
        source_url="https://example.com/rag-guide",
    )
    assert result.source_type == "website"
    assert result.source_url == "https://example.com/rag-guide"
    assert "Understanding Retrieval-Augmented Generation" in result.title
    assert "Retrieval-Augmented Generation (RAG) combines" in result.content
    assert "PostgreSQL and pgvector" in result.content


def test_extract_website_via_mock_transport() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=SAMPLE_ARTICLE_HTML)

    transport = httpx.MockTransport(handler)
    result = extract_website(
        "https://example.com/articles/rag",
        transport=transport,
        resolve_dns=False,
    )
    assert result.source_url == "https://example.com/articles/rag"
    assert "Retrieval-Augmented Generation" in result.content


@pytest.mark.parametrize(
    "bad_url",
    [
        "",
        "   ",
        "ftp://example.com/file.txt",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "http://",
        "not-a-url",
    ],
)
def test_reject_invalid_urls(bad_url: str) -> None:
    with pytest.raises(InvalidURLError):
        validate_url_and_check_ssrf(bad_url, resolve_dns=False)


@pytest.mark.parametrize(
    "ssrf_url",
    [
        "http://localhost:8000/health",
        "http://127.0.0.1:8080/admin",
        "http://0.0.0.0/",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/secret",
        "http://192.168.1.1/router",
        "http://172.16.0.5/internal",
        "http://[::1]:8000/",
        "https://service.internal/api",
        "http://printer.local/",
    ],
)
def test_block_ssrf_targets(ssrf_url: str) -> None:
    with pytest.raises(SSRFBlockedError):
        validate_url_and_check_ssrf(ssrf_url, resolve_dns=False)


def test_handle_http_error_and_oversized_response() -> None:
    error_transport = httpx.MockTransport(lambda req: httpx.Response(404, text="Not Found"))
    with pytest.raises(WebFetchError, match="HTTP 404"):
        extract_website(
            "https://example.com/missing",
            transport=error_transport,
            resolve_dns=False,
        )

    huge_transport = httpx.MockTransport(lambda req: httpx.Response(200, content=b"x" * 2048))
    with pytest.raises(WebFetchError, match="exceeds maximum size"):
        extract_website(
            "https://example.com/huge",
            max_bytes=512,
            transport=huge_transport,
            resolve_dns=False,
        )


def test_reject_empty_html_content() -> None:
    with pytest.raises(EmptyWebContentError):
        extract_website_from_html("<html><body></body></html>", "https://example.com/empty")
