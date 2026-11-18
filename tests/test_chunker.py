import pytest

from app.ingestion.chunker import DocumentChunker
from app.ingestion.pdf_loader import ExtractedPDF, PDFPage
from app.ingestion.web_loader import ExtractedWebsite


def test_chunk_pdf_preserves_page_numbers_and_sequential_indexes() -> None:
    pdf = ExtractedPDF(
        filename="financials.pdf",
        pages=[
            PDFPage(
                page_number=1,
                content="Paragraph A is about Q1 revenue growth. " * 5,
            ),
            PDFPage(
                page_number=4,
                content="Paragraph B is about Q4 operating margins. " * 5,
            ),
        ],
    )

    chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
    chunks = chunker.chunk_pdf(pdf)

    assert len(chunks) >= 4
    # Verify sequential 0-based indexing across all pages
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    # Every chunk stays within configured size and preserves metadata
    for chunk in chunks:
        assert len(chunk.content) <= 100
        assert chunk.page_number in (1, 4)
        assert chunk.source_url is None
        assert chunk.chunk_metadata["source_type"] == "pdf"
        assert chunk.chunk_metadata["filename"] == "financials.pdf"
        assert chunk.chunk_metadata["page_number"] == chunk.page_number

    assert chunks[0].page_number == 1
    assert chunks[-1].page_number == 4


def test_chunk_website_preserves_url_and_title_metadata() -> None:
    website = ExtractedWebsite(
        source_url="https://example.com/blog/rag-architecture",
        title="Modern RAG Architecture",
        content=("PostgreSQL with pgvector enables unified vector search. " * 8).strip(),
    )

    chunker = DocumentChunker(chunk_size=120, chunk_overlap=20)
    chunks = chunker.chunk_website(website)

    assert len(chunks) >= 3
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    for chunk in chunks:
        assert len(chunk.content) <= 120
        assert chunk.page_number is None
        assert chunk.source_url == "https://example.com/blog/rag-architecture"
        assert chunk.chunk_metadata == {
            "source_type": "website",
            "title": "Modern RAG Architecture",
            "source_url": "https://example.com/blog/rag-architecture",
        }


def test_chunker_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        DocumentChunker(chunk_size=0, chunk_overlap=0)

    with pytest.raises(ValueError, match="chunk_overlap"):
        DocumentChunker(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError, match="chunk_overlap"):
        DocumentChunker(chunk_size=100, chunk_overlap=-5)
