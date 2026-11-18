from pathlib import Path

import pymupdf
import pytest

from app.ingestion.pdf_loader import (
    CorruptedPDFError,
    EmptyPDFContentError,
    extract_pdf_from_bytes,
    extract_pdf_from_path,
    normalize_text,
)


def _build_pdf_bytes(page_texts: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def test_normalize_text_collapses_whitespace_and_newlines() -> None:
    raw = "  First   line\twith   tabs. \r\n\r\n\r\n\r\n  Second paragraph.  "
    assert normalize_text(raw) == "First line with tabs.\n\nSecond paragraph."


def test_extract_pdf_preserves_page_numbers_and_skips_blank_pages() -> None:
    pdf_bytes = _build_pdf_bytes(
        [
            "Page one   with   extra   spaces.\nSecond line.",
            "   ",
            "Page three content here.",
        ]
    )

    result = extract_pdf_from_bytes(pdf_bytes, filename="report.pdf")
    assert result.filename == "report.pdf"
    assert result.source_type == "pdf"
    assert len(result.pages) == 2

    assert result.pages[0].page_number == 1
    assert result.pages[0].content == "Page one with extra spaces.\nSecond line."

    # Page 2 was blank, so the next extracted page must retain page_number == 3
    assert result.pages[1].page_number == 3
    assert result.pages[1].content == "Page three content here."


def test_extract_pdf_from_path(tmp_path: Path) -> None:
    pdf_path = tmp_path / "annual_report.pdf"
    pdf_path.write_bytes(_build_pdf_bytes(["Revenue grew 15% in 2026."]))

    result = extract_pdf_from_path(pdf_path)
    assert result.filename == "annual_report.pdf"
    assert len(result.pages) == 1
    assert result.pages[0].page_number == 1
    assert "Revenue grew 15% in 2026." in result.pages[0].content


def test_reject_corrupted_pdf() -> None:
    with pytest.raises(CorruptedPDFError):
        extract_pdf_from_bytes(b"not a valid pdf byte stream", filename="bad.pdf")

    with pytest.raises(CorruptedPDFError):
        extract_pdf_from_bytes(b"", filename="empty.pdf")


def test_reject_empty_text_pdf() -> None:
    blank_pdf = _build_pdf_bytes(["", "   \n\t  "])
    with pytest.raises(EmptyPDFContentError, match="No extractable text found"):
        extract_pdf_from_bytes(blank_pdf, filename="scanned.pdf")


def test_normalize_text_strips_nul_and_control_chars() -> None:
    raw = "λ\x00(ℓ) = x\x01\x1f\x7f\tok\nnext"
    assert normalize_text(raw) == "λ(ℓ) = x ok\nnext"
