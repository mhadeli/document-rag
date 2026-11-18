import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf


class PDFExtractionError(Exception):
    """Base exception for PDF extraction failures."""


class CorruptedPDFError(PDFExtractionError):
    """Raised when a PDF file is corrupted, unreadable, or password-protected."""


class EmptyPDFContentError(PDFExtractionError):
    """Raised when a PDF contains no extractable text (e.g., scanned image-only PDF)."""


@dataclass(slots=True)
class PDFPage:
    page_number: int
    content: str


@dataclass(slots=True)
class ExtractedPDF:
    filename: str
    pages: list[PDFPage] = field(default_factory=list)
    source_type: str = "pdf"


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf_from_bytes(
    pdf_bytes: bytes,
    filename: str = "document.pdf",
) -> ExtractedPDF:
    if not pdf_bytes:
        raise CorruptedPDFError("PDF content is empty.")

    try:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise CorruptedPDFError(f"Failed to open PDF '{filename}': {exc}") from exc

    with doc:
        if doc.needs_pass:
            raise CorruptedPDFError(f"PDF '{filename}' is password-protected.")

        if doc.page_count == 0:
            raise EmptyPDFContentError(f"PDF '{filename}' has no pages.")

        pages: list[PDFPage] = []
        for page_num, page in enumerate(doc, start=1):
            raw_text = page.get_text("text")
            cleaned = normalize_text(raw_text)
            if cleaned:
                pages.append(PDFPage(page_number=page_num, content=cleaned))

    if not pages:
        raise EmptyPDFContentError(
            f"No extractable text found in '{filename}'. "
            "Scanned image-only PDFs are not supported."
        )

    return ExtractedPDF(filename=filename, pages=pages)


def extract_pdf_from_path(
    path: Path | str,
    filename: str | None = None,
) -> ExtractedPDF:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    resolved_name = filename or file_path.name
    return extract_pdf_from_bytes(file_path.read_bytes(), filename=resolved_name)
