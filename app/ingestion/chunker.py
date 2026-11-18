from dataclasses import dataclass, field
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.ingestion.pdf_loader import ExtractedPDF
from app.ingestion.web_loader import ExtractedWebsite


@dataclass(slots=True)
class TextChunk:
    chunk_index: int
    content: str
    page_number: int | None = None
    source_url: str | None = None
    chunk_metadata: dict[str, Any] = field(default_factory=dict)


class DocumentChunker:
    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        settings = get_settings()
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and strictly less than chunk_size.")

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_pdf(self, extracted_pdf: ExtractedPDF) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        chunk_idx = 0

        for page in extracted_pdf.pages:
            raw_splits = self._splitter.split_text(page.content)
            for piece in raw_splits:
                cleaned = piece.strip()
                if not cleaned:
                    continue
                chunks.append(
                    TextChunk(
                        chunk_index=chunk_idx,
                        content=cleaned,
                        page_number=page.page_number,
                        source_url=None,
                        chunk_metadata={
                            "source_type": "pdf",
                            "filename": extracted_pdf.filename,
                            "page_number": page.page_number,
                        },
                    )
                )
                chunk_idx += 1

        return chunks

    def chunk_website(self, extracted_web: ExtractedWebsite) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        raw_splits = self._splitter.split_text(extracted_web.content)

        for chunk_idx, piece in enumerate(
            (p.strip() for p in raw_splits if p.strip()),
            start=0,
        ):
            chunks.append(
                TextChunk(
                    chunk_index=chunk_idx,
                    content=piece,
                    page_number=None,
                    source_url=extracted_web.source_url,
                    chunk_metadata={
                        "source_type": "website",
                        "title": extracted_web.title,
                        "source_url": extracted_web.source_url,
                    },
                )
            )

        return chunks
