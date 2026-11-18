import uuid
from abc import ABC, abstractmethod
from pathlib import Path

from app.config import get_settings

PDF_MAGIC_HEADER = b"%PDF-"


class StorageError(Exception):
    """Base exception for file storage errors."""


class FileTooLargeError(StorageError):
    """Raised when an uploaded file exceeds the maximum allowed size."""


class InvalidPDFError(StorageError):
    """Raised when uploaded content is empty or not a valid PDF."""


class PathTraversalError(StorageError):
    """Raised when a storage key attempts to escape the base upload directory."""


class FileStorage(ABC):
    @abstractmethod
    def save_pdf(self, content: bytes, key: str | None = None) -> str:
        """Validate and save PDF bytes, returning the storage key."""

    @abstractmethod
    def read(self, key: str) -> bytes:
        """Read bytes for the given storage key."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Delete the file associated with the given storage key if it exists."""


class LocalFileStorage(FileStorage):
    def __init__(
        self,
        base_dir: Path | str | None = None,
        max_upload_mb: int | None = None,
    ) -> None:
        settings = get_settings()
        self.base_dir = Path(base_dir or settings.upload_dir).resolve()
        self.max_upload_mb = max_upload_mb if max_upload_mb is not None else settings.max_upload_mb
        self.max_upload_bytes = self.max_upload_mb * 1024 * 1024
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, key: str) -> Path:
        if not key or "/" in key or "\\" in key or ".." in key:
            raise PathTraversalError(f"Invalid storage key: {key!r}")

        target_path = (self.base_dir / key).resolve()
        if target_path.parent != self.base_dir:
            raise PathTraversalError(f"Storage key escapes upload directory: {key!r}")

        return target_path

    def validate_pdf(self, content: bytes) -> None:
        if not content:
            raise InvalidPDFError("Uploaded file is empty.")

        if len(content) > self.max_upload_bytes:
            raise FileTooLargeError(
                f"PDF exceeds the maximum allowed size of {self.max_upload_mb} MB."
            )

        if not content.startswith(PDF_MAGIC_HEADER):
            raise InvalidPDFError("Uploaded file is not a valid PDF (missing %PDF- header).")

    def save_pdf(self, content: bytes, key: str | None = None) -> str:
        self.validate_pdf(content)
        if key is None:
            storage_key = f"{uuid.uuid4()}.pdf"
        else:
            if not key.strip():
                raise PathTraversalError("Storage key cannot be empty.")
            storage_key = key if key.lower().endswith(".pdf") else f"{key}.pdf"

        target_path = self.resolve_path(storage_key)
        target_path.write_bytes(content)
        return storage_key

    def read(self, key: str) -> bytes:
        target_path = self.resolve_path(key)
        if not target_path.is_file():
            raise FileNotFoundError(f"Stored file not found for key: {key}")
        return target_path.read_bytes()

    def delete(self, key: str) -> None:
        target_path = self.resolve_path(key)
        target_path.unlink(missing_ok=True)
