from pathlib import Path

import pytest

from app.storage.local import (
    FileTooLargeError,
    InvalidPDFError,
    LocalFileStorage,
    PathTraversalError,
)


def test_save_read_and_delete_pdf(tmp_path: Path) -> None:
    storage = LocalFileStorage(base_dir=tmp_path, max_upload_mb=1)
    pdf_bytes = b"%PDF-1.7\nSample PDF content"

    key = storage.save_pdf(pdf_bytes)
    assert key.endswith(".pdf")
    assert (tmp_path / key).is_file()
    assert storage.read(key) == pdf_bytes

    storage.delete(key)
    assert not (tmp_path / key).exists()
    # Deleting again should be idempotent
    storage.delete(key)


def test_reject_invalid_pdf_content(tmp_path: Path) -> None:
    storage = LocalFileStorage(base_dir=tmp_path, max_upload_mb=1)

    with pytest.raises(InvalidPDFError, match="empty"):
        storage.save_pdf(b"")

    with pytest.raises(InvalidPDFError, match="not a valid PDF"):
        storage.save_pdf(b"This is a plain text file pretending to be a PDF")


def test_reject_oversized_pdf(tmp_path: Path) -> None:
    storage = LocalFileStorage(base_dir=tmp_path, max_upload_mb=1)
    oversized_bytes = b"%PDF-1.7\n" + (b"a" * (1024 * 1024 + 10))

    with pytest.raises(FileTooLargeError, match="exceeds the maximum allowed size"):
        storage.save_pdf(oversized_bytes)


@pytest.mark.parametrize(
    "malicious_key",
    [
        "../secret.pdf",
        "../../etc/passwd",
        "subdir/file.pdf",
        "\\windows\\path.pdf",
        "",
    ],
)
def test_prevent_path_traversal(tmp_path: Path, malicious_key: str) -> None:
    storage = LocalFileStorage(base_dir=tmp_path, max_upload_mb=1)
    valid_pdf = b"%PDF-1.7\nhello"

    with pytest.raises(PathTraversalError):
        storage.save_pdf(valid_pdf, key=malicious_key)

    with pytest.raises(PathTraversalError):
        storage.read(malicious_key)

    with pytest.raises(PathTraversalError):
        storage.delete(malicious_key)
