

from pathlib import Path


SUPPORTED_TYPES: set[str] = {"pdf", "txt"}


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Extract plain text from a PDF or TXT file.

    Args:
        file_bytes: Raw file content.
        filename:   Original filename used to infer file type.

    Returns:
        Extracted text as a single cleaned string.

    Raises:
        ValueError: If the file extension is not supported.
    """
    suffix = Path(filename).suffix.lower().lstrip(".")
    if suffix not in SUPPORTED_TYPES:
        raise ValueError(
            f"Unsupported file type: .{suffix}. Accepted: {sorted(SUPPORTED_TYPES)}"
        )

    if suffix == "txt":
        return _extract_txt(file_bytes)
    return _extract_pdf(file_bytes)


def _extract_txt(file_bytes: bytes) -> str:
    """Decode TXT bytes — tries UTF-8 then falls back to latin-1."""
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        text = file_bytes.decode("latin-1")
    return _clean_text(text)


def _extract_pdf(file_bytes: bytes) -> str:
    """
    Extract text from PDF using pdfminer.six directly.
    No unstructured, no ML models, no unstructured-inference required.
    """
    import io
    from pdfminer.high_level import extract_text as pdfminer_extract

    pdf_file = io.BytesIO(file_bytes)
    text = pdfminer_extract(pdf_file)
    return _clean_text(text or "")


def _clean_text(text: str) -> str:
    """Strip whitespace per line and collapse blank lines."""
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)