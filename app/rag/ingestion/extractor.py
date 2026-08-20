"""PDF text extraction, one page at a time."""
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.core.exceptions import AppException


@dataclass
class PageText:
    page_number: int  # 1-indexed, for citations later
    text: str


class PDFExtractor:
    """Reliable text extraction with clear, typed failures.

    Corrupt files, password-protected PDFs and scanned/image-only PDFs
    all fail in predictable, distinguishable ways rather than raising
    whatever pypdf happens to throw.
    """

    def extract(self, file_path: Path) -> list[PageText]:
        try:
            reader = PdfReader(str(file_path))
        except (PdfReadError, OSError) as exc:
            raise AppException(f"Could not open PDF file: {exc}", status_code=422) from exc

        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception as exc:
                raise AppException("PDF is password-protected", status_code=422) from exc

        pages: list[PageText] = []
        for index, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text() or ""
            except Exception:
                text = ""
            pages.append(PageText(page_number=index, text=text))
        return pages
