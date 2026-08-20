"""Unit tests for PDF text extraction."""
from pathlib import Path

import pytest
from reportlab.pdfgen import canvas

from app.core.exceptions import AppException
from app.rag.ingestion.extractor import PDFExtractor


def _make_pdf(path: Path, pages_text: list[str]) -> None:
    c = canvas.Canvas(str(path))
    for text in pages_text:
        c.drawString(72, 720, text)
        c.showPage()
    c.save()


def test_extract_returns_text_per_page(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    _make_pdf(pdf_path, ["Hello World", "Second page content"])

    pages = PDFExtractor().extract(pdf_path)

    assert len(pages) == 2
    assert pages[0].page_number == 1
    assert "Hello World" in pages[0].text
    assert pages[1].page_number == 2
    assert "Second page content" in pages[1].text


def test_extract_handles_multi_page_documents(tmp_path):
    pdf_path = tmp_path / "multi.pdf"
    _make_pdf(pdf_path, [f"Page number {i}" for i in range(1, 6)])

    pages = PDFExtractor().extract(pdf_path)

    assert len(pages) == 5
    assert [p.page_number for p in pages] == [1, 2, 3, 4, 5]
    assert "Page number 3" in pages[2].text


def test_extract_raises_app_exception_on_invalid_pdf(tmp_path):
    bad_path = tmp_path / "not_a_pdf.pdf"
    bad_path.write_bytes(b"this is definitely not a pdf file")

    with pytest.raises(AppException):
        PDFExtractor().extract(bad_path)
