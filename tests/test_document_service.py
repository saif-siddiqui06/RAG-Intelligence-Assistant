"""Integration tests for DocumentService, using a fake embedder and an
isolated FAISS index so nothing here hits the real OpenAI API or the
developer's real data/ directory.
"""
import io
import uuid

import pytest
from reportlab.pdfgen import canvas

from app.core.config import Settings
from app.rag.embeddings.base import BaseEmbedder
from app.rag.ingestion.chunker import ChunkingConfig, RecursiveCharacterChunker
from app.rag.ingestion.extractor import PDFExtractor
from app.rag.ingestion.pipeline import IngestionPipeline
from app.rag.vectorstore.faiss_store import FaissVectorStore
from app.services import document_service as document_service_module
from app.services.document_service import DocumentService, DuplicateDocumentError


class FakeEmbedder(BaseEmbedder):
    """Deterministic, network-free stand-in for OpenAIEmbedder."""

    dimension_size = 8

    @property
    def dimension(self) -> int:
        return self.dimension_size

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)

    def _vector(self, text):
        seed = sum(ord(c) for c in text) or 1
        return [((seed * (i + 1)) % 97) / 97 for i in range(self.dimension_size)]


class _FakeUploadFile:
    """Minimal stand-in for FastAPI's UploadFile."""

    def __init__(self, filename: str, data: bytes):
        self.filename = filename
        self.content_type = "application/pdf"
        self.file = io.BytesIO(data)


def _make_pdf_bytes(tmp_path, text: str) -> bytes:
    path = tmp_path / f"{uuid.uuid4()}.pdf"
    c = canvas.Canvas(str(path))
    c.drawString(72, 720, text)
    c.showPage()
    c.save()
    return path.read_bytes()


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        upload_dir=tmp_path / "uploads",
        processed_dir=tmp_path / "processed",
        vectorstore_dir=tmp_path / "vectorstore",
        log_dir=tmp_path / "logs",
    )


@pytest.fixture(autouse=True)
def _patch_pipeline(monkeypatch, tmp_path):
    """Swap the real OpenAI/FAISS singletons for fast, network-free fakes."""
    embedder = FakeEmbedder()
    vector_store = FaissVectorStore(
        index_path=tmp_path / "test_index.faiss", dimension=embedder.dimension_size
    )

    def _fake_build_pipeline(chunk_size=None, chunk_overlap=None):
        config = ChunkingConfig(chunk_size=chunk_size or 500, chunk_overlap=chunk_overlap or 50)
        return IngestionPipeline(
            extractor=PDFExtractor(),
            chunker=RecursiveCharacterChunker(config),
            embedder=embedder,
            vector_store=vector_store,
        )

    monkeypatch.setattr(document_service_module, "build_ingestion_pipeline", _fake_build_pipeline)
    monkeypatch.setattr(document_service_module, "get_vector_store_instance", lambda: vector_store)
    return vector_store


def test_duplicate_upload_is_rejected(db_session, settings, tmp_path):
    service = DocumentService(db_session, settings)
    data = _make_pdf_bytes(tmp_path, "Duplicate detection test content.")

    first = service.upload(_FakeUploadFile("a.pdf", data))
    assert first.status == "completed"
    assert first.num_chunks and first.num_chunks > 0

    with pytest.raises(DuplicateDocumentError):
        service.upload(_FakeUploadFile("a-renamed-copy.pdf", data))


def test_different_files_are_both_ingested(db_session, settings, tmp_path):
    service = DocumentService(db_session, settings)

    first = service.upload(_FakeUploadFile("a.pdf", _make_pdf_bytes(tmp_path, "Document A content.")))
    second = service.upload(
        _FakeUploadFile("b.pdf", _make_pdf_bytes(tmp_path, "Document B has totally different text."))
    )

    assert first.document_id != second.document_id
    assert {d.document_id for d in service.list_documents()} == {first.document_id, second.document_id}


def test_delete_removes_document_chunks_and_vectors(db_session, settings, tmp_path, _patch_pipeline):
    service = DocumentService(db_session, settings)
    vector_store = _patch_pipeline
    uploaded = service.upload(_FakeUploadFile("a.pdf", _make_pdf_bytes(tmp_path, "Some content to delete.")))

    assert vector_store.count() > 0

    service.delete_document(uploaded.document_id)

    assert vector_store.count() == 0
    assert service.list_documents() == []


def test_non_pdf_upload_is_rejected(db_session, settings):
    service = DocumentService(db_session, settings)

    from app.core.exceptions import AppException

    with pytest.raises(AppException):
        service.upload(_FakeUploadFile("not-a-pdf.txt", b"hello"))
