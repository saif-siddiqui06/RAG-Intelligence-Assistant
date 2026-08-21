"""Tests for semantic retrieval, metadata filtering and context
selection (near-duplicate removal), using a network-free keyword-based
fake embedder and a real (tmp_path-scoped) FAISS index.
"""
import uuid

import pytest

from app.core.config import Settings
from app.database.models import ChunkRecord, DocumentRecord
from app.rag.retrieval.vector_retriever import VectorRetriever
from app.rag.vectorstore.faiss_store import FaissVectorStore
from app.services.retrieval_service import RetrievalService
from tests.fakes import KeywordFakeEmbedder


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        upload_dir=tmp_path / "uploads",
        processed_dir=tmp_path / "processed",
        vectorstore_dir=tmp_path / "vectorstore",
        log_dir=tmp_path / "logs",
        retrieval_top_k=3,
    )


@pytest.fixture()
def embedder() -> KeywordFakeEmbedder:
    return KeywordFakeEmbedder()


@pytest.fixture()
def vector_store(tmp_path, embedder) -> FaissVectorStore:
    return FaissVectorStore(index_path=tmp_path / "index.faiss", dimension=embedder.dimension)


def _seed_document(db_session, vector_store, embedder, filename, document_type, chunks_text):
    doc = DocumentRecord(
        filename=filename,
        document_type=document_type,
        status="completed",
        file_hash=str(uuid.uuid4()),
        storage_path=f"/tmp/{filename}",
        file_size_bytes=100,
        num_pages=1,
        num_chunks=len(chunks_text),
    )
    db_session.add(doc)
    db_session.flush()

    vectors = embedder.embed_documents(chunks_text)
    vector_ids = vector_store.add(vectors)
    for idx, (text, vector_id) in enumerate(zip(chunks_text, vector_ids)):
        db_session.add(
            ChunkRecord(
                document_id=doc.id,
                vector_id=vector_id,
                chunk_index=idx,
                page_number=idx + 1,
                content=text,
            )
        )
    db_session.commit()
    return doc


def test_semantic_retrieval_ranks_relevant_chunks_higher(db_session, settings, vector_store, embedder):
    _seed_document(
        db_session,
        vector_store,
        embedder,
        "smote.pdf",
        "pdf",
        [
            "SMOTE oversamples the minority class by generating synthetic samples.",
            "A disadvantage of SMOTE is that it can introduce noise and cause overfitting.",
        ],
    )
    _seed_document(
        db_session,
        vector_store,
        embedder,
        "django.pdf",
        "pdf",
        ["Django is a Python web framework for building web applications."],
    )

    service = RetrievalService(db_session, VectorRetriever(embedder, vector_store), settings)
    results = service.retrieve("What is SMOTE?", top_k=3)

    assert results
    assert results[0].filename == "smote.pdf"
    assert "smote" in results[0].content.lower()


def test_document_id_filter_restricts_to_one_document(db_session, settings, vector_store, embedder):
    doc_a = _seed_document(
        db_session, vector_store, embedder, "smote_a.pdf", "pdf", ["SMOTE oversamples the minority class."]
    )
    _seed_document(
        db_session,
        vector_store,
        embedder,
        "smote_b.pdf",
        "pdf",
        ["SMOTE oversamples the minority class in a second document."],
    )

    service = RetrievalService(db_session, VectorRetriever(embedder, vector_store), settings)
    results = service.retrieve("SMOTE", top_k=5, document_id=doc_a.id)

    assert results
    assert all(r.document_id == doc_a.id for r in results)


def test_document_type_filter_restricts_by_type(db_session, settings, vector_store, embedder):
    _seed_document(db_session, vector_store, embedder, "smote.pdf", "pdf", ["SMOTE oversamples the minority class."])
    _seed_document(
        db_session, vector_store, embedder, "smote_notes.txt", "txt", ["SMOTE synthetic minority oversampling notes."]
    )

    service = RetrievalService(db_session, VectorRetriever(embedder, vector_store), settings)
    results = service.retrieve("SMOTE", top_k=5, document_type="pdf")

    assert results
    assert all(r.filename.endswith(".pdf") for r in results)


def test_filter_matching_no_documents_returns_empty(db_session, settings, vector_store, embedder):
    _seed_document(db_session, vector_store, embedder, "smote.pdf", "pdf", ["SMOTE oversamples the minority class."])

    service = RetrievalService(db_session, VectorRetriever(embedder, vector_store), settings)
    results = service.retrieve("SMOTE", top_k=5, document_id="does-not-exist")

    assert results == []


def test_empty_vector_store_returns_empty(db_session, settings, vector_store, embedder):
    service = RetrievalService(db_session, VectorRetriever(embedder, vector_store), settings)

    assert service.retrieve("anything", top_k=5) == []


def test_near_duplicate_chunks_are_deduplicated(db_session, settings, vector_store, embedder):
    _seed_document(
        db_session,
        vector_store,
        embedder,
        "smote.pdf",
        "pdf",
        [
            "SMOTE oversamples the minority class by generating synthetic samples for training.",
            "SMOTE oversamples the minority class by generating synthetic samples for training data.",
            "A disadvantage of SMOTE is that it can introduce noise and cause overfitting.",
        ],
    )

    service = RetrievalService(db_session, VectorRetriever(embedder, vector_store), settings)
    results = service.retrieve("SMOTE", top_k=3)

    contents = [r.content for r in results]
    assert len(contents) == len(set(contents))
    assert len(results) <= 2
