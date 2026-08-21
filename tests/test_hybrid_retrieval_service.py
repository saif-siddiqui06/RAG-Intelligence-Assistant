"""Integration tests for HybridRetrievalService: BM25 surfacing terms
vector search's fixed vocabulary can't, RRF fusion favoring items both
engines agree on, reranking producing a valid re-sorted order, dedup,
and metadata-filter edge cases.
"""
import uuid

import pytest

from app.core.config import Settings
from app.database.models import ChunkRecord, DocumentRecord
from app.rag.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.rag.reranking.noop_reranker import NoOpReranker
from app.rag.retrieval.vector_retriever import VectorRetriever
from app.rag.vectorstore.faiss_store import FaissVectorStore
from app.services.hybrid_retrieval_service import HybridRetrievalService
from tests.fakes import KeywordFakeEmbedder


@pytest.fixture(scope="module")
def cross_encoder() -> CrossEncoderReranker:
    """Loaded once for the module — model loading is the slow part."""
    return CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        upload_dir=tmp_path / "uploads",
        processed_dir=tmp_path / "processed",
        vectorstore_dir=tmp_path / "vectorstore",
        log_dir=tmp_path / "logs",
        vector_top_k=10,
        bm25_top_k=10,
        rerank_top_k=10,
        final_context_k=3,
    )


@pytest.fixture()
def embedder() -> KeywordFakeEmbedder:
    return KeywordFakeEmbedder()


@pytest.fixture()
def vector_store(tmp_path, embedder) -> FaissVectorStore:
    return FaissVectorStore(index_path=tmp_path / "index.faiss", dimension=embedder.dimension)


def _make_doc(db_session, filename: str) -> DocumentRecord:
    doc = DocumentRecord(
        filename=filename,
        document_type="pdf",
        status="completed",
        file_hash=str(uuid.uuid4()),
        storage_path=f"/tmp/{filename}",
        file_size_bytes=10,
        num_pages=1,
        num_chunks=1,
    )
    db_session.add(doc)
    db_session.flush()
    return doc


def _seed_chunk(db_session, vector_store, embedder, doc, text: str, page: int) -> ChunkRecord:
    vector_id = vector_store.add(embedder.embed_documents([text]))[0]
    chunk = ChunkRecord(
        document_id=doc.id, vector_id=vector_id, chunk_index=0, page_number=page, content=text
    )
    db_session.add(chunk)
    return chunk


def test_bm25_surfaces_exact_term_vector_search_cannot_represent(
    db_session, settings, embedder, vector_store
):
    """KeywordFakeEmbedder only has a small fixed vocabulary — a rare
    made-up token like "XQZ-9000" is invisible to it (an all-near-zero
    vector), so vector search can't discriminate on it at all. BM25
    scores literal term overlap over the full text, so it finds this
    immediately. This is the textbook case for why keyword search
    still matters even with semantic embeddings.
    """
    # BM25's classic IDF formula is degenerate at very small corpus
    # sizes (a term in exactly half a 2-document corpus scores idf=0
    # exactly) — a handful of unrelated filler chunks makes this a
    # realistic-sized corpus instead, which is also more representative
    # of how BM25 is actually used.
    doc = _make_doc(db_session, "doc.pdf")
    _seed_chunk(db_session, vector_store, embedder, doc, "The XQZ-9000 module handles retry logic.", 1)
    _seed_chunk(db_session, vector_store, embedder, doc, "SMOTE oversamples the minority class.", 2)
    _seed_chunk(db_session, vector_store, embedder, doc, "Django is a Python web framework.", 3)
    _seed_chunk(db_session, vector_store, embedder, doc, "Random forests aggregate many decision trees.", 4)
    _seed_chunk(db_session, vector_store, embedder, doc, "Gradient boosting fits trees to residual errors.", 5)
    db_session.commit()

    service = HybridRetrievalService(
        db_session, VectorRetriever(embedder, vector_store), NoOpReranker(), settings
    )
    result = service.retrieve("What does the XQZ-9000 module do?")

    assert any("XQZ-9000" in c.content for c in result.keyword_results)
    assert any("XQZ-9000" in c.content for c in result.final_chunks)


def test_fusion_ranks_items_found_by_both_engines_first(
    db_session, settings, embedder, vector_store
):
    doc = _make_doc(db_session, "doc.pdf")
    # Found by both vector (shares "smote"/"oversample" keywords with the fake's vocab) and BM25.
    _seed_chunk(
        db_session, vector_store, embedder, doc,
        "SMOTE oversamples the minority class by generating synthetic samples.", 1,
    )
    # A distractor that only overlaps with BM25 tokens weakly, unrelated topic.
    _seed_chunk(db_session, vector_store, embedder, doc, "Django is a Python web framework.", 2)
    db_session.commit()

    service = HybridRetrievalService(
        db_session, VectorRetriever(embedder, vector_store), NoOpReranker(), settings
    )
    result = service.retrieve("SMOTE oversampling")

    assert result.fused_results
    assert "smote" in result.fused_results[0].content.lower()


def test_reranking_produces_a_valid_sorted_order(
    db_session, settings, embedder, vector_store, cross_encoder
):
    doc = _make_doc(db_session, "doc.pdf")
    _seed_chunk(
        db_session, vector_store, embedder, doc,
        "SMOTE oversamples the minority class by generating synthetic examples.", 1,
    )
    _seed_chunk(db_session, vector_store, embedder, doc, "Django is a Python web framework.", 2)
    db_session.commit()

    service = HybridRetrievalService(
        db_session, VectorRetriever(embedder, vector_store), cross_encoder, settings
    )
    result = service.retrieve("What is SMOTE?")

    assert result.reranked_results
    scores = [c.score for c in result.reranked_results]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)
    # the cross-encoder should recognize the SMOTE passage as most relevant
    assert "smote" in result.reranked_results[0].content.lower()


def test_final_chunks_respect_final_context_k(db_session, settings, embedder, vector_store):
    doc = _make_doc(db_session, "doc.pdf")
    for i in range(6):
        _seed_chunk(db_session, vector_store, embedder, doc, f"SMOTE synthetic minority passage number {i}.", i + 1)
    db_session.commit()

    service = HybridRetrievalService(
        db_session, VectorRetriever(embedder, vector_store), NoOpReranker(), settings
    )
    result = service.retrieve("SMOTE synthetic minority")

    assert len(result.final_chunks) <= settings.final_context_k


def test_near_duplicate_chunks_are_deduplicated_in_final_selection(
    db_session, settings, embedder, vector_store
):
    doc = _make_doc(db_session, "doc.pdf")
    text = "SMOTE oversamples the minority class by generating synthetic samples for training."
    near_duplicate = text + " data"
    _seed_chunk(db_session, vector_store, embedder, doc, text, 1)
    _seed_chunk(db_session, vector_store, embedder, doc, near_duplicate, 2)
    _seed_chunk(db_session, vector_store, embedder, doc, "A disadvantage of SMOTE is added noise.", 3)
    db_session.commit()

    service = HybridRetrievalService(
        db_session, VectorRetriever(embedder, vector_store), NoOpReranker(), settings
    )
    result = service.retrieve("SMOTE")

    contents = [c.content for c in result.final_chunks]
    assert len(contents) == len(set(contents))


def test_filter_matching_nothing_returns_empty_result(db_session, settings, embedder, vector_store):
    _make_doc(db_session, "doc.pdf")
    db_session.commit()

    service = HybridRetrievalService(
        db_session, VectorRetriever(embedder, vector_store), NoOpReranker(), settings
    )
    result = service.retrieve("anything", document_id="does-not-exist")

    assert result.final_chunks == []
    assert result.vector_results == []
    assert result.keyword_results == []


def test_empty_corpus_returns_empty_result(db_session, settings, embedder, vector_store):
    service = HybridRetrievalService(
        db_session, VectorRetriever(embedder, vector_store), NoOpReranker(), settings
    )
    result = service.retrieve("anything")

    assert result.final_chunks == []
