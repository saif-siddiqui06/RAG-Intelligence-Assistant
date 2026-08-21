"""Integration tests for document_search_tool and document_summary_tool
— the two agent tools that call back into the existing RAG pipeline —
using the same fake embedder/chat-model pattern as the chat service
tests, no network calls.
"""
import uuid

import pytest

from app.agents.tools.document_search import DocumentSearchTool
from app.agents.tools.document_summary import DocumentSummaryTool
from app.core.config import Settings
from app.database.models import ChunkRecord, DocumentRecord
from app.rag.generation.query_rewriter import QueryRewriter
from app.rag.retrieval.vector_retriever import VectorRetriever
from app.rag.vectorstore.faiss_store import FaissVectorStore
from app.services import chat_service as chat_service_module
from tests.fakes import FakeChatModel, KeywordFakeEmbedder


@pytest.fixture()
def settings(tmp_path) -> Settings:
    return Settings(
        upload_dir=tmp_path / "uploads",
        processed_dir=tmp_path / "processed",
        vectorstore_dir=tmp_path / "vectorstore",
        log_dir=tmp_path / "logs",
        min_relevance_score=0.1,
    )


@pytest.fixture()
def embedder() -> KeywordFakeEmbedder:
    return KeywordFakeEmbedder()


@pytest.fixture()
def vector_store(tmp_path, embedder) -> FaissVectorStore:
    return FaissVectorStore(index_path=tmp_path / "index.faiss", dimension=embedder.dimension)


def _seed_document(db_session, vector_store, embedder, filename="smote.pdf") -> DocumentRecord:
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
    text = "SMOTE oversamples the minority class by generating synthetic samples."
    vector_id = vector_store.add(embedder.embed_documents([text]))[0]
    db_session.add(
        ChunkRecord(document_id=doc.id, vector_id=vector_id, chunk_index=0, page_number=12, content=text)
    )
    db_session.commit()
    return doc


def test_document_search_tool_returns_grounded_answer_and_sources(
    db_session, settings, embedder, vector_store, monkeypatch
):
    _seed_document(db_session, vector_store, embedder)
    chat_model = FakeChatModel(responses=["SMOTE oversamples the minority class [1]."])
    monkeypatch.setattr(
        chat_service_module, "get_vector_retriever", lambda: VectorRetriever(embedder, vector_store)
    )
    monkeypatch.setattr(chat_service_module, "get_query_rewriter", lambda: QueryRewriter(chat_model))
    monkeypatch.setattr(chat_service_module, "get_answer_generator", lambda: _answer_gen(chat_model))

    tool = DocumentSearchTool(db_session, settings)
    result = tool.run(query="What is SMOTE?")

    assert "SMOTE" in result.output
    assert result.sources
    assert result.sources[0].tool == "document_search_tool"
    assert result.sources[0].page_number == 12


def test_document_search_tool_reports_missing_query():
    from app.core.config import Settings as _Settings

    tool = DocumentSearchTool(db=None, settings=_Settings())
    result = tool.run()
    assert result.error == "missing_query"


def test_document_summary_tool_summarizes_by_filename(db_session, settings, embedder, vector_store):
    _seed_document(db_session, vector_store, embedder, filename="research_paper.pdf")
    chat_model = FakeChatModel(responses=["This paper introduces SMOTE for class imbalance."])

    tool = DocumentSummaryTool(db_session, settings, chat_model)
    result = tool.run(filename="research_paper.pdf")

    assert result.output == "This paper introduces SMOTE for class imbalance."
    assert result.sources[0].document_name == "research_paper.pdf"
    assert result.error is None


def test_document_summary_tool_handles_unknown_document(db_session, settings):
    chat_model = FakeChatModel(responses=["should not be used"])
    tool = DocumentSummaryTool(db_session, settings, chat_model)

    result = tool.run(filename="does-not-exist.pdf")

    assert result.error == "not_found"
    assert chat_model.calls == []


def _answer_gen(chat_model):
    from app.rag.generation.answer_generator import AnswerGenerator

    return AnswerGenerator(chat_model)
