"""Integration tests for the conversational RAG orchestrator: citation
generation, the hallucination-prevention guard, and conversational
query rewriting across turns — using a fake embedder and a scripted
fake LLM client, no network calls.
"""
import pytest

from app.core.config import Settings
from app.database.models import ChunkRecord, DocumentRecord
from app.models.chat import ChatRequest
from app.rag.generation.answer_generator import AnswerGenerator
from app.rag.generation.prompts import NO_CONTEXT_SENTINEL
from app.rag.generation.query_rewriter import QueryRewriter
from app.rag.retrieval.vector_retriever import VectorRetriever
from app.rag.vectorstore.faiss_store import FaissVectorStore
from app.services import chat_service as chat_service_module
from app.services.chat_service import ChatService
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


def _seed_smote_document(db_session, vector_store, embedder) -> DocumentRecord:
    doc = DocumentRecord(
        filename="smote_paper.pdf",
        document_type="pdf",
        status="completed",
        file_hash="hash-1",
        storage_path="/tmp/smote.pdf",
        file_size_bytes=10,
        num_pages=2,
        num_chunks=2,
    )
    db_session.add(doc)
    db_session.flush()

    texts = [
        "SMOTE oversamples the minority class by generating synthetic samples.",
        "A disadvantage of SMOTE is that it can introduce noise and cause overfitting.",
    ]
    vectors = embedder.embed_documents(texts)
    vector_ids = vector_store.add(vectors)
    for idx, (text, vector_id, page) in enumerate(zip(texts, vector_ids, [12, 15])):
        db_session.add(
            ChunkRecord(
                document_id=doc.id,
                vector_id=vector_id,
                chunk_index=idx,
                page_number=page,
                content=text,
            )
        )
    db_session.commit()
    return doc


def _patch_dependencies(monkeypatch, embedder, vector_store, chat_model):
    retriever = VectorRetriever(embedder, vector_store)
    monkeypatch.setattr(chat_service_module, "get_vector_retriever", lambda: retriever)
    monkeypatch.setattr(chat_service_module, "get_query_rewriter", lambda: QueryRewriter(chat_model))
    monkeypatch.setattr(
        chat_service_module, "get_answer_generator", lambda: AnswerGenerator(chat_model)
    )


def test_citation_generation_maps_markers_to_correct_sources(
    db_session, settings, embedder, vector_store, monkeypatch
):
    _seed_smote_document(db_session, vector_store, embedder)
    chat_model = FakeChatModel(
        responses=["SMOTE oversamples the minority class [1]. A disadvantage is added noise [2]."]
    )
    _patch_dependencies(monkeypatch, embedder, vector_store, chat_model)

    service = ChatService(db_session, settings)
    response = service.ask(ChatRequest(question="What is SMOTE and its disadvantages?"))

    assert response.answer.startswith("SMOTE oversamples")
    assert [s.index for s in response.sources] == [1, 2]
    assert {s.document_name for s in response.sources} == {"smote_paper.pdf"}
    assert {s.page_number for s in response.sources} == {12, 15}
    assert all(s.chunk_id for s in response.sources)


def test_no_relevant_chunks_triggers_hallucination_guard_without_calling_llm(
    db_session, settings, embedder, vector_store, monkeypatch
):
    # Nothing ingested at all — nothing can possibly be relevant.
    chat_model = FakeChatModel(responses=["should never be used"])
    _patch_dependencies(monkeypatch, embedder, vector_store, chat_model)

    service = ChatService(db_session, settings)
    response = service.ask(ChatRequest(question="What is quantum computing?"))

    assert response.answer == NO_CONTEXT_SENTINEL
    assert response.sources == []
    assert response.confidence == "low"
    assert chat_model.calls == []  # generation LLM was never invoked


def test_model_declining_forces_empty_sources(db_session, settings, embedder, vector_store, monkeypatch):
    _seed_smote_document(db_session, vector_store, embedder)
    chat_model = FakeChatModel(responses=[NO_CONTEXT_SENTINEL])
    _patch_dependencies(monkeypatch, embedder, vector_store, chat_model)

    service = ChatService(db_session, settings)
    response = service.ask(ChatRequest(question="What is SMOTE?"))

    assert response.answer == NO_CONTEXT_SENTINEL
    assert response.sources == []
    assert response.confidence == "low"


def test_answer_with_no_citation_markers_falls_back_to_all_sources(
    db_session, settings, embedder, vector_store, monkeypatch
):
    _seed_smote_document(db_session, vector_store, embedder)
    chat_model = FakeChatModel(responses=["SMOTE is an oversampling technique with some drawbacks."])
    _patch_dependencies(monkeypatch, embedder, vector_store, chat_model)

    service = ChatService(db_session, settings)
    response = service.ask(ChatRequest(question="What is SMOTE?"))

    assert len(response.sources) == len(response.retrieved_chunks)


def test_conversational_followup_is_rewritten_using_history(
    db_session, settings, embedder, vector_store, monkeypatch
):
    _seed_smote_document(db_session, vector_store, embedder)
    chat_model = FakeChatModel(
        responses=[
            "SMOTE oversamples the minority class [1].",  # answer #1
            "What are the disadvantages of SMOTE?",  # rewrite of the follow-up
            "A disadvantage of SMOTE is added noise [1].",  # answer #2
        ]
    )
    _patch_dependencies(monkeypatch, embedder, vector_store, chat_model)

    service = ChatService(db_session, settings)
    first = service.ask(ChatRequest(question="What is SMOTE?"))
    second = service.ask(
        ChatRequest(session_id=first.session_id, question="What are its disadvantages?")
    )

    assert second.rewritten_query == "What are the disadvantages of SMOTE?"
    assert second.session_id == first.session_id
    # the rewrite call (2nd LLM call overall) actually saw the first turn
    rewrite_call_messages = chat_model.calls[1]
    assert "What is SMOTE?" in rewrite_call_messages[-1]["content"]


def test_document_filter_is_applied_to_chat_requests(
    db_session, settings, embedder, vector_store, monkeypatch
):
    other_doc = DocumentRecord(
        filename="other.pdf",
        document_type="pdf",
        status="completed",
        file_hash="hash-2",
        storage_path="/tmp/other.pdf",
        file_size_bytes=10,
        num_pages=1,
        num_chunks=1,
    )
    db_session.add(other_doc)
    db_session.flush()
    vector_ids = vector_store.add(embedder.embed_documents(["SMOTE is mentioned here too."]))
    db_session.add(
        ChunkRecord(
            document_id=other_doc.id, vector_id=vector_ids[0], chunk_index=0, page_number=1, content="SMOTE is mentioned here too."
        )
    )
    db_session.commit()

    smote_doc = _seed_smote_document(db_session, vector_store, embedder)
    chat_model = FakeChatModel(responses=["SMOTE oversamples the minority class [1]."])
    _patch_dependencies(monkeypatch, embedder, vector_store, chat_model)

    service = ChatService(db_session, settings)
    response = service.ask(ChatRequest(question="What is SMOTE?", document_id=smote_doc.id))

    assert all(c.document_id == smote_doc.id for c in response.retrieved_chunks)
