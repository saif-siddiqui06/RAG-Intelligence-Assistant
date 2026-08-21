"""Tests for conversation CRUD, source persistence, and title
generation — the core of production-style conversational memory.
"""
import pytest

from app.core.config import Settings
from app.core.exceptions import NotFoundError
from app.database.models import ChunkRecord, DocumentRecord
from app.models.chat import ChatRequest
from app.rag.retrieval.vector_retriever import VectorRetriever
from app.rag.vectorstore.faiss_store import FaissVectorStore
from app.services import chat_service as chat_service_module
from app.services.chat_service import ChatService
from app.services.conversation_service import ConversationService, get_or_create_default_user
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


def test_get_or_create_default_user_is_idempotent(db_session):
    first = get_or_create_default_user(db_session)
    second = get_or_create_default_user(db_session)
    assert first.id == second.id


def test_create_list_get_delete_conversation(db_session):
    service = ConversationService(db_session)

    created = service.create_conversation()
    assert created.title is None

    listed = service.list_conversations()
    assert [c.id for c, _ in listed] == [created.id]
    assert listed[0][1] == 0  # no messages yet

    fetched = service.get_conversation(created.id)
    assert fetched.id == created.id

    service.delete_conversation(created.id)
    with pytest.raises(NotFoundError):
        service.get_conversation(created.id)


def test_get_missing_conversation_raises_not_found(db_session):
    with pytest.raises(NotFoundError):
        ConversationService(db_session).get_conversation("does-not-exist")


def _patch_dependencies(monkeypatch, embedder, vector_store, chat_model):
    monkeypatch.setattr(
        chat_service_module, "get_vector_retriever", lambda: VectorRetriever(embedder, vector_store)
    )
    from app.rag.generation.answer_generator import AnswerGenerator
    from app.rag.generation.query_rewriter import QueryRewriter

    monkeypatch.setattr(chat_service_module, "get_query_rewriter", lambda: QueryRewriter(chat_model))
    monkeypatch.setattr(
        chat_service_module, "get_answer_generator", lambda: AnswerGenerator(chat_model)
    )
    monkeypatch.setattr(chat_service_module, "get_chat_model", lambda: chat_model)


def test_first_turn_generates_a_title_and_persists_sources(db_session, settings, monkeypatch, tmp_path):
    embedder = KeywordFakeEmbedder()
    vector_store = FaissVectorStore(index_path=tmp_path / "index.faiss", dimension=embedder.dimension)
    doc = DocumentRecord(
        filename="smote.pdf", document_type="pdf", status="completed", file_hash="h1",
        storage_path="/tmp/smote.pdf", file_size_bytes=1, num_pages=1, num_chunks=1,
    )
    db_session.add(doc)
    db_session.flush()
    text = "SMOTE oversamples the minority class."
    vector_id = vector_store.add(embedder.embed_documents([text]))[0]
    db_session.add(ChunkRecord(document_id=doc.id, vector_id=vector_id, chunk_index=0, page_number=7, content=text))
    db_session.commit()

    chat_model = FakeChatModel(responses=["SMOTE is a technique [1].", "SMOTE Basics"])
    _patch_dependencies(monkeypatch, embedder, vector_store, chat_model)

    response = ChatService(db_session, settings).ask(ChatRequest(question="What is SMOTE?"))

    conversation = ConversationService(db_session).get_conversation(response.session_id)
    assert conversation.title == "SMOTE Basics"
    assert conversation.user_id is not None
    assistant_message = [m for m in conversation.messages if m.role == "assistant"][0]
    assert len(assistant_message.sources) == 1
    assert assistant_message.sources[0].page_number == 7


def test_title_generation_falls_back_on_llm_failure(db_session, settings, monkeypatch, tmp_path):
    # No document seeded -> retrieval finds nothing -> the no-context path
    # never calls answer_generator, so the *only* chat_model call in this
    # whole request is title generation. Making it always raise isolates
    # exactly that failure.
    embedder = KeywordFakeEmbedder()
    vector_store = FaissVectorStore(index_path=tmp_path / "index.faiss", dimension=embedder.dimension)

    class BoomChatModel(FakeChatModel):
        def complete(self, messages, temperature=0):
            raise RuntimeError("quota exceeded")

    chat_model = BoomChatModel()
    _patch_dependencies(monkeypatch, embedder, vector_store, chat_model)

    long_question = "What is SMOTE and why does it matter for imbalanced datasets?"
    response = ChatService(db_session, settings).ask(ChatRequest(question=long_question))

    conversation = ConversationService(db_session).get_conversation(response.session_id)
    assert conversation.title is not None
    assert conversation.title.endswith("...")  # truncated-question fallback
