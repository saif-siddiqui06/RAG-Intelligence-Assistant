"""Chat service: the conversational RAG orchestrator.

Request lifecycle (see README for the full diagram):

    question -> memory (bounded history) -> query rewrite -> retrieval
             -> context selection -> generation -> citations -> response

This module is the only thing that sequences those steps; each step
itself lives in its own module (app.rag.generation for rewriting/
generation, app.services.retrieval_service / hybrid_retrieval_service
for retrieval) so any one of them can be swapped or tested without
touching the others.

Retrieval itself branches on `Settings.retrieval_mode`:
- "vector" (default) — the unchanged Milestone 2 path, RetrievalService.
- "hybrid" — vector + BM25, fused (RRF) and reranked (cross-encoder),
  via HybridRetrievalService. Only this branch populates
  `retrieval_diagnostics` on the response.

`prepare()` is deliberately a plain method, not a generator — it does
query rewriting and retrieval (which embeds the query, a real failure
point) eagerly, so a bad API key or empty vector store raises a normal
exception the registered FastAPI handlers can turn into a clean JSON
error *before* a streaming response has started. Only the LLM answer
call itself streams — that's the one place a "mid-response" failure is
actually unavoidable, since it's the thing being streamed.
"""
from collections.abc import Iterator
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.database.models import ConversationRecord, MessageRecord
from app.models.chat import (
    ChatRequest,
    ChatResponse,
    ChatStreamMeta,
    RetrievalDiagnostics,
    RetrievedChunkOut,
    SourceCitation,
)
from app.rag.dependencies import (
    get_answer_generator,
    get_query_rewriter,
    get_reranker,
    get_vector_retriever,
)
from app.rag.generation.answer_generator import extract_cited_indices, is_no_context_answer
from app.rag.generation.prompts import NO_CONTEXT_SENTINEL
from app.services.chunk_lookup import RetrievedChunk
from app.services.hybrid_retrieval_service import HybridRetrievalResult, HybridRetrievalService
from app.services.retrieval_service import RetrievalService

STREAM_META_DELIMITER = "\n<<<META>>>\n"


@dataclass
class PreparedChat:
    conversation: ConversationRecord
    original_question: str
    rewritten_query: str
    retrieved: list[RetrievedChunk]
    relevant: list[RetrievedChunk]
    diagnostics: HybridRetrievalResult | None = None


class ChatService:
    def __init__(self, db: Session, settings: Settings) -> None:
        self.db = db
        self.settings = settings
        self.retrieval_service = RetrievalService(db, get_vector_retriever(), settings)
        self.query_rewriter = get_query_rewriter()
        self.answer_generator = get_answer_generator()

    def prepare(self, request: ChatRequest) -> PreparedChat:
        """Memory -> query rewrite -> retrieval -> context selection.
        Everything up to (not including) the answer-generation LLM call.
        """
        conversation = self._get_or_create_conversation(request.session_id)
        history = self._load_history(conversation)
        rewritten_query = self.query_rewriter.rewrite(history, request.question)

        document_type = request.document_type.value if request.document_type else None
        diagnostics: HybridRetrievalResult | None = None

        if self.settings.retrieval_mode == "hybrid":
            hybrid_service = HybridRetrievalService(
                self.db, get_vector_retriever(), get_reranker(), self.settings
            )
            diagnostics = hybrid_service.retrieve(
                rewritten_query, document_id=request.document_id, document_type=document_type
            )
            retrieved = diagnostics.final_chunks
        else:
            retrieved = self.retrieval_service.retrieve(
                rewritten_query,
                top_k=request.top_k,
                document_id=request.document_id,
                document_type=document_type,
            )

        relevant = [c for c in retrieved if c.score >= self.settings.min_relevance_score]
        return PreparedChat(
            conversation, request.question, rewritten_query, retrieved, relevant, diagnostics
        )

    def ask(self, request: ChatRequest) -> ChatResponse:
        prepared = self.prepare(request)
        diagnostics_out = _to_diagnostics_out(prepared.diagnostics)

        if not prepared.relevant:
            self._persist_turn(prepared.conversation, prepared.original_question, NO_CONTEXT_SENTINEL)
            return ChatResponse(
                answer=NO_CONTEXT_SENTINEL,
                sources=[],
                retrieved_chunks=[_to_chunk_out(c) for c in prepared.retrieved],
                confidence="low",
                rewritten_query=prepared.rewritten_query,
                session_id=prepared.conversation.id,
                retrieval_diagnostics=diagnostics_out,
            )

        answer_text = self.answer_generator.generate(prepared.rewritten_query, prepared.relevant)
        sources, confidence = self._finalize(answer_text, prepared.relevant)
        self._persist_turn(prepared.conversation, prepared.original_question, answer_text)

        return ChatResponse(
            answer=answer_text,
            sources=sources,
            retrieved_chunks=[_to_chunk_out(c) for c in prepared.retrieved],
            confidence=confidence,
            rewritten_query=prepared.rewritten_query,
            session_id=prepared.conversation.id,
            retrieval_diagnostics=diagnostics_out,
        )

    def stream_answer(self, prepared: PreparedChat) -> Iterator[str]:
        """Pure generator over an already-`prepare()`d request. Yields
        answer text deltas, then one final chunk:
        STREAM_META_DELIMITER + a ChatStreamMeta JSON blob.
        """
        if not prepared.relevant:
            answer_text = NO_CONTEXT_SENTINEL
            yield answer_text
            sources: list[SourceCitation] = []
            confidence = "low"
        else:
            parts: list[str] = []
            for delta in self.answer_generator.generate_stream(
                prepared.rewritten_query, prepared.relevant
            ):
                parts.append(delta)
                yield delta
            answer_text = "".join(parts)
            sources, confidence = self._finalize(answer_text, prepared.relevant)

        self._persist_turn(prepared.conversation, prepared.original_question, answer_text)

        meta = ChatStreamMeta(
            sources=sources,
            retrieved_chunks=[_to_chunk_out(c) for c in prepared.retrieved],
            confidence=confidence,
            rewritten_query=prepared.rewritten_query,
            session_id=prepared.conversation.id,
            retrieval_diagnostics=_to_diagnostics_out(prepared.diagnostics),
        )
        yield STREAM_META_DELIMITER + meta.model_dump_json()

    def _finalize(
        self, answer_text: str, relevant: list[RetrievedChunk]
    ) -> tuple[list[SourceCitation], str]:
        """Turn inline [n] markers into a Sources list + a confidence
        label. Defends against the model still refusing (or citing
        nothing) even though retrieval cleared the relevance bar.
        """
        if is_no_context_answer(answer_text):
            return [], "low"

        cited_indices = extract_cited_indices(answer_text, max_index=len(relevant))
        if not cited_indices:
            # Answered substantively but cited nothing — credit all
            # provided sources rather than showing an answer with none.
            cited_indices = list(range(1, len(relevant) + 1))

        sources = [
            SourceCitation(
                index=i,
                document_name=relevant[i - 1].filename,
                page_number=relevant[i - 1].page_number,
                chunk_id=relevant[i - 1].chunk_id,
            )
            for i in cited_indices
        ]
        cited_scores = [relevant[i - 1].score for i in cited_indices]
        return sources, self._compute_confidence(cited_scores)

    def _compute_confidence(self, cited_scores: list[float]) -> str:
        """A heuristic proxy, not a calibrated metric — real faithfulness
        scoring is the evaluation milestone's job. This only combines the
        model's own citation behavior with how similar the cited chunks
        actually were to the query.
        """
        if not cited_scores:
            return "low"
        avg_score = sum(cited_scores) / len(cited_scores)
        if avg_score >= self.settings.confidence_high_threshold:
            return "high"
        if avg_score >= self.settings.confidence_medium_threshold:
            return "medium"
        return "low"

    def _get_or_create_conversation(self, session_id: str | None) -> ConversationRecord:
        if session_id:
            existing = self.db.get(ConversationRecord, session_id)
            if existing:
                return existing
            conversation = ConversationRecord(id=session_id)
        else:
            conversation = ConversationRecord()
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def _load_history(self, conversation: ConversationRecord) -> list[tuple[str, str]]:
        """Bounded window, not the full transcript — this is what keeps
        query rewriting cheap and keeps the answer-generation call from
        ever seeing raw chat history at all (see module docstring).
        """
        window = self.settings.conversation_history_window
        recent = conversation.messages[-window:] if window > 0 else []
        return [(m.role, m.content) for m in recent]

    def _persist_turn(self, conversation: ConversationRecord, question: str, answer: str) -> None:
        self.db.add(MessageRecord(conversation_id=conversation.id, role="user", content=question))
        self.db.add(
            MessageRecord(conversation_id=conversation.id, role="assistant", content=answer)
        )
        self.db.commit()


def _to_chunk_out(chunk: RetrievedChunk) -> RetrievedChunkOut:
    return RetrievedChunkOut(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        filename=chunk.filename,
        page_number=chunk.page_number,
        chunk_index=chunk.chunk_index,
        content=chunk.content,
        score=chunk.score,
    )


def _to_diagnostics_out(diagnostics: HybridRetrievalResult | None) -> RetrievalDiagnostics | None:
    if diagnostics is None:
        return None
    return RetrievalDiagnostics(
        vector_results=[_to_chunk_out(c) for c in diagnostics.vector_results],
        keyword_results=[_to_chunk_out(c) for c in diagnostics.keyword_results],
        fused_results=[_to_chunk_out(c) for c in diagnostics.fused_results],
        reranked_results=[_to_chunk_out(c) for c in diagnostics.reranked_results],
    )
