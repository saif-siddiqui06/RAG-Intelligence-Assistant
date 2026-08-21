"""API schemas for the conversational RAG (chat) endpoints."""
from pydantic import BaseModel, Field

from app.models.document import DocumentType


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = None
    document_id: str | None = None
    document_type: DocumentType | None = None
    top_k: int | None = Field(default=None, gt=0, le=20)


class RetrievedChunkOut(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    content: str
    score: float


class SourceCitation(BaseModel):
    index: int  # matches the [n] marker inline in `answer`
    document_name: str
    page_number: int
    chunk_id: str


class RetrievalDiagnostics(BaseModel):
    """Per-stage output of the hybrid retrieval pipeline, for
    development/debugging — only populated when RETRIEVAL_MODE=hybrid;
    None in vector-only mode.
    """

    vector_results: list[RetrievedChunkOut]
    keyword_results: list[RetrievedChunkOut]
    fused_results: list[RetrievedChunkOut]
    reranked_results: list[RetrievedChunkOut]


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    retrieved_chunks: list[RetrievedChunkOut]
    confidence: str
    rewritten_query: str
    session_id: str
    retrieval_diagnostics: RetrievalDiagnostics | None = None


class ChatStreamMeta(BaseModel):
    """Everything in ChatResponse except `answer` — sent as a trailing
    JSON blob after the streamed answer text (see
    app.services.chat_service.STREAM_META_DELIMITER).
    """

    sources: list[SourceCitation]
    retrieved_chunks: list[RetrievedChunkOut]
    confidence: str
    rewritten_query: str
    session_id: str
    retrieval_diagnostics: RetrievalDiagnostics | None = None
