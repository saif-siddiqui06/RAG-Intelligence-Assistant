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


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    retrieved_chunks: list[RetrievedChunkOut]
    confidence: str
    rewritten_query: str
    session_id: str


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
