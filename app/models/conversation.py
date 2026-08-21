"""API schemas for conversation management."""
from datetime import datetime

from pydantic import BaseModel


class ConversationSummary(BaseModel):
    conversation_id: str
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    message_count: int


class MessageSourceOut(BaseModel):
    index: int
    document_name: str | None = None
    page_number: int | None = None
    chunk_id: str | None = None


class MessageOut(BaseModel):
    message_id: str
    role: str
    content: str
    created_at: datetime
    sources: list[MessageSourceOut] = []


class ConversationDetail(BaseModel):
    conversation_id: str
    title: str | None = None
    created_at: datetime
    updated_at: datetime
    messages: list[MessageOut]


class DeleteConversationResponse(BaseModel):
    conversation_id: str
    deleted: bool
    message: str
