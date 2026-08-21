"""SQLAlchemy ORM models for document/conversation metadata.

Deliberately separate from `app.models.*` (the Pydantic API contracts)
— these describe on-disk rows, not HTTP payloads.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.session import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class DocumentRecord(Base):
    """One row per uploaded source document (currently: PDF)."""

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    filename: Mapped[str] = mapped_column(String(255))
    document_type: Mapped[str] = mapped_column(String(20), default="pdf")
    status: Mapped[str] = mapped_column(String(20), default="pending")
    file_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    storage_path: Mapped[str] = mapped_column(String(500))
    file_size_bytes: Mapped[int] = mapped_column(Integer)
    num_pages: Mapped[int | None] = mapped_column(Integer, nullable=True)
    num_chunks: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    upload_timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    processed_timestamp: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    chunks: Mapped[list["ChunkRecord"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class ChunkRecord(Base):
    """One row per chunk. `vector_id` is the integer id used inside the
    vector store's index — this table is the metadata side-channel every
    vector store backend (FAISS today, Chroma/Qdrant/pgvector later) is
    joined against to recover text + citation metadata for a hit.
    """

    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id"), index=True)
    vector_id: Mapped[int] = mapped_column(Integer, unique=True, index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    page_number: Mapped[int] = mapped_column(Integer)
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    document: Mapped["DocumentRecord"] = relationship(back_populates="chunks")


class UserRecord(Base):
    """Minimal user record — this project has no auth system yet, so
    every conversation is owned by one auto-provisioned default user
    (see app.services.conversation_service.get_or_create_default_user).
    The schema is shaped for real multi-user auth to slot in later
    without a conversations/messages table change.
    """

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    conversations: Mapped[list["ConversationRecord"]] = relationship(back_populates="user")


class ConversationRecord(Base):
    """One row per chat session."""

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True, index=True)
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    user: Mapped["UserRecord | None"] = relationship(back_populates="conversations")
    messages: Mapped[list["MessageRecord"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="MessageRecord.created_at",
    )


class MessageRecord(Base):
    """One row per turn. Read in a bounded window (see
    Settings.conversation_history_window) — never the full history —
    when rewriting a follow-up question.
    """

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    role: Mapped[str] = mapped_column(String(20))  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    conversation: Mapped["ConversationRecord"] = relationship(back_populates="messages")
    sources: Mapped[list["MessageSourceRecord"]] = relationship(
        back_populates="message", cascade="all, delete-orphan"
    )


class MessageSourceRecord(Base):
    """One row per citation attached to an assistant message — persisted
    so past conversations still show their sources, not just the live
    response. Only ever attached to "assistant"-role messages.
    """

    __tablename__ = "message_sources"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_new_id)
    message_id: Mapped[str] = mapped_column(ForeignKey("messages.id"), index=True)
    index: Mapped[int] = mapped_column(Integer)  # matches the [n] marker in the message content
    document_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    page_number: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_id: Mapped[str | None] = mapped_column(String(36), nullable=True)

    message: Mapped["MessageRecord"] = relationship(back_populates="sources")
