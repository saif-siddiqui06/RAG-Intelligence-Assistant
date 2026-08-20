"""API schemas for document ingestion and management."""
from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class DocumentType(str, Enum):
    pdf = "pdf"


class DocumentStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"


class DocumentMetadataResponse(BaseModel):
    document_id: str
    filename: str
    document_type: DocumentType
    status: DocumentStatus
    file_hash: str
    file_size_bytes: int
    num_pages: int | None = None
    num_chunks: int | None = None
    upload_timestamp: datetime
    processed_timestamp: datetime | None = None
    error_message: str | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentMetadataResponse]
    total: int


class ChunkPreview(BaseModel):
    chunk_id: str
    page_number: int
    chunk_index: int
    content: str


class DeleteResponse(BaseModel):
    document_id: str
    deleted: bool
    message: str


class IngestionStats(BaseModel):
    total_documents: int
    total_chunks: int
    vector_count: int
