"""Document ingestion & management endpoints.

Upload runs the full extraction -> cleaning -> chunking -> embedding ->
vector-store pipeline synchronously and returns once it's done. Kept
simple (no background jobs) for this milestone; large batches will
want an async task queue later.
"""
import logging

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from sqlalchemy.orm import Session

from app.api.deps import get_app_settings, get_db
from app.core.config import Settings, get_settings
from app.core.rate_limit import limiter
from app.models.document import (
    ChunkPreview,
    DeleteResponse,
    DocumentListResponse,
    DocumentMetadataResponse,
    IngestionStats,
)
from app.services.document_service import DocumentService, document_to_response

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=list[DocumentMetadataResponse], status_code=201)
@limiter.limit(get_settings().rate_limit_upload)
def upload_documents(
    request: Request,
    files: list[UploadFile] = File(..., description="One or more PDF files"),
    chunk_size: int | None = Query(default=None, gt=0),
    chunk_overlap: int | None = Query(default=None, ge=0),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
) -> list[DocumentMetadataResponse]:
    service = DocumentService(db, settings)
    return [service.upload(f, chunk_size, chunk_overlap) for f in files]


@router.get("", response_model=DocumentListResponse)
def list_documents(
    db: Session = Depends(get_db), settings: Settings = Depends(get_app_settings)
) -> DocumentListResponse:
    service = DocumentService(db, settings)
    documents = service.list_documents()
    return DocumentListResponse(documents=documents, total=len(documents))


@router.get("/stats/summary", response_model=IngestionStats)
def get_stats(
    db: Session = Depends(get_db), settings: Settings = Depends(get_app_settings)
) -> IngestionStats:
    return DocumentService(db, settings).get_stats()


@router.get("/{document_id}", response_model=DocumentMetadataResponse)
def get_document(
    document_id: str, db: Session = Depends(get_db), settings: Settings = Depends(get_app_settings)
) -> DocumentMetadataResponse:
    service = DocumentService(db, settings)
    return document_to_response(service.get_document(document_id))


@router.get("/{document_id}/chunks", response_model=list[ChunkPreview])
def get_document_chunks(
    document_id: str, db: Session = Depends(get_db), settings: Settings = Depends(get_app_settings)
) -> list[ChunkPreview]:
    service = DocumentService(db, settings)
    return [
        ChunkPreview(chunk_id=c.id, page_number=c.page_number, chunk_index=c.chunk_index, content=c.content)
        for c in service.get_chunks(document_id)
    ]


@router.delete("/{document_id}", response_model=DeleteResponse)
def delete_document(
    document_id: str, db: Session = Depends(get_db), settings: Settings = Depends(get_app_settings)
) -> DeleteResponse:
    service = DocumentService(db, settings)
    service.delete_document(document_id)
    return DeleteResponse(document_id=document_id, deleted=True, message="Document deleted")


@router.post("/{document_id}/reindex", response_model=DocumentMetadataResponse)
def reindex_document(
    document_id: str,
    chunk_size: int | None = Query(default=None, gt=0),
    chunk_overlap: int | None = Query(default=None, ge=0),
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
) -> DocumentMetadataResponse:
    service = DocumentService(db, settings)
    return service.reindex_document(document_id, chunk_size, chunk_overlap)
