"""Business logic for document ingestion and management.

Talks to app.rag (extraction/chunking/embeddings/vector store) and
app.database (metadata persistence). API endpoints never touch either
directly — they only call methods on DocumentService.
"""
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import UploadFile
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.exceptions import AppException, NotFoundError
from app.database.models import ChunkRecord, DocumentRecord
from app.models.document import DocumentMetadataResponse, IngestionStats
from app.rag.dependencies import build_ingestion_pipeline, get_vector_store_instance
from app.rag.ingestion.hasher import compute_file_hash

logger = logging.getLogger(__name__)

ALLOWED_SUFFIX = ".pdf"


class DuplicateDocumentError(AppException):
    def __init__(self, document_id: str) -> None:
        super().__init__(f"Document already ingested (document_id={document_id})", status_code=409)
        self.document_id = document_id


def document_to_response(record: DocumentRecord) -> DocumentMetadataResponse:
    """Explicit field-by-field mapping — the ORM's `id` becomes the
    API's `document_id`, so this can't be done with blind attribute
    auto-mapping (model_validate) alone.
    """
    return DocumentMetadataResponse(
        document_id=record.id,
        filename=record.filename,
        document_type=record.document_type,
        status=record.status,
        file_hash=record.file_hash,
        file_size_bytes=record.file_size_bytes,
        num_pages=record.num_pages,
        num_chunks=record.num_chunks,
        upload_timestamp=record.upload_timestamp,
        processed_timestamp=record.processed_timestamp,
        error_message=record.error_message,
    )


class DocumentService:
    def __init__(self, db: Session, settings: Settings) -> None:
        self.db = db
        self.settings = settings

    def upload(
        self,
        file: UploadFile,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> DocumentMetadataResponse:
        filename = file.filename or "upload.pdf"
        if not filename.lower().endswith(ALLOWED_SUFFIX):
            raise AppException("Only PDF files are supported", status_code=415)

        raw_bytes = file.file.read()
        if not raw_bytes:
            raise AppException("Uploaded file is empty", status_code=400)

        file_hash = compute_file_hash(raw_bytes)
        existing = self.db.scalar(select(DocumentRecord).where(DocumentRecord.file_hash == file_hash))
        if existing:
            logger.warning("Rejected duplicate upload filename=%s existing_id=%s", filename, existing.id)
            raise DuplicateDocumentError(existing.id)

        document_id = str(uuid.uuid4())
        storage_path = Path(self.settings.upload_dir) / f"{document_id}.pdf"
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage_path.write_bytes(raw_bytes)

        record = DocumentRecord(
            id=document_id,
            filename=filename,
            document_type="pdf",
            status="processing",
            file_hash=file_hash,
            storage_path=str(storage_path),
            file_size_bytes=len(raw_bytes),
        )
        self.db.add(record)
        self.db.commit()

        try:
            self._run_ingestion(record, storage_path, chunk_size, chunk_overlap)
        except Exception as exc:
            record.status = "failed"
            record.error_message = str(exc)
            self.db.commit()
            logger.exception("Ingestion failed for document_id=%s", document_id)
            raise

        self.db.refresh(record)
        return document_to_response(record)

    def _run_ingestion(
        self,
        record: DocumentRecord,
        storage_path: Path,
        chunk_size: int | None,
        chunk_overlap: int | None,
    ) -> None:
        pipeline = build_ingestion_pipeline(chunk_size, chunk_overlap)
        output = pipeline.ingest(storage_path)

        for chunk in output.chunks:
            self.db.add(
                ChunkRecord(
                    id=str(uuid.uuid4()),
                    document_id=record.id,
                    vector_id=chunk.vector_id,
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    content=chunk.content,
                )
            )

        record.status = "completed"
        record.num_pages = output.num_pages
        record.num_chunks = len(output.chunks)
        record.processed_timestamp = datetime.now(timezone.utc)
        self.db.commit()
        logger.info(
            "Ingested document_id=%s pages=%d chunks=%d", record.id, output.num_pages, len(output.chunks)
        )

    def list_documents(self) -> list[DocumentMetadataResponse]:
        records = self.db.scalars(
            select(DocumentRecord).order_by(DocumentRecord.upload_timestamp.desc())
        ).all()
        return [document_to_response(r) for r in records]

    def get_document(self, document_id: str) -> DocumentRecord:
        record = self.db.get(DocumentRecord, document_id)
        if not record:
            raise NotFoundError(f"Document {document_id} not found")
        return record

    def get_chunks(self, document_id: str) -> list[ChunkRecord]:
        record = self.get_document(document_id)
        return sorted(record.chunks, key=lambda c: (c.page_number, c.chunk_index))

    def delete_document(self, document_id: str) -> None:
        record = self.get_document(document_id)
        vector_ids = [c.vector_id for c in record.chunks]
        if vector_ids:
            get_vector_store_instance().delete(vector_ids)

        storage_path = Path(record.storage_path)
        if storage_path.exists():
            storage_path.unlink()

        self.db.delete(record)  # cascades to ChunkRecord rows
        self.db.commit()
        logger.info("Deleted document_id=%s (chunks=%d)", document_id, len(vector_ids))

    def reindex_document(
        self,
        document_id: str,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> DocumentMetadataResponse:
        record = self.get_document(document_id)
        storage_path = Path(record.storage_path)
        if not storage_path.exists():
            raise AppException("Original file missing on disk, cannot re-index", status_code=410)

        old_vector_ids = [c.vector_id for c in record.chunks]
        if old_vector_ids:
            get_vector_store_instance().delete(old_vector_ids)
        for chunk in list(record.chunks):
            self.db.delete(chunk)
        record.status = "processing"
        record.error_message = None
        self.db.commit()

        try:
            self._run_ingestion(record, storage_path, chunk_size, chunk_overlap)
        except Exception as exc:
            record.status = "failed"
            record.error_message = str(exc)
            self.db.commit()
            logger.exception("Re-index failed for document_id=%s", document_id)
            raise

        self.db.refresh(record)
        return document_to_response(record)

    def get_stats(self) -> IngestionStats:
        total_documents = self.db.scalar(select(func.count()).select_from(DocumentRecord)) or 0
        total_chunks = self.db.scalar(select(func.count()).select_from(ChunkRecord)) or 0
        vector_count = get_vector_store_instance().count()
        return IngestionStats(
            total_documents=total_documents,
            total_chunks=total_chunks,
            vector_count=vector_count,
        )
