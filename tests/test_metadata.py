"""Unit tests for document/chunk metadata creation."""
import uuid
from datetime import datetime, timezone

from app.database.models import ChunkRecord, DocumentRecord
from app.services.document_service import document_to_response


def test_document_record_defaults_apply_on_flush(db_session):
    record = DocumentRecord(
        filename="paper.pdf",
        file_hash="a" * 64,
        storage_path="/tmp/paper.pdf",
        file_size_bytes=1234,
    )

    db_session.add(record)
    db_session.flush()

    assert record.document_type == "pdf"
    assert record.status == "pending"
    assert uuid.UUID(record.id)  # a valid uuid4 was generated
    assert record.upload_timestamp is not None


def test_chunk_record_preserves_page_and_chunk_position():
    chunk = ChunkRecord(
        document_id=str(uuid.uuid4()),
        vector_id=7,
        chunk_index=2,
        page_number=5,
        content="some extracted, cleaned chunk text",
    )

    assert chunk.page_number == 5
    assert chunk.chunk_index == 2
    assert chunk.vector_id == 7
    assert "extracted" in chunk.content


def test_document_to_response_maps_id_to_document_id():
    record = DocumentRecord(
        id=str(uuid.uuid4()),
        filename="paper.pdf",
        document_type="pdf",
        file_hash="b" * 64,
        storage_path="/tmp/paper.pdf",
        file_size_bytes=42,
        status="completed",
        num_pages=3,
        num_chunks=10,
        upload_timestamp=datetime.now(timezone.utc),
    )

    response = document_to_response(record)

    assert response.document_id == record.id
    assert response.filename == "paper.pdf"
    assert response.status.value == "completed"
    assert response.num_chunks == 10
    assert response.document_type.value == "pdf"


def test_upload_timestamp_and_hash_survive_a_round_trip(db_session):
    record = DocumentRecord(
        filename="report.pdf",
        file_hash="c" * 64,
        storage_path="/tmp/report.pdf",
        file_size_bytes=99,
    )
    db_session.add(record)
    db_session.commit()

    fetched = db_session.get(DocumentRecord, record.id)

    assert fetched is not None
    assert fetched.file_hash == "c" * 64
    assert fetched.upload_timestamp is not None
