"""Tests for Milestone 6 production-hardening features: request-id
propagation, rate limiting, and upload file validation.
"""
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.core.rate_limit import limiter
from app.main import app
from app.services.document_service import DocumentService


@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture()
def dev_settings(tmp_path):
    return Settings(
        upload_dir=tmp_path / "uploads",
        processed_dir=tmp_path / "processed",
        vectorstore_dir=tmp_path / "vectorstore",
        log_dir=tmp_path / "logs",
    )


def test_response_carries_a_request_id(client):
    response = client.get("/api/v1/health")
    assert "x-request-id" in response.headers
    assert len(response.headers["x-request-id"]) > 0


def test_request_id_is_echoed_back_when_supplied(client):
    response = client.get("/api/v1/health", headers={"X-Request-ID": "test-fixed-id"})
    assert response.headers["x-request-id"] == "test-fixed-id"


def test_upload_rejects_non_pdf(client):
    limiter.reset()
    response = client.post(
        "/api/v1/documents/upload",
        files={"files": ("notes.txt", b"hello world", "text/plain")},
    )
    assert response.status_code == 415


def test_upload_rejects_oversized_file(client, monkeypatch):
    limiter.reset()
    monkeypatch.setattr(get_settings(), "max_upload_size_mb", 1)
    oversized = b"%PDF-1.4\n" + b"0" * (2 * 1024 * 1024)
    response = client.post(
        "/api/v1/documents/upload",
        files={"files": ("big.pdf", oversized, "application/pdf")},
    )
    assert response.status_code == 413


def test_upload_endpoint_rate_limits_after_the_configured_threshold(client):
    limiter.reset()
    statuses = [
        client.post(
            "/api/v1/documents/upload",
            files={"files": ("notes.txt", b"hello world", "text/plain")},
        ).status_code
        for _ in range(15)
    ]
    assert 429 in statuses


def test_document_service_rejects_file_over_the_configured_size_limit(dev_settings, db_session):
    from io import BytesIO

    from fastapi import UploadFile

    from app.core.exceptions import AppException

    dev_settings.max_upload_size_mb = 1
    service = DocumentService(db_session, dev_settings)
    oversized = UploadFile(filename="big.pdf", file=BytesIO(b"0" * (2 * 1024 * 1024)))

    with pytest.raises(AppException) as exc_info:
        service.upload(oversized)
    assert exc_info.value.status_code == 413
