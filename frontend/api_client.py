"""Thin HTTP client the Streamlit UI uses to talk to the FastAPI backend.

Kept separate from streamlit_app.py so UI code never builds URLs or
calls `requests` directly — that stays here, one place to change if the
backend's base URL or routes move.
"""
import os

import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
_DOCUMENTS_URL = f"{API_BASE_URL}/api/v1/documents"


def get_health(timeout: float = 3.0) -> dict:
    """Call the backend health endpoint. Raises on network/HTTP failure."""
    response = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=timeout)
    response.raise_for_status()
    return response.json()


def upload_documents(files, chunk_size: int | None = None, chunk_overlap: int | None = None) -> list[dict]:
    """`files` is a list of Streamlit UploadedFile objects."""
    multipart = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
    params = {k: v for k, v in {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap}.items() if v}
    response = requests.post(f"{_DOCUMENTS_URL}/upload", files=multipart, params=params, timeout=180)
    response.raise_for_status()
    return response.json()


def list_documents() -> dict:
    response = requests.get(_DOCUMENTS_URL, timeout=10)
    response.raise_for_status()
    return response.json()


def get_document_chunks(document_id: str) -> list[dict]:
    response = requests.get(f"{_DOCUMENTS_URL}/{document_id}/chunks", timeout=10)
    response.raise_for_status()
    return response.json()


def delete_document(document_id: str) -> dict:
    response = requests.delete(f"{_DOCUMENTS_URL}/{document_id}", timeout=30)
    response.raise_for_status()
    return response.json()


def reindex_document(document_id: str) -> dict:
    response = requests.post(f"{_DOCUMENTS_URL}/{document_id}/reindex", timeout=180)
    response.raise_for_status()
    return response.json()


def get_stats() -> dict:
    response = requests.get(f"{_DOCUMENTS_URL}/stats/summary", timeout=10)
    response.raise_for_status()
    return response.json()


def api_error_detail(exc: requests.exceptions.HTTPError) -> str:
    if exc.response is None:
        return str(exc)
    try:
        return exc.response.json().get("detail", str(exc))
    except ValueError:
        return str(exc)
