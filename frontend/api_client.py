"""Thin HTTP client the Streamlit UI uses to talk to the FastAPI backend.

Kept separate from streamlit_app.py so UI code never builds URLs or
calls `requests` directly — that stays here, one place to change if the
backend's base URL or routes move.
"""
import json
import os

import requests

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
_DOCUMENTS_URL = f"{API_BASE_URL}/api/v1/documents"
_CHAT_URL = f"{API_BASE_URL}/api/v1/chat"

# Must match app.services.chat_service.STREAM_META_DELIMITER exactly —
# frontend and backend are separate deployables, so this can't be a
# shared import, only a matching literal.
_STREAM_META_DELIMITER = "\n<<<META>>>\n"


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


def _chat_payload(question, session_id, document_id, document_type, top_k) -> dict:
    payload: dict = {"question": question}
    if session_id:
        payload["session_id"] = session_id
    if document_id:
        payload["document_id"] = document_id
    if document_type:
        payload["document_type"] = document_type
    if top_k:
        payload["top_k"] = top_k
    return payload


def ask_chat(
    question: str,
    session_id: str | None = None,
    document_id: str | None = None,
    document_type: str | None = None,
    top_k: int | None = None,
) -> dict:
    """Non-streaming: returns the full ChatResponse in one call."""
    payload = _chat_payload(question, session_id, document_id, document_type, top_k)
    response = requests.post(_CHAT_URL, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def ask_chat_stream(
    question: str,
    session_id: str | None = None,
    document_id: str | None = None,
    document_type: str | None = None,
    top_k: int | None = None,
):
    """Returns (generator, result). Feed the generator to
    `st.write_stream` — it yields only clean answer text. Once fully
    consumed, `result["meta"]` holds the parsed ChatStreamMeta dict (or
    `result["error"]` holds a message if the request failed).
    """
    payload = _chat_payload(question, session_id, document_id, document_type, top_k)
    result: dict = {"meta": None, "error": None}

    def _generate():
        buffer = ""
        meta_chunks: list[str] = []
        found_delimiter = False
        try:
            with requests.post(
                f"{_CHAT_URL}/stream", json=payload, stream=True, timeout=180
            ) as response:
                response.raise_for_status()
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if not chunk:
                        continue
                    if found_delimiter:
                        meta_chunks.append(chunk)
                        continue
                    buffer += chunk
                    if _STREAM_META_DELIMITER in buffer:
                        text_part, _, rest = buffer.partition(_STREAM_META_DELIMITER)
                        if text_part:
                            yield text_part
                        meta_chunks.append(rest)
                        found_delimiter = True
                        buffer = ""
                    else:
                        # Hold back a delimiter-length tail in case the
                        # delimiter is split across two chunk reads.
                        safe_len = max(0, len(buffer) - len(_STREAM_META_DELIMITER))
                        if safe_len:
                            yield buffer[:safe_len]
                            buffer = buffer[safe_len:]
        except requests.exceptions.RequestException as exc:
            result["error"] = api_error_detail(exc) if isinstance(
                exc, requests.exceptions.HTTPError
            ) else str(exc)
            return

        if not found_delimiter:
            if buffer:
                yield buffer
            result["error"] = "Stream ended without response metadata"
            return

        try:
            result["meta"] = json.loads("".join(meta_chunks))
        except ValueError:
            result["error"] = "Could not parse response metadata"

    return _generate(), result


def api_error_detail(exc: requests.exceptions.HTTPError) -> str:
    if exc.response is None:
        return str(exc)
    try:
        return exc.response.json().get("detail", str(exc))
    except ValueError:
        return str(exc)
