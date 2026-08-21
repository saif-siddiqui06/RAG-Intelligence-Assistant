"""Conversational RAG endpoints.

`POST /chat` returns the full structured response in one shot.
`POST /chat/stream` streams the answer text as it's generated, followed
by one final chunk carrying the structured metadata (sources, retrieved
chunks, confidence, rewritten query, session id) — see
app.services.chat_service.STREAM_META_DELIMITER for the exact protocol.
"""
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from app.api.deps import get_app_settings, get_db
from app.core.config import Settings
from app.models.chat import ChatRequest, ChatResponse
from app.services.chat_service import ChatService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def ask(
    request: ChatRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
) -> ChatResponse:
    return ChatService(db, settings).ask(request)


@router.post("/stream")
def ask_stream(
    request: ChatRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
) -> StreamingResponse:
    service = ChatService(db, settings)
    # Query rewriting + retrieval (embeds the query — a real failure
    # point) run here, synchronously, so a failure raises a normal
    # exception the registered handlers turn into a clean JSON error
    # *before* the streaming response starts.
    prepared = service.prepare(request)
    return StreamingResponse(
        service.stream_answer(prepared), media_type="text/plain; charset=utf-8"
    )
