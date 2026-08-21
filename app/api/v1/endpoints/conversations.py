"""Conversation management endpoints.

"Continue a conversation" has no endpoint of its own — pass its id as
`session_id` to POST /chat or /chat/stream.
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.models.conversation import (
    ConversationDetail,
    ConversationSummary,
    DeleteConversationResponse,
    MessageOut,
    MessageSourceOut,
)
from app.services.conversation_service import ConversationService

router = APIRouter(prefix="/conversations", tags=["conversations"])


@router.post("", response_model=ConversationSummary, status_code=201)
def create_conversation(db: Session = Depends(get_db)) -> ConversationSummary:
    conversation = ConversationService(db).create_conversation()
    return _to_summary(conversation, message_count=0)


@router.get("", response_model=list[ConversationSummary])
def list_conversations(db: Session = Depends(get_db)) -> list[ConversationSummary]:
    return [_to_summary(c, count) for c, count in ConversationService(db).list_conversations()]


@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str, db: Session = Depends(get_db)) -> ConversationDetail:
    conversation = ConversationService(db).get_conversation(conversation_id)
    return ConversationDetail(
        conversation_id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[
            MessageOut(
                message_id=m.id,
                role=m.role,
                content=m.content,
                created_at=m.created_at,
                sources=[
                    MessageSourceOut(
                        index=s.index,
                        document_name=s.document_name,
                        page_number=s.page_number,
                        chunk_id=s.chunk_id,
                    )
                    for s in m.sources
                ],
            )
            for m in conversation.messages
        ],
    )


@router.delete("/{conversation_id}", response_model=DeleteConversationResponse)
def delete_conversation(conversation_id: str, db: Session = Depends(get_db)) -> DeleteConversationResponse:
    ConversationService(db).delete_conversation(conversation_id)
    return DeleteConversationResponse(
        conversation_id=conversation_id, deleted=True, message="Conversation deleted"
    )


def _to_summary(conversation, message_count: int) -> ConversationSummary:
    return ConversationSummary(
        conversation_id=conversation.id,
        title=conversation.title,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        message_count=message_count,
    )
