"""Conversation CRUD — separate from ChatService (which only ever reads
a bounded history window to answer a question). This is what backs
"create conversation / view previous conversations / delete
conversation" in the API and Streamlit's sidebar.

"Continue a conversation" has no dedicated endpoint: it's just passing
an existing conversation's id as `session_id` to POST /chat(/stream),
same as it's worked since Milestone 2.
"""
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.exceptions import NotFoundError
from app.database.models import ConversationRecord, UserRecord

DEFAULT_USER_EMAIL = "default-user@local"


def get_or_create_default_user(db: Session) -> UserRecord:
    """This project has no auth system yet — every conversation is
    owned by one auto-provisioned user so the schema is ready for real
    multi-user auth later without another migration.
    """
    user = db.scalar(select(UserRecord).where(UserRecord.email == DEFAULT_USER_EMAIL))
    if user:
        return user
    user = UserRecord(email=DEFAULT_USER_EMAIL)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


class ConversationService:
    def __init__(self, db: Session) -> None:
        self.db = db

    def create_conversation(self) -> ConversationRecord:
        user = get_or_create_default_user(self.db)
        conversation = ConversationRecord(user_id=user.id)
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        return conversation

    def list_conversations(self) -> list[tuple[ConversationRecord, int]]:
        """Returns (conversation, message_count) pairs, newest first."""
        conversations = self.db.scalars(
            select(ConversationRecord).order_by(ConversationRecord.updated_at.desc())
        ).all()
        return [(c, len(c.messages)) for c in conversations]

    def get_conversation(self, conversation_id: str) -> ConversationRecord:
        conversation = self.db.get(ConversationRecord, conversation_id)
        if conversation is None:
            raise NotFoundError(f"Conversation {conversation_id} not found")
        return conversation

    def delete_conversation(self, conversation_id: str) -> None:
        conversation = self.get_conversation(conversation_id)
        self.db.delete(conversation)  # cascades to messages + message_sources
        self.db.commit()
