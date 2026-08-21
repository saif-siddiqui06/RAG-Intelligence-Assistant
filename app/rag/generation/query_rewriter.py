"""LLM call #1: fold bounded conversation history into one standalone,
retrieval-optimized query. This is what lets the system understand a
follow-up like "What are its disadvantages?" without ever sending the
full conversation to the answer-generation call — only the rewritten,
self-contained query and the retrieved chunks go there.
"""
import logging

from app.rag.generation.base import BaseChatModel
from app.rag.generation.prompts import build_rewrite_messages

logger = logging.getLogger(__name__)


class QueryRewriter:
    def __init__(self, chat_model: BaseChatModel) -> None:
        self._chat_model = chat_model

    def rewrite(self, history: list[tuple[str, str]], question: str) -> str:
        """Returns `question` unchanged when there's no history yet —
        nothing to resolve pronouns against, and it skips an LLM call on
        every conversation's first turn.

        Degrades gracefully (falls back to the raw question) on any LLM
        failure rather than blocking the whole chat request — rewriting
        is an optimization, not a hard dependency of being able to answer.
        """
        if not history:
            return question

        messages = build_rewrite_messages(history, question)
        try:
            rewritten = self._chat_model.complete(messages, temperature=0).strip()
        except Exception:
            logger.exception("Query rewrite failed; falling back to the original question")
            return question

        return rewritten or question
