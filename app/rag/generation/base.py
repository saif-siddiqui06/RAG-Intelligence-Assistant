"""Chat-model provider interface.

QueryRewriter and AnswerGenerator depend only on this — never on a
specific provider's SDK shape. That's what makes swapping the LLM
provider (as happened going from OpenAI to Gemini) a matter of adding
one new class here, not touching the generation logic at all.
"""
from abc import ABC, abstractmethod
from collections.abc import Iterator


class BaseChatModel(ABC):
    @abstractmethod
    def complete(self, messages: list[dict], temperature: float = 0) -> str:
        """Return the full response text for a system+user message list."""

    @abstractmethod
    def stream(self, messages: list[dict], temperature: float = 0) -> Iterator[str]:
        """Yield response text deltas as they're generated."""
