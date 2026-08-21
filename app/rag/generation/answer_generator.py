"""LLM call #2: rewritten query + retrieved chunks -> a grounded,
inline-cited answer. Supports both a plain call and a streaming one so
the same prompt/model powers both the JSON chat endpoint and the
streaming one Streamlit consumes.

Citations are extracted from the answer text itself (`[1]`, `[2]`...)
rather than requested as structured/JSON output — that's what makes
token-by-token streaming to the UI possible: the client can render the
raw text as it arrives and only needs to parse citation markers once,
after the stream ends.
"""
import re
from collections.abc import Iterator

from app.rag.generation.base import BaseChatModel
from app.rag.generation.prompts import NO_CONTEXT_SENTINEL, CitableChunk, build_answer_messages

_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


class AnswerGenerator:
    def __init__(self, chat_model: BaseChatModel) -> None:
        self._chat_model = chat_model

    def generate(self, query: str, chunks: list[CitableChunk]) -> str:
        """Non-streaming: returns the full answer text."""
        messages = build_answer_messages(query, chunks)
        return self._chat_model.complete(messages, temperature=0).strip()

    def generate_stream(self, query: str, chunks: list[CitableChunk]) -> Iterator[str]:
        """Streaming: yields text deltas as the model produces them."""
        messages = build_answer_messages(query, chunks)
        yield from self._chat_model.stream(messages, temperature=0)


def is_no_context_answer(answer_text: str) -> bool:
    return answer_text.strip().lower().startswith(NO_CONTEXT_SENTINEL.lower())


def extract_cited_indices(answer_text: str, max_index: int) -> list[int]:
    """1-indexed source numbers actually cited in the answer, ascending,
    deduplicated, and clamped to the range of sources that were provided
    (guards against the model inventing an out-of-range citation).
    """
    found = {int(match) for match in _CITATION_PATTERN.findall(answer_text)}
    return sorted(i for i in found if 1 <= i <= max_index)
