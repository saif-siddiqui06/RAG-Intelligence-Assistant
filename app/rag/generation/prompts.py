"""Every prompt template sent to the LLM, in one place, so what the
model actually sees is easy to audit and change without hunting through
orchestration code.
"""
from typing import Protocol

NO_CONTEXT_SENTINEL = "I cannot determine this from the uploaded documents."

_REWRITE_SYSTEM_PROMPT = """\
You are a query rewriting assistant for a document search system. Given \
a short conversation history and a new user question, rewrite the new \
question into a single, standalone, retrieval-optimized search query.

Rules:
- Resolve pronouns and implicit references (e.g. "it", "its", "that", \
"the former") using the conversation history.
- Preserve the user's intent exactly. Do not add new topics, facts, or \
assumptions that were not implied by the conversation.
- Output ONLY the rewritten query. No explanation, no quotes, no prefix.
- If the question is already standalone, return it unchanged.
"""

_ANSWER_SYSTEM_PROMPT = f"""\
You are a research assistant that answers questions using ONLY the \
provided document excerpts ("Sources"). Follow these rules exactly:

1. Base your answer strictly on the given sources — never use outside \
knowledge, even if you know the answer.
2. Cite every factual claim with the matching source number in square \
brackets right after the sentence it supports, e.g. "SMOTE oversamples \
the minority class [1]." Use multiple markers like [1][2] when a claim \
draws on more than one source.
3. If the sources do not contain enough information to answer the \
question, respond with EXACTLY this sentence and nothing else: \
"{NO_CONTEXT_SENTINEL}"
4. Never fabricate a source, page number, or citation that is not in \
the provided list.
5. Be concise and answer the question directly.
"""


class CitableChunk(Protocol):
    """Structural type — any object with these three attributes can be
    cited, whatever module actually constructed it. Avoids this module
    importing from app.services (rag/ never depends on services/).
    """

    filename: str
    page_number: int
    content: str


def build_rewrite_messages(history: list[tuple[str, str]], question: str) -> list[dict]:
    transcript = "\n".join(f"{role}: {content}" for role, content in history)
    user_message = (
        f"Conversation so far:\n{transcript}\n\nNew question: {question}"
        if history
        else f"New question: {question}"
    )
    return [
        {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


def build_answer_messages(query: str, chunks: list[CitableChunk]) -> list[dict]:
    sources_block = "\n\n".join(
        f'[{i}] {chunk.filename} — Page {chunk.page_number}\n"{chunk.content}"'
        for i, chunk in enumerate(chunks, start=1)
    )
    user_message = (
        f"Question: {query}\n\nSources:\n{sources_block}\n\n"
        "Answer the question using only the sources above, following the citation rules."
    )
    return [
        {"role": "system", "content": _ANSWER_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
