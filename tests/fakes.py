"""Shared test doubles — network-free stand-ins for provider-backed
components, used across the retrieval/chat test suite so no test needs
a real API key or network access.
"""
from collections.abc import Iterator

from app.rag.embeddings.base import BaseEmbedder
from app.rag.generation.base import BaseChatModel

_VOCAB = [
    "smote",
    "oversample",
    "minority",
    "disadvantage",
    "noise",
    "overfit",
    "synthetic",
    "imbalance",
    "django",
    "web",
    "framework",
    "python",
]


class KeywordFakeEmbedder(BaseEmbedder):
    """Deterministic, network-free embedder: cosine similarity between
    two texts is proportional to how many keywords they share from a
    small fixed vocabulary. Enough to test ranking/filtering without a
    real embedding model.
    """

    dimension_size = len(_VOCAB)

    @property
    def dimension(self) -> int:
        return self.dimension_size

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        lowered = text.lower()
        raw = [1.0 if word in lowered else 0.0 for word in _VOCAB]
        norm = sum(v * v for v in raw) ** 0.5
        if norm == 0:
            raw[0] = 1e-6  # avoid an all-zero row going into FAISS's normalize_L2
            norm = raw[0]
        return [v / norm for v in raw]


class FakeChatModel(BaseChatModel):
    """Stands in for a real BaseChatModel implementation — returns
    scripted responses in call order instead of reaching the network.
    """

    def __init__(self, responses: list[str] | None = None, stream_chunks: list[str] | None = None):
        self._responses = list(responses or [])
        self._stream_chunks = stream_chunks or []
        self.calls: list[list[dict]] = []

    def complete(self, messages: list[dict], temperature: float = 0) -> str:
        self.calls.append(messages)
        return self._responses.pop(0) if self._responses else ""

    def stream(self, messages: list[dict], temperature: float = 0) -> Iterator[str]:
        self.calls.append(messages)
        yield from self._stream_chunks
