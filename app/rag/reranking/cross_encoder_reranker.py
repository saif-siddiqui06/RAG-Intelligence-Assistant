"""Cross-encoder reranker (sentence-transformers).

A cross-encoder scores a (query, document) pair jointly through one
model pass, instead of comparing two independently-computed embedding
vectors (a "bi-encoder", which is what vector search does). That joint
attention is why cross-encoders are consistently more accurate at
judging relevance — and why they're too slow to run over a whole
corpus, only ever over a short reranking shortlist.

Runs fully locally — no API key, no network call — using the free,
widely-used MS MARCO MiniLM cross-encoder by default.
"""
import logging
import math

from app.rag.reranking.base import BaseReranker

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model_name)

    def score(self, query: str, documents: list[str]) -> list[float]:
        if not documents:
            return []
        pairs = [(query, doc) for doc in documents]
        raw_scores = self._model.predict(pairs)
        # ms-marco-MiniLM's raw output is an unbounded logit, not a
        # probability — squash it so scores stay comparable to the
        # ~0-1 relevance scores the rest of the app already works with
        # (cosine similarity, BM25-derived confidence heuristics).
        return [_sigmoid(float(s)) for s in raw_scores]


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
