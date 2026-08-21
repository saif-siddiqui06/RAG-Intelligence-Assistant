"""Passthrough reranker — used when RERANKER_BACKEND=none.

Assigns monotonically decreasing scores that preserve the incoming
(fusion) order exactly, so sorting by score afterward is a no-op. This
is what makes reranking an optional stage rather than a load-bearing
one: turning it off should change nothing else about the pipeline.
"""
from app.rag.reranking.base import BaseReranker


class NoOpReranker(BaseReranker):
    def score(self, query: str, documents: list[str]) -> list[float]:
        n = len(documents)
        return [1.0 - i / n for i in range(n)] if n else []
