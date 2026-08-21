"""Reranker provider interface.

HybridRetrievalService depends only on this — never on a specific
reranking library — matching the same swappability pattern already
used for embeddings, the vector store, and the chat model.
"""
from abc import ABC, abstractmethod


class BaseReranker(ABC):
    @abstractmethod
    def score(self, query: str, documents: list[str]) -> list[float]:
        """Return one relevance score per document, in the same order
        as `documents`. Higher means more relevant. The caller sorts —
        this only scores.
        """
