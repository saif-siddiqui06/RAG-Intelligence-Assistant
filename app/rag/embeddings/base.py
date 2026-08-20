"""Embedding provider interface.

Ingestion embeds documents; the future retrieval/chat pipeline will
embed queries with the same interface — both depend only on this
abstract class, never on a concrete provider.
"""
from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality this embedder produces."""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunk texts, preserving input order."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
