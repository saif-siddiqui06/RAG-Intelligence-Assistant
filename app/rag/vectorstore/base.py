"""Vector store interface.

Deliberately minimal: integer ids in, (id, score) pairs out. All chunk
text/metadata lives in the SQL metadata store (app.database.models),
keyed by the same id — so a vector store swap never touches how
metadata is stored or queried.
"""
from abc import ABC, abstractmethod


class VectorStore(ABC):
    @abstractmethod
    def add(self, vectors: list[list[float]]) -> list[int]:
        """Add vectors, returning the ids assigned to them (input order)."""

    @abstractmethod
    def search(
        self, query_vector: list[float], top_k: int, allowed_ids: set[int] | None = None
    ) -> list[tuple[int, float]]:
        """Return up to `top_k` (id, similarity_score) pairs, best first.

        `allowed_ids`, when given, restricts the search to only those ids —
        this is how metadata filtering (a specific document, a document
        type) is implemented: the caller resolves the filter to a set of
        vector ids via SQL, then passes it here. Backends with native
        filtered search (Qdrant/pgvector) would take a real filter
        expression instead; FAISS (no native filtering) does an exact
        brute-force score over just the allowed ids.
        """

    @abstractmethod
    def delete(self, ids: list[int]) -> None:
        """Remove vectors by id."""

    @abstractmethod
    def count(self) -> int:
        """Total number of vectors currently stored."""
