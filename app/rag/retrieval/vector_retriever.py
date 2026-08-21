"""Pure semantic vector retrieval — no database, no HTTP.

Hybrid (vector + BM25) search and reranking are explicitly out of scope
for this milestone; this is vector-only, exact (FAISS's flat index),
optionally restricted to a caller-resolved set of ids for metadata
filtering.
"""
from dataclasses import dataclass

from app.rag.embeddings.base import BaseEmbedder
from app.rag.vectorstore.base import VectorStore


@dataclass
class VectorHit:
    vector_id: int
    score: float


class VectorRetriever:
    def __init__(self, embedder: BaseEmbedder, vector_store: VectorStore) -> None:
        self.embedder = embedder
        self.vector_store = vector_store

    def search(self, query: str, top_k: int, allowed_ids: set[int] | None = None) -> list[VectorHit]:
        if allowed_ids is not None and not allowed_ids:
            return []  # filter resolved to nothing — skip embedding the query for nothing
        if self.vector_store.count() == 0:
            return []  # nothing ingested yet — skip embedding the query for nothing
        query_vector = self.embedder.embed_query(query)
        hits = self.vector_store.search(query_vector, top_k=top_k, allowed_ids=allowed_ids)
        return [VectorHit(vector_id=vid, score=score) for vid, score in hits]
