"""Retrieval service: the DB-aware half of vector-only retrieval.

Ties app.rag.retrieval.vector_retriever (pure vector search) together
with SQL: resolving document/document_type filters into a set of
allowed vector ids, joining hits back to chunk content + citation
metadata, and dropping near-duplicate chunks before the caller ever
sees them. Kept as its own service (per the "retrieval, prompting and
generation as separate services" requirement) — app.services.chat_service
is the only thing that calls it.

This is the unchanged Milestone 2 path (`RETRIEVAL_MODE=vector`).
`HybridRetrievalService` is the Milestone 3 addition
(`RETRIEVAL_MODE=hybrid`) — it reuses the SQL helpers below via
app.services.chunk_lookup but does not touch this class.
"""
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.rag.retrieval.dedup import is_near_duplicate
from app.rag.retrieval.vector_retriever import VectorRetriever
from app.services.chunk_lookup import (
    RetrievedChunk,
    load_chunks_by_vector_id,
    resolve_allowed_vector_ids,
    to_retrieved_chunk,
)


class RetrievalService:
    def __init__(self, db: Session, retriever: VectorRetriever, settings: Settings) -> None:
        self.db = db
        self.retriever = retriever
        self.settings = settings

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        document_id: str | None = None,
        document_type: str | None = None,
    ) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.retrieval_top_k

        allowed_ids = resolve_allowed_vector_ids(self.db, document_id, document_type)
        if allowed_ids is not None and not allowed_ids:
            return []  # filter matched no chunks at all

        overfetch = top_k * self.settings.retrieval_overfetch_multiplier
        hits = self.retriever.search(query, top_k=overfetch, allowed_ids=allowed_ids)
        if not hits:
            return []

        chunks_by_vector_id = load_chunks_by_vector_id(self.db, {hit.vector_id for hit in hits})

        candidates: list[RetrievedChunk] = []
        for hit in hits:
            chunk = chunks_by_vector_id.get(hit.vector_id)
            if chunk is None:
                continue  # vector exists but metadata missing (e.g. raced with a delete) — skip safely
            candidates.append(to_retrieved_chunk(chunk, hit.score))

        return self._select_context(candidates, top_k)

    def _select_context(
        self, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Keep the top_k highest-scoring candidates, skipping any whose
        text is a near-duplicate of one already kept (e.g. the same
        passage chunked twice, or two documents sharing boilerplate).
        """
        selected: list[RetrievedChunk] = []
        for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
            if any(
                is_near_duplicate(candidate.content, kept.content, self.settings.dedup_similarity_threshold)
                for kept in selected
            ):
                continue
            selected.append(candidate)
            if len(selected) >= top_k:
                break
        return selected
