"""Retrieval service: the DB-aware half of retrieval.

Ties app.rag.retrieval.vector_retriever (pure vector search) together
with SQL: resolving document/document_type filters into a set of
allowed vector ids, joining hits back to chunk content + citation
metadata, and dropping near-duplicate chunks before the caller ever
sees them. Kept as its own service (per the "retrieval, prompting and
generation as separate services" requirement) — app.services.chat_service
is the only thing that calls it.
"""
from dataclasses import dataclass
from difflib import SequenceMatcher

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.database.models import ChunkRecord, DocumentRecord
from app.rag.retrieval.vector_retriever import VectorRetriever


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    content: str
    score: float


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

        allowed_ids = self._resolve_allowed_ids(document_id, document_type)
        if allowed_ids is not None and not allowed_ids:
            return []  # filter matched no chunks at all

        overfetch = top_k * self.settings.retrieval_overfetch_multiplier
        hits = self.retriever.search(query, top_k=overfetch, allowed_ids=allowed_ids)
        if not hits:
            return []

        chunks_by_vector_id = self._load_chunks({hit.vector_id for hit in hits})

        candidates: list[RetrievedChunk] = []
        for hit in hits:
            chunk = chunks_by_vector_id.get(hit.vector_id)
            if chunk is None:
                continue  # vector exists but metadata missing (e.g. raced with a delete) — skip safely
            candidates.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    filename=chunk.document.filename,
                    page_number=chunk.page_number,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    score=hit.score,
                )
            )

        return self._select_context(candidates, top_k)

    def _resolve_allowed_ids(
        self, document_id: str | None, document_type: str | None
    ) -> set[int] | None:
        """None means "no filter" (search all documents). An empty set
        means the filter matched nothing.
        """
        if not document_id and not document_type:
            return None

        query = select(ChunkRecord.vector_id)
        if document_id:
            query = query.where(ChunkRecord.document_id == document_id)
        if document_type:
            query = query.join(DocumentRecord).where(DocumentRecord.document_type == document_type)
        return set(self.db.scalars(query).all())

    def _load_chunks(self, vector_ids: set[int]) -> dict[int, ChunkRecord]:
        rows = self.db.scalars(
            select(ChunkRecord).where(ChunkRecord.vector_id.in_(vector_ids))
        ).all()
        return {row.vector_id: row for row in rows}

    def _select_context(
        self, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        """Keep the top_k highest-scoring candidates, skipping any whose
        text is a near-duplicate of one already kept (e.g. the same
        passage chunked twice, or two documents sharing boilerplate).
        """
        selected: list[RetrievedChunk] = []
        for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
            if any(self._is_near_duplicate(candidate.content, kept.content) for kept in selected):
                continue
            selected.append(candidate)
            if len(selected) >= top_k:
                break
        return selected

    def _is_near_duplicate(self, a: str, b: str) -> bool:
        return SequenceMatcher(None, a, b).ratio() >= self.settings.dedup_similarity_threshold
