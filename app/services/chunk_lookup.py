"""Shared SQL helpers + the RetrievedChunk data model.

Used by both RetrievalService (vector-only) and HybridRetrievalService
(vector + BM25 + rerank) so "resolve a metadata filter" and "turn a
chunk row into a citable, scored chunk" each have one implementation,
not two copies that could drift.
"""
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.models import ChunkRecord, DocumentRecord


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_index: int
    content: str
    score: float


def resolve_allowed_vector_ids(
    db: Session, document_id: str | None, document_type: str | None
) -> set[int] | None:
    """None means "no filter" (search everything). An empty set means
    the filter matched nothing.
    """
    if not document_id and not document_type:
        return None

    query = select(ChunkRecord.vector_id)
    if document_id:
        query = query.where(ChunkRecord.document_id == document_id)
    if document_type:
        query = query.join(DocumentRecord).where(DocumentRecord.document_type == document_type)
    return set(db.scalars(query).all())


def load_chunks_by_vector_id(db: Session, vector_ids: set[int]) -> dict[int, ChunkRecord]:
    if not vector_ids:
        return {}
    rows = db.scalars(select(ChunkRecord).where(ChunkRecord.vector_id.in_(vector_ids))).all()
    return {row.vector_id: row for row in rows}


def load_chunk_corpus(
    db: Session, document_id: str | None, document_type: str | None
) -> list[ChunkRecord]:
    """All chunk rows matching the optional filter — this is BM25's
    corpus. Rebuilt from SQL on every hybrid search since rank_bm25 has
    no incremental index API; fine at portfolio scale, a documented
    scaling limit (same honesty as FAISS's exact flat-index tradeoff)
    for a corpus large enough that this matters.
    """
    query = select(ChunkRecord)
    if document_id:
        query = query.where(ChunkRecord.document_id == document_id)
    if document_type:
        query = query.join(DocumentRecord).where(DocumentRecord.document_type == document_type)
    return list(db.scalars(query).all())


def to_retrieved_chunk(chunk: ChunkRecord, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk.id,
        document_id=chunk.document_id,
        filename=chunk.document.filename,
        page_number=chunk.page_number,
        chunk_index=chunk.chunk_index,
        content=chunk.content,
        score=score,
    )
