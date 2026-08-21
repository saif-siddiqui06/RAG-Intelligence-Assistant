"""Hybrid retrieval: vector search + BM25, fused via RRF, reranked with
a cross-encoder. This is the Milestone 3 addition, gated behind
`Settings.retrieval_mode == "hybrid"` — the Milestone 2 vector-only
path (`RetrievalService`) is untouched and still used when
`retrieval_mode == "vector"` (the default).

    query ─┬─► vector search (VectorRetriever)  ─┐
           └─► BM25 search (BM25Retriever)       ─┴─► RRF fusion ─► rerank ─► dedup + top-k

Every stage's output is kept (not just the final one) so callers can
surface retrieval diagnostics for development/debugging, per the
milestone's requirement #9.
"""
from dataclasses import dataclass, replace

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.rag.reranking.base import BaseReranker
from app.rag.retrieval.bm25_retriever import BM25Retriever
from app.rag.retrieval.dedup import is_near_duplicate
from app.rag.retrieval.fusion import reciprocal_rank_fusion
from app.rag.retrieval.vector_retriever import VectorRetriever
from app.services.chunk_lookup import (
    RetrievedChunk,
    load_chunk_corpus,
    load_chunks_by_vector_id,
    resolve_allowed_vector_ids,
    to_retrieved_chunk,
)


@dataclass
class HybridRetrievalResult:
    vector_results: list[RetrievedChunk]
    keyword_results: list[RetrievedChunk]
    fused_results: list[RetrievedChunk]
    reranked_results: list[RetrievedChunk]
    final_chunks: list[RetrievedChunk]


class HybridRetrievalService:
    def __init__(
        self,
        db: Session,
        vector_retriever: VectorRetriever,
        reranker: BaseReranker,
        settings: Settings,
    ) -> None:
        self.db = db
        self.vector_retriever = vector_retriever
        self.reranker = reranker
        self.settings = settings

    def retrieve(
        self,
        query: str,
        document_id: str | None = None,
        document_type: str | None = None,
    ) -> HybridRetrievalResult:
        allowed_ids = resolve_allowed_vector_ids(self.db, document_id, document_type)
        if allowed_ids is not None and not allowed_ids:
            return HybridRetrievalResult([], [], [], [], [])

        vector_results = self._search_vector(query, allowed_ids)
        keyword_results = self._search_bm25(query, document_id, document_type)
        fused_results = self._fuse(vector_results, keyword_results)
        reranked_results = self._rerank(query, fused_results[: self.settings.rerank_top_k])
        final_chunks = self._select_final(reranked_results)

        return HybridRetrievalResult(
            vector_results=vector_results,
            keyword_results=keyword_results,
            fused_results=fused_results,
            reranked_results=reranked_results,
            final_chunks=final_chunks,
        )

    def _search_vector(self, query: str, allowed_ids: set[int] | None) -> list[RetrievedChunk]:
        hits = self.vector_retriever.search(query, top_k=self.settings.vector_top_k, allowed_ids=allowed_ids)
        if not hits:
            return []
        chunks_by_vector_id = load_chunks_by_vector_id(self.db, {hit.vector_id for hit in hits})
        results = []
        for hit in hits:
            chunk = chunks_by_vector_id.get(hit.vector_id)
            if chunk is None:
                continue  # stale reference (e.g. raced with a delete) — skip safely
            results.append(to_retrieved_chunk(chunk, hit.score))
        return results

    def _search_bm25(
        self, query: str, document_id: str | None, document_type: str | None
    ) -> list[RetrievedChunk]:
        corpus_rows = load_chunk_corpus(self.db, document_id, document_type)
        bm25 = BM25Retriever([(row.id, row.content) for row in corpus_rows])
        hits = bm25.search(query, top_k=self.settings.bm25_top_k)
        rows_by_id = {row.id: row for row in corpus_rows}
        results = []
        for hit in hits:
            row = rows_by_id.get(hit.chunk_id)
            if row is None:
                continue
            results.append(to_retrieved_chunk(row, hit.score))
        return results

    def _fuse(
        self, vector_results: list[RetrievedChunk], keyword_results: list[RetrievedChunk]
    ) -> list[RetrievedChunk]:
        vector_ranked_ids = [c.chunk_id for c in vector_results]
        keyword_ranked_ids = [c.chunk_id for c in keyword_results]
        fused_scores = reciprocal_rank_fusion(
            [vector_ranked_ids, keyword_ranked_ids], k=self.settings.rrf_k
        )

        by_chunk_id = {c.chunk_id: c for c in vector_results}
        for chunk in keyword_results:
            by_chunk_id.setdefault(chunk.chunk_id, chunk)

        return [replace(by_chunk_id[chunk_id], score=score) for chunk_id, score in fused_scores]

    def _rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not candidates:
            return []
        scores = self.reranker.score(query, [c.content for c in candidates])
        rescored = [replace(c, score=s) for c, s in zip(candidates, scores)]
        return sorted(rescored, key=lambda c: c.score, reverse=True)

    def _select_final(self, reranked: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Same near-duplicate-removal + top-k truncation as the
        vector-only path, applied to the reranked (already best-first)
        list instead of a raw score sort.
        """
        selected: list[RetrievedChunk] = []
        for candidate in reranked:
            if any(
                is_near_duplicate(candidate.content, kept.content, self.settings.dedup_similarity_threshold)
                for kept in selected
            ):
                continue
            selected.append(candidate)
            if len(selected) >= self.settings.final_context_k:
                break
        return selected
