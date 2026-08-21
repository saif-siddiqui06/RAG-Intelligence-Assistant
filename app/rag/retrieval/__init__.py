"""Retrieval: turning a query into scored candidate chunks.

- `vector_retriever.py` — semantic search over `embeddings/` + `vectorstore/`.
- `bm25_retriever.py` — keyword search (rank_bm25), built fresh from a
  caller-supplied corpus each time (no incremental index — see its
  docstring).
- `fusion.py` — Reciprocal Rank Fusion, combining the two engines'
  ranked lists without needing comparable score scales.
- `dedup.py` — near-duplicate chunk-text detection, shared by both the
  vector-only and hybrid retrieval services.

All of the above are DB-agnostic — they take/return plain ids and
scores, never a database session. Resolving hits into chunk text/
citations and applying SQL-side metadata filters is a DB-aware
concern, handled one layer up by `app.services.retrieval_service`
(vector-only) and `app.services.hybrid_retrieval_service` (hybrid).
"""
