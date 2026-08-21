"""Retrieval: turning a query into scored (vector_id, score) hits.

Deliberately as thin as ingestion is — `vector_retriever.py` only knows
about `embeddings/` and `vectorstore/`, never the database. Resolving
those hits into chunk text/citations and applying SQL-side metadata
filters is a DB-aware concern, handled one layer up by
`app.services.retrieval_service`.
"""
