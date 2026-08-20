"""Vector storage, behind a provider-agnostic interface.

FAISS is today's implementation; the interface in base.py is the only
thing ingestion (and later, retrieval) code depends on, so switching to
Chroma/Qdrant/pgvector later means adding one new class and changing
`get_vector_store`'s factory branch — nothing else in the app changes.
"""
