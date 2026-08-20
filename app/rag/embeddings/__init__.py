"""Embedding generation, behind a provider-agnostic interface.

Only app.rag.embeddings.base.BaseEmbedder is meant to be depended on
outside this package, so swapping OpenAI for another provider later
never touches ingestion or (future) retrieval code.
"""
