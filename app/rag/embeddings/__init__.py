"""Embedding generation, behind a provider-agnostic interface.

Only app.rag.embeddings.base.BaseEmbedder is meant to be depended on
outside this package, so swapping providers (OpenAI -> Gemini already
happened once; anything else later) never touches ingestion or
retrieval code.
"""
