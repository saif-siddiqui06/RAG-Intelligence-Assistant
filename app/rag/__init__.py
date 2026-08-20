"""RAG pipeline.

- `ingestion/` — PDF extraction, cleaning, chunking, file hashing. Implemented.
- `embeddings/` — provider-agnostic embedding interface (OpenAI today). Implemented.
- `vectorstore/` — provider-agnostic vector store interface (FAISS today). Implemented.
- `dependencies.py` — cached singletons wiring the above into an IngestionPipeline.

Hybrid retrieval, reranking, query rewriting and citations (the
chat/answer side of RAG) are separate, not-yet-built modules that will
depend on `embeddings/` and `vectorstore/` too, but never on `ingestion/`
directly — ingestion and chat stay independent.
"""
