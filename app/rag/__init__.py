"""RAG pipeline.

- `ingestion/` — PDF extraction, cleaning, chunking, file hashing.
- `embeddings/` — provider-agnostic embedding interface (Gemini today).
- `vectorstore/` — provider-agnostic vector store interface (FAISS today).
- `retrieval/` — pure semantic vector retrieval (embed query -> search).
- `generation/` — provider-agnostic chat-model interface (Gemini today),
  query rewriting, and grounded/cited answer generation.
- `dependencies.py` — cached singletons wiring all of the above together.

Ingestion stays independent of the chat/answer side by design: it never
imports anything from `retrieval/` or `generation/`, only `embeddings/`
and `vectorstore/`. Hybrid (BM25) search and reranking are not built yet.
"""
