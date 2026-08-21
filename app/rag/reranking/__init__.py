"""Reranking: re-scoring a small candidate set with a model that looks
at the (query, document) pair jointly, rather than comparing two
independently-computed vectors.

- `base.py` — BaseReranker, the provider-agnostic interface.
- `cross_encoder_reranker.py` — sentence-transformers cross-encoder
  implementation (`RERANKER_BACKEND=cross_encoder`, the default).
- `noop_reranker.py` — passthrough (`RERANKER_BACKEND=none`), preserves
  fusion's ordering unchanged — this is what makes reranking optional
  rather than load-bearing.

HybridRetrievalService depends only on BaseReranker.
"""
