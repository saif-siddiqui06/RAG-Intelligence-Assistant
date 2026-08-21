"""Evaluation harness: retrieval quality, context relevance, answer
correctness and faithfulness/hallucination checks.

- `dataset.py` — a small, hand-built retrieval eval set (14 passages,
  22 questions with known-relevant passage ids).
- `metrics.py` — Recall@k, MRR (pure functions over ranked id lists).
- `retrieval_benchmark.py` — compares vector-only / BM25-only / hybrid
  (fusion) / hybrid+reranking on that dataset. Run with
  `python -m app.evaluation.retrieval_benchmark`.

Answer correctness and faithfulness/hallucination checks (evaluating
*generation*, not just retrieval) are not built yet — that's the rest
of this milestone, still to come.
"""
