"""Retrieval quality metrics — pure functions over ranked id lists, so
they're usable both by the benchmark script and by unit tests without
touching embeddings, BM25, or any database.
"""


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """1.0 if any relevant id appears in the top-k retrieved ids, else 0.0."""
    top_k = set(retrieved_ids[:k])
    return 1.0 if top_k & set(relevant_ids) else 0.0


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """1 / rank of the first relevant id found (1-indexed); 0.0 if none
    of the relevant ids appear anywhere in `retrieved_ids`.
    """
    relevant = set(relevant_ids)
    for rank, item_id in enumerate(retrieved_ids, start=1):
        if item_id in relevant:
            return 1.0 / rank
    return 0.0


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
