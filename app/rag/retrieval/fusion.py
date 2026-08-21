"""Reciprocal Rank Fusion (RRF).

Combines multiple ranked candidate lists into one using only each
item's *rank position* in each list — never raw scores. That's the
point: vector cosine-similarity and BM25's score are on completely
different, incomparable scales, so naively averaging or summing them
would be meaningless. RRF sidesteps that entirely.

Reference: Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion
Outperforms Condorcet and Individual Rank Learning Methods" (SIGIR 2009).
k=60 is the constant that paper found robust across collections, and
the value most hybrid-search implementations default to.
"""


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """ranked_lists: each a list of ids, best result first.

    Returns (id, fused_score) pairs sorted by fused_score descending.
    An id that appears in more lists, or ranks higher within a list,
    accumulates a higher score — this is what lets a chunk that both
    engines agree on outrank one only a single engine found.
    """
    scores: dict[str, float] = {}
    for ranked_ids in ranked_lists:
        for rank, item_id in enumerate(ranked_ids, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
