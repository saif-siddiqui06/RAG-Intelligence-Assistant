"""Tests for Reciprocal Rank Fusion."""
from app.rag.retrieval.fusion import reciprocal_rank_fusion


def test_item_in_both_lists_ranks_above_item_in_only_one():
    vector_ranked = ["a", "b", "c"]
    keyword_ranked = ["b", "d", "e"]

    fused = reciprocal_rank_fusion([vector_ranked, keyword_ranked], k=60)
    fused_ids = [item_id for item_id, _ in fused]

    assert fused_ids[0] == "b"  # found by both engines


def test_higher_rank_in_a_single_list_scores_higher():
    fused = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
    scores = dict(fused)

    assert scores["a"] > scores["b"] > scores["c"]


def test_empty_lists_produce_empty_fusion():
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_single_list_preserves_relative_order():
    fused = reciprocal_rank_fusion([["x", "y", "z"]], k=60)

    assert [item_id for item_id, _ in fused] == ["x", "y", "z"]


def test_disjoint_lists_still_produce_a_full_ranking():
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "d"]], k=60)

    assert {item_id for item_id, _ in fused} == {"a", "b", "c", "d"}


def test_smaller_k_increases_the_influence_of_top_ranks():
    # With a smaller k, the gap between rank 1 and rank 2 is proportionally larger.
    fused_small_k = dict(reciprocal_rank_fusion([["a", "b"]], k=1))
    fused_large_k = dict(reciprocal_rank_fusion([["a", "b"]], k=1000))

    gap_small_k = fused_small_k["a"] - fused_small_k["b"]
    gap_large_k = fused_large_k["a"] - fused_large_k["b"]
    assert gap_small_k > gap_large_k
