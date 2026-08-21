"""Tests for the retrieval evaluation metrics."""
from app.evaluation.metrics import mean, reciprocal_rank, recall_at_k


def test_recall_at_k_hits_when_relevant_id_present():
    assert recall_at_k(["a", "b", "c"], ["b"], k=3) == 1.0


def test_recall_at_k_misses_when_relevant_id_outside_k():
    assert recall_at_k(["a", "b", "c", "d"], ["d"], k=2) == 0.0


def test_recall_at_k_handles_multiple_acceptable_answers():
    assert recall_at_k(["a", "b"], ["x", "b"], k=2) == 1.0


def test_reciprocal_rank_first_position():
    assert reciprocal_rank(["a", "b", "c"], ["a"]) == 1.0


def test_reciprocal_rank_second_position():
    assert reciprocal_rank(["a", "b", "c"], ["b"]) == 0.5


def test_reciprocal_rank_not_found_is_zero():
    assert reciprocal_rank(["a", "b"], ["z"]) == 0.0


def test_mean_of_empty_list_is_zero():
    assert mean([]) == 0.0


def test_mean_averages_values():
    assert mean([1.0, 0.0, 1.0, 0.0]) == 0.5
