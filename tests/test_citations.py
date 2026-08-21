"""Tests for citation-marker extraction and no-context-sentinel
detection — the pure logic answer_generator uses to turn model output
into a Sources list.
"""
from app.rag.generation.answer_generator import extract_cited_indices, is_no_context_answer
from app.rag.generation.prompts import NO_CONTEXT_SENTINEL


def test_extracts_cited_source_numbers_in_ascending_order():
    text = "SMOTE oversamples the minority class [1]. It can introduce noise [2][1]."

    assert extract_cited_indices(text, max_index=2) == [1, 2]


def test_ignores_out_of_range_citations():
    text = "Some claim [1] and a bogus one [99]."

    assert extract_cited_indices(text, max_index=1) == [1]


def test_no_citation_markers_returns_empty_list():
    assert extract_cited_indices("An answer with no citations at all.", max_index=3) == []


def test_detects_no_context_sentinel_case_and_whitespace_insensitively():
    assert is_no_context_answer(NO_CONTEXT_SENTINEL)
    assert is_no_context_answer(NO_CONTEXT_SENTINEL.lower())
    assert is_no_context_answer(f"  {NO_CONTEXT_SENTINEL}  ")


def test_normal_answer_is_not_mistaken_for_the_sentinel():
    assert not is_no_context_answer("SMOTE oversamples the minority class [1].")
