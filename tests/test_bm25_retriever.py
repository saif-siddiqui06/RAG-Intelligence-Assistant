"""Tests for BM25 keyword retrieval."""
from app.rag.retrieval.bm25_retriever import BM25Retriever, tokenize


def test_tokenize_lowercases_and_strips_punctuation():
    assert tokenize("SMOTE's Algorithm!") == ["smote", "s", "algorithm"]


def test_exact_keyword_match_ranks_highest():
    corpus = [
        ("a", "SMOTE oversamples the minority class using synthetic examples."),
        ("b", "Random forests build many decision trees and aggregate votes."),
        ("c", "Gradient boosting fits new trees to residual errors sequentially."),
    ]
    retriever = BM25Retriever(corpus)

    results = retriever.search("SMOTE synthetic minority", top_k=3)

    assert results
    assert results[0].chunk_id == "a"


def test_no_term_overlap_returns_empty():
    corpus = [("a", "SMOTE oversamples the minority class.")]
    retriever = BM25Retriever(corpus)

    results = retriever.search("completely unrelated query about baking bread", top_k=5)

    assert results == []


def test_empty_corpus_returns_empty():
    retriever = BM25Retriever([])

    assert retriever.search("anything", top_k=5) == []


def test_top_k_limits_results():
    corpus = [(str(i), f"repeated keyword apple number {i}") for i in range(10)]
    retriever = BM25Retriever(corpus)

    results = retriever.search("apple", top_k=3)

    assert len(results) == 3
