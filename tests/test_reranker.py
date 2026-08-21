"""Tests for reranking: the real cross-encoder (model is small and
already cached locally, so this stays fast — no network call, no
download on a warm cache) and the no-op passthrough.
"""
import pytest

from app.rag.reranking.cross_encoder_reranker import CrossEncoderReranker
from app.rag.reranking.noop_reranker import NoOpReranker


def test_noop_reranker_preserves_input_order():
    scores = NoOpReranker().score("query", ["first", "second", "third"])

    assert scores[0] > scores[1] > scores[2]


def test_noop_reranker_handles_empty_input():
    assert NoOpReranker().score("query", []) == []


@pytest.fixture(scope="module")
def cross_encoder() -> CrossEncoderReranker:
    return CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")


def test_cross_encoder_scores_relevant_document_higher(cross_encoder):
    scores = cross_encoder.score(
        "What is SMOTE?",
        [
            "SMOTE oversamples the minority class by generating synthetic examples.",
            "Django is a Python web framework for building web applications.",
        ],
    )

    assert scores[0] > scores[1]
    assert all(0.0 <= s <= 1.0 for s in scores)  # sigmoid-normalized


def test_cross_encoder_handles_empty_input(cross_encoder):
    assert cross_encoder.score("query", []) == []
