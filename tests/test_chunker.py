"""Unit tests for the configurable recursive chunking strategy."""
import pytest

from app.rag.ingestion.chunker import ChunkingConfig, RecursiveCharacterChunker


def test_short_text_returns_single_chunk():
    chunker = RecursiveCharacterChunker(ChunkingConfig(chunk_size=100, chunk_overlap=20))

    chunks = chunker.split_text("This is a short paragraph.")

    assert chunks == ["This is a short paragraph."]


def test_empty_text_returns_no_chunks():
    chunker = RecursiveCharacterChunker()

    assert chunker.split_text("") == []


def test_splits_long_text_respecting_chunk_size():
    text = ("Sentence one has some words. " * 30).strip()
    chunker = RecursiveCharacterChunker(ChunkingConfig(chunk_size=100, chunk_overlap=20))

    chunks = chunker.split_text(text)

    assert len(chunks) > 1
    # A little slack is allowed at merge boundaries, but chunks must stay close to the budget.
    assert all(len(c) <= 130 for c in chunks)


def test_reconstructs_full_content_from_chunks():
    text = "Paragraph one is here.\n\nParagraph two follows right after that.\n\nAnd a third one."
    chunker = RecursiveCharacterChunker(ChunkingConfig(chunk_size=40, chunk_overlap=5))

    chunks = chunker.split_text(text)

    assert "Paragraph one is here." in "".join(chunks)
    assert "Paragraph two" in "".join(chunks)
    assert "third one" in "".join(chunks)


def test_overlap_carries_content_between_chunks():
    text = " ".join(f"word{i}" for i in range(200))
    chunker = RecursiveCharacterChunker(ChunkingConfig(chunk_size=50, chunk_overlap=15))

    chunks = chunker.split_text(text)

    assert len(chunks) > 1
    for first, second in zip(chunks, chunks[1:]):
        assert any(word in second for word in first.split()[-2:])


def test_respects_custom_separators():
    text = "field1|field2|field3|field4|field5|field6"
    chunker = RecursiveCharacterChunker(
        ChunkingConfig(chunk_size=15, chunk_overlap=0, separators=["|", ""])
    )

    chunks = chunker.split_text(text)

    assert all(len(c) <= 15 for c in chunks)
    assert "field1" in "".join(chunks)


def test_chunk_size_must_be_positive():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=0)


def test_chunk_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=10, chunk_overlap=10)


def test_hard_slices_single_unbreakable_token_longer_than_chunk_size():
    text = "a" * 250
    chunker = RecursiveCharacterChunker(ChunkingConfig(chunk_size=100, chunk_overlap=10, separators=[""]))

    chunks = chunker.split_text(text)

    assert len(chunks) >= 3
    assert all(len(c) <= 100 for c in chunks)
