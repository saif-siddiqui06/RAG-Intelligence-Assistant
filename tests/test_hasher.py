"""Unit tests for the file-hashing duplicate-detection primitive."""
from app.rag.ingestion.hasher import compute_file_hash


def test_identical_bytes_produce_identical_hash():
    data = b"identical content"

    assert compute_file_hash(data) == compute_file_hash(data)


def test_different_bytes_produce_different_hashes():
    assert compute_file_hash(b"one") != compute_file_hash(b"two")


def test_hash_is_a_sha256_hex_digest():
    digest = compute_file_hash(b"some file bytes")

    assert len(digest) == 64
    int(digest, 16)  # raises ValueError if not valid hex


def test_single_byte_difference_changes_the_hash():
    # Duplicate detection relies on this: even a near-identical file must
    # not collide with the original.
    assert compute_file_hash(b"payload-A") != compute_file_hash(b"payload-B")
