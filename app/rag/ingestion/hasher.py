"""Content hashing used to detect duplicate document uploads."""
import hashlib


def compute_file_hash(data: bytes) -> str:
    """SHA-256 hex digest of the raw file bytes.

    Hashing content (not filename) means renamed-but-identical files
    are still caught as duplicates.
    """
    return hashlib.sha256(data).hexdigest()
