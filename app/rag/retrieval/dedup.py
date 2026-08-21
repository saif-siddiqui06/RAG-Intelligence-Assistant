"""Near-duplicate detection for retrieved chunk content — shared by
both the vector-only and hybrid retrieval services.
"""
from difflib import SequenceMatcher


def is_near_duplicate(a: str, b: str, threshold: float) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= threshold
