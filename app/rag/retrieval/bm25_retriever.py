"""Pure BM25 keyword retrieval — no database, no HTTP.

Mirrors vector_retriever.py's shape (a `search()` method returning
scored hits) so HybridRetrievalService can treat both engines
uniformly. Unlike FAISS, rank_bm25 has no incremental index API: the
corpus is passed in fresh at construction time (built from SQL by the
caller — see app.services.chunk_lookup.load_chunk_corpus).
"""
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

# Classic BM25 IDF (Robertson-Sparck-Jones) goes *negative* for a term
# appearing in more than half the corpus — with a small corpus, a bare
# "the" or "what" can otherwise swamp genuine signal from the rare
# terms that actually distinguish documents. Every production BM25
# setup (Lucene/Elasticsearch included) strips stopwords before
# indexing for exactly this reason.
_STOPWORDS = frozenset(
    """
    a an the this that these those is are was were be been being
    to of in on for with at by from as it its it's and or but if
    not no do does did done doing have has had having
    what which who whom whose when where why how
    i you he she we they me him her us them my your his its our their
    can could will would shall should may might must
    so than then there here just also very
    """.split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_PATTERN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass
class KeywordHit:
    chunk_id: str
    score: float


class BM25Retriever:
    def __init__(self, corpus: list[tuple[str, str]]) -> None:
        """corpus: (chunk_id, text) pairs to index."""
        self._chunk_ids = [chunk_id for chunk_id, _ in corpus]
        tokenized_corpus = [tokenize(text) for _, text in corpus]
        self._bm25 = BM25Okapi(tokenized_corpus) if tokenized_corpus else None

    def search(self, query: str, top_k: int) -> list[KeywordHit]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self._chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
        # A score at or below 0 means no meaningful term overlap — not
        # a real candidate, just BM25's dense scoring returning a value
        # for every document in the corpus.
        return [KeywordHit(chunk_id=cid, score=float(s)) for cid, s in ranked[:top_k] if s > 0]
