"""FAISS-backed implementation of VectorStore.

Persists to a single index file plus a tiny JSON sidecar (just the
next-id counter, so ids stay unique across restarts and deletions).
Nothing outside this module imports faiss directly.
"""
import json
import threading
from pathlib import Path

import faiss
import numpy as np

from app.rag.vectorstore.base import VectorStore


class FaissVectorStore(VectorStore):
    def __init__(self, index_path: Path, dimension: int) -> None:
        self._index_path = Path(index_path)
        self._meta_path = self._index_path.with_suffix(".meta.json")
        self._dimension = dimension
        self._lock = threading.Lock()
        self._next_id = 0
        self._index = self._load_or_create()

    def _load_or_create(self):
        if self._index_path.exists():
            index = faiss.read_index(str(self._index_path))
            if self._meta_path.exists():
                self._next_id = json.loads(self._meta_path.read_text()).get("next_id", 0)
            return index
        return faiss.IndexIDMap2(faiss.IndexFlatIP(self._dimension))

    def add(self, vectors: list[list[float]]) -> list[int]:
        with self._lock:
            matrix = np.asarray(vectors, dtype="float32")
            faiss.normalize_L2(matrix)
            ids = list(range(self._next_id, self._next_id + len(vectors)))
            self._index.add_with_ids(matrix, np.asarray(ids, dtype="int64"))
            self._next_id += len(vectors)
            self._persist()
            return ids

    def search(self, query_vector: list[float], top_k: int = 5) -> list[tuple[int, float]]:
        with self._lock:
            if self._index.ntotal == 0:
                return []
            matrix = np.asarray([query_vector], dtype="float32")
            faiss.normalize_L2(matrix)
            scores, ids = self._index.search(matrix, min(top_k, self._index.ntotal))
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def delete(self, ids: list[int]) -> None:
        if not ids:
            return
        with self._lock:
            self._index.remove_ids(np.asarray(ids, dtype="int64"))
            self._persist()

    def count(self) -> int:
        return self._index.ntotal

    def _persist(self) -> None:
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self._index_path))
        self._meta_path.write_text(json.dumps({"next_id": self._next_id}))
