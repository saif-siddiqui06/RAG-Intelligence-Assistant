"""Vector store selection — the one place a backend swap happens."""
from app.core.config import Settings
from app.rag.vectorstore.base import VectorStore
from app.rag.vectorstore.faiss_store import FaissVectorStore


def get_vector_store(settings: Settings, dimension: int) -> VectorStore:
    backend = settings.vector_store_backend.lower()
    if backend == "faiss":
        index_path = settings.vectorstore_dir / "index.faiss"
        return FaissVectorStore(index_path=index_path, dimension=dimension)
    # Add Chroma/Qdrant/pgvector branches here later — callers never
    # need to change, they only ever see the VectorStore interface.
    raise ValueError(f"Unsupported vector store backend: {backend!r}")
