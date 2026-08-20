"""Cached singletons for the RAG pipeline's stateful components.

Kept separate from app.api.deps so app.rag never imports FastAPI — it
stays usable from scripts/tests without the web framework running.
"""
from functools import lru_cache

from app.core.config import get_settings
from app.rag.embeddings.base import BaseEmbedder
from app.rag.embeddings.openai_embedder import OpenAIEmbedder
from app.rag.ingestion.chunker import ChunkingConfig, RecursiveCharacterChunker
from app.rag.ingestion.extractor import PDFExtractor
from app.rag.ingestion.pipeline import IngestionPipeline
from app.rag.vectorstore.base import VectorStore
from app.rag.vectorstore.factory import get_vector_store


@lru_cache
def get_embedder() -> BaseEmbedder:
    return OpenAIEmbedder(get_settings())


@lru_cache
def get_vector_store_instance() -> VectorStore:
    return get_vector_store(get_settings(), dimension=get_embedder().dimension)


def build_ingestion_pipeline(
    chunk_size: int | None = None, chunk_overlap: int | None = None
) -> IngestionPipeline:
    settings = get_settings()
    config = ChunkingConfig(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        separators=list(settings.chunk_separators),
    )
    return IngestionPipeline(
        extractor=PDFExtractor(),
        chunker=RecursiveCharacterChunker(config),
        embedder=get_embedder(),
        vector_store=get_vector_store_instance(),
    )
