"""Cached singletons for the RAG pipeline's stateful components.

Kept separate from app.api.deps so app.rag never imports FastAPI — it
stays usable from scripts/tests without the web framework running.
"""
from functools import lru_cache

from app.core.config import get_settings
from app.rag.embeddings.base import BaseEmbedder
from app.rag.embeddings.gemini_embedder import GeminiEmbedder
from app.rag.generation.answer_generator import AnswerGenerator
from app.rag.generation.base import BaseChatModel
from app.rag.generation.gemini_chat_model import GeminiChatModel
from app.rag.generation.query_rewriter import QueryRewriter
from app.rag.ingestion.chunker import ChunkingConfig, RecursiveCharacterChunker
from app.rag.ingestion.extractor import PDFExtractor
from app.rag.ingestion.pipeline import IngestionPipeline
from app.rag.retrieval.vector_retriever import VectorRetriever
from app.rag.vectorstore.base import VectorStore
from app.rag.vectorstore.factory import get_vector_store


@lru_cache
def get_embedder() -> BaseEmbedder:
    return GeminiEmbedder(get_settings())


@lru_cache
def get_vector_store_instance() -> VectorStore:
    return get_vector_store(get_settings(), dimension=get_embedder().dimension)


@lru_cache
def get_chat_model() -> BaseChatModel:
    settings = get_settings()
    return GeminiChatModel(api_key=settings.gemini_api_key, model=settings.gemini_model)


@lru_cache
def get_vector_retriever() -> VectorRetriever:
    return VectorRetriever(embedder=get_embedder(), vector_store=get_vector_store_instance())


@lru_cache
def get_query_rewriter() -> QueryRewriter:
    return QueryRewriter(get_chat_model())


@lru_cache
def get_answer_generator() -> AnswerGenerator:
    return AnswerGenerator(get_chat_model())


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
