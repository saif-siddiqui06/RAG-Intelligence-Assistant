"""Gemini implementation of BaseEmbedder.

Uses task_type to get Gemini's asymmetric retrieval embeddings — chunks
are embedded as RETRIEVAL_DOCUMENT, queries as RETRIEVAL_QUERY, which
the model was specifically trained to make more similar to each other
than a naive symmetric embedding would.
"""
import logging

from google import genai
from google.genai import types

from app.core.config import Settings
from app.core.exceptions import AppException
from app.rag.embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)

# Known output dimensions for Gemini's current embedding models.
_MODEL_DIMENSIONS = {
    "text-embedding-004": 768,
    "gemini-embedding-001": 3072,
}

_BATCH_SIZE = 100


class GeminiEmbedder(BaseEmbedder):
    def __init__(self, settings: Settings) -> None:
        if not settings.gemini_api_key:
            raise AppException(
                "GEMINI_API_KEY is not configured — set it in .env before ingesting documents",
                status_code=500,
            )
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.embedding_model
        self._dimension = _MODEL_DIMENSIONS.get(self._model, 768)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text], task_type="RETRIEVAL_QUERY")[0]

    def _embed(self, texts: list[str], task_type: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            try:
                response = self._client.models.embed_content(
                    model=self._model,
                    contents=batch,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
            except Exception as exc:
                logger.exception("Gemini embedding request failed")
                raise AppException(f"Embedding provider request failed: {exc}", status_code=502) from exc
            vectors.extend(embedding.values for embedding in response.embeddings)
        return vectors
