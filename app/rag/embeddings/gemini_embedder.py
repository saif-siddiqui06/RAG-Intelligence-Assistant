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
    "gemini-embedding-2": 3072,
    "gemini-embedding-2-preview": 3072,
    "gemini-embedding-001": 3072,
    "text-embedding-004": 768,  # legacy model, kept for backward compatibility
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
        self._dimension = _MODEL_DIMENSIONS.get(self._model, 3072)

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
            # Each text MUST be its own Content — passing a flat list[str] as
            # `contents` gets interpreted as multiple *parts of one input* and
            # silently returns a single aggregate embedding for the whole
            # batch instead of one per text (confirmed against the live API).
            contents = [types.Content(parts=[types.Part(text=t)]) for t in batch]
            try:
                response = self._client.models.embed_content(
                    model=self._model,
                    contents=contents,
                    config=types.EmbedContentConfig(task_type=task_type),
                )
            except Exception as exc:
                logger.exception("Gemini embedding request failed")
                raise AppException(f"Embedding provider request failed: {exc}", status_code=502) from exc

            if len(response.embeddings) != len(batch):
                # Defense in depth: never let a count mismatch silently
                # misalign chunks and vectors (that's exactly how a 37-page
                # document once ended up with a single stored chunk).
                raise AppException(
                    f"Embedding provider returned {len(response.embeddings)} vectors for "
                    f"{len(batch)} inputs — refusing to continue with misaligned data.",
                    status_code=502,
                )
            vectors.extend(embedding.values for embedding in response.embeddings)
        return vectors
