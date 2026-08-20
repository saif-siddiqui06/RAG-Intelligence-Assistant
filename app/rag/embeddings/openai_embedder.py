"""OpenAI implementation of BaseEmbedder."""
import logging

from openai import OpenAI

from app.core.config import Settings
from app.core.exceptions import AppException
from app.rag.embeddings.base import BaseEmbedder

logger = logging.getLogger(__name__)

# Known dimensions for OpenAI's current embedding models, so callers can
# size a vector store before the first real API call.
_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_BATCH_SIZE = 100


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, settings: Settings) -> None:
        if not settings.openai_api_key:
            raise AppException(
                "OPENAI_API_KEY is not configured — set it in .env before ingesting documents",
                status_code=500,
            )
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model
        self._dimension = _MODEL_DIMENSIONS.get(self._model, 1536)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = texts[start : start + _BATCH_SIZE]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        try:
            response = self._client.embeddings.create(model=self._model, input=batch)
        except Exception as exc:
            logger.exception("OpenAI embedding request failed")
            raise AppException(f"Embedding provider request failed: {exc}", status_code=502) from exc
        return [item.embedding for item in response.data]
