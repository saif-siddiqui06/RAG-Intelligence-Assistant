"""Orchestrates the ingestion flow:

    PDF -> extraction -> cleaning -> chunking -> metadata -> embeddings -> vector store

Pure orchestration — no FastAPI, no database session. `app.services
.document_service` calls this and is the only place results get
persisted to SQL.
"""
from dataclasses import dataclass
from pathlib import Path

from app.core.exceptions import AppException
from app.rag.embeddings.base import BaseEmbedder
from app.rag.ingestion.chunker import RecursiveCharacterChunker
from app.rag.ingestion.cleaner import clean_text
from app.rag.ingestion.extractor import PDFExtractor
from app.rag.vectorstore.base import VectorStore


@dataclass
class ChunkResult:
    page_number: int
    chunk_index: int
    content: str
    vector_id: int


@dataclass
class IngestionOutput:
    num_pages: int
    chunks: list[ChunkResult]


class IngestionPipeline:
    def __init__(
        self,
        extractor: PDFExtractor,
        chunker: RecursiveCharacterChunker,
        embedder: BaseEmbedder,
        vector_store: VectorStore,
    ) -> None:
        self.extractor = extractor
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def ingest(self, file_path: Path) -> IngestionOutput:
        pages = self.extractor.extract(file_path)
        if not pages:
            raise AppException("PDF has no pages", status_code=422)

        pending: list[tuple[int, int, str]] = []
        for page in pages:
            cleaned = clean_text(page.text)
            if not cleaned:
                continue
            for chunk_index, piece in enumerate(self.chunker.split_text(cleaned)):
                pending.append((page.page_number, chunk_index, piece))

        if not pending:
            raise AppException(
                "No extractable text found in this PDF (it may be scanned/image-only)",
                status_code=422,
            )

        texts = [item[2] for item in pending]
        vectors = self.embedder.embed_documents(texts)
        if len(vectors) != len(texts):
            # zip() below would otherwise silently truncate to the shorter
            # list — that's exactly how a provider bug once turned a
            # 37-page document into a single stored chunk with no error.
            raise AppException(
                f"Embedder returned {len(vectors)} vectors for {len(texts)} chunks — "
                "refusing to continue with misaligned data.",
                status_code=502,
            )
        vector_ids = self.vector_store.add(vectors)

        chunks = [
            ChunkResult(page_number=page_number, chunk_index=chunk_index, content=text, vector_id=vector_id)
            for (page_number, chunk_index, text), vector_id in zip(pending, vector_ids)
        ]
        return IngestionOutput(num_pages=len(pages), chunks=chunks)
