"""Document summary tool — summarizes one uploaded document's full
content (not just a retrieved handful of chunks). Looks the document
up by id or filename, concatenates its chunk text (bounded by
Settings.summary_max_chars — a portfolio-scale simplification; a real
system would map-reduce over chunks instead of truncating), and asks
the chat model for a summary.
"""
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agents.tools.base import BaseTool, ToolResult, ToolSource
from app.core.config import Settings
from app.database.models import DocumentRecord
from app.rag.generation.base import BaseChatModel

_SUMMARY_SYSTEM_PROMPT = (
    "You are a research assistant. Summarize the following document excerpt "
    "clearly and concisely for someone who hasn't read it: cover its main "
    "purpose, key points, and any notable conclusions. Do not invent "
    "details that aren't in the text."
)


class DocumentSummaryTool(BaseTool):
    name = "document_summary_tool"
    description = (
        "Summarizes one uploaded document, identified by filename or "
        "document id. Use this when the user asks to 'summarize' a "
        "document, not for narrow factual questions (use "
        "document_search_tool for those)."
    )
    parameters = {
        "document_id": "Optional: the document's id, if known.",
        "filename": "Optional: the document's filename (or part of it), if the id isn't known.",
    }

    def __init__(self, db: Session, settings: Settings, chat_model: BaseChatModel) -> None:
        self._db = db
        self._settings = settings
        self._chat_model = chat_model

    def run(
        self, document_id: str | None = None, filename: str | None = None, **kwargs
    ) -> ToolResult:
        record = self._resolve_document(document_id, filename)
        if record is None:
            return ToolResult(
                output=f"No matching document found for {filename or document_id or '(none given)'}.",
                error="not_found",
            )
        if record.status != "completed" or not record.chunks:
            return ToolResult(
                output=f"'{record.filename}' has no processed content to summarize (status: {record.status}).",
                error="no_content",
            )

        chunks = sorted(record.chunks, key=lambda c: (c.page_number, c.chunk_index))
        text_parts: list[str] = []
        budget = self._settings.summary_max_chars
        for chunk in chunks:
            if sum(len(p) for p in text_parts) >= budget:
                break
            text_parts.append(chunk.content)
        document_text = "\n\n".join(text_parts)[:budget]

        messages = [
            {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": f"Document: {record.filename}\n\n{document_text}"},
        ]
        try:
            summary = self._chat_model.complete(messages, temperature=0).strip()
        except Exception as exc:
            return ToolResult(output=f"Summarization failed: {exc}", error=str(exc))

        source = ToolSource(
            tool=self.name,
            document_name=record.filename,
            page_number=chunks[0].page_number,
            chunk_id=chunks[0].id,
        )
        return ToolResult(output=summary, sources=[source])

    def _resolve_document(
        self, document_id: str | None, filename: str | None
    ) -> DocumentRecord | None:
        if document_id:
            return self._db.get(DocumentRecord, document_id)
        if filename:
            return self._db.scalar(
                select(DocumentRecord).where(DocumentRecord.filename.ilike(f"%{filename}%"))
            )
        return None
