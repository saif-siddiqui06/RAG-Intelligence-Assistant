"""Document search tool — the agent's window into the existing advanced
RAG pipeline: query rewriting, hybrid (or vector-only) retrieval,
reranking, and metadata filtering, followed by grounded, cited
generation. This tool does not reimplement any of that; it just calls
app.services.chat_service.ChatService, exactly as the plain chat
endpoint does, and forwards its answer + citations as an observation.
"""
from sqlalchemy.orm import Session

from app.agents.tools.base import BaseTool, ToolResult, ToolSource
from app.core.config import Settings
from app.models.chat import ChatRequest
from app.services.chat_service import ChatService


class DocumentSearchTool(BaseTool):
    name = "document_search_tool"
    description = (
        "Searches the user's uploaded documents using the full advanced RAG "
        "pipeline (query rewriting, hybrid retrieval, reranking) and returns "
        "a grounded, cited answer. Use this for any question about 'my "
        "document(s)', 'my paper', or their content."
    )
    parameters = {
        "query": "The question to answer from the uploaded documents.",
        "document_id": "Optional: restrict the search to one document's id.",
        "document_type": "Optional: restrict the search to one document type (e.g. 'pdf').",
    }

    def __init__(self, db: Session, settings: Settings, session_id: str | None = None) -> None:
        self._db = db
        self._settings = settings
        self._session_id = session_id

    def run(
        self,
        query: str = "",
        document_id: str | None = None,
        document_type: str | None = None,
        **kwargs,
    ) -> ToolResult:
        if not query:
            return ToolResult(output="No query provided.", error="missing_query")
        request = ChatRequest(
            question=query,
            session_id=self._session_id,
            document_id=document_id or None,
            document_type=document_type or None,
        )
        try:
            response = ChatService(self._db, self._settings).ask(request)
        except Exception as exc:
            return ToolResult(output=f"Document search failed: {exc}", error=str(exc))

        sources = [
            ToolSource(
                tool=self.name,
                document_name=s.document_name,
                page_number=s.page_number,
                chunk_id=s.chunk_id,
            )
            for s in response.sources
        ]
        return ToolResult(output=response.answer, sources=sources)
