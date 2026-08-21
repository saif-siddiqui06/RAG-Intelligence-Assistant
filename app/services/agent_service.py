"""Agent service — separate from ChatService (the plain RAG service), as
required: this is the DB/settings-aware layer that builds the four
tools per-request, wires them into an AgentOrchestrator, runs it, and
converts the result into the API response shape.
"""
from sqlalchemy.orm import Session

from app.agents.orchestrator import AgentOrchestrator
from app.agents.tools.calculator import CalculatorTool
from app.agents.tools.document_search import DocumentSearchTool
from app.agents.tools.document_summary import DocumentSummaryTool
from app.agents.tools.web_search import WebSearchTool
from app.core.config import Settings
from app.models.agent import AgentRequest, AgentResponse, ToolSourceOut
from app.rag.dependencies import get_chat_model


class AgentService:
    def __init__(self, db: Session, settings: Settings) -> None:
        self.db = db
        self.settings = settings

    def run(self, request: AgentRequest) -> AgentResponse:
        chat_model = get_chat_model()
        tools = [
            DocumentSearchTool(self.db, self.settings, session_id=request.session_id),
            WebSearchTool(max_results=self.settings.web_search_max_results),
            CalculatorTool(),
            DocumentSummaryTool(self.db, self.settings, chat_model),
        ]
        orchestrator = AgentOrchestrator(
            chat_model,
            tools,
            max_iterations=self.settings.agent_max_iterations,
            tool_timeout=self.settings.agent_tool_timeout_seconds,
        )
        result = orchestrator.run(request.query)

        return AgentResponse(
            answer=result.answer,
            tools_used=result.tools_used,
            sources=[
                ToolSourceOut(
                    tool=s.tool,
                    document_name=s.document_name,
                    page_number=s.page_number,
                    chunk_id=s.chunk_id,
                    url=s.url,
                    title=s.title,
                )
                for s in result.sources
            ],
            reasoning_summary=result.reasoning_summary,
            execution_time=result.execution_time,
        )
