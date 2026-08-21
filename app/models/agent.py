"""API schemas for the agent endpoint."""
from pydantic import BaseModel, Field

from app.models.document import DocumentType


class AgentRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: str | None = None
    document_id: str | None = None
    document_type: DocumentType | None = None


class ToolSourceOut(BaseModel):
    tool: str
    document_name: str | None = None
    page_number: int | None = None
    chunk_id: str | None = None
    url: str | None = None
    title: str | None = None


class AgentResponse(BaseModel):
    answer: str
    tools_used: list[str]
    sources: list[ToolSourceOut]
    reasoning_summary: str
    execution_time: float
