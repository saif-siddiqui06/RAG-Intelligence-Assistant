"""Agentic endpoint: the LLM picks which tool(s) to use, if any."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_app_settings, get_db
from app.core.config import Settings
from app.models.agent import AgentRequest, AgentResponse
from app.services.agent_service import AgentService

router = APIRouter(prefix="/agent", tags=["agent"])


@router.post("", response_model=AgentResponse)
def run_agent(
    request: AgentRequest,
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
) -> AgentResponse:
    return AgentService(db, settings).run(request)
