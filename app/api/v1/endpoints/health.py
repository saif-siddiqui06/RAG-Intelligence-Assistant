"""Health check endpoint — used by the Streamlit UI, Docker healthcheck
and manual smoke testing. Checks real DB connectivity rather than just
reporting "the process is up", since a dead database is the most common
way this service actually fails in production.
"""
import logging

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_app_settings, get_db
from app.core.config import Settings
from app.models.schemas import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["health"])
def health_check(
    settings: Settings = Depends(get_app_settings), db: Session = Depends(get_db)
) -> HealthResponse:
    try:
        db.execute(text("SELECT 1"))
        database_status = "ok"
    except Exception:
        logger.exception("Health check: database connectivity failed")
        database_status = "unavailable"

    return HealthResponse(
        status="ok" if database_status == "ok" else "degraded",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
        database=database_status,
    )
