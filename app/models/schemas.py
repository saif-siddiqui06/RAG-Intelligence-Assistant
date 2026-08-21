"""Shared Pydantic schemas (request/response models).

Document, chat and evaluation schemas will be added here as those
milestones land. Only the health check schema exists for now.
"""
from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    environment: str
    database: str  # "ok" | "unavailable"
