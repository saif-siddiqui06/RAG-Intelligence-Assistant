"""Shared FastAPI dependency-injection providers.

Endpoints should depend on functions here rather than reaching into
`app.core` or `app.database` directly, so wiring changes (e.g. swapping
how the DB session is created) happen in one place.
"""
from app.core.config import Settings, get_settings
from app.database.session import get_db

__all__ = ["get_app_settings", "get_db"]


def get_app_settings() -> Settings:
    return get_settings()
