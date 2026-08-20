"""SQLAlchemy engine/session setup.

Defaults to a local SQLite file (see Settings.database_url) so document
ingestion works with zero extra infra. Swapping to Postgres later is a
one-line env var change — nothing here or in app/database/models.py
needs to change.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import get_settings

settings = get_settings()

_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, pool_pre_ping=True, future=True, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency that yields a request-scoped DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create tables for any ORM models registered on Base.metadata.

    Imported lazily inside the function (not at module top) because
    app.database.models imports Base from this module — importing it
    up front would be a circular import.
    """
    from app.database import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
