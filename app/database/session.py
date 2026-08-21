"""SQLAlchemy engine/session setup.

Defaults to a local SQLite file (see Settings.database_url) so document
ingestion works with zero extra infra. Swapping to Postgres later is a
one-line env var change — nothing here or in app/database/models.py
needs to change.
"""
import logging

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import get_settings

logger = logging.getLogger(__name__)
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

    In production, schema is owned by Alembic (`alembic upgrade head`,
    run as a deploy step / Docker entrypoint) since create_all() can add
    new tables but can never ALTER an existing one — it would silently
    skip new columns added to an already-created table. In dev/test,
    create_all() is kept as a zero-friction convenience so a fresh clone
    or an in-memory/tmp SQLite test DB works with no extra setup step.

    Imported lazily inside the function (not at module top) because
    app.database.models imports Base from this module — importing it
    up front would be a circular import.
    """
    from app.database import models  # noqa: F401

    if settings.environment == "production":
        logger.info("Skipping create_all() in production; schema is managed by Alembic migrations")
        return
    Base.metadata.create_all(bind=engine)
