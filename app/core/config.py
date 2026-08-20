"""Centralized application configuration.

All environment-driven settings are declared here so the rest of the
codebase never reads `os.environ` directly. Values are loaded from a
`.env` file (see `.env.example`) with sane defaults for local dev.
"""
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_name: str = "Agentic RAG Research Assistant"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = True

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000
    api_v1_prefix: str = "/api/v1"

    # --- CORS ---
    backend_cors_origins: list[str] = ["http://localhost:8501"]

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # --- Database ---
    # SQLite by default so ingestion works with zero extra infra. Swap to a
    # postgresql:// URL later (the persistence milestone) — the SQLAlchemy
    # code in app/database/ doesn't change either way.
    database_url: str = f"sqlite:///{(BASE_DIR / 'data' / 'processed' / 'metadata.db').as_posix()}"

    # --- RAG / ingestion ---
    vector_store_backend: str = "faiss"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    chunk_separators: list[str] = ["\n\n", "\n", ". ", " ", ""]

    # --- Storage paths ---
    upload_dir: Path = BASE_DIR / "data" / "uploads"
    processed_dir: Path = BASE_DIR / "data" / "processed"
    vectorstore_dir: Path = BASE_DIR / "data" / "vectorstore"

    # --- Logging ---
    log_level: str = "INFO"
    log_dir: Path = BASE_DIR / "logs"

    # --- Frontend ---
    api_base_url: str = "http://localhost:8000"


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached Settings instance.

    Also ensures the data/log directories exist, so a fresh clone of
    the repo works without any manual setup beyond copying `.env`.
    """
    settings = Settings()
    for path in (settings.upload_dir, settings.processed_dir, settings.vectorstore_dir, settings.log_dir):
        path.mkdir(parents=True, exist_ok=True)
    return settings
