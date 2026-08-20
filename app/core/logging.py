"""Logging configuration for the whole application.

`configure_logging()` must run once, as early as possible — it is
called from `app.main.create_app()` before any router or service
imports emit a log record.
"""
import logging.config
from pathlib import Path

from app.core.config import Settings


def configure_logging(settings: Settings) -> None:
    log_file = Path(settings.log_dir) / "app.log"

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "default",
                "level": settings.log_level,
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "default",
                "filename": str(log_file),
                "maxBytes": 5 * 1024 * 1024,
                "backupCount": 3,
                "level": settings.log_level,
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": settings.log_level,
        },
        "loggers": {
            "uvicorn": {"level": "INFO", "propagate": True},
            "uvicorn.error": {"level": "INFO", "propagate": True},
            "uvicorn.access": {"level": "WARNING", "propagate": True},
        },
    }

    logging.config.dictConfig(logging_config)
