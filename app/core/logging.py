"""Logging configuration for the whole application.

`configure_logging()` must run once, as early as possible — it is
called from `app.main.create_app()` before any router or service
imports emit a log record.

`Settings.log_format`:
- "text" (default, best for local dev) — human-readable single line.
- "json" (for production/log aggregators — CloudWatch, Loki, ELK) —
  one JSON object per line, machine-parseable, includes request_id.
Both include the per-request X-Request-ID (see app.core.request_id) so
one request's log lines can be grepped/queried across every layer.
"""
import json
import logging.config
from pathlib import Path

from app.core.config import Settings
from app.core.request_id import RequestIDLogFilter

_TEXT_FORMAT = "%(asctime)s | %(levelname)-8s | request_id=%(request_id)s | %(name)s | %(message)s"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(settings: Settings) -> None:
    log_file = Path(settings.log_dir) / "app.log"
    formatter_name = "json" if settings.log_format == "json" else "text"

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {"request_id": {"()": RequestIDLogFilter}},
        "formatters": {
            "text": {"format": _TEXT_FORMAT, "datefmt": "%Y-%m-%d %H:%M:%S"},
            "json": {"()": JsonFormatter},
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": formatter_name,
                "filters": ["request_id"],
                "level": settings.log_level,
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": formatter_name,
                "filters": ["request_id"],
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
