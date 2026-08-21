"""FastAPI application factory and entrypoint.

Kept intentionally thin: it wires together config, logging, exception
handlers, middleware and routers, and delegates everything else to
`app.core` / `app.api`. Run with:

    uvicorn app.main:app --reload
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging
from app.core.rate_limit import limiter
from app.core.request_id import RequestIDMiddleware
from app.database.session import init_db

_DESCRIPTION = """
Agentic RAG Research Assistant — upload PDFs, ask grounded questions
with citations, and let an LLM-orchestrated agent reach for web search,
a calculator or document summarization when the question calls for it.

All responses are traceable: every request carries an `X-Request-ID`
header, and log lines across every layer (retrieval, generation, tool
calls) include that same ID.
"""


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings)
    init_db()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=_DESCRIPTION,
        debug=settings.debug,
    )

    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.backend_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(SlowAPIMiddleware)
    # Added last so it ends up outermost (Starlette wraps most-recently-added
    # middleware around everything else) — every request gets an id, even
    # ones CORS or the rate limiter reject.
    app.add_middleware(RequestIDMiddleware)

    register_exception_handlers(app)
    app.include_router(api_router, prefix=settings.api_v1_prefix)

    @app.get("/", tags=["root"])
    def root() -> dict:
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "docs": "/docs",
            "health": f"{settings.api_v1_prefix}/health",
        }

    return app


app = create_app()
