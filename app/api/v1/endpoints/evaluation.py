"""Retrieval evaluation endpoint — runs app.evaluation.retrieval_benchmark
(vector-only vs BM25-only vs hybrid vs hybrid+rerank) against the
built-in 22-question dataset and returns the same metrics the CLI
script prints, as JSON for the Streamlit Evaluation page.

Heavily rate-limited: each run makes ~20+ real Gemini embedding calls
against a 20-request/day free-tier quota, so this must stay an
explicit, user-triggered action — never called automatically.
"""
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from app.api.deps import get_app_settings
from app.core.config import Settings, get_settings
from app.core.exceptions import AppException
from app.core.rate_limit import limiter
from app.evaluation.retrieval_benchmark import run_benchmark

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


class BenchmarkResponse(BaseModel):
    results: dict
    per_question: list[dict]


@router.post("/benchmark", response_model=BenchmarkResponse)
@limiter.limit(get_settings().rate_limit_evaluation)
def benchmark(request: Request, settings: Settings = Depends(get_app_settings)) -> BenchmarkResponse:
    if not settings.gemini_api_key:
        raise AppException("GEMINI_API_KEY is not configured; cannot run the benchmark", status_code=503)
    return BenchmarkResponse(**run_benchmark())
