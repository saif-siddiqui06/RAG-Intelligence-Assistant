"""Aggregates all v1 endpoint routers into a single router.

`app.main` mounts this one router under the configured API prefix, so
new endpoint modules only need to be registered here, not in main.py.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import agent, chat, documents, health

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(chat.router)
api_router.include_router(agent.router)
