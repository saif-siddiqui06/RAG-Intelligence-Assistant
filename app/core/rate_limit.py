"""Shared slowapi rate limiter.

One `Limiter` instance for the whole app: `app.main` registers it on
`app.state` + wires the 429 handler and middleware, and individual
endpoints import this same instance to decorate their routes with
`@limiter.limit(...)`.
"""
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
