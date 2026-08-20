"""HTTP layer: routers and request-scoped dependencies.

Endpoints here stay thin — they parse/validate input and delegate all
real work to `app.services`. Versioned under `v1/` so future breaking
changes can live alongside it as `v2/`.
"""
