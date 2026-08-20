"""SQLAlchemy persistence layer.

Backed by SQLite today (see app.core.config.Settings.database_url),
swappable to Postgres by changing that URL alone. Holds document/chunk
metadata (app.database.models) for the ingestion pipeline; conversation
memory will land here too in a later milestone.
"""
