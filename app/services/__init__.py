"""Business-logic layer between API endpoints and rag/agents/database.

Endpoints stay thin and delegate real work to services here.
`document_service.py` orchestrates document upload/list/delete/reindex;
a chat/query service will be added alongside the retrieval milestone.
"""
