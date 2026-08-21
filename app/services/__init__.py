"""Business-logic layer between API endpoints and rag/agents/database.

Endpoints stay thin and delegate real work to services here.
`document_service.py` orchestrates document upload/list/delete/reindex.
`retrieval_service.py` and `chat_service.py` orchestrate the
conversational RAG pipeline (query rewrite -> retrieval -> generation ->
citations) — kept as two services per the milestone's "keep retrieval,
prompting and generation separate" requirement.
"""
