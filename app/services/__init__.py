"""Business-logic layer between API endpoints and rag/agents/database.

Endpoints stay thin and delegate real work to services here.
`document_service.py` orchestrates document upload/list/delete/reindex.
`chunk_lookup.py` holds SQL helpers shared by both retrieval services.
`retrieval_service.py` (vector-only) and `hybrid_retrieval_service.py`
(vector + BM25 + rerank) are two interchangeable retrieval strategies,
selected by `Settings.retrieval_mode`; `chat_service.py` is the
orchestrator that picks one and sequences it with query rewriting and
generation — kept separate per the "retrieval, prompting and
generation as separate services" requirement.
`agent_service.py` is the agent's DB-aware wiring layer — separate from
`chat_service.py` on purpose (Milestone 4): it builds the four tools
per-request and hands them to `app.agents.orchestrator`, but never
implements RAG logic itself (document_search_tool calls back into
ChatService for that).
"""
