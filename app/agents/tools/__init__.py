"""The four agent tools, each implementing BaseTool.

- `document_search.py` — wraps the existing RAG pipeline (query
  rewrite -> hybrid retrieval -> rerank -> cited generation) via ChatService.
- `web_search.py` — free DuckDuckGo search (ddgs), no API key.
- `calculator.py` — safe AST-based arithmetic (never eval()/exec()).
- `document_summary.py` — summarizes one document's full chunk content.
"""
