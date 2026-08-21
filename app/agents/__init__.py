"""Agent layer (Milestone 4): an LLM-orchestrated tool-use loop.

    User -> Agent -> Tool Selection -> Tools -> Observation -> Final Answer

- `tools/` — the four tools (document search, web search, calculator,
  document summary), each implementing `tools.base.BaseTool`.
- `prompts.py` — the agent's system prompt (JSON-action protocol).
- `parsing.py` — parses the model's action JSON; builds the user-safe
  `reasoning_summary` from tool *names* only, never the model's own text.
- `orchestrator.py` — the loop itself: max iterations, per-tool
  timeouts, error handling, graceful fallback.

`app.services.agent_service` is the DB/settings-aware layer that wires
tools together per-request and exposes this to the API — kept separate
from `app.services.chat_service` (the plain RAG service), per the
milestone's explicit requirement.
"""
