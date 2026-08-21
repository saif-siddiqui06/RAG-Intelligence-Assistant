"""Web search tool — free, no API key (DuckDuckGo via the `ddgs` package).

Used for anything the uploaded documents wouldn't cover: current
events, general facts, "what happened this week"-style queries.
"""
import logging

from app.agents.tools.base import BaseTool, ToolResult, ToolSource

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    name = "web_search_tool"
    description = (
        "Searches the public web. Use this for current events, general "
        "knowledge, or anything not covered by the uploaded documents — "
        "never for questions about 'my paper' or 'my document'."
    )
    parameters = {"query": "The search query text."}

    def __init__(self, max_results: int = 5) -> None:
        self._max_results = max_results

    def run(self, query: str = "", **kwargs) -> ToolResult:
        if not query:
            return ToolResult(output="No search query provided.", error="missing_query")
        try:
            from ddgs import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=self._max_results))
        except Exception as exc:
            logger.exception("Web search failed")
            return ToolResult(output=f"Web search failed: {exc}", error=str(exc))

        if not results:
            return ToolResult(output="No web results found for this query.")

        lines = []
        sources = []
        for i, r in enumerate(results, start=1):
            title, body, url = r.get("title", ""), r.get("body", ""), r.get("href", "")
            lines.append(f"[{i}] {title}\n{body}\nURL: {url}")
            sources.append(ToolSource(tool=self.name, title=title, url=url))
        return ToolResult(output="\n\n".join(lines), sources=sources)
