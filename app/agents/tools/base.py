"""Tool interface for the agent.

Every tool declares its own name/description/parameter schema (rendered
into the agent's system prompt — see app.agents.prompts) so the LLM
knows what's available and how to call it correctly, and returns a
plain ToolResult the orchestrator feeds back as an observation. Tools
never talk to the LLM's calling convention directly; only the
orchestrator does.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class ToolSource:
    """A citation-like reference a tool's output is grounded in —
    document-based (page_number/chunk_id) or web-based (url/title).
    """

    tool: str
    document_name: str | None = None
    page_number: int | None = None
    chunk_id: str | None = None
    url: str | None = None
    title: str | None = None


@dataclass
class ToolResult:
    output: str  # fed back to the LLM as the observation text
    sources: list[ToolSource] = field(default_factory=list)
    error: str | None = None


class BaseTool(ABC):
    name: str
    description: str
    # {param_name: "human-readable description of what it is and its type"}
    parameters: dict[str, str]

    @abstractmethod
    def run(self, **kwargs) -> ToolResult:
        """Execute the tool. Must not raise for expected failure modes
        (not-found, empty results, upstream errors) — return a ToolResult
        with `error` set instead, so the agent loop can keep going and
        let the LLM decide what to do next.
        """
