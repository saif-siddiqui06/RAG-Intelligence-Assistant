"""The agent loop:

    User -> Agent (LLM decides an action) -> Tool Selection -> Tools
         -> Observation -> ... (repeat) ... -> Final Answer

Built on BaseChatModel (not a provider-specific function-calling API),
using the JSON-action protocol in app.agents.prompts — this keeps the
agent portable across any BaseChatModel implementation, same as
query_rewriter/answer_generator.
"""
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass

from app.agents.parsing import build_reasoning_summary, parse_agent_action
from app.agents.prompts import build_observation_message, build_system_prompt
from app.agents.tools.base import BaseTool, ToolSource
from app.rag.generation.base import BaseChatModel

logger = logging.getLogger(__name__)

FALLBACK_MESSAGE = (
    "I wasn't able to fully resolve this within the allowed number of steps. "
    "Here is my best answer based on what I found so far."
)


@dataclass
class AgentResult:
    answer: str
    tools_used: list[str]
    sources: list[ToolSource]
    reasoning_summary: str
    execution_time: float


class AgentOrchestrator:
    def __init__(
        self,
        chat_model: BaseChatModel,
        tools: list[BaseTool],
        max_iterations: int = 5,
        tool_timeout: float = 20.0,
    ) -> None:
        self._chat_model = chat_model
        self._tools_by_name = {t.name: t for t in tools}
        self._system_prompt = build_system_prompt(tools)
        self._max_iterations = max_iterations
        self._tool_timeout = tool_timeout

    def run(self, query: str) -> AgentResult:
        start = time.perf_counter()
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": query},
        ]
        tools_used: list[str] = []
        sources: list[ToolSource] = []
        last_observation = ""

        for _ in range(self._max_iterations):
            raw = self._chat_model.complete(messages, temperature=0)
            action = parse_agent_action(raw)

            if action is None:
                # Didn't follow the JSON protocol — fall back to treating
                # its raw text as the answer rather than erroring out.
                return self._finish(raw.strip(), tools_used, sources, start)

            if action.name == "final_answer":
                answer = action.input.get("answer") or raw.strip()
                return self._finish(answer, tools_used, sources, start)

            tool = self._tools_by_name.get(action.name)
            if tool is None:
                available = ", ".join(self._tools_by_name)
                observation = f"Error: unknown tool '{action.name}'. Available tools: {available}."
            else:
                observation, tool_sources = self._run_tool_safely(tool, action.input)
                tools_used.append(action.name)
                sources.extend(tool_sources)

            last_observation = observation
            messages.append({"role": "assistant", "content": raw})
            messages.append(
                {"role": "user", "content": build_observation_message(action.name, observation)}
            )

        # Max iterations exceeded — a deliberate, graceful fallback instead
        # of crashing or looping forever.
        answer = f"{FALLBACK_MESSAGE}\n\n{last_observation}" if last_observation else FALLBACK_MESSAGE
        return self._finish(answer, tools_used, sources, start)

    def _run_tool_safely(self, tool: BaseTool, tool_input: dict) -> tuple[str, list[ToolSource]]:
        # Deliberately not a `with` block: on timeout we must return
        # immediately, not block waiting for shutdown() to join a thread
        # that may be genuinely hung (e.g. a stalled network call).
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(tool.run, **tool_input)
        try:
            result = future.result(timeout=self._tool_timeout)
        except FutureTimeoutError:
            logger.warning("Tool %s timed out after %ss", tool.name, self._tool_timeout)
            executor.shutdown(wait=False)
            return f"Error: {tool.name} timed out after {self._tool_timeout}s.", []
        except Exception as exc:
            logger.exception("Tool %s raised unexpectedly", tool.name)
            executor.shutdown(wait=False)
            return f"Error running {tool.name}: {exc}", []
        executor.shutdown(wait=False)
        return result.output, result.sources

    def _finish(
        self, answer: str, tools_used: list[str], sources: list[ToolSource], start: float
    ) -> AgentResult:
        return AgentResult(
            answer=answer,
            tools_used=tools_used,
            sources=sources,
            reasoning_summary=build_reasoning_summary(tools_used),
            execution_time=time.perf_counter() - start,
        )
