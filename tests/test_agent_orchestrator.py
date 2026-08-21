"""Tests for the agent loop: tool selection, tool execution, error
handling, timeouts, max-iteration fallback, and that no raw model
text/JSON ever leaks into reasoning_summary.
"""
import time

from app.agents.orchestrator import FALLBACK_MESSAGE, AgentOrchestrator
from app.agents.tools.base import BaseTool, ToolResult, ToolSource
from tests.fakes import FakeChatModel


class RecordingTool(BaseTool):
    name = "recording_tool"
    description = "A test tool that records the args it was called with."
    parameters = {"value": "any string"}

    def __init__(self, output="ok", sources=None):
        self.calls: list[dict] = []
        self._output = output
        self._sources = sources or []

    def run(self, **kwargs) -> ToolResult:
        self.calls.append(kwargs)
        return ToolResult(output=self._output, sources=self._sources)


class RaisingTool(BaseTool):
    name = "raising_tool"
    description = "Always raises."
    parameters = {}

    def run(self, **kwargs) -> ToolResult:
        raise RuntimeError("boom")


class SlowTool(BaseTool):
    name = "slow_tool"
    description = "Sleeps longer than the configured timeout."
    parameters = {}

    def run(self, **kwargs) -> ToolResult:
        time.sleep(2)
        return ToolResult(output="finished (should not be seen)")


def _action(name: str, action_input: dict | None = None) -> str:
    import json

    return json.dumps({"action": name, "action_input": action_input or {}})


def test_selects_and_calls_the_named_tool_with_its_input():
    tool = RecordingTool()
    chat_model = FakeChatModel(
        responses=[
            _action("recording_tool", {"value": "hello"}),
            _action("final_answer", {"answer": "done"}),
        ]
    )
    orchestrator = AgentOrchestrator(chat_model, [tool], max_iterations=5)

    result = orchestrator.run("do the thing")

    assert tool.calls == [{"value": "hello"}]
    assert result.tools_used == ["recording_tool"]
    assert result.answer == "done"


def test_multi_tool_query_calls_both_tools_in_order():
    tool_a = RecordingTool(output="fact A")
    tool_b = RecordingTool(output="42")
    tool_b.name = "other_tool"
    chat_model = FakeChatModel(
        responses=[
            _action("recording_tool", {"value": "q1"}),
            _action("other_tool", {"value": "q2"}),
            _action("final_answer", {"answer": "combined answer"}),
        ]
    )
    orchestrator = AgentOrchestrator(chat_model, [tool_a, tool_b], max_iterations=5)

    result = orchestrator.run("compound question")

    assert result.tools_used == ["recording_tool", "other_tool"]
    assert result.answer == "combined answer"
    # names not in the known-tool phrase map pass through unchanged
    assert result.reasoning_summary == "Used recording_tool and other_tool."


def test_sources_are_aggregated_from_tool_results():
    source = ToolSource(tool="recording_tool", document_name="doc.pdf", page_number=3, chunk_id="c1")
    tool = RecordingTool(output="fact", sources=[source])
    chat_model = FakeChatModel(
        responses=[_action("recording_tool", {"value": "q"}), _action("final_answer", {"answer": "a"})]
    )
    orchestrator = AgentOrchestrator(chat_model, [tool], max_iterations=5)

    result = orchestrator.run("question")

    assert result.sources == [source]


def test_unknown_tool_name_feeds_back_an_error_observation_without_crashing():
    chat_model = FakeChatModel(
        responses=[
            _action("nonexistent_tool", {}),
            _action("final_answer", {"answer": "recovered"}),
        ]
    )
    orchestrator = AgentOrchestrator(chat_model, [], max_iterations=5)

    result = orchestrator.run("question")

    assert result.answer == "recovered"
    assert result.tools_used == []  # unknown tool never actually "used"
    # the model saw the error and adapted — confirm via the second call's prompt
    assert "unknown tool" in chat_model.calls[1][-1]["content"].lower()


def test_tool_exception_is_caught_and_fed_back_as_an_observation():
    chat_model = FakeChatModel(
        responses=[
            _action("raising_tool", {}),
            _action("final_answer", {"answer": "handled the error"}),
        ]
    )
    orchestrator = AgentOrchestrator(chat_model, [RaisingTool()], max_iterations=5)

    result = orchestrator.run("question")

    assert result.answer == "handled the error"
    assert "boom" in chat_model.calls[1][-1]["content"]


def test_tool_timeout_returns_promptly_with_an_error_observation():
    chat_model = FakeChatModel(
        responses=[
            _action("slow_tool", {}),
            _action("final_answer", {"answer": "gave up waiting"}),
        ]
    )
    orchestrator = AgentOrchestrator(chat_model, [SlowTool()], max_iterations=5, tool_timeout=0.2)

    t0 = time.perf_counter()
    result = orchestrator.run("question")
    elapsed = time.perf_counter() - t0

    assert elapsed < 1.5  # well under SlowTool's 2s sleep — timeout, not a hang
    assert "timed out" in chat_model.calls[1][-1]["content"].lower()
    assert result.answer == "gave up waiting"


def test_max_iterations_triggers_graceful_fallback_not_a_hang():
    tool = RecordingTool()
    # Always calls the tool, never gives a final_answer.
    chat_model = FakeChatModel(responses=[_action("recording_tool", {"value": "x"})] * 10)
    orchestrator = AgentOrchestrator(chat_model, [tool], max_iterations=3)

    result = orchestrator.run("question")

    assert len(tool.calls) == 3  # never exceeds max_iterations
    assert FALLBACK_MESSAGE in result.answer


def test_unparseable_model_output_is_treated_as_the_final_answer():
    chat_model = FakeChatModel(responses=["Just a plain text answer, no JSON."])
    orchestrator = AgentOrchestrator(chat_model, [], max_iterations=5)

    result = orchestrator.run("question")

    assert result.answer == "Just a plain text answer, no JSON."
    assert result.tools_used == []


def test_reasoning_summary_never_contains_raw_model_text():
    tool = RecordingTool()
    chat_model = FakeChatModel(
        responses=[
            _action("recording_tool", {"value": "some very specific internal reasoning detail"}),
            _action("final_answer", {"answer": "final"}),
        ]
    )
    orchestrator = AgentOrchestrator(chat_model, [tool], max_iterations=5)

    result = orchestrator.run("question")

    assert "specific internal reasoning detail" not in result.reasoning_summary
    assert "action" not in result.reasoning_summary  # no leaked JSON protocol keywords
    assert result.reasoning_summary == "Used recording_tool."


def test_execution_time_is_recorded_and_positive():
    chat_model = FakeChatModel(responses=[_action("final_answer", {"answer": "x"})])
    orchestrator = AgentOrchestrator(chat_model, [], max_iterations=5)

    result = orchestrator.run("question")

    assert result.execution_time >= 0
