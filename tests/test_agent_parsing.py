"""Tests for parsing the agent's JSON action and building the
user-safe reasoning summary.
"""
from app.agents.parsing import build_reasoning_summary, parse_agent_action


def test_parses_plain_json_action():
    action = parse_agent_action('{"action": "calculator_tool", "action_input": {"expression": "1+1"}}')
    assert action.name == "calculator_tool"
    assert action.input == {"expression": "1+1"}


def test_parses_json_wrapped_in_markdown_fence():
    text = '```json\n{"action": "final_answer", "action_input": {"answer": "hi"}}\n```'
    action = parse_agent_action(text)
    assert action.name == "final_answer"
    assert action.input == {"answer": "hi"}


def test_extracts_json_object_from_surrounding_text():
    text = 'Sure, here it is:\n{"action": "web_search_tool", "action_input": {"query": "news"}}\nDone.'
    action = parse_agent_action(text)
    assert action.name == "web_search_tool"


def test_missing_action_key_returns_none():
    assert parse_agent_action('{"foo": "bar"}') is None


def test_garbage_text_returns_none():
    assert parse_agent_action("I think the answer is 42.") is None


def test_missing_action_input_defaults_to_empty_dict():
    action = parse_agent_action('{"action": "final_answer"}')
    assert action.input == {}


def test_reasoning_summary_single_tool():
    assert build_reasoning_summary(["calculator_tool"]) == "Used calculator."


def test_reasoning_summary_multiple_tools_matches_example_format():
    summary = build_reasoning_summary(["document_search_tool", "calculator_tool"])
    assert summary == "Used document search and calculator."


def test_reasoning_summary_dedupes_repeated_tools():
    summary = build_reasoning_summary(["calculator_tool", "calculator_tool"])
    assert summary == "Used calculator."


def test_reasoning_summary_no_tools():
    assert build_reasoning_summary([]) == "Answered directly without using any tools."
