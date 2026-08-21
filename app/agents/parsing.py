"""Parsing the model's JSON action, and building the user-safe
reasoning summary.

`reasoning_summary` is built here from the list of tool *names* that
were actually called — never from the model's own text — which is
what guarantees no hidden chain-of-thought ever reaches the response.
"""
import json
import re
from dataclasses import dataclass

_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)
_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)

_TOOL_PHRASES = {
    "document_search_tool": "document search",
    "web_search_tool": "web search",
    "calculator_tool": "calculator",
    "document_summary_tool": "document summarization",
}


@dataclass
class AgentAction:
    name: str
    input: dict


def parse_agent_action(text: str) -> AgentAction | None:
    """Returns None if the model's response isn't a recognizable action
    (the orchestrator's fallback is to treat the raw text as the final
    answer in that case).
    """
    cleaned = _CODE_FENCE.sub("", text.strip()).strip()

    data = _try_json(cleaned)
    if data is None:
        match = _JSON_OBJECT.search(cleaned)
        if match:
            data = _try_json(match.group(0))

    if not isinstance(data, dict) or not isinstance(data.get("action"), str):
        return None
    action_input = data.get("action_input")
    if not isinstance(action_input, dict):
        action_input = {}
    return AgentAction(name=data["action"], input=action_input)


def _try_json(text: str):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return None


def build_reasoning_summary(tools_used: list[str]) -> str:
    if not tools_used:
        return "Answered directly without using any tools."
    seen = list(dict.fromkeys(tools_used))  # de-dupe, preserve order
    phrases = [_TOOL_PHRASES.get(name, name) for name in seen]
    if len(phrases) == 1:
        return f"Used {phrases[0]}."
    return f"Used {', '.join(phrases[:-1])} and {phrases[-1]}."
