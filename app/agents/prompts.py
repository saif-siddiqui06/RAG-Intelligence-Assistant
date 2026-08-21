"""The agent's system prompt: a JSON-action ("ReAct"-style) protocol
built on top of the existing BaseChatModel interface — no provider-
specific function-calling API needed, so the agent works with any
BaseChatModel implementation, matching this project's "swap providers
without rewriting" pattern.

Kept in its own file, same reasoning as app.rag.generation.prompts:
what's actually sent to the LLM should be easy to audit in one place.
"""
from app.agents.tools.base import BaseTool

FINAL_ANSWER_ACTION = "final_answer"

_SYSTEM_TEMPLATE = """\
You are an assistant that answers questions by reasoning step by step \
and calling tools when you genuinely need information you don't \
already have. Never call a tool just because it exists — if you can \
answer directly and confidently, do that immediately.

Available tools:
{tool_descriptions}

Respond with ONLY a single JSON object and nothing else — no markdown \
fences, no explanation outside the JSON. It must match exactly one of \
these shapes:
{{"action": "<tool_name>", "action_input": {{...}}}}
{{"action": "final_answer", "action_input": {{"answer": "<your complete final answer>"}}}}

Rules:
1. You may call multiple tools, one at a time, using each observation \
to decide your next step — this is how you handle questions that need \
more than one tool (e.g. a document fact AND a calculation).
2. Never fabricate what a tool returned — always wait for the real \
observation.
3. If a tool's observation contains an error, either try a different \
approach or acknowledge the limitation in your final answer — never \
pretend it worked.
4. Base any claim about "my document(s)"/"my paper" only on \
document_search_tool or document_summary_tool observations, never on \
outside knowledge.
5. When you have everything you need, respond with the final_answer \
action, combining everything you learned into one coherent answer with \
no mention of tools, steps, or reasoning — just the answer itself.
"""


def build_system_prompt(tools: list[BaseTool]) -> str:
    blocks = []
    for tool in tools:
        params = ", ".join(f"{name} ({desc})" for name, desc in tool.parameters.items())
        blocks.append(f"- {tool.name}: {tool.description}\n  Parameters: {params}")
    return _SYSTEM_TEMPLATE.format(tool_descriptions="\n".join(blocks))


def build_observation_message(tool_name: str, observation: str) -> str:
    return f"Observation from {tool_name}:\n{observation}"
