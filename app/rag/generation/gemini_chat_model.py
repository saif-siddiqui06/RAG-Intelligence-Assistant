"""Gemini implementation of BaseChatModel.

Converts the OpenAI-style {"role": "system"|"user"|"assistant", "content": ...}
message list this codebase builds (see prompts.py) into Gemini's shape:
a `system_instruction` config field plus a `contents` list using
"user"/"model" roles (Gemini has no "system" role on Content itself).
"""
import logging
from collections.abc import Iterator

from google import genai
from google.genai import types

from app.core.exceptions import AppException
from app.rag.generation.base import BaseChatModel

logger = logging.getLogger(__name__)

_ROLE_MAP = {"user": "user", "assistant": "model"}


def _split_messages(messages: list[dict]) -> tuple[str | None, list[dict]]:
    system_parts: list[str] = []
    contents: list[dict] = []
    for message in messages:
        role = message["role"]
        if role == "system":
            system_parts.append(message["content"])
        else:
            contents.append({"role": _ROLE_MAP[role], "parts": [{"text": message["content"]}]})
    system_instruction = "\n\n".join(system_parts) if system_parts else None
    return system_instruction, contents


class GeminiChatModel(BaseChatModel):
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise AppException(
                "GEMINI_API_KEY is not configured — set it in .env before chatting",
                status_code=500,
            )
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def complete(self, messages: list[dict], temperature: float = 0) -> str:
        system_instruction, contents = _split_messages(messages)
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction, temperature=temperature
                ),
            )
        except Exception as exc:
            logger.exception("Gemini generate_content request failed")
            raise AppException(f"Answer generation failed: {exc}", status_code=502) from exc
        return response.text or ""

    def stream(self, messages: list[dict], temperature: float = 0) -> Iterator[str]:
        system_instruction, contents = _split_messages(messages)
        try:
            for chunk in self._client.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction, temperature=temperature
                ),
            ):
                if chunk.text:
                    yield chunk.text
        except Exception as exc:
            logger.exception("Gemini generate_content_stream request failed")
            raise AppException(f"Answer generation failed: {exc}", status_code=502) from exc
