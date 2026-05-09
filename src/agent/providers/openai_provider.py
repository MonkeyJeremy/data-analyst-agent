"""OpenAI provider — wraps the OpenAI SDK, converts tool schemas on the fly."""
from __future__ import annotations

import json
import time
from typing import Any

from src.agent.base import AgentResponse, BaseLLMClient, TokenUsage, ToolCall

_DEFAULT_MODEL = "gpt-4o"
_MAX_TOKENS = 4096


class OpenAIClient(BaseLLMClient):
    """OpenAI chat-completions client with tool-use support.

    Tool schemas are stored internally in Anthropic-style ``input_schema``
    format and converted to OpenAI's ``function`` wrapper format inside
    :meth:`call`.

    History format differences vs. Anthropic:
    - System prompt is a regular ``{"role": "system", ...}`` message.
    - Assistant tool-call turns use ``tool_calls`` key.
    - Tool results are individual ``{"role": "tool", ...}`` messages, not
      wrapped in a single user message.

    Raises
    ------
    ImportError
        If the ``openai`` package is not installed.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI as _OpenAI  # lazy import — optional dep

            self._client = _OpenAI(api_key=api_key)
        except ImportError as exc:
            raise ImportError(
                "OpenAI provider requires the 'openai' package.\n"
                "Install with:  pip install openai"
            ) from exc

        self._model = model or _DEFAULT_MODEL
        self.usage = TokenUsage()

    # ── BaseLLMClient interface ───────────────────────────────────────────────

    def call(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> AgentResponse:
        """Call the OpenAI chat-completions API.

        The system prompt is prepended as a ``role: system`` message.
        Internal tool schemas (``input_schema`` format) are converted to
        OpenAI's ``type: function`` wrapper format before the call.
        """
        full_messages = [{"role": "system", "content": system}] + messages
        provider_tools = [self._convert_tool(t) for t in tools] if tools else None

        try:
            from openai import APIStatusError
        except ImportError:  # pragma: no cover
            APIStatusError = Exception  # type: ignore[misc,assignment]

        for attempt in range(2):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=_MAX_TOKENS,
                    messages=full_messages,   # type: ignore[arg-type]
                    tools=provider_tools,     # type: ignore[arg-type]
                )
                self.usage._add_openai(response.usage)
                return self._normalize(response)
            except APIStatusError as exc:
                if attempt == 0 and getattr(exc, "status_code", 0) in (
                    429, 500, 502, 503, 529
                ):
                    time.sleep(2)
                    continue
                raise
        raise RuntimeError("Unreachable")  # pragma: no cover

    def build_assistant_entry(self, response: AgentResponse) -> dict:
        """Build the assistant history entry from the raw OpenAI message."""
        message = response._raw.choices[0].message
        entry: dict = {
            "role": "assistant",
            "content": message.content,
        }
        if message.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        return entry

    def build_tool_result_entries(self, results: list[dict]) -> list[dict]:
        """OpenAI expects one ``role: tool`` message per tool result."""
        return [
            {
                "role": "tool",
                "tool_call_id": r["id"],
                "content": r["content"],
            }
            for r in results
        ]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _normalize(self, response: Any) -> AgentResponse:
        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        text = message.content or ""
        tool_calls: list[ToolCall] = []

        if message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append(
                    ToolCall(
                        id=tc.id,
                        name=tc.function.name,
                        input=json.loads(tc.function.arguments),
                    )
                )

        stop = "tool_use" if finish_reason == "tool_calls" else "end_turn"
        return AgentResponse(
            stop_reason=stop,
            text=text.strip(),
            tool_calls=tuple(tool_calls),
            _raw=response,
        )

    @staticmethod
    def _convert_tool(tool: dict) -> dict:
        """Convert Anthropic-style ``input_schema`` to OpenAI ``function`` wrapper."""
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": tool.get(
                    "input_schema",
                    {"type": "object", "properties": {}},
                ),
            },
        }
