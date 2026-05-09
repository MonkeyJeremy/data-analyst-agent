"""Anthropic provider — wraps the Anthropic SDK with prompt caching."""
from __future__ import annotations

import time
from typing import Any

import anthropic

from src.agent.base import AgentResponse, BaseLLMClient, TokenUsage, ToolCall
from src.config import MAX_TOKENS, MODEL_NAME


class AnthropicClient(BaseLLMClient):
    """Claude API client with prompt caching and cumulative token metering.

    The system prompt is automatically wrapped in a ``cache_control: ephemeral``
    block so repeated turns within a session pay ~10 % of full input rate.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model or MODEL_NAME
        self.usage = TokenUsage()

    # ── BaseLLMClient interface ───────────────────────────────────────────────

    def call(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> AgentResponse:
        """Call the Claude API.  Tools are already in Anthropic format."""
        system_block: list[dict] = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]

        for attempt in range(2):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=MAX_TOKENS,
                    system=system_block,       # type: ignore[arg-type]
                    messages=messages,
                    tools=tools,               # type: ignore[arg-type]
                )
                self.usage._add_anthropic(response.usage)
                return self._normalize(response)
            except anthropic.APIStatusError as exc:
                if attempt == 0 and exc.status_code in (429, 500, 502, 503, 529):
                    time.sleep(2)
                    continue
                raise
        raise RuntimeError("Unreachable")  # pragma: no cover

    def build_assistant_entry(self, response: AgentResponse) -> dict:
        return {"role": "assistant", "content": response._raw.content}

    def build_tool_result_entries(self, results: list[dict]) -> list[dict]:
        """Anthropic expects all tool results in one ``role: user`` message."""
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": r["id"],
                        "content": r["content"],
                    }
                    for r in results
                ],
            }
        ]

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _normalize(self, response: Any) -> AgentResponse:
        text = ""
        tool_calls: list[ToolCall] = []

        for block in response.content:
            if hasattr(block, "text"):
                text += block.text
            if getattr(block, "type", None) == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        input=dict(block.input),
                    )
                )

        stop = "end_turn" if response.stop_reason == "end_turn" else "tool_use"
        return AgentResponse(
            stop_reason=stop,
            text=text.strip(),
            tool_calls=tuple(tool_calls),
            _raw=response,
        )
