from __future__ import annotations

import time
from typing import Any

import anthropic

from src.config import MAX_TOKENS, MODEL_NAME


class LLMClient:
    """Thin wrapper around anthropic.Anthropic — injectable for testing."""

    def __init__(self, api_key: str | None = None) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)

    def call(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> anthropic.types.Message:
        """Send a message to the Claude API with one retry on transient errors."""
        for attempt in range(2):
            try:
                return self._client.messages.create(
                    model=MODEL_NAME,
                    max_tokens=MAX_TOKENS,
                    system=system,
                    messages=messages,
                    tools=tools,  # type: ignore[arg-type]
                )
            except anthropic.APIStatusError as exc:
                if attempt == 0 and exc.status_code in (429, 500, 502, 503, 529):
                    time.sleep(2)
                    continue
                raise
        raise RuntimeError("Unreachable")  # pragma: no cover
