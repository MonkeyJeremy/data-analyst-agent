from __future__ import annotations

import time
from dataclasses import dataclass, field

import anthropic

from src.config import MAX_TOKENS, MODEL_NAME


@dataclass
class TokenUsage:
    """Cumulative token counts for a single LLMClient session.

    ``cache_read`` counts tokens served from the prompt cache (billed at ~10%
    of normal input rate).  ``cache_write`` counts tokens written into the
    cache on the first call (billed at ~125% of normal input rate, amortised
    over subsequent cache hits).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def _add(self, response_usage: object) -> None:
        """Accumulate usage from a Message.usage object (mutates in place)."""
        self.input_tokens += getattr(response_usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(response_usage, "output_tokens", 0) or 0
        self.cache_read_tokens += (
            getattr(response_usage, "cache_read_input_tokens", 0) or 0
        )
        self.cache_write_tokens += (
            getattr(response_usage, "cache_creation_input_tokens", 0) or 0
        )


class LLMClient:
    """Thin wrapper around anthropic.Anthropic — injectable for testing.

    Enables prompt caching on the system prompt (``cache_control: ephemeral``)
    so the static schema + EDA context is served from cache on every turn after
    the first, reducing input-token billing by ~80–90 %.

    Accumulates per-session token usage in :attr:`usage`.
    """

    def __init__(self, api_key: str | None = None) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self.usage: TokenUsage = TokenUsage()

    def call(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> anthropic.types.Message:
        """Send a message to the Claude API with one retry on transient errors.

        The *system* string is wrapped in a cached content block so the token
        budget for repeated schema injections is charged at the cache-read rate
        (≈ 10% of full input rate) after the first call.
        """
        # Wrap system prompt in a cached block for prompt caching
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
                    model=MODEL_NAME,
                    max_tokens=MAX_TOKENS,
                    system=system_block,  # type: ignore[arg-type]
                    messages=messages,
                    tools=tools,  # type: ignore[arg-type]
                )
                self.usage._add(response.usage)
                return response
            except anthropic.APIStatusError as exc:
                if attempt == 0 and exc.status_code in (429, 500, 502, 503, 529):
                    time.sleep(2)
                    continue
                raise
        raise RuntimeError("Unreachable")  # pragma: no cover
