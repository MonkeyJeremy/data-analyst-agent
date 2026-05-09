"""Provider-agnostic types and abstract base class for all LLM clients.

Every concrete client (Anthropic, OpenAI, …) must:
  1. Translate the provider-specific API response into :class:`AgentResponse`.
  2. Build provider-specific history entries via :meth:`build_assistant_entry`
     and :meth:`build_tool_result_entries` so the ReAct loop never imports
     anything provider-specific.
  3. Convert internal (Anthropic-style ``input_schema``) tool schemas to its
     own wire format inside :meth:`call`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ── Shared value types ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    input: dict


@dataclass
class AgentResponse:
    """Normalised model response — independent of provider SDK types.

    The loop only touches this dataclass; provider details live inside
    ``_raw`` and are used exclusively by the client's history-building
    methods.
    """

    stop_reason: str                  # ``"end_turn"`` or ``"tool_use"``
    text: str                         # full assistant text (may be "" on tool_use)
    tool_calls: tuple[ToolCall, ...]  # populated when stop_reason == "tool_use"
    _raw: Any = field(default=None, repr=False)


@dataclass
class TokenUsage:
    """Cumulative token counts for one :class:`BaseLLMClient` session.

    ``cache_read_tokens`` are tokens served from the prompt cache (billed at
    ~10 % of full input rate).  ``cache_write_tokens`` are tokens written into
    the cache on the first call (billed at ~125 % of full input rate,
    amortised over later hits).
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def _add_anthropic(self, usage: Any) -> None:
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        self.cache_read_tokens += (
            getattr(usage, "cache_read_input_tokens", 0) or 0
        )
        self.cache_write_tokens += (
            getattr(usage, "cache_creation_input_tokens", 0) or 0
        )

    def _add_openai(self, usage: Any) -> None:
        self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
        # OpenAI cached token count lives in prompt_tokens_details
        details = getattr(usage, "prompt_tokens_details", None)
        if details:
            self.cache_read_tokens += getattr(details, "cached_tokens", 0) or 0


# ── Abstract base client ──────────────────────────────────────────────────────

class BaseLLMClient(ABC):
    """Contract every provider client must fulfil.

    The :class:`~src.agent.loop` only calls methods defined here, keeping
    the loop itself completely provider-agnostic.
    """

    usage: TokenUsage

    @abstractmethod
    def call(
        self,
        *,
        system: str,
        messages: list[dict],
        tools: list[dict],
    ) -> AgentResponse:
        """Send a turn and return a normalised response.

        Parameters
        ----------
        system:
            System prompt string.
        messages:
            Full conversation history in the provider's expected format.
        tools:
            Tool definitions in **internal (Anthropic-style ``input_schema``)
            format**.  The implementation is responsible for converting these
            to its own wire format before calling the API.

        Returns
        -------
        AgentResponse
            Normalised response with ``stop_reason``, ``text``, and
            ``tool_calls`` populated.
        """
        ...

    @abstractmethod
    def build_assistant_entry(self, response: AgentResponse) -> dict:
        """Return the assistant history dict to append after a model response."""
        ...

    @abstractmethod
    def build_tool_result_entries(
        self, results: list[dict]
    ) -> list[dict]:
        """Return the list of history dicts to extend with after tool execution.

        Parameters
        ----------
        results:
            ``[{"id": str, "content": str}, ...]``

        Returns
        -------
        list[dict]
            One or more message dicts.  Anthropic wraps all results in a
            single ``role: user`` message; OpenAI uses separate
            ``role: tool`` messages — hence a list rather than a single dict.
        """
        ...
