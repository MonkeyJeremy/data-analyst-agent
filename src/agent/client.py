"""LLM client factory and backward-compatible re-exports.

Import surface kept stable so existing code and tests that do
``from src.agent.client import LLMClient, TokenUsage`` continue to work.
"""
from __future__ import annotations

from src.agent.base import AgentResponse, BaseLLMClient, TokenUsage, ToolCall
from src.agent.providers.anthropic_provider import AnthropicClient
from src.agent.providers.openai_provider import OpenAIClient

# ── Backward-compatible alias ─────────────────────────────────────────────────
# Code written before multi-provider support used ``LLMClient`` directly.
LLMClient = AnthropicClient

__all__ = [
    "LLMClient",
    "AnthropicClient",
    "OpenAIClient",
    "BaseLLMClient",
    "TokenUsage",
    "ToolCall",
    "AgentResponse",
    "create_client",
]


def create_client(
    provider: str,
    api_key: str | None,
    model: str | None = None,
) -> BaseLLMClient:
    """Instantiate the appropriate provider client.

    Parameters
    ----------
    provider:
        Display name of the provider — must match a key in
        :data:`src.config.PROVIDERS` (e.g. ``"Anthropic"``, ``"OpenAI"``).
    api_key:
        Provider API key.  ``None`` falls back to the relevant environment
        variable (``ANTHROPIC_API_KEY`` or ``OPENAI_API_KEY``).
    model:
        Model name override.  ``None`` uses the provider's default model.

    Raises
    ------
    ValueError
        If *provider* is not recognised.
    ImportError
        If the provider's SDK package is not installed.
    """
    if provider == "Anthropic":
        return AnthropicClient(api_key=api_key, model=model)
    if provider == "OpenAI":
        return OpenAIClient(api_key=api_key, model=model)
    raise ValueError(
        f"Unknown provider: '{provider}'. Supported providers: Anthropic, OpenAI"
    )
