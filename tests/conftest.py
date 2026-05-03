"""Shared fixtures and test doubles."""
from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import pytest

from src.data.schema import describe_schema


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["Alice", "Bob", "Carol", "Dave"],
            "age": [30, 25, 35, 28],
            "salary": [70000.0, 55000.0, 85000.0, 62000.0],
            "department": ["Eng", "Mkt", "Eng", "Mkt"],
        }
    )


@pytest.fixture
def sample_schema(sample_df):
    return describe_schema(sample_df)


# ── Fake Anthropic SDK types ──────────────────────────────────────────────────

@dataclass
class FakeTextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class FakeToolUseBlock:
    type: str = "tool_use"
    id: str = "tu_001"
    name: str = "execute_python"
    input: dict = field(default_factory=dict)


@dataclass
class FakeMessage:
    stop_reason: str
    content: list


# ── FakeLLMClient ─────────────────────────────────────────────────────────────

class FakeLLMClient:
    """Deterministic stand-in for LLMClient.

    Pass a list of FakeMessage objects; each call() pops the next one.
    """

    def __init__(self, responses: list[FakeMessage]) -> None:
        self._queue: deque[FakeMessage] = deque(responses)
        self.calls: list[dict] = []  # recorded for assertions

    def call(self, *, system: str, messages: list[dict], tools: list[dict]) -> FakeMessage:
        # deepcopy so mutations after this call don't corrupt recorded history
        self.calls.append({"system": system, "messages": copy.deepcopy(messages), "tools": tools})
        if not self._queue:
            raise RuntimeError("FakeLLMClient ran out of responses")
        return self._queue.popleft()
