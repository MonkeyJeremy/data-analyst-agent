"""Shared fixtures and test doubles."""
from __future__ import annotations

import copy
import pathlib
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import openpyxl
import pandas as pd
import pytest

from src.agent.client import TokenUsage
from src.data.schema import describe_schema

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session", autouse=True)
def sales_xlsx() -> pathlib.Path:
    """Generate a synthetic sales.xlsx fixture once per test session."""
    path = FIXTURES_DIR / "sales.xlsx"
    if path.exists():
        return path

    import datetime
    import random

    random.seed(42)
    products = ["Widget A", "Widget B", "Widget C"]
    regions = ["North", "South", "East", "West"]
    rows = []
    base = datetime.date(2024, 1, 1)
    for i in range(20):
        rows.append(
            {
                "Date": base + datetime.timedelta(days=i * 7),
                "Product": products[i % len(products)],
                "Region": regions[i % len(regions)],
                "Revenue": round(random.uniform(500, 5000), 2),
                "Units": random.randint(1, 50),
            }
        )

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sales"
    headers = ["Date", "Product", "Region", "Revenue", "Units"]
    ws.append(headers)
    for row in rows:
        ws.append([row[h] for h in headers])
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    wb.save(path)
    return path


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
        self.usage: TokenUsage = TokenUsage()  # mirrors LLMClient interface

    def call(self, *, system: str, messages: list[dict], tools: list[dict]) -> FakeMessage:
        # deepcopy so mutations after this call don't corrupt recorded history
        self.calls.append({"system": system, "messages": copy.deepcopy(messages), "tools": tools})
        if not self._queue:
            raise RuntimeError("FakeLLMClient ran out of responses")
        return self._queue.popleft()
