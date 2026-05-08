"""Tests for src/text/analyzer.py — nested-Claude text labelling."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.text.analyzer import analyze_text_batch, _MAX_TEXTS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_fake_client(labels: list[str] | None = None, raw_override: str | None = None):
    """Return a minimal fake LLMClient whose call() returns structured JSON."""

    def _call(*, system: str, messages: list[dict], tools: list[dict]):
        if raw_override is not None:
            text = raw_override
        else:
            count = len(messages[0]["content"].splitlines()) if messages else 0
            # Build a JSON array matching the texts count extracted from prompt
            _labels = labels or ["positive"] * 50
            results = [
                {"index": i + 1, "label": _labels[i % len(_labels)],
                 "confidence": "high", "note": "looks fine"}
                for i in range(count)
            ]
            text = json.dumps(results)
        block = SimpleNamespace(text=text)
        return SimpleNamespace(content=[block])

    return SimpleNamespace(call=_call)


def _make_client_for_n(n: int, label: str = "positive"):
    """Return a fake client that produces exactly *n* results."""

    def _call(*, system, messages, tools):
        results = [
            {"index": i + 1, "label": label, "confidence": "high", "note": "ok"}
            for i in range(n)
        ]
        block = SimpleNamespace(text=json.dumps(results))
        return SimpleNamespace(content=[block])

    return SimpleNamespace(call=_call)


# ── analyze_text_batch ────────────────────────────────────────────────────────

def test_analyze_text_returns_markdown_table():
    """Valid JSON response → markdown table in summary, no error."""
    texts = ["Great product!", "Terrible service.", "Average quality."]
    client = _make_client_for_n(3, "positive")
    result = analyze_text_batch(client, texts, "sentiment")

    assert result.error is None
    assert "| # |" in result.summary
    assert "Label" in result.summary
    assert "Confidence" in result.summary
    assert "positive" in result.summary
    assert "sentiment" in result.summary


def test_analyze_text_caps_at_50():
    """100 texts passed → only _MAX_TEXTS (50) sent to the inner Claude call."""
    texts = [f"text number {i}" for i in range(100)]
    sent_counts: list[int] = []

    def _call(*, system, messages, tools):
        # Count numbered lines in the prompt to infer how many texts were sent
        prompt = messages[0]["content"]
        sent = sum(1 for line in prompt.splitlines() if line[:3].strip().rstrip(".").isdigit())
        sent_counts.append(sent)
        results = [
            {"index": i + 1, "label": "neutral", "confidence": "low", "note": "ok"}
            for i in range(sent)
        ]
        block = SimpleNamespace(text=json.dumps(results))
        return SimpleNamespace(content=[block])

    client = SimpleNamespace(call=_call)
    result = analyze_text_batch(client, texts, "topic")

    assert result.error is None
    assert sent_counts[0] == _MAX_TEXTS
    assert f"capped at {_MAX_TEXTS}" in result.summary


def test_analyze_text_empty_list():
    """Empty text list → error ExecutionResult, no crash."""
    client = SimpleNamespace(call=lambda **kw: None)  # should never be called
    result = analyze_text_batch(client, [], "sentiment")

    assert result.error is not None
    assert result.error  # non-empty error message
    assert "ERROR" in result.summary


def test_analyze_text_malformed_json():
    """Client returns bad JSON → error in ExecutionResult, no crash."""
    client = _make_fake_client(raw_override="this is not json {{{{")
    result = analyze_text_batch(client, ["hello", "world"], "sentiment")

    assert result.error is not None
    assert "ERROR" in result.summary


def test_analyze_text_strips_markdown_fences():
    """JSON wrapped in ```json fences is still parsed correctly."""
    texts = ["Nice item"]
    raw = '```json\n[{"index": 1, "label": "positive", "confidence": "high", "note": "good"}]\n```'
    client = _make_fake_client(raw_override=raw)
    result = analyze_text_batch(client, texts, "sentiment")

    assert result.error is None
    assert "positive" in result.summary


def test_analyze_text_single_text():
    """Single text returns a one-row table."""
    texts = ["Absolutely love this!"]
    client = _make_client_for_n(1, "positive")
    result = analyze_text_batch(client, texts, "sentiment")

    assert result.error is None
    # Exactly one data row (header + sep = 2 non-data lines)
    lines = [l for l in result.summary.splitlines() if l.startswith("|")]
    assert len(lines) == 3  # header, separator, 1 data row


# ── dispatch routing for analyze_text ────────────────────────────────────────

def test_dispatch_routes_analyze_text():
    """dispatch_tool('analyze_text', ..., client=fake) routes to analyze_text_batch."""
    from src.agent.tools import dispatch_tool

    texts = ["good product", "bad service"]
    client = _make_client_for_n(2)
    result = dispatch_tool(
        "analyze_text",
        {"texts": texts, "task": "sentiment", "purpose": "test"},
        client=client,
    )
    assert result.error is None
    assert "| # |" in result.summary


def test_dispatch_analyze_text_no_client():
    """dispatch_tool('analyze_text', ..., client=None) → error, no crash."""
    from src.agent.tools import dispatch_tool

    result = dispatch_tool(
        "analyze_text",
        {"texts": ["hello"], "task": "sentiment", "purpose": "test"},
        client=None,
    )
    assert result.error is not None
    assert "ERROR" in result.summary
