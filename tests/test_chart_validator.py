"""Tests for src/execution/chart_validator.py."""
from __future__ import annotations

import json

import plotly.graph_objects as go
import plotly.io as pio
import pytest

from src.execution.chart_validator import ValidationResult, validate_figures


def _fig_to_json(fig: go.Figure) -> str:
    return pio.to_json(fig)


def _bar_fig(n_cats: int, title: str = "Sales by Region") -> str:
    fig = go.Figure(
        go.Bar(
            x=[f"Cat{i}" for i in range(n_cats)],
            y=list(range(n_cats)),
        )
    )
    fig.update_layout(title=title, xaxis_title="Category", yaxis_title="Value")
    return _fig_to_json(fig)


def _pie_fig(n_slices: int) -> str:
    fig = go.Figure(
        go.Pie(
            labels=[f"Slice{i}" for i in range(n_slices)],
            values=list(range(1, n_slices + 1)),
        )
    )
    fig.update_layout(title="Market Share")
    return _fig_to_json(fig)


def _empty_fig() -> str:
    return _fig_to_json(go.Figure())


def test_valid_bar_chart_passes():
    result = validate_figures((_bar_fig(10),))
    assert result.valid
    assert result.issues == []
    assert result.correction_prompt == ""


def test_valid_pie_chart_passes():
    assert validate_figures((_pie_fig(5),)).valid


def test_empty_figure_is_invalid():
    result = validate_figures((_empty_fig(),))
    assert not result.valid
    assert any("empty" in issue for issue in result.issues)


def test_too_many_categories_detected():
    result = validate_figures((_bar_fig(35),))
    assert not result.valid
    assert any("35 categories" in issue for issue in result.issues)


def test_pie_too_many_slices_detected():
    result = validate_figures((_pie_fig(12),))
    assert not result.valid
    assert any("12 slices" in issue for issue in result.issues)


def test_missing_axis_labels_detected():
    fig = go.Figure(go.Bar(x=["A", "B"], y=[1, 2]))
    fig.update_layout(title="My Chart")
    result = validate_figures((_fig_to_json(fig),))
    assert not result.valid
    issues_text = " ".join(result.issues)
    assert "x-axis" in issues_text or "y-axis" in issues_text


def test_missing_title_detected():
    fig = go.Figure(go.Bar(x=["A", "B"], y=[1, 2]))
    fig.update_layout(xaxis_title="X", yaxis_title="Y")
    result = validate_figures((_fig_to_json(fig),))
    assert not result.valid
    assert any("title" in issue for issue in result.issues)


def test_correction_prompt_present_when_invalid():
    result = validate_figures((_empty_fig(),))
    assert not result.valid
    assert len(result.correction_prompt) > 0


def test_multiple_figures_combines_issues():
    result = validate_figures((_bar_fig(5), _empty_fig()))
    assert not result.valid


def test_empty_tuple_is_valid():
    assert validate_figures(()).valid


def test_invalid_json_skipped():
    assert validate_figures(("not json",)).valid
