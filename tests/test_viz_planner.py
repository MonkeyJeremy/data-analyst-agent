"""Tests for src/agent/viz_planner.py."""
from __future__ import annotations

import pytest

from src.agent.viz_planner import VizPlan, build_viz_hint, plan_visualization


@pytest.mark.parametrize("question,expected_intent", [
    ("Show sales over time by month", "trend"),
    ("What is the revenue trend for 2024?", "trend"),
    ("Plot monthly growth", "trend"),
    ("What is the distribution of age?", "distribution"),
    ("Show the spread of salaries", "distribution"),
    ("Are there any outliers in the price column?", "distribution"),
    ("What is the relationship between height and weight?", "relationship"),
    ("Correlation between salary and experience", "relationship"),
    ("Scatter plot of revenue vs cost", "relationship"),
    ("Top 10 customers by revenue", "ranking"),
    ("Which products have the highest sales?", "ranking"),
    ("Show the bottom 5 performers", "ranking"),
    ("Breakdown of sales by category", "composition"),
    ("What proportion of orders are returned?", "composition"),
    ("Market share by region", "composition"),
    ("Show the correlation matrix", "correlation"),
    ("Create a heatmap of column correlations", "correlation"),
    ("Compare the sales performance across regions", "comparison"),
    ("Difference between male and female scores", "comparison"),
])
def test_intent_detected(question, expected_intent):
    plan = plan_visualization(question)
    assert plan.intent == expected_intent, (
        f"Expected '{expected_intent}' for: '{question}', got '{plan.intent}'"
    )


@pytest.mark.parametrize("question", [
    "What is the mean of the price column?",
    "How many rows are there?",
    "List all unique values in the status column",
    "What does this dataset contain?",
])
def test_no_intent_for_non_viz_questions(question):
    plan = plan_visualization(question)
    assert plan.intent is None
    assert plan.recommended_chart is None


def test_chart_output_type_for_plot_questions():
    assert plan_visualization("Plot the distribution of age").output_type == "chart"


def test_both_output_type_for_ambiguous_questions():
    assert plan_visualization("Show distribution of salaries").output_type in ("both", "chart")


def test_plan_is_frozen():
    plan = plan_visualization("trend over time")
    with pytest.raises((AttributeError, TypeError)):
        plan.intent = "something"  # type: ignore[misc]


def test_recommended_chart_present_when_intent_detected():
    plan = plan_visualization("Show the trend over time")
    assert plan.intent is not None
    assert plan.recommended_chart is not None
    assert len(plan.recommended_chart) > 0


def test_hint_empty_when_no_intent():
    plan = plan_visualization("What is the mean?")
    assert build_viz_hint(plan) == ""


def test_hint_contains_intent_and_chart():
    plan = plan_visualization("Show trend over time")
    hint = build_viz_hint(plan)
    assert "trend" in hint
    assert "line chart" in hint
    assert "VISUALIZATION HINT" in hint
