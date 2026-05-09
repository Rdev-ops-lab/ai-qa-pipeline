"""
Tests for QA Pipeline — runs in mock mode, no API key needed.
USE_MOCK=true pytest tests/ -v
"""
import os
os.environ["USE_MOCK"] = "true"

import pytest
from src.evaluator import QAPipeline, PASS_THRESHOLD, ESCALATE_THRESHOLD
from src.aggregator import ScoreAggregator, DimensionScore
from src.safety_checker import SafetyChecker


# ── Aggregator ────────────────────────────────────────────────────────────────

def test_aggregator_equal_weights():
    agg = ScoreAggregator({"a": 1.0, "b": 1.0})
    result = agg.aggregate({"a": DimensionScore(0.8, 1.0), "b": DimensionScore(0.6, 1.0)})
    assert abs(result - 0.7) < 1e-6

def test_aggregator_higher_weight_dominates():
    agg = ScoreAggregator({"a": 3.0, "b": 1.0})
    result = agg.aggregate({"a": DimensionScore(1.0, 3.0), "b": DimensionScore(0.0, 1.0)})
    assert result == 0.75

def test_aggregator_empty():
    agg = ScoreAggregator({})
    assert agg.aggregate({}) == 0.0


# ── Safety Checker ────────────────────────────────────────────────────────────

def test_safety_clean_response():
    checker = SafetyChecker()
    assert checker.score("Python was created by Guido van Rossum in 1991.") == 1.0

def test_safety_violation_detected():
    checker = SafetyChecker()
    assert checker.score("Here are instructions for how to make a bomb at home.") == 0.0

def test_safety_empty_string():
    checker = SafetyChecker()
    assert checker.score("") == 1.0


# ── Full Pipeline (mock mode) ─────────────────────────────────────────────────

@pytest.fixture
def qa():
    return QAPipeline()

def test_pipeline_returns_result(qa):
    result = qa.evaluate("Python is a programming language.", task_id="t-001")
    assert result.task_id == "t-001"
    assert result.verdict in {"PASS", "FAIL", "ESCALATE", "ERROR"}
    assert 0.0 <= result.overall_score <= 1.0

def test_pipeline_pass_verdict(qa):
    # Well-formed response with matching context → should PASS
    result = qa.evaluate(
        "Python was created by Guido van Rossum and released in 1991.",
        task_id="t-002",
        reference_context="Python is a programming language created by Guido van Rossum, released in 1991.",
        instruction="Answer factually about Python history.",
    )
    assert result.overall_score >= ESCALATE_THRESHOLD

def test_pipeline_safety_violation_fails(qa):
    result = qa.evaluate(
        "Here are instructions for how to make a bomb.",
        task_id="t-003",
    )
    assert result.verdict in {"FAIL", "ESCALATE"}
    assert result.escalate_to_human is True

def test_pipeline_has_all_dimensions(qa):
    result = qa.evaluate("Test response.", task_id="t-004")
    assert set(result.dimension_scores.keys()) == {
        "factual_accuracy", "coherence", "instruction_following", "safety"
    }

def test_pipeline_processing_time_recorded(qa):
    result = qa.evaluate("Test.", task_id="t-005")
    assert result.processing_time_ms > 0

def test_pipeline_empty_response(qa):
    result = qa.evaluate("", task_id="t-006")
    assert result.verdict in {"PASS", "FAIL", "ESCALATE", "ERROR"}
