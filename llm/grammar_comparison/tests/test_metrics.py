"""Tests for evaluation metrics.

Mock data is constructed so that expected values are deterministic and
easy to verify by hand.
"""

from __future__ import annotations

import math

import pytest

from llm.grammar_comparison.evaluation.metrics import (
    expressibility,
    spearman_rank_correlation,
    top1_accuracy,
    weighted_log_probability,
)

# ---------------------------------------------------------------------------
# Fixtures: reusable mock data
# ---------------------------------------------------------------------------


def _agreeing_rule() -> list[dict]:
    """Rule A: grammar agrees with LLM.

    rank 1 has the highest log_prob (closest to 0), rank 3 the lowest.
    Expected Spearman rho is -1.0 (perfect negative correlation between
    rank and log_prob).
    """
    return [
        {"rule_id": "rule_a", "rank": 1, "log_prob": -2.0},
        {"rule_id": "rule_a", "rank": 2, "log_prob": -5.0},
        {"rule_id": "rule_a", "rank": 3, "log_prob": -9.0},
    ]


def _disagreeing_rule() -> list[dict]:
    """Rule B: grammar disagrees with LLM.

    rank 1 has the *lowest* log_prob; rank 3 has the highest.
    Expected Spearman rho is +1.0 (positive correlation).
    """
    return [
        {"rule_id": "rule_b", "rank": 1, "log_prob": -20.0},
        {"rule_id": "rule_b", "rank": 2, "log_prob": -10.0},
        {"rule_id": "rule_b", "rank": 3, "log_prob": -3.0},
    ]


def _mixed_data() -> list[dict]:
    """Both rules combined."""
    return _agreeing_rule() + _disagreeing_rule()


# ---------------------------------------------------------------------------
# spearman_rank_correlation
# ---------------------------------------------------------------------------


class TestSpearmanRankCorrelation:
    def test_perfect_agreement(self):
        """When grammar fully agrees, rho should be -1.0."""
        rho = spearman_rank_correlation(_agreeing_rule())
        assert rho == pytest.approx(-1.0)

    def test_perfect_disagreement(self):
        """When grammar fully disagrees, rho should be +1.0."""
        rho = spearman_rank_correlation(_disagreeing_rule())
        assert rho == pytest.approx(1.0)

    def test_average_across_rules(self):
        """Average of -1.0 and +1.0 should be 0.0."""
        rho = spearman_rank_correlation(_mixed_data())
        assert rho == pytest.approx(0.0)

    def test_empty_list(self):
        assert spearman_rank_correlation([]) == 0.0

    def test_single_hypothesis_per_rule(self):
        """Rules with < 2 hypotheses are skipped → return 0.0."""
        data = [{"rule_id": "solo", "rank": 1, "log_prob": -3.0}]
        assert spearman_rank_correlation(data) == 0.0

    def test_identical_log_probs_skipped(self):
        """All-identical log_probs give undefined correlation → skipped."""
        data = [
            {"rule_id": "flat", "rank": 1, "log_prob": -5.0},
            {"rule_id": "flat", "rank": 2, "log_prob": -5.0},
            {"rule_id": "flat", "rank": 3, "log_prob": -5.0},
        ]
        assert spearman_rank_correlation(data) == 0.0

    def test_returns_float(self):
        assert isinstance(spearman_rank_correlation(_agreeing_rule()), float)


# ---------------------------------------------------------------------------
# weighted_log_probability
# ---------------------------------------------------------------------------


class TestWeightedLogProbability:
    def test_basic_computation(self):
        """Hand-computed: 5*(-2) + 4*(-5) + 3*(-9) = -10 -20 -27 = -57."""
        result = weighted_log_probability(_agreeing_rule())
        assert result == pytest.approx(-57.0)

    def test_skips_neg_inf(self):
        """Hypotheses with -inf log_prob contribute 0."""
        data = [
            {"rule_id": "x", "rank": 1, "log_prob": -4.0},
            {"rule_id": "x", "rank": 2, "log_prob": float("-inf")},
        ]
        # Only rank 1 contributes: 5 * (-4) = -20
        assert weighted_log_probability(data) == pytest.approx(-20.0)

    def test_empty_list(self):
        assert weighted_log_probability([]) == 0.0

    def test_all_neg_inf(self):
        data = [
            {"rule_id": "x", "rank": 1, "log_prob": float("-inf")},
            {"rule_id": "x", "rank": 2, "log_prob": float("-inf")},
        ]
        assert weighted_log_probability(data) == 0.0

    def test_returns_float(self):
        assert isinstance(weighted_log_probability(_agreeing_rule()), float)


# ---------------------------------------------------------------------------
# top1_accuracy
# ---------------------------------------------------------------------------


class TestTop1Accuracy:
    def test_grammar_agrees(self):
        """Rule A: rank 1 has highest log_prob → correct."""
        assert top1_accuracy(_agreeing_rule()) == pytest.approx(1.0)

    def test_grammar_disagrees(self):
        """Rule B: rank 1 has lowest log_prob → incorrect."""
        assert top1_accuracy(_disagreeing_rule()) == pytest.approx(0.0)

    def test_mixed_rules(self):
        """One correct, one incorrect → 0.5."""
        assert top1_accuracy(_mixed_data()) == pytest.approx(0.5)

    def test_empty_list(self):
        assert top1_accuracy([]) == 0.0

    def test_returns_float(self):
        assert isinstance(top1_accuracy(_agreeing_rule()), float)

    def test_bounded_zero_one(self):
        acc = top1_accuracy(_mixed_data())
        assert 0.0 <= acc <= 1.0


# ---------------------------------------------------------------------------
# expressibility
# ---------------------------------------------------------------------------


class TestExpressibility:
    def test_all_finite(self):
        assert expressibility(_agreeing_rule()) == pytest.approx(1.0)

    def test_some_neg_inf(self):
        data = [
            {"rule_id": "x", "rank": 1, "log_prob": -3.0},
            {"rule_id": "x", "rank": 2, "log_prob": float("-inf")},
            {"rule_id": "x", "rank": 3, "log_prob": -7.0},
            {"rule_id": "x", "rank": 4, "log_prob": float("-inf")},
        ]
        # 2 finite out of 4 → 0.5
        assert expressibility(data) == pytest.approx(0.5)

    def test_all_neg_inf(self):
        data = [
            {"rule_id": "x", "rank": 1, "log_prob": float("-inf")},
            {"rule_id": "x", "rank": 2, "log_prob": float("-inf")},
        ]
        assert expressibility(data) == pytest.approx(0.0)

    def test_empty_list(self):
        assert expressibility([]) == 0.0

    def test_returns_float(self):
        assert isinstance(expressibility(_agreeing_rule()), float)

    def test_bounded_zero_one(self):
        expr = expressibility(_mixed_data())
        assert 0.0 <= expr <= 1.0
