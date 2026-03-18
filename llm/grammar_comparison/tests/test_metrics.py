"""Tests for evaluation metrics.

Mock data is constructed so that expected values are deterministic and
easy to verify by hand.
"""

from __future__ import annotations

import math

import pytest

from llm.grammar_comparison.evaluation.metrics import (
    correct_rank,
    expressibility,
    rule_difficulty_correlation,
    spearman_agreement,
    spearman_rank_correlation,
    top1_accuracy,
    top1_accuracy_corrected,
    weighted_log_probability,
)

# ---------------------------------------------------------------------------
# Fixtures: reusable mock data
# ---------------------------------------------------------------------------


def _agreeing_rule() -> list[dict]:
    """Rule A: grammar agrees with LLM.

    rank 1 has the highest log_posterior (closest to 0), rank 3 the lowest.
    Raw Spearman rho is -1.0 (perfect negative correlation between rank and
    log_posterior).  After negation, spearman_agreement returns +1.0.
    """
    return [
        {"rule_id": "rule_a", "rank": 1, "log_posterior": -2.0},
        {"rule_id": "rule_a", "rank": 2, "log_posterior": -5.0},
        {"rule_id": "rule_a", "rank": 3, "log_posterior": -9.0},
    ]


def _disagreeing_rule() -> list[dict]:
    """Rule B: grammar disagrees with LLM.

    rank 1 has the *lowest* log_posterior; rank 3 has the highest.
    Raw Spearman rho is +1.0; after negation, spearman_agreement returns -1.0.
    """
    return [
        {"rule_id": "rule_b", "rank": 1, "log_posterior": -20.0},
        {"rule_id": "rule_b", "rank": 2, "log_posterior": -10.0},
        {"rule_id": "rule_b", "rank": 3, "log_posterior": -3.0},
    ]


def _mixed_data() -> list[dict]:
    """Both rules combined."""
    return _agreeing_rule() + _disagreeing_rule()


def _legacy_log_prob_data() -> list[dict]:
    """Data using the old ``log_prob`` key (backward compatibility)."""
    return [
        {"rule_id": "rule_a", "rank": 1, "log_prob": -2.0},
        {"rule_id": "rule_a", "rank": 2, "log_prob": -5.0},
        {"rule_id": "rule_a", "rank": 3, "log_prob": -9.0},
    ]


# ---------------------------------------------------------------------------
# Backward compatibility: log_prob fallback
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """All metrics should work with both log_posterior and log_prob keys."""

    def test_spearman_agreement_with_log_prob(self):
        result = spearman_agreement(_legacy_log_prob_data())
        assert result == pytest.approx(1.0)

    def test_weighted_log_probability_with_log_prob(self):
        # 5*(-2) + 4*(-5) + 3*(-9) = -10 - 20 - 27 = -57
        result = weighted_log_probability(_legacy_log_prob_data())
        assert result == pytest.approx(-57.0)

    def test_top1_accuracy_corrected_with_log_prob(self):
        result = top1_accuracy_corrected(_legacy_log_prob_data())
        # k=3, hit=1, chance=1/3, corrected = (1 - 1/3) / (1 - 1/3) = 1.0
        assert result == pytest.approx(1.0)

    def test_expressibility_with_log_prob(self):
        result = expressibility(_legacy_log_prob_data())
        assert result == pytest.approx(1.0)

    def test_log_posterior_takes_priority(self):
        """When both keys present, log_posterior wins."""
        data = [
            {"rule_id": "r", "rank": 1, "log_posterior": -1.0, "log_prob": -100.0},
            {"rule_id": "r", "rank": 2, "log_posterior": -5.0, "log_prob": -0.1},
            {"rule_id": "r", "rank": 3, "log_posterior": -9.0, "log_prob": -0.01},
        ]
        # If log_posterior is used: agreement (rank 1 has highest) → +1.0
        # If log_prob were used: disagreement → -1.0
        assert spearman_agreement(data) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# spearman_agreement
# ---------------------------------------------------------------------------


class TestSpearmanAgreement:
    def test_positive_when_grammar_agrees(self):
        """When grammar fully agrees, negated rho should be +1.0."""
        result = spearman_agreement(_agreeing_rule())
        assert result == pytest.approx(1.0)

    def test_negative_when_grammar_disagrees(self):
        """When grammar fully disagrees, negated rho should be -1.0."""
        result = spearman_agreement(_disagreeing_rule())
        assert result == pytest.approx(-1.0)

    def test_average_across_rules(self):
        """Average of +1.0 and -1.0 should be 0.0."""
        result = spearman_agreement(_mixed_data())
        assert result == pytest.approx(0.0)

    def test_empty_list(self):
        assert spearman_agreement([]) == 0.0

    def test_single_hypothesis_per_rule(self):
        """Rules with < 2 hypotheses are skipped -> return 0.0."""
        data = [{"rule_id": "solo", "rank": 1, "log_posterior": -3.0}]
        assert spearman_agreement(data) == 0.0

    def test_identical_log_posteriors_skipped(self):
        """All-identical log_posteriors give undefined correlation -> skipped."""
        data = [
            {"rule_id": "flat", "rank": 1, "log_posterior": -5.0},
            {"rule_id": "flat", "rank": 2, "log_posterior": -5.0},
            {"rule_id": "flat", "rank": 3, "log_posterior": -5.0},
        ]
        assert spearman_agreement(data) == 0.0

    def test_returns_float(self):
        assert isinstance(spearman_agreement(_agreeing_rule()), float)


# ---------------------------------------------------------------------------
# Deprecated spearman_rank_correlation alias
# ---------------------------------------------------------------------------


class TestSpearmanRankCorrelationAlias:
    def test_returns_raw_rho(self):
        """The deprecated alias returns the non-negated value."""
        assert spearman_rank_correlation(_agreeing_rule()) == pytest.approx(-1.0)

    def test_relationship_to_agreement(self):
        """Should be the negation of spearman_agreement."""
        data = _mixed_data()
        assert spearman_rank_correlation(data) == pytest.approx(
            -spearman_agreement(data)
        )


# ---------------------------------------------------------------------------
# weighted_log_probability
# ---------------------------------------------------------------------------


class TestWeightedLogProbability:
    def test_basic_computation(self):
        """Hand-computed: 5*(-2) + 4*(-5) + 3*(-9) = -10 -20 -27 = -57."""
        result = weighted_log_probability(_agreeing_rule())
        assert result == pytest.approx(-57.0)

    def test_skips_neg_inf(self):
        """Hypotheses with -inf log_posterior contribute 0."""
        data = [
            {"rule_id": "x", "rank": 1, "log_posterior": -4.0},
            {"rule_id": "x", "rank": 2, "log_posterior": float("-inf")},
        ]
        # Only rank 1 contributes: 5 * (-4) = -20
        assert weighted_log_probability(data) == pytest.approx(-20.0)

    def test_empty_list(self):
        assert weighted_log_probability([]) == 0.0

    def test_all_neg_inf(self):
        data = [
            {"rule_id": "x", "rank": 1, "log_posterior": float("-inf")},
            {"rule_id": "x", "rank": 2, "log_posterior": float("-inf")},
        ]
        assert weighted_log_probability(data) == 0.0

    def test_returns_float(self):
        assert isinstance(weighted_log_probability(_agreeing_rule()), float)


# ---------------------------------------------------------------------------
# top1_accuracy_corrected
# ---------------------------------------------------------------------------


class TestTop1AccuracyCorrected:
    def test_perfect_agreement(self):
        """Rule A: rank 1 has highest log_posterior -> corrected = 1.0.

        k=3, hit=1, chance=1/3, corrected = (1 - 1/3) / (1 - 1/3) = 1.0
        """
        assert top1_accuracy_corrected(_agreeing_rule()) == pytest.approx(1.0)

    def test_grammar_disagrees(self):
        """Rule B: rank 1 has lowest log_posterior -> miss.

        k=3, hit=0, chance=1/3, corrected = (0 - 1/3) / (1 - 1/3) = -0.5
        """
        assert top1_accuracy_corrected(_disagreeing_rule()) == pytest.approx(-0.5)

    def test_zero_when_at_chance(self):
        """If accuracy exactly equals chance, corrected should be 0.

        Two rules: one hit, one miss, both k=2.
        Per-rule corrected: hit → (1 - 0.5) / (1 - 0.5) = 1.0
                            miss → (0 - 0.5) / (1 - 0.5) = -1.0
        Average = 0.0
        """
        data = [
            # Rule where grammar agrees (rank 1 = best log_posterior)
            {"rule_id": "r1", "rank": 1, "log_posterior": -1.0},
            {"rule_id": "r1", "rank": 2, "log_posterior": -5.0},
            # Rule where grammar disagrees
            {"rule_id": "r2", "rank": 1, "log_posterior": -10.0},
            {"rule_id": "r2", "rank": 2, "log_posterior": -1.0},
        ]
        assert top1_accuracy_corrected(data) == pytest.approx(0.0)

    def test_negative_when_worse_than_chance(self):
        """All rules incorrect should give negative corrected accuracy."""
        result = top1_accuracy_corrected(_disagreeing_rule())
        assert result < 0.0

    def test_mixed_rules(self):
        """One correct (k=3), one incorrect (k=3).

        Correct: (1 - 1/3) / (1 - 1/3) = 1.0
        Incorrect: (0 - 1/3) / (1 - 1/3) = -0.5
        Average = 0.25
        """
        assert top1_accuracy_corrected(_mixed_data()) == pytest.approx(0.25)

    def test_single_hypothesis_rule(self):
        """A rule with k=1 always matches -> corrected = 0.0 (no info)."""
        data = [{"rule_id": "solo", "rank": 1, "log_posterior": -3.0}]
        assert top1_accuracy_corrected(data) == pytest.approx(0.0)

    def test_empty_list(self):
        assert top1_accuracy_corrected([]) == 0.0

    def test_returns_float(self):
        assert isinstance(top1_accuracy_corrected(_agreeing_rule()), float)


# ---------------------------------------------------------------------------
# Deprecated top1_accuracy alias
# ---------------------------------------------------------------------------


class TestTop1AccuracyAlias:
    def test_grammar_agrees(self):
        assert top1_accuracy(_agreeing_rule()) == pytest.approx(1.0)

    def test_grammar_disagrees(self):
        assert top1_accuracy(_disagreeing_rule()) == pytest.approx(0.0)

    def test_mixed_rules(self):
        assert top1_accuracy(_mixed_data()) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# expressibility
# ---------------------------------------------------------------------------


class TestExpressibility:
    def test_all_finite(self):
        assert expressibility(_agreeing_rule()) == pytest.approx(1.0)

    def test_some_neg_inf(self):
        data = [
            {"rule_id": "x", "rank": 1, "log_posterior": -3.0},
            {"rule_id": "x", "rank": 2, "log_posterior": float("-inf")},
            {"rule_id": "x", "rank": 3, "log_posterior": -7.0},
            {"rule_id": "x", "rank": 4, "log_posterior": float("-inf")},
        ]
        # 2 finite out of 4 -> 0.5
        assert expressibility(data) == pytest.approx(0.5)

    def test_all_neg_inf(self):
        data = [
            {"rule_id": "x", "rank": 1, "log_posterior": float("-inf")},
            {"rule_id": "x", "rank": 2, "log_posterior": float("-inf")},
        ]
        assert expressibility(data) == pytest.approx(0.0)

    def test_empty_list(self):
        assert expressibility([]) == 0.0

    def test_returns_float(self):
        assert isinstance(expressibility(_agreeing_rule()), float)

    def test_bounded_zero_one(self):
        expr = expressibility(_mixed_data())
        assert 0.0 <= expr <= 1.0


# ---------------------------------------------------------------------------
# correct_rank
# ---------------------------------------------------------------------------


class TestCorrectRank:
    def test_always_ranked_first(self):
        """Ground truth has highest log_posterior -> rank 1.0."""
        data = [
            {"rule_id": "r1", "rank": 1, "log_posterior": -1.0, "fingerprint": "fp_correct"},
            {"rule_id": "r1", "rank": 2, "log_posterior": -5.0, "fingerprint": "fp_other"},
            {"rule_id": "r1", "rank": 3, "log_posterior": -9.0, "fingerprint": "fp_third"},
        ]
        gt = {"r1": "fp_correct"}
        assert correct_rank(data, gt) == pytest.approx(1.0)

    def test_ranked_last(self):
        """Ground truth has lowest log_posterior -> rank 3."""
        data = [
            {"rule_id": "r1", "rank": 1, "log_posterior": -1.0, "fingerprint": "fp_a"},
            {"rule_id": "r1", "rank": 2, "log_posterior": -5.0, "fingerprint": "fp_b"},
            {"rule_id": "r1", "rank": 3, "log_posterior": -9.0, "fingerprint": "fp_correct"},
        ]
        gt = {"r1": "fp_correct"}
        assert correct_rank(data, gt) == pytest.approx(3.0)

    def test_average_across_rules(self):
        """Two rules: correct ranked 1st and 3rd -> average 2.0."""
        data = [
            # Rule 1: correct is best
            {"rule_id": "r1", "rank": 1, "log_posterior": -1.0, "fingerprint": "fp1_correct"},
            {"rule_id": "r1", "rank": 2, "log_posterior": -5.0, "fingerprint": "fp1_other"},
            {"rule_id": "r1", "rank": 3, "log_posterior": -9.0, "fingerprint": "fp1_third"},
            # Rule 2: correct is worst
            {"rule_id": "r2", "rank": 1, "log_posterior": -2.0, "fingerprint": "fp2_a"},
            {"rule_id": "r2", "rank": 2, "log_posterior": -4.0, "fingerprint": "fp2_b"},
            {"rule_id": "r2", "rank": 3, "log_posterior": -8.0, "fingerprint": "fp2_correct"},
        ]
        gt = {"r1": "fp1_correct", "r2": "fp2_correct"}
        assert correct_rank(data, gt) == pytest.approx(2.0)

    def test_skips_rule_without_gt(self):
        """Rules not in ground_truth_fingerprints are skipped."""
        data = [
            {"rule_id": "r1", "rank": 1, "log_posterior": -1.0, "fingerprint": "fp"},
            {"rule_id": "r1", "rank": 2, "log_posterior": -5.0, "fingerprint": "fp2"},
        ]
        gt = {"r_other": "fp"}
        assert correct_rank(data, gt) == float("inf")

    def test_skips_rule_without_fingerprint_field(self):
        """Rules where no hypothesis has fingerprint are skipped."""
        data = [
            {"rule_id": "r1", "rank": 1, "log_posterior": -1.0},
            {"rule_id": "r1", "rank": 2, "log_posterior": -5.0},
        ]
        gt = {"r1": "fp"}
        assert correct_rank(data, gt) == float("inf")

    def test_skips_when_gt_fingerprint_not_found(self):
        """If GT fingerprint doesn't match any hypothesis, skip rule."""
        data = [
            {"rule_id": "r1", "rank": 1, "log_posterior": -1.0, "fingerprint": "fp_a"},
            {"rule_id": "r1", "rank": 2, "log_posterior": -5.0, "fingerprint": "fp_b"},
        ]
        gt = {"r1": "fp_nonexistent"}
        assert correct_rank(data, gt) == float("inf")

    def test_empty_scored(self):
        assert correct_rank([], {"r1": "fp"}) == float("inf")

    def test_backward_compat_log_prob(self):
        """Works with log_prob fallback."""
        data = [
            {"rule_id": "r1", "rank": 1, "log_prob": -1.0, "fingerprint": "fp_correct"},
            {"rule_id": "r1", "rank": 2, "log_prob": -5.0, "fingerprint": "fp_other"},
        ]
        gt = {"r1": "fp_correct"}
        assert correct_rank(data, gt) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# rule_difficulty_correlation
# ---------------------------------------------------------------------------


class TestRuleDifficultyCorrelation:
    def test_positive_when_expensive_rules_harder(self):
        """Harder rules (group 3) have more negative log_prior -> positive."""
        data = [
            {"rule_id": "easy", "rank": 1, "log_prior": -5.0, "log_posterior": -10.0},
            {"rule_id": "medium", "rank": 1, "log_prior": -15.0, "log_posterior": -20.0},
            {"rule_id": "hard", "rank": 1, "log_prior": -30.0, "log_posterior": -40.0},
        ]
        groups = {"easy": 1, "medium": 2, "hard": 3}
        result = rule_difficulty_correlation(data, groups)
        # log_prior: [-5, -15, -30], groups: [1, 2, 3]
        # Raw Spearman: negative (as log_prior decreases, group increases)
        # Negated: positive
        assert result > 0.0
        assert result == pytest.approx(1.0)  # perfect monotonic

    def test_negative_when_cheap_rules_harder(self):
        """If cheaper rules are marked harder, correlation should be negative."""
        data = [
            {"rule_id": "easy", "rank": 1, "log_prior": -30.0, "log_posterior": -40.0},
            {"rule_id": "medium", "rank": 1, "log_prior": -15.0, "log_posterior": -20.0},
            {"rule_id": "hard", "rank": 1, "log_prior": -5.0, "log_posterior": -10.0},
        ]
        groups = {"easy": 1, "medium": 2, "hard": 3}
        result = rule_difficulty_correlation(data, groups)
        assert result < 0.0
        assert result == pytest.approx(-1.0)

    def test_returns_zero_for_single_rule(self):
        """Need at least 2 rules."""
        data = [{"rule_id": "r1", "rank": 1, "log_prior": -5.0, "log_posterior": -10.0}]
        groups = {"r1": 1}
        assert rule_difficulty_correlation(data, groups) == 0.0

    def test_returns_zero_for_empty(self):
        assert rule_difficulty_correlation([], {}) == 0.0

    def test_skips_rules_not_in_groups(self):
        """Rules not in rule_groups are excluded."""
        data = [
            {"rule_id": "r1", "rank": 1, "log_prior": -5.0, "log_posterior": -10.0},
            {"rule_id": "r2", "rank": 1, "log_prior": -15.0, "log_posterior": -20.0},
            {"rule_id": "unknown", "rank": 1, "log_prior": -25.0, "log_posterior": -30.0},
        ]
        groups = {"r1": 1, "r2": 2}
        # Only r1 and r2 used: log_prior [-5, -15], groups [1, 2] -> positive
        result = rule_difficulty_correlation(data, groups)
        assert result > 0.0

    def test_skips_hypotheses_without_log_prior(self):
        """If rank-1 hypothesis lacks log_prior, skip that rule."""
        data = [
            {"rule_id": "r1", "rank": 1, "log_posterior": -10.0},  # no log_prior
            {"rule_id": "r2", "rank": 1, "log_prior": -15.0, "log_posterior": -20.0},
        ]
        groups = {"r1": 1, "r2": 2}
        # Only r2 usable -> fewer than 2 -> return 0.0
        assert rule_difficulty_correlation(data, groups) == 0.0

    def test_identical_log_priors_returns_zero(self):
        """All-identical log_priors -> no variance -> 0.0."""
        data = [
            {"rule_id": "r1", "rank": 1, "log_prior": -10.0, "log_posterior": -15.0},
            {"rule_id": "r2", "rank": 1, "log_prior": -10.0, "log_posterior": -20.0},
        ]
        groups = {"r1": 1, "r2": 2}
        assert rule_difficulty_correlation(data, groups) == 0.0
