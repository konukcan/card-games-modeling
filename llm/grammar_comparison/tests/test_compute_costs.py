"""
Tests for the log-probability scorer (compute_costs module).

Verifies that:
  - score_hypothesis returns negative floats for valid programs
  - Simpler programs get higher (less negative) scores than complex ones
  - Inexpressible programs return -inf
  - score_all_hypotheses returns dicts with the correct fields
  - Scores differ between grammar/cost combinations

Written TDD-style: tests define expected behaviour before implementation.
"""

import math
import sys
from pathlib import Path
from unittest.mock import patch

# Allow importing from the main src/ tree
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import pytest

from llm.grammar_comparison.evaluation.compute_costs import (
    score_hypothesis,
    score_all_hypotheses,
    parse_hypothesis,
    score_program,
    REQUEST_TYPE,
)
from llm.grammar_comparison.grammars.grammar_factory import (
    build_grammar,
    CostStructure,
)
from dreamcoder_core.type_system import HAND, BOOL, arrow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_uniform_grammar():
    """The base grammar with uniform costs — simplest combination."""
    return build_grammar("base", CostStructure.UNIFORM)


@pytest.fixture
def base_tiered_grammar():
    """The base grammar with tiered costs."""
    return build_grammar("base", CostStructure.TIERED)


@pytest.fixture
def minimal_uniform_grammar():
    """The minimal grammar with uniform costs — fewer primitives."""
    return build_grammar("minimal", CostStructure.UNIFORM)


# ---------------------------------------------------------------------------
# Request type
# ---------------------------------------------------------------------------

class TestRequestType:
    """Verify the request type constant is correctly constructed."""

    def test_request_type_is_hand_to_bool(self):
        """REQUEST_TYPE should be arrow(HAND, BOOL) = list(card) -> bool."""
        expected = arrow(HAND, BOOL)
        assert REQUEST_TYPE == expected

    def test_request_type_string(self):
        """Sanity check on the string representation."""
        assert "list(card)" in str(REQUEST_TYPE)
        assert "bool" in str(REQUEST_TYPE)


# ---------------------------------------------------------------------------
# score_hypothesis — basic scoring
# ---------------------------------------------------------------------------

class TestScoreHypothesisBasic:
    """Test that score_hypothesis returns sensible values for valid programs."""

    def test_valid_program_returns_negative_float(self, base_uniform_grammar):
        """A valid, expressible program should get a finite negative score.

        We use a simple program: (λ all (λ eq (get_suit $0) CLUBS) $0)
        which means 'all cards are clubs'. This uses only base primitives.
        """
        sexpr = "(λ all (λ eq (get_suit $0) CLUBS) $0)"
        ll = score_hypothesis(sexpr, base_uniform_grammar)

        # Should be a finite negative number (log-probability < 0)
        assert isinstance(ll, float)
        assert ll < 0
        assert math.isfinite(ll)

    def test_another_valid_program(self, base_uniform_grammar):
        """Test with a different valid program for robustness.

        (λ gt (length $0) 3) means 'hand has more than 3 cards'.
        """
        sexpr = "(λ gt (length $0) 3)"
        ll = score_hypothesis(sexpr, base_uniform_grammar)

        assert isinstance(ll, float)
        assert ll < 0
        assert math.isfinite(ll)


# ---------------------------------------------------------------------------
# score_hypothesis — inexpressible / invalid programs
# ---------------------------------------------------------------------------

class TestScoreHypothesisInexpressible:
    """Test that inexpressible or invalid programs return -inf."""

    def test_parse_error_returns_neg_inf(self, base_uniform_grammar):
        """A syntactically invalid s-expression should return -inf."""
        sexpr = "(((broken syntax"
        ll = score_hypothesis(sexpr, base_uniform_grammar)
        assert ll == float('-inf')

    def test_unknown_primitive_returns_neg_inf(self, base_uniform_grammar):
        """A program using a primitive not in the registry returns -inf."""
        sexpr = "(λ nonexistent_primitive $0)"
        ll = score_hypothesis(sexpr, base_uniform_grammar)
        assert ll == float('-inf')

    def test_empty_string_returns_neg_inf(self, base_uniform_grammar):
        """An empty string should return -inf."""
        ll = score_hypothesis("", base_uniform_grammar)
        assert ll == float('-inf')

    def test_primitive_not_in_grammar(self, minimal_uniform_grammar):
        """A program using a primitive missing from the minimal grammar.

        The minimal grammar doesn't include 'count_suit', so a program
        using it should be inexpressible (return -inf).
        """
        # count_suit is NOT in _MINIMAL_KEEP
        sexpr = "(λ gt (count_suit HEARTS $0) 2)"
        ll = score_hypothesis(sexpr, minimal_uniform_grammar)
        assert ll == float('-inf')


# ---------------------------------------------------------------------------
# score_hypothesis — complexity comparison
# ---------------------------------------------------------------------------

class TestScoreHypothesisComplexity:
    """Test that simpler programs score higher than complex ones.

    Under any reasonable grammar, a program using fewer primitives/applications
    should have a higher (less negative) log-probability than one using more,
    since each production choice multiplies probabilities (adds log-probs).
    """

    def test_simple_beats_complex(self, base_uniform_grammar):
        """A simpler program should have a higher log-probability.

        Simple:  (λ gt (length $0) 3)              — 4 choices
        Complex: (λ all (λ eq (get_suit $0) CLUBS) $0) — more choices
        """
        simple = "(λ gt (length $0) 3)"
        complex_ = "(λ all (λ eq (get_suit $0) CLUBS) $0)"

        ll_simple = score_hypothesis(simple, base_uniform_grammar)
        ll_complex = score_hypothesis(complex_, base_uniform_grammar)

        # Both should be finite
        assert math.isfinite(ll_simple)
        assert math.isfinite(ll_complex)

        # Simpler should have higher (less negative) log-probability
        assert ll_simple > ll_complex, (
            f"Expected simpler program to score higher: "
            f"{ll_simple:.2f} vs {ll_complex:.2f}"
        )


# ---------------------------------------------------------------------------
# score_hypothesis — different grammars/costs produce different scores
# ---------------------------------------------------------------------------

class TestScoreHypothesisDifferences:
    """Verify that different grammar/cost combinations produce different scores."""

    def test_different_cost_structures_differ(
        self, base_uniform_grammar, base_tiered_grammar
    ):
        """The same program under different cost structures should score differently.

        Under TIERED costs, Tier 1 primitives (like eq, get_suit, CLUBS) are
        boosted, so a program using only Tier 1 primitives should score
        differently than under UNIFORM.
        """
        sexpr = "(λ all (λ eq (get_suit $0) CLUBS) $0)"

        ll_uniform = score_hypothesis(sexpr, base_uniform_grammar)
        ll_tiered = score_hypothesis(sexpr, base_tiered_grammar)

        # Both should be finite
        assert math.isfinite(ll_uniform)
        assert math.isfinite(ll_tiered)

        # They should differ (tiered boosts tier-1 prims like eq, get_suit)
        assert ll_uniform != ll_tiered, (
            f"Expected different scores: uniform={ll_uniform:.4f}, "
            f"tiered={ll_tiered:.4f}"
        )

    def test_different_grammars_differ(self, base_uniform_grammar):
        """The same program under a different grammar should score differently.

        The 'add-both' grammar has more primitives, so uniform probabilities
        are spread thinner, producing a different score.
        """
        sexpr = "(λ gt (length $0) 3)"

        ll_base = score_hypothesis(sexpr, base_uniform_grammar)

        add_both_grammar = build_grammar("add-both", CostStructure.UNIFORM)
        ll_add_both = score_hypothesis(sexpr, add_both_grammar)

        # Both should be finite
        assert math.isfinite(ll_base)
        assert math.isfinite(ll_add_both)

        # They should differ (different number of primitives = different uniform weight)
        assert ll_base != ll_add_both


# ---------------------------------------------------------------------------
# score_all_hypotheses — batch scoring
# ---------------------------------------------------------------------------

class TestScoreAllHypotheses:
    """Test the batch scoring function."""

    def test_returns_list_of_dicts(self):
        """score_all_hypotheses should return a list of dicts with correct fields."""
        results = score_all_hypotheses("base", CostStructure.UNIFORM, limit=5)

        assert isinstance(results, list)
        assert len(results) <= 5

        if len(results) > 0:
            r = results[0]
            # Check all required fields are present
            assert "rule_id" in r
            assert "rank" in r
            assert "confidence" in r
            assert "nl_description" in r
            assert "log_prob" in r
            assert "grammar_name" in r
            assert "cost_structure" in r

            # Check field types
            assert isinstance(r["rule_id"], str)
            assert isinstance(r["rank"], int)
            assert isinstance(r["log_prob"], float)
            assert r["grammar_name"] == "base"
            assert r["cost_structure"] == "uniform"

    def test_log_prob_is_negative_or_neg_inf(self):
        """All log-probabilities should be <= 0 (negative or -inf)."""
        results = score_all_hypotheses("base", CostStructure.UNIFORM, limit=10)

        for r in results:
            assert r["log_prob"] <= 0 or r["log_prob"] == float('-inf'), (
                f"log_prob should be <= 0, got {r['log_prob']} for "
                f"{r['rule_id']} rank {r['rank']}"
            )

    def test_limit_zero_returns_all(self):
        """limit=0 (default) should return all hypotheses."""
        results_limited = score_all_hypotheses("base", CostStructure.UNIFORM, limit=3)
        results_all = score_all_hypotheses("base", CostStructure.UNIFORM, limit=0)

        # The unlimited version should have at least as many as the limited
        assert len(results_all) >= len(results_limited)

    def test_different_grammars_produce_different_scores(self):
        """Batch scoring with different grammars should give different results.

        We score the same hypotheses under UNIFORM and TIERED costs
        and check that at least some scores differ.
        """
        results_uniform = score_all_hypotheses(
            "base", CostStructure.UNIFORM, limit=10
        )
        results_tiered = score_all_hypotheses(
            "base", CostStructure.TIERED, limit=10
        )

        # Find hypotheses that have finite scores in both
        finite_pairs = [
            (u["log_prob"], t["log_prob"])
            for u, t in zip(results_uniform, results_tiered)
            if math.isfinite(u["log_prob"]) and math.isfinite(t["log_prob"])
        ]

        if len(finite_pairs) > 0:
            # At least one pair should differ
            any_differ = any(u != t for u, t in finite_pairs)
            assert any_differ, (
                "Expected at least one hypothesis to score differently "
                "under UNIFORM vs TIERED"
            )

    def test_hypothesis_without_dsl_code_uses_python_fallback(self):
        """Hypotheses with dsl_code=None but valid python_code should get a
        finite score via the Python-to-AST fallback path.

        We mock load_phase1b_hypotheses to return a hypothesis without dsl_code
        but with a python_code that the python_parser can handle.
        """
        mock_hypotheses = [
            {
                "rule_id": "test_rule",
                "rank": 1,
                "confidence": "HIGH",
                "nl_description": "Test hypothesis",
                "dsl_code": None,
                "python_code": "lambda hand: all(card.suit == Suit.CLUBS for card in hand)",
                "judge_verdict": "PASS",
                "source_model": "test",
            }
        ]

        with patch(
            "llm.grammar_comparison.evaluation.compute_costs.load_phase1b_hypotheses",
            return_value=mock_hypotheses,
        ):
            results = score_all_hypotheses("base", CostStructure.UNIFORM)

        assert len(results) == 1
        assert math.isfinite(results[0]["log_prob"])
        assert results[0]["log_prob"] < 0

    def test_hypothesis_with_both_none_gets_neg_inf(self):
        """Hypotheses with both dsl_code=None and python_code=None should
        get -inf log_prob since there is nothing to parse.
        """
        mock_hypotheses = [
            {
                "rule_id": "test_rule",
                "rank": 1,
                "confidence": "HIGH",
                "nl_description": "Test hypothesis",
                "dsl_code": None,
                "python_code": None,
                "judge_verdict": "PASS",
                "source_model": "test",
            }
        ]

        with patch(
            "llm.grammar_comparison.evaluation.compute_costs.load_phase1b_hypotheses",
            return_value=mock_hypotheses,
        ):
            results = score_all_hypotheses("base", CostStructure.UNIFORM)

        assert len(results) == 1
        assert results[0]["log_prob"] == float('-inf')


# ---------------------------------------------------------------------------
# parse_hypothesis -- fallback logic
# ---------------------------------------------------------------------------

class TestParseHypothesisFallback:
    """Test the parse_hypothesis() helper and its fallback chain."""

    def test_dsl_code_none_python_code_valid_returns_program(self):
        """When dsl_code is None but python_code is valid, parse_hypothesis
        should return a Program AST (not None).
        """
        hyp = {
            "dsl_code": None,
            "python_code": "lambda hand: all(card.suit == Suit.CLUBS for card in hand)",
        }
        program = parse_hypothesis(hyp)
        assert program is not None

    def test_both_none_returns_none(self):
        """When both dsl_code and python_code are None, parse_hypothesis
        should return None.
        """
        hyp = {"dsl_code": None, "python_code": None}
        program = parse_hypothesis(hyp)
        assert program is None

    def test_fallback_produces_same_score_as_direct(self, base_uniform_grammar):
        """When a hypothesis has BOTH dsl_code and python_code, the score
        from the s-expression path should match the score from the Python
        fallback path.

        This validates that both parsers produce equivalent ASTs for the
        same underlying rule.
        """
        # A simple rule: all cards are clubs
        dsl_code = "(λ all (λ eq (get_suit $0) CLUBS) $0)"
        python_code = "lambda hand: all(card.suit == Suit.CLUBS for card in hand)"

        # Score via the direct s-expression path
        score_sexpr = score_hypothesis(dsl_code, base_uniform_grammar)

        # Score via the Python fallback path
        hyp_python_only = {"dsl_code": None, "python_code": python_code}
        program_from_python = parse_hypothesis(hyp_python_only)
        assert program_from_python is not None
        score_python = score_program(program_from_python, base_uniform_grammar)

        # Both should be finite and equal
        assert math.isfinite(score_sexpr)
        assert math.isfinite(score_python)
        assert score_sexpr == score_python, (
            f"Scores differ: s-expr={score_sexpr:.4f}, python={score_python:.4f}"
        )
