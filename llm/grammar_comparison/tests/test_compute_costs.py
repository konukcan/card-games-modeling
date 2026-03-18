"""
Tests for the log-probability scorer (compute_costs module).

Verifies posterior = prior + likelihood, backward compatibility of score_hypothesis,
fingerprint/extension fields, and exemplar consistency checks.
"""

import math
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import pytest

from llm.grammar_comparison.evaluation.compute_costs import (
    score_hypothesis,
    score_all_hypotheses,
    parse_hypothesis,
    score_program,
    _fingerprint_to_string,
    _make_predicate,
    REQUEST_TYPE,
)
from llm.grammar_comparison.grammars.grammar_factory import (
    build_grammar,
    CostStructure,
)
from dreamcoder_core.type_system import HAND, BOOL, arrow
from rules.cards import Card, Suit, Rank


def _make_hand_all_clubs():
    return [Card(Suit.CLUBS, r) for r in [Rank.TWO, Rank.THREE, Rank.FOUR, Rank.FIVE, Rank.SIX, Rank.SEVEN]]

def _make_hand_mixed():
    return [
        Card(Suit.CLUBS, Rank.TWO), Card(Suit.HEARTS, Rank.THREE),
        Card(Suit.DIAMONDS, Rank.FOUR), Card(Suit.SPADES, Rank.FIVE),
        Card(Suit.CLUBS, Rank.SIX), Card(Suit.HEARTS, Rank.SEVEN),
    ]

def _mock_hypothesis(dsl_code="(\u03bb all (\u03bb eq (get_suit $0) CLUBS) $0)", python_code=None,
                     exemplar_hands=None, rule_id="test_rule", rank=1):
    if exemplar_hands is None:
        exemplar_hands = [_make_hand_all_clubs() for _ in range(6)]
    return {
        "rule_id": rule_id, "rank": rank, "confidence": "HIGH",
        "nl_description": "Test hypothesis", "dsl_code": dsl_code,
        "python_code": python_code, "judge_verdict": "PASS",
        "source_model": "test", "exemplar_hands": exemplar_hands,
    }


@pytest.fixture
def base_uniform_grammar():
    return build_grammar("base", CostStructure.UNIFORM)

@pytest.fixture
def base_tiered_grammar():
    return build_grammar("base", CostStructure.TIERED)

@pytest.fixture
def minimal_uniform_grammar():
    return build_grammar("minimal", CostStructure.UNIFORM)


class TestRequestType:
    def test_request_type_is_hand_to_bool(self):
        assert REQUEST_TYPE == arrow(HAND, BOOL)


class TestScoreHypothesisBasic:
    def test_valid_program_returns_negative_float(self, base_uniform_grammar):
        ll = score_hypothesis("(\u03bb all (\u03bb eq (get_suit $0) CLUBS) $0)", base_uniform_grammar)
        assert isinstance(ll, float) and ll < 0 and math.isfinite(ll)

    def test_another_valid_program(self, base_uniform_grammar):
        ll = score_hypothesis("(\u03bb gt (length $0) 3)", base_uniform_grammar)
        assert isinstance(ll, float) and ll < 0 and math.isfinite(ll)


class TestScoreHypothesisInexpressible:
    def test_parse_error_returns_neg_inf(self, base_uniform_grammar):
        assert score_hypothesis("(((broken syntax", base_uniform_grammar) == float('-inf')

    def test_unknown_primitive_returns_neg_inf(self, base_uniform_grammar):
        assert score_hypothesis("(\u03bb nonexistent_primitive $0)", base_uniform_grammar) == float('-inf')

    def test_empty_string_returns_neg_inf(self, base_uniform_grammar):
        assert score_hypothesis("", base_uniform_grammar) == float('-inf')

    def test_primitive_not_in_grammar(self, minimal_uniform_grammar):
        assert score_hypothesis("(\u03bb gt (count_suit HEARTS $0) 2)", minimal_uniform_grammar) == float('-inf')


class TestScoreHypothesisComplexity:
    def test_simple_beats_complex(self, base_uniform_grammar):
        ll_s = score_hypothesis("(\u03bb gt (length $0) 3)", base_uniform_grammar)
        ll_c = score_hypothesis("(\u03bb all (\u03bb eq (get_suit $0) CLUBS) $0)", base_uniform_grammar)
        assert math.isfinite(ll_s) and math.isfinite(ll_c) and ll_s > ll_c


class TestScoreHypothesisDifferences:
    def test_different_cost_structures_differ(self, base_uniform_grammar, base_tiered_grammar):
        sexpr = "(\u03bb all (\u03bb eq (get_suit $0) CLUBS) $0)"
        assert score_hypothesis(sexpr, base_uniform_grammar) != score_hypothesis(sexpr, base_tiered_grammar)

    def test_different_grammars_differ(self, base_uniform_grammar):
        sexpr = "(\u03bb gt (length $0) 3)"
        assert score_hypothesis(sexpr, base_uniform_grammar) != score_hypothesis(sexpr, build_grammar("add-both", CostStructure.UNIFORM))


class TestHelpers:
    def test_fingerprint_to_string_basic(self):
        assert _fingerprint_to_string((True, False, True, None, False)) == "101X0"

    def test_fingerprint_to_string_empty(self):
        assert _fingerprint_to_string(()) == ""

    def test_make_predicate_valid(self):
        from llm.grammar_comparison.translation.sexpr_parser import parse_hypothesis_sexpr
        pred = _make_predicate(parse_hypothesis_sexpr("(\u03bb all (\u03bb eq (get_suit $0) CLUBS) $0)"))
        assert pred is not None
        assert pred(_make_hand_all_clubs()) is True
        assert pred(_make_hand_mixed()) is False

    def test_make_predicate_returns_none_on_failure(self):
        from dreamcoder_core.program import Index
        assert _make_predicate(Index(0)) is None


class TestScoreAllHypothesesPosterior:

    def test_returns_list_of_dicts_with_new_fields(self):
        with patch("llm.grammar_comparison.evaluation.compute_costs.load_phase1b_hypotheses",
                   return_value=[_mock_hypothesis()]):
            results = score_all_hypotheses("base", CostStructure.UNIFORM)
        r = results[0]
        for f in ["log_prior", "log_likelihood", "log_posterior", "base_rate",
                   "fingerprint", "exemplars_consistent", "grammar_name", "cost_structure"]:
            assert f in r, f"Missing field: {f}"
        assert isinstance(r["log_prior"], float) and isinstance(r["log_posterior"], float)
        assert r["grammar_name"] == "base" and r["cost_structure"] == "uniform"

    def test_posterior_is_sum_of_prior_and_likelihood(self):
        with patch("llm.grammar_comparison.evaluation.compute_costs.load_phase1b_hypotheses",
                   return_value=[_mock_hypothesis()]):
            r = score_all_hypotheses("base", CostStructure.UNIFORM)[0]
        if math.isfinite(r["log_prior"]) and math.isfinite(r["log_likelihood"]):
            assert abs(r["log_posterior"] - (r["log_prior"] + r["log_likelihood"])) < 1e-10

    def test_log_prior_is_negative(self):
        with patch("llm.grammar_comparison.evaluation.compute_costs.load_phase1b_hypotheses",
                   return_value=[_mock_hypothesis()]):
            assert score_all_hypotheses("base", CostStructure.UNIFORM)[0]["log_prior"] <= 0

    def test_log_likelihood_is_negative(self):
        with patch("llm.grammar_comparison.evaluation.compute_costs.load_phase1b_hypotheses",
                   return_value=[_mock_hypothesis()]):
            assert score_all_hypotheses("base", CostStructure.UNIFORM)[0]["log_likelihood"] <= 0

    def test_specific_hypothesis_higher_likelihood_than_vague(self):
        specific = _mock_hypothesis(rule_id="specific", rank=1)
        vague = _mock_hypothesis(dsl_code="(\u03bb gt (length $0) 0)", rule_id="vague", rank=2)
        with patch("llm.grammar_comparison.evaluation.compute_costs.load_phase1b_hypotheses",
                   return_value=[specific, vague]):
            results = score_all_hypotheses("base", CostStructure.UNIFORM)
        s = next(r for r in results if r["rule_id"] == "specific")
        v = next(r for r in results if r["rule_id"] == "vague")
        assert s["log_likelihood"] > v["log_likelihood"]

    def test_inconsistent_exemplars_give_neg_inf_likelihood(self):
        hyp = _mock_hypothesis(exemplar_hands=[_make_hand_mixed() for _ in range(6)])
        with patch("llm.grammar_comparison.evaluation.compute_costs.load_phase1b_hypotheses",
                   return_value=[hyp]):
            r = score_all_hypotheses("base", CostStructure.UNIFORM)[0]
        assert r["log_likelihood"] == float('-inf')
        assert r["exemplars_consistent"] is False
        assert r["log_posterior"] == float('-inf')

    def test_unparseable_hypothesis_gets_all_neg_inf(self):
        with patch("llm.grammar_comparison.evaluation.compute_costs.load_phase1b_hypotheses",
                   return_value=[_mock_hypothesis(dsl_code=None, python_code=None)]):
            r = score_all_hypotheses("base", CostStructure.UNIFORM)[0]
        assert r["log_prior"] == r["log_likelihood"] == r["log_posterior"] == float('-inf')
        assert r["fingerprint"] == "" and r["exemplars_consistent"] is False

    def test_limit_parameter_works(self):
        hyps = [_mock_hypothesis(rule_id=f"rule_{i}", rank=i) for i in range(3)]
        with patch("llm.grammar_comparison.evaluation.compute_costs.load_phase1b_hypotheses",
                   return_value=hyps):
            assert len(score_all_hypotheses("base", CostStructure.UNIFORM, limit=2)) == 2

    def test_different_grammars_same_likelihood(self):
        with patch("llm.grammar_comparison.evaluation.compute_costs.load_phase1b_hypotheses",
                   return_value=[_mock_hypothesis()]):
            r_base = score_all_hypotheses("base", CostStructure.UNIFORM)[0]
            r_add = score_all_hypotheses("add-both", CostStructure.UNIFORM)[0]
        if math.isfinite(r_base["log_prior"]) and math.isfinite(r_add["log_prior"]):
            assert r_base["log_prior"] != r_add["log_prior"]
        assert r_base["log_likelihood"] == r_add["log_likelihood"]

    def test_fingerprint_is_populated(self):
        with patch("llm.grammar_comparison.evaluation.compute_costs.load_phase1b_hypotheses",
                   return_value=[_mock_hypothesis()]):
            r = score_all_hypotheses("base", CostStructure.UNIFORM)[0]
        assert len(r["fingerprint"]) > 0 and all(c in "01X" for c in r["fingerprint"])

    def test_different_cost_structures_same_likelihood(self):
        with patch("llm.grammar_comparison.evaluation.compute_costs.load_phase1b_hypotheses",
                   return_value=[_mock_hypothesis()]):
            ru = score_all_hypotheses("base", CostStructure.UNIFORM)[0]
            rt = score_all_hypotheses("base", CostStructure.TIERED)[0]
        if math.isfinite(ru["log_prior"]) and math.isfinite(rt["log_prior"]):
            assert ru["log_prior"] != rt["log_prior"]
        assert ru["log_likelihood"] == rt["log_likelihood"]

    def test_python_fallback_with_exemplar_hands(self):
        with patch("llm.grammar_comparison.evaluation.compute_costs.load_phase1b_hypotheses",
                   return_value=[_mock_hypothesis(dsl_code=None,
                       python_code="lambda hand: all(card.suit == Suit.CLUBS for card in hand)")]):
            r = score_all_hypotheses("base", CostStructure.UNIFORM)[0]
        assert math.isfinite(r["log_prior"]) and r["log_prior"] < 0


class TestParseHypothesisFallback:
    def test_dsl_code_none_python_code_valid(self):
        assert parse_hypothesis({"dsl_code": None, "python_code": "lambda hand: all(card.suit == Suit.CLUBS for card in hand)"}) is not None

    def test_both_none(self):
        assert parse_hypothesis({"dsl_code": None, "python_code": None}) is None

    def test_fallback_same_score(self, base_uniform_grammar):
        dsl = "(\u03bb all (\u03bb eq (get_suit $0) CLUBS) $0)"
        py = "lambda hand: all(card.suit == Suit.CLUBS for card in hand)"
        s1 = score_hypothesis(dsl, base_uniform_grammar)
        p = parse_hypothesis({"dsl_code": None, "python_code": py})
        s2 = score_program(p, base_uniform_grammar)
        assert math.isfinite(s1) and s1 == s2
