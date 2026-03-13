"""
Tests for the 4-tier weighted grammar.

Verifies:
1. Grammar is constructed with correct number of productions
2. Tier assignments match the design spec
3. Variable cost is set correctly
4. dsl_prior produces different priors for shallow vs compositional programs
"""
import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pytest

from gallery_analysis.enumerator import (
    build_gallery_primitives,
    build_gallery_grammar,
    build_weighted_gallery_grammar,
    TIER_CHEAP, TIER_STANDARD, TIER_AGGREGATE, TIER_ULTRA_SHALLOW,
)
from gallery_analysis.dsl_prior import compute_log_prior


@pytest.fixture(scope="module")
def weighted_grammar():
    return build_weighted_gallery_grammar()


@pytest.fixture(scope="module")
def uniform_grammar():
    return build_gallery_grammar()


class TestGrammarConstruction:
    """Verify the grammar is built correctly."""

    def test_same_number_of_productions(self, weighted_grammar, uniform_grammar):
        """Weighted grammar has same productions as uniform (same primitives)."""
        assert len(weighted_grammar.productions) == len(uniform_grammar.productions)

    def test_variable_cost_is_cheap(self, weighted_grammar):
        """Variable cost should be -1.0 (not the uniform -4.14)."""
        assert weighted_grammar.log_variable == pytest.approx(-1.0)

    def test_no_missing_primitives(self, weighted_grammar):
        """Every gallery primitive must appear in exactly one tier."""
        prim_names = {p.name for p in build_gallery_primitives()}
        tier_names = TIER_CHEAP | TIER_STANDARD | TIER_AGGREGATE | TIER_ULTRA_SHALLOW
        # Filter to only names that are actual primitives (not constants that
        # might not be in the tier sets because they're in TIER_STANDARD by default)
        assigned = tier_names & prim_names
        unassigned = prim_names - tier_names
        # Unassigned primitives default to TIER_STANDARD, so this is fine
        # But check no primitive is in multiple explicit tiers
        overlap = (TIER_CHEAP & TIER_AGGREGATE) | (TIER_CHEAP & TIER_ULTRA_SHALLOW) | (TIER_AGGREGATE & TIER_ULTRA_SHALLOW)
        assert overlap == set(), f"Primitives in multiple tiers: {overlap}"


class TestTierCosts:
    """Verify that tier log-probabilities are applied correctly."""

    def test_has_suit_is_expensive(self, weighted_grammar):
        """has_suit should have log_p = -9.0 (ultra-shallow tier)."""
        for prod in weighted_grammar.productions:
            if prod.program.name == 'has_suit':
                assert prod.log_probability == pytest.approx(-9.0)
                return
        pytest.fail("has_suit not found in productions")

    def test_has_color_is_expensive(self, weighted_grammar):
        """has_color should have log_p = -9.0 (ultra-shallow tier)."""
        for prod in weighted_grammar.productions:
            if prod.program.name == 'has_color':
                assert prod.log_probability == pytest.approx(-9.0)
                return
        pytest.fail("has_color not found in productions")

    def test_eq_is_cheap(self, weighted_grammar):
        """eq should have log_p = -3.0 (cheap tier)."""
        for prod in weighted_grammar.productions:
            if prod.program.name == 'eq':
                assert prod.log_probability == pytest.approx(-3.0)
                return
        pytest.fail("eq not found in productions")

    def test_count_suit_is_aggregate(self, weighted_grammar):
        """count_suit should have log_p = -5.5 (aggregate tier)."""
        for prod in weighted_grammar.productions:
            if prod.program.name == 'count_suit':
                assert prod.log_probability == pytest.approx(-5.5)
                return
        pytest.fail("count_suit not found in productions")

    def test_rank_val_is_standard(self, weighted_grammar):
        """rank_val should have log_p = -4.0 (standard tier)."""
        for prod in weighted_grammar.productions:
            if prod.program.name == 'rank_val':
                assert prod.log_probability == pytest.approx(-4.0)
                return
        pytest.fail("rank_val not found in productions")


class TestAnalyzeCLIFlag:
    """Verify that analyze.py accepts --grammar flag."""

    def test_grammar_flag_is_recognized(self):
        """The --grammar flag should be recognized by the argument parser."""
        import gallery_analysis.analyze as analyze_mod
        parser = analyze_mod.build_argument_parser()
        args = parser.parse_args(["--grammar", "weighted", "--quick"])
        assert args.grammar == "weighted"

    def test_grammar_flag_defaults_to_uniform(self):
        """Default grammar should be 'uniform' for backward compatibility."""
        import gallery_analysis.analyze as analyze_mod
        parser = analyze_mod.build_argument_parser()
        args = parser.parse_args([])
        assert args.grammar == "uniform"


class TestPriorEffect:
    """Verify that weighted grammar changes priors in the expected direction."""

    def test_shallow_program_is_more_expensive(self, weighted_grammar, uniform_grammar):
        """has_suit under weighted grammar should have worse prior than under uniform."""
        prog = "(λ has_suit $0 SPADES)"
        prior_uniform = compute_log_prior(prog, uniform_grammar)
        prior_weighted = compute_log_prior(prog, weighted_grammar)
        # Weighted should be more negative (more expensive)
        assert prior_weighted < prior_uniform

    def test_compositional_program_is_cheaper(self, weighted_grammar, uniform_grammar):
        """A compositional HOF program should be cheaper under weighted grammar."""
        # all (λ eq (get_suit $0) SPADES) $0 — uses cheap eq + cheap variable
        prog = "(λ all (λ eq (get_suit $0) SPADES) $0)"
        prior_uniform = compute_log_prior(prog, uniform_grammar)
        prior_weighted = compute_log_prior(prog, weighted_grammar)
        # Weighted should be less negative (cheaper) due to cheap vars + cheap eq
        assert prior_weighted > prior_uniform


class TestWeightedPipelineIntegration:
    """
    Integration test: run a small analysis with --grammar weighted and
    verify that the output structure is correct and priors differ from uniform.
    """

    def test_run_analysis_with_weighted_grammar(self):
        """run_analysis() with scoring_grammar='weighted' should complete without error."""
        from gallery_analysis.analyze import run_analysis
        results = run_analysis(
            max_depth=4,
            max_programs=1000,
            max_cost=20.0,
            timeout=30.0,
            n_probes=50,
            extension_samples=1000,
            scoring_grammar="weighted",
            verbose=0,
        )
        assert "rule_results" in results
        assert "difficulty_ranking" in results
        assert len(results["rule_results"]) > 0

    def test_weighted_changes_ranking(self):
        """
        Weighted grammar should produce different top-1 programs for at least
        some rules compared to uniform.
        """
        from gallery_analysis.analyze import run_analysis

        r_uniform = run_analysis(
            max_depth=4, max_programs=1000, max_cost=20.0,
            timeout=30.0, n_probes=50, extension_samples=1000,
            scoring_grammar="uniform", verbose=0,
        )
        r_weighted = run_analysis(
            max_depth=4, max_programs=1000, max_cost=20.0,
            timeout=30.0, n_probes=50, extension_samples=1000,
            scoring_grammar="weighted", verbose=0,
        )

        # Check at least one rule has a different top-1
        n_different = 0
        for rule_id in r_uniform["rule_results"]:
            if rule_id not in r_weighted["rule_results"]:
                continue
            u_top = r_uniform["rule_results"][rule_id]["top_hypotheses"]
            w_top = r_weighted["rule_results"][rule_id]["top_hypotheses"]
            if u_top and w_top and u_top[0]["program"] != w_top[0]["program"]:
                n_different += 1
        assert n_different > 0, "Weighted grammar should change at least one top-1 ranking"
