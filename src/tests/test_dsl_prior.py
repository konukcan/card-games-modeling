"""
Tests for dsl_prior.compute_log_prior().

Verifies that:
1. Simple programs have finite negative log-priors
2. Computed priors match what the enumerator produces (within tolerance)
3. Deeper programs have lower (more negative) priors than shallow ones
"""

import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from gallery_analysis.dsl_prior import compute_log_prior
from gallery_analysis.enumerator import build_gallery_grammar, enumerate_hypotheses


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture(scope="module")
def grammar():
    """Build the gallery grammar once for all tests."""
    return build_gallery_grammar()


@pytest.fixture(scope="module")
def enumerated_programs(grammar):
    """
    Enumerate a small set of programs at depth 4 to use as ground truth.

    Returns list of (program_string, log_prior) tuples.
    We use depth 4 and a small count to keep tests fast.
    """
    results = enumerate_hypotheses(
        max_depth=4,
        max_programs=50,
        max_cost=50.0,
        timeout=30.0,
        grammar=grammar,
        syntactic_filter=True,
    )
    # Each result is (prog_str, pred_fn, log_prob)
    return [(prog_str, log_prob) for prog_str, _, log_prob in results]


# =========================================================================
# Test 1: Simple programs have finite negative log-priors
# =========================================================================

class TestBasicPrior:
    """Basic sanity checks on log-prior values."""

    def test_simple_program_has_finite_negative_prior(self, grammar):
        """A valid program should have a finite, negative log-prior."""
        # (λ has_color $0 RED) — "hand has color red"
        lp = compute_log_prior("(λ has_color $0 RED)", grammar)
        assert math.isfinite(lp), f"Expected finite log-prior, got {lp}"
        assert lp < 0, f"Expected negative log-prior, got {lp}"

    def test_another_simple_program(self, grammar):
        """Another simple program should also have a finite negative prior."""
        # (λ has_suit $0 HEARTS) — "hand has suit hearts"
        lp = compute_log_prior("(λ has_suit $0 HEARTS)", grammar)
        assert math.isfinite(lp), f"Expected finite log-prior, got {lp}"
        assert lp < 0, f"Expected negative log-prior, got {lp}"

    def test_invalid_program_raises(self, grammar):
        """An unparseable program should raise ValueError."""
        with pytest.raises(ValueError):
            compute_log_prior("(λ nonexistent_primitive $0)", grammar)


# =========================================================================
# Test 2: Priors match the enumerator's log-probs
# =========================================================================

class TestMatchesEnumerator:
    """
    The enumerator computes log-priors during enumeration.
    compute_log_prior should produce the same values when given
    the same program strings and grammar.
    """

    def test_priors_match_enumerator(self, grammar, enumerated_programs):
        """
        For each enumerated program, compute_log_prior should match
        the enumerator's log_prob within tolerance 0.01.
        """
        assert len(enumerated_programs) > 0, "No programs were enumerated"

        mismatches = []
        for prog_str, enum_log_prob in enumerated_programs:
            computed_lp = compute_log_prior(prog_str, grammar)

            if not math.isclose(computed_lp, enum_log_prob, abs_tol=0.01):
                mismatches.append(
                    f"  {prog_str}: enumerator={enum_log_prob:.4f}, "
                    f"computed={computed_lp:.4f}, "
                    f"diff={abs(computed_lp - enum_log_prob):.4f}"
                )

        if mismatches:
            detail = "\n".join(mismatches)
            pytest.fail(
                f"{len(mismatches)}/{len(enumerated_programs)} priors "
                f"don't match enumerator:\n{detail}"
            )


# =========================================================================
# Test 3: Deeper programs have lower priors
# =========================================================================

class TestDepthOrdering:
    """Deeper (more complex) programs should have lower log-priors."""

    def test_deeper_program_has_lower_prior(self, grammar):
        """
        A deeper program should have a more negative log-prior than
        a shallower one, since each additional node multiplies in
        another probability factor < 1.

        Shallow: (λ has_color $0 RED)
            — 4 nodes: λ, has_color, $0, RED

        Deep: (λ and (has_color $0 RED) (has_suit $0 HEARTS))
            — 8 nodes: λ, and, has_color, $0, RED, has_suit, $0, HEARTS
        """
        shallow_lp = compute_log_prior("(λ has_color $0 RED)", grammar)
        deep_lp = compute_log_prior(
            "(λ and (has_color $0 RED) (has_suit $0 HEARTS))", grammar
        )

        assert math.isfinite(shallow_lp), f"Shallow prior not finite: {shallow_lp}"
        assert math.isfinite(deep_lp), f"Deep prior not finite: {deep_lp}"
        assert deep_lp < shallow_lp, (
            f"Expected deeper program to have lower prior. "
            f"Shallow={shallow_lp:.4f}, Deep={deep_lp:.4f}"
        )
