"""
Tests for the MCMC program sampler.

Verifies that sample_program() produces complete, type-correct programs
from the grammar prior, with reproducibility via seeding.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from dreamcoder_core.type_system import (
    Arrow, BOOL, HAND, INT, CARD, SUIT, RANK, TypeContext,
)
from dreamcoder_core.program import has_holes, Hole
from gallery_analysis.enumerator import build_gallery_grammar
from gallery_analysis.mcmc_search import sample_program


@pytest.fixture
def grammar():
    """Build the gallery grammar (no boolean constants, ~57 primitives)."""
    return build_gallery_grammar()


# --------------------------------------------------------------------------- #
# Test 1: Sampled programs are complete (no holes)
# --------------------------------------------------------------------------- #

def test_samples_complete_program(grammar):
    """sample_program should return a fully expanded AST with no Hole nodes."""
    for seed in range(10):
        program = sample_program(grammar, Arrow(HAND, BOOL), max_depth=6, seed=seed)
        assert not has_holes(program), (
            f"seed={seed} produced a program with holes: {program}"
        )


# --------------------------------------------------------------------------- #
# Test 2: Sampled programs type-check against the request type
# --------------------------------------------------------------------------- #

def test_samples_correct_type(grammar):
    """The inferred type of every sampled program should unify with hand -> bool.

    Note: some sampled programs contain polymorphic subexpressions whose type
    variables may not be fully resolved (e.g., map with a lambda that doesn't
    constrain its input type). The inferred type will be something like
    list('a) -> bool, which is MORE GENERAL than list(card) -> bool and
    therefore unifies with it. We test unifiability, not strict equality.
    """
    request = Arrow(HAND, BOOL)
    for seed in range(10):
        program = sample_program(grammar, request, max_depth=6, seed=seed)
        # Type-check the program
        ctx = TypeContext()
        inferred = program.infer_type(ctx, env=[])
        resolved = ctx.apply(inferred)
        # The resolved type should unify with hand -> bool.
        # Use a fresh context for unification to avoid polluting the inference context.
        unify_ctx = TypeContext()
        try:
            unify_ctx.unify(resolved, request)
        except Exception as e:
            raise AssertionError(
                f"seed={seed}: inferred type {resolved} does not unify with "
                f"{request} for program {program}: {e}"
            )


# --------------------------------------------------------------------------- #
# Test 3: Different seeds produce different programs
# --------------------------------------------------------------------------- #

def test_samples_different_programs(grammar):
    """Different seeds should (almost certainly) produce different programs.

    With ~57 primitives, the probability that 20 independent samples all
    produce the exact same program is vanishingly small.
    """
    programs = set()
    for seed in range(20):
        p = sample_program(grammar, Arrow(HAND, BOOL), max_depth=6, seed=seed)
        programs.add(str(p))
    # At least 2 distinct programs out of 20 seeds
    assert len(programs) >= 2, (
        f"All 20 seeds produced the same program: {programs}"
    )


# --------------------------------------------------------------------------- #
# Test 4: Respects max_depth bound
# --------------------------------------------------------------------------- #

def test_respects_max_depth(grammar):
    """Sampled programs should have bounded depth.

    The sampler uses max_depth as a soft limit: beyond this depth it strongly
    prefers terminals and variables. However, some types in the gallery grammar
    (notably BOOL, which has no constant productions since true/false are
    excluded) require at least one non-terminal production even at max_depth.
    Additionally, lambdas (which are free) add to the AST's .depth() count.

    We verify that the depth stays within a reasonable multiple of max_depth,
    which ensures the sampler terminates and doesn't produce runaway recursion.
    The absolute depth limit in the sampler (_ABSOLUTE_DEPTH_LIMIT=20)
    provides the hard guarantee; this test checks that programs are
    practically bounded.
    """
    max_depth = 5
    for seed in range(10):
        program = sample_program(grammar, Arrow(HAND, BOOL), max_depth=max_depth, seed=seed)
        # The AST .depth() includes lambdas and the grace overflow for types
        # without terminal productions. A 4x multiple of max_depth is generous
        # enough to avoid false failures while catching truly runaway recursion.
        assert program.depth() <= max_depth * 4, (
            f"seed={seed}: depth {program.depth()} exceeds {max_depth * 4} "
            f"for program {program}"
        )


# --------------------------------------------------------------------------- #
# Test 5: Same seed produces same program (determinism)
# --------------------------------------------------------------------------- #

def test_deterministic_with_seed(grammar):
    """Calling sample_program with the same seed should return the identical AST."""
    for seed in [0, 42, 99, 12345]:
        p1 = sample_program(grammar, Arrow(HAND, BOOL), max_depth=6, seed=seed)
        p2 = sample_program(grammar, Arrow(HAND, BOOL), max_depth=6, seed=seed)
        assert str(p1) == str(p2), (
            f"seed={seed} not deterministic: {p1} vs {p2}"
        )
