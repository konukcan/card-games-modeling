"""
Tests for the MCMC program sampler and chain runner.

Verifies that sample_program() produces complete, type-correct programs
from the grammar prior, with reproducibility via seeding. Also tests the
full MH chain (MCMCChain) and likelihood computation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import math

from dreamcoder_core.type_system import (
    Arrow, BOOL, HAND, INT, CARD, SUIT, RANK, TypeContext, Type,
)
from dreamcoder_core.program import has_holes, Hole, Program
from gallery_analysis.enumerator import build_gallery_grammar
from gallery_analysis.mcmc_search import (
    sample_program, collect_subtree_sites, propose_regeneration,
    replace_subtree, SubtreeSite,
    MCMCConfig, MCMCResult, MCMCChain, compute_mcmc_log_likelihood,
)
from gallery_analysis.exemplars import load_exemplars, generate_probe_set


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


# =========================================================================== #
# Tests for collect_subtree_sites
# =========================================================================== #

def test_collect_subtree_sites_finds_nodes(grammar):
    """Should find multiple subtree sites in a non-trivial program."""
    prog = sample_program(grammar, Arrow(HAND, BOOL), max_depth=4, seed=42)
    sites = collect_subtree_sites(prog, Arrow(HAND, BOOL))
    assert len(sites) > 0, (
        f"Expected at least 1 subtree site, got 0 for program: {prog}"
    )


def test_subtree_sites_have_types_and_env(grammar):
    """Each site should have type, env, path, and subtree."""
    prog = sample_program(grammar, Arrow(HAND, BOOL), max_depth=4, seed=42)
    sites = collect_subtree_sites(prog, Arrow(HAND, BOOL))
    for site in sites:
        assert isinstance(site.type, Type), (
            f"site.type should be a Type, got {type(site.type)}"
        )
        assert isinstance(site.env, list), (
            f"site.env should be a list, got {type(site.env)}"
        )
        assert isinstance(site.path, tuple), (
            f"site.path should be a tuple, got {type(site.path)}"
        )
        assert isinstance(site.subtree, Program), (
            f"site.subtree should be a Program, got {type(site.subtree)}"
        )


def test_subtree_sites_skip_root(grammar):
    """Root should not be in the site list."""
    prog = sample_program(grammar, Arrow(HAND, BOOL), max_depth=4, seed=42)
    sites = collect_subtree_sites(prog, Arrow(HAND, BOOL))
    for site in sites:
        assert len(site.path) > 0, "Root node should not be a subtree site"


# =========================================================================== #
# Tests for replace_subtree
# =========================================================================== #

def test_replace_subtree_identity(grammar):
    """Replacing a subtree with itself should produce the same program."""
    prog = sample_program(grammar, Arrow(HAND, BOOL), max_depth=4, seed=42)
    sites = collect_subtree_sites(prog, Arrow(HAND, BOOL))
    if sites:
        site = sites[0]
        replaced = replace_subtree(prog, site.path, site.subtree)
        assert str(replaced) == str(prog), (
            f"Identity replacement changed the program:\n"
            f"  original: {prog}\n"
            f"  replaced: {replaced}\n"
            f"  path: {site.path}"
        )


# =========================================================================== #
# Tests for propose_regeneration
# =========================================================================== #

def test_propose_regeneration_returns_valid(grammar):
    """Proposal should return a complete program and finite log probs."""
    prog = sample_program(grammar, Arrow(HAND, BOOL), max_depth=5, seed=42)
    new_prog, log_q_fwd, log_q_rev = propose_regeneration(
        grammar, prog, Arrow(HAND, BOOL), max_depth=5, seed=100
    )
    assert not has_holes(new_prog), (
        f"Proposed program has holes: {new_prog}"
    )
    # Log probs should be finite or -inf (valid probability values).
    assert math.isfinite(log_q_fwd) or log_q_fwd == float('-inf'), (
        f"Forward log prob is not a valid number: {log_q_fwd}"
    )
    assert math.isfinite(log_q_rev) or log_q_rev == float('-inf'), (
        f"Reverse log prob is not a valid number: {log_q_rev}"
    )


def test_propose_regeneration_changes_program(grammar):
    """At least some proposals should produce different programs."""
    prog = sample_program(grammar, Arrow(HAND, BOOL), max_depth=5, seed=42)
    different_count = 0
    for seed in range(20):
        new_prog, _, _ = propose_regeneration(
            grammar, prog, Arrow(HAND, BOOL), max_depth=5, seed=seed
        )
        if str(new_prog) != str(prog):
            different_count += 1
    assert different_count > 0, (
        f"All 20 proposals produced the same program as the original: {prog}"
    )


# =========================================================================== #
# Tests for compute_mcmc_log_likelihood
# =========================================================================== #

@pytest.fixture
def exemplars():
    """Load frozen exemplar hands from the gallery experiment."""
    return load_exemplars()


def test_likelihood_returns_float(grammar, exemplars):
    """Likelihood should return a finite float (or -inf), never NaN."""
    hands = exemplars['all_red']['hands_primary']
    probes = generate_probe_set(n_probes=1000, seed=42)
    prog = sample_program(grammar, Arrow(HAND, BOOL), max_depth=5, seed=42)
    ll = compute_mcmc_log_likelihood(prog, hands, noise_epsilon=0.01, ext_probe_hands=probes)
    assert isinstance(ll, float), f"Expected float, got {type(ll)}"
    assert not math.isnan(ll), f"Likelihood is NaN for program {prog}"


def test_likelihood_noise_prevents_neg_inf(grammar, exemplars):
    """With noise_epsilon > 0, likelihood should be finite (not -inf)
    for programs that can at least be evaluated, even if they miss all exemplars."""
    hands = exemplars['all_red']['hands_primary']
    probes = generate_probe_set(n_probes=1000, seed=42)
    # Try several seeds — at least one program should evaluate without crashing
    # and produce a finite (not -inf) likelihood due to noise floor.
    finite_count = 0
    for seed in range(20):
        prog = sample_program(grammar, Arrow(HAND, BOOL), max_depth=5, seed=seed)
        ll = compute_mcmc_log_likelihood(prog, hands, noise_epsilon=0.01, ext_probe_hands=probes)
        if math.isfinite(ll):
            finite_count += 1
    assert finite_count > 0, (
        "All 20 programs produced -inf likelihood despite noise_epsilon=0.01"
    )


# =========================================================================== #
# Tests for MCMCChain
# =========================================================================== #

def test_chain_runs_without_error(grammar, exemplars):
    """Chain should complete N steps without crashing."""
    config = MCMCConfig(n_steps=100, max_depth=5, seed=42)
    hands = exemplars['all_red']['hands_primary']
    result = MCMCChain(grammar, config).run(
        request_type=Arrow(HAND, BOOL),
        exemplar_hands=hands,
    )
    assert result is not None
    assert result.n_steps == 100


def test_chain_collects_unique_programs(grammar, exemplars):
    """Chain should find distinct programs."""
    config = MCMCConfig(n_steps=500, max_depth=5, seed=42)
    hands = exemplars['all_red']['hands_primary']
    result = MCMCChain(grammar, config).run(
        request_type=Arrow(HAND, BOOL),
        exemplar_hands=hands,
    )
    assert result.n_unique > 0


def test_chain_acceptance_rate(grammar, exemplars):
    """Acceptance rate should be between 0 and 1."""
    config = MCMCConfig(n_steps=500, max_depth=5, seed=42)
    hands = exemplars['all_red']['hands_primary']
    result = MCMCChain(grammar, config).run(
        request_type=Arrow(HAND, BOOL),
        exemplar_hands=hands,
    )
    assert 0.0 <= result.acceptance_rate <= 1.0


def test_chain_has_top_hypotheses(grammar, exemplars):
    """Result should have ranked hypotheses with required fields."""
    config = MCMCConfig(n_steps=1000, max_depth=5, seed=42)
    hands = exemplars['all_red']['hands_primary']
    result = MCMCChain(grammar, config).run(
        request_type=Arrow(HAND, BOOL),
        exemplar_hands=hands,
    )
    assert len(result.top_hypotheses) > 0
    for hyp in result.top_hypotheses[:5]:
        assert 'program' in hyp
        assert 'visit_count' in hyp
        assert 'log_posterior' in hyp
        assert 'first_seen_step' in hyp


def test_chain_visit_counts_sum(grammar, exemplars):
    """Total visit counts should equal n_steps + 1 (initial state + all steps)."""
    config = MCMCConfig(n_steps=200, max_depth=5, seed=42)
    hands = exemplars['all_red']['hands_primary']
    result = MCMCChain(grammar, config).run(
        request_type=Arrow(HAND, BOOL),
        exemplar_hands=hands,
    )
    total_visits = sum(result.visit_counts.values())
    # n_steps + 1 because we count the initial program plus one count per step.
    assert total_visits == config.n_steps + 1, (
        f"Expected {config.n_steps + 1} total visits, got {total_visits}"
    )
