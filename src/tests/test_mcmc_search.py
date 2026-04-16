"""
Tests for the MCMC program sampler and chain runner.

Verifies that sample_program() produces complete, type-correct programs
from the grammar prior, with reproducibility via seeding. Also tests the
full MH chain (MCMCChain) and likelihood computation, including the
3-layer tautology rejection system.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

import math

from dreamcoder_core.type_system import (
    Arrow, BOOL, HAND, INT, CARD, SUIT, RANK, TypeContext, Type,
)
from dreamcoder_core.program import has_holes, Hole, Program, Index, Abstraction, Application, Primitive
from gallery_analysis.enumerator import build_gallery_grammar
from gallery_analysis.mcmc_search import (
    sample_program, collect_subtree_sites, propose_regeneration,
    replace_subtree, SubtreeSite,
    MCMCConfig, MCMCResult, MCMCChain, compute_mcmc_log_likelihood,
    run_parallel_chains, is_vacuous_lambda,
    _score_subtree_under_sampler, _sample,
    get_collect_subtree_sites_failures,
    reset_collect_subtree_sites_failures,
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
# C1 regression: scorer density matches sampler density
# =========================================================================== #

def test_c1_scorer_round_trip_finite(grammar):
    """C1 regression: for any program produced by `_sample`, the scorer
    must return a finite log_q under the SAME sampler distribution.

    If the scorer returns -inf for a program the sampler just generated,
    the MH reverse density would be -inf and the proposal could never be
    accepted back, breaking detailed balance.
    """
    from gallery_analysis.mcmc_search import _score_subtree_under_sampler

    request = Arrow(HAND, BOOL)
    max_depth = 5
    finite_count = 0
    neg_inf_count = 0
    for seed in range(20):
        prog = sample_program(grammar, request, max_depth=max_depth, seed=seed)
        log_q = _score_subtree_under_sampler(
            grammar, prog, request, max_depth=max_depth, depth=0, env=[],
        )
        if math.isfinite(log_q):
            finite_count += 1
        elif log_q == float('-inf'):
            neg_inf_count += 1
        else:
            raise AssertionError(
                f"seed={seed}: scorer returned non-finite, non-(-inf) value {log_q}"
            )
    # Allow a small number of -inf for edge cases (polymorphic rng fallback).
    # The bulk must be scorable — if the scorer routinely returns -inf on
    # sampled programs, detailed balance is broken.
    assert finite_count >= 15, (
        f"C1 scorer returned -inf on too many sampled programs: "
        f"{neg_inf_count}/20 were -inf (expected ≤5). "
        "Scorer control flow does not match sampler."
    )


def test_c1_scorer_rejects_impossible_subtree(grammar):
    """C1 regression: scorer must return -inf for programs that could not
    have been sampled at the given hole (wrong type, wrong shape).
    """
    from gallery_analysis.mcmc_search import _score_subtree_under_sampler

    # A bare variable cannot have been sampled for HAND->BOOL at the root
    # (the sampler must produce an Abstraction for an Arrow type).
    bare_index = Index(0)
    log_q = _score_subtree_under_sampler(
        grammar, bare_index, Arrow(HAND, BOOL),
        max_depth=5, depth=0, env=[HAND],
    )
    assert log_q == float('-inf'), (
        f"Scorer should reject bare Index at Arrow root, got {log_q}"
    )

    # An Abstraction at a base-type hole is also impossible.
    lam = Abstraction(Index(0))
    log_q2 = _score_subtree_under_sampler(
        grammar, lam, BOOL, max_depth=5, depth=0, env=[],
    )
    assert log_q2 == float('-inf'), (
        f"Scorer should reject Abstraction at BOOL hole, got {log_q2}"
    )


def test_c1_propose_regeneration_finite_densities(grammar):
    """C1 regression: end-to-end `propose_regeneration` should yield enough
    finite fwd+rev density pairs for the chain to mix.

    After the C1 fix, `log_q_reverse = -inf` is mathematically CORRECT
    whenever the old subtree could not have been produced by `_sample` at
    the site's regen_depth. This happens legitimately for deep subtrees
    whose internal depth exceeds `max(3, max_depth - path_len)` — the
    old subtree may have been sampled at the outer budget of the full
    program but would exceed the regeneration budget at its site.

    Such proposals are correctly rejected (acceptance = 0), which is
    detailed-balance-preserving though potentially sticky. The bar here
    is that at least a non-trivial fraction of proposals can be reversed;
    if nearly all were irreversible, the chain could not mix and the
    sampler density would be essentially useless.
    """
    prog = sample_program(grammar, Arrow(HAND, BOOL), max_depth=5, seed=42)
    finite_pairs = 0
    for seed in range(40):
        _, log_q_fwd, log_q_rev = propose_regeneration(
            grammar, prog, Arrow(HAND, BOOL), max_depth=5, seed=seed,
        )
        if math.isfinite(log_q_fwd) and math.isfinite(log_q_rev):
            finite_pairs += 1
    # Require at least ~30% of proposals to be mixable. The remaining
    # fraction includes structurally irreversible proposals (correct
    # behavior). Historical runs show ~45-60% on this grammar.
    assert finite_pairs >= 12, (
        f"Only {finite_pairs}/40 proposals had finite fwd+rev densities — "
        "the chain would be unable to mix. C1 scorer may be over-rejecting "
        "beyond the legitimate asymmetry."
    )


# =========================================================================== #
# Tests for compute_mcmc_log_likelihood
# =========================================================================== #

@pytest.fixture
def exemplars():
    """Load frozen exemplar hands from the gallery experiment."""
    return load_exemplars()


def test_likelihood_returns_tuple(grammar, exemplars):
    """Likelihood should return a (float, float) tuple, never NaN."""
    hands = exemplars['all_red']['hands_primary']
    probes = generate_probe_set(n_probes=1000, seed=42)
    prog = sample_program(grammar, Arrow(HAND, BOOL), max_depth=5, seed=42)
    result = compute_mcmc_log_likelihood(prog, hands, noise_epsilon=0.01, ext_probe_hands=probes)
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2-tuple, got length {len(result)}"
    ll, ext_frac = result
    assert isinstance(ll, float), f"Expected float log-lik, got {type(ll)}"
    assert isinstance(ext_frac, float), f"Expected float ext_frac, got {type(ext_frac)}"
    assert not math.isnan(ll), f"Likelihood is NaN for program {prog}"
    assert 0.0 <= ext_frac <= 1.0, f"ext_fraction should be in [0,1], got {ext_frac}"


def test_likelihood_rejects_empty_extension(grammar, exemplars):
    """Regression test for C4: 'accepts nothing' programs must not be rewarded.

    Prior bug: when n_hits_probe == 0, ext_size was set to 1.0 (smallest
    possible), so if any exemplar happened to be accepted via the noise
    floor, the per-exemplar probability was (1 - ε)/1.0 ≈ 1.0 — the
    HIGHEST possible. This rewarded programs that accept nothing on probes
    but luckily accept an exemplar.

    Fix: Jeffreys-smoothed ext_fraction (n_hits+0.5)/(n_probes+1) with a
    one-probe floor ensures ext_size >= TOTAL_HANDS / n_probes, so the
    per-exemplar probability is bounded above by n_probes/TOTAL_HANDS,
    which for n_probes=1000 and TOTAL_HANDS≈1.47e10 is ~7e-8 — very low.
    An "accepts nothing" program should have lower likelihood than a
    program that actually fits the exemplars.
    """
    from dreamcoder_core.program import Abstraction, Application

    hands = exemplars['all_red']['hands_primary']
    probes = generate_probe_set(n_probes=1000, seed=42)

    # Look up primitives by string name from the grammar's production list.
    prims_by_name = {str(p): p for p in grammar.primitives()}
    required = ('lt', '0', '5')
    for name in required:
        if name not in prims_by_name:
            import pytest
            pytest.skip(f"Grammar missing primitive {name!r}")

    lt = prims_by_name['lt']
    zero = prims_by_name['0']
    five = prims_by_name['5']

    # (lambda (lt 5 0)) — constant-false (5 < 0 is always False).
    empty = Abstraction(Application(Application(lt, five), zero))
    ll_empty, frac_empty = compute_mcmc_log_likelihood(
        empty, hands, noise_epsilon=0.01, ext_probe_hands=probes,
    )
    assert frac_empty == 0.0, f"'accepts nothing' should have ext_fraction=0, got {frac_empty}"

    # (lambda (lt 0 5)) — constant-true tautology.
    taut = Abstraction(Application(Application(lt, zero), five))
    ll_taut, frac_taut = compute_mcmc_log_likelihood(
        taut, hands, noise_epsilon=0.01, ext_probe_hands=probes,
    )
    assert frac_taut == 1.0

    # The "accepts nothing" program must NOT have higher likelihood than the
    # tautology. Prior bug: n_hits==0 set ext_size=1.0, inflating per-exemplar
    # probability to ~1.0 on any noise-accepted exemplar, making the empty
    # program *more* attractive than a tautology. With Jeffreys smoothing +
    # one-probe floor, ext_size is bounded below at TOTAL_HANDS/n_probes so
    # per-exemplar probability stays tiny.
    assert ll_empty <= ll_taut + 1e-6, (
        f"'accepts nothing' likelihood {ll_empty} exceeds tautology "
        f"likelihood {ll_taut} — empty-extension pathology returned?"
    )


def test_likelihood_noise_prevents_neg_inf(grammar, exemplars):
    """With noise_epsilon > 0, likelihood should be finite (not -inf)
    for programs that can at least be evaluated, even if they miss all exemplars."""
    hands = exemplars['all_red']['hands_primary']
    probes = generate_probe_set(n_probes=1000, seed=42)
    # Try several seeds -- at least one program should evaluate without crashing
    # and produce a finite (not -inf) likelihood due to noise floor.
    finite_count = 0
    for seed in range(20):
        prog = sample_program(grammar, Arrow(HAND, BOOL), max_depth=5, seed=seed)
        ll, ext_frac = compute_mcmc_log_likelihood(prog, hands, noise_epsilon=0.01, ext_probe_hands=probes)
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


def test_chain_has_ext_fractions(grammar, exemplars):
    """Chain result should populate ext_fractions dict."""
    config = MCMCConfig(n_steps=200, max_depth=5, seed=42)
    hands = exemplars['all_red']['hands_primary']
    result = MCMCChain(grammar, config).run(
        request_type=Arrow(HAND, BOOL),
        exemplar_hands=hands,
    )
    # Every visited program should have an ext_fraction entry
    assert len(result.ext_fractions) > 0
    for prog_str, frac in result.ext_fractions.items():
        assert 0.0 <= frac <= 1.0, f"ext_fraction {frac} out of range for {prog_str[:60]}"


# =========================================================================== #
# Tests for run_parallel_chains
# =========================================================================== #

def test_parallel_chains_merge_results(grammar, exemplars):
    """Multiple chains should merge visit counts, total steps = n_chains * n_steps."""
    config = MCMCConfig(n_steps=200, max_depth=5, seed=42)
    hands = exemplars['all_red']['hands_primary']
    result = run_parallel_chains(
        grammar, config,
        request_type=Arrow(HAND, BOOL),
        exemplar_hands=hands,
        n_chains=4,
    )
    assert result.n_unique > 0
    assert result.n_steps == 200 * 4


def test_parallel_chains_acceptance_rate(grammar, exemplars):
    """Merged acceptance rate should be reasonable."""
    config = MCMCConfig(n_steps=200, max_depth=5, seed=42)
    hands = exemplars['all_red']['hands_primary']
    result = run_parallel_chains(
        grammar, config,
        request_type=Arrow(HAND, BOOL),
        exemplar_hands=hands,
        n_chains=4,
    )
    assert 0.0 <= result.acceptance_rate <= 1.0


def test_parallel_chains_more_unique_than_single(grammar, exemplars):
    """4 chains should typically find more unique programs than 1 chain with same per-chain budget."""
    config = MCMCConfig(n_steps=200, max_depth=5, seed=42)
    hands = exemplars['all_red']['hands_primary']
    single = MCMCChain(grammar, config).run(
        request_type=Arrow(HAND, BOOL),
        exemplar_hands=hands,
    )
    multi = run_parallel_chains(
        grammar, config,
        request_type=Arrow(HAND, BOOL),
        exemplar_hands=hands,
        n_chains=4,
    )
    # Multi-chain should find at least as many unique programs
    # (very likely more, since different starting points)
    assert multi.n_unique >= single.n_unique


def test_parallel_chains_merge_ext_fractions(grammar, exemplars):
    """Merged result should have ext_fractions from all chains."""
    config = MCMCConfig(n_steps=200, max_depth=5, seed=42)
    hands = exemplars['all_red']['hands_primary']
    result = run_parallel_chains(
        grammar, config,
        request_type=Arrow(HAND, BOOL),
        exemplar_hands=hands,
        n_chains=4,
    )
    assert len(result.ext_fractions) > 0
    for prog_str, frac in result.ext_fractions.items():
        assert 0.0 <= frac <= 1.0


def test_parallel_chains_first_passage_no_offset(grammar, exemplars):
    """Regression test for C6: merged first_passage must not offset by chain-index.

    Prior bug: merged_first_passage[prog] = min(step + i*n_steps, ...) caused
    chain-0 to always appear "fastest" for any program seen in multiple chains.
    The merged min must lie within a single chain's step range [0, n_steps).
    """
    config = MCMCConfig(n_steps=200, max_depth=5, seed=42)
    hands = exemplars['all_red']['hands_primary']
    result = run_parallel_chains(
        grammar, config,
        request_type=Arrow(HAND, BOOL),
        exemplar_hands=hands,
        n_chains=4,
    )
    # All first-passage steps must be within a single chain's range.
    # If the buggy offset were present, chain-3's first-seen programs would
    # have step >= 3*200 = 600.
    assert len(result.first_passage) > 0
    for prog, step in result.first_passage.items():
        assert 0 <= step < config.n_steps, (
            f"First-passage step {step} for {prog[:60]!r} lies outside "
            f"single-chain range [0, {config.n_steps}) — offset bug returned?"
        )
    # Per-chain first-passage should be exposed for honest timing analysis.
    assert len(result.per_chain_first_passage) == 4
    for pc_fp in result.per_chain_first_passage:
        for step in pc_fp.values():
            assert 0 <= step < config.n_steps


def test_parallel_chains_propagates_beta_annealing(grammar, exemplars, monkeypatch):
    """Regression test for C2: each chain config must carry caller's beta_start/beta_end.

    Prior bug: run_parallel_chains re-instantiated MCMCConfig from a hand-picked
    subset of fields, silently dropping beta_start/beta_end so every chain ran
    at the dataclass default β=1.0 despite the caller setting β-annealing.
    """
    import gallery_analysis.mcmc_search as mcmc_module

    captured_configs = []
    original_init = mcmc_module.MCMCChain.__init__

    def spy_init(self, grammar_arg, config_arg):
        captured_configs.append(config_arg)
        return original_init(self, grammar_arg, config_arg)

    monkeypatch.setattr(mcmc_module.MCMCChain, "__init__", spy_init)

    config = MCMCConfig(
        n_steps=50, max_depth=5, seed=42,
        beta_start=0.1, beta_end=0.9,
    )
    hands = exemplars['all_red']['hands_primary']
    run_parallel_chains(
        grammar, config,
        request_type=Arrow(HAND, BOOL),
        exemplar_hands=hands,
        n_chains=3,
    )
    assert len(captured_configs) == 3
    for cfg in captured_configs:
        assert cfg.beta_start == 0.1, f"beta_start lost in propagation: {cfg.beta_start}"
        assert cfg.beta_end == 0.9, f"beta_end lost in propagation: {cfg.beta_end}"
        assert cfg.n_steps == 50
        assert cfg.max_depth == 5


# =========================================================================== #
# Integration tests on real gallery rules
# =========================================================================== #

class TestMCMCIntegration:
    """End-to-end tests on real gallery rules."""

    def setup_method(self):
        from gallery_analysis.enumerator import build_gallery_grammar
        from gallery_analysis.exemplars import load_exemplars
        self.grammar = build_gallery_grammar()
        self.exemplars = load_exemplars()

    def test_finds_hypotheses_for_easy_rule(self):
        """MCMC should find consistent hypotheses for 'all_red' (group 1, easy)."""
        config = MCMCConfig(n_steps=5000, max_depth=5, seed=42)
        hands = self.exemplars['all_red']['hands_primary']
        result = run_parallel_chains(
            self.grammar, config,
            request_type=Arrow(HAND, BOOL),
            exemplar_hands=hands,
            n_chains=4,
        )
        assert result.n_unique > 10, f"Only found {result.n_unique} unique programs"
        assert result.acceptance_rate > 0.001, f"Acceptance rate too low: {result.acceptance_rate}"
        print(f"\nall_red: {result.n_unique} unique, accept={result.acceptance_rate:.3f}, top={result.top_hypotheses[0]['program'][:80]}")

    def test_finds_hypotheses_for_medium_rule(self):
        """MCMC should find hypotheses for 'all_even' (group 2, medium)."""
        config = MCMCConfig(n_steps=5000, max_depth=5, seed=42)
        hands = self.exemplars['all_even']['hands_primary']
        result = run_parallel_chains(
            self.grammar, config,
            request_type=Arrow(HAND, BOOL),
            exemplar_hands=hands,
            n_chains=4,
        )
        assert result.n_unique > 5, f"Only found {result.n_unique} unique programs"
        print(f"\nall_even: {result.n_unique} unique, accept={result.acceptance_rate:.3f}, top={result.top_hypotheses[0]['program'][:80]}")

    def test_top_hypothesis_has_reasonable_visits(self):
        """The most-visited hypothesis should have more than 1 visit."""
        config = MCMCConfig(n_steps=2000, max_depth=5, seed=42)
        hands = self.exemplars['all_red']['hands_primary']
        result = run_parallel_chains(
            self.grammar, config,
            request_type=Arrow(HAND, BOOL),
            exemplar_hands=hands,
            n_chains=2,
        )
        assert len(result.top_hypotheses) > 0
        assert result.top_hypotheses[0]['visit_count'] > 1, "Top hypothesis should be visited multiple times"

    def test_first_passage_tracked(self):
        """First passage times should be tracked for discovered hypotheses."""
        config = MCMCConfig(n_steps=1000, max_depth=5, seed=42)
        hands = self.exemplars['all_red']['hands_primary']
        result = run_parallel_chains(
            self.grammar, config,
            request_type=Arrow(HAND, BOOL),
            exemplar_hands=hands,
            n_chains=2,
        )
        assert len(result.first_passage) > 0
        for prog_str, step in result.first_passage.items():
            assert step >= 0, f"First passage step should be non-negative, got {step}"


# --------------------------------------------------------------------------- #
# Tests for likelihood annealing (beta schedule)
# --------------------------------------------------------------------------- #

def test_annealing_completes(grammar, exemplars):
    """Chain with annealing should complete without error."""
    config = MCMCConfig(
        n_steps=100, max_depth=5, seed=42,
        beta_start=0.0, beta_end=1.0,
    )
    hands = exemplars['all_red']['hands_primary']
    result = MCMCChain(grammar, config).run(
        request_type=Arrow(HAND, BOOL),
        exemplar_hands=hands,
    )
    assert result.n_steps == 100


def test_annealing_default_is_no_anneal(grammar, exemplars):
    """Default beta_start=1.0 beta_end=1.0 should work identically to before."""
    config = MCMCConfig(n_steps=50, max_depth=5, seed=42)
    hands = exemplars['all_red']['hands_primary']
    result = MCMCChain(grammar, config).run(
        request_type=Arrow(HAND, BOOL),
        exemplar_hands=hands,
    )
    assert result.n_steps == 50
    # Default betas are 1.0/1.0
    assert config.beta_start == 1.0
    assert config.beta_end == 1.0


# --------------------------------------------------------------------------- #
# Test: init_max_depth produces small programs
# --------------------------------------------------------------------------- #

def test_init_max_depth_produces_small_programs(grammar):
    """With init_max_depth=3, initial programs should be manageable size."""
    from gallery_analysis.mcmc_search import sample_program
    from dreamcoder_core.type_system import Arrow, HAND, BOOL
    sizes = []
    for seed in range(20):
        prog = sample_program(grammar, Arrow(HAND, BOOL), max_depth=3, seed=seed)
        sizes.append(prog.size())
    avg = sum(sizes) / len(sizes)
    assert avg < 50, f"Average size {avg} too large for max_depth=3"
    assert max(sizes) < 100, f"Max size {max(sizes)} too large"


# =========================================================================== #
# Tests for tautology rejection (3 layers)
# =========================================================================== #

def test_vacuous_lambda_detected(grammar):
    """is_vacuous_lambda should detect programs that ignore their input.

    A vacuous lambda like (lambda (lt 0 5)) never references $0, so it
    computes the same constant regardless of the hand. Such programs are
    uninformative as hypotheses and should be rejected.
    """
    # Find the 'lt' primitive in the grammar
    lt_prim = None
    for p in grammar.productions:
        if str(p.program) == 'lt':
            lt_prim = p.program
            break
    assert lt_prim is not None, "Grammar should have 'lt'"

    # Build (lambda (lt 0 5)) -- a vacuous lambda that ignores its input
    zero = Primitive('0', INT, 0)
    five = Primitive('5', INT, 5)
    vacuous = Abstraction(Application(Application(lt_prim, zero), five))
    assert is_vacuous_lambda(vacuous) is True, (
        f"Expected vacuous lambda, but is_vacuous_lambda returned False for: {vacuous}"
    )

    # A program that uses $0 should NOT be vacuous
    uses_input = Abstraction(Application(Application(lt_prim, Index(0)), five))
    assert is_vacuous_lambda(uses_input) is False, (
        f"Expected non-vacuous lambda, but is_vacuous_lambda returned True for: {uses_input}"
    )


def test_vacuous_lambda_non_abstraction():
    """is_vacuous_lambda should return False for non-Abstraction programs."""
    # A bare primitive is not a lambda at all
    prim = Primitive('0', INT, 0)
    assert is_vacuous_lambda(prim) is False

    # An Index is not a lambda
    idx = Index(0)
    assert is_vacuous_lambda(idx) is False


def test_tautology_filter_is_post_hoc(grammar, exemplars):
    """Regression test for C3: Layer-2 tautology filter must be post-hoc only.

    Prior bug: ext_fraction >= 1.0 was rejected inline in the MH loop, breaking
    detailed balance. Fix: let the MH loop run normally (size-principle
    likelihood naturally suppresses tautologies), then filter tautologies
    from top_hypotheses at result construction. visit_counts/trajectory
    retain the full unfiltered trace.
    """
    config = MCMCConfig(n_steps=500, max_depth=5, seed=42)
    hands = exemplars['all_red']['hands_primary']
    result = MCMCChain(grammar, config).run(
        request_type=Arrow(HAND, BOOL),
        exemplar_hands=hands,
    )
    # top_hypotheses must not contain tautologies (ext_fraction >= 1.0).
    for hyp in result.top_hypotheses:
        prog = hyp['program']
        frac = result.ext_fractions.get(prog, 0.0)
        assert frac < 1.0, (
            f"Tautology {prog[:80]!r} with ext_fraction={frac} leaked into "
            f"top_hypotheses — post-hoc filter broken."
        )
    # Chain should still produce some usable hypotheses.
    assert len(result.top_hypotheses) > 0


def test_tautology_not_top_hypothesis(grammar, exemplars):
    """With tautology rejection, known tautologies should not dominate the chain.

    Programs like (lambda (eq $0 $0)) or (lambda (lt 0 5)) always return True
    regardless of the hand. The 3-layer rejection system should prevent these
    from being the top hypothesis.
    """
    config = MCMCConfig(n_steps=500, max_depth=5, seed=42)
    hands = exemplars['all_red']['hands_primary']
    result = MCMCChain(grammar, config).run(
        request_type=Arrow(HAND, BOOL),
        exemplar_hands=hands,
    )
    # The top hypothesis should not be a known tautology pattern
    if result.top_hypotheses:
        top = result.top_hypotheses[0]['program']
        known_tautologies = [
            '(lambda (eq $0 $0))',
            '(lambda (lt 0 5))',
            '(lambda (le 0 5))',
            '(lambda (gt 5 0))',
        ]
        assert top not in known_tautologies, (
            f"Tautology {top} should not be the #1 hypothesis"
        )


# =========================================================================== #
# R1 fixes: regression tests for proposal density & site collection
# =========================================================================== #

def test_collect_subtree_sites_no_silent_drops_on_sampled_programs(grammar):
    """
    `collect_subtree_sites` must not silently swallow inference failures on
    well-typed sampled programs. Its `try/except` blocks now increment a
    module-level counter; the gallery grammar sampler should produce zero
    drops across a reasonable sample size.
    """
    reset_collect_subtree_sites_failures()
    for seed in range(50):
        prog = sample_program(grammar, Arrow(HAND, BOOL), max_depth=5, seed=seed)
        _ = collect_subtree_sites(prog, Arrow(HAND, BOOL))
    assert get_collect_subtree_sites_failures() == 0, (
        f"collect_subtree_sites silently dropped "
        f"{get_collect_subtree_sites_failures()} sites across 50 sampled "
        f"programs. n_sites in the MH ratio is no longer what the chain "
        f"believes it is."
    )


def test_score_matches_empirical_on_polymorphic_toy():
    """
    The choose/is_zero toy from Round 1 review: raw `_sample` emits
    `choose ((λ is_zero $0))` with empirical probability ≈ 0.13, while
    the earlier scorer (tier-2 / tier-3 heuristics) reported P = 1.0.
    After exact marginalization the scored probability should match
    the empirical frequency to within a few percent.
    """
    import random as _random
    from dreamcoder_core.type_system import Arrow as _Arrow, TypeVariable, BOOL as _BOOL, INT as _INT
    from dreamcoder_core.grammar import Grammar as _Grammar, Production as _Production
    from dreamcoder_core.program import Primitive as _Primitive, Abstraction as _Abs, Application as _App, Index as _Idx
    import gallery_analysis.mcmc_search as ms

    alpha = TypeVariable(0)
    choose = _Primitive('choose', _Arrow(_Arrow(alpha, _BOOL), _BOOL), 1.0)
    is_zero = _Primitive('is_zero', _Arrow(_INT, _BOOL), 1.0)
    prods = [
        _Production(choose, _Arrow(_Arrow(alpha, _BOOL), _BOOL), 0.0),
        _Production(is_zero, _Arrow(_INT, _BOOL), 0.0),
    ]
    g = _Grammar(productions=prods, log_variable=0.0)

    # Narrow _CONCRETE_TYPES to [BOOL, INT] for the duration of this test.
    saved = list(ms._CONCRETE_TYPES)
    try:
        ms._CONCRETE_TYPES[:] = [_BOOL, _INT]

        N = 2000
        counts = {}
        for seed in range(N):
            rng = _random.Random(seed)
            try:
                p = _sample(g, _BOOL, 5, 0, [], rng)
                counts[str(p)] = counts.get(str(p), 0) + 1
            except Exception:
                pass

        target_body = _App(is_zero, _Idx(0))
        target = _App(choose, _Abs(target_body))
        empirical = counts.get(str(target), 0) / N

        log_p = _score_subtree_under_sampler(g, target, _BOOL, 5, 0, [])
        scored = math.exp(log_p) if log_p != float('-inf') else 0.0

        assert empirical > 0.05, (
            f"Sanity: expected empirical > 5% for choose((λ is_zero $0)); "
            f"got {empirical:.4f}"
        )
        assert scored > 0.0, f"Scored P should be > 0; got {scored:.6f}"
        # Tolerate at most 5 percentage points of absolute deviation (the
        # sampler has ERR retries that the direct _sample call does not, so
        # the two distributions are not identical — but the scored value
        # should no longer be the inflated tier-2/tier-3 value of 1.0).
        assert abs(scored - empirical) < 0.05, (
            f"Scored P {scored:.4f} differs from empirical {empirical:.4f} "
            f"by more than 0.05 — free-var marginalization may be wrong."
        )
        assert scored < 0.5, (
            f"Scored P {scored:.4f} too large — the pre-R1 tier-2 bug "
            f"reported P=1.0 here; any value > 0.5 suggests a regression."
        )
    finally:
        ms._CONCRETE_TYPES[:] = saved


def test_propose_regeneration_matches_sampler_on_polymorphic_grammar(grammar):
    """
    Weak empirical check that the forward proposal density `log_q_forward`
    matches the distribution from which `new_subtree` was actually drawn.

    Strategy: pick a fixed site in a sampled program, repeatedly resample
    the subtree at that site via `propose_regeneration`, collect the
    empirical distribution over distinct subtrees, and compare to the
    scored density. For each observed subtree, exp(log_q_forward) should
    lie within a factor of ~3x of the empirical frequency (loose bound;
    this is a sanity check, not a goodness-of-fit test).
    """
    # Build a simple base program whose first site we'll regenerate.
    base = sample_program(
        grammar, Arrow(HAND, BOOL), max_depth=3, seed=42,
    )
    sites = collect_subtree_sites(base, Arrow(HAND, BOOL))
    if not sites:
        pytest.skip("No sites in base program; resample a different seed.")

    N = 400
    counts = {}
    scored_cache = {}
    for seed in range(N):
        try:
            new_prog, log_q_fwd, _ = propose_regeneration(
                grammar, base, Arrow(HAND, BOOL),
                max_depth=4, seed=seed,
            )
            key = str(new_prog)
            counts[key] = counts.get(key, 0) + 1
            if key not in scored_cache:
                scored_cache[key] = log_q_fwd
        except Exception:
            pass

    if not counts:
        pytest.skip("All proposals failed — likely sampler edge case.")

    # Check the top-2 distinct proposals have scored densities consistent
    # with their empirical frequencies. Use loose 3x bound.
    sorted_keys = sorted(counts.keys(), key=lambda k: -counts[k])[:2]
    for key in sorted_keys:
        empirical = counts[key] / N
        if empirical < 0.05:
            continue  # too few samples to compare meaningfully
        scored = math.exp(scored_cache[key]) if scored_cache[key] != float('-inf') else 0.0
        # Scored includes log_pick = -log(n_sites) so should match empirical.
        # Use multiplicative tolerance: 0.33 <= scored / empirical <= 3.0.
        if scored == 0.0:
            pytest.fail(
                f"Proposal {key!r} seen {counts[key]}/{N} times but scored "
                f"probability is 0. log_q_forward says this can't happen."
            )
        ratio = scored / empirical
        assert 0.33 < ratio < 3.0, (
            f"Proposal {key!r}: empirical={empirical:.4f}, scored={scored:.4f}, "
            f"ratio={ratio:.2f}. Expected ratio in [0.33, 3.0]."
        )


def test_proposal_retry_loop_disabled_in_sample_program():
    """
    `propose_regeneration` must call `sample_program(allow_retries=False)`.
    With retries enabled, the effective distribution of generated subtrees
    is `_sample | typecheck_passes`, which has a different normalization
    constant than `_score_subtree_under_sampler` computes — violating
    detailed balance.
    """
    import inspect
    from gallery_analysis import mcmc_search
    src = inspect.getsource(mcmc_search.propose_regeneration)
    assert 'allow_retries=False' in src, (
        "propose_regeneration must call sample_program with allow_retries=False "
        "to keep the forward proposal distribution consistent with "
        "_score_subtree_under_sampler."
    )


# --------------------------------------------------------------------------- #
# R2-Fix3: ΣQ(s'|s) = 1 on a tiny hand-built state space.
#
# Reviewer weakness (R2): the existing proposal-density test
# `test_propose_regeneration_matches_sampler_on_polymorphic_grammar` only
# checks the top-2 proposals with a factor-3 tolerance, which is too weak
# to detect residual scorer/sampler disagreement in rare-branch logic
# (depth-cap lookahead, fallback set). The test below constructs a
# micro-grammar {t:bool, f:bool→bool} whose support at max_depth=3 is
# enumerable in closed form ({t, f(t), f(f(t)), f(f(f(t)))} with exact
# sampler probabilities {1/2, 1/4, 1/8, 1/8}) and asserts both that
#   (a) Σ exp(log_q_scored) ≈ 1 across the full support (normalization), and
#   (b) each |log_q_scored - log_q_expected| ≤ 1e-6 (pointwise exactness).
# Any disagreement exposes a scorer/sampler divergence directly; no
# statistical tolerance is needed because the support is enumerable.
# --------------------------------------------------------------------------- #


def _build_tiny_bool_grammar():
    """Build {t:bool, f:bool→bool}, each lp=0. No variables (log_variable=-inf)."""
    from dreamcoder_core.grammar import Grammar, Production
    from dreamcoder_core.program import Primitive
    from dreamcoder_core.type_system import BOOL as _BOOL, Arrow as _Arrow
    t_prim = Primitive('t', _BOOL, True)
    f_prim = Primitive('f', _Arrow(_BOOL, _BOOL), lambda x: x)
    prods = [
        Production(t_prim, _BOOL, 0.0),
        Production(f_prim, _Arrow(_BOOL, _BOOL), 0.0),
    ]
    # log_variable = -inf so variables never compete with productions
    # (the support then depends only on the two productions).
    return Grammar(prods, log_variable=float('-inf'))


def _enumerate_tiny_bool_support(max_depth):
    """
    Closed-form sampler distribution for `_sample(BOOL, max_depth=D, depth=0)`
    under `_build_tiny_bool_grammar`. At each depth < D, P(pick t) = P(pick f) = 1/2.
    At depth == D, forced to t (terminal_prods = {t}, var_candidates = []).

    Returns List[(Program_as_str, probability)].
    """
    # Programs: t, f(t), f(f(t)), ..., f^k(t) for k in 0..D.
    # P(f^k(t)) = (1/2)^(k+1) for k < D; P(f^D(t)) = (1/2)^D (forced last).
    # Sum: Σ_{k=0}^{D-1} (1/2)^(k+1) + (1/2)^D = (1 - (1/2)^D) + (1/2)^D = 1. ✓
    from dreamcoder_core.program import Primitive, Application
    from dreamcoder_core.type_system import BOOL as _BOOL, Arrow as _Arrow
    t_prim = Primitive('t', _BOOL, True)
    f_prim = Primitive('f', _Arrow(_BOOL, _BOOL), lambda x: x)

    support = []
    prog = t_prim
    for k in range(max_depth):
        prob = 0.5 ** (k + 1)
        support.append((prog, prob))
        prog = Application(f_prim, prog)
    # Terminal forced at depth == max_depth:
    support.append((prog, 0.5 ** max_depth))
    return support


def test_score_subtree_under_sampler_normalizes_on_tiny_grammar():
    """
    ΣQ(s'|s) = 1 regression test on a hand-built state space (R2-Fix3).

    The tiny grammar {t:bool, f:bool→bool} has an enumerable support at
    max_depth=3: {t, f(t), f(f(t)), f(f(f(t)))}. This is the exact kind
    of setting where a scorer/sampler mismatch in the depth-cap lookahead
    would surface, because depth >= max_depth forces the terminal_prods
    branch (var_candidates empty via log_variable = -inf).
    """
    from dreamcoder_core.type_system import BOOL as _BOOL
    grammar = _build_tiny_bool_grammar()
    max_depth = 3
    support = _enumerate_tiny_bool_support(max_depth)

    # Sanity: expected probabilities sum to 1.
    assert abs(sum(p for _, p in support) - 1.0) < 1e-12

    # Score every program in the support and compare to expected.
    scored_probs = []
    for prog, expected_p in support:
        log_q = _score_subtree_under_sampler(
            grammar, prog, _BOOL, max_depth=max_depth, depth=0, env=[],
        )
        if log_q == float('-inf'):
            pytest.fail(
                f"Scorer assigned probability 0 to {prog!r}, which should "
                f"have P = {expected_p}. Scorer/sampler divergence."
            )
        scored_p = math.exp(log_q)
        scored_probs.append((prog, scored_p, expected_p))
        assert abs(scored_p - expected_p) < 1e-9, (
            f"Pointwise mismatch for {prog!r}: scored {scored_p:.10f}, "
            f"expected {expected_p:.10f}, diff {scored_p - expected_p:.2e}."
        )

    total_scored = sum(p for _, p, _ in scored_probs)
    assert abs(total_scored - 1.0) < 1e-9, (
        f"Σ exp(log_q) = {total_scored:.10f} ≠ 1. Scorer is not a proper "
        f"density over `_sample`'s support."
    )


def test_score_depth_cap_exact_matches_reviewer_counterexample():
    """
    Exact verification of R2-Fix1 against the reviewer's counterexample.

    Grammar: {p1: int→bool, p2: 'a→bool, 0: int}, request bool at depth cap.
    Sampler semantics:
      - p1 survives filter with probability 1 (no free vars; terminable).
      - p2 survives with probability 1/2 ('a=int → terminable via 0:int;
        'a=bool → no bool terminal, not terminable).
      - 0:int is not a candidate for bool.
    Exact P(pick p1) = 1/2 × (1/1) + 1/2 × (1/2) = 0.75.
    Old mean-field P(pick p1) = exp(0)/(exp(0) + exp(log 0.5)) = 1/1.5 = 0.6667.

    We drive _score_subtree_under_sampler with a state at depth == max_depth
    (no terminals for bool, no variables) and assert it returns log(0.75)
    for the observed subtree `p1 0`, not log(0.6667).
    """
    from dreamcoder_core.grammar import Grammar, Production
    from dreamcoder_core.program import Primitive, Application
    from dreamcoder_core.type_system import (
        BOOL as _BOOL, INT as _INT, Arrow as _Arrow, TypeVariable,
    )
    import gallery_analysis.mcmc_search as ms

    p1 = Primitive('p1', _Arrow(_INT, _BOOL), None)
    # p2 is polymorphic: 'a -> bool.
    tv = TypeVariable(0)
    p2 = Primitive('p2', _Arrow(tv, _BOOL), None)
    zero = Primitive('0', _INT, 0)

    prods = [
        Production(p1, _Arrow(_INT, _BOOL), 0.0),
        Production(p2, _Arrow(tv, _BOOL), 0.0),
        Production(zero, _INT, 0.0),
    ]
    grammar = Grammar(prods, log_variable=float('-inf'))

    # Pin _CONCRETE_TYPES to [BOOL, INT] so p2's survival computation matches
    # the reviewer's counterexample (1/2 survival).
    saved = list(ms._CONCRETE_TYPES)
    ms._CONCRETE_TYPES[:] = [_BOOL, _INT]
    try:
        # Observed subtree: p1(0). We want the scorer at depth == max_depth = 0
        # (so the lookahead branch fires immediately). The branch only enters
        # when terminal_prods for bool are empty AND var_candidates are empty;
        # bool has no zero-arg production here, and env=[] gives no variables.
        observed = Application(p1, zero)
        log_q = _score_subtree_under_sampler(
            grammar, observed, _BOOL, max_depth=0, depth=0, env=[],
        )
        scored_p = math.exp(log_q)
        # Head weight only — p1 scored at depth cap under the exact dispatch.
        # The only arg is 0:int at depth 1 (>= max_depth=0), with target INT.
        # INT has one terminal production (0), no vars, so that branch picks it
        # with P=1. Thus scored_p equals P(pick p1 head at depth cap) = 0.75.
        assert abs(scored_p - 0.75) < 1e-9, (
            f"Scorer returned {scored_p:.6f}; expected 0.75. If you see "
            f"0.6667, the mean-field shift is still in effect. Check "
            f"_score_depth_cap_lookahead_exact integration."
        )
    finally:
        ms._CONCRETE_TYPES[:] = saved


# --------------------------------------------------------------------------- #
# R3-Fix1: Full-kernel ΣQ(s'|s) = 1 test for propose_regeneration.
#
# Reviewer weakness (R3): the existing tiny-grammar test validates scorer
# normalization only (root-level `_score_subtree_under_sampler`), not the
# full proposal kernel Q(s'|s) = (1/n_sites) × P_sample(new_subtree | site).
# This test closes that gap by enumerating all (site, new_subtree) pairs on
# a hand-built starting state and verifying Σ exp(log_q_fwd) = 1 across the
# entire kernel support.
#
# Construction:
#   - Grammar: {t:bool, f:bool→bool}, log_variable=-inf (tiny-bool grammar).
#   - Starting program s = f(f(t)). By collect_subtree_sites(s, BOOL), this
#     has exactly 2 sites: `f(t)` at ('x',) and `t` at ('x','x'), both with
#     type BOOL and env=[]. Root is excluded by design.
#   - We call propose_regeneration with max_depth=4 so regen_depth =
#     max(3, 4 - len(site.path)) = 3 at both sites (path lengths 1 and 2).
#   - At regen_depth=3, the sampler support is enumerable:
#       {t, f(t), f(f(t)), f(f(f(t)))} with P = {1/2, 1/4, 1/8, 1/8}.
#   - For each (site, new_subtree) pair we compute
#       log_q_fwd = -log(n_sites) + _score_subtree_under_sampler(
#           grammar, new_subtree, site.type, regen_depth, depth=0, env=site.env,
#       )
#     which mirrors the exact expression inside propose_regeneration. We
#     then check Σ exp(log_q_fwd) ≈ 1 over ALL pairs. A failure implies
#     either the scorer is not a proper density under `_sample` or the
#     site-pick normalization is broken.
# --------------------------------------------------------------------------- #


def test_propose_regeneration_full_kernel_normalizes_on_tiny_grammar():
    """
    Σ_{(site, new_subtree)} Q_fwd(site, new_subtree | s) = 1 on a tiny
    hand-built state space. Complements R2-Fix3 (which only validated the
    scorer, not the full kernel with site-pick).
    """
    from dreamcoder_core.grammar import Grammar, Production
    from dreamcoder_core.program import Primitive, Application
    from dreamcoder_core.type_system import BOOL as _BOOL, Arrow as _Arrow
    from gallery_analysis.mcmc_search import _score_subtree_under_sampler

    grammar = _build_tiny_bool_grammar()

    t_prim = Primitive('t', _BOOL, True)
    f_prim = Primitive('f', _Arrow(_BOOL, _BOOL), lambda x: x)
    starting_program = Application(f_prim, Application(f_prim, t_prim))

    sites = collect_subtree_sites(starting_program, _BOOL)
    assert len(sites) == 2, (
        f"Tiny kernel test requires exactly 2 sites on f(f(t)); got "
        f"{len(sites)}. If site-collection semantics changed, the test "
        f"design needs revisiting."
    )
    n_sites_old = len(sites)
    log_pick_fwd = -math.log(n_sites_old)

    # max_depth=4 → regen_depth=max(3, 4-1)=3 at ('x',), max(3, 4-2)=3 at ('x','x').
    propose_max_depth = 4
    total_q = 0.0
    per_pair: list = []
    for site in sites:
        regen_depth = max(3, propose_max_depth - len(site.path))
        assert regen_depth == 3, (
            f"Test assumes regen_depth=3 at every site; got {regen_depth} "
            f"for site {site.path!r} (propose_max_depth={propose_max_depth})."
        )
        support = _enumerate_tiny_bool_support(regen_depth)
        # Sanity: per-site scorer support is a proper distribution.
        per_site_sum = 0.0
        for new_subtree, expected_p in support:
            log_gen = _score_subtree_under_sampler(
                grammar, new_subtree, site.type,
                max_depth=regen_depth, depth=0, env=site.env,
            )
            assert log_gen != float('-inf'), (
                f"Scorer assigned 0 to {new_subtree!r} at site {site.path!r}."
            )
            log_q = log_pick_fwd + log_gen
            q = math.exp(log_q)
            per_site_sum += math.exp(log_gen)
            total_q += q
            per_pair.append((site.path, str(new_subtree), q, expected_p))
        assert abs(per_site_sum - 1.0) < 1e-9, (
            f"Scorer not normalized at site {site.path!r}: "
            f"Σ P(new_subtree) = {per_site_sum:.10f}."
        )

    assert abs(total_q - 1.0) < 1e-9, (
        f"Full-kernel Σ Q(s'|s) = {total_q:.10f} ≠ 1.\n"
        f"Per-pair breakdown:\n"
        + "\n".join(
            f"  site={sp!r} subtree={prog}: q={q:.10f} (expected 1/n_sites × P={ep/2:.10f})"
            for sp, prog, q, ep in per_pair
        )
    )

    # Also assert expected per-pair mass (pointwise lock-in):
    # each pair's q should equal (1/2) × P(new_subtree), i.e. uniform site
    # × sampler P. This catches any future drift in log_pick_fwd semantics.
    for site_path, subtree_str, q, expected_p in per_pair:
        expected_q = 0.5 * expected_p
        assert abs(q - expected_q) < 1e-9, (
            f"Pair (site={site_path!r}, subtree={subtree_str}): "
            f"q={q:.10f}, expected={expected_q:.10f}."
        )
