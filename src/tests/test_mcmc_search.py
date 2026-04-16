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
