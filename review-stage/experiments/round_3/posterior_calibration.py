"""
Posterior-calibration experiment for the C1 fix (Round 3).

Goal
----
Round 2 reviewer: "run chain on toy grammar with analytically tractable
posterior, verify empirical visit frequencies match within Monte-Carlo
error." This confirms the MH ratio built from `_score_subtree_under_sampler`
(the C1 fix) actually yields a well-behaved sampler, not just algebraic
correctness on paper.

Setup
-----
* Toy grammar over BOOL: productions {not, and, or} with uniform
  log-probabilities, request type = BOOL -> BOOL.
* Analytical universe: enumerate all programs up to `MAX_DEPTH`; this is
  the support of a Metropolis chain that starts from such a program.
* Target distribution π(p) ∝ exp(log_prior(p) - LENGTH_BETA * length(p)).
  - `log_prior(p)` is the C1-fixed scorer evaluated at the root:
    `_score_subtree_under_sampler(grammar, p, request_type, MAX_DEPTH, 0, [])`.
    Using this as the "prior" is the natural Bayesian prior under the
    sampler's own distribution.
  - `LENGTH_BETA * length(p)` is a simple auxiliary log-likelihood that
    makes π NON-IDENTICAL to the proposal density: this way the MH
    accept/reject logic is actually exercised (otherwise every move
    accepts with probability 1 and detailed-balance is untestable).
* Proposal: `propose_regeneration` (the C1-fixed sampler path).
* Accept with min(1, exp((log π(new) - log π(old)) + (log_q_rev - log_q_fwd))).

Metric
------
Total-variation distance between empirical visit frequencies and analytical
π, computed over the support. Expected MC error at N samples (effective) is
O(1/sqrt(N_eff)); we report TV, the 95% MC threshold (Bernstein-style
bound on TV for multinomial), and a per-program ratio table.

Usage
-----
    python3 review-stage/experiments/round_3/posterior_calibration.py
"""

import json
import math
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from dreamcoder_core.grammar import Grammar, Production
from dreamcoder_core.primitives import Primitive
from dreamcoder_core.program import (
    Abstraction,
    Application,
    Index,
    Invented,
    Program,
)
from dreamcoder_core.type_system import BOOL, Arrow
from gallery_analysis.mcmc_search import (
    _score_subtree_under_sampler,
    collect_subtree_sites,
    propose_regeneration,
    sample_program,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_DEPTH = 3
LENGTH_BETA = 0.7          # per-node penalty in target log-density
N_STEPS = 200_000          # MH steps
N_BURNIN = 20_000          # discard
SEED = 20260416

OUT_DIR = Path(__file__).parent
OUT_JSON = OUT_DIR / "posterior_calibration_result.json"
OUT_LOG = OUT_DIR / "posterior_calibration_log.txt"


# ---------------------------------------------------------------------------
# Toy grammar
# ---------------------------------------------------------------------------
def build_toy_grammar() -> Grammar:
    """Uniform grammar over boolean ops, request = BOOL -> BOOL."""
    def _not(x):
        return not x

    def _and(x):
        return lambda y: x and y

    def _or(x):
        return lambda y: x or y

    prims = [
        Primitive('not', Arrow(BOOL, BOOL), _not),
        Primitive('and', Arrow(BOOL, Arrow(BOOL, BOOL)), _and),
        Primitive('or', Arrow(BOOL, Arrow(BOOL, BOOL)), _or),
    ]
    productions = [Production(p, p.tp, -1.0) for p in prims]
    return Grammar(productions, log_variable=-0.5)


# ---------------------------------------------------------------------------
# Enumeration of the support (all programs up to MAX_DEPTH)
# ---------------------------------------------------------------------------
def enumerate_programs(
    grammar: Grammar, target_type, max_depth: int, depth: int, env,
) -> List[Program]:
    """
    Enumerate every program of the given type reachable at the given depth.

    Mirrors `_sample` structurally, but over all choices: at Arrow types,
    wrap in Abstraction and recurse; at base types, enumerate every
    production + variable and recurse on arg slots.
    """
    if isinstance(target_type, Arrow):
        new_env = [target_type.arg] + list(env)
        body_depth = depth if depth < max_depth else depth + 1
        bodies = enumerate_programs(
            grammar, target_type.ret, max_depth, body_depth, new_env,
        )
        return [Abstraction(body) for body in bodies]

    programs: List[Program] = []

    # Variable leaves (only at/below max_depth; always allowed).
    for idx, var_type in enumerate(env):
        if var_type == target_type:
            programs.append(Index(idx))

    # Production heads.
    from dreamcoder_core.type_system import TypeContext

    ctx = TypeContext()
    prod_candidates = grammar.candidates_for_type(
        target_type, ctx, env, normalize=False,
    )

    # If at depth cap, keep only zero-arg productions (mirror _sample logic).
    at_cap = depth >= max_depth
    for prod, inst_type, _lp in prod_candidates:
        arg_types = list(inst_type.arguments)
        if at_cap and len(arg_types) > 0:
            continue
        if len(arg_types) == 0:
            programs.append(prod.program)
            continue

        # Recurse into each arg, Cartesian product.
        arg_options: List[List[Program]] = []
        skip = False
        for arg_type in arg_types:
            # We do NOT mirror `_sample`'s full free-var resolution for
            # enumeration — the toy grammar has no polymorphism, so arg_type
            # is concrete.
            subs = enumerate_programs(
                grammar, arg_type, max_depth, depth + 1, env,
            )
            if not subs:
                skip = True
                break
            arg_options.append(subs)
        if skip:
            continue

        def _product(options):
            if not options:
                yield []
                return
            head, *rest = options
            for h in head:
                for tail in _product(rest):
                    yield [h] + tail

        for combo in _product(arg_options):
            app: Program = prod.program
            for arg in combo:
                app = Application(app, arg)
            programs.append(app)

    return programs


def program_length(p: Program) -> int:
    """Tree-node count — matches `Program.show` tokenization roughly."""
    if isinstance(p, (Primitive, Invented, Index)):
        return 1
    if isinstance(p, Abstraction):
        return 1 + program_length(p.body)
    if isinstance(p, Application):
        return 1 + program_length(p.f) + program_length(p.x)
    raise TypeError(f"unexpected {type(p)}")


def canonical_str(p: Program) -> str:
    """Stable string key (lambda form)."""
    return str(p)


# ---------------------------------------------------------------------------
# Target density and analytical posterior
# ---------------------------------------------------------------------------
def log_target(p: Program, grammar: Grammar, request_type) -> float:
    """log π(p) = log_prior(p) - LENGTH_BETA * length(p)."""
    lp = _score_subtree_under_sampler(
        grammar, p, request_type, MAX_DEPTH, 0, [],
    )
    if lp == float('-inf'):
        return float('-inf')
    return lp - LENGTH_BETA * program_length(p)


def analytical_posterior(
    programs: List[Program], grammar: Grammar, request_type,
) -> dict:
    """Normalize target over enumerated support."""
    log_ps = {}
    for p in programs:
        lp = log_target(p, grammar, request_type)
        if math.isfinite(lp):
            log_ps[canonical_str(p)] = lp
    if not log_ps:
        return {}
    max_lp = max(log_ps.values())
    unnorm = {k: math.exp(v - max_lp) for k, v in log_ps.items()}
    Z = sum(unnorm.values())
    return {k: v / Z for k, v in unnorm.items()}


# ---------------------------------------------------------------------------
# Manual Metropolis-Hastings chain (C1-fixed proposal density)
# ---------------------------------------------------------------------------
def run_mh(
    grammar: Grammar,
    request_type,
    n_steps: int,
    n_burnin: int,
    seed: int,
):
    rng = random.Random(seed)

    # Seed a non-trivial starting program: sample until `propose_regeneration`
    # can find a site (i.e., program has ≥1 non-root subtree site). Starting
    # from `λ $0` leaves the chain with no moves because the body is a bare
    # Index at the root path.
    tries = 0
    while True:
        tries += 1
        if tries > 500:
            raise RuntimeError("cannot find starting program with regen sites")
        current = sample_program(
            grammar, request_type, max_depth=MAX_DEPTH, seed=seed + tries,
        )
        current_lt = log_target(current, grammar, request_type)
        if not math.isfinite(current_lt):
            continue
        try:
            propose_regeneration(
                grammar, current, request_type,
                max_depth=MAX_DEPTH,
                seed=seed + tries + 1000,
            )
            break
        except RuntimeError:
            continue

    visits = Counter()
    accept = 0
    proposed = 0

    for step in range(n_steps):
        try:
            new_program, log_q_fwd, log_q_rev = propose_regeneration(
                grammar, current, request_type,
                max_depth=MAX_DEPTH,
                seed=rng.randint(0, 2**31 - 1),
            )
        except RuntimeError:
            # No sites to regenerate.
            if step >= n_burnin:
                visits[canonical_str(current)] += 1
            continue

        new_lt = log_target(new_program, grammar, request_type)
        proposed += 1

        if not math.isfinite(new_lt) or not math.isfinite(log_q_fwd):
            # Reject: infinite forward density means sampler couldn't have
            # proposed this (shouldn't happen for a valid proposal), or
            # target rejects.
            pass
        else:
            if not math.isfinite(log_q_rev):
                # Reverse density -inf → proposal is irreversible; reject.
                pass
            else:
                log_alpha = (new_lt - current_lt) + (log_q_rev - log_q_fwd)
                if math.log(rng.random()) < log_alpha:
                    current = new_program
                    current_lt = new_lt
                    accept += 1

        if step >= n_burnin:
            visits[canonical_str(current)] += 1

    return visits, accept, proposed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    grammar = build_toy_grammar()
    request_type = Arrow(BOOL, BOOL)

    log(f"[{time.time()-t0:6.1f}s] enumerating programs up to depth={MAX_DEPTH} ...")
    programs = enumerate_programs(
        grammar, request_type, MAX_DEPTH, 0, [],
    )
    # Deduplicate by canonical string.
    seen = {}
    for p in programs:
        k = canonical_str(p)
        seen.setdefault(k, p)
    programs = list(seen.values())
    log(f"  |support| = {len(programs)} (after dedup)")

    log(f"[{time.time()-t0:6.1f}s] computing analytical posterior ...")
    analytical = analytical_posterior(programs, grammar, request_type)
    log(f"  |reachable under scorer| = {len(analytical)}")
    if not analytical:
        raise RuntimeError("no programs reachable under scorer; grammar setup bug")

    log(f"[{time.time()-t0:6.1f}s] running MH: n_steps={N_STEPS}, burnin={N_BURNIN}")
    visits, accept, proposed = run_mh(
        grammar, request_type, N_STEPS, N_BURNIN, SEED,
    )
    n_post_burnin = sum(visits.values())
    unique_visited = len(visits)
    log(f"  post-burnin samples: {n_post_burnin}")
    log(f"  unique states visited: {unique_visited}")
    log(f"  acceptance rate: {accept / max(proposed, 1):.3f} "
        f"(accept={accept}, proposed={proposed})")

    # Empirical distribution over the analytical support only.
    empirical = {k: visits.get(k, 0) / n_post_burnin for k in analytical}

    # Total-variation distance.
    tv = 0.5 * sum(abs(empirical[k] - analytical[k]) for k in analytical)
    # How much mass did the chain put OUTSIDE the enumerated support?
    outside_mass = sum(
        v / n_post_burnin for k, v in visits.items() if k not in analytical
    )

    # Per-program comparison (sorted by analytical prob descending).
    rows = sorted(
        [(k, analytical[k], empirical[k]) for k in analytical],
        key=lambda r: -r[1],
    )

    log("")
    log("Per-program comparison (top 20 by analytical prob):")
    log(f"  {'program':<50} {'analytical':>10} {'empirical':>10} {'ratio':>8}")
    for k, pa, pe in rows[:20]:
        ratio = pe / pa if pa > 0 else float('inf')
        log(f"  {k[:50]:<50} {pa:10.4f} {pe:10.4f} {ratio:8.2f}")

    # Naive 95% MC bound: TV ≤ sqrt(|support| / (2 * n_eff)) at worst case.
    # We use n_eff = n_post_burnin * (1 - rejection_rate) as a rough proxy.
    # For a strong bound, we'd need autocorrelation analysis.
    rej_rate = 1 - accept / max(proposed, 1)
    n_eff_proxy = n_post_burnin * (1 - rej_rate)
    mc_bound_95 = math.sqrt(len(analytical) / (2 * max(n_eff_proxy, 1)))

    log("")
    log(f"=== SUMMARY ===")
    log(f"Total variation (empirical vs analytical): {tv:.4f}")
    log(f"Rough 95% MC bound on TV (n_eff~{n_eff_proxy:.0f}): {mc_bound_95:.4f}")
    log(f"Mass outside enumerated support: {outside_mass:.4f}")
    log(f"Pass: {tv < 2 * mc_bound_95 and outside_mass < 0.01}")

    OUT_LOG.write_text("\n".join(log_lines) + "\n")
    OUT_JSON.write_text(json.dumps({
        'config': {
            'max_depth': MAX_DEPTH,
            'length_beta': LENGTH_BETA,
            'n_steps': N_STEPS,
            'n_burnin': N_BURNIN,
            'seed': SEED,
        },
        'support_size': len(analytical),
        'n_post_burnin': n_post_burnin,
        'unique_visited': unique_visited,
        'accept_rate': accept / max(proposed, 1),
        'total_variation': tv,
        'mc_bound_95': mc_bound_95,
        'outside_mass': outside_mass,
        'pass': tv < 2 * mc_bound_95 and outside_mass < 0.01,
        'rows': [{'program': k, 'analytical': pa, 'empirical': pe}
                 for k, pa, pe in rows],
        'wall_seconds': time.time() - t0,
    }, indent=2))
    log(f"\nResults written to {OUT_JSON.name}, {OUT_LOG.name}")


if __name__ == "__main__":
    main()
