"""
Round-1 Fix #5 re-run of the posterior-calibration experiment.

Compared to night1/round_3/posterior_calibration.py, this version responds to
three reviewer objections:

(1) H-methodology: the night-1 run used the scorer itself as the Bayesian
    prior, so any scorer/sampler mismatch cancelled in the acceptance ratio.
    Here the target is log π(p) = log_π_indep(p) - BETA * length(p) where
    log_π_indep(p) is computed from a FIXED independent grammar (uniform
    log-probs, unrelated to the sampler's grammar) via a vanilla top-down
    recursion that never touches _score_subtree_under_sampler. If the
    scorer/proposal densities disagree, TV will grow instead of cancel.

(2) C1-gallery: the night-1 toy used a monomorphic BOOL grammar, so the C1
    polymorphic-resolution code paths were not exercised. Here the grammar
    includes {not, and, or, eq, if} over types {BOOL, INT} with the
    polymorphic `eq : 'a -> 'a -> bool` and `if : bool -> 'a -> 'a -> 'a`.
    We also pin `_CONCRETE_TYPES` to [BOOL, INT] for this run so
    rng.choice resolution is exercised.

(3) H-autocorr + H-seeds: night 1 used n_eff_proxy = N * accept_rate which is
    not an ESS. Here we compute a Geyer IPS autocorrelation-based ESS per
    seed (on the program-length-signed indicator 1{p == p_modal}), and we
    run FIVE independent seeds to assess run-to-run variability.

Usage
-----
    python3 review-stage/experiments/round_1/posterior_calibration_v2.py
"""
import json
import math
import random
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dreamcoder_core.grammar import Grammar, Production
from dreamcoder_core.primitives import Primitive
from dreamcoder_core.program import (
    Abstraction, Application, Index, Invented, Program,
)
from dreamcoder_core.type_system import (
    BOOL, INT, Arrow, TypeContext, Type, TypeVariable,
)

import gallery_analysis.mcmc_search as ms
from gallery_analysis.mcmc_search import (
    _score_subtree_under_sampler, propose_regeneration, sample_program,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_DEPTH = 2
BETA = 0.9                  # per-node length penalty in target
N_STEPS = 50_000
N_BURNIN = 5_000
SEEDS = [20260417, 20260418, 20260419, 20260420, 20260421]

OUT_DIR = Path(__file__).parent
OUT_JSON = OUT_DIR / "posterior_calibration_v2_result.json"
OUT_LOG = OUT_DIR / "posterior_calibration_v2_log.txt"


# ---------------------------------------------------------------------------
# Polymorphic toy grammar over {BOOL, INT}
# ---------------------------------------------------------------------------
def build_polymorphic_grammar() -> Grammar:
    """Grammar exercising polymorphic free-var resolution at INT→BOOL.

    * not:  bool → bool
    * and:  bool → bool → bool
    * or:   bool → bool → bool
    * eq:   'a → 'a → bool         (polymorphic)
    * if:   bool → 'a → 'a → 'a    (polymorphic)
    * 0:    int
    * 1:    int
    """
    a = TypeVariable(0)

    def _not(x):
        return not x

    def _and(x):
        return lambda y: x and y

    def _or(x):
        return lambda y: x or y

    def _eq(x):
        return lambda y: x == y

    def _if(c):
        return lambda t: lambda e: t if c else e

    prims = [
        Primitive('not', Arrow(BOOL, BOOL), _not),
        Primitive('and', Arrow(BOOL, Arrow(BOOL, BOOL)), _and),
        Primitive('or', Arrow(BOOL, Arrow(BOOL, BOOL)), _or),
        Primitive('eq', Arrow(a, Arrow(a, BOOL)), _eq),
        Primitive('if', Arrow(BOOL, Arrow(a, Arrow(a, a))), _if),
        Primitive('0', INT, 0),
        Primitive('1', INT, 1),
    ]
    prods = [Production(p, p.tp, -1.0) for p in prims]
    return Grammar(prods, log_variable=-0.5)


def build_independent_grammar() -> Grammar:
    """Prior grammar, uniform but with DIFFERENT log-probs than the sampler.

    Using a different grammar makes the target density an *independent*
    Bayesian prior — the MH ratio can no longer cancel scorer/sampler
    disagreement by construction.
    """
    g = build_polymorphic_grammar()
    # Reweight: mix of -0.8 and -1.2 depending on whether arity == 0.
    productions = [
        Production(
            p.program,
            p.program.tp,
            (-0.8 if len(list(p.program.tp.arguments)) == 0 else -1.2),
        )
        for p in g.productions
    ]
    return Grammar(productions, log_variable=-0.4)


# ---------------------------------------------------------------------------
# Independent log-prior (top-down recursion, no _score_subtree_under_sampler)
# ---------------------------------------------------------------------------
def log_prior_indep(
    program: Program, grammar: Grammar, request_type: Type, env: List[Type],
) -> float:
    """Top-down log-density of `program` under `grammar`, ignoring scorer.

    Matches a canonical PCFG recursion: at each Application, look up the
    head's normalized log-prob among all productions/variables consistent
    with the current target type, and recurse on arguments with arg types
    obtained by unifying the candidate's polymorphic type with the actual
    argument's inferred type. This keeps the prior independent of
    `_score_subtree_under_sampler` while still resolving polymorphism.
    """
    if isinstance(request_type, Arrow):
        if not isinstance(program, Abstraction):
            return float('-inf')
        new_env = [request_type.arg] + env
        return log_prior_indep(program.body, grammar, request_type.ret, new_env)

    # Gather candidates with normalized log-probs summed with log_variable.
    ctx_cands = TypeContext()
    cands = grammar.candidates_for_type(
        request_type, ctx_cands, env, normalize=True,
    )
    var_entries = [
        idx for idx, var_t in enumerate(env) if var_t == request_type
    ]

    # Decompose program head.
    head: Program = program
    args: List[Program] = []
    while isinstance(head, Application):
        args.insert(0, head.x)
        head = head.f

    # Locate head in candidates / variables.
    log_head = float('-inf')
    head_arg_types: Optional[List[Type]] = None

    if isinstance(head, Index):
        if head.i < len(env) and env[head.i] == request_type:
            total = len(cands) + len(var_entries)
            log_head = grammar.log_variable - math.log(max(total, 1))
            head_arg_types = []
    elif isinstance(head, (Primitive, Invented)):
        # Re-derive arg types in a FRESH context so we can unify with each
        # arg's inferred type (resolves 'a in `eq : 'a → 'a → bool`).
        for prod, _inst_type, lp in cands:
            if prod.program.name == head.name:
                try:
                    ctx = TypeContext()
                    inst = ctx.instantiate(prod.program.tp)
                    ctx.unify(inst.returns, request_type)
                    resolved_args: List[Type] = []
                    for arg_prog, arg_t in zip(args, inst.arguments):
                        arg_inferred = arg_prog.infer_type(ctx, env)
                        ctx.unify(arg_t, arg_inferred)
                        resolved_args.append(ctx.apply(arg_t))
                    log_head = lp
                    head_arg_types = resolved_args
                except Exception:
                    pass
                break

    if not math.isfinite(log_head) or head_arg_types is None:
        return float('-inf')
    if len(args) != len(head_arg_types):
        return float('-inf')

    total = log_head
    for arg, arg_t in zip(args, head_arg_types):
        sub = log_prior_indep(arg, grammar, arg_t, env)
        if not math.isfinite(sub):
            return float('-inf')
        total += sub
    return total


# ---------------------------------------------------------------------------
# Enumeration for analytical posterior
# ---------------------------------------------------------------------------
def enumerate_programs(
    grammar: Grammar,
    target_type: Type,
    max_depth: int,
    depth: int,
    env: List[Type],
) -> List[Program]:
    """Exhaustively enumerate programs of the given type up to max_depth."""
    if isinstance(target_type, Arrow):
        new_env = [target_type.arg] + list(env)
        # Lambda doesn't consume depth budget (matches _sample).
        body_depth = depth if depth < max_depth else depth + 1
        bodies = enumerate_programs(
            grammar, target_type.ret, max_depth, body_depth, new_env,
        )
        return [Abstraction(body) for body in bodies]

    programs: List[Program] = []

    # Variables.
    for idx, var_t in enumerate(env):
        if var_t == target_type:
            programs.append(Index(idx))

    ctx = TypeContext()
    cands = grammar.candidates_for_type(
        target_type, ctx, env, normalize=False,
    )
    at_cap = depth >= max_depth
    for prod, inst_type, _lp in cands:
        arg_types = list(inst_type.arguments)
        if at_cap and len(arg_types) > 0:
            continue
        # Resolve free type variables by trying each concrete type.
        free = set()
        for at in arg_types:
            free |= at.free_type_variables()
        if not free:
            assignments = [{}]
        else:
            import itertools
            free_sorted = sorted(free)
            # Use BOOL/INT here since that matches _CONCRETE_TYPES for run.
            assignments = []
            for combo in itertools.product([BOOL, INT], repeat=len(free_sorted)):
                assignments.append(dict(zip(free_sorted, combo)))
        for subst in assignments:
            resolved_args = [a.apply_substitution(subst) for a in arg_types]
            if len(resolved_args) == 0:
                programs.append(prod.program)
                continue
            # Enumerate each arg recursively.
            per_arg: List[List[Program]] = []
            ok = True
            for at in resolved_args:
                subs = enumerate_programs(
                    grammar, at, max_depth, depth + 1, env,
                )
                if not subs:
                    ok = False
                    break
                per_arg.append(subs)
            if not ok:
                continue

            def _product(options):
                if not options:
                    yield []
                    return
                head, *rest = options
                for h in head:
                    for tail in _product(rest):
                        yield [h] + tail

            for combo in _product(per_arg):
                app: Program = prod.program
                for a in combo:
                    app = Application(app, a)
                programs.append(app)
    return programs


def program_length(p: Program) -> int:
    if isinstance(p, (Primitive, Invented, Index)):
        return 1
    if isinstance(p, Abstraction):
        return 1 + program_length(p.body)
    if isinstance(p, Application):
        return 1 + program_length(p.f) + program_length(p.x)
    raise TypeError(f"unexpected {type(p)}")


def canonical_str(p: Program) -> str:
    return str(p)


# ---------------------------------------------------------------------------
# Target and analytical posterior
# ---------------------------------------------------------------------------
def log_target(
    p: Program,
    prior_grammar: Grammar,
    request_type: Type,
) -> float:
    lp = log_prior_indep(p, prior_grammar, request_type, [])
    if not math.isfinite(lp):
        return float('-inf')
    return lp - BETA * program_length(p)


def analytical_posterior(
    programs: List[Program], prior_grammar: Grammar, request_type: Type,
) -> Dict[str, float]:
    log_ps = {}
    for p in programs:
        lp = log_target(p, prior_grammar, request_type)
        if math.isfinite(lp):
            log_ps[canonical_str(p)] = lp
    if not log_ps:
        return {}
    max_lp = max(log_ps.values())
    unnorm = {k: math.exp(v - max_lp) for k, v in log_ps.items()}
    Z = sum(unnorm.values())
    return {k: v / Z for k, v in unnorm.items()}


# ---------------------------------------------------------------------------
# Geyer IPS ESS
# ---------------------------------------------------------------------------
def geyer_ips_ess(x: List[float]) -> float:
    """Initial Positive Sequence ESS (Geyer 1992).

    Computes autocovariance gamma_t, pairs (gamma_{2t} + gamma_{2t+1}), stops
    at the first non-positive pair. Returns N / (1 + 2 sum_{t>=1} rho_t).
    """
    n = len(x)
    if n < 8:
        return float(max(n, 1))
    mean = sum(x) / n
    centered = [xi - mean for xi in x]
    var = sum(c * c for c in centered) / n
    if var == 0.0:
        return float(n)
    # Autocovariance up to n/4.
    max_lag = n // 4
    gammas = [var]
    for t in range(1, max_lag + 1):
        g = sum(centered[i] * centered[i + t] for i in range(n - t)) / n
        gammas.append(g)
    # Pairwise sums; keep only initial positive run.
    tau = 1.0
    t = 1
    while t + 1 <= max_lag:
        pair = gammas[t] + gammas[t + 1]
        if pair <= 0:
            break
        tau += 2.0 * pair / var
        t += 2
    return n / max(tau, 1.0)


# ---------------------------------------------------------------------------
# MH chain (C1-fixed proposal, independent-prior target)
# ---------------------------------------------------------------------------
def run_mh_one_seed(
    sampler_grammar: Grammar,
    prior_grammar: Grammar,
    request_type: Type,
    n_steps: int,
    n_burnin: int,
    seed: int,
) -> Dict:
    rng = random.Random(seed)

    # Seed a starting program with ≥1 regen site and finite target.
    tries = 0
    while True:
        tries += 1
        if tries > 500:
            raise RuntimeError("cannot find starting program")
        current = sample_program(
            sampler_grammar, request_type, max_depth=MAX_DEPTH,
            seed=seed + tries,
        )
        lt = log_target(current, prior_grammar, request_type)
        if not math.isfinite(lt):
            continue
        try:
            propose_regeneration(
                sampler_grammar, current, request_type,
                max_depth=MAX_DEPTH, seed=seed + tries + 1000,
            )
            break
        except RuntimeError:
            continue
    current_lt = lt

    visits = Counter()
    indicator_seq: List[float] = []  # for ESS
    accept = 0
    proposed = 0
    # Modal estimate updated online from post-burnin samples so the
    # indicator isn't fixed before the chain has any data.
    modal: Optional[str] = None
    modal_count = 0

    for step in range(n_steps):
        try:
            new_prog, log_q_fwd, log_q_rev = propose_regeneration(
                sampler_grammar, current, request_type,
                max_depth=MAX_DEPTH, seed=rng.randint(0, 2**31 - 1),
            )
        except RuntimeError:
            if step >= n_burnin:
                k = canonical_str(current)
                visits[k] += 1
                if visits[k] > modal_count:
                    modal = k
                    modal_count = visits[k]
                indicator_seq.append(1.0 if k == modal else 0.0)
            continue

        proposed += 1
        new_lt = log_target(new_prog, prior_grammar, request_type)
        if (math.isfinite(new_lt) and math.isfinite(log_q_fwd)
                and math.isfinite(log_q_rev)):
            log_alpha = (new_lt - current_lt) + (log_q_rev - log_q_fwd)
            if math.log(rng.random()) < log_alpha:
                current = new_prog
                current_lt = new_lt
                accept += 1

        if step >= n_burnin:
            k = canonical_str(current)
            visits[k] += 1
            if visits[k] > modal_count:
                modal = k
                modal_count = visits[k]
            indicator_seq.append(1.0 if k == modal else 0.0)

    return {
        'visits': dict(visits),
        'accept': accept,
        'proposed': proposed,
        'indicator_seq': indicator_seq,
        'modal': modal,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    log_lines: List[str] = []

    def log(msg: str) -> None:
        import sys
        print(msg, flush=True)
        sys.stdout.flush()
        log_lines.append(msg)
        OUT_LOG.write_text("\n".join(log_lines) + "\n")

    # Pin _CONCRETE_TYPES so rng.choice resolution is exercised on [BOOL, INT]
    # only — avoids spurious CARD/SUIT/RANK resolution that the toy grammar
    # cannot produce.
    old_concrete = list(ms._CONCRETE_TYPES)
    ms._CONCRETE_TYPES = [BOOL, INT]
    try:
        sampler_grammar = build_polymorphic_grammar()
        prior_grammar = build_independent_grammar()
        request_type = Arrow(INT, BOOL)

        log(f"[{time.time()-t0:6.1f}s] enumerating support at depth={MAX_DEPTH}")
        programs = enumerate_programs(
            sampler_grammar, request_type, MAX_DEPTH, 0, [],
        )
        seen = {}
        for p in programs:
            seen.setdefault(canonical_str(p), p)
        programs = list(seen.values())
        log(f"  |support|={len(programs)} after dedup")

        analytical = analytical_posterior(
            programs, prior_grammar, request_type,
        )
        log(f"  |reachable|={len(analytical)}")
        if not analytical:
            raise RuntimeError("empty analytical posterior; grammar bug")

        per_seed = []
        for seed in SEEDS:
            log(f"[{time.time()-t0:6.1f}s] MH seed={seed} ...")
            res = run_mh_one_seed(
                sampler_grammar, prior_grammar, request_type,
                N_STEPS, N_BURNIN, seed,
            )
            visits = res['visits']
            n_post = sum(visits.values())
            empirical = {
                k: visits.get(k, 0) / n_post for k in analytical
            }
            tv = 0.5 * sum(
                abs(empirical[k] - analytical[k]) for k in analytical
            )
            outside = sum(
                v / n_post for k, v in visits.items() if k not in analytical
            )
            accept_rate = res['accept'] / max(res['proposed'], 1)
            ess = geyer_ips_ess(res['indicator_seq'])
            mc_bound = math.sqrt(len(analytical) / (2.0 * max(ess, 1.0)))
            per_seed.append({
                'seed': seed,
                'n_post_burnin': n_post,
                'accept_rate': accept_rate,
                'ess_geyer_ips': ess,
                'tv': tv,
                'mc_bound_95': mc_bound,
                'outside_mass': outside,
                'pass': tv < 2 * mc_bound and outside < 0.01,
            })
            log(f"  TV={tv:.4f} ESS={ess:.0f} accept={accept_rate:.3f} "
                f"outside={outside:.4f} pass={per_seed[-1]['pass']}")

        # Aggregate.
        mean_tv = sum(s['tv'] for s in per_seed) / len(per_seed)
        max_tv = max(s['tv'] for s in per_seed)
        n_pass = sum(1 for s in per_seed if s['pass'])
        log("")
        log("=== SUMMARY (5 seeds, polymorphic INT→BOOL, independent prior) ===")
        log(f"Mean TV: {mean_tv:.4f}, max TV: {max_tv:.4f}, "
            f"passed {n_pass}/{len(per_seed)} seeds")

        OUT_LOG.write_text("\n".join(log_lines) + "\n")
        OUT_JSON.write_text(json.dumps({
            'config': {
                'max_depth': MAX_DEPTH,
                'beta': BETA,
                'n_steps': N_STEPS,
                'n_burnin': N_BURNIN,
                'seeds': SEEDS,
                'request_type': str(request_type),
                'concrete_types': [str(t) for t in ms._CONCRETE_TYPES],
            },
            'support_size': len(analytical),
            'per_seed': per_seed,
            'mean_tv': mean_tv,
            'max_tv': max_tv,
            'n_seeds_pass': n_pass,
            'n_seeds_total': len(per_seed),
            'wall_seconds': time.time() - t0,
        }, indent=2))
        log(f"Results written to {OUT_JSON.name}, {OUT_LOG.name}")
    finally:
        ms._CONCRETE_TYPES = old_concrete


if __name__ == "__main__":
    main()
