"""
Night 4 C1 — MCMC attribution runner
====================================

6 rules × 4 chains × 100k steps, β = 0.5→1.0 under ``piecewise_half`` schedule
(ramps over first 50%, flat at 1.0 for remaining 50%).  Schema-v2 output keeps
the merged ``visit_counts`` field that ``run_comparison.py`` reads, **and**
adds ``chain_visit_counts: List[Dict[prog_str, int]]`` so we can compute
per-n-chains posterior-TV post-hoc and answer the "did one chain already
converge?" question without rerunning.

Fork of ``run_mcmc_night3.py``.  Key deltas:

* ``n_steps``: 20_000 → 100_000.
* ``beta_start / beta_end / beta_schedule``: 0.3→1.0 linear → 0.5→1.0
  ``piecewise_half`` (new field, see commit 044444f).
* Output dir: ``mcmc_50k_4chains`` → ``mcmc_100k_4chains`` (distinct from
  Night 3's outputs — never overwrite prior data).
* Rule bundle: 18 rules → 6-rule attribution bundle locked in the Night 4
  design doc + Codex consensus review (``colors_palindrome`` dropped,
  ``all_red`` added).
* Persistence: schema_version 2, per-chain visit_counts preserved.

The worker stays stateless: it runs one chain, returns the full visit Counter
for that chain, and the parent merges on flush.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))


CONFIG = json.loads((HERE / "config.json").read_text())
RULE_IDS: List[str] = CONFIG["rules"]
MCMC_CFG = CONFIG["mcmc"]
PERSISTENCE_CFG = CONFIG["persistence"]
SCHEMA_VERSION = PERSISTENCE_CFG["schema_version"]

OUT_DIR = HERE / PERSISTENCE_CFG["output_dir"]
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "raw_visits").mkdir(exist_ok=True)
(OUT_DIR / "checkpoints").mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Worker: runs ONE chain to completion; returns a serialisable dict.
# Top-level so ProcessPoolExecutor can pickle it.
# ---------------------------------------------------------------------------

def _run_one_chain(task: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single MCMC chain for one rule. Returns serialisable summary.

    task keys:
        rule_id, chain_idx, seed, n_steps, max_depth, noise_epsilon,
        max_nodes, init_max_depth, beta_start, beta_end, beta_schedule,
        exemplar_hands (pickled bytes), ext_probes (pickled bytes),
        checkpoint_steps (List[int])
    """
    import pickle as pkl

    sys.path.insert(0, str(REPO / "src"))
    from dreamcoder_core.type_system import Arrow, BOOL, HAND
    from gallery_analysis.enumerator import build_gallery_grammar
    from gallery_analysis.mcmc_search import MCMCChain, MCMCConfig

    exemplar_hands = pkl.loads(task["exemplar_hands"])
    ext_probes = pkl.loads(task["ext_probes"])

    grammar = build_gallery_grammar()
    config = MCMCConfig(
        n_steps=task["n_steps"],
        max_depth=task["max_depth"],
        noise_epsilon=task["noise_epsilon"],
        max_nodes=task["max_nodes"],
        top_k=250,
        seed=task["seed"],
        verbose=0,
        init_max_depth=task["init_max_depth"],
        beta_start=task["beta_start"],
        beta_end=task["beta_end"],
        beta_schedule=task["beta_schedule"],
    )

    t0 = time.time()
    chain = MCMCChain(grammar, config)
    result = chain.run(
        request_type=Arrow(HAND, BOOL),
        exemplar_hands=exemplar_hands,
        ext_probe_hands=ext_probes,
    )
    elapsed = time.time() - t0

    # Per-checkpoint running visit counters. Incremental counting avoids
    # re-scanning the prefix for each checkpoint (same pattern as Night 3).
    ckpt_steps = sorted(task["checkpoint_steps"])
    checkpoints: Dict[int, Dict[str, int]] = {}
    trajectory = result.trajectory
    running: Counter = Counter()
    prev = 0
    for cp in ckpt_steps:
        cp_end = min(cp + 1, len(trajectory))  # inclusive of step=cp
        for j in range(prev, cp_end):
            running[trajectory[j]] += 1
        prev = cp_end
        checkpoints[cp] = dict(running)

    return {
        "rule_id": task["rule_id"],
        "chain_idx": task["chain_idx"],
        "seed": task["seed"],
        "n_steps": result.n_steps,
        "n_accepted": result.n_accepted,
        "n_unique": result.n_unique,
        "acceptance_rate": result.acceptance_rate,
        "best_program": result.best_program,
        "best_log_posterior": result.best_log_posterior,
        "visit_counts": dict(result.visit_counts),
        "first_passage": dict(result.first_passage),
        "ext_fractions": dict(result.ext_fractions),
        "checkpoints": checkpoints,
        "elapsed_s": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Per-rule merging and flushing.
# ---------------------------------------------------------------------------

def _merge_chains(chain_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum visit_counts, take min first_passage, avg ext_fractions."""
    merged_visits: Counter = Counter()
    merged_fp: Dict[str, int] = {}
    ext_sum: Dict[str, float] = {}
    ext_n: Dict[str, int] = {}
    best_prog, best_lp = None, float("-inf")
    total_accepted = 0
    total_steps = 0

    for cr in chain_results:
        for prog, count in cr["visit_counts"].items():
            merged_visits[prog] += count
        for prog, step in cr["first_passage"].items():
            if prog not in merged_fp or step < merged_fp[prog]:
                merged_fp[prog] = step
        for prog, ef in cr["ext_fractions"].items():
            ext_sum[prog] = ext_sum.get(prog, 0.0) + ef
            ext_n[prog] = ext_n.get(prog, 0) + 1
        if cr["best_log_posterior"] > best_lp:
            best_lp = cr["best_log_posterior"]
            best_prog = cr["best_program"]
        total_accepted += cr["n_accepted"]
        total_steps += cr["n_steps"]

    merged_ext = {p: ext_sum[p] / ext_n[p] for p in ext_sum}

    return {
        "visit_counts": dict(merged_visits),
        "first_passage": merged_fp,
        "ext_fractions": merged_ext,
        "best_program": best_prog,
        "best_log_posterior": best_lp,
        "total_accepted": total_accepted,
        "total_steps": total_steps,
        "n_unique_merged": len(merged_visits),
    }


def _merge_checkpoint(chain_results: List[Dict[str, Any]], step: int) -> Dict[str, int]:
    """Sum visit_counts across chains at a single checkpoint step."""
    agg: Counter = Counter()
    for cr in chain_results:
        for prog, count in cr["checkpoints"].get(step, {}).items():
            agg[prog] += count
    return dict(agg)


def _flush_rule(
    rid: str,
    chains: List[Dict[str, Any]],
    ckpt_steps: List[int],
) -> None:
    """Persist schema-v2 merged+per-chain results + per-checkpoint merged visits."""
    chains_sorted = sorted(chains, key=lambda c: c["chain_idx"])
    merged = _merge_chains(chains_sorted)

    out = {
        # --- schema & metadata ---------------------------------------------
        "schema_version": SCHEMA_VERSION,
        "rule_id": rid,
        "n_chains": len(chains_sorted),
        "n_unique_merged": merged["n_unique_merged"],
        "n_unique_per_chain": [c["n_unique"] for c in chains_sorted],
        "acceptance_rate_per_chain": [c["acceptance_rate"] for c in chains_sorted],
        "total_accepted": merged["total_accepted"],
        "total_steps": merged["total_steps"],
        "best_program": merged["best_program"],
        "best_log_posterior": merged["best_log_posterior"],
        # --- v1 shape preserved for run_comparison.py --------------------
        "visit_counts": merged["visit_counts"],
        "first_passage": merged["first_passage"],
        "ext_fractions": merged["ext_fractions"],
        # --- v2 additions: per-chain dicts (for chain-scaling post-hoc) --
        "chain_visit_counts": [c["visit_counts"] for c in chains_sorted],
        "chain_first_passage": [c["first_passage"] for c in chains_sorted],
        "chain_seeds": [c["seed"] for c in chains_sorted],
        "chain_best_program": [c["best_program"] for c in chains_sorted],
        "chain_best_log_posterior": [c["best_log_posterior"] for c in chains_sorted],
        "chain_elapsed_s": [c["elapsed_s"] for c in chains_sorted],
    }
    with open(OUT_DIR / "raw_visits" / f"{rid}.json", "w") as f:
        json.dump(out, f)

    # Per-checkpoint merged visits (run_comparison.py consumes these for
    # convergence diagnostics; same shape as Night 3).
    ckpt_dir = OUT_DIR / "checkpoints" / rid
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for cp in ckpt_steps:
        cp_merged = _merge_checkpoint(chains_sorted, cp)
        with open(ckpt_dir / f"checkpoint_{cp}.json", "w") as f:
            json.dump({
                "rule_id": rid,
                "step": cp,
                "visit_counts": cp_merged,
                "n_unique": len(cp_merged),
            }, f)

    print(
        f"  [flush] {rid} → raw_visits/{rid}.json (v{SCHEMA_VERSION}) "
        f"+ {len(ckpt_steps)} checkpoints",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main driver.
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n-workers",
        type=int,
        default=CONFIG["parallelism"]["n_mcmc_workers"],
        help="Parallel chain workers (default from config.parallelism).",
    )
    ap.add_argument(
        "--skip-if-exists",
        action="store_true",
        help="Skip rules that already have raw_visits/{rule}.json.",
    )
    ap.add_argument(
        "--only-rules",
        type=str,
        default=None,
        help="Comma-separated subset of rule ids to run.",
    )
    ap.add_argument(
        "--dry-run-steps",
        type=int,
        default=None,
        help="Override n_steps for smoke testing (e.g. --dry-run-steps 1000).",
    )
    args = ap.parse_args()

    from gallery_analysis.exemplars import load_exemplars, generate_probe_set
    from gallery_analysis.gallery_rules import GALLERY_RULES

    print("[MCMC] Loading exemplars + generating probes...", flush=True)
    frozen = load_exemplars()
    ext_probes = generate_probe_set(
        n_probes=MCMC_CFG["ext_probes_native"],
        seed=MCMC_CFG["base_seed"],
    )
    ext_probes_pkl = pickle.dumps(ext_probes)

    rule_ids = RULE_IDS
    if args.only_rules:
        rule_ids = [r.strip() for r in args.only_rules.split(",") if r.strip()]

    runnable = [
        rid for rid in rule_ids
        if rid in GALLERY_RULES and rid in frozen
    ]
    skipped = [rid for rid in rule_ids if rid not in runnable]
    if skipped:
        print(f"[MCMC] Skipping unknown rules: {skipped}", flush=True)

    n_steps = args.dry_run_steps if args.dry_run_steps is not None else MCMC_CFG["n_steps"]
    if args.dry_run_steps is not None:
        print(
            f"[MCMC] DRY-RUN OVERRIDE: n_steps = {n_steps} (config had {MCMC_CFG['n_steps']})",
            flush=True,
        )

    n_ckpt = MCMC_CFG["n_checkpoints"]
    ckpt_steps = [
        int(round(n_steps * (i + 1) / n_ckpt)) - 1
        for i in range(n_ckpt)
    ]
    print(f"[MCMC] n_steps={n_steps}, checkpoints at: {ckpt_steps}", flush=True)
    print(
        f"[MCMC] β schedule: {MCMC_CFG['beta_schedule']} "
        f"(β = {MCMC_CFG['beta_start']} → {MCMC_CFG['beta_end']})",
        flush=True,
    )

    tasks: List[Dict[str, Any]] = []
    base_seed = MCMC_CFG["base_seed"]
    for rid in runnable:
        if args.skip_if_exists and (OUT_DIR / "raw_visits" / f"{rid}.json").exists():
            print(f"[MCMC] Skipping (exists): {rid}", flush=True)
            continue
        exemplars = frozen[rid]["hands_primary"]
        exemplars_pkl = pickle.dumps(exemplars)
        rule_seed_offset = hash(rid) % 100_000
        for cidx in range(MCMC_CFG["n_chains"]):
            tasks.append({
                "rule_id": rid,
                "chain_idx": cidx,
                "seed": base_seed + rule_seed_offset + cidx * 1000,
                "n_steps": n_steps,
                "max_depth": MCMC_CFG["max_depth"],
                "noise_epsilon": MCMC_CFG["noise_epsilon"],
                "max_nodes": MCMC_CFG["max_nodes"],
                "init_max_depth": MCMC_CFG["init_max_depth"],
                "beta_start": MCMC_CFG["beta_start"],
                "beta_end": MCMC_CFG["beta_end"],
                "beta_schedule": MCMC_CFG["beta_schedule"],
                "exemplar_hands": exemplars_pkl,
                "ext_probes": ext_probes_pkl,
                "checkpoint_steps": ckpt_steps,
            })

    print(
        f"[MCMC] Launching {len(tasks)} chain-tasks "
        f"({len(set(t['rule_id'] for t in tasks))} rules × "
        f"{MCMC_CFG['n_chains']} chains) on {args.n_workers} workers...",
        flush=True,
    )

    per_rule: Dict[str, List[Dict[str, Any]]] = {}
    t_start = time.time()

    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {pool.submit(_run_one_chain, t): t for t in tasks}
        n_done = 0
        for fut in as_completed(futures):
            try:
                res = fut.result()
            except Exception as e:
                t = futures[fut]
                print(
                    f"[MCMC][ERR] {t['rule_id']} chain {t['chain_idx']}: {e}",
                    flush=True,
                )
                continue
            rid = res["rule_id"]
            per_rule.setdefault(rid, []).append(res)
            n_done += 1
            print(
                f"  [{n_done}/{len(tasks)}] {rid} ch{res['chain_idx']}: "
                f"accept={res['acceptance_rate']:.3f} "
                f"unique={res['n_unique']:,} "
                f"{res['elapsed_s']:.1f}s",
                flush=True,
            )
            if len(per_rule[rid]) == MCMC_CFG["n_chains"]:
                _flush_rule(rid, per_rule[rid], ckpt_steps)

    total_elapsed = time.time() - t_start
    summary = {
        "schema_version": SCHEMA_VERSION,
        "n_rules_completed": len(per_rule),
        "n_chains_per_rule": MCMC_CFG["n_chains"],
        "n_steps_per_chain": n_steps,
        "checkpoint_steps": ckpt_steps,
        "beta_schedule": MCMC_CFG["beta_schedule"],
        "beta_start": MCMC_CFG["beta_start"],
        "beta_end": MCMC_CFG["beta_end"],
        "total_wall_seconds": round(total_elapsed, 1),
        "config": MCMC_CFG,
    }
    with open(OUT_DIR / "mcmc_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(
        f"[MCMC] Done. {len(per_rule)} rules in {total_elapsed / 60:.1f} min",
        flush=True,
    )


if __name__ == "__main__":
    main()
