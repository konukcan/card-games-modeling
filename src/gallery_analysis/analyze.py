"""
Main analysis pipeline: Bayesian rule induction over all 60 gallery rules.

This is the top-level orchestrator that wires together:
  1. Enumeration of hand→bool programs from the DSL
  2. Syntactic pruning (built into enumerator)
  3. Trivial filtering on curated exemplar hands
  4. Fingerprinting into equivalence classes
  5. Extension size estimation (shared across rules)
  6. Per-rule Bayesian scoring and difficulty computation

ARCHITECTURE NOTE:
  Steps 1-5 are rule-independent — the hypothesis pool and equivalence classes
  are shared across all 60 rules. Only step 6 (hit vector computation and
  scoring) is rule-specific. This means we enumerate once and score 60 times,
  rather than enumerating 60 times.

Usage:
    cd src
    python -m gallery_analysis.analyze [--depth 7] [--max-programs 300000] [--quick]

    # Quick test (~2-3 minutes)
    python -m gallery_analysis.analyze --quick

    # Full analysis (~15-30 minutes depending on depth)
    python -m gallery_analysis.analyze --depth 7 --max-programs 300000
"""
import sys
import time
import json
import math
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Callable
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Hand
from gallery_analysis.enumerator import enumerate_hypotheses_with_stats
from gallery_analysis.exemplars import load_exemplars, generate_probe_set
from gallery_analysis.hypothesis_table import (
    filter_trivial, compute_fingerprint, estimate_extension_size,
)
from gallery_analysis.gallery_rules import GALLERY_RULES
from gallery_analysis.bayesian_scorer import (
    compute_log_likelihood_strict,
    compute_log_likelihood_noisy,
    normalize_posteriors,
    compute_rule_difficulty,
    ScoredHypothesis,
    TOTAL_HANDS,
)


# =========================================================================
# Step 1-4: Build shared hypothesis pool
# =========================================================================

def build_hypothesis_pool(
    max_depth: int = 7,
    max_programs: int = 300_000,
    max_cost: float = 35.0,
    timeout: float = 600.0,
    n_probes: int = 500,
    probe_seed: int = 42,
    verbose: int = 1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Build the shared pool of equivalence classes (rule-independent).

    Pipeline: enumerate → syntactic filter → trivial filter → fingerprint → deduplicate

    Args:
        max_depth: Maximum AST depth for enumeration.
        max_programs: Maximum programs to enumerate.
        max_cost: Maximum cost (-log probability) to explore.
        timeout: Enumeration timeout in seconds.
        n_probes: Number of random probe hands for fingerprinting.
        probe_seed: Random seed for probe generation.
        verbose: 0=silent, 1=summary, 2=detailed.

    Returns:
        (equivalence_classes, pipeline_stats)
        Each equivalence class is a dict with: canonical_program, canonical_prior,
        summed_prior, n_expressions, all_programs, fingerprint, predicate.
    """
    pipeline_stats = {}
    t_start = time.time()

    # --- Step 1+2: Enumerate with syntactic filter ---
    if verbose >= 1:
        print(f"Step 1: Enumerating programs (depth={max_depth}, max={max_programs:,})...",
              flush=True)

    t0 = time.time()
    programs, enum_stats = enumerate_hypotheses_with_stats(
        max_depth=max_depth,
        max_programs=max_programs,
        max_cost=max_cost,
        timeout=timeout,
    )
    t_enum = time.time() - t0

    pipeline_stats["enumeration"] = {
        **enum_stats,
        "time_seconds": round(t_enum, 1),
    }
    if verbose >= 1:
        print(f"  Yielded: {enum_stats['total_yielded']:,}, "
              f"syntactic rejected: {enum_stats['syntactic_rejected']:,}, "
              f"accepted: {enum_stats['accepted']:,} ({t_enum:.1f}s)", flush=True)

    # --- Step 3: Trivial filter ---
    if verbose >= 1:
        print("Step 2: Trivial filter (360 curated exemplar hands)...", flush=True)

    t0 = time.time()
    exemplars = load_exemplars()
    all_exemplar_hands = []
    for rule_id, data in exemplars.items():
        all_exemplar_hands.extend(data["hands_primary"])

    survivors, trivial_stats = filter_trivial(programs, all_exemplar_hands)
    t_trivial = time.time() - t0

    pipeline_stats["trivial_filter"] = {
        **trivial_stats,
        "time_seconds": round(t_trivial, 1),
    }
    if verbose >= 1:
        print(f"  Survivors: {trivial_stats['survivors']:,} / {trivial_stats['total']:,} "
              f"({100*trivial_stats['survivors']/max(trivial_stats['total'],1):.1f}%) "
              f"({t_trivial:.1f}s)", flush=True)

    # --- Step 4: Fingerprint into equivalence classes ---
    if verbose >= 1:
        print(f"Step 3: Fingerprinting ({n_probes} probes)...", flush=True)

    t0 = time.time()
    probes = generate_probe_set(n_probes=n_probes, seed=probe_seed)

    # Group by fingerprint
    fp_groups: Dict[str, List[Tuple[str, Callable, float]]] = {}
    for prog_str, pred_fn, log_prior in survivors:
        fp = compute_fingerprint(pred_fn, probes)
        if fp not in fp_groups:
            fp_groups[fp] = []
        fp_groups[fp].append((prog_str, pred_fn, log_prior))

    # Build equivalence classes
    equivalence_classes = []
    for fp, group in fp_groups.items():
        # Sort by prior (highest first = least negative)
        group.sort(key=lambda x: -x[2])
        canonical_str, canonical_pred, canonical_prior = group[0]

        # Summed prior across all expressions in the class
        summed_prior = math.log(sum(math.exp(lp) for _, _, lp in group))

        equivalence_classes.append({
            "canonical_program": canonical_str,
            "canonical_prior": canonical_prior,
            "summed_prior": summed_prior,
            "n_expressions": len(group),
            "all_programs": [p for p, _, _ in group],
            "fingerprint": fp,
            "predicate": canonical_pred,
        })

    # Sort by canonical prior
    equivalence_classes.sort(key=lambda c: -c["canonical_prior"])

    t_fp = time.time() - t0
    pipeline_stats["fingerprinting"] = {
        "n_equivalence_classes": len(equivalence_classes),
        "dedup_ratio": round(1 - len(equivalence_classes) / max(len(survivors), 1), 3),
        "time_seconds": round(t_fp, 1),
    }
    if verbose >= 1:
        print(f"  Equivalence classes: {len(equivalence_classes):,} "
              f"(dedup {pipeline_stats['fingerprinting']['dedup_ratio']*100:.1f}%) "
              f"({t_fp:.1f}s)", flush=True)

    pipeline_stats["total_time_seconds"] = round(time.time() - t_start, 1)
    return equivalence_classes, pipeline_stats


# =========================================================================
# Step 5: Extension size estimation (shared across rules)
# =========================================================================

def estimate_extensions(
    equivalence_classes: List[Dict[str, Any]],
    n_samples: int = 100_000,
    seed: int = 123,
    verbose: int = 1,
) -> List[Tuple[int, float]]:
    """
    Estimate extension sizes for all equivalence classes.

    This is rule-independent (a property of each hypothesis, not the data).
    We compute it once and reuse across all 60 rules.

    Returns:
        List of (extension_size, base_rate) tuples, parallel to equivalence_classes.
    """
    if verbose >= 1:
        print(f"Step 4: Estimating extension sizes ({n_samples:,} MC samples, "
              f"{len(equivalence_classes):,} classes)...", flush=True)

    t0 = time.time()
    extensions = []
    for i, cls in enumerate(equivalence_classes):
        ext_size, base_rate = estimate_extension_size(
            cls["predicate"], n_samples=n_samples, seed=seed
        )
        extensions.append((ext_size, base_rate))

        if verbose >= 2 and (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(equivalence_classes)} estimated...", flush=True)

    t_ext = time.time() - t0
    if verbose >= 1:
        # Summary statistics
        nonzero = sum(1 for e, _ in extensions if e > 0)
        print(f"  Done in {t_ext:.1f}s. Non-zero extensions: {nonzero:,} / {len(extensions):,}",
              flush=True)

    return extensions


# =========================================================================
# Step 6: Per-rule scoring
# =========================================================================

def score_rule(
    rule_id: str,
    exemplar_hands: List[Hand],
    equivalence_classes: List[Dict[str, Any]],
    extensions: List[Tuple[int, float]],
    epsilon: float = 0.01,
    prior_mode: str = "summed",
) -> Dict[str, Any]:
    """
    Score all hypotheses for a single rule and compute difficulty.

    Args:
        rule_id: Gallery rule identifier.
        exemplar_hands: The rule's exemplar hands (typically 6).
        equivalence_classes: Shared equivalence classes from build_hypothesis_pool.
        extensions: Shared extension size estimates from estimate_extensions.
        epsilon: Noise parameter for noisy likelihood.
        prior_mode: "canonical" or "summed".

    Returns:
        Dict with: rule_id, difficulty_metrics, top_hypotheses, n_hypotheses_scored,
        n_with_any_hit, n_with_all_hits.
    """
    n_exemplars = len(exemplar_hands)
    scored = []

    n_with_any_hit = 0
    n_with_all_hits = 0

    for cls, (ext_size, base_rate) in zip(equivalence_classes, extensions):
        # Compute hit vector for this rule's exemplars
        pred = cls["predicate"]
        hit_vector = []
        n_hits = 0
        for hand in exemplar_hands:
            try:
                result = pred(hand)
                hit_vector.append(result)
                if result:
                    n_hits += 1
            except Exception:
                hit_vector.append(False)

        if n_hits > 0:
            n_with_any_hit += 1
        if n_hits == n_exemplars:
            n_with_all_hits += 1

        # Compute likelihoods
        log_lik_strict = compute_log_likelihood_strict(n_hits, n_exemplars, ext_size)
        log_lik_noisy = compute_log_likelihood_noisy(n_hits, n_exemplars, ext_size, epsilon)

        # Select prior
        if prior_mode == "canonical":
            log_prior = cls["canonical_prior"]
        else:
            log_prior = cls["summed_prior"]

        log_post_strict = log_prior + log_lik_strict
        log_post_noisy = log_prior + log_lik_noisy

        scored.append(ScoredHypothesis(
            canonical_program=cls["canonical_program"],
            n_expressions=cls["n_expressions"],
            log_prior_canonical=cls["canonical_prior"],
            log_prior_summed=cls["summed_prior"],
            hit_vector=hit_vector,
            n_hits=n_hits,
            n_exemplars=n_exemplars,
            extension_size=ext_size,
            base_rate=base_rate,
            log_likelihood_strict=log_lik_strict,
            log_likelihood_noisy=log_lik_noisy,
            log_posterior_strict=log_post_strict,
            log_posterior_noisy=log_post_noisy,
            fingerprint=cls["fingerprint"],
            all_programs=cls["all_programs"],
        ))

    # Sort by noisy posterior (best first)
    scored.sort(key=lambda s: -s.log_posterior_noisy)

    # Normalize posteriors
    normalized = normalize_posteriors(scored, mode="noisy")

    # Compute difficulty
    difficulty = compute_rule_difficulty(normalized)

    # Top hypotheses for reporting
    top_hyps = []
    for sh, prob in normalized[:10]:
        top_hyps.append({
            "program": sh.canonical_program,
            "probability": round(prob, 6),
            "n_hits": sh.n_hits,
            "n_expressions": sh.n_expressions,
            "extension_size": sh.extension_size,
            "base_rate": round(sh.base_rate, 6),
            "log_prior": round(sh.log_prior_summed, 2),
            "log_likelihood": round(sh.log_likelihood_noisy, 2),
        })

    return {
        "rule_id": rule_id,
        "n_exemplars": n_exemplars,
        "n_hypotheses_scored": len(scored),
        "n_with_any_hit": n_with_any_hit,
        "n_with_all_hits": n_with_all_hits,
        "difficulty": difficulty,
        "top_hypotheses": top_hyps,
    }


# =========================================================================
# Full pipeline
# =========================================================================

def run_analysis(
    max_depth: int = 7,
    max_programs: int = 300_000,
    max_cost: float = 35.0,
    timeout: float = 600.0,
    n_probes: int = 500,
    extension_samples: int = 100_000,
    epsilon: float = 0.01,
    prior_mode: str = "summed",
    verbose: int = 1,
) -> Dict[str, Any]:
    """
    Run the full Bayesian rule induction analysis over all 60 gallery rules.

    Returns a results dict with pipeline_stats, per-rule results, and
    difficulty rankings.
    """
    t_total_start = time.time()

    # Build shared hypothesis pool
    equiv_classes, pipeline_stats = build_hypothesis_pool(
        max_depth=max_depth,
        max_programs=max_programs,
        max_cost=max_cost,
        timeout=timeout,
        n_probes=n_probes,
        verbose=verbose,
    )

    # Estimate extension sizes (shared)
    extensions = estimate_extensions(
        equiv_classes,
        n_samples=extension_samples,
        verbose=verbose,
    )

    # Score each rule
    if verbose >= 1:
        print(f"\nStep 5: Scoring {len(GALLERY_RULES)} rules...", flush=True)

    exemplars = load_exemplars()
    rule_results = {}
    t0 = time.time()

    for i, (rule_id, rule_info) in enumerate(GALLERY_RULES.items()):
        if rule_id not in exemplars:
            if verbose >= 2:
                print(f"  Skipping {rule_id}: no exemplars", flush=True)
            continue

        result = score_rule(
            rule_id=rule_id,
            exemplar_hands=exemplars[rule_id]["hands_primary"],
            equivalence_classes=equiv_classes,
            extensions=extensions,
            epsilon=epsilon,
            prior_mode=prior_mode,
        )
        result["group"] = rule_info["group"]
        result["answer"] = rule_info["answer"]
        rule_results[rule_id] = result

        if verbose >= 2:
            d = result["difficulty"]
            top = result["top_hypotheses"][0] if result["top_hypotheses"] else {}
            print(f"  {rule_id:<30} entropy={d['posterior_entropy']:.2f}  "
                  f"top1={d['top1_probability']:.3f}  "
                  f"n_eff={d['n_effective_hypotheses']:.1f}  "
                  f"hits_all={result['n_with_all_hits']}  "
                  f"top: {top.get('program', '?')[:50]}", flush=True)

    t_scoring = time.time() - t0
    if verbose >= 1:
        print(f"  Done in {t_scoring:.1f}s", flush=True)

    # Build difficulty ranking
    ranking = []
    for rule_id, result in rule_results.items():
        ranking.append({
            "rule_id": rule_id,
            "group": result["group"],
            "answer": result["answer"],
            "posterior_entropy": result["difficulty"]["posterior_entropy"],
            "n_effective_hypotheses": result["difficulty"]["n_effective_hypotheses"],
            "top1_probability": result["difficulty"]["top1_probability"],
            "top5_probability": result["difficulty"]["top5_probability"],
            "n_with_all_hits": result["n_with_all_hits"],
        })

    # Sort by entropy (highest = hardest)
    ranking.sort(key=lambda r: -r["posterior_entropy"])

    t_total = time.time() - t_total_start
    pipeline_stats["scoring_time_seconds"] = round(t_scoring, 1)
    pipeline_stats["grand_total_seconds"] = round(t_total, 1)

    if verbose >= 1:
        print(f"\nTotal pipeline time: {t_total:.1f}s", flush=True)

    return {
        "pipeline_stats": pipeline_stats,
        "rule_results": rule_results,
        "difficulty_ranking": ranking,
        "config": {
            "max_depth": max_depth,
            "max_programs": max_programs,
            "max_cost": max_cost,
            "n_probes": n_probes,
            "extension_samples": extension_samples,
            "epsilon": epsilon,
            "prior_mode": prior_mode,
        },
    }


# =========================================================================
# Reporting
# =========================================================================

def print_difficulty_report(results: Dict[str, Any]):
    """Print a human-readable difficulty ranking report."""
    ranking = results["difficulty_ranking"]
    stats = results["pipeline_stats"]
    config = results["config"]

    print("\n" + "=" * 80)
    print("BAYESIAN RULE INDUCTION — DIFFICULTY RANKING")
    print("=" * 80)

    print(f"\nConfig: depth={config['max_depth']}, programs={config['max_programs']:,}, "
          f"probes={config['n_probes']}, MC_samples={config['extension_samples']:,}, "
          f"ε={config['epsilon']}, prior={config['prior_mode']}")

    enum_s = stats.get("enumeration", {})
    triv_s = stats.get("trivial_filter", {})
    fp_s = stats.get("fingerprinting", {})
    print(f"\nPipeline: {enum_s.get('accepted',0):,} programs → "
          f"{triv_s.get('survivors',0):,} after trivial filter → "
          f"{fp_s.get('n_equivalence_classes',0):,} equivalence classes")
    print(f"Total time: {stats.get('grand_total_seconds',0):.0f}s")

    # Group-level summary
    group_entropies = {1: [], 2: [], 3: []}
    for r in ranking:
        if r["group"] in group_entropies:
            group_entropies[r["group"]].append(r["posterior_entropy"])

    print(f"\nMean posterior entropy by difficulty group:")
    for g in [1, 2, 3]:
        ents = group_entropies[g]
        label = {1: "Easy", 2: "Medium", 3: "Hard"}[g]
        if ents:
            mean_e = sum(ents) / len(ents)
            print(f"  Group {g} ({label:>6}, n={len(ents):>2}): mean entropy = {mean_e:.2f}")

    # Full ranking
    print(f"\n{'Rank':<6} {'Rule':<30} {'Grp':>4} {'Entropy':>8} {'N_eff':>7} "
          f"{'Top1%':>7} {'AllHit':>7} {'Top hypothesis'}")
    print(f"{'─'*6} {'─'*30} {'─'*4} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*40}")

    for i, r in enumerate(ranking):
        rule_id = r["rule_id"]
        rule_res = results["rule_results"][rule_id]
        top_prog = rule_res["top_hypotheses"][0]["program"][:40] if rule_res["top_hypotheses"] else "?"

        print(f"{i+1:<6} {rule_id:<30} {r['group']:>4} "
              f"{r['posterior_entropy']:>8.2f} {r['n_effective_hypotheses']:>7.1f} "
              f"{r['top1_probability']*100:>6.1f}% {r['n_with_all_hits']:>7} "
              f"{top_prog}")

    # Easiest and hardest
    print(f"\n{'─'*80}")
    print("EASIEST 5:")
    for r in ranking[-5:][::-1]:
        print(f"  {r['rule_id']:<30} entropy={r['posterior_entropy']:.2f}  "
              f"top1={r['top1_probability']*100:.1f}%")

    print("\nHARDEST 5:")
    for r in ranking[:5]:
        print(f"  {r['rule_id']:<30} entropy={r['posterior_entropy']:.2f}  "
              f"top1={r['top1_probability']*100:.1f}%")


# =========================================================================
# CLI
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="Bayesian rule induction analysis")
    parser.add_argument("--depth", type=int, default=7, help="Max AST depth")
    parser.add_argument("--max-programs", type=int, default=300_000, help="Max programs to enumerate")
    parser.add_argument("--max-cost", type=float, default=35.0, help="Max cost to explore")
    parser.add_argument("--timeout", type=float, default=600.0, help="Enumeration timeout")
    parser.add_argument("--probes", type=int, default=500, help="Number of probe hands")
    parser.add_argument("--mc-samples", type=int, default=100_000, help="MC samples for extension size")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Noise parameter")
    parser.add_argument("--prior", choices=["canonical", "summed"], default="summed", help="Prior mode")
    parser.add_argument("--verbose", type=int, default=2, help="Verbosity (0-2)")
    parser.add_argument("--quick", action="store_true", help="Quick test (depth 5, 50K programs)")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON to this path")
    args = parser.parse_args()

    if args.quick:
        args.depth = 5
        args.max_programs = 50_000
        args.max_cost = 25.0
        args.timeout = 120.0
        args.mc_samples = 10_000

    results = run_analysis(
        max_depth=args.depth,
        max_programs=args.max_programs,
        max_cost=args.max_cost,
        timeout=args.timeout,
        n_probes=args.probes,
        extension_samples=args.mc_samples,
        epsilon=args.epsilon,
        prior_mode=args.prior,
        verbose=args.verbose,
    )

    print_difficulty_report(results)

    # Save results if requested
    if args.output:
        # Make JSON-serializable (remove predicate functions, hit vectors)
        save_results = {
            "pipeline_stats": results["pipeline_stats"],
            "config": results["config"],
            "difficulty_ranking": results["difficulty_ranking"],
            "rule_details": {},
        }
        for rule_id, rr in results["rule_results"].items():
            save_results["rule_details"][rule_id] = {
                "rule_id": rr["rule_id"],
                "group": rr["group"],
                "answer": rr["answer"],
                "difficulty": rr["difficulty"],
                "top_hypotheses": rr["top_hypotheses"],
                "n_hypotheses_scored": rr["n_hypotheses_scored"],
                "n_with_any_hit": rr["n_with_any_hit"],
                "n_with_all_hits": rr["n_with_all_hits"],
            }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(save_results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
