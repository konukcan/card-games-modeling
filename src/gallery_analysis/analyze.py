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
    max_list_chain: int = 2,
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
        max_list_chain: Maximum consecutive list→list transforms (default 2).
            Set to None to disable. Eliminates "deeply wrapped shallow
            predicates" that carry negligible posterior mass.
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
        chain_str = f", max_list_chain={max_list_chain}" if max_list_chain is not None else ""
        print(f"Step 1: Enumerating programs (depth={max_depth}, max={max_programs:,}{chain_str})...",
              flush=True)

    t0 = time.time()
    programs, enum_stats = enumerate_hypotheses_with_stats(
        max_depth=max_depth,
        max_programs=max_programs,
        max_cost=max_cost,
        timeout=timeout,
        max_list_chain=max_list_chain,
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
    cache_path: str = None,
) -> List[Tuple[int, float]]:
    """
    Estimate extension sizes for all equivalence classes.

    This is rule-independent (a property of each hypothesis, not the data).
    We compute it once and reuse across all 60 rules.

    If cache_path is provided, loads cached extension sizes by fingerprint
    and only computes for classes not in the cache. Saves updated cache
    after computation.

    Returns:
        List of (extension_size, base_rate) tuples, parallel to equivalence_classes.
    """
    # Load cache if available
    cache: Dict[str, Tuple[int, float]] = {}
    if cache_path:
        cache_file = Path(cache_path)
        if cache_file.exists():
            with open(cache_file) as f:
                raw = json.load(f)
            cache = {k: tuple(v) for k, v in raw.items()}
            if verbose >= 1:
                print(f"  Loaded extension cache: {len(cache):,} entries from {cache_path}",
                      flush=True)

    # Count how many we need to compute
    n_cached = sum(1 for cls in equivalence_classes if cls["fingerprint"] in cache)
    n_to_compute = len(equivalence_classes) - n_cached

    if verbose >= 1:
        print(f"Step 4: Estimating extension sizes ({n_samples:,} MC samples, "
              f"{len(equivalence_classes):,} classes, {n_cached:,} cached, "
              f"{n_to_compute:,} to compute)...", flush=True)

    t0 = time.time()
    extensions = []
    n_computed = 0
    for i, cls in enumerate(equivalence_classes):
        fp = cls["fingerprint"]
        if fp in cache:
            extensions.append(cache[fp])
        else:
            ext_size, base_rate = estimate_extension_size(
                cls["predicate"], n_samples=n_samples, seed=seed
            )
            extensions.append((ext_size, base_rate))
            cache[fp] = (ext_size, base_rate)
            n_computed += 1

            if verbose >= 2 and n_computed % 500 == 0:
                print(f"  {n_computed}/{n_to_compute} computed...", flush=True)

    t_ext = time.time() - t0
    if verbose >= 1:
        nonzero = sum(1 for e, _ in extensions if e > 0)
        print(f"  Done in {t_ext:.1f}s. Non-zero extensions: {nonzero:,} / {len(extensions):,}",
              flush=True)

    # Save updated cache
    if cache_path:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump({k: list(v) for k, v in cache.items()}, f)
        if verbose >= 1:
            print(f"  Extension cache saved: {len(cache):,} entries to {cache_path}",
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
    true_rule_fingerprint: str = None,
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
        true_rule_fingerprint: Fingerprint of the equivalence class containing the
            true rule for this gallery rule. Used for true-rule tracking, confusion
            profiles, and diagnosticity analysis.

    Returns:
        Dict with: rule_id, difficulty_metrics, top_hypotheses, n_hypotheses_scored,
        n_with_any_hit, n_with_all_hits, plus true_rule_* fields and
        exemplar_diagnosticity when true_rule_fingerprint is provided.
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

    # --- True-rule tracking ---
    # Find the true rule in the posterior ranking by its fingerprint.
    true_rule_rank = None
    true_rule_posterior_mass = None
    true_rule_program = None
    true_rule_log_prior = None
    true_rule_hit_vector = None

    if true_rule_fingerprint:
        for i, (sh, prob) in enumerate(normalized):
            if sh.fingerprint == true_rule_fingerprint:
                true_rule_rank = i + 1  # 1-indexed
                true_rule_posterior_mass = prob
                true_rule_program = sh.canonical_program
                true_rule_log_prior = sh.log_prior_summed
                true_rule_hit_vector = sh.hit_vector
                break

    # --- Confusion profiles ---
    # For each top-10 competitor, record per-exemplar agreement with the true rule.
    # Since the true rule's exemplars are all hits by definition, the agreement
    # vector is simply whether the competitor also returns True for each exemplar.
    if true_rule_hit_vector is not None:
        for idx, hyp_dict in enumerate(top_hyps):
            sh_top = normalized[idx][0]
            agrees = [
                sh_top.hit_vector[j] == true_rule_hit_vector[j]
                for j in range(n_exemplars)
            ]
            hyp_dict["agrees_on_exemplars"] = agrees
            hyp_dict["n_exemplar_agreements"] = sum(agrees)

    # --- Diagnosticity analysis ---
    # For each exemplar hand, compute what fraction of the top-10 posterior mass
    # agrees with the true rule on that hand. Hands where many competitors
    # disagree are "diagnostic" — they help distinguish the true rule.
    exemplar_diagnosticity = None
    if true_rule_hit_vector is not None:
        exemplar_diagnosticity = []
        for hand_idx in range(n_exemplars):
            true_val = true_rule_hit_vector[hand_idx]
            agree_mass = 0.0
            total_mass = 0.0
            for sh, prob in normalized[:10]:
                total_mass += prob
                if sh.hit_vector[hand_idx] == true_val:
                    agree_mass += prob
            agreement_rate = agree_mass / max(total_mass, 1e-10)
            exemplar_diagnosticity.append({
                "hand_idx": hand_idx,
                "agreement_rate": round(agreement_rate, 4),
                "diagnostic": agreement_rate < 0.90,
            })

    return {
        "rule_id": rule_id,
        "n_exemplars": n_exemplars,
        "n_hypotheses_scored": len(scored),
        "n_with_any_hit": n_with_any_hit,
        "n_with_all_hits": n_with_all_hits,
        "difficulty": difficulty,
        "top_hypotheses": top_hyps,
        # True-rule tracking fields
        "true_rule_rank": true_rule_rank,
        "true_rule_posterior_mass": true_rule_posterior_mass,
        "true_rule_program": true_rule_program,
        "true_rule_log_prior": true_rule_log_prior,
        # Diagnosticity (None when no true rule fingerprint provided)
        "exemplar_diagnosticity": exemplar_diagnosticity,
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
    inject_path: str = None,
    extension_cache: str = None,
    max_list_chain: int = 2,
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
        max_list_chain=max_list_chain,
        verbose=verbose,
    )

    # --- Step 3b: Merge injected hypotheses (if provided) ---
    if inject_path:
        from gallery_analysis.injection import load_and_validate_injections, merge_injected
        from gallery_analysis.enumerator import build_gallery_grammar

        if verbose >= 1:
            print(f"\nStep 3b: Loading and merging injected hypotheses...", flush=True)

        grammar = build_gallery_grammar()

        # Get the enumerated prior range for calibration warnings
        if equiv_classes:
            enum_priors = [c["canonical_prior"] for c in equiv_classes]
            enumerated_prior_range = (min(enum_priors), max(enum_priors))
        else:
            enumerated_prior_range = None

        injected = load_and_validate_injections(
            inject_path, grammar=grammar,
            enumerated_prior_range=enumerated_prior_range,
        )

        # Regenerate the same probes used during fingerprinting (same seed)
        probes = generate_probe_set(n_probes=n_probes, seed=42)

        n_before = len(equiv_classes)
        equiv_classes = merge_injected(equiv_classes, injected, probes)
        n_after = len(equiv_classes)

        if verbose >= 1:
            print(f"  Loaded {len(injected)} hypotheses, "
                  f"{n_after - n_before} novel classes added, "
                  f"{len(injected) - (n_after - n_before)} merged into existing",
                  flush=True)

        pipeline_stats["injection"] = {
            "n_injected": len(injected),
            "n_novel_classes": n_after - n_before,
            "n_merged": len(injected) - (n_after - n_before),
        }

    # Estimate extension sizes (shared, with optional cache)
    extensions = estimate_extensions(
        equiv_classes,
        n_samples=extension_samples,
        verbose=verbose,
        cache_path=extension_cache,
    )

    # Build true-rule fingerprint lookup from equivalence classes.
    # Each equivalence class that was injected as a true rule has a
    # "true_for_rule" field mapping it to the gallery rule it represents.
    true_rule_fps = {}  # rule_id -> fingerprint
    for cls in equiv_classes:
        # Use true_for_rules list (handles fingerprint collisions where
        # multiple true rules map to the same equivalence class).
        true_for_list = cls.get("true_for_rules", [])
        if not true_for_list:
            # Fallback for backward compat with single-value field
            single = cls.get("true_for_rule")
            if single:
                true_for_list = [single]
        for rule_id in true_for_list:
            true_rule_fps[rule_id] = cls["fingerprint"]

    if verbose >= 1:
        n_found = len(true_rule_fps)
        print(f"\n  True-rule fingerprints found: {n_found} / {len(GALLERY_RULES)}", flush=True)

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
            true_rule_fingerprint=true_rule_fps.get(rule_id),
        )
        result["group"] = rule_info["group"]
        result["answer"] = rule_info["answer"]
        rule_results[rule_id] = result

        if verbose >= 2:
            d = result["difficulty"]
            top = result["top_hypotheses"][0] if result["top_hypotheses"] else {}
            true_rank_str = (f"  true_rank={result['true_rule_rank']}"
                             if result.get('true_rule_rank') is not None else "")
            print(f"  {rule_id:<30} entropy={d['posterior_entropy']:.2f}  "
                  f"top1={d['top1_probability']:.3f}  "
                  f"n_eff={d['n_effective_hypotheses']:.1f}  "
                  f"hits_all={result['n_with_all_hits']}{true_rank_str}  "
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
            "true_rule_rank": result.get("true_rule_rank"),
            "true_rule_posterior_mass": result.get("true_rule_posterior_mass"),
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

def print_difficulty_report(results: Dict[str, Any], verbose: int = 1):
    """
    Print a human-readable difficulty ranking report.

    Args:
        results: Full results dict from run_analysis().
        verbose: Verbosity level.
            0 = summary only
            1 = ranking table + group summary (default)
            2 = + per-rule confusion profiles and diagnostic exemplars
    """
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

    # Group-level summary with true-rule tracking
    group_data = {1: {"entropy": [], "true_rank": [], "true_mass": []},
                  2: {"entropy": [], "true_rank": [], "true_mass": []},
                  3: {"entropy": [], "true_rank": [], "true_mass": []}}
    for r in ranking:
        g = r.get("group")
        if g in group_data:
            group_data[g]["entropy"].append(r["posterior_entropy"])
            if r.get("true_rule_rank") is not None:
                group_data[g]["true_rank"].append(r["true_rule_rank"])
            if r.get("true_rule_posterior_mass") is not None:
                group_data[g]["true_mass"].append(r["true_rule_posterior_mass"])

    print(f"\nMean by group:")
    for g in [1, 2, 3]:
        gd = group_data[g]
        label = {1: "Easy", 2: "Medium", 3: "Hard"}[g]
        n = len(gd["entropy"])
        if not n:
            continue
        mean_e = sum(gd["entropy"]) / n
        parts = [f"entropy={mean_e:.2f}"]
        if gd["true_rank"]:
            mean_rank = sum(gd["true_rank"]) / len(gd["true_rank"])
            parts.append(f"true_rank={mean_rank:.1f}")
        if gd["true_mass"]:
            mean_mass = sum(gd["true_mass"]) / len(gd["true_mass"])
            parts.append(f"true_mass={mean_mass*100:.1f}%")
        print(f"  Group {g} ({label:>6}, n={n:>2}): {('  ').join(parts)}")

    # Full ranking table with TrueRk and TrueP% columns
    print(f"\n{'Rank':<6} {'Rule':<30} {'Grp':>4} {'Entropy':>8} {'N_eff':>7} "
          f"{'Top1%':>7} {'TrueRk':>7} {'TrueP%':>7} {'Top hypothesis'}")
    print(f"{'─'*6} {'─'*30} {'─'*4} {'─'*8} {'─'*7} {'─'*7} {'─'*7} {'─'*7} {'─'*40}")

    for i, r in enumerate(ranking):
        rule_id = r["rule_id"]
        rule_res = results["rule_results"][rule_id]
        top_prog = rule_res["top_hypotheses"][0]["program"][:40] if rule_res["top_hypotheses"] else "?"

        true_rk_str = f"{r['true_rule_rank']:>7}" if r.get("true_rule_rank") is not None else "      —"
        true_mass_str = (f"{r['true_rule_posterior_mass']*100:>6.1f}%"
                         if r.get("true_rule_posterior_mass") is not None else "      —")

        print(f"{i+1:<6} {rule_id:<30} {r['group']:>4} "
              f"{r['posterior_entropy']:>8.2f} {r['n_effective_hypotheses']:>7.1f} "
              f"{r['top1_probability']*100:>6.1f}% {true_rk_str} {true_mass_str} "
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

    # --- Cross-rule summary: rules where true rule not in top 10 ---
    not_in_top10 = [
        r for r in ranking
        if r.get("true_rule_rank") is not None and r["true_rule_rank"] > 10
    ]
    missing_true = [
        r for r in ranking
        if r.get("true_rule_rank") is None
    ]

    if not_in_top10 or missing_true:
        print(f"\n{'─'*80}")
        print("RULES WHERE TRUE RULE NOT IN TOP 10:")
        for r in sorted(not_in_top10, key=lambda x: -x["true_rule_rank"]):
            print(f"  {r['rule_id']:<30} (rank={r['true_rule_rank']})")
        for r in missing_true:
            print(f"  {r['rule_id']:<30} (not found in hypothesis pool)")

    # --- Entropy vs true-rule mass correlation ---
    paired = [
        (r["posterior_entropy"], r["true_rule_posterior_mass"])
        for r in ranking
        if r.get("true_rule_posterior_mass") is not None
    ]
    if len(paired) >= 3:
        # Compute Spearman rank correlation (no scipy dependency)
        n_p = len(paired)
        ent_vals = [p[0] for p in paired]
        mass_vals = [p[1] for p in paired]

        def _rank_values(vals):
            """Assign ranks (1-based, average ties)."""
            indexed = sorted(enumerate(vals), key=lambda x: x[1])
            ranks = [0.0] * len(vals)
            i = 0
            while i < len(indexed):
                j = i
                while j < len(indexed) - 1 and indexed[j + 1][1] == indexed[j][1]:
                    j += 1
                avg_rank = (i + j) / 2.0 + 1
                for k in range(i, j + 1):
                    ranks[indexed[k][0]] = avg_rank
                i = j + 1
            return ranks

        ent_ranks = _rank_values(ent_vals)
        mass_ranks = _rank_values(mass_vals)

        d_sq_sum = sum((er - mr) ** 2 for er, mr in zip(ent_ranks, mass_ranks))
        spearman_rho = 1 - (6 * d_sq_sum) / (n_p * (n_p ** 2 - 1))

        print(f"\nEntropy vs true-rule mass: Spearman rho = {spearman_rho:.3f} (n={n_p})")

    # --- Per-rule detail section (verbose >= 2) ---
    if verbose >= 2:
        print(f"\n{'='*80}")
        print("PER-RULE DETAIL: TOP 5 COMPETITORS AND DIAGNOSTIC EXEMPLARS")
        print(f"{'='*80}")

        for r in ranking:
            rule_id = r["rule_id"]
            rule_res = results["rule_results"][rule_id]
            top_hyps = rule_res.get("top_hypotheses", [])
            diagnosticity = rule_res.get("exemplar_diagnosticity")
            true_rank = rule_res.get("true_rule_rank")
            true_mass = rule_res.get("true_rule_posterior_mass")

            true_rk_str = str(true_rank) if true_rank is not None else "—"
            true_mass_str = f"{true_mass*100:.1f}%" if true_mass is not None else "—"

            print(f"\n{'─'*70}")
            print(f"{rule_id}  (group={r['group']}, entropy={r['posterior_entropy']:.2f}, "
                  f"true_rank={true_rk_str}, true_mass={true_mass_str})")

            # Top 5 competitors with confusion profile
            for j, hyp in enumerate(top_hyps[:5]):
                prog = hyp["program"][:55]
                prob_pct = hyp["probability"] * 100
                n_hits = hyp["n_hits"]
                agrees = hyp.get("agrees_on_exemplars")
                agree_str = ""
                if agrees is not None:
                    agree_str = "  agree=[" + "".join("Y" if a else "n" for a in agrees) + "]"
                print(f"  {j+1}. ({prob_pct:5.1f}%) hits={n_hits} {agree_str}  {prog}")

            # Diagnostic exemplars
            if diagnosticity:
                diag_hands = [d for d in diagnosticity if d["diagnostic"]]
                if diag_hands:
                    idxs = ", ".join(f"h{d['hand_idx']}({d['agreement_rate']*100:.0f}%)"
                                     for d in diag_hands)
                    print(f"  Diagnostic exemplars: {idxs}")
                else:
                    print(f"  Diagnostic exemplars: none (all exemplars >90% agreement)")


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
    parser.add_argument("--inject", type=str, default=None,
                        help="Path to injection JSON file with additional hypotheses")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON to this path")
    parser.add_argument("--extension-cache", type=str, default=None,
                        help="Path to cache extension sizes (skips MC estimation on re-runs)")
    parser.add_argument("--max-list-chain", type=int, default=2,
                        help="Max consecutive list→list transforms (default 2, None to disable)")
    parser.add_argument("--no-list-chain-limit", action="store_true",
                        help="Disable list→list chain limit (enumerate all programs)")
    args = parser.parse_args()

    # Handle list chain limit
    max_list_chain = None if args.no_list_chain_limit else args.max_list_chain

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
        inject_path=args.inject,
        extension_cache=args.extension_cache,
        max_list_chain=max_list_chain,
        verbose=args.verbose,
    )

    print_difficulty_report(results, verbose=args.verbose)

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
                "true_rule_rank": rr.get("true_rule_rank"),
                "true_rule_posterior_mass": rr.get("true_rule_posterior_mass"),
                "true_rule_program": rr.get("true_rule_program"),
                "true_rule_log_prior": rr.get("true_rule_log_prior"),
                "exemplar_diagnosticity": rr.get("exemplar_diagnosticity"),
            }

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(save_results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
