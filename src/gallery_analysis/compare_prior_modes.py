"""
Compare summed vs canonical prior modes and their interaction with Phase 1b injection.

This script answers three questions:

1. Which equivalence classes benefit most from summed priors?
   → Classes with many programs (n_expressions) get the biggest boost because
   summed prior = log(sum(exp(lp_i))) across all i programs, while canonical
   prior = max(lp_i). More programs → bigger gap.

2. How does Phase 1b injection interact with the class prior?
   → By design (Round 1 Finding 1 fix in `injection.py`), merging an injected
   hypothesis into an existing class does NOT inflate `summed_prior`: the
   enumerated summed prior encodes grammar expressibility only, not LLM
   agreement. `summed_prior_with_injections` is tracked separately for
   diagnostic use. For merged classes, membership changes (n_expressions
   increases); the prior does not. For novel classes, the new injected
   class contributes a full new mass term. We therefore show, per class,
   (i) the unchanged enumerated `summed_prior`, (ii) the diagnostic
   `summed_prior_with_injections`, and (iii) the class-membership delta.

3. Would posterior rankings change meaningfully under canonical priors?
   → For ~10 rules spanning easy/medium/hard, compare true rule rank, top-1
   hypothesis, and posterior entropy under both prior modes.

Usage:
    cd src
    python -m gallery_analysis.compare_prior_modes
"""

import sys
import math
import time
from pathlib import Path
from typing import Dict, List, Any, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from gallery_analysis.analyze import (
    build_hypothesis_pool,
    estimate_extensions,
    merge_injections_and_extend,
    score_rule,
)
from gallery_analysis.injection import load_and_validate_injections, merge_injected
from gallery_analysis.enumerator import build_gallery_grammar
from gallery_analysis.exemplars import load_exemplars, generate_probe_set
from gallery_analysis.gallery_rules import GALLERY_RULES
from gallery_analysis.bayesian_scorer import normalize_posteriors


# =========================================================================
# Configuration
# =========================================================================

# Quick-equivalent settings for speed
DEPTH = 5
MAX_PROGRAMS = 50_000
MAX_COST = 35.0
TIMEOUT = 300.0
N_PROBES = 500
EXTENSION_SAMPLES = 100_000

# Representative rules spanning easy / medium / hard
# Selected to cover a range of difficulty and structural types.
SELECTED_RULES = [
    # Easy (Group 1) — simple property checks
    "strict_increasing",       # Group 3 but structurally easy for the model
    "all_same_suit",           # Group 1 — single aggregate check
    "all_same_color",          # Group 1 — color version
    # Medium (Group 2) — moderate compositional structure
    "all_even",                # Group 2 — requires rank → parity check
    "all_odd",                 # Group 2 — similar structure to all_even
    "three_spades",            # Group 2 — counting + threshold
    "both_halves_uniform_color",  # Group 2 — split + check each half
    # Hard (Group 3) — complex compositional rules
    "both_halves_have_pair_rank",  # Group 2 but hard for model
    "four_kind_adjacent_any",      # Group 1 but structurally interesting
    "ranks_palindrome",            # Group 3 — requires reversal + comparison
]

# Path to Phase 1b injection file
INJECT_PATH = str(
    Path(__file__).parent / "data" / "injected_hypotheses.json"
)


# =========================================================================
# Utility functions
# =========================================================================

def compute_entropy(normalized: List[Tuple[Any, float]]) -> float:
    """
    Compute Shannon entropy of a normalized posterior distribution.

    H = -sum(p * log2(p)) for p > 0

    Higher entropy means the posterior is more spread out (more uncertain).
    Lower entropy means mass is concentrated on fewer hypotheses.
    """
    entropy = 0.0
    for _, p in normalized:
        if p > 1e-15:
            entropy -= p * math.log2(p)
    return entropy


def fmt_program(prog: str, max_len: int = 50) -> str:
    """Truncate a program string for table display."""
    if len(prog) <= max_len:
        return prog
    return prog[:max_len - 3] + "..."


def print_header(title: str):
    """Print a section header."""
    print()
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print()


def score_rule_both_modes(
    rule_id: str,
    exemplar_hands: List,
    equiv_classes: List[Dict],
    extensions: List[Tuple[int, float]],
    true_rule_fps: Dict[str, str],
) -> Dict[str, Any]:
    """
    Score a rule under both summed and canonical priors, returning
    key comparison metrics.

    Returns a dict with:
      - rule_id
      - For each mode ('summed', 'canonical'):
        - true_rule_rank
        - top1_program
        - top1_posterior
        - entropy
    """
    results = {"rule_id": rule_id}

    for mode in ("summed", "canonical"):
        result = score_rule(
            rule_id=rule_id,
            exemplar_hands=exemplar_hands,
            equivalence_classes=equiv_classes,
            extensions=extensions,
            prior_mode=mode,
            true_rule_fingerprint=true_rule_fps.get(rule_id),
        )

        # Recompute normalized posteriors to get entropy.
        # score_rule already sorts by noisy posterior, so we can use the
        # top hypotheses directly. But we need the full normalized list
        # for entropy, so we re-run the normalization.
        # Actually, score_rule returns top_hypotheses (top 10) and
        # true_rule_rank, true_rule_posterior_mass, which is what we need.
        # For entropy, we approximate using the returned data.

        # For a cleaner entropy calculation, we'll score manually:
        from gallery_analysis.bayesian_scorer import (
            compute_log_likelihood_noisy_from_base_rate, ScoredHypothesis,
        )

        scored = []
        for cls, (ext_size, base_rate) in zip(equiv_classes, extensions):
            pred = cls["predicate"]
            n_hits = 0
            for hand in exemplar_hands:
                try:
                    if pred(hand):
                        n_hits += 1
                except Exception:
                    pass
            n_exemplars = len(exemplar_hands)

            log_lik = compute_log_likelihood_noisy_from_base_rate(
                n_hits, n_exemplars, base_rate
            )

            if mode == "canonical":
                log_prior = cls["canonical_prior"]
            else:
                log_prior = cls["summed_prior"]

            log_post = log_prior + log_lik

            scored.append(ScoredHypothesis(
                canonical_program=cls["canonical_program"],
                n_expressions=cls["n_expressions"],
                log_prior_canonical=cls["canonical_prior"],
                log_prior_summed=cls["summed_prior"],
                hit_vector=[],  # not needed for this analysis
                n_hits=n_hits,
                n_exemplars=n_exemplars,
                extension_size=ext_size,
                base_rate=base_rate,
                log_likelihood_strict=float('-inf'),
                log_likelihood_noisy=log_lik,
                log_posterior_strict=float('-inf'),
                log_posterior_noisy=log_post,
                fingerprint=cls["fingerprint"],
                all_programs=cls["all_programs"],
            ))

        scored.sort(key=lambda s: -s.log_posterior_noisy)
        normalized = normalize_posteriors(scored, mode="noisy")

        entropy = compute_entropy(normalized)

        # Find true rule rank
        true_fp = true_rule_fps.get(rule_id)
        true_rank = None
        true_post = None
        if true_fp:
            for i, (sh, prob) in enumerate(normalized):
                if sh.fingerprint == true_fp:
                    true_rank = i + 1
                    true_post = prob
                    break

        # Top-1
        top1_sh, top1_prob = normalized[0]

        results[mode] = {
            "true_rank": true_rank,
            "true_post": true_post,
            "top1_program": top1_sh.canonical_program,
            "top1_prob": top1_prob,
            "top1_n_expr": top1_sh.n_expressions,
            "entropy": entropy,
        }

    return results


# =========================================================================
# Main analysis
# =========================================================================

def main():
    t0 = time.time()

    # ------------------------------------------------------------------
    # Step 1: Build hypothesis pool (shared, no injection)
    # ------------------------------------------------------------------
    print_header("BUILDING HYPOTHESIS POOL (no injection)")

    equiv_classes_base, stats = build_hypothesis_pool(
        max_depth=DEPTH,
        max_programs=MAX_PROGRAMS,
        max_cost=MAX_COST,
        timeout=TIMEOUT,
        n_probes=N_PROBES,
        verbose=1,
    )

    n_base = len(equiv_classes_base)
    print(f"\nBase equivalence classes: {n_base:,}")

    # ------------------------------------------------------------------
    # Step 2+3: Injected pool + extensions via shared helper
    # (Round 2 ruling, Findings 1/2/4): reuses the exact probes used during
    # fingerprinting, re-runs _strict_split_classes after merge, and passes
    # the probe hash so stale extension caches are detected.
    # ------------------------------------------------------------------
    print_header("MERGING PHASE 1b INJECTIONS + ESTIMATING EXTENSIONS")

    # Shallow copy of pipeline_stats so the base-pool build_hypothesis_pool
    # invocation above isn't mutated by the helper's strict_split_post_inject
    # bookkeeping.
    stats_for_injected = dict(stats)

    equiv_classes_injected, extensions_injected, probes, _ = merge_injections_and_extend(
        list(equiv_classes_base),
        stats_for_injected,
        inject_path=INJECT_PATH,
        verbose=1,
    )
    n_injected = len(equiv_classes_injected)
    print(f"Injected pool: {n_injected:,} classes "
          f"({n_injected - n_base:+d} vs base)")

    extensions_base = estimate_extensions(
        equiv_classes_base,
        n_samples=EXTENSION_SAMPLES,
        verbose=1,
    )

    # ------------------------------------------------------------------
    # ANALYSIS 1: Which equivalence classes benefit most from summed priors?
    # ------------------------------------------------------------------
    print_header("ANALYSIS 1: CLASSES WITH LARGEST SUMMED-vs-CANONICAL PRIOR GAP")

    print("These are the equivalence classes where summing over multiple programs")
    print("gives the largest boost relative to using only the canonical (shortest) program.")
    print()

    # Compute the gap for each class
    gaps = []
    for cls in equiv_classes_base:
        gap = cls["summed_prior"] - cls["canonical_prior"]
        gaps.append({
            "program": cls["canonical_program"],
            "n_expressions": cls["n_expressions"],
            "canonical_prior": cls["canonical_prior"],
            "summed_prior": cls["summed_prior"],
            "gap": gap,
            "boost_factor": math.exp(gap),  # multiplicative boost in probability
        })

    # Sort by gap (largest first)
    gaps.sort(key=lambda x: -x["gap"])

    # Print top 20
    print(f"{'Rank':<5} {'n_expr':<8} {'Canon LP':<10} {'Summed LP':<10} "
          f"{'Gap':<8} {'Boost':<8} {'Canonical Program'}")
    print("-" * 110)
    for i, g in enumerate(gaps[:20]):
        print(f"{i+1:<5} {g['n_expressions']:<8} {g['canonical_prior']:<10.2f} "
              f"{g['summed_prior']:<10.2f} {g['gap']:<8.3f} "
              f"{g['boost_factor']:<8.2f}x {fmt_program(g['program'], 50)}")

    # Summary statistics
    print()
    mean_gap = sum(g["gap"] for g in gaps) / len(gaps)
    median_gap = sorted(g["gap"] for g in gaps)[len(gaps) // 2]
    n_single = sum(1 for g in gaps if g["n_expressions"] == 1)
    print(f"Summary: {len(gaps)} classes total, {n_single} singletons (gap=0)")
    print(f"  Mean gap:   {mean_gap:.4f} (mean boost: {math.exp(mean_gap):.3f}x)")
    print(f"  Median gap: {median_gap:.4f} (median boost: {math.exp(median_gap):.3f}x)")
    print(f"  Max gap:    {gaps[0]['gap']:.4f} (max boost: {gaps[0]['boost_factor']:.2f}x)")

    # Distribution of n_expressions
    expr_counts = [g["n_expressions"] for g in gaps]
    print(f"\n  n_expressions distribution:")
    for threshold in [1, 2, 3, 5, 10, 20, 50]:
        count = sum(1 for n in expr_counts if n >= threshold)
        print(f"    >= {threshold:<3}: {count:>6} classes ({100*count/len(gaps):.1f}%)")

    # ------------------------------------------------------------------
    # ANALYSIS 2: Phase 1b injection impact on class membership and priors
    # ------------------------------------------------------------------
    print_header("ANALYSIS 2: PHASE 1b INJECTION — MEMBERSHIP + DIAGNOSTIC PRIOR DELTA")

    print("Top 20 classes by n_expressions AFTER injection.")
    print("By design (Round 1 Finding 1 fix), `summed_prior` on merged")
    print("classes is UNCHANGED by injection. `SumLP+inj` shows the")
    print("diagnostic `summed_prior_with_injections`; the `LP+inj delta`")
    print("column reports the difference. Membership changes appear in the")
    print("n_base / n_inj / delta columns.")
    print()

    # Build lookup from fingerprint to base class
    base_lookup = {cls["fingerprint"]: cls for cls in equiv_classes_base}

    # Find classes with most expressions after injection
    injected_sorted = sorted(
        equiv_classes_injected,
        key=lambda c: -c["n_expressions"],
    )

    print(f"{'Rank':<5} {'n_base':<8} {'n_inj':<8} {'delta':<7} "
          f"{'Base SumLP':<11} {'SumLP+inj':<11} {'LP+inj delta':<13} "
          f"{'Source':<10} {'Program'}")
    print("-" * 136)

    for i, cls in enumerate(injected_sorted[:20]):
        fp = cls["fingerprint"]
        base_cls = base_lookup.get(fp)
        if base_cls:
            n_base = base_cls["n_expressions"]
            base_sum_lp = base_cls["summed_prior"]
        else:
            n_base = 0
            base_sum_lp = float('-inf')

        n_inj = cls["n_expressions"]
        delta_n = n_inj - n_base
        # Use the diagnostic `summed_prior_with_injections` when present;
        # fall back to enumerated `summed_prior` when no injection touched
        # the class. Injection does NOT inflate `summed_prior` itself.
        inj_sum_lp = cls.get("summed_prior_with_injections", cls["summed_prior"])
        delta_lp = inj_sum_lp - base_sum_lp if math.isfinite(base_sum_lp) else float('inf')

        source = cls.get("source", "enumerated")

        delta_lp_str = f"{delta_lp:.3f}" if math.isfinite(delta_lp) else "NEW"
        base_lp_str = f"{base_sum_lp:.2f}" if math.isfinite(base_sum_lp) else "N/A"

        print(f"{i+1:<5} {n_base:<8} {n_inj:<8} {'+' + str(delta_n):<7} "
              f"{base_lp_str:<11} {inj_sum_lp:<11.2f} {delta_lp_str:<13} "
              f"{source:<10} {fmt_program(cls['canonical_program'], 45)}")

    # Count how many classes were affected by injection
    n_merged = sum(1 for cls in equiv_classes_injected if cls.get("source") == "merged")
    n_novel = sum(1 for cls in equiv_classes_injected if cls.get("source") == "injected")
    print(f"\nInjection summary: {n_merged} classes gained programs, "
          f"{n_novel} novel classes added")
    print(f"Total classes: {len(equiv_classes_base)} (base) -> {n_injected} (injected)")

    # ------------------------------------------------------------------
    # ANALYSIS 3: Posterior comparison across prior modes
    # ------------------------------------------------------------------
    print_header("ANALYSIS 3: POSTERIOR RANKINGS UNDER SUMMED vs CANONICAL PRIORS")

    print(f"Comparing {len(SELECTED_RULES)} rules across easy/medium/hard.")
    print("Using Phase 1b-augmented hypothesis pool.")
    print()

    # Build true-rule fingerprint lookup
    true_rule_fps = {}
    for cls in equiv_classes_injected:
        true_for_list = cls.get("true_for_rules", [])
        if not true_for_list:
            single = cls.get("true_for_rule")
            if single:
                true_for_list = [single]
        for rid in true_for_list:
            true_rule_fps[rid] = cls["fingerprint"]

    exemplars = load_exemplars()

    # Run comparison for each selected rule
    comparisons = []
    for rule_id in SELECTED_RULES:
        if rule_id not in GALLERY_RULES:
            print(f"  WARNING: {rule_id} not in GALLERY_RULES, skipping")
            continue
        if rule_id not in exemplars:
            print(f"  WARNING: {rule_id} has no exemplars, skipping")
            continue

        print(f"  Scoring {rule_id}...", flush=True)
        comp = score_rule_both_modes(
            rule_id=rule_id,
            exemplar_hands=exemplars[rule_id]["hands_primary"],
            equiv_classes=equiv_classes_injected,
            extensions=extensions_injected,
            true_rule_fps=true_rule_fps,
        )
        comparisons.append(comp)

    # Print comparison table
    print()
    print(f"{'Rule':<32} {'Group':<6} "
          f"{'Rank(S)':<8} {'Rank(C)':<8} {'Delta':<7} "
          f"{'Top1 Same?':<11} "
          f"{'H(S)':<7} {'H(C)':<7} {'dH':<7}")
    print("-" * 100)

    for comp in comparisons:
        rule_id = comp["rule_id"]
        group = GALLERY_RULES[rule_id]["group"]

        s = comp["summed"]
        c = comp["canonical"]

        rank_s = s["true_rank"] if s["true_rank"] is not None else "N/F"
        rank_c = c["true_rank"] if c["true_rank"] is not None else "N/F"

        if isinstance(rank_s, int) and isinstance(rank_c, int):
            delta_rank = rank_c - rank_s  # positive = summed is better
            delta_str = f"{delta_rank:+d}"
        else:
            delta_str = "N/A"

        # Check if top-1 hypothesis is the same under both modes
        top1_same = "YES" if s["top1_program"] == c["top1_program"] else "NO"

        entropy_s = s["entropy"]
        entropy_c = c["entropy"]
        delta_h = entropy_c - entropy_s

        print(f"{rule_id:<32} {group:<6} "
              f"{str(rank_s):<8} {str(rank_c):<8} {delta_str:<7} "
              f"{top1_same:<11} "
              f"{entropy_s:<7.2f} {entropy_c:<7.2f} {delta_h:<+7.2f}")

    # Detailed view: where top-1 differs
    print()
    print_header("DETAILED: RULES WHERE TOP-1 DIFFERS BETWEEN MODES")

    any_differ = False
    for comp in comparisons:
        s = comp["summed"]
        c = comp["canonical"]
        if s["top1_program"] != c["top1_program"]:
            any_differ = True
            print(f"Rule: {comp['rule_id']}")
            print(f"  Summed  top-1: {fmt_program(s['top1_program'], 70)} "
                  f"(p={s['top1_prob']:.4f}, n_expr={s['top1_n_expr']})")
            print(f"  Canonical top-1: {fmt_program(c['top1_program'], 70)} "
                  f"(p={c['top1_prob']:.4f}, n_expr={c['top1_n_expr']})")
            print()

    if not any_differ:
        print("All selected rules have the SAME top-1 hypothesis under both modes.")
        print("The prior mode primarily affects posterior mass distribution,")
        print("not the winner identity.")

    # Detailed view: true rule posterior mass comparison
    print()
    print_header("TRUE RULE POSTERIOR MASS: SUMMED vs CANONICAL")

    print(f"{'Rule':<32} {'P_true(S)':<14} {'P_true(C)':<14} {'Ratio S/C':<12}")
    print("-" * 75)

    def fmt_prob(p):
        """Format a probability, using scientific notation for tiny values."""
        if p is None:
            return "N/F"
        if p >= 0.0001:
            return f"{p:.6f}"
        return f"{p:.2e}"

    for comp in comparisons:
        rule_id = comp["rule_id"]
        s = comp["summed"]
        c = comp["canonical"]

        ps = s["true_post"]
        pc = c["true_post"]

        ps_str = fmt_prob(ps)
        pc_str = fmt_prob(pc)

        if ps is not None and pc is not None and pc > 1e-30:
            ratio = ps / pc
            if ratio > 1e6:
                ratio_str = f"{ratio:.1e}x"
            else:
                ratio_str = f"{ratio:.2f}x"
        else:
            ratio_str = "N/A"

        print(f"{rule_id:<32} {ps_str:<14} {pc_str:<14} {ratio_str}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_header("SUMMARY")

    # Count rank changes
    rank_changes = []
    for comp in comparisons:
        s = comp["summed"]
        c = comp["canonical"]
        if isinstance(s["true_rank"], int) and isinstance(c["true_rank"], int):
            rank_changes.append(c["true_rank"] - s["true_rank"])

    n_top1_differ = sum(
        1 for comp in comparisons
        if comp["summed"]["top1_program"] != comp["canonical"]["top1_program"]
    )

    print(f"Rules analyzed: {len(comparisons)}")
    print(f"Top-1 differs between modes: {n_top1_differ} / {len(comparisons)}")

    if rank_changes:
        print(f"True rule rank shifts (canonical - summed):")
        print(f"  Mean: {sum(rank_changes)/len(rank_changes):+.1f}")
        print(f"  Range: [{min(rank_changes):+d}, {max(rank_changes):+d}]")
        n_better = sum(1 for d in rank_changes if d > 0)
        n_worse = sum(1 for d in rank_changes if d < 0)
        n_same = sum(1 for d in rank_changes if d == 0)
        print(f"  Summed better: {n_better}, Same: {n_same}, Canonical better: {n_worse}")

    entropy_diffs = [
        comp["canonical"]["entropy"] - comp["summed"]["entropy"]
        for comp in comparisons
    ]
    print(f"\nEntropy difference (canonical - summed):")
    print(f"  Mean: {sum(entropy_diffs)/len(entropy_diffs):+.3f} bits")
    print(f"  Range: [{min(entropy_diffs):+.3f}, {max(entropy_diffs):+.3f}]")

    elapsed = time.time() - t0
    print(f"\nTotal runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
