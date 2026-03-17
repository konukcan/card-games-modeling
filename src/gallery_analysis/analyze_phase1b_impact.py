"""
Comprehensive analysis of Phase 1b LLM-generated hypothesis injection impact.

Compares the Bayesian hypothesis space and posterior distributions WITH vs
WITHOUT Phase 1b hypotheses across all 60 gallery rules.

Usage:
    cd src
    python3 gallery_analysis/analyze_phase1b_impact.py
"""
import sys
import time
import json
import math
import copy
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from gallery_analysis.analyze import (
    build_hypothesis_pool,
    estimate_extensions,
    score_rule,
)
from gallery_analysis.injection import load_and_validate_injections, merge_injected
from gallery_analysis.enumerator import build_gallery_grammar
from gallery_analysis.exemplars import load_exemplars, generate_probe_set
from gallery_analysis.hypothesis_table import compute_fingerprint
from gallery_analysis.gallery_rules import GALLERY_RULES
from gallery_analysis.bayesian_scorer import normalize_posteriors


# =========================================================================
# Helpers
# =========================================================================

def log_sum_exp(values):
    """Numerically stable log-sum-exp."""
    if not values:
        return float('-inf')
    mx = max(values)
    if mx == float('-inf'):
        return float('-inf')
    return mx + math.log(sum(math.exp(v - mx) for v in values))


def separate_injections(all_injected):
    """Split validated injections into Phase 0 and Phase 1b."""
    phase0 = [h for h in all_injected if not h.get('id', '').startswith('phase1b__')]
    phase1b = [h for h in all_injected if h.get('id', '').startswith('phase1b__')]
    return phase0, phase1b


# =========================================================================
# Main analysis
# =========================================================================

def main():
    t_start = time.time()
    inject_path = 'gallery_analysis/data/injected_hypotheses.json'

    print("=" * 80)
    print("PHASE 1b INJECTION IMPACT ANALYSIS")
    print("=" * 80)

    # --- Step 1: Build shared enumerated hypothesis pool ---
    print("\n[1/6] Building enumerated hypothesis pool (depth=5)...")
    equiv_classes_enum, pipeline_stats = build_hypothesis_pool(
        max_depth=5,
        max_programs=50_000,
        max_cost=25.0,
        timeout=120.0,
        n_probes=500,
        max_list_chain=2,
        verbose=1,
    )
    n_enum_classes = len(equiv_classes_enum)
    print(f"  Enumerated equivalence classes: {n_enum_classes}")

    # --- Step 2: Load and separate injections ---
    print("\n[2/6] Loading and separating injections...")
    grammar = build_gallery_grammar()

    enum_priors = [c["canonical_prior"] for c in equiv_classes_enum]
    enumerated_prior_range = (min(enum_priors), max(enum_priors))

    all_injected = load_and_validate_injections(
        inject_path, grammar=grammar,
        enumerated_prior_range=enumerated_prior_range,
    )
    phase0, phase1b = separate_injections(all_injected)
    print(f"  Total injected: {len(all_injected)}")
    print(f"  Phase 0 (true rules + early LLM): {len(phase0)}")
    print(f"  Phase 1b (LLM foils): {len(phase1b)}")

    # --- Step 3: Build three versions of the hypothesis space ---
    print("\n[3/6] Building hypothesis space variants...")
    probes = generate_probe_set(n_probes=500, seed=42)

    # Version A: enum + Phase 0 only (no Phase 1b)
    classes_no_1b = merge_injected(equiv_classes_enum, phase0, probes)
    n_no_1b = len(classes_no_1b)

    # Version B: enum + all injections (Phase 0 + Phase 1b)
    classes_with_1b = merge_injected(equiv_classes_enum, all_injected, probes)
    n_with_1b = len(classes_with_1b)

    print(f"  Without Phase 1b: {n_no_1b} equivalence classes")
    print(f"  With Phase 1b:    {n_with_1b} equivalence classes")
    print(f"  Novel classes from Phase 1b: {n_with_1b - n_no_1b}")

    # --- Step 3b: Classify Phase 1b hypotheses as novel vs merged ---
    print("\n  Classifying Phase 1b hypotheses...")
    # Build fingerprint index from the no-1b space
    fp_index_no_1b = {c["fingerprint"]: i for i, c in enumerate(classes_no_1b)}

    phase1b_novel = []
    phase1b_merged = []
    phase1b_novel_fps = set()
    for inj in phase1b:
        fp = compute_fingerprint(inj["predicate"], probes)
        if fp in fp_index_no_1b:
            phase1b_merged.append(inj)
        else:
            phase1b_novel.append(inj)
            phase1b_novel_fps.add(fp)

    print(f"  Phase 1b novel (new fingerprint): {len(phase1b_novel)}")
    print(f"  Phase 1b merged (existing fingerprint): {len(phase1b_merged)}")
    print(f"  Unique novel fingerprints: {len(phase1b_novel_fps)}")

    # --- Step 4: Estimate extensions for both variants ---
    print("\n[4/6] Estimating extension sizes...")
    ext_no_1b = estimate_extensions(
        classes_no_1b, n_samples=10_000, verbose=1,
        cache_path='/tmp/ext_cache_no1b.json',
    )
    ext_with_1b = estimate_extensions(
        classes_with_1b, n_samples=10_000, verbose=1,
        cache_path='/tmp/ext_cache_with1b.json',
    )

    # --- Step 5: Score all 60 rules both ways ---
    print("\n[5/6] Scoring all 60 rules (both variants)...")
    exemplars = load_exemplars()

    # Build true-rule fingerprint lookups for both
    def get_true_rule_fps(classes):
        fps = {}
        for cls in classes:
            for rid in cls.get("true_for_rules", []):
                fps[rid] = cls["fingerprint"]
            single = cls.get("true_for_rule")
            if single and single not in fps:
                fps[single] = cls["fingerprint"]
        return fps

    true_fps_no_1b = get_true_rule_fps(classes_no_1b)
    true_fps_with_1b = get_true_rule_fps(classes_with_1b)

    results_no_1b = {}
    results_with_1b = {}
    n_rules = 0

    for rule_id, rule_info in GALLERY_RULES.items():
        if rule_id not in exemplars:
            continue
        n_rules += 1

        hands = exemplars[rule_id]["hands_primary"]

        r_no = score_rule(
            rule_id=rule_id,
            exemplar_hands=hands,
            equivalence_classes=classes_no_1b,
            extensions=ext_no_1b,
            true_rule_fingerprint=true_fps_no_1b.get(rule_id),
        )
        results_no_1b[rule_id] = r_no

        r_with = score_rule(
            rule_id=rule_id,
            exemplar_hands=hands,
            equivalence_classes=classes_with_1b,
            extensions=ext_with_1b,
            true_rule_fingerprint=true_fps_with_1b.get(rule_id),
        )
        results_with_1b[rule_id] = r_with

        if n_rules % 10 == 0:
            print(f"  Scored {n_rules} rules...", flush=True)

    print(f"  Done: {n_rules} rules scored.")

    # --- Step 6: Analysis & Reporting ---
    print("\n[6/6] Analyzing results...\n")

    # =====================================================================
    # ANALYSIS 1: Phase 1b hypotheses in top-10 posteriors
    # =====================================================================
    print("=" * 80)
    print("ANALYSIS 1: PHASE 1b HYPOTHESES IN TOP-10 POSTERIORS")
    print("=" * 80)

    # Build a set of Phase 1b injection IDs for lookup
    phase1b_ids = {h["id"] for h in phase1b}

    # For each rule, check if any top-10 hypothesis in the with-1b results
    # belongs to a class that contains Phase 1b injections
    phase1b_in_top10 = []

    # Build fingerprint -> Phase 1b IDs mapping in the with_1b classes
    fp_to_phase1b_ids = defaultdict(set)
    for inj in phase1b:
        fp = compute_fingerprint(inj["predicate"], probes)
        fp_to_phase1b_ids[fp].add(inj["id"])

    # Also need fingerprints for the with_1b classes
    fp_to_class_idx = {}
    for i, cls in enumerate(classes_with_1b):
        fp_to_class_idx[cls["fingerprint"]] = i

    for rule_id in sorted(results_with_1b.keys()):
        r = results_with_1b[rule_id]
        # We need to check fingerprints of top-10 hypotheses
        # The top_hypotheses dict doesn't store fingerprints, so we need
        # to match via the full scoring results. Let's re-score minimally.
        # Actually, top_hypotheses has 'program' - we can match via classes_with_1b

        for rank_idx, hyp in enumerate(r["top_hypotheses"][:10]):
            prog = hyp["program"]
            # Find the class this program belongs to
            for cls in classes_with_1b:
                if cls["canonical_program"] == prog:
                    fp = cls["fingerprint"]
                    p1b_ids_in_class = fp_to_phase1b_ids.get(fp, set())
                    if p1b_ids_in_class:
                        phase1b_in_top10.append({
                            "rule_id": rule_id,
                            "rank": rank_idx + 1,
                            "posterior_mass": hyp["probability"],
                            "program": prog,
                            "phase1b_ids": sorted(p1b_ids_in_class),
                            "n_expressions": hyp["n_expressions"],
                            "is_novel_class": fp in phase1b_novel_fps,
                        })
                    break

    if phase1b_in_top10:
        print(f"\nFound {len(phase1b_in_top10)} Phase 1b appearances in top-10 posteriors:\n")
        print(f"{'Rule':<35} {'Rank':>4} {'Post%':>7} {'Novel?':>6} {'Program':<50} {'Phase1b IDs'}")
        print("-" * 140)
        for entry in sorted(phase1b_in_top10, key=lambda x: (x["rule_id"], x["rank"])):
            ids_str = ", ".join(entry["phase1b_ids"][:3])
            if len(entry["phase1b_ids"]) > 3:
                ids_str += f" (+{len(entry['phase1b_ids'])-3} more)"
            novel_str = "YES" if entry["is_novel_class"] else "no"
            print(f"{entry['rule_id']:<35} {entry['rank']:>4} "
                  f"{entry['posterior_mass']*100:>6.2f}% {novel_str:>6} "
                  f"{entry['program'][:50]:<50} {ids_str}")
    else:
        print("\nNo Phase 1b hypotheses appear in top-10 posteriors for any rule.")

    # =====================================================================
    # ANALYSIS 2: Equivalence class expansion per rule
    # =====================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 2: HYPOTHESIS SPACE EXPANSION (WITH vs WITHOUT Phase 1b)")
    print("=" * 80)

    print(f"\nOverall:")
    print(f"  Classes without Phase 1b: {n_no_1b}")
    print(f"  Classes with Phase 1b:    {n_with_1b}")
    print(f"  New classes added:        {n_with_1b - n_no_1b}")
    print(f"  Phase 1b hypotheses that merged into existing: {len(phase1b_merged)}")
    print(f"  Phase 1b hypotheses that created novel classes: {len(phase1b_novel)}")
    print(f"  (Some novel hypotheses share fingerprints: {len(phase1b_novel)} hyps -> {len(phase1b_novel_fps)} unique classes)")

    # Per-rule: count how many hypotheses hit all exemplars (confusers)
    print(f"\nPer-rule confuser counts (hypotheses hitting all 6 exemplars):")
    print(f"{'Rule':<35} {'No 1b':>7} {'With 1b':>8} {'Delta':>6} {'Delta%':>7}")
    print("-" * 70)

    confuser_deltas = []
    for rule_id in sorted(results_no_1b.keys()):
        c_no = results_no_1b[rule_id]["n_with_all_hits"]
        c_with = results_with_1b[rule_id]["n_with_all_hits"]
        delta = c_with - c_no
        pct = (delta / max(c_no, 1)) * 100
        confuser_deltas.append((rule_id, c_no, c_with, delta, pct))
        print(f"{rule_id:<35} {c_no:>7} {c_with:>8} {delta:>+6} {pct:>+6.1f}%")

    # Summary
    total_delta = sum(d[3] for d in confuser_deltas)
    avg_delta = total_delta / len(confuser_deltas)
    print(f"\nTotal confuser increase across all rules: {total_delta}")
    print(f"Average confuser increase per rule: {avg_delta:.1f}")

    # =====================================================================
    # ANALYSIS 3: Prior distribution comparison
    # =====================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 3: LOG-PRIOR DISTRIBUTION COMPARISON")
    print("=" * 80)

    # Collect log-priors by source
    enum_priors_all = [c["canonical_prior"] for c in equiv_classes_enum]
    phase0_priors = [h["log_prior"] for h in phase0]
    phase1b_priors = [h["log_prior"] for h in phase1b]

    def prior_stats(priors, label):
        if not priors:
            print(f"  {label}: (empty)")
            return
        mn = min(priors)
        mx = max(priors)
        avg = sum(priors) / len(priors)
        med = sorted(priors)[len(priors) // 2]
        print(f"  {label} (n={len(priors):>4}): "
              f"min={mn:>8.2f}  max={mx:>7.2f}  mean={avg:>7.2f}  median={med:>7.2f}")

    print()
    prior_stats(enum_priors_all, "Enumerated    ")
    prior_stats(phase0_priors,   "Phase 0 (LLM) ")
    prior_stats(phase1b_priors,  "Phase 1b (LLM)")

    # Histogram buckets
    print("\n  Log-prior distribution (histogram):")
    all_lps = enum_priors_all + phase0_priors + phase1b_priors
    bucket_min = math.floor(min(all_lps))
    bucket_max = math.ceil(max(all_lps))
    buckets = list(range(bucket_min, bucket_max + 1, 2))

    print(f"  {'Bucket':<14} {'Enum':>6} {'Ph0':>6} {'Ph1b':>6}")
    for b_start in buckets:
        b_end = b_start + 2
        n_e = sum(1 for p in enum_priors_all if b_start <= p < b_end)
        n_0 = sum(1 for p in phase0_priors if b_start <= p < b_end)
        n_1 = sum(1 for p in phase1b_priors if b_start <= p < b_end)
        if n_e + n_0 + n_1 > 0:
            print(f"  [{b_start:>5},{b_end:>5}) {n_e:>6} {n_0:>6} {n_1:>6}")

    # =====================================================================
    # ANALYSIS 4: Per-rule coverage gain
    # =====================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 4: PER-RULE COVERAGE GAIN FROM PHASE 1b")
    print("=" * 80)

    # Sort by delta (descending)
    confuser_deltas.sort(key=lambda x: -x[3])

    print(f"\nTop 10 rules gaining MOST new confusers from Phase 1b:")
    print(f"{'Rule':<35} {'Before':>7} {'After':>7} {'Delta':>6}")
    print("-" * 60)
    for rule_id, c_no, c_with, delta, pct in confuser_deltas[:10]:
        print(f"{rule_id:<35} {c_no:>7} {c_with:>7} {delta:>+6}")

    print(f"\nRules gaining FEWEST (or zero) new confusers from Phase 1b:")
    print(f"{'Rule':<35} {'Before':>7} {'After':>7} {'Delta':>6}")
    print("-" * 60)
    for rule_id, c_no, c_with, delta, pct in confuser_deltas[-10:]:
        print(f"{rule_id:<35} {c_no:>7} {c_with:>7} {delta:>+6}")

    # =====================================================================
    # ANALYSIS 5: Posterior impact (entropy & rank changes)
    # =====================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 5: POSTERIOR IMPACT (ENTROPY & RANK CHANGES)")
    print("=" * 80)

    print(f"\n{'Rule':<35} {'Ent_no':>7} {'Ent_w':>7} {'dEnt':>6} "
          f"{'TrRk_no':>7} {'TrRk_w':>7} {'Top1_no':>7} {'Top1_w':>7}")
    print("-" * 100)

    entropy_changes = []
    for rule_id in sorted(results_no_1b.keys()):
        r_no = results_no_1b[rule_id]
        r_with = results_with_1b[rule_id]
        e_no = r_no["difficulty"]["posterior_entropy"]
        e_with = r_with["difficulty"]["posterior_entropy"]
        d_e = e_with - e_no
        t1_no = r_no["difficulty"]["top1_probability"]
        t1_with = r_with["difficulty"]["top1_probability"]
        tr_no = r_no.get("true_rule_rank", "—")
        tr_with = r_with.get("true_rule_rank", "—")
        tr_no_str = f"{tr_no:>7}" if isinstance(tr_no, int) else f"{'—':>7}"
        tr_with_str = f"{tr_with:>7}" if isinstance(tr_with, int) else f"{'—':>7}"

        entropy_changes.append((rule_id, e_no, e_with, d_e, t1_no, t1_with, tr_no, tr_with))
        print(f"{rule_id:<35} {e_no:>7.3f} {e_with:>7.3f} {d_e:>+6.3f} "
              f"{tr_no_str} {tr_with_str} "
              f"{t1_no*100:>6.1f}% {t1_with*100:>6.1f}%")

    # Summary stats
    d_ents = [x[3] for x in entropy_changes]
    print(f"\nEntropy change summary:")
    print(f"  Mean delta entropy: {sum(d_ents)/len(d_ents):+.4f}")
    print(f"  Max increase: {max(d_ents):+.4f}")
    print(f"  Max decrease: {min(d_ents):+.4f}")
    print(f"  Rules with increased entropy: {sum(1 for d in d_ents if d > 0.001)}")
    print(f"  Rules with decreased entropy: {sum(1 for d in d_ents if d < -0.001)}")
    print(f"  Rules unchanged (|d| < 0.001): {sum(1 for d in d_ents if abs(d) <= 0.001)}")

    # True rule rank changes
    rank_changes = []
    for rule_id, _, _, _, _, _, tr_no, tr_with in entropy_changes:
        if isinstance(tr_no, int) and isinstance(tr_with, int):
            rank_changes.append((rule_id, tr_no, tr_with, tr_with - tr_no))

    if rank_changes:
        print(f"\nTrue-rule rank changes (where both known):")
        worsened = [(r, n, w, d) for r, n, w, d in rank_changes if d > 0]
        improved = [(r, n, w, d) for r, n, w, d in rank_changes if d < 0]
        unchanged = [(r, n, w, d) for r, n, w, d in rank_changes if d == 0]
        print(f"  Improved: {len(improved)}, Unchanged: {len(unchanged)}, Worsened: {len(worsened)}")
        if worsened:
            print(f"  Worsened rules:")
            for r, n, w, d in sorted(worsened, key=lambda x: -x[3]):
                print(f"    {r:<35} rank {n} -> {w} ({d:+d})")
        if improved:
            print(f"  Improved rules:")
            for r, n, w, d in sorted(improved, key=lambda x: x[3]):
                print(f"    {r:<35} rank {n} -> {w} ({d:+d})")

    # =====================================================================
    # ANALYSIS 6: Most interesting novel Phase 1b equivalence classes
    # =====================================================================
    print("\n" + "=" * 80)
    print("ANALYSIS 6: MOST INTERESTING NOVEL PHASE 1b EQUIVALENCE CLASSES")
    print("=" * 80)

    # Find novel classes in classes_with_1b that are NOT in classes_no_1b
    fps_no_1b = {c["fingerprint"] for c in classes_no_1b}
    novel_classes = [c for c in classes_with_1b if c["fingerprint"] not in fps_no_1b]

    print(f"\n  Total novel classes from Phase 1b: {len(novel_classes)}")

    # Sort by summed_prior (highest = most plausible under grammar)
    novel_classes.sort(key=lambda c: -c["summed_prior"])

    print(f"\n  Top 20 novel classes by prior (most plausible new confusers):")
    print(f"  {'Rank':<5} {'Log-Prior':>10} {'#Expr':>6} {'Program':<65} {'IDs'}")
    print("  " + "-" * 130)
    for i, cls in enumerate(novel_classes[:20]):
        ids = cls.get("injection_ids", [])
        ids_str = ", ".join(ids[:2])
        if len(ids) > 2:
            ids_str += f" (+{len(ids)-2})"
        print(f"  {i+1:<5} {cls['summed_prior']:>10.2f} {cls['n_expressions']:>6} "
              f"{cls['canonical_program'][:65]:<65} {ids_str}")

    # Which rules do these novel classes confuse?
    print(f"\n  Which rules do novel Phase 1b classes confuse? (top confusers)")
    # For each novel class, check which rules' exemplars it hits
    novel_confusion_map = defaultdict(list)
    for cls in novel_classes:
        pred = cls["predicate"]
        for rule_id in sorted(GALLERY_RULES.keys()):
            if rule_id not in exemplars:
                continue
            hands = exemplars[rule_id]["hands_primary"]
            hits = sum(1 for h in hands if _safe_eval(pred, h))
            if hits == len(hands):
                novel_confusion_map[rule_id].append(cls)

    rules_by_novel_confusers = sorted(
        novel_confusion_map.items(), key=lambda x: -len(x[1])
    )
    print(f"  {'Rule':<35} {'Novel confusers':>15}")
    print("  " + "-" * 55)
    for rule_id, clss in rules_by_novel_confusers[:15]:
        print(f"  {rule_id:<35} {len(clss):>15}")

    # =====================================================================
    # SUMMARY
    # =====================================================================
    t_total = time.time() - t_start
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
  Phase 1b injection: {len(phase1b)} hypotheses
  Novel equivalence classes added: {n_with_1b - n_no_1b}
  Merged into existing classes: {len(phase1b_merged)}

  Hypothesis space: {n_no_1b} -> {n_with_1b} classes ({n_with_1b - n_no_1b:+d}, {((n_with_1b - n_no_1b)/n_no_1b)*100:+.1f}%)

  Posterior impact:
    Mean entropy change: {sum(d_ents)/len(d_ents):+.4f} nats
    Rules with entropy increase: {sum(1 for d in d_ents if d > 0.001)}
    Rules with entropy decrease: {sum(1 for d in d_ents if d < -0.001)}

  Phase 1b in top-10 posteriors: {len(phase1b_in_top10)} appearances across {len(set(e['rule_id'] for e in phase1b_in_top10))} rules

  Total analysis time: {t_total:.1f}s
""")


def _safe_eval(pred, hand):
    """Safely evaluate a predicate on a hand."""
    try:
        return pred(hand)
    except Exception:
        return False


if __name__ == "__main__":
    main()
