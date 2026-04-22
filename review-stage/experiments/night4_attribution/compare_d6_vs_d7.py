"""
Night 4 — Pool-level comparison between night3 (d6, 300k) and night4 (d7, 500k).

C2 analysis script. Tests the budget-cap hypothesis from RESIDUAL_FINDING.md
by asking three questions:

  Q1 (pool shape): How many classes did d7 add? How many are entirely new
      (direct fingerprint not present in d6)? How does the strict-split
      sub-class population change?

  Q2 (posterior shape per rule): For each of the 6 rules, how much of d7's
      retained posterior mass lives on fingerprints that were NOT in d6's
      direct-fp universe? Is the top-1 class stable? Does d7's top-20
      overlap d6's top-20?

  Q3 (budget-cap signal): If d7 rescues residual mass from d6 (same
      fingerprints appearing in both, but d6 had truncated), we expect
      higher retained_mass_d7 but similar top-K set. If d7 finds GENUINELY
      new classes (depth-7-only programs), we expect new fingerprints with
      non-trivial posterior mass.

Pure post-processing. No enumeration, no MCMC. Reads two already-computed
enum artifacts off disk.

USAGE:
  python3 compare_d6_vs_d7.py            # reads config.json for rule list
  python3 compare_d6_vs_d7.py --rules all_red all_same_suit   # subset

OUTPUTS:
  compare_output/pool_stats.json         — pool-level overlap counts
  compare_output/per_rule.json           — rule-by-rule mass movement
  stdout human-readable summary table
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import cloudpickle as pickle  # predicates are local closures

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

CONFIG = json.loads((HERE / "config.json").read_text())
OUT_DIR = HERE / "compare_output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

D6_POOL = REPO / "review-stage/experiments/night3_mcmc_remediation/enum_depth6_300k/pool.pkl"
D6_POST_DIR = REPO / "review-stage/experiments/night3_mcmc_remediation/enum_depth6_300k/posteriors"
D7_POOL = HERE / CONFIG["enumeration"]["output_dir"] / "pool.pkl"
D7_POST_DIR = HERE / CONFIG["enumeration"]["output_dir"] / "posteriors"


def _pool_fp_sets(equiv: List[Dict[str, Any]]) -> Tuple[Set[str], Set[str], int, int]:
    """Return (direct_fps, parent_fps, n_direct, n_split_sub).

    direct_fps: fingerprints of unsplit classes.
    parent_fps: parent_fingerprints of strict-split sub-classes.
    """
    direct_fps: Set[str] = set()
    parent_fps: Set[str] = set()
    n_direct = 0
    n_split_sub = 0
    for c in equiv:
        parent = c.get("parent_fingerprint")
        if parent:
            parent_fps.add(parent)
            n_split_sub += 1
        else:
            direct_fps.add(c["fingerprint"])
            n_direct += 1
    return direct_fps, parent_fps, n_direct, n_split_sub


def compute_pool_stats() -> Dict[str, Any]:
    """Load both pools and compare class/fingerprint counts."""
    print(f"[pool] Loading d6 pool: {D6_POOL}", flush=True)
    with open(D6_POOL, "rb") as f:
        d6 = pickle.load(f)
    print(f"[pool] Loading d7 pool: {D7_POOL}", flush=True)
    with open(D7_POOL, "rb") as f:
        d7 = pickle.load(f)

    d6_direct, d6_parents, n6_direct, n6_split = _pool_fp_sets(d6["equiv"])
    d7_direct, d7_parents, n7_direct, n7_split = _pool_fp_sets(d7["equiv"])

    # Fingerprint-level overlap on direct classes (the universe that MCMC
    # programs typically land in via the first-stage lookup).
    shared = d6_direct & d7_direct
    d6_only = d6_direct - d7_direct
    d7_only = d7_direct - d6_direct

    # Probe-set hash should be identical since both runs use seed=42, n=500.
    stats = {
        "d6_n_classes_total":   len(d6["equiv"]),
        "d7_n_classes_total":   len(d7["equiv"]),
        "d6_n_direct_fps":      n6_direct,
        "d7_n_direct_fps":      n7_direct,
        "d6_n_split_subclasses": n6_split,
        "d7_n_split_subclasses": n7_split,
        "d6_n_parents_with_split": len(d6_parents),
        "d7_n_parents_with_split": len(d7_parents),
        "shared_direct_fps":    len(shared),
        "d6_only_direct_fps":   len(d6_only),
        "d7_only_direct_fps":   len(d7_only),
        "d6_probe_hash":        d6.get("probe_hash"),
        "d7_probe_hash":        d7.get("probe_hash"),
        "probe_hash_match":     d6.get("probe_hash") == d7.get("probe_hash"),
    }
    return stats, d6_direct, d7_direct


def compare_rule_posteriors(
    rule_id: str, d6_direct: Set[str], d7_direct: Set[str]
) -> Dict[str, Any]:
    """For one rule, load d6 and d7 posteriors and quantify mass movement."""
    d6_path = D6_POST_DIR / f"{rule_id}.json"
    d7_path = D7_POST_DIR / f"{rule_id}.json"
    if not d6_path.exists() or not d7_path.exists():
        return {
            "rule_id": rule_id,
            "missing": {"d6": not d6_path.exists(), "d7": not d7_path.exists()},
        }

    d6 = json.loads(d6_path.read_text())
    d7 = json.loads(d7_path.read_text())

    # Full posterior: list of {prob, cls_idx, fingerprint}
    d6_post = d6["full_posterior"]
    d7_post = d7["full_posterior"]

    d6_fp_to_prob = {h["fingerprint"]: h["prob"] for h in d6_post}
    d7_fp_to_prob = {h["fingerprint"]: h["prob"] for h in d7_post}

    # Mass in d7 that lives on fingerprints NOT present in d6's direct-fp universe.
    # Using d6_direct (not d6_fp_to_prob) because a fp can exist in d6's pool but
    # not make the per-rule posterior threshold.
    mass_d7_on_new_fps = sum(
        p for fp, p in d7_fp_to_prob.items() if fp not in d6_direct
    )
    mass_d7_on_fps_absent_from_d6_posterior = sum(
        p for fp, p in d7_fp_to_prob.items() if fp not in d6_fp_to_prob
    )

    # Top-K overlap (K=20)
    d6_top20 = [h["fingerprint"] for h in d6["top20"]]
    d7_top20 = [h["fingerprint"] for h in d7["top20"]]
    overlap20 = len(set(d6_top20) & set(d7_top20))
    top1_stable = (d6_top20[0] == d7_top20[0]) if d6_top20 and d7_top20 else False

    # Total-variation distance on the union of fingerprints
    union = set(d6_fp_to_prob) | set(d7_fp_to_prob)
    tv = 0.5 * sum(
        abs(d6_fp_to_prob.get(fp, 0.0) - d7_fp_to_prob.get(fp, 0.0))
        for fp in union
    )

    return {
        "rule_id": rule_id,
        "d6_n_hyps": d6["n_hyps_in_posterior"],
        "d7_n_hyps": d7["n_hyps_in_posterior"],
        "d6_retained_mass": d6["enum_retained_mass"],
        "d7_retained_mass": d7["enum_retained_mass"],
        "mass_d7_on_new_direct_fps":            mass_d7_on_new_fps,
        "mass_d7_on_fps_absent_from_d6_posterior": mass_d7_on_fps_absent_from_d6_posterior,
        "top20_overlap":                         overlap20,
        "top1_stable":                           top1_stable,
        "d6_top1_fp":                            d6_top20[0] if d6_top20 else None,
        "d7_top1_fp":                            d7_top20[0] if d7_top20 else None,
        "d6_top1_prog":                          d6["top20"][0]["canonical_program"] if d6["top20"] else None,
        "d7_top1_prog":                          d7["top20"][0]["canonical_program"] if d7["top20"] else None,
        "total_variation_d6_vs_d7":              tv,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rules", nargs="+", default=None,
                    help="Rules to compare (defaults to config.rules).")
    args = ap.parse_args()

    rules = args.rules if args.rules else CONFIG["rules"]

    # Pre-flight existence checks
    missing: List[str] = []
    for p in (D6_POOL, D7_POOL):
        if not p.exists():
            missing.append(str(p))
    if missing:
        print("[error] Missing artifacts:", flush=True)
        for m in missing:
            print(f"  - {m}", flush=True)
        print("[error] Run the corresponding enum scripts before this comparison.",
              flush=True)
        sys.exit(1)

    # -------- Q1: pool-level --------
    pool_stats, d6_direct, d7_direct = compute_pool_stats()

    print()
    print("=== Pool-level comparison (d6 vs d7) ===", flush=True)
    print(f"{'metric':<35} {'d6':>12} {'d7':>12}", flush=True)
    print("-" * 61, flush=True)
    print(f"{'classes (total)':<35} {pool_stats['d6_n_classes_total']:>12,} {pool_stats['d7_n_classes_total']:>12,}", flush=True)
    print(f"{'direct fingerprints':<35} {pool_stats['d6_n_direct_fps']:>12,} {pool_stats['d7_n_direct_fps']:>12,}", flush=True)
    print(f"{'strict-split sub-classes':<35} {pool_stats['d6_n_split_subclasses']:>12,} {pool_stats['d7_n_split_subclasses']:>12,}", flush=True)
    print(f"{'parents with split':<35} {pool_stats['d6_n_parents_with_split']:>12,} {pool_stats['d7_n_parents_with_split']:>12,}", flush=True)
    print(f"shared direct fingerprints: {pool_stats['shared_direct_fps']:,}", flush=True)
    print(f"d6-only direct fingerprints: {pool_stats['d6_only_direct_fps']:,}", flush=True)
    print(f"d7-only direct fingerprints: {pool_stats['d7_only_direct_fps']:,}  (new-at-d7)", flush=True)
    print(f"probe hash match: {pool_stats['probe_hash_match']}", flush=True)

    (OUT_DIR / "pool_stats.json").write_text(json.dumps(pool_stats, indent=2))
    print(f"\nwrote {OUT_DIR / 'pool_stats.json'}", flush=True)

    # -------- Q2/Q3: per-rule --------
    print()
    print("=== Per-rule posterior mass movement ===", flush=True)
    hdr = (
        f"{'rule':<28}  {'d6_hyp':>6}  {'d7_hyp':>6}  "
        f"{'d6_ret':>7}  {'d7_ret':>7}  "
        f"{'d7_new_fp_mass':>14}  {'top1_same':>9}  {'top20_ov':>8}  {'TV(d6,d7)':>9}"
    )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)

    per_rule: List[Dict[str, Any]] = []
    for rule in rules:
        r = compare_rule_posteriors(rule, d6_direct, d7_direct)
        per_rule.append(r)
        if r.get("missing"):
            print(f"{rule:<28}  [MISSING: d6={r['missing']['d6']} d7={r['missing']['d7']}]",
                  flush=True)
            continue
        print(
            f"{rule:<28}  "
            f"{r['d6_n_hyps']:>6,}  "
            f"{r['d7_n_hyps']:>6,}  "
            f"{r['d6_retained_mass']:>7.4f}  "
            f"{r['d7_retained_mass']:>7.4f}  "
            f"{r['mass_d7_on_new_direct_fps']:>14.4f}  "
            f"{str(r['top1_stable']):>9}  "
            f"{r['top20_overlap']:>8}  "
            f"{r['total_variation_d6_vs_d7']:>9.4f}",
            flush=True,
        )

    (OUT_DIR / "per_rule.json").write_text(
        json.dumps({"rules": per_rule}, indent=2)
    )
    print(f"\nwrote {OUT_DIR / 'per_rule.json'}", flush=True)

    print()
    print("Reading guide:", flush=True)
    print("  d7_new_fp_mass  = posterior mass at d7 on fingerprints NOT in d6's", flush=True)
    print("                    direct-fp universe. Large = depth-7-only classes", flush=True)
    print("                    matter. Small + d7_ret >> d6_ret = d6 was budget-", flush=True)
    print("                    truncated on the same classes (rescue).", flush=True)
    print("  TV(d6,d7)       = total-variation of the two rule-posteriors over", flush=True)
    print("                    the union of fingerprints.", flush=True)


if __name__ == "__main__":
    main()
