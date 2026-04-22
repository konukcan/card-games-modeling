"""
Night 4 — final per-rule attribution synthesis.

Cross-references C1 (night4 MCMC 100k, per-chain, β=1 tail) against BOTH the
d6 pool (night3) and the d7 pool (night4/C2). For each of the 6 rules we
produce three attributions:

  A_d6  = attribution of night4 MCMC visits using d6 pool
          (what the gap would look like without C2)
  A_d7  = attribution of night4 MCMC visits using d7 pool
          (what C2 rescues)
  ΔC2   = A_d6.mass_mapped  →  A_d7.mass_mapped  (budget-cap rescue mass)

Then ΔA (MCMC convergence rescue) can be read from the design doc by comparing
A_d6 here vs the night3 pre-flight A_d6 on the night3 50k MCMC — both against
the same d6 pool. If longer + β=1-tail MCMC reduces unmapped mass on its own,
we attribute the delta to Source A. If C2 explains most of the residual on
top of that, we attribute to budget-cap.

Pure post-processing. Reads from disk only. Reuses attribute_rule from
attribute_unmapped.py so the 2-stage fingerprint mapper is identical.

USAGE:
  python3 synthesize_attribution.py                # all rules in config
  python3 synthesize_attribution.py --rules all_red all_same_suit  # subset
  python3 synthesize_attribution.py --skip-d7      # only d6 pass (useful
                                                     before C2 finishes)

OUTPUTS:
  synthesis/<rule>_attribution_d6.json    — full predicate vectors, d6 pool
  synthesis/<rule>_attribution_d7.json    — full predicate vectors, d7 pool
  synthesis/summary.json                  — cross-rule table (machine-readable)
  stdout human-readable cross-rule table
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cloudpickle as pickle  # predicates are local closures

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))  # for attribute_unmapped
sys.path.insert(0, str(REPO / "src"))

from attribute_unmapped import attribute_rule  # noqa: E402

CONFIG = json.loads((HERE / "config.json").read_text())
OUT_DIR = HERE / "synthesis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

D6_POOL = REPO / "review-stage/experiments/night3_mcmc_remediation/enum_depth6_300k/pool.pkl"
D7_POOL = HERE / CONFIG["enumeration"]["output_dir"] / "pool.pkl"
MCMC_DIR = HERE / CONFIG["persistence"]["output_dir"]


# Rule buckets from the Night 4 consensus review (see config.json _rule_bucket_note)
RULE_BUCKETS = {
    "all_red":                  "tight-agree",
    "all_same_suit":            "tight-agree",
    "ranks_palindrome":         "tight-agree",
    "pair_5s_adjacent":         "borderline",
    "all_even":                 "divergent",
    "four_of_a_kind_adjacent":  "divergent",
}


def run_attribution_one_pool(
    pool_path: Path, visits_path: Path, d7_fp_to_cls: Optional[Dict[str, int]] = None
) -> Dict[str, Any]:
    """Thin wrapper: load pool, load visits, run attribute_rule, return result."""
    with open(pool_path, "rb") as f:
        pool = pickle.load(f)
    blob = json.loads(visits_path.read_text())
    rule_id = blob["rule_id"]
    visit_counts = blob["visit_counts"]
    from gallery_analysis.enumerator import MAX_LIST_CHAIN  # local import
    result = attribute_rule(
        rule_id=rule_id,
        visit_counts=visit_counts,
        equiv_classes=pool["equiv"],
        exemplar_hands=pool["stats"]["_exemplar_hands"],
        fp_probes=pool["stats"]["_probes"],
        max_list_chain=MAX_LIST_CHAIN,
        d7_fp_to_cls=d7_fp_to_cls,
    )
    return result


def _unpack_summary(result: Dict[str, Any]) -> Dict[str, float]:
    """Extract the scalar fields we want in the cross-rule table."""
    s = result["summary"]
    df = s["attribution_depth_first_mass_frac"]
    pf = s["attribution_pruner_first_mass_frac"]
    return {
        "mass_mapped":            s["mass_mapped_frac"],
        "mass_unmapped":          s["mass_unmapped_frac"],
        "mass_parse_fail":        s["mass_parse_fail_frac"],
        "src_C_depth_first":      df.get("C", 0.0),
        "src_B_depth_first":      df.get("B", 0.0),
        "src_D_depth_first":      df.get("D", 0.0),
        "src_residual_depth_first": df.get("C_residual", 0.0),
        "src_C_pruner_first":     pf.get("C", 0.0),
        "src_B_pruner_first":     pf.get("B", 0.0),
        "src_D_pruner_first":     pf.get("D", 0.0),
        "src_residual_pruner_first": pf.get("C_residual", 0.0),
    }


def build_d7_direct_fp_index(d7_pool_path: Path) -> Dict[str, int]:
    """For `in_d7_pool_actual` predicate-vector flag (direct fps only)."""
    with open(d7_pool_path, "rb") as f:
        d7 = pickle.load(f)
    return {
        c["fingerprint"]: i
        for i, c in enumerate(d7["equiv"])
        if not c.get("parent_fingerprint")
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rules", nargs="+", default=None,
                    help="Rules to synthesize (defaults to config.rules).")
    ap.add_argument("--skip-d7", action="store_true",
                    help="Only run the d6 attribution pass (useful if d7 pool "
                         "not yet built).")
    args = ap.parse_args()

    rules = args.rules if args.rules else CONFIG["rules"]

    # Pre-flight existence checks (pools + MCMC outputs).
    missing: List[str] = []
    if not D6_POOL.exists():
        missing.append(str(D6_POOL))
    if not args.skip_d7 and not D7_POOL.exists():
        missing.append(str(D7_POOL))
    mcmc_raw = MCMC_DIR / "raw_visits"
    if not mcmc_raw.exists():
        missing.append(str(mcmc_raw))
    if missing:
        print("[error] Missing artifacts:", flush=True)
        for m in missing:
            print(f"  - {m}", flush=True)
        print("[error] Run the corresponding jobs first, or pass --skip-d7.",
              flush=True)
        sys.exit(1)

    d7_fp_to_cls: Optional[Dict[str, int]] = None
    if not args.skip_d7:
        print(f"[synth] Building d7 direct-fp index from {D7_POOL}...", flush=True)
        d7_fp_to_cls = build_d7_direct_fp_index(D7_POOL)
        print(f"[synth]   {len(d7_fp_to_cls):,} direct-fp classes.", flush=True)

    cross_rule: List[Dict[str, Any]] = []
    for rule in rules:
        visits_path = mcmc_raw / f"{rule}.json"
        if not visits_path.exists():
            print(f"[skip] {rule}: no MCMC visits at {visits_path}", flush=True)
            cross_rule.append({"rule_id": rule, "missing_visits": True})
            continue

        print(f"[synth] {rule} — attributing against d6 pool...", flush=True)
        t0 = time.time()
        res_d6 = run_attribution_one_pool(D6_POOL, visits_path, d7_fp_to_cls)
        (OUT_DIR / f"{rule}_attribution_d6.json").write_text(
            json.dumps(res_d6, indent=2, default=str)
        )
        sum_d6 = _unpack_summary(res_d6)
        elapsed_d6 = time.time() - t0
        print(f"[synth]   d6: mapped={sum_d6['mass_mapped']:.4f} "
              f"unmapped={sum_d6['mass_unmapped']:.4f} ({elapsed_d6:.1f}s)",
              flush=True)

        sum_d7 = None
        if not args.skip_d7:
            print(f"[synth] {rule} — attributing against d7 pool...", flush=True)
            t0 = time.time()
            res_d7 = run_attribution_one_pool(D7_POOL, visits_path, d7_fp_to_cls)
            (OUT_DIR / f"{rule}_attribution_d7.json").write_text(
                json.dumps(res_d7, indent=2, default=str)
            )
            sum_d7 = _unpack_summary(res_d7)
            elapsed_d7 = time.time() - t0
            print(f"[synth]   d7: mapped={sum_d7['mass_mapped']:.4f} "
                  f"unmapped={sum_d7['mass_unmapped']:.4f} ({elapsed_d7:.1f}s)",
                  flush=True)

        row: Dict[str, Any] = {
            "rule_id":        rule,
            "bucket":         RULE_BUCKETS.get(rule, "unknown"),
            "total_visits":   res_d6["summary"]["total_visits"],
            "d6":             sum_d6,
        }
        if sum_d7 is not None:
            row["d7"] = sum_d7
            row["d7_minus_d6_mapped"] = sum_d7["mass_mapped"] - sum_d6["mass_mapped"]
        cross_rule.append(row)

    # Cross-rule summary
    summary = {
        "config": {
            "rules":   rules,
            "d6_pool": str(D6_POOL),
            "d7_pool": str(D7_POOL) if not args.skip_d7 else None,
            "mcmc":    str(MCMC_DIR),
        },
        "rows": cross_rule,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {OUT_DIR / 'summary.json'}", flush=True)

    # Human-readable cross-rule table
    print()
    print("=== Cross-rule attribution summary ===", flush=True)
    if args.skip_d7:
        hdr = (
            f"{'rule':<28}  {'bucket':<12}  {'mapped':>7}  {'unmap':>6}  "
            f"{'C':>6}  {'B':>6}  {'D':>6}  {'resid':>6}  (depth-first @ d6)"
        )
    else:
        hdr = (
            f"{'rule':<28}  {'bucket':<12}  "
            f"{'d6_map':>7}  {'d7_map':>7}  {'Δmap':>7}  "
            f"{'d7_resid':>9}  {'d7_B':>6}  {'d7_D':>6}"
        )
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for row in cross_rule:
        if row.get("missing_visits"):
            print(f"{row['rule_id']:<28}  [no MCMC visits]", flush=True)
            continue
        d6 = row["d6"]
        if args.skip_d7:
            print(
                f"{row['rule_id']:<28}  {row['bucket']:<12}  "
                f"{d6['mass_mapped']:>7.4f}  {d6['mass_unmapped']:>6.4f}  "
                f"{d6['src_C_depth_first']:>6.4f}  "
                f"{d6['src_B_depth_first']:>6.4f}  "
                f"{d6['src_D_depth_first']:>6.4f}  "
                f"{d6['src_residual_depth_first']:>6.4f}",
                flush=True,
            )
        else:
            d7 = row["d7"]
            print(
                f"{row['rule_id']:<28}  {row['bucket']:<12}  "
                f"{d6['mass_mapped']:>7.4f}  {d7['mass_mapped']:>7.4f}  "
                f"{row['d7_minus_d6_mapped']:>7.4f}  "
                f"{d7['src_residual_depth_first']:>9.4f}  "
                f"{d7['src_B_depth_first']:>6.4f}  "
                f"{d7['src_D_depth_first']:>6.4f}",
                flush=True,
            )

    print()
    print("Reading guide (per design doc v2):", flush=True)
    print("  Δmap = d7_mapped - d6_mapped  (Source-budget-cap rescue from C2)", flush=True)
    print("  d7_resid = depth-first residual under d7 pool (Source A floor —", flush=True)
    print("             what neither longer MCMC nor richer enum can explain)", flush=True)
    print("  B = type-invalid, D = pruner-rejected (should both be tiny)", flush=True)


if __name__ == "__main__":
    main()
