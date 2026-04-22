"""Decompose the 'd6_eligible_but_unmapped' bucket (Source E candidates).

Split each unmapped program that (parse_ok AND type_ok AND pruner_ok AND depth<=6)
into:
  (a) strict-split miss: direct fp matches a PARENT class in the d6 pool, but the
      program's composite (holdout-hand) fingerprint does not match any sub-class.
      Interpretation: probe-set / holdout partition instability (Source E).
  (b) truly unseen fingerprint: direct fp matches nothing in the pool.
      Interpretation: budget-cap truncation (enum capped at 300k programs) or a
      genuinely missing class.

For (a), additionally reports:
  - which parent classes absorb the misses (and how many sub-classes already exist)
  - example MCMC programs per parent

Usage:
  python3 investigate_residual.py <rule1> <rule2> ...   # reads preflight_output/
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
NIGHT3 = ROOT / "review-stage/experiments/night3_mcmc_remediation"
NIGHT4 = ROOT / "review-stage/experiments/night4_attribution"
POOL_PKL = NIGHT3 / "enum_depth6_300k/pool.pkl"
PREFLIGHT_DIR = NIGHT4 / "preflight_output"


def main(rules: list[str]) -> None:
    with open(POOL_PKL, "rb") as f:
        pool = pickle.load(f)

    direct_fp_to_cls: dict[str, int] = {}
    parent_to_subs: dict[str, set[str]] = defaultdict(set)
    for i, cls in enumerate(pool["equiv"]):
        pfp = cls.get("parent_fingerprint")
        if pfp:
            parent_to_subs[pfp].add(cls["fingerprint"])
        else:
            direct_fp_to_cls[cls["fingerprint"]] = i

    print(
        f"Pool: {len(pool['equiv'])} classes, "
        f"{len(direct_fp_to_cls)} direct, {len(parent_to_subs)} split parents "
        f"(total subclasses: {sum(len(v) for v in parent_to_subs.values())})"
    )
    print()

    for rule in rules:
        pf = PREFLIGHT_DIR / f"{rule}_unmapped_predicates.json"
        if not pf.exists():
            print(f"[skip] {rule}: no preflight output at {pf}")
            continue
        with open(pf) as f:
            d = json.load(f)

        preds = d["unmapped_predicates"]
        d6_elig = [
            p for p in preds
            if p["parse_ok"] and p["type_ok"] and p["pruner_ok"] and p["depth"] <= 6
        ]

        ss_miss = [p for p in d6_elig if p["fingerprint"] in parent_to_subs]
        unseen = [p for p in d6_elig if p["fingerprint"] not in parent_to_subs
                  and p["fingerprint"] not in direct_fp_to_cls]

        ss_mass = sum(p["visit_mass"] for p in ss_miss)
        un_mass = sum(p["visit_mass"] for p in unseen)

        print(f"=== {rule} ===")
        print(f"  unmapped mass (total visits): {d['summary']['mass_unmapped_frac']:.4f}")
        print(f"  d6_eligible_but_unmapped: {len(d6_elig)} progs, {ss_mass + un_mass:.4f} mass")
        print(f"    (a) strict-split miss : {len(ss_miss)} progs, {ss_mass:.4f} mass")
        print(f"    (b) truly unseen fp   : {len(unseen)} progs, {un_mass:.4f} mass")

        if ss_miss:
            by_parent: dict[str, list] = defaultdict(list)
            for p in ss_miss:
                by_parent[p["fingerprint"]].append(p)
            parent_rows = sorted(
                (
                    (pfp, sum(x["visit_mass"] for x in progs), len(progs),
                     len(parent_to_subs[pfp]))
                    for pfp, progs in by_parent.items()
                ),
                key=lambda r: -r[1],
            )
            print(f"    SS-miss targets {len(parent_rows)} distinct parent classes:")
            for pfp, mass, n_progs, n_subs in parent_rows[:5]:
                print(
                    f"      parent {pfp[:12]}... : mass={mass:.4f}, "
                    f"n_mcmc_progs={n_progs}, n_subs_in_pool={n_subs}"
                )

        if unseen:
            print(f"    truly-unseen depth histogram:")
            depth_mass: dict[int, float] = defaultdict(float)
            depth_n: dict[int, int] = defaultdict(int)
            for p in unseen:
                depth_mass[p["depth"]] += p["visit_mass"]
                depth_n[p["depth"]] += 1
            for d_ in sorted(depth_mass.keys()):
                print(f"      depth={d_}: n={depth_n[d_]}, mass={depth_mass[d_]:.5f}")

        print()


if __name__ == "__main__":
    rules = sys.argv[1:] if len(sys.argv) > 1 else [
        "all_red", "all_even", "all_same_suit", "ranks_palindrome",
    ]
    main(rules)
