"""Targeted 5k-holdout probe: does using 5000 holdout hands (vs the pool's 1000)
rescue the strict-split misses?

For each parent class that absorbs SS-miss mass:
  1. Build a 5k-holdout fingerprint for every sub-class (using its canonical
     predicate). This mimics what _strict_split_classes would produce if it
     used n_holdout=5000 instead of 1000.
  2. For each MCMC SS-miss program targeting this parent, compute its 5k
     fingerprint. Check whether it now matches any sub-class's 5k fingerprint.

Reports per rule:
  - ss_mass_at_1k:       SS-miss mass with pool's 1000-holdout partition (status quo).
  - ss_mass_rescued_5k:  mass that now maps to some sub-class at 5k.
  - ss_mass_still_unmapped_5k: mass that STILL doesn't match any sub-class at 5k.

Interpretation:
  - rescued@5k dominant → Source E is holdout-count-dependent; fix by bumping n_holdout.
  - still@5k  dominant → MCMC programs are genuinely new sub-classes (Source C / enum-budget).
"""
from __future__ import annotations

import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from dreamcoder_core.program import parse_program, Primitive  # noqa: E402
from gallery_analysis.enumerator import (  # noqa: E402
    build_gallery_grammar,
    _make_evaluator,
)
from gallery_analysis.exemplars import generate_probe_set  # noqa: E402

NIGHT3 = REPO / "review-stage/experiments/night3_mcmc_remediation"
NIGHT4 = REPO / "review-stage/experiments/night4_attribution"
POOL_PKL = NIGHT3 / "enum_depth6_300k/pool.pkl"
PREFLIGHT_DIR = NIGHT4 / "preflight_output"


def bitvec(pred, probes) -> str:
    """Same holdout-signature construction as analyze._strict_split_classes."""
    bits = []
    for h in probes:
        try:
            bits.append("1" if pred(h) else "0")
        except Exception:
            bits.append("E")
    return "".join(bits)


def main() -> None:
    with open(POOL_PKL, "rb") as f:
        pool = pickle.load(f)

    parent_to_sub_idxs: dict[str, list[int]] = defaultdict(list)
    for i, cls in enumerate(pool["equiv"]):
        pfp = cls.get("parent_fingerprint")
        if pfp:
            parent_to_sub_idxs[pfp].append(i)

    # 5k holdout probes, seed=9999 (prefix-compatible with pool's 1k at same seed).
    # Also use the same exemplar_hands the pool used (to match _strict_split
    # signature construction: exemplars + holdouts).
    print("Generating 5000 holdout probes (seed=9999)...", flush=True)
    probes_5k = generate_probe_set(n_probes=5000, seed=9999)
    exemplar_hands = pool["stats"].get("_exemplar_hands") or []
    check_hands = list(exemplar_hands) + list(probes_5k)
    print(f"  exemplars: {len(exemplar_hands)}; total check hands: {len(check_hands)}",
          flush=True)

    # Build 5k signature for every sub-class, grouped by parent.
    # Use the sub-class's stored predicate (a canonical_program closure).
    print("Computing 5k signatures for pool sub-classes...", flush=True)
    parent_sub_5k: dict[str, dict[str, int]] = {}
    n_total_subs = sum(len(v) for v in parent_to_sub_idxs.values())
    done = 0
    for pfp, sub_idxs in parent_to_sub_idxs.items():
        sub_sigs: dict[str, int] = {}
        for idx in sub_idxs:
            cls = pool["equiv"][idx]
            pred = cls.get("predicate")
            if pred is None:
                continue
            sig = bitvec(pred, check_hands)
            sub_sigs.setdefault(sig, idx)
            done += 1
        parent_sub_5k[pfp] = sub_sigs
    print(f"  done: {done}/{n_total_subs} sub-class signatures computed", flush=True)

    # Parse MCMC programs once using the project's primitive dict.
    grammar = build_gallery_grammar()
    prim_dict = {
        prod.program.name: prod.program
        for prod in grammar.productions
        if isinstance(prod.program, Primitive)
    }

    rules = ["all_red", "all_even", "all_same_suit", "ranks_palindrome"]
    print()
    print(f"{'rule':<20}  {'ss@1k_mass':>10}  {'rescued@5k':>10}  "
          f"{'still@5k':>10}  {'eval_fail':>10}")
    print("-" * 75)

    summary_rows = []
    for rule in rules:
        pf = PREFLIGHT_DIR / f"{rule}_unmapped_predicates.json"
        with open(pf) as f:
            d = json.load(f)
        preds = d["unmapped_predicates"]

        ss_miss = [
            p for p in preds
            if p["parse_ok"] and p["type_ok"] and p["pruner_ok"]
            and p["depth"] <= 6 and p["fingerprint"] in parent_to_sub_idxs
        ]
        total_ss_mass = sum(p["visit_mass"] for p in ss_miss)

        mass_rescued = 0.0
        mass_still = 0.0
        mass_eval_fail = 0.0
        for p in ss_miss:
            try:
                prog = parse_program(p["prog_str"], prim_dict)
                pred = _make_evaluator(prog)
                sig = bitvec(pred, check_hands)
            except Exception:
                mass_eval_fail += p["visit_mass"]
                continue

            sub_sigs = parent_sub_5k.get(p["fingerprint"], {})
            if sig in sub_sigs:
                mass_rescued += p["visit_mass"]
            else:
                mass_still += p["visit_mass"]

        print(f"{rule:<20}  {total_ss_mass:>10.4f}  {mass_rescued:>10.4f}  "
              f"{mass_still:>10.4f}  {mass_eval_fail:>10.4f}")
        summary_rows.append({
            "rule": rule,
            "ss_mass_at_1k": total_ss_mass,
            "rescued_at_5k": mass_rescued,
            "still_at_5k": mass_still,
            "eval_fail": mass_eval_fail,
        })

    # Write summary
    out_path = NIGHT4 / "preflight_output" / "holdout_5k_probe.json"
    out_path.write_text(json.dumps({"rows": summary_rows}, indent=2))
    print()
    print(f"Saved: {out_path.relative_to(REPO)}")
    print()
    print("Interpretation:")
    print("  rescued@5k dominant → Source E is holdout-count-dependent; bump n_holdout.")
    print("  still@5k  dominant → MCMC programs are new sub-classes (Source C / budget).")


if __name__ == "__main__":
    main()
