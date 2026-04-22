"""Night 4 — multi-label C3: attribute unmapped MCMC mass to sources.

Per design doc v2, for every MCMC-visited program that is UNMAPPED (its
fingerprint is not in the enum pool under either direct or 2-stage lookup),
we record a predicate vector:

    predicates[prog_str] = {
        "visit_count":              int,
        "visit_mass":               float,    # count / total visits (rule)
        "parse_ok":                 bool,
        "type_ok":                  bool,
        "pruner_ok":                bool,
        "depth":                    int | None,
        "size":                     int | None,
        "in_d6_pool_hypothetical":  bool,     # type_ok AND pruner_ok AND depth <= 6
        "in_d7_pool_hypothetical":  bool,     # type_ok AND pruner_ok AND depth <= 7
        "fingerprint":              str | None,
        "canonical_type":           str | None,
    }

From the predicate vectors we derive SINGLE-LABEL attribution under TWO
priority orders (design doc v2 §"Multi-label C3"):

    depth-first:  depth > 6 → C.  else type_fail → B.  else pruner_rejected → D.  else residual (C).
    pruner-first: pruner_rejected → D.  else type_fail → B.  else depth > 6 → C.  else residual.

And the CO-OCCURRENCE MATRIX so the two orders can be compared (a program
that is both depth>6 and pruner-rejected gets attributed to C under
depth-first and to D under pruner-first — we show the overlap directly).

INPUTS:
  - Pool .pkl from an enum run (pool['equiv'], pool['stats']).
  - Raw visits JSON from an MCMC run (raw_visits/<rule>.json).
  - Optional d7 pool .pkl (for in_d7_pool lookup; falls back to depth<=7 heuristic).
  - max_list_chain for the pruner (default: MAX_LIST_CHAIN from enumerator).

OUTPUTS (per rule):
  - {out_dir}/{rule}_unmapped_predicates.json: per-program predicate vector +
    rule-level totals, priority-order attributions, co-occurrence matrix.

USAGE:
  # Pre-flight on Night 3 data (no d7 pool yet):
  python3 attribute_unmapped.py \\
      --pool  ../night3_mcmc_remediation/enum_depth6_300k/pool.pkl \\
      --visits ../night3_mcmc_remediation/mcmc_50k_4chains/raw_visits/all_red.json \\
      --out    preflight_output/

  # Full Night 4 run with d7 pool:
  python3 attribute_unmapped.py \\
      --pool  enum_depth7_500k/pool.pkl \\
      --d7-pool enum_depth7_500k/pool.pkl \\
      --visits mcmc_night4/raw_visits/all_red.json \\
      --out    night4_output/
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from dreamcoder_core.program import parse_program, Primitive, TypeContext  # noqa: E402
from dreamcoder_core.type_system import BOOL, HAND, Arrow  # noqa: E402
from gallery_analysis.enumerator import (  # noqa: E402
    MAX_LIST_CHAIN,
    build_gallery_grammar,
    is_syntactically_redundant,
    _make_evaluator,
)
from gallery_analysis.exemplars import generate_probe_set  # noqa: E402
from gallery_analysis.hypothesis_table import compute_fingerprint  # noqa: E402


# ---------------------------------------------------------------------------
# Pool indexing — same 2-stage fingerprint lookup as run_comparison.py
# ---------------------------------------------------------------------------

def build_fingerprint_indices(
    equiv_classes: List[Dict[str, Any]],
) -> Tuple[Dict[str, int], Dict[str, List[Tuple[str, int]]]]:
    """Return (direct_fp_to_cls, parent_fp_to_subclasses).

    Unsplit classes index directly by `fingerprint`. Strict-split sub-classes
    store `sha256(parent_fp|sub_fp)` and are indexed by parent_fp.
    """
    direct: Dict[str, int] = {}
    parent_to_subs: Dict[str, List[Tuple[str, int]]] = {}
    for i, c in enumerate(equiv_classes):
        parent = c.get("parent_fingerprint")
        if parent:
            parent_to_subs.setdefault(parent, []).append((c["fingerprint"], i))
        else:
            direct[c["fingerprint"]] = i
    return direct, parent_to_subs


def build_member_fp_fn(
    exemplar_hands, holdout_seed: int = 9999, n_holdout: int = 1000
):
    """Same holdout fingerprint function as analyze._strict_split_classes.

    Must match seed 9999 and n_holdout 1000 used in the night-3 pool.
    """
    holdout = generate_probe_set(n_probes=n_holdout, seed=holdout_seed)
    check_hands = list(exemplar_hands) + list(holdout)

    def member_fp(pred):
        bits = []
        for h in check_hands:
            try:
                bits.append("1" if pred(h) else "0")
            except Exception:
                bits.append("E")
        return "".join(bits)
    return member_fp


def compose_fp(parent_fp: str, sub_fp: str) -> str:
    return hashlib.sha256(f"{parent_fp}|{sub_fp}".encode()).hexdigest()


# ---------------------------------------------------------------------------
# Per-program predicate vector
# ---------------------------------------------------------------------------

def compute_predicates(
    prog_str: str,
    count: int,
    total_visits: int,
    prim_dict: Dict[str, Primitive],
    fp_probes,
    max_list_chain: int,
    d7_fp_to_cls: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Compute the predicate vector for a single unmapped MCMC program.

    Returns None fields when a preceding check fails (e.g., depth is None
    if parse_ok is False).
    """
    vec: Dict[str, Any] = {
        "prog_str": prog_str,
        "visit_count": count,
        "visit_mass": count / total_visits if total_visits else 0.0,
        "parse_ok": False,
        "type_ok": False,
        "pruner_ok": not is_syntactically_redundant(prog_str, max_list_chain),
        "depth": None,
        "size": None,
        "in_d6_pool_hypothetical": False,
        "in_d7_pool_hypothetical": False,
        "in_d7_pool_actual": None,   # None when d7 pool not provided
        "fingerprint": None,
        "canonical_type": None,
    }

    # Step 1: parse
    try:
        prog = parse_program(prog_str, prim_dict)
    except Exception as exc:
        vec["parse_error"] = f"{type(exc).__name__}: {exc}"
        return vec
    vec["parse_ok"] = True
    vec["depth"] = prog.depth()
    vec["size"] = prog.size()

    # Step 2: type check
    # Target type for a hand classifier is Arrow(HAND, BOOL) = list(card) -> bool.
    try:
        t = prog.infer_type(TypeContext(), [])
        t_str = str(t)
        target = Arrow(HAND, BOOL)
        vec["canonical_type"] = t_str
        # Compare by stringification to avoid type-var unification issues.
        vec["type_ok"] = (t_str == str(target))
    except Exception as exc:
        vec["type_error"] = f"{type(exc).__name__}: {exc}"
        vec["type_ok"] = False

    # Step 3: fingerprint for d7 lookup (only if type_ok — otherwise evaluator
    # will throw on most probes).
    if vec["type_ok"]:
        try:
            pred = _make_evaluator(prog)
            fp = compute_fingerprint(pred, fp_probes)
            vec["fingerprint"] = fp
            if d7_fp_to_cls is not None:
                vec["in_d7_pool_actual"] = fp in d7_fp_to_cls
        except Exception as exc:
            vec["fingerprint_error"] = f"{type(exc).__name__}: {exc}"

    # Step 4: derived hypotheticals
    # in_d6_pool_hypothetical = "would have been in d6 pool with unlimited budget"
    # ≈ type_ok AND pruner_ok AND depth <= 6
    vec["in_d6_pool_hypothetical"] = (
        vec["parse_ok"]
        and vec["type_ok"]
        and vec["pruner_ok"]
        and vec["depth"] is not None
        and vec["depth"] <= 6
    )
    vec["in_d7_pool_hypothetical"] = (
        vec["parse_ok"]
        and vec["type_ok"]
        and vec["pruner_ok"]
        and vec["depth"] is not None
        and vec["depth"] <= 7
    )

    return vec


# ---------------------------------------------------------------------------
# Single-label attribution under two priority orders
# ---------------------------------------------------------------------------

def label_depth_first(vec: Dict[str, Any]) -> str:
    """Attribution order: depth>6 → C. else type_fail → B. else pruner_rejected → D. else C_residual."""
    if not vec["parse_ok"]:
        return "PARSE_FAIL"
    if vec["depth"] is not None and vec["depth"] > 6:
        return "C"
    if not vec["type_ok"]:
        return "B"
    if not vec["pruner_ok"]:
        return "D"
    # Depth<=6 AND type_ok AND pruner_ok — should have been in d6 pool.
    # Most likely it WAS in pool but our fp differs (probe-set aliasing, E)
    # or the budget cap kicked in before hitting it.
    return "C_residual"


def label_pruner_first(vec: Dict[str, Any]) -> str:
    """Attribution order: pruner_rejected → D. else type_fail → B. else depth>6 → C. else C_residual."""
    if not vec["parse_ok"]:
        return "PARSE_FAIL"
    if not vec["pruner_ok"]:
        return "D"
    if not vec["type_ok"]:
        return "B"
    if vec["depth"] is not None and vec["depth"] > 6:
        return "C"
    return "C_residual"


# ---------------------------------------------------------------------------
# Per-rule driver
# ---------------------------------------------------------------------------

def attribute_rule(
    rule_id: str,
    visit_counts: Dict[str, int],
    equiv_classes: List[Dict[str, Any]],
    exemplar_hands,
    fp_probes,
    max_list_chain: int,
    d7_fp_to_cls: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    grammar = build_gallery_grammar()
    prim_dict = {
        prod.program.name: prod.program
        for prod in grammar.productions
        if isinstance(prod.program, Primitive)
    }

    direct_fp, parent_to_subs = build_fingerprint_indices(equiv_classes)
    member_fp_fn = build_member_fp_fn(exemplar_hands)

    total_visits = sum(visit_counts.values())

    unmapped_predicates: List[Dict[str, Any]] = []
    mass_mapped = 0
    mass_parse_fail = 0
    n_mapped = 0
    n_parse_fail = 0

    for prog_str, count in visit_counts.items():
        # First try to map it via the same path as run_comparison.py.
        try:
            prog = parse_program(prog_str, prim_dict)
            pred = _make_evaluator(prog)
            fp = compute_fingerprint(pred, fp_probes)
        except Exception:
            # Can't parse or evaluator fails: treat as parse_fail for
            # the mapping stage, but still record a predicate vector
            # for diagnostic purposes.
            n_parse_fail += 1
            mass_parse_fail += count
            vec = compute_predicates(
                prog_str, count, total_visits, prim_dict, fp_probes,
                max_list_chain, d7_fp_to_cls,
            )
            vec["mapped"] = False
            vec["unmapped_reason"] = "parse_or_eval_error"
            unmapped_predicates.append(vec)
            continue

        mapped = False
        if fp in direct_fp:
            mapped = True
        elif fp in parent_to_subs:
            try:
                sub = member_fp_fn(pred)
                comp = compose_fp(fp, sub)
                for stored_comp, _ in parent_to_subs[fp]:
                    if stored_comp == comp:
                        mapped = True
                        break
            except Exception:
                pass

        if mapped:
            n_mapped += 1
            mass_mapped += count
        else:
            vec = compute_predicates(
                prog_str, count, total_visits, prim_dict, fp_probes,
                max_list_chain, d7_fp_to_cls,
            )
            vec["mapped"] = False
            vec["unmapped_reason"] = "fp_not_in_pool"
            unmapped_predicates.append(vec)

    # Per-rule totals
    mass_unmapped = total_visits - mass_mapped - mass_parse_fail
    per_source_mass_depth_first: Dict[str, int] = {}
    per_source_mass_pruner_first: Dict[str, int] = {}
    cooccur: Dict[str, int] = {}  # "depth>6 & pruner_rejected" etc.

    for vec in unmapped_predicates:
        lbl_d = label_depth_first(vec)
        lbl_p = label_pruner_first(vec)
        per_source_mass_depth_first[lbl_d] = (
            per_source_mass_depth_first.get(lbl_d, 0) + vec["visit_count"]
        )
        per_source_mass_pruner_first[lbl_p] = (
            per_source_mass_pruner_first.get(lbl_p, 0) + vec["visit_count"]
        )
        bits = []
        if vec["depth"] is not None and vec["depth"] > 6:
            bits.append("depth>6")
        if vec["parse_ok"] and not vec["type_ok"]:
            bits.append("type_fail")
        if not vec["pruner_ok"]:
            bits.append("pruner_rejected")
        if not vec["parse_ok"]:
            bits.append("parse_fail")
        key = "+".join(bits) if bits else "d6_eligible_but_unmapped"
        cooccur[key] = cooccur.get(key, 0) + vec["visit_count"]

    # Normalize to fractions of total_visits for readability.
    def _frac(d: Dict[str, int]) -> Dict[str, float]:
        return {k: (v / total_visits if total_visits else 0.0) for k, v in d.items()}

    summary = {
        "rule_id": rule_id,
        "total_visits": total_visits,
        "n_distinct_programs": len(visit_counts),
        "n_mapped": n_mapped,
        "n_unmapped": len(unmapped_predicates) - n_parse_fail,
        "n_parse_fail": n_parse_fail,
        "mass_mapped_frac": mass_mapped / total_visits if total_visits else 0.0,
        "mass_unmapped_frac": mass_unmapped / total_visits if total_visits else 0.0,
        "mass_parse_fail_frac": mass_parse_fail / total_visits if total_visits else 0.0,
        "attribution_depth_first_mass_frac": _frac(per_source_mass_depth_first),
        "attribution_pruner_first_mass_frac": _frac(per_source_mass_pruner_first),
        "cooccurrence_mass_frac": _frac(cooccur),
        "max_list_chain": max_list_chain,
        "d7_pool_provided": d7_fp_to_cls is not None,
    }

    return {
        "summary": summary,
        "unmapped_predicates": unmapped_predicates,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", type=Path, required=True,
                    help="Path to enum pool.pkl (e.g. enum_depth6_300k/pool.pkl).")
    ap.add_argument("--visits", type=Path, required=True,
                    help="Path to raw_visits/<rule>.json for the rule to attribute.")
    ap.add_argument("--d7-pool", type=Path, default=None,
                    help="Optional: path to d7 pool.pkl for in_d7_pool lookup.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output directory; writes <rule>_unmapped_predicates.json.")
    ap.add_argument("--max-list-chain", type=int, default=MAX_LIST_CHAIN,
                    help=f"Pruner max_list_chain (default from enumerator: {MAX_LIST_CHAIN}).")
    args = ap.parse_args()

    print(f"Loading pool {args.pool}...")
    with open(args.pool, "rb") as f:
        pool = pickle.load(f)
    equiv = pool["equiv"]
    fp_probes = pool["stats"]["_probes"]
    exemplar_hands = pool["stats"]["_exemplar_hands"]
    print(f"  {len(equiv)} classes, {len(fp_probes)} probes, {len(exemplar_hands)} exemplars.")

    d7_fp_to_cls: Optional[Dict[str, int]] = None
    if args.d7_pool is not None:
        print(f"Loading d7 pool {args.d7_pool}...")
        with open(args.d7_pool, "rb") as f:
            d7_pool = pickle.load(f)
        # Only direct fps (no 2-stage for d7 lookup; lookup is a heuristic bool).
        d7_fp_to_cls = {
            c["fingerprint"]: i
            for i, c in enumerate(d7_pool["equiv"])
            if not c.get("parent_fingerprint")
        }
        print(f"  {len(d7_fp_to_cls)} direct-fp classes in d7 pool.")

    print(f"Loading visits {args.visits}...")
    blob = json.loads(args.visits.read_text())
    rule_id = blob["rule_id"]
    visit_counts = blob["visit_counts"]
    print(f"  rule={rule_id}, {len(visit_counts)} programs, "
          f"total_steps={blob.get('total_steps')}.")

    print(f"Attributing {rule_id}...")
    result = attribute_rule(
        rule_id=rule_id,
        visit_counts=visit_counts,
        equiv_classes=equiv,
        exemplar_hands=exemplar_hands,
        fp_probes=fp_probes,
        max_list_chain=args.max_list_chain,
        d7_fp_to_cls=d7_fp_to_cls,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"{rule_id}_unmapped_predicates.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"Wrote {out_path}")
    print("\nSummary:")
    s = result["summary"]
    print(f"  mass_mapped_frac:      {s['mass_mapped_frac']:.4f}")
    print(f"  mass_unmapped_frac:    {s['mass_unmapped_frac']:.4f}")
    print(f"  mass_parse_fail_frac:  {s['mass_parse_fail_frac']:.4f}")
    print(f"\n  Depth-first attribution (fractions of total visits):")
    for k, v in sorted(s["attribution_depth_first_mass_frac"].items(), key=lambda kv: -kv[1]):
        print(f"    {k:>20}: {v:.4f}")
    print(f"\n  Pruner-first attribution (fractions of total visits):")
    for k, v in sorted(s["attribution_pruner_first_mass_frac"].items(), key=lambda kv: -kv[1]):
        print(f"    {k:>20}: {v:.4f}")
    print(f"\n  Co-occurrence (bitsets among unmapped, fractions of total visits):")
    for k, v in sorted(s["cooccurrence_mass_frac"].items(), key=lambda kv: -kv[1]):
        print(f"    {k:>40}: {v:.4f}")


if __name__ == "__main__":
    main()
