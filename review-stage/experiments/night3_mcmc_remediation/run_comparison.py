"""
Night 3 — class-aggregated posterior comparison.

For each rule:
  1. Parse every MCMC-visited program string, compile to predicate, compute
     its FINGERPRINT on the same 500 probes the enum pool was fingerprinted
     with, and map it to an enum equivalence class (or mark as MCMC-only).
  2. Build MCMC posterior over CLASSES by summing visit counts per class,
     renormalizing over the mapped subset.  Report unmapped + parse-failure
     mass as validity diagnostics.
  3. Compare the enum and MCMC posteriors over the same class universe:
       * per-class absolute difference (KL/JS optional)
       * top-k overlap
       * extensional p_accept agreement on 5000 shared comparison probes
  4. Gate with Night 2's VALIDITY_THRESHOLDS, augmented with two new
     MCMC-side diagnostics:
       - max_mcmc_unmapped_mass
       - max_mcmc_parse_failure_mass

Outputs:
    comparison/question_a/{rule}.json   — using shared adaptive-ladder exts
    comparison/convergence_diagnostics/{rule}.json
                                        — per-checkpoint comparison snapshot
    comparison/summary.json             — pass/fail counts, headline metrics
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import cloudpickle

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from dreamcoder_core.program import parse_program, Primitive
from gallery_analysis.enumerator import build_gallery_grammar, _make_evaluator
from gallery_analysis.exemplars import load_exemplars, generate_probe_set
from gallery_analysis.hand_diagnosticity import compute_posteriors_for_rule
from gallery_analysis.hypothesis_table import compute_fingerprint


CONFIG = json.loads((HERE / "config.json").read_text())
ENUM_DIR = HERE / "enum_depth6_300k"
MCMC_DIR = HERE / "mcmc_50k_4chains"
OUT_DIR = HERE / "comparison"
(OUT_DIR / "question_a").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "convergence_diagnostics").mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Validity thresholds (pre-registered; extended from Night 2 with MCMC-side
# class-mapping diagnostics).
# ---------------------------------------------------------------------------
VALIDITY_THRESHOLDS = dict(CONFIG["validity_thresholds"])
VALIDITY_THRESHOLDS.setdefault("max_mcmc_unmapped_mass", 0.10)
VALIDITY_THRESHOLDS.setdefault("max_mcmc_parse_failure_mass", 0.05)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_prim_dict(grammar) -> Dict[str, Primitive]:
    out: Dict[str, Primitive] = {}
    for prod in grammar.productions:
        if isinstance(prod.program, Primitive):
            out[prod.program.name] = prod.program
    return out


def _build_comparison_probes(n_probes: int, seed: int):
    return generate_probe_set(n_probes=n_probes, seed=seed)


def _spearman(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")

    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def _jensen_shannon(p: List[float], q: List[float]) -> float:
    """Jensen-Shannon divergence between two prob distributions (base 2)."""
    if len(p) != len(q):
        return float("nan")
    sp = sum(p)
    sq = sum(q)
    if sp <= 0 or sq <= 0:
        return float("nan")
    p = [x / sp for x in p]
    q = [x / sq for x in q]
    m = [(p[i] + q[i]) / 2 for i in range(len(p))]

    def kl(a, b):
        s = 0.0
        for i in range(len(a)):
            if a[i] > 0 and b[i] > 0:
                s += a[i] * math.log2(a[i] / b[i])
        return s

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


# ---------------------------------------------------------------------------
# Class aggregation: MCMC programs → enum classes via fingerprint
# ---------------------------------------------------------------------------

def _build_fingerprint_index(
    equiv_classes: List[Dict[str, Any]],
) -> Dict[str, int]:
    """fingerprint → cls_idx lookup."""
    return {c["fingerprint"]: i for i, c in enumerate(equiv_classes)}


def _aggregate_mcmc_to_classes(
    visit_counts: Dict[str, int],
    grammar,
    prim_dict: Dict[str, Primitive],
    fp_probes,
    fp_to_cls: Dict[str, int],
) -> Tuple[Dict[int, int], Dict[str, Any]]:
    """Map each MCMC program → fingerprint → class idx; sum counts per class.

    Returns:
        class_counts: {cls_idx: total_visit_count_mapped_there}
        audit: {
            total_visits, n_programs,
            n_parse_fail, mass_parse_fail,
            n_predicate_fail, mass_predicate_fail,
            n_mapped, mass_mapped,
            n_unmapped, mass_unmapped,
            n_mcmc_only_fps, mcmc_only_fps: [(fp, mass), ...]  # top 20
        }
    """
    total_visits = sum(visit_counts.values())
    if total_visits == 0:
        return {}, {"total_visits": 0, "error": "empty_visit_counts"}

    class_counts: Dict[int, int] = {}
    unmapped_fp_counts: Dict[str, int] = {}
    n_parse_fail = mass_parse_fail = 0
    n_predicate_fail = mass_predicate_fail = 0
    n_mapped = mass_mapped = 0
    n_unmapped = mass_unmapped = 0

    for prog_str, count in visit_counts.items():
        try:
            program = parse_program(prog_str, prim_dict)
        except Exception:
            n_parse_fail += 1
            mass_parse_fail += count
            continue
        try:
            pred = _make_evaluator(program)
            fp = compute_fingerprint(pred, fp_probes)
        except Exception:
            n_predicate_fail += 1
            mass_predicate_fail += count
            continue
        if fp in fp_to_cls:
            idx = fp_to_cls[fp]
            class_counts[idx] = class_counts.get(idx, 0) + count
            n_mapped += 1
            mass_mapped += count
        else:
            unmapped_fp_counts[fp] = unmapped_fp_counts.get(fp, 0) + count
            n_unmapped += 1
            mass_unmapped += count

    top_unmapped = sorted(
        unmapped_fp_counts.items(), key=lambda kv: -kv[1],
    )[:20]
    audit = {
        "total_visits": total_visits,
        "n_programs": len(visit_counts),
        "n_parse_fail": n_parse_fail,
        "mass_parse_fail_frac": mass_parse_fail / total_visits,
        "n_predicate_fail": n_predicate_fail,
        "mass_predicate_fail_frac": mass_predicate_fail / total_visits,
        "n_mapped_programs": n_mapped,
        "mass_mapped_frac": mass_mapped / total_visits,
        "n_unmapped_programs": n_unmapped,
        "mass_unmapped_frac": mass_unmapped / total_visits,
        "n_distinct_unmapped_fps": len(unmapped_fp_counts),
        "top_unmapped_fps": [
            {"fingerprint": fp[:16] + "...", "count": c,
             "mass_frac": c / total_visits}
            for fp, c in top_unmapped
        ],
    }
    return class_counts, audit


# ---------------------------------------------------------------------------
# Per-rule comparison
# ---------------------------------------------------------------------------

def _enum_p_accept_per_hand(
    posteriors, equiv_classes, probe_hands,
) -> Tuple[List[float], int]:
    out = []
    exc = 0
    for h in probe_hands:
        p = 0.0
        for prob, cls_idx, _ in posteriors:
            try:
                if equiv_classes[cls_idx]["predicate"](h):
                    p += prob
            except Exception:
                exc += 1
        out.append(p)
    return out, exc


def _mcmc_class_p_accept_per_hand(
    class_posterior: Dict[int, float],
    equiv_classes,
    probe_hands,
) -> Tuple[List[float], int]:
    out = []
    exc = 0
    for h in probe_hands:
        p = 0.0
        for cls_idx, w in class_posterior.items():
            try:
                if equiv_classes[cls_idx]["predicate"](h):
                    p += w
            except Exception:
                exc += 1
        out.append(p)
    return out, exc


def _normalize(counts: Dict[int, int]) -> Dict[int, float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counts.items()}


def _union_top_k(
    enum_post: List[Tuple[float, int, Any]],
    mcmc_post: Dict[int, float],
    k: int,
) -> Dict[str, Any]:
    enum_top = [(p, ci) for (p, ci, _) in enum_post[:k]]
    mcmc_top = sorted(
        mcmc_post.items(), key=lambda kv: -kv[1],
    )[:k]
    enum_top_cls = {ci for _, ci in enum_top}
    mcmc_top_cls = {ci for ci, _ in mcmc_top}
    overlap = enum_top_cls & mcmc_top_cls
    return {
        "k": k,
        "enum_top_k_classes": sorted(enum_top_cls),
        "mcmc_top_k_classes": sorted(mcmc_top_cls),
        "overlap_size": len(overlap),
        "overlap_frac_of_k": len(overlap) / max(1, k),
        "enum_top1_cls": enum_top[0][1] if enum_top else None,
        "mcmc_top1_cls": mcmc_top[0][0] if mcmc_top else None,
    }


def _divergence_metrics(
    enum_post: List[Tuple[float, int, Any]],
    mcmc_post_classes: Dict[int, float],
    n_classes: int,
) -> Dict[str, float]:
    """Compute over the full enum+mcmc class universe."""
    all_cls = set(ci for _, ci, _ in enum_post) | set(mcmc_post_classes.keys())
    enum_dict = {ci: p for p, ci, _ in enum_post}
    xs = [enum_dict.get(ci, 0.0) for ci in all_cls]
    ys = [mcmc_post_classes.get(ci, 0.0) for ci in all_cls]
    abs_diffs = [abs(a - b) for a, b in zip(xs, ys)]
    return {
        "n_union_classes": len(all_cls),
        "n_enum_classes": len(enum_dict),
        "n_mcmc_classes": len(mcmc_post_classes),
        "total_variation": sum(abs_diffs) / 2,
        "max_abs_diff_on_class": max(abs_diffs) if abs_diffs else 0.0,
        "jensen_shannon": _jensen_shannon(xs, ys),
        "spearman_rank_corr": _spearman(xs, ys),
    }


def _build_validity_flags(
    parse_audit: Dict[str, Any],
    extensional: Dict[str, float],
    enum_retained_mass: Optional[float],
    n_enum_exc: int,
    n_mcmc_exc: int,
) -> Dict[str, bool]:
    t = VALIDITY_THRESHOLDS
    enum_trunc = (
        enum_retained_mass is not None
        and enum_retained_mass < t["min_enum_retained_mass"]
    )
    return {
        "mcmc_parse_drop_excessive": (
            parse_audit["mass_parse_fail_frac"] > t["max_mcmc_parse_failure_mass"]
        ),
        "mcmc_unmapped_excessive": (
            parse_audit["mass_unmapped_frac"] > t["max_mcmc_unmapped_mass"]
        ),
        "enum_truncation_excessive": enum_trunc,
        "rare_rule_blind_spot": (
            max(extensional["probe_hit_rate_enum"],
                extensional["probe_hit_rate_mcmc"])
            < t["min_probe_hit_rate_either"]
        ),
        "predicate_exceptions_present": (
            (n_enum_exc + n_mcmc_exc) > t["max_predicate_exceptions"]
        ),
    }


def _summarize_extensional(
    enum_pacc: List[float], mcmc_pacc: List[float],
) -> Dict[str, float]:
    n = len(enum_pacc)
    diffs = [abs(a - b) for a, b in zip(enum_pacc, mcmc_pacc)]
    return {
        "n_probes": n,
        "mean_abs_diff": sum(diffs) / max(1, n),
        "max_abs_diff": max(diffs) if diffs else 0.0,
        "spearman_rank_corr": _spearman(enum_pacc, mcmc_pacc),
        "fraction_disagreement_above_0_5": (
            sum(1 for d in diffs if d > 0.5) / max(1, n)
        ),
        "probe_hit_rate_enum": sum(enum_pacc) / max(1, n),
        "probe_hit_rate_mcmc": sum(mcmc_pacc) / max(1, n),
    }


def compare_one_rule(
    rule_id: str,
    equiv_classes,
    fp_probes,
    fp_to_cls: Dict[str, int],
    grammar,
    prim_dict: Dict[str, Primitive],
    enum_posteriors,
    enum_retained_mass: float,
    mcmc_visit_counts: Dict[str, int],
    comparison_probes,
    top_k_summary: int,
) -> Dict[str, Any]:
    t0 = time.time()
    # Aggregate MCMC to classes
    class_counts, parse_audit = _aggregate_mcmc_to_classes(
        mcmc_visit_counts, grammar, prim_dict, fp_probes, fp_to_cls,
    )
    mcmc_cls_post = _normalize(class_counts)

    # Divergence metrics on class universe
    divergence = _divergence_metrics(
        enum_posteriors, mcmc_cls_post, len(equiv_classes),
    )
    topk = _union_top_k(enum_posteriors, mcmc_cls_post, top_k_summary)

    # Extensional comparison on shared comparison probes
    enum_pacc, n_enum_exc = _enum_p_accept_per_hand(
        enum_posteriors, equiv_classes, comparison_probes,
    )
    mcmc_pacc, n_mcmc_exc = _mcmc_class_p_accept_per_hand(
        mcmc_cls_post, equiv_classes, comparison_probes,
    )
    extensional = _summarize_extensional(enum_pacc, mcmc_pacc)

    # Validity gating
    flags = _build_validity_flags(
        parse_audit, extensional, enum_retained_mass,
        n_enum_exc, n_mcmc_exc,
    )
    comparison_valid = not any(flags.values())

    # Top classes from each side for human inspection
    enum_top = [
        {"rank": i + 1, "prob": float(p), "cls_idx": int(ci),
         "fingerprint": equiv_classes[ci]["fingerprint"],
         "canonical_program": equiv_classes[ci]["canonical_program"]}
        for i, (p, ci, _) in enumerate(enum_posteriors[:20])
    ]
    mcmc_top = sorted(
        mcmc_cls_post.items(), key=lambda kv: -kv[1],
    )[:20]
    mcmc_top_records = [
        {"rank": i + 1, "prob": float(w), "cls_idx": int(ci),
         "fingerprint": equiv_classes[ci]["fingerprint"],
         "canonical_program": equiv_classes[ci]["canonical_program"]}
        for i, (ci, w) in enumerate(mcmc_top)
    ]

    return {
        "rule_id": rule_id,
        "elapsed_s": round(time.time() - t0, 2),
        "parse_audit": parse_audit,
        "enum_retained_mass": enum_retained_mass,
        "divergence": divergence,
        "top_k": topk,
        "extensional": extensional,
        "n_enum_exc": n_enum_exc,
        "n_mcmc_exc": n_mcmc_exc,
        "validity_flags": flags,
        "comparison_valid": comparison_valid,
        "enum_top20": enum_top,
        "mcmc_top20": mcmc_top_records,
    }


# ---------------------------------------------------------------------------
# Convergence diagnostics (per-checkpoint)
# ---------------------------------------------------------------------------

def run_convergence_for_rule(
    rule_id: str,
    equiv_classes,
    fp_probes,
    fp_to_cls: Dict[str, int],
    grammar,
    prim_dict,
    enum_posteriors,
    enum_retained_mass: float,
    ckpt_dir: Path,
    comparison_probes,
    top_k_summary: int,
) -> Dict[str, Any]:
    """Replay the comparison at each MCMC checkpoint."""
    ckpt_files = sorted(
        ckpt_dir.glob("checkpoint_*.json"),
        key=lambda p: int(p.stem.split("_", 1)[1]),
    )
    records = []
    first_valid_step = None
    for cf in ckpt_files:
        cp_data = json.loads(cf.read_text())
        step = cp_data["step"]
        visits = cp_data["visit_counts"]
        result = compare_one_rule(
            rule_id, equiv_classes, fp_probes, fp_to_cls,
            grammar, prim_dict, enum_posteriors, enum_retained_mass,
            visits, comparison_probes, top_k_summary,
        )
        rec = {
            "step": step,
            "total_variation": result["divergence"]["total_variation"],
            "jensen_shannon": result["divergence"]["jensen_shannon"],
            "ext_mean_abs_diff": result["extensional"]["mean_abs_diff"],
            "topk_overlap": result["top_k"]["overlap_size"],
            "comparison_valid": result["comparison_valid"],
            "mass_mapped": result["parse_audit"]["mass_mapped_frac"],
            "mass_unmapped": result["parse_audit"]["mass_unmapped_frac"],
            "n_mcmc_classes": result["divergence"]["n_mcmc_classes"],
            "validity_flags": result["validity_flags"],
        }
        records.append(rec)
        if result["comparison_valid"] and first_valid_step is None:
            first_valid_step = step
    return {
        "rule_id": rule_id,
        "n_checkpoints": len(records),
        "first_valid_step": first_valid_step,
        "trajectory": records,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-convergence", action="store_true",
                    help="Only run final-state comparison (no per-checkpoint).")
    ap.add_argument("--only-rules", type=str, default=None)
    args = ap.parse_args()

    print("[CMP] Loading enum pool...", flush=True)
    with open(ENUM_DIR / "pool.pkl", "rb") as f:
        pool_payload = cloudpickle.load(f)
    equiv = pool_payload["equiv"]
    enum_probes = pool_payload["stats"]["_probes"]
    print(f"[CMP] Pool: {len(equiv):,} classes, "
          f"{len(enum_probes)} fingerprint probes", flush=True)

    fp_to_cls = _build_fingerprint_index(equiv)
    grammar = build_gallery_grammar()
    prim_dict = _build_prim_dict(grammar)

    comp_probes = _build_comparison_probes(
        n_probes=CONFIG["comparison"]["n_probes"],
        seed=CONFIG["comparison"]["probe_seed"],
    )
    print(f"[CMP] Comparison probes: {len(comp_probes)} "
          f"(seed={CONFIG['comparison']['probe_seed']})", flush=True)

    top_k_summary = CONFIG["comparison"]["top_k_summary"]

    # Which rules have both enum posteriors and mcmc raw_visits?
    enum_avail = {p.stem for p in (ENUM_DIR / "posteriors").glob("*.json")}
    mcmc_avail = {p.stem for p in (MCMC_DIR / "raw_visits").glob("*.json")}
    rules = sorted(enum_avail & mcmc_avail)
    if args.only_rules:
        rules = [r for r in rules if r in args.only_rules.split(",")]
    print(f"[CMP] Rules with both enum + mcmc outputs: {len(rules)}",
          flush=True)
    print(f"  enum-only: {sorted(enum_avail - mcmc_avail)}")
    print(f"  mcmc-only: {sorted(mcmc_avail - enum_avail)}")

    summary: Dict[str, Any] = {
        "rules": {},
        "n_rules_total": len(rules),
        "n_comparison_valid": 0,
        "headline_metrics": {},
        "validity_thresholds": VALIDITY_THRESHOLDS,
    }

    for i, rid in enumerate(rules):
        print(f"\n[CMP] {i+1}/{len(rules)} {rid}...", flush=True)
        enum_post_blob = json.loads(
            (ENUM_DIR / "posteriors" / f"{rid}.json").read_text()
        )
        # Rebuild enum_posteriors as List[(prob, cls_idx, None)]
        enum_posteriors = [
            (e["prob"], e["cls_idx"], None)
            for e in enum_post_blob["full_posterior"]
        ]
        enum_retained = enum_post_blob["enum_retained_mass"]

        mcmc_blob = json.loads(
            (MCMC_DIR / "raw_visits" / f"{rid}.json").read_text()
        )
        visit_counts = mcmc_blob["visit_counts"]

        result = compare_one_rule(
            rid, equiv, enum_probes, fp_to_cls, grammar, prim_dict,
            enum_posteriors, enum_retained, visit_counts, comp_probes,
            top_k_summary,
        )
        with open(OUT_DIR / "question_a" / f"{rid}.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        if not args.skip_convergence:
            cdir = MCMC_DIR / "checkpoints" / rid
            if cdir.exists():
                conv = run_convergence_for_rule(
                    rid, equiv, enum_probes, fp_to_cls, grammar, prim_dict,
                    enum_posteriors, enum_retained, cdir, comp_probes,
                    top_k_summary,
                )
                with open(OUT_DIR / "convergence_diagnostics" / f"{rid}.json",
                          "w") as f:
                    json.dump(conv, f, indent=2, default=str)

        summary["rules"][rid] = {
            "comparison_valid": result["comparison_valid"],
            "total_variation": result["divergence"]["total_variation"],
            "jensen_shannon": result["divergence"]["jensen_shannon"],
            "ext_mean_abs_diff": result["extensional"]["mean_abs_diff"],
            "topk_overlap": result["top_k"]["overlap_size"],
            "mass_mapped": result["parse_audit"]["mass_mapped_frac"],
            "mass_unmapped": result["parse_audit"]["mass_unmapped_frac"],
            "mass_parse_fail": result["parse_audit"]["mass_parse_fail_frac"],
            "enum_retained_mass": result["enum_retained_mass"],
            "validity_flags": result["validity_flags"],
        }
        if result["comparison_valid"]:
            summary["n_comparison_valid"] += 1

        # Live progress print
        d = result["divergence"]
        e = result["extensional"]
        pa = result["parse_audit"]
        print(
            f"  valid={result['comparison_valid']}  "
            f"TV={d['total_variation']:.3f}  "
            f"JS={d['jensen_shannon']:.3f}  "
            f"ext.|Δ|={e['mean_abs_diff']:.3f}  "
            f"top-{top_k_summary} overlap={result['top_k']['overlap_size']}  "
            f"mapped={pa['mass_mapped_frac']:.2f} "
            f"unmapped={pa['mass_unmapped_frac']:.2f}",
            flush=True,
        )

    # Headline summary
    if rules:
        tvs = [summary["rules"][r]["total_variation"] for r in rules]
        jss = [summary["rules"][r]["jensen_shannon"] for r in rules
               if summary["rules"][r]["jensen_shannon"] == summary["rules"][r]["jensen_shannon"]]  # not nan
        summary["headline_metrics"] = {
            "median_total_variation": sorted(tvs)[len(tvs) // 2] if tvs else None,
            "mean_total_variation": sum(tvs) / len(tvs) if tvs else None,
            "median_jensen_shannon": sorted(jss)[len(jss) // 2] if jss else None,
            "n_comparison_valid": summary["n_comparison_valid"],
            "n_rules": len(rules),
        }

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(
        f"\n[CMP] Done. "
        f"{summary['n_comparison_valid']}/{len(rules)} rules passed "
        f"VALIDITY gate.",
        flush=True,
    )


if __name__ == "__main__":
    main()
