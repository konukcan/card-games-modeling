"""Enumeration-vs-MCMC posterior comparison.

For each rule we already have:
  - Enumeration posterior over EQUIVALENCE CLASSES, computed by
    ``compute_posteriors_for_rule`` (output of ``analyze.py`` plus the
    Night 1 fixes). Each class has a canonical_program string and an
    associated predicate.
  - MCMC visit-frequency posterior over PROGRAM STRINGS, produced by
    ``analyze_mcmc.py``.

These are NOT directly comparable in the program-string space:
  * Enumeration collapses to canonical-program-per-class; MCMC samples raw
    programs, so the same equivalence class can be represented by many MCMC
    program strings.
  * Enumeration's "summed prior" for a class lumps together ALL syntactic
    variants; MCMC just visits whichever variant the chain happened to land
    in.

We compare in two intrinsically-defined spaces that don't depend on string
canonicalization:

A. **Extensional agreement on a held-out probe set.** For ``n_probes`` random
   6-card hands, compute ``p_accept`` from each method and report
   - mean / max absolute disagreement,
   - rank correlation of per-hand ``p_accept`` between methods,
   - top-1 hand agreement (does the same hand maximize entropy?).

B. **Top-K coverage.** Enumeration top-K classes vs MCMC top-K programs.
   Map each MCMC program to its equivalence class by re-fingerprinting
   against the enumeration probe set (the predicate's hit-vector on a
   shared probe set IS the fingerprint). Then ask: how many of MCMC's top-K
   classes appear in enumeration's top-K?

This script is INTENTIONALLY decoupled from the live audit run; it loads
saved enumeration outputs from the standard analyze.py results dir and
saved MCMC outputs from analyze_mcmc.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from gallery_analysis.enumerator import build_gallery_grammar, _make_evaluator
from gallery_analysis.gallery_rules import GALLERY_RULES
from gallery_analysis.exemplars import load_exemplars
from gallery_analysis.analyze import (
    build_hypothesis_pool, _strict_split_classes, estimate_extensions,
)
from gallery_analysis.injection import (
    load_and_validate_injections, merge_injected,
)
from gallery_analysis.hand_diagnosticity import compute_posteriors_for_rule
from dreamcoder_core.program import parse_program, Primitive
from rules.cards import Card, Suit, Rank


# ---------------------------------------------------------------------------
# Probe-set helpers
# ---------------------------------------------------------------------------

def _build_probe_hands(n_probes: int, seed: int) -> List[List[Card]]:
    """Random 6-card probe hands, deterministic per seed."""
    rng = random.Random(seed)
    deck = [Card(s, r) for s in Suit for r in Rank]
    return [rng.sample(deck, 6) for _ in range(n_probes)]


def _enum_p_accept_per_hand(
    posteriors, equiv_classes, probe_hands,
) -> List[float]:
    """Predictive p_accept of each probe hand under the enumeration posterior."""
    out: List[float] = []
    for h in probe_hands:
        p = 0.0
        for prob, cls_idx, _ in posteriors:
            try:
                if equiv_classes[cls_idx]["predicate"](h):
                    p += prob
            except Exception:
                pass
        out.append(p)
    return out


def _mcmc_p_accept_per_hand(
    mcmc_posterior: List[Tuple[str, float, Callable]],
    probe_hands,
) -> List[float]:
    """Same, under an MCMC empirical posterior weighted by visit fraction.

    ``mcmc_posterior`` is ``[(program_str, weight, predicate), ...]`` —
    weights renormalized to 1 over RECONSTRUCTED programs (see _reconstruct
    below).
    """
    out: List[float] = []
    for h in probe_hands:
        p = 0.0
        for _prog, w, pred in mcmc_posterior:
            try:
                if pred(h):
                    p += w
            except Exception:
                pass
        out.append(p)
    return out


def _build_prim_dict(grammar) -> Dict[str, Primitive]:
    """name -> Primitive lookup needed by ``parse_program``."""
    out: Dict[str, Primitive] = {}
    for prod in grammar.productions:
        if isinstance(prod.program, Primitive):
            out[prod.program.name] = prod.program
    return out


def _reconstruct_mcmc_predicates(
    mcmc_top: Sequence[Dict[str, Any]],
    grammar,
) -> List[Tuple[str, float, Callable]]:
    """Given MCMC top-k visit-frequency entries, reconstruct callable
    predicates by parsing each program string against the gallery grammar
    (``parse_program``) and wrapping with ``_make_evaluator``.

    Returns ``[(program_str, normalized_weight, predicate), ...]``,
    skipping entries we cannot reconstruct. Weights renormalize over the
    successfully reconstructed subset.
    """
    prim_dict = _build_prim_dict(grammar)
    out: List[Tuple[str, float, Callable]] = []
    raw_weights: List[float] = []
    for entry in mcmc_top:
        prog_str = entry["program"]
        w = entry.get("empirical_posterior", 0.0)
        try:
            program = parse_program(prog_str, prim_dict)
            pred = _make_evaluator(program)
        except Exception:
            continue
        raw_weights.append(w)
        out.append((prog_str, w, pred))
    total = sum(raw_weights)
    if total <= 0:
        return []
    return [(p, w / total, pred) for (p, w, pred) in out]


# ---------------------------------------------------------------------------
# Pairwise summary
# ---------------------------------------------------------------------------

def _spearman_rank_corr(xs: List[float], ys: List[float]) -> float:
    """Spearman rank correlation. Pure-Python, O(n log n)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")

    def _rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0  # ranks are 1-indexed
            for k in range(i, j + 1):
                ranks[order[k]] = avg
            i = j + 1
        return ranks

    rx = _rank(xs)
    ry = _rank(ys)
    n = len(xs)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    num = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rx[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((ry[i] - mean_ry) ** 2 for i in range(n)))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def _summarize_per_hand(
    enum_paccept: List[float], mcmc_paccept: List[float],
) -> Dict[str, float]:
    n = len(enum_paccept)
    diffs = [abs(a - b) for a, b in zip(enum_paccept, mcmc_paccept)]
    return {
        "n_probes": n,
        "mean_abs_diff": sum(diffs) / max(1, n),
        "max_abs_diff": max(diffs) if diffs else 0.0,
        "spearman_rank_corr": _spearman_rank_corr(enum_paccept, mcmc_paccept),
        "fraction_disagreement_above_0_5": (
            sum(1 for d in diffs if d > 0.5) / max(1, n)
        ),
    }


def compare_one_rule(
    rule_id: str,
    posteriors,
    equiv_classes,
    mcmc_summary: Dict[str, Any],
    grammar,
    n_probes: int = 5_000,
    probe_seed: int = 99,
    top_k: int = 50,
) -> Dict[str, Any]:
    """Run extensional comparison for one rule. Returns a JSON-friendly dict."""
    probe_hands = _build_probe_hands(n_probes, probe_seed)

    enum_paccept = _enum_p_accept_per_hand(posteriors, equiv_classes, probe_hands)

    mcmc_top = (
        mcmc_summary.get("top_hypotheses")
        or mcmc_summary.get("frequency_ranking")
        or []
    )[:top_k]
    mcmc_posterior = _reconstruct_mcmc_predicates(mcmc_top, grammar)
    if not mcmc_posterior:
        return {
            "rule_id": rule_id,
            "error": "MCMC top-k could not be parsed; comparison aborted.",
            "mcmc_top_size": len(mcmc_top),
        }
    mcmc_paccept = _mcmc_p_accept_per_hand(mcmc_posterior, probe_hands)

    return {
        "rule_id": rule_id,
        "extensional": _summarize_per_hand(enum_paccept, mcmc_paccept),
        "n_enum_classes_in_posterior": len(posteriors),
        "n_mcmc_programs_used": len(mcmc_posterior),
        "mcmc_top_k_requested": top_k,
        "mcmc_top_k_parsed": len(mcmc_posterior),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--depth", type=int, default=7)
    p.add_argument("--max-programs", type=int, default=300_000)
    p.add_argument("--mcmc-results", type=str, required=True,
                   help="Path to MCMC results JSON from analyze_mcmc.py.")
    p.add_argument("--rules", type=str, default="",
                   help="Comma-separated rule IDs (default: ALL rules in MCMC file).")
    p.add_argument("--inject", type=str,
                   default="src/gallery_analysis/data/injected_hypotheses.json")
    p.add_argument("--n-probes", type=int, default=5_000)
    p.add_argument("--probe-seed", type=int, default=99)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--output", type=str,
                   default="review-stage/experiments/night2/enum_vs_mcmc.json")
    args = p.parse_args()

    print(f"[CMP] Loading MCMC results: {args.mcmc_results}", flush=True)
    with open(args.mcmc_results) as f:
        mcmc_payload = json.load(f)
    # analyze_mcmc.py writes {'config':..., 'rules': {rid: {...}}}; tolerate
    # either schema for back-compat with hand-rolled files.
    if isinstance(mcmc_payload, dict) and "rules" in mcmc_payload \
            and isinstance(mcmc_payload["rules"], dict):
        mcmc_results = mcmc_payload["rules"]
    else:
        mcmc_results = mcmc_payload

    print(f"[CMP] Building enumeration pool depth={args.depth} "
          f"max_programs={args.max_programs}...", flush=True)
    t0 = time.time()
    equiv, stats = build_hypothesis_pool(
        max_depth=args.depth, max_programs=args.max_programs, verbose=1,
    )
    print(f"[CMP] Base pool: {len(equiv):,} classes in {time.time()-t0:.1f}s",
          flush=True)

    probes = stats["_probes"]
    exemplars_disamb = stats.get("_exemplar_hands") or []
    grammar = build_gallery_grammar()
    enum_priors = [c["canonical_prior"] for c in equiv]
    rng_prior = (min(enum_priors), max(enum_priors)) if enum_priors else None
    inject_path = args.inject
    if not os.path.isabs(inject_path):
        inject_path = str(ROOT / inject_path)
    injected = load_and_validate_injections(
        inject_path, grammar=grammar, enumerated_prior_range=rng_prior,
    )
    equiv = merge_injected(list(equiv), injected, probes)
    if exemplars_disamb:
        equiv, _ = _strict_split_classes(
            equiv, exemplar_hands=exemplars_disamb, main_probes=probes, verbose=0,
        )
    print(f"[CMP] Post-inject pool: {len(equiv):,} classes", flush=True)

    print(f"[CMP] Estimating extensions ({10_000:,} samples each)...",
          flush=True)
    t1 = time.time()
    extensions = estimate_extensions(equiv, n_samples=10_000)
    print(f"[CMP] Extensions in {time.time()-t1:.1f}s", flush=True)

    frozen = load_exemplars()

    if args.rules:
        rule_ids = [r.strip() for r in args.rules.split(",") if r.strip()]
    else:
        rule_ids = sorted(mcmc_results.keys())

    out = {
        "args": vars(args),
        "n_rules": 0,
        "rules": {},
    }
    for rid in rule_ids:
        if rid not in mcmc_results:
            print(f"[CMP] {rid}: not in MCMC results, skip", flush=True)
            continue
        if rid not in GALLERY_RULES or rid not in frozen:
            print(f"[CMP] {rid}: missing rule or exemplars, skip", flush=True)
            continue
        exemplars = frozen[rid]["hands_primary"]

        print(f"\n[CMP] === {rid} ===", flush=True)
        t = time.time()
        posteriors, retained = compute_posteriors_for_rule(
            equiv_classes=equiv,
            extensions=extensions,
            exemplar_hands=exemplars,
            mass_threshold=0.001,
            return_retained_mass=True,
        )
        print(f"  enum posterior: {len(posteriors)} hyps, "
              f"retained={retained:.4f} in {time.time()-t:.2f}s", flush=True)

        rule_result = compare_one_rule(
            rid, posteriors, equiv,
            mcmc_summary=mcmc_results[rid],
            grammar=grammar,
            n_probes=args.n_probes,
            probe_seed=args.probe_seed,
            top_k=args.top_k,
        )
        rule_result["enum_retained_mass"] = retained
        out["rules"][rid] = rule_result
        print(f"  ext: mean|Δ|={rule_result['extensional']['mean_abs_diff']:.4f} "
              f"max|Δ|={rule_result['extensional']['max_abs_diff']:.4f} "
              f"ρ={rule_result['extensional']['spearman_rank_corr']:.4f}",
              flush=True)
        out["n_rules"] += 1

    output = ROOT / args.output if not os.path.isabs(args.output) else Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[CMP] Wrote {output}", flush=True)


if __name__ == "__main__":
    main()
