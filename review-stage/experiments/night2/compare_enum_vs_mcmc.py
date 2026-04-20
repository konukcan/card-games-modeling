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

We compare in EXTENSIONAL space, which is intrinsically defined and does
not depend on string canonicalization:

   For ``n_probes`` random 6-card hands, compute ``p_accept`` from each
   method and report
   - mean / max absolute disagreement,
   - rank correlation of per-hand ``p_accept`` between methods,
   - per-method probe hit-rate (mean p_accept) — diagnoses the rare-rule
     blind spot where both methods are near-zero on uniform random hands.

KNOWN LIMITATIONS (per Codex Round 1, 2026-04-20):

* Uniform 6-card random probes are predictively valid but have a blind
  spot for rare-condition rules (e.g. "all 4s and queens"): both methods
  may agree at ~0 on essentially every probe, producing meaningless
  high agreement. We report ``probe_hit_rate_*`` per method so the
  reader can detect this regime; we do NOT (yet) targeted-sample
  positive probes.

* MCMC is compared on its top-K visit-frequency entries only, with
  weights renormalized over the parsed subset. We report:
  - ``mcmc_parse_audit.mass_in_full_list`` — Σ empirical_posterior over the
    FULL ``top_hypotheses`` payload (before our top-K truncation); used to
    bound how much MCMC mass sits OUTSIDE the comparison.
  - ``mcmc_parse_audit.mass_in_top_k`` — Σ empirical_posterior over the
    TOP-K subset (i.e. AFTER our truncation, BEFORE parse-drop).
  - ``mcmc_parse_audit.mass_dropped_parse`` — Σ weight of top-K entries
    that failed to parse and were silently dropped.
  - ``mcmc_parse_audit.mass_used`` — Σ weight of top-K entries that
    survived parsing (BEFORE renormalization to 1).
  Compatibility should only be claimed when ``mass_in_full_list`` and
  ``mass_in_top_k`` are both near 1 and ``mass_dropped_parse`` is near 0
  (Codex Round 2 claim-gating recommendation).

* Spearman ρ is reported only as a monotonicity indicator. The
  calibration claim must be read off mean/max absolute difference, not ρ.

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
) -> Tuple[List[float], int]:
    """Predictive p_accept of each probe hand under the enumeration posterior.

    Returns ``(p_accept_per_hand, n_predicate_exceptions)`` so the caller can
    surface silent-failure rate alongside the comparison statistics.
    """
    out: List[float] = []
    n_exc = 0
    for h in probe_hands:
        p = 0.0
        for prob, cls_idx, _ in posteriors:
            try:
                if equiv_classes[cls_idx]["predicate"](h):
                    p += prob
            except Exception:
                n_exc += 1
        out.append(p)
    return out, n_exc


def _mcmc_p_accept_per_hand(
    mcmc_posterior: List[Tuple[str, float, Callable]],
    probe_hands,
) -> Tuple[List[float], int]:
    """Same, under an MCMC empirical posterior weighted by visit fraction.

    ``mcmc_posterior`` is ``[(program_str, weight, predicate), ...]`` —
    weights renormalized to 1 over RECONSTRUCTED programs (see _reconstruct
    below).
    """
    out: List[float] = []
    n_exc = 0
    for h in probe_hands:
        p = 0.0
        for _prog, w, pred in mcmc_posterior:
            try:
                if pred(h):
                    p += w
            except Exception:
                n_exc += 1
        out.append(p)
    return out, n_exc


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
    mcmc_full: Sequence[Dict[str, Any]] | None = None,
) -> Tuple[List[Tuple[str, float, Callable]], Dict[str, Any]]:
    """Given MCMC top-k visit-frequency entries, reconstruct callable
    predicates by parsing each program string against the gallery grammar
    (``parse_program``) and wrapping with ``_make_evaluator``.

    ``mcmc_full`` is the FULL ``top_hypotheses`` payload (BEFORE the
    caller's top-K slice). It is used to compute ``mass_in_full_list`` so
    the audit reports how much MCMC mass sits OUTSIDE the top-K window.
    If omitted, falls back to ``mcmc_top`` itself, in which case
    ``mass_in_full_list == mass_in_top_k``.

    Returns ``(posterior, audit)`` where:
      - ``posterior`` is ``[(program_str, normalized_weight, predicate), ...]``
        with weights renormalized over the successfully reconstructed subset.
      - ``audit`` reports raw / truncated / parsed mass so silent drops
        and truncation effects are both visible:
            ``mass_in_full_list`` — Σ empirical_posterior over the full
                                    top_hypotheses payload (BEFORE top-K
                                    truncation)
            ``mass_in_top_k``     — Σ empirical_posterior over mcmc_top
                                    (AFTER top-K truncation, BEFORE parse)
            ``n_dropped_parse``   — # top-K entries that failed to parse
            ``mass_dropped_parse``— Σ weight of dropped top-K entries
            ``mass_used``         — Σ weight of parsed top-K entries
                                    (BEFORE renormalization)
    """
    prim_dict = _build_prim_dict(grammar)
    out: List[Tuple[str, float, Callable]] = []
    raw_weights: List[float] = []
    n_dropped = 0
    mass_dropped = 0.0
    mass_in_top_k = 0.0
    for entry in mcmc_top:
        prog_str = entry["program"]
        w = entry.get("empirical_posterior", 0.0)
        mass_in_top_k += w
        try:
            program = parse_program(prog_str, prim_dict)
            pred = _make_evaluator(program)
        except Exception:
            n_dropped += 1
            mass_dropped += w
            continue
        raw_weights.append(w)
        out.append((prog_str, w, pred))
    if mcmc_full is None:
        mass_in_full_list = mass_in_top_k
    else:
        mass_in_full_list = sum(
            e.get("empirical_posterior", 0.0) for e in mcmc_full
        )
    audit = {
        "mass_in_full_list": mass_in_full_list,
        "mass_in_top_k": mass_in_top_k,
        "n_dropped_parse": n_dropped,
        "mass_dropped_parse": mass_dropped,
        "mass_used": sum(raw_weights),
    }
    total = sum(raw_weights)
    if total <= 0:
        return [], audit
    return [(p, w / total, pred) for (p, w, pred) in out], audit


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
        # Probe-set hit-rates flag the rare-rule blind spot: if BOTH are
        # ~0 across all probes, "agreement" is uninformative.
        "probe_hit_rate_enum": sum(enum_paccept) / max(1, n),
        "probe_hit_rate_mcmc": sum(mcmc_paccept) / max(1, n),
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
    enum_retained_mass: float | None = None,
) -> Dict[str, Any]:
    """Run extensional comparison for one rule. Returns a JSON-friendly dict.

    ``enum_retained_mass`` is the fraction of enumeration posterior mass
    that survived diagnosticity pruning. When < ``min_enum_retained_mass``
    (see ``VALIDITY_THRESHOLDS``), the ``enum_truncation_excessive`` flag
    trips and ``comparison_valid`` becomes False. This is the enum-side
    analogue of MCMC ``mass_in_top_k``.
    """
    probe_hands = _build_probe_hands(n_probes, probe_seed)

    enum_paccept, enum_n_exc = _enum_p_accept_per_hand(
        posteriors, equiv_classes, probe_hands,
    )

    # Loader has already validated that every entry has 'top_hypotheses';
    # we no longer fall back to 'frequency_ranking' here, so a schema drift
    # at this point would be an upstream bug, not silently absorbed.
    mcmc_full = mcmc_summary["top_hypotheses"]
    mcmc_top = mcmc_full[:top_k]
    mcmc_posterior, parse_audit = _reconstruct_mcmc_predicates(
        mcmc_top, grammar, mcmc_full=mcmc_full,
    )
    if not mcmc_posterior:
        return {
            "rule_id": rule_id,
            "error": "MCMC top-k could not be parsed; comparison aborted.",
            "mcmc_top_size": len(mcmc_top),
            "mcmc_parse_audit": parse_audit,
        }
    mcmc_paccept, mcmc_n_exc = _mcmc_p_accept_per_hand(
        mcmc_posterior, probe_hands,
    )

    extensional = _summarize_per_hand(enum_paccept, mcmc_paccept)

    # Hard claim-gating (Codex Rounds 3–4 accept-condition). Any flag set
    # ⇒ comparison_valid=False ⇒ caller MUST NOT report extensional
    # agreement as evidence of compatibility for this rule.
    validity_flags = _build_validity_flags(
        parse_audit=parse_audit,
        extensional=extensional,
        n_enum_exc=enum_n_exc,
        n_mcmc_exc=mcmc_n_exc,
        enum_retained_mass=enum_retained_mass,
    )
    comparison_valid = not any(validity_flags.values())

    return {
        "rule_id": rule_id,
        "extensional": extensional,
        "n_enum_classes_in_posterior": len(posteriors),
        "n_mcmc_programs_used": len(mcmc_posterior),
        "mcmc_top_k_requested": top_k,
        "mcmc_top_k_parsed": len(mcmc_posterior),
        # Honest accounting of silent-failure surfaces (Codex Rounds 1–3
        # comparison-framework review):
        #   * mcmc_parse_audit.mass_dropped_parse > 0 ⇒ truncation may bias
        #     the comparison; renormalization makes it look better.
        #   * mass_in_top_k < ~1.0 ⇒ top-K truncation excludes long tail.
        #   * mass_in_full_list < ~1.0 ⇒ MCMC payload itself was incomplete
        #     (analyze_mcmc.py only stored a partial top-N).
        #   * predicate_exception_count_* > 0 ⇒ silent False-on-throw
        #     contaminated p_accept.
        #   * comparison_valid==False ⇒ at least one threshold tripped;
        #     do NOT report extensional agreement as compatibility.
        "mcmc_parse_audit": parse_audit,
        "predicate_exception_count_enum": enum_n_exc,
        "predicate_exception_count_mcmc": mcmc_n_exc,
        "validity_flags": validity_flags,
        "comparison_valid": comparison_valid,
    }


# ---------------------------------------------------------------------------
# Validity gating
# ---------------------------------------------------------------------------

# Per-rule flag thresholds. A rule's comparison is reported as "valid"
# only if ALL of these are satisfied. Tuned conservatively so that any
# rule the audit considers "compatible" has high coverage on both sides.
# Pre-registered for the Night 2 paper appendix (Codex Round 4 accept
# condition) — DO NOT tune post-hoc on observed results.
VALIDITY_THRESHOLDS: Dict[str, float] = {
    # MCMC-side coverage (parse + truncation + payload completeness):
    "min_mass_in_full_list": 0.90,
    "min_mass_in_top_k": 0.80,
    "max_mass_dropped_parse": 0.05,
    # Enum-side coverage: diagnosticity pruning may drop posterior mass.
    # Symmetric to MCMC mass_in_top_k.
    "min_enum_retained_mass": 0.95,
    # Probe-set hit rate: if BOTH methods are nearly always 0, the
    # comparison is vacuous regardless of mean|Δ|.
    "min_probe_hit_rate_either": 0.001,
    # Silent predicate failures should never happen on a well-formed
    # pool; nonzero is a bug the caller should investigate.
    "max_predicate_exceptions": 0,
}


def _build_validity_flags(
    parse_audit: Dict[str, Any],
    extensional: Dict[str, float],
    n_enum_exc: int,
    n_mcmc_exc: int,
    enum_retained_mass: float | None = None,
) -> Dict[str, bool]:
    """Compute per-rule comparison-validity flags. ``True`` ⇒ tripped."""
    t = VALIDITY_THRESHOLDS
    enum_trunc = (
        enum_retained_mass is not None
        and enum_retained_mass < t["min_enum_retained_mass"]
    )
    return {
        "mcmc_payload_truncated": (
            parse_audit["mass_in_full_list"] < t["min_mass_in_full_list"]
        ),
        "topk_truncation_excessive": (
            parse_audit["mass_in_top_k"] < t["min_mass_in_top_k"]
        ),
        "parse_drop_excessive": (
            parse_audit["mass_dropped_parse"] > t["max_mass_dropped_parse"]
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
    p.add_argument("--allow-legacy-schema", action="store_true",
                   help="Accept flat {rule_id: {...}} MCMC payload instead "
                        "of analyze_mcmc.py's nested {rules: {...}} schema.")
    args = p.parse_args()

    print(f"[CMP] Loading MCMC results: {args.mcmc_results}", flush=True)
    with open(args.mcmc_results) as f:
        mcmc_payload = json.load(f)

    # analyze_mcmc.py writes {'config':..., 'rules': {rid: {...}}}; we also
    # tolerate a flat {rid: {...}} payload for legacy / hand-rolled files.
    # We log which schema branch we took, validate per-rule keys, and
    # require --allow-legacy-schema to accept the flat form so it cannot
    # mask an upstream bug.
    if isinstance(mcmc_payload, dict) and "rules" in mcmc_payload \
            and isinstance(mcmc_payload["rules"], dict):
        mcmc_results = mcmc_payload["rules"]
        print("[CMP] Detected nested schema (analyze_mcmc.py).", flush=True)
    else:
        if not args.allow_legacy_schema:
            raise ValueError(
                "MCMC payload missing top-level 'rules' dict. "
                "Pass --allow-legacy-schema to accept a flat "
                "{rule_id: {...}} payload."
            )
        mcmc_results = mcmc_payload
        print("[CMP] Detected legacy flat schema (rule_id keys at top level).",
              flush=True)

    _required_per_rule = {"top_hypotheses"}
    for rid, entry in mcmc_results.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"MCMC entry for rule {rid!r} is not a dict; got {type(entry)}"
            )
        if not (entry.keys() & _required_per_rule):
            raise ValueError(
                f"MCMC entry for rule {rid!r} missing 'top_hypotheses'; "
                f"keys present: {sorted(entry.keys())}"
            )

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
            enum_retained_mass=retained,
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
