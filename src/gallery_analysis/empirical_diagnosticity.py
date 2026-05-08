"""
Empirical diagnosticity — find diagnostic hands from human-derived posteriors.

The pipeline
------------
For each rule, we have N participant lambdas (DSL programs translated from
free-text responses). Multiple participants may write semantically equivalent
lambdas; we collapse those into equivalence classes using fingerprinting on
a probe hand set. The class-size distribution gives a count-based "empirical
posterior" over hypotheses, which we then feed to the same
``find_most_diagnostic_hands`` machinery used for the model posterior.

The output is the top-k hands the *aggregate human population* is most
collectively split on for each rule. These are candidates for the next round
of stimuli, or for cross-checking against hands the model flags as diagnostic.

Input format
------------
A JSON file mapping rule_id → list of DSL program strings. Two accepted
shapes:

    {"all_red": ["(λ all_color $0 RED)", "(λ ...)", ...], ...}

or with per-participant metadata (the script ignores the participant_id but
preserves it through to the per-class manifest):

    {"all_red": [
        {"participant_id": "p001", "dsl": "(λ all_color $0 RED)"},
        ...
    ], ...}

Usage
-----
    cd src
    python -m gallery_analysis.empirical_diagnosticity \\
        --participant-data /path/to/participant_lambdas.json \\
        --output /path/to/empirical_diagnosticity.json \\
        [--n-candidates 50000] [--top-k 100] [--n-probes 1580]

Notes
-----
* The probe set used for fingerprinting empirical lambdas does NOT have to
  match the model's probe set. Any sufficiently large probe set (~500+) will
  give reliable equivalence classes. We default to Config I (1,580 hands)
  because it's the model's default and is well-tested.
* This script is standalone — it does not need the model's pool.pkl. Each
  participant lambda is its own predicate; we don't need to look anything
  up in the model's enumeration.
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.program import Primitive, parse_program
from gallery_analysis.adversarial_hands import (
    AdversarialHand,
    find_most_diagnostic_hands,
    adversarial_hand_to_dict,
)
from gallery_analysis.enumerator import build_gallery_grammar, _make_evaluator
from gallery_analysis.exemplars import generate_probe_set
from gallery_analysis.gallery_rules import GALLERY_RULES
from gallery_analysis.hypothesis_table import compute_fingerprint


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _normalize_participant_entry(entry: Any) -> Tuple[str, str]:
    """Accept either a bare DSL string or a {participant_id, dsl, ...} dict.
    Returns (participant_id, dsl_program).
    """
    if isinstance(entry, str):
        return ("anonymous", entry)
    if isinstance(entry, dict):
        return (str(entry.get("participant_id", "anonymous")), entry["dsl"])
    raise ValueError(f"Cannot read participant entry: {entry!r}")


def load_participant_data(path: Path) -> Dict[str, List[Tuple[str, str]]]:
    """Load and normalize {rule_id: [(participant_id, dsl), ...]}."""
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(
            f"Expected top-level dict {{rule_id: [...]}}, got {type(raw).__name__}"
        )
    out: Dict[str, List[Tuple[str, str]]] = {}
    for rule_id, entries in raw.items():
        out[rule_id] = [_normalize_participant_entry(e) for e in entries]
    return out


# ---------------------------------------------------------------------------
# Per-rule pipeline
# ---------------------------------------------------------------------------

def build_empirical_pool(
    rule_id: str,
    participant_lambdas: List[Tuple[str, str]],
    probes: List,
    prim_dict: Dict[str, Primitive],
    verbose: int = 1,
) -> Tuple[List[Dict[str, Any]], List[Tuple[float, int, List[bool]]], Dict[str, Any]]:
    """Group participant lambdas into equivalence classes by fingerprint.

    Returns
    -------
    equiv_classes : list of class dicts (canonical_program, predicate, etc.)
    posteriors    : list of (prob, cls_idx, hit_vector) — count-based posterior
                    in the format ``find_most_diagnostic_hands`` expects.
    stats         : metadata dict with parse failures, eval failures, n classes
    """
    parse_fail: List[Dict[str, str]] = []
    fingerprint_fail: List[Dict[str, str]] = []

    # Step 1: parse every DSL string into a predicate.
    parsed: List[Tuple[str, str, Any]] = []  # (participant_id, dsl, predicate)
    for participant_id, dsl in participant_lambdas:
        try:
            program = parse_program(dsl, prim_dict)
            predicate = _make_evaluator(program)
            parsed.append((participant_id, dsl, predicate))
        except (ValueError, KeyError) as e:
            parse_fail.append({"participant_id": participant_id, "dsl": dsl,
                               "error": f"{type(e).__name__}: {e}"})

    if verbose >= 1 and parse_fail:
        print(f"  [{rule_id}] {len(parse_fail)} participant lambdas failed to parse",
              flush=True)

    # Step 2: fingerprint each predicate on the probe set.
    # Multiple participants with semantically-equivalent lambdas land on the
    # same fingerprint and merge into one empirical class.
    fp_to_members: Dict[str, List[Tuple[str, str, Any]]] = {}
    for triple in parsed:
        participant_id, dsl, predicate = triple
        try:
            fp = compute_fingerprint(predicate, probes)
        except Exception as e:
            fingerprint_fail.append({"participant_id": participant_id, "dsl": dsl,
                                     "error": f"{type(e).__name__}: {e}"})
            continue
        fp_to_members.setdefault(fp, []).append(triple)

    # Step 3: build equiv_classes and posteriors. Class size → probability.
    n_total = sum(len(members) for members in fp_to_members.values())
    equiv_classes: List[Dict[str, Any]] = []
    posteriors: List[Tuple[float, int, List[bool]]] = []
    for fp, members in fp_to_members.items():
        # Use the first participant's lambda as the canonical representative.
        # All members are fingerprint-equivalent so any choice works.
        _, canonical_dsl, canonical_pred = members[0]
        cls_idx = len(equiv_classes)
        prob = len(members) / n_total
        # The hit_vector is the predicate's accept pattern on the probes,
        # which find_most_diagnostic_hands uses for splitter annotation.
        hit_vector = [bool(canonical_pred(h)) for h in probes]
        equiv_classes.append({
            "fingerprint": fp,
            "canonical_program": canonical_dsl,
            "all_programs": [m[1] for m in members],
            "n_expressions": len(members),
            "predicate": canonical_pred,
            "participant_ids": [m[0] for m in members],
            "canonical_prior": 0.0,   # irrelevant for the diagnosticity search
            "summed_prior": 0.0,
        })
        posteriors.append((prob, cls_idx, hit_vector))

    # Sort posterior descending by probability (just for readability of output).
    posteriors.sort(key=lambda t: -t[0])

    stats = {
        "n_participants_input": len(participant_lambdas),
        "n_parsed": len(parsed),
        "n_parse_failures": len(parse_fail),
        "n_fingerprint_failures": len(fingerprint_fail),
        "n_used_in_posterior": n_total,
        "n_empirical_classes": len(equiv_classes),
        "parse_failures": parse_fail,
        "fingerprint_failures": fingerprint_fail,
    }
    return equiv_classes, posteriors, stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    participant_data_path: Path,
    output_path: Path,
    n_candidates: int = 50_000,
    top_k: int = 100,
    n_probes: int = 1_580,
    seed_base: int = 12345,
    verbose: int = 1,
) -> None:
    print(f"Loading participant data from {participant_data_path}", flush=True)
    data = load_participant_data(participant_data_path)
    print(f"  {len(data)} rules, "
          f"{sum(len(v) for v in data.values())} total participant lambdas",
          flush=True)

    # Build probe set + grammar primitives once. Random probes are sufficient
    # for empirical fingerprinting — any 500+ hands gives reliable equivalence
    # classes between participant lambdas. We don't need the model's targeted-
    # probe set since we're not joining classes back to the model's pool.
    probes = generate_probe_set(n_probes=n_probes, seed=42)
    grammar = build_gallery_grammar()
    prim_dict: Dict[str, Primitive] = {
        prod.program.name: prod.program
        for prod in grammar.productions
        if isinstance(prod.program, Primitive)
    }

    output: Dict[str, Any] = {
        "config": {
            "participant_data_path": str(participant_data_path),
            "n_candidates": n_candidates,
            "top_k": top_k,
            "n_probes": len(probes),
            "seed_base": seed_base,
        },
        "rules": {},
    }

    n_rules_processed = 0
    for rule_id, participant_lambdas in data.items():
        if rule_id not in GALLERY_RULES:
            print(f"  WARN: rule {rule_id!r} not in GALLERY_RULES — skipping",
                  flush=True)
            continue

        if verbose >= 1:
            print(f"\n[{rule_id}] {len(participant_lambdas)} participant lambdas",
                  flush=True)

        equiv_classes, posteriors, stats = build_empirical_pool(
            rule_id, participant_lambdas, probes, prim_dict, verbose=verbose,
        )

        if not posteriors:
            print(f"  [{rule_id}] no usable posteriors — skipping search",
                  flush=True)
            output["rules"][rule_id] = {
                "stats": stats,
                "empirical_posterior": [],
                "diagnostic_hands": [],
                "ground_truth_acceptance_rate_in_top_k": None,
            }
            continue

        # Stable per-rule seed so results are reproducible per rule.
        rule_seed = (seed_base + zlib.crc32(rule_id.encode())) & 0xFFFFFFFF

        # Ground-truth predicate (the gallery rule).
        # If absent for some reason, pass None and skip the truth annotation.
        rule_pred = GALLERY_RULES[rule_id].get("predicate")

        diag_hands = find_most_diagnostic_hands(
            posteriors=posteriors,
            equiv_classes=equiv_classes,
            n_candidates=n_candidates,
            top_k=top_k,
            seed=rule_seed,
            diversity=True,
            ground_truth_pred=rule_pred,
            retained_mass=1.0,   # the empirical posterior is its own full distribution
        )

        if verbose >= 1:
            print(f"  {stats['n_empirical_classes']} empirical classes; "
                  f"top diagnostic p_accept = "
                  f"{diag_hands[0].p_accept:.3f} (entropy "
                  f"{diag_hands[0].entropy_bits:.3f} bits)",
                  flush=True)

        # Per-class summary in the output (canonical DSL + count + prob).
        empirical_posterior_summary = [
            {
                "fingerprint": equiv_classes[cls_idx]["fingerprint"],
                "canonical_program": equiv_classes[cls_idx]["canonical_program"],
                "n_participants": equiv_classes[cls_idx]["n_expressions"],
                "probability": prob,
                "participant_ids": equiv_classes[cls_idx]["participant_ids"],
            }
            for prob, cls_idx, _ in posteriors
        ]

        output["rules"][rule_id] = {
            "stats": stats,
            "rule_seed": rule_seed,
            "empirical_posterior": empirical_posterior_summary,
            "diagnostic_hands": [adversarial_hand_to_dict(h) for h in diag_hands],
        }
        n_rules_processed += 1

    output["summary"] = {
        "n_rules_processed": n_rules_processed,
        "n_rules_input": len(data),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nWrote {output_path}", flush=True)
    print(f"  {n_rules_processed} rules, {top_k} diagnostic hands per rule",
          flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--participant-data", required=True, type=Path,
                        help="JSON file mapping rule_id → list of participant lambdas")
    parser.add_argument("--output", required=True, type=Path,
                        help="Path to write the diagnosticity JSON")
    parser.add_argument("--n-candidates", type=int, default=50_000,
                        help="Random candidate hands to score per rule (default 50000)")
    parser.add_argument("--top-k", type=int, default=100,
                        help="Top diagnostic hands to keep per rule (default 100)")
    parser.add_argument("--n-probes", type=int, default=1_500,
                        help="Probe set size for fingerprinting (default 1500). "
                             "Any 500+ random hands gives reliable equivalence "
                             "classes between participant lambdas.")
    parser.add_argument("--seed-base", type=int, default=12345)
    parser.add_argument("--verbose", type=int, default=1)
    args = parser.parse_args()

    run(
        participant_data_path=args.participant_data,
        output_path=args.output,
        n_candidates=args.n_candidates,
        top_k=args.top_k,
        n_probes=args.n_probes,
        seed_base=args.seed_base,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
