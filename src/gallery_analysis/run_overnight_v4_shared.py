"""
Overnight pipeline v4 (shared-pool variant loop).

Builds the equivalence-class pool ONCE, then runs all 10 scoring variants
in the same Python process — sharing enumeration, trivial filter, and the
initial fingerprint/strict-split pass across variants.

Equivalent output to run_overnight_pipeline.py but ~2 hours faster:
  Per-variant work is now just:
    inject + post-inject strict split + extension top-up (cache hit) +
    prior precompute + score 60 rules + write JSON.

Used to add the extension_fingerprint and full_posterior fields to
score_rule's output (commit 203903f).

Usage:
    cd src
    nohup caffeinate -d -i -s python -u gallery_analysis/run_overnight_v4_shared.py \
        > /tmp/overnight_v4.log 2>&1 &
"""

import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from gallery_analysis.analyze import (
    build_hypothesis_pool,
    estimate_extensions,
    merge_injections_and_extend,
    _recompute_class_prior,
    score_rule,
    GALLERY_RULES,
)
from gallery_analysis.exemplars import load_exemplars
from gallery_analysis.provenance import compute_probe_hash, compute_provenance


RESULTS_DIR = Path("gallery_analysis/results")
INJECT_PATH = "gallery_analysis/data/injected_hypotheses.json"
EXT_CACHE = str(RESULTS_DIR / "extension_cache_depth6_v3.json")
PREFIX = "v3"

DEPTH = 6
MAX_PROGRAMS = 1_000_000
MC_SAMPLES = 100_000
N_PROBES = 500            # ignored when use_targeted_probes=True
USE_TARGETED_PROBES = True
EPSILON = 0.01
ENUM_GRAMMAR = "uniform"   # matches the v3_* outputs we already have on disk

# (name, prior_mode, scoring_grammar, inject_mode, likelihood_mode)
# inject_mode is "all" / "true_only" / None
VARIANTS: List[Tuple[str, str, str, str, str]] = [
    ("weighted_canonical_inject",    "canonical", "weighted", "all",       "noisy"),
    ("weighted_summed_inject",       "summed",    "weighted", "all",       "noisy"),
    ("weighted_canonical_trueonly",  "canonical", "weighted", "true_only", "noisy"),
    ("weighted_summed_trueonly",     "summed",    "weighted", "true_only", "noisy"),
    ("uniform_canonical_inject",     "canonical", "uniform",  "all",       "noisy"),
    ("uniform_summed_inject",        "summed",    "uniform",  "all",       "noisy"),
    ("uniform_canonical_trueonly",   "canonical", "uniform",  "true_only", "noisy"),
    ("uniform_summed_trueonly",      "summed",    "uniform",  "true_only", "noisy"),
    ("weighted_canonical_strict",    "canonical", "weighted", "all",       "strict"),
    ("weighted_summed_strict",       "summed",    "weighted", "all",       "strict"),
]


def score_one_variant(
    base_classes: List[Dict[str, Any]],
    base_pipeline_stats: Dict[str, Any],
    variant_name: str,
    prior_mode: str,
    scoring_grammar: str,
    inject_mode: str,
    likelihood_mode: str,
) -> Dict[str, Any]:
    """Run one variant on a fresh deep-copy of the base pool."""
    t_var = time.time()
    print(f"\n{'='*70}\nVARIANT: {variant_name}", flush=True)
    print(f"  prior={prior_mode}  scoring_grammar={scoring_grammar}  "
          f"inject={inject_mode}  lik={likelihood_mode}", flush=True)

    # Deep-copy so variants don't leak into each other.
    equiv_classes = copy.deepcopy(base_classes)
    pipeline_stats = copy.deepcopy(base_pipeline_stats)

    # Inject + post-inject strict split + extension top-up (mostly cache hit).
    inject_path = INJECT_PATH if inject_mode in ("all", "true_only") else None
    inject_true_only = (inject_mode == "true_only")
    equiv_classes, extensions, probes, probe_hash = merge_injections_and_extend(
        equiv_classes,
        pipeline_stats,
        inject_path=inject_path,
        extension_cache=EXT_CACHE,
        n_probes=N_PROBES,
        extension_samples=MC_SAMPLES,
        inject_true_only=inject_true_only,
        verbose=1,
    )

    # True-rule fingerprint lookup.
    true_rule_fps: Dict[str, str] = {}
    for cls in equiv_classes:
        true_for_list = cls.get("true_for_rules") or (
            [cls["true_for_rule"]] if cls.get("true_for_rule") else []
        )
        for rule_id in true_for_list:
            true_rule_fps[rule_id] = cls["fingerprint"]
    print(f"  True-rule fingerprints found: {len(true_rule_fps)} / {len(GALLERY_RULES)}",
          flush=True)

    # Build scoring grammar object.
    scoring_grammar_obj = None
    if scoring_grammar == "weighted":
        from gallery_analysis.enumerator import build_weighted_gallery_grammar
        scoring_grammar_obj = build_weighted_gallery_grammar()

    # Precompute priors under the scoring grammar.
    precomputed_priors = None
    if scoring_grammar_obj is not None:
        t0 = time.time()
        precomputed_priors = [
            _recompute_class_prior(cls, scoring_grammar_obj,
                                   prior_mode=prior_mode, strict=False)
            for cls in equiv_classes
        ]
        print(f"  Precomputed {len(precomputed_priors)} priors in "
              f"{time.time()-t0:.1f}s", flush=True)

    # Score each rule.
    print(f"  Scoring {len(GALLERY_RULES)} rules...", flush=True)
    exemplars = load_exemplars()
    rule_results: Dict[str, Any] = {}
    t_score = time.time()
    for rule_id, rule_info in GALLERY_RULES.items():
        if rule_id not in exemplars:
            continue
        result = score_rule(
            rule_id=rule_id,
            exemplar_hands=exemplars[rule_id]["hands_primary"],
            equivalence_classes=equiv_classes,
            extensions=extensions,
            epsilon=EPSILON,
            prior_mode=prior_mode,
            true_rule_fingerprint=true_rule_fps.get(rule_id),
            grammar=scoring_grammar_obj,
            likelihood_exponent=1.0,
            likelihood_mode=likelihood_mode,
            precomputed_priors=precomputed_priors,
            strict_priors=False,
        )
        result["group"] = rule_info["group"]
        result["answer"] = rule_info["answer"]
        rule_results[rule_id] = result
    print(f"  Scoring done in {time.time()-t_score:.1f}s", flush=True)

    # Difficulty ranking.
    ranking = sorted(
        [
            {
                "rule_id": r,
                "group": rr["group"],
                "answer": rr["answer"],
                "posterior_entropy": rr["difficulty"]["posterior_entropy"],
                "n_effective_hypotheses": rr["difficulty"]["n_effective_hypotheses"],
                "top1_probability": rr["difficulty"]["top1_probability"],
                "top5_probability": rr["difficulty"]["top5_probability"],
                "n_with_all_hits": rr["n_with_all_hits"],
                "true_rule_rank": rr.get("true_rule_rank"),
                "true_rule_posterior_mass": rr.get("true_rule_posterior_mass"),
            }
            for r, rr in rule_results.items()
        ],
        key=lambda d: -d["posterior_entropy"],
    )

    elapsed = time.time() - t_var
    pipeline_stats["scoring_time_seconds"] = round(time.time() - t_score, 1)
    pipeline_stats["grand_total_seconds"] = round(elapsed, 1)
    print(f"  Variant total: {elapsed:.1f}s", flush=True)

    provenance = compute_provenance(
        probe_seed=42,
        n_probes=N_PROBES,
        probes=probes,
        inject_path=inject_path,
        n_equiv_classes=len(equiv_classes),
    )

    # Strip non-serializable fields (predicates, hit vectors) and use the
    # `rule_details` key that the visualization layer expects (matches what
    # analyze.py's main() produces).
    rule_details: Dict[str, Any] = {}
    for rule_id, rr in rule_results.items():
        rule_details[rule_id] = {
            "rule_id": rr["rule_id"],
            "group": rr["group"],
            "answer": rr["answer"],
            "difficulty": rr["difficulty"],
            "top_hypotheses": rr["top_hypotheses"],
            "full_posterior": rr.get("full_posterior"),
            "n_hypotheses_scored": rr["n_hypotheses_scored"],
            "n_with_any_hit": rr["n_with_any_hit"],
            "n_with_all_hits": rr["n_with_all_hits"],
            "true_rule_rank": rr.get("true_rule_rank"),
            "true_rule_posterior_mass": rr.get("true_rule_posterior_mass"),
            "true_rule_program": rr.get("true_rule_program"),
            "true_rule_log_prior": rr.get("true_rule_log_prior"),
            "true_rule_extension_size": rr.get("true_rule_extension_size"),
            "true_rule_base_rate": rr.get("true_rule_base_rate"),
            "true_rule_log_likelihood": rr.get("true_rule_log_likelihood"),
            "true_rule_n_expressions": rr.get("true_rule_n_expressions"),
            "true_rule_approximate": rr.get("true_rule_approximate"),
            "exemplar_diagnosticity": rr.get("exemplar_diagnosticity"),
        }

    return {
        "pipeline_stats": pipeline_stats,
        "rule_details": rule_details,
        "difficulty_ranking": ranking,
        "config": {
            "max_depth": DEPTH,
            "max_programs": MAX_PROGRAMS,
            "n_probes": N_PROBES,
            "extension_samples": MC_SAMPLES,
            "epsilon": EPSILON,
            "prior_mode": prior_mode,
            "scoring_grammar": scoring_grammar,
            "likelihood_exponent": 1.0,
            "likelihood_mode": likelihood_mode,
            "strict_priors": False,
            "is_tempered_posterior": False,
            "posterior_kind": "standard_bayesian",
        },
        "provenance": provenance,
    }


def main() -> None:
    t_total = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("OVERNIGHT PIPELINE v4 — SHARED-POOL VARIANT LOOP")
    print("=" * 70)
    print(f"Variants: {len(VARIANTS)}")
    print(f"Enumeration grammar: {ENUM_GRAMMAR} (shared across all variants)")
    print(f"Depth: {DEPTH}, Max programs: {MAX_PROGRAMS:,}")
    print(f"Extension cache: {EXT_CACHE}", flush=True)

    # --------------------------------------------------------------- pool once
    print("\n" + "=" * 70 + "\nPHASE 0: Build base equivalence-class pool (one-time)\n"
          + "=" * 70, flush=True)
    base_classes, base_stats = build_hypothesis_pool(
        max_depth=DEPTH,
        max_programs=MAX_PROGRAMS,
        max_cost=35.0,
        timeout=600.0,
        n_probes=N_PROBES,
        max_list_chain=2,
        verbose=1,
        use_targeted_probes=USE_TARGETED_PROBES,
        enumeration_grammar=ENUM_GRAMMAR,
    )
    print(f"Base pool: {len(base_classes)} classes", flush=True)

    # --------------------------------------------------------- variants in proc
    print("\n" + "=" * 70 + "\nPHASE 1: Score all variants on the shared pool\n"
          + "=" * 70, flush=True)
    succeeded = 0
    for name, prior, sg, inject, lik in VARIANTS:
        try:
            results = score_one_variant(
                base_classes=base_classes,
                base_pipeline_stats=base_stats,
                variant_name=name,
                prior_mode=prior,
                scoring_grammar=sg,
                inject_mode=inject,
                likelihood_mode=lik,
            )
            output_path = RESULTS_DIR / f"{PREFIX}_{name}.json"
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"  ✓ Wrote {output_path} ({output_path.stat().st_size/1024:.0f} KB)",
                  flush=True)
            succeeded += 1
        except Exception as e:
            print(f"  ✗ FAILED {name}: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    elapsed = time.time() - t_total
    print(f"\n{'='*70}\nTOTAL TIME: {elapsed/3600:.2f} hours "
          f"({succeeded}/{len(VARIANTS)} variants succeeded)\n{'='*70}", flush=True)


if __name__ == "__main__":
    main()
