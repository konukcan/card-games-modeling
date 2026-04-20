"""Driver: run adversarial hand generation on 5 representative rules.

Uses the BALD entropy-proxy method (find_most_diagnostic_hands) plus the
confident-but-wrong probe (find_most_adversarial_hands).

For each rule:
  1. Build pool + estimate extensions + compute posterior (default settings).
  2. Find top-100 most diagnostic hands by posterior predictive entropy.
  3. Find top-50 false positives + top-50 false negatives vs ground truth.
  4. Dump everything to JSON.

Output: review-stage/experiments/night2/adversarial_hands_results.json
"""
import sys
import os
import time
import json
import zlib
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "..", "src"))

from gallery_analysis.analyze import (
    build_hypothesis_pool,
    _strict_split_classes,
    estimate_extensions,
)
from gallery_analysis.injection import (
    load_and_validate_injections,
    merge_injected,
)
from gallery_analysis.enumerator import build_gallery_grammar
from gallery_analysis.gallery_rules import GALLERY_RULES
from gallery_analysis.exemplars import load_exemplars
from gallery_analysis.hand_diagnosticity import compute_posteriors_for_rule
from gallery_analysis.adversarial_hands import (
    find_most_diagnostic_hands,
    find_most_adversarial_hands,
    adversarial_hand_to_dict,
)


REPRESENTATIVE_RULES = [
    "all_red",                    # easy
    "all_same_suit",              # medium-easy
    "all_even",                   # medium
    "triple_2s_pos234",           # hard (specific positions)
    "all_but_one_same_color",     # complex composition
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=int, default=7)
    parser.add_argument("--max-programs", type=int, default=300_000)
    parser.add_argument("--rules", type=str,
                        default=",".join(REPRESENTATIVE_RULES))
    parser.add_argument("--n-candidates", type=int, default=50_000)
    parser.add_argument("--top-k-diagnostic", type=int, default=20)
    parser.add_argument("--top-k-adversarial", type=int, default=10)
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--mass-threshold", type=float, default=0.001)
    parser.add_argument("--extension-samples", type=int, default=10_000)
    parser.add_argument("--inject", type=str,
                        default="src/gallery_analysis/data/injected_hypotheses.json")
    parser.add_argument("--output-dir", type=str,
                        default="review-stage/experiments/night2")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "adversarial_hands_results.json"

    print(f"[ADV] Building pool: depth={args.depth}, max_programs={args.max_programs}",
          flush=True)
    t0 = time.time()
    equiv, stats = build_hypothesis_pool(
        max_depth=args.depth, max_programs=args.max_programs, verbose=1,
    )
    print(f"[ADV] Base pool: {len(equiv):,} classes in {time.time()-t0:.1f}s",
          flush=True)

    probes = stats["_probes"]
    exemplars_disamb = stats.get("_exemplar_hands") or []
    grammar = build_gallery_grammar()
    enum_priors = [c["canonical_prior"] for c in equiv]
    rng_prior = (min(enum_priors), max(enum_priors)) if enum_priors else None

    inject_path = args.inject
    if not os.path.isabs(inject_path):
        inject_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "..", "..", "..", inject_path)
    injected = load_and_validate_injections(
        inject_path, grammar=grammar, enumerated_prior_range=rng_prior,
    )
    print(f"[ADV] Merging {len(injected)} injections...", flush=True)
    equiv = merge_injected(list(equiv), injected, probes)
    if exemplars_disamb:
        equiv, _ = _strict_split_classes(
            equiv, exemplar_hands=exemplars_disamb, main_probes=probes, verbose=0,
        )
    print(f"[ADV] Post-inject pool: {len(equiv):,} classes", flush=True)

    print(f"[ADV] Estimating extensions ({args.extension_samples:,} samples each)...",
          flush=True)
    t1 = time.time()
    extensions = estimate_extensions(equiv, n_samples=args.extension_samples)
    print(f"[ADV] Extensions in {time.time()-t1:.1f}s", flush=True)

    frozen_exemplars = load_exemplars()
    rule_ids = [r.strip() for r in args.rules.split(",") if r.strip()]

    results = {
        "args": vars(args),
        "pool_size": len(equiv),
        "rules": {},
    }

    for rule_id in rule_ids:
        rule = GALLERY_RULES.get(rule_id)
        if rule is None:
            print(f"[ADV] {rule_id}: not in GALLERY_RULES, skip", flush=True)
            continue
        ex_dict = frozen_exemplars.get(rule_id)
        if ex_dict is None or "hands_primary" not in ex_dict:
            print(f"[ADV] {rule_id}: no frozen exemplars, skip", flush=True)
            continue
        exemplars = ex_dict["hands_primary"]
        rule_predicate = rule["predicate"]

        print(f"\n[ADV] === {rule_id}: {rule['answer']} ===", flush=True)

        # Compute posterior under default settings (summed prior, k=1, mass-thresholded)
        t2 = time.time()
        posteriors, retained_mass = compute_posteriors_for_rule(
            equiv_classes=equiv,
            extensions=extensions,
            exemplar_hands=exemplars,
            mass_threshold=args.mass_threshold,
            return_retained_mass=True,
        )
        print(f"  posterior: {len(posteriors)} hyps survive prune (retained={retained_mass:.4f}) "
              f"in {time.time()-t2:.2f}s", flush=True)

        # Per-rule deterministic seed derivation (Round 1 finding #6, fixed
        # in Round 2 finding #1). We use zlib.crc32 because Python's built-in
        # ``hash()`` is per-process randomized (PEP 456) unless
        # PYTHONHASHSEED is fixed — that would silently drift the candidate
        # set across overnight runs.
        rule_crc = zlib.crc32(rule_id.encode("utf-8"))
        seed_diag = (12345 + rule_crc) & 0xFFFFFFFF
        seed_adv = (23456 + rule_crc) & 0xFFFFFFFF

        # ============================================================
        # 1. Most diagnostic hands (BALD-on-survivors)
        # ============================================================
        # ``retained_mass`` is threaded through so the search both warns
        # (UserWarning) when below floor and stamps each returned hand with
        # the parent posterior's mass — required by Round 1 finding #1.
        t3 = time.time()
        diag = find_most_diagnostic_hands(
            posteriors, equiv,
            n_candidates=args.n_candidates,
            top_k=args.top_k_diagnostic,
            ground_truth_pred=rule_predicate,
            seed=seed_diag,
            retained_mass=retained_mass,
        )
        print(f"  diagnostic: {len(diag)} hands (top entropy={diag[0].entropy_bits:.4f} "
              f"p_accept={diag[0].p_accept:.3f}) in {time.time()-t3:.2f}s", flush=True)

        # ============================================================
        # 2. Adversarial: confident-but-wrong
        # ============================================================
        t4 = time.time()
        adv = find_most_adversarial_hands(
            posteriors, equiv,
            rule_predicate=rule_predicate,
            n_candidates=args.n_candidates,
            top_k=args.top_k_adversarial,
            confidence_threshold=args.confidence_threshold,
            seed=seed_adv,
            retained_mass=retained_mass,
        )
        n_fp = len(adv["false_positives"])
        n_fn = len(adv["false_negatives"])
        print(f"  adversarial: {n_fp} FP / {n_fn} FN (τ={args.confidence_threshold}) "
              f"in {time.time()-t4:.2f}s", flush=True)

        results["rules"][rule_id] = {
            "answer": rule["answer"],
            "n_exemplars": len(exemplars),
            "n_posterior_hyps": len(posteriors),
            "retained_posterior_mass": retained_mass,
            "diagnostic_hands": [adversarial_hand_to_dict(h) for h in diag],
            "adversarial_false_positives": [
                adversarial_hand_to_dict(h) for h in adv["false_positives"]
            ],
            "adversarial_false_negatives": [
                adversarial_hand_to_dict(h) for h in adv["false_negatives"]
            ],
            "diagnostic_summary": {
                "max_entropy_bits": diag[0].entropy_bits if diag else 0.0,
                "min_entropy_bits": diag[-1].entropy_bits if diag else 0.0,
                "max_p_accept": max((h.p_accept for h in diag), default=0.0),
                "min_p_accept": min((h.p_accept for h in diag), default=0.0),
                "fraction_correct": (
                    sum(1 for h in diag if h.correct_prediction) / len(diag)
                    if diag else 0.0
                ),
            },
            "adversarial_summary": {
                "n_false_positives": n_fp,
                "n_false_negatives": n_fn,
                "max_p_accept_fp": max((h.p_accept for h in adv["false_positives"]),
                                       default=0.0),
                "min_p_accept_fn": min((h.p_accept for h in adv["false_negatives"]),
                                       default=1.0),
            },
        }

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n[ADV] Wrote {out_json}", flush=True)
    print("\n=== SUMMARY ===", flush=True)
    for rule_id, r in results["rules"].items():
        ds = r["diagnostic_summary"]
        as_ = r["adversarial_summary"]
        print(f"  {rule_id}: max_H={ds['max_entropy_bits']:.4f} bits, "
              f"FP={as_['n_false_positives']}, FN={as_['n_false_negatives']}",
              flush=True)


if __name__ == "__main__":
    main()
