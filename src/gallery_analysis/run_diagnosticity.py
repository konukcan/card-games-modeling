"""
CLI for hand diagnosticity analysis.

Rates how diagnostic (easy/hard to classify) random candidate hands are for
each gallery rule, given the (optionally tempered) posterior over hypotheses
from the exemplar hands. When ``--likelihood-exponent != 1`` the distribution
is a power / tempered posterior π_k(h|D) ∝ π(h)·P(D|h)^k, not the standard
Bayesian posterior (Finding 7). Produces a "diagnostic spectrum" showing the
distribution of classification confidence across random hands.

Usage:
    cd src

    # Single rule:
    python gallery_analysis/run_diagnosticity.py \
        --rule all_red \
        --n-candidates 10000 \
        --extension-cache gallery_analysis/results/extension_cache_depth6.json \
        --inject gallery_analysis/injected_hypotheses.json \
        --output gallery_analysis/results/diagnosticity_all_red.json \
        --verbose 2

    # All 60 rules:
    python gallery_analysis/run_diagnosticity.py \
        --all-rules \
        --n-candidates 5000 \
        --extension-cache gallery_analysis/results/extension_cache_depth6.json \
        --inject gallery_analysis/injected_hypotheses.json \
        --output gallery_analysis/results/diagnosticity_all_rules.json

    # Specific rules:
    python gallery_analysis/run_diagnosticity.py \
        --rule all_red --rule three_of_a_kind --rule zigzag_ranks \
        --n-candidates 10000 \
        --extension-cache gallery_analysis/results/extension_cache_depth6.json \
        --inject gallery_analysis/injected_hypotheses.json \
        --verbose 2
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from gallery_analysis.analyze import build_hypothesis_pool, estimate_extensions
from gallery_analysis.bayesian_scorer import compute_log_likelihood_noisy
from gallery_analysis.enumerator import build_gallery_grammar
from gallery_analysis.exemplars import load_exemplars, generate_probe_set
from gallery_analysis.gallery_rules import GALLERY_RULES
from gallery_analysis.hand_diagnosticity import (
    DiagnosticityReport,
    DiagnosticSpectrum,
    compute_posteriors_for_rule,
    generate_diagnostic_spectrum,
)
from gallery_analysis.injection import load_and_validate_injections, merge_injected
from gallery_analysis.provenance import compute_provenance


def _card_to_str(card) -> str:
    """Format a Card as a short string like 'A♠' or 'K♥'."""
    suit_symbols = {"HEARTS": "♥", "DIAMONDS": "♦", "CLUBS": "♣", "SPADES": "♠"}
    return f"{card.rank.value}{suit_symbols.get(card.suit.value, card.suit.value)}"


def _hand_to_str(hand) -> str:
    """Format a Hand as a compact string."""
    return " ".join(_card_to_str(c) for c in hand)


def _report_to_dict(report: DiagnosticityReport) -> Dict[str, Any]:
    """Convert a DiagnosticityReport to a JSON-serializable dict."""
    return {
        "hand": [{"suit": c.suit.value, "rank": c.rank.value} for c in report.hand],
        "hand_str": _hand_to_str(report.hand),
        "rule_id": report.rule_id,
        "p_accept": round(report.p_accept, 6),
        "confidence": round(report.confidence, 6),
        "ground_truth": report.ground_truth,
        "correct_prediction": report.correct_prediction,
        "top_hypotheses_votes": [
            {
                "program": v["program"],
                "prob": round(v["prob"], 6),
                "accepts_hand": v["accepts_hand"],
            }
            for v in report.top_hypotheses_votes
        ],
    }


def _spectrum_to_dict(spectrum: DiagnosticSpectrum) -> Dict[str, Any]:
    """Convert a DiagnosticSpectrum to a JSON-serializable dict."""
    return {
        "rule_id": spectrum.rule_id,
        "group": spectrum.group,
        "n_candidates": spectrum.n_candidates,
        "mean_p_accept": round(spectrum.mean_p_accept, 6),
        "std_p_accept": round(spectrum.std_p_accept, 6),
        "mean_confidence": round(spectrum.mean_confidence, 6),
        "fraction_high_confidence": round(spectrum.fraction_high_confidence, 6),
        "fraction_ambiguous": round(spectrum.fraction_ambiguous, 6),
        "accuracy": round(spectrum.accuracy, 6),
        "p_accept_histogram": spectrum.p_accept_histogram,
        "gt_histogram": spectrum.gt_histogram,
        "balanced_gt_histogram": spectrum.balanced_gt_histogram,
        "hand_summaries": spectrum.hand_summaries,
        "balanced_hand_summaries": spectrum.balanced_hand_summaries,
        "easy_accept_hands": [_report_to_dict(r) for r in spectrum.easy_accept_hands],
        "easy_reject_hands": [_report_to_dict(r) for r in spectrum.easy_reject_hands],
        "ambiguous_hands": [_report_to_dict(r) for r in spectrum.ambiguous_hands],
        "balanced_reports": [_report_to_dict(r) for r in spectrum.balanced_reports],
        # Round 3 Finding 4: ``balanced_n`` now reports the achieved
        # balanced count, not the target. ``balanced_n_target`` preserves
        # the requested value; _accept_actual / _reject_actual break down
        # what was obtained; _attempts is the total rejection-sampling
        # attempts. Downstream readers can detect time-outs via
        # balanced_n < balanced_n_target.
        "balanced_n": spectrum.balanced_n,
        "balanced_n_target": spectrum.balanced_n_target,
        "balanced_n_accept_actual": spectrum.balanced_n_accept_actual,
        "balanced_n_reject_actual": spectrum.balanced_n_reject_actual,
        "balanced_attempts": spectrum.balanced_attempts,
        # Finding 5 (R1) + Finding 4 (R2): serialize the pruning truncation
        # so the JSON artefact carries the total-variation bound on the
        # reported posterior. Downstream readers should treat retained mass
        # < 1.0 as the posterior being conditional on "survived threshold".
        "retained_posterior_mass": round(spectrum.retained_posterior_mass, 6),
        "discarded_posterior_mass": round(spectrum.discarded_posterior_mass, 6),
    }


def print_spectrum_report(spectrum: DiagnosticSpectrum, verbose: int = 1):
    """Print a human-readable summary of a diagnostic spectrum."""
    rule_info = GALLERY_RULES.get(spectrum.rule_id, {})
    answer = rule_info.get("answer", "?")
    group_labels = {1: "Easy", 2: "Medium", 3: "Hard"}
    group_label = group_labels.get(spectrum.group, "?")

    print(f"\n{'─'*70}")
    print(f"{spectrum.rule_id}  (group={spectrum.group}/{group_label}, \"{answer}\")")
    print(f"  Candidates: {spectrum.n_candidates:,}")
    print(f"  Mean P(accept):  {spectrum.mean_p_accept:.3f} ± {spectrum.std_p_accept:.3f}")
    print(f"  Mean confidence: {spectrum.mean_confidence:.3f}")
    print(f"  High confidence: {spectrum.fraction_high_confidence*100:.1f}% (conf > 0.8)")
    print(f"  Ambiguous:       {spectrum.fraction_ambiguous*100:.1f}% (conf < 0.2)")
    print(f"  Accuracy:        {spectrum.accuracy*100:.1f}%")
    # Finding 5 (R1) + Finding 4 (R2): surface pruning truncation so it's
    # not hidden in the JSON.
    if spectrum.discarded_posterior_mass > 0:
        print(
            f"  Pruned posterior: retained={spectrum.retained_posterior_mass:.4f}, "
            f"discarded={spectrum.discarded_posterior_mass:.4f} "
            f"(TV bound on reported posterior ≤ discarded)"
        )

    # Histogram
    print(f"\n  P(accept) distribution:")
    max_count = max(spectrum.p_accept_histogram.values()) if spectrum.p_accept_histogram else 1
    for bin_label, count in spectrum.p_accept_histogram.items():
        bar_len = int(40 * count / max(max_count, 1))
        bar = "█" * bar_len
        pct = count / spectrum.n_candidates * 100
        print(f"    {bin_label}: {bar:<40} {count:>5} ({pct:5.1f}%)")

    if verbose >= 2:
        # Representative hands
        if spectrum.easy_accept_hands:
            print(f"\n  Easy ACCEPT hands (high confidence, P(accept) > 0.5):")
            for r in spectrum.easy_accept_hands[:3]:
                gt_marker = "✓" if r.ground_truth else "✗"
                print(f"    {_hand_to_str(r.hand)}  P={r.p_accept:.3f}  "
                      f"conf={r.confidence:.3f}  truth={gt_marker}")

        if spectrum.easy_reject_hands:
            print(f"\n  Easy REJECT hands (high confidence, P(accept) ≤ 0.5):")
            for r in spectrum.easy_reject_hands[:3]:
                gt_marker = "✓" if not r.ground_truth else "✗"
                print(f"    {_hand_to_str(r.hand)}  P={r.p_accept:.3f}  "
                      f"conf={r.confidence:.3f}  truth={gt_marker}")

        if spectrum.ambiguous_hands:
            print(f"\n  Ambiguous hands (confidence < 0.2):")
            for r in spectrum.ambiguous_hands[:3]:
                print(f"    {_hand_to_str(r.hand)}  P={r.p_accept:.3f}  "
                      f"conf={r.confidence:.3f}  truth={'T' if r.ground_truth else 'F'}")


def main():
    parser = argparse.ArgumentParser(
        description="Hand diagnosticity analysis for gallery rules"
    )
    parser.add_argument(
        "--rule", type=str, action="append", default=None,
        help="Rule ID(s) to analyze (can specify multiple times)"
    )
    parser.add_argument(
        "--all-rules", action="store_true",
        help="Analyze all 60 gallery rules"
    )
    parser.add_argument(
        "--n-candidates", type=int, default=10_000,
        help="Number of random candidate hands per rule (default: 10000)"
    )
    parser.add_argument(
        "--depth", type=int, default=6,
        help=("Enumeration depth BUDGET passed to the top-down enumerator "
              "(application-depth budget, NOT Program.depth()). Emitted "
              "programs can reach Program.depth() == max_depth+1. Default: 6.")
    )
    parser.add_argument(
        "--max-programs", type=int, default=500_000,
        help="Max programs to enumerate (default: 500000)"
    )
    parser.add_argument(
        "--extension-cache", type=str, default=None,
        help="Path to extension size cache JSON"
    )
    parser.add_argument(
        "--inject", type=str, default=None,
        help="Path to injected hypotheses JSON"
    )
    parser.add_argument(
        "--mass-threshold", type=float, default=0.001,
        help="Posterior mass threshold for pruning (default: 0.001 = 0.1%%)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Save results JSON to this path"
    )
    parser.add_argument(
        "--verbose", type=int, default=1,
        help="Verbosity: 0=silent, 1=summary, 2=detailed"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for candidate hand generation (default: 42)"
    )
    parser.add_argument(
        "--balanced", type=int, default=500,
        help="Balanced sampling: generate this many accept + this many reject "
             "hands per rule via rejection sampling (default: 500, 0 to disable)"
    )
    parser.add_argument(
        "--grammar", choices=["uniform", "weighted"], default="uniform",
        help="Scoring grammar: 'uniform' (baseline) or 'weighted' (4-tier)"
    )
    parser.add_argument(
        "--prior", choices=["canonical", "summed"], default="summed",
        help="Prior mode: 'canonical' (single cheapest) or 'summed' (log-sum-exp)"
    )
    parser.add_argument(
        "--likelihood-exponent", type=float, default=1.0,
        help="Exponent k on P(D|h)^k. k>1 inflates size principle (default 1.0)"
    )
    parser.add_argument(
        "--targeted-probes", action="store_true", default=False,
        help="Use Config I probes (targeted + near-miss) instead of random"
    )
    parser.add_argument(
        "--enumeration-grammar", choices=["uniform", "weighted"],
        default="uniform",
        help="Grammar driving top-down enumeration (Finding 2). Combine with "
             "--grammar weighted for honest weighted-model runs."
    )
    parser.add_argument(
        "--strict-priors", action="store_true",
        help="Fail fast on non-finite per-program priors (Finding 3). "
             "Recommended for official runs."
    )
    parser.add_argument(
        "--n-probes", type=int, default=500,
        help="Random probe hands for fingerprinting (ignored when --targeted-probes)"
    )
    args = parser.parse_args()

    # Determine which rules to analyze
    if args.all_rules:
        rule_ids = list(GALLERY_RULES.keys())
    elif args.rule:
        rule_ids = args.rule
        # Validate rule IDs
        for rid in rule_ids:
            if rid not in GALLERY_RULES:
                print(f"❌ Unknown rule: {rid}", flush=True)
                print(f"   Available rules: {', '.join(sorted(GALLERY_RULES.keys())[:10])}...",
                      flush=True)
                sys.exit(1)
    else:
        print("❌ Specify --rule <id> or --all-rules", flush=True)
        sys.exit(1)

    t_total_start = time.time()

    print("=" * 70)
    print("HAND DIAGNOSTICITY ANALYSIS")
    print("=" * 70)
    balanced_info = f", Balanced: {args.balanced}+{args.balanced}" if args.balanced > 0 else ""
    print(f"Rules: {len(rule_ids)}, Candidates/rule: {args.n_candidates:,}, "
          f"Mass threshold: {args.mass_threshold}{balanced_info}", flush=True)

    # Step 1: Build hypothesis pool
    # Finding 2 (R2): honour --enumeration-grammar so weighted scoring runs
    # use a weighted-enumerated pool rather than post-hoc rescoring a uniform
    # pool (those supports differ under truncation).
    equiv_classes, pipeline_stats = build_hypothesis_pool(
        max_depth=args.depth,
        max_programs=args.max_programs,
        n_probes=args.n_probes,
        verbose=args.verbose,
        use_targeted_probes=getattr(args, 'targeted_probes', False),
        enumeration_grammar=args.enumeration_grammar,
    )
    print(f"  {len(equiv_classes):,} equivalence classes", flush=True)

    # Reuse the exact probes and exemplars stashed by build_hypothesis_pool
    # (Finding 4 + Finding 1 R2). Falling back to a fresh random set would
    # use different fingerprints than the pool was built on.
    probes = pipeline_stats.get("_probes") or generate_probe_set(
        n_probes=args.n_probes, seed=42
    )
    exemplar_hands_for_split = pipeline_stats.get("_exemplar_hands") or []

    # Step 2: Inject hypotheses (if provided)
    if args.inject:
        from gallery_analysis.analyze import _strict_split_classes
        grammar = build_gallery_grammar()
        print(f"\nInjecting from {args.inject}...", flush=True)
        injected = load_and_validate_injections(args.inject, grammar=grammar)
        n_before = len(equiv_classes)
        equiv_classes = merge_injected(equiv_classes, injected, probes)
        print(f"  {len(equiv_classes) - n_before} novel classes added", flush=True)

        # Finding 1 R2: re-run strict split after injection merge — injection
        # can reintroduce mixed classes that the first split rejected.
        if exemplar_hands_for_split:
            equiv_classes, resplit_stats = _strict_split_classes(
                equiv_classes,
                exemplar_hands=exemplar_hands_for_split,
                main_probes=probes,
                verbose=args.verbose,
            )
            if resplit_stats["n_split"] > 0 and args.verbose >= 1:
                print(
                    f"  [Finding 1 R2] Post-injection strict split: "
                    f"{resplit_stats['n_split']} classes disagreed → "
                    f"{resplit_stats['n_subclasses']} sub-classes; "
                    f"total now {len(equiv_classes):,}",
                    flush=True,
                )

    # Step 3: Extension sizes (Finding 4 R2: pass probe_hash so stale cache
    # reuse is detected when probes change).
    from gallery_analysis.provenance import compute_probe_hash
    probe_hash = compute_probe_hash(probes)
    print(f"\nEstimating extension sizes...", flush=True)
    extensions = estimate_extensions(
        equiv_classes,
        verbose=args.verbose,
        cache_path=args.extension_cache,
        _probe_hash=probe_hash,
    )

    # Compute provenance metadata using the real probes used above.
    provenance = compute_provenance(
        probe_seed=42,
        n_probes=len(probes),
        probes=probes,
        inject_path=args.inject if args.inject else None,
        n_equiv_classes=len(equiv_classes),
    )

    # Build scoring grammar object if weighted
    scoring_grammar_obj = None
    if args.grammar == "weighted":
        from gallery_analysis.enumerator import build_weighted_gallery_grammar
        scoring_grammar_obj = build_weighted_gallery_grammar()
        print(f"\nUsing WEIGHTED scoring grammar (4-tier)", flush=True)

    # Step 4: Load exemplars
    exemplars = load_exemplars()

    # Step 5: Analyze each rule
    print(f"\nAnalyzing {len(rule_ids)} rules...", flush=True)
    all_spectrums = {}

    for rule_idx, rule_id in enumerate(rule_ids):
        if rule_id not in exemplars:
            print(f"  Skipping {rule_id}: no exemplars", flush=True)
            continue

        t_rule_start = time.time()
        rule_info = GALLERY_RULES[rule_id]
        exemplar_hands = exemplars[rule_id]["hands_primary"]

        # Compute posteriors for this rule (with mass threshold pruning).
        # Finding 5: capture retained_mass so we can report the pruning error.
        posteriors, retained_mass = compute_posteriors_for_rule(
            equiv_classes, extensions, exemplar_hands,
            epsilon=0.01,
            prior_mode=args.prior,
            mass_threshold=args.mass_threshold,
            grammar=scoring_grammar_obj,
            likelihood_exponent=args.likelihood_exponent,
            return_retained_mass=True,
            strict_priors=args.strict_priors,
        )
        discarded_mass = 1.0 - retained_mass

        if args.verbose >= 2:
            print(f"\n  {rule_id}: {len(posteriors)} hypotheses above "
                  f"{args.mass_threshold*100:.1f}% mass threshold "
                  f"(retained={retained_mass:.4f}, "
                  f"TV-error bound={discarded_mass:.4f})", flush=True)
        # Warn loudly if a material fraction of the posterior was discarded.
        # 0.01 is a soft threshold — anything below this means the published
        # p_accept numbers have non-negligible truncation error.
        if discarded_mass > 0.01 and args.verbose >= 1:
            print(f"  WARNING: {rule_id} discarded {discarded_mass*100:.2f}% "
                  f"posterior mass during pruning (threshold={args.mass_threshold}). "
                  f"p_accept TV-error is bounded by this fraction.", flush=True)

        # Generate diagnostic spectrum
        spectrum = generate_diagnostic_spectrum(
            rule_id=rule_id,
            posteriors=posteriors,
            equiv_classes=equiv_classes,
            ground_truth_pred=rule_info["predicate"],
            n_candidates=args.n_candidates,
            seed=args.seed,
            group=rule_info["group"],
            balanced_n=args.balanced,
            verbose=args.verbose,
            retained_mass=retained_mass,
        )

        all_spectrums[rule_id] = spectrum

        t_rule = time.time() - t_rule_start
        if args.verbose >= 1:
            print(f"  [{rule_idx+1}/{len(rule_ids)}] {rule_id:<30} "
                  f"conf={spectrum.mean_confidence:.3f}  "
                  f"hi={spectrum.fraction_high_confidence*100:.0f}%  "
                  f"amb={spectrum.fraction_ambiguous*100:.0f}%  "
                  f"acc={spectrum.accuracy*100:.0f}%  "
                  f"({t_rule:.1f}s)", flush=True)

    # Print detailed reports
    if args.verbose >= 1:
        print(f"\n{'='*70}")
        print("DIAGNOSTIC SPECTRUMS")
        print(f"{'='*70}")

        # Sort by mean confidence (most diagnostic first)
        sorted_spectrums = sorted(
            all_spectrums.values(),
            key=lambda s: -s.mean_confidence,
        )

        for spectrum in sorted_spectrums:
            print_spectrum_report(spectrum, verbose=args.verbose)

        # Summary table
        print(f"\n{'='*70}")
        print("SUMMARY TABLE")
        print(f"{'='*70}")
        print(f"{'Rule':<30} {'Grp':>3} {'MeanConf':>8} {'Hi%':>5} {'Amb%':>5} "
              f"{'Acc%':>5} {'MeanP':>6}")
        print("─" * 70)

        for s in sorted_spectrums:
            print(f"  {s.rule_id:<28} {s.group:>3} "
                  f"{s.mean_confidence:>8.3f} "
                  f"{s.fraction_high_confidence*100:>4.0f}% "
                  f"{s.fraction_ambiguous*100:>4.0f}% "
                  f"{s.accuracy*100:>4.0f}% "
                  f"{s.mean_p_accept:>6.3f}")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "config": {
                "n_candidates": args.n_candidates,
                "mass_threshold": args.mass_threshold,
                "depth": args.depth,
                "seed": args.seed,
                "n_equiv_classes": len(equiv_classes),
                "balanced_n": args.balanced,
            },
            "provenance": provenance,
            "spectrums": {
                rule_id: _spectrum_to_dict(spectrum)
                for rule_id, spectrum in all_spectrums.items()
            },
        }

        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {output_path}", flush=True)

    t_total = time.time() - t_total_start
    print(f"\nTotal time: {t_total:.1f}s", flush=True)


if __name__ == "__main__":
    main()
