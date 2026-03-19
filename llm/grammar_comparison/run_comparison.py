"""Main entry point for the grammar comparison and ablation pipeline.

Runs either a broad comparison (Stage 1) across grammar x cost-structure
combinations with 6 metrics and 5-fold cross-validation, or a fine-grained
ablation analysis (Stage 2) around a single base grammar.

Usage
-----
Stage 1 -- broad comparison across all 7 grammars x 3 cost structures:
    python -m llm.grammar_comparison.run_comparison --stage 1

Stage 1 -- subset of grammars, limited hypotheses (quick test):
    python -m llm.grammar_comparison.run_comparison --stage 1 --limit 10 --grammars base minimal

Stage 2 -- ablation around a specific grammar:
    python -m llm.grammar_comparison.run_comparison --stage 2 --base-grammar swap-both
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup: allow importing from the main src/ tree
# ---------------------------------------------------------------------------
_SRC_DIR = str(Path(__file__).resolve().parent.parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from llm.grammar_comparison.grammars.grammar_factory import (
    CostStructure,
    GRAMMAR_NAMES,
)
from llm.grammar_comparison.evaluation.compute_costs import score_all_hypotheses
from llm.grammar_comparison.evaluation.metrics import (
    spearman_agreement,
    weighted_log_probability,
    top1_accuracy_corrected,
    expressibility,
    correct_rank,
    rule_difficulty_correlation,
)
from llm.grammar_comparison.evaluation.ablation import (
    leave_one_out,
    leave_one_in,
    cross_validate,
)


# ---------------------------------------------------------------------------
# Ground-truth helpers
# ---------------------------------------------------------------------------

def _load_rule_groups() -> Dict[str, int]:
    """Load rule difficulty groups from the Phase 1b JSON files.

    Each Phase 1b JSON file has a top-level 'rule_group' field (1 = easy,
    2 = medium, 3 = hard). We scan the directory and build a mapping from
    rule_id to rule_group integer.

    Returns:
        Dict mapping rule_id -> difficulty group (int).
    """
    phase1b_dir = (
        Path(__file__).resolve().parent.parent / "results" / "phase1b"
    )
    if not phase1b_dir.exists():
        return {}

    rule_groups: Dict[str, int] = {}
    for json_file in sorted(phase1b_dir.glob("*.json")):
        # Only read dsl-constrained files to avoid duplicates
        parts = json_file.stem.split("__")
        if len(parts) < 3:
            continue
        if parts[1] != "dsl-constrained":
            continue

        with open(json_file, "r") as f:
            data = json.load(f)

        rule_id = data.get("rule_id")
        rule_group = data.get("rule_group")
        if rule_id and rule_group is not None:
            rule_groups[rule_id] = int(rule_group)

    return rule_groups


def _load_ground_truth_fingerprints() -> Dict[str, str]:
    """Compute fingerprints for ground-truth rules on the 200-probe set.

    Loads the catalogue rules from src/rules/catalogue.py, evaluates each
    rule's predicate on the 200 probe hands, and returns a mapping from
    rule_id to fingerprint string ('1'/'0'/'X').

    Returns:
        Dict mapping rule_id -> fingerprint string.
    """
    from rules.catalogue import create_all_rules
    from llm.grammar_comparison.translation.verification import load_probe_hands

    try:
        probes = load_probe_hands()
    except FileNotFoundError:
        return {}

    rules = create_all_rules()
    gt_fingerprints: Dict[str, str] = {}

    for rule in rules:
        fp_chars = []
        for hand in probes:
            try:
                result = rule.predicate(hand)
                fp_chars.append('1' if result else '0')
            except Exception:
                fp_chars.append('X')
        gt_fingerprints[rule.id] = ''.join(fp_chars)

    return gt_fingerprints


# ---------------------------------------------------------------------------
# Cross-validation helper for Stage 1
# ---------------------------------------------------------------------------

def _cross_validate_metrics(
    scored: List[Dict],
    ground_truth_fps: Dict[str, str],
    rule_groups: Dict[str, int],
    k: int = 5,
) -> Dict[str, Tuple[float, float]]:
    """Compute all 6 metrics with k-fold cross-validation.

    Splits the scored hypotheses into k folds (by hypothesis index, not by
    rule). For each fold, computes the 6 metrics on the held-out subset.
    Returns mean +/- std for each metric.

    The scoring (prior + likelihood) is computed once on all data; only the
    metric computation is done per-fold on the held-out subset.

    Args:
        scored: List of scored hypothesis dicts from score_all_hypotheses.
        ground_truth_fps: Dict mapping rule_id -> ground-truth fingerprint.
        rule_groups: Dict mapping rule_id -> difficulty group (int).
        k: Number of folds (default 5).

    Returns:
        Dict mapping metric_name -> (mean, std).
    """
    n = len(scored)
    if n == 0:
        return {
            "agree": (0.0, 0.0),
            "wtd_lp": (0.0, 0.0),
            "top1": (0.0, 0.0),
            "crank": (float("inf"), 0.0),
            "rdiff": (0.0, 0.0),
            "expr": (0.0, 0.0),
        }

    effective_k = min(k, n)
    fold_size = n // effective_k

    # Collect per-fold metric values
    fold_results: Dict[str, List[float]] = defaultdict(list)

    for i in range(effective_k):
        start = i * fold_size
        end = n if i == effective_k - 1 else start + fold_size
        fold = scored[start:end]

        fold_results["agree"].append(spearman_agreement(fold))
        fold_results["wtd_lp"].append(weighted_log_probability(fold))
        fold_results["top1"].append(top1_accuracy_corrected(fold))
        fold_results["expr"].append(expressibility(fold))
        fold_results["crank"].append(correct_rank(fold, ground_truth_fps))
        fold_results["rdiff"].append(rule_difficulty_correlation(fold, rule_groups))

    # Compute mean and std for each metric
    results: Dict[str, Tuple[float, float]] = {}
    for name, values in fold_results.items():
        # Filter out inf values for mean/std computation (correct_rank returns inf
        # when no rules can be evaluated)
        finite_values = [v for v in values if not math.isinf(v)]
        if not finite_values:
            results[name] = (float("inf"), 0.0)
            continue
        mean = sum(finite_values) / len(finite_values)
        variance = sum((v - mean) ** 2 for v in finite_values) / len(finite_values)
        std = math.sqrt(variance)
        results[name] = (mean, std)

    return results


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_HEADER_WIDTH = 105


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace inf/nan floats with JSON-safe string representations."""
    if isinstance(obj, float):
        if math.isinf(obj):
            return "Infinity" if obj > 0 else "-Infinity"
        if math.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _fmt(value: float, width: int = 8) -> str:
    """Format a float for table display, handling -inf and inf gracefully."""
    if math.isinf(value):
        sign = "-" if value < 0 else ""
        return f"{sign + 'inf':>{width}}"
    return f"{value:>{width}.4f}"


def _fmt_pm(mean: float, std: float, width: int = 14) -> str:
    """Format a mean +/- std value for table display.

    Produces a compact 'mean+/-std' string (e.g., '0.42+/-0.05').
    Handles inf values gracefully.
    """
    if math.isinf(mean):
        sign = "-" if mean < 0 else ""
        return f"{sign + 'inf':>{width}}"
    return f"{mean:.2f}\u00b1{std:.2f}".rjust(width)


def _print_header(title: str) -> None:
    """Print a section header with double-line border."""
    print()
    print(title)
    print("=" * _HEADER_WIDTH)


def _print_separator() -> None:
    """Print a single-line separator."""
    print("-" * _HEADER_WIDTH)


# ---------------------------------------------------------------------------
# Stage 1: Broad grammar comparison
# ---------------------------------------------------------------------------

def run_stage1(
    grammar_names: List[str],
    limit: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run Stage 1: score all hypotheses under each grammar x cost combination.

    For each grammar and cost structure pair:
      1. Call score_all_hypotheses to get per-hypothesis log-posteriors with
         fingerprints.
      2. Compute all 6 evaluation metrics with 5-fold cross-validation.
      3. Print a summary row and collect results for JSON export.

    The 6 metrics are:
      - spearman_agreement: Average negated Spearman rho (positive = agreement)
      - weighted_log_probability: Rank-weighted sum of log-posteriors
      - top1_accuracy_corrected: Chance-corrected top-1 accuracy
      - expressibility: Fraction of hypotheses with finite log-prob
      - correct_rank: Average rank of ground-truth hypothesis (lower = better)
      - rule_difficulty_correlation: Correlation with human difficulty groups

    Args:
        grammar_names: List of grammar variant names to evaluate.
        limit: If > 0, only process the first `limit` hypotheses.
        output_dir: Directory to write stage1_results.json.

    Returns:
        Dict with 'rows' (list of per-configuration result dicts) and
        'timing_seconds' (total wall-clock time).
    """
    cost_structures = list(CostStructure)

    # Load auxiliary data for correct_rank and rule_difficulty_correlation
    ground_truth_fps = _load_ground_truth_fingerprints()
    rule_groups = _load_rule_groups()

    _print_header("Grammar Comparison -- Stage 1 (5-fold cross-validated, posterior scoring)")
    print(f"Grammars: {', '.join(grammar_names)}")
    print(f"Cost structures: {', '.join(c.value for c in cost_structures)}")
    if limit > 0:
        print(f"Hypothesis limit: {limit}")
    print(f"Ground-truth fingerprints: {len(ground_truth_fps)} rules")
    print(f"Rule difficulty groups: {len(rule_groups)} rules")
    print()

    # Table header — compact format with mean+/-std
    header = (
        f"{'Grammar':<21}"
        f"{'Cost':<10}"
        f"{'Agree':>14}"
        f"{'WtdLP':>18}"
        f"{'Top1':>14}"
        f"{'CRank':>14}"
        f"{'RDiff':>8}"
        f"{'Expr':>8}"
    )
    print(header)
    _print_separator()

    rows: List[Dict[str, Any]] = []
    t0 = time.time()

    for grammar_name in grammar_names:
        for cost in cost_structures:
            # Score all hypotheses under this grammar + cost
            scored = score_all_hypotheses(grammar_name, cost, limit=limit)

            if not scored:
                # Graceful handling of empty results
                print(
                    f"{grammar_name:<21}{cost.value:<10}"
                    f"{'  (no data)':>60}"
                )
                rows.append({
                    "grammar": grammar_name,
                    "cost": cost.value,
                    "n_hypotheses": 0,
                    "metrics": {},
                })
                continue

            # Compute all 6 metrics with cross-validation
            cv_metrics = _cross_validate_metrics(
                scored, ground_truth_fps, rule_groups, k=5
            )

            # Also compute full-sample (non-CV) expressibility and rdiff
            # since those don't benefit much from CV and are useful as single values
            full_expr = expressibility(scored)
            full_rdiff = rule_difficulty_correlation(scored, rule_groups)

            # Print table row
            agree_m, agree_s = cv_metrics["agree"]
            wtd_m, wtd_s = cv_metrics["wtd_lp"]
            top1_m, top1_s = cv_metrics["top1"]
            crank_m, crank_s = cv_metrics["crank"]

            print(
                f"{grammar_name:<21}"
                f"{cost.value:<10}"
                f"{_fmt_pm(agree_m, agree_s, 14)}"
                f"{_fmt_pm(wtd_m, wtd_s, 18)}"
                f"{_fmt_pm(top1_m, top1_s, 14)}"
                f"{_fmt_pm(crank_m, crank_s, 14)}"
                f"{_fmt(full_rdiff, 8)}"
                f"{_fmt(full_expr, 8)}"
            )

            rows.append({
                "grammar": grammar_name,
                "cost": cost.value,
                "n_hypotheses": len(scored),
                "metrics": {
                    "spearman_agreement": {
                        "mean": agree_m, "std": agree_s,
                    },
                    "weighted_log_prob": {
                        "mean": wtd_m, "std": wtd_s,
                    },
                    "top1_accuracy_corrected": {
                        "mean": top1_m, "std": top1_s,
                    },
                    "correct_rank": {
                        "mean": crank_m, "std": crank_s,
                    },
                    "rule_difficulty_correlation": full_rdiff,
                    "expressibility": full_expr,
                },
            })

    elapsed = time.time() - t0
    print()
    print(f"Completed in {elapsed:.1f}s")

    # Save results JSON
    results = {"rows": rows, "timing_seconds": round(elapsed, 2)}
    output_path = output_dir / "stage1_results.json"

    with open(output_path, "w") as f:
        json.dump(_sanitize_for_json(results), f, indent=2)
    print(f"Results saved to {output_path}")

    return results


# ---------------------------------------------------------------------------
# Stage 2: Ablation analysis
# ---------------------------------------------------------------------------

def run_stage2(
    base_grammar: str,
    limit: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """Run Stage 2: ablation analysis around a single base grammar.

    Performs three analyses:
      1. Leave-one-out: remove each primitive one at a time and measure
         impact on the Spearman metric.
      2. Leave-one-in: add candidate primitives one at a time and measure
         impact on the Spearman metric.
      3. Cross-validation: 5-fold CV of the Spearman metric.

    Note: The ablation framework currently uses prior-only scoring via
    score_hypothesis (backward-compatible). A future update will switch
    to posterior scoring by refactoring _score_hypotheses_with_grammar.

    Args:
        base_grammar: The grammar name to ablate (e.g. "swap-both").
        limit: If > 0, only process the first `limit` hypotheses.
        output_dir: Directory to write stage2_results.json.

    Returns:
        Dict with 'leave_one_out', 'leave_one_in', 'cross_validate' results
        and 'timing_seconds'.
    """
    _print_header(f"Ablation Analysis -- Stage 2 (base: {base_grammar})")
    if limit > 0:
        print(f"Hypothesis limit: {limit}")
    print()

    t0 = time.time()
    all_results: Dict[str, Any] = {"base_grammar": base_grammar}

    # --- Leave-one-out ---
    print("Leave-One-Out Analysis")
    print(f"  Removing each primitive from '{base_grammar}' one at a time...")
    loo_results = leave_one_out(base_grammar, limit=limit)
    all_results["leave_one_out"] = loo_results

    top_n = min(10, len(loo_results))
    print(f"\n  Top {top_n} most impactful removals (by |delta| on Spearman):")
    _print_separator()
    print(f"  {'Removed':<25}{'Metric':>10}{'Baseline':>10}{'Delta':>10}")
    _print_separator()
    for r in loo_results[:top_n]:
        print(
            f"  {r['removed']:<25}"
            f"{_fmt(r['metric_value'], 10)}"
            f"{_fmt(r['baseline_value'], 10)}"
            f"{_fmt(r['delta'], 10)}"
        )
    print()

    # --- Leave-one-in ---
    print("Leave-One-In Analysis")
    print(f"  Adding candidate primitives to '{base_grammar}' one at a time...")
    loi_results = leave_one_in(base_grammar, limit=limit)
    all_results["leave_one_in"] = loi_results

    top_n = min(10, len(loi_results))
    print(f"\n  Top {top_n} most beneficial additions (by |delta| on Spearman):")
    _print_separator()
    print(f"  {'Added':<25}{'Metric':>10}{'Baseline':>10}{'Delta':>10}")
    _print_separator()
    for r in loi_results[:top_n]:
        print(
            f"  {r['added']:<25}"
            f"{_fmt(r['metric_value'], 10)}"
            f"{_fmt(r['baseline_value'], 10)}"
            f"{_fmt(r['delta'], 10)}"
        )
    print()

    # --- Cross-validation ---
    print("Cross-Validation (5-fold)")
    cv_mean, cv_std = cross_validate(base_grammar, limit=limit)
    all_results["cross_validate"] = {"mean": cv_mean, "std": cv_std}
    print(f"  Spearman (mean +/- std): {cv_mean:.4f} +/- {cv_std:.4f}")
    print()

    elapsed = time.time() - t0
    all_results["timing_seconds"] = round(elapsed, 2)
    print(f"Completed in {elapsed:.1f}s")

    # Save results JSON
    output_path = output_dir / "stage2_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {output_path}")

    return all_results


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments for the grammar comparison pipeline.

    Args:
        argv: Optional list of arguments (defaults to sys.argv[1:]).

    Returns:
        Parsed argparse.Namespace with stage, grammars, limit, base_grammar,
        and output_dir attributes.
    """
    parser = argparse.ArgumentParser(
        description="Grammar comparison and ablation analysis pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m llm.grammar_comparison.run_comparison --stage 1\n"
            "  python -m llm.grammar_comparison.run_comparison --stage 1 --limit 10 --grammars base minimal\n"
            "  python -m llm.grammar_comparison.run_comparison --stage 2 --base-grammar swap-both\n"
        ),
    )

    parser.add_argument(
        "--stage",
        type=int,
        required=True,
        choices=[1, 2],
        help="Stage to run: 1 = broad comparison, 2 = ablation analysis.",
    )

    parser.add_argument(
        "--grammars",
        nargs="+",
        default=GRAMMAR_NAMES,
        choices=GRAMMAR_NAMES,
        metavar="GRAMMAR",
        help=(
            "Space-separated list of grammar names to evaluate. "
            f"Choices: {', '.join(GRAMMAR_NAMES)}. Default: all 7."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit the number of hypotheses processed (0 = all). Default: 0.",
    )

    parser.add_argument(
        "--base-grammar",
        type=str,
        default="base",
        choices=GRAMMAR_NAMES,
        metavar="GRAMMAR",
        help=(
            "For stage 2: which grammar to perform ablation around. "
            f"Choices: {', '.join(GRAMMAR_NAMES)}. Default: base."
        ),
    )

    default_output = str(
        Path(__file__).resolve().parent.parent / "results" / "grammar_comparison"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output,
        help=f"Directory to save results JSON. Default: {default_output}",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    """Entry point: parse arguments and dispatch to Stage 1 or Stage 2.

    Args:
        argv: Optional argument list (for testing). Defaults to sys.argv.
    """
    args = parse_args(argv)

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.stage == 1:
        run_stage1(args.grammars, args.limit, output_dir)
    elif args.stage == 2:
        run_stage2(args.base_grammar, args.limit, output_dir)


if __name__ == "__main__":
    main()
