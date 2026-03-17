"""Main entry point for the grammar comparison and ablation pipeline.

Runs either a broad comparison (Stage 1) across grammar x cost-structure
combinations, or a fine-grained ablation analysis (Stage 2) around a single
base grammar.

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
from pathlib import Path
from typing import Any, Dict, List

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
    spearman_rank_correlation,
    weighted_log_probability,
    top1_accuracy,
    expressibility,
)
from llm.grammar_comparison.evaluation.ablation import (
    leave_one_out,
    leave_one_in,
    cross_validate,
)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

_HEADER_WIDTH = 75


def _fmt(value: float, width: int = 8) -> str:
    """Format a float for table display, handling -inf gracefully."""
    if math.isinf(value):
        return f"{'  -inf':>{width}}"
    return f"{value:>{width}.4f}"


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
      1. Call score_all_hypotheses to get per-hypothesis log-probabilities.
      2. Compute the four evaluation metrics (Spearman, weighted log-prob,
         top-1 accuracy, expressibility).
      3. Print a summary row and collect results for JSON export.

    Args:
        grammar_names: List of grammar variant names to evaluate.
        limit: If > 0, only process the first `limit` hypotheses.
        output_dir: Directory to write stage1_results.json.

    Returns:
        Dict with 'rows' (list of per-configuration result dicts) and
        'timing_seconds' (total wall-clock time).
    """
    cost_structures = list(CostStructure)

    _print_header("Grammar Comparison -- Stage 1")
    print(f"Grammars: {', '.join(grammar_names)}")
    print(f"Cost structures: {', '.join(c.value for c in cost_structures)}")
    if limit > 0:
        print(f"Hypothesis limit: {limit}")
    print()

    # Table header
    header = (
        f"{'Grammar':<21}"
        f"{'Cost':<10}"
        f"{'Spearman':>9}"
        f"{'WtdLogP':>10}"
        f"{'Top1':>8}"
        f"{'Express':>9}"
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
                    f"{'  (no data)':>36}"
                )
                rows.append({
                    "grammar": grammar_name,
                    "cost": cost.value,
                    "spearman": None,
                    "weighted_log_prob": None,
                    "top1": None,
                    "expressibility": None,
                })
                continue

            # Compute the four metrics
            sp = spearman_rank_correlation(scored)
            wlp = weighted_log_probability(scored)
            t1_acc = top1_accuracy(scored)
            expr = expressibility(scored)

            # Print table row
            print(
                f"{grammar_name:<21}"
                f"{cost.value:<10}"
                f"{_fmt(sp, 9)}"
                f"{_fmt(wlp, 10)}"
                f"{_fmt(t1_acc, 8)}"
                f"{_fmt(expr, 9)}"
            )

            rows.append({
                "grammar": grammar_name,
                "cost": cost.value,
                "spearman": sp,
                "weighted_log_prob": wlp,
                "top1": t1_acc,
                "expressibility": expr,
                "n_hypotheses": len(scored),
            })

    elapsed = time.time() - t0
    print()
    print(f"Completed in {elapsed:.1f}s")

    # Save results JSON
    results = {"rows": rows, "timing_seconds": round(elapsed, 2)}
    output_path = output_dir / "stage1_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
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
