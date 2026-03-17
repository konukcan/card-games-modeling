"""
Systematic comparison of pipeline variants across three dimensions:
  1. Grammar: uniform vs weighted (4-tier production costs)
  2. Prior mode: summed (log-sum-exp across expressions) vs canonical (single cheapest)
  3. Injection: with vs without LLM-generated hypotheses (~401 entries)

Produces a comparison report with:
  - True rule recovery: mean rank, median rank, rank-1 count, rank-in-top-10 count
  - Shallow dominance: how many rules have a shallow/vacuous top-1 hypothesis
  - Posterior concentration: mean entropy, mean top-1 probability
  - Per-rule rank deltas between variants
  - Group-level breakdowns (Easy/Medium/Hard)

Usage:
    cd src
    python -m gallery_analysis.compare_variants \
        --results-dir gallery_analysis/results/ \
        [--output gallery_analysis/results/comparison_report.json] \
        [--html gallery_analysis/results/reports/comparison.html]
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── Known shallow/vacuous program fragments ──
# Programs containing these fragments are considered "shallow" confusers.
SHALLOW_FRAGMENTS = [
    "has_suit", "has_color",
    "(λ lt 1 (n_unique_suits",
    "(λ lt 1 (n_unique_colors",
    "(λ gt (n_unique_suits",
    "(λ gt (n_unique_colors",
    "(λ le 1 (n_unique",
]


def _is_shallow(program: str) -> bool:
    """Check if a program is a shallow/vacuous hypothesis."""
    return any(frag in program for frag in SHALLOW_FRAGMENTS)


def load_variant(path: Path) -> Optional[Dict[str, Any]]:
    """Load a results JSON, returning None if not found."""
    if not path.exists():
        print(f"  Warning: {path.name} not found, skipping", flush=True)
        return None
    with open(path) as f:
        return json.load(f)


def extract_metrics(results: Dict[str, Any], label: str) -> Dict[str, Any]:
    """
    Extract comparison metrics from a single variant's results.

    Returns a dict with aggregate metrics and per-rule details.
    """
    config = results.get("config", {})
    ranking = results.get("difficulty_ranking", [])
    rule_details = results.get("rule_details", {})

    # ── Aggregate metrics ──
    true_ranks = []
    true_masses = []
    entropies = []
    top1_probs = []
    shallow_top1_count = 0
    rank1_count = 0
    top10_count = 0

    # Group-level tracking
    group_metrics = defaultdict(lambda: {
        "true_ranks": [], "entropies": [], "top1_probs": [], "count": 0
    })

    per_rule = {}

    for r in ranking:
        rule_id = r["rule_id"]
        group = r.get("group", 0)
        entropy = r["posterior_entropy"]
        top1_prob = r["top1_probability"]

        entropies.append(entropy)
        top1_probs.append(top1_prob)

        group_metrics[group]["entropies"].append(entropy)
        group_metrics[group]["top1_probs"].append(top1_prob)
        group_metrics[group]["count"] += 1

        # True rule tracking
        true_rank = r.get("true_rule_rank")
        true_mass = r.get("true_rule_posterior_mass")

        if true_rank is not None:
            true_ranks.append(true_rank)
            group_metrics[group]["true_ranks"].append(true_rank)
            if true_rank == 1:
                rank1_count += 1
            if true_rank <= 10:
                top10_count += 1
        if true_mass is not None:
            true_masses.append(true_mass)

        # Shallow top-1 check
        detail = rule_details.get(rule_id, {})
        top_hyps = detail.get("top_hypotheses", [])
        top1_prog = top_hyps[0]["program"] if top_hyps else ""
        is_shallow_top1 = _is_shallow(top1_prog)
        if is_shallow_top1:
            shallow_top1_count += 1

        per_rule[rule_id] = {
            "true_rank": true_rank,
            "true_mass": true_mass,
            "entropy": entropy,
            "top1_prob": top1_prob,
            "top1_program": top1_prog[:80],
            "shallow_top1": is_shallow_top1,
            "group": group,
        }

    n = len(ranking)
    n_true = len(true_ranks)

    return {
        "label": label,
        "config": {
            "grammar": config.get("scoring_grammar", "uniform"),
            "prior_mode": config.get("prior_mode", "summed"),
            "has_injection": "injection" in results.get("pipeline_stats", {}),
        },
        "n_rules": n,
        "aggregate": {
            # True rule recovery
            "n_with_true_rank": n_true,
            "mean_true_rank": sum(true_ranks) / n_true if n_true else None,
            "median_true_rank": sorted(true_ranks)[n_true // 2] if n_true else None,
            "rank_1_count": rank1_count,
            "top_10_count": top10_count,
            "mean_true_mass": sum(true_masses) / len(true_masses) if true_masses else None,
            # Posterior concentration
            "mean_entropy": sum(entropies) / n if n else 0,
            "mean_top1_prob": sum(top1_probs) / n if n else 0,
            # Shallow dominance
            "shallow_top1_count": shallow_top1_count,
        },
        "group_summary": {
            g: {
                "count": gm["count"],
                "mean_entropy": sum(gm["entropies"]) / gm["count"] if gm["count"] else 0,
                "mean_top1_prob": sum(gm["top1_probs"]) / gm["count"] if gm["count"] else 0,
                "mean_true_rank": (
                    sum(gm["true_ranks"]) / len(gm["true_ranks"])
                    if gm["true_ranks"] else None
                ),
                "rank_1_count": sum(1 for r in gm["true_ranks"] if r == 1),
            }
            for g, gm in sorted(group_metrics.items())
        },
        "per_rule": per_rule,
    }


def print_comparison(variants: List[Dict[str, Any]]):
    """Print a formatted comparison table across all variants."""
    print("\n" + "=" * 100)
    print("SYSTEMATIC VARIANT COMPARISON")
    print("=" * 100)

    # ── Header row ──
    labels = [v["label"] for v in variants]
    col_w = 18
    header = f"{'Metric':<35}" + "".join(f"{l:>{col_w}}" for l in labels)
    print(f"\n{header}")
    print("─" * (35 + col_w * len(labels)))

    def _row(metric_name: str, values: List, fmt: str = "{}"):
        """Print one row of the comparison table."""
        cells = []
        for v in values:
            if v is None:
                cells.append(f"{'—':>{col_w}}")
            else:
                cells.append(f"{fmt.format(v):>{col_w}}")
        print(f"{metric_name:<35}" + "".join(cells))

    # ── True rule recovery ──
    print("\n  TRUE RULE RECOVERY")
    _row("  Rules with true rank",
         [v["aggregate"]["n_with_true_rank"] for v in variants],
         "{}")
    _row("  Mean true rank",
         [v["aggregate"]["mean_true_rank"] for v in variants],
         "{:.1f}")
    _row("  Median true rank",
         [v["aggregate"]["median_true_rank"] for v in variants],
         "{}")
    _row("  Rank-1 count",
         [v["aggregate"]["rank_1_count"] for v in variants],
         "{}")
    _row("  Top-10 count",
         [v["aggregate"]["top_10_count"] for v in variants],
         "{}")
    _row("  Mean true mass",
         [v["aggregate"]["mean_true_mass"] for v in variants],
         "{:.3f}")

    # ── Posterior concentration ──
    print("\n  POSTERIOR CONCENTRATION")
    _row("  Mean entropy",
         [v["aggregate"]["mean_entropy"] for v in variants],
         "{:.2f}")
    _row("  Mean top-1 probability",
         [v["aggregate"]["mean_top1_prob"] for v in variants],
         "{:.3f}")

    # ── Shallow dominance ──
    print("\n  SHALLOW DOMINANCE")
    _row("  Shallow top-1 count",
         [v["aggregate"]["shallow_top1_count"] for v in variants],
         "{}")

    # ── Group-level ──
    group_labels = {1: "Easy", 2: "Medium", 3: "Hard"}
    for g in [1, 2, 3]:
        label = group_labels[g]
        print(f"\n  GROUP {g} ({label})")
        _row(f"    Mean entropy",
             [v["group_summary"].get(g, {}).get("mean_entropy") for v in variants],
             "{:.2f}")
        _row(f"    Mean true rank",
             [v["group_summary"].get(g, {}).get("mean_true_rank") for v in variants],
             "{:.1f}")
        _row(f"    Rank-1 count",
             [v["group_summary"].get(g, {}).get("rank_1_count") for v in variants],
             "{}")

    # ── Per-rule rank deltas ──
    # Show rules where rank changes most between first and last variant
    if len(variants) >= 2:
        print(f"\n{'─' * 100}")
        print("PER-RULE TRUE RANK CHANGES (largest deltas)")
        print(f"{'─' * 100}")

        first = variants[0]
        last = variants[-1]
        deltas = []
        for rule_id in first["per_rule"]:
            r1 = first["per_rule"][rule_id].get("true_rank")
            r2 = last["per_rule"].get(rule_id, {}).get("true_rank")
            if r1 is not None and r2 is not None:
                deltas.append((rule_id, r1, r2, r2 - r1))

        deltas.sort(key=lambda x: x[3])  # improvements first

        print(f"\n  {'Rule':<30} {first['label']:>12} {last['label']:>12} {'Delta':>8}")
        print(f"  {'─' * 30} {'─' * 12} {'─' * 12} {'─' * 8}")

        # Show top 10 improvements
        for rule_id, r1, r2, delta in deltas[:10]:
            arrow = "↑" if delta < 0 else "↓" if delta > 0 else "="
            print(f"  {rule_id:<30} {r1:>12} {r2:>12} {delta:>+7} {arrow}")

        # Show top 5 regressions
        if any(d[3] > 0 for d in deltas):
            print(f"\n  Worst regressions:")
            for rule_id, r1, r2, delta in reversed(deltas[-5:]):
                if delta > 0:
                    print(f"  {rule_id:<30} {r1:>12} {r2:>12} {delta:>+7} ↓")


def generate_html_report(variants: List[Dict[str, Any]], output_path: Path):
    """Generate an HTML comparison report."""
    labels = [v["label"] for v in variants]
    n_variants = len(variants)

    # Build the HTML table rows
    def _metric_row(name, values, fmt="{}", highlight_best="min"):
        """Generate an HTML table row, highlighting the best value."""
        cells = []
        numeric_vals = [v for v in values if v is not None]
        best = None
        if numeric_vals and highlight_best:
            if highlight_best == "min":
                best = min(numeric_vals)
            elif highlight_best == "max":
                best = max(numeric_vals)

        for v in values:
            if v is None:
                cells.append("<td>—</td>")
            else:
                bold = " style='font-weight:bold; color:#2563eb;'" if v == best else ""
                cells.append(f"<td{bold}>{fmt.format(v)}</td>")
        return f"<tr><td>{name}</td>{''.join(cells)}</tr>"

    header_cells = "".join(f"<th>{l}</th>" for l in labels)

    rows = []
    # True rule recovery
    rows.append("<tr><th colspan='{}' style='text-align:left; padding-top:12px;'>True Rule Recovery</th></tr>".format(n_variants + 1))
    rows.append(_metric_row("Mean true rank",
        [v["aggregate"]["mean_true_rank"] for v in variants], "{:.1f}", "min"))
    rows.append(_metric_row("Median true rank",
        [v["aggregate"]["median_true_rank"] for v in variants], "{}", "min"))
    rows.append(_metric_row("Rank-1 count",
        [v["aggregate"]["rank_1_count"] for v in variants], "{}", "max"))
    rows.append(_metric_row("Top-10 count",
        [v["aggregate"]["top_10_count"] for v in variants], "{}", "max"))
    rows.append(_metric_row("Mean true mass",
        [v["aggregate"]["mean_true_mass"] for v in variants], "{:.3f}", "max"))

    # Posterior concentration
    rows.append("<tr><th colspan='{}' style='text-align:left; padding-top:12px;'>Posterior Concentration</th></tr>".format(n_variants + 1))
    rows.append(_metric_row("Mean entropy",
        [v["aggregate"]["mean_entropy"] for v in variants], "{:.2f}", None))
    rows.append(_metric_row("Mean top-1 prob",
        [v["aggregate"]["mean_top1_prob"] for v in variants], "{:.3f}", "max"))

    # Shallow dominance
    rows.append("<tr><th colspan='{}' style='text-align:left; padding-top:12px;'>Shallow Dominance</th></tr>".format(n_variants + 1))
    rows.append(_metric_row("Shallow top-1 count",
        [v["aggregate"]["shallow_top1_count"] for v in variants], "{}", "min"))

    # Group-level
    group_labels = {1: "Easy", 2: "Medium", 3: "Hard"}
    for g in [1, 2, 3]:
        rows.append("<tr><th colspan='{}' style='text-align:left; padding-top:12px;'>Group {} ({})</th></tr>".format(
            n_variants + 1, g, group_labels[g]))
        rows.append(_metric_row(f"  Mean entropy",
            [v["group_summary"].get(g, {}).get("mean_entropy") for v in variants], "{:.2f}", None))
        rows.append(_metric_row(f"  Mean true rank",
            [v["group_summary"].get(g, {}).get("mean_true_rank") for v in variants], "{:.1f}", "min"))
        rows.append(_metric_row(f"  Rank-1 count",
            [v["group_summary"].get(g, {}).get("rank_1_count") for v in variants], "{}", "max"))

    table_rows = "\n".join(rows)

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Variant Comparison</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1200px; margin: 40px auto; padding: 0 20px; }}
  h1 {{ color: #1e293b; }}
  table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
  th, td {{ padding: 6px 12px; text-align: right; border-bottom: 1px solid #e2e8f0; }}
  th {{ background: #f8fafc; font-weight: 600; }}
  td:first-child, th:first-child {{ text-align: left; }}
  tr:hover {{ background: #f1f5f9; }}
  .config {{ background: #f0fdf4; padding: 12px; border-radius: 8px; margin: 20px 0; font-size: 14px; }}
</style>
</head><body>
<h1>Bayesian Rule Induction — Variant Comparison</h1>
<div class="config">
  <strong>Dimensions:</strong> Grammar (uniform vs weighted) × Prior (summed vs canonical) × Injection (with/without LLM hypotheses)<br>
  <strong>Variants:</strong> {n_variants} | <strong>Rules:</strong> 60
</div>
<table>
<thead><tr><th>Metric</th>{header_cells}</tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
</body></html>"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"\nHTML report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare pipeline variants")
    parser.add_argument("--results-dir", type=Path, default=Path("gallery_analysis/results/"),
                        help="Directory containing results JSON files")
    parser.add_argument("--output", type=Path, default=None,
                        help="Save comparison JSON to this path")
    parser.add_argument("--html", type=Path, default=None,
                        help="Save HTML comparison report to this path")
    args = parser.parse_args()

    d = args.results_dir

    # Define the 8 variants in a logical order:
    # Group by grammar, then prior, then injection
    variant_files = [
        ("Uni+Sum", d / "uniform_depth6_summed_noinject.json"),
        ("Uni+Sum+Inj", d / "uniform_depth6_summed_inject.json"),
        ("Uni+Can", d / "uniform_depth6_canonical_noinject.json"),
        ("Uni+Can+Inj", d / "uniform_depth6_canonical_inject.json"),
        ("Wt+Sum", d / "weighted_depth6_summed_noinject.json"),
        ("Wt+Sum+Inj", d / "weighted_depth6_results.json"),
        ("Wt+Can", d / "weighted_depth6_canonical_noinject.json"),
        ("Wt+Can+Inj", d / "weighted_depth6_canonical_results.json"),
    ]

    print("Loading variants...", flush=True)
    variants = []
    for label, path in variant_files:
        data = load_variant(path)
        if data is not None:
            metrics = extract_metrics(data, label)
            variants.append(metrics)
            print(f"  Loaded {label}: {metrics['n_rules']} rules", flush=True)

    if not variants:
        print("No variants found. Check --results-dir path.", flush=True)
        sys.exit(1)

    print_comparison(variants)

    if args.output:
        # Save JSON (strip per_rule to keep it manageable)
        save_data = []
        for v in variants:
            save_v = {k: v[k] for k in v if k != "per_rule"}
            save_data.append(save_v)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(save_data, f, indent=2, default=str)
        print(f"\nJSON saved to: {args.output}")

    if args.html:
        generate_html_report(variants, args.html)


if __name__ == "__main__":
    main()
