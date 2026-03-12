"""CLI entry point for generating Bayesian rule-induction HTML reports.

Orchestrates the full pipeline: loads results JSON and exemplars, registers
the shared Altair theme, then generates one summary index page and one
detail page per rule.

Usage::

    cd .worktrees/bayesian-rule-induction/src
    python -m gallery_analysis.visualization.generate_reports \\
        --results gallery_analysis/results/depth6_injected.json \\
        --exemplars ../../../card-games/rule-gallery/frozen-exemplars.json \\
        --card-images ../../../card-games/stim/ \\
        --output gallery_analysis/results/reports/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ── Imports with sys.path fallback (same pattern as sibling modules) ──
try:
    from gallery_analysis.visualization.data import load_results
    from gallery_analysis.visualization.cards import load_exemplars
    from gallery_analysis.visualization.report_summary import generate_summary
    from gallery_analysis.visualization.report_rule import generate_rule_page
    from shared.theme import register_theme
except ImportError:
    _this_dir = Path(__file__).resolve().parent
    _src_dir = _this_dir.parent.parent
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))

    from gallery_analysis.visualization.data import load_results
    from gallery_analysis.visualization.cards import load_exemplars
    from gallery_analysis.visualization.report_summary import generate_summary
    from gallery_analysis.visualization.report_rule import generate_rule_page
    from shared.theme import register_theme


def main() -> None:
    """Parse arguments and generate all report pages."""
    parser = argparse.ArgumentParser(
        description="Generate Bayesian rule-induction HTML reports."
    )
    parser.add_argument(
        "--results",
        type=Path,
        required=True,
        help="Path to the results JSON (e.g. gallery_analysis/results/depth6_injected.json)",
    )
    parser.add_argument(
        "--exemplars",
        type=Path,
        required=True,
        help="Path to frozen-exemplars.json",
    )
    parser.add_argument(
        "--card-images",
        type=str,
        required=True,
        help="Relative path from output dir to card image PNGs (e.g. ../../../card-games/stim/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("gallery_analysis/results/reports/"),
        help="Output directory for generated HTML (default: gallery_analysis/results/reports/)",
    )
    args = parser.parse_args()

    # ── Step 1: Register shared Altair theme ──────────────────────────
    register_theme()
    print("Registered shared Altair theme.")

    # ── Step 2: Load data ─────────────────────────────────────────────
    results = load_results(args.results)
    print(f"Loaded results: {len(results.difficulty_df)} rules.")

    exemplars = load_exemplars(args.exemplars)
    print(f"Loaded exemplars: {len(exemplars)} rules.")

    # ── Step 3: Read cards.js from the same directory as this file ────
    cards_js_path = Path(__file__).resolve().parent / "cards.js"
    cards_js = cards_js_path.read_text(encoding="utf-8")
    print(f"Loaded cards.js ({len(cards_js)} chars).")

    # ── Step 4: Generate summary page ─────────────────────────────────
    output_dir = args.output
    summary_path = generate_summary(results, output_dir)
    print(f"Generated summary: {summary_path}")

    # ── Step 5: Build sorted rule list (by entropy, hardest first) ────
    # This ordering is used for prev/next navigation links.
    sorted_rules = (
        results.difficulty_df
        .sort_values("posterior_entropy", ascending=False)["rule_id"]
        .tolist()
    )

    # ── Step 6: Generate per-rule detail pages ────────────────────────
    # Rule pages live in output_dir/rules/, which is one level deeper
    # than the summary page.  Prepend "../" to the card-images path so
    # relative paths resolve correctly from that subdirectory.
    rules_dir = output_dir / "rules"
    rule_card_images = "../" + args.card_images

    n_rules = len(sorted_rules)
    for i, rule_id in enumerate(sorted_rules):
        prev_rule = sorted_rules[i - 1] if i > 0 else None
        next_rule = sorted_rules[i + 1] if i < n_rules - 1 else None

        generate_rule_page(
            rule_id=rule_id,
            results=results,
            exemplars=exemplars,
            card_images_path=rule_card_images,
            cards_js=cards_js,
            output_dir=rules_dir,
            prev_rule=prev_rule,
            next_rule=next_rule,
        )
        print(f"  [{i + 1}/{n_rules}] {rule_id}")

    print(f"\nDone. Generated {n_rules} rule pages + 1 summary in {output_dir}")


if __name__ == "__main__":
    main()
