"""End-to-end test for the Bayesian report generation pipeline.

Generates a summary page and one rule detail page using real data,
verifies the output structure and content, then cleans up.

Usage::

    cd .worktrees/bayesian-rule-induction/src
    python3 gallery_analysis/visualization/test_e2e.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

# ── sys.path setup (same pattern as sibling modules) ──────────────────
_this_dir = Path(__file__).resolve().parent
_src_dir = _this_dir.parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from gallery_analysis.visualization.data import load_results
from gallery_analysis.visualization.cards import load_exemplars
from gallery_analysis.visualization.report_summary import generate_summary
from gallery_analysis.visualization.report_rule import generate_rule_page
from shared.theme import register_theme


# ── Paths ─────────────────────────────────────────────────────────────

# Results JSON lives inside the worktree's src/ directory.
_RESULTS_PATH = _src_dir / "gallery_analysis" / "results" / "depth6_injected.json"

# Exemplars are in the sibling card-games repo.
# From the worktree root: ../../../card-games/rule-gallery/frozen-exemplars.json
# The worktree is at: card-games-modelling/.worktrees/bayesian-rule-induction/
# So from _src_dir (which is src/), we go up to the worktree root, then
# up three more levels to self-explanations-project/.
_PROJECT_ROOT = _src_dir.parent.parent.parent.parent
_EXEMPLARS_PATH = _PROJECT_ROOT / "card-games" / "rule-gallery" / "frozen-exemplars.json"

# Temporary output directory (cleaned up after test).
_TEST_OUTPUT = _src_dir / "gallery_analysis" / "results" / "_test_reports"


def _cleanup() -> None:
    """Remove the test output directory if it exists."""
    if _TEST_OUTPUT.exists():
        shutil.rmtree(_TEST_OUTPUT)


def test_e2e() -> None:
    """Run the end-to-end report generation test."""
    _cleanup()
    passed = 0
    failed = 0

    try:
        # ── Setup ─────────────────────────────────────────────────────
        register_theme()

        print(f"Results path: {_RESULTS_PATH}")
        print(f"Exemplars path: {_EXEMPLARS_PATH}")
        assert _RESULTS_PATH.exists(), f"Results not found: {_RESULTS_PATH}"
        assert _EXEMPLARS_PATH.exists(), f"Exemplars not found: {_EXEMPLARS_PATH}"

        results = load_results(_RESULTS_PATH)
        exemplars = load_exemplars(_EXEMPLARS_PATH)

        # Read cards.js from the visualization directory.
        cards_js_path = _this_dir / "cards.js"
        cards_js = cards_js_path.read_text(encoding="utf-8")

        # ── Test 1: Summary page ──────────────────────────────────────
        print("\n[Test 1] Generating summary page...")
        summary_path = generate_summary(results, _TEST_OUTPUT)

        assert summary_path.exists(), "index.html was not created"
        html = summary_path.read_text(encoding="utf-8")
        assert "Bayesian Rule Induction" in html, "Missing title in index.html"
        assert "vegaEmbed" in html, "Missing vegaEmbed in index.html"
        print("  PASS: index.html exists, contains expected content.")
        passed += 1

        # ── Test 2: Rule detail page ──────────────────────────────────
        # Pick the first rule from the results.
        first_rule = results.difficulty_df["rule_id"].iloc[0]
        print(f"\n[Test 2] Generating detail page for rule: {first_rule}")

        rules_dir = _TEST_OUTPUT / "rules"
        rule_path = generate_rule_page(
            rule_id=first_rule,
            results=results,
            exemplars=exemplars,
            card_images_path="../../stim",
            cards_js=cards_js,
            output_dir=rules_dir,
            prev_rule=None,
            next_rule=None,
        )

        assert rule_path.exists(), f"{first_rule}.html was not created"
        rule_html = rule_path.read_text(encoding="utf-8")
        assert first_rule in rule_html, f"Missing rule_id '{first_rule}' in detail page"
        assert "CardRenderer" in rule_html, "Missing CardRenderer in detail page"
        assert "vegaEmbed" in rule_html, "Missing vegaEmbed in detail page"
        print("  PASS: rule detail page exists, contains expected content.")
        passed += 1

        # ── Test 3: File structure ────────────────────────────────────
        print("\n[Test 3] Verifying file structure...")
        assert (_TEST_OUTPUT / "index.html").exists(), "index.html missing from output"
        assert (_TEST_OUTPUT / "rules").is_dir(), "rules/ subdirectory missing"
        assert rule_path.exists(), "Rule HTML missing from rules/"
        print("  PASS: file structure is correct (index.html + rules/).")
        passed += 1

    except Exception as e:
        print(f"\n  FAIL: {e}")
        failed += 1
        raise

    finally:
        # ── Cleanup ───────────────────────────────────────────────────
        _cleanup()
        print(f"\nCleaned up {_TEST_OUTPUT}")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("All tests passed.")
    else:
        print("SOME TESTS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    test_e2e()
