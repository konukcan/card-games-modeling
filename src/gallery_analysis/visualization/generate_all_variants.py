"""Generate HTML reports for all result variants + a switcher page.

Each variant gets its own output directory under reports/.
Each summary page has a dropdown to switch between variants.
A top-level switcher page (reports/switcher.html) also links them all.

Usage:
    cd src
    python gallery_analysis/visualization/generate_all_variants.py
"""

import json
import os
import sys
from pathlib import Path

# Ensure src/ is on sys.path for imports.
_src_dir = Path(__file__).resolve().parent.parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from gallery_analysis.visualization.data import (
    load_results,
    load_depth_decomposition,
    load_diagnosticity_spectrums,
)
from gallery_analysis.visualization.cards import load_exemplars
from gallery_analysis.visualization.report_summary import generate_summary
from gallery_analysis.visualization.report_rule import generate_rule_page
from shared.theme import register_theme

RESULTS_DIR = Path("gallery_analysis/results")
EXEMPLARS = Path("/Users/cankonuk/Documents/self-explanations-project/card-games/rule-gallery/frozen-exemplars.json")
CARD_IMAGES = Path("/Users/cankonuk/Documents/self-explanations-project/card-games/stim/")
REPORTS_BASE = RESULTS_DIR / "reports"
DEPTH_DECOMP = RESULTS_DIR / "depth_decomposition_data.json"

# Each variant: (short_name, results_file, diagnosticity_file_or_None, label)
VARIANTS = [
    ("weighted-canonical-inject",   "weighted_depth6_canonical_results.json",  "diagnosticity_all_rules_weighted_canonical.json", "Weighted · Canonical · +Inject"),
    ("weighted-summed-inject",      "weighted_depth6_results.json",            "diagnosticity_all_rules_weighted.json",           "Weighted · Summed · +Inject"),
    ("weighted-canonical-noinject", "weighted_depth6_canonical_noinject.json",  None,                                              "Weighted · Canonical · No Inject"),
    ("weighted-summed-noinject",    "weighted_depth6_summed_noinject.json",     None,                                              "Weighted · Summed · No Inject"),
    ("uniform-canonical-inject",    "uniform_depth6_canonical_inject.json",     None,                                              "Uniform · Canonical · +Inject"),
    ("uniform-summed-inject",       "uniform_depth6_summed_inject.json",        None,                                              "Uniform · Summed · +Inject"),
    ("uniform-canonical-noinject",  "uniform_depth6_canonical_noinject.json",   None,                                              "Uniform · Canonical · No Inject"),
    ("uniform-summed-noinject",     "uniform_depth6_summed_noinject.json",      None,                                              "Uniform · Summed · No Inject"),
    ("weighted-canonical-strict",   "weighted_depth6_canonical_strict.json",    None,                                              "Weighted · Canonical · Strict"),
    ("weighted-summed-strict",      "weighted_depth6_summed_strict.json",       None,                                              "Weighted · Summed · Strict"),
]


def _build_all_variants_list() -> list[dict]:
    """Build the list of all variants for the dropdown switcher.

    Each entry has keys: name, label, path (relative URL from any
    variant's index.html to the other variant's index.html), has_diag.
    """
    entries = []
    for name, _, diag_file, label in VARIANTS:
        results_path = RESULTS_DIR / _
        if not results_path.exists():
            continue
        has_diag = diag_file is not None and (RESULTS_DIR / diag_file).exists()
        # Path from <variant>/index.html to <other_variant>/index.html
        # is ../<other_variant>/index.html
        entries.append({
            "name": name,
            "label": label,
            "path": f"../{name}/index.html",
            "has_diag": has_diag,
        })
    return entries


def generate_variant(
    name: str,
    results_file: str,
    diag_file: str | None,
    label: str,
    exemplars: dict,
    cards_js: str,
    all_variants_list: list[dict],
) -> bool:
    """Generate summary + 60 rule pages for one variant. Returns True on success."""
    results_path = RESULTS_DIR / results_file
    output_dir = REPORTS_BASE / name

    if not results_path.exists():
        print(f"  SKIP (missing: {results_file})")
        return False

    # Load results
    results = load_results(results_path)

    # Load depth decomposition (optional, shared across all variants)
    depth_results = None
    if DEPTH_DECOMP.exists():
        depth_results = load_depth_decomposition(DEPTH_DECOMP)

    # Load diagnosticity (optional)
    diag_results = None
    if diag_file and (RESULTS_DIR / diag_file).exists():
        diag_results = load_diagnosticity_spectrums(RESULTS_DIR / diag_file)
        print(f"   + diagnosticity: {diag_file}")
    else:
        print(f"   (no diagnosticity)")

    # Build variant_info for the dropdown
    variant_info = {
        "variant_name": name,
        "variant_label": label,
        "all_variants": all_variants_list,
    }

    # Generate summary page
    generate_summary(
        results, output_dir,
        depth_results=depth_results,
        diag_results=diag_results,
        variant_info=variant_info,
    )

    # Generate per-rule pages
    sorted_rules = (
        results.difficulty_df
        .sort_values("posterior_entropy", ascending=False)["rule_id"]
        .tolist()
    )

    rules_dir = output_dir / "rules"
    rules_dir_abs = rules_dir.resolve()
    card_images_abs = CARD_IMAGES.resolve()
    rule_card_images = os.path.relpath(card_images_abs, rules_dir_abs)

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
            diag_results=diag_results,
        )

    print(f"   Done: {n_rules} rule pages + 1 summary")
    return True


def generate_switcher(generated: list[tuple[str, str, bool]]) -> None:
    """Create the switcher HTML page listing all generated variants."""
    cards_html = ""
    for name, label, has_diag in generated:
        parts = label.split(" · ")
        scoring = parts[0] if len(parts) > 0 else ""
        prior = parts[1] if len(parts) > 1 else ""
        inject_part = parts[2] if len(parts) > 2 else ""

        scoring_tag = f'<span class="tag tag-scoring">{scoring}</span>'
        prior_tag = f'<span class="tag tag-prior">{prior}</span>'

        if "Strict" in inject_part:
            inject_tag = '<span class="tag tag-strict">Strict</span>'
        elif "No" in inject_part:
            inject_tag = '<span class="tag tag-noinject">No Inject</span>'
        else:
            inject_tag = '<span class="tag tag-inject">+Inject</span>'

        diag_tag = '<span class="tag tag-diag">+Diagnosticity</span>' if has_diag else ""

        cards_html += f'''    <a class="card" href="{name}/index.html">
      <h2>{label}</h2>
      <div class="tags">{scoring_tag}{prior_tag}{inject_tag}{diag_tag}</div>
    </a>
'''

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Bayesian Rule Induction — Variant Switcher</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: system-ui, -apple-system, sans-serif; background: #f8f8f8; padding: 2rem; }}
    h1 {{ font-size: 1.5rem; margin-bottom: 0.5rem; }}
    p.sub {{ color: #666; font-size: 0.9rem; margin-bottom: 1.5rem; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1rem; }}
    .card {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 1.2rem;
            transition: box-shadow 0.15s; cursor: pointer; text-decoration: none; color: inherit; display: block; }}
    .card:hover {{ box-shadow: 0 4px 12px rgba(0,0,0,0.1); border-color: #4A90D9; }}
    .card h2 {{ font-size: 1.05rem; margin-bottom: 0.4rem; }}
    .card .tags {{ display: flex; gap: 6px; flex-wrap: wrap; }}
    .tag {{ font-size: 0.72rem; padding: 2px 8px; border-radius: 12px; }}
    .tag-scoring {{ background: #E8F0FE; color: #1A73E8; }}
    .tag-prior {{ background: #FFF3E0; color: #E65100; }}
    .tag-inject {{ background: #E8F5E9; color: #2E7D32; }}
    .tag-noinject {{ background: #FCE4EC; color: #C62828; }}
    .tag-diag {{ background: #F3E5F5; color: #6A1B9A; }}
    .tag-strict {{ background: #FFF9C4; color: #F57F17; }}
  </style>
</head>
<body>
  <h1>Bayesian Rule Induction — Report Variants</h1>
  <p class="sub">Click any variant to open its summary page. Each contains 60 per-rule detail pages with full navigation. Summary pages also have a dropdown to switch variants inline.</p>
  <div class="grid">
{cards_html}  </div>
</body>
</html>
'''

    switcher_path = REPORTS_BASE / "switcher.html"
    REPORTS_BASE.mkdir(parents=True, exist_ok=True)
    switcher_path.write_text(html, encoding="utf-8")
    print(f"\nSwitcher page: {switcher_path}")


def main() -> None:
    print("=" * 60)
    print("Generating all report variants")
    print("=" * 60)

    # One-time setup
    register_theme()
    print("Registered shared Altair theme.")

    exemplars = load_exemplars(EXEMPLARS)
    print(f"Loaded exemplars: {len(exemplars)} rules.")

    cards_js_path = Path(__file__).resolve().parent / "cards.js"
    cards_js = cards_js_path.read_text(encoding="utf-8")
    print(f"Loaded cards.js ({len(cards_js)} chars).")

    # Build the variant list for dropdowns (only include variants whose
    # results file actually exists).
    all_variants_list = _build_all_variants_list()

    generated: list[tuple[str, str, bool]] = []

    for name, results_file, diag_file, label in VARIANTS:
        print(f"\n── {label} ──")
        print(f"   Results: {results_file}")

        ok = generate_variant(
            name, results_file, diag_file, label,
            exemplars, cards_js, all_variants_list,
        )
        if ok:
            has_diag = diag_file is not None and (RESULTS_DIR / diag_file).exists()
            generated.append((name, label, has_diag))

    print(f"\n{'=' * 60}")
    print(f"Generated {len(generated)}/{len(VARIANTS)} variants")
    print("=" * 60)

    generate_switcher(generated)
    print(f"\nDone. Open {REPORTS_BASE / 'switcher.html'} to browse all variants.")


if __name__ == "__main__":
    main()
