"""Regenerate viz reports using v3 result files (post-ARIS-fix pipeline)."""
import sys, os
from pathlib import Path
sys.path.insert(0, '.')

from gallery_analysis.visualization.data import (
    load_results, load_depth_decomposition, load_diagnosticity_spectrums,
)
from gallery_analysis.visualization.cards import load_exemplars
from gallery_analysis.visualization.report_summary import generate_summary
from gallery_analysis.visualization.report_rule import generate_rule_page
from shared.theme import register_theme

RESULTS_DIR = Path("gallery_analysis/results")
EXEMPLARS = Path("/Users/cankonuk/Documents/self-explanations-project/card-games/rule-gallery/frozen-exemplars.json")
CARD_IMAGES = Path("/Users/cankonuk/Documents/self-explanations-project/card-games/stim/")
REPORTS_BASE = RESULTS_DIR / "reports_v3"

DIAG_V3 = "v3_diagnosticity.json"
DEPTH_DECOMP = RESULTS_DIR / "depth_decomposition_data.json"

VARIANTS = [
    ("weighted-canonical-inject",    "v3_weighted_canonical_inject.json",    DIAG_V3, "Weighted · Canonical · +Inject"),
    ("weighted-summed-inject",       "v3_weighted_summed_inject.json",       DIAG_V3, "Weighted · Summed · +Inject"),
    ("weighted-canonical-trueonly",  "v3_weighted_canonical_trueonly.json",  None,    "Weighted · Canonical · True Only"),
    ("weighted-summed-trueonly",     "v3_weighted_summed_trueonly.json",     None,    "Weighted · Summed · True Only"),
    ("uniform-canonical-inject",     "v3_uniform_canonical_inject.json",     DIAG_V3, "Uniform · Canonical · +Inject"),
    ("uniform-summed-inject",        "v3_uniform_summed_inject.json",        DIAG_V3, "Uniform · Summed · +Inject"),
    ("uniform-canonical-trueonly",   "v3_uniform_canonical_trueonly.json",   None,    "Uniform · Canonical · True Only"),
    ("uniform-summed-trueonly",      "v3_uniform_summed_trueonly.json",      None,    "Uniform · Summed · True Only"),
    ("weighted-canonical-strict",    "v3_weighted_canonical_strict.json",    None,    "Weighted · Canonical · Strict"),
    ("weighted-summed-strict",       "v3_weighted_summed_strict.json",       None,    "Weighted · Summed · Strict"),
]


def all_variants_list():
    entries = []
    for name, results_file, diag_file, label in VARIANTS:
        if not (RESULTS_DIR / results_file).exists():
            continue
        has_diag = diag_file is not None and (RESULTS_DIR / diag_file).exists()
        entries.append({"name": name, "label": label, "path": f"../{name}/index.html", "has_diag": has_diag})
    return entries


def gen_one(name, results_file, diag_file, label, exemplars, cards_js, all_v):
    results_path = RESULTS_DIR / results_file
    if not results_path.exists():
        print(f"  SKIP {name}: {results_file} missing")
        return False

    output_dir = REPORTS_BASE / name
    results = load_results(results_path)
    depth_results = load_depth_decomposition(DEPTH_DECOMP) if DEPTH_DECOMP.exists() else None
    diag_results = None
    if diag_file and (RESULTS_DIR / diag_file).exists():
        diag_results = load_diagnosticity_spectrums(RESULTS_DIR / diag_file)

    variant_info = {"variant_name": name, "variant_label": label, "all_variants": all_v}
    generate_summary(results, output_dir, depth_results=depth_results,
                     diag_results=diag_results, variant_info=variant_info)

    sorted_rules = (
        results.difficulty_df.sort_values("posterior_entropy", ascending=False)["rule_id"].tolist()
    )
    rules_dir = output_dir / "rules"
    card_images_path = os.path.relpath(CARD_IMAGES.resolve(), rules_dir.resolve())
    n = len(sorted_rules)
    for i, rid in enumerate(sorted_rules):
        prev_rule = sorted_rules[i-1] if i > 0 else None
        next_rule = sorted_rules[i+1] if i < n-1 else None
        generate_rule_page(
            rule_id=rid, results=results, exemplars=exemplars,
            card_images_path=card_images_path, cards_js=cards_js,
            output_dir=rules_dir, prev_rule=prev_rule, next_rule=next_rule,
            diag_results=diag_results,
        )
    print(f"  Done: {name} ({n} rule pages)")
    return True


register_theme()
exemplars = load_exemplars(EXEMPLARS)
cards_js = Path("gallery_analysis/visualization/cards.js").read_text()

print(f"Generating v3 reports to {REPORTS_BASE}")
REPORTS_BASE.mkdir(parents=True, exist_ok=True)

all_v = all_variants_list()
succeeded = 0
for name, results_file, diag_file, label in VARIANTS:
    if gen_one(name, results_file, diag_file, label, exemplars, cards_js, all_v):
        succeeded += 1

print(f"\n{succeeded}/{len(VARIANTS)} variants generated")

# Comparison dashboard
try:
    import gallery_analysis.visualization.report_comparison as rc
    rc.VARIANT_FILES = [
        ("v3_weighted_canonical_inject.json",    "weighted", "canonical", "inject",   "noisy"),
        ("v3_weighted_summed_inject.json",       "weighted", "summed",    "inject",   "noisy"),
        ("v3_weighted_canonical_trueonly.json",  "weighted", "canonical", "trueonly", "noisy"),
        ("v3_weighted_summed_trueonly.json",     "weighted", "summed",    "trueonly", "noisy"),
        ("v3_uniform_canonical_inject.json",     "uniform",  "canonical", "inject",   "noisy"),
        ("v3_uniform_summed_inject.json",        "uniform",  "summed",    "inject",   "noisy"),
        ("v3_uniform_canonical_trueonly.json",   "uniform",  "canonical", "trueonly", "noisy"),
        ("v3_uniform_summed_trueonly.json",      "uniform",  "summed",    "trueonly", "noisy"),
        ("v3_weighted_canonical_strict.json",    "weighted", "canonical", "inject",   "strict"),
        ("v3_weighted_summed_strict.json",       "weighted", "summed",    "inject",   "strict"),
    ]
    rc.generate_comparison_page(str(RESULTS_DIR), str(REPORTS_BASE / "comparison.html"))
    print(f"Comparison: {REPORTS_BASE / 'comparison.html'}")
except Exception as e:
    print(f"Comparison failed: {e}")

# Switcher page
switcher = REPORTS_BASE / "switcher.html"
cards = ""
for name, results_file, diag_file, label in VARIANTS:
    if not (RESULTS_DIR / results_file).exists():
        continue
    cards += f'<a class="card" href="{name}/index.html"><h2>{label}</h2></a>\n'

switcher.write_text(f"""<!DOCTYPE html><html><head><title>v3 Variants</title>
<style>body{{font-family:system-ui;max-width:1200px;margin:0 auto;padding:2rem}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:1rem}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:1.2rem;text-decoration:none;color:inherit;display:block}}
.card:hover{{box-shadow:0 4px 12px rgba(0,0,0,0.1)}}
.card h2{{font-size:1rem;margin:0}}</style></head><body>
<h1>v3 Variants (post-ARIS fixes)</h1>
<p><a href="comparison.html">→ Comparison Dashboard</a></p>
<div class="grid">{cards}</div></body></html>""")
print(f"Switcher: {switcher}")
