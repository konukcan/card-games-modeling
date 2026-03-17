#!/bin/bash
# Generate HTML reports for all result variants.
#
# Each variant gets its own output directory under reports/.
# A top-level switcher page (reports/switcher.html) links them all.
#
# Usage:
#   cd src
#   bash gallery_analysis/visualization/generate_all_variants.sh

set -e

RESULTS_DIR="gallery_analysis/results"
EXEMPLARS="/Users/cankonuk/Documents/self-explanations-project/card-games/rule-gallery/frozen-exemplars.json"
CARD_IMAGES="/Users/cankonuk/Documents/self-explanations-project/card-games/stim/"
REPORTS_BASE="${RESULTS_DIR}/reports"

# Map: short_name -> results_file -> diagnosticity_file (empty = none)
# Diagnosticity files only match weighted+inject variants (same posterior).
declare -A VARIANTS
declare -A DIAG_MAP
declare -A LABELS

VARIANTS[weighted-canonical-inject]="weighted_depth6_canonical_results.json"
DIAG_MAP[weighted-canonical-inject]="diagnosticity_all_rules_weighted_canonical.json"
LABELS[weighted-canonical-inject]="Weighted · Canonical · +Inject"

VARIANTS[weighted-summed-inject]="weighted_depth6_results.json"
DIAG_MAP[weighted-summed-inject]="diagnosticity_all_rules_weighted.json"
LABELS[weighted-summed-inject]="Weighted · Summed · +Inject"

VARIANTS[weighted-canonical-noinject]="weighted_depth6_canonical_noinject.json"
DIAG_MAP[weighted-canonical-noinject]=""
LABELS[weighted-canonical-noinject]="Weighted · Canonical · No Inject"

VARIANTS[weighted-summed-noinject]="weighted_depth6_summed_noinject.json"
DIAG_MAP[weighted-summed-noinject]=""
LABELS[weighted-summed-noinject]="Weighted · Summed · No Inject"

VARIANTS[uniform-canonical-inject]="uniform_depth6_canonical_inject.json"
DIAG_MAP[uniform-canonical-inject]=""
LABELS[uniform-canonical-inject]="Uniform · Canonical · +Inject"

VARIANTS[uniform-summed-inject]="uniform_depth6_summed_inject.json"
DIAG_MAP[uniform-summed-inject]=""
LABELS[uniform-summed-inject]="Uniform · Summed · +Inject"

VARIANTS[uniform-canonical-noinject]="uniform_depth6_canonical_noinject.json"
DIAG_MAP[uniform-canonical-noinject]=""
LABELS[uniform-canonical-noinject]="Uniform · Canonical · No Inject"

VARIANTS[uniform-summed-noinject]="uniform_depth6_summed_noinject.json"
DIAG_MAP[uniform-summed-noinject]=""
LABELS[uniform-summed-noinject]="Uniform · Summed · No Inject"

VARIANTS[weighted-canonical-strict]="weighted_depth6_canonical_strict.json"
DIAG_MAP[weighted-canonical-strict]=""
LABELS[weighted-canonical-strict]="Weighted · Canonical · Strict"

VARIANTS[weighted-summed-strict]="weighted_depth6_summed_strict.json"
DIAG_MAP[weighted-summed-strict]=""
LABELS[weighted-summed-strict]="Weighted · Summed · Strict"

echo "=============================================="
echo "Generating all report variants"
echo "=============================================="

for name in $(echo "${!VARIANTS[@]}" | tr ' ' '\n' | sort); do
    results_file="${RESULTS_DIR}/${VARIANTS[$name]}"
    diag_file="${DIAG_MAP[$name]}"
    output_dir="${REPORTS_BASE}/${name}"
    label="${LABELS[$name]}"

    echo ""
    echo "── ${label} ──"
    echo "   Results: ${VARIANTS[$name]}"

    DIAG_FLAG=""
    if [ -n "$diag_file" ] && [ -f "${RESULTS_DIR}/${diag_file}" ]; then
        DIAG_FLAG="--diagnosticity ${RESULTS_DIR}/${diag_file}"
        echo "   Diagnosticity: ${diag_file}"
    else
        echo "   Diagnosticity: (none)"
    fi

    python -m gallery_analysis.visualization.generate_reports \
        --results "$results_file" \
        --exemplars "$EXEMPLARS" \
        --card-images "$CARD_IMAGES" \
        $DIAG_FLAG \
        --output "$output_dir" 2>&1 | tail -1

done

echo ""
echo "=============================================="
echo "All variants generated under ${REPORTS_BASE}/"
echo "=============================================="
echo ""
echo "Generating switcher page..."

# Generate the switcher HTML using Python (easier for template generation)
python3 -c "
import json
from pathlib import Path

variants = json.loads('''$(python3 -c "
import json
v = {
$(for name in $(echo "${!VARIANTS[@]}" | tr ' ' '\n' | sort); do
    echo "    \"$name\": {\"label\": \"${LABELS[$name]}\", \"has_diag\": $([ -n "${DIAG_MAP[$name]}" ] && echo 'true' || echo 'false')},"
done)
}
print(json.dumps(v))
")''')

html = '''<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>Bayesian Rule Induction — Variant Switcher</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: system-ui, -apple-system, sans-serif; background: #f8f8f8; padding: 2rem; }
    h1 { font-size: 1.5rem; margin-bottom: 0.5rem; }
    p.sub { color: #666; font-size: 0.9rem; margin-bottom: 1.5rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1rem; }
    .card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 1.2rem;
            transition: box-shadow 0.15s; cursor: pointer; text-decoration: none; color: inherit; }
    .card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); border-color: #4A90D9; }
    .card h2 { font-size: 1.05rem; margin-bottom: 0.4rem; }
    .card .tags { display: flex; gap: 6px; flex-wrap: wrap; }
    .tag { font-size: 0.72rem; padding: 2px 8px; border-radius: 12px; }
    .tag-scoring { background: #E8F0FE; color: #1A73E8; }
    .tag-prior { background: #FFF3E0; color: #E65100; }
    .tag-inject { background: #E8F5E9; color: #2E7D32; }
    .tag-noinject { background: #FCE4EC; color: #C62828; }
    .tag-diag { background: #F3E5F5; color: #6A1B9A; }
    .tag-strict { background: #FFF9C4; color: #F57F17; }
    .current { border: 2px solid #4A90D9; }
  </style>
</head>
<body>
  <h1>Bayesian Rule Induction — Report Variants</h1>
  <p class=\"sub\">Click any variant to open its summary page. Each has 60 per-rule detail pages.</p>
  <div class=\"grid\">
'''

for name, info in sorted(variants.items()):
    label = info['label']
    parts = label.split(' · ')
    scoring = parts[0] if len(parts) > 0 else ''
    prior = parts[1] if len(parts) > 1 else ''
    inject_part = parts[2] if len(parts) > 2 else ''

    scoring_tag = f'<span class=\"tag tag-scoring\">{scoring}</span>'
    prior_tag = f'<span class=\"tag tag-prior\">{prior}</span>'

    if 'Strict' in inject_part:
        inject_tag = '<span class=\"tag tag-strict\">Strict</span>'
    elif 'No' in inject_part:
        inject_tag = '<span class=\"tag tag-noinject\">No Inject</span>'
    else:
        inject_tag = '<span class=\"tag tag-inject\">+Inject</span>'

    diag_tag = '<span class=\"tag tag-diag\">+Diagnosticity</span>' if info['has_diag'] else ''

    html += f'''    <a class=\"card\" href=\"{name}/index.html\">
      <h2>{label}</h2>
      <div class=\"tags\">{scoring_tag}{prior_tag}{inject_tag}{diag_tag}</div>
    </a>
'''

html += '''  </div>
</body>
</html>
'''

Path('${REPORTS_BASE}/switcher.html').write_text(html)
print('Switcher page: ${REPORTS_BASE}/switcher.html')
"

echo "Done. Open ${REPORTS_BASE}/switcher.html to browse all variants."
