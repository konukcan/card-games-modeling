# Visualization Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Overhaul the Bayesian rule induction gallery visualization to fix fingerprint collisions, consolidate redundant charts, and improve information density and aesthetics across both summary and per-rule pages.

**Architecture:** The pipeline has a clean 3-layer architecture: data layer (data.py) → plot layer (plots.py) → report layer (report_summary.py, report_rule.py, templates). Changes flow top-down: fix data issues first, then update chart functions, then update templates. A new DSL-to-English translator module is added as a utility.

**Tech Stack:** Python 3, pandas, Altair/Vega-Lite, Jinja2, HTML/CSS/JS

## Execution Guidelines

- Present 2+ options for each major decision, wait for selection
- Explain code as you write it
- Test each step before proceeding
- Keep existing structure when editing code
- Start simple, build up

---

### Task 1: Fingerprint Resolution Diagnostic Script

**Files:**
- Create: `src/gallery_analysis/fingerprint_resolution.py`
- Read: `src/gallery_analysis/analyze.py:61-190` (build_hypothesis_pool)
- Read: `src/gallery_analysis/hypothesis_table.py:132-150` (compute_fingerprint)
- Read: `src/gallery_analysis/exemplars.py` (load_exemplars, generate_probe_set)

**Step 1: Write the resolution curve script**

Create `fingerprint_resolution.py` that:
1. Runs enumeration + trivial filter once (reuses `build_hypothesis_pool` internals up to fingerprinting)
2. Generates one large probe set (N=20000, seed=42)
3. For each probe count in [100, 200, 500, 1000, 2000, 5000, 10000, 20000], takes the first N probes, fingerprints all ~79,011 survivors, counts distinct fingerprints
4. Also loads the 42 true-rule injections, fingerprints them at each N, counts how many are uniquely resolved (i.e. no two true rules share a fingerprint)
5. Adds a "targeted" variant: 360 exemplar hands + N random probes, measures same metrics
6. Saves results JSON: `{probe_counts: [...], random: {n_classes: [...], n_true_resolved: [...]}, targeted: {n_classes: [...], n_true_resolved: [...]}}`
7. Prints a summary table

Key implementation detail: to avoid re-running enumeration for each probe count, enumerate once, store all 79,011 `(prog_str, pred_fn, log_prior)` tuples, then loop over probe counts applying `compute_fingerprint` to each predicate.

**Step 2: Run it**

```bash
cd src
python gallery_analysis/fingerprint_resolution.py --output gallery_analysis/results/fingerprint_resolution.json --verbose 1
```

Expected: ~10-20 minutes (fingerprinting 79k programs × 8 probe counts).

**Step 3: Commit**

```bash
git add src/gallery_analysis/fingerprint_resolution.py gallery_analysis/results/fingerprint_resolution.json
git commit -m "feat: add fingerprint resolution diagnostic script"
```

---

### Task 2: Targeted Probe Integration

**Files:**
- Modify: `src/gallery_analysis/analyze.py:61-190` (build_hypothesis_pool — add `extra_probes` parameter)
- Modify: `src/gallery_analysis/hypothesis_table.py:132-150` (no changes needed — fingerprint already takes any probe list)
- Modify: `src/gallery_analysis/exemplars.py` (add function to get all exemplar hands as probes)

**Step 1: Add exemplar probe extraction**

In `exemplars.py`, add:
```python
def get_exemplar_probes() -> List[Hand]:
    """Return all 360 exemplar hands (6 per rule × 60 rules) for use as targeted probes."""
    exemplars = load_exemplars()
    probes = []
    for rule_id, data in exemplars.items():
        probes.extend(data["hands_primary"])
    return probes
```

**Step 2: Modify build_hypothesis_pool to accept extra probes**

Add `extra_probes: List[Hand] = None` parameter. When provided, prepend them to the random probe set before fingerprinting. Update the probe hash to include both sets.

**Step 3: Run a test**

```bash
cd src
python -c "from gallery_analysis.exemplars import get_exemplar_probes; print(len(get_exemplar_probes()))"
```

Expected: `360`

**Step 4: Commit**

```bash
git add src/gallery_analysis/exemplars.py src/gallery_analysis/analyze.py
git commit -m "feat: support targeted exemplar probes in fingerprinting"
```

---

### Task 3: Re-run Full Pipeline with Fixed Fingerprinting

**Files:**
- Run: `src/gallery_analysis/analyze.py` with `--extra-probes exemplars` flag
- Output: New result files in `gallery_analysis/results/`

**Step 1: Add --extra-probes CLI flag to analyze.py**

In the argparse section of `analyze.py`'s `main()`, add a `--extra-probes` flag that loads exemplar hands and passes them to `build_hypothesis_pool`.

**Step 2: Re-run all needed variants**

```bash
cd src
# Re-run the canonical weighted variant with targeted probes
nohup caffeinate -d -i -s python gallery_analysis/analyze.py \
    --depth 6 --max-programs 500000 \
    --inject gallery_analysis/data/injected_hypotheses.json \
    --extra-probes exemplars \
    --prior-mode canonical --scoring-grammar weighted \
    --output gallery_analysis/results/weighted_depth6_canonical_targeted.json \
    --verbose 1 > /tmp/targeted_run.log 2>&1 &
```

**Step 3: Verify true-rule resolution**

```bash
python -c "
import json
with open('gallery_analysis/results/weighted_depth6_canonical_targeted.json') as f:
    d = json.load(f)
n_found = sum(1 for r in d['rule_details'].values() if r.get('true_rule_posterior_mass', 0) > 0)
print(f'True rules with nonzero mass: {n_found}/60')
"
```

Expected: 60/60 (all true rules uniquely resolved).

**Step 4: Commit**

```bash
git add src/gallery_analysis/analyze.py
git commit -m "feat: add --extra-probes flag for targeted fingerprinting"
```

---

### Task 4: Depth Decomposition Data Loader Fix

**Files:**
- Modify: `src/gallery_analysis/visualization/data.py:267-320` (load_depth_decomposition)

**Step 1: Update the loader to handle the new JSON format**

The current `depth_mass_analysis.py` outputs:
```json
{"provenance": {...}, "rules": {"all_red": {"group": 1, "depth_mass": {"2": 0.99, ...}, ...}}}
```

But `load_depth_decomposition` expects `metadata.depth_population`. Update the loader to:
1. Accept the new format (no `metadata` key)
2. Derive `depth_population_df` by aggregating across rules
3. Build `depth_rule_df` from `rules[rule_id].depth_mass`
4. Build `rule_summary_df` from `rules[rule_id].true_rule_rank`, `.true_rule_mass`

**Step 2: Test the loader**

```bash
cd src
python -c "
from gallery_analysis.visualization.data import load_depth_decomposition
dd = load_depth_decomposition('gallery_analysis/results/depth_decomposition_data.json')
print(f'Rules: {len(dd.rule_summary_df)}, Depths: {len(dd.depth_population_df)}')
print(dd.depth_population_df.head())
"
```

**Step 3: Commit**

```bash
git add src/gallery_analysis/visualization/data.py
git commit -m "fix: update depth decomposition loader for new JSON format"
```

---

### Task 5: DSL-to-English Translator

**Files:**
- Create: `src/gallery_analysis/visualization/dsl_translator.py`

**Step 1: Build the S-expression parser**

Write a simple recursive parser that converts `(λ all (λ eq (mod (rank_val $0) 2) 0) $0)` into an AST of nested tuples.

**Step 2: Build the translation table**

Map each DSL primitive to an English fragment generator:

```python
TRANSLATIONS = {
    "all": lambda args: f"all cards {args[0]}",
    "any": lambda args: f"some card {args[0]}",
    "ge": lambda args: f"{args[0]} >= {args[1]}",
    "le": lambda args: f"{args[0]} <= {args[1]}",
    "eq": lambda args: f"{args[0]} = {args[1]}",
    "gt": lambda args: f"{args[0]} > {args[1]}",
    "lt": lambda args: f"{args[0]} < {args[1]}",
    "and": lambda args: f"{args[0]} and {args[1]}",
    "or": lambda args: f"{args[0]} or {args[1]}",
    "not": lambda args: f"not {args[0]}",
    "rank_val": lambda args: f"rank of {args[0]}",
    "get_suit": lambda args: f"suit of {args[0]}",
    "get_color": lambda args: f"color of {args[0]}",
    "count_color": lambda args: f"count of {args[1]} cards",
    "count_suit": lambda args: f"count of {args[1]} cards",
    "has_suit": lambda args: f"hand has {args[1]}",
    "has_color": lambda args: f"hand has {args[1]}",
    "n_unique_suits": lambda args: f"number of distinct suits",
    "n_unique_ranks": lambda args: f"number of distinct ranks",
    "max_suit_count": lambda args: f"max cards of one suit",
    "mod": lambda args: f"{args[0]} mod {args[1]}",
    "+": lambda args: f"{args[0]}+{args[1]}",
    "head": lambda args: f"first card",
    "last": lambda args: f"last card",
    "at": lambda args: f"card at position {args[1]}",
    "take": lambda args: f"first {args[0]} cards",
    "drop": lambda args: f"cards after position {args[0]}",
    "filter": lambda args: f"cards where {args[0]}",
    "length": lambda args: f"hand size",
    # ... etc for remaining primitives
}
```

**Step 3: Write the recursive translator**

```python
def translate_dsl(program: str) -> str:
    """Translate a DSL S-expression to English. Falls back to raw DSL on failure."""
    try:
        ast = parse_sexpr(program)
        return _translate_node(ast)
    except Exception:
        return program
```

**Step 4: Test on known programs**

```python
assert translate_dsl("(λ all (λ eq (mod (rank_val $0) 2) 0) $0)") == "all cards have even rank"
assert translate_dsl("(λ ge (count_color $0 RED) 3)") == "at least 3 red cards"
assert translate_dsl("(λ not (has_color $0 BLACK))") == "no black cards"
```

These won't be exact string matches — the translator produces natural-sounding English, so test for key phrases rather than exact strings.

**Step 5: Add caching**

```python
_cache: Dict[str, str] = {}

def translate_dsl(program: str) -> str:
    if program in _cache:
        return _cache[program]
    result = _translate_uncached(program)
    _cache[program] = result
    return result
```

**Step 6: Commit**

```bash
git add src/gallery_analysis/visualization/dsl_translator.py
git commit -m "feat: add rule-based DSL-to-English translator"
```

---

### Task 6: Balanced Test Hand Generation

**Files:**
- Modify: `src/gallery_analysis/hand_diagnosticity.py:284-340` (generate_diagnostic_spectrum)
- Modify: `src/gallery_analysis/run_diagnosticity.py`

**Step 1: Add balanced sampling to generate_diagnostic_spectrum**

Add `balanced_n: int = 0` parameter. When > 0, additionally generate N accept + N reject hands using rejection sampling against the ground truth predicate. Store these in a new field `balanced_reports` on the `DiagnosticSpectrum` dataclass.

**Step 2: Update DiagnosticSpectrum dataclass**

Add fields:
```python
balanced_reports: List[DiagnosticityReport] = field(default_factory=list)
balanced_n: int = 0
```

**Step 3: Update run_diagnosticity.py**

Add `--balanced` CLI flag (default 500). Pass to `generate_diagnostic_spectrum`. Serialize balanced reports in the output JSON alongside the uniform ones.

**Step 4: Run for all 60 rules**

```bash
cd src
nohup caffeinate -d -i -s python gallery_analysis/run_diagnosticity.py \
    --all-rules --n-candidates 10000 --balanced 500 \
    --extension-cache gallery_analysis/results/extension_cache_depth6.json \
    --inject gallery_analysis/data/injected_hypotheses.json \
    --output gallery_analysis/results/diagnosticity_all_rules_balanced.json \
    --verbose 1 > /tmp/diag_balanced.log 2>&1 &
```

**Step 5: Commit**

```bash
git add src/gallery_analysis/hand_diagnosticity.py src/gallery_analysis/run_diagnosticity.py
git commit -m "feat: add balanced sampling mode to diagnosticity analysis"
```

---

### Task 7: Summary Page — Heatmap Table

**Files:**
- Modify: `src/gallery_analysis/visualization/templates/summary.html`
- Modify: `src/gallery_analysis/visualization/report_summary.py:78-210`

**Step 1: Remove old charts from report_summary.py**

Remove generation of: `chart_strip`, `chart_recovery`, `chart_equiv`. Remove their imports from plots.py. Keep `chart_scatter` (will be modified in Task 8).

**Step 2: Build heatmap table data in report_summary.py**

Compute column min/max for heatmap coloring. Pass to template as `rules` list (already exists) plus `heatmap_ranges` dict with min/max per column.

**Step 3: Rewrite summary.html table section**

Replace the old `<table id="rule-table">` and the chart-strip/recovery/equiv divs with a single heatmap table. Use inline CSS `background-color` computed from cell value relative to column range. Sortable via existing JS.

Heatmap coloring: use a blue-to-red diverging scale for entropy (blue = low/easy, red = high/hard). Use green-to-red for true rule mass (green = high, red = low). Use the shared difficulty colors for the group badge.

**Step 4: Remove old chart divs and vegaEmbed calls from summary.html**

Remove `#chart-strip`, `#chart-recovery`, `#chart-equiv` and their vegaEmbed calls.

**Step 5: Verify generation**

```bash
cd src
python -m gallery_analysis.visualization.generate_reports \
    --results gallery_analysis/results/weighted_depth6_canonical_results.json \
    --exemplars /path/to/frozen-exemplars.json \
    --card-images /path/to/stim/ \
    --output /tmp/test_reports/
open /tmp/test_reports/index.html
```

**Step 6: Commit**

```bash
git add src/gallery_analysis/visualization/report_summary.py src/gallery_analysis/visualization/templates/summary.html
git commit -m "feat: replace strip/recovery/equiv charts with heatmap table"
```

---

### Task 8: Summary Page — Entropy vs True Rule Mass Scatter (Log Scale)

**Files:**
- Modify: `src/gallery_analysis/visualization/plots.py:100-148` (difficulty_scatter)

**Step 1: Update difficulty_scatter to use log y-axis**

Already uses true_rule_posterior_mass on y-axis. Add:
- `alt.Scale(type="log")` on the y encoding
- Filter out rows where `true_rule_posterior_mass` is 0 or None
- Floor at the smallest nonzero value
- Ensure tooltip shows the true rule mass in scientific notation

**Step 2: Verify**

Generate reports, open summary, confirm log scale renders correctly with data points spread across orders of magnitude.

**Step 3: Commit**

```bash
git add src/gallery_analysis/visualization/plots.py
git commit -m "feat: log scale y-axis for entropy vs true rule mass scatter"
```

---

### Task 9: Summary Page — Entropy vs Accuracy Scatter

**Files:**
- Modify: `src/gallery_analysis/visualization/plots.py` (replace `confusability_vs_entropy` with `entropy_vs_accuracy`)
- Modify: `src/gallery_analysis/visualization/report_summary.py`
- Modify: `src/gallery_analysis/visualization/templates/summary.html`

**Step 1: Write entropy_vs_accuracy chart function**

Replace `confusability_vs_entropy` in plots.py:
```python
def entropy_vs_accuracy(merged_df: pd.DataFrame) -> alt.Chart:
    """Scatter of posterior entropy vs weighted-vote accuracy."""
```
X = posterior_entropy, Y = accuracy, colored by difficulty group.

**Step 2: Update report_summary.py**

Replace `confusability_vs_entropy` call with `entropy_vs_accuracy`. Update the merge logic to use accuracy column from diagnosticity data.

**Step 3: Update summary.html**

Replace `#chart-confusability` div/vegaEmbed with `#chart-accuracy`.

**Step 4: Commit**

```bash
git add src/gallery_analysis/visualization/plots.py src/gallery_analysis/visualization/report_summary.py src/gallery_analysis/visualization/templates/summary.html
git commit -m "feat: replace confusability chart with entropy vs accuracy scatter"
```

---

### Task 10: Summary Page — Calibration Plot

**Files:**
- Modify: `src/gallery_analysis/visualization/plots.py` (add `calibration_plot`)
- Modify: `src/gallery_analysis/visualization/data.py` (add calibration data builder)
- Modify: `src/gallery_analysis/visualization/report_summary.py`
- Modify: `src/gallery_analysis/visualization/templates/summary.html`

**Step 1: Build calibration data in data.py**

Add function `build_calibration_df(diag_results, difficulty_df)` that:
1. For each rule, bins test hands by P(accept) into 10 bins
2. Within each bin, computes fraction where ground truth = accept
3. Tags each row with the rule's difficulty group
4. Aggregates by (bin, group) → mean observed acceptance rate

**Step 2: Write calibration_plot chart function in plots.py**

```python
def calibration_plot(cal_df: pd.DataFrame) -> alt.LayerChart:
    """Calibration curves by difficulty group with diagonal reference line."""
```
Three colored lines (Easy/Medium/Hard) + diagonal reference line.

**Step 3: Wire into report_summary.py and summary.html**

Generate the calibration data, serialize chart spec, add `#chart-calibration` div.

**Step 4: Commit**

```bash
git add src/gallery_analysis/visualization/plots.py src/gallery_analysis/visualization/data.py src/gallery_analysis/visualization/report_summary.py src/gallery_analysis/visualization/templates/summary.html
git commit -m "feat: add calibration plot by difficulty group"
```

---

### Task 11: Summary Page — Clean Up Removed Charts

**Files:**
- Modify: `src/gallery_analysis/visualization/plots.py` (remove dead functions)
- Modify: `src/gallery_analysis/visualization/templates/summary.html` (remove old depth chart divs except heatmap)

**Step 1: Remove unused plot functions**

Remove from plots.py: `difficulty_strip`, `true_rule_recovery`, `equiv_class_bars`, `depth_population`, `depth_vs_difficulty`, `depth_prior_range`, `diagnosticity_overview_scatter`, `confusability_vs_entropy`.

Keep: `difficulty_scatter`, `depth_posterior_heatmap`, `entropy_vs_accuracy`, `calibration_plot`, and all per-rule functions.

**Step 2: Remove old depth chart divs from summary.html**

Remove `#chart-depth-pop`, `#chart-depth-prior`, `#chart-depth-vs-diff` and their vegaEmbed calls. Keep `#chart-depth-heatmap`.

**Step 3: Remove old diagnosticity section from summary.html**

Remove `#chart-diag-overview` div and the explanatory note about exemplar diagnosticity.

**Step 4: Update imports in report_summary.py**

Remove imports for deleted functions.

**Step 5: Verify no broken references**

```bash
cd src
python -m gallery_analysis.visualization.generate_reports \
    --results gallery_analysis/results/weighted_depth6_canonical_results.json \
    --exemplars /path/to/frozen-exemplars.json \
    --card-images /path/to/stim/ \
    --output /tmp/test_cleanup/
open /tmp/test_cleanup/index.html
```

**Step 6: Commit**

```bash
git add src/gallery_analysis/visualization/plots.py src/gallery_analysis/visualization/report_summary.py src/gallery_analysis/visualization/templates/summary.html
git commit -m "chore: remove redundant chart functions and template sections"
```

---

### Task 12: Per-Rule Page — Combined Posterior Decomposition Chart

**Files:**
- Modify: `src/gallery_analysis/visualization/plots.py` (add `posterior_decomposition`)
- Modify: `src/gallery_analysis/visualization/report_rule.py`
- Modify: `src/gallery_analysis/visualization/templates/rule_detail.html`
- Read: `src/gallery_analysis/visualization/dsl_translator.py` (from Task 5)

**Step 1: Write posterior_decomposition chart function**

```python
def posterior_decomposition(hyp_df: pd.DataFrame, rule_id: str) -> alt.Chart:
    """Stacked horizontal bars showing posterior = prior contribution + likelihood contribution."""
```

For each hypothesis:
- `prior_share = |log_prior| / (|log_prior| + |log_likelihood|)`
- `likelihood_share = 1 - prior_share`
- Bar total width = posterior probability
- Left segment (blue-toned) = prior_share × probability
- Right segment (amber-toned) = likelihood_share × probability
- True rule gets green stroke
- Y-axis labels = NL translations from dsl_translator

Implementation: create a long-form DataFrame with two rows per hypothesis (one for prior segment, one for likelihood segment), use `alt.Chart.mark_bar()` with `alt.X` stacked encoding.

**Step 2: Add minimum bar width and probability labels**

Use `alt.condition` or a separate text mark layer to show exact probability at bar end. Set minimum bar width via padding or a floor value.

**Step 3: Update report_rule.py**

Replace `posterior_bars` and `prior_vs_likelihood` calls with single `posterior_decomposition` call. Pass the NL-translated labels.

**Step 4: Update rule_detail.html**

Replace `#chart-posterior` and `#chart-prior-lik` with single `#chart-decomposition` div. Remove old vegaEmbed calls.

**Step 5: Commit**

```bash
git add src/gallery_analysis/visualization/plots.py src/gallery_analysis/visualization/report_rule.py src/gallery_analysis/visualization/templates/rule_detail.html
git commit -m "feat: combined posterior decomposition chart with NL labels"
```

---

### Task 13: Per-Rule Page — Combined P(accept) Chart with Ground Truth

**Files:**
- Modify: `src/gallery_analysis/visualization/plots.py` (add `p_accept_ground_truth`)
- Modify: `src/gallery_analysis/visualization/report_rule.py`
- Modify: `src/gallery_analysis/visualization/templates/rule_detail.html`
- Modify: `src/gallery_analysis/visualization/data.py` (load balanced data)

**Step 1: Update DiagnosticityResults to include balanced data**

Add `balanced_histogram_data` and `balanced_representative_hands` fields. Update `load_diagnosticity_spectrums` to load balanced data when present in JSON.

**Step 2: Write p_accept_ground_truth chart function**

```python
def p_accept_ground_truth(histogram_data: list, rule_id: str) -> alt.LayerChart:
    """Histogram with green/red ground-truth split + rug strip below."""
```

The histogram data needs to include ground_truth breakdown per bin. This requires pre-computing it in the diagnosticity pipeline or in data.py.

Implementation approach:
1. In data.py, add a function that takes raw per-hand reports and bins them by P(accept), splitting each bin by ground_truth → produces `[{bin: "0.0-0.1", true_accept: 45, true_reject: 890}, ...]`
2. The chart function renders stacked bars (green/red) + a rug strip below using `mark_tick` or `mark_rect` with very thin rectangles

**Step 3: Render two side-by-side panels (uniform + balanced)**

In report_rule.py, generate two chart specs (one for uniform, one for balanced sampling). Pass both to the template. Template renders them side-by-side with labels "Uniform Sampling" and "Balanced Sampling".

**Step 4: Update rule_detail.html**

Replace `#chart-p-accept` and `#chart-diag` with `#chart-p-accept-uniform` and `#chart-p-accept-balanced` side by side.

**Step 5: Commit**

```bash
git add src/gallery_analysis/visualization/plots.py src/gallery_analysis/visualization/report_rule.py src/gallery_analysis/visualization/templates/rule_detail.html src/gallery_analysis/visualization/data.py
git commit -m "feat: combined P(accept) histogram with ground-truth split and rug strip"
```

---

### Task 14: Per-Rule Page — Metrics Box and Table Updates

**Files:**
- Modify: `src/gallery_analysis/visualization/templates/rule_detail.html`
- Modify: `src/gallery_analysis/visualization/report_rule.py`

**Step 1: Remove N_eff from metrics box**

In rule_detail.html, remove the N_eff metric div.

**Step 2: Add true rule row to hypotheses table**

In report_rule.py, after building the `hypotheses` list, check if the true rule is already present. If not, append it with its actual rank, highlighted. Pass a `true_rule_hypothesis` dict to the template.

In rule_detail.html, after the `{% for h in hypotheses %}` loop, add a conditional row for the true rule if not already shown:
```html
{% if true_rule_hypothesis and not true_rule_in_top %}
<tr class="true-rule-row">
  <td>{{ true_rule_hypothesis.rank }}</td>
  ...
</tr>
{% endif %}
```

**Step 3: Card hands aesthetic polish**

In `cards.js`, update:
- Card gap from 6px to 4px
- Add `box-shadow: 0 1px 3px rgba(0,0,0,0.12)` to card images
- Remove flat grey background, use white with subtle border
- Smaller metrics text (0.72rem instead of 0.78rem)
- Category labels as colored pills

**Step 4: Commit**

```bash
git add src/gallery_analysis/visualization/templates/rule_detail.html src/gallery_analysis/visualization/report_rule.py src/gallery_analysis/visualization/cards.js
git commit -m "feat: metrics box cleanup, true-rule table row, card aesthetic polish"
```

---

### Task 15: Per-Rule Page — Remove Dead Chart Code

**Files:**
- Modify: `src/gallery_analysis/visualization/plots.py` (remove dead per-rule functions)
- Modify: `src/gallery_analysis/visualization/report_rule.py`

**Step 1: Remove unused plot functions**

Remove from plots.py: `posterior_bars`, `prior_vs_likelihood`, `diagnosticity_bars`, `p_accept_histogram`.

**Step 2: Update imports in report_rule.py**

Remove imports for deleted functions. Update to import new functions: `posterior_decomposition`, `p_accept_ground_truth`.

**Step 3: Commit**

```bash
git add src/gallery_analysis/visualization/plots.py src/gallery_analysis/visualization/report_rule.py
git commit -m "chore: remove replaced per-rule chart functions"
```

---

### Task 16: Full Pipeline Regeneration

**Files:**
- Modify: `src/gallery_analysis/visualization/generate_all_variants.py` (ensure depth data is loaded)
- Run: Full regeneration of all 10 variants

**Step 1: Verify generate_all_variants.py passes depth data**

Already done in the brainstorming session — verify the `DEPTH_DECOMP` path is correct and the loader works with the fixed format from Task 4.

**Step 2: Regenerate all variants**

```bash
cd src
python gallery_analysis/visualization/generate_all_variants.py
```

**Step 3: Visual verification**

Open `gallery_analysis/results/reports/switcher.html` and spot-check:
- Heatmap table renders with colored cells
- Entropy vs true rule mass scatter has log y-axis
- Depth posterior heatmap renders
- Entropy vs accuracy scatter renders
- Calibration plot shows three curves
- Per-rule pages show combined decomposition chart with NL labels
- Per-rule pages show combined P(accept) chart with green/red split
- True rule appears in hypotheses table even when not in top 10
- Variant switcher works

**Step 4: Commit**

```bash
git add -A
git commit -m "feat: complete visualization redesign — all variants regenerated"
```

---

### Task 17: Fingerprint Resolution — Collision Characterization

**Files:**
- Modify: `src/gallery_analysis/fingerprint_resolution.py` (add characterization analysis)

**Step 1: Add collision characterization**

After running the resolution curve, analyze which programs collide:
- Group colliding programs by their shared fingerprint
- For each collision group, record: extension size distribution, AST depth distribution, program types (which primitives appear)
- Identify the top-10 largest collision groups and describe their common properties
- Save characterization to the output JSON

**Step 2: Run and review**

```bash
cd src
python gallery_analysis/fingerprint_resolution.py \
    --output gallery_analysis/results/fingerprint_resolution.json \
    --characterize --verbose 2
```

**Step 3: Commit**

```bash
git add src/gallery_analysis/fingerprint_resolution.py gallery_analysis/results/fingerprint_resolution.json
git commit -m "feat: add collision characterization to fingerprint resolution analysis"
```
