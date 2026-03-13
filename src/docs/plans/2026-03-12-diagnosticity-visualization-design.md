# Hand Diagnosticity Visualization — Design Sheet

## Goal

Integrate the hand diagnosticity analysis results into the existing HTML report visualization pipeline. The diagnosticity data shows how easy/hard it is for the Bayesian ideal learner to classify random candidate hands for each rule, and includes representative hand examples at different confidence levels.

## What Already Exists

### Diagnosticity Data Source

The diagnosticity analysis (`run_diagnosticity.py`) produces a JSON file with this structure:

```json
{
  "config": {
    "n_candidates": 10000,
    "mass_threshold": 0.001,
    "depth": 6,
    "seed": 42,
    "n_equiv_classes": 3741
  },
  "spectrums": {
    "all_red": {
      "rule_id": "all_red",
      "group": 1,
      "n_candidates": 10000,
      "mean_p_accept": 0.00992,
      "std_p_accept": 0.098495,
      "mean_confidence": 0.999758,
      "fraction_high_confidence": 1.0,
      "fraction_ambiguous": 0.0,
      "accuracy": 1.0,
      "p_accept_histogram": {
        "0.0-0.1": 9902, "0.1-0.2": 0, "0.2-0.3": 0,
        "0.3-0.4": 0, "0.4-0.5": 0, "0.5-0.6": 0,
        "0.6-0.7": 0, "0.7-0.8": 0, "0.8-0.9": 0,
        "0.9-1.0": 98
      },
      "easy_accept_hands": [
        {
          "hand": [{"suit": "HEARTS", "rank": "6"}, ...],
          "hand_str": "6♥ J♦ 10♦ J♥ 10♥ K♦",
          "rule_id": "all_red",
          "p_accept": 0.999927,
          "confidence": 0.999854,
          "ground_truth": true,
          "correct_prediction": true,
          "top_hypotheses_votes": [
            {"program": "(λ not (has_color $0 BLACK))", "prob": 0.990373, "accepts_hand": true},
            ...
          ]
        },
        ...  // up to 5 hands
      ],
      "easy_reject_hands": [...],  // up to 5 hands, same structure
      "ambiguous_hands": [...]      // up to 5 hands, same structure
    },
    ...  // one entry per rule
  }
}
```

Each representative hand has:
- `hand`: array of `{suit, rank}` card objects (same format as frozen exemplars)
- `hand_str`: human-readable compact string like "6♥ J♦ 10♦ J♥ 10♥ K♦"
- `p_accept`: posterior predictive probability (0–1)
- `confidence`: `|p_accept - 0.5| × 2` (0–1, higher = more diagnostic)
- `ground_truth`: whether the hand actually satisfies the rule
- `correct_prediction`: whether the model's prediction matches ground truth
- `top_hypotheses_votes`: top 5 hypotheses with their vote on this hand

### Existing Visualization Pipeline

```
gallery_analysis/visualization/
├── data.py              ← Load JSON → BayesianResults dataclass (DataFrames)
├── plots.py             ← Pure Altair chart functions
├── cards.py             ← Card → PNG filename mapping, hands_to_json()
├── cards.js             ← Client-side card renderer (CardRenderer.renderHand())
├── report_summary.py    ← Generate index.html (summary charts + sortable table)
├── report_rule.py       ← Generate rules/{rule_id}.html (cards + charts + table)
├── generate_reports.py  ← CLI orchestrator
└── templates/
    ├── summary.html     ← Jinja2: Vega-Lite charts, sortable JS table
    └── rule_detail.html ← Jinja2: card images, Vega-Lite charts, hypothesis table
```

**Key patterns:**
- **Data layer** (`data.py`): JSON → pandas DataFrames in a `@dataclass`. Each new data source gets its own load function and dataclass.
- **Plots layer** (`plots.py`): Pure functions taking DataFrames, returning Altair chart objects. Charts embedded as Vega-Lite JSON specs via `vegaEmbed()`.
- **Cards layer** (`cards.py` + `cards.js`): `hands_to_json()` adds `image_path` fields. `CardRenderer.renderHand(handData)` creates `<img>` elements from card PNG files.
- **Reports** (`report_rule.py`): Jinja2 template rendering. Charts serialized to JSON and embedded in `<script>` tags. Cards rendered client-side via inlined `cards.js`.
- **CLI** (`generate_reports.py`): Loads all data, registers Altair theme, generates summary + per-rule pages.

### Existing Template Placeholder

The `rule_detail.html` template already has a "Test Hands" panel (lines 208-213) with a "Coming soon" placeholder:

```html
<div class="panel" style="margin-top: 1rem;">
  <h3>Test Hands</h3>
  <div id="test-hands-panel">
    <p style="color: #999; font-size: 0.85rem;">Coming soon</p>
  </div>
</div>
```

This is where the diagnosticity representative hands should go.

## What to Build

### 1. Data Layer: `data.py` — Add `load_diagnosticity_spectrums()`

Add a new function and dataclass to `data.py`:

```python
@dataclass
class DiagnosticityResults:
    """Container for hand diagnosticity spectrum data."""
    spectrum_df: pd.DataFrame
        # One row per rule. Columns: rule_id, group, n_candidates,
        # mean_p_accept, std_p_accept, mean_confidence,
        # fraction_high_confidence, fraction_ambiguous, accuracy
    representative_hands: Dict[str, Dict[str, List[Dict]]]
        # Keyed by rule_id, then by category:
        # {"easy_accept": [...], "easy_reject": [...], "ambiguous": [...]}
        # Each hand dict has: hand (card list), p_accept, confidence,
        # ground_truth, correct_prediction, top_hypotheses_votes
    config: Dict[str, Any]

def load_diagnosticity_spectrums(path: Union[str, Path]) -> DiagnosticityResults:
    """Load diagnosticity JSON into normalized structures."""
    ...
```

The `spectrum_df` feeds the summary charts. The `representative_hands` dict feeds the per-rule card rendering.

### 2. Plots Layer: `plots.py` — Add 2 new chart functions

#### a) `p_accept_histogram(spectrum_row)` — Per-rule detail page

A horizontal bar chart showing the P(accept) distribution across 10 bins. This replaces the current "Diagnosticity" chart position (or sits alongside it).

- X-axis: count of hands in each bin
- Y-axis: P(accept) bin labels ("0.0-0.1", ..., "0.9-1.0")
- Color: gradient from blue (reject) through grey (ambiguous) to green (accept)
- Tooltip: bin range, count, percentage of total

#### b) `diagnosticity_overview_scatter(spectrum_df)` — Summary page

A scatter plot of all 60 rules: mean_confidence (x) vs fraction_ambiguous (y), colored by difficulty group. Reveals which rules have the most ambiguous test hands.

- X-axis: mean confidence (0–1)
- Y-axis: fraction ambiguous (0–1)
- Color: difficulty group (Easy/Medium/Hard)
- Size: encode accuracy
- Tooltip: rule_id, all metrics

### 3. Template Changes: `rule_detail.html` — Fill "Test Hands" panel

Replace the "Coming soon" placeholder with rendered diagnostic hands. Three subsections:

```
┌─ Test Hands ──────────────────────────────────────────────────────┐
│                                                                    │
│  Easy ACCEPT (high confidence, model says "yes")                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ [6♥][J♦][10♦][J♥][10♥][K♦]  P=0.999  conf=0.999  truth=✓  │  │
│  │ [9♥][7♦][ 5♥][8♥][ A♥][3♥]  P=1.000  conf=1.000  truth=✓  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Easy REJECT (high confidence, model says "no")                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ [3♠][9♣][3♣][10♠][6♦][4♦]  P=0.000  conf=1.000  truth=✓   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  Ambiguous (hypotheses disagree)                                   │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │ [8♦][6♥][7♦][5♦][2♦][2♣]  P=0.532  conf=0.064  truth=F    │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

Each hand is rendered using the existing `CardRenderer.renderHand()` from `cards.js`, with a metadata line below showing P(accept), confidence, and ground truth status.

**Implementation approach:**
- `report_rule.py` passes `test_hands_json` to the template — same format as `hands_json` but with extra metadata fields per hand
- `cards.js` gets a new function `CardRenderer.renderTestHands(data, container)` that renders the three categories with labels and metadata annotations
- Alternative: render entirely server-side in the Jinja2 template with `<img>` tags (simpler, no JS changes needed)

### 4. Template Changes: `summary.html` — Add diagnosticity columns & chart

#### a) Summary table: 3 new columns

Add to the sortable table: `MeanConf`, `Ambig%`, `Accuracy`

#### b) New chart panel

Add the `diagnosticity_overview_scatter` chart to the summary page, either in the existing chart grid or in a new "Hand Diagnosticity" section.

### 5. CLI Changes: `generate_reports.py` — Add `--diagnosticity` flag

```bash
python -m gallery_analysis.visualization.generate_reports \
    --results gallery_analysis/results/depth6_injected.json \
    --exemplars .../frozen-exemplars.json \
    --card-images .../stim/ \
    --diagnosticity gallery_analysis/results/diagnosticity_all_rules.json \
    --output gallery_analysis/results/reports/
```

The diagnosticity data is optional — if not provided, the "Test Hands" panel stays as "Coming soon" and the summary table omits the diagnosticity columns.

### 6. Report Rule: `report_rule.py` — Wire diagnosticity data

In `generate_rule_page()`:
- Accept optional `diagnosticity_results: DiagnosticityResults`
- Extract representative hands for this rule
- Convert to JSON using `hands_to_json()` (or a variant that includes the metadata)
- Build the P(accept) histogram chart
- Pass `test_hands_json`, `chart_p_accept_hist`, and `has_test_hands` to template

## Data Flow Summary

```
diagnosticity_all_rules.json
         ↓
    data.py:load_diagnosticity_spectrums()
         ↓
DiagnosticityResults:
  - spectrum_df (60 rows × 9 metrics)
  - representative_hands (dict: rule → category → hand list)
         ↓
generate_reports.py (CLI, --diagnosticity flag)
         ↓
    ├── report_summary.py
    │     ├── plots.diagnosticity_overview_scatter(spectrum_df)
    │     └── template: summary.html (new columns + chart)
    │
    └── report_rule.py (for each rule)
          ├── plots.p_accept_histogram(spectrum_row)
          ├── cards.hands_to_json(representative_hands[rule_id])
          └── template: rule_detail.html (test hands panel + histogram)
```

## Integration Constraints

- **Optional data**: The diagnosticity JSON is optional. All template sections must be conditional ({% if has_test_hands %}).
- **Card rendering**: Representative hand cards use the same PNG images and CardRenderer as the exemplar hands. The `image_path` field must use the same relative path computed by `generate_reports.py`.
- **Consistent styling**: Follow the existing `.panel`, `.metric`, badge, and table CSS classes. No new CSS files — inline styles in the template.
- **Altair theme**: Use the registered shared theme (`shared.theme.register_theme()`) for all new charts.

## Files to Modify/Create

| File | Action | What |
|------|--------|------|
| `data.py` | MODIFY | Add `DiagnosticityResults` dataclass + `load_diagnosticity_spectrums()` |
| `plots.py` | MODIFY | Add `p_accept_histogram()` + `diagnosticity_overview_scatter()` |
| `cards.py` | MODIFY (maybe) | Add helper to enrich diagnosticity hands with `image_path` |
| `report_rule.py` | MODIFY | Accept + wire diagnosticity data to template |
| `report_summary.py` | MODIFY | Accept + wire diagnosticity data to template |
| `generate_reports.py` | MODIFY | Add `--diagnosticity` CLI flag, load + pass data |
| `templates/rule_detail.html` | MODIFY | Replace "Coming soon" with test hands + histogram |
| `templates/summary.html` | MODIFY | Add diagnosticity columns to table + new chart |
| `cards.js` | MODIFY (optional) | Add `renderTestHands()` if client-side rendering preferred |

## Sample Results for Reference

From the analysis of 3 rules (10K candidates each):

| Rule | Group | MeanConf | Hi% | Amb% | Acc% | Interpretation |
|------|-------|----------|-----|------|------|----------------|
| all_red | Easy | 1.000 | 100% | 0% | 100% | Perfectly classifiable |
| three_spades | Medium | 0.970 | 100% | 0% | 100% | Very clean separation |
| zigzag_ranks | Hard | 0.688 | 64% | 7% | 8% | Model confused, low accuracy |

The zigzag_ranks result is the most interesting: 34 active hypotheses, 7% ambiguous hands, and only 8% accuracy — the posterior is dominated by overly-permissive hypotheses.

## Existing File for Testing

The sample diagnosticity JSON is at:
`gallery_analysis/results/diagnosticity_sample.json`

It contains spectrums for `all_red`, `three_spades`, and `zigzag_ranks` (3 rules, 10K candidates each). Use this for development and testing. For the full 60-rule version, run:

```bash
cd src
python gallery_analysis/run_diagnosticity.py \
    --all-rules --n-candidates 5000 \
    --extension-cache gallery_analysis/results/extension_cache_depth6.json \
    --inject gallery_analysis/data/injected_hypotheses.json \
    --output gallery_analysis/results/diagnosticity_all_rules.json
```
