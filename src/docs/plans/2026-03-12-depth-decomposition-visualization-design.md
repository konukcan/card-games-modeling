# Depth Decomposition Visualization — Design Sheet

## Context

We have a Bayesian rule induction engine that models an ideal learner reasoning about 60 card game rules. Each rule defines a category of valid 6-card hands. The learner entertains ~4,135 hypotheses (equivalence classes of DSL programs), scores them via:

```
P(hypothesis | data) ∝ P(hypothesis) × P(data | hypothesis)
```

- **Prior P(h)**: Grammar-based cost — shorter/simpler programs have higher prior.
- **Likelihood P(data|h)**: Size principle — hypotheses whose *extension* (set of consistent hands) is smaller get exponentially higher likelihood per exemplar, because they make sharper predictions.
- **Extension size**: How many of the ~20.4M possible 6-card hands satisfy the hypothesis.
- **AST depth**: Max nesting depth of the s-expression program (1–6). Depth is a proxy for structural complexity.

### Key finding: prior dominates, likelihood barely bites

The grammar prior spans **12.6 log units** across depths (d=1 at -8.7 vs d=5 at -21.4), while the size-principle likelihood varies only **~1 log unit** (because mean extension sizes differ by only ~10% across depths). With only 6 exemplars, the likelihood ratio between a 92%-coverage hypothesis and an 81%-coverage hypothesis is just ~0.7 log units — nowhere near enough to overcome the prior advantage of shallow programs.

## Data Source

**File**: `gallery_analysis/results/depth_decomposition_data.json` (~1 MB)

### Structure

```json
{
  "metadata": {
    "total_equivalence_classes": 4135,
    "total_possible_hands": 20358520,
    "max_depth": 6,
    "n_exemplars_per_rule": 6,
    "epsilon": 0.01,
    "depth_population": {"1": 6, "2": 338, "3": 1739, "4": 1352, "5": 473, "6": 207},
    "depth_prior_ranges": {
      "1": {"min": -8.85, "max": -8.60, "mean": -8.76},
      "2": {"min": -21.9, "max": -10.4, "mean": -16.6},
      ...
    }
  },
  "rules": {
    "<rule_id>": {
      "rule_id": "all_red",
      "group": 1,
      "group_label": "Easy",
      "true_rule_rank": 1,
      "true_rule_mass": 0.9904,
      "true_rule_depth": 2,
      "n_exemplars": 6,
      "depth_decomposition": {
        "1": {
          "n_total": 6,
          "n_all_hits": 2,
          "n_any_hits": 6,
          "posterior_mass": 0.001,
          "mean_log_prior_allhit": -8.60,
          "mean_log_lik_allhit": -100.91,
          "mean_ext_size_allhit": 20132133,
          "median_ext_size_allhit": 20132540,
          "min_ext_size_allhit": 20131726,
          "max_ext_size_allhit": 20132540
        },
        "2": { ... },
        ...
      },
      "top_competitors": [
        {
          "rank": 1,
          "program": "(λ not (has_color $0 BLACK))",
          "depth": 2,
          "probability": 0.9904,
          "log_prior": -11.25,
          "log_likelihood": -74.05,
          "n_hits": 6,
          "extension_size": 226793,
          "base_rate": 0.01114,
          "n_expressions": 461,
          "is_true_rule": true,
          "agrees_on_exemplars": [true, true, true, true, true, true],
          "source": "merged"
        },
        ...  // up to 20 competitors
      ]
    }
  }
}
```

### Key fields

| Field | Meaning |
|-------|---------|
| `depth_decomposition.{d}.posterior_mass` | Fraction of posterior at depth d |
| `depth_decomposition.{d}.n_all_hits` | # hypotheses at depth d matching all 6 exemplars |
| `depth_decomposition.{d}.mean_log_prior_allhit` | Avg grammar prior (log) for all-hit hypotheses at depth d |
| `depth_decomposition.{d}.mean_log_lik_allhit` | Avg size-principle likelihood (log) for all-hit hypotheses at depth d |
| `depth_decomposition.{d}.mean_ext_size_allhit` | Avg extension size for all-hit hypotheses at depth d |
| `top_competitors[i].agrees_on_exemplars` | Boolean vector: does competitor i agree with each exemplar? |
| `top_competitors[i].is_true_rule` | Whether this competitor IS the correct rule |

## Visualizations Requested

### 1. Depth-Mass Heatmap (the "depth budget" view)

**What**: A 60×6 heatmap where rows = rules (grouped by difficulty group 1/2/3), columns = depth 1–6, and cell color = fraction of posterior mass at that depth.

**Design**:
- Rows sorted: first by group (Easy/Medium/Hard), then within group by true_rule_rank (ascending)
- Color scale: white (0%) → deep blue (100%), log-scaled since many cells are <1%
- Annotate cells with mass percentage when ≥ 5%
- Left sidebar: thin color bar indicating group (green=Easy, amber=Medium, red=Hard)
- Right sidebar: true_rule_rank and true_rule_mass columns
- Title: "Where does posterior mass live? Depth-stratified analysis (depth 6, 4,135 hypotheses)"

**Source fields**: `rules.{id}.depth_decomposition.{d}.posterior_mass`

### 2. Three-Factor Decomposition Bar Chart (the "why each depth wins" view)

**What**: For each depth level 1–5 (skip 6, negligible), show three grouped bars representing the three factors that determine posterior mass:
1. **Count** (log scale): avg number of all-hit hypotheses at that depth
2. **Prior** (log scale): mean log-prior at that depth
3. **Likelihood** (log scale): mean log-likelihood at that depth

Do this three times — one panel per difficulty group — so the reader can see how the decomposition differs across Easy/Medium/Hard.

**Design**:
- 3 panels (columns): Easy, Medium, Hard
- X-axis: depth 1–5
- Y-axis: log-scale values
- Three grouped bars per depth, colored differently: Count (blue), Prior (orange), Likelihood (green)
- Key insight annotation: arrow or text showing "12.6 log unit span" on prior vs "~1 log unit span" on likelihood
- Title: "Posterior mass decomposition: Count × Prior × Likelihood by depth"

**Source fields**: Average across rules in each group of `mean_log_prior_allhit`, `mean_log_lik_allhit`, `n_all_hits` from `depth_decomposition.{d}`

### 3. Prior vs Likelihood Tradeoff Scatter (the "size principle isn't biting" view)

**What**: One scatter plot per group. Each dot = one (rule, depth) combination. X-axis = mean log-prior at that depth, Y-axis = mean log-likelihood at that depth. Dot size = posterior mass. Color = depth level (d=1 red, d=2 orange, d=3 yellow, d=4 green, d=5 blue).

**Design**:
- 3 panels (Easy / Medium / Hard)
- Annotate the diagonal where prior + likelihood = constant (iso-posterior lines)
- The key insight: dots are spread widely on the x-axis (prior varies a lot) but tightly banded on the y-axis (likelihood barely varies) — the "pancake" shape demonstrates prior dominance
- Add text annotation: "Prior spans 12.6 log units" (horizontal arrow) and "Likelihood spans ~1 log unit" (vertical arrow)
- Title: "Prior vs Likelihood tradeoff across depths"

**Source fields**: For each rule × depth combination: `mean_log_prior_allhit`, `mean_log_lik_allhit`, `posterior_mass`

### 4. Extension Size Distribution by Depth (why likelihood doesn't discriminate)

**What**: Violin or box plots showing the distribution of extension sizes for all-hit hypotheses at each depth level.

**Design**:
- X-axis: depth 1–5
- Y-axis: extension size (as fraction of 20.4M total hands, i.e., base rate)
- Show that the distributions overlap heavily — deeper programs don't have much tighter extensions
- Add horizontal line at the grand mean
- Add annotation: "Only ~10% difference between d=1 (92%) and d=5 (81%)"
- Reference line showing what extension size would be needed to overcome the prior penalty (the "break-even" extension)
- Title: "Extension sizes by depth — why the size principle can't overcome the prior"

**Source fields**: `mean_ext_size_allhit`, `median_ext_size_allhit`, `min_ext_size_allhit`, `max_ext_size_allhit` from `depth_decomposition.{d}` (or recompute from raw if needed)

### 5. Confusion Profile Cards (per-rule detail view)

**What**: For each rule, a compact "card" showing:
- Rule name, group, true_rank, true_mass
- Top 10 competitors as rows with: rank, program text, depth, probability, extension size, per-exemplar agreement string (YYYYYY / YYYYYn etc.)
- Highlight the true rule row in green
- Mark injected/merged sources differently

**Design**:
- One card per rule, arranged in a grid or scrollable list
- Sorted by group then by true_rule_rank
- Color-code the agreement string: Y=green, n=red
- Maybe show only for a subset of "interesting" rules: (a) all top-10 true rank rules, (b) a sample of mid-rank, (c) a sample of bottom-rank
- Title: "What does the learner think each rule is?"

**Source fields**: `top_competitors` array per rule

### 6. Depth-6 Anatomy (why depth 6 is all noise)

**What**: A focused panel showing:
- All 207 depth-6 programs categorized by their outermost primitive (has_color: 52, has_suit: 104, not: 18, lt/gt: 26, other: 7)
- Example programs at depth 6 to illustrate the "deeply wrapped shallow predicate" pattern
- Mass contribution of depth 6 per rule (bar chart or dot strip — all near zero)

**Key insight to communicate**: Depth 6 programs are almost entirely `has_color/has_suit` wrapped in 5 layers of list transformations (`first_half`, `reverse`, `sort_by_rank`, etc.). They are semantically equivalent to "does some subset contain some color/suit" — the nesting adds no new discriminative power, just prior penalty. This is why depth 7 would be futile.

**Source fields**: The population counts in metadata, plus a static table of example programs (provided below)

**Static content for this panel**:
```
Depth-6 program categories (207 total):
  has_suit: 104 (50%) — "Does [some reordered subset] contain [suit]?"
  has_color: 52 (25%) — "Does [some reordered subset] contain [color]?"
  not:       18 (9%)  — Negations of has_color/has_suit on subsets
  lt/gt:     26 (13%) — suit_to_int comparisons on specific card positions
  other:      7 (3%)  — and/or/all/any/eq combinations

Examples:
  (λ has_color (first_half (reverse (first_half (sort_by_rank (reverse $0))))) RED)
  (λ has_suit (second_half (sort_by_rank (reverse (second_half (sort_by_rank $0))))) CLUBS)
  (λ lt (suit_to_int (get_suit (head (second_half (sort_by_rank $0))))) 3)

These are "deeply wrapped shallow predicates" — 5 layers of list manipulation
wrapping the same simple {has_color, has_suit} checks available at depth 1.
```

### 7. "What Would Help" Summary Panel

**What**: A visual summary comparing the leverage of three potential improvements:

| Intervention | Mechanism | Expected impact |
|-------------|-----------|-----------------|
| More exemplars (6→12) | Likelihood ratio scales as (ext_ratio)^n; doubling n squares the discrimination | **High** — would kill overly-general shallow hypotheses |
| Depth 7 | Adds ~3-4 log units prior penalty, ~0 new discriminative programs | **Negligible** — 0% mass at d=6 already |
| New primitives | Could express structural rules (adjacency, ordering) at lower depth | **High** — but requires domain-specific engineering |

Visualize as a simple ranked bar chart or matrix comparing "cost" vs "expected gain" for each intervention.

## Group Average Reference Values

For use in annotations and summary statistics:

### Group averages by depth (all-hit hypotheses)

| | d=1 | d=2 | d=3 | d=4 | d=5 |
|---|---|---|---|---|---|
| **Group 1 (Easy)** |
| Avg mass | 5.6% | 50.0% | 23.2% | 12.3% | 8.8% |
| Avg N_allhit | 3.6 | 92.8 | 390.4 | 245.9 | 66.8 |
| Mean log-prior | -8.72 | -16.37 | -19.64 | -20.28 | -21.32 |
| Mean log-lik | -100.45 | -99.40 | -99.46 | -99.14 | -98.98 |
| Mean ext size | 18,699,098 | 17,097,756 | 16,775,050 | 16,147,177 | 15,794,627 |
| **Group 2 (Medium)** |
| Avg mass | 16.3% | 62.3% | 15.8% | 3.9% | 1.7% |
| Avg N_allhit | 2.6 | 82.2 | 332.6 | 202.6 | 59.8 |
| Mean log-prior | -8.70 | -16.41 | -19.67 | -20.19 | -21.33 |
| Mean log-lik | -100.51 | -99.75 | -99.88 | -99.76 | -99.59 |
| Mean ext size | 18,905,231 | 17,609,549 | 17,478,870 | 17,090,720 | 16,762,219 |
| **Group 3 (Hard)** |
| Avg mass | 37.6% | 38.1% | 14.9% | 8.8% | 0.5% |
| Avg N_allhit | 2.6 | 72.1 | 312.1 | 195.7 | 52.8 |
| Mean log-prior | -8.70 | -16.42 | -19.61 | -20.21 | -21.39 |
| Mean log-lik | -100.51 | -100.14 | -100.08 | -99.82 | -99.76 |
| Mean ext size | 18,883,853 | 18,345,230 | 17,906,555 | 17,255,588 | 17,053,922 |

### Grand summary

| Depth | Pop | Avg N_allhit | Mean LogPrior | Mean LogLik | Mean Ext | Avg Mass |
|-------|-----|-------------|---------------|-------------|----------|----------|
| d=1 | 6 | 3.0 | -8.70 | -100.49 | 18,828,367 | 19.9% |
| d=2 | 338 | 82.4 | -16.40 | -99.77 | 17,684,178 | 50.1% |
| d=3 | 1,739 | 345.0 | -19.64 | -99.80 | 17,386,825 | 18.0% |
| d=4 | 1,352 | 214.7 | -20.23 | -99.57 | 16,831,162 | 8.4% |
| d=5 | 473 | 59.8 | -21.35 | -99.44 | 16,536,923 | 3.7% |
| d=6 | 207 | ~7.4 | -23.5 | ~-100.5 | ~18,500,000 | <0.01% |

### Approximate log-decomposition (per-hypothesis)

```
log U(d) ≈ log(N_allhit) + mean_log_prior + mean_log_lik

d=1:  1.1 + (-8.7)  + (-100.5) = -108.1  (best)
d=2:  4.4 + (-16.4) + (-99.8)  = -111.8  (40× worse)
d=3:  5.8 + (-19.6) + (-99.8)  = -113.6  (240× worse)
d=4:  5.4 + (-20.2) + (-99.6)  = -114.4  (570× worse)
d=5:  4.1 + (-21.4) + (-99.4)  = -116.7  (6000× worse)
```

## Style Guidance

- Use a consistent color palette: group colors (green/amber/red), depth colors (spectral from warm→cool)
- Prefer clean, publication-quality figures (suitable for a cognitive science paper)
- Label axes clearly; avoid jargon — a reader should understand "grammar prior" and "size principle likelihood" from the axis labels
- Include panel letters (A, B, C...) if combining into a multi-panel figure
- All log values are natural log (not log10)
