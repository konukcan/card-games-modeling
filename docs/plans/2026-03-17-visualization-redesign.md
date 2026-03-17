# Visualization Pipeline Redesign

## Context

The Bayesian rule induction gallery visualization has a summary page (60 rules) and 60 per-rule detail pages. After an initial build-out, a systematic review revealed design improvements, redundancies, and a critical fingerprint resolution bug. This document captures all agreed changes.

---

## Part 1: Fingerprint Resolution Fix

### Problem

500 random probes are insufficient to distinguish highly restrictive rules. 16 true rules (e.g. `all_even`, `ranks_palindrome`, `triple_2s_pos234`) all accept 0/500 probes and collapse into a single equivalence class with 2,113 programs. This corrupts true-rule tracking, posterior mass attribution, and the strict-mode analysis.

### Strategy (three parts)

**1a. Resolution curve diagnostic**
- Generate probe sets at N = 100, 200, 500, 1000, 2000, 5000, 10000, 20000
- For each N, fingerprint all 79,011 surviving programs (pre-dedup) and count distinct equivalence classes
- Also track: how many of the 60 true rules are uniquely resolved at each N
- Plot N_probes vs N_equiv_classes and N_probes vs N_true_rules_resolved
- This does NOT require re-enumeration — only re-fingerprinting

**1b. Targeted probes**
- Include the 360 exemplar hands (6 per rule x 60 rules) in the probe set
- These guarantee that each true rule has a distinct fingerprint (the true rule for `all_even` accepts its own exemplars but not those of `ranks_palindrome`, etc.)
- Can be combined with random probes: e.g. 360 targeted + N random

**1c. Collision characterization**
- Analyze which program properties predict fingerprint collisions: extension size, AST depth, predicate type
- Programs with very small extensions (low base rate) are collision-prone
- This informs whether the hypothesis space itself is distorted (beyond just true-rule tracking)

### Impact scope
- Re-fingerprinting and re-scoring: does NOT require re-enumeration
- The enumeration pipeline yields raw programs; fingerprinting happens after
- All downstream results (scoring, diagnosticity, visualization) need regeneration after the fix

---

## Part 2: Summary Page Redesign

### Current state (10 charts + 1 table)

| # | Element | Decision |
|---|---------|----------|
| 1 | `difficulty_strip` (entropy dot plot) | **REMOVE** — merged into heatmap table |
| 2 | `difficulty_scatter` (entropy vs top-1%) | **KEEP + MODIFY** — change to entropy vs true rule mass, log y-axis |
| 3 | `true_rule_recovery` (dot plot of mass per rule) | **REMOVE** — merged into heatmap table |
| 4 | `equiv_class_bars` (all-hits count per rule) | **REMOVE** — merged into heatmap table |
| 5 | `depth_population` (equiv classes per depth) | **REMOVE** |
| 6 | `depth_prior_range` (log-prior range per depth) | **REMOVE** |
| 7 | `depth_vs_difficulty` (true depth vs entropy) | **REMOVE** |
| 8 | `depth_posterior_heatmap` (rules x depths) | **KEEP** |
| 9 | `diagnosticity_overview_scatter` (confidence vs ambiguous) | **REMOVE** — circular (confidence and ambiguity are near-inverses) |
| 10 | `confusability_vs_entropy` (entropy vs fraction ambiguous) | **REPLACE** — with entropy vs accuracy scatter |
| — | Rule index table | **REMOVE** — merged into heatmap table |

### New summary page layout

1. **Variant dropdown** (top bar) — switch between all 10 result variants inline
2. **Stats bar** — enumeration counts, timing
3. **Heatmap table** (new) — replaces charts 1, 3, 4 and the old index table
4. **Entropy vs true rule mass scatter** — log y-axis, floor at smallest nonzero value
5. **Depth posterior heatmap** — rules x depths, log-color posterior mass
6. **Entropy vs accuracy scatter** — posterior entropy (x) vs weighted-vote accuracy (y)
7. **Calibration plot** — binned P(accept) vs observed true acceptance rate, one curve per difficulty group (Easy/Medium/Hard)

### Heatmap table specification

Single sortable table replacing multiple charts and the old index table.

| Column | Type | Heatmap colored? |
|--------|------|-----------------|
| Rule (link to detail page) | string | no |
| Group (difficulty badge) | badge | no |
| Answer (full text) | string | no |
| Posterior Entropy | number | yes |
| True Rule Posterior Mass | number | yes |
| N All-Hits (equiv class count) | number | yes |
| True Rule Rank | number | no |
| True Rule Depth | number | no |

- Sortable by clicking any column header
- Heatmap cells colored by relative magnitude within each column
- Default sort: by posterior entropy descending (hardest first)

### Entropy vs true rule mass scatter

- X-axis: posterior entropy
- Y-axis: true rule posterior mass, **log scale**
- Floor at smallest nonzero value; rules with mass=0 excluded
- Sized by N_eff, colored by difficulty group
- Tooltip: rule_id, entropy, true rule mass, true rule rank, N_eff

### Entropy vs accuracy scatter

- X-axis: posterior entropy
- Y-axis: weighted-vote accuracy (fraction of test hands correctly classified by posterior majority)
- Colored by difficulty group
- Replaces the confusability_vs_entropy chart

### Calibration plot

- X-axis: binned P(accept) (e.g. 10 bins from 0-1)
- Y-axis: observed fraction actually accepted by the true rule within each bin
- Three curves: Easy (blue), Medium (amber), Hard (red)
- Diagonal line = perfect calibration
- Pooled across all rules within each difficulty group

---

## Part 3: Per-Rule Page Redesign

### Current state (7 elements)

| # | Element | Decision |
|---|---------|----------|
| 1 | Metrics box | **MODIFY** — remove N_eff (redundant with entropy) |
| 2 | `posterior_bars` | **REPLACE** — combined posterior decomposition chart |
| 3 | `prior_vs_likelihood` | **REPLACE** — absorbed into combined chart |
| 4 | `diagnosticity_bars` | **REMOVE** — always 1.0, not informative |
| 5 | `p_accept_histogram` | **REPLACE** — combined histogram + rug strip |
| 6 | Card hands (exemplars + test) | **KEEP** — minor aesthetic polish |
| 7 | Hypotheses table | **KEEP** — add true rule row always visible |

### New per-rule page layout

**Metrics box**: Entropy, Top-1%, True Rule Rank, True Rule Posterior

**Combined posterior decomposition chart** (replaces posterior_bars + prior_vs_likelihood):
- Horizontal stacked bars sorted by posterior mass (highest at top)
- Bar total length = posterior probability
- Each bar split into two colored segments: prior contribution (blue-toned) and likelihood contribution (amber-toned)
- Prior share = |log_prior| / (|log_prior| + |log_likelihood|), likelihood share = complement
- True rule bar gets green border/outline
- Y-axis labels: natural language translations of DSL programs (rule-based translator)
- Aesthetic: Option A — gradient fill with subtle grid, clean sans-serif labels
- Minimum bar width so small posteriors remain visible; exact probability printed at bar end

**DSL-to-English translator**:
- Rule-based recursive translator walking the S-expression parse tree
- ~20 primitives to cover (all, any, ge, le, eq, count_color, rank_val, has_suit, etc.)
- Fallback to raw DSL if translation fails
- Cached by program string (many programs repeat across rules)

**Combined P(accept) chart** (replaces p_accept_histogram + diagnosticity_bars):
Two side-by-side panels, one per sampling method:
- **Uniform panel**: 10,000 random hands (natural base rate)
- **Balanced panel**: 500 accept + 500 reject (forced 50/50)

Each panel contains:
- **Top**: Histogram bars (P(accept) bins), each bar split green (true accept) / red (true reject) by ground truth — shows distribution shape + error direction
- **Bottom strip** (~30px): Rug/separation plot — all hands sorted by P(accept), each a thin vertical stripe colored green/red by ground truth — shows separation quality at a glance

**Card hands panel**: Minor aesthetic polish:
- Tighter card spacing (3-4px gap)
- Subtle shadow on card images instead of flat grey background
- Colored pill badges for test hand categories
- Slightly smaller/lighter metrics text below test hands

**Hypotheses table**: Kept as-is with one addition:
- True rule always shown — at its natural rank position if in top 10, or appended as a final row with actual rank (e.g. "Rank 3771") highlighted in green

---

## Part 4: Infrastructure

### Variant generation
- `generate_all_variants.py` produces 10 variant directories + switcher page
- Each summary page has an inline dropdown to switch variants
- Depth decomposition data loader needs updating to match current JSON format

### Depth decomposition data format reconciliation
- Current `depth_mass_analysis.py` outputs: `{provenance, rules: {rule_id: {group, depth_mass, cumulative_mass, ...}}}`
- `load_depth_decomposition()` in data.py expects: `{metadata: {depth_population, depth_prior_ranges}, rules: {...}}`
- Need to reconcile — either update the loader or the analysis script output

### Balanced test hand generation
- Add a `--balanced` sampling mode to `run_diagnosticity.py`
- For each rule, sample N hands that the true rule accepts + N hands it rejects
- Store alongside the existing uniform sample in the diagnosticity JSON
