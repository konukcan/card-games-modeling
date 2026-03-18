# Grammar Comparison Design v2: Empirical DSL Primitive Selection

**Created:** 2026-03-17T22:24:31Z
**Updated:** 2026-03-18
**Branch:** `feat/grammar-comparison`
**Status:** Design revised after initial Stage 1 results and brainstorming review
**Previous version:** `docs/plans/2026-03-17-grammar-comparison-design.md`

---

## 1. Motivation

Phase 0 translation calibration revealed that our current DSL fails to express ~15% of hypotheses in DSL-constrained format, with failures clustering into 4 categories:

1. **Arithmetic constant limits** — constants 0–5 only; rank values 8, 9, 13 require verbose arithmetic
2. **No positional indexing** — can't express sliding windows or position-dependent patterns concisely
3. **No count aggregation** — can't check distribution shapes (e.g., "two suits with 2 cards each")
4. **Combinatorial nesting** — 3-way card relationships explode in curried syntax

Rather than making ad-hoc fixes, we apply the methodology of Piantadosi, Tenenbaum & Goodman (2016) to **empirically test which primitive set best predicts observed hypothesis distributions**, using LLM-generated hypotheses as a proxy for human cognition (with human data validation to follow).

### Theoretical Grounding

Piantadosi et al. (2016) showed that:
- Humans use a **rich, non-minimal** set of logical primitives (AND, OR, NOT, IMPLIES, IFF — not just NAND)
- Different primitive sets, even if computationally equivalent, predict different concept learning difficulties
- The best primitive set can be identified by comparing model predictions against human learning curves
- **PCFG priors** (not flat description length) best capture human complexity judgments
- First-order quantification is used; second-order is not
- Small-cardinality primitives ("exactly 1", "exactly 2") improve fit

We adapt this methodology to our card-game domain.

### What Changed in v2

Initial Stage 1 results (prior-only scoring) revealed that:
- All grammars showed **positive** Spearman correlation (grammar disagrees with LLM) because the prior favors syntactically simple but semantically vague hypotheses
- A **likelihood term** (size principle) is needed to compute proper posteriors
- The AST rewriter was incomplete, conflating "inexpressible" with "rewriter can't handle it"
- The redundant grammar (G6) showed no improvement because hypotheses weren't rewritten to use shortcuts

These issues are addressed in the revised pipeline below.

---

## 2. Goals

### Workstream A: DSL Design
Choose a revised primitive library with cost structure that produces a cognitively realistic prior — one that:
- Assigns appropriate relative difficulty to the 45 catalogue rules
- Generates hypotheses matching what humans (and LLMs) actually produce
- Leaves room for library learning to explain transfer effects

### Workstream B: Empirical Validation
Build a pipeline to test candidate grammars against observed hypothesis data:
- **Now:** LLM hypotheses from Phase 1b (~270 passed, with confidence rankings)
- **Later:** Human hypotheses from the behavioral experiment

---

## 3. Candidate Grammars

Seven grammar families, each representing a coherent theory about cognitive vocabulary:

### G1: Base (control)
Current 57 primitives, unchanged. The baseline to beat.

### G2: Swap-Positional
**Add:** `slice(i, j, hand)`, `shifted_match(k, pred, hand)`
**Remove:** `take`, `drop`, `first_half`, `second_half`, `adjacent_pairs`, `shifted_pairs`

Rationale: General positional primitives replace ad-hoc shortcuts. Net primitive count stays similar. Tests whether general position operations improve fit despite higher per-use cost.

### G3: Swap-Distributional
**Add:** `count_where(pred, hand)`, `sorted_counts(key_fn, hand)`
**Remove:** `count_suit`, `count_rank`, `count_color`

Rationale: General counting/distribution primitives replace suit/rank/color-specific ones. Tests whether generalized counting improves fit.

### G4: Swap-Both
G2 + G3 combined. All new primitives added, all replaced primitives removed.

### G5: +Both (additive)
Add all 5 new primitives (`slice`, `shifted_match`, `stride`, `count_where`, `sorted_counts`). Keep all existing primitives. Tests pure addition without removal — does a larger vocabulary help even with increased branching?

### G6: Redundant (Piantadosi-inspired)
Base + cognitive shortcuts as additional cheap primitives:
- `all_same(key_fn, hand)` — "all cards have same X"
- `all_different(key_fn, hand)` — "all cards have different X"
- `is_sorted(key_fn, hand)` — "cards sorted by X"
- `exactly_n(n, pred, hand)` — "exactly n cards satisfy pred"
- `at_least_n(n, pred, hand)` — "at least n cards satisfy pred"
- `n_unique(key_fn, hand)` — "number of distinct values of X"
- `is_run(hand)` — "cards form consecutive ranks"
- `has_pair(key_fn, hand)` — "at least two cards share X"

Tests Piantadosi's finding that redundant/non-minimal primitives improve cognitive fit.

**Important (v2):** Requires bidirectional rewriting — detect patterns in the AST that match a shortcut and compress them (e.g., `all(λc. eq(get_suit c)(get_suit(head hand)), hand)` → `all_same(get_suit, hand)`). Without this, shortcuts are never used and the grammar shows no benefit.

### G7: Minimal
Stripped to orthogonal core:
- **List ops:** `head`, `at`, `map`, `filter`, `all`, `any`, `zip_with`, `length`, `unique`, `reverse`
- **Arithmetic:** `+`, `-`, `mod`, constants 0–5
- **Logic:** `eq`, `lt`, `gt`, `not`, `and`, `or`
- **Card accessors:** `get_suit`, `get_rank`, `get_rank_val`, `get_color`
- **Constants:** `CLUBS`, `DIAMONDS`, `HEARTS`, `SPADES`, `RED`, `BLACK`

Tests the "DreamCoder-minimal" hypothesis. Uses LLM-assisted translation as fallback for hypotheses the mechanical rewriter can't handle (verified by fingerprints).

---

## 4. New Primitive Definitions

### `slice(i, j, hand)` — Positional window extraction
**Type:** `int → int → [a] → [a]`
**Semantics:** Returns elements from position i (inclusive) to j (exclusive).
**Subsumes:** `take(n, hand) = slice(0, n, hand)`, `drop(n, hand) = slice(n, 6, hand)`, `first_half = slice(0, 3)`, `second_half = slice(3, 6)`

### `shifted_match(k, pred, hand)` — Pairwise check at offset k
**Type:** `int → (a → a → bool) → [a] → bool`
**Semantics:** For all valid positions i, checks `pred(hand[i], hand[i+k])`. Returns True iff all pairs match.
**Subsumes:** `adjacent_pairs` + `all` patterns (when k=1), `zip_with` + `take`/`drop` + `all` compositions

### `stride(k, hand)` — Every k-th element
**Type:** `int → [a] → [a]`
**Semantics:** Returns elements at positions 0, k, 2k, 3k, ...
**Enables:** Even/odd position rules (currently inexpressible)

### `count_where(pred, hand)` — Count satisfying a predicate
**Type:** `(a → bool) → [a] → int`
**Semantics:** Count elements where pred returns True.
**Subsumes:** `count_suit`, `count_rank`, `count_color` (all are special cases)

### `sorted_counts(key_fn, hand)` — Sorted frequency distribution
**Type:** `(a → b) → [a] → [int]`
**Semantics:** Group cards by key_fn, count each group, sort counts descending.
**Example:** `sorted_counts(get_suit, [S,S,S,H,H,D])` → `[3, 2, 1]`
**Enables:** Distribution-shape rules ("two suits with 2 cards each" = `sorted_counts(get_suit) == [2,2,1,1]`)

---

## 5. Cost Structures

Three cost variants tested in parallel for each grammar:

### Uniform
Every primitive has `log_prob = log(1/N)` where N = number of primitives returning that type. Pure description length. Serves as baseline.

### Tiered (3-tier)
Primitives assigned to cognitive accessibility tiers. Within each type, Tier 1 primitives get 3× the probability of Tier 2, which get 3× Tier 3.

- **Tier 1 (most natural):** Card accessors (`get_suit`, `get_color`, `get_rank_val`), basic comparisons (`eq`), quantifiers (`all`, `any`), suit/color constants, small integers (0–3)
- **Tier 2 (common but deliberate):** `map`, `filter`, `length`, `head`, `last`, `count_where`, `lt`, `gt`, `not`, `and`, `or`, `+`, `-`, integers 4–5
- **Tier 3 (specialized):** `at`, `zip_with`, `slice`, `shifted_match`, `sorted_counts`, `mod`, `unique`, `reverse`, integers 6+

### LOTlib3-style (Piantadosi's implementation)
Two features combined:

1. **Inverse-square weighting for numeric constants:** `weight(n) = 10/n²`. Makes small numbers cheap (1 → 10.0, 2 → 2.5, 3 → 1.11) and large numbers expensive (10 → 0.1, 13 → 0.059). Captures the cognitive accessibility gradient for numbers.

2. **Terminal upweighting (5×):** All terminal productions (constants, leaf primitives) get 5× weight multiplier. Biases grammar toward shorter, shallower programs. Prevents deep recursive expansions from dominating.

---

## 6. Data

### Source
Phase 1b hypotheses: ~271 judge-verified hypotheses across 60 rules.

Each hypothesis has:
- **Natural language description**
- **Confidence rank** (1–5, where 1 = highest confidence)
- **Confidence level** (HIGH / MEDIUM / LOW)
- **Code translations** in multiple formats
- **Judge verdict** (PASS/FAIL with explanation)
- **Source model** (gemini-pro)
- **6 frozen exemplar hands** that the LLM was shown when generating hypotheses

### Dependent Variable
The confidence rank (1–5) assigned by the LLM to each hypothesis. A good grammar should assign higher posterior probability to higher-confidence hypotheses.

### Usage
All 271 hypotheses used. Both correct and incorrect hypotheses are included — incorrect hypotheses reveal the prior's bias toward simpler/more accessible rules. The goal is descriptive (reverse-engineer the LLM's implicit "language of thought"), not normative (find the "right" answer).

---

## 7. Scoring Pipeline (v2)

**This is the core of the revised design.** Each hypothesis receives a posterior score combining a grammar-specific prior with a grammar-independent likelihood.

### Step 1: GET THE PROGRAM
Parse the hypothesis into a Program AST via s-expression parser (Path A, 252 hypotheses) or Python-to-AST converter (Path B, remaining 19 hypotheses). Rewrite AST for the target grammar.

**Critical requirement (v2):** The rewriter must NEVER raise InexpressibleError. All grammars are computationally universal over this domain. If a primitive is not in the target grammar, decompose it into available primitives. For G7 (Minimal), use LLM-assisted translation as fallback, verified by fingerprints.

### Step 2: COMPUTE THE PRIOR (grammar-specific)
```
prior(h, G) = G.program_log_likelihood(AST, HAND → BOOL)
```
The PCFG log-probability of the program under grammar G. Varies across grammars.

### Step 3: COMPUTE THE EXTENSION SIZE (grammar-independent)
Execute the hypothesis on a large random sample of hands. Count how many it classifies as True.
```
extension_size = (hits / n_samples) × C(52,6)
```

**Adaptive sampling (v2):**
- Start with 1,000,000 random 6-card hands (cached once, reused across all grammars)
- If base rate is extreme (<0.1% or >99.9%, i.e., fewer than ~1000 hits or misses), escalate to 10,000,000
- If still extreme, escalate to 100,000,000
- Cache extension sizes by fingerprint to avoid recomputation

Extension size is grammar-independent because the rewriter preserves semantics (verified by fingerprint checks).

### Step 4: COMPUTE THE LIKELIHOOD via size principle (grammar-independent)
The 6 frozen exemplar hands shown to the LLM are the data. All are positive examples (hands satisfying the ground-truth rule).
```
log_likelihood(h) = -n × log(extension_size)
```
where n = 6 (number of exemplar hands).

If h does NOT classify all 6 exemplar hands as True, likelihood = 0 (log = -∞).

Intuition: A vague hypothesis (large extension) makes the 6 specific exemplars unsurprising → weak evidence. A specific hypothesis (small extension) makes seeing those exact 6 exemplars surprising → strong evidence.

### Step 5: COMPUTE THE POSTERIOR SCORE
```
log_posterior(h, G) = prior(h, G) + log_likelihood(h)
```

### Step 6: EVALUATE GRAMMAR FIT
Compare the posterior ranking against the LLM's confidence ranking using the metrics in Section 8.

---

## 8. Evaluation Metrics (v2)

Six metrics computed for each grammar × cost structure combination:

### Primary (descriptive — does the grammar predict the LLM's full ranking?)

1. **Spearman agreement** (negated ρ, so higher = better): Per-rule Spearman correlation between grammar posterior and LLM rank, negated and averaged across rules. Tests ordinal agreement.

2. **Weighted log-probability:** Sum of `(6 - rank) × log_posterior` across all hypotheses. Weights high-confidence hypotheses more. Now that the rewriter is complete, all grammars score the same number of hypotheses, removing the expressibility confound.

3. **Top-1 accuracy (chance-corrected):** For each rule, does the grammar's highest-posterior hypothesis match the LLM's rank-1? Corrected by `(accuracy - 1/k) / (1 - 1/k)` where k = number of hypotheses for that rule.

4. **Expressibility:** Fraction of hypotheses with finite posterior. Should be 100% for all grammars once the rewriter is complete; serves as a sanity check.

### Supplementary (normative — does the grammar lead to correct inferences?)

5. **Correct-rank:** For each rule, where does the ground-truth hypothesis fall in the grammar's posterior ranking? Lower = better. Averaged across rules.

6. **Rule-difficulty correlation:** Correlation between the grammar's prior cost of the ground-truth program and some measure of rule difficulty (e.g., rule group 1/2/3). Tests whether the grammar's complexity ordering matches the expected difficulty ordering.

---

## 9. Methodology (v2)

### Stage 1: Broad Comparison (cross-validated, no grammar penalty)
Evaluate all 7 grammars × 3 cost structures = **21 configurations**.

For each configuration:
1. Split ~271 hypotheses into k folds (k=5)
2. For each fold: compute posteriors on training set, evaluate all 6 metrics on held-out set
3. Report mean ± std across folds

No grammar complexity penalty. The held-out evaluation naturally penalizes overly complex grammars (they spread prior mass too thinly) without requiring an arbitrary penalty weight. This follows the Piantadosi et al. approach.

### Stage 2: Fine-Grained Ablations (MDL-guided, cross-validated)
Around the top-performing grammar(s) from Stage 1:

1. **MDL generates candidates:** Use `grammar.grammar_description_length()` to identify primitives that cost more to include than they save in program costs → candidates for removal.
2. **Leave-one-out:** For each candidate, remove it and evaluate on held-out data.
3. **Leave-one-in:** Starting from a sparser grammar, add one candidate at a time and evaluate.
4. **Accept/reject:** Only accept changes that improve held-out metrics. MDL guides the search, held-out performance decides.

This ensures MDL and held-out metrics are never directly compared — they play different roles (hypothesis generation vs. hypothesis testing).

### Stage 3: Human Data Validation
When behavioral experiment data arrives:
- Re-run the winning grammar(s) against human hypotheses
- Compare grammar rankings: LLM vs human
- Calibrate tier assignments against human difficulty ratings

### Future Direction: Marginal Likelihood (parked)
A more theoretically principled approach would compute the marginal likelihood:
```
P(data | G) = Σ_h P(data | h) × P(h | G)
```
This automatically penalizes complex grammars via Bayesian Occam's razor. Currently parked because the sum requires enumerating programs, which is computationally expensive at depths > 5–6. May become feasible if enumeration efficiency improves (see separate MCMC brainstorming prompt at `docs/prompts/mcmc-wake-phase-brainstorming.md`).

---

## 10. Translation Pipeline (v2)

### Step 1: Source → Base AST

**Path A (252 hypotheses):** Parse existing DSL s-expressions via the s-expression parser (wraps `parse_program()` with full primitive registry).

**Path B (19 remaining):** Convert Python-freeform code to AST using the Python-to-AST converter (pattern matching with Python `ast` module). These 19 hypotheses had no s-expression match due to text normalization differences in the injection pipeline.

### Step 2: Base AST → Grammar-Specific ASTs

**Bidirectional mechanical rewriting.** The rewriter must handle two directions:

**Expansion (for swap/minimal grammars):** Decompose compound primitives into the target grammar's available primitives.
- `first_half(hand)` → `slice(0, 3, hand)` (swap-positional)
- `count_suit(hand, S)` → `count_where(λc. eq(get_suit c) S, hand)` (swap-distributional)
- `count_suit(hand, S)` → `length(filter(λc. eq(get_suit c) S, hand))` (minimal)
- Every primitive not in the target grammar MUST have a decomposition rule

**Compression (for redundant grammar):** Detect patterns that match cognitive shortcuts and replace them.
- `all(λc. eq(key c)(key(head hand)), hand)` → `all_same(key, hand)`
- `eq(length(unique(map(key, hand))))(length(hand))` → `all_different(key, hand)`
- etc. for all 8 shortcuts

**Fallback for Minimal (G7):** When the mechanical rewriter can't decompose a pattern, use LLM-assisted translation, verified by fingerprint matching on the 200-probe set.

**The rewriter must NEVER raise InexpressibleError.** All grammars are computationally universal.

### Step 3: Compute Posterior

See Section 7 (Scoring Pipeline).

### Verification Protocol

1. **Dual-path check (252 hypotheses):** Parse both s-expression and Python paths. Compute fingerprints on 200-probe set. Must match exactly.

2. **Python-only check (19 hypotheses):** Parse Python → AST, execute on probes, compare fingerprint against original Python lambda.

3. **Rewriting check (all grammars):** After rewriting Base AST → Grammar-specific AST, re-compute fingerprint. Must match Base AST fingerprint.

4. **Regression suite:** 10–15 hand-written examples covering all pattern types.

---

## 11. Extension Size Estimation (v2 — new section)

Extension size is critical for the size principle likelihood. It must be estimated accurately, especially for rules with very small or very large base rates.

### Methodology
- **Reference population:** Random 6-card hands drawn uniformly from C(52,6)
- **Adaptive sampling:**
  - Level 1: 1,000,000 samples (default)
  - Level 2: 10,000,000 samples (if base rate < 0.1% or > 99.9% at Level 1)
  - Level 3: 100,000,000 samples (if still extreme at Level 2)
- **Caching:** Cache by fingerprint. Extension size is grammar-independent, so compute once and reuse across all 21 configurations.
- **Consistency with Bayesian model:** The existing `estimate_extension_size()` in `src/gallery_analysis/hypothesis_table.py` uses 100,000 samples. We use 10× more because we have far fewer hypotheses to estimate (hundreds vs. thousands).

### Exemplar Hands
The 6 frozen exemplar hands per rule (from the `hands_shown` field in Phase 1b JSONs) are the **data** for the likelihood computation. A hypothesis must classify all 6 as True to receive a non-zero likelihood.

---

## 12. Isolation Protocol

All work on a dedicated branch: `feat/grammar-comparison`.

### Directory Structure
```
llm/grammar_comparison/          # All new code lives here
├── grammars/                    # Grammar definitions (primitive sets + costs)
│   └── grammar_factory.py       # build_grammar(name, cost) → Grammar
├── primitives/                  # New primitive implementations (isolated)
│   └── definitions.py           # slice, shifted_match, stride, count_where, sorted_counts
├── translation/                 # AST translation pipeline
│   ├── sexpr_parser.py          # S-expression → Base AST
│   ├── python_parser.py         # Python → Base AST
│   ├── rewriter.py              # Base AST → Grammar-specific AST (bidirectional)
│   └── verification.py          # Fingerprint checks
├── evaluation/                  # Scoring and comparison
│   ├── compute_costs.py         # Prior + likelihood → posterior scoring
│   ├── metrics.py               # 6 evaluation metrics
│   ├── ablation.py              # Leave-one-out, leave-one-in, cross-validation
│   └── extension.py             # Adaptive extension size estimation
├── tests/                       # Verification test suite
├── docs/                        # Pipeline documentation
│   └── pipeline_guide.md        # Full walkthrough of the evaluation pipeline
└── run_comparison.py            # Main entry point
```

### Rules
- **No modifications** to any existing file in `src/dreamcoder_core/`, `src/rules/`, or `src/experiments/`
- New code **reads** from existing modules (grammar, program, type_system) but never writes to them
- New primitives defined only within `llm/grammar_comparison/primitives/`, not in the main `primitives.py`
- Results written to `llm/results/grammar_comparison/`

---

## 13. Documentation Requirements (v2 — new section)

A detailed pipeline guide (`llm/grammar_comparison/docs/pipeline_guide.md`) must be written covering:

1. **End-to-end pipeline overview** — the 6-step scoring pipeline from Section 7, with a worked example for one hypothesis
2. **What each metric measures** — formal definitions, how to interpret values, what "good" looks like
3. **The transcription chain** — how hypotheses go from NL → s-expression → AST → grammar-specific AST → score, with data flow diagrams
4. **The size principle likelihood** — what it is, why it's needed, how extension sizes are estimated, the adaptive sampling protocol
5. **How to interpret results** — what the Stage 1 table means, how to read ablation output, what constitutes a "significant" difference
6. **How to add a new grammar** — step-by-step instructions for defining a new grammar family, including rewriting rules

---

## 14. Key References

- Piantadosi, Tenenbaum & Goodman (2016). "The Logical Primitives of Thought." *Psychological Review.* https://colala.berkeley.edu/papers/piantadosi2016logical.pdf
- LOTlib3 (Piantadosi). https://github.com/piantado/LOTlib3
- Ellis et al. (2021). "DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning." *PLDI 2021.*
- Tenenbaum & Griffiths (2001). "Generalization, similarity, and Bayesian inference." *Behavioral and Brain Sciences.* (Size principle)
- Balog et al. (2017). "DeepCoder: Learning to Write Programs." *ICLR 2017.*
- Rule et al. (2020). "The Child as Hacker." *Trends in Cognitive Sciences.*
- Rule et al. (2024). "Symbolic metaprogram search." *Nature Communications.*
- Gulwani (2011). "Automating String Processing in Spreadsheets Using Input-Output Examples." *POPL 2011.* (FlashFill)

---

## 15. Resolved Design Decisions

| # | Decision | Resolution | Rationale |
|---|---|---|---|
| 1 | Prior only vs. posterior | Posterior (prior + size principle likelihood) | Prior alone favors vague hypotheses; need likelihood to match LLM's evidence-conditioned confidence |
| 2 | Likelihood model | Size principle: P(data\|h) ∝ 1/\|ext(h)\|^n | Grammar-independent; consistent with Bayesian model in separate analysis |
| 3 | Exemplar hands | Use the 6 frozen hands per rule from Phase 1b `hands_shown` field | These are the actual data the LLM conditioned on |
| 4 | Extension sampling | Adaptive: 1M → 10M → 100M for extreme base rates | Rare rules need more samples for stable estimates |
| 5 | Missing 19 hypotheses | Recover via Python-to-AST converter (Path B) | 7% data loss is avoidable |
| 6 | Rewriter completeness | Must never give up; all grammars are universal | Previous InexpressibleErrors measured rewriter bugs, not grammar limits |
| 7 | Redundant grammar rewriting | Add shortcut detection (compression direction) | Without it, shortcuts are never used and G6 shows no effect |
| 8 | Minimal grammar fallback | LLM-assisted translation, verified by fingerprints | Complete mechanical decomposition is too expensive to engineer |
| 9 | Spearman sign | Negate and label "agreement" (higher = better) | Consistent with other metrics where higher = better |
| 10 | Top-1 correction | Chance-corrected: (acc - 1/k) / (1 - 1/k) | Rules with different numbers of hypotheses have different baselines |
| 11 | Weighted log-prob confound | Fixed by making rewriter complete (all grammars score all hypotheses) | Previously confounded with expressibility |
| 12 | Correct-rank metric | Add as supplementary normative metric | Tests whether grammar leads to correct inferences, complementary to descriptive metrics |
| 13 | Rule-difficulty metric | Add: correlate grammar cost of ground-truth with rule group | Tests whether complexity ordering matches difficulty ordering |
| 14 | Descriptive vs. normative | Spearman primary (descriptive), correct-rank supplementary (normative) | Descriptive captures the full distribution including errors; normative checks accuracy |
| 15 | Grammar penalty | Stage 1: none (cross-validation). Stage 2: MDL as guide, validated by held-out | Avoids arbitrary penalty weight; MDL generates candidates, held-out decides |
| 16 | Marginal likelihood | Parked for future | Requires expensive enumeration; may revisit when enumeration improves |
| 17 | Data: all vs. correct only | All hypotheses (including errors) | Errors are informative — they reveal the prior's biases |
