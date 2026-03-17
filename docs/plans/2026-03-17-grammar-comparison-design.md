# Grammar Comparison Design: Empirical DSL Primitive Selection

**Created:** 2026-03-17T22:24:31Z
**Branch:** `feat/grammar-comparison`
**Status:** Design complete, pending implementation planning

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

### G7: Minimal
Stripped to orthogonal core:
- **List ops:** `head`, `at`, `map`, `filter`, `all`, `any`, `zip_with`, `length`, `unique`, `reverse`
- **Arithmetic:** `+`, `-`, `mod`, constants 0–5
- **Logic:** `eq`, `lt`, `gt`, `not`, `and`, `or`
- **Card accessors:** `get_suit`, `get_rank`, `get_rank_val`, `get_color`
- **Constants:** `CLUBS`, `DIAMONDS`, `HEARTS`, `SPADES`, `RED`, `BLACK`

Tests the "DreamCoder-minimal" hypothesis — whether a small orthogonal core is sufficient.

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
Phase 1b hypotheses: ~270 judge-verified hypotheses across 60 rules.

Each hypothesis has:
- **Natural language description**
- **Confidence rank** (1–5, where 1 = highest confidence)
- **Confidence level** (HIGH / MEDIUM / LOW)
- **Code translations** in multiple formats
- **Judge verdict** (PASS/FAIL with explanation)
- **Source model** (gemini-pro)

### Dependent Variable
The confidence rank (1–5) assigned by the LLM to each hypothesis. A good grammar should assign higher log-probability to higher-confidence hypotheses.

### Usage
All 270 hypotheses used (option A from design discussion). Both correct and incorrect hypotheses are included — incorrect hypotheses reveal the prior's bias toward simpler/more accessible rules, which is exactly what we want to capture.

Sensitivity check: re-run with only correct hypotheses to verify ranking stability.

---

## 7. Evaluation Metrics

Four metrics computed for each grammar × cost structure combination:

1. **Spearman rank correlation:** Between grammar log-probability and LLM confidence rank, per rule. Tests ordinal agreement.

2. **Weighted log-probability:** Assign weights proportional to confidence rank (rank 1 → weight 5, rank 5 → weight 1). Compute weighted sum of log-probabilities. Tests whether high-confidence hypotheses get disproportionately high probability.

3. **Top-1 accuracy:** For each rule, does the grammar's highest-probability hypothesis match the LLM's rank-1 hypothesis? Tests mode agreement.

4. **Expressibility:** Fraction of hypotheses with finite log-probability (i.e., expressible in the grammar at all). Penalizes grammars that can't represent observed hypotheses.

---

## 8. Methodology

### Stage 1+3 (combined): Broad Comparison
Evaluate all 7 grammars × 3 cost structures = **21 configurations**.

For each configuration:
- Translate all ~270 hypotheses into grammar-specific ASTs
- Compute log-probability for each
- Compute all 4 evaluation metrics

Compute cost: microseconds per hypothesis × grammar. Total: < 1 second.

### Stage 2: Fine-Grained Ablations
Around the top-performing grammar(s) from Stage 1:

- **Leave-one-out:** Remove each primitive, recompute. Which removal hurts most?
- **Leave-one-in:** Add each proposed primitive individually. Which addition helps most?
- **Pairwise swaps:** For each (old, new) pair, swap and measure net effect.
- **Greedy forward/backward selection:** Build up from Minimal or strip down from +Both.

**Overfitting mitigation:**
- 70/30 train/test split on hypotheses
- k-fold cross-validation
- Report confidence intervals

### Stage 4: Human Data Validation
When behavioral experiment data arrives:
- Re-run the winning grammar(s) against human hypotheses
- Compare grammar rankings: LLM vs human
- Calibrate tier assignments against human difficulty ratings

---

## 9. Translation Pipeline

### Step 1: Source → Base AST

**Path A (248 hypotheses):** Parse existing DSL s-expressions via the existing s-expression parser in `src/dreamcoder_core/program.py`.

**Path B (~22 remaining):** Build a Python-to-AST converter using Python's `ast` module. Pattern-match common idioms:
- `all(... for x in hand)` → `Application(all, Abstraction(...), hand)`
- `card.suit` → `Application(get_suit, $0)`
- `x in (A, B)` → `Application(or, Application(eq, x, A), Application(eq, x, B))`
- `RANK_VALUES[card.rank]` → `Application(rank_val, $0)`
- `len(set(...))` → `Application(length, Application(unique, ...))`

### Step 2: Base AST → Grammar-Specific ASTs

Mechanical rewriting rules for each grammar. Examples for G2 (Swap-Positional):
- `first_half(hand)` → `slice(0, 3, hand)`
- `second_half(hand)` → `slice(3, 6, hand)`
- `take(n, hand)` → `slice(0, n, hand)`
- `drop(n, hand)` → `slice(n, 6, hand)`
- `all(..., adjacent_pairs(hand))` → `shifted_match(1, ..., hand)`

For G7 (Minimal): decompose compound primitives into core operations:
- `all_same(key_fn, hand)` → `all(λc. eq(key_fn(c), key_fn(head(hand))), hand)`
- `count_suit(hand, S)` → `length(filter(λc. eq(get_suit(c), S), hand))`

### Step 3: Compute Log-Probability

Call `grammar.log_probability(ast)` — tree walk summing production log-probs. O(n) in AST size.

### Verification Protocol

1. **Dual-path check (248 hypotheses):** Parse both s-expression and Python paths. Compute fingerprints on 200-probe set. Must match exactly. Any disagreement flags a parser bug.

2. **Python-only check (~22 hypotheses):** Parse Python → AST, execute AST on 200-probe set, compare fingerprint against original Python lambda. Must match.

3. **Rewriting check (all grammars):** After rewriting Base AST → Grammar-specific AST, re-compute fingerprint. Must match Base AST fingerprint. Rewriting must never change semantics.

4. **Regression suite:** Build a test set of 10–15 hand-written (Python → AST → grammar AST) examples covering all pattern types. Run automatically on any parser/rewriter change.

---

## 10. Isolation Protocol

All work on a dedicated branch: `feat/grammar-comparison`.

### Directory Structure
```
llm/grammar_comparison/          # All new code lives here
├── grammars/                    # Grammar definitions (primitive sets + costs)
│   ├── base.py
│   ├── swap_positional.py
│   ├── swap_distributional.py
│   ├── swap_both.py
│   ├── add_both.py
│   ├── redundant.py
│   └── minimal.py
├── primitives/                  # New primitive implementations (isolated)
│   ├── slice.py
│   ├── shifted_match.py
│   ├── stride.py
│   ├── count_where.py
│   └── sorted_counts.py
├── translation/                 # AST translation pipeline
│   ├── python_to_ast.py         # Python → Base AST converter
│   ├── rewriter.py              # Base AST → Grammar-specific AST
│   └── verification.py          # Fingerprint checks
├── evaluation/                  # Scoring and comparison
│   ├── compute_costs.py         # Log-probability computation
│   ├── metrics.py               # Spearman, weighted log-prob, top-1, expressibility
│   └── ablation.py              # Leave-one-out, greedy selection
├── tests/                       # Verification test suite
│   ├── test_python_parser.py
│   ├── test_rewriting.py
│   └── test_fingerprints.py
└── run_comparison.py            # Main entry point
```

### Rules
- **No modifications** to any existing file in `src/dreamcoder_core/`, `src/rules/`, or `src/experiments/`
- New code **reads** from existing modules (grammar, program, type_system) but never writes to them
- New primitives defined only within `llm/grammar_comparison/primitives/`, not in the main `primitives.py`
- Results written to `llm/results/grammar_comparison/`

---

## 11. Key References

- Piantadosi, Tenenbaum & Goodman (2016). "The Logical Primitives of Thought." *Psychological Review.* https://colala.berkeley.edu/papers/piantadosi2016logical.pdf
- LOTlib3 (Piantadosi). https://github.com/piantado/LOTlib3
- Ellis et al. (2021). "DreamCoder: Bootstrapping Inductive Program Synthesis with Wake-Sleep Library Learning." *PLDI 2021.*
- Balog et al. (2017). "DeepCoder: Learning to Write Programs." *ICLR 2017.*
- Rule et al. (2020). "The Child as Hacker." *Trends in Cognitive Sciences.*
- Rule et al. (2024). "Symbolic metaprogram search." *Nature Communications.*
- Gulwani (2011). "Automating String Processing in Spreadsheets Using Input-Output Examples." *POPL 2011.* (FlashFill)

---

## 12. Open Questions (for implementation planning)

1. **Tier assignments for Tiered cost structure:** Current assignments are judgment-based. Should we treat these as hyperparameters to optimize, or fix them a priori?

2. **LOTlib3-style constant handling for rank values:** Should rank enum values (ACE, TWO, ..., KING) get inverse-square weights mapped to their numeric values (1–13), or should they all be weighted equally as enum constants?

3. **How to handle hypotheses inexpressible in a grammar:** Currently scored as -∞. Should we instead assign a penalty value (e.g., cost of the longest expressible hypothesis + some constant)?

4. **Human data format:** When behavioral data arrives, what format will it be in? NL hypotheses? Classification responses? Both?
