# Grammar Comparison Pipeline Guide

This document explains the end-to-end pipeline for comparing 7 candidate DSL grammars against LLM-generated hypotheses about card game rules. The goal is to find which primitive set best predicts the distribution of hypotheses that an LLM (and eventually humans) produce when trying to explain card game rules.

**Key references:**
- Design document: `docs/plans/2026-03-18-grammar-comparison-design-v2.md`
- Entry point: `llm/grammar_comparison/run_comparison.py`
- Theoretical basis: Piantadosi, Tenenbaum & Goodman (2016), "The Logical Primitives of Thought"

---

## 1. End-to-End Pipeline Overview

Every hypothesis receives a **posterior score** under each grammar. The pipeline has 6 steps:

```
Hypothesis text
    |
    v
[Step 1] Parse -> Program AST (base grammar)
    |
    v
[Step 2] Rewrite AST -> Grammar-specific AST
    |
    v
[Step 3] Compute log_prior = Grammar.program_log_likelihood(AST, HAND -> BOOL)
    |
    v
[Step 4] Compute extension_size via adaptive Monte Carlo (1M -> 100M samples)
    |
    v
[Step 5] Compute log_likelihood = -n * log(extension_size)
    |
    v
[Step 6] log_posterior = log_prior + log_likelihood
```

After scoring all hypotheses, the posterior rankings are compared against the LLM's confidence rankings using 6 evaluation metrics.

### Worked Example: "All cards have an even rank"

Consider hypothesis rank 1 for the `all_even` rule. The natural language description is "All cards have an even rank." We trace it through all 6 steps.

**Step 1: Parse.** The hypothesis has a DSL s-expression from the injection pipeline:

```
(lambda (all (lambda (eq (mod (rank_val $0) 2) 0)) $0))
```

The s-expression parser (`sexpr_parser.py`) calls `parse_program()` with the full primitive registry, producing a Program AST:

```
Abstraction(
  Application(
    Application(
      Primitive("all"),
      Abstraction(
        Application(
          Application(
            Primitive("eq"),
            Application(
              Application(Primitive("mod"), Application(Primitive("rank_val"), Index(0))),
              Primitive("2")
            )
          ),
          Primitive("0")
        )
      )
    ),
    Index(0)
  )
)
```

In readable form: `lambda hand. all (lambda c. eq (mod (rank_val c) 2) 0) hand`.

**Step 2: Rewrite for target grammar.** Suppose we are scoring under the **base** grammar. The rewriter returns the AST unchanged (identity rewrite). For the **redundant** grammar, the compressor would not find a matching shortcut pattern here (there is no `all_same`-type pattern), so the AST stays the same. For the **minimal** grammar, every primitive in this hypothesis (`all`, `eq`, `mod`, `rank_val`, `0`, `2`) is in the minimal set, so again no rewriting is needed.

**Step 3: Compute log_prior.** Under the base grammar with uniform costs (64 primitives + 1 variable slot):

```
log_prior = Grammar.program_log_likelihood(AST, HAND -> BOOL)
```

This walks the AST, computing at each production point the log-probability of choosing that primitive from among all primitives of the correct return type. The sum over all production choices gives the total log-prior. For uniform costs with N = 64 primitives, each choice contributes approximately `log(1/N)` (the exact value depends on per-type normalization). A program with ~8 primitive applications would get roughly `8 * log(1/N) ~ -33` (illustrative -- the actual value depends on type-specific branching factors).

**Step 4: Compute extension size.** We execute the hypothesis as a predicate on 1,000,000 randomly sampled 6-card hands (drawn uniformly from C(52,6) = 20,358,520 possible hands). "All cards have even rank" means every card's rank value is even (2, 4, 6, 8, 10, Q). There are 24 such cards out of 52.

The base rate is approximately C(24,6)/C(52,6) = 134,596/20,358,520 ~ 0.66%. Since the base rate is above 0.1%, the adaptive sampler stays at the 1M level. With 1M samples:

```
hits ~ 6,600 (0.66% of 1,000,000)
extension_size = (6600 / 1,000,000) * 20,358,520 ~ 134,366
```

**Step 5: Compute log_likelihood.** The 6 frozen exemplar hands shown to the LLM are the data. All 6 must satisfy the predicate (if any fail, log_likelihood = -inf). For `all_even`, assuming all exemplar hands do have even ranks:

```
log_likelihood = -6 * log(134,366) = -6 * 11.81 = -70.84
```

This is the size principle at work: a hypothesis that accepts only ~134K of the ~20M possible hands makes the 6 specific exemplars moderately surprising, yielding a moderate (not extreme) likelihood.

**Step 6: Compute log_posterior.**

```
log_posterior = log_prior + log_likelihood
             ~ -33 + (-70.84)
             = -103.84
```

This score is then compared against the posterior scores of other hypotheses for the same rule to evaluate whether the grammar's ranking matches the LLM's confidence ranking.

---

## 2. What Each Metric Measures

The pipeline computes 6 metrics for each grammar x cost-structure configuration. The first 4 are **descriptive** (does the grammar predict the LLM's ranking?). The last 2 are **supplementary/normative** (does the grammar lead to correct inferences?).

### 2.1 Spearman Agreement

**Formal definition:**

For each rule r with hypotheses h_1, ..., h_k:
1. Compute Spearman rho between (LLM ranks) and (grammar log-posteriors).
2. Negate: `agreement_r = -rho_r`.
3. Average across all rules with k >= 2.

```
spearman_agreement = -(1/|R|) * sum_r rho(LLM_ranks_r, log_posteriors_r)
```

**What "good" looks like:** Positive values indicate agreement. Because LLM rank 1 = best should correspond to the highest (least negative) log-posterior, the raw rho is negative when they agree; negation makes positive = good. Range: [-1, +1]. A value of +0.3 to +0.5 would indicate meaningful agreement.

**Caveats:** Rules with only 1 hypothesis or with all-identical log-posteriors are skipped. If all remaining hypotheses for a rule have `-inf` log-posterior, the rule contributes nothing. Spearman correlation is ordinal, so it ignores the magnitude of score differences.

### 2.2 Weighted Log-Probability

**Formal definition:**

```
weighted_log_prob = sum_h (6 - rank_h) * log_posterior_h
```

where rank_h is the LLM's confidence rank (1-5) and log_posterior_h is the grammar's posterior score. Hypotheses with `-inf` log-posterior are excluded from the sum.

**What "good" looks like:** Higher (less negative) is better. This metric rewards grammars that assign high posterior to hypotheses the LLM was confident about (rank 1 gets weight 5, rank 5 gets weight 1). The absolute magnitude depends on the number of hypotheses and the grammar's probability scale, so it is most useful for comparing grammars against each other, not as an absolute measure.

**Caveats:** Previously confounded with expressibility (grammars that couldn't score some hypotheses got artificially lower values). Now that the rewriter is complete, all grammars score all hypotheses, removing this confound.

### 2.3 Top-1 Accuracy (Chance-Corrected)

**Formal definition:**

For each rule r with k_r hypotheses:
1. Check if the LLM's rank-1 hypothesis is the grammar's highest-posterior hypothesis.
2. Apply chance correction: `corrected_r = (hit_r - 1/k_r) / (1 - 1/k_r)`.
3. Average across rules.

```
top1_accuracy = (1/|R|) * sum_r corrected_r
```

**What "good" looks like:** Positive values mean better than chance. Range: approximately [-1/(k-1), 1]. A value of 0.0 means chance-level performance. Rules with only 1 hypothesis contribute 0.0 (trivially correct).

**Caveats:** The chance correction is necessary because rules with different numbers of hypotheses (k = 2 to 5) have different random baselines. Binary; does not capture partial agreement (e.g., if the grammar ranks the LLM's top pick 2nd, that counts the same as ranking it last).

### 2.4 Expressibility

**Formal definition:**

```
expressibility = (# hypotheses with finite log-posterior) / (# total hypotheses)
```

**What "good" looks like:** Should be 1.0 (100%) for all grammars once the rewriter is complete. Any value below 1.0 indicates a rewriter bug or a genuinely inexpressible hypothesis (only expected for the minimal grammar with fold/reduce-dependent primitives). This metric serves primarily as a sanity check.

**Caveats:** A grammar that assigns `-inf` prior to a hypothesis (because the program uses a primitive not in the grammar and the rewriter cannot decompose it) will show reduced expressibility. This is not the same as the grammar being "bad" -- it means the translation pipeline is incomplete.

### 2.5 Correct-Rank

**Formal definition:**

For each rule r:
1. Sort hypotheses by log-posterior (highest = rank 1).
2. Find the ground-truth hypothesis (identified by fingerprint matching).
3. Record its position in the posterior ranking.
4. Average across rules.

```
correct_rank = (1/|R|) * sum_r position_of_ground_truth_r
```

**What "good" looks like:** Lower is better. A perfect score of 1.0 means the ground-truth hypothesis always has the highest posterior. A value equal to (k+1)/2 would indicate chance-level performance.

**Caveats:** Requires ground-truth fingerprints for each rule. Rules without a ground-truth hypothesis in the scored set are skipped. This is a normative metric (does the grammar lead to the *correct* answer?) as opposed to a descriptive one (does the grammar predict *what the LLM said*?).

### 2.6 Rule-Difficulty Correlation

**Formal definition:**

For each rule r with a known difficulty group (1 = easy, 2 = medium, 3 = hard):
1. Find the LLM's rank-1 hypothesis.
2. Read its `log_prior` (the grammar's prior cost, before likelihood).
3. Compute Spearman rho between log-priors and difficulty groups.
4. Negate (so positive = harder rules have more negative log-prior).

```
rule_difficulty_corr = -spearmanr(log_priors_rank1, difficulty_groups)
```

**What "good" looks like:** Positive values indicate that the grammar's complexity ordering matches the expected difficulty ordering (harder rules require more complex programs). Range: [-1, +1].

**Caveats:** Requires rule-group assignments (not all rules may have them). Uses only the rank-1 hypothesis per rule, so is sensitive to ties at rank 1. The "difficulty" grouping is currently coarse (3 levels).

---

## 3. The Transcription Chain

Hypotheses begin as natural language + code translations in Phase 1b JSON files and must be converted into scored Program ASTs. There are two paths, depending on what code representation is available.

### 3.1 Path A: S-Expression (252 hypotheses)

Most hypotheses were translated into DSL s-expressions during the Phase 0/Phase 1b injection pipeline. These are stored in `injected_hypotheses.json` and cross-referenced by the data loader.

**Flow:**

```
Phase 1b JSON
    |  data_loader.load_phase1b_hypotheses()
    |  Cross-references NL text with injected_hypotheses.json
    v
hypothesis dict with dsl_code field
    |  sexpr_parser.parse_hypothesis_sexpr()
    |  Calls parse_program() with full primitive registry
    v
Program AST (base grammar)
    |  rewriter.rewrite_ast(ast, target_grammar)
    v
Grammar-specific AST
    |  grammar.program_log_likelihood()
    v
log_prior score
```

The s-expression parser (`sexpr_parser.py`) wraps the existing `parse_program()` function from `src/dreamcoder_core/program.py`. It builds a primitive registry once (cached as a module singleton) that includes all 64+ base primitives plus aliases for names the LLM uses (`sum_vals` -> `sum_ranks`, `n_cards` -> `length`, etc.) and an extra multiplication primitive.

**What can go wrong:**
- **Parse error:** The s-expression has a syntax error or uses an unknown primitive name. Result: `log_prior = -inf` for this hypothesis.
- **Missing dsl_code:** The data loader could not find a matching entry in `injected_hypotheses.json` (19 hypotheses fall into this category due to text normalization differences). These fall through to Path B.

### 3.2 Path B: Python Fallback (19 hypotheses)

For hypotheses without s-expression translations, the pipeline falls back to converting the Python lambda code (which every hypothesis has) into a Program AST.

**Flow:**

```
hypothesis dict with python_code field (no dsl_code)
    |  python_parser.python_to_ast()
    |  Uses Python ast module + pattern matching
    v
Program AST (base grammar)
    |  (same rewriting and scoring as Path A)
    v
log_prior score
```

The Python parser (`python_parser.py`) handles common patterns:
- `all(expr for c in hand)` -> `(all (lambda. expr') hand)`
- `len(set(expr for c in hand))` -> `(length (unique (map (lambda. expr') hand)))`
- `sum(1 for c in hand if pred)` -> `(length (filter (lambda. pred') hand))`
- Card attribute access, comparisons, boolean operators, arithmetic
- De Bruijn index tracking through nested lambdas

**What can go wrong:**
- **Unsupported pattern:** The Python code uses a construct the parser does not handle (e.g., list comprehensions with multiple `for` clauses, dictionary operations). Result: `NotImplementedError` is raised, and `parse_hypothesis()` in `compute_costs.py` returns `None`, leading to `log_prior = -inf`.
- **Complex nesting:** Deeply nested expressions may produce incorrect de Bruijn indices if the parser's environment tracking has a bug.

### 3.3 The Rewriter

Once a base-grammar AST is obtained (via either path), it must be rewritten for the target grammar. The rewriter (`rewriter.py`) handles two directions:

**Expansion** (for swap and minimal grammars): Decompose compound primitives into the target grammar's available primitives.

| Source primitive | Target grammar | Rewritten form |
|---|---|---|
| `first_half(hand)` | swap-positional | `slice(0, 3, hand)` |
| `second_half(hand)` | swap-positional | `slice(3, 6, hand)` |
| `take(n, hand)` | swap-positional | `slice(0, n, hand)` |
| `drop(n, hand)` | swap-positional | `slice(n, 6, hand)` |
| `all(pred, adjacent_pairs(hand))` | swap-positional | `shifted_match(1, pred', hand)` |
| `count_suit(hand, S)` | swap-distributional | `count_where(lambda c. eq(get_suit c) S, hand)` |
| `count_suit(hand, S)` | minimal | `length(filter(lambda c. eq(get_suit c) S, hand))` |
| `n_unique_suits(hand)` | minimal | `length(unique(map(get_suit, hand)))` |
| `le(x, y)` | minimal | `or(lt(x, y), eq(x, y))` |

**Compression** (for redundant grammar G6): Detect patterns in the AST that match cognitive shortcuts and compress them.

| Base-grammar pattern | Compressed form |
|---|---|
| `eq(length(unique(map(key, hand))), length(hand))` | `all_different(key, hand)` |
| `lt(length(unique(map(key, hand))), 2)` or `eq(..., 1)` | `all_same(key, hand)` |
| `eq(length(filter(pred, hand)), n)` | `exactly_n(n, pred, hand)` |
| `ge(length(filter(pred, hand)), n)` | `at_least_n(n, pred, hand)` |
| `length(unique(map(key, hand)))` | `n_unique(key, hand)` |

The compression rewriter works bottom-up: it first recurses into child nodes, then attempts pattern matching on the rebuilt tree. This ensures nested patterns are compressed correctly.

**What can go wrong:**
- **InexpressibleError (minimal grammar only):** Some primitives genuinely cannot be decomposed into the minimal set because they require fold/reduce or sorting operations that are not available. The full list: `sum_ranks`, `max_rank`, `min_rank`, `sort_by_rank`, `max_suit_count`, `n_repeated_ranks`, `n_repeated_suits`, `running_sum`, `suit_to_int`, `signum`. For these, the hypothesis gets `log_prior = -inf` under the minimal grammar.
- **Identity grammars:** For `base` and `add-both`, the rewriter returns the AST unchanged (these grammars are supersets of the base primitives).
- **Missed compression patterns:** The redundant grammar compressor only detects specific AST shapes. A semantically equivalent but syntactically different arrangement will not trigger compression, meaning the shortcut primitives go unused. This is a known limitation.

### 3.4 Verification Protocol

Correctness is ensured through fingerprint matching at multiple points:

1. **Dual-path check (252 hypotheses with both representations):** Parse both the s-expression and the Python code. Compute fingerprints (200-probe boolean vector) for each. They must match exactly. This verifies that the s-expression parser and Python parser produce semantically equivalent ASTs.

2. **Python-only check (19 hypotheses):** Parse the Python code to AST, evaluate on 200 probes, compare the fingerprint against the original Python lambda executed directly. This verifies the Python-to-AST conversion is correct.

3. **Rewriting check (all grammars):** After rewriting a base AST for a target grammar, re-compute the fingerprint. It must match the base AST's fingerprint. This verifies the rewriter preserves semantics.

A fingerprint is a 200-bit string: evaluate the hypothesis-as-predicate on each of 200 fixed probe hands (loaded from `llm/results/probe_set_200.json`), producing a `1` for True and `0` for False. Two programs with identical fingerprints are extensionally equivalent on those 200 hands, which gives very high confidence of semantic equivalence.

---

## 4. The Size Principle Likelihood

### 4.1 Why Prior-Only Scoring Fails

The initial Stage 1 results (v1 design, prior only) revealed a fundamental problem: all grammars showed **positive** raw Spearman correlation (grammar *disagrees* with LLM). The reason is that the prior favors syntactically simple, short programs. But syntactically simple programs tend to be semantically **vague** -- they accept many hands. For example, `lambda hand. true` has the highest prior (it is the shortest program) but is maximally uninformative.

The LLM's rank-1 hypothesis is typically *specific* (it accurately describes the actual rule), not vague. So a prior that rewards simplicity alone will systematically rank vague hypotheses above specific ones, contradicting the LLM's confidence ordering.

### 4.2 What the Size Principle Is

The size principle (Tenenbaum & Griffiths, 2001) provides a likelihood term that penalizes vague hypotheses:

```
P(data | h) = (1 / |ext(h)|)^n
```

where:
- `ext(h)` is the **extension** of hypothesis h: the set of all possible observations consistent with h. In our domain, this is the set of all 6-card hands that h classifies as True.
- `|ext(h)|` is the size of that set.
- `n` is the number of observed data points (the 6 exemplar hands shown to the LLM).

**Intuition:** If h accepts only 1,000 out of 20 million possible hands, then seeing 6 specific hands that all happen to be in that set is strong evidence for h. But if h accepts 10 million hands, the same 6 hands are unsurprising -- almost any 6 hands would satisfy h. The size principle mathematically captures this: specific hypotheses (small extension) get high likelihood, vague hypotheses (large extension) get low likelihood.

In log space:

```
log_likelihood = -n * log(|ext(h)|)
```

This is always negative (since |ext(h)| >= 1), and becomes more negative as the extension grows.

### 4.3 How Extension Sizes Are Estimated

Since enumerating all C(52,6) = 20,358,520 possible 6-card hands is computationally feasible but slow when repeated for hundreds of hypotheses, we estimate extension sizes via Monte Carlo sampling.

**Procedure (in `extension.py`):**

1. Generate a random sample of 6-card hands drawn uniformly from the deck of 52 cards (sampling without replacement within each hand).
2. Evaluate the hypothesis predicate on each sampled hand.
3. Count the number of "hits" (hands where the predicate returns True).
4. Estimate: `extension_size = (hits / n_samples) * C(52, 6)`.

The samples are cached at the module level and reused across all hypotheses, so the expensive sampling step happens only once.

### 4.4 The Adaptive Sampling Protocol

Some hypotheses have extreme base rates -- they accept either almost no hands (< 0.1%) or almost all hands (> 99.9%). For these, 1 million samples may not provide enough hits (or misses) for a reliable estimate.

The protocol escalates through three tiers:

| Tier | Samples | Triggered when |
|---|---|---|
| 1 | 1,000,000 | Always (starting point) |
| 2 | 10,000,000 | Base rate < 0.1% at Tier 1 (fewer than ~1,000 hits) |
| 3 | 100,000,000 | Base rate < 0.1% at Tier 2 |

Note: near-universal predicates (base rate > 99.9%) do not trigger escalation because they have huge extensions, and the exact extension size does not materially affect the ranking (all near-universal hypotheses get approximately the same very low likelihood).

Each tier uses its own cached sample set (keyed by `(n_samples, seed)`), so escalation does not waste previous computation -- it generates a fresh, larger sample.

### 4.5 Why Likelihood Is Grammar-Independent

The extension size depends only on the *semantics* of the hypothesis -- which hands it classifies as True -- not on how it is expressed. Because the rewriter preserves semantics (verified by fingerprint checks), the extension size is the same regardless of which grammar the hypothesis is expressed in. This means extension sizes (and therefore likelihoods) are computed once and cached by fingerprint for reuse across all 21 grammar x cost-structure configurations.

### 4.6 The Exemplar Hands

Each rule in the Phase 1b data has 6 frozen exemplar hands (stored in the `hands_shown` field of the JSON files). These are the specific hands that the LLM was shown when generating its hypotheses. They serve as the **data** in the Bayesian computation.

A critical consistency check: every exemplar hand must satisfy the hypothesis predicate. If any exemplar fails, the hypothesis is **inconsistent with the observed data**, and its likelihood is set to 0 (log_likelihood = -inf). This is checked before Monte Carlo sampling proceeds.

The data loader (`data_loader.py`) parses hand strings like `"2S 5H KC 3D 7S JH"` into lists of Card objects, making them available as `hypothesis["exemplar_hands"]`.

---

## 5. How to Interpret Results

### 5.1 The Stage 1 Table

Running `python -m llm.grammar_comparison.run_comparison --stage 1` produces a table like:

```
Grammar              Cost      Spearman   WtdLogP    Top1  Express
---------------------------------------------------------------------------
base                 uniform     0.1234  -1234.56    0.12     1.00
base                 tiered      0.1456  -1200.34    0.15     1.00
base                 lotlib3     0.1678  -1180.23    0.18     1.00
swap-positional      uniform     0.1345  -1220.45    0.13     1.00
...
minimal              lotlib3     0.0890  -1350.78    0.08     0.95
```

**Reading the columns:**

- **Grammar + Cost:** The configuration being evaluated (7 grammars x 3 cost structures = 21 rows).
- **Spearman:** Average negated Spearman rho. **Higher is better.** Positive means the grammar's posterior ranking agrees with the LLM's confidence ranking.
- **WtdLogP:** Weighted log-probability. **Higher (less negative) is better.** Large differences (e.g., 50+ units) between grammars are meaningful; small differences may be noise.
- **Top1:** Chance-corrected top-1 accuracy. **Higher is better.** Positive means better than random; 0.0 is chance level.
- **Express:** Expressibility. **Should be 1.00.** Anything less indicates incomplete translation.

**Identifying the best grammar:** Look for the configuration(s) with the highest Spearman agreement, since this is the primary metric. If two grammars have similar Spearman values, use WtdLogP and Top1 as tiebreakers. The cost structure dimension tells you whether cognitive-accessibility weighting (tiered or LOTlib3) helps beyond uniform description length.

### 5.2 How to Read Ablation Output (Stage 2)

Stage 2 produces three analyses around a chosen base grammar:

**Leave-one-out:** Shows the impact of removing each primitive. A large *negative* delta means removing that primitive *hurts* performance -- the primitive is important. A positive delta means the grammar improves without it (the primitive is diluting probability mass).

```
Top 10 most impactful removals (by |delta| on Spearman):
Removed                      Metric   Baseline     Delta
---------------------------------------------------------------------------
get_suit                     0.0800     0.1400   -0.0600   <-- important
shift_pairs                  0.1500     0.1400    0.0100   <-- dispensable
```

**Leave-one-in:** Shows the impact of adding candidate primitives (from other grammars). A large positive delta means adding that primitive improves performance.

**Cross-validation:** Reports the 5-fold CV mean and standard deviation of the Spearman metric. This estimates how stable the grammar's performance is. A standard deviation larger than the difference between two grammars suggests the difference is not reliable.

### 5.3 What Constitutes a Meaningful Difference

With ~271 hypotheses across ~60 rules, statistical power is limited. Rules of thumb:

- **Spearman difference > 0.05** between grammars is likely meaningful, especially if consistent across cost structures.
- **Spearman difference < 0.02** is probably noise.
- **Cross-validation std > difference** means the difference is not reliable.
- **Expressibility < 1.0** for any grammar (except minimal) indicates a pipeline bug that must be fixed before interpreting other metrics.
- Compare grammars across all three cost structures. A grammar that wins under uniform but loses under tiered/LOTlib3 may just be benefiting from an accident of uniform weighting.

### 5.4 The Cross-Validation Protocol

Stage 1 can be run with 5-fold cross-validation (via the ablation module). The ~271 hypotheses are split into 5 folds. For each fold:
1. Compute posteriors on the training set (4/5 of hypotheses).
2. Evaluate all 6 metrics on the held-out set (1/5 of hypotheses).
3. Report mean and standard deviation across folds.

The held-out evaluation naturally penalizes overly complex grammars: a grammar with too many primitives spreads prior mass too thinly, reducing the posterior of any specific hypothesis. This eliminates the need for an explicit grammar complexity penalty, following the approach of Piantadosi et al. (2016).

---

## 6. How to Add a New Grammar

Adding a new grammar involves 4 files. Here is the step-by-step process.

### Step 1: Define the primitive set in `grammar_factory.py`

1. Add a name to the `GRAMMAR_NAMES` list:

```python
GRAMMAR_NAMES = [
    "base",
    "swap-positional",
    # ... existing entries ...
    "my-new-grammar",    # <-- add here
]
```

2. If the grammar introduces new primitives, create builder functions (similar to `_make_new_positional_primitives()`). Each primitive needs a name, a type (using the type constructors from `type_system.py`), and a Python implementation:

```python
def _make_my_primitives() -> List[Primitive]:
    a = TypeVariable(0)
    return [
        Primitive(
            "my_prim",
            arrow(CARD, BOOL),           # type signature
            lambda c: c.rank.value > 5,  # implementation
        ),
    ]
```

3. Add a branch to `_select_primitives()`:

```python
elif name == "my-new-grammar":
    base = build_primitives()
    # Remove unwanted primitives
    filtered = [p for p in base if p.name not in {"prim_to_remove"}]
    # Add new primitives
    filtered.extend(_make_my_primitives())
    return filtered
```

4. Assign tiers for the `TIERED` cost structure. Add your new primitive names to `TIER1_NAMES`, `TIER2_NAMES`, or leave them in the default Tier 3.

### Step 2: Add rewriting rules in `rewriter.py`

If your grammar removes base primitives (swap or minimal style), you need **expansion** rules. If it adds cognitive shortcuts (redundant style), you need **compression** rules.

**For expansion:** Create a new `ProgramTransformer` subclass:

```python
class _MyRewriter(ProgramTransformer):
    def transform_application(self, program: Application) -> Program:
        func, args = _uncurry(program)
        if isinstance(func, Primitive):
            if func.name == "prim_to_remove" and len(args) == N:
                # Build the equivalent expression using available primitives
                return _curry(_prim("replacement"), [...])
        # Default: recurse
        new_f = self.transform(program.f)
        new_x = self.transform(program.x)
        if new_f is program.f and new_x is program.x:
            return program
        return Application(new_f, new_x)
```

**For compression:** Add pattern matchers to a compressor class (see `_RedundantCompressor` for examples). The key is to recurse into children first (bottom-up), then try to match patterns on the rebuilt tree.

Add a dispatch branch to `rewrite_ast()`:

```python
if target_grammar == "my-new-grammar":
    return _MyRewriter().transform(program)
```

### Step 3: Assign tier costs

In `grammar_factory.py`, add your new primitives to the appropriate tier sets (`TIER1_NAMES`, `TIER2_NAMES`). Primitives not assigned to Tier 1 or 2 default to Tier 3 (weight 1).

### Step 4: Verify

1. **Expressibility check:** Run the pipeline with `--stage 1 --grammars my-new-grammar --limit 20` and verify expressibility is 1.0 (or close to it for minimal-style grammars).

2. **Fingerprint verification:** For a few representative hypotheses, verify that the rewritten AST produces the same fingerprint as the base AST:

```python
from llm.grammar_comparison.translation.verification import (
    verify_rewrite_preserves_semantics, load_probe_hands,
)
probes = load_probe_hands()
match, details = verify_rewrite_preserves_semantics(
    "(lambda (all (lambda (eq (get_suit $0) CLUBS)) $0))",
    "my-new-grammar",
    probes,
)
assert match, f"Rewrite changed semantics: {details}"
```

3. **Full run:** Run `--stage 1` with all grammars including your new one, and compare metrics.

### Checklist Summary

- [ ] Name added to `GRAMMAR_NAMES`
- [ ] Primitive builder function(s) created (if new primitives)
- [ ] `_select_primitives()` branch added
- [ ] Tier assignments updated
- [ ] Rewriter class created (if removing or adding shortcut primitives)
- [ ] `rewrite_ast()` dispatch branch added
- [ ] Expressibility verified at 1.0
- [ ] Fingerprint verification passes for representative hypotheses
- [ ] Full Stage 1 comparison run completes successfully
