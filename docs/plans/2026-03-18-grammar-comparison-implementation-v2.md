# Grammar Comparison Implementation Plan v2

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Update the grammar comparison pipeline with posterior scoring (prior + size principle likelihood), complete rewriting, and revised metrics. Builds on the existing v1 implementation (Tasks 1–12 complete, 285 tests passing).

**What exists:** The v1 pipeline is functional on branch `feat/grammar-comparison`. All modules exist in `llm/grammar_comparison/`. This plan adds/modifies modules to address issues identified in the brainstorming review.

**Architecture change:** Score = `prior(h, G) + log_likelihood(h)` where likelihood uses the size principle. Extension sizes estimated via adaptive Monte Carlo sampling (1M–100M hands).

**Tech Stack:** Same as v1 plus `scipy.stats` for correlations.

## Execution Guidelines
- Explain code as you write it
- Test each step before proceeding
- Commit after each task
- NEVER modify files in `src/` — only read/import from them
- Run full test suite (`python3 -m pytest llm/grammar_comparison/tests/ -v`) after each task to check for regressions

---

### Task A: Recover 19 missing hypotheses via Python-to-AST fallback

**Files:**
- Modify: `llm/grammar_comparison/evaluation/compute_costs.py`
- Modify: `llm/grammar_comparison/data_loader.py` (if needed)
- Test: `llm/grammar_comparison/tests/test_compute_costs.py`

**What:** Currently 19 of 271 hypotheses have `dsl_code=None` because the text cross-reference failed. These hypotheses DO have `python_code`. Modify `score_hypothesis()` (or add a wrapper) to fall back to the Python-to-AST converter when `dsl_code` is missing.

Flow:
1. If `dsl_code` exists → parse via `parse_hypothesis_sexpr()` (existing path)
2. If `dsl_code` is None but `python_code` exists → parse via `python_to_ast()` (fallback)
3. If neither → return -inf

Add tests verifying that hypotheses without dsl_code still get scored.

**Commit:** `feat: fall back to Python parser for hypotheses without s-expressions`

---

### Task B: Adaptive extension size estimation

**Files:**
- Create: `llm/grammar_comparison/evaluation/extension.py`
- Test: `llm/grammar_comparison/tests/test_extension.py`

**What:** Implement `estimate_extension_adaptive(predicate, exemplar_hands, ...)` that:

1. Samples 1,000,000 random 6-card hands from C(52,6) using a fixed seed
2. Evaluates the predicate on each hand, counts hits
3. If base_rate < 0.001 or base_rate > 0.999 (fewer than ~1000 hits or misses):
   - Escalate to 10,000,000 samples
4. If still extreme at 10M:
   - Escalate to 100,000,000 samples
5. Checks that all 6 exemplar hands satisfy the predicate (consistency check)
   - If any exemplar fails → predicate is inconsistent with data → return (0, -inf log_likelihood)
6. Computes log_likelihood = -n × log(extension_size / TOTAL_HANDS) where n = len(exemplar_hands)
7. Caches results by fingerprint (grammar-independent)

Return a dataclass:
```python
@dataclass
class ExtensionResult:
    extension_size: int         # estimated |ext(h)| out of C(52,6)
    base_rate: float            # hits / n_samples
    n_samples: int              # how many samples were used (1M, 10M, or 100M)
    log_likelihood: float       # -n * log(base_rate), or -inf if inconsistent
    exemplars_consistent: bool  # do all exemplar hands satisfy h?
```

Also implement `build_hand_cache(n_samples, seed)` that pre-generates the random hands and caches them for reuse.

Reference: `src/gallery_analysis/hypothesis_table.py:estimate_extension_size()` for the existing approach (100K samples, no adaptive scaling).

**Tests:**
- Known simple predicate (all_even) returns reasonable base rate (~1.8%)
- Exemplar consistency check catches mismatches
- Adaptive escalation triggers for a rare predicate
- Results are cached by fingerprint (second call is instant)
- log_likelihood is finite and negative for valid predicates

**Commit:** `feat: add adaptive extension size estimation with size principle likelihood`

---

### Task C: Complete the AST rewriter (never give up)

**Files:**
- Modify: `llm/grammar_comparison/translation/rewriter.py`
- Test: `llm/grammar_comparison/tests/test_rewriter.py`

**What:** The rewriter currently raises `InexpressibleError` when it encounters a primitive not in the target grammar. This is wrong — all grammars are computationally universal. The rewriter must decompose any primitive into available ones.

**Expansion rules to add (for swap/minimal grammars):**

Every primitive not in the target grammar needs a decomposition. Read the existing `rewriter.py` to see what's already handled, then add rules for everything missing. Key ones:

For swap-positional:
- `adjacent_pairs(hand)` used standalone (not inside all/any) → decompose to list comprehension equivalent or mark which context it appears in

For swap-distributional:
- Any remaining count_* primitives that aren't handled

For minimal — decompose ALL compound primitives:
- `first_half(hand)` → `take(3, hand)`
- `second_half(hand)` → `drop(3, hand)`
- `all_same_suit(hand)` → `all(λc. eq(get_suit c)(get_suit(head hand)), hand)`
- `count_suit(hand, S)` → `length(filter(λc. eq(get_suit c) S, hand))`
- `n_unique_suits(hand)` → `length(unique(map(get_suit, hand)))`
- `n_unique_ranks(hand)` → `length(unique(map(get_rank, hand)))`
- `sum_vals(hand)` → decompose using fold or map+sum
- etc. — enumerate ALL primitives in base that aren't in minimal

For **any** remaining cases the mechanical rewriter can't handle in minimal: use LLM-assisted translation via Gemini Flash (reuse Phase 0 pipeline), verified by fingerprint matching.

**Compression rules to add (for redundant grammar):**

Detect AST patterns that match shortcuts and replace them:
- `all(λc. eq(KEY c)(KEY(head $0)), $0)` → `all_same(KEY, $0)`
- `eq(length(unique(map KEY $0)))(length $0)` → `all_different(KEY, $0)`
- Pattern for `is_sorted`, `exactly_n`, `at_least_n`, `n_unique`, `is_run`, `has_pair`

**Remove InexpressibleError entirely** — the rewriter should always return a valid AST.

**Tests:**
- Every grammar achieves 100% expressibility on the 252 test hypotheses
- Rewriting preserves fingerprints (semantic equivalence)
- Compression rules correctly detect and replace patterns for G6

**Commit:** `feat: complete AST rewriter — bidirectional, never fails`

---

### Task D: Update scorer with posterior = prior + likelihood

**Files:**
- Modify: `llm/grammar_comparison/evaluation/compute_costs.py`
- Test: `llm/grammar_comparison/tests/test_compute_costs.py`

**What:** Update `score_hypothesis()` and `score_all_hypotheses()` to compute posterior scores:

```python
def score_hypothesis(sexpr_or_python, grammar, exemplar_hands, extension_cache) -> dict:
    """
    Returns dict with:
        log_prior: float      — grammar PCFG score
        log_likelihood: float — size principle score
        log_posterior: float  — prior + likelihood
        extension_result: ExtensionResult
    """
```

The data loader needs to also load the `hands_shown` field from Phase 1b files and pass it through to the scorer.

**Update `score_all_hypotheses()`** to:
1. Load hypotheses WITH exemplar hands
2. Build extension cache (compute once, reuse across grammars)
3. For each hypothesis: compute prior (grammar-specific) + likelihood (cached)
4. Return list of dicts with: rule_id, rank, confidence, log_prior, log_likelihood, log_posterior

**Tests:**
- Posterior is sum of prior + likelihood
- Specific hypotheses (small extension) get likelihood boost
- Vague hypotheses (large extension) get likelihood penalty
- Hypotheses inconsistent with exemplars get -inf likelihood
- Extension cache is reused across grammars (verify by timing)

**Commit:** `feat: update scorer with posterior = prior + size principle likelihood`

---

### Task E: Update data loader to include exemplar hands

**Files:**
- Modify: `llm/grammar_comparison/data_loader.py`
- Test: `llm/grammar_comparison/tests/test_data_loader.py`

**What:** The Phase 1b JSON files contain a `hands_shown` field with the 6 exemplar hands. Add this to the loaded hypothesis dicts.

Parse the hand strings (e.g., `"2♠ 5♥ K♣ 3♦ 7♠ J♥"`) into `List[Card]` objects. Reference: `llm/grammar_comparison/translation/verification.py:load_probe_hands()` for hand parsing logic.

Each hypothesis dict gets a new field: `exemplar_hands: List[List[Card]]` (the 6 hands, each a list of 6 Card objects).

**Tests:**
- Loaded hypotheses have exemplar_hands field
- Each exemplar_hands has 6 hands of 6 cards each
- Cards are valid Card objects with correct suits and ranks

**Commit:** `feat: include exemplar hands in loaded Phase 1b data`

---

### Task F: Update metrics

**Files:**
- Modify: `llm/grammar_comparison/evaluation/metrics.py`
- Test: `llm/grammar_comparison/tests/test_metrics.py`

**What:** Update existing metrics and add new ones:

1. **Spearman agreement** — negate the correlation so higher = better. Use `log_posterior` instead of `log_prob`.

2. **Weighted log-probability** — use `log_posterior` instead of `log_prob`.

3. **Top-1 accuracy (chance-corrected)** — apply correction: `(accuracy - 1/k) / (1 - 1/k)` per rule, where k = number of hypotheses for that rule. Average across rules.

4. **Expressibility** — unchanged, but should now be 100% for all grammars (sanity check).

5. **Correct-rank** (NEW) — for each rule, find the ground-truth hypothesis (the one that matches the rule's actual definition). Compute its rank in the grammar's posterior ordering (1 = best). Average across rules. Lower = better.

   Requires knowing which hypothesis is correct. Options:
   - Compare each hypothesis's fingerprint against the ground-truth rule's fingerprint
   - Use the `rule_id` to look up the ground-truth predicate from the catalogue

6. **Rule-difficulty correlation** (NEW) — for each of the 60 rules, compute the grammar's prior cost (description length) of the ground-truth program. Correlate with rule group (1=easy, 2=medium, 3=hard). Report Spearman correlation.

   Requires: ground-truth programs in each grammar. These can be obtained from the catalogue rules, translated to ASTs, rewritten per grammar.

**Tests:**
- Spearman agreement: positive value when grammar agrees with LLM
- Chance-corrected top-1: returns 0 when accuracy equals chance
- Correct-rank: returns 1.0 when ground truth is always ranked first
- Rule-difficulty: returns positive correlation when expensive rules are in higher groups

**Commit:** `feat: update metrics — negated Spearman, chance-corrected top-1, correct-rank, rule-difficulty`

---

### Task G: Update main runner and re-run Stage 1

**Files:**
- Modify: `llm/grammar_comparison/run_comparison.py`

**What:** Update the runner to:
1. Use posterior scores (prior + likelihood)
2. Report all 6 metrics
3. Run k-fold cross-validation for Stage 1 (k=5)
4. Report mean ± std for each metric across folds
5. Print updated summary table

**Stage 1 output format:**
```
Grammar Comparison — Stage 1 (5-fold cross-validated)
═══════════════════════════════════════════════════════════════════════════════
Grammar              Cost      Agree±std  WtdLP±std  Top1±std  CRank±std  RDiff
───────────────────────────────────────────────────────────────────────────────
base                 uniform   0.42±0.05  -1234±89   0.35±0.08  2.1±0.3   0.45
swap-distributional  lotlib3   0.58±0.04  -987±72    0.48±0.06  1.8±0.2   0.52
...
```

**Commit:** `feat: update runner with posterior scoring and cross-validation`

---

### Task H: Write pipeline documentation

**Files:**
- Create: `llm/grammar_comparison/docs/pipeline_guide.md`

**What:** Comprehensive documentation covering:

1. **End-to-end pipeline overview** — the 6-step scoring pipeline with a worked example for one hypothesis (e.g., all_even rank 1). Show actual numbers at each step: AST size, prior cost, extension size, likelihood, posterior.

2. **What each metric measures** — formal definitions with equations, how to interpret values, what "good" looks like, known caveats.

3. **The transcription chain** — how hypotheses flow from NL → s-expression → AST → grammar-specific AST → score. Include what can go wrong at each step and how errors are caught.

4. **The size principle likelihood** — what it is, why it's needed (the prior-only problem), how extension sizes are estimated, the adaptive sampling protocol, why it's grammar-independent.

5. **How to interpret results** — what the Stage 1 table means, how to read ablation output, what constitutes a meaningful difference between grammars.

6. **How to add a new grammar** — step-by-step: define primitive set in grammar_factory.py, add rewriting rules in rewriter.py, add tier assignments, test.

**Commit:** `docs: add comprehensive pipeline guide for grammar comparison`

---

### Task I: Run full revised comparison and analyze

**Steps:**
1. Run full test suite: `python3 -m pytest llm/grammar_comparison/tests/ -v`
2. Run Stage 1: `python3 -m llm.grammar_comparison.run_comparison --stage 1`
3. Analyze results — compare with v1 results, check if Spearman flipped to positive agreement
4. Run Stage 2 ablations around top performer
5. Commit results

**Commit:** `results: Stage 1+2 grammar comparison with posterior scoring`

---

## Dependency Graph

```
Task A (recover 19 missing)     ─┐
Task B (extension estimation)    ├── Task D (posterior scorer) ──┐
Task E (exemplar hands in data) ─┘                               │
                                                                  ├── Task G (runner) → Task I (execute)
Task C (complete rewriter)  ─────────────────────────────────────┤
                                                                  │
Task F (update metrics) ─────────────────────────────────────────┘

Task H (documentation) — can run in parallel with any task
```

Tasks A, B, C, E can run in parallel.
Task D depends on A, B, E.
Task F is independent.
Task G depends on C, D, F.
Task H is independent.
Task I depends on G.
