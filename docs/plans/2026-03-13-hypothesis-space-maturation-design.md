# Hypothesis Space Maturation Design

## Goal

Mature the Bayesian rule induction hypothesis space to produce richer, more credible confuser/foil profiles for human participants. Two complementary changes: (1) non-uniform prior weighting to suppress shallow dominance, (2) expanded LLM-generated foil injection.

## Problem Statement

Under the current uniform grammar (`log_p = -log(63) ≈ -4.14` for all 62 primitives), ultra-shallow hypotheses like `has_suit $0 SPADES` dominate the posterior for 18/60 rules (30%). This is counterintuitive — these depth-1 programs are implausible as human hypotheses for most rules, yet they top the charts because:

1. **Cheapest possible prior**: depth-1 programs have minimal cost under uniform weights
2. **6-exemplar tolerance**: with only 6 exemplars, many shallow predicates pass all positive examples
3. **Size Principle reward**: `has_suit` has extension ~60K/311K — smaller than "always true" but not small enough to be penalized

The goal is NOT to optimize posterior sharpness for the true rule, but to produce a **richer, more psychologically plausible** space of top competitors (confusers/foils) that includes compositional alternatives alongside shallow ones.

## Approach: Two-Phase Strategy

### Phase 1 — Non-Uniform Prior Weighting (no re-enumeration)

**Core insight**: The existing hypothesis pool (8,126+ equivalence classes at depth 6-7, plus 153 injected hypotheses) is sufficient. We don't need to re-enumerate — we just need to re-score with a grammar that penalizes ultra-shallow shortcuts and rewards compositional depth.

**4-Tier Grammar (F_extreme scheme)**:

| Tier | log_p | Primitives | Count | Rationale |
|------|-------|-----------|-------|-----------|
| **Cheap** | -3.0 | `eq`, `lt`, `le`, `gt`, `ge`, `and`, `or`, `not`, `all`, `any`, `if` | 11 | Compositional glue — should be nearly free to use |
| **Standard** | -4.0 | All accessors (`rank_val`, `get_suit`, `get_color`), constants (`HEARTS`, `RED`, `0`-`13`), arithmetic (`+`, `-`, `mod`), list ops (`reverse`, `filter`, `head`, `last`, etc.) | ~38 | Default building blocks |
| **Aggregate** | -5.5 | `count_suit`, `count_color`, `n_unique_suits`, `n_unique_ranks`, `n_unique_colors`, `max_suit_count`, `n_repeated_ranks`, `n_repeated_suits`, `sum_ranks`, `max_rank`, `min_rank` | 11 | Prepackaged aggregate queries — useful but shouldn't dominate over compositional equivalents |
| **Ultra-shallow** | -9.0 | `has_suit`, `has_color` | 2 | Boolean shortcut queries — extremely penalized to prevent shallow dominance |

**Variable cost**: `log_variable = -1.0` (was -4.14). Cheap variables encourage compositional depth — using a lambda variable should be nearly free since it's just "referring to the thing you already have."

**Empirical validation** (from `/private/tmp/compare_aggressive_weights.py`):
- Shallow `has_suit`/`has_color` top-1 dominance: 18/60 → 1/60 rules
- True rule rank improved for 42/60 rules, worsened for 3, unchanged for 15
- Compositional hypotheses (e.g., `all (λ not (eq ...)) $0`) rise dramatically in posterior mass
- The one remaining shallow-dominated rule is `all_red`, where `has_color $0 RED` is semantically correct

**Implementation**: Add `build_weighted_gallery_grammar()` to `enumerator.py`, wire into `injection.py` and `analyze.py`. The prior re-computation uses existing `dsl_prior.py::compute_log_prior()` — no changes to the prior computation logic itself.

### Phase 2 — Expanded LLM Foil Injection

**Current state**: 153 injected hypotheses (111 LLM foils, 40 true rules, 2 approximate). These come from Gemini Flash generating alternative hypotheses for each gallery rule.

**Goal**: Expand the injection set with additional LLM-generated confusers from the adjacent card-games project. These hypotheses were generated during experiment design and represent plausible participant misconceptions.

**Process**:
1. Identify untranslated LLM hypotheses from the adjacent project
2. Translate Python lambdas → DSL program strings (using the established translation patterns)
3. Validate via `injection.py::load_and_validate_injections()`
4. Merge into the hypothesis pool via `merge_injected()`

**Key constraint**: The injection pipeline already handles grammar-based prior computation, fingerprinting, and merging. It works with ANY grammar, so Phase 2 automatically benefits from Phase 1's non-uniform weights.

## What We Explicitly Skip

- **No depth-8 enumeration**: Would take days, and most depth-8 programs get crushed by prior anyway (50× worse than depth-7 under uniform, even more under weighted grammar)
- **No re-enumeration at depth 6-7**: The existing pool is saturated — the 8,126 classes already capture the significant behavioral distinctions. Re-enumerating with different parameters would mostly find duplicates
- **No grammar restructuring**: The 4-tier weighting achieves the same effect as restructuring without changing the enumerator's combinatorics

## Design Decisions

### Why -9.0 for ultra-shallow and not -7.0 or -11.0?

At -7.0 (scheme E), 2/60 rules still had shallow top-1 (not just `all_red`). At -9.0, only the semantically correct case survives. Going further (e.g., -11.0) would effectively eliminate `has_suit`/`has_color` from contention entirely, which is too aggressive — they should still appear as low-ranked alternatives.

### Why -1.0 for variables and not -2.0 or 0.0?

Variables at -1.0 make lambdas nearly free, which is the right behavior — a HOF like `all (λ ...) $0` should not be penalized for using its bound variable. At -2.0, the improvement was noticeable but less dramatic. At 0.0, variables would be free, which doesn't match any reasonable psycholinguistic theory (referring to something still has minimal cognitive cost).

### Why not change the enumeration grammar?

The enumeration grammar (used by `TopDownEnumerator`) controls which programs get generated. Changing it would alter the hypothesis pool itself. The scoring grammar (used by `dsl_prior.py`) controls how we weight those programs post-hoc. Changing the scoring grammar is:
- **Faster**: No re-enumeration needed (~seconds vs ~hours)
- **Reversible**: Can try different weights without regenerating the pool
- **Composable**: Can combine with injection without interference

The user noted they may want to fine-tune weights further after observing results — keeping these separate makes that trivial.

## Files Changed

### Phase 1 (Prior Re-weighting)

| File | Change |
|------|--------|
| `src/gallery_analysis/enumerator.py` | Add `build_weighted_gallery_grammar()` with 4-tier weighting |
| `src/gallery_analysis/injection.py` | Accept optional grammar parameter (already does, just needs to use weighted grammar by default) |
| `src/gallery_analysis/analyze.py` | Add `--grammar` flag (`uniform` vs `weighted`), pass grammar through pipeline |
| `src/gallery_analysis/dsl_prior.py` | No changes — already grammar-agnostic |

### Phase 2 (Foil Injection Expansion)

| File | Change |
|------|--------|
| `src/gallery_analysis/data/injected_hypotheses.json` | Append new translated LLM hypotheses |
| `src/gallery_analysis/data/llm_hypotheses_raw.json` | Update with any newly discovered raw hypotheses |

### Validation Script

| File | Change |
|------|--------|
| `src/gallery_analysis/validate_weights.py` (new) | Quick-check script: loads pool, re-scores with weighted grammar, prints summary table comparing uniform vs weighted across all 60 rules |

## Tier Assignment Reference

Complete tier assignment for all 62 primitives (excluding `true`/`false`):

**Cheap (-3.0)**: `eq`, `lt`, `le`, `gt`, `ge`, `and`, `or`, `not`, `all`, `any`, `if`

**Standard (-4.0)**: `rank_val`, `get_suit`, `get_color`, `head`, `last`, `at`, `len`, `reverse`, `first_half`, `second_half`, `sort_by_rank`, `unique`, `take`, `drop`, `filter`, `adjacent_pairs`, `map`, `zip_with_index`, `HEARTS`, `DIAMONDS`, `CLUBS`, `SPADES`, `RED`, `BLACK`, `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`, `10`, `11`, `12`, `13`, `+`, `-`, `mod`

**Aggregate (-5.5)**: `count_suit`, `count_color`, `n_unique_suits`, `n_unique_ranks`, `n_unique_colors`, `max_suit_count`, `n_repeated_ranks`, `n_repeated_suits`, `sum_ranks`, `max_rank`, `min_rank`

**Ultra-shallow (-9.0)**: `has_suit`, `has_color`

**Variables (-1.0)**: All bound variables (`$0`, `$1`, ...)

## Success Criteria

1. **Shallow dominance reduced**: `has_suit`/`has_color` as top-1 in ≤ 2/60 rules (down from 18/60)
2. **Compositional competitors visible**: For rules like `three_or_more_same_suit`, `all_odd`, `colors_palindrome`, the top-10 should include structurally meaningful alternatives (HOFs, multi-step comparisons), not just trivial boolean checks
3. **No regression on well-recovered rules**: Rules like `all_same_suit` (entropy 0.00) and `three_spades` (entropy 0.33) should remain sharp
4. **Expanded foil set**: At least 50 additional LLM-generated confusers beyond the current 153
5. **Validation script passes**: `validate_weights.py` produces a clean comparison table with no errors
