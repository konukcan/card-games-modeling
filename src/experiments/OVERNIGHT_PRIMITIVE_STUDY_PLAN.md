# Comprehensive Overnight Primitive Library Study

**Goal:** Determine the optimal primitive library through systematic ablation and addition experiments, including wake-sleep dynamics for library evolution.

**Runtime Target:** 10-12 hours (overnight)

---

## Executive Summary

This plan covers three major experimental tracks:

| Track | Focus | Duration | Purpose |
|-------|-------|----------|---------|
| **A: Fine-Grained Ablation** | Remove individual/grouped primitives | ~3 hours | Find essential vs redundant primitives |
| **B: Addition Experiments** | Add back removed technical primitives | ~2 hours | Test if abstract combinators help |
| **C: Wake-Sleep Dynamics** | Multi-iteration learning with library evolution | ~5-6 hours | Test compounding abstractions |

---

## Track A: Fine-Grained Ablation Experiments

### A1: Individual Gestalt Primitive Ablation (~1 hour)

Test removing each gestalt primitive individually to measure impact:

```
Variant                    Removes                     Composition Alternative
─────────────────────────────────────────────────────────────────────────────
baseline                   (none)                      -
no_all_same_suit           all_same_suit               eq 1 (n_unique_suits hand)
no_all_same_color          all_same_color              eq 1 (n_unique_colors hand)
no_n_unique_suits          n_unique_suits              length (unique (map get_suit hand))
no_n_unique_ranks          n_unique_ranks              length (unique (map get_rank hand))
no_n_unique_colors         n_unique_colors             length (unique (map get_color hand))
no_has_suit                has_suit                    any (λ eq SUIT (get_suit $0)) hand
no_has_color               has_color                   any (λ eq COLOR (get_color $0)) hand
no_count_suit              count_suit                  length (filter (λ eq SUIT (get_suit $0)) hand)
no_count_color             count_color                 length (filter (λ eq COLOR (get_color $0)) hand)
no_first_half              first_half                  take (half_len hand) hand
no_second_half             second_half                 drop (half_len hand) hand
```

**Parameters:**
- Tasks: 44 pretraining + 45 catalogue = 89 tasks
- Iterations: 1
- Budget: 100,000 programs per task
- Timeout: 60s per task
- Estimated: 12 variants × 89 tasks × ~2s = ~35 minutes

### A2: Grouped Ablation by Category (~1 hour)

Test removing entire categories of primitives:

```
Variant                    Removes                                        # Removed
─────────────────────────────────────────────────────────────────────────────────────
no_gestalt_perception      all_same_suit, all_same_color                  2
no_uniqueness_queries      n_unique_suits, n_unique_ranks, n_unique_colors  3
no_membership_queries      has_suit, has_color                            2
no_counting_queries        count_suit, count_color                        2
no_halves                  first_half, second_half                        2
no_aggregates              sum_ranks, max_rank, min_rank                  3
no_list_slicing            take, drop, half_len, adjacent_pairs           4
no_position_access         head, last, at, reverse                        4
no_comparison_pairs        lt + gt (keep le, ge, eq)                      2
minimal_gestalt            all gestalt (9) + halves (2) = 11 removed      11
minimal_hof                no map, filter (keep all, any, unique)         2
ultra_minimal              all removable primitives (15+)                 15+
```

**Parameters:**
- Tasks: 89 tasks
- Iterations: 1
- Budget: 100,000 programs per task
- Estimated: 12 variants × 89 tasks × ~2s = ~35 minutes

### A3: Interaction Effects (~45 minutes)

Test if removing combinations has non-additive effects:

```
Variant                           Removes (testing interaction)
────────────────────────────────────────────────────────────────────
no_gestalt_no_halves              all_same_*, halves
no_count_no_has                   count_*, has_*
no_unique_no_gestalt              n_unique_*, all_same_*
no_slicing_no_position            take, drop, head, last, at
no_hof_no_gestalt                 map, filter + all_same_*
```

---

## Track B: Addition Experiments (Testing Removed Technical Primitives)

These primitives were intentionally removed from v3 but may provide useful expressiveness:

### B1: Abstract Combinators (~40 minutes)

```
Primitive    Type                      Semantic                    Composition Enables
─────────────────────────────────────────────────────────────────────────────────────
compose      (b→c) → (a→b) → a → c     Function composition        Chained transformations
flip         (a→b→c) → b → a → c       Argument swap               Different argument orders
const        a → b → a                  Constant function           Ignoring second arg
id           a → a                      Identity                    Placeholder in HOFs
```

**Test Variants:**
```
Variant                   Adds
──────────────────────────────
baseline                  (none - current library)
add_compose               compose only
add_flip                  flip only
add_const                 const only
add_all_combinators       compose + flip + const + id
```

### B2: Removed Operators (~40 minutes)

```
Primitive    Type             Reason Removed          Why Test Adding
───────────────────────────────────────────────────────────────────────────
neq          a → a → bool     Use "not (eq x y)"      May be more efficient
fold         (a→b→b)→b→[a]→b  Complex, rare use       Enables aggregates
tail         [a] → [a]        Rarely needed           List manipulation
cons         a → [a] → [a]    List construction       Building new lists
```

**Test Variants:**
```
Variant                   Adds
──────────────────────────────
add_neq                   neq only
add_fold                  fold only
add_tail                  tail only
add_list_construction     cons + tail
add_neq_fold              neq + fold
```

### B3: Rank Constants (~30 minutes)

These were removed because they make rules too specific:

```
Constants      Description               Why Removed
─────────────────────────────────────────────────────
10, 11, 12     Face card values          Too specific to poker
13, 14         King, Ace high            May bias search
17, 21         Blackjack thresholds      Game-specific
```

**Test Variants:**
```
Variant                   Adds
──────────────────────────────
add_face_cards            10, 11, 12, 13, 14
add_blackjack             17, 21
add_all_rank_constants    All of above
```

---

## Track C: Wake-Sleep Dynamics

This is the most important experiment - testing whether abstractions compound over iterations.

### C1: Wake-Sleep With Library Learning (~2.5 hours)

Full DreamCoder-style loop:
1. **Wake phase:** Enumerate programs, find solutions
2. **Sleep (Recognition):** Train neural model on solutions
3. **Sleep (Abstraction):** Compress library, find shared patterns
4. **Iterate:** Use improved grammar for next iteration

**Variants to test:**

```yaml
Library Configuration:
  baseline_wakesleep:
    primitives: full library (59)
    iterations: 6
    library_learning: enabled

  minimal_wakesleep:
    primitives: minimal (48)
    iterations: 6
    library_learning: enabled

  no_gestalt_wakesleep:
    primitives: no gestalt (57)
    iterations: 6
    library_learning: enabled
```

**Parameters per variant:**
- Tasks: 44 pretraining rules
- Iterations: 6
- Budget: 150,000 programs/task
- Dreams: 100 per iteration
- Recognition epochs: 15
- Estimated: 3 variants × ~50 minutes = ~2.5 hours

### C2: Abstraction Recovery Test (~1.5 hours)

Key question: If we remove gestalt primitives, can DreamCoder rediscover them as abstractions?

```
Test: Start with minimal library, see if compression recovers:
  - "eq 1 (n_unique_suits $0)" → compressed to single abstraction?
  - This tests the MDL-driven library learning
```

**Parameters:**
- Start: ultra_minimal library
- Iterations: 8
- Budget: 200,000 programs/task
- Library learning: aggressive (lower compression threshold)
- Estimated: ~1.5 hours

### C3: Recognition Model Guidance Impact (~2 hours)

Test how neural guidance interacts with library choices:

```
Condition A: No neural guidance (uniform prior)
Condition B: Neural guidance after 3 iterations
Condition C: Neural guidance from start

Cross with:
  - Full library
  - Minimal library

= 6 conditions total
```

**Parameters:**
- Per condition: 4 iterations
- Budget: 100,000 programs/task
- Estimated: 6 conditions × ~20 min = ~2 hours

---

## Complete Overnight Pipeline

### Schedule

```
Phase    Duration   Description
────────────────────────────────────────────────────────────────
Setup    5 min      Load tasks, validate grammar, initialize
A1       35 min     Individual primitive ablation
A2       35 min     Grouped category ablation
A3       30 min     Interaction effects
B1       40 min     Abstract combinator addition
B2       40 min     Removed operator addition
B3       30 min     Rank constant addition
C1       2.5 hr     Wake-sleep with library learning
C2       1.5 hr     Abstraction recovery test
C3       2 hr       Recognition model guidance impact
Report   10 min     Generate summary report
────────────────────────────────────────────────────────────────
TOTAL    ~10 hours
```

### Output Structure

```
results_overnight_primitive_study/
├── track_A_ablation/
│   ├── A1_individual/
│   ├── A2_grouped/
│   └── A3_interactions/
├── track_B_addition/
│   ├── B1_combinators/
│   ├── B2_operators/
│   └── B3_constants/
├── track_C_wakesleep/
│   ├── C1_library_learning/
│   ├── C2_abstraction_recovery/
│   └── C3_recognition_impact/
├── summary_report.json
└── recommendation.md
```

### Metrics to Track

For each experiment:

1. **Solve Rate:** % of tasks solved
2. **Solution Depth:** Average depth of found solutions
3. **Search Effort:** Programs enumerated to find solutions
4. **Time per Task:** Average enumeration time
5. **Library Size (C):** Number of primitives + learned abstractions
6. **Recognition Accuracy (C):** Neural model prediction quality

### Decision Criteria

After overnight run, use these criteria:

| Metric | Threshold | Interpretation |
|--------|-----------|----------------|
| Solve rate drop > 10% | Primitive is essential |
| Solve rate drop < 5% | Primitive is redundant/derivable |
| Addition solve rate increase > 5% | Consider adding |
| Abstraction recovery success | Library can compensate |

---

## Implementation Notes

### Primitives to Test Adding

These are the "technical" primitives removed in v3 that could help:

```python
ADDITION_CANDIDATES = {
    # Abstract combinators
    'compose': {
        'type': '(b→c) → (a→b) → a → c',
        'impl': lambda f: lambda g: lambda x: f(g(x)),
        'rationale': 'Enables chained transformations without explicit lambda'
    },
    'flip': {
        'type': '(a→b→c) → b → a → c',
        'impl': lambda f: lambda x: lambda y: f(y)(x),
        'rationale': 'Swap argument order for partial application'
    },
    'const': {
        'type': 'a → b → a',
        'impl': lambda x: lambda _: x,
        'rationale': 'Return constant, useful with map/filter'
    },
    'id': {
        'type': 'a → a',
        'impl': lambda x: x,
        'rationale': 'Identity function for composition'
    },

    # Operators
    'neq': {
        'type': 'a → a → bool',
        'impl': lambda x: lambda y: x != y,
        'rationale': 'More direct than not (eq x y)'
    },
    'fold': {
        'type': '(a→b→b) → b → [a] → b',
        'impl': lambda f: lambda z: lambda xs: reduce(lambda b, a: f(a)(b), xs, z),
        'rationale': 'General aggregation, enables sum/max/min without special primitives'
    },
    'tail': {
        'type': '[a] → [a]',
        'impl': lambda xs: xs[1:] if xs else [],
        'rationale': 'Rest of list after head'
    },

    # Constants
    **{f'RANK_{v}': {'type': 'int', 'impl': v, 'rationale': f'Rank value {v}'}
       for v in [10, 11, 12, 13, 14, 17, 21]}
}
```

### Ablation Categories Summary

```python
ABLATION_CATEGORIES = {
    'gestalt_perception': ['all_same_suit', 'all_same_color'],
    'uniqueness_queries': ['n_unique_suits', 'n_unique_ranks', 'n_unique_colors'],
    'membership_queries': ['has_suit', 'has_color'],
    'counting_queries': ['count_suit', 'count_color'],
    'halves': ['first_half', 'second_half'],
    'aggregates': ['sum_ranks', 'max_rank', 'min_rank'],
    'list_slicing': ['take', 'drop', 'half_len', 'adjacent_pairs'],
    'position_access': ['head', 'last', 'at', 'reverse'],
    'higher_order': ['map', 'filter'],  # Keep all, any, unique
}
```

---

## Expected Insights

By morning, we should be able to answer:

1. **Which gestalt primitives are truly essential?**
   - If removing `all_same_suit` causes >10% solve rate drop → essential
   - If removing it causes <5% drop → can be derived from `n_unique_suits`

2. **Do abstract combinators help or hurt?**
   - Adding `compose` may help chain operations
   - May hurt by expanding search space unnecessarily

3. **Can library learning compensate for missing primitives?**
   - If abstraction recovery works → primitives are redundant
   - If not → primitives encode irreducible knowledge

4. **What's the optimal primitive set?**
   - Trade-off: fewer primitives = smaller search space
   - But: some primitives provide essential expressiveness

5. **How much does neural guidance matter?**
   - If guidance helps with minimal library → recognition compensates
   - If not → library choice is fundamental

---

## Launch Command (DO NOT RUN YET)

```bash
nohup caffeinate -d -i -s python3 experiments/run_overnight_primitive_study.py \
    --tracks A B C \
    --output-dir results_overnight_primitive_study \
    > overnight_primitive_study.out 2>&1 &
```

This document serves as the comprehensive game plan. The actual implementation script will be created next.
