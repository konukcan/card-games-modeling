# Overnight Wake-Sleep Library Study

**Goal:** Determine optimal primitive library by testing variants under **real wake-sleep conditions**, including neural transfer (recognition model) and library evolution (compression/abstraction).

**Runtime:** 10-12 hours (overnight)

---

## Why Full Wake-Sleep Matters

The previous ablation study (Track A/B/C) only tested **static enumeration** - measuring how many tasks could be solved with different library configurations in a single pass. This misses two critical dynamics:

### 1. Recognition Model Transfer
After solving some tasks, the recognition model learns which primitives are useful for which task patterns:
```
Task "Uniform_color" → model learns to predict: all_same_color, get_color, eq
Task "Sorted_ranks" → model learns to predict: adjacent_pairs, lt, all
```

If we remove `all_same_color`, does the model learn to predict the longer composition `eq 1 (n_unique_colors hand)` instead? **This requires multi-iteration training to observe.**

### 2. Abstraction Compounding
The compression phase finds shared patterns across solutions:
```
Iteration 1: Solves 10 tasks using (all_same_suit hand) pattern
Iteration 2: Compression creates #(λ. all_same_suit $0) abstraction
Iteration 3: New tasks can use this abstraction directly
```

Without running full wake-sleep, we can't test whether removing gestalt primitives leads to the model **rediscovering** equivalent abstractions through compression.

---

## Experimental Design

### Track A: Ablation Experiments

| Variant | Description | Primitives | Hypothesis |
|---------|-------------|------------|------------|
| `baseline_full_library` | Current full library | 59 | Baseline for comparison |
| `minimal_no_gestalt` | Remove all gestalt primitives | 48 | Forces compositional solutions |

**Gestalt primitives removed:**
- `all_same_suit`, `all_same_color` (perception)
- `n_unique_suits`, `n_unique_ranks`, `n_unique_colors` (uniqueness)
- `has_suit`, `has_color` (membership)
- `count_suit`, `count_color` (counting)
- `first_half`, `second_half` (halves)

### Track B: Addition Experiments

| Variant | Description | Primitives | Hypothesis |
|---------|-------------|------------|------------|
| `minimal_plus_combinators` | Minimal + abstract combinators | 52 | Technical primitives may compensate for removed gestalt |

**Primitives added:**
- `compose`: Function composition - enables chained transformations
- `flip`: Argument swap - enables different partial application patterns
- `fold`: General aggregation - can express sum, max, count
- `neq`: Not-equal - more direct than `not (eq x y)`

---

## What We Measure (Comprehensive Logging)

### Per-Iteration Metrics

| Category | Metric | Why It Matters |
|----------|--------|----------------|
| **Wake** | Programs/second | Enumeration efficiency |
| | Tasks solved (cumulative) | Primary outcome |
| | Newly solved this iteration | Learning progress |
| **Recognition** | Loss per epoch | Training dynamics |
| | Per-task predictions | Detect model collapse |
| | Prediction entropy | Diversity of predictions |
| **Compression** | Abstractions learned | Library evolution |
| | Abstraction size/savings | Quality of abstractions |
| **Dreaming** | Dreams generated | Synthetic training data |
| | Contrastive ratio | Near-miss vs standard |

### Model Collapse Detection

**What is model collapse?**
If the recognition model predicts the same primitives for every task (e.g., always predicting `eq`, `filter`, `map`), it provides no search guidance. We detect this by:

1. **Prediction entropy**: High entropy across all tasks = collapsed
2. **Top-5 overlap**: If all tasks have identical top-5, model is collapsed
3. **Per-task analysis**: Save predictions for every task at every iteration

### Saved Artifacts (Per Iteration)

```
results_overnight_wakesleep/study_YYYYMMDD_HHMMSS/
├── experiment_config.json          # Full configuration
├── baseline_full_library/
│   ├── iter_01_log.json            # Iteration summary
│   ├── iter_02_log.json
│   ├── ...
│   ├── checkpoints/
│   │   ├── model_iter_01.pt        # Model weights
│   │   ├── model_iter_02.pt
│   │   ├── embeddings_iter_01.json # Task embeddings (for t-SNE)
│   │   ├── embeddings_iter_02.json
│   │   └── ...
│   ├── predictions/
│   │   ├── iter_01_predictions.json # Per-task predictions
│   │   ├── iter_02_predictions.json
│   │   └── ...
│   ├── dreams/
│   │   ├── iter_02_dreams.json     # Dream program contents
│   │   └── ...
│   └── final_result.json           # Complete result
├── minimal_no_gestalt/
│   └── ... (same structure)
├── minimal_plus_combinators/
│   └── ... (same structure)
└── summary.json                    # Cross-variant comparison
```

---

## Time Budget

| Phase | Time per Iteration | 6 Iterations |
|-------|-------------------|--------------|
| Wake (enumeration) | ~25 min | 2.5 hours |
| Sleep (compression) | ~3 min | 18 min |
| Sleep (recognition) | ~8 min | 48 min |
| Sleep (dreaming) | ~5 min | 30 min |
| Analysis/saving | ~2 min | 12 min |
| **Total per variant** | | **~4 hours** |

**3 variants × 4 hours = 12 hours total**

---

## Expected Insights

### Question 1: Are gestalt primitives essential or derivable?

**If baseline >> minimal:**
→ Gestalt primitives encode irreducible domain knowledge
→ They're worth keeping despite larger search space

**If baseline ≈ minimal (after 6 iterations):**
→ Gestalt primitives are syntactic sugar
→ Compression can recover equivalent abstractions
→ Consider removing them to reduce search space

### Question 2: Do technical combinators help?

**If minimal+combinators > minimal:**
→ Abstract combinators provide useful expressiveness
→ `fold` can compensate for missing aggregates
→ Consider adding to final library

**If minimal+combinators ≈ minimal:**
→ Combinators add search space overhead
→ Not worth the complexity

### Question 3: How does recognition transfer evolve?

**Watch for:**
- Does mean prediction entropy decrease over iterations? (learning)
- Does entropy variance increase? (task differentiation)
- Do predictions for unsolved tasks change? (generalization)

### Question 4: What abstractions emerge?

**If minimal library learns gestalt-equivalent abstractions:**
- Document the patterns: Are they identical or different structure?
- Measure: How many iterations until emergence?
- Compare: Abstraction quality vs hand-designed primitives

---

## Launch Commands

### Full overnight run (recommended)
```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling/src
nohup caffeinate -d -i -s python3 experiments/run_overnight_wakesleep_study.py \
    --variants baseline_full_library minimal_no_gestalt minimal_plus_combinators \
    --iterations 6 \
    --budget 150000 \
    > overnight_wakesleep.out 2>&1 &
echo "PID: $!"
```

### Quick test (30 min)
```bash
python3 experiments/run_overnight_wakesleep_study.py \
    --variants baseline_full_library minimal_no_gestalt \
    --quick-test
```

### Dry run (show config only)
```bash
python3 experiments/run_overnight_wakesleep_study.py --dry-run
```

### Monitor progress
```bash
tail -f overnight_wakesleep.out
```

---

## Post-Experiment Analysis

After the run completes:

### 1. Compare solve rates
```python
import json
with open('results_overnight_wakesleep/study_*/summary.json') as f:
    summary = json.load(f)
for name, data in summary['variants'].items():
    print(f"{name}: {data['solve_rate']:.1%} ({data['solved']} tasks)")
```

### 2. Visualize task embeddings (t-SNE)
```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load embeddings from final iteration
with open('.../checkpoints/embeddings_iter_06.json') as f:
    embeddings = json.load(f)

X = np.array([embeddings[t] for t in sorted(embeddings)])
X_tsne = TSNE(n_components=2).fit_transform(X)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1])
```

### 3. Analyze model collapse
```python
# Load all predictions
import glob
files = sorted(glob.glob('.../predictions/iter_*_predictions.json'))
for f in files:
    with open(f) as fp:
        preds = json.load(fp)
    entropies = [p['prediction_entropy'] for p in preds]
    print(f"{f}: mean entropy = {np.mean(entropies):.4f}")
```

### 4. Review learned abstractions
```python
with open('.../final_result.json') as f:
    result = json.load(f)
print(f"Learned {len(result['all_abstractions_learned'])} abstractions:")
for a in result['all_abstractions_learned']:
    print(f"  {a}")
```

---

## Decision Criteria

| Outcome | Recommendation |
|---------|----------------|
| Baseline solves >60%, minimal solves <40% | Keep gestalt primitives |
| Baseline ≈ minimal within 10% | Remove gestalt primitives |
| Minimal+combinators > minimal by >15% | Add technical combinators |
| Minimal learns equivalent abstractions | Gestalt are derivable |
| Model collapse detected (entropy flat) | Recognition needs tuning |

---

## Appendix: Technical Primitive Definitions

```python
# compose: (b→c) → (a→b) → a → c
# Enables: (compose f g x) = f(g(x))
# Example: (compose length filter) = count

# flip: (a→b→c) → b → a → c
# Enables: (flip f x y) = f y x
# Example: (flip eq 5) checks if something equals 5

# fold: (a→b→b) → b → [a] → b
# Enables: (fold + 0 [1,2,3]) = 6
# Can express: sum, max, count, any, all

# neq: a → a → bool
# More direct than: (not (eq x y))
```
