# Recognition × Dream Strategy Experiment Plan

**Created**: December 2024
**Status**: Ready to launch (pending model review)
**Purpose**: Compare recognition model architectures crossed with dream generation strategies

---

## Overview

This experiment systematically compares two recognition model architectures against three dream generation strategies, creating a 2×3 factorial design with 6 experimental conditions.

### Research Questions

1. **Does contrastive recognition outperform GRU-based recognition?**
   - The contrastive model uses τ = mean(pos) - mean(neg) encoding
   - The GRU model pools examples without distinguishing pos/neg

2. **Do near-miss dreams provide better training signal than random dreams?**
   - Standard: whatever balance emerges naturally
   - Balanced: equal pos/neg, but random negatives
   - Contrastive: equal pos/neg, with near-miss negatives (1 card different)

3. **Is there an interaction between recognition architecture and dream strategy?**
   - Perhaps contrastive recognition benefits more from contrastive dreams?
   - Perhaps GRU recognition needs the diversity of standard dreams?

---

## Experimental Design

### Independent Variables

#### Factor A: Recognition Model (2 levels)

| Level | Model | Key Features |
|-------|-------|--------------|
| `gru` | GRU-based (legacy) | Bidirectional GRU, pools all examples, softmax output |
| `contrastive` | Contrastive (new) | Factored embeddings, τ = mean(pos)-mean(neg), sigmoid output, structural similarity loss |

#### Factor B: Dream Strategy (3 levels)

| Level | Strategy | Positive Examples | Negative Examples |
|-------|----------|-------------------|-------------------|
| `standard` | StandardDreamer | Whatever emerges | Whatever emerges |
| `balanced` | BalancedDreamer | Rejection-sampled N/2 | Rejection-sampled N/2 (random) |
| `contrastive` | ContrastiveDreamer | Rejection-sampled N/2 | Near-miss (1 card different) |

### Conditions (6 total)

| Condition | Recognition | Dreams | Nickname |
|-----------|-------------|--------|----------|
| 1 | GRU | Standard | `gru_standard` |
| 2 | GRU | Balanced | `gru_balanced` |
| 3 | GRU | Contrastive | `gru_contrastive` |
| 4 | Contrastive | Standard | `contrastive_standard` |
| 5 | Contrastive | Balanced | `contrastive_balanced` |
| 6 | Contrastive | Contrastive | `contrastive_contrastive` |

### Dependent Variables

Primary metrics (per iteration):
- **Tasks solved**: Number of rules with perfect programs found
- **Programs enumerated**: Total search effort
- **Recognition loss**: Training loss of recognition model
- **Time per iteration**: Wall-clock time

Secondary metrics:
- **Learning curve**: Tasks solved over iterations
- **Transfer effects**: Does solving task A help solve task B?
- **Abstraction quality**: Inventions discovered, reuse frequency
- **Search efficiency**: Programs per solution

---

## Task Set

Use the full pretraining rules set from `rules/pretraining_rules.py`:

### Easy Rules (22 rules)
- Uniform patterns: Uniform_color, Uniform_suit
- Endpoint patterns: Ends_same_color, Ends_same_suit, Ends_same_rank
- Count patterns: Has_pair, Has_three_of_a_kind, Has_four_of_a_kind
- Sequence patterns: Sorted_asc, Sorted_desc
- All_same variants, All_diff variants
- And more...

### Hard Rules (21 additional rules)
- Palindrome patterns: Suits_palindrome, Colors_palindrome
- Complex logic: Exactly_one_pair, Exactly_two_pairs
- Combined patterns: Monotonic_ranks, Alternating_colors
- And more...

**Total**: 43 rules

---

## Hyperparameters

### Fixed across all conditions

```python
# Wake phase
enumeration_budget = 200_000      # Programs per task
enumeration_timeout = 120.0       # Seconds per task
max_depth = 8                     # AST depth limit

# Compression
use_compression = True
max_inventions_per_iteration = 5
min_compression_savings = 2.0

# Dreaming
dreams_per_iteration = 100
dream_temperature = 1.0
n_examples_per_dream = 10

# Training
keep_top_k = 5                    # Solutions per task
max_iterations = 10

# Task examples
n_examples_per_task = 100
n_holdout_per_task = 20
```

### Recognition-specific

```python
# GRU model
gru_hidden_dim = 128
gru_learning_rate = 1e-3
gru_epochs_per_iteration = 10

# Contrastive model
contrastive_card_out = 32
contrastive_pred_hidden = 64
contrastive_learning_rate = 1e-3
contrastive_epochs_per_iteration = 10
structural_similarity_weight = 0.1
```

---

## Implementation

### File Structure

```
experiments/
├── recognition_dream_experiment_plan.md    # This file
├── run_recognition_dream_experiment.py     # Main runner script
├── results/
│   ├── gru_standard/
│   │   ├── run_1/
│   │   ├── run_2/
│   │   └── run_3/
│   ├── gru_balanced/
│   ├── gru_contrastive/
│   ├── contrastive_standard/
│   ├── contrastive_balanced/
│   └── contrastive_contrastive/
└── analysis/
    ├── learning_curves.py
    ├── statistical_tests.py
    └── generate_figures.py
```

### Runner Script Outline

```python
# run_recognition_dream_experiment.py

CONDITIONS = [
    {'recognition': 'gru', 'dreams': 'standard'},
    {'recognition': 'gru', 'dreams': 'balanced'},
    {'recognition': 'gru', 'dreams': 'contrastive'},
    {'recognition': 'contrastive', 'dreams': 'standard'},
    {'recognition': 'contrastive', 'dreams': 'balanced'},
    {'recognition': 'contrastive', 'dreams': 'contrastive'},
]

N_RUNS_PER_CONDITION = 3  # For statistical reliability

def run_condition(recognition_type, dream_strategy, run_id):
    # 1. Build grammar
    grammar = build_lean_grammar()

    # 2. Create tasks from rules
    tasks = create_tasks_from_rules(get_all_pretraining_rules())

    # 3. Initialize recognition model
    if recognition_type == 'gru':
        recognition = NeuralRecognitionModel(grammar, hidden_dim=128)
    else:
        recognition = ContrastiveRecognitionModel(grammar, card_out=32)

    # 4. Initialize dreamer
    dreamer = ConfigurableDreamer(
        grammar, eval_fn, sample_hand_fn, sample_card_fn,
        strategy=dream_strategy
    )

    # 5. Run wake-sleep loop
    results = run_wake_sleep(
        grammar, tasks, recognition, dreamer,
        max_iterations=10,
        log_dir=f"results/{recognition_type}_{dream_strategy}/run_{run_id}"
    )

    return results

def main():
    for condition in CONDITIONS:
        for run_id in range(1, N_RUNS_PER_CONDITION + 1):
            run_condition(
                condition['recognition'],
                condition['dreams'],
                run_id
            )
```

---

## Analysis Plan

### 1. Learning Curves

For each condition, plot:
- X-axis: Iteration (1-10)
- Y-axis: Tasks solved (0-43)
- Lines: Mean across runs, shaded regions for ±1 SE

Compare:
- GRU vs Contrastive (collapsing across dream strategies)
- Standard vs Balanced vs Contrastive dreams (collapsing across recognition)
- All 6 conditions

### 2. Final Performance

After 10 iterations:
- Bar plot: Tasks solved by condition
- Statistical tests: 2-way ANOVA (Recognition × Dreams)
- Post-hoc: Tukey HSD for pairwise comparisons

### 3. Search Efficiency

- Programs enumerated per solution
- Time per task
- Compare enumeration efficiency across conditions

### 4. Recognition Model Analysis

For contrastive model:
- Task embedding clusters (t-SNE)
- Primitive prediction accuracy
- Structural similarity loss trajectory

For GRU model:
- Task embedding clusters (t-SNE)
- Primitive prediction accuracy

### 5. Abstraction Analysis

- Number of inventions per condition
- Invention reuse frequency
- Quality of discovered abstractions

---

## Expected Outcomes

### Hypotheses

**H1**: Contrastive recognition outperforms GRU recognition
- Rationale: τ = mean(pos) - mean(neg) captures decision boundary directly

**H2**: Contrastive dreams outperform balanced dreams outperform standard dreams
- Rationale: Near-miss negatives provide maximum information about decision boundary

**H3**: Interaction effect: Contrastive recognition benefits more from contrastive dreams
- Rationale: The contrastive encoding τ is specifically designed to leverage pos/neg differences

### Alternative outcomes

- GRU may benefit from dream diversity (standard strategy)
- Balanced dreams may be sufficient (no need for near-miss complexity)
- Recognition architecture may matter more than dream strategy (or vice versa)

---

## Resource Estimates

### Per condition (1 run)
- ~10 iterations × ~5 minutes/iteration = ~50 minutes
- Memory: ~2GB (models + grammar)

### Full experiment
- 6 conditions × 3 runs = 18 runs
- 18 runs × 50 minutes = ~15 hours total
- Can parallelize across machines

### Recommended execution
```bash
# Use caffeinate to prevent sleep
nohup caffeinate -d -i -s python3 run_recognition_dream_experiment.py \
    > experiment_log.txt 2>&1 &
```

---

## Checkpoints and Recovery

Each run saves:
1. Per-iteration checkpoints (model weights, frontiers)
2. Final results JSON
3. Log file with timestamps

If interrupted:
```python
# Resume from last checkpoint
resume_experiment(run_dir="results/gru_contrastive/run_2")
```

---

## Launch Checklist

Before launching:

- [ ] Verify all dreamers work: `python3 dreamcoder_core/contrastive_dreaming.py`
- [ ] Verify contrastive recognition: `python3 dreamcoder_core/contrastive_recognition.py`
- [ ] Verify wake-sleep integration: `python3 dreamcoder_core/contrastive_wake_sleep.py`
- [ ] Verify GRU recognition still works: `python3 dreamcoder_core/neural_recognition.py`
- [ ] Create runner script: `experiments/run_recognition_dream_experiment.py`
- [ ] Test on 2-3 rules first (quick smoke test)
- [ ] Verify caffeinate is working
- [ ] Ensure sufficient disk space (~1GB per condition)

---

## Notes

### Why three dream strategies?

The three strategies isolate different factors:

1. **Standard vs Balanced**: Effect of class balance
   - Standard: imbalanced (e.g., 95% negative for rare patterns)
   - Balanced: guaranteed 50/50

2. **Balanced vs Contrastive**: Effect of near-miss
   - Both have 50/50 balance
   - Difference is whether negatives are random or near-miss

This allows us to attribute effects to either:
- Class balance (Standard → Balanced)
- Near-miss information (Balanced → Contrastive)

### Why include GRU baseline?

The GRU model is the standard DreamCoder architecture. Including it allows:
- Comparison with prior work
- Attribution of improvements to contrastive encoding vs other factors
- Understanding if contrastive dreams help even with standard architecture
