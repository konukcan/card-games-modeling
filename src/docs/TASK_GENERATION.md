# Task Generation System

This document explains how training and holdout tasks are generated for the DreamCoder card game modeling system.

---

## Overview

Each **Task** consists of:
- **Training examples** (40): What the model sees during enumeration
- **Holdout examples** (40): Used for verification, never seen during training

A program is only accepted as a solution if it passes **both** training AND holdout.

---

## Architecture: Three Disjoint Pools

The task generation system uses three **completely separate** pools of positive examples to prevent data leakage:

```
┌─────────────────────────────────────────────────────────────────┐
│  TRAINING       │  SEED (hidden)   │  HOLDOUT                   │
│  POSITIVES      │  for near-miss   │  POSITIVES                 │
│  (20 hands)     │  generation      │  (20 hands)                │
│                 │  (20 hands)      │                            │
│  Model sees     │  Model NEVER     │  For verification          │
│  these          │  sees these      │  only                      │
└─────────────────┴──────────────────┴────────────────────────────┘
```

**Why three pools?**

1. **Training positives**: Direct positive examples the model learns from
2. **Seed positives**: Hidden pool used ONLY to generate near-miss negatives
3. **Holdout positives**: Verification set to catch spurious solutions

The separation of seed positives prevents the model from learning the near-miss generation pattern.

---

## Near-Miss Negative Generation

Instead of random negatives, we generate **near-miss negatives** that are "close" to positives:

**Algorithm:**
1. Take a positive hand from the SEED pool (not training!)
2. Flip exactly ONE card to a random different card
3. Check if the resulting hand is negative
4. If yes, use as a training negative

**Example:**
```
Positive (SEED):  [7♥, 7♠, K♣, 2♦, 9♥, 4♠]  → has_pair = TRUE
                      ↓ flip one card
Near-miss negative: [7♥, 3♠, K♣, 2♦, 9♥, 4♠]  → has_pair = FALSE
```

This creates "hard" negatives that force the model to learn precise rule boundaries.

---

## Task Structure

```python
@dataclass
class Task:
    name: str                          # e.g., "poker_has_pair"
    request_type: Type                 # arrow(HAND, BOOL)
    examples: List[Tuple[Hand, bool]]  # Training (20+, 20-)
    holdout: List[Tuple[Hand, bool]]   # Holdout (20+, 20-)
    family: str                        # e.g., "poker"
    difficulty_level: int              # 1-5
    rule_fn: Optional[Callable]        # Original rule function
```

---

## Prerecorded Tasks

Tasks are pre-generated and stored in JSON for reproducibility:

```
src/data/prerecorded_tasks/
├── catalogue_tasks.json     # 45 rules from catalogue.py
├── pretraining_tasks.json   # 44 rules from pretraining_rules.py
└── combined_tasks.json      # All 89 rules (deduplicated)
```

**Generation:**
```bash
cd src
python3 generate_prerecorded_tasks.py
```

**Configuration used:**
```python
TaskGenerationConfig(
    n_training_positives=20,
    n_seed_positives=20,      # Hidden, for near-miss generation
    n_training_negatives=20,
    n_holdout_positives=20,
    n_holdout_negatives=20,
    hand_size=6,
    max_sampling_attempts=200_000,
    use_near_miss_negatives=True,
)
```

---

## Solution Verification

In the wake-sleep loop, solutions are verified on BOTH training and holdout:

```python
# 1. Check training examples
training_correct = sum(
    1 for inp, expected in task.examples
    if eval_program(program, inp) == expected
)

if training_correct == len(task.examples):
    # 2. ALSO check holdout examples
    holdout_correct = sum(
        1 for inp, expected in task.holdout
        if eval_program(program, inp) == expected
    )

    if holdout_correct == len(task.holdout):
        # Accept solution
```

This prevents **spurious solutions** that overfit to training examples.

---

## Visual Inspection

To inspect generated tasks visually, see the PDF report:

```
src/docs/prerecorded_tasks_report.pdf
```

Or regenerate it:
```bash
cd src
python3 generate_tasks_pdf_report.py
```

The report shows:
- Sample hands for each task
- Color-coded positive (green) and negative (red) examples
- Balance statistics
- Near-miss generation success rates

---

## Handling Rare Rules

Some rules have very low positive rates (e.g., `sym_ranks_palindrome` requires specific patterns). The system handles this by:

1. **Increased sampling**: Up to 200,000 attempts per rule
2. **Proportional adjustment**: If not enough positives found, scales down all counts proportionally
3. **Failure reporting**: If < 80% of target positives found, the rule is skipped with a warning

Failed rules are logged in the dataset metadata:
```json
{
  "metadata": {
    "failures": [
      {"rule_id": "sym_ranks_palindrome", "reason": "Could not find enough positives..."}
    ]
  }
}
```

---

## Key Files

| File | Purpose |
|------|---------|
| `dreamcoder_core/task_generation.py` | Main generation logic |
| `dreamcoder_core/task.py` | Task dataclass definition |
| `generate_prerecorded_tasks.py` | Script to regenerate all tasks |
| `generate_tasks_pdf_report.py` | Visual inspection report |
| `data/prerecorded_tasks/*.json` | Pre-generated task datasets |

---

## Guarantees

1. **Balance**: Equal positives and negatives in training set
2. **Disjoint pools**: Training, seed, and holdout are completely separate
3. **Reproducibility**: Same seed produces identical tasks
4. **Near-miss quality**: Negatives differ from positives by exactly one card
5. **Holdout verification**: Solutions must generalize beyond training examples
