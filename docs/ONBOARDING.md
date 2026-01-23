# Onboarding Guide

Welcome to the DreamCoder Card Game Modeling project! This guide will help you understand the project structure and get started with the codebase.

**Last Updated**: January 2025

---

## Project Overview

This is a DreamCoder-inspired program synthesis system for modeling how humans learn card game rules. The system:

1. **Enumerates programs** in a domain-specific language (DSL) of card operations
2. **Learns a recognition model** that predicts which primitives are useful for each task
3. **Extracts abstractions** (library learning) to compress discovered solutions
4. **Iteratively improves** through a wake-sleep training loop

### Research Questions

- Can program induction mechanisms explain human self-explanation effects?
- How do learned abstractions evolve over training iterations?
- What primitives are critical for different rule families?

---

## Quick Start

### 1. Environment Setup

```bash
# Navigate to the project
cd card-games-modelling/src

# Ensure Python 3.9+ is available
python3 --version

# Install dependencies (if not already done)
pip install torch numpy matplotlib tqdm
```

### 2. Run a Small Experiment

```bash
# Quick recognition model evaluation (< 5 minutes)
python3 experiments/train_and_evaluate_recognition.py
```

### 3. Full Overnight Run

```bash
cd src
nohup caffeinate -d -i -s python3 experiments/run_reference_wakesleep.py > overnight.out 2>&1 &
echo $!  # Note the PID

# Monitor progress
tail -f overnight.out
```

---

## Directory Structure

```
card-games-modelling/
├── src/                              # Main source code
│   ├── dreamcoder_core/              # Core DreamCoder modules
│   │   ├── primitives.py             # ★ Authoritative primitives (57 ops)
│   │   ├── enumeration.py            # Program enumeration
│   │   ├── contrastive_recognition.py # ★ Primary recognition model
│   │   ├── grammar.py                # Probabilistic grammar
│   │   └── wake_sleep.py             # Wake-sleep loop
│   │
│   ├── rules/                        # Card game rules
│   │   ├── catalogue.py              # 45 core rules
│   │   ├── pretraining_rules.py      # 44 alternative rules
│   │   └── cards.py                  # Card representation
│   │
│   ├── experiments/                  # Experiment scripts
│   │   ├── archive/                  # Archived diagnostic scripts
│   │   └── *.py                      # Active experiments
│   │
│   └── experiments/                  # Experiment scripts
│       └── run_reference_wakesleep.py  # ★ Primary entry point
│
├── docs/                             # Documentation
│   ├── ARCHITECTURE.md               # System architecture
│   ├── KNOWN_ISSUES.md               # Bug documentation
│   └── ONBOARDING.md                 # This file
│
└── archived/                         # Archived code
    ├── legacy_runners/               # Old runner scripts
    └── legacy_recognition/           # Superseded recognition models
```

---

## Key Concepts

### 1. Tasks

A **task** is a card rule to learn:
- **Positive examples**: Hands that satisfy the rule
- **Negative examples**: Hands that don't satisfy the rule
- **Holdout set**: Unseen examples for verification

```python
from rules.catalogue import get_all_rules
rules = get_all_rules()
print(f"Total rules: {len(rules)}")  # 45 rules
```

### 2. Primitives

**Primitives** are the building blocks of programs (60 total):
- Constants (TRUE, FALSE, 0-5, SPADES, HEARTS, RED, BLACK, etc.)
- Card accessors (get_rank, get_suit, get_color, rank_val)
- List ops (head, last, take, drop, map, filter, all, any)
- Queries (has_suit, count_color, n_unique_suits, etc.)
- Comparisons and arithmetic (eq, lt, +, -, mod)

```python
from dreamcoder_core.primitives import build_lean_grammar
grammar = build_lean_grammar()
print(f"Total primitives: {len(grammar.primitives)}")  # 57
```

### 3. Programs

**Programs** are compositions of primitives:
```
(all (map get_color $0) (lambda (eq $0 RED)))
```
This checks if all cards are red.

### 4. Grammar

The **grammar** assigns probabilities to primitives:
```python
from dreamcoder_core.primitives import build_lean_grammar
grammar = build_lean_grammar()
```

### 5. Recognition Model

The **recognition model** predicts useful primitives for each task:
```python
from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
model = ContrastiveRecognitionModel(n_primitives=60, output_mode='softmax')
```

---

## Common Workflows

### Running Experiments

1. **Wake-sleep training**:
   ```bash
   cd src
   nohup caffeinate -d -i -s python3 experiments/run_reference_wakesleep.py > overnight.out 2>&1 &
   ```

2. **Quick test** (reduced iterations):
   ```bash
   python3 experiments/run_reference_wakesleep.py --quick --verbose 3
   ```

### Analyzing Results

Results are saved in `src/results_reference/`:
```
run_YYYYMMDD_HHMMSS/
├── iter_01/               # Per-iteration checkpoints
│   ├── model.pt           # Recognition model
│   ├── grammar.json       # Grammar state
│   ├── frontiers.json     # Solved programs
│   └── metrics.json       # Performance metrics
├── iter_02/
└── final_results.json     # Summary
```

### Debugging

1. **Check logs**: `tail -f overnight.out`
2. **Check process**: `ps aux | grep python`
3. **Check known issues**: `src/KNOWN_ISSUES.md`

---

## Critical Things to Know

### 1. Always Use Memoized Enumeration

```python
# GOOD - 1000x faster
for prog, cost in enumerator.enumerate_memoized(...):
    ...

# BAD - naive enumeration is very slow
for prog, cost in enumerator.enumerate(...):
    ...
```

### 2. Always Use Caffeinate for Long Runs

```bash
# System will sleep without this!
nohup caffeinate -d -i -s python3 script.py > output.out 2>&1 &
```

### 3. Task-Result Mapping in Parallel Code

```python
# CORRECT - use dictionary keyed by task name
results_by_name = {}
for future in as_completed(futures):
    result = future.result()
    results_by_name[result['task_name']] = result

# WRONG - order is not preserved!
results = [f.result() for f in as_completed(futures)]
```

### 4. Authoritative Files

| Purpose | File |
|---------|------|
| Primitives | `dreamcoder_core/primitives.py` |
| Recognition model | `dreamcoder_core/contrastive_recognition.py` |
| Rules | `rules/catalogue.py` |
| Main runner | `experiments/run_reference_wakesleep.py` |
| Known bugs | `docs/KNOWN_ISSUES.md` |

---

## Rule Families

Rules are classified by difficulty (see `docs/rule_difficulty_classification.md`):

| Phase | Difficulty | Count | Example |
|-------|------------|-------|---------|
| 1 | Easy | 8 | `Uniform_color` (all same color) |
| 2 | Medium | 12 | `Has_Ace_of_Spades`, `Sorted_by_rank` |
| 3 | Hard | 18 | `Suits_palindrome`, `Halves_copy_suits` |
| 4 | Very Hard | 7 | `Well_formed_brackets_by_suit` |

**Key Finding**: Adding list primitives (take, drop, zip_with) improved solve rate from ~8/45 to ~30-40/45.

---

## Getting Help

1. **Documentation**: Check `docs/` directory
2. **Known Issues**: See `docs/KNOWN_ISSUES.md`
3. **Architecture**: See `docs/ARCHITECTURE.md`
