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

### 2. Run the Demo

```bash
# Interactive demo showing the system in action
python3 main_demo.py
```

### 3. Run a Small Experiment

```bash
# Quick recognition model evaluation (< 5 minutes)
python3 experiments/train_and_evaluate_recognition.py
```

### 4. Full Overnight Run

```bash
# IMPORTANT: Always use caffeinate for long runs!
nohup caffeinate -d -i -s python3 run_incremental_wakesleep.py > overnight.out 2>&1 &
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
│   │   ├── lean_primitives.py        # ★ Authoritative primitives (60 ops)
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
│   ├── logs/                         # Organized log files
│   │   ├── production/               # Main runs
│   │   ├── experiments/              # Experiment logs
│   │   └── historical/               # Important historical logs
│   │
│   ├── run_incremental_wakesleep.py  # ★ Current wake-sleep runner
│   ├── run_progressive_wakesleep.py  # Alternative curriculum runner
│   └── generate_systematic_report.py # Report generation
│
├── docs/                             # Documentation
│   ├── MODULE_STATUS.md              # Module reference
│   ├── FEATURE_STATUS.md             # Feature reference
│   ├── rule_difficulty_classification.md  # Rule taxonomy
│   └── ...                           # Other docs
│
├── archived/                         # Archived code
│   ├── legacy_runners/               # Old runner scripts
│   └── legacy_recognition/           # Superseded recognition models
│
└── CLAUDE.md                         # Agent guidelines
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
- Level 0: Constants (TRUE, FALSE, 0-13, SPADES, etc.)
- Level 1: Basic ops (eq, lt, add, sub, etc.)
- Level 2: Card accessors (get_rank, get_suit, get_color)
- Level 3: List ops (map, filter, all, any, etc.)
- Level 4: Aggregates (count, sum, unique, etc.)

```python
from dreamcoder_core.lean_primitives import ALL_PRIMITIVES
print(f"Total primitives: {len(ALL_PRIMITIVES)}")  # 60
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
from dreamcoder_core.grammar import Grammar
grammar = Grammar.uniform(ALL_PRIMITIVES)
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

1. **Wake-sleep training** (current recommended approach):
   ```bash
   # Incremental wake-sleep with ContrastiveRecognitionModel
   nohup caffeinate -d -i -s python3 run_incremental_wakesleep.py > overnight.out 2>&1 &

   # Or progressive curriculum
   nohup caffeinate -d -i -s python3 run_progressive_wakesleep.py > overnight.out 2>&1 &
   ```

2. **Generate report**:
   ```bash
   python3 generate_systematic_report.py --run-dir results_*/
   ```

3. **Quick evaluation** (experiments directory):
   ```bash
   python3 experiments/train_and_evaluate_recognition.py
   ```

### Analyzing Results

Results are saved in `results_*/` directories:
```
results_overnight_20250101_123456/
├── results.json           # Main results data
├── report.html            # Visual report
├── models/                # Saved model checkpoints
│   └── recognition_iter_N.pt
└── iteration_N/           # Per-iteration data
```

### Debugging

1. **Check logs**: `tail -f overnight.out`
2. **Check system**: `ps aux | grep python` and `ps aux | grep caffeinate`
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
| Primitives | `dreamcoder_core/lean_primitives.py` |
| Recognition model | `dreamcoder_core/contrastive_recognition.py` |
| Rules | `rules/catalogue.py` |
| Main runners | `run_incremental_wakesleep.py`, `run_progressive_wakesleep.py` |
| Known bugs | `KNOWN_ISSUES.md` |

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
2. **Known Issues**: See `src/KNOWN_ISSUES.md`
3. **Module Status**: See `docs/MODULE_STATUS.md`
4. **Feature Status**: See `docs/FEATURE_STATUS.md`
5. **Agent Guidelines**: See `CLAUDE.md`

---

## Next Steps for New Contributors

1. **Run the demo** (`main_demo.py`) to see the system in action
2. **Read MODULE_STATUS.md** to understand the codebase
3. **Read KNOWN_ISSUES.md** to learn from past bugs
4. **Try a small experiment** with reduced iterations
5. **Explore results** using `generate_systematic_report.py`

Welcome to the project!
