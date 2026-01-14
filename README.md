# Card Game Rule Learning: DreamCoder Modeling

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Directory Structure](#directory-structure)
- [Core Components](#core-components)
- [Running Experiments](#running-experiments)
- [Configuration Reference](#configuration-reference)
- [Results & Reporting](#results--reporting)
- [Troubleshooting](#troubleshooting)
- [Development Guide](#development-guide)
- [Research Context](#research-context)
- [Citation](#citation)

---

## Overview

This repository implements a computational model of rule learning using **program synthesis**. The system learns to synthesize programs (rules) that classify card hands as satisfying or not satisfying compositional predicates.

### Key Features

| Feature | Description |
|---------|-------------|
| **57 Primitives in 10 Categories** | Cognitively-realistic DSL for card game rules |
| **Memoized Enumeration** | 1000x+ speedup over naive search |
| **Contrastive Recognition** | Neural network predicts useful primitives from examples |
| **Library Learning** | MDL-based compression with quality filters |
| **Wake-Sleep Training** | Iterative improvement loop |

### Connection to DreamCoder

This implementation follows the architecture of [Ellis et al. (2023)](https://doi.org/10.1098/rsta.2022.0050):

> Ellis, K., Wong, L., Nye, M., et al. **DreamCoder: growing generalizable, interpretable knowledge with wake–sleep Bayesian program learning.** *Phil. Trans. R. Soc. A*, 381: 20220050.

**Key adaptations for this domain:**
- **Domain**: Card game predicates instead of list/graphics/text tasks
- **DSL**: 10-category compositional grammar designed for cognitive realism
- **Recognition**: Predicts primitives from task examples (not full programs)
- **Compression**: Quality filters prevent degenerate abstractions

---

## Quick Start

### Prerequisites

- Python 3.9+
- PyTorch 1.12+
- (Optional) PyPy 3.9+ for accelerated enumeration workers

### Installation

```bash
git clone https://github.com/[your-username]/card-games-modelling.git
cd card-games-modelling
pip install -r requirements.txt
```

### Run a Quick Demo

```bash
cd src
python ../examples/main_demo.py
```

This runs a small demo showing rule loading, task generation, and visualization.

### Run a Full Experiment

```bash
cd src

# Launch canonical wake-sleep experiment with caffeinate (REQUIRED for runs > 30 min)
nohup caffeinate -d -i -s python3 experiments/run_reference_wakesleep.py > overnight.out 2>&1 &

# Quick test run (15 minutes)
python3 experiments/run_reference_wakesleep.py --quick --verbose 3

# Monitor progress
tail -f overnight.out
```

### Generate a Report

```bash
cd src
python generate_systematic_report.py --run-dir ../results/run_YYYYMMDD_HHMMSS/
```

---

## Architecture

### System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DreamCoder System                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐  │
│  │   TASKS     │───▶│  RECOGNITION    │───▶│      ENUMERATION            │  │
│  │  (45 rules) │    │    MODEL        │    │   (Best-first search)       │  │
│  │             │    │                 │    │                             │  │
│  │ + examples  │    │ τ = pos - neg   │    │  P(prog) × P(prims|task)   │  │
│  │ - examples  │    │ → P(primitive)  │    │  Memoized (1000x speedup)   │  │
│  └─────────────┘    └─────────────────┘    └──────────────┬──────────────┘  │
│                                                           │                  │
│                                                           ▼                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         FRONTIERS                                    │    │
│  │           Programs that solve each task (solutions)                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│              ┌───────────────┴───────────────┐                              │
│              ▼                               ▼                              │
│  ┌─────────────────────┐        ┌─────────────────────────────────────┐    │
│  │    COMPRESSION      │        │     SLEEP (Recognition Training)    │    │
│  │                     │        │                                     │    │
│  │ Extract patterns    │        │ Train on solved tasks               │    │
│  │ MDL scoring         │        │ Multi-hot primitive labels          │    │
│  │ Quality filters     │        │ Contrastive + count + bigram loss   │    │
│  │ Update grammar      │        │                                     │    │
│  └─────────────────────┘        └─────────────────────────────────────┘    │
│              │                               │                              │
│              └───────────────┬───────────────┘                              │
│                              ▼                                               │
│                     ┌─────────────────┐                                     │
│                     │  ITERATE        │                                     │
│                     │  (Wake-Sleep)   │                                     │
│                     └─────────────────┘                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Recognition Model Architecture

The recognition model predicts which primitives are useful for a task based on its positive and negative examples. The key insight is **contrastive encoding**: by subtracting the mean negative embedding from the mean positive embedding, we get a vector that captures what distinguishes "yes" hands from "no" hands.

**Stage 1: Card → Vector (Factored Embeddings)**

Each card is represented by three learned embeddings concatenated together:

| Property | Values | Embedding Dims |
|----------|--------|----------------|
| Suit | ♣ ♦ ♥ ♠ (4) | 8 |
| Rank | 2-A (13) | 16 |
| Position | 1st-8th (8) | 8 |
| **Total** | | **32 per card** |

*Why factored?* A single embedding table for all 52×8 card-position combinations would need 416 vectors. Factoring lets us learn just 4+13+8=25 embeddings that combine compositionally.

**Stage 2: Hand → Vector (Mean Pooling)**

Each card embedding passes through an MLP, then we average across all cards:
```
cards → MLP(32 → 64 → 32) per card → mean pool → 32-dim hand vector
```

**Stage 3: Task → Vector (Contrastive Encoding)**

A task has positive examples (hands satisfying the rule) and negative examples. We compute:
```
τ = mean(positive hand vectors) − mean(negative hand vectors)
```

This difference vector encodes the decision boundary. For example, if the rule is "all red cards," positive hands have high "redness" in their embedding, negatives don't, and τ captures this contrast.

**Stage 4: Vector → Primitive Probabilities**

An MLP maps the 32-dim task vector to probabilities over 57 primitives:
```
τ → Linear(32 → 64) → ReLU → Linear(64 → 57) → Softmax → P(primitive | task)
```

These probabilities guide program search by biasing toward likely-useful primitives.

### Data Flow

```
Behavioral Experiment (card-games/)
           │
           ▼
   Participant responses (.csv)
           │
           ▼
   Model fitting (this repo)
           │
           ▼
   Predicted learning curves
           │
           ▼
   Compare to human data
```

---

## Directory Structure

```
card-games-modelling/
├── src/                              # Main implementation
│   ├── dreamcoder_core/              # Core DreamCoder components
│   │   ├── primitives.py             # 🔑 AUTHORITATIVE primitive library (57 primitives)
│   │   ├── contrastive_recognition.py# 🔑 PRIMARY recognition model
│   │   ├── enumeration.py            # TopDownEnumerator with memoization
│   │   ├── compression/              # Library learning package (10 modules)
│   │   │   ├── compress.py           # Main compression algorithm
│   │   │   ├── quality_filters.py    # Abstraction quality checks
│   │   │   ├── recognition_guided.py # Recognition-guided compression
│   │   │   ├── mdl_scoring.py        # MDL-based scoring
│   │   │   └── ...                   # Anti-unification, rewriting, etc.
│   │   ├── grammar.py                # Probabilistic context-free grammar
│   │   ├── program.py                # Program representation and parsing
│   │   ├── type_system.py            # Hindley-Milner type system
│   │   ├── task.py                   # Task definition
│   │   └── wake_sleep.py             # Wake-sleep training loop
│   │
│   ├── rules/                        # Card game domain
│   │   ├── catalogue.py              # 🔑 45 core experimental rules
│   │   ├── pretraining_rules.py      # 44 alternative pre-training rules
│   │   ├── cards.py                  # Card/Hand representations
│   │   └── primitives.py             # Python helper functions
│   │
│   ├── experiments/                  # Canonical experiment scripts (7)
│   │   ├── run_reference_wakesleep.py       # ⭐ PRIMARY entry point
│   │   ├── run_comprehensive_library_ablation.py
│   │   ├── run_recognition_guided_ablation.py
│   │   ├── run_transfer_study.py
│   │   ├── run_targeted_ablation_study.py
│   │   └── ...
│   │
│   ├── tests/                        # Consolidated unit tests (7 files)
│   └── generate_systematic_report.py # Report generator
│
├── examples/                         # Simple demonstrations
│   ├── main_demo.py                  # Interactive demo
│   └── README.md                     # How to run examples
│
├── docs/                             # Documentation
│   ├── ARCHITECTURE.md               # System architecture
│   ├── KNOWN_ISSUES.md               # 🔑 Bug documentation
│   ├── MODULE_STATUS.md              # Module status reference
│   ├── FEATURE_STATUS.md             # Feature implementation status
│   ├── ONBOARDING.md                 # Getting started guide
│   └── EXPERIMENTS_STATUS.md         # Active experiments
│
├── archived/                         # Historical code (52 files)
│   ├── parameter_tuning/             # Ablation & comparison scripts
│   ├── analysis/                     # Evaluation & diagnostic scripts
│   ├── model_development/            # Training variant scripts
│   ├── legacy_recognition/           # Old recognition models (GRU, Set Transformer)
│   ├── legacy_runners/               # Old runner scripts
│   └── laps_research/                # LAPS subproject (paused)
│
├── data/                             # Input data and samples
│   └── sample_results/               # Example outputs for testing
│
├── results/                          # Experiment outputs (gitignored)
├── CLAUDE.md                         # AI coding agent guidelines
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## Core Components

### 1. Primitives (`primitives.py`)

The DSL contains **57 primitives in 10 categories**, designed for cognitive realism:

| Category | Examples | Count |
|----------|----------|-------|
| Constants | `true`, `false`, `0-5`, `CLUBS`, `HEARTS`, `RED`, `BLACK` | 14 |
| Card Accessors | `get_rank`, `get_suit`, `get_color`, `rank_val` | 4 |
| Position Ops | `head`, `last`, `at`, `length`, `reverse` | 5 |
| List Slicing | `take`, `drop`, `zip_with`, `adjacent_pairs`, `first_half`, `second_half` | 7 |
| Direct Queries | `has_suit`, `has_color`, `count_suit`, `count_color`, `n_unique_suits` | 7 |
| Aggregates | `sum_ranks`, `max_rank`, `min_rank` | 3 |
| Comparisons | `eq`, `lt`, `le`, `gt`, `ge` | 5 |
| Boolean Ops | `and`, `or`, `not`, `if` | 4 |
| Higher-Order | `map`, `filter`, `all`, `any`, `unique` | 5 |
| Arithmetic | `+`, `-`, `mod` | 3 |

**Example rules in DSL notation:**
```
# "First card is red"
(λ hand. (eq RED (get_color (head hand))))

# "Hand contains a spade"
(λ hand. (has_suit hand SPADES))

# "All cards are the same suit" (flush)
(λ hand. (lt (n_unique_suits hand) 2))

# "Hand has exactly two colors"
(λ hand. (eq 2 (n_unique_colors hand)))
```

### 2. Recognition Model (`contrastive_recognition.py`)

**ContrastiveRecognitionModel** - the PRIMARY recognition architecture:

```python
from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.primitives import build_lean_grammar

grammar = build_lean_grammar()  # 57 primitives
model = ContrastiveRecognitionModel(
    grammar=grammar,            # Model derives num_primitives from grammar
    card_hidden=64,
    card_out=32,
    pred_hidden=64,
    output_mode='softmax',      # Use 'softmax' for search ranking
)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(100):
    loss = model.compute_loss(tasks, primitive_labels)
    loss.backward()
    optimizer.step()

# Inference
primitive_probs = model.predict_primitives(task)  # Shape: (num_primitives,)
```

### 3. Enumeration (`enumeration.py`)

**Memoized best-first search** with 1000x+ speedup:

```python
from dreamcoder_core.enumeration import TopDownEnumerator
from dreamcoder_core.primitives import build_lean_grammar

grammar = build_lean_grammar()
enumerator = TopDownEnumerator(grammar, max_depth=10, max_programs=50000)

# Use memoized enumeration (ALWAYS use this)
for program, log_prob in enumerator.enumerate_memoized(
    request_type=arrow(HAND, BOOL),
    max_cost=50.0,
    timeout_seconds=300
):
    if task.evaluate(program):
        print(f"Solution: {program}")
        break
```

**Performance characteristics:**

| Depth | Programs/Second | 50K Budget Time |
|-------|-----------------|-----------------|
| 6 | ~300,000 | < 1 second |
| 7 | ~70,000 | < 1 second |
| 8 | ~200,000 | < 1 second |
| 10+ | ~50,000 | 1-2 seconds |

### 4. Compression (`compression/`)

Library learning with **quality filters** to prevent degenerate abstractions:

```python
from dreamcoder_core.compression import compress_frontiers

# Extract useful abstractions from solved programs
new_grammar, compressions = compress_frontiers(
    grammar=current_grammar,
    frontiers=solved_frontiers,
    min_uses=3,                    # Minimum frequency
    max_abstractions=10,           # Per iteration
    recognition_model=model,       # Optional: recognition-guided
    use_recognition_guidance=True
)
```

**Quality filters (fixed Jan 2025):**
- `is_nontrivial()`: Requires ≥2 primitives or duplicated variable uses
- `is_eta_reducible()`: Catches `(λ x. f x)` wrappers
- `is_single_task_abstraction()`: Prevents overly specific patterns

### 5. Wake-Sleep Loop (`wake_sleep.py`)

```python
from dreamcoder_core.wake_sleep import wake_sleep_iteration

for iteration in range(n_iterations):
    # Wake: Neural-guided enumeration
    frontiers = enumerate_with_recognition(tasks, grammar, model)

    # Sleep: Train recognition on solved tasks
    model = train_recognition(model, frontiers)

    # Compress: Learn new abstractions
    grammar = compress_frontiers(grammar, frontiers)

    # Dream (optional): Generate synthetic tasks
    synthetic_tasks = dream_from_grammar(grammar)
```

---

## Running Experiments

### Available Experiment Scripts

All scripts are in `src/experiments/`. The canonical entry point is marked with ⭐.

| Script | Purpose | Typical Duration |
|--------|---------|------------------|
| `run_reference_wakesleep.py` ⭐ | **Canonical** wake-sleep training | 6-10 hours |
| `run_overnight_wakesleep_study.py` | Full overnight study | 8-12 hours |
| `run_comprehensive_library_ablation.py` | Library learning ablation | 4-8 hours |
| `run_recognition_guided_ablation.py` | Recognition + compression | 4-8 hours |
| `run_recognition_compression_ablation.py` | Recognition vs compression | 4-6 hours |
| `run_targeted_ablation_study.py` | Component ablations | 2-4 hours |
| `run_transfer_study.py` | Transfer learning | 2-4 hours |

### Overnight Run Protocol

**CRITICAL**: Any run expected to exceed 30 minutes MUST use caffeinate:

```bash
cd src

# Full wake-sleep experiment
nohup caffeinate -d -i -s python3 experiments/run_overnight_wakesleep_study.py > overnight.out 2>&1 &

# Get the process ID
PID=$!
echo "Started process $PID"

# Verify caffeinate is protecting it
ps aux | grep caffeinate

# Monitor progress
tail -f overnight.out
```

**Caffeinate flags:**
- `-d`: Prevent display sleep
- `-i`: Prevent system idle sleep
- `-s`: Prevent system sleep (on AC power)
- `-w PID`: Wait for process (alternative usage)

### Quick Validation Run

Before overnight runs, validate with small budgets:

```bash
cd src
python -c "
from experiments.run_overnight_wakesleep_study import run_study
run_study(n_iterations=2, budget=5000, timeout=60)  # Quick test
"
```

---

## Configuration Reference

### Recognition Model Parameters

The model takes a `grammar` object and derives `num_primitives` automatically.

```python
# ContrastiveRecognitionModel constructor parameters
model = ContrastiveRecognitionModel(
    grammar=grammar,           # Required: provides primitives
    d_suit=8,                  # Suit embedding dimension
    d_rank=16,                 # Rank embedding dimension
    d_pos=8,                   # Position embedding dimension
    card_hidden=64,            # Card MLP hidden size
    card_out=32,               # Final card/task embedding dimension
    pred_hidden=64,            # Primitive predictor hidden size
    output_mode='softmax',     # 'softmax' for search ranking, 'sigmoid' for classification
    normalize_embeddings=True, # Apply LayerNorm to task embeddings
    device='cpu'               # 'cpu' or 'cuda'
)
```

### Enumeration Parameters

```python
config = {
    'max_depth': 10,        # Maximum program tree depth
    'budget': 50000,        # Maximum programs to enumerate
    'timeout': 300,         # Timeout in seconds
    'max_cost': 50.0,       # Maximum log-probability cost
}
```

### Wake-Sleep Parameters

```python
config = {
    'n_iterations': 10,     # Number of wake-sleep iterations
    'recognition_epochs': 10,  # Epochs per sleep phase
    'learning_rate': 1e-3,
    'blend_factor': 0.5,    # Blend frontiers with grammar
    'min_uses': 3,          # Minimum abstraction frequency
}
```

---

## Results & Reporting

### Output Structure

Each experiment creates a timestamped directory:

```
results_overnight_wakesleep/study_20260102_225624/
├── experiment_config.json       # Run configuration
├── final_result.json            # Summary metrics
├── report.html                  # Visual HTML report
├── iteration_checkpoints/       # Per-iteration state
│   ├── iteration_0001.json
│   ├── iteration_0002.json
│   └── ...
├── models/                      # Saved model weights
│   ├── recognition_model_iter_01.pt
│   └── ...
├── grammars/                    # Grammar snapshots
│   ├── grammar_iter_01.json
│   └── ...
└── frontiers/                   # Solved programs
    ├── frontiers_iter_01.json
    └── ...
```

### Generating Reports

```bash
# Generate HTML report for a completed run
python generate_systematic_report.py --run-dir results_overnight_wakesleep/study_20260102_225624/

# View in browser
open results_overnight_wakesleep/study_20260102_225624/report.html
```

### Key Metrics to Monitor

| Metric | Good Value | Concern If |
|--------|------------|------------|
| Tasks solved | 30+/45 | < 20/45 |
| Recognition loss | < 0.1 | > 1.0 or near 0 |
| Enumeration rate | > 10K prog/s | < 1K prog/s |
| Cache hit rate | > 95% | < 80% |
| Abstractions learned | 5-20/iter | 0 or > 50 |

---

## Troubleshooting

### Common Issues

#### "Process killed" or system becomes unresponsive
**Cause**: System went to sleep during overnight run
**Fix**: Always use `caffeinate -d -i -s` (see [Overnight Run Protocol](#overnight-run-protocol))

#### Zero tasks solved despite long runtime
**Cause**: Likely the task-result scrambling bug (if using old scripts)
**Fix**: Use dictionary keying by task name (see `KNOWN_ISSUES.md`)

#### Very slow enumeration (< 100 prog/s)
**Cause**: Not using memoized enumeration
**Fix**: Use `enumerator.enumerate_memoized()` instead of `enumerate()`

#### "MaybeEncodingError" in logs
**Cause**: Program objects contain lambdas that can't be pickled
**Fix**: Return `program_str` (string) not `program` (object) from workers

#### Recognition model loss goes to exactly 0
**Cause**: Overfitting or bug in loss computation
**Fix**: Add validation split, check for data leakage

### Diagnostic Commands

```bash
# Check if process is still running
ps aux | grep python3

# Check if caffeinate is active
ps aux | grep caffeinate

# Monitor memory usage
top -pid $(pgrep -f "run_overnight")

# Check recent logs
tail -100 overnight.out

# Find crash reports (macOS)
ls -la /Library/Logs/DiagnosticReports/ | grep python
```

---

## Development Guide

### Running Tests

```bash
cd src
python -m pytest tests/ -v
```

### Code Style

- **Type hints** on all function signatures
- **Docstrings** for all public functions
- Use `primitives.py` as the authoritative primitive source
- Follow existing patterns for new experiment scripts

### Adding a New Experiment

1. Create script in `src/experiments/`
2. Use `run_reference_wakesleep.py` as a template
3. Results save to `results/` (gitignored - not committed)
4. Document in `docs/EXPERIMENTS_STATUS.md`

### Bug Documentation

When fixing bugs, document in `docs/KNOWN_ISSUES.md`:
1. Severity level
2. Symptoms observed
3. Root cause analysis
4. Fix with code example
5. Lessons learned

---

## Research Context

### Self-Explanation and Program Induction

This project tests whether DreamCoder's mechanisms can explain self-explanation effects in human learning:

| Self-Explanation | Program Induction |
|------------------|-------------------|
| Generalization from examples | Library learning: extracting reusable primitives |
| Implicit → explicit knowledge | Representational exchange between systems |
| Information compression | MDL-based abstraction |

### Behavioral Experiment

The companion experiment is at: [card-games](https://github.com/konukcan/card-games)

- Participants learn card game rules through examples
- We measure learning curves, transfer, and explanations
- This model predicts these patterns computationally

### Key Research Questions

1. **Do humans follow similar compositional structure?**
2. **What predicts rule difficulty?** (program length, search depth)
3. **How does library learning improve efficiency?**
4. **Which primitives are cognitively natural?**

---

## Citation

This project builds upon the following foundational work:

**DreamCoder** — The wake-sleep program synthesis framework this project implements:

```bibtex
@article{ellis2023dreamcoder,
  title={DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning},
  author={Ellis, Kevin and Wong, Catherine and Nye, Maxwell and Mathew, Sable-Meyer and
          Cary, Luke and Morales, Luc and Hewitt, Luke and Solar-Lezama, Armando and Tenenbaum, Joshua B},
  journal={Philosophical Transactions of the Royal Society A},
  volume={381},
  number={2251},
  pages={20220050},
  year={2023},
  doi={10.1098/rsta.2022.0050}
}
```

- **GitHub**: [https://github.com/ellisk42/ec](https://github.com/ellisk42/ec)
- **Documentation**: [https://ellisk42.github.io/ec/](https://ellisk42.github.io/ec/)

**DreamDecompiler** — Recognition-guided compression used in our library learning:

```bibtex
@inproceedings{palmarini2024dreamdecompiler,
  title={Bayesian Program Learning by Decompiling Amortized Knowledge},
  author={Palmarini, Alessandro B and Lucas, Christopher G and Siddharth, N},
  booktitle={International Conference on Machine Learning (ICML)},
  pages={39042--39055},
  year={2024},
  volume={235},
  series={PMLR},
  url={https://arxiv.org/abs/2306.07856}
}
```

### Additional Influences

- **Wong et al. (2021)** — LAPS language-guided synthesis: "Leveraging Language to Learn Program Abstractions and Search Heuristics." *ICML 2021*. [arXiv:2106.11053](https://arxiv.org/abs/2106.11053)
- **Chi et al. (1994)** — Self-explanation in learning: "Eliciting self-explanations improves understanding." *Cognitive Science*, 18(3).

