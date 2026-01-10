# Card Game Rule Learning: DreamCoder Modeling

A **DreamCoder-inspired program synthesis system** for modeling how humans learn compositional rules in card game tasks. This project is part of PhD research connecting self-explanation (cognitive science) to program induction mechanisms.

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
| **60 Primitives in 5 Levels** | Cognitively-realistic DSL for card game rules |
| **Memoized Enumeration** | 1000x+ speedup over naive search |
| **Contrastive Recognition** | Neural network predicts useful primitives from examples |
| **Library Learning** | MDL-based compression with quality filters |
| **Wake-Sleep Training** | Iterative improvement loop |

### Connection to DreamCoder

This implementation follows the architecture of [Ellis et al. (2023)](https://doi.org/10.1098/rsta.2022.0050):

> Ellis, K., Wong, L., Nye, M., et al. **DreamCoder: growing generalizable, interpretable knowledge with wake–sleep Bayesian program learning.** *Phil. Trans. R. Soc. A*, 381: 20220050.

**Key adaptations for this domain:**
- **Domain**: Card game predicates instead of list/graphics/text tasks
- **DSL**: 5-level compositional grammar extracted from cognitive analysis
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
python main_demo.py
```

This runs a small enumeration task and shows the system finding solutions.

### Run a Full Experiment

```bash
cd src

# Launch overnight run with caffeinate (REQUIRED for runs > 30 min)
nohup caffeinate -d -i -s python3 experiments/run_overnight_wakesleep_study.py > overnight.out 2>&1 &

# Monitor progress
tail -f overnight.out
```

### Generate a Report

```bash
python generate_systematic_report.py --run-dir results_overnight_wakesleep/study_YYYYMMDD_HHMMSS/
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

```
Input: Task with positive and negative card hands
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Factored Card Embeddings                                       │
│  ├─ E_suit(suit): 4 → 16 dims                                  │
│  ├─ E_rank(rank): 13 → 16 dims                                 │
│  ├─ E_position(pos): 5 → 16 dims                               │
│  └─ Concatenate: 48 dims per card                              │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Hand Encoding                                                  │
│  ├─ Linear(48 → 64) per card                                   │
│  └─ Mean pooling → 64-dim hand embedding                       │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Contrastive Task Encoding                                      │
│  ├─ τ_pos = mean(positive hand embeddings)                     │
│  ├─ τ_neg = mean(negative hand embeddings)                     │
│  └─ τ = τ_pos - τ_neg  (captures decision boundary)            │
└─────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Primitive Predictor                                            │
│  ├─ Linear(64 → 128) + ReLU                                    │
│  ├─ Linear(128 → 60) + Softmax                                 │
│  └─ Output: P(primitive | task) for 60 primitives              │
└─────────────────────────────────────────────────────────────────┘
```

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
│   │   ├── lean_primitives.py        # 🔑 AUTHORITATIVE primitive library (60 primitives)
│   │   ├── contrastive_recognition.py# 🔑 PRIMARY recognition model
│   │   ├── enumeration.py            # TopDownEnumerator with memoization
│   │   ├── compression/              # Library learning package (9 modules)
│   │   │   ├── compress.py           # Main compression algorithm
│   │   │   ├── quality_filters.py    # Abstraction quality checks
│   │   │   ├── recognition_guided.py # Recognition-guided compression
│   │   │   ├── mdl_scoring.py        # MDL-based scoring
│   │   │   └── ...                   # Anti-unification, rewriting, etc.
│   │   ├── grammar.py                # Probabilistic context-free grammar
│   │   ├── program.py                # Program representation and parsing
│   │   ├── type_system.py            # Hindley-Milner type system
│   │   ├── task.py                   # Task definition
│   │   ├── wake_sleep.py             # Wake-sleep training loop
│   │   ├── interpretability.py       # Feature importance, embeddings
│   │   └── html_report.py            # Report generation
│   │
│   ├── rules/                        # Card game domain
│   │   ├── catalogue.py              # 🔑 45 core experimental rules
│   │   ├── pretraining_rules.py      # 44 alternative pre-training rules
│   │   ├── cards.py                  # Card/Hand representations
│   │   └── primitives.py             # Python helper functions
│   │
│   ├── experiments/                  # Experiment scripts (50+)
│   │   ├── run_overnight_wakesleep_study.py  # Full wake-sleep training
│   │   ├── run_targeted_ablation_study.py    # Component ablations
│   │   ├── run_recognition_guided_ablation.py # Recognition + compression
│   │   ├── run_transfer_study.py             # Transfer learning
│   │   ├── compare_*.py                      # Comparison studies
│   │   └── ...
│   │
│   ├── tests/                        # Unit tests
│   ├── visualization/                # Plotting utilities
│   ├── results_*/                    # Experiment outputs (38+ directories)
│   │
│   │  # Entry Point Scripts
│   ├── run_incremental_wakesleep.py  # Incremental wake-sleep runner
│   ├── run_progressive_wakesleep.py  # Curriculum-based runner
│   ├── generate_systematic_report.py # Report generator
│   ├── main_demo.py                  # Interactive demo
│   └── KNOWN_ISSUES.md               # 🔑 Bug documentation
│
├── docs/                             # Documentation
│   ├── MODULE_STATUS.md              # Module status reference
│   ├── FEATURE_STATUS.md             # Feature implementation status
│   ├── EXPERIMENTS_STATUS.md         # Active experiments
│   ├── rule_difficulty_classification.md
│   └── ...
│
├── archived/                         # Legacy code
│   ├── legacy_recognition/           # Old recognition models (GRU, Set Transformer)
│   └── deprecated_experiments/       # Superseded experiment scripts
│
├── CLAUDE.md                         # AI coding agent guidelines
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## Core Components

### 1. Primitives (`lean_primitives.py`)

The DSL contains **60 primitives in 5 levels**, designed for cognitive realism:

| Level | Category | Examples | Count |
|-------|----------|----------|-------|
| 0 | Constants | `TRUE`, `FALSE`, `0-5`, `CLUBS`, `HEARTS`... | 18 |
| 1 | Basic Operations | `eq`, `lt`, `gt`, `add`, `sub`, `and`, `or`, `not` | 12 |
| 2 | Card Accessors | `get_rank`, `get_suit`, `get_color`, `first`, `last` | 8 |
| 3 | List Operations | `map`, `filter`, `all`, `any`, `length`, `count` | 14 |
| 4 | Aggregates | `n_unique_ranks`, `n_unique_suits`, `has_pair`, `is_sorted` | 8 |

**Example rule in DSL notation:**
```
# "Hand is sorted by rank"
(all (λx. λy. (lt (get_rank x) (get_rank y))) (shifted_pairs hand))
```

### 2. Recognition Model (`contrastive_recognition.py`)

**ContrastiveRecognitionModel** - the PRIMARY recognition architecture:

```python
from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel

model = ContrastiveRecognitionModel(
    n_primitives=60,
    card_hidden=64,
    card_out=64,
    pred_hidden=128,
    output_mode='softmax',  # Use 'softmax' for search ranking
    use_count_head=True,
    use_bigram_head=True
)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(100):
    loss = model.compute_loss(tasks, primitive_labels)
    loss.backward()
    optimizer.step()

# Inference
primitive_probs = model(task)  # Shape: (60,)
```

### 3. Enumeration (`enumeration.py`)

**Memoized best-first search** with 1000x+ speedup:

```python
from dreamcoder_core.enumeration import TopDownEnumerator
from dreamcoder_core.lean_primitives import create_grammar

grammar = create_grammar()
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

| Script | Purpose | Typical Duration |
|--------|---------|------------------|
| `run_overnight_wakesleep_study.py` | Full wake-sleep training | 6-10 hours |
| `run_targeted_ablation_study.py` | Component ablations | 2-4 hours |
| `run_recognition_guided_ablation.py` | Recognition + compression | 4-8 hours |
| `run_transfer_study.py` | Transfer learning | 2-4 hours |
| `compare_normalization_*.py` | Normalization strategies | 1-2 hours |

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

```python
config = {
    'card_hidden': 64,      # Card encoder hidden size
    'card_out': 64,         # Card embedding dimension
    'pred_hidden': 128,     # Primitive predictor hidden size
    'n_primitives': 60,     # Number of primitives in grammar
    'output_mode': 'softmax',  # 'softmax' for search, 'sigmoid' for classification
    'use_count_head': True,    # Predict number of primitives
    'use_bigram_head': True,   # Predict primitive co-occurrence
    'normalization': 'layernorm_scale',  # 'l2', 'layernorm_scale', or None
}
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
- Use `lean_primitives.py` as the authoritative primitive source
- Follow existing patterns for new experiment scripts

### Adding a New Experiment

1. Create script in `src/experiments/`
2. Use existing scripts as templates
3. Save results to `src/results_<experiment_name>/`
4. Document in `docs/EXPERIMENTS_STATUS.md`

### Bug Documentation

When fixing bugs, document in `src/KNOWN_ISSUES.md`:
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

If you use this code, please cite:

```bibtex
@software{cardgames_dreamcoder_2025,
  author = {Can Konuk},
  title = {DreamCoder for Card Game Rule Learning},
  year = {2025},
  url = {https://github.com/konukcan/card-games-modelling}
}

@article{ellis2023dreamcoder,
  title={DreamCoder: growing generalizable, interpretable knowledge with wake--sleep Bayesian program learning},
  author={Ellis, Kevin and Wong, Catherine and Nye, Maxwell and others},
  journal={Philosophical Transactions of the Royal Society A},
  volume={381},
  number={2251},
  pages={20220050},
  year={2023}
}
```

---

## License

MIT License - see LICENSE file

## Contact

For questions about this modeling work, open an issue or contact the repository owner.

For the behavioral experiment, see the main [card-games repository](https://github.com/konukcan/card-games).
