# Card Game Rule Learning: DreamCoder Modeling

This repository implements a **DreamCoder-inspired program synthesis system** for modeling how participants learn compositional rules in card game tasks.

## Overview

This is a computational modeling companion to the behavioral card game experiment in [card-games](https://github.com/konukcan/card-games). We implement:

1. **Compositional DSL** for card game rules (56 rules decomposed into primitives)
2. **Program enumeration** with best-first search
3. **Neural recognition network** to guide search
4. **Library learning** through compression
5. **Comprehensive visualizations** of learning dynamics

## Connection to Ellis et al.'s DreamCoder

This implementation follows the architecture of:

> Ellis, K., Wong, L., Nye, M., et al. (2023). **DreamCoder: growing generalizable, interpretable knowledge with wake–sleep Bayesian program learning.** *Philosophical Transactions of the Royal Society A*, 381: 20220050.

**Key adaptations:**

- **Domain**: Card game predicates instead of list/graphics/text tasks
- **DSL**: 5-level compositional grammar extracted from domain analysis
- **Recognition model**: Predicts primitives from task examples (not full programs)
- **Search**: Enumerate + neural guide (no refactoring/sketch-and-fill)

See `docs/DREAMCODER_INTEGRATION.md` for detailed comparison.

## Repository Structure

```
card-games-modeling/
├── src/
│   ├── dreamcoder_core/         # Core DreamCoder components (active)
│   │   ├── lean_primitives.py   # Authoritative primitive library (60 base primitives)
│   │   ├── neural_recognition.py # Neural recognition network
│   │   ├── enumeration.py       # Program enumeration with PyPy workers
│   │   ├── compression.py       # Library learning
│   │   ├── interpretability.py  # Feature importance, embeddings
│   │   └── cython_src/          # Cython modules (not currently active)
│   ├── dreamcoder/              # Legacy components (reference only)
│   ├── rules/                   # Card game domain
│   │   ├── cards.py             # Card/Hand representations
│   │   └── catalogue.py         # All 56 rules
│   ├── visualization/           # Analysis & plotting
│   ├── results/                 # Experiment outputs
│   │   └── overnight_v3/        # Current overnight runs
│   ├── run_overnight_v3.py      # Main overnight experiment script
│   ├── resume_overnight_v3.py   # Resume interrupted runs
│   └── KNOWN_ISSUES.md          # Bug documentation
├── docs/                        # Documentation
├── CLAUDE.md                    # Coding agent guidelines
└── requirements.txt             # Python dependencies
```

## Quick Start

### Installation

```bash
cd card-games-modeling
pip install -r requirements.txt
```

### Running Overnight Experiments

The main experiment script runs a multi-phase wake-sleep training loop:

```bash
cd src

# Launch overnight run with caffeinate (prevents system sleep)
nohup caffeinate -d -i -s python3 run_overnight_v3.py > overnight_v3.out 2>&1 &

# Monitor progress
tail -f overnight_v3.out

# Resume an interrupted run
python3 resume_overnight_v3.py --run-dir results/overnight_v3/run_v3_YYYYMMDD_HHMMSS
```

### Output

Each run generates a timestamped directory under `src/results/overnight_v3/` containing:

1. **Iteration checkpoints**: `iteration_checkpoints/iteration_NNNN.json`
2. **Model weights**: `recognition_model_*.pt`
3. **Grammar snapshots**: `grammar_phase*.json`
4. **Frontiers**: `frontiers_phase*.json` (solved programs)
5. **Summary report**: `report.html`

## Key Features

### 1. Compositional DSL

Our grammar has 5 levels:

- **Level 0**: Atomic primitives (`getSuit`, `getRank`, `first`, `last`, ...)
- **Level 1**: List combinators (`map`, `filter`, `count`, `palindrome`, ...)
- **Level 2**: Structural operators (`halves`, `terminals`, `shiftedPairs`, ...)
- **Level 3**: Domain algorithms (`hasAP`, `bracketMatch`, `cycleMap`, ...)
- **Level 4**: Meta-combinators (`halvesEqual_F`, `seqPalindrome_P`, ...)

See `src/dreamcoder_core/lean_primitives.py` for implementation.

### 2. Neural Recognition Model

The recognition model predicts which primitives are likely useful for solving a task, given input/output examples. This guides enumeration by prioritizing promising programs.

**Architecture:**

```
Task with M examples [(hand₁, bool₁), (hand₂, bool₂), ...]
                    ↓
┌─────────────────────────────────────────────────────────┐
│  Per-Card Feature Extraction (24 dimensions/card)       │
│  ├─ Suit one-hot: 4 dims (♣♦♥♠)                        │
│  ├─ Rank one-hot: 13 dims (2-A)                        │
│  ├─ Color one-hot: 2 dims (red/black)                  │
│  ├─ Normalized rank: 1 dim (0-1)                       │
│  └─ Binary features: 4 dims (face, ace, even, odd)     │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│  CardEncoder (per card → sequence → hand embedding)     │
│  ├─ Linear(24 → 64) + ReLU + Dropout(0.1)              │
│  ├─ Linear(64 → 64)                                     │
│  ├─ Bidirectional GRU(64 → 64×2)                       │
│  └─ Linear(128 → 64) [combine forward/backward]        │
│  Output: 64-dim hand embedding                          │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│  ExampleEncoder (hand + output → example embedding)     │
│  ├─ CardEncoder(hand) → 64 dims                        │
│  ├─ Linear(2 → 64) + ReLU [output: True/False]         │
│  ├─ Concat → 128 dims                                   │
│  └─ Linear(128 → 64) + ReLU + Linear(64 → 64)         │
│  Output: 64-dim example embedding (×M examples)         │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│  TaskEncoder (M examples → single task embedding)       │
│  ├─ Attention weights: Linear(64→32→1) + Softmax       │
│  ├─ Weighted pooling across examples                   │
│  └─ Linear(64 → 64) + ReLU + Linear(64 → 64)          │
│  Output: 64-dim task embedding                          │
│  [Permutation-invariant over examples]                  │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│  PrimitivePredictor (task → primitive log-probs)        │
│  ├─ Linear(64 → 128) + ReLU + Dropout(0.1)             │
│  ├─ Linear(128 → 64) + ReLU + Dropout(0.1)             │
│  └─ Linear(64 → num_primitives) + LogSoftmax           │
│  Output: log P(primitive | task) for ~60 primitives     │
└─────────────────────────────────────────────────────────┘
```

**Key Design Choices:**
- **Bidirectional GRU**: Captures left-to-right and right-to-left card dependencies
- **Attention-based pooling**: Learns which examples are most informative (not max-pool)
- **Permutation invariance**: Example order doesn't affect predictions
- **Multi-hot training targets**: Multiple primitives can be correct per task
- **~25-30K parameters**: Small enough for rapid training in sleep phase

**Training:**
- **Signal**: Primitives used in solved programs (from enumeration phase)
- **Loss**: Cross-entropy over multi-hot primitive targets
- **Optimizer**: Adam (lr=0.001)
- **Epochs per iteration**: 10 (configurable)

**Files:**
- `dreamcoder_core/neural_recognition.py`: Main implementation
- `dreamcoder_core/interpretability.py`: Feature importance, embedding visualization

### 3. Program Enumeration

Best-first search prioritized by:
```
priority(program) = log P(program | primitives) + log P_neural(primitives | examples)
```

where:
- `P(program | primitives)` = grammar prior
- `P_neural(primitives | examples)` = recognition network output

### 4. Library Learning

Compression through:
- **Frequency heuristic**: Abstract patterns used ≥3 times
- **MDL principle**: Minimize total description length
- **Retraining**: Update recognition network with new abstractions

## Visualizations

The HTML report (`report.html`) in each run directory includes:

1. **Training progress**: Loss curves, solve rates per iteration
2. **Grammar growth**: Primitives added through library learning
3. **Solution details**: Programs discovered for each task
4. **Timing analysis**: Enumeration and training durations

## Experimental Questions

This model enables testing:

1. **Do humans follow similar compositional structure?**
   - Compare model's learned abstractions to participant transfer patterns

2. **What predicts rule difficulty?**
   - Correlate search depth / description length with human accuracy

3. **How does library learning improve efficiency?**
   - Bootstrap experiments: does early exposure to related rules help?

4. **Which primitives are cognitively natural?**
   - Compare model's induced library to human explanations

## Integration with Behavioral Experiment

The main experiment repository is at: [card-games](https://github.com/konukcan/card-games)

**Data flow:**

```
Behavioral Experiment
    ↓
Participant responses (.csv)
    ↓
Model fitting (this repo)
    ↓
Predicted learning curves
    ↓
Compare to human data
```

## Development Status

**Current (v0.3 - Working Pipeline)**:
- ✅ Complete DSL with 60 base primitives
- ✅ All 56 rules implemented in catalogue
- ✅ Neural recognition network (bidirectional GRU + attention)
- ✅ PyPy-accelerated parallel enumeration
- ✅ Library learning with compression
- ✅ Wake-sleep training loop
- ✅ Multi-phase pretraining on 43 rules
- ✅ Checkpoint/resume system for long runs

**Latest Results (Nov 2024)**:
- 26/43 pretraining tasks solved (60.5%)
- Grammar grew from 60 to 170 primitives via abstraction
- Recognition model loss: 5.60 → 4.82

**Next Steps**:
- Scale to full 56 rules
- Optimize memory usage for longer runs
- Fit to human behavioral data

## Citation

If you use this code, please cite:

```bibtex
@software{cardgames_dreamcoder_2025,
  author = {[Your Name]},
  title = {DreamCoder for Card Game Rule Learning},
  year = {2025},
  url = {https://github.com/[your-repo]}
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

## License

MIT License - see LICENSE file

## Contact

For questions about this modeling work, open an issue or contact [your email].

For the behavioral experiment, see the main [card-games repository](https://github.com/konukcan/card-games).
