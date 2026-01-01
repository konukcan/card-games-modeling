# Card Game Rule Learning: DreamCoder Modeling

This repository implements a **DreamCoder-inspired program synthesis system** for modeling how participants learn compositional rules in card game tasks.

## Overview

This is a computational modeling companion to the behavioral card game experiment in [card-games](https://github.com/konukcan/card-games). We implement:

1. **Compositional DSL** for card game rules (45 core rules decomposed into primitives)
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
│   │   ├── catalogue.py         # Core 45 experimental rules
│   │   └── pretraining_rules.py # Alternative 44 pre-training rules
│   ├── visualization/           # Analysis & plotting
│   ├── results*/                # Experiment outputs
│   ├── run_incremental_wakesleep.py  # Current wake-sleep runner
│   ├── run_progressive_wakesleep.py  # Alternative curriculum runner
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

### Running Experiments

The main experiment scripts run wake-sleep training with ContrastiveRecognitionModel:

```bash
cd src

# Launch overnight run with caffeinate (prevents system sleep)
nohup caffeinate -d -i -s python3 run_incremental_wakesleep.py > overnight.out 2>&1 &

# Or use progressive curriculum approach
nohup caffeinate -d -i -s python3 run_progressive_wakesleep.py > overnight.out 2>&1 &

# Monitor progress
tail -f overnight.out
```

### Output

Each run generates a timestamped results directory containing:

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

### 2. Contrastive Recognition Model

The recognition model predicts which primitives are likely useful for solving a task, given positive and negative examples. This guides enumeration by prioritizing promising programs.

**Architecture** (`ContrastiveRecognitionModel`):

```
Task with positive (satisfies rule) and negative (doesn't satisfy) examples
                    ↓
┌─────────────────────────────────────────────────────────┐
│  Factored Card Embeddings                               │
│  ├─ E_suit(suit): 4 → 16 dims                          │
│  ├─ E_rank(rank): 13 → 16 dims                         │
│  ├─ E_position(pos): 5 → 16 dims                       │
│  └─ Concatenate: 48 dims per card                      │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│  Hand Encoding (mean pooling over cards)               │
│  ├─ Linear(48 → 64) → per-card                        │
│  └─ Mean pool → 64-dim hand embedding                 │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│  Contrastive Task Encoding                             │
│  ├─ τ_pos = mean(positive hand embeddings)            │
│  ├─ τ_neg = mean(negative hand embeddings)            │
│  └─ τ = τ_pos - τ_neg (decision boundary)             │
│  Output: 64-dim task embedding capturing what          │
│  distinguishes positive from negative examples         │
└─────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────┐
│  Primitive Predictor (task → primitive probs)          │
│  ├─ Linear(64 → 128) + ReLU                           │
│  ├─ Linear(128 → 60) + Softmax (for search ranking)   │
│  Output: P(primitive | task) for 60 primitives         │
└─────────────────────────────────────────────────────────┘
```

**Key Design Choices:**
- **Contrastive encoding** (τ = mean(pos) - mean(neg)): Directly captures the decision boundary
- **Factored embeddings**: Learned embeddings for suit, rank, position (not one-hot)
- **Softmax output**: Provides primitive **ranking** for search guidance (critical for enumeration)
- **Auxiliary heads**: Count head (predicts # primitives), Bigram head (predicts co-occurrence)

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
- ✅ All 45 core rules implemented in catalogue
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
- Scale to full 45 core rules
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
