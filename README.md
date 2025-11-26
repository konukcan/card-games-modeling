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
│   ├── dreamcoder/          # Core DreamCoder components
│   │   ├── dsl.py           # Type system + primitive library
│   │   ├── enumeration.py   # Program enumeration
│   │   ├── recognition.py   # Neural recognition network
│   │   ├── compression.py   # Library learning
│   │   └── wake_sleep.py    # Training loop
│   ├── rules/               # Card game domain
│   │   ├── cards.py         # Card/Hand representations
│   │   ├── primitives.py    # Level 0-4 primitives
│   │   └── catalogue.py     # All 56 rules
│   └── visualization/       # Analysis & plotting
│       ├── plots.py         # Visualization suite
│       └── analysis.py      # Metrics & statistics
├── data/                    # Generated tasks
├── results/                 # Outputs & figures
├── docs/                    # Documentation
└── tests/                   # Unit tests
```

## Quick Start

### Installation

```bash
# Clone this repository
cd card-games-modeling

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m pytest tests/
```

### Running the Demo

```bash
# Run on 10 representative rules
python src/main.py --demo

# Full run on all 56 rules (takes ~30 min)
python src/main.py --full

# Custom subset
python src/main.py --rules Sorted_by_rank,Has_pair_ranks,Halves_copy_suits
```

### Output

The demo generates:

1. **Search visualizations**: Program enumeration traces
2. **Recognition analysis**: Neural network predictions
3. **Compression metrics**: Library learning over iterations
4. **Transfer analysis**: Cross-rule generalization
5. **Summary report**: `results/report.html`

## Key Features

### 1. Compositional DSL

Our grammar has 5 levels:

- **Level 0**: Atomic primitives (`getSuit`, `getRank`, `first`, `last`, ...)
- **Level 1**: List combinators (`map`, `filter`, `count`, `palindrome`, ...)
- **Level 2**: Structural operators (`halves`, `terminals`, `shiftedPairs`, ...)
- **Level 3**: Domain algorithms (`hasAP`, `bracketMatch`, `cycleMap`, ...)
- **Level 4**: Meta-combinators (`halvesEqual_F`, `seqPalindrome_P`, ...)

See `src/rules/primitives.py` for implementation.

### 2. Recognition Network

Architecture:
```
Task Examples (8 hands × 104 features)
    ↓
Example Encoder (104 → 128 → 64) per hand
    ↓
Set Aggregator (max-pool, permutation-invariant)
    ↓
Task Embedding (64 dims)
    ↓
Primitive Predictor (64 → 128 → num_primitives)
    ↓
Probability distribution over primitives
```

Trained on synthetic tasks with ground truth decompositions.

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

All plots saved to `results/`:

1. **`search_trace.png`**: Enumeration tree with neural scores
2. **`recognition_heatmap.png`**: Predicted vs. true primitives
3. **`compression_curve.png`**: Description length over iterations
4. **`transfer_matrix.png`**: Cross-rule generalization
5. **`embedding_space.png`**: t-SNE of task representations
6. **`primitive_usage.png`**: Frequency distribution

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

See `docs/INTEGRATION.md` for details on fitting procedure.

## Development Status

**Current (v0.1 - Foundation)**:
- ✅ Complete DSL implementation
- ✅ All 56 rules ported to Python
- ✅ Basic enumeration search
- ✅ Recognition network architecture
- ✅ Visualization suite
- ✅ Demo on 10 rules

**Next Steps (v0.2 - Full Pipeline)**:
- ⏳ Library learning with compression
- ⏳ Wake-sleep training loop
- ⏳ Scaling to all 56 rules
- ⏳ Integration with Ellis et al.'s codebase

**Future (v1.0 - Empirical Modeling)**:
- ⏳ Fit to human behavioral data
- ⏳ Predict transfer patterns
- ⏳ Generate experimental curricula

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
# Sample improvement for CodeRabbit review
