# Module Status Reference

This document describes the status and purpose of each module in the DreamCoder card game modeling system.

**Last Updated**: January 2025

---

## Core Modules (`src/dreamcoder_core/`)

### Production-Ready

| Module | Purpose | Status | Notes |
|--------|---------|--------|-------|
| `lean_primitives.py` | **Authoritative primitive library** (60 primitives in 5 levels) | STABLE | Always use this for primitives |
| `program.py` | Program representation, parsing, evaluation | STABLE | Core data structures |
| `grammar.py` | Probabilistic context-free grammar | STABLE | Handles primitive weights |
| `type_system.py` | Hindley-Milner type system | STABLE | Type inference and checking |
| `task.py` | Task definition and evaluation | STABLE | Input/output specification |
| `enumeration.py` | TopDownEnumerator with memoization | STABLE | Use `enumerate_memoized()` for best performance |
| `compression.py` | Library learning / abstraction extraction | STABLE | MDL-based compression |
| `wake_sleep.py` | Main wake-sleep training loop | STABLE | Core DreamCoder algorithm |
| `html_report.py` | HTML report generation | STABLE | Visualization of results |
| `visualization.py` | Matplotlib-based visualizations | STABLE | Training curves, metrics |

### Recognition Models

| Module | Purpose | Status | Notes |
|--------|---------|--------|-------|
| `contrastive_recognition.py` | **PRIMARY** - Factored embeddings + contrastive encoding | PRODUCTION | τ = mean(pos) - mean(neg), supports sigmoid/softmax output |
| `recognition_variants.py` | Architectural variants (EnhancedCardEncoder, BigramHead, etc.) | EXPERIMENTAL | Component library for testing |

#### Archived Recognition Models (`archived/legacy_recognition/`)

| Module | Purpose | Why Archived |
|--------|---------|--------------|
| `neural_recognition.py` | GRU + attention (DreamCoder-style) | Sequential processing suboptimal for cards; limited transfer |
| `set_transformer_recognition.py` | Set Transformer | Embedding collapse problem - all tasks had cosine similarity ≈ 1.0 |

See `archived/legacy_recognition/README.md` for detailed explanation of why these models were superseded.

### Experimental / Supporting

| Module | Purpose | Status | Notes |
|--------|---------|--------|-------|
| `contextual_grammar.py` | Grammar with contextual information | EXPERIMENTAL | December 2024 addition |
| `contrastive_dreaming.py` | Contrastive learning in dream phase | EXPERIMENTAL | Not fully integrated |
| `contrastive_wake_sleep.py` | Wake-sleep with contrastive loss | EXPERIMENTAL | Alternative training loop |
| `interpretability.py` | Model interpretability tools | EXPERIMENTAL | Feature importance, attention visualization |
| `dreamcoder_original.py` | Original DreamCoder algorithms | REFERENCE | For comparison with paper |
| `enumeration_optimized.py` | Alternative enumeration strategies | DEPRECATED | Use main `enumeration.py` instead |
| `test_enumeration_optimized.py` | Tests for enumeration | TEST | Unit tests |

---

## Rules Modules (`src/rules/`)

| Module | Purpose | Status | Notes |
|--------|---------|--------|-------|
| `catalogue.py` | **45 core experimental rules** | STABLE | Main rule set for experiments |
| `pretraining_rules.py` | 44 alternative rules for pre-training | STABLE | Warmstart experiments |
| `cards.py` | Card representation (rank, suit) | STABLE | Core data model |
| `primitives.py` | Python helper functions for rule definitions | STABLE | NOT DSL primitives - just Python helpers |

---

## Runner Scripts (`src/`)

### Active

| Script | Purpose | Status |
|--------|---------|--------|
| `generate_systematic_report.py` | HTML report generation | STABLE |
| `run_incremental_wakesleep.py` | Incremental wake-sleep with ContrastiveRecognitionModel | EXPERIMENTAL |
| `run_progressive_wakesleep.py` | Progressive curriculum with ContrastiveRecognitionModel | EXPERIMENTAL |

### Legacy (use ContrastiveRecognitionModel instead)

| Script | Purpose | Status |
|--------|---------|--------|
| `run_overnight_v3.py` | Old overnight runner with NeuralRecognitionModel | LEGACY |
| `resume_overnight_v3.py` | Resume for run_overnight_v3 | LEGACY |
| `run_experimental_rules.py` | Set Transformer experiments | LEGACY |
| `run_overnight_set_transformer.py` | Set Transformer overnight | LEGACY |

### Supporting

| Script | Purpose | Status |
|--------|---------|--------|
| `enumeration_worker.py` | Worker process for parallel enumeration | STABLE |
| `enumeration_worker_set_transformer.py` | Worker for Set Transformer | STABLE |
| `main_demo.py` | Interactive demo | STABLE |
| `benchmark_enumeration.py` | Enumeration performance testing | UTILITY |
| `deep_enumeration_test.py` | Deep enumeration testing | UTILITY |
| `parallel_primitive_experiment.py` | Primitive experiment parallelization | UTILITY |
| `test_*.py` | Unit tests | TEST |

---

## Experiment Scripts (`src/experiments/`)

### Active Experiments

| Script | Purpose | Status |
|--------|---------|--------|
| `comprehensive_variant_comparison.py` | Compare all recognition variants | ACTIVE |
| `run_embedding_variants.py` | Test embedding approaches | ACTIVE |
| `train_and_evaluate_recognition.py` | Recognition model evaluation | ACTIVE |
| `train_recognition_improved.py` | Improved training procedure | ACTIVE |
| `run_warmstart_experiment.py` | Pre-training experiments | ACTIVE |
| `ablate_lambda_weights*.py` | Lambda hyperparameter ablation | ACTIVE |
| `compare_*.py` | Various comparison studies | ACTIVE |

### Analysis Scripts

| Script | Purpose |
|--------|---------|
| `analyze_factorial_results.py` | Factorial experiment analysis |
| `evaluate_*.py` | Evaluation scripts |
| `generate_prediction_comparison.py` | Prediction quality analysis |
| `run_interpretability_on_saved.py` | Post-hoc interpretability |

### Archived (`experiments/archive/`)

20 diagnostic scripts from December 2024 development. See `experiments/archive/README.md`.

---

## Documentation (`docs/`)

| Document | Purpose | Status |
|----------|---------|--------|
| `rule_difficulty_classification.md` | Rule difficulty taxonomy (Easy/Medium/Hard/Very Hard) | REFERENCE |
| `model_comparison_table.md` | Recognition model comparison | REFERENCE |
| `ENUMERATION_ARCHITECTURE.md` | Enumeration system design | REFERENCE |
| `DREAMCODER_INTEGRATION.md` | DreamCoder algorithm integration | REFERENCE |
| `experimental_design_warmstart_pretraining.md` | Warmstart experiment design | REFERENCE |
| `description_generator_*.md` | LAPS subproject docs | PAUSED |
| `LAPS_*.md` | LAPS integration analysis | PAUSED |
| `set_transformer_*.md` | Set Transformer implementation | REFERENCE |

---

## Key Architectural Notes

### What's Working Well
1. **Memoized enumeration** (`enumerate_memoized()`) - 1000x+ speedup over naive enumeration
2. **BiGRU recognition** - Best primitive prediction accuracy
3. **4-phase curriculum** - Effective for progressive difficulty
4. **PyPy workers** - 7-15x speedup without Cython complexity

### What's Experimental
1. **Contrastive recognition** - Promising but needs tuning
2. **Set Transformer** - Good for position-invariant features
3. **Contextual grammar** - Recently added, evaluation ongoing

### What's Deprecated
1. **Cython modules** - Removed (pickle serialization issues with multiprocessing)
2. **Legacy GRU recognition** - Superseded by BiGRU with attention
3. **Cost-banding enumeration** - Removed (was causing 6x slowdown)

---

## Module Dependencies

```
                          ┌─────────────────┐
                          │  type_system.py │
                          └────────┬────────┘
                                   │
                          ┌────────▼────────┐
                          │    program.py   │
                          └────────┬────────┘
                                   │
           ┌───────────────────────┼───────────────────────┐
           │                       │                       │
   ┌───────▼───────┐      ┌────────▼────────┐     ┌────────▼────────┐
   │  grammar.py   │      │ enumeration.py  │     │ lean_primitives │
   └───────┬───────┘      └────────┬────────┘     └────────┬────────┘
           │                       │                       │
           └───────────────────────┼───────────────────────┘
                                   │
                          ┌────────▼────────┐
                          │   wake_sleep.py │
                          └────────┬────────┘
                                   │
                   ┌───────────────┼───────────────┐
                   │               │               │
          ┌────────▼────────┐  ┌───▼───┐  ┌───────▼───────┐
          │ recognition.py  │  │task.py│  │compression.py │
          └─────────────────┘  └───────┘  └───────────────┘
```
