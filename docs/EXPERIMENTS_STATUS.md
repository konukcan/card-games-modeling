# Experiments Status Reference

This document tracks the current state of experiments in the DreamCoder card game modeling system.

**Last Updated**: January 2025

---

## Current Research Focus

The project is actively investigating **recognition model optimization**:
- How different architectural choices affect primitive prediction
- How different normalization strategies affect embedding quality
- How training on pretraining rules transfers to catalogue rules

### Primary Recognition Model

**ContrastiveRecognitionModel** (`contrastive_recognition.py`)
- Task encoding: τ = mean(positive_embeddings) - mean(negative_embeddings)
- Factored card embeddings: suit + rank + position
- Supports sigmoid (classification) and softmax (ranking) output modes

---

## Active Experiments

### Lambda Weight Ablation
**Scripts**: `ablate_lambda_weights.py`, `ablate_lambda_weights_v2.py`
**Question**: How do different weights for auxiliary losses affect performance?
**Status**: Completed, results in `results_lambda_ablation/`, `results_lambda_grid/`

### Normalization Comparison
**Scripts**: `compare_normalization_strategies.py`, `compare_normalization_wakesleep.py`
**Question**: LayerNorm+Scale vs L2 Normalization vs no normalization?
**Status**: Completed, results in `results_normalization_comparison/`, `results_normalization_wakesleep/`

### Sigmoid vs Softmax Output
**Scripts**: `compare_sigmoid_softmax.py`
**Question**: Which output activation gives better search guidance?
**Key Finding**: Softmax critical for search ranking (provides primitive ordering)
**Status**: Completed, results in `results_sigmoid_softmax/`

### Bigram Training
**Scripts**: `compare_bigram_training.py`
**Question**: Does the bigram head improve primitive co-occurrence prediction?
**Status**: Completed, results in `results_bigram_comparison/`

### Pipeline A/B Comparison
**Scripts**: `pipeline_ab_comparison.py`
**Question**: Train on pretraining rules → evaluate on catalogue rules
**Status**: Completed, results in `results_pipeline_ab/`

### Interpretability Analysis
**Scripts**: `run_interpretability_on_saved.py`
**Purpose**: Analyze saved model checkpoints for feature importance
**Status**: Active, results in `results_interpretability/`

### Warmstart Experiment
**Scripts**: `run_warmstart_experiment.py`
**Question**: Does pretraining on simpler rules help learn harder rules?
**Design**: See `docs/experimental_design_warmstart_pretraining.md`
**Status**: Partially completed, results in `src/results/warmstart_experiment/`

---

## Results Directories

### Recent (December 2024 - January 2025)

| Directory | Purpose | Date |
|-----------|---------|------|
| `results_ablation/` | Primitive ablation studies | Jan 2025 |
| `results_interpretability/` | Model interpretability | Jan 2025 |
| `results_normalization_wakesleep/` | Normalization in wake-sleep | Jan 2025 |
| `results_pipeline_ab/` | Pipeline A/B comparison | Jan 2025 |
| `results_sigmoid_softmax/` | Output activation comparison | Jan 2025 |
| `results_lambda_grid/` | Lambda hyperparameter grid | Dec 2024 |
| `results_lambda_ablation/` | Lambda ablation | Dec 2024 |
| `results_bigram_comparison/` | Bigram head comparison | Dec 2024 |
| `results_normalization_comparison/` | Normalization strategies | Dec 2024 |
| `results_new_heads_comparison/` | Additional prediction heads | Dec 2024 |

### Warmstart Experiments

| Directory | Condition | Recognition Model |
|-----------|-----------|-------------------|
| `results/warmstart_experiment/WARM_*` | Pretraining → catalogue | Various |
| `results/warmstart_experiment/COLD_*` | Direct on catalogue | Various |
| `results/warmstart_experiment/BOTH_*` | Both conditions | Various |

### Legacy (Pre-December 2024)

| Directory | Purpose | Notes |
|-----------|---------|-------|
| `results_archive/` | Archived old results | May contain NeuralRecognitionModel runs |
| `results_embedding_variants_*` | Embedding experiments | Dec 2024 |
| `results_random_contrast_*` | Random contrast variants | Dec 2024 |

---

## Key Findings

### 1. Output Activation Matters for Search
- **Finding**: Softmax output is critical for search guidance
- **Why**: Enumeration needs primitive **ranking**, not just classification
- **Implication**: Always use `output_mode='softmax'` for production

### 2. Contrastive Encoding Works Best
- **Finding**: τ = mean(pos) - mean(neg) captures decision boundary directly
- **Why**: This representation explicitly encodes what distinguishes positive from negative examples
- **See**: `archived/legacy_recognition/README.md` for comparison with GRU and Set Transformer

### 3. Embedding Normalization
- **Finding**: LayerNorm + learned scale factor provides good balance
- **Alternative**: L2 normalization is simpler but may limit expressiveness
- **Results**: `results_normalization_comparison/`

### 4. Pretraining Transfer (Preliminary)
- **Finding**: Mixed results on warm-start transfer
- **Challenge**: Recognition model may overfit to pretraining distribution
- **See**: `docs/experimental_design_warmstart_pretraining.md`

---

## Archived Experiments

### Diagnostic Scripts (`experiments/archive/`)
20 diagnostic scripts from December 2024 development:
- 15 `diagnose_*.py` scripts for component analysis
- 2 `check_*.py` scripts for verification
- 3 `test_*.py` scripts for integration testing

See `experiments/archive/README.md` for details.

### Factorial Experiments
Log files in `experiments/`:
- `factorial_experiment.log`
- `factorial_experiment_v2.log`
- `factorial_experiment_v3_highbudget.log`
- `factorial_overnight.log`

---

## Running New Experiments

### Recommended Entry Points

1. **Quick evaluation**: `experiments/train_and_evaluate_recognition.py`
2. **Full wake-sleep**: `run_incremental_wakesleep.py` or `run_progressive_wakesleep.py`
3. **Comparison study**: Use `experiments/compare_*.py` as templates

### Configuration

Most experiments share common parameters:
```python
# Recognition model
card_hidden = 64
card_out = 64
pred_hidden = 128
n_primitives = 60
output_mode = 'softmax'  # Use for search

# Training
learning_rate = 1e-3
n_epochs = 100
batch_size = 32

# Wake-sleep
budget = 50000
max_depth = 10
blend_factor = 0.5
```

### Results Format

Experiments typically save:
- `results.json` - Main metrics and configuration
- `models/` - Model checkpoints (PyTorch state dicts)
- `logs/` - Training logs and per-iteration metrics

---

## Paused / Incomplete

| Experiment | Status | Reason |
|------------|--------|--------|
| Set Transformer variants | PAUSED | Embedding collapse problem |
| GRU recognition | DEPRECATED | Sequential processing suboptimal |
| LAPS integration | PAUSED | Waiting for core model to stabilize |

---

## Next Steps

1. **Consolidate warmstart findings** - Complete analysis of transfer conditions
2. **Production wake-sleep run** - Full overnight run with optimized ContrastiveRecognitionModel
3. **Interpretability report** - Document what the model has learned
