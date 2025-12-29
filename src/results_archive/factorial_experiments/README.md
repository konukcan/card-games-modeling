# Factorial Experiments

2×3×3 factorial experiments comparing recognition models, dream strategies, and primitive sets.

## Runner Script
- **Script**: `src/experiments/run_recognition_dream_experiment.py` (ACTIVE)

## Experimental Design

### Factors
1. **Recognition Model** (2 levels): GRU (legacy) vs Contrastive (new)
2. **Dream Strategy** (3 levels): Standard vs Balanced vs Contrastive
3. **Primitive Set** (3 levels): Minimal vs Lean vs Lean+Fold

### Total Conditions
18 conditions × multiple runs = comprehensive comparison

## Directory Contents

### results_factorial/
**Date**: December 20-21, 2024
**Status**: Main factorial results

Contains:
- `gru_standard_lean/` - GRU model with standard dreaming on lean primitives
- `gru_standard_lean_plus_fold/` - GRU model with fold primitives
- `gru_standard_minimal/` - GRU model with minimal primitives
- `experiment_*.log` - Execution logs

### results_factorial_backup_20251220_023847/
**Date**: December 20, 2024
**Status**: Backup of earlier runs

Early factorial runs before debugging issues.

### results_factorial_lowbudget_backup_20251220_013756/
**Date**: December 20, 2024
**Status**: Low-budget testing runs

Quick iterations to validate experiment framework:
- Lower enumeration budget
- Fewer iterations
- Used for debugging

### results_factorial_pretraining_backup_20251219_193420/
**Date**: December 19, 2024
**Status**: First pretraining factorial attempt

Most comprehensive early run with all conditions:
- `gru_standard_{lean,lean_plus_fold,minimal}/`
- `gru_balanced_{lean,lean_plus_fold,minimal}/`
- `gru_contrastive_{lean,lean_plus_fold,minimal}/`
- `contrastive_standard_{lean,lean_plus_fold,minimal}/`

Multiple experiment logs showing iteration progression.

## Key Findings

1. **GRU vs Contrastive**: Contrastive model shows better task discrimination
2. **Dream Strategy**: Balanced dreaming helps on hard tasks
3. **Primitives**: Lean+Fold set solves more rules than minimal

## Log File Format

```
2025-12-19 17:28:24,763 - INFO - 2×3×3 FACTORIAL EXPERIMENT
2025-12-19 17:28:24,763 - INFO - Conditions: 18
2025-12-19 17:28:24,763 - INFO - Iterations per condition: 5
```

## How to Read Results

1. Check `experiment_*.log` for execution flow
2. Each condition subdirectory contains:
   - Run results JSON files
   - Model checkpoints
   - Per-iteration metrics
3. Compare solve rates across conditions
