# Logs Directory

This directory contains organized output logs from experiment runs (Nov-Dec 2024).

## Structure

```
logs/
├── production/           # Main overnight runs
│   ├── overnight_v3.out  # Primary production run (Nov 28)
│   └── resume_v3.out     # Resumed run
│
├── experiments/          # Active experiment logs
│   ├── warmstart/        # Warmstart pretraining experiments
│   │   ├── warmstart.out
│   │   ├── warmstart_neural.out
│   │   ├── warmstart_contrastive.out
│   │   └── warmstart_contrastive_softmax.out
│   │
│   ├── lambda_ablation/  # Lambda weight ablation studies
│   │   ├── lambda_grid.out
│   │   └── ablation_lambda.out
│   │
│   ├── contrastive/      # Contrastive learning experiments
│   │   ├── contrastive_full.log
│   │   └── run_lean_contrastive.out
│   │
│   └── variants/         # Wake-sleep variants
│       ├── progressive_wakesleep.out
│       └── incremental_wakesleep.out
│
├── historical/           # Logs documenting key findings
│   ├── overnight_listprims.out  # List primitives → 30-40/45 solved
│   └── dcyp_overnight.out       # DreamCoder+Y&P extended primitives
│
└── README.md
```

## Log Descriptions

### Production (2 files, ~180KB)
- **overnight_v3.out**: Main production run with `run_overnight_v3.py`. Shows 4-phase curriculum, 26/43 rules solved.
- **resume_v3.out**: Resumed run after interruption.

### Experiments (10 files, ~85KB)

#### Warmstart
Tests whether pretraining on synthetic rules improves performance on experimental rules.
- Associated script: `run_warmstart_experiment.py`
- Results directory: `results/warmstart_experiment/`

#### Lambda Ablation
Grid search over λ_struct, λ_count, λ_bigram hyperparameters.
- Associated scripts: `experiments/ablate_lambda_weights*.py`
- Results directory: `results_lambda_grid/`, `results_lambda_ablation/`

#### Contrastive
Contrastive recognition model experiments.
- Associated module: `dreamcoder_core/contrastive_recognition.py`

#### Variants
Alternative wake-sleep loop implementations.
- `progressive_wakesleep.out`: Progressive curriculum variant
- `incremental_wakesleep.out`: Incremental learning variant

### Historical (2 files, ~97KB)
These logs document key findings from archived experiments:

- **overnight_listprims.out**: Documents the finding that list primitives (take, drop, zip_with) dramatically improve solve rate from ~8/45 to ~30-40/45 rules.
  - Associated runner: `archived/legacy_runners/run_overnight_listprims.py` (deleted, info extracted to `docs/rule_difficulty_classification.md`)

- **dcyp_overnight.out**: Extended primitives experiment adding DreamCoder and Y&P primitives (fold, foldr, cons, empty, tail). Grammar grew to 76 primitives.

## Deleted Logs (January 2025)

The following logs were deleted as low-value:
- Logs tied to deleted runners (twophase, phase6, 5iter)
- Trivial/empty logs (speed_test, pretraining_test, etc.)
- Ad-hoc test logs (deep_enum, parallel_exp, sanity_check, etc.)

## Current Practice

New experiment logs should be placed in the appropriate results directory:
- `results/overnight_v3/` for production runs
- `results_*/` for specific experiments
