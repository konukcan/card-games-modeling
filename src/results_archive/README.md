# Results Archive

This directory contains organized experiment results from the DreamCoder card game modeling project.

## Directory Structure

```
results_archive/
├── overnight_v3_runs/      # Main production overnight runs (run_overnight_v3.py)
├── factorial_experiments/  # 2×3×3 factorial experiments (run_recognition_dream_experiment.py)
├── contrastive_experiments/ # Contrastive wake-sleep experiments
├── specialized_experiments/ # List primitives, MDL testing, etc.
└── test_runs/              # Quick tests, smoke tests, debugging runs
```

## Quick Navigation

| If you're looking for... | Go to... |
|--------------------------|----------|
| Main overnight results | `overnight_v3_runs/` |
| Recognition model comparisons | `factorial_experiments/` |
| Contrastive learning results | `contrastive_experiments/` |
| List primitive experiments | `specialized_experiments/` |
| Debug/test runs | `test_runs/` |

## Runner Script → Results Mapping

| Runner Script | Results Location |
|---------------|------------------|
| `run_overnight_v3.py` | `overnight_v3_runs/` |
| `resume_overnight_v3.py` | `overnight_v3_runs/` (creates resume subdirs) |
| `run_recognition_dream_experiment.py` | `factorial_experiments/` |
| `run_experimental_rules.py` | `contrastive_experiments/` |
| `run_overnight_set_transformer.py` | `contrastive_experiments/` |
| `legacy_runners/run_overnight_listprims.py` | `specialized_experiments/` |
| `legacy_runners/run_medium_mdl_test.py` | `specialized_experiments/` |
| `legacy_runners/run_5iter_memoized.py` | `specialized_experiments/` |

## Understanding Result Files

### Common File Types

- `run_config.json` - Configuration used for the run
- `experiment_*.log` - Detailed execution logs
- `checkpoint_*.pt` - PyTorch model checkpoints
- `grammar_*.json` - Grammar state at checkpoints
- `frontiers_*.json` - Solved tasks and their programs
- `report.html` - Generated analysis report (if available)
- `*_results_*.json` - Summary results with learning curves

### Reading Results

1. **Start with the log file** to understand what happened
2. **Check run_config.json** for parameters used
3. **Look at learning_curve in results JSON** for performance over time
4. **Open report.html** for visualizations (if generated)

## Key Experiments Timeline

- **Nov 28-30, 2024**: First successful overnight v3 runs, discovered task-result scrambling bug
- **Dec 1-3, 2024**: List primitives experiments, transfer learning tests
- **Dec 14, 2024**: MDL compression testing, top-down enumeration
- **Dec 19-21, 2024**: Factorial experiment framework, recognition model comparisons
- **Dec 22-23, 2024**: Contrastive learning, memoized enumeration, deep enumeration tests
