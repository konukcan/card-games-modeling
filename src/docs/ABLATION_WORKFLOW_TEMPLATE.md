# Ablation Study Workflow Template

This document defines the **standard two-phase transfer learning workflow** used for ablation studies. Copy and modify for future experiments.

## Workflow Structure

### Two-Phase Design

| Phase | Task Source | Purpose |
|-------|-------------|---------|
| Phase 1 | `pretraining_tasks.json` (44 rules) | Build initial library, train recognition model |
| Phase 2 | `catalogue_tasks.json` (45 rules) | Test transfer with learned abstractions and model |

### Transfer Between Phases

At the end of Phase 1, save and transfer:
1. **Grammar with learned abstractions** - All `Invented` primitives
2. **Recognition model weights** - Trained neural network
3. **Abstraction history** - For logging continuity

## Iteration Budget Schedule

| Iteration | Program Budget | Max Depth | Rationale |
|-----------|---------------|-----------|-----------|
| 1 | 250,000 | 7 | Warm-up, find easy solutions |
| 2 | 500,000 | 7 | Standard exploration |
| 3 | 500,000 | 7 | Standard exploration |
| 4 | 500,000 | 7 | Standard exploration |
| 5 | 1,000,000 | 8 | Final deep search |

**Implementation:**
```python
def get_budget(self, iteration: int) -> int:
    if iteration == 1:
        return 250_000
    elif iteration == 5:
        return 1_000_000
    else:
        return 500_000

def get_max_depth(self, iteration: int) -> int:
    return 8 if iteration == 5 else 7
```

## Standard Configuration

```python
@dataclass
class WorkflowConfig:
    # Phase structure
    n_phases: int = 2
    iterations_per_phase: int = 5

    # Task sources
    phase1_tasks: str = "pretraining_tasks.json"
    phase2_tasks: str = "catalogue_tasks.json"

    # Recognition settings
    recognition_epochs: int = 15
    recognition_lr: float = 1e-3
    recognition_hidden_dim: int = 64

    # Compression settings
    max_inventions_per_iteration: int = 3
    min_compression_savings: float = 2.0
    recognition_alpha: float = 0.7

    # Enumeration settings
    enumeration_timeout: float = 180.0

    # Seeds
    seed: int = 42
```

## Required Logging

Every iteration must log:

### Solutions Found
```python
@dataclass
class SolutionLog:
    task_name: str
    task_family: str
    program: str
    program_size: int
    primitives_used: List[str]
    program_index: int  # Which program number found this
    iteration: int
    phase: int
    log_probability: float
```

### Abstractions Learned
```python
@dataclass
class AbstractionLog:
    abstraction_name: str
    abstraction_body: str
    size: int
    iteration: int
    phase: int
    backward_savings: float
    forward_score: Optional[float]  # If recognition-guided
```

### Iteration Summary
```python
@dataclass
class IterationLog:
    phase: int
    iteration: int
    tasks_solved_cumulative: int
    tasks_solved_new: int
    tasks_total: int
    total_programs_enumerated: int
    programs_budget: int
    max_depth: int
    wake_time_seconds: float
    solutions_found: List[SolutionLog]
    abstractions_found: List[AbstractionLog]
    compression_time_seconds: float
    grammar_size: int
    recognition_loss_final: float
    recognition_time_seconds: float
```

## File Structure

```
results_{experiment_name}/study_{timestamp}/
├── experiment_config.json       # Full configuration
├── summary.json                 # Final comparison summary
│
├── {variant_1}/
│   ├── phase1_iter01.json      # Phase 1, iteration 1 log
│   ├── phase1_iter02.json
│   ├── ...
│   ├── phase1_transfer_state.pkl  # Model + grammar for transfer
│   ├── phase2_iter01.json      # Phase 2, iteration 1 log
│   ├── ...
│   └── full_result.json        # Complete variant result
│
├── {variant_2}/
│   └── ...
└── {variant_3}/
    └── ...
```

## Time Estimates

Based on previous experiments with 44-45 tasks:

| Phase | Iterations | Programs/Iter | Estimated Time |
|-------|------------|---------------|----------------|
| 1 | 5 | 250k-1M | ~2-3 hours |
| 2 | 5 | 250k-1M | ~2-3 hours |

**Per variant: ~4-6 hours**

**3 variants: ~12-18 hours total**

### Scaling Factors
- Programs/second: ~30-50 (pure Python), ~500-1000 (PyPy workers)
- Recognition training: ~30-60 seconds per iteration
- Compression: ~10-30 seconds per iteration

## Quick Test Mode

For validation before overnight runs:

```python
if quick_test:
    workflow.iterations_per_phase = 2
    workflow.recognition_epochs = 5
    # Cap budgets at 50k
```

**Quick test time: ~30-45 minutes total**

## Execution

```bash
# Full overnight run
nohup caffeinate -d -i -s python3 experiments/run_{experiment}_ablation.py \
    > output.out 2>&1 &

# Quick validation
python3 experiments/run_{experiment}_ablation.py --quick-test

# Dry run (config only)
python3 experiments/run_{experiment}_ablation.py --dry-run
```

## Adapting for New Ablations

1. **Copy the template script** (`run_recognition_guided_ablation.py`)
2. **Modify VariantConfig** for your ablation variables
3. **Keep WorkflowConfig unchanged** (unless testing workflow variations)
4. **Update variant definitions** in `main()`
5. **Run quick test** before overnight run

### Example: MDL Weight Ablation

```python
variants = [
    VariantConfig(
        name="MDL_weight_0.5",
        description="Aggressive MDL (grammar_weight=0.5)",
        grammar_weight=0.5,
        ...
    ),
    VariantConfig(
        name="MDL_weight_1.0",
        description="Balanced MDL (grammar_weight=1.0)",
        grammar_weight=1.0,
        ...
    ),
    VariantConfig(
        name="MDL_weight_2.0",
        description="Conservative MDL (grammar_weight=2.0)",
        grammar_weight=2.0,
        ...
    ),
]
```

---

*Template version: 1.0*
*Last updated: January 3, 2026*
