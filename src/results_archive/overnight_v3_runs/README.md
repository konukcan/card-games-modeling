# Overnight V3 Runs

These are the main production overnight experiments using `run_overnight_v3.py`.

## Runner Script
- **Script**: `src/run_overnight_v3.py` (ACTIVE)
- **Resume**: `src/resume_overnight_v3.py` (ACTIVE)

## Key Features
- Task-result scrambling fix (uses dict keyed by task_name)
- Per-iteration checkpoints
- 4-phase curriculum learning
- PyPy worker parallelism

## Directory Contents

### overnight_v3/
Contains successful overnight v3 runs.

#### run_v3_20251128_215044/
**Date**: November 28, 2024
**Duration**: ~10 hours
**Outcome**: Partially successful (crashed in Phase 4)

Configuration:
- 4 phases: Foundation → Expansion → Deep Search → Intensive Push
- 43 total tasks (22 easy + 21 harder)
- Budget progression: 150K → 250K → 400K → 600K

Results:
- Phase 1: 18/22 easy tasks solved
- Phase 2: 25/43 all tasks solved
- Phase 3: 25/43 tasks solved
- Phase 4: 26/43 tasks (CRASHED at iteration 23)

Key files:
- `run_config.json` - Full configuration
- `checkpoint_phase{1,2,3}_*.pt` - Model checkpoints
- `grammar_phase{1,2,3}_*.json` - Grammar evolution
- `frontiers_phase{1,2,3}_*.json` - Solved programs
- `iteration_checkpoints/` - Per-iteration snapshots
- `report.html` - Analysis report

#### resume_v3_20251129_144431/
**Date**: November 29, 2024
**Purpose**: Resume crashed Phase 4 from checkpoint

This run resumed from the Phase 3 checkpoint and completed the remaining iterations.

## Lessons Learned

1. **System sleep kills long runs**: Must use `caffeinate -d -i -s` wrapper
2. **Checkpoint early and often**: Phase-level checkpoints saved this run
3. **Task-result mapping is critical**: Without dict keying, results scramble
4. **26/43 solve rate** is the baseline for this architecture

## How to Read Results

1. Open `report.html` for visualized analysis
2. Check `run_config.json` for exact parameters
3. Look at `iteration_checkpoints/` for learning curves
4. Load `checkpoint_*.pt` files to resume training
