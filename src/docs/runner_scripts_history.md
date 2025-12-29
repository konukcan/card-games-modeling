# Runner Scripts History and Documentation

This document provides historical context for all experiment runner scripts in the project.
Last updated: 2024-12-23

## Active Scripts (Keep)

### run_overnight_v3.py
**Status**: PRIMARY PRODUCTION RUNNER
**First created**: ~Nov 2024
**Results directory**: `results/overnight_v3/`

The main overnight training script with all bug fixes and optimizations:
- Fixed task-result scrambling bug (uses dict keyed by task_name)
- Per-iteration checkpoints for analysis
- 4-phase curriculum learning (easy→medium→hard→intensive)
- Pre-flight validation
- PyPy worker parallelism

This is the canonical overnight runner. Use this for production experiments.

### resume_overnight_v3.py
**Status**: ACTIVE (companion to v3)
**Results directory**: Same as v3 (creates resume subdirectory)

Resumes crashed v3 runs from checkpoints. Designed for the specific case of a crashed Phase 4 run but can be adapted for other checkpoints.

### run_experimental_rules.py
**Status**: ACTIVE EXPERIMENTAL
**Results directory**: Various `results_*` directories

Runs DreamCoder on the 57 experimental rules from `catalogue.py` (the actual human study rules) rather than pretraining rules. Uses Set Transformer recognition model with raw feature correlation encoding.

### experiments/run_recognition_dream_experiment.py
**Status**: ACTIVE EXPERIMENTAL
**Results directory**: `results_factorial/`, `results_calibration/`, etc.

Implements the 2×3 factorial experiment design:
- 2 Recognition Models: GRU (legacy) vs Contrastive (new)
- 3 Dream Strategies: Standard vs Balanced vs Contrastive

This is the most sophisticated experiment framework for comparing different model configurations.

---

## Archived Scripts (Historical Reference)

### run_overnight_v4.py
**Status**: SUPERSEDED (never fully deployed)
**Reason for archival**: Extended v3 with mini-batch wake-sleep that was never fully tested

Attempted to add:
- Part A: Standard batch wake-sleep
- Part B: Mini-batch wake-sleep (compression every N solutions)

The mini-batch approach was promising but never completed due to the shift to Set Transformer experiments.

### run_overnight_cython.py
**Status**: ROTTEN (Cython disabled)
**Results directory**: `results/overnight_cython/` (empty)

Early attempt at Cython acceleration. The approach failed because:
1. Cython modules require Cython-native Primitive objects
2. `build_lean_grammar()` creates Python Primitive objects
3. PyPy workers can't load `.so` files anyway

Key lesson: The main speedup comes from PyPy workers, not Cython. `USE_CYTHON = False` everywhere.

### run_overnight_full_cython.py
**Status**: NEVER WORKED (import fails)

Attempted full Cython pipeline but the Cython modules were never fully built.
This script will crash on import. Delete without remorse.

### run_overnight_optimized.py
**Status**: SUPERSEDED by v3
**Reason**: Early PyPy optimization without the curriculum structure of v3

Added PyPy parallelism but lacked:
- Task-result scrambling fix
- Per-iteration checkpoints
- Curriculum phases

### run_overnight_pretraining.py
**Status**: SUPERSEDED by v3
**Reason**: No parallelism, older architecture

"Option C+ Enhanced" pretraining script.
Key parameters: 256 hidden dim, 150 dreams/iteration, 20 recognition epochs.
Superseded by v3 which added parallelism and better checkpointing.

### run_overnight_listprims.py
**Status**: SPECIALIZED (imports from cython)
**Results directory**: `results/overnight_listprims/`

Multi-phase curriculum using list primitives (take, drop, zip_with, adjacent_pairs, half_len).
Depends on `run_overnight_cython.py` for base classes.
Achieved ~30-40 of 45 rules solved vs 8 without list primitives.

Lesson learned: List primitives significantly improve solve rate.

### run_overnight_smallhands.py
**Status**: UNTESTED EXPERIMENTAL

Hypothesis: Smaller hands (3 cards) have simpler search space.
Curriculum: 3-card → 4-card → 5-card → 6-card hands.
Never fully tested. The hypothesis may still be worth exploring.

### run_twophase_overnight.py
**Status**: SUPERSEDED
**Results directory**: `results/overnight_twophase/`

Two-phase structure:
- Phase 5: Intensive pretraining consolidation (43 rules, 10 iterations)
- Phase 6: Transfer test to full catalogue (50 rules, 8 iterations)

Superseded by more comprehensive v3/v4 approaches.

### run_phase6_transfer.py
**Status**: SUPERSEDED
**Results directory**: `results/phase6_transfer/`

Standalone Phase 6 transfer test that loads trained model from Phase 5.
Requires specific checkpoint files to exist.

### run_topdown_test.py
**Status**: TEST SCRIPT
**Results directory**: `results/topdown_test/`

Tests TopDownEnumerator with full wake-sleep architecture.
22 easy pre-training rules, 3-5 iterations.
Useful for comparing bottom-up vs top-down enumeration approaches.

### run_medium_mdl_test.py
**Status**: TEST SCRIPT
**Results directory**: `results/medium_mdl_test/`

Tests Phase 1 (program refactoring) and Phase 2 (MDL scoring) compression improvements.
Medium-sized run: 5-8 iterations, 30-60 minutes.

### run_5iter_memoized.py
**Status**: SPECIALIZED TEST
**Results directory**: `results_5iter/`

5-iteration test of memoized enumeration with contrastive wake-sleep.
Expected runtime: 20-25 hours.

---

## Deleted Scripts (No Longer Present)

### run_pretraining.py
**Deleted**: 2024-12-23
**Reason**: Very early script using old DreamCoderV2 API, completely superseded

### run_overnight_experiments.py
**Deleted**: 2024-12-23
**Reason**: Used old difficulty/transfer experiment API that no longer exists

### run_overnight_full_cython.py
**Deleted**: 2024-12-23
**Reason**: Never worked (Cython imports fail)

---

## Results Directory Mapping

| Results Directory | Runner Script | Purpose |
|-------------------|---------------|---------|
| `results/overnight_v3/` | run_overnight_v3.py | Main overnight runs |
| `results/overnight_cython/` | run_overnight_cython.py | Cython attempt (empty) |
| `results/overnight_listprims/` | run_overnight_listprims.py | List primitives |
| `results/overnight_twophase/` | run_twophase_overnight.py | Two-phase transfer |
| `results/phase6_transfer/` | run_phase6_transfer.py | Transfer evaluation |
| `results/topdown_test/` | run_topdown_test.py | Top-down enumeration |
| `results/medium_mdl_test/` | run_medium_mdl_test.py | MDL compression |
| `results_factorial*/` | run_recognition_dream_experiment.py | Factorial experiments |
| `results_5iter/` | run_5iter_memoized.py | Memoized enumeration |

---

## Key Lessons Learned

1. **Task-result mapping**: Always use dictionary keyed by task name with `as_completed()`. Never rely on list ordering.

2. **Cython approach failed**: PyPy workers provide the real speedup (~3-6x). Cython modules have serialization issues with multiprocessing.

3. **Curriculum learning helps**: Starting with easy rules builds useful abstractions.

4. **Set Transformer promising**: Newer recognition architecture shows better task discrimination.

5. **List primitives critical**: Adding take/drop/zip_with dramatically improves solve rate.

6. **Checkpointing essential**: Long runs crash. Save checkpoints after every phase.
