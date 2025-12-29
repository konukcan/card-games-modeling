# Legacy Runner Scripts

This directory contains archived experiment runner scripts that are no longer actively used but are preserved for historical reference.

**DO NOT DELETE THESE FILES** - They document previous approaches and lessons learned.

For detailed documentation on each script, see: `../docs/runner_scripts_history.md`

## Contents

| Script | Original Purpose | Why Archived |
|--------|------------------|--------------|
| run_overnight_v4.py | Mini-batch wake-sleep experiment | Never fully deployed |
| run_overnight_cython.py | Cython optimization | Cython approach failed (PyPy better) |
| run_overnight_optimized.py | Early PyPy parallelism | Superseded by v3 |
| run_overnight_pretraining.py | "Option C+" pretraining | Superseded by v3 |
| run_overnight_listprims.py | List primitives curriculum | Specialized, depends on cython runner |
| run_overnight_smallhands.py | Hand size curriculum | Never fully tested |
| run_twophase_overnight.py | Two-phase transfer | Superseded by v3/v4 |
| run_phase6_transfer.py | Transfer evaluation only | Requires specific checkpoints |
| run_topdown_test.py | TopDownEnumerator testing | Test script only |
| run_medium_mdl_test.py | MDL compression testing | Test script only |
| run_5iter_memoized.py | Memoized enumeration test | Specialized test |

## Active Scripts (in parent directory)

The following scripts are still actively used and remain in `src/`:

- `run_overnight_v3.py` - Primary production overnight runner
- `resume_overnight_v3.py` - Resume crashed v3 runs
- `run_experimental_rules.py` - Set Transformer on catalogue rules
- `run_overnight_set_transformer.py` - Set Transformer experiments
- `experiments/run_recognition_dream_experiment.py` - Factorial experiments

## Deleted Scripts

The following scripts were deleted (not just archived) because they were completely broken or obsolete:

- `run_overnight_full_cython.py` - Import failed (Cython modules don't exist)
- `run_pretraining.py` - Used obsolete DreamCoderV2 API
- `run_overnight_experiments.py` - Used old difficulty/transfer API that no longer exists
