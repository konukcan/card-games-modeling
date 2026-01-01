# Legacy Runner Scripts (Archived)

This directory contains archived test scripts that are preserved for their unique testing infrastructure.

**Note**: These scripts are NOT actively used. Current production runners use ContrastiveRecognitionModel:
- `src/run_incremental_wakesleep.py`
- `src/run_progressive_wakesleep.py`

## Remaining Scripts

| Script | Purpose | Why Kept |
|--------|---------|----------|
| `run_topdown_test.py` | Tests TopDownEnumerator with full wake-sleep | Useful test infrastructure for enumeration experiments |
| `run_medium_mdl_test.py` | Tests MDL-based vs heuristic compression | Only script that compares `compress_frontiers_mdl` |

## Deleted Scripts (January 2025)

The following scripts were deleted because their unique information was either:
- Already documented elsewhere (KNOWN_ISSUES.md, docs/)
- Superseded by `run_overnight_v3.py`
- Never tested / incomplete

| Script | Reason for Deletion |
|--------|---------------------|
| `run_overnight_v4.py` | Mini-batch wake-sleep, never deployed |
| `run_overnight_cython.py` | Cython failed, lesson in KNOWN_ISSUES.md |
| `run_overnight_optimized.py` | Fully superseded by v3 |
| `run_overnight_pretraining.py` | Just config values, superseded by v3 |
| `run_overnight_listprims.py` | Rule classification extracted to `docs/rule_difficulty_classification.md` |
| `run_overnight_smallhands.py` | Hand size curriculum, never tested |
| `run_twophase_overnight.py` | Two-phase structure, v3 has curriculum |
| `run_phase6_transfer.py` | Required specific checkpoints we don't have |
| `run_5iter_memoized.py` | Trivial wrapper script |

## Associated Logs

Output logs from the deleted runners are preserved in `src/logs/historical/`:

| Deleted Runner | Log File | Key Content |
|----------------|----------|-------------|
| `run_overnight_listprims.py` | `logs/historical/overnight_listprims.out` | Documents list primitives finding: 30-40/45 rules solved vs 8 without |
| Extended primitives experiment | `logs/historical/dcyp_overnight.out` | DreamCoder+Y&P primitives (76 primitives total) |

Logs for other deleted runners (twophase, phase6, 5iter) were deleted as they provided no additional insight beyond what's documented.

## Key Lessons Preserved

The important lessons from these scripts are documented in:
- `src/KNOWN_ISSUES.md` - Bug fixes and architectural decisions
- `docs/rule_difficulty_classification.md` - Rule difficulty taxonomy (extracted from listprims)
- `docs/runner_scripts_history.md` - Historical context for all runners
- `src/logs/README.md` - Log organization and descriptions

## Current Scripts (in src/)

### Active (use ContrastiveRecognitionModel)
- `run_incremental_wakesleep.py` - **Current production runner**
- `run_progressive_wakesleep.py` - Alternative curriculum approach

### Legacy (use archived recognition models)
- `run_overnight_v3.py` - Old runner using NeuralRecognitionModel
- `run_overnight_set_transformer.py` - Old runner using SetTransformerRecognitionModel
- `run_experimental_rules.py` - Set Transformer experiments
