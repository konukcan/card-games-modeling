# Specialized Experiments

Experiments testing specific hypotheses or approaches.

## Runner Scripts (in legacy_runners/)
- `run_overnight_listprims.py` - List primitives curriculum
- `run_twophase_overnight.py` - Two-phase transfer learning
- `run_phase6_transfer.py` - Transfer evaluation only
- `run_medium_mdl_test.py` - MDL compression testing
- `run_topdown_test.py` - Top-down enumeration
- `run_overnight_cython.py` - Cython acceleration attempt

## Directory Contents

### overnight_listprims/
**Date**: December 1, 2024
**Focus**: List primitives hypothesis

Hypothesis: Adding list primitives (take, drop, zip_with, adjacent_pairs, half_len)
dramatically improves solve rate.

Results:
- **Significant improvement**: ~30-40 rules solved vs ~8 without list primitives
- List operations are essential for position-based rules
- Confirmed that primitive set matters a lot

Key lesson: Choose primitives carefully based on domain requirements.

### overnight_twophase/
**Date**: November 30, 2024
**Focus**: Two-phase pretraining→transfer

Phase 5: Intensive consolidation on 43 pretraining rules
Phase 6: Transfer test to 50 catalogue rules

Results:
- Good pretraining consolidation
- Mixed transfer results

### phase6_transfer/
**Date**: December 1, 2024
**Focus**: Transfer evaluation

Loads pretrained model and tests on experimental rules.
Requires checkpoint from overnight_twophase or similar.

### overnight_extended/
**Date**: December 3, 2024
**Focus**: Extended overnight run

Longer run with more iterations to see if performance plateaus.

### medium_mdl_test/
**Date**: December 14, 2024
**Focus**: MDL compression testing

Tests Phase 1 (program refactoring) and Phase 2 (MDL scoring):
- Compares heuristic vs MDL-based compression
- 5-8 iterations, 30-60 minutes

Key findings:
- MDL scoring is more principled but slower
- Heuristic works well in practice

### topdown_test/
**Date**: December 14, 2024
**Focus**: Top-down enumeration

Tests TopDownEnumerator vs bottom-up enumeration:
- 22 easy pretraining rules
- Full wake-sleep architecture

Compares:
- Programs explored per solution
- Time to solution
- Memory efficiency

### overnight_cython/
**Date**: November 28, 2024 (empty)
**Focus**: Cython acceleration attempt

**THIS APPROACH FAILED**

Why it failed:
1. Cython modules require Cython-native Primitive objects
2. build_lean_grammar() creates Python Primitives
3. PyPy workers can't load .so files anyway

Lesson learned: PyPy workers provide the real speedup (~3-6x), not Cython.

## Key Lessons from Specialized Experiments

1. **List primitives matter**: 4x improvement in solve rate
2. **Transfer learning is hard**: Pretraining doesn't always generalize
3. **MDL vs heuristic**: Heuristic is good enough, MDL is slow
4. **Cython doesn't help**: PyPy workers are the answer
5. **Top-down may help memory**: But not speed
