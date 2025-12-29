# Test Runs

Quick tests, smoke tests, debugging runs, and validation experiments.

These are NOT production experiments - they're short runs used to:
- Validate code changes work correctly
- Debug issues
- Test specific hypotheses quickly
- Verify timing and performance

## Directory Contents

### results_calibration/
**Date**: December 19, 2024
**Purpose**: Calibration testing for factorial experiment

Validates that:
- All recognition models work
- All dream strategies work
- Task creation is correct
- Enumeration functions properly

### results_parallel_test/
**Date**: December 19, 2024
**Purpose**: Test parallel execution

Validates PyPy worker parallelism works correctly.

### results_quick_test/
**Date**: December 20, 2024
**Purpose**: Quick smoke test

2-minute validation that everything runs.

### results_sanity_check/
**Date**: December 22, 2024
**Purpose**: Sanity check after code changes

Multiple experiment logs showing incremental testing.

### results_smoke_test/
**Date**: December 21, 2024
**Purpose**: Component validation

Tests:
1. Primitive variants
2. Recognition models
3. Dreamers
4. Task creation
5. Enumeration
6. Mini experiment run

### results_speed_test/
**Date**: December 21, 2024
**Purpose**: Performance benchmarking

Measures execution time for different configurations.

### results_test_multi/
**Date**: December 19, 2024
**Purpose**: Multi-condition testing

Tests running multiple conditions in sequence.

### results_timing_check/ & results_timing_test/
**Date**: December 19, 2024
**Purpose**: Timing validation

Measures how long different operations take to validate
time estimates for overnight runs.

### results_fixed_3iter/
**Date**: December 21, 2024
**Purpose**: 3-iteration test after fixes

Validates bug fixes work correctly over multiple iterations.

### results_fixed_fast/
**Date**: December 21, 2024
**Purpose**: Fast validation after fixes

Quick test that fixed code runs without errors.

## When to Use These Results

**Don't use these for analysis** - they're too short and may have debugging code.

Use them to:
- Understand what was being tested at a point in time
- See what parameters were tried
- Debug similar issues in the future
- Understand the debugging process

## Log File Structure

All test runs contain `experiment_*.log` files with format:
```
2025-12-21 13:13:39 - INFO - SMOKE TEST: Validating experiment components
2025-12-21 13:13:39 - INFO -
1. Testing primitive variants...
2025-12-21 13:13:39 - INFO -    ✅ Primitive variants OK
...
```

This shows the validation checklist that runs before experiments.
