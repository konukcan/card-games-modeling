# Experimental Parallel Enumeration

**Status**: NOT CURRENTLY USED - Pending testing and integration

This folder contains experimental code for parallel enumeration that was developed but never integrated into the main pipeline.

## Files

| File | Purpose | Status |
|------|---------|--------|
| `enumeration_optimized.py` | Multiprocessing variants using `mp.Pool` | Untested in production |
| `enumeration_worker.py` | PyPy subprocess worker (reads stdin, writes stdout) | Never called by any code |
| `test_enumeration_optimized.py` | Test file for enumeration_optimized | Only consumer of the optimized code |

## Why These Exist

These files were created to explore parallelization strategies:
- **`enumeration_optimized.py`**: Uses Python's `multiprocessing.Pool` for parallel task enumeration
- **`enumeration_worker.py`**: Designed to be spawned as subprocess workers running in PyPy for better performance

## Why They're Not Used

1. The main `TopDownEnumerator` in `enumeration.py` is sufficient for current experiments
2. The PyPy worker integration was never completed
3. No production script imports or calls these modules

## Known Issues

- Both files import `build_lean_grammar` which may have stale import paths
- `enumeration_worker.py` references the old `primitives` module name
- No integration tests with the actual wake-sleep pipeline

## To Integrate

If you want to use these:
1. Update imports to use current module names (`primitives.py`)
2. Write integration tests with `run_reference_wakesleep.py`
3. Benchmark against single-threaded enumeration
4. Handle task-result mapping carefully (see KNOWN_ISSUES.md for the scrambling bug)
