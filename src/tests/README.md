# Tests

Unit and integration tests for the DreamCoder card game modeling system.

## Test Files

| File | Purpose |
|------|---------|
| `test_memoized_enumeration.py` | Memoization correctness tests |
| `test_deep_enumeration.py` | Deep enumeration integration tests |
| `test_compression_refactoring.py` | Tests for compression module refactoring |
| `test_task_generation.py` | Tests for task generation from rules |
| `test_validation_fixes.py` | Tests for solution validation fixes |
| `test_ab_comparison.py` | Tests for A/B pipeline comparison |

**Note**: `test_enumeration_optimized.py` was moved to `dreamcoder_core/experimental_parallel/` (pending integration).

## Running Tests

From the `src/` directory:

```bash
cd src

# Run all tests
python -m pytest tests/

# Run a specific test file
python -m pytest tests/test_deep_enumeration.py

# Run with verbose output
python -m pytest tests/ -v
```

## Test Coverage

These tests cover:
- Enumeration correctness and performance
- Compression/library learning
- Task generation from rule catalogue
- Solution validation edge cases

## Adding New Tests

1. Create a file named `test_<component>.py`
2. Use pytest conventions (`def test_*():`)
3. Import modules from the parent package:
   ```python
   import sys
   sys.path.insert(0, '..')
   from dreamcoder_core.enumeration import ...
   ```
