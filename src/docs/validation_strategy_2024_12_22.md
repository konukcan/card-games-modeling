# Validation Strategy for Recent Fixes

**Date:** 2024-12-22
**Purpose:** Verify the correctness of three critical fixes
**Test File:** `src/tests/test_validation_fixes.py`
**Run Time:** Under 5 minutes

---

## Executive Summary

| Fix | Status | Validation Result |
|-----|--------|-------------------|
| 1. Recognition Guidance (remove normalize) | **IMPLEMENTED** | PASS - No slowdown |
| 2. Dream Quality (sample_requiring_variable) | **NOT YET IMPLEMENTED** | Confirms need (only 33.7% use $0) |
| 3. enumerate_simple Architecture | **VERIFIED** | PASS - Architecture sound |

---

## Fix 1: Recognition Guidance Speed

### The Bug
In `neural_recognition.py` line ~600-605, the old code called `.normalize_probabilities()` after blending grammar weights:

```python
# OLD (BUGGY) - DO NOT USE
return Grammar(new_productions, self.grammar.log_variable).normalize_probabilities()
```

### The Fix
We removed the `normalize_probabilities()` call because:
1. The blended weights are already valid log-probabilities
2. `TopDownEnumerator` uses relative ordering, not absolute values
3. Normalization was causing unnecessary overhead

```python
# NEW (CORRECT)
return Grammar(new_productions, self.grammar.log_variable)
```

### Validation Method
The test compares enumeration speed with uniform vs biased grammars:

```python
def test_recognition_guidance_speedup():
    # Create baseline (uniform) and guided (biased) grammars
    # Enumerate 1000 programs with each
    # Compare times - guided should NOT be slower

    slowdown_ratio = guided_time / baseline_time
    assert slowdown_ratio <= 1.5  # Allow some variance
```

### Test Results
```
Baseline: 1000 programs in 0.272s (3671/sec)
Guided:   1000 programs in 0.267s (3743/sec)
Slowdown ratio: 0.98x

PASS: Recognition guidance does not slow down enumeration
```

### How to Verify Recognition Guidance Actually Helps
Beyond just "not slowing down," to verify guidance HELPS:
1. Create a task requiring specific primitives (e.g., `filter`, `all`)
2. Boost those primitives in the grammar
3. Count programs enumerated before finding solution
4. Guided should find solution in fewer programs

---

## Fix 2: Dream Quality (Variable Usage)

### The Bug
`Grammar.sample()` often returns programs that don't use the input variable `$0`. For dreams of type `HAND -> BOOL`, we want programs that actually examine the hand, not just return `true` or `false`.

### Current Behavior (BUGGY)
```
Sample 1: (λ and (is_valid $0) (not true)) - uses $0 ✓
Sample 2: (λ false)                        - NO $0! ✗
Sample 4: (λ true)                         - NO $0! ✗
...
Only 33.7% of dreams use $0
```

### The Proposed Fix
Add `sample_requiring_variable()` method to `Grammar` that:
1. Uses rejection sampling to filter out programs that don't use `$0`
2. Or modifies the sampling procedure to guarantee variable usage

### Validation Method
```python
def test_dream_quality_variable_usage():
    # Sample 100 dreams of type HAND -> BOOL
    # Check if each uses $0 (the hand argument)
    # Expected: >80% should use $0 after fix

    usage_rate = uses_variable_count / total_valid
    assert usage_rate >= 0.8
```

### Test Results (Current - Before Fix)
```
Valid samples: 86
Using $0: 29 (33.7%)

WARNING: Only 33.7% of dreams use $0
This indicates the dream quality fix is still needed!
```

### Implementation Approach for sample_requiring_variable()
```python
def sample_requiring_variable(
    self,
    request_type: Type,
    max_depth: int = 6,
    max_attempts: int = 100,
    temperature: float = 1.0
) -> Optional[Tuple[Program, float]]:
    """
    Sample a program that uses the input variable.

    Uses rejection sampling: keep trying until we get a program
    that actually references $0.
    """
    for _ in range(max_attempts):
        result = self.sample(request_type, max_depth, temperature)
        if result is None:
            continue

        program, log_prob = result
        if _uses_variable(program, 0):
            return result

    # Fallback: return any sample (better than nothing)
    return self.sample(request_type, max_depth, temperature)
```

---

## Fix 3: enumerate_simple Architecture

### The Architecture
```
Main Process (Python/CPython)
├── run_overnight_v3.py
│   ├── Uses TopDownEnumerator with recognition guidance
│   └── Falls back to workers for PyPy acceleration

Worker Process (PyPy)
├── enumeration_worker.py
│   └── Uses enumerate_simple (no neural model access)
```

### Why enumerate_simple in Workers?
1. **PyPy Compatibility**: PyPy doesn't work well with PyTorch (neural recognition)
2. **Serialization**: Grammars with neural models can't be pickled easily
3. **Simplicity**: Workers just need basic enumeration, not guided search

### Validation Method
```python
def test_enumerate_simple_architecture():
    # Test 1: enumerate_simple produces programs
    # Test 2: Programs are in depth order (iterative deepening)
    # Test 3: Programs evaluate correctly
    # Test 4: Function types produce lambdas
```

### Test Results
```
Test 3.1: enumerate_simple produces programs - PASS
Test 3.2: Programs are in depth order - PASS
Test 3.3: Programs evaluate correctly - PASS
Test 3.4: Function type enumeration - PASS

Sample programs:
  (λ $0)      f(3)=3
  (λ 1)       f(3)=1
  (λ + $0 1)  f(3)=4
```

### Architecture Is Sound Because:
1. Workers can enumerate without neural models
2. Results are returned as JSON (task_name -> solution)
3. Main process handles neural guidance for initial search
4. Workers handle parallelism for extended search

---

## Running the Validation Suite

### Quick Run (All Tests)
```bash
cd card-games-modelling/src
python3 tests/test_validation_fixes.py
```

### Run Individual Tests
```bash
# Using pytest
python3 -m pytest tests/test_validation_fixes.py::test_recognition_guidance_speedup -v
python3 -m pytest tests/test_validation_fixes.py::test_dream_quality_variable_usage -v
python3 -m pytest tests/test_validation_fixes.py::test_enumerate_simple_architecture -v
```

### Expected Output Summary
```
TEST SUMMARY
======================================================================
  recognition_speed: PASS       # Fix 1 validated
  dream_quality: FAIL/WARNING   # Fix 2 not yet implemented
  enumerate_simple: PASS        # Fix 3 validated
  integration: PASS             # End-to-end works
  normalization_perf: PASS      # Shows overhead saved
```

---

## Next Steps

1. **Fix 2 Implementation**: Add `sample_requiring_variable()` to `grammar.py`
2. **Re-run Tests**: Verify dream quality improves to >80%
3. **Overnight Run**: Launch with all fixes and measure:
   - Programs/second with recognition guidance
   - Dream quality in sleep phase
   - Task solve rate improvement

---

## Appendix: Performance Numbers

### Normalization Overhead
```
Grammar size: 67 primitives
Per normalize_probabilities() call: 0.0147ms

In overnight run (~40,000 calls):
  Overhead eliminated: 0.6 seconds
```

Note: The overhead is small in absolute terms, but removing unnecessary work is still correct practice.

### Enumeration Rates
```
Simple grammar (5 primitives):  ~3700 programs/sec
Full grammar (67 primitives):   ~11 programs/sec (with HAND->BOOL type)
```

The difference is due to the complexity of HAND->BOOL programs requiring card primitives.
