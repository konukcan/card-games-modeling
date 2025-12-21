# Known Issues and Development Notes

This document chronicles bugs, architectural decisions, and lessons learned during the development of the DreamCoder card game modeling system.

---

## Critical Bugs (Fixed)

### 1. Task-Result Scrambling Bug

**Severity**: CRITICAL
**Status**: FIXED (November 2025)
**Location**: `run_overnight_cython.py` lines 866-928

**Symptoms:**
- Programs like `(eq 14 (max_rank $0))` (perfect solution for `poker_has_ace`) were being rejected
- Log showed "Rejected spurious: poker_two_suits - (λ eq 14 (max_rank $0))"
- Valid solutions were lost; wrong programs were attributed to wrong tasks
- 9/43 tasks solved, but 3+ additional solutions existed in the search space

**Root Cause:**
Python's `concurrent.futures.as_completed()` returns futures in **completion order**, not submission order. The code was incorrectly zipping results with tasks:

```python
# BROKEN CODE:
for future in as_completed(futures):
    result = future.result()
    results.append(result)  # Appended in completion order!

# Then later:
for task, result in zip(tasks, results):  # WRONG MAPPING!
    process_result(task, result)
```

When tasks complete at different times, the mapping becomes scrambled.

**Impact:**
- Programs enumerated for task A were verified against task B's holdout examples
- Valid solutions rejected because they didn't match the wrong task's semantics
- ~50% of mappings still correct by chance (uniform completion times)

**Fix:**
Use a dictionary keyed by task name, extracted from the result itself:

```python
# FIXED CODE:
results_by_name = {}
for future in as_completed(futures):
    result = future.result()
    task_name = result.get('task_name')  # Task name is in the result
    if task_name:
        results_by_name[task_name] = result

# Reconstruct in correct order:
results = [results_by_name.get(task.name, default) for task in tasks]
```

**Lesson Learned:**
Always verify mapping correctness in parallel processing. Never assume order is preserved through asynchronous operations.

---

### 2. Card Object Access TypeError

**Severity**: LOW
**Status**: FIXED
**Location**: Various primitive implementations

**Symptoms:**
```
TypeError: 'Card' object is not subscriptable
```

**Root Cause:**
Code was using `card[0]` and `card[1]` to access rank and suit, but `Card` is a dataclass with named attributes.

**Fix:**
```python
# BROKEN:
rank = card[0]
suit = card[1]

# FIXED:
rank = card.rank
suit = card.suit
```

---

### 3. Multiprocessing Pickle Failure on Program Objects

**Severity**: CRITICAL
**Status**: FIXED (December 2025)
**Location**: `experiments/run_factorial_experiment.py` lines 215-230, 743-759

**Symptoms:**
- Experiments show 0 tasks solved despite workers finding solutions
- Tasks that find solutions show "0 programs enumerated, 0.0s"
- Tasks that DON'T find solutions show normal program counts
- `MaybeEncodingError` in logs (at DEBUG level, easy to miss)

**Root Cause:**
Workers were returning `Program` objects through `ProcessPoolExecutor`, but `Program` objects contain `Primitive.value` fields which are lambda functions. Lambda functions cannot be pickled.

```python
# BROKEN CODE in worker:
entries_data.append({
    'program': entry.program,  # Program contains lambdas - CAN'T PICKLE!
    'program_str': str(entry.program),
    ...
})
```

When a worker found a solution, the return value contained the unpicklable Program object, causing:
```
MaybeEncodingError: "Can't get local object 'make_direct_queries.<locals>.<lambda>'"
```

The exception was caught at DEBUG level, so the result appeared as 0 programs, 0 seconds (failed return masquerading as empty result).

**Why Solutions Appeared to Fail But Non-Solutions Succeeded:**
- Workers that FOUND solutions tried to return Program objects → pickle failed → result lost
- Workers that DIDN'T find solutions returned empty frontiers → no Program objects → success

**Fix:**
1. Worker returns `program_str` (string) instead of `program` (object):
```python
entries_data.append({
    'program_str': str(entry.program),  # STRING only!
    # 'program': entry.program,  # REMOVED
    ...
})
```

2. Receiving code parses the string back to Program:
```python
from dreamcoder_core.program import parse_program
primitives_dict = {str(p): p for p in self.grammar.primitives()}
program = parse_program(entry_data['program_str'], primitives_dict)
```

**Lesson Learned:**
- Never pass objects containing lambdas through multiprocessing
- Test multiprocessing paths with actual solutions, not just empty results
- Log errors at INFO level for multiprocessing failures, not DEBUG
- When "successful" tasks show 0 programs while "failed" tasks show normal counts, suspect return path failures

---

### 4. Cost-Banding Redundant Enumeration Bug

**Severity**: HIGH
**Status**: FIXED (December 2025)
**Location**: `experiments/run_factorial_experiment.py` lines 152-210 (now removed)

**Symptoms:**
- Enumeration running 5-6x slower than expected (~130 prog/s instead of ~700 prog/s)
- Tasks taking 5-10 minutes instead of 30-60 seconds
- Identical tasks solved, just much slower

**Root Cause:**
The "cost-banding" implementation created a NEW `TopDownEnumerator` for each cost band:

```python
# BROKEN CODE:
cost_bound = 15.0
while cost_bound <= 50.0 and not solution_found:
    # BUG: New enumerator resets the `seen` set!
    enumerator = TopDownEnumerator(grammar, ...)
    for program, log_prob in enumerator.enumerate(max_cost=cost_bound):
        # process...
    cost_bound += 5.0  # 15 → 20 → 25 → 30 → 35 → 40 → 45 → 50
```

Each `TopDownEnumerator` has its own `seen: Set[str] = set()`. When cost_bound increases from 15 → 20, a NEW enumerator is created and **re-enumerates all programs with cost ≤ 15 again**.

With 8 cost bands, low-cost programs were enumerated up to **8 times**. This turned O(n) into O(8n).

**Why Cost-Banding Was Wrong:**
`TopDownEnumerator` already implements **best-first search** using a priority queue sorted by cost. Low-cost programs are naturally enumerated first. Cost-banding is completely redundant - the priority queue already guarantees this ordering!

**Impact:**
- Before fix: 131 programs/second
- After fix: 789 programs/second
- **6x speedup** from removing redundant work

**Fix:**
Use a single enumeration pass instead of multiple cost bands:

```python
# FIXED CODE:
enumerator = TopDownEnumerator(grammar, max_depth=max_depth, max_programs=budget)

for program, log_prob in enumerator.enumerate(
    request_type,
    max_cost=50.0,  # Single high bound - priority queue handles ordering
    timeout_seconds=timeout
):
    # process...
```

**Lesson Learned:**
- Understand the algorithm before adding "optimizations" - `TopDownEnumerator` already implements best-first search
- Always benchmark before and after changes to catch performance regressions
- A 6x slowdown was masked by increasing timeouts from 30s to 300s, making the bug harder to notice

---

## Architecture Decisions

### Cython Implementation Status

**Current State:**
- **Code**: 100% complete for core modules (2,696 lines in `.pyx` files)
- **Compilation**: All `.so` files built successfully (1.45 MB total)
- **Usage**: DISABLED (`USE_CYTHON = False`)

**Why Cython Isn't Currently Used:**

| Issue | Severity | Details |
|-------|----------|---------|
| Pickle serialization | HIGH | Cython extension types don't pickle by default |
| Multiprocessing | HIGH | Workers need to pickle tasks/results between processes |
| Checkpoint/resume | HIGH | Saving state requires pickling grammar |
| PyPy workers | HIGH | PyPy cannot load `.so` files compiled for CPython |

**Technical Details:**

The format difference between Python and Cython primitives:

```python
# Python Primitive (program.py)
@dataclass(frozen=True)
class Primitive(Program):
    name: str
    tp: Type
    value: Any  # Python callable

# Cython Primitive (program_cy.pyx)
cdef class Primitive(Program):
    cdef readonly str name
    cdef readonly object tp
    cdef readonly object value
```

These are structurally identical but Cython `cdef class` types are C extension types that don't support Python's pickle protocol by default.

**Is This Insurmountable?**

**NO** - Engineering solutions exist:

1. **Hybrid Mode (Recommended, 1 day work)**:
   - Use Cython for enumeration only (the O(n²) hot path)
   - Keep Python primitives (fast enough)
   - Keep Python for neural recognition and compression
   - Result: ~60% of possible speedup, zero compatibility issues

2. **Full Cython with Pickle Fix (2-3 days work)**:
   ```cython
   cpdef tuple __reduce__(self):
       """Enable pickle support for Cython primitives."""
       return (Primitive, (self.name, self.tp, self.value))
   ```
   Add `__reduce__` to all Cython types to enable pickling.

3. **Protocol-Based Adapter (4-5 days, most elegant)**:
   - Define `IPrimitive` protocol both implementations satisfy
   - Grammar works with protocol, not concrete types
   - Add conversion utilities between representations

**Cost-Benefit Summary:**

| Approach | Speedup | Dev Time | Risk |
|----------|---------|----------|------|
| Current (PyPy only) | 7-15x | Done | None |
| Hybrid Cython | 10-20x | 1 day | Low |
| Full Cython | 15-30x | 3-5 days | Medium |

**Current Speedup Strategy (Without Cython):**
- PyPy JIT workers: 3-6x speedup
- Early pruning: 1.5-2x speedup
- Parallel workers (4x): Near-linear scaling
- **Combined**: ~7-15x speedup over naive Python

---

### Per-Iteration Model Tracking

**Current State:**
Only final model weights are saved. Per-iteration data is lost:
- Task embeddings
- Attention weights
- Feature importance scores
- Primitive predictions

**What Should Be Tracked (for future runs):**

```python
iteration_checkpoint = {
    'iteration': i,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),

    # Per-task analysis:
    'task_embeddings': {task_name: tensor},
    'attention_weights': {task_name: weights},
    'primitive_predictions': {task_name: logprobs},

    # Interpretability:
    'feature_importance': {task_name: {feature: score}},
    'embedding_pca': pca_components,

    # Performance:
    'accuracy_per_primitive': {prim: acc},
    'recognition_accuracy_on_unseen': float,
}
```

This enables post-hoc analysis of:
1. How recognition accuracy evolves over iterations
2. Which features become important when
3. How task embeddings cluster and evolve

---

## Known Limitations

### Not Yet Implemented

1. **Transfer Analysis**: No mechanism to measure how well learned abstractions transfer to new task families.

2. **Compression Phase**: The compression/abstraction learning phase is simplified compared to full DreamCoder.

3. **Dream Phase**: No dreaming of new tasks from learned grammar.

### Code Quality Issues

1. **Bare Exception Handlers**: Some modules use `except:` without specifying exception types, which can mask unexpected errors.

2. **Hardcoded Paths**: Some scripts have hardcoded paths that may need adjustment for different environments.

---

## Lessons Learned

1. **Parallel Processing**: Always verify mapping correctness. Use explicit keys/IDs rather than relying on order.

2. **Cython Integration**: Plan serialization strategy before writing Cython code. The performance gains are real but integration complexity is non-trivial.

3. **Logging**: Include task identifiers in all log messages when processing in parallel.

4. **Holdout Verification**: Essential for preventing spurious solutions. Always verify on held-out examples.

---

## File Cleanup Notes

The following files were identified as orphaned and removed:
- `src/view_report.py` - For old demo_report.json format, replaced by `generate_overnight_report.py`
- `src/run_demo_with_report.py` - Demo-only runner, superseded by `run_curriculum.py` and overnight scripts

The following files are actively used and must be kept:
- `dreamcoder_core/html_report.py` - Used by `run_curriculum.py` (lines 37, 439-444)
- `generate_full_report.py` - Comprehensive rule catalogue with real neural network data
- `generate_overnight_report.py` - Primary overnight report generator
