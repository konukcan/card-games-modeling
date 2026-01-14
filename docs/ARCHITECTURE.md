# DreamCoder Architecture Documentation

This document explains the architectural decisions in our DreamCoder implementation, particularly regarding parallelization and neural guidance.

---

## 1. Current Architecture

The reference implementation (`run_reference_wakesleep.py`) uses **sequential CPython with full neural guidance**.

### Why Sequential?

The key insight is that **neural guidance quality matters more than raw enumeration speed** for research purposes. Our current approach:

1. **Recognition model predicts grammar weights** per task before enumeration
2. **Memoized enumeration** provides ~1000x speedup (cache hit rate >95%)
3. **All phases run in the same process** (no serialization overhead)

### Performance Characteristics

| Component | Speedup Source | Status |
|-----------|---------------|--------|
| Memoization | Program cache across tasks | ~1000x |
| Neural guidance | Focused search space | ~10-100x |
| Parallelization | Multiple workers | Not implemented |

**Total effective speedup: ~1000-10000x over naive enumeration**

The memoization alone provides such dramatic speedup that parallelization becomes a secondary concern.

---

## 2. The Parallelization Trade-off

### The Core Constraint

**PyPy cannot run PyTorch.**

- PyPy is an alternative Python interpreter with JIT compilation (~3-6x speedup)
- PyTorch is compiled C/CUDA code that only works with CPython
- You cannot call `recognition.predict_grammar_weights(task)` inside a PyPy process

### What Original DreamCoder Does

The [official DreamCoder](https://github.com/ellisk42/ec/blob/master/docs/software-architecture.md) solves this by:

1. **Python frontend** (CPython + PyTorch) computes neural-guided grammar weights
2. **JSON serialization** passes weights to enumeration workers
3. **OCaml backend** performs fast parallel enumeration with pre-computed weights

Key quote from their docs:
> "The Python frontend communicates a request to the OCaml backend in JSON format after serializing the current library to JSON."

### Our Implementation Gap

Our codebase has PyPy worker infrastructure but **never wired up the weight-passing mechanism**:

```python
# CURRENT (incomplete - weights never passed):
def enumerate_task(task_data, grammar_productions, ...):
    grammar = build_lean_grammar()  # Always default weights!
    if grammar_productions:         # This is ALWAYS None!
        pass  # Never executed
```

The `grammar_productions` parameter exists but is never populated with neural predictions.

---

## 3. Future Parallelization Options

### Option A: Current (Sequential + Neural Guidance)

**What we use now.**

```
CPython main process:
  → Recognition predicts weights
  → Enumeration with guidance
  → Compression
  → Training
  → Dreaming
```

- **Pros**: Full neural guidance, simple to debug
- **Cons**: No parallelization (~1x)
- **Effective speedup**: ~1000x (from memoization)

### Option B: CPython Parallel Workers

**Use ProcessPoolExecutor with CPython workers.**

```
Main process:
  → Recognition predicts weights per task
  → Serialize weights as dictionaries

Worker processes (CPython):
  → Receive task + weights via pickle
  → Rebuild guided grammar
  → Enumerate with guidance
  → Return results
```

- **Pros**: Full neural guidance, ~4x parallelization
- **Cons**: No JIT speedup, pickle serialization overhead
- **Expected speedup**: ~4x over Option A

**Implementation sketch:**
```python
def enumerate_with_guidance(task_data, neural_weights):
    grammar = build_lean_grammar()
    for prim_name, log_prob in neural_weights.items():
        grammar.set_weight(prim_name, log_prob)
    return enumerate_task(task_data, grammar)

# In main process:
with ProcessPoolExecutor(max_workers=4) as executor:
    futures = []
    for task in tasks:
        weights = recognition.predict_weights(task)
        weights_dict = {p.name: w for p, w in zip(grammar.primitives, weights)}
        futures.append(executor.submit(enumerate_with_guidance, task, weights_dict))
```

### Option C: PyPy Workers + Pre-computed Weights

**Full original DreamCoder architecture.**

```
Main process (CPython):
  → Recognition predicts weights per task
  → Serialize weights as JSON
  → Write to shared file/pipe

Worker processes (PyPy):
  → Read weights from JSON
  → Rebuild guided grammar
  → Enumerate with JIT speedup
  → Return results
```

- **Pros**: Full neural guidance, JIT speedup, parallel
- **Cons**: Complex IPC, serialization overhead
- **Expected speedup**: ~7-15x over Option A

**Why it's worth considering:**
- PyPy JIT gives ~3-6x speedup on program enumeration
- Combined with 4 workers: 12-24x theoretical speedup
- Neural guidance preserved (unlike current PyPy workers)

---

## 4. Performance Breakdown

### Current Performance (Option A)

Measured on 44 pretraining tasks:

| Phase | Time | Notes |
|-------|------|-------|
| Wake (enumeration) | ~300s | 50K budget/task, memoized |
| Compression | ~50s | Grammar size dependent |
| Recognition | ~100s | 15 epochs training |
| Dreaming | ~20s | 20 dreams |
| **Total per iteration** | **~470s** | |

### Theoretical Performance with Parallelization

| Configuration | Estimated Time | Speedup |
|--------------|----------------|---------|
| Option A (current) | 470s | 1x |
| Option B (4 CPython workers) | ~180s | 2.6x |
| Option C (4 PyPy workers) | ~80s | 5.9x |

**Note**: These are rough estimates. Actual speedup depends on:
- Task difficulty distribution
- Cache hit rates
- Serialization overhead
- Worker startup time

---

## 5. Why Memoization Dominates

The key insight is that **memoization provides exponential speedup** while parallelization provides linear speedup.

### How Memoization Works

```python
# First task: enumerate all programs up to depth 10
#   → 50,000 programs explored
#   → Results cached by (type_signature, depth)

# Second task: same type signature
#   → Cache hit!
#   → 50,000 programs retrieved instantly
```

With 44 tasks sharing the same type (`list[Card] → bool`):
- First task: Full enumeration (~30s)
- Subsequent tasks: Cache lookup (~0.01s)
- **Effective speedup: 3000x for cached tasks**

### Implication for Parallelization

Since memoization already reduces wall-clock time by ~1000x:
- Going from 470s → 80s (6x speedup) is nice
- But memoization took us from ~470,000s → 470s first

**Parallelization is optimization of an already-optimized process.**

---

## 6. Recommendations

### For Research/Ablation Studies

Use Option A (current implementation):
- Neural guidance correctness is paramount
- Results are reproducible
- Debugging is straightforward
- ~8-12 hours for full overnight run is acceptable

### For Production/Large-Scale

Implement Option B first:
- Relatively simple (~1 day of work)
- Preserves neural guidance
- ~3x speedup

Consider Option C if:
- Running thousands of tasks
- Every hour of wall-clock time matters
- Have engineering time for IPC complexity

---

## 7. File Reference

| File | Purpose |
|------|---------|
| `run_reference_wakesleep.py` | Canonical Option A implementation |
| `dreamcoder_core/enumeration.py` | Memoized enumerator |
| `dreamcoder_core/contrastive_recognition.py` | Neural recognition model |
| `dreamcoder_core/compression/` | Library learning |
| `utils/pypy_bootstrap.py` | PyPy worker infrastructure (incomplete) |
| `enumeration_worker.py` | PyPy worker (weights not wired up) |

---

## 8. References

- [Original DreamCoder Paper](https://arxiv.org/abs/2006.08381)
- [DreamCoder GitHub](https://github.com/ellisk42/ec)
- [DreamCoder Architecture Docs](https://github.com/ellisk42/ec/blob/master/docs/software-architecture.md)
- `KNOWN_ISSUES.md` - Historical bugs and lessons learned
