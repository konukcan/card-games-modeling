# Comprehensive Code Review - December 22, 2024

This document captures the findings from a 6-agent parallel code review of the card-games-modelling project.

---

## Executive Overview

| Review            | Key Finding                                                              | Severity |
|-------------------|--------------------------------------------------------------------------|----------|
| Python Quality    | compression.py (3,351 lines) needs decomposition; 8+ bare except: blocks | HIGH     |
| Pattern Detection | 14+ runner scripts with 80-90% code duplication (~8,000 lines redundant) | CRITICAL |
| Performance       | O(n²) anti-unification bottleneck; 5-20x speedup possible                | HIGH     |
| Silent Failures   | 26 CRITICAL + 18 HIGH severity error handling issues                     | CRITICAL |
| Architecture      | Good layering, but compression.py violates Single Responsibility         | MEDIUM   |
| Simplification    | ~70% LOC reduction possible (12,000 → 3,500 lines)                       | HIGH     |

---

## Progress Tracking

### Completed

- [x] **Replace bare `except:` blocks** (26 locations) - Fixed 2024-12-22
  - All bare except blocks replaced with specific exception types
  - Pattern used: `except (ValueError, TypeError, ZeroDivisionError, IndexError, KeyError, AttributeError, RecursionError):`

- [x] **Unify Task definition** - Fixed 2024-12-23
  - Created canonical `dreamcoder_core/task.py` with unified Task class
  - All 5 duplicate definitions now import from single source
  - Backward compatible (existing imports still work)

- [x] **Add structural hashing for deduplication** - Fixed 2024-12-23
  - Replaced string-based deduplication with hash-based O(1) lookups
  - Updated enumeration.py (30+ locations), all TaskFrontier classes, compression.py
  - Expected 2-3x speedup in deduplication-heavy code paths

- [x] **Rename dreamcoder_v2.py → dreamcoder_original.py** - Fixed 2024-12-23
  - Clarified purpose as reference implementation
  - Updated all 17 files that imported from it

- [x] **Reorganize runner scripts** - Fixed 2024-12-23
  - Reviewed all 19 runner scripts for status (active/outdated/rotten)
  - Created `docs/runner_scripts_history.md` with detailed documentation
  - Moved 11 archived scripts to `legacy_runners/` directory
  - Deleted 3 broken/obsolete scripts (full_cython, pretraining, experiments)
  - Kept 4 active scripts: run_overnight_v3.py, resume_overnight_v3.py, run_experimental_rules.py, run_overnight_set_transformer.py
  - Added legacy_runners/README.md explaining each archived script

### Pending

- [ ] **Consolidate 4 active runner scripts → 1** (DEFERRED - now only 4 active)
- [ ] **Decompose compression.py** (3,351 lines → 6 modules, 1 day)
- [ ] **Cache program size/depth** (program.py, 2h, 2-3x hot path speedup)
- [ ] **Fingerprint-based anti-unification filtering** (compression.py:357-399, 1-2 days, 5-20x speedup)
- [ ] **Extract domain encoder interface** (neural_recognition.py, 4h)
- [ ] **Remove PreFlightValidator** (run_overnight_v3.py:85-442, 30m, -357 LOC)
- [ ] **Type compatibility index** (grammar.py, 4-6h, 5-10x candidate lookup speedup)

---

## Top 10 Action Items (Prioritized)

### Critical (Do First)

| #   | Issue                                    | Location                     | Effort | Impact                               | Status |
|-----|------------------------------------------|------------------------------|--------|--------------------------------------|--------|
| 1   | Replace bare except: blocks              | 26 locations across codebase | 4h     | Prevents silent failures             | DONE   |
| 2   | Reorganize runner scripts                | src/run_*.py                 | 2h     | Cleanup, 11→legacy, 3 deleted        | DONE     |
| 3   | Add structural hashing for deduplication | enumeration.py               | 3h     | 2-3x enumeration speedup             | DONE   |

### High Priority

| #   | Issue                                        | Location                | Effort   | Impact                    | Status  |
|-----|----------------------------------------------|-------------------------|----------|---------------------------|---------|
| 4   | Decompose compression.py                     | 3,351 lines → 6 modules | 1 day    | Maintainability           | PENDING |
| 5   | Cache program size/depth                     | program.py              | 2h       | 2-3x hot path speedup     | PENDING |
| 6   | Fingerprint-based anti-unification filtering | compression.py:357-399  | 1-2 days | 5-20x compression speedup | PENDING |
| 7   | Extract domain encoder interface             | neural_recognition.py   | 4h       | Domain portability        | PENDING |

### Medium Priority

| #   | Issue                     | Location                   | Effort | Impact                         | Status  |
|-----|---------------------------|----------------------------|--------|--------------------------------|---------|
| 8   | Unify Task definition     | 2 duplicate definitions    | 1h     | Single source of truth         | DONE    |
| 9   | Remove PreFlightValidator | run_overnight_v3.py:85-442 | 30m    | -357 LOC                       | PENDING |
| 10  | Type compatibility index  | grammar.py                 | 4-6h   | 5-10x candidate lookup speedup | PENDING |

---

## Detailed Findings by Category

### 1. Error Handling (26 CRITICAL issues) - FIXED

Pattern found across codebase:
```python
# BAD - catches KeyboardInterrupt, MemoryError, etc.
try:
    result = evaluate_program(program, hand)
except:
    return None  # Silent failure!
```

Fix pattern applied:
```python
except (ValueError, TypeError, ZeroDivisionError, IndexError, KeyError, AttributeError, RecursionError) as e:
    # Expected evaluation errors from malformed programs
    return None
```

### 2. Code Duplication (~8,000 redundant lines)

Duplicated 12+ times:
- `format_time()` - identical in 12 files
- `print_banner()` - identical in 12 files
- `serialize_task_for_worker()` - 5 files
- `sample_balanced_examples()` - 2 implementations with different defaults
- `PhaseConfig` dataclass - 10 definitions with slight variations
- `WORKER_SCRIPT` - 130-line string embedded in 4 files

Solution (when ready): Create `experiment_utils.py`:
```python
# Single source of truth for all utilities
from experiment_utils import (
    format_time, print_banner,
    serialize_task_for_worker,
    sample_balanced_examples,
    PhaseConfig
)
```

### 3. Performance Bottlenecks

| Bottleneck            | Current Complexity       | Impact at Scale          | Fix                       |
|-----------------------|--------------------------|--------------------------|---------------------------|
| Anti-unification      | O(n² × m)                | 100x slower at 10x tasks | Fingerprint pre-filtering |
| MDL re-computation    | O(candidates × programs) | 10x slower at 10x tasks  | Incremental caching       |
| String deduplication  | O(n × program_size)      | Memory bloat             | Structural hashing        |
| Type candidate lookup | O(productions) per call  | CPU overhead             | Pre-computed index        |

Quick wins (4-6h total):
1. Structural hashing → 2-3x enumeration speedup
2. Cache program size/depth → 2-3x hot path speedup
3. Sort examples by discrimination → 10-30% faster evaluation

### 4. Architecture Assessment

**Good:**
- Clean layered dependencies (no circular imports)
- Well-designed type/program abstractions
- Correct dependency direction

**Needs work:**
- `compression.py` (3,351 lines) = God Class
- `neural_recognition.py` has domain-specific code that prevents reuse
- Duplicate Task definition in 2 modules

Recommended decomposition for compression.py:
```
compression/
├── __init__.py           # Re-exports
├── anti_unification.py   # anti_unify, find_anti_unified_patterns
├── subtree_finder.py     # find_common_subtrees
├── abstraction.py        # abstract_subtree
├── rewriting.py          # rewrite_with_invention
├── mdl_scoring.py        # compute_mdl, evaluate_invention_mdl
└── core.py               # compress_frontiers
```

### 5. Simplification Opportunities - REORGANIZED

**Previous state**: 14+ separate runner scripts
**Current state**: 4 active scripts + 11 archived in `legacy_runners/`

Active scripts kept:
- `run_overnight_v3.py` - Primary production overnight runner
- `resume_overnight_v3.py` - Resume crashed v3 runs
- `run_experimental_rules.py` - Set Transformer on catalogue rules
- `run_overnight_set_transformer.py` - Set Transformer experiments

Archived to `legacy_runners/`:
- run_overnight_v4.py, run_overnight_cython.py, run_overnight_optimized.py
- run_overnight_pretraining.py, run_overnight_listprims.py, run_overnight_smallhands.py
- run_twophase_overnight.py, run_phase6_transfer.py, run_topdown_test.py
- run_medium_mdl_test.py, run_5iter_memoized.py

Deleted (broken/obsolete):
- run_overnight_full_cython.py, run_pretraining.py, run_overnight_experiments.py

See `docs/runner_scripts_history.md` for full documentation of each script's purpose and lessons learned.

---

## Recommended Roadmap

### Phase 1: Quick Performance Wins (Current Focus)
- [x] Replace all bare `except:` blocks (4h) - DONE
- [ ] Add structural hashing to enumeration (3h)
- [ ] Cache program size/depth (2h)

### Phase 2: Code Quality
- [ ] Decompose compression.py into submodules (8h)
- [ ] Extract domain encoder interface (4h)
- [ ] Unify Task definition (1h)

### Phase 3: Performance Deep Dive
- [ ] Implement fingerprint-based anti-unification filtering (8-16h)
- [ ] Add incremental MDL computation (6-8h)

### Phase 4: Consolidation (Deferred)
- [ ] Create experiment_utils.py with shared code (4h)
- [ ] Build unified run_dreamcoder.py (8h)
- [ ] Delete redundant runner scripts (2h)

---

## Agent IDs for Follow-up

If you want to dive deeper into any specific review, these agent IDs can be resumed:

| Review            | Agent ID |
|-------------------|----------|
| Python Quality    | a5330c9  |
| Pattern Detection | a87d298  |
| Performance       | ade6a1d  |
| Silent Failures   | a990ef6  |
| Architecture      | a3b739b  |
| Simplification    | aa2342a  |

---

## Notes

- **Runner scripts reorganized 2024-12-23**: 11 scripts archived to `legacy_runners/`, 3 deleted, 4 kept active
- Historical scripts preserved in `legacy_runners/` with full documentation in `docs/runner_scripts_history.md`
- Key lessons from archived scripts documented (Cython failure, PyPy success, list primitives importance)
- **Results directories reorganized 2024-12-23**: All scattered `results_*` folders consolidated into `results_archive/`:
  - `results_archive/overnight_v3_runs/` - Main production runs
  - `results_archive/factorial_experiments/` - 2×3×3 factorial comparisons
  - `results_archive/contrastive_experiments/` - Contrastive learning, memoization, Set Transformer
  - `results_archive/specialized_experiments/` - List prims, MDL, transfer learning, Cython attempts
  - `results_archive/test_runs/` - Quick tests, smoke tests, debugging
  - Each directory has README.md explaining contents and lessons learned
