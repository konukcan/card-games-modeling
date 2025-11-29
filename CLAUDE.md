# CLAUDE.md - Coding Agent Guidelines

This document provides guidelines for AI coding agents working on this DreamCoder card game modeling project.

## Pre-Flight Checks (Before Running Any Script)

Before executing any script, especially overnight runs, verify:

1. **Imports resolve correctly** - Run a quick syntax check or import test
2. **Task-result mapping uses dictionary keying** - NEVER rely on list ordering with `as_completed()` (see KNOWN_ISSUES.md for the critical bug this caused)
3. **Holdout verification is enabled** - Solutions must be verified on held-out examples
4. **Primitives are from `lean_primitives.py`** - This is the authoritative primitive library

## Known Bug Patterns to Avoid

### Critical: Task-Result Scrambling
When using `concurrent.futures.as_completed()`, results return in **completion order**, not submission order. Always use task name as dictionary key:

```python
# WRONG - will scramble results
results = []
for future in as_completed(futures):
    results.append(future.result())
for task, result in zip(tasks, results):  # BROKEN MAPPING
    ...

# CORRECT - use dictionary keyed by task name
results_by_name = {}
for future in as_completed(futures):
    result = future.result()
    results_by_name[result['task_name']] = result
```

### Card Object Access
Use named attributes, not indexing:
```python
# WRONG
rank = card[0]
suit = card[1]

# CORRECT
rank = card.rank
suit = card.suit
```

### Exception Handling
Always specify exception types:
```python
# WRONG
except:
    pass

# CORRECT
except ValueError as e:
    logger.error(f"Value error: {e}")
```

### Path Handling
Use relative paths from file location:
```python
# WRONG
path = "/absolute/hardcoded/path"

# CORRECT
from pathlib import Path
path = Path(__file__).parent / "relative" / "path"
```

## Git Workflow

- **Commit regularly** after significant changes that work
- **Push to remote** after verifying changes
- **No Claude collaboration tags** - commits should appear as regular commits
  - No `Co-Authored-By: Claude`
  - No `Generated with Claude Code`
- **Commit message format**: Use conventional commits style
  - `feat: add new primitive for X`
  - `fix: correct task-result mapping in overnight runner`
  - `chore: remove deprecated files`
  - `docs: update README with architecture diagram`

## Logging Requirements

Logging should be **as detailed as possible** to enable post-hoc analysis:

- Track progression over iterations of all relevant model parameters
- Include per-iteration metrics:
  - Training/validation loss
  - Recognition accuracy (per-primitive and overall)
  - Primitive predictions for each task
  - Task embeddings (for later clustering analysis)
  - Attention weights
  - Feature importance scores
- Log timing information for performance analysis
- Include task identifiers in all parallel processing logs

## Updating KNOWN_ISSUES.md

When a salient bug is discovered and fixed, document it in `src/KNOWN_ISSUES.md`:

1. **Severity**: CRITICAL / HIGH / MEDIUM / LOW
2. **Status**: FIXED / OPEN / WORKAROUND
3. **Location**: File and line numbers
4. **Symptoms**: What behavior was observed
5. **Root Cause**: Technical explanation
6. **Fix**: Code showing the solution
7. **Lessons Learned**: What to watch for in future

This serves as institutional memory for future development.

## Code Conventions

- **Type hints** on all function signatures
- **Docstrings** for all public functions
- Use `lean_primitives.py` as the authoritative primitive library
- Cython modules exist but are **NOT currently active** (`USE_CYTHON = False`)
  - Main speedup comes from PyPy workers, not Cython
  - See KNOWN_ISSUES.md for why Cython isn't used (pickle serialization issues)

## File Authority

| Purpose | Authoritative File |
|---------|-------------------|
| Primitives | `src/dreamcoder_core/lean_primitives.py` |
| Main overnight script | `src/run_overnight_cython.py` |
| Known issues | `src/KNOWN_ISSUES.md` |
| Rules catalogue | `src/rules/catalogue.py` |
| Card representations | `src/rules/cards.py` |

## Testing Requirements

Before committing changes to experiment scripts:

1. **Run with small dataset first** - Use a subset of rules before overnight runs
2. **Verify task-result mapping** - Check logs to confirm correct task-solution pairing
3. **Check solution validity** - Ensure valid solutions are not being rejected
4. **Verify holdout accuracy** - Solutions should pass holdout verification

## Project Context

This is a DreamCoder-inspired program synthesis system for modeling card game rule learning. Key components:

- **Recognition Network**: Predicts which primitives are useful for a task
- **Program Enumeration**: Best-first search guided by neural predictions
- **Library Learning**: Compression to find reusable abstractions
- **Wake-Sleep Loop**: Iterative improvement of recognition and enumeration

The companion behavioral experiment is at: https://github.com/konukcan/card-games
