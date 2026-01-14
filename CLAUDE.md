# CLAUDE.md - Coding Agent Guidelines

This document provides guidelines for AI coding agents working on this DreamCoder card game modeling project.

---

## ⚠️ CRITICAL: Overnight Run Protocol

**BEFORE LAUNCHING ANY OVERNIGHT/LONG-RUNNING SCRIPT, YOU MUST:**

### 1. Prevent System Sleep with Caffeinate

```bash
# Launch any script, then attach caffeinate to its PID
python3 your_experiment.py &
PID=$!
caffeinate -d -i -s -w $PID &
echo "Process $PID protected by caffeinate"

# OR use nohup + caffeinate in one command:
nohup caffeinate -d -i -s python3 your_experiment.py > output.out 2>&1 &
```

**Caffeinate flags explained:**
- `-d` : Prevent display sleep (monitor can still sleep if `-d` omitted)
- `-i` : Prevent system idle sleep
- `-s` : Prevent system sleep (on AC power)
- `-w PID` : Wait for process PID to finish, then exit caffeinate

### 2. Verify Caffeinate is Running

```bash
ps aux | grep caffeinate | grep -v grep
# Should show: caffeinate -d -i -s -w <PID>
```

### 3. Use nohup for Session Independence

The process should survive terminal/session closure:
```bash
nohup python3 your_experiment.py > output.out 2>&1 &
```

**This is NON-NEGOTIABLE for any run expected to take more than 30 minutes.**

---

## Experiment Scripts

All experiment scripts are in `src/experiments/`.

### Canonical Reference Script

**`run_reference_wakesleep.py`** - The canonical reference implementation for DreamCoder wake-sleep learning. Use this as the template for all new experiments.

```bash
# Quick test (~10-15 minutes)
python3 src/experiments/run_reference_wakesleep.py --quick --verbose 3

# Overnight run (~12 hours)
cd src
nohup caffeinate -d -i -s python3 experiments/run_reference_wakesleep.py --overnight > ref.out 2>&1 &

# Resume from checkpoint
python3 src/experiments/run_reference_wakesleep.py --resume results/run_YYYYMMDD_HHMMSS/
```

**Verbose levels:**
- `--verbose 1`: Iteration summaries only (default)
- `--verbose 2`: + Phase progress (wake/compression/recognition/dreaming)
- `--verbose 3`: + Per-task details and diagnostic info

See `src/experiments/ARCHITECTURE.md` for detailed explanation of design decisions.

### Other Experiment Scripts

| Script | Purpose |
|--------|---------|
| `run_overnight_wakesleep_study.py` | Full wake-sleep loop with recognition training |
| `run_targeted_ablation_study.py` | Ablation studies for model components |
| `run_transfer_study.py` | Transfer learning experiments |
| `run_recognition_compression_ablation.py` | Recognition + compression ablations |

**Legacy scripts** (in `src/`, kept for reference):
- `run_overnight_v3.py` - Original overnight runner
- `run_progressive_wakesleep.py` - Progressive training variant

**Running any experiment**:
```bash
cd src
nohup caffeinate -d -i -s python3 experiments/<script>.py > output.out 2>&1 &
```

---

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

### Context Preservation Commits

**⚠️ MANDATORY - STRICTLY ENFORCED**: To prevent loss of work history when conversation context is summarized:

#### When to Commit (ALL of these are REQUIRED):

1. **IMMEDIATELY after context summarization** - When the conversation is summarized due to context limits, THE VERY FIRST ACTION must be to commit all uncommitted work. Do NOT continue with any other task until this is done.

2. **Before overnight/long runs** - Always commit the current state before launching any long-running script

3. **After each logical unit of work** - Don't batch too many changes; commit after:
   - Creating or significantly modifying a script
   - Completing a debugging session
   - Finishing an analysis
   - Making architectural decisions

4. **At minimum every 30 minutes of active work** - Even if changes seem minor

#### Commit Routine (FOLLOW EXACTLY):

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling
git add -A
git status  # Review what's being committed
git diff --cached --stat  # See summary of changes
git commit -m "feat/fix/chore: <description>

<brief summary of what was accomplished>"
```

#### Self-Check Question:
**Before starting any new task, ask yourself: "Has it been more than 30 minutes or a logical milestone since the last commit?"** If yes, COMMIT FIRST.

This ensures that even if conversation context is lost, the git history preserves what was accomplished. Failure to commit regularly has caused loss of work context in the past.

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

⚠️ CRITICAL: **Also, make sure that for every run we are able to log intermediate results before the end of the run, so we can quick check it along the way.

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
| Recognition model | `src/dreamcoder_core/contrastive_recognition.py` |
| Task generation | `src/dreamcoder_core/task_generation.py` |
| Reference experiment | `src/experiments/run_reference_wakesleep.py` |
| Architecture docs | `src/experiments/ARCHITECTURE.md` |
| Task generation docs | `src/docs/TASK_GENERATION.md` |
| Task visual report | `src/docs/prerecorded_tasks_report.pdf` |
| Known issues | `src/KNOWN_ISSUES.md` |
| Rules catalogue | `src/rules/catalogue.py` |
| Card representations | `src/rules/cards.py` |
| Module status | `docs/MODULE_STATUS.md` |
| Feature status | `docs/FEATURE_STATUS.md` |

### Legacy/Archived
- `archived/legacy_recognition/` - Old recognition models (GRU, Set Transformer) - see README there for why they were superseded
- `archived/legacy_runners/` - Old runner scripts - preserved for reference

## Testing Requirements

Before committing changes to experiment scripts:

1. **Run with small dataset first** - Use a subset of rules before overnight runs
2. **Verify task-result mapping** - Check logs to confirm correct task-solution pairing
3. **Check solution validity** - Ensure valid solutions are not being rejected
4. **Verify holdout accuracy** - Solutions should pass holdout verification

## Overnight Run Debrief Protocol

When the user asks for a "debrief" of an overnight run (or similar phrasing like "what happened last night", "how did the run go", etc.), provide a **comprehensive analysis** that includes:

### 1. Provide Text Summary
Include the following in your text debrief:

- **Run Summary**: Duration, tasks solved/total, success rate, grammar growth
- **Learning Curve**: How metrics evolved across iterations
- **Task Analysis**: Solved tasks by family, search effort distribution
- **Library Evolution**: Number and examples of learned abstractions
- **Unsolved Tasks**: List with brief analysis of why they might be hard
- **Recognition Model**: Training progress, loss trajectory
- **Timing Breakdown**: Enumeration time, programs/second throughput
- **Phase Comparison** (if multi-phase): Performance differences between phases
- **Any anomalies or bugs** observed in logs

### 2. Check for System Issues
If the user mentions system problems (crashes, freezes, unresponsiveness):
- Check `/Library/Logs/DiagnosticReports/` for JetsamEvent files (memory pressure)
- Check for crash reports
- Correlate timestamps with the run timeline
- Provide diagnosis and recommendations

### 3. Results Location
Always mention where the results are saved:
- Main results JSON
- HTML report location
- Iteration checkpoints
- Model files

---

## Project Context

This is a DreamCoder-inspired program synthesis system for modeling card game rule learning. Key components:

- **Recognition Network**: Predicts which primitives are useful for a task
- **Program Enumeration**: Best-first search guided by neural predictions
- **Library Learning**: Compression to find reusable abstractions
- **Wake-Sleep Loop**: Iterative improvement of recognition and enumeration

The companion behavioral experiment is at: https://github.com/konukcan/card-games

---

## Paused Subprojects

### LAPS (Language-Assisted Program Synthesis) - PAUSED

**Status**: Paused as of January 2025
**Location**: `../laps-subproject-PAUSED/` (moved outside main project)
**Resume when**: Core non-language DreamCoder model is working well

This subproject explored using natural language descriptions to guide program synthesis. Key findings:
- LLM descriptions had 50.5% error rate initially (hallucinated poker features)
- Improved prompting reduced errors to 7.5% but lost discrimination
- LLMs focus on surface features (colors 63%, ranks 62%) not structural patterns (8%)

**To resume**: Read `../laps-subproject-PAUSED/LAPS_SUBPROJECT_REPORT.md` for full context.

Files archived:
- `description_generator/` - Core Python module for feature extraction
- `results_llm_annotation/` - All experiment data and PDFs
- `LAPS_SUBPROJECT_REPORT.md` - Comprehensive documentation
