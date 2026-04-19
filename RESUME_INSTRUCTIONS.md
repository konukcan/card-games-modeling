# Resume Instructions — Night 3 (if the overnight session died)

**Night context:** This is **Night 3** — the final correctness push for the
MCMC program-search review. Night 2 ended at 8.0/10 ("Almost"). Night 1 is
archived at `night1-archive/`. Night 2's artifacts live at the worktree root
(`MORNING_REPORT.md` etc.) and in `review-stage/`. Night 3 state will overlay
`review-stage/` as new rounds complete.

## Environment

- **Worktree:** `~/Documents/self-explanations-project/card-games-modelling/.worktrees/aris-mcmc-night3`
- **Branch:** `aris/mcmc-review-night3` (off `feature/mcmc-search`)
- **Python:** `~/miniforge3/bin/python`
- **Codex MCP:** expected `✓ Connected` (verify: `claude mcp list | grep -i codex`)
- **Hard time budget:** 10 h from launch

## Scenario A — The `claude` session is still open but stuck on an error

Type this into the existing session:

> The network dropped. Please retry the last action and continue the run.
> If the retry fails on the same call, fall back per
> `ARIS_LAUNCH_PROMPT_NIGHT3.md` (Claude sub-agent reviewer). Keep going.

## Scenario B — Terminal is closed or `claude` exited

```bash
cd ~/Documents/self-explanations-project/card-games-modelling/.worktrees/aris-mcmc-night3
claude
```

Paste into the new session:

> Resume the Night 3 MCMC review run. Read `ARIS_LAUNCH_PROMPT_NIGHT3.md`,
> then `review-stage/REVIEW_STATE.json` to see where we stopped. Check
> `review-stage/experiments/round_N/` for any experiments that completed while
> the session was dead. Then continue the `/auto-review-loop` from the next
> round. Do not restart completed rounds.

## Night 3 focus areas (from ARIS_LAUNCH_PROMPT_NIGHT3.md)

1. **Depth-3 calibration** — OOMs on 24 GB. Evaluate workaround strategies
   A–E (smaller grammar, streaming enumeration, sampling-based calibration,
   targeted path-coverage tests, or a custom approach).
2. **Runtime guardrails** for silent approximation caps:
   - `_DEPTH_CAP_EXACT_ENUM_CAP = 16` @ `src/gallery_analysis/mcmc_search.py:793`
     (fallback at :823)
   - `_MARGINALIZATION_FREE_VAR_CAP = 3` @ `src/gallery_analysis/mcmc_search.py:630`
     (fallback at :679 and :769)
   - Reviewer R4: "Add gallery-level assertion or zero-counter. Do not leave
     these as comments only."
3. **Lambda/bound-variable kernel test variant** — add to complement the
   empty-env BOOL-only full-kernel ΣQ=1 test.
4. Open-ended: look for new issues the previous nights may have missed.

## Scope rails (CRITICAL — do not violate)

- **MAY freely modify:** `src/gallery_analysis/mcmc_search.py`,
  `mcmc_hypothesis_collector.py`, `analyze_mcmc.py`, `gallery_rules.py`
  (bug fixes only), diagnostic code, `src/tests/`, `review-stage/experiments/`.
- **MAY make conservative changes to:** `src/dreamcoder_core/grammar.py`
  (weights only), `src/gallery_analysis/dsl_prior.py` (weight tweaks only).
- **FORBIDDEN:** `src/rules/catalogue.py`, `src/rules/cards.py`,
  `src/dreamcoder_core/primitives.py`, `lean_primitives.py`, `type_system.py`,
  anything under `archived/`, `data/`, `night1-archive/` (read-only).
- **If forbidden change is needed:** stop, append to `NOTES_FOR_HUMAN.md`,
  use a workaround.

## Execution rules

- **Caffeinate required** for any script >30 min:
  `nohup caffeinate -d -i -s ~/miniforge3/bin/python … > logfile 2>&1 &`
- **Experiment logs:** `review-stage/experiments/round_N/`
- **PID/CMD files:** write alongside logs
- **90-min cap** per experiment; note overruns in `NOTES_FOR_HUMAN.md`
- **Commit after every logical unit** (conventional prefixes)
- **No `Co-Authored-By: Claude` tags, no `--no-verify`, no force push**
- **Nested engineering review** after each round's fixes:
  `kieran-python-reviewer` → `performance-oracle` → `code-simplicity-reviewer`
  (cap at 2 passes per ARIS round, critical/high-priority findings only)
- **Codex fallback:** if Codex MCP fails, spawn a Claude general-purpose
  sub-agent with harsh-NeurIPS-area-chair persona; note fallback in
  `AUTO_REVIEW.md`
- **Stop condition:** 4 rounds, no early stop (POSITIVE_THRESHOLD requires
  score ≥ 6 AND ≥ 3 rounds completed)

## Sanity checks before resuming

```bash
# See where the run got to
cat review-stage/REVIEW_STATE.json 2>/dev/null || echo "No state yet"

# See what's been committed tonight (N3 commits should start after 9e3aeb0)
git log --oneline aris/mcmc-review-night3 -15

# See if any experiments are still running
for pid_file in review-stage/experiments/round_*/*.pid; do
  [ -f "$pid_file" ] || continue
  pid=$(cat "$pid_file")
  if ps -p "$pid" > /dev/null 2>&1; then
    echo "STILL RUNNING: $pid_file (pid $pid)"
  else
    echo "finished: $pid_file"
  fi
done
```

## Morning report (`MORNING_REPORT.md` — overwrite at end)

Must cover:

1. TL;DR (~5 lines) — rounds / score progression / biggest fixes
2. Per-round breakdown (GPT-5 critique → fix → experiment → internal review)
3. Night 2 follow-ups status (depth-3 / guardrails / lambda test / init
   scoping / H5) — fixed / partially fixed / deferred / new info
4. New issues surfaced
5. `NOTES_FOR_HUMAN.md` flags
6. Final verdict on theoretical correctness — is the MH implementation
   defensible for the paper? What scope limitations remain?
7. Recommended next step for the benchmark-chasing run
