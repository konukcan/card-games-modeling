# ARIS Overnight Run — MCMC Rule Induction Review

**Paste everything below the divider into a FRESH Claude Code session launched from this worktree directory.** This prompt is self-contained; do not assume any prior conversation context.

---

## Your mission tonight

You are running a fully autonomous overnight research-review loop on an MCMC program-search system for card-game rule induction. The goal is to **improve MCMC quality** — so far the sampler converges on contrived programs, shows suspicious biases, and is in its infancy stage. The end goal is exploratory + proof-of-concept quality that the researcher (a PhD cognitive scientist) can share with collaborators.

**You do not need to determine the root cause yourself.** That's what the review loop is for. The external reviewer (GPT-5 via Codex MCP) will diagnose whether issues live in the grammar, proposal distribution, likelihood, annealing schedule, or some combination.

## Working environment — READ CAREFULLY

- **Working directory:** `/Users/cankonuk/Documents/self-explanations-project/card-games-modelling/.worktrees/aris-mcmc-review`
- **Branch:** `aris/mcmc-review-20260411` (isolated — based on `feature/mcmc-search`)
- **DO NOT touch the main repo.** You are in a git worktree. All your commits go to the `aris/mcmc-review-20260411` branch. The original `feature/mcmc-search` branch must remain untouched.
- **Python:** `~/miniforge3/bin/python` — conda/miniforge env. All existing scripts assume this.
- **Caffeinate requirement:** per project convention, any script expected to run >30 min MUST be launched with `nohup caffeinate -d -i -s ... &`. The project's `CLAUDE.md` (read it) is explicit about this.

## Core MCMC files (where quality issues likely live)

- `src/gallery_analysis/mcmc_search.py` — Metropolis-Hastings chain, subtree-regeneration proposals, size-principle likelihood, β-annealing
- `src/gallery_analysis/mcmc_hypothesis_collector.py` — collects MCMC samples into hypothesis pools
- `src/gallery_analysis/analyze_mcmc.py` — gallery-wide MCMC orchestrator
- `src/gallery_analysis/run_overnight_pipeline.py` — overnight runner entry point
- `src/tests/test_mcmc_search.py`, `src/tests/test_mcmc_hypothesis_collector.py` — existing tests (run these before any changes)
- `docs/plans/2026-03-17-mcmc-program-search.md` — design doc for MCMC system
- `docs/plans/2026-03-18-mcmc-fixes-design.md` + `2026-03-18-mcmc-fixes-plan.md` — recent fix plans

## Scope rules — STRICTLY ENFORCED

### ✅ You MAY freely modify:
- `src/gallery_analysis/mcmc_search.py` — the core sampler (proposals, likelihood, annealing, acceptance logic, diagnostics)
- `src/gallery_analysis/mcmc_hypothesis_collector.py`
- `src/gallery_analysis/analyze_mcmc.py`
- Any diagnostic/plotting/evaluation code
- New files under `src/gallery_analysis/` for diagnostics, experiments, ablations
- `src/tests/` — you should add tests as you fix things

### ⚠️ You MAY make CONSERVATIVE changes to:
- `src/dreamcoder_core/grammar.py` — **only** production probabilities / prior weights
- `src/gallery_analysis/dsl_prior.py` — weight tweaks only
- **FORBIDDEN in grammar/primitives code:** adding or removing primitives, changing primitive type signatures, restructuring the grammar hierarchy, changing de Bruijn index handling

### ❌ You MAY NOT touch:
- `src/rules/catalogue.py` — ground-truth rule catalogue (must stay frozen for result comparability)
- `src/rules/cards.py` — card representations
- `src/dreamcoder_core/primitives.py` (and `lean_primitives.py`) — primitive definitions/signatures
- `src/dreamcoder_core/type_system.py` — type system
- Anything under `archived/`, `data/`
- The experimental behavioral data

### 🚨 If you believe a forbidden change is necessary:
Stop. Do not make the change. Instead, append a block to `NOTES_FOR_HUMAN.md` (create if needed) explaining:
1. What you wanted to change and why
2. What you did instead as a workaround
3. Why the reviewer raised this

## The loop itself

You will invoke the installed ARIS skill `/auto-review-loop` with these overrides:

```
/auto-review-loop "Improve MCMC program search quality for card-game rule induction. The sampler converges on contrived programs and shows biases. Root causes unknown — could be in grammar weights, subtree-regeneration proposals, likelihood, β-annealing, or initialization. Focus on diagnosing and fixing whatever the reviewer identifies as highest-priority. Scope rules and forbidden files are in ARIS_LAUNCH_PROMPT.md — respect them." — compact: true, human checkpoint: false, difficulty: hard
```

The skill will:
1. Send project context to GPT-5 via Codex MCP for review
2. Receive structured weaknesses + suggestions + score
3. You implement fixes
4. Before proceeding to the next round: **run the nested engineering review** (see below)
5. Loop up to MAX_ROUNDS = 4 or until score ≥ 6/10 or verdict is "accept"/"sufficient"/"ready for submission"

### Nested engineering review (critical — do this after each round's fixes, BEFORE looping back)

After implementing GPT-5's suggested fixes in each round:

1. **Run `kieran-python-reviewer`** (via Agent tool, subagent_type `compound-engineering:review:kieran-python-reviewer`) on your diff
2. **Run `performance-oracle`** (via Agent tool, subagent_type `compound-engineering:review:performance-oracle`) — MCMC is perf-sensitive
3. **Run `code-simplicity-reviewer`** (via Agent tool, subagent_type `compound-engineering:review:code-simplicity-reviewer`)
4. Address **critical** and **high-priority** findings only. Skip nit-level style.
5. **Cap the internal review loop at 2 passes per ARIS round.** If reviewers still complain after 2 passes, commit what you have and move on — the next ARIS round will catch residual issues.

### Codex quota fallback

The user is on ChatGPT Plus ($20/mo), not Pro. If any Codex MCP call fails with a quota/auth error:

1. Do NOT abort the loop
2. Spawn a Claude sub-agent via the Agent tool (general-purpose subagent) with this system-ish instruction inside the prompt:
   > You are a harsh, skeptical scientific reviewer — think NeurIPS area chair who hates the paper. Given the following MCMC research code and results, produce a structured review with a score 0-10, a ranked list of weaknesses, and concrete suggestions. Start from the assumption that the implementation has bugs until proven otherwise. Do NOT be nice.
3. Feed it the same context you would have sent Codex (current diff, latest MCMC run logs, any notes)
4. Treat its response as the reviewer output and continue the loop
5. Note in `AUTO_REVIEW.md` that this round used the Claude fallback

## Experiment execution rules

When the reviewer suggests running an experiment to validate a fix:

1. **Caffeinate always:** `nohup caffeinate -d -i -s ~/miniforge3/bin/python ... > <logfile> 2>&1 &`
2. **Use small runs for validation:** Don't launch the full overnight pipeline unless it's the final round. For quick validation, use something like `--n-steps 5000 --n-chains 4` or whatever the equivalent flag is — inspect the CLI args first.
3. **Log location:** all experiment output to `review-stage/experiments/round_N/` (create as needed)
4. **Time-box experiments:** no single experiment should exceed 90 minutes. If a diagnostic run needs longer, note it in `NOTES_FOR_HUMAN.md` and skip it.

## Hard time budget: 10 hours from launch

Set a mental budget. If more than 10 hours elapse, **stop cleanly after the current round's commit**, write the morning report, and exit. Do not start a new round after 10 hours.

## Commit discipline

Per this project's (strictly enforced) CLAUDE.md:

- Commit after each logical unit of work (at minimum after each round's fixes and after each experiment)
- Conventional commit prefixes: `feat:`, `fix:`, `chore:`, `docs:`, `test:`
- **No `Co-Authored-By: Claude` or "Generated with Claude Code" tags** — commits should look like regular commits
- Commit BEFORE any experiment launch (so we can diff after)

## The morning report — what the user wakes up to

At the end (either all 4 rounds done, positive verdict reached, or 10h hard stop), write `MORNING_REPORT.md` at the worktree root with:

### 1. Top-level summary (TL;DR, ~5 lines)
- Rounds completed / score progression (e.g., "Round 1: 3/10 → Round 2: 5/10 → Round 3: 6/10 — verdict: sufficient")
- Biggest issues identified
- Biggest fixes made
- Whether MCMC quality demonstrably improved (with numbers if possible)

### 2. Per-round breakdown
For each round:
- What GPT-5 said (score, top 3 weaknesses)
- What you fixed (bullet list, file:line references)
- Experiment results (if any): key numbers, trace plots if made
- What engineering reviewers caught on top

### 3. Uncertainty flags / human attention needed
- Anything you skipped and why
- Grammar/primitive questions the reviewer raised that you couldn't resolve within scope
- Anything from `NOTES_FOR_HUMAN.md`

### 4. Comparison: before vs. after
- Key MCMC metrics before round 1 vs. after final round (acceptance rate, chain mixing, posterior spread, top programs found)
- If you added new diagnostics, show them

### 5. Recommended next steps
- What you'd do if you had another night
- What human judgment calls are pending

## Resilience to network interruptions (READ THIS)

This session may die if the user's Wi-Fi drops for more than a couple of minutes (Claude Code is an API client; long network outages kill the session). Your job is to make sure **nothing is lost** if that happens, so a morning relaunch can resume cleanly.

### Checkpoint-frequently rules

1. **Commit after EVERY meaningful unit of work**, not just end of round. Rule of thumb: if you'd be annoyed to redo it, commit it. Specifically commit after:
   - Every ~3-5 file edits
   - Before launching any experiment
   - After each reviewer response is logged
   - After each fix is implemented
2. **Update `review-stage/REVIEW_STATE.json` aggressively.** The ARIS skill writes it at end-of-round; you should ALSO touch it after each of: receiving a review, starting fixes, finishing fixes, launching an experiment. Add a `last_checkpoint_action` field documenting what was just completed.
3. **Append to `review-stage/AUTO_REVIEW.md` as you go**, not in bulk at end-of-round. Cumulative log = survivable.

### Experiments must survive session death

When launching any MCMC run or diagnostic experiment:
- **ALWAYS** use `nohup caffeinate -d -i -s ~/miniforge3/bin/python ... > <logfile> 2>&1 &` pattern
- This makes the experiment a child of init, not of the Claude session — it keeps running even if Claude dies
- Write the launched PID to `review-stage/experiments/round_N/<exp_name>.pid` so the resume session can find and check on it
- Write a `review-stage/experiments/round_N/<exp_name>.cmd` file with the exact command used (for human debugging)

### Resume protocol (this kicks in automatically)

If a morning session is launched with the same prompt:
- ARIS will detect `REVIEW_STATE.json`, see `status: in_progress` and a recent timestamp, and resume at the next round
- **Before resuming,** you must:
  1. Check any PID files in `review-stage/experiments/round_N/` — use `ps -p <PID>` to see if the experiment is still running, finished, or was killed
  2. Read any experiment log files completed overnight — they may already have the results the reviewer asked for
  3. Confirm the git state is clean (if there are uncommitted edits from the dying session, commit them first as `chore: resume overnight run - recovered uncommitted edits`)
- Do NOT restart completed rounds. Do NOT re-run completed experiments.

### If the user launches a morning session manually before you've finished

The user may wake up, see the session is dead, and relaunch. Your resume logic handles this. But if the user relaunches while you're MID-round (the unusual case of network coming back and user relaunching simultaneously), you might have two sessions. Check `review-stage/REVIEW_STATE.json` first thing — if its timestamp is <5 min old and status is `in_progress`, ANOTHER session is active. In that case, exit immediately with a message to the user: "Another session appears active; aborting to avoid conflicts."

### Write `RESUME_INSTRUCTIONS.md` early in your run

As one of your first actions, write `RESUME_INSTRUCTIONS.md` at the worktree root with:
- The exact command to relaunch if the session died
- What files to check to see where the run left off
- Any known-long-running experiments and their PID file paths

This way, if the user wakes up to a dead session, they have a 30-second path to resume without asking you.

## Safety rails

- **No destructive git operations.** No `git reset --hard`, no `git clean -f`, no force push. If you get into a confused state, commit what you have and write a note.
- **No `--no-verify` on commits.**
- **Do not push to remote.** Everything stays local on the `aris/mcmc-review-20260411` branch.
- **Do not modify anything outside this worktree.** No edits to `~/.claude/`, no edits to `~/Documents/self-explanations-project/card-games-modelling/` directly (only the `.worktrees/aris-mcmc-review/` copy).
- **If a command prompts for input interactively, kill it and find a non-interactive equivalent.** Your session will be unattended.

## Getting started — checklist

1. Confirm working directory with `pwd` — should end in `.worktrees/aris-mcmc-review`
2. Confirm branch with `git branch --show-current` — should be `aris/mcmc-review-20260411`
3. Read the project `CLAUDE.md` (at worktree root — it's the card-games-modelling one)
4. Read `docs/plans/2026-03-17-mcmc-program-search.md` and `docs/plans/2026-03-18-mcmc-fixes-*.md` to understand current state
5. Read `src/gallery_analysis/mcmc_search.py` (full file) to understand the sampler
6. Run the existing MCMC tests to establish a baseline: `cd src && ~/miniforge3/bin/python -m pytest tests/test_mcmc_search.py tests/test_mcmc_hypothesis_collector.py -v`
7. Launch the loop: invoke `/auto-review-loop` with the exact argument string from the "The loop itself" section above

You are unattended. Be thoughtful, be methodical, be honest in the morning report. Good luck.
