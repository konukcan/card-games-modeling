# ARIS Overnight Run — MCMC Rule Induction Review (NIGHT 3)

**Paste everything below the divider into a FRESH Claude Code session launched from a worktree directory (see setup instructions at bottom).** This prompt is self-contained; do not assume any prior conversation context.

---

## Your mission tonight (Night 3)

You are running a fully autonomous overnight research-review loop on an MCMC program-search system for card-game rule induction. Two previous overnight sessions have completed:

- **Night 1** (3 rounds, score 4.0 → 5.5 → 6.5): Fixed β-annealing, tautology filter, Jeffreys smoothing, first-passage merge, scorer density mismatch.
- **Night 2** (4 rounds, score 4.5 → 5.5 → 7.0 → 8.0): Fixed proposal retry alignment, exact marginalization, depth-cap lookahead mirror, vacuous-lambda target, calibration v2, walker TypeContext threading, exact depth-cap scorer, full-kernel ΣQ=1 test.

**Night 2 ended at 8.0/10 ("almost ready").** The reviewer's final verdict:

> "If the paper says 'this MH implementation targets the post-burn-in posterior under the current gallery grammar' and is honest about the depth-2 calibration limit, the theoretical-correctness section is now in good shape. If it claims full-gallery calibration, or uses early discovery timing as evidence, it is still not ready."

Tonight is the **final correctness push** before the project pivots to benchmark optimization. The goal is to close the remaining gaps and, if possible, reach a "Yes" from the reviewer.

### Prior artifacts (READ THESE FIRST, in this order)

1. `night1-archive/MORNING_REPORT.md` — Night 1 summary (read for background only)
2. `MORNING_REPORT.md` — Night 2 summary (**most important** — read fully)
3. `review-stage/AUTO_REVIEW.md` — Night 2 chronological reviewer output, all 4 rounds
4. `review-stage/REVIEW_STATE.json` — Night 2 final state
5. `review-stage/REVIEWER_MEMORY.md` — reviewer's accumulated memory across Night 2's 4 rounds
6. The commit log: `git log --oneline` — Night 1 + Night 2 commits are all here

**Do not re-do Night 1's or Night 2's fixes.** They are already committed. Tonight starts from where Night 2 left off.

### Tonight's focus: remaining correctness gaps + new issues

This run is still about **theoretical correctness and internal coherence**. Benchmark-chasing (rule recovery rate) is a SEPARATE future run. However, tonight is also **open-ended**: the reviewer should look for any new issues beyond the known list, including ones that might be revealed by fixing the known ones.

### Known remaining issues from Night 2 (prioritized)

**HIGH — the reviewer's remaining conditions for a "Yes":**

1. **Depth-3 calibration.** Night 2's calibration validated at depth=2 only (5/5 seeds pass, mean TV=0.0955). Depth=3 was attempted but OOM-killed on a 24GB machine. **This is a test-infrastructure problem, not a sampler defect.** The bottleneck is `enumerate_programs()` in the calibration script (`review-stage/experiments/round_1/posterior_calibration_v2.py`, lines 215-294), which recursively enumerates all programs up to a depth cap. At depth=3, this produces a combinatorial explosion that exhausts memory. **The sampler itself runs fine at depth=3** — it's only the exhaustive enumeration needed for ground-truth comparison that OOMs. See the section below for workaround strategies.

2. **Runtime guardrails for exactness caps.** Two approximation boundaries exist:
   - `_DEPTH_CAP_EXACT_ENUM_CAP = 16` (mcmc_search.py) — if more than 16 competing lookahead candidates exist at a depth-cap node, the exact 2^n-subset enumeration falls back to a mean-field approximation.
   - `_MARGINALIZATION_FREE_VAR_CAP = 3` — if more than 3 free type variables need marginalization, falls back.
   
   Neither cap is hit in the current gallery grammar, but they are silent: if the grammar grows and hits them, exactness silently degrades. The reviewer's Round 4 recommendation: "Add gallery-level assertion or zero-counter. Do not leave these as comments only."

**MEDIUM:**

3. **Lambda/bound-variable kernel test variant.** The full-kernel ΣQ=1 test (`test_propose_regeneration_full_kernel_normalizes_on_tiny_grammar`) uses an empty-env BOOL-only grammar `{t:BOOL, f:BOOL→BOOL}`. A variant with lambda terms and bound variables would strengthen regression coverage. Reviewer: "not a blocker" but recommended.

**LOW:**

4. **Init-resampling scoping.** An inline documentation block at mcmc_search.py:1981 states that init retries bias early-trajectory analysis. The paper's claims must honor this — no first-passage or cognitive-timing claims treating early trajectory as samples from π. This is a manuscript discipline item, not a code fix.

5. **H5 — sequential chains.** `run_parallel_chains` (mcmc_search.py:2234) runs chains sequentially despite the name. Low priority for correctness.

### Depth-3 calibration: the problem and workaround strategies

The depth-3 calibration bottleneck is NOT about the sampler — it's about the **test harness**. The calibration script needs to enumerate the *entire* support of the grammar (every possible program up to a given depth) to compute the exact posterior, then compare the MCMC chain's empirical distribution against it.

At depth=2, this enumeration produces ~990 programs — tractable. At depth=3, the combinatorial explosion makes it intractable on a 24GB machine.

**Possible workarounds (you should evaluate these and choose the most cost-efficient, or invent your own):**

- **Option A: Smaller grammar.** Design a grammar that is tiny enough for depth-3 exhaustive enumeration (e.g., 2-3 productions), but still exercises the code paths that depth=2 doesn't cover (deeper nesting, more type resolution layers). This doesn't test "full gallery grammar at depth=3" but it tests "depth=3 mechanics are correct."

- **Option B: Streaming/batched enumeration.** Rewrite `enumerate_programs()` to count/accumulate statistics in a streaming fashion without holding all programs in memory simultaneously. Only needs to compute P(program) under the prior and check whether each program satisfies the data — doesn't need to store all programs at once.

- **Option C: Sampling-based calibration.** Instead of exhaustive enumeration, use a sampling-based test: run a long MCMC chain and compare its empirical distribution against properties that must hold under the true posterior (e.g., Geweke test, Cook-Gelman-Rubin, simulation-based calibration). This doesn't require enumerating the support at all.

- **Option D: Targeted path-coverage tests.** Instead of full calibration, write unit tests that specifically exercise depth-3 code paths: verify that the scorer and sampler agree on specific depth-3 programs, that the depth-cap logic triggers correctly at depth boundaries, etc. This is more surgical than full calibration but may be sufficient combined with the depth-2 calibration already passing.

- **Option E: Your own approach.** The reviewer and you may think of something better. Exercise creativity.

The goal is to give the reviewer enough evidence that the depth-3 code paths are correct, without requiring a >24GB machine.

### Gallery rules update

The gallery rule file (`src/gallery_analysis/gallery_rules.py`) was just synced with the JS experiment. Two changes:
- `radial_decreasing` → `radial_increasing` (label fix — logic was already correct)
- `right_half_diamonds` added (group 2)
- File now has 61 rules, matching the JS gallery exactly

These changes are already committed on the branch. If any test references `radial_decreasing` by name, update it.

## Working environment

- **Working directory:** The worktree you create (see setup instructions below)
- **Branch:** Create a new branch `aris/mcmc-review-night3` from `feature/mcmc-search`
- **DO NOT touch the main repo.** You are in a git worktree.
- **Python:** `~/miniforge3/bin/python` — conda/miniforge env
- **Caffeinate requirement:** any script expected to run >30 min MUST be launched with `nohup caffeinate -d -i -s ... &` per project CLAUDE.md convention

## Scope rules — STRICTLY ENFORCED

### You MAY freely modify:
- `src/gallery_analysis/mcmc_search.py` — the core sampler
- `src/gallery_analysis/mcmc_hypothesis_collector.py`
- `src/gallery_analysis/analyze_mcmc.py`
- `src/gallery_analysis/gallery_rules.py` — ONLY bug fixes or test helpers, NOT rule semantics
- Any diagnostic/plotting/evaluation code
- New files under `src/gallery_analysis/` for diagnostics and calibration experiments
- `src/tests/` — add regression tests for each fix
- `review-stage/experiments/` — calibration scripts and artifacts

### You MAY make CONSERVATIVE changes to:
- `src/dreamcoder_core/grammar.py` — ONLY production probabilities / prior weights, NOT structure
- `src/gallery_analysis/dsl_prior.py` — weight tweaks only
- **FORBIDDEN in grammar/primitives code:** adding or removing primitives, changing type signatures, restructuring the grammar hierarchy, changing de Bruijn index handling

### You MAY NOT touch or READ:
- `src/rules/catalogue.py` — DreamCoder wake-sleep rule catalogue (FROZEN — irrelevant to MCMC gallery analysis)
- `src/rules/cards.py` — card representations
- `src/dreamcoder_core/primitives.py`, `lean_primitives.py` — primitive definitions
- `src/dreamcoder_core/type_system.py` — type system
- Anything under `archived/`, `data/`, `night1-archive/` (except for READING artifacts as reference)

**NOTE on the two rule catalogues:** This project has two distinct rule files:
- `src/rules/catalogue.py` (55 rules) — for DreamCoder wake-sleep experiments. **NOT relevant tonight.**
- `src/gallery_analysis/gallery_rules.py` (61 rules) — for MCMC/Bayesian analysis. **This is what the MCMC code uses.**
Do not confuse them.

### If you believe a forbidden change is necessary:
Stop. Append a block to `NOTES_FOR_HUMAN.md` (create if needed) explaining what you wanted to change, why, and what workaround you used instead.

## The loop itself

Invoke the installed ARIS skill `/auto-review-loop` with these overrides:

```
/auto-review-loop "Night 3 (final correctness push) of MCMC program-search review. Night 2 ended at 8.0/10 with three remaining gaps: (1) depth-3 calibration — OOMs on 24GB, need a creative workaround (see ARIS_LAUNCH_PROMPT_NIGHT3.md for strategies), (2) runtime guardrails for silent approximation caps (_DEPTH_CAP_EXACT_ENUM_CAP=16, _MARGINALIZATION_FREE_VAR_CAP=3) — add assertions/counters, (3) lambda/bound-variable kernel test variant. Also open-ended: look for any new issues the previous nights may have missed or that fixing the above may reveal. Read MORNING_REPORT.md and review-stage/AUTO_REVIEW.md for full context. Focus on theoretical correctness, NOT rule-recovery benchmarking." — compact: true, human checkpoint: false, difficulty: hard
```

The skill will:
1. Send project context to GPT-5 via Codex MCP for review (**verify connection with `claude mcp list | grep codex`** before starting)
2. Receive structured weaknesses + suggestions + score
3. You implement fixes
4. Run nested engineering review (see below)
5. Loop up to MAX_ROUNDS = 4

**Stop condition:** 4 rounds, no early stop. Even if score reaches 9+ or 10 in an early round, continue — the reviewer should push for the deepest possible diagnosis. Treat `POSITIVE_THRESHOLD` as requiring BOTH score ≥ 6 AND at least 3 rounds completed.

### Nested engineering review (critical)

After implementing the reviewer's suggested fixes in each round:

1. **Run `kieran-python-reviewer`** (Agent tool, subagent_type `compound-engineering:review:kieran-python-reviewer`) on your diff
2. **Run `performance-oracle`** (Agent tool, subagent_type `compound-engineering:review:performance-oracle`)
3. **Run `code-simplicity-reviewer`** (Agent tool, subagent_type `compound-engineering:review:code-simplicity-reviewer`)
4. Address critical and high-priority findings only. Skip nit-level style.
5. **Cap the internal review loop at 2 passes per ARIS round.**

### Codex fallback

Codex is registered and expected to work. If any Codex MCP call fails:

1. Do NOT abort
2. Spawn a Claude general-purpose sub-agent with a harsh-NeurIPS-area-chair reviewer persona
3. Feed it the same context you would have sent Codex
4. Note in `AUTO_REVIEW.md` that this round used the Claude fallback

## Experiment execution rules

1. **Caffeinate always:** `nohup caffeinate -d -i -s ~/miniforge3/bin/python ... > <logfile> 2>&1 &`
2. **Log location:** `review-stage/experiments/round_N/` (create as needed)
3. **Time-box experiments:** no single experiment >90 min. If a diagnostic needs longer, note it in `NOTES_FOR_HUMAN.md` and skip.
4. **Write PID files:** `review-stage/experiments/round_N/<exp_name>.pid` and `.cmd`

## Hard time budget: 10 hours from launch

Stop cleanly after 10 hours. Write `MORNING_REPORT.md` even if round 4 hasn't finished.

## Commit discipline

- Commit after each logical unit of work
- Conventional prefixes: `feat:`, `fix:`, `chore:`, `docs:`, `test:`
- **No `Co-Authored-By: Claude` or "Generated with Claude Code" tags**
- Commit BEFORE any experiment launch

## The morning report

At the end, write `MORNING_REPORT.md` at the worktree root with:

1. **TL;DR (~5 lines)** — rounds completed / score progression / biggest fixes
2. **Per-round breakdown** — what GPT-5 said, what you fixed (file:line refs), experiment results, internal review findings
3. **Night 2 follow-ups status** — for each of: depth-3 calibration, runtime guardrails, lambda kernel test, init-resampling scoping, H5 sequential chains: fixed / partially fixed / deferred / new info
4. **New issues surfaced** — anything the reviewer raised that wasn't on the prior list
5. **Uncertainty flags / human attention** — anything in `NOTES_FOR_HUMAN.md`
6. **Final verdict on theoretical correctness** — is the MH implementation now defensible for the paper? What scope limitations remain?
7. **Recommended next step** — what should the benchmark-chasing run focus on?

## Resilience to network interruptions

- Commit after every meaningful unit of work (every ~3-5 edits)
- Update `review-stage/REVIEW_STATE.json` aggressively, not just end-of-round
- Append to `review-stage/AUTO_REVIEW.md` as you go
- All experiments via `nohup` so they survive session death
- Write `RESUME_INSTRUCTIONS.md` at the worktree root early in the run

## Safety rails

- No destructive git operations (no `git reset --hard`, no `git clean -f`, no force push)
- No `--no-verify` on commits
- Do not push to remote
- Do not modify anything outside this worktree
- If a command prompts for interactive input, kill it and find a non-interactive equivalent

## Setup instructions (run these BEFORE pasting this prompt)

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling

# Create a fresh worktree for Night 3
git worktree add .worktrees/aris-mcmc-night3 -b aris/mcmc-review-night3 feature/mcmc-search

# Launch Claude Code from the worktree
cd .worktrees/aris-mcmc-night3
```

Then paste everything above the setup section into the fresh Claude Code session.

## Getting started — checklist

1. `pwd` → confirm you're in the worktree
2. `git branch --show-current` → should be `aris/mcmc-review-night3`
3. `claude mcp list | grep -i codex` → should show `✓ Connected`. If not, fall back to Claude sub-agent reviewer and flag in `NOTES_FOR_HUMAN.md`.
4. Read `MORNING_REPORT.md` (Night 2 summary — full read)
5. Skim `review-stage/AUTO_REVIEW.md` (especially Rounds 3-4)
6. Read `review-stage/REVIEWER_MEMORY.md` — this is the reviewer's accumulated understanding
7. Run the existing test suite to confirm green baseline: `cd src && ~/miniforge3/bin/python -m pytest tests/test_mcmc_search.py -v`
   Expected: 47/47 green
8. Write `RESUME_INSTRUCTIONS.md` at worktree root
9. Launch the loop with the exact argument string from "The loop itself" section

You are unattended. Be thoughtful, be methodical, be honest in the morning report. Tonight is the final correctness push — aim for a "Yes" from the reviewer. Good luck.
