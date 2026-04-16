# ARIS Overnight Run — MCMC Rule Induction Review (NIGHT 2)

**Paste everything below the divider into a FRESH Claude Code session launched from this worktree directory.** This prompt is self-contained; do not assume any prior conversation context.

---

## Your mission tonight (Night 2)

You are running a fully autonomous overnight research-review loop on an MCMC program-search system for card-game rule induction. A previous overnight session (Night 1) ran 3 rounds, terminating early at score 6.5/10 when the positive threshold was reached. Tonight is a **continuation focused on the open theoretical and methodological issues Night 1 identified but did not resolve.**

### Night 1 artifacts (READ THESE FIRST, in this order)

1. `night1-archive/MORNING_REPORT.md` — full summary of Night 1: what was fixed (C2, C3, C4, C6, C1), what open concerns remain, score trajectory 4.0 → 5.5 → 6.5
2. `night1-archive/AUTO_REVIEW.md` — chronological reviewer output across all 3 rounds
3. `night1-archive/REVIEW_STATE.json` — final state, including reviewer's explicit follow-up list
4. The commit log on this branch (`git log --oneline aris/mcmc-review-20260411`) — the last 10 commits are Night 1's work

**Do not re-do Night 1's fixes.** They are already committed on this branch. Tonight starts from where Night 1 left off.

### Tonight's focus: internal coherence, NOT benchmark-chasing

**IMPORTANT FRAMING:** The user is deliberately NOT asking you to optimize for "how many true rules from the catalogue does the sampler recover." That benchmark-chasing objective will be a SEPARATE future overnight run, with different infrastructure (a locked benchmark script + held-out rule split). Tonight is still about **theoretical correctness and internal coherence** — the same target-independent diagnosis that Night 1 did.

Your reviewer (GPT-5 via Codex MCP, see below) should focus on soundness questions: is the MH ratio correct in polymorphic regimes? Does detailed balance hold? Is the calibration methodology actually testing scorer/sampler consistency? Etc.

Do NOT attempt to read the rule catalogue, do NOT try to validate fixes by running the sampler against ground-truth rules, and do NOT optimize for anything resembling "recovery rate." If the reviewer asks you to do such things, push back in the `AUTO_REVIEW.md` record and note: "this is scope for a separate future run; tonight is internal-coherence only."

### Open issues from Night 1 — tonight's expected target list

From `night1-archive/MORNING_REPORT.md` §Open issues, in priority order:

**Critical:**
- **C3-tier2** — Scorer's tier-2 observed-type resolution via `infer_type` (mcmc_search.py:~810–827) is a scoring-side invention NOT in `_sample`'s forward path. Re-introduces forward/reverse asymmetry in polymorphic regimes (up to 5× over-crediting). Recommended: delete tier 2; charge `-log(|_CONCRETE_TYPES|)` for every env-unresolved free var.
- **C1-gallery** — Toy calibration used `{not, and, or}` BOOL→BOOL. Zero polymorphism, no HOFs, no lists. CogSci Bayesian claim is over the FULL gallery grammar. Need second calibration on `{not, and, or, eq, if}` with `_CONCRETE_TYPES=[BOOL, INT]`, depth-3, 500K steps, 5 seeds.
- **C2-cap** — Scorer skips `_sample`'s depth-cap `_all_args_terminable` lookahead branch. Unmeasured; gallery BOOL has no true/false terminals so BOOL holes with no variable in scope go straight into this branch. Need to instrument a counter reporting fraction of `_sample` calls entering this branch; if >1%, must be scored correctly.

**High:**
- **H-methodology** — The calibration target used `_score_subtree_under_sampler` as the prior, which makes the MH ratio independent of whether the scorer matches `_sample`. Current calibration cannot detect scorer/sampler mismatch. Must re-run with `grammar.program_log_likelihood` as the prior.
- **H-autocorr** — MC bound used `n_eff_proxy = n_post_burnin * (1 - rejection_rate)` — not standard ESS. True ESS requires IAT estimation.
- **H-seeds** — Single-seed calibration; need N≥5 seeds.
- **H-n_sites-invariance** — `collect_subtree_sites` excludes Application heads; site counts aren't invariant under trivial restructurings. Need a test that `∑_{s'} Q(s'|s) = 1` at several states.

**Deferred from Night 1 Round 1:**
- **C5** — `sample_program` seed-shift retry breaks determinism.
- **H5** — `run_parallel_chains` runs chains sequentially despite the name.

The reviewer is free to raise additional issues beyond this list. This list is a starting prior, not a constraint.

## Working environment — same as Night 1

- **Working directory:** `/Users/cankonuk/Documents/self-explanations-project/card-games-modelling/.worktrees/aris-mcmc-review`
- **Branch:** `aris/mcmc-review-20260411` — continues the same branch as Night 1; DO NOT create a new branch
- **DO NOT touch the main repo.** You are in a git worktree. The original `feature/mcmc-search` branch must remain untouched.
- **Python:** `~/miniforge3/bin/python` — conda/miniforge env
- **Caffeinate requirement:** any script expected to run >30 min MUST be launched with `nohup caffeinate -d -i -s ... &` per project CLAUDE.md convention

## Scope rules — STRICTLY ENFORCED (same as Night 1)

### ✅ You MAY freely modify:
- `src/gallery_analysis/mcmc_search.py` — the core sampler
- `src/gallery_analysis/mcmc_hypothesis_collector.py`
- `src/gallery_analysis/analyze_mcmc.py`
- Any diagnostic/plotting/evaluation code
- New files under `src/gallery_analysis/` for diagnostics and calibration experiments
- `src/tests/` — add regression tests for each fix

### ⚠️ You MAY make CONSERVATIVE changes to:
- `src/dreamcoder_core/grammar.py` — ONLY production probabilities / prior weights, NOT structure
- `src/gallery_analysis/dsl_prior.py` — weight tweaks only
- **FORBIDDEN in grammar/primitives code:** adding or removing primitives, changing type signatures, restructuring the grammar hierarchy, changing de Bruijn index handling

### ❌ You MAY NOT touch or READ:
- `src/rules/catalogue.py` — ground-truth rule catalogue (FROZEN — do not Read, do not Grep)
- `src/rules/cards.py` — card representations
- `src/dreamcoder_core/primitives.py`, `lean_primitives.py` — primitive definitions
- `src/dreamcoder_core/type_system.py` — type system
- Anything under `archived/`, `data/`, `night1-archive/` (except for READING night1 artifacts as reference)

**Special tonight:** since the benchmark-chasing run is explicitly deferred, you MUST NOT read any file under `src/rules/`. If a fix seems to require knowing about specific catalogue rules, stop and note in `NOTES_FOR_HUMAN.md` — the human will decide whether that scope expansion is warranted.

### 🚨 If you believe a forbidden change is necessary:
Stop. Append a block to `NOTES_FOR_HUMAN.md` (create if needed) explaining what you wanted to change, why, and what workaround you used instead.

## The loop itself

Invoke the installed ARIS skill `/auto-review-loop` with these overrides:

```
/auto-review-loop "Night 2 of MCMC program-search review. Night 1 left off at 6.5/10 with C3-tier2 (scorer tier-2 type resolution not matching _sample in polymorphic regimes), H-methodology (calibration used scorer as prior, masking scorer/sampler mismatch), C1-gallery (calibration only exercised BOOL, not polymorphism/HOFs/lists), C2-cap (unmeasured depth-cap branch), H-autocorr (n_eff_proxy not ESS), H-seeds (single seed), H-n_sites-invariance, plus deferred C5 + H5. Focus on theoretical correctness and internal coherence. Do NOT attempt rule-recovery benchmarking — that is a separate future run. Scope rules in ARIS_LAUNCH_PROMPT.md (night2). Read night1-archive/MORNING_REPORT.md first for context." — compact: true, human checkpoint: false, difficulty: hard
```

The skill will:
1. Send project context to GPT-5 via Codex MCP for review (**Codex is registered and connected tonight** — verify with `claude mcp list | grep codex` before starting; should show `✓ Connected`)
2. Receive structured weaknesses + suggestions + score
3. You implement fixes
4. Run nested engineering review (see below)
5. Loop up to MAX_ROUNDS = 4 or until score ≥ 6/10

**Intended stop condition for Night 2:** aim for 4 rounds even if score ≥6/10 is reached earlier. The user explicitly wants 4 rounds of diagnosis this time, not early-stop. To achieve this, in your ARIS loop invocation, treat the `POSITIVE_THRESHOLD` as requiring BOTH score ≥6 AND at least 3 rounds completed. If score hits 6+ in round 1 or 2, do NOT early-stop; continue so the reviewer can push for deeper fixes (7.5/10+, minor revision, or accept).

### Nested engineering review (same as Night 1 — critical)

After implementing GPT-5's suggested fixes in each round:

1. **Run `kieran-python-reviewer`** (Agent tool, subagent_type `compound-engineering:review:kieran-python-reviewer`) on your diff
2. **Run `performance-oracle`** (Agent tool, subagent_type `compound-engineering:review:performance-oracle`)
3. **Run `code-simplicity-reviewer`** (Agent tool, subagent_type `compound-engineering:review:code-simplicity-reviewer`)
4. Address critical and high-priority findings only. Skip nit-level style.
5. **Cap the internal review loop at 2 passes per ARIS round.**

### Codex fallback (same as Night 1)

Codex is registered and expected to work tonight. If any Codex MCP call nonetheless fails with a quota/auth error:

1. Do NOT abort
2. Spawn a Claude general-purpose sub-agent with a harsh-NeurIPS-area-chair reviewer persona
3. Feed it the same context you would have sent Codex
4. Note in `AUTO_REVIEW.md` that this round used the Claude fallback

## Experiment execution rules

When the reviewer suggests running an experiment:

1. **Caffeinate always:** `nohup caffeinate -d -i -s ~/miniforge3/bin/python ... > <logfile> 2>&1 &`
2. **Log location:** `review-stage/experiments/round_N/` (create as needed)
3. **Time-box experiments:** no single experiment >90 min. If a diagnostic needs longer, note it in `NOTES_FOR_HUMAN.md` and skip.
4. **Write PID files:** `review-stage/experiments/round_N/<exp_name>.pid` and `.cmd` for resume-ability
5. **Tonight's specific expected experiments** (based on Night 1's reviewer recommendations):
   - **H-methodology re-calibration:** toy grammar `{not, and, or}`, target uses `grammar.program_log_likelihood` as prior (NOT `_score_subtree_under_sampler`). ~30 min.
   - **C1-gallery polymorphic calibration:** `{not, and, or, eq, if}`, `_CONCRETE_TYPES=[BOOL, INT]`, `INT→BOOL`, depth-3, 500K steps, 5 seeds, ESS-corrected MC bound. ~1 hour.
   - **C2-cap instrumentation:** counter in `_sample` reporting fraction of calls entering `_all_args_terminable` branch. Run gallery. ~15 min.

## Hard time budget: 10 hours from launch

Stop cleanly after 10 hours. Write `MORNING_REPORT.md` even if round 4 hasn't finished.

## Commit discipline (same as Night 1 — strictly enforced)

- Commit after each logical unit of work
- Conventional prefixes: `feat:`, `fix:`, `chore:`, `docs:`, `test:`
- **No `Co-Authored-By: Claude` or "Generated with Claude Code" tags**
- Commit BEFORE any experiment launch

## The morning report

At the end, write `MORNING_REPORT.md` at the worktree root with:

1. **TL;DR (~5 lines)** — rounds completed / score progression / biggest fixes
2. **Per-round breakdown** — what GPT-5 said, what you fixed (file:line refs), experiment results, internal review findings
3. **Specific Night 1 follow-ups status** — for each of C3-tier2, C1-gallery, C2-cap, H-methodology, H-autocorr, H-seeds, H-n_sites-invariance, C5, H5: fixed / partially fixed / deferred / new info
4. **New issues surfaced** — anything the reviewer raised that wasn't on Night 1's list
5. **Uncertainty flags / human attention** — anything in `NOTES_FOR_HUMAN.md`
6. **Verdict on Bayesian-soundness claim** — given tonight's fixes, is the posterior claim now defensible for the CogSci paper? If not, what specifically still blocks it?
7. **Recommended next overnight** — given tonight's progress, what should the benchmark-chasing run (future overnight) focus on?

## Resilience to network interruptions (same as Night 1)

- Commit after every meaningful unit of work (every ~3-5 edits)
- Update `review-stage/REVIEW_STATE.json` aggressively, not just end-of-round
- Append to `review-stage/AUTO_REVIEW.md` as you go
- All experiments via `nohup` so they survive session death
- Write `RESUME_INSTRUCTIONS.md` at the worktree root early in the run

If a morning session is launched with the same prompt, ARIS resume logic handles the rest.

## Safety rails

- No destructive git operations (no `git reset --hard`, no `git clean -f`, no force push)
- No `--no-verify` on commits
- Do not push to remote
- Do not modify anything outside this worktree
- If a command prompts for interactive input, kill it and find a non-interactive equivalent

## Getting started — checklist

1. `pwd` → confirm `.worktrees/aris-mcmc-review`
2. `git branch --show-current` → should be `aris/mcmc-review-20260411`
3. `claude mcp list | grep -i codex` → should show `✓ Connected`. If not, STOP and note in `NOTES_FOR_HUMAN.md` — the user said Codex should work tonight; if it doesn't, fall back to Claude sub-agent reviewer and flag this.
4. Read `night1-archive/MORNING_REPORT.md` (full), skim `night1-archive/AUTO_REVIEW.md` (round 3 section especially), and `night1-archive/REVIEW_STATE.json`
5. Read `src/gallery_analysis/mcmc_search.py` around lines 592–878 (the C1 fix from Night 1) — this is the main surface area for C3-tier2 and C2-cap
6. Run the existing MCMC test suite to confirm green baseline: `cd src && ~/miniforge3/bin/python -m pytest tests/test_mcmc_search.py tests/test_mcmc_hypothesis_collector.py -v`
   Expected: 47/47 green (the state Night 1 left it in)
7. Write `RESUME_INSTRUCTIONS.md` at worktree root
8. Launch the loop with the exact argument string from "The loop itself" section

You are unattended. Be thoughtful, be methodical, be honest in the morning report. The goal tonight is depth, not speed — 4 rounds of rigorous diagnosis, not early-stop. Good luck.
