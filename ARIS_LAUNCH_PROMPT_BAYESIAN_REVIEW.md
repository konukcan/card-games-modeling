# ARIS Overnight Run — Bayesian Enumeration Model Review (Night 1)

**Paste everything below the divider into a FRESH Claude Code session launched from a worktree directory (see setup instructions at bottom).** This prompt is self-contained; do not assume any prior conversation context.

---

## Your mission tonight

You are running a fully autonomous overnight research-review loop on a **Bayesian program-enumeration system** for card-game rule induction. This system enumerates all programs up to a given depth under a typed DSL, groups them into equivalence classes, scores them against observed exemplar hands using the size principle, and computes posteriors to predict human classification difficulty.

This is the **first overnight review** — the codebase has not been adversarially stress-tested for theoretical correctness before. A sister project (MCMC-based sampler on a separate branch) went through 3 nights of review (score 4.0 to 8.0/10) and surfaced 10+ real bugs. Expect to find issues here too.

### What this system does (for the reviewer)

A participant sees 6 poker hands that all satisfy some hidden rule (e.g., "all cards are red" or "the hand is sorted by rank"). The model asks: **given these 6 exemplar hands, what is the posterior probability of each candidate rule?**

The pipeline:
1. **Enumerate** all typed programs up to depth 6 under a 64-primitive DSL
2. **Prune** syntactically redundant programs (double negation, etc.)
3. **Inject** ~260 LLM-generated hypotheses (translated from human NL descriptions)
4. **Group** extensionally equivalent programs into equivalence classes via fingerprinting
5. **Estimate** each class's extension size (how many of the 20M possible 6-card hands satisfy it) via Monte Carlo
6. **Score** using the size principle: P(data | hypothesis) = (1/|extension|)^n, where n = number of exemplars
7. **Compute posteriors** via Bayes' rule with either a canonical prior (shortest program) or summed prior (log-sum-exp across class)
8. **Predict** which new hands humans will find hardest to classify (posterior predictive)

The system runs 10 variant configurations across 4 dimensions: grammar (uniform/weighted), prior mode (canonical/summed), injection (with/without LLM hypotheses), likelihood (strict/noisy).

## Review scope — the FULL Bayesian inference pipeline

The review covers **8 subsystems**. Each must be checked for theoretical correctness, internal coherence, and silent failure modes.

### 1. Enumeration pipeline (`enumerator.py`, 573 lines)
- Correctness of top-down program enumeration (completeness, deduplication, depth handling)
- Syntactic pruning: are redundant programs (reverse(reverse(X)), not(not(X))) correctly rejected?
- Grammar construction: are the 64 primitives and their types correct?
- **Critical question:** Is post-hoc tier re-scoring mathematically equivalent to enumerating under the weighted grammar? **You must answer this with a proof or counterexample.** The enumeration engine enumerates under a uniform grammar, then re-scores programs post-hoc with tier weights. Does this change the support (set of enumerated programs) or only the prior weights? If the tier weights would cause some programs to be pruned (below a log-probability threshold), then post-hoc re-scoring is NOT equivalent.

**Tier weights (defined in `enumerator.py`):**
- TIER_CHEAP (-3.0): eq, lt, le, gt, ge, and, or, not, all, any, if
- TIER_STANDARD (-4.0): rank_val, get_suit, map, filter, sort_by_rank, arithmetic, etc.
- TIER_AGGREGATE (-5.5): count_suit, n_unique_ranks, sum_ranks, etc.
- TIER_ULTRA_SHALLOW (-9.0): has_suit, has_color

### 2. LLM injection pipeline (`injection.py`, 326 lines; `translate_hypotheses.py`, 820 lines; `write_true_rules.py`, 719 lines)
- Correctness of LLM hypothesis loading, parsing, and validation
- Deduplication: are duplicate DSL programs correctly identified and merged?
- Prior computation: do injected hypotheses get correct log-priors via `dsl_prior.py`?
- Merge logic: are injected hypotheses correctly merged into enumerated equivalence classes?
- Data integrity of `data/injected_hypotheses.json` (~401 entries, ~260 unique after dedup)

### 3. Equivalence class tracking (`hypothesis_table.py`, 419 lines)
- Fingerprinting correctness: do extensionally equivalent programs get the same fingerprint?
- Probe set adequacy: 500 random probes currently collapse 16 true rules into the same class — is this acceptable or a bug?
- Trivial filtering: are programs correctly classified as constant (all-True/all-False) on the 360 curated exemplar hands?
- Edge case: rare-but-meaningful programs that don't fire on any exemplar hand

### 4. Extension size estimation (`hypothesis_table.py`, `exemplars.py`, 114 lines)
- Monte Carlo correctness: is P(hand satisfies program) correctly estimated?
- Adaptive two-pass sampling: does the budget allocation work correctly?
- TOTAL_HANDS constant (C(52,6) = 20,358,520) — verify it's used correctly throughout
- Edge cases: programs with empty extensions, programs that timeout during evaluation

### 5. Scoring and posterior computation (`bayesian_scorer.py`, 379 lines; `dsl_prior.py`, 294 lines; `analyze.py`, 1204 lines)
- Size-principle likelihood: is P(D|h) = (1/|ext(h)|)^n correctly computed?
- Noisy likelihood variant: is the epsilon parameter handled correctly?
- Prior modes: canonical (shortest program) vs summed (log-sum-exp across class) — are both correct?
- Posterior normalization: is log-sum-exp stable across ~50K-300K equivalence classes?
- Rule difficulty metrics (entropy, top-1 mass, true-rule rank) — correctly derived?

### 6. Diagnosticity pipeline (`hand_diagnosticity.py`, 474 lines; `run_diagnosticity.py`, 415 lines)
- Posterior predictive P(hand in rule | data): correctly computed from the posterior?
- Sampling: are the 5K-10K random candidate hands representative?
- Accuracy/confidence metrics: correctly aggregated?
- Balanced sampling logic: equal accept/reject hand pairs

### 7. Visualization pipeline (`visualization/`, ~3000 lines total)
- Data loading (`data.py`, 670 lines): are results JSON files correctly parsed into DataFrames?
- DSL-to-English translation (`dsl_translator.py`, 790 lines): are translations correct?
- Chart correctness (`plots.py` 905 lines, `report_rule.py` 288 lines, `report_summary.py` 218 lines)
- Lower priority than subsystems 1-6, but still reviewable

### 8. Variant management (`compare_variants.py` 433 lines, `run_overnight_pipeline.py` 179 lines)
- 10 official variants across grammar x prior x injection x likelihood dimensions
- Are all combinations correctly configured and comparable?
- Is the variant comparison logic (ranking, delta computation) mathematically sound?

## Files in scope

### Core — MUST review:
- `src/gallery_analysis/enumerator.py` — grammar construction, enumeration, syntactic pruning
- `src/gallery_analysis/injection.py` — LLM hypothesis loading, dedup, merge
- `src/gallery_analysis/hypothesis_table.py` — equivalence classes, fingerprinting, extension estimation
- `src/gallery_analysis/bayesian_scorer.py` — likelihood, prior modes, posterior assembly
- `src/gallery_analysis/dsl_prior.py` — prior computation matching enumerator
- `src/gallery_analysis/analyze.py` — pipeline orchestration
- `src/gallery_analysis/hand_diagnosticity.py` — posterior predictive, diagnosticity
- `src/gallery_analysis/run_diagnosticity.py` — diagnosticity orchestration
- `src/gallery_analysis/exemplars.py` — frozen exemplar loading, probe generation
- `src/gallery_analysis/provenance.py` — reproducibility metadata
- `src/tests/` — new regression tests for each fix

### Supporting — SHOULD review:
- `src/gallery_analysis/visualization/data.py` — result loading
- `src/gallery_analysis/visualization/plots.py` — chart functions
- `src/gallery_analysis/visualization/dsl_translator.py` — DSL-to-English
- `src/gallery_analysis/compare_variants.py` — variant comparison
- `src/gallery_analysis/run_overnight_pipeline.py` — batch runner
- `src/gallery_analysis/depth_mass_analysis.py` — posterior mass by depth
- `src/gallery_analysis/fingerprint_resolution.py` — fingerprint distinctiveness

### NOT in scope:
- `src/gallery_analysis/mcmc_search.py` — MCMC sampler (separate branch, already reviewed)
- `src/gallery_analysis/mcmc_hypothesis_collector.py` — MCMC analysis (already reviewed)
- `src/gallery_analysis/analyze_mcmc.py` — MCMC analysis (already reviewed)
- `src/rules/catalogue.py` — DreamCoder wake-sleep catalogue (FROZEN, irrelevant to gallery analysis)
- `src/dreamcoder_core/` — frozen (primitives, types, grammar infrastructure)
- `src/gallery_analysis/gallery_rules.py` — ground truth predicates, frozen (except bug fixes)
- `src/gallery_analysis/explore_*.py` — exploration/profiling scripts (informational only)
- `src/gallery_analysis/translate_phase1b.py` — phase-specific, not core pipeline

**NOTE on two rule catalogues:** This project has two distinct rule files:
- `src/rules/catalogue.py` (55 rules) — for DreamCoder wake-sleep experiments. **NOT relevant tonight.**
- `src/gallery_analysis/gallery_rules.py` (61 rules) — for Bayesian analysis. **This is what the enumeration code uses.**
Do not confuse them.

## Working environment

- **Working directory:** The worktree you create (see setup instructions below)
- **Branch:** `aris/bayesian-review` branched from `feat/grammar-comparison`
- **DO NOT touch the main repo.** You are in a git worktree.
- **Python:** `~/miniforge3/bin/python` — conda/miniforge env
- **Caffeinate requirement:** any script expected to run >30 min MUST be launched with `nohup caffeinate -d -i -s ... &` per project CLAUDE.md convention

## Scope rules — STRICTLY ENFORCED

### You MAY freely modify:
- `src/gallery_analysis/enumerator.py` — the enumeration engine
- `src/gallery_analysis/injection.py` — injection pipeline
- `src/gallery_analysis/hypothesis_table.py` — equivalence classes
- `src/gallery_analysis/bayesian_scorer.py` — scoring engine
- `src/gallery_analysis/dsl_prior.py` — prior computation
- `src/gallery_analysis/analyze.py` — pipeline orchestration
- `src/gallery_analysis/hand_diagnosticity.py` — diagnosticity computation
- `src/gallery_analysis/run_diagnosticity.py` — diagnosticity orchestration
- `src/gallery_analysis/exemplars.py` — exemplar handling
- `src/gallery_analysis/compare_variants.py` — variant comparison
- `src/gallery_analysis/run_overnight_pipeline.py` — batch runner
- `src/gallery_analysis/visualization/` — all visualization code
- `src/gallery_analysis/gallery_rules.py` — ONLY bug fixes or test helpers, NOT rule semantics
- New files under `src/gallery_analysis/` for diagnostics and calibration experiments
- `src/tests/` — add regression tests for each fix
- `review-stage/experiments/` — calibration scripts and artifacts

### You MAY make CONSERVATIVE changes to:
- `src/dreamcoder_core/grammar.py` — ONLY production probabilities / prior weights, NOT structure
- **FORBIDDEN in grammar/primitives code:** adding or removing primitives, changing type signatures, restructuring the grammar hierarchy, changing de Bruijn index handling

### You MAY NOT touch:
- `src/rules/catalogue.py` — DreamCoder wake-sleep rule catalogue (FROZEN)
- `src/rules/cards.py` — card representations
- `src/dreamcoder_core/primitives.py`, `lean_primitives.py` — primitive definitions
- `src/dreamcoder_core/type_system.py` — type system
- Anything under `archived/`, `data/` (except reading for reference)
- `src/gallery_analysis/mcmc_search.py` — MCMC code (separate branch)

### If you believe a forbidden change is necessary:
Stop. Append a block to `NOTES_FOR_HUMAN.md` (create if needed) explaining what you wanted to change, why, and what workaround you used instead.

## The loop itself

Invoke the installed ARIS skill `/auto-review-loop` with these overrides:

```
/auto-review-loop "Night 1 review of Bayesian enumeration-based inference pipeline for card-game rule induction. This codebase has NOT been adversarially reviewed before — expect to find issues. Review the full pipeline: (1) top-down enumeration with syntactic pruning, (2) LLM hypothesis injection and dedup, (3) equivalence class fingerprinting, (4) Monte Carlo extension size estimation, (5) size-principle Bayesian scoring with canonical/summed priors, (6) posterior predictive diagnosticity, (7) visualization pipeline, (8) 10-variant configuration management. Critical question: is post-hoc tier re-scoring equivalent to enumerating under the weighted grammar? Answer with proof or counterexample. Read all 8 subsystem descriptions in ARIS_LAUNCH_PROMPT_BAYESIAN_REVIEW.md for detailed scope. Focus on theoretical correctness and internal coherence, NOT benchmarking." — compact: true, human checkpoint: false, difficulty: hard
```

The skill will:
1. Send project context to GPT-5 via Codex MCP for review (**verify connection with `claude mcp list | grep codex`** before starting)
2. Receive structured weaknesses + suggestions + score
3. You implement fixes
4. Run nested engineering review (see below)
5. Loop up to MAX_ROUNDS = 4

**Stop condition:** 4 rounds, no early stop. Even if score reaches 9+ or 10 in an early round, continue — the reviewer should push for the deepest possible diagnosis. Treat `POSITIVE_THRESHOLD` as requiring BOTH score >= 6 AND at least 3 rounds completed.

**Triage gate:** If score < 5/10 after round 1, this may indicate fundamental design issues. Write a detailed assessment in `NOTES_FOR_HUMAN.md` and continue, but flag that human review may be needed before the remaining rounds are useful.

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
3. **Critical question answer** — Is post-hoc tier re-scoring equivalent to enumerating under the weighted grammar? Document the proof or counterexample.
4. **Subsystem status** — for each of the 8 subsystems: reviewed / issues found / fixed / deferred
5. **New issues surfaced** — anything the reviewer raised that wasn't anticipated
6. **Uncertainty flags / human attention** — anything in `NOTES_FOR_HUMAN.md`
7. **Final verdict on theoretical correctness** — is the Bayesian enumeration pipeline defensible for the paper? What scope limitations remain?
8. **Recommended next steps** — what should the prior-realignment conversation focus on?

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

## Existing test baseline

The branch has ~69 tests across the enumeration/bayesian pipeline:
- `test_bayesian_scorer.py` (8 tests) — posterior computation, size principle
- `test_deep_enumeration.py` (16 tests) — program enumeration behavior
- `test_hypothesis_table.py` (8 tests) — fingerprinting, equivalence classes
- `test_injection.py` (18 tests) — LLM hypothesis validation
- `test_injected_hypotheses.py` (4 tests) — injection edge cases
- `test_hand_diagnosticity.py` (15 tests) — posterior predictive confidence
- `test_dsl_prior.py` (5 tests) — grammar log-prior computation
- `test_memoized_enumeration.py` (2 tests) — enumeration caching
- `test_fingerprint_refinement.py` (2 tests) — fingerprint distinctiveness
- `test_pipeline_integration.py` (1 test) — end-to-end integration
- `test_validation_fixes.py` — validation fixes

Run the full relevant test suite as your first action to establish a green baseline.

## Setup instructions (run these BEFORE pasting this prompt)

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling

# Tag current state for safety
git tag bayesian-pre-review feat/grammar-comparison

# Create a fresh worktree
git worktree add .worktrees/aris-bayesian-review -b aris/bayesian-review feat/grammar-comparison

# Launch Claude Code from the worktree
cd .worktrees/aris-bayesian-review
```

Then paste everything above the setup section into the fresh Claude Code session.

## Getting started — checklist

1. `pwd` — confirm you're in the worktree
2. `git branch --show-current` — should be `aris/bayesian-review`
3. `claude mcp list | grep -i codex` — should show connected. If not, fall back to Claude sub-agent reviewer and flag in `NOTES_FOR_HUMAN.md`.
4. Read this prompt file (`ARIS_LAUNCH_PROMPT_BAYESIAN_REVIEW.md`) in full — especially the 8 subsystem descriptions
5. Run the existing test suite to confirm green baseline:
   ```bash
   cd src && ~/miniforge3/bin/python -m pytest tests/test_bayesian_scorer.py tests/test_deep_enumeration.py tests/test_hypothesis_table.py tests/test_injection.py tests/test_injected_hypotheses.py tests/test_hand_diagnosticity.py tests/test_dsl_prior.py tests/test_memoized_enumeration.py tests/test_fingerprint_refinement.py tests/test_pipeline_integration.py tests/test_validation_fixes.py -v
   ```
6. Write `RESUME_INSTRUCTIONS.md` at worktree root
7. Launch the loop with the exact argument string from "The loop itself" section

You are unattended. Be thoughtful, be methodical, be honest in the morning report. This is the first review — expect to find real issues. Good luck.
