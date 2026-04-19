# Resume Instructions — ARIS Bayesian Review

**Start timestamp:** 2026-04-19T20:32:32Z
**Hard deadline:** 10 hours from start → 2026-04-20T06:32:32Z
**Branch:** `aris/bayesian-review` (worktree at `.worktrees/aris-bayesian-review`)
**Safety tag:** `bayesian-pre-review` marks pre-review state of `feat/grammar-comparison`.

## If this session dies, resume with:

1. `cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling/.worktrees/aris-bayesian-review`
2. Read `review-stage/REVIEW_STATE.json` to see where the loop left off.
3. Read `review-stage/AUTO_REVIEW.md` to see the in-progress log.
4. Check which round was active: `ls review-stage/experiments/`
5. Check running experiments: `ls review-stage/experiments/round_*/*.pid` and `ps -p $(cat <pid file>)`
6. If a round was mid-way: re-invoke `/auto-review-loop` with `--resume` or launch the next round manually.

## Baseline state

- 83/84 tests pass.
- **Pre-existing failure:** `tests/test_injection.py::test_merge_updates_summed_prior`
  - `-2.0 != -1.6867383124817772` — summed-prior merge appears to overwrite rather than log-sum-exp with existing class members.
  - This is likely a real bug in summed-prior mode and is in-scope for review. Flag in round 1.

## Codex MCP

- Connected: `codex: codex mcp-server - ✓ Connected` (verified at launch).
- Fallback: spawn Claude general-purpose sub-agent as harsh-NeurIPS reviewer if any Codex call fails.

## Loop parameters

- MAX_ROUNDS = 4, no early stop.
- POSITIVE_THRESHOLD: score ≥ 6 AND ≥ 3 rounds completed.
- Nested engineering reviews per round: kieran-python-reviewer, performance-oracle, code-simplicity-reviewer (cap 2 passes).

## Scope reminders (STRICT)

- FREE: `src/gallery_analysis/*.py` (except mcmc_*), `src/tests/`, `review-stage/`, new diagnostic files.
- CONSERVATIVE: `src/dreamcoder_core/grammar.py` production probs only.
- FORBIDDEN: `src/rules/catalogue.py`, `src/dreamcoder_core/primitives.py`, `lean_primitives.py`, `type_system.py`, anything under `archived/`, MCMC files.
- If forbidden change is needed: append to `NOTES_FOR_HUMAN.md` with justification + workaround.

## Critical question to answer by end

**Is post-hoc tier re-scoring mathematically equivalent to enumerating under the weighted grammar?**
Prove or give counterexample. Tier weights: CHEAP −3.0, STANDARD −4.0, AGGREGATE −5.5, ULTRA_SHALLOW −9.0. Engine enumerates under uniform grammar then re-scores.
