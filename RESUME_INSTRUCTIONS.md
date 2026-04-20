# Resume Instructions — ARIS Bayesian Review (Night 2)

**Night 2 start:** 2026-04-20T06:55Z
**Hard deadline:** 10 hours from start → 2026-04-20T16:55Z
**Branch:** `aris/bayesian-review` (worktree at `.worktrees/aris-bayesian-review`)

## Night 1 status (completed)
- 4 rounds complete, score 2 → 4 → 5 → 6 ("almost").
- All Night 1 fixes committed in `4a8bd60`. Review artifacts in `0f1a17d`.
- MCMC files cherry-picked in `d525079`.
- See `review-stage/AUTO_REVIEW.md` for full Night 1 log.
- See `REVIEWER_MEMORY.md` for accumulated reviewer state.

## Night 2 workstreams
1. **Full-scale sensitivity audit** (background compute)
2. **Adversarial hand generation** (BALD entropy proxy)
3. **Enumeration-vs-MCMC posterior comparison**
4. **Visualization end-to-end stress test**

## If this session dies, resume with:
1. `cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling/.worktrees/aris-bayesian-review`
2. Read `review-stage/REVIEW_STATE.json` for last loop state.
3. Read `review-stage/AUTO_REVIEW.md` for chronological log (will be appended to during Night 2).
4. Check active experiments:
   - `ls review-stage/experiments/night2/*.pid`
   - `ps -p $(cat review-stage/experiments/night2/<exp>.pid)` per pid file
   - `tail -50 review-stage/experiments/night2/<exp>.log` for progress
5. Sensitivity audit incremental dump: `review-stage/experiments/night2/full_sensitivity_partial.json`
6. To continue: re-launch the auto-review skill, pointing to the round you were in.

## Codex MCP
- Connected: `codex: codex mcp-server - ✓ Connected`
- Thread: `019da777-8714-71e3-9784-5bea4a10bed1` (Night 1 thread; reuse if appropriate or start fresh for Night 2)
- Fallback: spawn Claude general-purpose sub-agent as harsh-NeurIPS reviewer if any Codex call fails.

## Loop parameters
- MAX_ROUNDS = 4, no early stop.
- POSITIVE_THRESHOLD: score >= 6 AND >= 3 rounds.
- Nested engineering reviews: kieran-python-reviewer, performance-oracle, code-simplicity-reviewer (cap 2 passes).

## Scope (STRICT)
- FREE: all files Night 1 modified, plus new files under `src/gallery_analysis/` for Night 2 work.
- import-only adjustments OK on `analyze_mcmc.py` and `mcmc_hypothesis_collector.py`.
- FORBIDDEN: `mcmc_search.py` (cherry-picked verbatim, no logic changes).
- Same Night 1 forbidden list otherwise.

## Test baseline
- 83/84 pass. Pre-existing failure: `tests/test_injection.py::test_merge_updates_summed_prior`.

## Memory monitoring
- 24GB shared between background audit and review compute.
- Pause audit if `top -l 1 -s 0 | grep PhysMem` shows <2GB free.
