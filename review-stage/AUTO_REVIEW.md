# Auto-Review Loop — MCMC Program Search

**Session started:** 2026-04-16T05:03:07Z
**Branch:** `aris/mcmc-review-20260411`
**Loop topic:** Improve MCMC program search quality for Bayesian rule induction over card-game rules
**Difficulty:** hard (4 rounds max, stop early on score ≥ 6/10 or verdict accept/sufficient/ready-for-submission)
**Codex MCP:** unavailable (confirmed via `claude mcp list` — only Gmail/Drive/Calendar/context7/gmail-personal/slack present). Using **Claude general-purpose sub-agent** as harsh NeurIPS-area-chair reviewer per ARIS_LAUNCH_PROMPT.md fallback protocol.

---

## Round 0: Bootstrap

- Created `review-stage/` with `experiments/` subdir.
- Confirmed worktree clean, branch correct, no prior REVIEW_STATE.json (fresh run, not a resume).
- **Baseline: 40/40 tests pass in 1173s (~19.5 min).**
  - Command: `cd src && ~/miniforge3/bin/python -m pytest tests/test_mcmc_search.py tests/test_mcmc_hypothesis_collector.py -v`
  - No pre-existing regressions; this is the reference for any future changes.


## Round 1: Review (2026-04-16T05:15Z)

**Reviewer:** Claude general-purpose sub-agent (Codex MCP unavailable — fallback per ARIS_LAUNCH_PROMPT.md).
**Agent IDs:** first attempt `ab9a0f705ef731f77` (ConnectionRefused at ~395s); retry `aeb9b1200669b676e` (completed, 135s, 6 tool_uses).

**Verdict:** Major revision.
**Score:** 4/10.

**Critical (6):**
- **C1.** Proposal density ≠ sampler density. `_sample` uses `candidates_for_type(normalize=False)` + custom softmax over productions ∪ variables, but `program_log_likelihood` uses normalized type-indexed dist. MH ratio (mcmc_search.py:864-874) inconsistent; detailed balance broken even at β=1.
- **C2.** `run_parallel_chains` (mcmc_search.py:1443-1452) silently drops `beta_start`/`beta_end` — every gallery run has actually run at β=1.0 defaults despite `--beta-start 0.1`. Fix: `dataclasses.replace(config, seed=chain_seed)`.
- **C3.** Layer-2 tautology rejection (mcmc_search.py:1247-1252) inside MH loop breaks detailed balance; biases toward over-specific hypotheses. Fix: apply post-hoc on visit table.
- **C4.** Monte-Carlo `ext_size` plugged into likelihood as exact (mcmc_search.py:1053-1080). `n_hits==0 → ext_size=1.0` pathologically rewards programs accepting nothing. Fix: Laplace smoothing + one-probe floor.
- **C5.** `sample_program` retry loop with seed-shift (mcmc_search.py:130-156) breaks determinism in practice — number of RNG calls varies with subtree content.
- **C6.** First-passage merge (mcmc_search.py:1477-1489) uses `offset_step = step + i * config.n_steps` — treats independent chains as concatenated timeline. Biases low-index chains to appear "faster." Drop offset, use true `min(step)`.

**High (6):**
- H1. β<1 stationary ≠ posterior; visit counts mix stationary dists.
- H2. 10K probes insufficient at P(52,6)=14.66B for rare-extension rules (e.g. strict_increasing ~0.0002).
- H3. Few exemplars + size principle fragile.
- H4. Shared probes correlate Monte-Carlo noise across chains; Python's non-deterministic string `hash` in `seed_offset`.
- H5. `run_parallel_chains` runs chains sequentially despite name. Use `ProcessPoolExecutor`.
- H6. `collect_subtree_sites` silent-except truncates n_sites, biases MH ratio via `log_pick_fwd`.

**Medium (8):** top-level-only vacuous lambda detection (M1); fallback ill-typed (M2); ext_fractions "latest" merge (M3); brittle 4-branch visit_counts increments (M4); test_annealing doesn't verify effect (M5); TOTAL_HANDS cross-module duplication (M6); visit_counts test checks arithmetic not states (M7); top_k=250 too small for gallery (M8).

**Priority fix order (author's):** C1 → C2 → C3 → C5/C6 → C4/H2 → H5.
**Our execution order (impact/effort):** C2 → C6 → C3 → C4 → C1 → H5 (C2/C6 are one-liners unblocking all downstream experiments).

**What's defensible:** AST surgery (subtree collection, replacement, de Bruijn env tracking); trajectory data structures (`visit_counts`, `first_passage`, `consecutive_dwelling_times`).

Full review text cached at `/tmp/round1_review.txt`.

