# Night 2 Morning Report — ARIS MCMC Review

**Date:** 2026-04-16
**Branch:** `aris/mcmc-review-20260411`
**Worktree:** `.worktrees/aris-mcmc-review`
**Reviewer:** GPT-5.4 via Codex MCP (`xhigh` reasoning effort, thread `019d9579-b4c2-7b23-97c2-8a2316c6d104`)
**Difficulty:** hard (reviewer memory + debate protocol + direct code access)
**Scope:** Internal coherence / theoretical correctness of the MCMC program-search implementation. Explicitly NOT rule-recovery benchmarking.

---

## TL;DR

**Final score: 8.0/10 — Almost READY for submission.**

Four full rounds of adversarial external review completed. Score trajectory: **4.5 → 5.5 → 7.0 → 8.0**. The MH kernel now has no remaining fatal detailed-balance / proposal-density defect in the current gallery regime. The theoretical-correctness story is strong enough to support a NeurIPS/ICML MCMC-sampler description, **provided the paper's claims are scoped to "post-burn-in stationary sampling under the current gallery grammar"** and the depth-2 calibration limit is acknowledged honestly.

**Three open items block a "Yes" verdict:**
1. Depth-3 calibration (hardware-bounded — OOMs on 24 GB machine).
2. Runtime guardrails for silent approximation caps (`_DEPTH_CAP_EXACT_ENUM_CAP=16`, `_MARGINALIZATION_FREE_VAR_CAP=3`).
3. `run_parallel_chains` is still sequential (H5 sustained, low priority).

---

## Score trajectory

| Round | Score | Verdict | Key gain |
|-------|-------|---------|----------|
| 1 | 4.5/10 | not ready | Reviewer ran code, confirmed 4 concrete counterexamples |
| 2 | 5.5/10 | not ready | C3/H-methodology/H-autocorr/H-seeds OVERRULED; C2-cap sustained |
| 3 | 7.0/10 | almost | C2-cap OVERRULED (exact depth-cap scorer); max_nodes init OVERRULED |
| 4 | 8.0/10 | almost | Full-kernel ΣQ=1 verified; H-n_sites-invariance OVERRULED |

---

## Commits (chronological)

| Commit | Round | Summary |
|--------|-------|---------|
| `3ac343c` | R1 | Retry-conditioned proposal alignment, exact marginalization in scorer, depth-cap lookahead mirror, vacuous-lambda as −∞ target, calibration v2, regression tests |
| `3bdf156` | R1 | Calibration v2 artifacts (5/5 seeds pass depth=2, mean TV=0.0955, Geyer IPS ESS 909-1607) |
| `072788b` | R1 | Posterior calibration experiment for C1 (passes TV=0.0014) |
| `23134f4` | R2 | Exact depth-cap scorer (2^n-subset enumeration), max_nodes init consistency, site-drop walker fix, proposal-normalization tests |
| `a96edcf` | R2 | Archived night 1 artifacts + rewrote launch prompt |
| `a05a59c` | R3 | Full-kernel ΣQ=1 test on tiny hand-built state space (R3-Fix1), init-resampling caveat inline doc (R3-Fix2) |

---

## What got fixed

### Round 1 → Round 2 (score 4.5 → 5.5)
Six concrete fixes landed:
1. **Proposal retry alignment** — `propose_regeneration` uses `allow_retries=False`; init path uses bounded retry with hard `RuntimeError` on exhaustion. Proposal distribution == scorer distribution.
2. **Exact marginalization in scorer** — Replaced tier-2/tier-3 heuristics with enumeration over `_CONCRETE_TYPES^k` + log-sum-exp, capped at `_MARGINALIZATION_FREE_VAR_CAP=3`. Empirically verified on toy `{choose, is_zero}` grammar: scorer P=0.125 matches empirical 0.1355 within 5% (was P=1.0 under tier-2).
3. **Depth-cap lookahead mirror** — Scorer shifts production log-probs by `_all_args_terminable` survival probability.
4. **Vacuous-lambda as −∞** — Target-coded inside MH body instead of pre-MH hard rejection.
5. **Independent-prior calibration v2** — Perturbed-grammar PCFG, polymorphic `{not,and,or,eq,if}` at `INT→BOOL`, `_CONCRETE_TYPES=[BOOL,INT]`, 5 seeds × 50k steps, Geyer IPS ESS. 5/5 seeds pass at depth=2, mean TV=0.0955.
6. **`collect_subtree_sites` walker fix** — Root-threaded `TypeContext`, identity-keyed node-type map. Post-fix: 0 silent drops across 50 samples (was 20/50).

### Round 2 → Round 3 (score 5.5 → 7.0)
Two major fixes:
1. **Exact depth-cap scorer** — `_log_expected_softmax_over_random_filter` does 2^n-subset enumeration of random filter sets, capped at `_DEPTH_CAP_EXACT_ENUM_CAP=16` competing lookahead candidates (gallery max observed: 16, never hit fallback). Reviewer stress-tested 3-competitor case: empirical 0.58085 vs scored 0.58333.
2. **`max_nodes` on init path** — Consistent with MH-body enforcement at `mcmc_search.py:1974`.

### Round 3 → Round 4 (score 7.0 → 8.0)
Two surgical fixes:
1. **R3-Fix1: Full-kernel ΣQ=1 test** (`test_propose_regeneration_full_kernel_normalizes_on_tiny_grammar`) — Tiny-bool grammar `{t:BOOL, f:BOOL→BOOL}` with `log_variable=-inf`, starting program `s = f(f(t))` giving exactly 2 sites, `regen_depth=3` at both, closed-form support `{t, f(t), f(f(t)), f(f(f(t)))}` with probs `{1/2, 1/4, 1/8, 1/8}`. Three assertions (per-site sum, full-kernel sum, pointwise lock-in) all pass within 1e-9. Reviewer verified it matches the exact kernel factorization at `mcmc_search.py:1667`.
2. **R3-Fix2: Init-resampling caveat** — Documentation block at `mcmc_search.py:1981` stating init retries filter vacuous/tautological/oversized draws from an *ad hoc* prior-with-rejection, preserves MH stationarity but biases early-trajectory / first-passage / cognitive-timing analyses. Paper scope excludes early-trajectory claims.

---

## Tests

**Baseline (start of Night 2):** 47/47 green in 28:52.
**After Round 1 fixes:** 44/44 in 1:01:27 (marginalization enumeration ~125× slower worst case; correctness verified).
**After Round 2 fixes:** 46/46 in 2:49:27 (2 new tests + exponential 2^n depth-cap enum).
**After Round 3 fixes:** 47/47 in 2:49:34 (1 new test: `test_propose_regeneration_full_kernel_normalizes_on_tiny_grammar`).
**Focused R3 run:** 12/12 in 16s.

---

## Final rulings from reviewer

### OVERRULED (no longer a concern)
- C3-tier2 (R1), H-methodology (R1), H-autocorr (R1), H-seeds (R1), retry-conditioned proposal law (R1), vacuous-lambda hard reject (R1), silent site-drop path (R1).
- C2-cap depth-cap (R2), max_nodes init (R2), hidden-free-type-draws for gallery (R2).
- R3 full-kernel normalization priority (R3), H-n_sites-invariance (R3), hidden-free-type caps for current gallery (R3).

### PARTIALLY SUSTAINED (doc/scope-gated, not kernel defects)
- **Init-resampling** — documented, but incompatible with any early-trajectory claim. **Paper must honor this.**
- **C1-gallery / calibration-support depth=2** — meaningful, but not depth-3. **Hardware-bounded.**
- **C5 init retries** — init-policy caveat, not kernel defect.
- **Explicit-cap weaknesses** — `_DEPTH_CAP_EXACT_ENUM_CAP=16`, `_MARGINALIZATION_FREE_VAR_CAP=3`. If hit, exactness silently degrades.

### SUSTAINED (open, non-blocking for Night 2 soundness)
- **H5 sequential chains** — `run_parallel_chains` (mcmc_search.py:2234) still sequential. Low priority for soundness target.

---

## Remaining open items (prioritized)

### HIGH (if reviewer is to say "Yes" next time)
1. **Depth-3 calibration run.** Needs machine with >24 GB RAM. Current 5/5 depth-2 result stands; depth-3 is the reviewer's last remaining substantive request.
2. **Runtime guardrails for exactness caps.** Add gallery-level assertion or zero-counter for `_DEPTH_CAP_EXACT_ENUM_CAP=16` and `_MARGINALIZATION_FREE_VAR_CAP=3`. Reviewer's direct Round 4 recommendation: "Do not leave these as comments only."

### MEDIUM (minor coverage gap)
3. **Env-bound lambda variant of full-kernel test.** Current test is empty-env BOOL only. A lambda/bound-variable variant would strengthen regression coverage (reviewer: "not a blocker").

### LOW (scope / process)
4. **Manuscript discipline.** Init caveat now exists in code comments; the paper text must honor it — no first-passage or cognitive-timing claims that treat early trajectory as samples from π.
5. **H5 parallel chains.** Convert `run_parallel_chains` from sequential to true parallel execution. Reviewer flagged as low priority.

---

## Recommendations

1. **If going to submission soon** — scope paper claims to "this MH implementation targets the post-burn-in posterior under the current gallery grammar," be explicit about depth-2 calibration limit, avoid any early-trajectory claims. Night 2 theoretical-correctness section is sufficient at this scope.

2. **Before next review cycle (Night 3 or later)** —
   - Provision a >24 GB machine for depth-3 calibration.
   - Add the runtime guardrails (small PR; tests check counters stay at zero on gallery run).
   - Add the env-bound lambda variant test (small PR).

3. **Not worth doing now** — H5 parallelization. The reviewer explicitly downgraded it; it affects throughput, not correctness.

---

## Reviewer final verdict (verbatim, Round 4)

> "If the paper says 'this MH implementation targets the post-burn-in posterior under the current gallery grammar' and is honest about the depth-2 calibration limit, the theoretical-correctness section is now in good shape. If it claims full-gallery calibration, or uses early discovery timing as evidence, it is still not ready."

---

## Artifacts

- `review-stage/round_1_review.txt` — Round 1 verbatim response (4.5/10).
- `review-stage/round_2_review.txt` — Round 2 verbatim response (5.5/10).
- `review-stage/round_3_review.txt` — Round 3 verbatim response (7.0/10).
- `review-stage/round_4_review.txt` — Round 4 verbatim response (8.0/10).
- `review-stage/REVIEWER_MEMORY.md` — Full reviewer memory across all 4 rounds.
- `review-stage/AUTO_REVIEW.md` — Cumulative auto-review log.
- `review-stage/REVIEW_STATE.json` — Final state (status=completed, final_score=8.0).
- `review-stage/experiments/round_1/posterior_calibration_v2.py` — Calibration v2 script.
- `night1-archive/` — Night 1 artifacts (MORNING_REPORT, round 1-3 reviews, commits).

## Forbidden-scope log

None. All fixes kept strictly within the MCMC sampler + its tests + its calibration script. No primitive, type-system, or grammar-structure changes.
