# Night 3 Morning Report — ARIS MCMC Review (FINAL)

**Date:** 2026-04-19
**Branch:** `aris/mcmc-review-night3`
**Worktree:** `.worktrees/aris-mcmc-night3`
**Final commit:** `26f55f4`
**Final score:** **9.0/10 — Ready**
**Thread ID:** `019da4d5-c81e-77b2-9c4d-e64b2b463c93`

## TL;DR

Night 3 closed the three theoretical-correctness gaps that Night 2 ended at 8.0/10 Almost with. Score progressed 5.0 → 7.5 → 8.5 → 9.0 across four rounds and terminated with reviewer verdict **Ready** and the statement:

> "If the paper says 'this MH implementation targets the post-burn-in posterior under the current gallery grammar' and is honest about the depth-2 calibration limit, the theoretical-correctness section is now in good shape."

No new correctness blockers remain under the claim scope. Only non-blocking manuscript/engineering caveats are open.

## Score trajectory

| Round | Commit reviewed | Score | Verdict | Headline |
|-------|-----------------|-------|---------|----------|
| 1 | `0bf49b3` | 5.0 | No | Reviewer probe exposed W1: `collect_subtree_sites` site-metadata corruption (id-keyed cache collided on Primitive singletons). ~5.7% bad-site rate on gallery programs. |
| 2 | `7f14a2f` | 7.5 | Almost | W1 fixed (path-keyed cache + single-pass `_annotate`). Three new findings: F-R2-1 root-type in retry path, F-R2-2 overclaiming typability test, W2 counters harness-wired but not experiment-wired. |
| 3 | `9d97392` | 8.5 | Almost | R2 fixes mostly OVERRULED (root-type agreement clean, binary grammar closure clean). Three closure gaps: F1 no-terminal lookahead branch uncovered; F2 experiment-side W2 assertion missing; F3 prose/assert rename. |
| 4 | `26f55f4` | **9.0** | **Ready** | F1/F2/F3 all OVERRULED. No new regressions. |

## What closed each night

### Night 3 Round 1 → 2 (W1 closure)
- **Path-keyed `collect_subtree_sites` cache**: replaced `id(node)` key with path tuple, eliminating Primitive-singleton collisions.
- **Single-pass `_annotate` mirroring `infer_type` semantics**: eliminated re-instantiation-of-Primitive stale TVs that blocked `ctx.apply` from resolving polymorphic site types. 0/30 ill-typed proposals (previously 10/20).
- **3 per-site approximation-cap counters + accessors**: `_arg_marginalization_fallbacks`, `_survival_prob_fallbacks`, `_depth_cap_mean_field_fallbacks` + `get_`/`reset_approximation_fallback_counters()`.
- **Lambda/bound-variable full-kernel ΣQ=1 test** on reviewer's exact spec (BOOL→BOOL, `{t:BOOL, f:BOOL→BOOL}`, `log_variable=0`, starting state `λ f(f($0))`, closed-form 8-program support at `{1/3, 1/3, 1/9, 1/9, 1/27, 1/27, 1/54, 1/54}`).

### Night 3 Round 2 → 3 (R2 fixes)
- **R2-Fix1** — `sample_program(allow_retries=True)` retry path unifies resolved type with `request_type` at `mcmc_search.py:~148`. Regression pins seed 1001 max_depth=6 and 0-leak on seeds 900..1199.
- **R2-Fix2** — `MCMCResult.approximation_fallback_counters` field; `MCMCChain.run` resets + captures; `run_parallel_chains` sums across chains.
- **R2-Fix3** — weakened `test_propose_regeneration_preserves_typability_and_reversibility` to the actual `len(new_sites) > 0` invariant + typed-OK ≥ tried // 2 smoke bound. Docstring explicitly disclaims "every regeneration type-checks".
- **R2-Fix4** — tiny exact binary grammar `{c:BOOL, g:BOOL→BOOL→BOOL}` with structural `_binary_program_key` (avoids Primitive lambda-hex-address repr collisions). Parameterized depth-boundary tests at `{1:2, 2:5, 3:26}` closed-form + Monte Carlo N=20000 with 4σ tolerance.

### Night 3 Round 3 → 4 (closure of F1, F2, F3)
- **R3-Fix1 (F1)** — no-terminal-BOOL lookahead grammar `{p:INT→BOOL, q:INT→BOOL, zero:INT, one:INT}` (+ optional non-terminable `r:LIST_INT→BOOL`). Monkeypatch counter on `_score_depth_cap_lookahead_exact` verifies branch fires exactly `4 + len(forbidden_keys)` times with masses `1/4` for valid + `-inf` for forbidden. Sampler companion at N=20000.
- **R3-Fix2 (F2)** — wired `reset_approximation_fallback_counters` / `get_approximation_fallback_counters` into `review-stage/experiments/round_1/posterior_calibration_v2.py`. `run_mh_one_seed` resets at chain start, per-seed results carry counters, `main()` aggregates + asserts zero + writes `approximation_fallback_counters_sum` and `approximation_fallback_counters_all_zero` into the JSON artifact. Reviewer independently verified by monkeypatching `run_mh_one_seed` to return nonzero — assertion fired as intended.
- **R3-Fix3 (F3)** — renamed `reversible_given_typed` → `has_reverse_sites_given_typed`. Docstring now explicitly disclaims that `log_q_rev` may legitimately be `-inf` even when typability + non-empty reverse sites both hold. Reviewer reproduced this on (gallery seed=10, proposal seed=7929).

## Remaining non-blocking caveats

| # | Item | Nature |
|---|------|--------|
| 1 | Depth-2 calibration scope | Manuscript must stay honest: independent-prior TV calibration is at depth=2 only; no full-gallery depth-3 sign-off run. |
| 2 | Init-resampling language | `sample_program(allow_retries=True)` at init skips tautological starts. OK for stationary sampling after burn-in; NOT OK for first-passage / cognitive-timing claims. Already documented in code; manuscript must honor. |
| 3 | `run_parallel_chains` sequential | H5 Night-2 carry. Not a correctness issue; throughput only. Low priority. |

**None of these affect the forward/reverse-density soundness of the MH kernel under the scoped claim.**

## Verification (as of commit `26f55f4`)

```
python -m pytest tests/test_mcmc_search.py \
  -k "root_type_agreement or fallback_counters or no_terminal_bool or
      typability_and_reversibility or tiny_binary or full_kernel_normalizes" \
  -v -p no:cacheprovider

13 passed, 47 deselected in 24.73s
```

Reviewer independently verified via direct `python -c` probes against the repo (not just reading diffs).

## Files changed across Night 3

| File | Change |
|------|--------|
| `src/gallery_analysis/mcmc_search.py` | Path-keyed cache + single-pass `_annotate` (R1); 3 approximation-cap counters + accessors (R1); root-type unify in retry path (R2); MCMCResult counter field + run-level reset/capture + parallel sum (R2). |
| `src/tests/test_mcmc_search.py` | +~400 lines of regression tests (W1 site consistency, W4 bound-variable full kernel, fallback triggers, binary grammar at depth ∈ {1,2,3}, no-terminal-BOOL lookahead, root-type retry agreement). Docstring/variable rename (R3). |
| `review-stage/experiments/round_1/posterior_calibration_v2.py` | Fallback-counter reset + per-seed record + aggregate zero-assert + JSON fields (R3). |
| `review-stage/{REVIEWER_MEMORY.md, REVIEW_STATE.json, night3_round_{1,2,3,4}_review.txt, AUTO_REVIEW.md, findings.md}` | Full audit trail. |

## Claim scope accepted by reviewer

> "This MH implementation targets the post-burn-in posterior under the current gallery grammar, with depth-2 calibration."

Under this scope the theoretical-correctness section is ready for submission. The manuscript must reflect the three caveats above verbatim.

## What Night 3 explicitly did NOT close

- **Depth-3 full-gallery calibration.** Reviewer's B-option (streaming exact depth-3 enumerator on 24 GB) never attempted; A+D closure (tiny exact depth-3 grammar + boundary tests) landed in Night 2. Not a correctness blocker under scoped claim, but excludes full-gallery calibration claims.
- **First-passage / early-trajectory analyses.** Init resampling is off-policy for these. If the manuscript makes cognitive-timing claims, init policy must be fixed OR early trajectory must be excluded.
- **`run_parallel_chains` parallelism.** Still sequential. Throughput improvement deferred.

## Next steps (if any)

If the submission target claims expand beyond scope:
1. **For full-gallery depth-3 calibration**: implement streaming enumerator or find GPU/bigger-RAM host for 24-GB-OOM workaround.
2. **For first-passage claims**: either (a) fix init resampling to match posterior, or (b) discard burn-in + first-passage jointly in the experiment pipeline.
3. **For throughput**: parallelize `run_parallel_chains` (multiprocessing or threading on CPU-bound chain inner loop).

Otherwise: proceed to manuscript; reviewer bottom-line is accepted.

---

**Audit trail:**
- Full per-round reviewer responses: `review-stage/night3_round_{1,2,3,4}_review.txt`
- Cumulative review log: `review-stage/AUTO_REVIEW.md`
- Reviewer memory (GPT-5.4 persistent context): `review-stage/REVIEWER_MEMORY.md`
- Loop state snapshot: `review-stage/REVIEW_STATE.json`
- Findings one-liners (compact mode): `review-stage/findings.md`
