# Reviewer Memory

Persistent memory across rounds for the external GPT-5.4 reviewer (Codex MCP, `xhigh`).

---

## Round 1 — Score: 4.5/10, Verdict: not ready

**Thread ID:** `019d9579-b4c2-7b23-97c2-8a2316c6d104`

### Suspicions raised
- The current proposal density `q` is still wrong until the proposal generator, scorer, and latent type-resolution story are literally the same mathematical object. Multiple operational filters (retry loop in `sample_program`; pre-MH vacuous-lambda rejection) live outside the stated posterior / proposal density and silently truncate state space.

### Unresolved concerns carried forward
- `max_nodes` and other hard support truncations may also be off-book unless folded into the target explicitly.
- Independent-prior calibration may still fail even after tier-2 removal unless `sample_program` retry mismatch is fixed first.
- `collect_subtree_sites` swallows inference failures silently (lines ~953–962, 1013–1028); `n_sites` may drift undetected.

### Patterns the reviewer flagged
- Every major soundness failure so far comes from operational filters or retries that live outside the stated posterior / proposal density.

### Confirmed concrete counterexamples (reviewer ran code)
- Tier-2 scorer inversion: toy grammar `{choose: ('a->bool)->out, is_zero: int->bool}`, `_CONCRETE_TYPES=[BOOL,INT]`. Scorer returns `P=1.0` for `choose ((λ is_zero $0))`; raw `_sample` emits it 992/2000 ≈ 0.496. Tier-3 also fails — it picks one sentinel assignment, not a marginal.
- `_all_args_terminable` branch entry rate: 23.5% in gallery `HAND→BOOL` at `max_depth=5` (1955/8318 calls). Claim that it is "rare" is false.
- `sample_program` retry rate: 2.8% on 500 gallery root samples (14/500 retried). Toy distribution shifted: raw `_sample` `{992,471,537 ERR}` → `sample_program` `{1338,662}`.

### Night 2 Round 1 action list sent to Claude
1. Align proposal generator with scored density (remove retry or score exact retry-conditioned law).
2. Exact marginalization over env-unresolved type draws (enumerate `_CONCRETE_TYPES^k`, log-sum-exp).
3. Mirror depth-cap `_all_args_terminable` in the scorer.
4. Remove pre-MH vacuous-lambda rejection (encode as −∞ in target or defer to reporting).
5. Re-run calibration with independent prior, 5 seeds, IAT/Geyer ESS, polymorphic grammar `{not,and,or,eq,if}` at `INT→BOOL`.
6. Add proposal-normalization / site-collection regression tests (`ΣQ(s'|s)≈1`, zero silent site-drop failures).

### Reviewer memory update for future rounds (verbatim from GPT-5.4)
> Suspicion: the current q is still wrong until the proposal generator, scorer, and latent type-resolution story are literally the same object.
> Unresolved: `max_nodes` and other hard support truncations may also be off-book unless folded into the target explicitly.
> Patterns: every major soundness failure so far comes from operational filters or retries that live outside the stated posterior.
> Unresolved: an independent-prior calibration may fail even after tier-2 removal unless the `sample_program` retry mismatch is fixed first.

---

## Round 2 — Score: 5.5/10, Verdict: not ready

**Reviewer read commit `3bdf156` directly (not just prose summary).**

### Rulings on Round 1 weaknesses
- `C3-tier2`: **OVERRULED** — scorer backsolve bug gone; hidden arg-level type draws genuinely marginalized (mcmc_search.py:633).
- `C1-gallery`: **PARTIALLY SUSTAINED** — polymorphism + independent prior now exercised, but depth=2 still, not full-support exact.
- `C2-cap`: **SUSTAINED** — depth-cap scorer is still approximate ("mean-field" at mcmc_search.py:734). Counterexample reproduced: raw `_sample` gives `P(p1 0)=0.736`, scorer returns `0.667`.
- `C5`: **PARTIALLY SUSTAINED** — proposal kernel clean, retries remain on init/one-shot path.
- `H-methodology`: **OVERRULED** — independent-prior calibration is a real improvement (posterior_calibration_v2.py:135).
- `H-autocorr`: **OVERRULED** — Geyer IPS ESS used (posterior_calibration_v2.py:344).
- `H-seeds`: **OVERRULED** — five seeds present.
- `H-n_sites-invariance`: **PARTIALLY SUSTAINED** — silent-drop fixed (mcmc_search.py:1132), but no direct proposal-normalization test.
- `H5`: **SUSTAINED** — `run_parallel_chains` still sequential (mcmc_search.py:2039).
- `retry-conditioned proposal law`: **OVERRULED** — `allow_retries=False` at mcmc_search.py:1397.
- `vacuous-lambda hard reject`: **OVERRULED** — target-coded at mcmc_search.py:1826.
- `hidden free-type draws`: **PARTIALLY SUSTAINED** — fixed for ordinary arg scoring; not fixed for depth-cap filtered-candidate normalization.
- `silent site-drop path`: **OVERRULED**.

### NEW weaknesses introduced or exposed (Round 2)
- **Depth-cap correction still approximate** (mcmc_search.py:921). Folding survival probs into per-production weights is NOT the exact marginal of the sampler's random filtered set. **Main remaining soundness defect.**
- **`max_nodes` off-book on init** (mcmc_search.py:1731). MH body applies −∞ (:1824) but init path does not. 57/100 initialized gallery states exceeded `max_nodes=25`.
- **Init resampling = arbitrary prior over starts** (mcmc_search.py:1731). Vacuous/tautological starts skipped. Doesn't break stationarity but biases early visit counts / first-passage analyses.
- **Calibration support incomplete** (posterior_calibration_v2.py:243). Enumerator drops nonzero-arity productions at cap; result file shows nonzero outside-support mass on every seed.
- **Proposal-density regression test is weak** (test_mcmc_search.py:973). Checks only top-2 proposals with factor-3 tolerance. Would not catch the remaining depth-cap mismatch.

### Reviewer memory update for future rounds (verbatim from GPT-5.4, Round 2)
> Addressed: proposal retry-conditioning, scorer tier-2 inversion, silent site-drop path, ESS/seeds, Night 1 methodology flaw.
> Still open: exact depth-cap scorer/sampler agreement, init-path `max_nodes` inconsistency, arbitrary init resampling, lack of explicit `ΣQ(s'|s)=1` verification.
> Pattern update: the remaining failures are now concentrated in "rare branch" logic where the fix was approximate rather than exact.
> Next-round priority:
>   1. Make the depth-cap scorer exact or prove the branch is excluded in the final experimental regime.
>   2. Apply `max_nodes` consistently at init or state clearly that init is arbitrary and excluded from all timing/visit analyses.
>   3. Add a true proposal-normalization test on a tiny hand-built state space.
>   4. If possible, rerun calibration at depth 3 on a machine that can enumerate the support exactly.

---

## Round 3 — Score: 7.0/10, Verdict: Almost

**Reviewer read commit `23134f4` directly; stress-tested R2-Fix1 on extra 3-competitor case (empirical 0.58085 vs scored 0.58333).**

### Rulings on Round 2 weaknesses
- `C2-cap depth-cap`: **OVERRULED** — exact subset enumeration at mcmc_search.py:796 integrated at :1047. Gallery lookahead candidate count max = 16 (exactly cap); fallback not triggered.
- `max_nodes init`: **OVERRULED** — fixed at mcmc_search.py:1974.
- `init resampling`: **PARTIALLY SUSTAINED** — still skips tautological starts (mcmc_search.py:1992). Fine for stationary sampling after burn-in, NOT fine for first-passage/cognitive-timing claims.
- `calibration support`: **PARTIALLY SUSTAINED** — kernel evidence much better, but calibration still depth-2, not exact full-support depth-3.
- `weak proposal test`: **PARTIALLY SUSTAINED** — new tests (test_mcmc_search.py:1113, :1156) are good scorer regressions but NOT full proposal-kernel normalization test (no site-pick term coverage).
- `C1-gallery depth=2`: **PARTIALLY SUSTAINED** — better, not full depth-3 sign-off run.
- `C5 init retries`: **PARTIALLY SUSTAINED** — no longer a proposal-law problem, still an init-policy choice.
- `H-n_sites-invariance`: **PARTIALLY SUSTAINED** — site-drop bug fixed, no direct `propose_regeneration` normalization test yet.
- `hidden free-type draws`: **OVERRULED** for gallery (max free vars in gallery arg = 2, below `_MARGINALIZATION_FREE_VAR_CAP=3`).
- `H5 sequential chains`: **SUSTAINED** — still sequential (mcmc_search.py:2219), low priority.

### NEW weaknesses introduced by R2 fixes
- Code is exact only within explicit caps (`_DEPTH_CAP_EXACT_ENUM_CAP=16`, `_MARGINALIZATION_FREE_VAR_CAP=3`). Not hit in current gallery but still approximation boundaries.
- Exact depth-cap scorer is exponential in competing lookahead productions — scalability risk, not current correctness blocker.

### Reviewer memory update for future rounds (verbatim from GPT-5.4, Round 3)
> Addressed: exact depth-cap scorer/sampler agreement, init-path `max_nodes` inconsistency, scorer normalization regression strength.
> Still open: init resampling biases early trajectory / first-passage analyses; calibration remains depth-2; no full `∑_{s'}Q(s'|s)=1` test for subtree regeneration.
> Pattern update: the remaining issues are no longer fatal MH-ratio defects; they are now boundary-condition and methodology issues.
> Next-round priority:
>   1. Add a tiny full-kernel normalization test for `propose_regeneration`, not just the scorer.
>   2. Decide whether the paper will make first-passage / cognitive timing claims. If yes, fix init resampling or exclude early trajectory explicitly.
>   3. If hardware permits, do the depth-3 calibration run; if not, document the limitation clearly.

**Bottom line:** "If the paper's central claim is only that the chain targets a posterior after burn-in, this is close. If it also leans on early discovery timing or strong 'full gallery' calibration language, it is still not ready."

---

## Round 4 — Score: 8.0/10, Verdict: Almost (FINAL)

**Reviewer read commit `a05a59c` directly; verified full-kernel test and subtle duplicate-destination case (multiple `(site, subtree)` pairs mapping to same `new_program` — fine because the implemented kernel is a mixture over site-conditioned proposals).**

### Rulings on Round 3 priorities + carry-forward weaknesses
- `R3 full-kernel normalization priority`: **OVERRULED** — closed by `test_mcmc_search.py:1250`.
- `Init-resampling ruling after doc fix`: **PARTIALLY SUSTAINED** — correctly documented, but still incompatible with any early-trajectory claim.
- `C1-gallery depth=2`: **PARTIALLY SUSTAINED** — independent-prior calibration is meaningful, but still only depth-2.
- `Calibration-support depth=2`: **PARTIALLY SUSTAINED** — same reason.
- `C5 init retries`: **PARTIALLY SUSTAINED** — no longer a kernel defect, still an init-policy caveat.
- `H-n_sites-invariance`: **OVERRULED** — site-drop bug fixed, new kernel test closes prior normalization gap.
- `Hidden-free-type caps`: **OVERRULED for the current gallery regime** — no evidence the current gallery is relying on the fallback path.
- `Explicit-cap weaknesses`: **PARTIALLY SUSTAINED** — if caps are ever hit, exactness silently degrades.
- `H5 sequential chains`: **SUSTAINED**, low priority — `run_parallel_chains` (mcmc_search.py:2234) still sequential.

### NEW weaknesses introduced by R3 fixes
- No new algebraic correctness defect.
- Minor coverage gap: new kernel test is empty-env BOOL only. A lambda / bound-variable variant would strengthen regression coverage (not a blocker).
- Minor process gap: init caveat now exists in code comments; the manuscript still has to honor it.

### Scalability caps recommendation (Round 4 new)
> Do not leave `_DEPTH_CAP_EXACT_ENUM_CAP=16` (mcmc_search.py:793) and `_MARGINALIZATION_FREE_VAR_CAP=3` (mcmc_search.py:630) as comments only. Add a gallery-level assertion or runtime counter that must stay at zero for the reported experiments.

### Reviewer memory update for future rounds (verbatim from GPT-5.4, Round 4)
> Addressed: full-kernel site-conditioned proposal normalization; init caveat is now explicit and correct.
> Still open: depth-3 / fuller-gallery calibration remains missing; exactness caps need active guardrails; init policy still rules out early-trajectory claims.
> Pattern update: the fatal MH-ratio defects appear resolved. Remaining risk is now claim scope and silent future regression outside the current gallery regime.

**Final bottom line:** "If the paper says 'this MH implementation targets the post-burn-in posterior under the current gallery grammar' and is honest about the depth-2 calibration limit, the theoretical-correctness section is now in good shape. If it claims full-gallery calibration, or uses early discovery timing as evidence, it is still not ready."
