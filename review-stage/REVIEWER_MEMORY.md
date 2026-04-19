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

---

## Night 3 Round 1 — Score: 5.0/10, Verdict: No

**Thread ID:** `019da4d5-c81e-77b2-9c4d-e64b2b463c93`
**Reviewer read commit `0bf49b3` directly; ran direct probes on the repo.**

### New critical blocker (W1)
- `collect_subtree_sites` site-metadata corruption. Root-threaded type-annotation cache keyed on `id(node)`, which collides on Primitive singletons shared across AST positions. Direct probe: 18/314 collected sites (~5.7%) had `(site.type, site.env)` inconsistent with the subtree on 10 sampled gallery programs; 11/20 regeneration probes produced ill-typed proposals; 8/20 lost the chosen path in `new_sites`. Correctness blocker because the forward kernel uses site metadata for density, and reverse kernel reuses old-site metadata.

### Reviewer's Round 1 rulings
- **W1 (new blocker):** path-keyed (not id-keyed) annotation cache. Regression tests: every collected site on sampled gallery programs must type-check under `(site.env, site.type)`; `propose_regeneration` must preserve whole-program typability and retain the chosen path in `new_sites`.
- **W2 (still open, Night 2 carry):** three silent approximation-cap fallbacks at mcmc_search.py:679, :769, :823. Add explicit per-site counters, reset per run, surface in result artifact, paper-run fails if any counter is nonzero.
- **W3 (still open, Night 2 carry):** depth-3 calibration still closed only at depth-2. After W1, do either a streaming exact depth-3 enumerator on the toy OR a tiny exact depth-3 grammar + targeted depth-boundary tests.
- **W4 (still open, Night 2 carry):** full-kernel test is empty-env BOOL only. Add one tiny lambda / bound-variable full-kernel test with exact closed-form support. Reviewer provided exact spec: `BOOL -> BOOL`, grammar `{t:BOOL, f:BOOL→BOOL}`, `log_variable=0`, starting state `λ f(f($0))`, closed-form support of 8 programs with probabilities `{1/3, 1/3, 1/9, 1/9, 1/27, 1/27, 1/54, 1/54}`.

### Reviewer's rankings
- **G1 strategy preference (depth-3):** `B > (A + D) > D > C`. `A + D` is fastest acceptable closure under 24 GB.
- **G2 counter format:** Separate counters for arg-marginalization, survival-prob, depth-cap mean-field. Merged counter too lossy. Hard CI failure if any counter nonzero at run end.
- **G3 lambda kernel spec:** provided verbatim (above).

### Reviewer memory update for Round 2 (verbatim)
> Addressed: Night-2 scorer/proposal normalization fixes still look intact; the tiny-bool full-kernel test still supports the claim that the local scorer math is much improved.
> Still open: depth-3 closure, cap guardrails, lambda full-kernel coverage, init-resampling manuscript scoping.
> New suspicions: `collect_subtree_sites` is using occurrence-unsafe type caching; site metadata can be wrong even when the failure counter is zero; reverse-site recollection can fail on proposed states.
> Pattern: the remaining serious bugs are now in operational bookkeeping around site collection and context threading, not in the local softmax/marginalization algebra.

### Claude's Round 1 fixes (for Round 2 review)
- **W1 — path-keyed cache.** Replaced `id(node)` key with path tuple. Direct probe post-fix: ≤0.5% residual (attributable to an out-of-scope pre-existing `TypeContext.instantiate` identity bug in dreamcoder_core/type_system.py, which the probe neutralises by pre-bumping `_next_var`).
- **W1 — follow-on bug found and fixed in the same walker.** The original `_annotate` called `node.infer_type(ctx, env)` at every recursion level, re-instantiating primitives and producing stale TVs in the path map. Root-threaded `ctx.apply` could then not resolve polymorphic site types (e.g., `list('t578)` stayed un-narrowed). Fix: rewrote `_annotate` to mirror `infer_type`'s per-node semantics exactly in a single pass, sharing one TypeContext. End-to-end probe: 0/30 ill-typed proposals (previously 10/20).
- **W2 — three separate counters** `_arg_marginalization_fallbacks`, `_survival_prob_fallbacks`, `_depth_cap_mean_field_fallbacks`. Accessors `get_approximation_fallback_counters` / `reset_approximation_fallback_counters`. All three call sites wired.
- **W4 — lambda kernel test** on reviewer's exact spec. Passes: per-site sum = 1, pointwise probabilities match closed-form, full-kernel ΣQ = 1.
- **W1, W2 regression tests added.** Type-consistency probe on sampled gallery programs; propose-preserves-typability; counter triggers and stays-zero tests.
- **W3 deferred to Round 2** pending reviewer's verdict on the W1 fix.

---

## Night 3 Round 2 — Score: 7.5/10, Verdict: Almost

**Reviewer read commit `7f14a2f` directly via `python -c` probes.**

### Rulings on Round 1 blockers
- `W1` (site-metadata corruption): **OVERRULED** — 64-var-offset probe strategy shows 0/3256 bad sites over 50 sampled programs; 0 silent drops.
- `W2` (approximation-cap counters): **PARTIALLY SUSTAINED** — counters exist and stay zero on gallery probes, but NOT wired into any experiment artifact; reported runs can still ignore them.
- `W4` (bound-variable full kernel): **OVERRULED** — full-kernel `ΣQ=1` reproduced at `{1/3, 1/3, 1/9, ...}`.

### NEW findings (Round 2)
- **Medium — `sample_program(allow_retries=True)` root-type agreement**: at `mcmc_search.py:148`, retry path only checks that `infer_type` returns, not that the resolved type unifies with `request_type`. Counter-probe: `sample_program(build_gallery_grammar(), HAND→BOOL, max_depth=6, seed=1001)` returns a program that fails unification and gets `-inf` prior. Rate: 1/100 seeds at both `max_depth=5` and `max_depth=6`. Fix: unify resolved type with `request_type` before returning; add regression on seed 1001.
- **Low/Medium — `test_propose_regeneration_preserves_typability_and_reversibility` overclaims**: asserts every regeneration should whole-program type-check, but `propose_regeneration` calls `sample_program(allow_retries=False)` by design. Not a W1 resurrection — just an over-tight test. Fix: weaken to the actual invariant (correct site metadata + reverse-support existence) or scope as seed-limited smoke test.

### Reviewer memory update for Round 3 (verbatim)
> Addressed: W1 site-metadata corruption is fixed; W4 bound-variable full-kernel coverage is fixed; W2 per-fallback counters now exist.
> Still open: W2 harness/result wiring; W3 depth-3 closure; init-resampling manuscript caveat.
> New suspicions: `sample_program` retry mode still admits occasional root-mistyped programs; the new proposal-typability regression is stronger than the implementation's actual contract.
> Pattern: the fatal MH-ratio/site-context bug appears resolved. Remaining risk is now experiment guardrails plus helper-path type-contract slippage, not the core forward/reverse kernel math.

### Claude's Round 2 fixes (for Round 3 review)
- **R2-Fix1 — root-type agreement** in `sample_program(allow_retries=True)` at `mcmc_search.py:~148`: added `ctx.unify(inferred, ctx.instantiate(request_type))` before returning; regression `test_sample_program_retry_path_enforces_root_type_agreement` pins seed 1001 + 0-leak on `seeds 900..1199` at `max_depth ∈ {5,6}`.
- **R2-Fix2 — W2 counters attached to MCMCResult**: `MCMCResult.approximation_fallback_counters: Dict[str, int]`; `MCMCChain.run` resets counters at start and captures via `get_approximation_fallback_counters()` on return; `run_parallel_chains` sums counters across chains.
- **R2-Fix3 — weakened W1 proposal-typability test**: docstring + assertion now pin the actual invariant (reversibility conditional on typability + typed-OK ≥ tried // 2 smoke bound).
- **R2-Fix4 — W3 A+D tiny exact binary grammar**: `_build_tiny_bool_binary_grammar()` → `(grammar, c_prim, g_prim)`. Parameterized `test_score_subtree_under_sampler_normalizes_on_tiny_binary_grammar[1-2, 2-5, 3-26]` closed-form scorer normalization. Monte-Carlo `test_score_matches_sampler_empirics_on_tiny_binary_grammar` with N=20000 and 4σ tolerance. Structural `_binary_program_key(prog)` avoids Primitive lambda-hex-address repr collisions.

---

## Night 3 Round 3 — Score: 8.5/10, Verdict: Almost

**Reviewer read commit `9d97392` directly via `python -c` probes.**

### Rulings on Round 2 findings
- **R2-Fix1 root-type agreement**: **OVERRULED** — unify added at `mcmc_search.py:148`; regression `test_sample_program_retry_path_enforces_root_type_agreement` locks in the invariant.
- **R2-Fix2 MCMCResult counters**: **OVERRULED at the harness level**; **PARTIALLY SUSTAINED at the experiment-artifact level** — counters now flow through `MCMCChain.run` and `run_parallel_chains`, but no review-stage experiment script *asserts* zero on its reported result. `review-stage/experiments/*` can still silently take an approximation branch without any signal.
- **R2-Fix3 weakened typability test**: **PARTIALLY SUSTAINED** — prose/assert still slightly overstate the invariant. Reviewer reproduced `log_q_rev=-inf` on `seed=10+7919` where typability and non-empty `new_sites` both hold. The `reversible_given_typed` variable name promises finite-density but only checks non-empty reverse sites.
- **R2-Fix4 tiny binary grammar closure**: **OVERRULED** — structural key + Monte-Carlo at N=20000 close depth-3 exact-scorer agreement at the toy scale.

### NEW findings (Round 3)
- **F1 — no-terminal lookahead branch uncovered**: `_score_depth_cap_lookahead_exact` at `mcmc_search.py:~1192` fires when `depth >= max_depth` AND no terminal AND no var. Current tests do not exercise this branch with a grammar that has zero terminal productions at the target type. Recommended grammar: `{p:INT→BOOL, q:INT→BOOL, zero:INT, one:INT}` (+ optional non-terminable `r:LIST_INT→BOOL`). Closed-form support over `BOOL` at depth=0 max_depth=0 is 4 programs uniform at 1/4.
- **F2 — experiment-side W2 assertion missing**: `review-stage/experiments/round_1/posterior_calibration_v2.py` does not reset or record `approximation_fallback_counters`. Counters could be nonzero in a future run and never surface in the JSON or log.
- **F3 — prose/assert mismatch in R2-Fix3**: rename `reversible_given_typed` → `has_reverse_sites_given_typed` and explicitly disclaim that `log_q_rev` may legitimately be `-inf`, OR strengthen the assertion to `math.isfinite(log_q_rev)`.

### Reviewer memory update for Round 4 (verbatim)
> Addressed: R2-Fix1 root-type agreement is clean; R2-Fix2 harness-level counter wiring is clean; R2-Fix4 binary grammar closure is clean.
> Still open: no-terminal lookahead branch uncovered (F1); experiment-side W2 assertion missing (F2); R2-Fix3 prose/assert mismatch (F3); init-resampling manuscript caveat.
> New suspicions: silent approximation fallbacks still reachable on experiment scripts that bypass the MCMCChain harness.
> Pattern: theoretical-correctness core is firm; remaining risk is operational — coverage gaps at branch boundaries + experiment artifacts that don't surface the new guardrails.

### Claude's Round 3 fixes (for Round 4 review)
- **R3-Fix1 (F1)** — added `_build_tiny_no_terminal_bool_grammar(include_non_terminable=False)` returning `{p,q:INT→BOOL, zero,one:INT}` (+ optional `r:LIST_INT→BOOL`). `test_score_subtree_normalizes_on_no_terminal_bool_lookahead_grammar[False/True]` monkeypatches `_score_depth_cap_lookahead_exact` to count invocations, verifies 4-program uniform support, forbidden r-headed programs score `-inf`, branch fires exactly `4 + len(forbidden_keys)` times. `test_sample_matches_lookahead_scorer_empirics_on_no_terminal_bool` draws N=20000 from `_sample` at depth=0 max_depth=0 and asserts 0 out-of-support, 0 r-headed, 4σ tolerance on uniform 1/4.
- **R3-Fix2 (F2)** — wired `reset_approximation_fallback_counters` / `get_approximation_fallback_counters` into `review-stage/experiments/round_1/posterior_calibration_v2.py`. Each seed resets on chain start, records counters in its per-seed result; summary sums across seeds and asserts zero. JSON artifact records `approximation_fallback_counters_sum` and `approximation_fallback_counters_all_zero` — grep-visible W2 guardrail.
- **R3-Fix3 (F3)** — renamed `reversible_given_typed` → `has_reverse_sites_given_typed` in `test_propose_regeneration_preserves_typability_and_reversibility`; docstring now explicitly disclaims that `log_q_rev` can be `-inf` even when typability and non-empty reverse sites both hold; assertion and log message updated to match.

### Verification
13/13 R2+R3 regression tests pass in 24.7s (root-type agreement, fallback counters, no-terminal lookahead, full-kernel normalizes, proposal typability, tiny binary grammar).

---

## Night 3 Round 4 — Score: 9.0/10, Verdict: Ready (FINAL)

**Reviewer read commit `26f55f4` directly and ran direct `python -c` probes.**

### Rulings on Round 3 findings
- **F1 (no-terminal lookahead branch coverage): OVERRULED.** Monkeypatch counter at test_mcmc_search.py:1484 wraps `ms._score_depth_cap_lookahead_exact` and fires exactly `6` times (4 valid + 2 forbidden) on the `include_non_terminable=True` parameterization. Scored masses: `1/4` for `p/q × zero/one`, `-inf` for `r(zero/one)`. Sampler companion at :1546 hits only the 4 expected keys on a 2000-draw probe.
- **F2 (experiment-side W2 assertion): OVERRULED.** Guardrail is on the live reported-result path: `run_mh_one_seed()` resets at posterior_calibration_v2.py:397 and records at :468; `per_seed` carries counters at :541; `main()` aggregates and asserts zero at :562. Reviewer monkeypatched `run_mh_one_seed()` to return a nonzero count — `main()` raised the new assertion as intended.
- **F3 (prose/assert mismatch): OVERRULED.** Docstring at test_mcmc_search.py:2044 explicitly says the invariant is weaker than "finite reverse density" and tells callers to check `math.isfinite(log_q_rev)`. Reviewer independently reproduced `log_q_rev=-inf` on (seed=10, proposal seed=7929) with a typed proposal and non-empty reverse sites — new wording matches reality.

### No new regressions
Reviewer found no silent regression introduced by commit `26f55f4`. All remaining items are known non-blockers: depth-2 calibration scope, init-resampling manuscript caveat, `run_parallel_chains` sequentiality.

### Reviewer memory update (verbatim, Round 4 FINAL)
> Addressed: F1 is closed by the new no-terminal-BOOL lookahead tests, which genuinely hit `_score_depth_cap_lookahead_exact`; F2 is closed by wiring the fallback counters into the live calibration script path with a repo-visible zero-counter assertion and JSON/log reporting; F3 is closed because the proposal regression test now states exactly the weaker invariant it actually checks, and it explicitly warns that `log_q_rev` may be `-inf` even for typed proposals. Still open: no new correctness blockers under the scoped claim; only non-blocking scope/performance caveats remain, namely depth-2 calibration must stay honestly scoped in the manuscript, the init-resampling caveat must stay in the writeup, and `run_parallel_chains` is still sequential. Pattern: the Night 3 fixes resolved the remaining MH-kernel correctness risks; what remains is claim-scoping and benchmark-engineering, not a forward/reverse-density soundness defect.

**Final bottom line:** Theoretical-correctness section is ready for submission under the claim scope "this MH implementation targets the post-burn-in posterior under the current gallery grammar, with depth-2 calibration." Non-blocking caveats (depth-2 scope honesty, init-resampling language, sequential parallel chains) must be reflected in the manuscript.
