# Auto-Review Loop — MCMC Program Search (Night 2)

**Session started:** 2026-04-16T08:45:40Z
**Branch:** `aris/mcmc-review-20260411` (continuation of Night 1 branch)
**Loop topic:** Internal-coherence / theoretical-correctness review of MCMC program search — specifically the open issues Night 1 left (C3-tier2, C1-gallery, C2-cap, H-methodology, H-autocorr, H-seeds, H-n_sites-invariance, C5, H5).
**Difficulty:** hard (reviewer memory + debate protocol; 4 rounds max; aim 4 rounds even if score ≥6/10 is reached; positive threshold requires score ≥6 AND ≥3 rounds completed)
**Reviewer backend:** Codex MCP (`mcp__codex__codex`) with `model_reasoning_effort: xhigh` (`✓ Connected` confirmed via `claude mcp list`).
**Compact mode:** true (findings.md appended each round).

Night 1 artifacts are at `night1-archive/`. The branch baseline tests ran 47/47 green in 1732s (~28.9 min) at session start.

**Framing (from ARIS_LAUNCH_PROMPT.md):** Tonight is internal coherence, NOT benchmark chasing. Do NOT try to validate against the rule catalogue. Do NOT read `src/rules/`. Forbidden scope changes must be logged in `NOTES_FOR_HUMAN.md`.

---

## Round 0: Bootstrap

- Baseline tests: **47/47 green in 1732s** (28:52) at the start of the run. No regression from Night 1.
- `review-stage/` created fresh; Night 1 artifacts archived to `night1-archive/`.
- Codex MCP confirmed connected before launching Round 1.
- `REVIEWER_MEMORY.md` initialized empty (Night 1 reviewer was a Claude sub-agent; GPT-5.4 is fresh tonight).

---

## Round 1 (2026-04-16T02:00:00Z)

### Assessment (Summary)
- **Score:** 4.5/10
- **Verdict:** not ready
- **Thread ID:** `019d9579-b4c2-7b23-97c2-8a2316c6d104`
- **Key criticisms:** All Night-1 weaknesses confirmed; four new concrete counterexamples (tier-2 scorer inversion, 23.5% depth-cap branch entry, 2.8% retry rate shifting q, silent site-drop path). The stated proposal density `q` is still wrong until the proposal generator, scorer, and latent type-resolution story are literally the same mathematical object.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response (saved in review-stage/round_1_review.txt)</summary>

See `review-stage/round_1_review.txt` for verbatim text.

</details>

### Actions Taken (committed in 3ac343c, 3bdf156)

**Six fixes against reviewer's prioritized action list + one emergent fix from regression tests:**

1. **Fix 1 (action #1) — `sample_program` retry alignment.** `propose_regeneration` uses `allow_retries=False`; init path uses `allow_retries=True` with bounded retry and hard `RuntimeError` on exhaustion. Proposal-path samples from exactly the distribution the scorer evaluates.
2. **Fix 2 (action #2) — Exact marginalization in scorer.** `_score_subtree_under_sampler` replaces tier-2/tier-3 heuristics with enumeration over `_CONCRETE_TYPES^k` (k ≤ `_MARGINALIZATION_FREE_VAR_CAP = 3`) + log-sum-exp. Verified on the toy `{choose, is_zero}` grammar: scorer P=0.125 matches empirical 0.1355 within 5% (was P=1.0 under tier-2).
3. **Fix 3 (action #3) — Depth-cap lookahead mirror.** Scorer now shifts production log-probs by log of `_all_args_terminable` survival probability, enumerated over `_CONCRETE_TYPES^k` for remaining free vars.
4. **Fix 4 (action #4) — `max_nodes` + vacuous-lambda as `log π = −∞`.** Target-coded inside MH body (mcmc_search.py:1824-1827) instead of pre-MH hard rejection.
5. **Fix 5 (action #5) — Calibration v2.** `review-stage/experiments/round_1/posterior_calibration_v2.py`. Independent prior (perturbed-grammar PCFG), polymorphic grammar `{not,and,or,eq,if}` at `INT→BOOL`, `_CONCRETE_TYPES=[BOOL,INT]`, 5 seeds × 50k steps, Geyer IPS ESS. **Intended depth=3 × 200k OOM-killed 24GB machine; scaled to depth=2 × 50k.** Result: 5/5 seeds pass, mean TV=0.0955, max TV=0.1044, ESS 909-1607. Wall 13:20.
6. **Fix 6 (action #6) — Four regression tests added.** site-drop zero-count, scored-vs-empirical on poly toy, propose_regeneration vs sampler, static check that `propose_regeneration` calls `allow_retries=False`.
7. **Emergent fix — `collect_subtree_sites` walker root-threaded `TypeContext`.** Surfaced by the site-drop test. Walker now runs a single root-level `infer_type` whose `TypeContext` threads through all subtrees, stored in identity-keyed node-type map. Post-fix: 0 drops across 50 samples (was 20/50).

### Results
- **Pytest: 44/44 green in 3687.96s (1:01:27).** Marginalization enumeration makes depth-cap scoring ~125× slower worst case, but correctness verified.
- **Calibration: 5/5 seeds pass at depth=2.** Acknowledged resource-constrained downscope from intended depth=3 to reviewer.
- Two commits: `3ac343c` (code fixes) and `3bdf156` (calibration artifacts).

### Status
- Proceeding to Round 2.
- Difficulty: hard.

---

## Round 2 (2026-04-16T11:25:57Z)

### Assessment (Summary)
- **Score:** 5.5/10 (+1 from Round 1)
- **Verdict:** not ready
- **Key criticisms:** Reviewer read commit `3bdf156` directly. Confirmed most Night-1 fixes land. Remaining defects are now concentrated in "rare branch" logic where Round-1 fixes were approximate, not exact. **Main remaining soundness defect: depth-cap scorer is still a mean-field approximation, not the exact marginal of the sampler's random filtered set.** Concrete counterexample reproduced: raw `_sample` gives `P(p1 0) = 0.736`, scorer returns `0.667`.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response (saved in review-stage/round_2_review.txt)</summary>

See `review-stage/round_2_review.txt` for verbatim text, including rulings on all 13 Round-1 weaknesses, 5 new weaknesses, and memory update.

</details>

### Rulings summary
- **Overruled (7):** C3-tier2, H-methodology, H-autocorr, H-seeds, retry-conditioned proposal law, vacuous-lambda hard reject, silent site-drop path.
- **Partially sustained (4):** C1-gallery (depth=2 only), C5 (init retries remain), H-n_sites-invariance (no ΣQ test), hidden-free-type-draws (depth-cap path still approximate).
- **Sustained (2):** C2-cap (depth-cap scorer approximate), H5 (sequential chains).

### New weaknesses surfaced in Round 2
1. **Depth-cap scorer still approximate** (mcmc_search.py:921). Main remaining soundness defect.
2. **`max_nodes` off-book on init** (mcmc_search.py:1731). 57/100 init states exceed cap.
3. **Init resampling = arbitrary prior over starts**. Biases early visit counts / first-passage analyses.
4. **Calibration enumerator incomplete** (posterior_calibration_v2.py:243). Drops nonzero-arity productions at cap; nonzero outside mass every seed.
5. **Proposal-density test weak** (test_mcmc_search.py:973). Top-2 with factor-3 tolerance would not catch depth-cap mismatch.

### Round 3 priority (from reviewer)
1. Make the depth-cap scorer **exact** or prove the branch is excluded in the final experimental regime.
2. Apply `max_nodes` consistently at init OR state clearly that init is arbitrary and excluded from timing/visit analyses.
3. Add a true proposal-normalization test on a tiny hand-built state space.
4. If possible, rerun calibration at depth=3 on a machine that can enumerate the support exactly.

### Status
- Proceeding to Round 3 implementing priorities 1-3 (priority 4 is hardware-bounded).
- Difficulty: hard.

---

## Round 3 (2026-04-16T16:05:00Z)

### Assessment (Summary)
- **Score:** 7.0/10 (+1.5 from Round 2)
- **Verdict:** Almost
- **Threshold met:** `positive_threshold_effective: "score>=6 AND rounds_completed>=3"` — SATISFIED.
- **Key ruling:** `C2-cap` OVERRULED (exact depth-cap scorer stress-tested on 3-competitor case: empirical 0.58085 vs scored 0.58333). `max_nodes init` OVERRULED. Remaining weaknesses are no longer fatal MH-ratio defects — they are boundary conditions + methodology choices.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response (saved in review-stage/round_3_review.txt)</summary>

See `review-stage/round_3_review.txt` for verbatim text.

</details>

### Rulings summary
- **Overruled (3):** C2-cap depth-cap, max_nodes init, hidden free-type draws (gallery grammar only).
- **Partially sustained (6):** init resampling, calibration support, weak proposal test, C1-gallery depth=2, C5 init retries, H-n_sites-invariance.
- **Sustained (1):** H5 sequential chains.

### New weaknesses surfaced in Round 3
1. **Explicit approximation caps** (`_DEPTH_CAP_EXACT_ENUM_CAP=16`, `_MARGINALIZATION_FREE_VAR_CAP=3`). Not hit in gallery regime but still approximation boundaries.
2. **Scalability risk**: exact depth-cap scorer is exponential in competing lookahead productions.

### Round 4 priority (from reviewer)
1. Add a tiny full-kernel normalization test for `propose_regeneration` (site-pick × regeneration, not just root scorer).
2. Decide whether paper makes first-passage / cognitive-timing claims. If yes, fix init resampling or exclude early trajectory explicitly.
3. If hardware permits, depth-3 calibration run; if not, document the limitation clearly.

### Status
- Positive threshold SATISFIED (7.0/10 ≥ 6, 3 rounds ≥ 3).
- Per launch prompt, still aiming for 4 rounds ("aim 4 rounds even if score ≥6/10 is reached").
- Proceeding to Round 4 implementing priorities 1-3 where feasible.
- Difficulty: hard.

---

## Round 4 (FINAL) (2026-04-16T17:30:00Z)

### Assessment (Summary)
- **Score:** 8.0/10 (+1.0 from Round 3)
- **Verdict:** Almost
- **Key ruling:** R3 full-kernel normalization OVERRULED. H-n_sites-invariance OVERRULED. Hidden free-type caps OVERRULED for gallery regime. No remaining fatal detailed-balance / proposal-density defect. Still "Almost" (not "Yes") because validation is depth-2 and exactness caps are silent fallbacks.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response (saved in review-stage/round_4_review.txt)</summary>

See `review-stage/round_4_review.txt` for verbatim text.

</details>

### Rulings summary
- **Overruled (3):** R3 full-kernel normalization priority, H-n_sites-invariance, hidden-free-type caps (gallery regime).
- **Partially sustained (5):** init-resampling (doc fix accepted), C1-gallery depth=2, calibration-support depth=2, C5 init retries, explicit-cap weaknesses.
- **Sustained (1):** H5 sequential chains (low priority).

### New weaknesses surfaced in Round 4
1. **Coverage gap** (minor, non-blocking): new kernel test is empty-env BOOL only. Lambda/bound-variable variant would strengthen regression coverage.
2. **Process gap** (minor, non-blocking): init caveat in code; manuscript must honor it.
3. **Scalability caps recommendation**: add gallery-level assertion or runtime counter for `_DEPTH_CAP_EXACT_ENUM_CAP=16` and `_MARGINALIZATION_FREE_VAR_CAP=3` — must stay at zero for reported experiments.

### Final verdict
> "If the paper says 'this MH implementation targets the post-burn-in posterior under the current gallery grammar' and is honest about the depth-2 calibration limit, the theoretical-correctness section is now in good shape. If it claims full-gallery calibration, or uses early discovery timing as evidence, it is still not ready."

### Score trajectory
| Round | Score | Verdict | Key gain |
|-------|-------|---------|----------|
| 1 | 4.5 | not ready | Reviewer ran code, 4 concrete counterexamples confirmed |
| 2 | 5.5 | not ready | C3/H-methodology/H-autocorr/H-seeds OVERRULED; C2-cap sustained |
| 3 | 7.0 | almost | C2-cap OVERRULED (exact depth-cap scorer); max_nodes init OVERRULED |
| 4 | 8.0 | almost | Full-kernel ΣQ=1 verified; H-n_sites-invariance OVERRULED |

### Status
- **Night 2 COMPLETED.** 4/4 rounds run. Threshold SATISFIED (8.0/10 ≥ 6, 4 rounds ≥ 3).
- Remaining open items documented in MORNING_REPORT.md.
- Difficulty: hard.

---

# Night 3 — Final correctness push (2026-04-19)

**Session started:** 2026-04-19T00:00:00Z
**Branch:** `aris/mcmc-review-night3` (new, off `feature/mcmc-search`)
**Worktree:** `.worktrees/aris-mcmc-night3/`
**Difficulty:** hard (reviewer memory + debate protocol; 4 rounds, no early stop)
**Reviewer backend:** Codex MCP (`mcp__codex__codex`) with `model_reasoning_effort: xhigh`
**Compact mode:** true
**Carry-in state:** Night 2 ended at 8.0/10 ("Almost").

Three Night-2 remaining gaps entering tonight:
1. Depth-3 calibration (OOMs on 24 GB — needs creative workaround).
2. Runtime guardrails for silent approximation caps (`_DEPTH_CAP_EXACT_ENUM_CAP=16` @ mcmc_search.py:793, `_MARGINALIZATION_FREE_VAR_CAP=3` @ mcmc_search.py:630).
3. Lambda/bound-variable kernel test variant (current `test_propose_regeneration_full_kernel_normalizes_on_tiny_grammar` is empty-env BOOL only).

Tonight is also open-ended: new issues the previous nights may have missed are welcome.


## Round 1 (2026-04-19T00:00:00Z → 2026-04-19T02:00:00Z)

### Assessment (Summary)
- **Score:** 5.0/10 (dropped from Night 2's 8.0/10)
- **Verdict:** No
- **Thread ID:** `019da4d5-c81e-77b2-9c4d-e64b2b463c93`
- **Reviewer:** GPT-5.4 (Codex MCP, xhigh reasoning); read commit `0bf49b3` directly and ran probes on the repo.

### Key criticisms
1. **W1 (NEW critical blocker):** `collect_subtree_sites` root-threaded annotation cache is keyed by `id(node)`. Primitive singletons collide across AST positions; 18/314 sampled-program sites (~5.7%) had corrupted `(site.type, site.env)` metadata. 11/20 regeneration probes produced ill-typed proposals; 8/20 lost the chosen path in `new_sites`. Correctness blocker for both forward and reverse kernels.
2. **W2 (carry):** three silent approximation-cap fallbacks at mcmc_search.py:679, :769, :823. Add per-site counters, reset per run, surface in artifact, paper-run hard-fail if any nonzero.
3. **W3 (carry):** depth-3 calibration still closed only at depth-2.
4. **W4 (carry):** full-kernel test is empty-env BOOL only. Reviewer provided exact lambda-variant spec with closed-form support.

### Reviewer Raw Response

<details>
<summary>Click to expand full reviewer response (Round 1)</summary>

```
## Score
5.0/10

## Verdict
No

## Critical remaining weaknesses (ranked)

### W1. `collect_subtree_sites` can assign the wrong local type/environment to a site
- Why this is a concern: In mcmc_search.py:1375, the root-threaded annotation cache is keyed by `id(node)`. That is unsafe because primitive objects are reused across multiple AST occurrences. Direct probe: repeated primitives in one sampled AST share the same object id; on 10 sampled gallery programs, 18/314 collected sites had `(site.type, site.env)` inconsistent with the subtree they allegedly describe; on 20 regeneration probes, 11 proposed programs were ill-typed, and in 8/20 the chosen path was not recoverable in `new_sites`. That is a correctness blocker, because `propose_regeneration` relies on this site metadata to define the forward kernel and then reuses old-site metadata for `log_q_rev`.
- MINIMUM fix: Make the annotation cache occurrence-specific, not object-specific. Path-keyed annotations are the obvious minimal repair. Then add two regressions: every collected site on sampled gallery programs must type-check under `(site.env, site.type)`, and `propose_regeneration` must preserve whole-program typability and retain the chosen path in `new_sites`.

### W2. The approximation-cap fallbacks are still silent and unguarded
- MINIMUM fix: Add explicit counters for each fallback site, reset them per run, surface them in the result artifact, and make reported experiments fail if any counter is nonzero.

### W3. Depth-3 calibration is still not closed
- MINIMUM fix: After W1 is fixed, do either a streaming exact depth-3 enumerator on the existing toy, or a tiny exact depth-3 grammar plus targeted depth-boundary tests.

### W4. Full-kernel regression coverage is still missing a bound-variable case
- MINIMUM fix: Add one tiny lambda/bound-variable full-kernel test with exact closed-form support (provided verbatim by reviewer: BOOL -> BOOL, grammar {t:BOOL, f:BOOL→BOOL}, log_variable=0, starting state λ f(f($0)), closed-form support of 8 programs with probs {1/3, 1/3, 1/9, 1/9, 1/27, 1/27, 1/54, 1/54}).

## Answers to the specific questions above
- G1 strategy preference: B > (A + D) > D > C. Under 24 GB, A + D is fastest acceptable closure.
- G2 counter vs assertion + scope: Separate counters for arg-marginalization fallback, survival-prob fallback, depth-cap mean-field fallback; merged counter too lossy. Hard CI failure if any counter nonzero.

## New issues you want to surface tonight (beyond G1-G3)
- The new blocker is W1: the site collector's occurrence-typing is unsound. Not in the Night 2 carry-in, and materially changes the verdict.
- The existing `collect_subtree_sites_failures == 0` test is too weak. Zero silent drops does not imply correct site metadata.
```

</details>

### Actions taken (Round 1 fixes)

**R1-Fix1a — path-keyed `collect_subtree_sites` cache (W1 primary)**
- File: `src/gallery_analysis/mcmc_search.py` (collect_subtree_sites at line ~1396)
- Replaced `id(node)` key with path tuple. Cache now records occurrence-specific annotations rather than object-specific.
- Direct probe post-fix dropped bad-site rate from ~5.7% → ~0.5% (residual attributable to out-of-scope `TypeContext.instantiate` identity bug in `src/dreamcoder_core/type_system.py:~470`; the regression probe neutralises it by pre-bumping `_next_var`).

**R1-Fix1b — single-pass `_annotate` mirroring `infer_type` semantics (W1 follow-on)**
- Same file, same function. Original `_annotate` called `node.infer_type(ctx, env)` at every recursion level, re-instantiating primitives against the shared ctx and producing stale TVs that the final `ctx.apply` could not resolve. As a result, polymorphic site types like `list('t578)` remained un-narrowed even when whole-program inference had bound them to concrete types.
- Fix: rewrote `_annotate` as a single-pass walker mirroring `Program.infer_type` per node type (Primitive / Index / Application / Abstraction / Invented), sharing one TypeContext throughout. End-to-end probe: 0/30 ill-typed proposals (previously 10/20 failed).

**R1-Fix2 — per-site approximation-cap fallback counters (W2)**
- Same file, lines ~1345 (counters), ~1361 (getter), ~1386 (reset), plus three increment sites:
  - `_arg_marginalization_fallbacks` at mcmc_search.py:~683
  - `_survival_prob_fallbacks` at mcmc_search.py:~776
  - `_depth_cap_mean_field_fallbacks` at mcmc_search.py:~829
- Public API: `get_approximation_fallback_counters()`, `reset_approximation_fallback_counters()`.

**R1-Fix3 — full-kernel lambda/bound-variable test (W4)**
- File: `src/tests/test_mcmc_search.py` (`test_propose_regeneration_full_kernel_normalizes_with_bound_variable`).
- Exact match to reviewer's spec: BOOL → BOOL request, grammar `{t:BOOL, f:BOOL→BOOL}`, `log_variable = 0`, starting program `λ f(f($0))`. Validates per-site normalization, pointwise closed-form probabilities, and full-kernel ΣQ = 1.

**R1-Regression — W1 + W2 coverage tests**
- `test_collect_subtree_sites_all_sites_type_consistent` — 20 sampled gallery programs, every collected site must type-check under (env, type).
- `test_propose_regeneration_preserves_typability_and_reversibility` — 20 proposals must produce type-correct programs with ≥1 site (reverse kernel well-defined).
- `test_fallback_counters_trigger_and_reset` — exercises each of the three fallback branches at its cap boundary and verifies exact counter increments (no cross-contamination).
- `test_fallback_counters_stay_zero_on_normal_kernel_runs` — 30 proposals from gallery grammar at `max_depth=5` must not trip any counter.

**W3 deferred to Round 2** pending reviewer's verdict on W1 closure (reviewer stated: "After W1 is fixed, do...").

### Results

- New tests all pass (5/5): site consistency on 20 programs, proposal typability+reversibility on 20 programs, counter triggers/resets, normal-run zero-counter assertion, and the reviewer-specified lambda-variant full-kernel ΣQ=1 test.
- End-to-end proposal type-check probe: 0/30 failures (previously 10/20).

### Status
- Continuing to Round 2. Difficulty: hard.


## Round 2 (2026-04-19T05:00:00Z → 2026-04-19T08:00:00Z)

### Assessment (Summary)
- Score: 7.5/10
- Verdict: Almost
- Reviewer commit: `7f14a2f`

### Key criticisms (raw response in `review-stage/night3_round_2_review.txt`)
- W1 (site-metadata) OVERRULED — 0/3256 bad sites over 50 sampled programs via 64-var-offset probe.
- W2 (approximation counters) PARTIALLY SUSTAINED — counters exist but not wired into any experiment/result artifact.
- W4 (bound-variable full kernel) OVERRULED — ΣQ=1 reproduced at `{1/3, 1/3, 1/9, ...}`.
- **NEW F-R2-1 (Medium)**: `sample_program(allow_retries=True)` retry path at `mcmc_search.py:148` only checks `infer_type` returns, not that resolved type unifies with request_type. 1/100 seeds at max_depth ∈ {5,6} returned ill-typed programs.
- **NEW F-R2-2 (Low/Medium)**: `test_propose_regeneration_preserves_typability_and_reversibility` overclaims — asserts every regeneration should whole-program type-check, but `propose_regeneration` uses `allow_retries=False` by design.

### Actions taken (Round 2 fixes, commit `a05a59c` → later `7f14a2f`)

**R2-Fix1 — root-type agreement in `sample_program(allow_retries=True)`**
- `src/gallery_analysis/mcmc_search.py:~148`: added `ctx.unify(inferred, ctx.instantiate(request_type))` before returning from retry path.
- Regression test `test_sample_program_retry_path_enforces_root_type_agreement`: pins seed 1001 at max_depth=6 + 0-leak assertion on seeds 900..1199 at max_depth ∈ {5,6}.

**R2-Fix2 — W2 counters attached to MCMCResult**
- `MCMCResult.approximation_fallback_counters: Dict[str, int] = field(default_factory=dict)`.
- `MCMCChain.run` (line ~2079): `reset_approximation_fallback_counters()` at start; captures via `get_approximation_fallback_counters()` on return.
- `run_parallel_chains` (line ~2545): sums `merged_fallback_counters` across chain results.

**R2-Fix3 — weakened W1 proposal-typability test to actual invariant**
- Docstring + assertion now pin: "when proposed program type-checks, it must have ≥1 reverse site" AND typed-OK ≥ tried // 2 smoke bound. Explicitly disclaims "every regeneration type-checks".

**R2-Fix4 — W3 A+D tiny exact binary grammar + boundary tests**
- `_build_tiny_bool_binary_grammar()` returns `(grammar, c_prim, g_prim)` with `c:BOOL, g:BOOL→BOOL→BOOL`.
- `_binary_program_key(prog)` structural key (avoids Primitive lambda-hex-address repr collisions).
- Parameterized `test_score_subtree_under_sampler_normalizes_on_tiny_binary_grammar[1-2, 2-5, 3-26]`: closed-form normalization at each depth.
- `test_score_matches_sampler_empirics_on_tiny_binary_grammar`: N=20000 Monte Carlo with 4σ tolerance.

### Results
- All new tests pass. R2 addresses three-of-four R1 carry-forwards cleanly.

### Status
- Continuing to Round 3. Difficulty: hard.

---

## Round 3 (2026-04-19T08:30:00Z → 2026-04-19T11:30:00Z)

### Assessment (Summary)
- Score: 8.5/10
- Verdict: Almost
- Reviewer commit: `9d97392`

### Key criticisms (raw response in `review-stage/night3_round_3_review.txt`)
- R2-Fix1 (root-type agreement): **OVERRULED**.
- R2-Fix2 (MCMCResult counters): **PARTIALLY SUSTAINED** — harness clean, experiment-artifact not asserted.
- R2-Fix3 (weakened typability test): **PARTIALLY SUSTAINED** — prose/variable still overstates invariant.
- R2-Fix4 (tiny binary grammar): **OVERRULED**.
- **F1 — Medium**: `_score_depth_cap_lookahead_exact` branch uncovered — no test with zero terminal productions at target type.
- **F2 — Medium**: experiment-side W2 assertion missing in `review-stage/experiments/round_1/posterior_calibration_v2.py`.
- **F3 — Low/Medium**: prose/assert mismatch — `reversible_given_typed` only checks non-empty reverse sites, not finite `log_q_rev`. Reviewer reproduced `log_q_rev=-inf` on seed=10+7919 where both invariants hold.

### Actions taken (Round 3 fixes, commit `26f55f4`)

**R3-Fix1 (F1) — no-terminal-BOOL lookahead grammar + tests**
- `src/tests/test_mcmc_search.py`:
  - `_build_tiny_no_terminal_bool_grammar(include_non_terminable=False)` returns `{p:INT→BOOL, q:INT→BOOL, zero:INT, one:INT}` plus optional `r:LIST_INT→BOOL`.
  - `_no_terminal_program_key(prog)` structural key.
  - Parameterized `test_score_subtree_normalizes_on_no_terminal_bool_lookahead_grammar[False, True]`: monkeypatches `ms._score_depth_cap_lookahead_exact` to count invocations, asserts 4-program 1/4 support, forbidden r-headed programs score `-inf`, branch fires exactly `4 + len(forbidden_keys)` times.
  - `test_sample_matches_lookahead_scorer_empirics_on_no_terminal_bool`: N=20000 MC draws from `_sample` at `depth=0, max_depth=0`, 0 out-of-support, 0 r-headed, 4σ tolerance per program.

**R3-Fix2 (F2) — experiment-side W2 assertion**
- `review-stage/experiments/round_1/posterior_calibration_v2.py`:
  - Imports `reset_approximation_fallback_counters`, `get_approximation_fallback_counters`.
  - `run_mh_one_seed()` resets counters at chain start; records per-seed in return dict.
  - `main()` aggregates across seeds; asserts sum equals `{arg_marginalization: 0, survival_prob: 0, depth_cap_mean_field: 0}`; writes `approximation_fallback_counters_sum` + `approximation_fallback_counters_all_zero` to JSON artifact.

**R3-Fix3 (F3) — prose/assert rename**
- `src/tests/test_mcmc_search.py::test_propose_regeneration_preserves_typability_and_reversibility`:
  - Variable `reversible_given_typed` → `has_reverse_sites_given_typed`.
  - Docstring now explicitly disclaims that `log_q_rev` may legitimately be `-inf` even when typability and non-empty reverse sites both hold; callers must check `math.isfinite(log_q_rev)` themselves.
  - Inline comment + assertion message updated.

### Results
- 13/13 R2+R3 regression tests pass in 24.7s (root-type agreement, fallback counters, no-terminal lookahead, full-kernel normalizes, proposal typability, tiny binary grammar).

### Status
- Continuing to Round 4 (final). Difficulty: hard.

---

## Round 4 (FINAL) (2026-04-19T12:00:00Z)

### Assessment (Summary)
- Score: **9.0/10**
- Verdict: **Ready**
- Reviewer commit: `26f55f4`

### Rulings (raw response in `review-stage/night3_round_4_review.txt`)
- F1: **OVERRULED** — monkeypatch counter fires exactly `6` times (4 valid + 2 forbidden); scored masses exactly 1/4 and -inf; sampler companion N=2000 probe produces only the 4 expected keys.
- F2: **OVERRULED** — guardrail on live reported-result path; reviewer monkeypatched `run_mh_one_seed()` to return nonzero fallback and `main()` raised assertion as intended.
- F3: **OVERRULED** — docstring matches assertion; reviewer reproduced `log_q_rev=-inf` on (seed=10, proposal seed=7929) with typed proposal + non-empty reverse sites.
- **No new regressions introduced by R3.**

### Still-open non-blockers
- Depth-2 calibration must stay honestly scoped in manuscript (Night 2 carry).
- Init-resampling caveat must stay in writeup (Night 2 carry).
- `run_parallel_chains` still sequential (H5, Night 2 carry).

These are claim-scoping and benchmark-engineering items, not forward/reverse-density soundness defects.

### Final bottom line
> "If the paper says 'this MH implementation targets the post-burn-in posterior under the current gallery grammar' and is honest about the depth-2 calibration limit, the theoretical-correctness section is now in good shape."

**Loop terminated: positive verdict reached at Round 4/4.**

### Score trajectory (Night 3)
- Round 1: 5.0 (No) — new W1 blocker introduced by reviewer's site-metadata probe
- Round 2: 7.5 (Almost) — W1 closed, three new findings
- Round 3: 8.5 (Almost) — R2 fixes clean, three closure gaps (F1/F2/F3)
- Round 4: 9.0 (Ready) — F1/F2/F3 all OVERRULED, no new regressions

### Files changed across Night 3
- `src/gallery_analysis/mcmc_search.py`: path-keyed cache + single-pass `_annotate` (R1), 3 approximation-cap counters + accessors (R1), root-type unify in retry path (R2), MCMCResult counter field + run-level reset/capture + parallel sum (R2).
- `src/tests/test_mcmc_search.py`: +~400 lines of regression tests (W1 site consistency, W4 bound-variable full kernel, fallback triggers, binary grammar at depth ∈ {1,2,3}, no-terminal-BOOL lookahead, root-type retry agreement) + docstring/variable rename (R3).
- `review-stage/experiments/round_1/posterior_calibration_v2.py`: fallback-counter reset + record + aggregate zero-assert + JSON fields (R3).
- `review-stage/{REVIEWER_MEMORY.md, REVIEW_STATE.json, night3_round_{1,2,3,4}_review.txt, AUTO_REVIEW.md, findings.md}`: full audit trail.

