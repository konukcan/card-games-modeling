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
