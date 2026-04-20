# Claims from Results — Night 1 Adversarial Review

**Source:** `review-stage/AUTO_REVIEW.md` (4 rounds) + `/tmp/r3_f1_sensitivity_out.json`
**Generated:** 2026-04-20T04:14Z via `/result-to-claim` (Codex GPT-5.4, xhigh reasoning; thread `019da91c-45ad-7020-a677-ecd540f63f25`)
**Overall verdict:** "substantially repaired and adversarially audited implementation", NOT "fully verified method".

## Paper-safe bottom line

- **C2** (weighted variants are truly weighted) and **C3** (fail-closed on non-finite priors): supportable as stated.
- **C1** (method correctness) and **C5** (sidecar alignment): soften to "adversarially audited and code-verified implementation" rather than "correct/proven aligned".
- **C4** (approximate-equivalence bound): keep explicitly qualified — reduced-scale audit only — until full-scale residual-sensitivity audit is done.

---

## Claim C1 — Method correctness

- **Status:** partial
- **Confidence:** medium
- **Supported wording (Codex-recommended):** "The current branch implements the intended Bayesian rule-induction pipeline and passed four rounds of adversarial code review; we treat this as strong empirical validation of the implementation, not a formal proof of correctness."
- **What the results support:** 19 concrete defects found, all implemented and code-verified by R4. Branch cleared the review threshold (6/10, "almost") with no new blocker.
- **What the results do NOT support:** a blanket method-correctness claim. R4 still ended at 6/10; one regression test still encodes pre-fix semantics and needs updating; full-scale residual audit not run.
- **Missing evidence / next experiments:**
  - Full-scale end-to-end audit of posterior outputs.
  - Update stale baseline test `tests/test_injection.py::test_merge_updates_summed_prior` to match new `summed_prior_with_injections` semantics.
  - Golden fixture tests exercising posterior computation and pool construction across the full pipeline.

## Claim C2 — Weighted-grammar variants are weighted models

- **Status:** yes
- **Confidence:** medium
- **Supported wording:** "On the reviewed branch, variants labelled `weighted` are enumerated under a weighted grammar rather than obtained by post-hoc reweighting of a uniform pool."
- **What the results support:** R1 identified that the weighted variant used post-hoc reweighting; fix (true weighted enumeration path in `build_hypothesis_pool` + `--enumeration-grammar {uniform,weighted}` CLI) implemented and code-verified by R4.
- **What the results do NOT support:** a quantitative audit of weighted-enumeration traces. Evidence is code-review and regression-based.
- **Missing evidence / next experiments:** regression test showing the weighted branch's ordering / mass allocation differs from `uniform-plus-reweighting` in the expected direction on a hand-constructed small-grammar case.

## Claim C3 — Fail-closed invariance on non-finite priors

- **Status:** yes
- **Confidence:** medium
- **Supported wording:** "In strict mode, the scorer fails closed on non-finite log-priors instead of silently renormalizing them into the posterior."
- **What the results support:** R1 flagged `nan`/`-inf` prior poisoning (e.g., the polymorphic-type-failure string `(λ all ((λ $0)) (map ((λ at $1 (rank_val $0))) $0))` returning `-inf` from `compute_log_prior` and poisoning its class). `_recompute_class_prior(strict=True)` implemented and wired; code-verified by R4.
- **What the results do NOT support:** a broad numerical-stability theorem; this is about the specific failure mode.
- **Missing evidence / next experiments:** unit + integration tests exercising non-finite priors across main scorer AND both sidecars.

## Claim C4 — Approximate-equivalence bound

- **Status:** partial
- **Confidence:** medium
- **Supported wording:** "Fingerprint equivalence is an empirical approximation whose downstream effect appears small in reduced-scale audits; full-scale sensitivity remains to be quantified."
- **What the results support:** collision rate dropped from 1.7% (R1, 14/813) to ≤ 0.2% (R3, 3/1858 post-inject on 2k unseen hands at full scale). Reduced-scale residual-sensitivity audit (depth=6 / 100k / 5 rules): max |Δrank|=2 (on a rule with posterior ≈ 1.45e-22), max |Δprob|=1.68e-10, max mixed-class mass=7.70e-05, 0/5 top-1 flips.
- **What the results do NOT support:** an unqualified "small and quantified effect" at paper scale. Downstream sensitivity audit was reduced-scale only; full-scale audit was explicitly left as the one remaining R4 recommendation.
- **Missing evidence / next experiments:**
  - **Full-scale residual-sensitivity audit** at `depth=7 / max_programs=300k / all 60 rules / all official variants` — report mixed-class mass, posterior deltas, rank changes, top-1 stability. (This is the single pre-submission recommendation from R4.)
  - Report distributional summaries (median, p95), not just maxima.

## Claim C5 — Sidecar alignment

- **Status:** partial
- **Confidence:** medium
- **Supported wording:** "The reviewed branch aligns the sidecar analyses with the main scorer on the audited issues; parity is code-verified on reviewed paths, not exhaustively proven."
- **What the results support:** R2 identified pool-construction, prior-computation, and depth-semantics drift in `run_diagnosticity.py`, `depth_mass_analysis.py`, `compare_prior_modes.py`. Fixed via shared helper `merge_injections_and_extend` and unified `_recompute_class_prior` across scorers; code-verified by R4.
- **What the results do NOT support:** exhaustive universal parity. No automated cross-tool parity harness yet.
- **Missing evidence / next experiments:** golden tests comparing main scorer vs. each sidecar on shared inputs for (a) pool membership, (b) prior values, (c) depth-semantics labels, (d) retained-mass reporting.

---

## Routing

- **C2, C3** (status yes): record as supported claims for the paper.
- **C1, C4, C5** (status partial): keep paper wording explicitly qualified; schedule the listed next experiments before final submission. The single most impactful follow-up is **C4's full-scale residual-sensitivity audit** — Codex (R4): "if that bound stays in the same regime you already found, I would clear it."

## Codex trace

- Final R4 review thread: `019da777-8714-71e3-9784-5bea4a10bed1`
- Result-to-claim evaluation thread: `019da91c-45ad-7020-a677-ecd540f63f25`
