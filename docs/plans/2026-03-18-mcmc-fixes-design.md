# MCMC Search Fixes — Design Document

## Summary

Five fixes to the MCMC program search pipeline on branch `feature/mcmc-search`, addressing issues discovered during the first quick runs. All fixes target `src/gallery_analysis/mcmc_search.py` and `src/gallery_analysis/analyze_mcmc.py`; no changes to the enumeration pipeline on main.

---

## Fix 1: Dead Chains from Oversized Initial Programs

**Problem:** `sample_program()` with `max_depth=6` often produces programs of size 100-200+, far exceeding the `max_nodes=25` constraint. Every proposal is rejected, freezing the chain for its entire run. With 2 chains, this wastes 50% of compute and pollutes results (the frozen program becomes the #1 hypothesis by visit count).

**Fix:** Initialize chains with `max_depth=3` (almost always produces programs < 25 nodes). Keep `max_depth=6` for subtree regeneration during proposals.

**Rationale:** Cognitively plausible — learners start with simple hypotheses and build complexity through revision. The chain can grow program size through accepted proposals, so depth-9 programs are still reachable. LOTlib3 avoids this problem by using `maxnodes=25` during initialization too; we achieve the same effect via depth control.

**Implementation:** Add an `init_max_depth` parameter to `MCMCConfig` (default 3). Use it in `MCMCChain.run()` when calling `sample_program()` for the initial program. The `max_depth` parameter continues to govern proposal regeneration.

---

## Fix 2: Identical Trajectories Across Rules

**Problem:** The first-passage orderings for `all_even` and `strict_increasing` are byte-for-byte identical. The exemplar hands (which differ between rules) have zero influence on the chain's trajectory. This happens because with `noise_epsilon=0.01` and only 6 positive exemplars, the likelihood surface is too flat — most programs are either tautologies (same likelihood for all rules) or crash (likelihood = -inf).

**Fix (a): Likelihood annealing.** Multiply log-likelihood by a temperature parameter β that ramps from a low value (e.g., 0.1) to 1.0 over the course of the chain. Early steps explore broadly via the prior; later steps concentrate on data-fitting hypotheses.

The annealing schedule: linear ramp from `beta_start` to `beta_end` over the chain's steps.

```
β(step) = beta_start + (beta_end - beta_start) * (step / n_steps)
```

New MCMCConfig fields: `beta_start: float = 0.1`, `beta_end: float = 1.0`.

In the MH loop, the acceptance ratio becomes:
```
log_alpha = (β × proposed_log_lik + proposed_log_prior + log_q_rev)
          - (β × current_log_lik   + current_log_prior  + log_q_fwd)
```

**Cognitive interpretation:** Early learning is exploratory (prior-driven hypothesis generation), later learning is data-driven (likelihood sharpens as the learner attends more to the evidence). This maps onto theories of attention allocation during category learning.

**Fix (b): Per-rule seed variation.** Use `seed = base_seed + hash(rule_id) % 100000` so each rule gets a different random trajectory, even when the likelihood doesn't differentiate.

**Implementation:** Hash the rule_id and add it to the seed. This requires passing the rule_id (or a rule-specific seed offset) to `run_parallel_chains()`.

---

## Fix 3: Tautology Rejection

**Problem:** Programs like `(λ lt 0 5)` and `(λ eq $0 $0)` — which return True on every possible hand — appear as top hypotheses for every rule. `(λ lt 0 5)` doesn't even reference the hand variable; it computes a constant boolean from integer literals.

The type system allows this because it checks type *shape*, not data *flow*. `λ. (lt 0 5)` type-checks as `Hand → bool` because the lambda introduces a `Hand` variable that the body is free to ignore.

**Three-layer fix:**

**Layer 1 — Vacuous lambda rejection (syntactic, during proposals):** After generating a program `λ. body`, check whether `$0` appears free in `body`. If not, reject. This catches all programs that ignore their input entirely. Cost: near-zero (`free_indices()` is cached on every AST node). Precedent: LOTlib3's `check_lambdas` option.

**Layer 2 — 100%-on-probes rejection (during chain):** After evaluating a proposed program on the 10K probe hands for extension-size estimation, if it scores 100% (accepts every single probe hand), reject it. The probability that a non-tautological program with a meaningful extension scores 100% on 10K probes is < 10^{-4} even at 99.99% base rate. Cost: zero additional — the probe evaluation already happens in `compute_mcmc_log_likelihood()`.

**Layer 3 — Post-hoc extension-size filtering (in results):** When building `top_hypotheses`, flag programs whose extension exceeds 50% of TOTAL_HANDS. These are near-tautologies that passed the first two filters. Cost: zero (comparison on already-computed values). This is an analysis-time filter, not a chain modification.

**Implementation:** Layers 1 and 2 modify the MH loop (reject before computing the full MH ratio). Layer 3 modifies the results construction. Layer 2 requires `compute_mcmc_log_likelihood()` to return the extension fraction alongside the log-likelihood, or to be refactored slightly.

---

## Fix 4: Output File Overwriting

**Problem:** Successive `--quick` runs overwrite the same output file.

**Fix:** Add a timestamp to the output filename: `mcmc_quick_20260318_143022_results.json`.

**Implementation:** Use `datetime.now().strftime('%Y%m%d_%H%M%S')` in the filename construction in `analyze_mcmc.py`.

---

## Fix 5: Wrong TOTAL_HANDS Constant

**Problem:** `TOTAL_HANDS = C(52,6) = 20,358,520` counts unordered combinations. But hands in the gallery experiment are ordered (position matters for rules like `strict_increasing`, `colors_palindrome`). The probe set is sampled as ordered permutations via `rng.sample(deck, 6)`. The correct value is `P(52,6) = 14,658,134,400`.

The error doesn't affect relative hypothesis rankings (the 720× factor cancels in ratios). But it inflates the noise floor `ε/TOTAL_HANDS`, making the likelihood surface flatter than intended — contributing to Issue 2.

**Fix:** Change `TOTAL_HANDS` to `14_658_134_400` in `mcmc_search.py`. The same fix is needed in `bayesian_scorer.py` and `hypothesis_table.py` on main (to be done independently).

---

## Files Modified

| File | Fixes |
|------|-------|
| `src/gallery_analysis/mcmc_search.py` | 1, 2a, 3 (layers 1-2), 5 |
| `src/gallery_analysis/analyze_mcmc.py` | 2b, 4 |
| `src/gallery_analysis/mcmc_hypothesis_collector.py` | 3 (layer 3) |
| `src/tests/test_mcmc_search.py` | Updated tests for all fixes |
