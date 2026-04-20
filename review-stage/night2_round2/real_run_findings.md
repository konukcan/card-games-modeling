# Night 2 Round 2 — Real MCMC vs enumeration comparison

**Pipeline:** `compare_enum_vs_mcmc.py` (post-Codex Round 4 ACCEPT, with hard validity gating).
**Enumeration:** depth=6, max_programs=100,000, 1,094 base classes → 1,261 post-injection.
**MCMC:** `mcmc_10000steps_4chains_20260319_065344_results.json` (10k steps × 4 chains, top_k=250 stored, max_depth=6, ε=0.01).
**Probes:** 5,000 uniform random 6-card hands (seed=99).
**Top-K reconstruction:** 50.
**Output:** `real_mcmc_compare_7rules.json`.

## Headline finding

**Of 7 rules tested, 7/7 (100%) fail the comparison-validity filter.**
Two flags trip on every rule:
- `mcmc_payload_truncated`: `mass_in_full_list` is 0.13–0.52, far below the
  pre-registered threshold of 0.90.
- `topk_truncation_excessive`: same root cause (top-K = full payload here).

| rule                    | valid | mean&#124;Δ&#124; | max&#124;Δ&#124; | ρ      | hit_e  | hit_m  | mass_top_k |
|-------------------------|-------|---------|---------|--------|--------|--------|------------|
| all_red                 | False | 0.0764  | 0.0997  | 0.069  | 0.0121 | 0.0871 | 0.521      |
| all_same_suit           | False | 0.0685  | 0.4849  | nan    | 0.0000 | 0.0685 | 0.430      |
| all_even                | False | 0.5913  | 0.8751  | 0.608  | 0.0762 | 0.6570 | 0.258      |
| triple_any_adjacent     | False | 0.7225  | 0.9481  | 0.660  | 0.1728 | 0.8671 | 0.215      |
| strict_increasing       | False | 0.8318  | 1.0000  | nan    | 0.0000 | 0.8318 | 0.341      |
| four_of_a_kind_adjacent | False | 0.5617  | 0.9871  | 0.172  | 0.0011 | 0.5625 | 0.133      |
| ranks_palindrome        | False | 0.3687  | 0.4302  | 0.159  | 0.0034 | 0.3716 | 0.371      |

Even ignoring the validity flags (which we should NOT do), the raw mean|Δ|
ranges from 0.07 to 0.83, with `four_of_a_kind_adjacent` and
`triple_any_adjacent` showing >50% disagreement on the predictive
acceptance probability. ρ is often near zero or NaN, indicating the rank
ordering of probe hands also disagrees substantially.

## What the validity flag tells us

Why `mass_in_full_list` is small: MCMC visits THOUSANDS of distinct
program strings, many of which represent the same equivalence class via
syntactic variation. The top-50 (and even the top-250 retained by
`analyze_mcmc.py`) captures only the highest-frequency individual STRINGS
— not the highest-mass equivalence CLASSES. Cumulative visit fraction in
the top-K trails well below 1 because the chain's mass is spread thinly
across the long tail.

This is exactly the concern Codex flagged in Round 1 and that the
`comparison_valid` gate, added in Round 4, now enforces hard.

## Three honest readings

The framework, run on this MCMC payload, **cannot answer** the headline
question "do MCMC and enumeration give compatible posteriors at the
extensional level on uniform 6-card probes." Three viable next steps:

1. **Aggregate MCMC visits onto equivalence classes BEFORE comparison.**
   The MCMC payload is per-program-string; the enumeration posterior is
   per-equivalence-class. Mapping MCMC strings to enumeration classes
   (via `parse_program → fingerprint → class lookup`) and summing
   visit weights per class would produce a comparable mass vector. Then
   `mass_in_full_list` would sum to ~1 across classes (the chain's
   total time spent in any class) instead of trailing off in the
   per-string tail.

2. **Run MCMC with larger top_k retention** (e.g., top-2000 instead of
   top-250) and rerun. Validity should improve mechanically.

3. **Run MCMC with more steps so visit-frequency posterior is closer to
   the stationary distribution.** 10k steps × 4 chains may be too few
   to converge.

For Night 2 paper-prep purposes, the right narrative is:
> "Using the validity-gated comparison framework, we attempted to verify
> that MCMC and enumeration produce compatible posteriors. On a 7-rule
> reduced-scale test (depth=6 / 100k programs / MCMC 10k×4 chains),
> ALL 7 rules failed the pre-registered validity filter due to MCMC
> top-K mass coverage falling between 13% and 52%. We therefore make
> NO claim about cross-method compatibility from this run. We document
> three concrete remediations [above]."

## Validation that the framework is doing its job

This is a SUCCESS for the framework — exactly the failure mode it was
designed to catch. Without `validity_flags`, we would have reported
mean|Δ|=0.07 on `all_red` as "good agreement" and silently buried the
fact that we were extrapolating from 52% of MCMC mass. Codex's hard
gating (Round 4) makes this impossible.
