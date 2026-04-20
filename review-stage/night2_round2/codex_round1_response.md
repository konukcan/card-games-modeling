# Night 2 Round 2 — MCMC comparison framework, Codex Round 1 review

- **Reviewer thread**: 019da9dd-efac-75b0-934b-5b4c0805c077
- **Subject**: `review-stage/experiments/night2/compare_enum_vs_mcmc.py`
- **Score**: 6/10
- **Verdict**: needs work

## Findings (7)

1. **"Top-K coverage" promised in docstring but not implemented.** Headline
   claim of the comparison was unsupported by the metrics emitted.

2. **Predicate-eval exceptions silently swallowed.** Both
   `_enum_p_accept_per_hand` and `_mcmc_p_accept_per_hand` returned just
   the p_accept list; an exception during predicate eval was caught with
   bare `except` and contributed `False` to the count, masking
   contamination of `mean_abs_diff`.

3. **Top-K parse-failure mass not tracked.** When a top-K MCMC entry
   failed to parse, weight was silently dropped and remaining mass
   renormalized — making the comparison *look* better than it was. No
   accounting field was emitted.

4. **Probe-set blind spot for rare rules.** Uniform random 6-card hands
   essentially never satisfy "all 4s and queens"-class rules; both
   methods report ~0 p_accept on every probe; the resulting `mean_abs_diff`
   is ~0 and meaningless. No hit-rate diagnostic was emitted.

5. **Schema duck-typed silently.** Driver tolerated both
   `{rules: {...}}` (analyze_mcmc.py format) and flat `{rid: {...}}`
   (legacy) without flagging which path was taken; missing
   `top_hypotheses` would just produce empty results, no error.

6. **Field-name fallback masked schema drift.** `top_hypotheses or
   frequency_ranking` quietly fell through if neither was present.

7. **Calibration vs ρ confusion.** Spearman ρ was reported as a primary
   metric, but ρ measures monotonicity, not calibration; calibration is
   what the headline question asks about and must be read off
   mean/max abs diff.
