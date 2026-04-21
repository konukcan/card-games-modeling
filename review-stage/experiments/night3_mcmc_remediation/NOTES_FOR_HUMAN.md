# Notes for the Human — Night 3 Run

Quick "things you should know before reading the report" list. Short on purpose.

## Budget deviations from the design doc

| param | design doc | actual | reason |
|---|---:|---:|---|
| MCMC steps per chain | 50,000 | 20,000 | 50k × 4 × 18 on 10 workers was projected at 20 h; I cut to 20k to finish overnight. This is the biggest methodological deviation and probably explains some fraction of the convergence failures. |
| MCMC extended probes | 10,000 | 2,000 | Same wall-clock reason. Affects only the fine-grained comparison, not validity. |
| Everything else | as specified | as specified | Uniform grammar, 500 standard probes, depth 6, 1000 holdout probes, seed 42 base, validity thresholds unchanged. |

If you want the full-fat 50k × 10k-probe run, the config is a one-line edit in `config.json` and another ~12 h of compute.

## The bug you should know about

See "Critical methodological discovery" in the morning report. Short version: `pool.pkl` stores split sub-classes under **composed** fingerprints (`sha256(parent|child)`), not the raw fingerprints you get by running a predicate on the 500 standard probes. If you or anyone else ever writes code that consumes `pool.pkl` and does fingerprint-based lookups, **you need the 2-stage lookup from `run_comparison.py:_aggregate_mcmc_to_classes`** or you'll get a silent zero-match bug.

I almost missed this; the only reason I caught it was the hard validity gate producing `mass_mapped ≈ 0.01`, which was implausible enough to investigate rather than write up.

Commit `e2cd8d7` is the fix.

## What "comparison_valid" actually means in summary.json

It means `mass_mapped ≥ 0.90`, i.e., 90 % of MCMC's posterior lands on classes enum also enumerates. It does **not** mean the posteriors agree. For agreement you want TV < 0.2 AND valid. Only 3 rules meet that stricter bar.

## What I didn't do

- I did not rerun enumeration. The existing 300k/depth-6 pool from the prior session was used as-is. Sanity check: `enum_retained_mass ≥ 0.97` for every rule, so truncation is not the problem.
- I did not cross-check against Night 2's MCMC results. Different grammar (with injections) and different step budget, so probably not comparable.
- I did not write a "question B" analysis. The `comparison/question_b/` directory is empty; I kept the answering scope to question A + convergence. Question B was going to be about enum-only-classes that MCMC never visits, which becomes much more interpretable after you decide whether to extend MCMC to 50k steps.

## Files that are safe to delete

- `mcmc_50k_4chains/mcmc_summary.json` is a leftover from an earlier 500-step smoke test that I forgot to clean up. The real 20k results live in `raw_visits/` and `checkpoints/`, not the stale summary. Probably worth deleting next time you're in that directory.
- `experiments/night2/timing_test/` is untracked and leftover from a wall-clock test. Not mine, not modified; decide at your leisure.

## Things to sanity-check before you cite any of this

1. The `_member_fp` function in the fixed aggregator uses seed 9999 and n=1000 holdout probes, matching `analyze.py:381`. If that source ever drifts, the aggregator needs to drift with it.
2. The `spearman_rank_corr` in `question_a/*.json` is computed over the union of classes. For rules where enum has 1 class and MCMC has 17 (like `all_same_suit`), the correlation is not meaningful; treat TV and JS as the authoritative distance metrics.
3. `mass_parse_fail` is 0.0 across every rule. That is correct — the sampler only accepts programs that parse. Not a bug.

## Headline to remember

**With the fingerprint fix, MCMC and enumeration agree tightly on 3 of 18 rules, partially agree on 3 more, and diverge on the remaining 12. The primary failure mode is 10–33 % mass leaking to classes enum never enumerates — not Python errors, not parse failures, but legitimately-typed programs the enumeration pruner didn't produce.**
