# Night 2 Round 2 — MCMC comparison framework, Codex Round 3 review

- **Reviewer thread**: 019da9dd-efac-75b0-934b-5b4c0805c077 (continued)
- **Subject**: `review-stage/experiments/night2/compare_enum_vs_mcmc.py`
- **Score**: 9/10
- **Verdict**: almost

## Findings

No new fatal flaws. Round 2 nits both addressed:

- `mass_in_full_list` vs `mass_in_top_k` semantics now consistent in
  docstring (lines 40–51) and in `_reconstruct_mcmc_predicates`
  (lines 164–183, 204–216).
- `frequency_ranking` fallback removed from `compare_one_rule`
  (lines 296–303); loader is the single source of truth (lines
  383–393).

## One residual condition for full ACCEPT

- Codex wants HARD gating, not just docstring guidance, before
  ACCEPT: a `comparison_valid` boolean OR an explicit warning/error
  that flips false when `mass_in_full_list`,
  `mass_in_top_k`, `mass_dropped_parse`, `probe_hit_rate_*`, or
  predicate-exception counts cross thresholds.

## Rare-rule scope

Scoping conclusions to "rules with non-trivial probe_hit_rate under
uniform random 6-card hands" is acceptable for this round if:

- Headline explicitly names the evaluation distribution.
- Number of rules excluded by hit-rate filter is reported.
- No implicit claims about rare-rule compatibility.
