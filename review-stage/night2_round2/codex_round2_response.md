# Night 2 Round 2 — MCMC comparison framework, Codex Round 2 review

- **Reviewer thread**: 019da9dd-efac-75b0-934b-5b4c0805c077 (continued)
- **Subject**: `review-stage/experiments/night2/compare_enum_vs_mcmc.py`
- **Score**: 8/10 (up from 6)
- **Verdict**: almost

## Findings (2 residual nits)

1. **`mass_in_top_k` mislabelled "BEFORE truncation"** — it was actually
   computed over the post-truncation `mcmc_top = ...[:top_k]` slice. Either
   reword the docstring to "before parse-drop, after truncation", OR
   compute a true pre-truncation `mass_in_full_list`. Codex preferred the
   second since the MCMC payload itself can be partial.

2. **`frequency_ranking` fallback in `compare_one_rule`** is now dead code
   because the loader already validates `top_hypotheses`. Either delete
   it (clean), or extend the validator to accept `frequency_ranking`
   under `--allow-legacy-schema` (consistent).

## Methodological notes (not nits)

- **Headline phrasing**: agreement on this probe distribution does NOT
  imply agreement on rare submanifolds or other target distributions.
  Codex would prefer explicit "claim gating" in the writeup like:
  "only report compatibility when `mass_used` is high AND
  `mass_dropped_parse` is low AND `mass_in_top_k` is near 1".

- **Rare-rule blind spot**: `probe_hit_rate` reporting alone is enough
  *only* if conclusions are scoped to high-hit-rate rules and refuse
  to interpret low-|Δ| agreement when both hit-rates are ~0.
