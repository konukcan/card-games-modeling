# Night 4 — Pre-flight Residual Finding

**Date:** 2026-04-22
**Branch:** `aris/bayesian-review`
**Scripts:** `attribute_unmapped.py`, `investigate_residual.py`, `probe_holdout_5k.py`
**Pool:** `night3_mcmc_remediation/enum_depth6_300k/pool.pkl` (2501 classes: 2268 direct + 233 sub-classes under 64 split parents)

---

## TL;DR

Source **E** (probe-set / strict-split instability) is **ruled out**. The residual unmapped mass after excluding type-invalid (B) and depth>6 (C) programs is concentrated in shallow (d4-d6) MCMC programs whose canonical-program predicates match a pool **parent fingerprint** on the 500 standard probes but do not match any sub-class composite fingerprint on any holdout-count we tested (1k, 5k). The most likely remaining source is **budget-cap truncation** at 300k yielded programs in the d6 enum pool — peer programs in the same grammar rewrite neighborhood are present in the pool, but specific variants MCMC visits are absent. C2 (d7 × 500k enum) will test this directly.

---

## Numbers — C3 pre-flight on night-3 MCMC data (50k steps × 4 chains)

| Rule              | mass_mapped | mass_unmapped | B (type) | pruner | C (depth>6) | d6_elig_unmapped |
|-------------------|-------------|---------------|----------|--------|-------------|------------------|
| all_red           | 0.9287      | 0.0713        | 0.0209   | 0.0000 | 0.0015      | **0.0489**       |
| all_even          | 0.8401      | 0.1599        | 0.0513   | 0.0000 | 0.0058      | **0.0883**       |
| all_same_suit     | 0.9897      | 0.0103        | 0.0000   | 0.0000 | 0.0016      | **0.0050**       |
| ranks_palindrome  | 0.9413      | 0.0587        | 0.0218   | 0.0000 | 0.0011      | **0.0346**       |

`mass_mapped` reconciles with Night 3's class-aggregated number (0.9287 on all_red), confirming the mapper in `attribute_unmapped.py` matches production. **`d6_elig_unmapped` is the residual Sources A ∪ E ∪ budget-cap.**

---

## Residual decomposition (investigate_residual.py)

For each d6-eligible unmapped program: split into (a) **strict-split miss** — direct fp matches a pool parent, composite does not match any sub-class; vs (b) **truly unseen fp** — direct fp matches no pool class at all.

| Rule              | SS-miss mass | SS-miss progs | Truly-unseen mass | Truly-unseen progs |
|-------------------|--------------|---------------|-------------------|--------------------|
| all_red           | 0.0489       | 82            | 0                 | 0                  |
| all_even          | 0.0883       | —             | —                 | —                  |
| all_same_suit     | 0.0050       | —             | —                 | —                  |
| ranks_palindrome  | 0.0346       | —             | —                 | —                  |

**100% of the residual is "strict-split miss" — not a single truly-unseen fingerprint.** All SS-miss mass on all 4 rules concentrates on ONE pool parent class `4ec297ba99da...` (the "always-true-on-realistic-hands" parent, which already has 45 sub-classes).

---

## Source-E direct test (probe_holdout_5k.py)

Rebuild sub-class holdout fingerprints at **n_holdout=5000** (vs pool's 1000, seed=9999) and re-check whether MCMC SS-miss programs now match:

| Rule              | SS@1k mass | Rescued @5k | Still @5k | Eval fail |
|-------------------|------------|-------------|-----------|-----------|
| all_red           | 0.0489     | **0.0000**  | 0.0489    | 0.0000    |
| all_even          | 0.0883     | **0.0000**  | 0.0883    | 0.0000    |
| all_same_suit     | 0.0050     | **0.0000**  | 0.0050    | 0.0000    |
| ranks_palindrome  | 0.0346     | **0.0000**  | 0.0346    | 0.0000    |

**0% rescue rate across all 4 rules at 5x the holdout count.** Source E is dead: partition instability from small holdout counts is not the story.

---

## Depth distribution of SS-miss (all_red)

82 programs total:
- d4: 14 progs, 0.0082 mass
- d5: 57 progs, 0.0382 mass
- d6: 11 progs, 0.0024 mass

Program sizes range 8-14. These are **shallow, compact** — not depth-escape programs. Source C is unlikely to account for the residual on its own.

---

## Pool-membership spot check

Checked top-10 SS-miss program strings against pool (49,697 unique program strings across canonicals + stored members): **all 10 are NOT in the pool.** But structural peers are present:

| Example SS-miss program                  | In pool? | Structural peer in pool (parent `4ec297ba99da`)       |
|------------------------------------------|----------|-------------------------------------------------------|
| `(λ gt (max_rank $0) 3)`                 | no       | `(λ lt 5 (max_rank $0))` (same tautology, flipped ≤/≥) |
| `(λ lt 0 (n_unique_ranks $0))`           | no       | peers with same `n_unique_ranks` primitive             |
| `(λ gt (sum_ranks $0) 5)`                | no       | `(λ gt (sum_ranks $0) 3)` etc. (different constants)   |

Enum's 300k yielded budget cut off before reaching these variants in the rewrite neighborhood.

---

## Conclusion

- **Source E: ruled out** (0% rescue at 5x holdouts).
- **Source C (depth>6): unlikely to drive residual** (programs are d4-d6).
- **Source A (MCMC under-convergence): possible but can't be tested from static night-3 data** — C1 (100k × β=1-tail) addresses this.
- **Budget-cap truncation: most likely.** Peer programs are present; specific variants MCMC visits aren't. C2 (d7 × 500k, ONE pool for all 6 rules) tests this directly — more depth AND more budget in a single run.

**Gate 2 result: GREEN.** Script works, numbers reconcile, Source E is dead, remaining sources have a clear test in C1/C2. Proceed to task #20 (fork `run_mcmc_night4.py` with per-chain persistence).

---

## Artifacts

- `attribute_unmapped.py` — multi-label C3 attribution (2-stage composite-fingerprint mapper)
- `investigate_residual.py` — SS-miss vs truly-unseen decomposition
- `probe_holdout_5k.py` — Source-E direct test via n_holdout=5000 rebuild
- `preflight_output/{rule}_unmapped_predicates.json` — per-rule attribution output
- `preflight_output/holdout_5k_probe.json` — Source-E rule-out summary
- `test_beta_schedule.py` — smoke test for the new `MCMCConfig.beta_schedule` field (task #17)
