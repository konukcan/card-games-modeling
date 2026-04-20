# Night 2 Round 2 — MCMC comparison framework, Codex Round 4 review

- **Reviewer thread**: 019da9dd-efac-75b0-934b-5b4c0805c077 (continued)
- **Subject**: `review-stage/experiments/night2/compare_enum_vs_mcmc.py`
- **Score**: 9/10
- **Verdict**: **ACCEPT** (with one optional add-on)

## Disposition

Hard gating implemented cleanly:

- `validity_flags: {...}` (5 booleans).
- `comparison_valid = not any(validity_flags.values())`.
- `VALIDITY_THRESHOLDS` dict declares the gating constants explicitly
  for paper appendix (pre-registered).

Codex finds no remaining fatal methodological-honesty issue within the
stated scope.

## One optional add-on (post-ACCEPT)

- Add `enum_truncation_excessive` flag using `enum_retained_mass`.
  Currently `comparison_valid` can be True even if enumeration retained
  mass is low (the enum-side analogue of `mass_in_top_k`).

**Status**: APPLIED post-review, before commit. New flag wired through
`_build_validity_flags` and `compare_one_rule(..., enum_retained_mass=...)`.
Threshold 0.95 added to `VALIDITY_THRESHOLDS["min_enum_retained_mass"]`.

## Threshold defensibility (Codex's read)

Defaults (0.90 / 0.80 / 0.05 / 0.95 / 0.001 / 0) are defensible for a
paper IF:

- Per-rule audit table is published.
- Thresholds are described as "pre-registered validity filters",
  not tuned post-hoc.

Codex would NOT tighten across the board. Only `max_predicate_exceptions=0`
might be loosened to a *rate* (not count) for very large runs, but the
strict-default recommendation stands.

## Pre-scaling recommendations (deferred)

- Multiple probe seeds, mean±sd of mean|Δ| and ρ.
- Bootstrap-CI on `mean_abs_diff` near hit-rate boundary.

## Round 1→4 trajectory

| Round | Score | Verdict | Key change |
|-------|-------|---------|------------|
| 1 | 6/10 | needs work | 7 findings (silent failures + scope claims) |
| 2 | 8/10 | almost | 2 nits (mass_in_top_k semantics + dead fallback) |
| 3 | 9/10 | almost | 1 condition (need HARD gating, not docstring) |
| 4 | 9/10 | **accept** | 5-flag validity gating; +1 optional add-on (enum-side) |

Three-turn improvement: 6 → 8 → 9 → 9 (with verdict promotion to accept).
