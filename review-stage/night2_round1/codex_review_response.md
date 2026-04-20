# Night 2 Round 1 — Codex review response

- **Reviewer thread**: 019da9c1-da9f-78e1-9c7b-6410ddac7b06 (gpt-5.1)
- **Received**: 2026-04-20T07:22Z
- **Score**: 5/10
- **Verdict**: `needs work`

## Findings (verbatim, condensed)

1. **BALD claim is exact in prose, approximate in code (R1 pattern repeats).**
   `adversarial_hands.py:19-22` and `:154-156` claim binary predictive entropy
   equals mutual information. But `compute_posteriors_for_rule`
   (`hand_diagnosticity.py:138-155`, `:222-229`) prunes by mass threshold and
   renormalizes — semantics become "conditional on survival", with
   `1 - retained_mass` bounding predictive error. The driver records
   `retained_mass` (`run_adversarial_hands.py:181`) but the search/ranking code
   doesn't propagate it into outputs or warnings.

2. **Degenerate posteriors fail open.** `_evaluate_posterior_predictive`
   (`adversarial_hands.py:90-111`) returns `p_accept = 0.0` for empty
   posteriors. Both search functions then proceed normally — emitting arbitrary
   "most diagnostic" hands at H=0 and synthesizing FNs whenever the truth
   accepts. The driver does not guard against this.

3. **`splitting_hypotheses` doesn't find splitters.** Both search paths claim
   to surface high-mass disagreers (`:184-200`, `:289-303`), but actually sort
   ALL decisions by mass and take the top — i.e., they mostly surface majority
   voters. Interpretability story wrong-in-implementation.

4. **Diversity filter is exact-multiset dedup, not "structural".**
   `_hand_signature` (`:76-83`) uses sorted `(rank, suit)` multiset, so it only
   removes order variants / exact re-draws. Module doc claims "structural
   variety" — overclaim. Tests mirror the weakness (`test_adversarial_hands.py:39-57`,
   `:116-132`).

5. **Pure uniform Monte Carlo, no targeting.** Both functions sample uniformly
   and rank within sample only; no stratification, adaptive search, or
   guarantee of seeing rare divisive hands. Tests cover happy path only.

## Prioritized recommendations

1. Thread `retained_mass` into adversarial outputs; refuse / warn when below
   floor; label scores as conditional-on-survival unless mass ≈ 1.
2. Fail closed on empty/unusable posteriors.
3. Fix `splitting_hypotheses` to actually report disagreement contributors;
   add a test that fails if majority dominates.
4. Rename current diversity to "exact_dedup" OR implement coarser
   canonicalization; report pre/post counts.
5. Stop calling output "worst-case" without targeted search; per-rule seed
   derivation; add baseline.
6. Tests: empty posterior, low retained mass, rare-rule miss, splitter
   semantics, tie at p=0.5, τ sensitivity.

## Note

Codex could not run `pytest` in sandbox (no writable temp dir) — review is
static + small inline probes only. Tests are still 9/9 pass on our end.
