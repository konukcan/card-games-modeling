# Night 2 Round 1 — Codex review prompt (draft, awaiting full-scale results)

## Branch state
- Branch: `aris/bayesian-review` @ commit `dc6b449`
- Test baseline: 92/93 pass (1 pre-existing: `test_merge_updates_summed_prior`)
- Night 1 closed at score 6/10 ("almost") with one remaining recommendation:
  full-scale residual mixed-class sensitivity audit at depth=7 / 300k / all 60 rules.
  That audit is currently launching in background (PID file at
  `review-stage/experiments/night2/full_sensitivity_audit.pid`).

## What's new (Round 1, Night 2)
A new module + driver for **adversarial hand generation**: given a posterior
P(rule | exemplars), find hands that are maximally diagnostic (BALD-style
posterior predictive entropy) and hands where the learner is *confidently
wrong* about the true rule (false positives / false negatives). Goal:
quantitatively expose the worst-case learner mistakes the official scoring
hides behind a single per-rule top-1 probability.

## Files for review
1. `src/gallery_analysis/adversarial_hands.py` — core module (~290 lines).
   - `find_most_diagnostic_hands(posteriors, equiv_classes, n_candidates,
     top_k, seed, diversity, n_top_splitters, ground_truth_pred)` — sample
     `n_candidates` random 6-card hands, compute posterior predictive
     `p_accept = Σ_i posterior(h_i) · I[h_i(h)]`, score by binary entropy
     `H(p) = -p log p - (1-p) log(1-p)`, optionally dedup by hand signature,
     return top-k by descending entropy.
   - `find_most_adversarial_hands(...)` — same sampling, but partition into
     false_positives (`p_accept ≥ τ` AND `not rule(h)`) and
     false_negatives (`p_accept ≤ 1-τ` AND `rule(h)`); rank by confidence-of-error.
   - `_dedup_by_signature` — greedy keep-first by sorted (rank, suit) tuple.
   - Splitting-hypotheses annotation: top-`n_top_splitters` mass classes attached
     for interpretability.
2. `src/tests/test_adversarial_hands.py` — 9 tests, all pass in 0.15s.
   - Binary entropy extremes / symmetry / monotonicity.
   - Hand signature canonicalization + dedup-keeps-first.
   - Toy posterior smoke tests: top-k ordering, top hand has p_accept ≈ 0.5.
   - Diversity filter actually filters.
   - End-to-end FP/FN detection on a contrived disagreement.
   - JSON serializability.
3. `review-stage/experiments/night2/run_adversarial_hands.py` — driver for 5
   representative rules: `all_red`, `all_same_suit`, `all_even`,
   `triple_2s_pos234`, `all_but_one_same_color`. Outputs
   `adversarial_hands_results.json`. (Results pending full-scale run.)

## Specific design decisions to attack
1. **Entropy proxy vs full EIG.** I use binary entropy of the posterior
   predictive (BALD reduction for binary labels). Is this the right
   diagnosticity score, or do I need full Mutual Information
   `I(h; rule | D) = H(rule | D) - E_h[H(rule | D, h)]` over the *full*
   posterior over rules? Argument for BALD: each hypothesis deterministically
   labels the hand, so the binary entropy of the marginal predictive is the
   correct mutual information up to a constant — but my formulation may be
   subtly off when the posterior has many low-mass survivors that all agree
   with the majority.
2. **Uniform random sampling.** I draw `n_candidates` random 6-card hands.
   For rules with very narrow extensions (e.g., `all_red`: 26C6 / 52C6 ≈ 1.6%),
   the most-divisive hands may be exponentially rare under uniform sampling.
   Should I be doing rejection / importance sampling stratified by some
   coarse hand signature, or by extension-overlap of hypothesis pairs?
3. **Diversity filter granularity.** My signature is the sorted (rank, suit)
   multiset, which collapses position-equivalent hands but keeps suit-distinct
   structurally-similar ones. Is this the right granularity? Should it be
   coarser (suit multiset only) or finer (position-aware)?
4. **τ = 0.8 confidence threshold.** Arbitrary. Justify or replace with a
   learned/calibrated threshold (e.g., from observed posterior calibration on
   held-out hands).
5. **Ground-truth dependence.** `find_most_adversarial_hands` requires the
   true rule predicate, which is fine for synthetic experiments but assumes
   we know the answer. Is there an oracle-free version (e.g., consensus of
   top-k posterior mass acting as proxy ground truth) that would be more
   defensible for the paper?
6. **Determinism + seed leakage.** Default seeds 12345 (diagnostic) and
   23456 (adversarial). Same seed across rules → same candidate hands → may
   bias if some hands happen to be diagnostic for many rules. Should each
   rule get its own seed derived from rule_id hash?

## What's NOT in this round (deferred)
- Full-scale results across all 60 rules. The driver hits 5 rules at depth=7
  / 300k as a minimum viable demonstration; scaling is straightforward but
  bounded by the night's compute budget.
- Comparison with naive random hands (baseline). Adding this trivially.
- Connection to the experimental UI (showing adversarial hands to human
  participants) — that's a downstream question, not in scope tonight.

## Score this round on (request from prior reviewer pattern)
- Mathematical correctness of the BALD reduction.
- Sampling efficiency / does this actually find the worst hands.
- Whether the diversity filter is doing useful work or hiding interesting
  duplication.
- Failure modes on degenerate posteriors (all mass on one class; mass on
  many disagreeing classes; etc.).
- Whether the test coverage actually exercises the failure modes you'd care
  about.

## Reviewer instruction
Be harsh. Round 4 closed at "almost" with one outstanding gap; this is
new work and we have not yet checked whether it has its own latent
unsoundness pattern (the R1 pattern: claimed mathematical equivalences
that are only approximate-in-implementation). Look hard for that.
