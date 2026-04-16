# Round 2 Review — MCMC Bayesian Rule Induction

**Reviewer:** Claude general-purpose sub-agent (Codex MCP unavailable — fallback)
**Agent ID:** a97cf5cd84ae6c9b9
**Received:** 2026-04-16T08:15:00Z (approx)
**Verdict:** major revision
**Score:** 5.5/10 (up from 4/10)

## Assessment of Round 1 fixes

- **C2 (β-annealing propagation)** — adequate. Nit: test should iterate `dataclasses.fields()` rather than hand-pick β fields to catch future drops.
- **C3 (post-hoc tautology filter)** — adequate with caveat. `test_tautology_filter_is_post_hoc` is a smoke test — doesn't prove filter fired. Strengthen by asserting known tautology in `visit_counts` but not in `top_hypotheses`.
- **C4 (Jeffreys + one-probe floor)** — adequate. Test covers empty-extension vs tautology but doesn't hit the "accepts exemplars via noise, nothing on probes" corner case. Consider augmenting.
- **C6 (first-passage merge)** — adequate. Docstring update flags pooled-counter semantics correctly.

## Round 2 critical/high

### C1 (STILL UNFIXED) — sampler posterior-incorrect
Single pooled softmax over productions ∪ variables in `_sample` vs type-indexed-normalized in `program_log_likelihood`. Different distributions whenever >1 variable of target type in scope. Hastings ratio wrong. Stationary distribution ≠ posterior. For CogSci "human rule induction approximates Bayesian posterior" claim this is not a nitpick — it's the central theoretical commitment.

### New minor
- `init_max_depth` retry loop: `current_log_prior` computed before recomputation of `current_log_lik` — fragile under future edits.
- `test_tautology_filter_is_post_hoc` is smoke test only.

## Priority for remaining work
**C1 is the single blocker for CogSci.** Fix approach: rewrite `_sample` to use type-indexed normalized distribution (matching `program_log_likelihood`) — preserves DreamCoder compatibility. Alternative: rewrite likelihood to match sampler. Option A preferred.

## Bottom line
Major revision. 4/6 critical fixes with honest regression tests = meaningful progress. Not yet accept. **Authors can proceed to Round 3 on C1 alone IF Round 3 includes a posterior-calibration experiment** (toy grammar with analytical posterior, verify empirical visit frequencies match within Monte-Carlo error). Without calibration, reviewers will remain skeptical C1's algebraic fix yields a well-behaved sampler in practice.
