# Reviewer Memory

Persistent memory across rounds for the external GPT-5.4 reviewer (Codex MCP, `xhigh`).

---

## Round 1 — Score: 4.5/10, Verdict: not ready

**Thread ID:** `019d9579-b4c2-7b23-97c2-8a2316c6d104`

### Suspicions raised
- The current proposal density `q` is still wrong until the proposal generator, scorer, and latent type-resolution story are literally the same mathematical object. Multiple operational filters (retry loop in `sample_program`; pre-MH vacuous-lambda rejection) live outside the stated posterior / proposal density and silently truncate state space.

### Unresolved concerns carried forward
- `max_nodes` and other hard support truncations may also be off-book unless folded into the target explicitly.
- Independent-prior calibration may still fail even after tier-2 removal unless `sample_program` retry mismatch is fixed first.
- `collect_subtree_sites` swallows inference failures silently (lines ~953–962, 1013–1028); `n_sites` may drift undetected.

### Patterns the reviewer flagged
- Every major soundness failure so far comes from operational filters or retries that live outside the stated posterior / proposal density.

### Confirmed concrete counterexamples (reviewer ran code)
- Tier-2 scorer inversion: toy grammar `{choose: ('a->bool)->out, is_zero: int->bool}`, `_CONCRETE_TYPES=[BOOL,INT]`. Scorer returns `P=1.0` for `choose ((λ is_zero $0))`; raw `_sample` emits it 992/2000 ≈ 0.496. Tier-3 also fails — it picks one sentinel assignment, not a marginal.
- `_all_args_terminable` branch entry rate: 23.5% in gallery `HAND→BOOL` at `max_depth=5` (1955/8318 calls). Claim that it is "rare" is false.
- `sample_program` retry rate: 2.8% on 500 gallery root samples (14/500 retried). Toy distribution shifted: raw `_sample` `{992,471,537 ERR}` → `sample_program` `{1338,662}`.

### Night 2 Round 1 action list sent to Claude
1. Align proposal generator with scored density (remove retry or score exact retry-conditioned law).
2. Exact marginalization over env-unresolved type draws (enumerate `_CONCRETE_TYPES^k`, log-sum-exp).
3. Mirror depth-cap `_all_args_terminable` in the scorer.
4. Remove pre-MH vacuous-lambda rejection (encode as −∞ in target or defer to reporting).
5. Re-run calibration with independent prior, 5 seeds, IAT/Geyer ESS, polymorphic grammar `{not,and,or,eq,if}` at `INT→BOOL`.
6. Add proposal-normalization / site-collection regression tests (`ΣQ(s'|s)≈1`, zero silent site-drop failures).

### Reviewer memory update for future rounds (verbatim from GPT-5.4)
> Suspicion: the current q is still wrong until the proposal generator, scorer, and latent type-resolution story are literally the same object.
> Unresolved: `max_nodes` and other hard support truncations may also be off-book unless folded into the target explicitly.
> Patterns: every major soundness failure so far comes from operational filters or retries that live outside the stated posterior.
> Unresolved: an independent-prior calibration may fail even after tier-2 removal unless the `sample_program` retry mismatch is fixed first.
