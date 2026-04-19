# Night 3 Findings (compact log)

One-line per key finding per round, appended as we go.

---

- [R1] negative: W1 new blocker — `collect_subtree_sites` `id(node)`-keyed cache collides on Primitive singletons; ~5.7% bad site metadata → 50% ill-typed MH proposals. Score 8.0 → 5.0.
- [R1] positive: W1 path-keyed fix + single-pass `_annotate` mirroring `infer_type` drops bad-site rate to 0% (30/30 proposals type-check).
- [R1] positive: W2 three separate approximation-cap counters wired + exposed + reset; regression tests trigger each branch and confirm no cross-contamination.
- [R1] positive: W4 lambda/bound-variable full-kernel ΣQ=1 test passes on reviewer's exact spec (3 sites × 8-program support on `λ f(f($0))`).
- [R1] unexpected: the original `_annotate` had a latent bug independent of `id()`-keying: redundant `infer_type` calls per recursion level created stale fresh TVs that `ctx.apply` could never resolve. Only visible at the full-proposal level (50% proposal type-fail pre-fix).
- [R2] positive: W1 site-metadata corruption fixed via path-keyed cache + single-pass _annotate (end-to-end: 10/20 → 0/30 ill-typed proposals).
- [R2] positive: W4 bound-variable full-kernel ΣQ=1 reproduced at {1/3, 1/3, 1/9, ...}.
- [R2] unexpected: retry path in sample_program(allow_retries=True) admits root-mistyped programs at 1/100 seeds (F-R2-1).
- [R3] positive: R2-Fix1 root-type unify clean; R2-Fix4 binary grammar closure clean at depth ∈ {1,2,3} + N=20000 MC.
- [R3] negative: _score_depth_cap_lookahead_exact branch uncovered (F1); experiment script ignores approximation-cap counters (F2); prose/assert mismatch in typability test (F3).
- [R4] positive: all three F1/F2/F3 OVERRULED by reviewer via direct python -c probes. No new regressions introduced.
- [R4] terminal: Night 3 loop terminated READY at 9.0/10. Remaining items are claim-scoping + throughput, not kernel soundness.
