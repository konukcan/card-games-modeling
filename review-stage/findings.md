# Night 3 Findings (compact log)

One-line per key finding per round, appended as we go.

---

- [R1] negative: W1 new blocker — `collect_subtree_sites` `id(node)`-keyed cache collides on Primitive singletons; ~5.7% bad site metadata → 50% ill-typed MH proposals. Score 8.0 → 5.0.
- [R1] positive: W1 path-keyed fix + single-pass `_annotate` mirroring `infer_type` drops bad-site rate to 0% (30/30 proposals type-check).
- [R1] positive: W2 three separate approximation-cap counters wired + exposed + reset; regression tests trigger each branch and confirm no cross-contamination.
- [R1] positive: W4 lambda/bound-variable full-kernel ΣQ=1 test passes on reviewer's exact spec (3 sites × 8-program support on `λ f(f($0))`).
- [R1] unexpected: the original `_annotate` had a latent bug independent of `id()`-keying: redundant `infer_type` calls per recursion level created stale fresh TVs that `ctx.apply` could never resolve. Only visible at the full-proposal level (50% proposal type-fail pre-fix).
