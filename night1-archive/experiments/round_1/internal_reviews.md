# Round 1 Internal Reviews (Round 1 fixes: C2, C3, C4, C6)

**Commits reviewed:** d1eabe8, 35cfef4, 83e5e93
**Diff:** `review-stage/experiments/round_1/round1_fixes.diff` (419 lines)

## Kieran Python Reviewer (agent acf700d0f1f0d147f)

**Critical:** None.
**High:**
1. **Stale docstring** in `run_parallel_chains` (lines 1449-1451) — claims first_passage is offset by `step + i * config.n_steps`. Contradicts new implementation. *Action: fix docstring.*
2. `total_steps = n_chains * config.n_steps` reporting at line 1575 — correct but semantically asymmetric with new `first_passage` timeline; worth a one-liner comment.
3. Type-hint drift: `List[Dict[str, int]]` fine given file already imports typing generics. No action.
4. `ext_fraction_accum` division guarded by construction; add assertion for defensive future-proofing. Low priority.

**Verdict:** No critical Python issues. Detailed balance, resource handling, type safety intact.

## Performance Oracle (agent aec19ae2e9f88ccd0)

**Critical:** None.
**High:**
1. `merged_trajectory.extend` pre-existing O(Σ steps) — C3 slightly enlarges by no longer short-circuiting tautology steps. Not a Round-1 regression, flagging as informational.
2. C3 post-hoc filter O(U log U) — same complexity as prior sort. No regression.
3. C6 first-passage merge — strictly cheaper. No regression.
4. C2 `dc_replace` — one-shot per chain. No regression.
5. C4 Jeffreys smoothing — adds ~3 float ops per likelihood call, dominated by 10K-probe eval loop. No measurable regression.
6. `per_chain_first_passage` copy — bounded by unique programs per chain. No regression.

**Verdict:** Clean. No performance regressions.

## Code Simplicity Reviewer (agent a130eabf0a0025622)

**Critical:** None.
**High:**
1. **Redundant second pass** for `best_program` tautology exclusion (lines 1568-1573) duplicates the filter already applied to `sorted_programs`. *Action: derive best_program from top_hypotheses[0] or sorted_programs[0].*
2. `ext_fraction_accum` tuple accumulator is a two-step dance; defaultdict(list) or in-place sum+count cleaner. Minor.

**Verdict:** Total LOC reduction potential ~3-5% of diff (trivial). Proceed with optional polish.

## Actions taken

- [x] Fix stale docstring in `run_parallel_chains` (Kieran H1).
- [x] Collapse redundant best_program tautology filter (Simplicity H1).
- [ ] Defer Simplicity H2 (ext_fraction_accum refactor) — cosmetic, not worth risk to passing tests.
- [ ] Defer Kieran H2 (total_steps comment) — low value.
