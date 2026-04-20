# Findings Log (COMPACT mode)

One-line per key finding, per round. See `AUTO_REVIEW.md` for full context.

- [Round 1] negative: 7 critical bugs in Bayesian pipeline — fingerprint collisions, weighted-enum invalid, nan priors, stale extension cache, silent renorm, rounding cancellation, mislabelled tempering (score 2/10).
- [Round 1] positive: all 7 R1 findings implementable and closed same round — member-exact scoring, true weighted enumeration, strict fail-closed priors, probe-hash wiring, retained-mass reporting, base_rate direct, tempering relabel.
- [Round 2] negative: fixes to `run_analysis()` did NOT propagate to sidecars (`run_diagnosticity.py`, `depth_mass_analysis.py`, `compare_prior_modes.py`); injection re-contaminates strict-split classes; depth-cap and spectrum JSON fields drift from semantics (score 4/10).
- [Round 2] positive: factored `merge_injections_and_extend` shared helper; post-injection `_strict_split_classes` re-run; depth-cap debate overruled in favour of relabel + JSON semantics fields.
- [Round 3] negative: labelling / nomenclature drift — ast_depth measures paren-nesting not Program.depth(); balanced_n reports target not actual; compare_prior_modes narrative wrong (score 5/10).
- [Round 3] positive: unified prior computation across main scorer and diagnosticity scorer; residual collision rate reduced from 1.7% (R1) to ≤0.2% (R3) on 2k unseen hands at full scale.
- [Round 4] positive: POSITIVE_THRESHOLD met — Codex scores 6/10, "almost", no new concrete blocker beyond full-scale residual-sensitivity audit (metric: score 2 → 4 → 5 → 6).
- [Round 4] positive: residual mixed-class sensitivity bound (reduced scale, 5 rules): max |Δrank|=2 on 1e-22-posterior rule, max |Δprob|=1.7e-10, max mixed-class mass=7.7e-5, 0/5 top-1 flips.
