# Reviewer Memory (persistent across rounds)

Reviewer: Codex GPT-5.4 (xhigh reasoning), acting as adversarial senior ML reviewer.
Thread: `019da777-8714-71e3-9784-5bea4a10bed1`.

---

## Round 1 — Score: 2/10 (not ready)

### Confirmed findings
1. **Fingerprint collision is real** — 14/813 groups disagree on curated exemplars in a 60k-yield prefix. Summed-prior + canonical-only scoring is already false on observed data.
2. **Weighted variants are not the weighted model** — 208,356 / 155,009 support asymmetry under actual branch settings (`max_depth=7, max_cost=35, max_programs=300000`). Post-hoc reweighting is invalid once search is truncated.
3. **`nan` in weighted summed-prior recomputation** — any class member with `-inf` log-prior (polymorphic-type failure or type-variable ambiguity) poisons the class prior.
4. **Probe hash not wired** — `run_analysis()` does not pass `_probe_hash` to extension cache; silent stale reuse on probe-set changes.
5. **Diagnosticity pruning renormalizes silently** — retained mass / error bound never reported.
6. **`TOTAL_HANDS` cancellation is not exact** — `int(base_rate * TOTAL_HANDS)` rounding breaks the claimed cancellation.
7. **Tempering mislabeled as Bayesian posterior** — `likelihood_exponent != 1` is a power posterior.

### New suspicions to track in future rounds
- Quantify how many full 300k weighted-summed classes hit `compute_log_prior == -inf` / `nan`.
- Check why programs that parse to depth `>7` can still appear from a `max_depth=7` enumerator (reviewer saw `(λ all ((λ $0)) (map ((λ at $1 (rank_val $0))) $0))` — suspected depth accounting bug).
- Rerun the collision audit under targeted probes (Config I) and larger probe sets; measure posterior movement after splitting colliding groups.
- Wire `_probe_hash` into extension caching and verify stale-cache failure with a changed probe set.
- Report retained posterior mass for diagnosticity pruning; otherwise those numbers are only conditional-on-survival approximations.

### Patterns to watch
- The pipeline has a consistent pattern of claiming mathematical equivalence (`summed_prior ≡ class-marginal`, `post-hoc rescore ≡ weighted enumeration`, `pruned posterior ≡ full posterior`, `int(base_rate*N) cancels`) that turns out to be approximate-at-best in the implementation. Future reviews should aggressively check for any place where "we claim X ≡ Y" is not actually proved or not tight in practice.
- Failure modes are silent: `nan` reductions, stale caches, renormalized truncations. Need explicit `fail-closed` semantics for official runs.

---

## Round 2 — Score: 4/10 (not ready)

### Previous suspicions addressed?
- **F1 (fingerprint collision)** — MATERIALLY IMPROVED for base pool. 0 residual disagreements on 2000 fresh random hands after strict split. BUT: injection can reintroduce mixed classes (`_strict_split` runs before `merge_injected`). 1/~813 residual on `true__` injections after merge.
- **F2 (weighted enumeration)** — Fixed in `run_analysis()`. NOT fixed in sidecars (`run_diagnosticity.py`, `depth_mass_analysis.py`, `compare_prior_modes.py`).
- **F3 (nan on -inf prior)** — Root-cause fixed. But `compute_posteriors_for_rule()` lacks strict option; full strict-prior validation run still pending.
- **F4 (probe hash)** — Fixed in `run_analysis()`. NOT fixed in sidecars.
- **F5 (retained mass)** — Mass computed and warned on, but `_spectrum_to_dict` drops the fields so JSON artefacts still hide it.
- **F6 (int() rounding)** — Fully addressed in main scoring.
- **F7 (tempering)** — analyze.py labels correctly. Diagnosticity outputs don't carry `posterior_kind` metadata.

### New Round 2 findings (6)
1. Injection-after-split recontamination — minimum fix: re-run `_strict_split_classes()` after `merge_injected()` and before extension estimation / true-rule FP / prior precompute.
2. Sidecar CLI drift — minimum fix: factor pool-building / injection-merging into a shared helper and use it everywhere.
3. Depth-cap bug — `Program.depth()` (codebase's own metric) gives depth=8 for depth-7-enumerated programs. Codebase truncates on "application-depth budget", not `Program.depth()`. Minimum fix: either enforce `Program.depth() <= max_depth` or relabel parameter as "enumeration budget" and drop the "depth-7 hypothesis space" paper claim.
4. `_spectrum_to_dict` drops retained/discarded mass — minimum fix: serialize both fields + print in human report.
5. `compute_posteriors_for_rule` lacks strict option — minimum fix: thread strict_priors through.
6. `depth_mass_analysis.py` / `compare_prior_modes.py` stale — fold into shared helper.

### Debate rulings (Round 2)
- **F3 (depth): OVERRULED.** Codex's definition is the codebase's own `Program.depth()`, not my tree-walker. Two acceptable resolutions: (a) enforce `Program.depth() <= max_depth`, or (b) relabel and drop paper claim.
- All other findings: accepted without debate.

### Patterns (extended)
- R1 pattern still holds. NEW pattern: fixes must propagate to ALL runners, not just `run_analysis()`. Sidecar tools drift silently.
- NEW pattern: post-split / post-merge / post-anything invariants need re-verification, not one-shot.

---

## Round 3 preparation (Claude-side notes, pre-Codex-call)

### All Round 2 findings implemented
- F1 R2 (injection-after-split recontamination): `_strict_split_classes()` now re-runs in both `run_analysis` and the new shared helper `merge_injections_and_extend` after `merge_injected`.
- F2 R2 (sidecar drift): factored injection+split+extensions into `merge_injections_and_extend`. `run_diagnosticity.py`, `depth_mass_analysis.py`, `compare_prior_modes.py` all now use this single helper — no duplicated inline code.
- F3 R2 (depth-cap): took OVERRULED option (b) — relabelled as "enumeration depth budget", updated docstrings in `enumerator.enumerate_hypotheses` and `analyze.build_hypothesis_pool`, updated CLI help, added `depth_budget_semantics` + `max_depth_budget` to JSON stats. Did not add a `Program.depth() <= max_depth` post-filter (would change pool composition; follow-up only if paper reintroduces a strict depth-7 hypothesis-space-size claim).
- F4 R2 (spectrum drops retained_mass): fixed in `_spectrum_to_dict` + human report.
- F5 R2 (compute_posteriors_for_rule strict option): `strict_priors` threaded through, wired into `run_diagnosticity.py`.
- F6 R2 (sidecars stale): see F2 R2 — fully absorbed into shared helper.

### Full-scale R3 collision audit (new evidence for Codex)
Fresh audit on 2,000 unseen random 6-card hands (seed=98765, disjoint from any probe/exemplar seed) at full pipeline scale:
- Pool build: depth=7 budget, max_programs=300,000, max_cost=35 (branch defaults).
- Step 3 (fingerprint) yielded 2,565 raw classes; strict-split collapsed/split to **2,501 classes** (64 disagreed → 233 sub-classes).
- **Pre-injection residual: 1/1,793 classes-with-≥2-members disagree on audit hands** (`(λ has_suit (drop 2 $0) SPADES)`).
- Merged 401 injected hypotheses → 260 merged into existing classes, 141 novel; total 2,642.
- Post-inject strict split: 12 disagreed → 40 sub-classes; total **2,670 classes**.
- **Post-injection residual: 3/1,858 classes-with-≥2-members disagree on audit hands**. Examples: `(λ has_suit (drop 2 $0) SPADES)` (carried through from base pool), `(λ eq (n_unique_ranks $0) 2)`, `(λ and (eq (rank_val (at $0 3)) (+ 5 (+ 5 1))) ...)`.

Collision rate now ≤ 0.2% at full scale. Compared to R1's 14/813 (1.7%) on a 60k-yield prefix this is a ~10× improvement. Residual is real but small; likely driven by structural correlation in the 500-probe set missing edge-case hand distributions for deep list-slicing predicates.

### Tests
- 184 pass / 1 pre-existing baseline failure (`tests/test_injection.py::test_merge_updates_summed_prior` — known summed-prior merge invariance edge case, noted at branch creation).

### Fresh suspicions to flag to Codex
- Residual collision is non-zero. Does the pipeline need a second pass of strict-splitting on larger audit hands, or is 0.04–0.11% acceptable residual noise for the size-principle scoring?
- The depth-relabel option (b) survives into JSON output (`depth_budget_semantics`), but the NARRATIVE_REPORT generator may still carry old "depth-7 hypothesis space" prose. Check.

---

## Round 4 — Score: 6/10 (almost) — FINAL

### Previous suspicions addressed?
- **F1 R3 (residual sensitivity)** — PARTIALLY CLOSED. Ran the requested experiment: at reduced scale (depth=6 / 100k / 5 rules), max |Δrank|=2 on a rule with posterior ≈ 1e-22, max |Δprob|=1.68e-10, max mixed-class mass 7.70e-05, 0/5 top-1 flips. Numbers are small. But Codex flags that this is a reduced-scale 5-rule check, not a full-scale closure at depth=7 / 300k / 60 rules.
- All other R3 findings (F2–F6) verified in code by the reviewer. Sidecar drift pattern (R2) and labelling drift pattern (R3) both look resolved.

### Codex's Memory Update (verbatim, R4)
- Remaining live issue: residual mixed classes are now an approximation-quality question, not an obvious pipeline bug.
- The branch-wide drift problem looks much better; the scorer and sidecars are largely aligned now.
- I do not see a new concrete blocker beyond the need for a full-scale residual-sensitivity bound.
- If you submit without that run, the honest framing is: fingerprint classes are an empirical approximation with observed negligible effect on a reduced-scale stress test, not an exact equivalence construction.

### Single most impactful pre-submission fix (Codex recommendation)
Run the residual mixed-class sensitivity audit on the full `depth=7 / 300k` pool across all official variants and all 60 rules, then report max posterior mass on residual mixed classes and worst-case movement in the headline outputs. If that bound stays in the same regime as the R4 reduced-scale result, Codex would clear it.

### Loop result
POSITIVE_THRESHOLD met (score ≥ 6 AND verdict contains "almost"). Loop stops at Round 4/4.
