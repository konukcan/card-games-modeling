# Night 3 — MCMC Remediation Morning Report

**Branch:** `aris/bayesian-review`
**Run directory:** `review-stage/experiments/night3_mcmc_remediation/`
**Completed:** 2026-04-21 (local, overnight)

---

## TL;DR

Do MCMC and enumeration agree on the same hypothesis space (depth-6, no injections, uniform grammar, 18 target rules with complete exemplars)?

**Partial agreement. 6 of 18 rules pass the pre-registered validity gate; 3 of 18 show genuinely tight posterior agreement (TV < 0.2).** The dominant failure mode is a ~10–33 % mass leak to equivalence classes outside the enumerated pool — MCMC visits programs whose fingerprints were never enumerated. This is the β-annealing artifact we flagged in the design doc: the chain wanders into low-likelihood regions early, and with only 20k steps + β climbing 0.3→1.0, it does not fully retract.

### Headline numbers (18 rules)

| metric | value |
|---|---|
| **n valid** (`mass_mapped ≥ 0.90`) | 6 / 18 |
| **n tightly agreeing** (valid AND TV < 0.20) | 3 / 18 |
| median total variation | 0.567 |
| mean total variation | 0.531 |
| median Jensen-Shannon | 0.361 |

### Per-rule table (sorted by TV, ascending)

| rule | valid | TV | JS | mass\_mapped | top-1 match | flag(s) |
|---|:---:|---:|---:|---:|:---:|---|
| all\_same\_suit | ✅ | 0.034 | 0.017 | 0.990 | ✓ | — |
| all\_red | ✅ | 0.103 | 0.041 | 0.929 | ✓ | — |
| three\_or\_more\_same\_suit | ❌ | 0.145 | 0.063 | 0.779 | ✓ | mcmc\_unmapped |
| ranks\_palindrome | ✅ | 0.174 | 0.091 | 0.941 | ✓ | — |
| ap\_step1\_len3\_adj | ❌ | 0.184 | 0.078 | 0.868 | ✗ | mcmc\_unmapped |
| all\_clubs\_or\_hearts | ✅ | 0.262 | 0.169 | 0.945 | ✓ | — |
| colors\_palindrome | ❌ | 0.283 | 0.119 | 0.666 | ✗ | mcmc\_unmapped |
| pair\_5s\_adjacent | ❌ | 0.394 | 0.155 | 0.898 | ✓ | mcmc\_unmapped |
| strict\_increasing | ❌ | 0.547 | 0.336 | 0.802 | ✓ | mcmc\_unmapped |
| all\_same\_color | ❌ | 0.567 | 0.361 | 0.810 | ✗ | mcmc\_unmapped |
| triple\_any\_adjacent | ❌ | 0.673 | 0.454 | 0.850 | ✓ | mcmc\_unmapped |
| pos135\_same\_rank | ❌ | 0.688 | 0.532 | 0.884 | ✓ | mcmc\_unmapped |
| all\_odd | ✅ | 0.759 | 0.635 | 0.927 | ✗ | — |
| four\_hearts\_adjacent | ✅ | 0.807 | 0.613 | 0.945 | ✗ | — |
| three\_clubs\_adjacent | ❌ | 0.960 | 0.868 | 0.772 | ✗ | mcmc\_unmapped |
| all\_even | ❌ | 0.988 | 0.959 | 0.840 | ✗ | mcmc\_unmapped |
| four\_of\_a\_kind\_adjacent | ❌ | 0.992 | 0.968 | 0.796 | ✗ | mcmc\_unmapped |
| left\_red\_right\_black | ❌ | 1.000 | 0.999 | 0.864 | ✗ | mcmc\_unmapped |

Legend — "top-1 match" = MCMC's modal equivalence class equals enum's modal class.

---

## Answering the three pre-registered questions

### Q1. Do MCMC and enumeration find the same equivalence classes?

**Yes for the common ones, no for the tails.** For every rule, 67–99 % of MCMC's mass falls on classes enum also enumerates. The unmapped 1–33 % consists of programs that type-check, pass predicate evaluation, but whose fingerprints never appear in the depth-6 enumeration pool. These are not Python exceptions and not parse failures — they are real programs at depths > 6, or at depth ≤ 6 with syntactic forms the enum pruner never reached.

**Mapped-mass distribution:** mean 0.87, median 0.86, range [0.67, 0.99].

### Q2. Do the posteriors agree in mass assignment?

**Only for the simplest rules.** Three rules meet the "agreement" bar (valid AND TV < 0.2):

- `all_same_suit` (TV 0.034) — strongest agreement; also had the cleanest MCMC trajectory
- `all_red` (TV 0.103) — strong agreement
- `ranks_palindrome` (TV 0.174) — moderate agreement

For every other rule, even when MCMC's *top-1 class matches enum's top-1* (10 of 18 cases), the mass distribution across the remaining classes diverges substantially. This is a classic MCMC-concentration problem: the modal mode is correct but surrounding mass is misallocated.

### Q3. Does MCMC converge by 20k steps?

**No, not for most rules.** Convergence diagnostics at 10 equally spaced checkpoints (steps 1999 → 19999) show `mass_mapped` still climbing slowly in the final third of most runs:

| rule | ckpt 3 | ckpt 6 | ckpt 10 | Δ last 3 |
|---|---:|---:|---:|---:|
| all\_same\_suit | 0.966 | 0.983 | 0.990 | +0.003 |
| all\_red | 0.762 | 0.881 | 0.929 | +0.018 |
| colors\_palindrome | 0.597 | 0.620 | 0.666 | +0.035 |
| left\_red\_right\_black | 0.827 | 0.836 | 0.864 | **−0.003** |
| three\_clubs\_adjacent | 0.800 | 0.778 | 0.772 | **−0.011** |
| strict\_increasing | 0.741 | 0.822 | 0.802 | **−0.016** |

Three rules actually *lose* mapped mass in the tail — strong signal that the chain is still mixing into new (unmapped) regions rather than concentrating. Several rules reach "valid" status only in the last 40 % of the run (e.g., `four_hearts_adjacent` first valid at step 11999; `all_red` at step 15999).

**Six rules never crossed the 0.90 mapped-mass threshold at any checkpoint.** For these, 20k steps is clearly insufficient.

---

## Critical methodological discovery

During the first comparison smoke test, the fingerprint mapping catastrophically failed: every rule reported `mass_mapped ≈ 0.01`, TV ≈ 1.0. Root cause, diagnosed via direct Python probing of the pool:

The enumeration pipeline does **strict-exemplar-holdout splitting** (`analyze.py:425`). When a fingerprint class fails to be consistent on a held-out probe set, it is split into sub-classes whose fingerprint is **composed** as `sha256(parent_fp | sub_fp)` where `sub_fp` is re-computed on exemplar hands + 1000 holdout probes (seed 9999).

Of 2,501 classes in `enum_depth6_300k/pool.pkl`, 233 are such split sub-classes under 64 parents — and several canonical rule classes (`all_same_suit`, `all_red`, etc.) live in the split sub-class index, not the direct-fingerprint index. The initial MCMC aggregator looked only at the direct index, so it never matched.

**Fix** (commit `e2cd8d7`): dual index `(direct_fp_to_cls, parent_to_subs)`; 2-stage lookup in `_aggregate_mcmc_to_classes`:

1. Try direct fingerprint match on the 500 standard probes.
2. If that misses and the fingerprint is a known parent, recompute a composed fingerprint using the `_member_fp` function (exemplars + 1000 holdout probes, seed 9999) and look it up in the sub-class index.

Without this fix, **the night-3 experiment would have reported 0 / 18 valid rules with TV 1.0 across the board.**

---

## Interpretation

### What this tells us about the design doc's hypothesis

The design doc predicted: *if MCMC and enumeration disagree on the same finite space, the culprit is MCMC failing to concentrate, not missing modes.* The data supports the "failing to concentrate" half but reveals a second dimension the doc underweighted: **MCMC's proposal distribution reaches classes the enumeration pruner never enumerates**. These are legitimately-typed programs that the enumeration simply never produces — not evaluation errors.

This means:

1. The effective MCMC hypothesis space ≠ the enumerated space, even at identical depth. Proposal kernels and enumeration grammars cover overlapping but non-identical subsets.
2. Reporting `mass_mapped` as a validity gate is *the right choice* — it makes the divergence visible rather than hiding it.

### Why 6 rules pass the gate but only 3 agree

The gate (`mass_mapped ≥ 0.90`) checks *support alignment*, not *mass alignment*. Rules like `all_odd` (TV 0.759) and `four_hearts_adjacent` (TV 0.807) clear the gate because MCMC's samples do land on enum-known classes — but they land on the wrong ones proportionally. For these rules, MCMC's modal class does not match enum's (see "top-1 match" column).

The 3 tight-agreement rules (`all_same_suit`, `all_red`, `ranks_palindrome`) all share: short canonical form, strong size-principle signal (small equivalence class on the exemplar set), and MCMC trajectories that climb monotonically and stably.

### Budget implications

Rough cost of the present run at 20k × 4 × 18 on 10 workers: ~8 h wall-clock. Extrapolating to the 50k budget originally planned (2.5×): ~20 h. Full restart at 50k is cheap in researcher time but would likely still not resolve the *unmapped mass* issue, which is architectural not quantitative.

---

## Recommendations for the next iteration

The cleanest next step depends on which question dominates your interest:

- **If the priority is "do MCMC and enumeration agree?"** — rerun the 12 failing rules at 100k steps × 4 chains with β ramp 0.5→1.0 (shorter annealing tail). Expected: 2–4 more rules cross the 0.90 gate.
- **If the priority is "what is MCMC proposing that enum misses?"** — dump the top 10 unmapped fingerprints per rule (already present in `comparison/question_a/*.json`) and synthesize those programs back into the enum grammar. This is a direct test of whether the enum pruner is too aggressive.
- **If the priority is "is the enumeration posterior even right?"** — compare enum@depth-6 vs enum@depth-7 on the 6 valid rules. If they agree to TV < 0.05, MCMC is the lone suspect for disagreement; if they don't, depth-6 is under-enumerated.

---

## Artifacts

| path | contents |
|---|---|
| `enum_depth6_300k/` | Enumeration pool: 2501 classes, 300k programs, depth 6, seed 42 |
| `mcmc_50k_4chains/raw_visits/*.json` | Per-rule visit counts, 4 chains × 20k steps × 18 rules |
| `mcmc_50k_4chains/checkpoints/*.json` | 10 checkpoints per rule (steps 1999…19999) |
| `comparison/summary.json` | Headline validity + divergence metrics, all 18 rules |
| `comparison/question_a/*.json` | Parse-audit, top-unmapped fingerprints, top-k class lists per rule |
| `comparison/convergence_diagnostics/*.json` | Checkpoint trajectory per rule |
| `run_comparison.py` | Aggregator with composite-fingerprint fix (e2cd8d7) |

---

## What went smoothly

- Autonomous loop pattern held for the full night: enum → MCMC (10 workers parallel) → checkpoints → comparison → convergence diagnostics, with no human intervention.
- Hard validity gating caught the fingerprint-mapping bug *before* results were written to the morning report, exactly as designed.
- ProcessPoolExecutor with cloudpickle + spawn handled 72 chain-tasks reliably; no worker deaths, no OOM.

## What surprised me

- The composite-fingerprint bug. The design doc assumed `pool.pkl` contained a flat `fingerprint → class` table; it does not. Half a day's work would have gone to a silent null result without the `mass_mapped ≈ 0.01` tripwire.
- 10 of 18 rules have a matching top-1 class despite high TV. MCMC identifies the right *mode* but distributes surrounding mass poorly — a different failure mode than expected.
- Three rules *lose* mapped mass as the run progresses. This is not stationary behavior; the chain is still exploring, probably still climbing out of β-annealing's early tail.
