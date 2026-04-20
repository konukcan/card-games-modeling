# ARIS Overnight Run — MCMC Remediation (Night 3)

**Paste everything below the divider into a FRESH Claude Code session launched from the worktree `.worktrees/aris-bayesian-review`.** This prompt is self-contained; do not assume any prior conversation context.

---

## Your mission tonight

You are running a fully autonomous overnight MCMC remediation run. The goal: **determine whether the MCMC sampler and the enumeration engine produce compatible posteriors when run on the same hypothesis space** (depth-6, no injections, uniform grammar).

Night 2 found that ALL 7 rules tested failed the validity filter because:
1. MCMC visit_counts were truncated to top-20 by a hardcoded bug in `analyze_mcmc.py`
2. MCMC and enumeration operated at different depths (6 vs 7) with different hypothesis sets (injections vs none)
3. No class aggregation — MCMC strings were compared directly instead of being mapped to equivalence classes

Tonight fixes all three. You will:
1. Re-enumerate at depth-6/300k programs (no injections)
2. Re-run MCMC at 50k steps with full visit_counts serialization and annealing
3. Aggregate MCMC visits onto enumeration equivalence classes
4. Compare posteriors with the validity-gated framework from Night 2
5. Track convergence at 12 checkpoints to show whether/when agreement stabilizes

### Critical constraint

**DO NOT modify or overwrite** any files in:
- `src/gallery_analysis/results/` (existing depth-7 model outputs)
- The existing enumeration pool at depth-7 with injections

All new outputs go to `review-stage/experiments/night3_mcmc_remediation/`. This is a separate, self-contained experiment.

---

## Prior artifacts (READ THESE FIRST)

1. `review-stage/MCMC_REMEDIATION_DESIGN.md` — **the design doc. Read this fully. It IS your spec.**
2. `review-stage/MORNING_REPORT_NIGHT2.md` — Night 2 results and context
3. `review-stage/night2_round2/real_run_findings.md` — the failure that motivates this run
4. `review-stage/experiments/night2/compare_enum_vs_mcmc.py` — Night 2's comparison framework (extend this)
5. `src/gallery_analysis/mcmc_search.py` — the MCMC engine (lines 1727-1760: MCMCConfig, MCMCResult; line 2234: run_parallel_chains)
6. `src/gallery_analysis/analyze.py` — the enumeration engine (line 464: estimate_extensions; lines 361-393: adaptive ladder)
7. `src/gallery_analysis/gallery_rules.py` — 61 rules (your rule source)
8. `src/gallery_analysis/hypothesis_table.py` — fingerprint and class infrastructure

---

## Design decisions (already locked — do not deviate)

| Decision | Value | Why |
|----------|-------|-----|
| Enumeration depth | 6 | Match MCMC hard cap |
| Enumeration max_programs | 300,000 | Good class coverage at depth-6 |
| Injections | None | Fair comparison — MCMC can't discover injections |
| Grammar | Uniform (flat) | Same for both methods |
| MCMC n_steps | 50,000 | 5x current, enough for convergence diagnostics |
| MCMC n_chains | 4 | Independent, no communication |
| MCMC beta_start | 0.3 | Annealed exploration at start |
| MCMC beta_end | 1.0 | True posterior by end |
| MCMC top_k | UNLIMITED | Serialize ALL visit_counts — this is the #1 fix |
| Convergence checkpoints | 12 | Evenly spaced over 50k steps |
| Extension estimation (shared) | Adaptive ladder: 100k base, 1M for base_rate < 0.001 | High-precision for fair comparison |
| Extension estimation (native MCMC) | 10k probes | Record as-is for Question B |
| Comparison probes | 5,000 uniform random 6-card hands | Same as Night 2 |
| Rules | 20 (include Night 2's 7 + 13 more spanning difficulty spectrum) |
| Output dir | `review-stage/experiments/night3_mcmc_remediation/` | Isolated |

---

## Phase 0: Setup (first 30-60 minutes)

### 0.1 Verify test baseline

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling/.worktrees/aris-bayesian-review
cd src && ~/miniforge3/bin/python -m pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: 206+ pass / 1 pre-existing baseline fail. If more failures appear, diagnose and fix before proceeding.

### 0.2 Select rules

Pick 20 rules from `gallery_rules.py` that span:
- **Easy/common** (high base rate): e.g., all_red, all_same_color, three_or_more_same_suit
- **Medium**: e.g., all_same_suit, all_even, all_odd
- **Hard/rare** (low base rate): e.g., four_of_a_kind_adjacent, ranks_palindrome, strict_increasing
- **Structural/adjacency**: e.g., triple_any_adjacent, three_clubs_adjacent, ap_step1_len3_adj

Must include: `all_red, all_same_suit, all_even, triple_any_adjacent, strict_increasing, four_of_a_kind_adjacent, ranks_palindrome` (the Night 2 set).

Save the list to `review-stage/experiments/night3_mcmc_remediation/config.json`.

### 0.3 Fix the truncation bug

In `src/gallery_analysis/analyze_mcmc.py`, line 136:
```python
# BEFORE (hardcoded truncation — THE BUG)
freq_ranking = analyzer.frequency_ranking(top_k=20)

# AFTER
freq_ranking = analyzer.frequency_ranking(top_k=args.top_k if hasattr(args, 'top_k') else None)
```

And line 154:
```python
# BEFORE
'top_hypotheses': freq_ranking[:20]

# AFTER (no truncation — let the caller decide)
'top_hypotheses': freq_ranking
```

This fix is a prerequisite. Verify with a minimal test.

### 0.4 Implement annealing in mcmc_search.py

The MCMCConfig at line 1727 already has `beta_start` and `beta_end` fields (both default 1.0). The MH acceptance logic at lines 2109-2138 already uses beta for annealing. Verify that:
- Setting `beta_start=0.3, beta_end=1.0` produces a linear schedule over n_steps
- Visit_counts are tracked with the UNANNEALED posterior (this should already be the case — verify)
- The annealing schedule is: `beta(step) = beta_start + (beta_end - beta_start) * step / n_steps`

If the annealing schedule isn't implemented yet, add it. If it is, just verify and move on.

### 0.5 Implement checkpoint snapshots

Add to the MCMC runner: at each of 12 evenly-spaced intervals, dump a snapshot containing:
- Cumulative visit_counts up to that step
- Total unique programs seen
- Acceptance rate in the preceding interval
- Current beta value

Save as `checkpoints/rule_name/checkpoint_{step}.json`.

### 0.6 Implement full visit_counts serialization

Ensure `run_parallel_chains()` returns the FULL merged visit_counts dict (no top_k truncation). The output JSON per rule should contain:
```json
{
  "rule": "rule_name",
  "config": { ... },
  "visit_counts": { "program_string": count, ... },
  "n_unique": 1234,
  "total_visits": 200000,
  "checkpoints": [ ... ]
}
```

---

## Phase 1: Enumeration (parallel with Phase 2 prep)

### 1.1 Run depth-6 enumeration with 300k programs, no injections

Write a script `review-stage/experiments/night3_mcmc_remediation/run_enum_depth6.py`:
- Use `build_gallery_grammar()` with uniform weights
- Depth budget = 6, max_programs = 300,000
- NO injections (do not call `merge_injections_and_extend`)
- Build equivalence classes via fingerprinting
- Save pool (classes + fingerprints) to `enum_depth6_300k/pool.json`

### 1.2 Compute shared extensions (adaptive ladder)

For each equivalence class in the pool:
- Base estimate: 100,000 MC probes
- If base_rate < 0.001: re-estimate with 1,000,000 probes
- Save to `enum_depth6_300k/extensions.json`

### 1.3 Compute per-rule posteriors

For each of the 20 rules:
- Use the depth-6 pool + shared extensions
- Compute posterior under uniform prior + size-principle likelihood
- Save to `enum_depth6_300k/posteriors/{rule_name}.json`

**Launch enumeration early** — it should complete in 30-60 min and the MCMC needs the class table for aggregation (or you can do aggregation as a post-processing step).

---

## Phase 2: MCMC run

### 2.1 Launch MCMC

For each of 20 rules, run:
- 50,000 steps × 4 chains
- beta_start=0.3, beta_end=1.0
- max_depth=6, max_nodes=25, noise_epsilon=0.01
- Seeds: 42, 43, 44, 45
- Full visit_counts (no top_k cap)
- 12 convergence checkpoints

**Timing estimate:** Night 2's 10k steps × 4 chains took ~671s per rule. At 50k steps, expect ~3,350s (~56 min) per rule. 20 rules sequential = ~18.5 hours. This is TOO LONG for one night.

**CRITICAL: You must parallelize or reduce scope to fit in ~10 hours.**

Options (choose the best strategy given available compute):
- **Option A:** Run rules in parallel using multiprocessing (4-8 workers, memory permitting)
- **Option B:** Run 50k steps on the 7 Night 2 rules + 25k steps on 13 new rules (prioritize depth on known failures)
- **Option C:** Reduce to 30k steps across all 20 rules (still 3x current, with 12 checkpoints showing convergence trajectory)
- **Option D:** Run 50k on 12-15 rules instead of 20 (drop some easy rules where convergence is likely fast)

**Use your judgment.** The convergence checkpoints will show whether more steps help, so even a shorter run is informative. The goal is maximum information within the time budget, not rigid adherence to step count.

Save MCMC outputs to `mcmc_50k_4chains/raw_visits/{rule_name}.json`.

### 2.2 Monitor and adapt

Every 2 hours, check:
- Are processes still running? (`ps aux | grep python`)
- Memory pressure? (`top -l 1 -s 0 | grep PhysMem`)
- Any rules already converged at earlier checkpoints?

If memory pressure is high:
- Serialize completed rules to disk immediately
- Reduce concurrent workers
- Document the adaptation in `NOTES_FOR_HUMAN.md`

---

## Phase 3: Class aggregation + comparison

Once both enumeration and MCMC outputs exist:

### 3.1 Class aggregation

For each rule's MCMC visit_counts:
1. Parse each program string into AST
2. Compute extensional fingerprint (same method as `hypothesis_table.py`)
3. Look up fingerprint in enumeration's class table
4. Sum visit weights per class → MCMC class-level posterior
5. Track unmatched programs separately

Report:
- `mass_in_full_list`: fraction of total MCMC visits that successfully mapped to an enum class
- `n_unmatched_programs`: programs that failed parse/fingerprint/lookup
- `unmatched_mass_fraction`: weight in unmatched programs

### 3.2 Question A — shared extensions comparison

Using the adaptive-ladder shared extensions:
- Compute p_accept(hand) for each of 5,000 probe hands under BOTH posteriors
- Report: mean|Δ|, max|Δ|, Spearman ρ, top-50 overlap, KL divergence
- Apply validity thresholds (from design doc)
- Record `comparison_valid` flag

### 3.3 Question B — native extensions comparison

Same as above, but using MCMC's native 10k-probe extension estimates for the MCMC side.
This tests full pipeline agreement including estimation noise.

### 3.4 Convergence diagnostics

From the 12 checkpoints per rule:
- Compute class-aggregated posterior at each checkpoint
- Track how Question A metrics evolve over steps
- Plot (or tabulate): mass_in_full_list, mean|Δ|, Spearman ρ at each checkpoint
- Identify: at what step count does each rule first pass the validity filter (if ever)?

---

## Phase 4: Reporting

### 4.1 Write summary.json

```json
{
  "run_config": { ... },
  "per_rule": {
    "rule_name": {
      "enum_n_classes": 1234,
      "mcmc_n_unique": 5678,
      "mcmc_total_visits": 200000,
      "mass_in_full_list": 0.94,
      "n_unmatched": 45,
      "unmatched_mass_fraction": 0.02,
      "question_a": { "valid": true, "mean_delta": 0.03, "rho": 0.92, ... },
      "question_b": { "valid": true, "mean_delta": 0.05, "rho": 0.88, ... },
      "convergence": { "first_valid_step": 25000, "final_rho": 0.95 }
    }
  },
  "aggregate": {
    "n_rules_valid_a": 18,
    "n_rules_valid_b": 15,
    "mean_rho_a": 0.91,
    "worst_rule": "four_of_a_kind_adjacent",
    "best_rule": "all_red"
  }
}
```

### 4.2 Write MORNING_REPORT_NIGHT3.md

Same format as Night 2's morning report. Include:
- Headline finding (how many rules pass validity? what's the agreement level?)
- Convergence trajectory (does agreement improve monotonically with steps?)
- Per-rule breakdown table
- Comparison to Night 2 (which had 0/7 valid — how much did we improve?)
- Any unexpected findings or failures
- Recommendations for follow-up

### 4.3 Commit everything

```bash
git add review-stage/experiments/night3_mcmc_remediation/ review-stage/MORNING_REPORT_NIGHT3.md
git commit -m "feat: Night 3 MCMC remediation — [HEADLINE RESULT]"
```

---

## Autonomy and creative latitude

This run is designed for you to operate with significant autonomy. The design doc locks the WHAT (decisions above), but you have full latitude on the HOW:

- **If something doesn't work as expected** (imports fail, memory pressure, unexpected data shapes), find a workaround. Document what you did and why in `review-stage/experiments/night3_mcmc_remediation/NOTES_FOR_HUMAN.md`.

- **If you discover something interesting** during the run (e.g., a subset of rules converges beautifully while others diverge, or unmatched mass is concentrated in a specific program pattern), investigate it. Add findings to the morning report.

- **If the MCMC runtime exceeds budget**, adapt. Prioritize rules that are most informative (the Night 2 failures + hard rules). A thorough analysis of 12 rules with full convergence diagnostics is more valuable than a rushed run of 20 rules with no time for comparison.

- **If class aggregation reveals a systematic issue** (e.g., most MCMC programs fail to parse), that IS a finding. Document it, diagnose the root cause, and attempt a fix if tractable. If not tractable, report it clearly.

- **If you need infrastructure** that doesn't exist (e.g., a parser that handles a different program format, or a checkpoint mechanism), build it. Keep it minimal and focused.

- **Multi-round iteration:** If your first attempt at comparison reveals issues (e.g., the fingerprinting doesn't match, or there's a grammar mismatch between the two engines), treat each iteration as a round. Fix, re-run the affected piece, and report progress. The goal is to maximize what we learn tonight, not to rigidly execute a linear pipeline.

The North Star is: **by morning, we should know whether MCMC and enumeration agree at depth-6, and if not, exactly why not.**

---

## Caffeinate and process management

```bash
# Launch the main orchestration script
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling/.worktrees/aris-bayesian-review
nohup caffeinate -d -i -s ~/miniforge3/bin/python review-stage/experiments/night3_mcmc_remediation/run_all.py > review-stage/experiments/night3_mcmc_remediation/run.log 2>&1 &
echo $! > review-stage/experiments/night3_mcmc_remediation/run.pid
```

Or, if you prefer to orchestrate in phases yourself (launching each phase manually), that's fine too. Just ensure caffeinate wraps any long-running process.

**Memory budget:** ~24GB available. The enumeration at 300k/depth-6 is lighter than the existing depth-7/300k run. MCMC with full visit_counts for 20 rules could be 2-4GB in aggregate. Serialize to disk between rules if needed.

---

## Success criteria

By morning, one of these outcomes:

**Best case:** Most rules pass validity filter. Agreement metrics (ρ > 0.8, mean|Δ| < 0.05) demonstrate that MCMC and enumeration produce compatible posteriors at depth-6. Convergence diagnostics show at what step count this agreement stabilized.

**Good case:** Some rules pass, some fail. Clear pattern emerges (e.g., "high-base-rate rules converge quickly, rare rules need more steps"). Convergence trajectory shows the direction of travel.

**Acceptable case:** Mass coverage (mass_in_full_list) dramatically improves over Night 2 (from 13-52% to 80%+), even if agreement metrics aren't perfect yet. This confirms the truncation bug was the main culprit and more steps will finish the job.

**Informative failure:** Class aggregation reveals a systematic mismatch (e.g., grammar difference between engines). Documented clearly with root cause. Still valuable — tells us exactly what to fix next.

All of these are publishable findings. The framework is designed to catch its own failures.

---

## Python environment

```bash
~/miniforge3/bin/python  # Python 3.11, all dependencies installed
```

All imports should use:
```python
import sys
sys.path.insert(0, '/Users/cankonuk/Documents/self-explanations-project/card-games-modelling/.worktrees/aris-bayesian-review/src')
```

---

## Final notes

- **DO NOT** modify files outside the worktree
- **DO NOT** overwrite existing model outputs in `src/gallery_analysis/results/`
- **DO** commit intermediate progress (at minimum after each phase completes)
- **DO** write `NOTES_FOR_HUMAN.md` with anything surprising, any workarounds used, any judgment calls made
- **DO** prioritize information density over completeness — a clear answer on 12 rules beats a partial answer on 20
