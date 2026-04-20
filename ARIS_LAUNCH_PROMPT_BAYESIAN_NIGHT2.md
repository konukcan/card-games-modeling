# ARIS Overnight Run — Bayesian Enumeration Model Review (NIGHT 2)

**Paste everything below the divider into a FRESH Claude Code session launched from the existing worktree (see setup instructions at bottom).** This prompt is self-contained; do not assume any prior conversation context.

---

## Your mission tonight (Night 2)

You are running a fully autonomous overnight research-review loop on a **Bayesian program-enumeration system** for card-game rule induction. Night 1 has completed:

- **Night 1** (4 rounds, score 2.0 -> 4.0 -> 5.0 -> 6.0): Fixed fingerprint collision splitting, weighted-enumeration pool separation, nan-prior fail-closed, probe-hash wiring, retained-mass reporting, base-rate scoring (eliminates int() rounding), tempering relabeling, sidecar drift repair, depth-cap relabeling, prior-scorer unification, balanced-n truthful reporting.

**Night 1 ended at 6.0/10 ("almost").** The reviewer's final verdict:

> "The branch is no longer in the 'fundamentally incorrect pipeline' state from Round 1. The single most impactful minimum fix is: run the residual mixed-class sensitivity audit on the full depth=7 / 300k pool across all official variants and all 61 rules, then report the maximum posterior mass on residual mixed classes and the worst-case movement in the headline outputs."

Tonight has **four workstreams**, three of which are new:

1. **Full-scale sensitivity audit** (background compute) — close the Night 1 gap
2. **Adversarial hand generation** (new design + implementation) — find hands that are maximally diagnostic for an ideal Bayesian learner
3. **Enumeration-vs-MCMC posterior comparison** (new) — validate MCMC against the hardened enumeration gold standard
4. **Visualization end-to-end stress test** (new) — actually run the report generator and check output

### Prior artifacts (READ THESE FIRST, in this order)

1. `ARIS_LAUNCH_PROMPT_BAYESIAN_REVIEW.md` — Night 1 launch prompt (read for full pipeline description)
2. `review-stage/AUTO_REVIEW.md` — Night 1 chronological reviewer output, all 4 rounds (**read fully**)
3. `review-stage/REVIEW_STATE.json` — Night 1 final state
4. `REVIEWER_MEMORY.md` — reviewer's accumulated memory across Night 1's 4 rounds
5. The commit log: `git log --oneline` — Night 1 commits are here
6. The uncommitted diff: `git diff HEAD` — **Night 1's fixes (858 lines) are NOT YET COMMITTED. Your first action is to commit them.**

**Do not re-do Night 1's fixes.** They are in the working tree (uncommitted). Tonight starts from where Night 1 left off.

## Phase 0: Setup (first 30-60 minutes)

Do these before launching the review loop:

### 0.1 Commit Night 1's fixes

Night 1 left 858 lines of uncommitted changes across 8 files. Commit them:

```bash
git add src/gallery_analysis/analyze.py src/gallery_analysis/bayesian_scorer.py \
  src/gallery_analysis/compare_prior_modes.py src/gallery_analysis/depth_mass_analysis.py \
  src/gallery_analysis/enumerator.py src/gallery_analysis/hand_diagnosticity.py \
  src/gallery_analysis/injection.py src/gallery_analysis/run_diagnosticity.py
git commit -m "fix: Night 1 Bayesian review — 7 correctness fixes across 8 subsystems

Fixes from 4-round adversarial review (score 2->6/10):
- Strict equivalence-class splitting (post-fingerprint AND post-injection)
- Weighted-enumeration pool separation (--enumeration-grammar flag)
- Fail-closed nan-prior handling (--strict-priors flag)
- Probe-hash wiring to extension cache
- Retained-mass reporting in diagnosticity pruning
- Base-rate scoring (eliminates int() rounding in TOTAL_HANDS)
- Tempering relabeling (likelihood_exponent != 1)
- Sidecar drift repair (shared merge_injections_and_extend helper)
- Prior-scorer unification between main and diagnosticity paths
- Depth-cap relabeling (enumeration budget, not Program.depth())
- Balanced-n truthful reporting (actual vs target)"
```

### 0.2 Cherry-pick MCMC analysis files

The MCMC analysis code doesn't exist on this branch. Copy it from `feature/mcmc-search`:

```bash
# From the worktree root
git show feature/mcmc-search:src/gallery_analysis/analyze_mcmc.py > src/gallery_analysis/analyze_mcmc.py
git show feature/mcmc-search:src/gallery_analysis/mcmc_hypothesis_collector.py > src/gallery_analysis/mcmc_hypothesis_collector.py
```

Then verify imports work:
```bash
cd src && ~/miniforge3/bin/python -c "from gallery_analysis.analyze_mcmc import run_mcmc_analysis; print('OK')"
```

If imports fail due to changed function signatures in Night 1's modified files, fix the import paths. **Import path adjustments are permitted; no logic changes to MCMC code.**

Also verify grammar consistency:
```bash
~/miniforge3/bin/python -c "
from gallery_analysis.enumerator import build_gallery_grammar
g = build_gallery_grammar()
print(f'Primitives: {len(g.primitives)}')
for p in sorted(g.primitives, key=str)[:5]:
    print(f'  {p}')
"
```
Should show 64 primitives. If the count differs from what `feature/mcmc-search` sees, document the discrepancy in `NOTES_FOR_HUMAN.md`.

Commit the cherry-picked files:
```bash
git add src/gallery_analysis/analyze_mcmc.py src/gallery_analysis/mcmc_hypothesis_collector.py
git commit -m "chore: cherry-pick MCMC analysis files from feature/mcmc-search for comparison"
```

### 0.3 Run test baseline

```bash
cd src && ~/miniforge3/bin/python -m pytest tests/test_bayesian_scorer.py tests/test_deep_enumeration.py tests/test_hypothesis_table.py tests/test_injection.py tests/test_injected_hypotheses.py tests/test_hand_diagnosticity.py tests/test_dsl_prior.py tests/test_memoized_enumeration.py tests/test_fingerprint_refinement.py tests/test_pipeline_integration.py tests/test_validation_fixes.py -v
```

Expected: same baseline as Night 1 end (184 pass / 1 pre-existing failure in `test_merge_updates_summed_prior`).

### 0.4 Write and launch sensitivity audit

Adapt `/tmp/r3_f1_sensitivity.py` to full scale. First, run a 1-rule timing test:

```bash
# Time how long 1 rule takes at full scale
~/miniforge3/bin/python -c "
import time
# ... (adapt r3_f1_sensitivity.py for 1 rule at depth=7, max_programs=300000)
"
```

**Decision tree:**
- If 1 rule takes <5 minutes: run all 61 rules (~5h total). Go.
- If 1 rule takes 5-10 minutes: run 20 representative rules covering all 4 difficulty groups (~2-3h). Go.
- If 1 rule takes >10 minutes: run 10 rules (~2h). Document the timing constraint.

Save the script to `review-stage/experiments/night2/full_sensitivity_audit.py` and launch:

```bash
mkdir -p review-stage/experiments/night2
nohup caffeinate -d -i -s ~/miniforge3/bin/python review-stage/experiments/night2/full_sensitivity_audit.py > review-stage/experiments/night2/sensitivity.log 2>&1 &
echo $! > review-stage/experiments/night2/sensitivity.pid
```

**Memory warning:** The audit and review work share 24GB. Monitor with `top -l 1 -s 0 | grep PhysMem`. If memory pressure is hit, pause the audit until the review round's compute finishes.

### 0.5 Write RESUME_INSTRUCTIONS.md

Write at worktree root, early.

## Tonight's workstreams

### Workstream 1: Adversarial Hand Generation (Rounds 1-2 priority)

**Goal:** Given the posterior P(rule | 6 exemplars), find hands that are maximally diagnostic for an ideal Bayesian learner who saw those exemplars and must decide whether a new hand belongs to the same rule.

**The approach: posterior predictive entropy maximization (BALD proxy)**

The current diagnosticity pipeline (`hand_diagnosticity.py`) samples random hands and rates them. We want to actively search for the hardest hands.

For each candidate hand h:
1. Compute p(h) = P(accept | h, D) = sum over hypotheses of [posterior(hypothesis) * I(h satisfies hypothesis)]
2. Compute entropy: H(h) = -p(h)*log(p(h)) - (1-p(h))*log(1-p(h))
3. H is maximized when p(h) = 0.5 (posterior is maximally divided)

This is equivalent to BALD (Bayesian Active Learning by Disagreement, Houlsby et al. 2011) for binary classification. Complexity: O(candidates * hypotheses), tractable for 50K candidates * 2.5K classes.

**Implementation plan:**

New function `find_most_diagnostic_hands(posteriors, n_candidates=50000, top_k=100)`:
1. Sample candidate hands uniformly from the card space
2. For each candidate, compute P(accept) via weighted vote across posterior
3. Rank by entropy (proximity to 0.5)
4. Return top-k with scores

New function `find_most_adversarial_hands(posteriors, rule_predicate, n_candidates, top_k)`:
1. Find hands where model is confident but WRONG (false positives under the true rule where p_accept > 0.8, and false negatives where p_accept < 0.2)
2. These are the hands that would most surprise the learner

**The reviewer should critique:**
- Is the entropy proxy sufficient, or does the setting require full EIG?
- Are there smarter sampling strategies than uniform random? (e.g., stratified by suit/rank distribution, or rejection sampling near the decision boundary)
- Edge cases: rules with very small extensions (most hands are negative), rules with very large extensions (most hands are positive)
- Should there be a notion of "diversity" among the top-k diagnostic hands (avoid returning 100 nearly-identical hands)?

**Minimum viable deliverable:** entropy-proxy ranking with 50K candidates for 5 representative rules, plus documentation of the method. If full design takes too long, ship this and note the open questions.

### Workstream 2: Enumeration-vs-MCMC Comparison (Rounds 2-3)

**Goal:** For a shared set of rules under the uniform grammar, compare posteriors from exhaustive enumeration vs MCMC sampling.

**Implementation:**

New script `compare_enumeration_mcmc.py`:
1. Load enumeration results (from `analyze.py` output, uniform grammar variant)
2. Run MCMC analysis (using cherry-picked `analyze_mcmc.py`) on the same rules
3. For each rule, compute:
   - Top-10 hypothesis overlap (Jaccard on program strings or equivalence class fingerprints)
   - KL divergence between the two posteriors (over shared support)
   - Rank correlation (Spearman) of hypothesis rankings
   - True-rule rank agreement
   - Total variation distance
4. Report aggregate statistics and per-rule breakdown
5. Flag rules where the two engines disagree most

**Constraints:**
- Both engines use uniform grammar (`build_gallery_grammar()` with no tier weights)
- Verify MCMC has enough samples for convergence (check effective sample size, warn if < 100)
- Comparison is on the posterior over equivalence classes, not individual programs
- If MCMC chain files don't exist on this branch: run MCMC for a few representative rules (5-10) with enough burn-in. Time-box to 60 minutes total.

### Workstream 3: Visualization Stress Test (Rounds 3-4)

- Run `generate_reports.py` end-to-end on at least 2 variant results (e.g., uniform_summed_inject and weighted_canonical_inject)
- Open the generated HTML reports and verify:
  - Difficulty scatter: correct axes, group coloring, rule labels
  - Per-rule pages: exemplar images load, posterior bars sum to ~1, DSL translations readable
  - Comparison dashboard: variants load correctly, delta computation makes sense
- Run `dsl_translator.py` on the actual top-10 hypotheses for all 61 rules. Count how many fall through to raw DSL (should be <5%).
- Fix any chart bugs found.

### Workstream 4: Open-ended review (all rounds)

Continue looking for issues beyond the known list. Night 1 found issues we didn't anticipate (fingerprint collisions, depth-cap bug, sidecar drift pattern). Expect the same here, especially in the new adversarial and comparison code.

## Round allocation

- **Round 1:** Adversarial hand generation design + implementation (hardest problem, needs fresh reviewer attention). Sensitivity audit running in background.
- **Round 2:** Adversarial refinement based on reviewer feedback + start MCMC cherry-pick and comparison script.
- **Round 3:** MCMC comparison results + visualization stress test.
- **Round 4:** Close-out. Integrate sensitivity audit results (if complete). Address any new issues surfaced. Write morning report.

## The loop itself

Invoke the installed ARIS skill `/auto-review-loop` with these overrides:

```
/auto-review-loop "Night 2 of Bayesian enumeration pipeline review. Night 1 ended at 6.0/10 with one gap: full-scale sensitivity audit (running as background compute tonight). Three new workstreams: (1) Adversarial hand generation via posterior predictive entropy maximization (BALD proxy) — design + implement find_most_diagnostic_hands() and find_most_adversarial_hands(), (2) Enumeration-vs-MCMC posterior comparison under uniform grammar — cherry-picked MCMC files, new comparison script, KL divergence + rank correlation + TV distance, (3) Visualization end-to-end stress test — actually generate reports and check chart correctness. Read review-stage/AUTO_REVIEW.md and REVIEWER_MEMORY.md for Night 1 context. Focus on theoretical correctness and principled experimental design, NOT benchmarking." — compact: true, human checkpoint: false, difficulty: hard
```

The skill will:
1. Send project context to GPT-5 via Codex MCP for review (**verify connection with `claude mcp list | grep codex`** before starting)
2. Receive structured weaknesses + suggestions + score
3. You implement fixes
4. Run nested engineering review (see below)
5. Loop up to MAX_ROUNDS = 4

**Stop condition:** 4 rounds, no early stop. Even if score reaches 9+ or 10, continue. Treat `POSITIVE_THRESHOLD` as requiring BOTH score >= 6 AND at least 3 rounds completed.

### Nested engineering review (critical)

After implementing the reviewer's suggested fixes in each round:

1. **Run `kieran-python-reviewer`** (Agent tool, subagent_type `compound-engineering:review:kieran-python-reviewer`) on your diff
2. **Run `performance-oracle`** (Agent tool, subagent_type `compound-engineering:review:performance-oracle`)
3. **Run `code-simplicity-reviewer`** (Agent tool, subagent_type `compound-engineering:review:code-simplicity-reviewer`)
4. Address critical and high-priority findings only. Skip nit-level style.
5. **Cap the internal review loop at 2 passes per ARIS round.**

### Codex fallback

Codex is registered and expected to work. If any Codex MCP call fails:

1. Do NOT abort
2. Spawn a Claude general-purpose sub-agent with a harsh-NeurIPS-area-chair reviewer persona
3. Feed it the same context you would have sent Codex
4. Note in `AUTO_REVIEW.md` that this round used the Claude fallback

## Scope rules — STRICTLY ENFORCED

### You MAY freely modify:
- All files listed in Night 1 scope (see `ARIS_LAUNCH_PROMPT_BAYESIAN_REVIEW.md`)
- `src/gallery_analysis/analyze_mcmc.py` — import path adjustments ONLY, no logic changes
- `src/gallery_analysis/mcmc_hypothesis_collector.py` — import path adjustments ONLY
- New files under `src/gallery_analysis/` for adversarial hand generation and comparison
- `review-stage/experiments/night2/` — scripts and artifacts

### You MAY NOT touch:
- Same exclusions as Night 1 (see `ARIS_LAUNCH_PROMPT_BAYESIAN_REVIEW.md`)
- `src/gallery_analysis/mcmc_search.py` — MCMC sampler (frozen, already reviewed on separate branch)

### If you believe a forbidden change is necessary:
Append to `NOTES_FOR_HUMAN.md`.

## Experiment execution rules

1. **Caffeinate always:** `nohup caffeinate -d -i -s ~/miniforge3/bin/python ... > <logfile> 2>&1 &`
2. **Log location:** `review-stage/experiments/night2/` (create as needed)
3. **Time-box experiments:** no single experiment >90 min. If a diagnostic needs longer, note it in `NOTES_FOR_HUMAN.md` and skip.
4. **Write PID files:** `review-stage/experiments/night2/<exp_name>.pid` and `.cmd`
5. **Monitor memory:** if `PhysMem` shows <2GB free, pause background jobs until review compute finishes.

## Hard time budget: 10 hours from launch

Stop cleanly after 10 hours. Write `MORNING_REPORT.md` even if round 4 hasn't finished.

## Commit discipline

- Commit after each logical unit of work
- Conventional prefixes: `feat:`, `fix:`, `chore:`, `docs:`, `test:`
- **No `Co-Authored-By: Claude` or "Generated with Claude Code" tags**
- Commit BEFORE any experiment launch

## The morning report

At the end, write `MORNING_REPORT.md` at the worktree root with:

1. **TL;DR (~5 lines)** — rounds completed / score progression / biggest accomplishments
2. **Per-round breakdown** — what GPT-5 said, what you implemented, experiment results, internal review findings
3. **Sensitivity audit results** — full-scale residual bound (or partial results if still running). Max rank shift, max probability shift, any top-1 flips.
4. **Adversarial hand generation** — method description, example outputs for 5 rules, reviewer assessment
5. **MCMC comparison results** — KL divergence, rank correlation, disagreement analysis per rule
6. **Visualization findings** — any chart bugs, DSL translation coverage stats
7. **New issues surfaced** — anything beyond the planned workstreams
8. **Uncertainty flags / human attention** — anything in `NOTES_FOR_HUMAN.md`
9. **Final verdict** — overall pipeline readiness. What scope limitations remain for the paper?
10. **Recommended next steps** — what should the prior-realignment conversation (Phase 2 from the roadmap) focus on?

## Resilience to network interruptions

- Commit after every meaningful unit of work (every ~3-5 edits)
- Update `review-stage/REVIEW_STATE.json` aggressively
- Append to `review-stage/AUTO_REVIEW.md` as you go
- All experiments via `nohup` so they survive session death
- Update `RESUME_INSTRUCTIONS.md` after each major milestone

## Safety rails

- No destructive git operations (no `git reset --hard`, no `git clean -f`, no force push)
- No `--no-verify` on commits
- Do not push to remote
- Do not modify anything outside this worktree
- If a command prompts for interactive input, kill it and find a non-interactive equivalent

## Setup instructions (run these BEFORE pasting this prompt)

The worktree already exists from Night 1. Just navigate to it:

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling/.worktrees/aris-bayesian-review

# Verify state
git branch --show-current  # should be aris/bayesian-review
git diff --stat HEAD        # should show ~858 lines of uncommitted Night 1 fixes
```

Then launch Claude Code and paste everything above the setup section.

## Getting started — checklist

1. `pwd` — confirm you're in the worktree
2. `git branch --show-current` — should be `aris/bayesian-review`
3. `git diff --stat HEAD` — should show Night 1's uncommitted fixes
4. `claude mcp list | grep -i codex` — should show connected. If not, fall back to Claude sub-agent reviewer.
5. Read `review-stage/AUTO_REVIEW.md` fully (Night 1 context)
6. Read `REVIEWER_MEMORY.md` (reviewer's accumulated understanding)
7. **Execute Phase 0 in order:** commit fixes -> cherry-pick MCMC -> test baseline -> write+launch sensitivity audit -> write RESUME_INSTRUCTIONS.md
8. Launch the loop with the exact argument string from "The loop itself" section

You are unattended. Be thoughtful, be methodical, be honest in the morning report. Tonight adds new capabilities on top of a hardened pipeline. Good luck.
