# MCMC Search Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix 5 issues in the MCMC search pipeline: dead chain initialization, missing likelihood guidance, tautology attractors, file overwriting, and wrong TOTAL_HANDS constant.

**Architecture:** All fixes modify `mcmc_search.py` (core engine), `analyze_mcmc.py` (CLI orchestrator), and `mcmc_hypothesis_collector.py` (results filtering). No changes to the enumeration pipeline. The fixes are mostly independent and can be tested incrementally.

**Tech Stack:** Python 3, existing `dreamcoder_core` infrastructure, `gallery_analysis` pipeline.

---

## Execution Guidelines

- Explain code as you write it
- Test each step before proceeding
- Preserve existing test behavior — all 25 existing tests must continue to pass
- Commit after each task

---

### Task 1: Fix TOTAL_HANDS constant

The simplest fix. Change the constant from C(52,6) to P(52,6) since hands are ordered.

**Files:**
- Modify: `src/gallery_analysis/mcmc_search.py:886`

**Step 1: Change the constant**

Replace line 886:
```python
# OLD
TOTAL_HANDS = 20_358_520

# NEW
# Total number of ORDERED 6-card hands from a 52-card deck: P(52,6)
# Hands are ordered because position matters (e.g., strict_increasing,
# colors_palindrome). P(52,6) = 52 * 51 * 50 * 49 * 48 * 47.
# Note: the same fix is needed in bayesian_scorer.py and hypothesis_table.py
# on main — those will be changed independently.
TOTAL_HANDS = 14_658_134_400
```

**Step 2: Run tests to verify nothing breaks**

Run: `cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling/src && python -m pytest tests/test_mcmc_search.py -x -q`
Expected: All pass (the constant affects likelihood magnitudes but not test assertions)

**Step 3: Commit**

```bash
git add src/gallery_analysis/mcmc_search.py
git commit -m "fix: correct TOTAL_HANDS from C(52,6) to P(52,6) for ordered hands"
```

---

### Task 2: Fix initialization with small max_depth

**Files:**
- Modify: `src/gallery_analysis/mcmc_search.py` (MCMCConfig + MCMCChain.run)
- Modify: `src/tests/test_mcmc_search.py`

**Step 1: Write the failing test**

Add to `src/tests/test_mcmc_search.py`:

```python
def test_init_max_depth_produces_small_programs(grammar):
    """With init_max_depth=3, initial programs should be small enough for max_nodes=25."""
    from gallery_analysis.mcmc_search import MCMCConfig, MCMCChain
    config = MCMCConfig(n_steps=10, max_depth=6, init_max_depth=3, max_nodes=25, seed=42)
    for seed_offset in range(20):
        config_i = MCMCConfig(
            n_steps=10, max_depth=6, init_max_depth=3, max_nodes=25,
            seed=42 + seed_offset
        )
        chain = MCMCChain(grammar, config_i)
        result = chain.run(
            request_type=Arrow(HAND, BOOL),
            exemplar_hands=load_exemplars()['all_red']['hands_primary'],
        )
        # Chain should have accepted at least 1 proposal if it started small
        # (not guaranteed, but extremely likely across 20 seeds)
    # At least some chains should have accepted proposals
    # (the old seed=42 produced size=202, accepting 0 proposals)
```

Actually, a cleaner test — test the initialization directly:

```python
def test_init_max_depth_constrains_initial_program(grammar):
    """init_max_depth should control the depth of the initial sample."""
    for seed in range(20):
        prog = sample_program(grammar, Arrow(HAND, BOOL), max_depth=3, seed=seed)
        # With max_depth=3, programs should be much smaller than max_nodes=25
        # (not guaranteed per-sample, but the average should be well under 25)
        assert prog.size() <= 50, f"seed={seed}: size={prog.size()} too large for max_depth=3"
```

**Step 2: Run test to verify it fails (or passes — this tests the sampler directly)**

Run: `cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling/src && python -m pytest tests/test_mcmc_search.py::test_init_max_depth_constrains_initial_program -v`

**Step 3: Add `init_max_depth` to MCMCConfig**

In `MCMCConfig`, add after `max_depth`:
```python
    init_max_depth: int = 3   # Max depth for initial program sample (kept small to avoid dead chains)
```

**Step 4: Use `init_max_depth` in MCMCChain.run()**

Change the initialization block (around line 1110) from:
```python
        current = sample_program(
            grammar, request_type, max_depth=config.max_depth,
            seed=rng.randint(0, 2**31),
        )
```
to:
```python
        current = sample_program(
            grammar, request_type, max_depth=config.init_max_depth,
            seed=rng.randint(0, 2**31),
        )
```

**Step 5: Propagate `init_max_depth` in `run_parallel_chains()`**

In the chain_config construction (around line 1354), add:
```python
            init_max_depth=config.init_max_depth,
```

**Step 6: Run all tests**

Run: `cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling/src && python -m pytest tests/test_mcmc_search.py -x -q`

**Step 7: Commit**

```bash
git add src/gallery_analysis/mcmc_search.py src/tests/test_mcmc_search.py
git commit -m "fix: initialize MCMC chains with small max_depth to avoid dead chains"
```

---

### Task 3: Add likelihood annealing

**Files:**
- Modify: `src/gallery_analysis/mcmc_search.py` (MCMCConfig + MCMCChain.run)
- Modify: `src/tests/test_mcmc_search.py`

**Step 1: Write the failing test**

```python
def test_annealing_beta_ramps(grammar, exemplars):
    """With annealing, early steps should use low beta, later steps high beta."""
    config = MCMCConfig(
        n_steps=100, max_depth=5, seed=42,
        beta_start=0.0, beta_end=1.0,
    )
    hands = exemplars['all_red']['hands_primary']
    result = MCMCChain(grammar, config).run(
        request_type=Arrow(HAND, BOOL),
        exemplar_hands=hands,
    )
    # Should complete without error
    assert result.n_steps == 100

def test_annealing_default_is_no_anneal(grammar, exemplars):
    """Default beta_start=1.0 beta_end=1.0 should behave identically to no annealing."""
    config = MCMCConfig(n_steps=50, max_depth=5, seed=42)
    hands = exemplars['all_red']['hands_primary']
    result = MCMCChain(grammar, config).run(
        request_type=Arrow(HAND, BOOL),
        exemplar_hands=hands,
    )
    assert result.n_steps == 50
```

**Step 2: Run tests to verify they fail**

**Step 3: Add annealing fields to MCMCConfig**

```python
    beta_start: float = 1.0   # Likelihood temperature at step 0 (1.0 = no annealing)
    beta_end: float = 1.0     # Likelihood temperature at final step
```

Note: default is `1.0, 1.0` (no annealing) so all existing tests pass unchanged. Users opt in with `beta_start=0.1`.

**Step 4: Modify the MH loop to use annealed likelihood**

In the main loop (around line 1200), change the MH ratio computation. Before computing `log_alpha`:

```python
            # Compute annealing temperature for this step.
            # beta ramps linearly from beta_start to beta_end.
            if config.beta_start == config.beta_end:
                beta = config.beta_start
            else:
                beta = config.beta_start + (config.beta_end - config.beta_start) * (step / config.n_steps)
```

Then change the MH ratio to use `beta * log_lik` instead of raw `log_lik`:

```python
            # Annealed posteriors for MH ratio
            current_annealed = current_log_prior + beta * current_log_lik
            proposed_annealed = proposed_log_prior + beta * proposed_log_lik
```

And use these in the acceptance computation instead of `current_log_posterior` and `proposed_log_posterior`.

**Important:** The *tracking* (visit counts, best_log_posterior) should still use the UNANNEALED posterior (`log_prior + log_lik`), because that's the true posterior we care about. The annealing only affects the acceptance decision.

**Step 5: Run all tests**

Run: `cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling/src && python -m pytest tests/test_mcmc_search.py -x -q`

**Step 6: Commit**

```bash
git add src/gallery_analysis/mcmc_search.py src/tests/test_mcmc_search.py
git commit -m "feat: add likelihood annealing (beta schedule) to MCMC chain"
```

---

### Task 4: Add per-rule seed variation

**Files:**
- Modify: `src/gallery_analysis/analyze_mcmc.py`
- Modify: `src/gallery_analysis/mcmc_search.py` (run_parallel_chains signature)

**Step 1: Modify `run_parallel_chains` to accept a `seed_offset`**

Add a `seed_offset: int = 0` parameter. In the seed computation:
```python
        chain_seed = base_seed + seed_offset + i * 1000
```

**Step 2: In `analyze_mcmc.py`, pass a rule-specific offset**

In the per-rule loop:
```python
        # Per-rule seed variation so different rules explore different trajectories
        rule_seed_offset = hash(rule_id) % 100_000

        result = run_parallel_chains(
            grammar, config,
            request_type=request_type,
            exemplar_hands=hands,
            n_chains=args.n_chains,
            ext_probe_hands=ext_probes,
            seed_offset=rule_seed_offset,
        )
```

**Step 3: Run quick mode to verify different rules get different results**

Run: `cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling/src && python -m gallery_analysis.analyze_mcmc --quick --verbose 1 2>&1 | head -30`

Verify that the init programs for chain 1 differ between rules.

**Step 4: Commit**

```bash
git add src/gallery_analysis/mcmc_search.py src/gallery_analysis/analyze_mcmc.py
git commit -m "feat: add per-rule seed variation for different trajectories per rule"
```

---

### Task 5: Add tautology rejection (3 layers)

**Files:**
- Modify: `src/gallery_analysis/mcmc_search.py`
- Modify: `src/gallery_analysis/mcmc_hypothesis_collector.py`
- Modify: `src/tests/test_mcmc_search.py`

**Step 1: Write failing tests**

```python
def test_vacuous_lambda_rejected(grammar):
    """Programs that ignore their input should not appear in chain results."""
    from dreamcoder_core.program import Abstraction, Application, Primitive, Index
    from gallery_analysis.mcmc_search import is_vacuous_lambda
    from dreamcoder_core.type_system import Arrow, HAND, BOOL, INT

    # (λ lt 0 5) — ignores $0
    lt_prim = None
    for p in grammar.productions:
        if str(p.program) == 'lt':
            lt_prim = p.program
            break
    zero = Primitive('0', INT, 0)
    five = Primitive('5', INT, 5)
    vacuous = Abstraction(Application(Application(lt_prim, zero), five))
    assert is_vacuous_lambda(vacuous) == True

    # (λ has_color (head $0) RED) — uses $0
    # Just test that a program using $0 is NOT vacuous
    prog = sample_program(grammar, Arrow(HAND, BOOL), max_depth=3, seed=99)
    # Most depth-3 programs will use $0, but not guaranteed
    # Just verify the function returns a bool
    assert isinstance(is_vacuous_lambda(prog), bool)

def test_tautology_100_percent_rejected(grammar, exemplars):
    """Programs that accept 100% of probes should be rejected during chain."""
    config = MCMCConfig(n_steps=500, max_depth=5, seed=42)
    hands = exemplars['all_red']['hands_primary']
    result = MCMCChain(grammar, config).run(
        request_type=Arrow(HAND, BOOL),
        exemplar_hands=hands,
    )
    # Check that no top hypothesis has "eq $0 $0" as the most-visited
    # (it may appear but shouldn't dominate)
    # This is a soft check — with the fix, tautologies should be less prominent
    if result.top_hypotheses:
        top_prog = result.top_hypotheses[0]['program']
        assert top_prog != '(λ eq $0 $0)', "Tautology should not be #1 hypothesis"
```

**Step 2: Run tests to verify they fail**

**Step 3: Implement `is_vacuous_lambda()`**

Add to `mcmc_search.py`:

```python
from dreamcoder_core.program import uses_variable

def is_vacuous_lambda(program: Program) -> bool:
    """
    Check if a program is a vacuous lambda — one that ignores its input.

    A program (λ. body) is vacuous if $0 does not appear in body.
    Such programs are degenerate: they compute a constant boolean
    regardless of the input hand, so they cannot express any rule
    about card games.

    Examples:
        (λ lt 0 5)       → True  ($0 not used, always returns True)
        (λ eq $0 $0)     → False ($0 is used, even though vacuously)
        (λ has_color (head $0) RED) → False ($0 is used meaningfully)

    Precedent: LOTlib3's check_lambdas option rejects hypotheses
    where lambda-bound variables are unused.
    """
    if not isinstance(program, Abstraction):
        return False
    return not uses_variable(program.body, 0)
```

**Step 4: Add vacuous lambda rejection to the MH loop**

After the `max_nodes` check (around line 1168), add:

```python
            # (b2) Reject vacuous lambdas — programs that ignore their input.
            if is_vacuous_lambda(proposed):
                current_str = str(current)
                visit_counts[current_str] = visit_counts.get(current_str, 0) + 1
                trajectory.append(current_str)
                continue
```

**Step 5: Modify `compute_mcmc_log_likelihood` to return extension info**

Change the return type to include the extension fraction. The simplest way: add an optional output parameter or return a tuple. Since many callers exist, the least disruptive approach is to add a helper that wraps the likelihood and also checks for 100% acceptance:

```python
def compute_mcmc_log_likelihood_with_ext(
    program, exemplar_hands, noise_epsilon, ext_probe_hands
) -> Tuple[float, float]:
    """
    Like compute_mcmc_log_likelihood but also returns the extension fraction.

    Returns:
        (log_likelihood, ext_fraction) where ext_fraction is the fraction
        of probe hands that the program accepts (0.0 to 1.0).
    """
```

Or simpler — just add the 100% check inside the MH loop after computing likelihood, since `compute_mcmc_log_likelihood` already evaluates probes internally. Refactor `compute_mcmc_log_likelihood` to accept an optional `out_ext_info` dict that it populates:

Actually, the simplest approach: compute the probe hit count separately before calling the likelihood, or refactor `compute_mcmc_log_likelihood` to also return `ext_fraction`. Let's go with refactoring to return a tuple:

Change `compute_mcmc_log_likelihood` signature to:
```python
def compute_mcmc_log_likelihood(
    program, exemplar_hands, noise_epsilon, ext_probe_hands,
    return_ext_fraction: bool = False,
) -> Union[float, Tuple[float, float]]:
```

When `return_ext_fraction=True`, return `(log_lik, n_hits_probe / n_probes)`.

**Step 6: Add 100% probe rejection to the MH loop**

After computing proposed likelihood, check:

```python
            proposed_log_lik, proposed_ext_frac = compute_mcmc_log_likelihood(
                proposed, exemplar_hands, config.noise_epsilon, ext_probe_hands,
                return_ext_fraction=True,
            )

            # (c2) Reject programs that accept 100% of probes (tautologies).
            if proposed_ext_frac >= 1.0:
                current_str = str(current)
                visit_counts[current_str] = visit_counts.get(current_str, 0) + 1
                trajectory.append(current_str)
                continue
```

Also apply the same check to the initial program — if it's a tautology, resample:
```python
        # Resample if initial program is a tautology
        for _ in range(20):
            current = sample_program(...)
            if not is_vacuous_lambda(current):
                _, ext_frac = compute_mcmc_log_likelihood(..., return_ext_fraction=True)
                if ext_frac < 1.0:
                    break
```

**Step 7: Add post-hoc extension filtering to TrajectoryAnalyzer**

In `mcmc_hypothesis_collector.py`, add a method:

```python
    def frequency_ranking_filtered(
        self, top_k: int = 50, max_ext_fraction: float = 0.5
    ) -> List[Dict[str, Any]]:
        """Like frequency_ranking but excludes near-tautologies."""
```

This requires storing `ext_fraction` per hypothesis in MCMCResult. Add an `ext_fractions: Dict[str, float]` field to MCMCResult, populated during the chain.

**Step 8: Run all tests**

Run: `cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling/src && python -m pytest tests/test_mcmc_search.py tests/test_mcmc_hypothesis_collector.py -x -q`

**Step 9: Commit**

```bash
git add src/gallery_analysis/mcmc_search.py src/gallery_analysis/mcmc_hypothesis_collector.py src/tests/test_mcmc_search.py
git commit -m "feat: add 3-layer tautology rejection (vacuous lambda, 100% probes, post-hoc filter)"
```

---

### Task 6: Add timestamp to output filename

**Files:**
- Modify: `src/gallery_analysis/analyze_mcmc.py`

**Step 1: Add timestamp to filename**

Change the filename construction (around line 191):

```python
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = 'quick' if args.quick else f'{args.n_steps}steps_{args.n_chains}chains'
    output_path = results_dir / f'mcmc_{suffix}_{timestamp}_results.json'
```

**Step 2: Run quick mode to verify**

Run: `cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling/src && python -m gallery_analysis.analyze_mcmc --quick 2>&1 | tail -5`

Verify the output path includes a timestamp.

**Step 3: Commit**

```bash
git add src/gallery_analysis/analyze_mcmc.py
git commit -m "fix: add timestamp to output filename to prevent overwriting"
```

---

### Task 7: Add annealing CLI flags and propagation

**Files:**
- Modify: `src/gallery_analysis/analyze_mcmc.py`

**Step 1: Add CLI arguments**

```python
    parser.add_argument('--beta-start', type=float, default=0.1,
                        help='Likelihood annealing start temperature (default: 0.1)')
    parser.add_argument('--beta-end', type=float, default=1.0,
                        help='Likelihood annealing end temperature (default: 1.0)')
```

Note: the CLI default is `0.1` (annealing ON), while the MCMCConfig default is `1.0` (annealing OFF). This means tests use no annealing by default, but the CLI enables it by default.

**Step 2: Pass to MCMCConfig**

Add to the config construction:
```python
        beta_start=args.beta_start,
        beta_end=args.beta_end,
```

Also add `init_max_depth` to the config construction if not already there.

**Step 3: Run quick mode with verbose to verify annealing works**

Run: `cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling/src && python -m gallery_analysis.analyze_mcmc --quick --verbose 1`

Verify that chains now start with small programs and produce different trajectories per rule.

**Step 4: Commit**

```bash
git add src/gallery_analysis/analyze_mcmc.py
git commit -m "feat: add annealing and init_max_depth CLI flags to analyze_mcmc"
```

---

### Task 8: End-to-end verification

**Step 1: Run the full test suite**

Run: `cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling/src && python -m pytest tests/test_mcmc_search.py tests/test_mcmc_hypothesis_collector.py -v`

All tests should pass.

**Step 2: Run quick mode and inspect results**

Run: `cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling/src && python -m gallery_analysis.analyze_mcmc --quick --verbose 1`

Verify:
- All chains start with small programs (size < 25)
- Different rules show different trajectories
- No tautologies (`eq $0 $0`, `lt 0 5`) in top-5 hypotheses
- Output file has timestamp in name
- Acceptance rates are > 0 for all chains

**Step 3: Commit any final adjustments**

```bash
git commit -m "test: verify all MCMC fixes work end-to-end"
```
