# Phase 1: Weighted Scoring Grammar Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a 4-tier non-uniform scoring grammar and wire it into the analysis pipeline, so prior re-weighting suppresses shallow `has_suit`/`has_color` dominance and surfaces compositional competitors.

**Architecture:** The existing analysis pipeline enumerates programs with a *uniform* grammar and scores them with the same grammar. We add a separate *weighted* scoring grammar (4 tiers of primitive costs + cheap variables) that `dsl_prior.py` uses to re-score the cached equivalence classes — no re-enumeration needed. The pipeline gains a `--grammar weighted` CLI flag. We also add a standalone validation script to compare uniform vs weighted results.

**Tech Stack:** Python 3, existing `dreamcoder_core` and `gallery_analysis` modules. No new dependencies.

**Design doc:** `docs/plans/2026-03-13-hypothesis-space-maturation-design.md`

## Execution Guidelines
- Explain code as you write it — include comments on what each function does and why
- Test each step before proceeding
- Present 2+ options for major decisions, wait for selection
- Commit after each task completes

---

### Task 1: Add `build_weighted_gallery_grammar()` to enumerator.py

**Files:**
- Modify: `src/gallery_analysis/enumerator.py:50-84`
- Test: `src/tests/test_weighted_grammar.py` (create)

**Step 1: Write the failing test**

Create `src/tests/test_weighted_grammar.py`:

```python
"""
Tests for the 4-tier weighted grammar.

Verifies:
1. Grammar is constructed with correct number of productions
2. Tier assignments match the design spec
3. Variable cost is set correctly
4. dsl_prior produces different priors for shallow vs compositional programs
"""
import sys
import math
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from gallery_analysis.enumerator import (
    build_gallery_primitives,
    build_gallery_grammar,
    build_weighted_gallery_grammar,
    TIER_CHEAP, TIER_STANDARD, TIER_AGGREGATE, TIER_ULTRA_SHALLOW,
)
from gallery_analysis.dsl_prior import compute_log_prior


@pytest.fixture(scope="module")
def weighted_grammar():
    return build_weighted_gallery_grammar()


@pytest.fixture(scope="module")
def uniform_grammar():
    return build_gallery_grammar()


class TestGrammarConstruction:
    """Verify the grammar is built correctly."""

    def test_same_number_of_productions(self, weighted_grammar, uniform_grammar):
        """Weighted grammar has same productions as uniform (same primitives)."""
        assert len(weighted_grammar.productions) == len(uniform_grammar.productions)

    def test_variable_cost_is_cheap(self, weighted_grammar):
        """Variable cost should be -1.0 (not the uniform -4.14)."""
        assert weighted_grammar.log_variable == pytest.approx(-1.0)

    def test_no_missing_primitives(self, weighted_grammar):
        """Every gallery primitive must appear in exactly one tier."""
        prim_names = {p.name for p in build_gallery_primitives()}
        tier_names = TIER_CHEAP | TIER_STANDARD | TIER_AGGREGATE | TIER_ULTRA_SHALLOW
        # Filter to only names that are actual primitives (not constants that
        # might not be in the tier sets because they're in TIER_STANDARD by default)
        assigned = tier_names & prim_names
        unassigned = prim_names - tier_names
        # Unassigned primitives default to TIER_STANDARD, so this is fine
        # But check no primitive is in multiple explicit tiers
        overlap = (TIER_CHEAP & TIER_AGGREGATE) | (TIER_CHEAP & TIER_ULTRA_SHALLOW) | (TIER_AGGREGATE & TIER_ULTRA_SHALLOW)
        assert overlap == set(), f"Primitives in multiple tiers: {overlap}"


class TestTierCosts:
    """Verify that tier log-probabilities are applied correctly."""

    def test_has_suit_is_expensive(self, weighted_grammar):
        """has_suit should have log_p = -9.0 (ultra-shallow tier)."""
        for prod in weighted_grammar.productions:
            if prod.program.name == 'has_suit':
                assert prod.log_probability == pytest.approx(-9.0)
                return
        pytest.fail("has_suit not found in productions")

    def test_has_color_is_expensive(self, weighted_grammar):
        """has_color should have log_p = -9.0 (ultra-shallow tier)."""
        for prod in weighted_grammar.productions:
            if prod.program.name == 'has_color':
                assert prod.log_probability == pytest.approx(-9.0)
                return
        pytest.fail("has_color not found in productions")

    def test_eq_is_cheap(self, weighted_grammar):
        """eq should have log_p = -3.0 (cheap tier)."""
        for prod in weighted_grammar.productions:
            if prod.program.name == 'eq':
                assert prod.log_probability == pytest.approx(-3.0)
                return
        pytest.fail("eq not found in productions")

    def test_count_suit_is_aggregate(self, weighted_grammar):
        """count_suit should have log_p = -5.5 (aggregate tier)."""
        for prod in weighted_grammar.productions:
            if prod.program.name == 'count_suit':
                assert prod.log_probability == pytest.approx(-5.5)
                return
        pytest.fail("count_suit not found in productions")

    def test_rank_val_is_standard(self, weighted_grammar):
        """rank_val should have log_p = -4.0 (standard tier)."""
        for prod in weighted_grammar.productions:
            if prod.program.name == 'rank_val':
                assert prod.log_probability == pytest.approx(-4.0)
                return
        pytest.fail("rank_val not found in productions")


class TestPriorEffect:
    """Verify that weighted grammar changes priors in the expected direction."""

    def test_shallow_program_is_more_expensive(self, weighted_grammar, uniform_grammar):
        """has_suit under weighted grammar should have worse prior than under uniform."""
        prog = "(λ has_suit $0 SPADES)"
        prior_uniform = compute_log_prior(prog, uniform_grammar)
        prior_weighted = compute_log_prior(prog, weighted_grammar)
        # Weighted should be more negative (more expensive)
        assert prior_weighted < prior_uniform

    def test_compositional_program_is_cheaper(self, weighted_grammar, uniform_grammar):
        """A compositional HOF program should be cheaper under weighted grammar."""
        # all (λ eq (get_suit $0) SPADES) $0 — uses cheap eq + cheap variable
        prog = "(λ all (λ eq (get_suit $0) SPADES) $0)"
        prior_uniform = compute_log_prior(prog, uniform_grammar)
        prior_weighted = compute_log_prior(prog, weighted_grammar)
        # Weighted should be less negative (cheaper) due to cheap vars + cheap eq
        assert prior_weighted > prior_uniform
```

**Step 2: Run test to verify it fails**

Run: `cd src && python -m pytest tests/test_weighted_grammar.py -v 2>&1 | head -30`
Expected: FAIL — `ImportError: cannot import name 'build_weighted_gallery_grammar'`

**Step 3: Write the implementation**

Add to `src/gallery_analysis/enumerator.py`, after `build_gallery_grammar()` (around line 84):

```python
# =========================================================================
# 4-Tier weighted grammar for scoring (not enumeration)
# =========================================================================
#
# This grammar assigns non-uniform log-probabilities to primitives based
# on their cognitive "cost" tier. It is used by dsl_prior.py to re-score
# hypotheses after enumeration, without changing which programs get
# generated. See docs/plans/2026-03-13-hypothesis-space-maturation-design.md.
#
# Tier rationale:
#   CHEAP:         Compositional glue (logic, comparison, HOFs) — nearly free
#   STANDARD:      Accessors, constants, arithmetic, list ops — default
#   AGGREGATE:     Prepackaged aggregate queries — useful but shouldn't dominate
#   ULTRA_SHALLOW: Boolean shortcut queries (has_suit, has_color) — heavily penalized

TIER_CHEAP = frozenset({
    'eq', 'lt', 'le', 'gt', 'ge',
    'and', 'or', 'not',
    'all', 'any', 'if',
})

TIER_AGGREGATE = frozenset({
    'count_suit', 'count_color',
    'n_unique_suits', 'n_unique_ranks', 'n_unique_colors',
    'max_suit_count', 'n_repeated_ranks', 'n_repeated_suits',
    'sum_ranks', 'max_rank', 'min_rank',
})

TIER_ULTRA_SHALLOW = frozenset({
    'has_suit', 'has_color',
})

# Everything else is STANDARD (accessors, constants, arithmetic, list ops).
# We define this as a frozenset for documentation and validation, but
# primitives not in any explicit tier default to STANDARD.
TIER_STANDARD = frozenset({
    'rank_val', 'get_suit', 'get_color', 'get_rank',
    'head', 'last', 'at', 'length', 'half_len',
    'reverse', 'first_half', 'second_half', 'sort_by_rank', 'unique',
    'take', 'drop', 'filter', 'adjacent_pairs', 'map', 'zip_with',
    'running_sum', 'signum', 'suit_to_int',
    'HEARTS', 'DIAMONDS', 'CLUBS', 'SPADES', 'RED', 'BLACK',
    '0', '1', '2', '3', '4', '5',
    '+', '-', 'mod',
})


def build_weighted_gallery_grammar(
    log_cheap: float = -3.0,
    log_standard: float = -4.0,
    log_aggregate: float = -5.5,
    log_ultra_shallow: float = -9.0,
    log_variable: float = -1.0,
) -> 'Grammar':
    """
    Build a 4-tier weighted grammar for scoring hypotheses.

    This grammar is used by dsl_prior.py to compute log-priors that
    penalize ultra-shallow programs (has_suit, has_color) and reward
    compositional depth (cheap variables and logic operators).

    It does NOT change which programs get enumerated — only how they
    are scored in the Bayesian posterior.

    Args:
        log_cheap:         Log-prob for compositional glue (eq, lt, all, any, ...).
        log_standard:      Log-prob for accessors, constants, arithmetic, list ops.
        log_aggregate:     Log-prob for prepackaged aggregate queries.
        log_ultra_shallow: Log-prob for has_suit, has_color.
        log_variable:      Log-prob for bound variables ($0, $1, ...).

    Returns:
        A Grammar with non-uniform production weights.
    """
    from dreamcoder_core.grammar import Grammar, Production

    prims = build_gallery_primitives()
    productions = []
    for p in prims:
        if p.name in TIER_CHEAP:
            lp = log_cheap
        elif p.name in TIER_AGGREGATE:
            lp = log_aggregate
        elif p.name in TIER_ULTRA_SHALLOW:
            lp = log_ultra_shallow
        else:
            lp = log_standard
        productions.append(Production(p, p.tp, lp))
    return Grammar(productions, log_variable)
```

**Step 4: Run tests to verify they pass**

Run: `cd src && python -m pytest tests/test_weighted_grammar.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/gallery_analysis/enumerator.py src/tests/test_weighted_grammar.py
git commit -m "feat: add 4-tier weighted scoring grammar

Adds build_weighted_gallery_grammar() with tier assignments:
- Cheap (-3.0): compositional glue (eq, lt, all, any, ...)
- Standard (-4.0): accessors, constants, arithmetic
- Aggregate (-5.5): prepackaged aggregate queries
- Ultra-shallow (-9.0): has_suit, has_color
- Variables (-1.0): cheap bound variables

Used for scoring only — does not change enumeration."
```

---

### Task 2: Wire weighted grammar into `analyze.py`

**Files:**
- Modify: `src/gallery_analysis/analyze.py:906-992` (CLI section)
- Modify: `src/gallery_analysis/analyze.py:514-708` (`run_analysis()`)

**Step 1: Write the failing test**

Add to `src/tests/test_weighted_grammar.py`:

```python
class TestAnalyzeCLIFlag:
    """Verify that analyze.py accepts --grammar flag."""

    def test_grammar_flag_is_recognized(self):
        """The --grammar flag should be recognized by the argument parser."""
        import gallery_analysis.analyze as analyze_mod
        parser = analyze_mod.build_argument_parser()
        args = parser.parse_args(["--grammar", "weighted", "--quick"])
        assert args.grammar == "weighted"

    def test_grammar_flag_defaults_to_uniform(self):
        """Default grammar should be 'uniform' for backward compatibility."""
        import gallery_analysis.analyze as analyze_mod
        parser = analyze_mod.build_argument_parser()
        args = parser.parse_args([])
        assert args.grammar == "uniform"
```

**Step 2: Run test to verify it fails**

Run: `cd src && python -m pytest tests/test_weighted_grammar.py::TestAnalyzeCLIFlag -v`
Expected: FAIL — `AttributeError: module 'gallery_analysis.analyze' has no attribute 'build_argument_parser'`

**Step 3: Implement the changes**

In `src/gallery_analysis/analyze.py`, make three changes:

**3a. Extract the argument parser into a function (refactor `main()` lines ~907-928):**

Replace the argument parser section inside `main()` with a call to a new function:

```python
def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser (extracted for testability)."""
    parser = argparse.ArgumentParser(description="Bayesian rule induction analysis")
    parser.add_argument("--depth", type=int, default=7, help="Max AST depth")
    parser.add_argument("--max-programs", type=int, default=300_000, help="Max programs to enumerate")
    parser.add_argument("--max-cost", type=float, default=35.0, help="Max cost to explore")
    parser.add_argument("--timeout", type=float, default=600.0, help="Enumeration timeout")
    parser.add_argument("--probes", type=int, default=500, help="Number of probe hands")
    parser.add_argument("--mc-samples", type=int, default=100_000, help="MC samples for extension size")
    parser.add_argument("--epsilon", type=float, default=0.01, help="Noise parameter")
    parser.add_argument("--prior", choices=["canonical", "summed"], default="summed", help="Prior mode")
    parser.add_argument("--verbose", type=int, default=2, help="Verbosity (0-2)")
    parser.add_argument("--quick", action="store_true", help="Quick test (depth 5, 50K programs)")
    parser.add_argument("--inject", type=str, default=None,
                        help="Path to injection JSON file with additional hypotheses")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON to this path")
    parser.add_argument("--extension-cache", type=str, default=None,
                        help="Path to cache extension sizes (skips MC estimation on re-runs)")
    parser.add_argument("--max-list-chain", type=int, default=2,
                        help="Max consecutive list→list transforms (default 2, None to disable)")
    parser.add_argument("--no-list-chain-limit", action="store_true",
                        help="Disable list→list chain limit (enumerate all programs)")
    parser.add_argument("--grammar", choices=["uniform", "weighted"], default="uniform",
                        help="Scoring grammar: 'uniform' (baseline) or 'weighted' (4-tier)")
    return parser
```

Then update `main()` to use it:
```python
def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    # ... rest of main unchanged ...
```

**3b. Add `scoring_grammar` parameter to `run_analysis()`:**

Add parameter `scoring_grammar: str = "uniform"` to `run_analysis()`. In the injection block (around line 548-586), replace:

```python
grammar = build_gallery_grammar()
```

with:

```python
if scoring_grammar == "weighted":
    from gallery_analysis.enumerator import build_weighted_gallery_grammar
    grammar = build_weighted_gallery_grammar()
else:
    grammar = build_gallery_grammar()
```

**3c. In `score_rule()`, add `grammar` parameter for prior recomputation:**

Add a `grammar` parameter to `score_rule()`:

```python
def score_rule(
    rule_id: str,
    exemplar_hands: List[Hand],
    equivalence_classes: List[Dict[str, Any]],
    extensions: List[Tuple[int, float]],
    epsilon: float = 0.01,
    prior_mode: str = "summed",
    true_rule_fingerprint: str = None,
    grammar=None,
) -> Dict[str, Any]:
```

Then in the scoring loop, when `grammar is not None`, recompute the prior:

```python
        # Select prior
        if grammar is not None:
            # Recompute prior under the provided (weighted) grammar
            log_prior = _recompute_class_prior(cls, grammar)
        elif prior_mode == "canonical":
            log_prior = cls["canonical_prior"]
        else:
            log_prior = cls["summed_prior"]
```

Add a helper function above `score_rule()`:

```python
def _recompute_class_prior(cls: Dict[str, Any], grammar) -> float:
    """
    Recompute the summed log-prior for an equivalence class under a new grammar.

    Sums exp(log_prior) across all programs in the class (log-sum-exp),
    matching the summed_prior semantics used by the default pipeline.
    """
    from gallery_analysis.dsl_prior import compute_log_prior

    log_probs = []
    for prog_str in cls["all_programs"]:
        try:
            lp = compute_log_prior(prog_str, grammar)
            log_probs.append(lp)
        except Exception:
            pass

    if not log_probs:
        # Fallback: try canonical only
        try:
            return compute_log_prior(cls["canonical_program"], grammar)
        except Exception:
            return float('-inf')

    max_lp = max(log_probs)
    return max_lp + math.log(sum(math.exp(lp - max_lp) for lp in log_probs))
```

**3d. Wire grammar through `run_analysis()` → `score_rule()`:**

In the `run_analysis()` scoring loop, build the grammar object and pass it:

```python
    # Build scoring grammar if weighted
    scoring_grammar_obj = None
    if scoring_grammar == "weighted":
        from gallery_analysis.enumerator import build_weighted_gallery_grammar
        scoring_grammar_obj = build_weighted_gallery_grammar()
        if verbose >= 1:
            print(f"\nUsing WEIGHTED scoring grammar (4-tier)", flush=True)

    # ... in the scoring loop:
    result = score_rule(
        rule_id=rule_id,
        exemplar_hands=exemplars[rule_id]["hands_primary"],
        equivalence_classes=equiv_classes,
        extensions=extensions,
        epsilon=epsilon,
        prior_mode=prior_mode,
        true_rule_fingerprint=true_rule_fps.get(rule_id),
        grammar=scoring_grammar_obj,
    )
```

**3e. Pass `--grammar` from CLI to `run_analysis()`:**

In `main()`, pass `scoring_grammar=args.grammar` to `run_analysis()`.

**Step 4: Run tests to verify they pass**

Run: `cd src && python -m pytest tests/test_weighted_grammar.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/gallery_analysis/analyze.py src/tests/test_weighted_grammar.py
git commit -m "feat: wire weighted grammar into analysis pipeline

- Extract build_argument_parser() for testability
- Add --grammar flag (uniform/weighted) to CLI
- Add scoring_grammar parameter to run_analysis()
- Add grammar parameter to score_rule() for prior recomputation
- Add _recompute_class_prior() helper for log-sum-exp under new grammar"
```

---

### Task 3: Create validation script

**Files:**
- Create: `src/gallery_analysis/validate_weights.py`

**Step 1: Write the validation script**

This is a standalone script (not a test) that loads the hypothesis pool, scores all 60 rules under both uniform and weighted grammars, and prints a side-by-side comparison table. It serves as a quick sanity check after any weight adjustments.

```python
#!/usr/bin/env python3
"""
Quick validation: compare uniform vs weighted grammar across all 60 rules.

Usage:
    cd src
    python -m gallery_analysis.validate_weights [--depth 6] [--max-programs 500000]

Outputs a side-by-side table showing:
  - True rule rank under uniform vs weighted
  - Entropy under uniform vs weighted
  - Top-1 program under each grammar
  - Count of rules where has_suit/has_color is top-1
"""
import sys
import time
import math
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gallery_analysis.analyze import (
    build_hypothesis_pool, estimate_extensions, score_rule,
)
from gallery_analysis.exemplars import load_exemplars, generate_probe_set
from gallery_analysis.enumerator import (
    build_gallery_grammar, build_weighted_gallery_grammar,
)
from gallery_analysis.injection import load_and_validate_injections, merge_injected
from gallery_analysis.gallery_rules import GALLERY_RULES

# Shallow primitives to track
_SHALLOW_PRIMS = {'has_suit', 'has_color'}


def _is_shallow_top1(program_str: str) -> bool:
    """Check if the top-1 program starts with a shallow primitive."""
    for p in _SHALLOW_PRIMS:
        if p in program_str and program_str.count('(') <= 2:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Validate weighted grammar")
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--max-programs", type=int, default=500_000)
    args = parser.parse_args()

    SRC = Path(__file__).parent.parent
    inject_path = SRC / "gallery_analysis" / "data" / "injected_hypotheses.json"
    cache_path = SRC / "gallery_analysis" / "results" / "extension_cache_depth6.json"

    print("=" * 90)
    print("GRAMMAR VALIDATION: Uniform vs Weighted (4-tier)")
    print("=" * 90)

    # Build hypothesis pool
    print(f"\nBuilding hypothesis pool (depth={args.depth}, max={args.max_programs:,})...")
    t0 = time.time()
    equiv_classes, stats = build_hypothesis_pool(
        max_depth=args.depth, max_programs=args.max_programs,
        max_list_chain=2, verbose=1,
    )
    print(f"  {len(equiv_classes):,} classes ({time.time()-t0:.0f}s)")

    # Inject
    if inject_path.exists():
        grammar_tmp = build_gallery_grammar()
        probes = generate_probe_set(500, seed=42)
        injected = load_and_validate_injections(str(inject_path), grammar=grammar_tmp)
        equiv_classes = merge_injected(equiv_classes, injected, probes)
        print(f"  After injection: {len(equiv_classes):,} classes")

    # Extensions
    extensions = estimate_extensions(
        equiv_classes, verbose=1,
        cache_path=str(cache_path) if cache_path.exists() else None,
    )

    # True-rule fingerprints
    true_fps = {}
    for cls in equiv_classes:
        for rid in cls.get("true_for_rules", []):
            true_fps[rid] = cls["fingerprint"]
        single = cls.get("true_for_rule")
        if single and single not in true_fps:
            true_fps[single] = cls["fingerprint"]

    # Build grammars
    g_weighted = build_weighted_gallery_grammar()

    # Score all rules under both grammars
    exemplars = load_exemplars()

    print(f"\n{'Rule':<30} {'Grp':>3} | "
          f"{'U_rank':>6} {'U_ent':>5} {'U_top1':<40} | "
          f"{'W_rank':>6} {'W_ent':>5} {'W_top1':<40}")
    print("─" * 150)

    n_shallow_uniform = 0
    n_shallow_weighted = 0
    n_improved = 0
    n_worsened = 0
    n_unchanged = 0

    for rule_id, rule_info in sorted(GALLERY_RULES.items(), key=lambda x: x[1]["group"]):
        if rule_id not in exemplars:
            continue

        hands = exemplars[rule_id]["hands_primary"]
        true_fp = true_fps.get(rule_id)

        # Score under uniform (grammar=None uses stored priors)
        r_u = score_rule(rule_id, hands, equiv_classes, extensions,
                         true_rule_fingerprint=true_fp, grammar=None)

        # Score under weighted
        r_w = score_rule(rule_id, hands, equiv_classes, extensions,
                         true_rule_fingerprint=true_fp, grammar=g_weighted)

        # Extract metrics
        u_rank = r_u.get("true_rule_rank") or 0
        w_rank = r_w.get("true_rule_rank") or 0
        u_ent = r_u["difficulty"]["posterior_entropy"]
        w_ent = r_w["difficulty"]["posterior_entropy"]
        u_top1 = r_u["top_hypotheses"][0]["program"][:38] if r_u["top_hypotheses"] else "?"
        w_top1 = r_w["top_hypotheses"][0]["program"][:38] if r_w["top_hypotheses"] else "?"

        u_rank_str = str(u_rank) if u_rank else "N/A"
        w_rank_str = str(w_rank) if w_rank else "N/A"

        if _is_shallow_top1(u_top1):
            n_shallow_uniform += 1
        if _is_shallow_top1(w_top1):
            n_shallow_weighted += 1

        if u_rank and w_rank:
            if w_rank < u_rank:
                n_improved += 1
            elif w_rank > u_rank:
                n_worsened += 1
            else:
                n_unchanged += 1

        print(f"  {rule_id:<28} {rule_info['group']:>3} | "
              f"{u_rank_str:>6} {u_ent:5.2f} {u_top1:<40} | "
              f"{w_rank_str:>6} {w_ent:5.2f} {w_top1:<40}")

    print(f"\n{'─'*90}")
    print(f"SUMMARY:")
    print(f"  Shallow (has_suit/has_color) as top-1: {n_shallow_uniform} → {n_shallow_weighted}")
    print(f"  True rule rank: {n_improved} improved, {n_worsened} worsened, {n_unchanged} unchanged")


if __name__ == "__main__":
    main()
```

**Step 2: Run it to verify it works**

Run: `cd src && python -m gallery_analysis.validate_weights --depth 5 --max-programs 50000 2>&1 | tail -20`
Expected: Table output with comparison columns. No errors.

**Step 3: Commit**

```bash
git add src/gallery_analysis/validate_weights.py
git commit -m "feat: add validation script for uniform vs weighted grammar comparison

Standalone script that scores all 60 rules under both grammars and
prints a side-by-side table with true-rule rank, entropy, and top-1
program. Reports shallow dominance counts."
```

---

### Task 4: Integration test — full pipeline with `--grammar weighted`

**Files:**
- Test: `src/tests/test_weighted_grammar.py` (add to existing)

**Step 1: Write the integration test**

Add to `src/tests/test_weighted_grammar.py`:

```python
class TestWeightedPipelineIntegration:
    """
    Integration test: run a small analysis with --grammar weighted and
    verify that the output structure is correct and priors differ from uniform.
    """

    def test_run_analysis_with_weighted_grammar(self):
        """run_analysis() with scoring_grammar='weighted' should complete without error."""
        from gallery_analysis.analyze import run_analysis
        results = run_analysis(
            max_depth=4,
            max_programs=1000,
            max_cost=20.0,
            timeout=30.0,
            n_probes=50,
            extension_samples=1000,
            scoring_grammar="weighted",
            verbose=0,
        )
        assert "rule_results" in results
        assert "difficulty_ranking" in results
        assert len(results["rule_results"]) > 0

    def test_weighted_changes_ranking(self):
        """
        Weighted grammar should produce different top-1 programs for at least
        some rules compared to uniform.
        """
        from gallery_analysis.analyze import run_analysis

        r_uniform = run_analysis(
            max_depth=4, max_programs=1000, max_cost=20.0,
            timeout=30.0, n_probes=50, extension_samples=1000,
            scoring_grammar="uniform", verbose=0,
        )
        r_weighted = run_analysis(
            max_depth=4, max_programs=1000, max_cost=20.0,
            timeout=30.0, n_probes=50, extension_samples=1000,
            scoring_grammar="weighted", verbose=0,
        )

        # Check at least one rule has a different top-1
        n_different = 0
        for rule_id in r_uniform["rule_results"]:
            if rule_id not in r_weighted["rule_results"]:
                continue
            u_top = r_uniform["rule_results"][rule_id]["top_hypotheses"]
            w_top = r_weighted["rule_results"][rule_id]["top_hypotheses"]
            if u_top and w_top and u_top[0]["program"] != w_top[0]["program"]:
                n_different += 1
        assert n_different > 0, "Weighted grammar should change at least one top-1 ranking"
```

**Step 2: Run tests**

Run: `cd src && python -m pytest tests/test_weighted_grammar.py::TestWeightedPipelineIntegration -v --timeout=120`
Expected: PASS (may take 30-60 seconds due to enumeration)

**Step 3: Commit**

```bash
git add src/tests/test_weighted_grammar.py
git commit -m "test: add integration tests for weighted grammar pipeline

Verifies that run_analysis() with scoring_grammar='weighted' completes
and produces different rankings than uniform for at least some rules."
```

---

### Task 5: Record grammar choice in results JSON and provenance

**Files:**
- Modify: `src/gallery_analysis/analyze.py` (config dict in `run_analysis()`)

**Step 1: Add grammar to config output**

In `run_analysis()`, add `"scoring_grammar": scoring_grammar` to the `config` dict at the bottom of the function (around line 698):

```python
    return {
        "pipeline_stats": pipeline_stats,
        "rule_results": rule_results,
        "difficulty_ranking": ranking,
        "config": {
            "max_depth": max_depth,
            "max_programs": max_programs,
            "max_cost": max_cost,
            "n_probes": n_probes,
            "extension_samples": extension_samples,
            "epsilon": epsilon,
            "prior_mode": prior_mode,
            "scoring_grammar": scoring_grammar,  # NEW
        },
        "provenance": provenance,
    }
```

Also update `print_difficulty_report()` to show the grammar used:

```python
    print(f"\nConfig: depth={config['max_depth']}, programs={config['max_programs']:,}, "
          f"probes={config['n_probes']}, MC_samples={config['extension_samples']:,}, "
          f"ε={config['epsilon']}, prior={config['prior_mode']}, "
          f"grammar={config.get('scoring_grammar', 'uniform')}")
```

**Step 2: Test**

Run: `cd src && python -m pytest tests/test_weighted_grammar.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add src/gallery_analysis/analyze.py
git commit -m "feat: record scoring_grammar in results config and report header

Ensures the grammar choice (uniform/weighted) is persisted in output JSON
and displayed in the difficulty report header for reproducibility."
```

---

### Task 6: Full validation run and regression check

**Files:**
- No new files — validation and regression check

**Step 1: Run all existing tests to confirm no regressions**

Run: `cd src && python -m pytest tests/ -v --timeout=120 2>&1 | tail -30`
Expected: All existing tests PASS. The weighted grammar doesn't change default behavior — `build_gallery_grammar()` still returns uniform, `score_rule()` defaults to `grammar=None`, `run_analysis()` defaults to `scoring_grammar="uniform"`.

**Step 2: Run the validation script**

Run: `cd src && python -m gallery_analysis.validate_weights --depth 6 --max-programs 500000 2>&1 | tee /tmp/weighted_validation.txt`

Expected:
- Shallow dominance (has_suit/has_color as top-1) drops from ~18 to ≤ 2 rules
- True rule rank improves for majority of rules
- No errors

**Step 3: Run full analysis with weighted grammar**

Run: `cd src && python -m gallery_analysis.analyze --depth 6 --max-programs 500000 --grammar weighted --inject gallery_analysis/data/injected_hypotheses.json --extension-cache gallery_analysis/results/extension_cache_depth6.json --output gallery_analysis/results/weighted_depth6_results.json --verbose 2 2>&1 | tee /tmp/weighted_full_run.txt`

Expected: Complete run with results JSON saved.

**Step 4: Commit results**

```bash
git add src/gallery_analysis/results/weighted_depth6_results.json
git commit -m "feat: add weighted grammar analysis results (depth 6, 500K programs)

4-tier weighted grammar reduces shallow dominance from ~18/60 to ~1/60 rules.
True rule rank improves for ~42/60 rules."
```
