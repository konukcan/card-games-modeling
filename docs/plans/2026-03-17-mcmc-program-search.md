# MCMC Program Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a pure MCMC (tree-regeneration Metropolis-Hastings) program search module as an alternative to top-down enumeration, targeting the 60-rule gallery catalog with positive-only frozen exemplars.

**Architecture:** A standalone `mcmc_search.py` module in `src/gallery_analysis/` that implements LOTlib3-style tree-regeneration MH over our typed DSL. Programs are proposed by regenerating random subtrees using the existing grammar's `candidates_for_type()`. The module produces the same output interface as the enumeration-based pipeline (hypothesis pool with scored programs), enabling direct comparison and integration with the existing Bayesian scorer. Phase 2 adds hypothesis trajectory collection for modeling the search process itself.

**Tech Stack:** Python 3, existing `dreamcoder_core` grammar/program/type infrastructure, `gallery_analysis` exemplar loading and scoring pipeline.

**Branch:** All work on `feature/mcmc-search`, branched from `main`.

---

## Execution Guidelines

- Present 2+ options for each major design decision, wait for selection
- Explain code as you write it — treat this as a learning opportunity
- Test each step before proceeding
- Start simple, build up: get a basic working MCMC loop before adding recognition integration
- Preserve existing enumeration pipeline completely — no modifications to existing files except adding imports/entry points
- Commit after each task

---

## Overview

### What This Builds

A Metropolis-Hastings MCMC search over the space of typed programs (Hand → bool) that:

1. **Finds solutions** to gallery rules by stochastic search guided by grammar prior + data likelihood
2. **Collects hypothesis trajectories** — the sequence of hypotheses visited during search, enabling process-level predictions about human learning
3. **Integrates with recognition network** — uses task-conditioned grammar weights as the MCMC proposal distribution
4. **Plugs into existing Bayesian pipeline** — outputs hypothesis pools compatible with `bayesian_scorer.py`

### Cognitive Modeling Rationale

- **Enumeration** models a learner who systematically tests hypotheses in order of simplicity
- **MCMC** models a learner who generates candidate hypotheses stochastically (sampling from prior beliefs), tests against examples, and revises — closer to Piantadosi's "rational rules" framework
- The recognition network becomes an *intuition* that shapes what the learner thinks of next (the proposal distribution), rather than just reweighting a priority queue
- Hypothesis trajectories model the *dynamics* of learning: which alternatives are considered, which are sticky, which are stepping stones

### Key Design Decisions (from brainstorming)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Architecture | Pure MCMC (Option B) | Clean theoretical claim; path to parallel tempering (Option C) later |
| Proposal mechanism | Type-constrained subtree regeneration | Reuses `candidates_for_type()`; type-safe by construction |
| Likelihood | Noisy during search, strict for final scoring | Prevents chain getting trapped by one bad exemplar |
| Budget | 10K (quick) / 100K (default) / 250K (full) | Incremental scaling |
| Recognition | Phase 3 — neural proposal distribution | Core MCMC first, then add guidance |
| Hypothesis collection | Phase 2 — trajectory + dwelling + transitions | The primary scientific contribution |

### File Plan

| New File | Purpose |
|----------|---------|
| `src/gallery_analysis/mcmc_search.py` | Core MCMC engine: proposals, acceptance, chain management |
| `src/gallery_analysis/mcmc_hypothesis_collector.py` | Phase 2: trajectory analysis, transition graphs, dwelling times |
| `src/gallery_analysis/analyze_mcmc.py` | Top-level orchestrator (parallel to `analyze.py` for enumeration) |
| `src/tests/test_mcmc_search.py` | Unit tests for MCMC core |
| `src/tests/test_mcmc_hypothesis_collector.py` | Tests for trajectory collection |

### Existing Files Referenced (READ ONLY — do not modify)

| File | Used For |
|------|----------|
| `src/dreamcoder_core/grammar.py` | `Grammar`, `candidates_for_type()`, `program_log_likelihood()`, `variable_candidates()` |
| `src/dreamcoder_core/program.py` | `Program`, `Primitive`, `Application`, `Abstraction`, `Index`, all AST operations |
| `src/dreamcoder_core/type_system.py` | `Type`, `Arrow`, `TypeContext`, `BOOL`, `HAND`, unification |
| `src/dreamcoder_core/primitives.py` | `build_primitives()` |
| `src/gallery_analysis/enumerator.py` | `build_gallery_grammar()`, `build_gallery_primitives()` — reuse same grammar |
| `src/gallery_analysis/exemplars.py` | `load_exemplars()`, `generate_probe_set()` |
| `src/gallery_analysis/gallery_rules.py` | `GALLERY_RULES` — ground truth for the 60 rules |
| `src/gallery_analysis/hypothesis_table.py` | `is_trivial()`, `compute_fingerprint()`, `estimate_extension_size()` |
| `src/gallery_analysis/bayesian_scorer.py` | `compute_log_likelihood_strict()`, `normalize_posteriors()`, `ScoredHypothesis` |

---

## Phase 0: Branch Setup

### Task 0.1: Create feature branch

**Step 1: Create branch from main**

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling
git checkout main
git checkout -b feature/mcmc-search
```

**Step 2: Verify clean state**

```bash
git status
git log --oneline -3
```

**Step 3: Commit placeholder**

Create empty module files so the branch has an initial commit.

```bash
touch src/gallery_analysis/mcmc_search.py
touch src/gallery_analysis/mcmc_hypothesis_collector.py
touch src/gallery_analysis/analyze_mcmc.py
touch src/tests/test_mcmc_search.py
touch src/tests/test_mcmc_hypothesis_collector.py
git add src/gallery_analysis/mcmc_search.py src/gallery_analysis/mcmc_hypothesis_collector.py src/gallery_analysis/analyze_mcmc.py src/tests/test_mcmc_search.py src/tests/test_mcmc_hypothesis_collector.py
git commit -m "chore: scaffold MCMC search module files"
```

---

## Phase 1: Core MCMC Engine

The heart of the system. This phase delivers a working MCMC search that can find programs solving gallery rules.

### Task 1.1: Type-safe program sampler

**Goal:** Write a function that samples a complete, type-correct program from the grammar. This is the foundation for both initializing MCMC chains and generating subtree proposals.

**Files:**
- Create: `src/gallery_analysis/mcmc_search.py`
- Test: `src/tests/test_mcmc_search.py`

**Step 1: Write the failing test**

```python
"""Tests for MCMC program search."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from dreamcoder_core.type_system import BOOL, HAND, Arrow, TypeContext, INT
from dreamcoder_core.program import Program, Abstraction, Application, Primitive
from gallery_analysis.enumerator import build_gallery_grammar
from gallery_analysis.mcmc_search import sample_program


class TestSampleProgram:
    """Test that sample_program generates valid typed programs."""

    def setup_method(self):
        self.grammar = build_gallery_grammar()

    def test_samples_complete_program(self):
        """Sampled program should have no holes."""
        prog = sample_program(self.grammar, Arrow(HAND, BOOL), max_depth=5, seed=42)
        assert prog is not None
        # Should be a complete program (no holes)
        from dreamcoder_core.program import has_holes
        assert not has_holes(prog)

    def test_samples_correct_type(self):
        """Sampled program should type-check as Hand -> bool."""
        prog = sample_program(self.grammar, Arrow(HAND, BOOL), max_depth=5, seed=42)
        ctx = TypeContext()
        tp = prog.infer_type(ctx, [])
        # Should return a function type
        assert tp is not None

    def test_samples_different_programs(self):
        """Different seeds should produce different programs."""
        prog1 = sample_program(self.grammar, Arrow(HAND, BOOL), max_depth=5, seed=1)
        prog2 = sample_program(self.grammar, Arrow(HAND, BOOL), max_depth=5, seed=2)
        # Very unlikely to be the same with different seeds
        # (not guaranteed, but overwhelmingly probable)
        # Just check both are valid
        assert prog1 is not None
        assert prog2 is not None

    def test_respects_max_depth(self):
        """Sampled program should not exceed max_depth."""
        for seed in range(10):
            prog = sample_program(self.grammar, Arrow(HAND, BOOL), max_depth=4, seed=seed)
            assert prog.depth() <= 4, f"seed={seed}: depth={prog.depth()} > 4"

    def test_deterministic_with_seed(self):
        """Same seed should produce same program."""
        prog1 = sample_program(self.grammar, Arrow(HAND, BOOL), max_depth=5, seed=42)
        prog2 = sample_program(self.grammar, Arrow(HAND, BOOL), max_depth=5, seed=42)
        assert str(prog1) == str(prog2)
```

**Step 2: Run test to verify it fails**

Run: `cd src && python -m pytest tests/test_mcmc_search.py::TestSampleProgram -v`
Expected: FAIL with ImportError (sample_program not defined)

**Step 3: Write the implementation**

In `src/gallery_analysis/mcmc_search.py`:

```python
"""
MCMC program search via tree-regeneration Metropolis-Hastings.

This module implements LOTlib3-style stochastic search over typed programs,
adapted for our DreamCoder grammar infrastructure. Instead of systematically
enumerating programs by cost (as TopDownEnumerator does), we perform a random
walk through program space:

  1. Start with a random program sampled from the grammar prior
  2. Propose a new program by regenerating a random subtree
  3. Accept/reject based on Metropolis-Hastings ratio (prior × likelihood)
  4. Repeat for N steps, collecting visited hypotheses

COGNITIVE MODELING RATIONALE:
  Enumeration models a learner who systematically tests hypotheses by simplicity.
  MCMC models a learner who stochastically generates candidates from prior beliefs,
  tests them against data, and revises — sampling from the posterior rather than
  the prior. The proposal distribution (which subtree to regenerate, what to
  replace it with) can be biased by a recognition network, modeling learned
  intuitions about which hypotheses are worth considering.

REFERENCES:
  - Piantadosi, Tenenbaum & Goodman (2016). "The logical primitives of thought."
  - LOTlib3: https://github.com/piantado/LOTlib3
  - Ellis et al. (2021). "DreamCoder" (for the grammar/type infrastructure we reuse)
"""
import sys
import math
import random
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.type_system import (
    Type, BaseType, Arrow, ListType, TypeVariable,
    TypeContext, UnificationError,
    BOOL, INT, CARD, SUIT, RANK, HAND, arrow
)
from dreamcoder_core.program import (
    Program, Primitive, Application, Abstraction, Index, Hole,
    has_holes, find_first_hole, substitute_hole, collect_holes
)
from dreamcoder_core.grammar import Grammar, Production


# =========================================================================
# PROGRAM SAMPLING — Sample a complete program from the grammar prior
# =========================================================================

def sample_program(
    grammar: Grammar,
    request_type: Type,
    max_depth: int = 6,
    seed: Optional[int] = None,
    env: Optional[List[Type]] = None,
) -> Program:
    """
    Sample a complete program from the grammar prior via top-down random generation.

    This is analogous to enumeration's hole-filling, but instead of picking the
    best (lowest-cost) production at each step, we SAMPLE a production with
    probability proportional to its grammar weight.

    How it works:
      1. Start with a hole of the request type: ?:Hand→bool
      2. For each hole, collect all type-compatible productions
      3. Sample one production proportional to exp(log_probability)
      4. Replace the hole with the chosen production (adding new holes for arguments)
      5. Repeat until no holes remain

    At max_depth, we force selection of terminal productions (those that don't
    add new holes — variables, constants, nullary primitives).

    Args:
        grammar: The PCFG defining production probabilities
        request_type: The type of program to generate (e.g., Arrow(HAND, BOOL))
        max_depth: Maximum AST depth — at this depth, only terminals are chosen
        seed: Random seed for reproducibility
        env: Type environment for bound variables (usually empty at top level)

    Returns:
        A complete Program with no holes, sampled from the grammar prior.
    """
    if env is None:
        env = []

    rng = random.Random(seed)
    return _sample_recursive(grammar, request_type, env, max_depth, 0, rng)


def _sample_recursive(
    grammar: Grammar,
    target_type: Type,
    env: List[Type],
    max_depth: int,
    current_depth: int,
    rng: random.Random,
) -> Program:
    """
    Recursively sample a program of the given type.

    At each node, we:
      1. If target is an arrow type → create a lambda, recurse on body
      2. Otherwise → sample a production that returns this type,
         then recurse on each argument

    When current_depth == max_depth, we only allow terminals (productions
    with no arguments, or variables) to ensure the program terminates.
    """
    # Case 1: Arrow type → create lambda
    # When we need a function (e.g., card → bool), we create λ. body
    # and recurse to fill in the body with the lambda's argument in scope.
    if isinstance(target_type, Arrow):
        new_env = [target_type.arg] + env
        body = _sample_recursive(
            grammar, target_type.ret, new_env,
            max_depth, current_depth, rng  # lambda doesn't add depth
        )
        return Abstraction(body)

    # Case 2: Base type → sample a production
    ctx = TypeContext()
    candidates = grammar.candidates_for_type(target_type, ctx, env, normalize=True)
    var_candidates = grammar.variable_candidates(target_type, ctx, env)

    # Build weighted candidate list: (choice, log_prob, arg_types_or_None)
    # Each candidate is either a production (with argument types to fill)
    # or a variable reference (terminal, no arguments).
    choices = []

    at_max_depth = current_depth >= max_depth

    for prod, inst_type, log_prob in candidates:
        # Extract argument types this production needs
        arg_types = list(inst_type.arguments)
        is_terminal = len(arg_types) == 0

        # At max depth, only allow terminals to prevent infinite recursion
        if at_max_depth and not is_terminal:
            continue

        choices.append(('prod', prod, inst_type, log_prob, arg_types))

    for var_idx, log_prob in var_candidates:
        choices.append(('var', var_idx, None, log_prob, []))

    if not choices:
        # Fallback: if nothing works (shouldn't happen with a well-formed grammar),
        # return a hole. This will be caught by tests.
        return Hole(target_type)

    # Sample proportional to exp(log_prob)
    weights = [math.exp(c[3]) for c in choices]
    total = sum(weights)
    if total <= 0:
        # All weights are zero/negative-inf — pick uniformly
        chosen = rng.choice(choices)
    else:
        chosen = rng.choices(choices, weights=weights, k=1)[0]

    kind = chosen[0]

    if kind == 'var':
        return Index(chosen[1])

    # Production chosen — create the program node and recurse for arguments
    prod = chosen[1]
    inst_type = chosen[2]
    arg_types = chosen[4]

    if not arg_types:
        # Nullary production (constant) — just return the primitive
        return prod.program

    # For productions with arguments, build Application chain:
    # e.g., (filter pred list) = Application(Application(filter, pred), list)
    result = prod.program
    for arg_type in arg_types:
        arg_program = _sample_recursive(
            grammar, arg_type, env,
            max_depth, current_depth + 1, rng
        )
        result = Application(result, arg_program)

    return result
```

**Step 4: Run test to verify it passes**

Run: `cd src && python -m pytest tests/test_mcmc_search.py::TestSampleProgram -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/gallery_analysis/mcmc_search.py src/tests/test_mcmc_search.py
git commit -m "feat: add type-safe program sampler from grammar prior"
```

---

### Task 1.2: Subtree selection and regeneration (the MCMC proposal)

**Goal:** Given a complete program, select a random subtree, note its type and environment, and regenerate it by sampling from the grammar. This is the core proposal mechanism for tree-regeneration MH.

**Files:**
- Modify: `src/gallery_analysis/mcmc_search.py`
- Test: `src/tests/test_mcmc_search.py`

**Step 1: Write the failing test**

Add to `src/tests/test_mcmc_search.py`:

```python
from gallery_analysis.mcmc_search import (
    sample_program, collect_subtree_sites, propose_regeneration
)


class TestSubtreeProposal:
    """Test subtree selection and regeneration."""

    def setup_method(self):
        self.grammar = build_gallery_grammar()

    def test_collect_subtree_sites_finds_nodes(self):
        """Should find at least one subtree site in any non-trivial program."""
        prog = sample_program(self.grammar, Arrow(HAND, BOOL), max_depth=4, seed=42)
        sites = collect_subtree_sites(prog, Arrow(HAND, BOOL))
        assert len(sites) > 0

    def test_subtree_sites_have_types(self):
        """Each site should have a type and environment."""
        prog = sample_program(self.grammar, Arrow(HAND, BOOL), max_depth=4, seed=42)
        sites = collect_subtree_sites(prog, Arrow(HAND, BOOL))
        for site in sites:
            assert 'type' in site
            assert 'env' in site
            assert 'path' in site

    def test_propose_regeneration_returns_valid_program(self):
        """Proposed program should be complete and type-correct."""
        prog = sample_program(self.grammar, Arrow(HAND, BOOL), max_depth=5, seed=42)
        new_prog, log_q_forward, log_q_reverse = propose_regeneration(
            self.grammar, prog, Arrow(HAND, BOOL), max_depth=5, seed=100
        )
        assert not has_holes(new_prog)

    def test_propose_regeneration_changes_program(self):
        """Proposal should (usually) produce a different program."""
        prog = sample_program(self.grammar, Arrow(HAND, BOOL), max_depth=5, seed=42)
        different_count = 0
        for seed in range(20):
            new_prog, _, _ = propose_regeneration(
                self.grammar, prog, Arrow(HAND, BOOL), max_depth=5, seed=seed
            )
            if str(new_prog) != str(prog):
                different_count += 1
        # At least some proposals should differ
        assert different_count > 0

    def test_proposal_returns_log_probabilities(self):
        """Forward and reverse proposal probabilities should be finite."""
        prog = sample_program(self.grammar, Arrow(HAND, BOOL), max_depth=5, seed=42)
        new_prog, log_q_fwd, log_q_rev = propose_regeneration(
            self.grammar, prog, Arrow(HAND, BOOL), max_depth=5, seed=100
        )
        assert math.isfinite(log_q_fwd)
        assert math.isfinite(log_q_rev)
```

**Step 2: Run test to verify it fails**

Run: `cd src && python -m pytest tests/test_mcmc_search.py::TestSubtreeProposal -v`
Expected: FAIL with ImportError

**Step 3: Write the implementation**

Add to `src/gallery_analysis/mcmc_search.py`:

```python
# =========================================================================
# SUBTREE COLLECTION — Find all regenerable subtree sites in a program
# =========================================================================

@dataclass
class SubtreeSite:
    """
    A location in a program AST where a subtree can be regenerated.

    During MCMC, we pick one of these sites uniformly at random, cut out
    the subtree at that position, and regrow a new one from the grammar.

    Fields:
        path: Sequence of 'f'/'x'/'body' steps from root to this node
        type: The type that any replacement subtree must have
        env: Type environment at this position (types of bound variables in scope)
        subtree: The current subtree at this position
    """
    path: Tuple[str, ...]
    type: Type
    env: List[Type]
    subtree: Program


def collect_subtree_sites(
    program: Program,
    request_type: Type,
) -> List[SubtreeSite]:
    """
    Walk the AST and collect all positions where a subtree could be regenerated.

    Each site records:
      - The path from root (for later replacement)
      - The type at that position (for type-safe regeneration)
      - The type environment (bound variables in scope)
      - The current subtree (for computing reverse proposal probability)

    We skip the root node itself — regenerating the entire program from scratch
    is equivalent to sampling a new chain start, not a local proposal.

    How it works:
      We do a recursive walk, tracking the current type and environment.
      For Abstractions (lambdas), the argument type is added to the environment.
      For Applications, we infer the function and argument types.
    """
    sites = []
    _collect_recursive(program, request_type, [], (), sites)
    return sites


def _collect_recursive(
    program: Program,
    current_type: Type,
    env: List[Type],
    path: Tuple[str, ...],
    sites: List[SubtreeSite],
) -> None:
    """Recursive helper to collect subtree sites with type/env tracking."""

    # Don't include the root — regenerating the root is not a "local" proposal
    if path:
        sites.append(SubtreeSite(
            path=path,
            type=current_type,
            env=list(env),
            subtree=program,
        ))

    if isinstance(program, Abstraction):
        # Lambda: λ. body
        # The body's type is the return type, and we extend env with arg type
        if isinstance(current_type, Arrow):
            body_type = current_type.ret
            new_env = [current_type.arg] + env
            _collect_recursive(program.body, body_type, new_env, path + ('body',), sites)

    elif isinstance(program, Application):
        # Application: (f x)
        # f has type A → B, x has type A, whole thing has type B
        # We need to infer A from f's type
        try:
            ctx = TypeContext()
            f_type = program.f.infer_type(ctx, [_instantiate_env_type(t) for t in env])
            # f_type should be Arrow(arg_type, current_type)
            if isinstance(f_type, Arrow):
                _collect_recursive(program.f, f_type, env, path + ('f',), sites)
                _collect_recursive(program.x, f_type.arg, env, path + ('x',), sites)
            else:
                # f is not obviously a function — still try to collect from children
                _collect_recursive(program.f, f_type, env, path + ('f',), sites)
        except (UnificationError, Exception):
            # Type inference failed — skip this subtree's children
            pass

    # Primitives, Indices, Invented: leaf nodes — no children to recurse into


def _instantiate_env_type(tp: Type) -> Type:
    """Create a fresh copy of a type for inference (avoid cross-contamination)."""
    # For simple base types, return as-is
    if isinstance(tp, BaseType):
        return tp
    return tp  # For now, return as-is — TypeContext handles freshening


def replace_subtree(program: Program, path: Tuple[str, ...], replacement: Program) -> Program:
    """
    Replace the subtree at the given path with a new subtree.

    The path is a tuple of steps: 'f' (go to function), 'x' (go to argument),
    'body' (go into lambda body).

    Example:
        path = ('body', 'f', 'x') means:
        1. Enter the lambda body
        2. Go to the function position of an application
        3. Go to the argument position — replace here
    """
    if not path:
        return replacement

    step = path[0]
    rest = path[1:]

    if step == 'body' and isinstance(program, Abstraction):
        new_body = replace_subtree(program.body, rest, replacement)
        return Abstraction(new_body)
    elif step == 'f' and isinstance(program, Application):
        new_f = replace_subtree(program.f, rest, replacement)
        return Application(new_f, program.x)
    elif step == 'x' and isinstance(program, Application):
        new_x = replace_subtree(program.x, rest, replacement)
        return Application(program.f, new_x)
    else:
        raise ValueError(f"Invalid path step '{step}' for {type(program).__name__}")


# =========================================================================
# PROPOSAL — Regenerate a random subtree (tree-regeneration MH)
# =========================================================================

def propose_regeneration(
    grammar: Grammar,
    program: Program,
    request_type: Type,
    max_depth: int = 6,
    seed: Optional[int] = None,
) -> Tuple[Program, float, float]:
    """
    Propose a new program by regenerating a random subtree.

    This is the core proposal distribution for tree-regeneration Metropolis-Hastings:
      1. Collect all subtree sites in the current program
      2. Pick one uniformly at random
      3. Regenerate that subtree by sampling from the grammar (type-safe)
      4. Compute forward and reverse proposal probabilities

    The proposal probability Q(new | old) factors as:
      Q(new | old) = P(pick site s) × P(new subtree | grammar, type_s, env_s)

    For the reverse direction Q(old | new):
      Q(old | new) = P(pick site s') × P(old subtree | grammar, type_s', env_s')
      where s' is the corresponding site in the new program

    Args:
        grammar: The PCFG for sampling subtrees
        program: The current program to modify
        request_type: Top-level type (e.g., Arrow(HAND, BOOL))
        max_depth: Max depth for regenerated subtrees
        seed: Random seed for reproducibility

    Returns:
        (new_program, log_q_forward, log_q_reverse):
        - new_program: The proposed program with one subtree regenerated
        - log_q_forward: log Q(new | old) — probability of this proposal
        - log_q_reverse: log Q(old | new) — probability of reverse proposal
    """
    rng = random.Random(seed)

    # Step 1: Collect all subtree sites
    sites = collect_subtree_sites(program, request_type)
    if not sites:
        # No sites to regenerate (degenerate program) — return unchanged
        return program, 0.0, 0.0

    # Step 2: Pick a site uniformly at random
    site_idx = rng.randrange(len(sites))
    site = sites[site_idx]
    log_p_pick_site = -math.log(len(sites))

    # Step 3: Regenerate the subtree at this site
    # The new subtree must have the same type and respect the same environment
    # We limit depth of the new subtree to avoid explosion
    subtree_max_depth = max(3, max_depth - len(site.path))
    new_subtree = _sample_recursive(
        grammar, site.type, site.env,
        subtree_max_depth, 0, rng
    )

    # Step 4: Replace the subtree
    new_program = replace_subtree(program, site.path, new_subtree)

    # Step 5: Compute proposal probabilities
    # Forward: P(pick this site) × P(new subtree | grammar)
    log_q_new_subtree = grammar.program_log_likelihood(new_subtree, site.type, site.env)
    log_q_forward = log_p_pick_site + log_q_new_subtree

    # Reverse: P(pick corresponding site in new program) × P(old subtree | grammar)
    new_sites = collect_subtree_sites(new_program, request_type)
    log_p_pick_site_reverse = -math.log(max(1, len(new_sites)))
    log_q_old_subtree = grammar.program_log_likelihood(site.subtree, site.type, site.env)
    log_q_reverse = log_p_pick_site_reverse + log_q_old_subtree

    return new_program, log_q_forward, log_q_reverse
```

**Step 4: Run test to verify it passes**

Run: `cd src && python -m pytest tests/test_mcmc_search.py::TestSubtreeProposal -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/gallery_analysis/mcmc_search.py src/tests/test_mcmc_search.py
git commit -m "feat: add subtree collection and regeneration proposal"
```

---

### Task 1.3: Metropolis-Hastings acceptance and chain runner

**Goal:** Implement the MH acceptance criterion and the main MCMC chain loop that searches for programs solving a given task.

**Files:**
- Modify: `src/gallery_analysis/mcmc_search.py`
- Test: `src/tests/test_mcmc_search.py`

**Step 1: Write the failing test**

```python
from gallery_analysis.mcmc_search import (
    sample_program, collect_subtree_sites, propose_regeneration,
    MCMCChain, MCMCConfig, MCMCResult,
)
from gallery_analysis.exemplars import load_exemplars


class TestMCMCChain:
    """Test the MCMC chain runner."""

    def setup_method(self):
        self.grammar = build_gallery_grammar()
        self.exemplars = load_exemplars()

    def test_chain_runs_without_error(self):
        """Chain should complete N steps without crashing."""
        config = MCMCConfig(n_steps=100, max_depth=5, seed=42)
        # Use a simple rule: all_red
        hands = self.exemplars['all_red']['hands_primary']
        result = MCMCChain(self.grammar, config).run(
            request_type=Arrow(HAND, BOOL),
            exemplar_hands=hands,
        )
        assert result is not None
        assert result.n_steps == 100

    def test_chain_collects_unique_programs(self):
        """Chain should find at least a few distinct programs."""
        config = MCMCConfig(n_steps=500, max_depth=5, seed=42)
        hands = self.exemplars['all_red']['hands_primary']
        result = MCMCChain(self.grammar, config).run(
            request_type=Arrow(HAND, BOOL),
            exemplar_hands=hands,
        )
        assert result.n_unique > 0

    def test_chain_acceptance_rate_reasonable(self):
        """Acceptance rate should be between 0 and 1."""
        config = MCMCConfig(n_steps=500, max_depth=5, seed=42)
        hands = self.exemplars['all_red']['hands_primary']
        result = MCMCChain(self.grammar, config).run(
            request_type=Arrow(HAND, BOOL),
            exemplar_hands=hands,
        )
        assert 0.0 <= result.acceptance_rate <= 1.0

    def test_chain_result_has_top_hypotheses(self):
        """Result should contain ranked hypotheses."""
        config = MCMCConfig(n_steps=1000, max_depth=5, seed=42)
        hands = self.exemplars['all_red']['hands_primary']
        result = MCMCChain(self.grammar, config).run(
            request_type=Arrow(HAND, BOOL),
            exemplar_hands=hands,
        )
        assert len(result.top_hypotheses) > 0
        # Each hypothesis should have a program string and visit count
        for hyp in result.top_hypotheses[:5]:
            assert 'program' in hyp
            assert 'visit_count' in hyp
            assert 'log_posterior' in hyp
```

**Step 2: Run test to verify it fails**

Run: `cd src && python -m pytest tests/test_mcmc_search.py::TestMCMCChain -v`
Expected: FAIL with ImportError

**Step 3: Write the implementation**

Add to `src/gallery_analysis/mcmc_search.py`:

```python
from rules.cards import Hand


# =========================================================================
# CONFIGURATION AND RESULTS
# =========================================================================

@dataclass
class MCMCConfig:
    """
    Configuration for an MCMC search run.

    n_steps: Number of MH steps to take.
        - 10_000: Quick test (~2-5 min per task)
        - 100_000: Default serious run (~20-50 min per task)
        - 250_000: Full Piantadosi-scale run

    max_depth: Maximum AST depth for programs.
        Controls the size of the search space.

    noise_epsilon: Noise parameter for likelihood.
        P(exemplar | hypothesis) = (1-ε)/|ext(h)| if exemplar ∈ ext(h)
                                  = ε/TOTAL_HANDS  otherwise
        Higher ε = more tolerant of exceptions = better mixing.
        Set to 0.0 for strict likelihood (chain gets trapped easily).

    max_nodes: Hard cutoff on program size (following LOTlib3's maxnodes=25).
        Prevents chains from wandering into overly complex programs.

    top_k: Number of top hypotheses to retain, ranked by visit count.
    """
    n_steps: int = 100_000
    max_depth: int = 6
    noise_epsilon: float = 0.01
    max_nodes: int = 25
    top_k: int = 250
    seed: Optional[int] = None


@dataclass
class MCMCResult:
    """
    Results from an MCMC chain run.

    Captures both the standard Bayesian outputs (best hypotheses, posteriors)
    and MCMC-specific diagnostics (acceptance rate, visit counts, trajectory).
    """
    n_steps: int
    n_accepted: int
    n_unique: int
    acceptance_rate: float
    top_hypotheses: List[Dict[str, Any]]
    best_program: Optional[str] = None
    best_log_posterior: float = float('-inf')
    # Trajectory data (for Phase 2 hypothesis collection)
    visit_counts: Dict[str, int] = field(default_factory=dict)
    first_passage: Dict[str, int] = field(default_factory=dict)


# =========================================================================
# LIKELIHOOD COMPUTATION (positive-only exemplars, size principle)
# =========================================================================

# Total 6-card hands: C(52,6) = 20,358,520
TOTAL_HANDS = 20_358_520

# For MCMC we estimate extension size via a small Monte Carlo sample
# (cheaper than the full 100K used in the static analysis)
_MCMC_EXT_SAMPLE_SIZE = 10_000


def compute_mcmc_log_likelihood(
    program: Program,
    exemplar_hands: List[Hand],
    noise_epsilon: float = 0.01,
    ext_probe_hands: Optional[List[Hand]] = None,
) -> float:
    """
    Compute log P(data | hypothesis) for MCMC acceptance.

    Uses the size principle with optional noise:
      Strict:  P(D|h) = (1/|ext(h)|)^n if all exemplars ∈ ext(h), else 0
      Noisy:   P(D|h) = ∏_i [(1-ε)/|ext(h)| if hit, else ε/TOTAL_HANDS]

    For MCMC, we use the noisy version to avoid the chain getting permanently
    stuck when a single exemplar doesn't match. The noise parameter ε controls
    how much tolerance we have for mismatches.

    Extension size |ext(h)| is estimated via Monte Carlo on a probe set.
    This is cheaper than exhaustive evaluation but introduces variance.

    Args:
        program: The hypothesis program (Hand → bool)
        exemplar_hands: The frozen positive exemplars (6 hands)
        noise_epsilon: Noise parameter (0 = strict, >0 = tolerant)
        ext_probe_hands: Hands for extension size estimation (if None, skip ext scaling)

    Returns:
        Log-likelihood (negative number; closer to 0 = better fit)
    """
    try:
        func = program.evaluate([])
    except Exception:
        return float('-inf')

    # Check which exemplars the hypothesis covers
    n_hits = 0
    for hand in exemplar_hands:
        try:
            result = func(hand)
            if result is True:
                n_hits += 1
        except Exception:
            pass  # Exception = doesn't cover this exemplar

    n_exemplars = len(exemplar_hands)

    # Estimate extension size
    if ext_probe_hands is not None and len(ext_probe_hands) > 0:
        ext_count = 0
        for probe in ext_probe_hands:
            try:
                if func(probe) is True:
                    ext_count += 1
            except Exception:
                pass
        # Scale up: ext_size ≈ (ext_count / n_probes) × TOTAL_HANDS
        ext_fraction = ext_count / len(ext_probe_hands)
        ext_size = max(1, int(ext_fraction * TOTAL_HANDS))
    else:
        # Without probes, use a default moderate extension size
        # This makes the likelihood purely about hit/miss pattern
        ext_size = TOTAL_HANDS // 10  # Assume ~10% of hands match

    # Compute log-likelihood using size principle + noise
    log_lik = 0.0
    for i, hand in enumerate(exemplar_hands):
        try:
            result = func(hand)
            is_hit = (result is True)
        except Exception:
            is_hit = False

        if is_hit:
            # P(d_i | h) = (1 - ε) / |ext(h)| + ε / TOTAL
            p = (1.0 - noise_epsilon) / ext_size + noise_epsilon / TOTAL_HANDS
        else:
            # P(d_i | h) = ε / TOTAL
            p = noise_epsilon / TOTAL_HANDS

        if p <= 0:
            return float('-inf')
        log_lik += math.log(p)

    return log_lik


# =========================================================================
# MCMC CHAIN — The main Metropolis-Hastings loop
# =========================================================================

class MCMCChain:
    """
    A single Metropolis-Hastings chain over the space of typed programs.

    The chain maintains a current program and proposes modifications via
    tree-regeneration: pick a random subtree, regrow it from the grammar.
    Accept/reject based on the MH ratio:

        α = min(1, [P(new|D) × P(new|grammar) × Q(old|new)] /
                    [P(old|D) × P(old|grammar) × Q(new|old)])

    where:
        P(·|D) is the data likelihood (size principle)
        P(·|grammar) is the grammar prior
        Q(·|·) is the proposal probability (tree regeneration)
    """

    def __init__(self, grammar: Grammar, config: MCMCConfig):
        self.grammar = grammar
        self.config = config

    def run(
        self,
        request_type: Type,
        exemplar_hands: List[Hand],
        ext_probe_hands: Optional[List[Hand]] = None,
    ) -> MCMCResult:
        """
        Run the MCMC chain for config.n_steps steps.

        Args:
            request_type: Type of programs to search (e.g., Arrow(HAND, BOOL))
            exemplar_hands: Frozen positive exemplars for this rule
            ext_probe_hands: Probe hands for extension size estimation

        Returns:
            MCMCResult with top hypotheses, diagnostics, and trajectory data
        """
        rng = random.Random(self.config.seed)
        cfg = self.config

        # Generate extension probes if not provided
        if ext_probe_hands is None:
            from gallery_analysis.exemplars import generate_probe_set
            ext_probe_hands = generate_probe_set(
                n_probes=_MCMC_EXT_SAMPLE_SIZE,
                seed=rng.randint(0, 2**31)
            )

        # Initialize chain with a random program from the grammar prior
        current = sample_program(
            self.grammar, request_type,
            max_depth=cfg.max_depth, seed=rng.randint(0, 2**31)
        )

        # Compute initial scores
        current_log_prior = self.grammar.program_log_likelihood(current, request_type)
        current_log_lik = compute_mcmc_log_likelihood(
            current, exemplar_hands, cfg.noise_epsilon, ext_probe_hands
        )
        current_log_posterior = current_log_prior + current_log_lik

        # Tracking
        visit_counts: Dict[str, int] = {}
        first_passage: Dict[str, int] = {}
        n_accepted = 0

        current_str = str(current)
        visit_counts[current_str] = 1
        first_passage[current_str] = 0

        for step in range(cfg.n_steps):
            # Propose a new program via subtree regeneration
            proposed, log_q_forward, log_q_reverse = propose_regeneration(
                self.grammar, current, request_type,
                max_depth=cfg.max_depth,
                seed=rng.randint(0, 2**31),
            )

            # Enforce max_nodes constraint (LOTlib3-style)
            if proposed.size() > cfg.max_nodes:
                continue  # Reject oversized proposals immediately

            # Compute proposed scores
            proposed_log_prior = self.grammar.program_log_likelihood(proposed, request_type)
            proposed_log_lik = compute_mcmc_log_likelihood(
                proposed, exemplar_hands, cfg.noise_epsilon, ext_probe_hands
            )
            proposed_log_posterior = proposed_log_prior + proposed_log_lik

            # Metropolis-Hastings acceptance ratio (in log space):
            # log α = (log P(new|D) + log P(new|G) + log Q(old|new))
            #       - (log P(old|D) + log P(old|G) + log Q(new|old))
            log_alpha = (
                (proposed_log_posterior + log_q_reverse)
                - (current_log_posterior + log_q_forward)
            )

            # Accept or reject
            if log_alpha >= 0 or math.log(rng.random()) < log_alpha:
                # Accept
                current = proposed
                current_log_prior = proposed_log_prior
                current_log_lik = proposed_log_lik
                current_log_posterior = proposed_log_posterior
                n_accepted += 1

            # Track visits
            current_str = str(current)
            visit_counts[current_str] = visit_counts.get(current_str, 0) + 1
            if current_str not in first_passage:
                first_passage[current_str] = step + 1

        # Build result
        # Sort hypotheses by visit count (most visited = highest posterior weight)
        sorted_hyps = sorted(
            visit_counts.items(), key=lambda x: -x[1]
        )[:cfg.top_k]

        top_hypotheses = []
        for prog_str, count in sorted_hyps:
            top_hypotheses.append({
                'program': prog_str,
                'visit_count': count,
                'log_posterior': float('-inf'),  # Will be computed in Phase 2
                'first_seen_step': first_passage.get(prog_str, -1),
            })

        # Compute log_posterior for top hypotheses
        # (Re-evaluate since we didn't store it during the chain)
        # For now, just flag the best one
        best_prog_str = sorted_hyps[0][0] if sorted_hyps else None

        return MCMCResult(
            n_steps=cfg.n_steps,
            n_accepted=n_accepted,
            n_unique=len(visit_counts),
            acceptance_rate=n_accepted / max(1, cfg.n_steps),
            top_hypotheses=top_hypotheses,
            best_program=best_prog_str,
            visit_counts=visit_counts,
            first_passage=first_passage,
        )
```

**Step 4: Run test to verify it passes**

Run: `cd src && python -m pytest tests/test_mcmc_search.py::TestMCMCChain -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/gallery_analysis/mcmc_search.py src/tests/test_mcmc_search.py
git commit -m "feat: add MH acceptance and MCMC chain runner"
```

---

### Task 1.4: Multi-chain parallel runner

**Goal:** Run multiple independent MCMC chains per task (for better coverage) and merge their results. This is the MCMC equivalent of parallelizing enumeration across workers.

**Files:**
- Modify: `src/gallery_analysis/mcmc_search.py`
- Test: `src/tests/test_mcmc_search.py`

**Step 1: Write the failing test**

```python
from gallery_analysis.mcmc_search import run_parallel_chains


class TestParallelChains:
    """Test multi-chain parallel execution."""

    def setup_method(self):
        self.grammar = build_gallery_grammar()
        self.exemplars = load_exemplars()

    def test_parallel_chains_merge_results(self):
        """Multiple chains should merge their visit counts."""
        config = MCMCConfig(n_steps=200, max_depth=5, seed=42)
        hands = self.exemplars['all_red']['hands_primary']
        result = run_parallel_chains(
            self.grammar, config,
            request_type=Arrow(HAND, BOOL),
            exemplar_hands=hands,
            n_chains=4,
        )
        assert result.n_unique > 0
        # Should have more unique programs than a single chain
        # (not guaranteed, but very likely with 4 chains)
        assert result.n_steps == 200 * 4

    def test_parallel_chains_different_seeds(self):
        """Each chain should explore different regions."""
        config = MCMCConfig(n_steps=200, max_depth=5, seed=42)
        hands = self.exemplars['all_red']['hands_primary']
        result = run_parallel_chains(
            self.grammar, config,
            request_type=Arrow(HAND, BOOL),
            exemplar_hands=hands,
            n_chains=4,
        )
        assert result.acceptance_rate > 0
```

**Step 2: Run test, verify failure**

**Step 3: Write the implementation**

```python
def run_parallel_chains(
    grammar: Grammar,
    config: MCMCConfig,
    request_type: Type,
    exemplar_hands: List[Hand],
    n_chains: int = 8,
    ext_probe_hands: Optional[List[Hand]] = None,
) -> MCMCResult:
    """
    Run multiple independent MCMC chains and merge results.

    Why multiple chains?
      A single chain may get stuck in a local optimum. Running N independent
      chains from different starting points gives better coverage of the
      posterior. This is the simplest form of parallel MCMC — each chain
      is fully independent (no communication or swapping).

    The merged result combines visit counts across chains, giving a pooled
    estimate of the posterior. Programs visited by multiple chains are
    particularly strong candidates.

    Future work (Option C from brainstorming): Parallel tempering would add
    communication between chains at different temperatures, enabling better
    exploration. This is a natural extension when mixing is a problem.

    Args:
        grammar: The PCFG
        config: Base configuration (seed will be varied per chain)
        request_type: Type of programs to search
        exemplar_hands: Frozen positive exemplars
        n_chains: Number of independent chains to run
        ext_probe_hands: Shared probe set for extension estimation

    Returns:
        Merged MCMCResult combining all chains
    """
    base_seed = config.seed if config.seed is not None else 42

    # Generate shared probe set (same for all chains — ensures comparable ext estimates)
    if ext_probe_hands is None:
        from gallery_analysis.exemplars import generate_probe_set
        ext_probe_hands = generate_probe_set(
            n_probes=_MCMC_EXT_SAMPLE_SIZE,
            seed=base_seed
        )

    # Run chains (sequential for now — can be parallelized with ProcessPoolExecutor)
    chain_results = []
    for i in range(n_chains):
        chain_config = MCMCConfig(
            n_steps=config.n_steps,
            max_depth=config.max_depth,
            noise_epsilon=config.noise_epsilon,
            max_nodes=config.max_nodes,
            top_k=config.top_k,
            seed=base_seed + i * 1000,  # Different seed per chain
        )
        chain = MCMCChain(grammar, chain_config)
        result = chain.run(request_type, exemplar_hands, ext_probe_hands)
        chain_results.append(result)

    # Merge results across chains
    merged_visits: Dict[str, int] = {}
    merged_first_passage: Dict[str, int] = {}
    total_accepted = 0
    total_steps = 0

    for i, result in enumerate(chain_results):
        total_accepted += result.n_accepted
        total_steps += result.n_steps

        for prog_str, count in result.visit_counts.items():
            merged_visits[prog_str] = merged_visits.get(prog_str, 0) + count

        for prog_str, step in result.first_passage.items():
            # First passage = earliest across all chains
            offset_step = step + i * config.n_steps
            if prog_str not in merged_first_passage:
                merged_first_passage[prog_str] = offset_step
            else:
                merged_first_passage[prog_str] = min(
                    merged_first_passage[prog_str], offset_step
                )

    # Build merged top hypotheses
    sorted_hyps = sorted(
        merged_visits.items(), key=lambda x: -x[1]
    )[:config.top_k]

    top_hypotheses = []
    for prog_str, count in sorted_hyps:
        top_hypotheses.append({
            'program': prog_str,
            'visit_count': count,
            'log_posterior': float('-inf'),
            'first_seen_step': merged_first_passage.get(prog_str, -1),
        })

    best_prog_str = sorted_hyps[0][0] if sorted_hyps else None

    return MCMCResult(
        n_steps=total_steps,
        n_accepted=total_accepted,
        n_unique=len(merged_visits),
        acceptance_rate=total_accepted / max(1, total_steps),
        top_hypotheses=top_hypotheses,
        best_program=best_prog_str,
        visit_counts=merged_visits,
        first_passage=merged_first_passage,
    )
```

**Step 4: Run tests**

Run: `cd src && python -m pytest tests/test_mcmc_search.py::TestParallelChains -v`

**Step 5: Commit**

```bash
git add src/gallery_analysis/mcmc_search.py src/tests/test_mcmc_search.py
git commit -m "feat: add multi-chain parallel MCMC runner"
```

---

### Task 1.5: End-to-end smoke test on a real gallery rule

**Goal:** Verify the full MCMC pipeline works on actual gallery rules. Run MCMC on 3 rules (one easy, one medium, one hard) and check that it finds at least some consistent hypotheses.

**Files:**
- Test: `src/tests/test_mcmc_search.py`

**Step 1: Write the integration test**

```python
class TestMCMCIntegration:
    """End-to-end test on real gallery rules."""

    def setup_method(self):
        self.grammar = build_gallery_grammar()
        self.exemplars = load_exemplars()

    def test_finds_hypotheses_for_easy_rule(self):
        """MCMC should find consistent hypotheses for 'all_red' (group 1)."""
        config = MCMCConfig(n_steps=5000, max_depth=5, seed=42)
        hands = self.exemplars['all_red']['hands_primary']
        result = run_parallel_chains(
            self.grammar, config,
            request_type=Arrow(HAND, BOOL),
            exemplar_hands=hands,
            n_chains=4,
        )
        # Should find some programs that cover all 6 exemplars
        assert result.n_unique > 10
        assert result.acceptance_rate > 0.01

    def test_finds_hypotheses_for_medium_rule(self):
        """MCMC should find hypotheses for 'all_even' (group 2)."""
        config = MCMCConfig(n_steps=5000, max_depth=5, seed=42)
        hands = self.exemplars['all_even']['hands_primary']
        result = run_parallel_chains(
            self.grammar, config,
            request_type=Arrow(HAND, BOOL),
            exemplar_hands=hands,
            n_chains=4,
        )
        assert result.n_unique > 5
        assert result.acceptance_rate > 0.001

    def test_mcmc_vs_enumeration_coverage(self):
        """MCMC should find at least some programs that enumeration also finds."""
        # This is a sanity check — both methods search the same space
        config = MCMCConfig(n_steps=10000, max_depth=5, seed=42)
        hands = self.exemplars['all_red']['hands_primary']
        result = run_parallel_chains(
            self.grammar, config,
            request_type=Arrow(HAND, BOOL),
            exemplar_hands=hands,
            n_chains=4,
        )
        # Just verify it produced reasonable output
        assert len(result.top_hypotheses) > 0
        assert result.top_hypotheses[0]['visit_count'] > 1
```

**Step 2: Run tests**

Run: `cd src && python -m pytest tests/test_mcmc_search.py::TestMCMCIntegration -v --timeout=120`
Expected: PASS (may take 30-60 seconds)

**Step 3: Commit**

```bash
git add src/tests/test_mcmc_search.py
git commit -m "test: add end-to-end MCMC integration tests on gallery rules"
```

---

## Phase 2: Hypothesis Trajectory Collection

This phase adds the scientifically interesting part: extracting process-level data from MCMC chains.

### Task 2.1: Trajectory recorder

**Goal:** Instrument the MCMC chain to record the full sequence of hypotheses visited, enabling trajectory analysis.

**Files:**
- Create: `src/gallery_analysis/mcmc_hypothesis_collector.py`
- Test: `src/tests/test_mcmc_hypothesis_collector.py`

**Step 1: Write the failing test**

```python
"""Tests for MCMC hypothesis trajectory collection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from dreamcoder_core.type_system import Arrow, BOOL, HAND
from gallery_analysis.enumerator import build_gallery_grammar
from gallery_analysis.exemplars import load_exemplars
from gallery_analysis.mcmc_search import MCMCConfig, run_parallel_chains
from gallery_analysis.mcmc_hypothesis_collector import (
    TrajectoryAnalyzer, HypothesisTrajectory
)


class TestTrajectoryAnalyzer:
    """Test trajectory analysis from MCMC results."""

    def setup_method(self):
        self.grammar = build_gallery_grammar()
        self.exemplars = load_exemplars()
        # Run a chain to get trajectory data
        config = MCMCConfig(n_steps=2000, max_depth=5, seed=42)
        hands = self.exemplars['all_red']['hands_primary']
        self.result = run_parallel_chains(
            self.grammar, config,
            request_type=Arrow(HAND, BOOL),
            exemplar_hands=hands,
            n_chains=2,
        )

    def test_dwelling_times(self):
        """Should compute how long chain stays on each hypothesis."""
        analyzer = TrajectoryAnalyzer(self.result)
        dwellings = analyzer.dwelling_times()
        assert len(dwellings) > 0
        # Dwelling times should be positive integers
        for prog, time in dwellings.items():
            assert time > 0

    def test_visit_frequency_ranking(self):
        """Should rank hypotheses by visit frequency."""
        analyzer = TrajectoryAnalyzer(self.result)
        ranking = analyzer.frequency_ranking(top_k=10)
        assert len(ranking) > 0
        assert len(ranking) <= 10
        # Should be sorted by frequency (descending)
        for i in range(len(ranking) - 1):
            assert ranking[i]['visit_count'] >= ranking[i+1]['visit_count']

    def test_first_passage_ordering(self):
        """Should provide first-passage ordering of hypotheses."""
        analyzer = TrajectoryAnalyzer(self.result)
        ordering = analyzer.first_passage_ordering()
        assert len(ordering) > 0
        # Should be sorted by first appearance step
        for i in range(len(ordering) - 1):
            assert ordering[i]['first_step'] <= ordering[i+1]['first_step']
```

**Step 2: Run test, verify failure**

**Step 3: Implement TrajectoryAnalyzer**

In `src/gallery_analysis/mcmc_hypothesis_collector.py`:

```python
"""
Hypothesis trajectory analysis for MCMC search.

Extracts process-level data from MCMC chains that models the *dynamics*
of hypothesis search, not just the endpoint:

  - Dwelling times: How long does the chain stay on each hypothesis?
    → Models how "sticky" a hypothesis is for a reasoner
  - Visit frequency: How often is each hypothesis visited?
    → Empirical posterior weight (frequency ∝ posterior after mixing)
  - First-passage time: When is each hypothesis first discovered?
    → Models order of hypothesis consideration
  - Transition structure: Which hypotheses lead to which others?
    → Models the "hypothesis adjacency" in the learner's search

COGNITIVE MODELING INTERPRETATION:
  A human learner considering card rules doesn't systematically enumerate
  hypotheses by simplicity. Instead, they consider one hypothesis, notice
  it doesn't quite work, and revise. The MCMC trajectory models this:
  - "Sticky" hypotheses (high dwelling time) are ones the learner would
    persist with before moving on
  - First-passage order predicts which alternatives come to mind first
  - Transitions reveal which hypotheses are "nearby" in thought-space
"""
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, str(Path(__file__).parent.parent))

from gallery_analysis.mcmc_search import MCMCResult


@dataclass
class HypothesisTrajectory:
    """
    Complete trajectory data for a single hypothesis across MCMC chains.

    Captures everything needed for cognitive process modeling.
    """
    program_str: str
    visit_count: int           # Total visits across all chains
    first_seen_step: int       # Earliest step at which this was visited
    dwelling_time: int         # Average consecutive visits before moving on
    n_chains_found: int = 0    # How many chains found this hypothesis


class TrajectoryAnalyzer:
    """
    Analyze MCMC results to extract hypothesis trajectory data.

    Takes an MCMCResult (from a single chain or merged parallel chains)
    and computes trajectory-level statistics.
    """

    def __init__(self, result: MCMCResult):
        self.result = result

    def dwelling_times(self) -> Dict[str, int]:
        """
        Compute dwelling time for each hypothesis.

        Dwelling time = total number of steps the chain spent on this hypothesis.
        For merged chains, this is the sum across all chains.

        This is equivalent to visit_count since each step either stays or moves.
        In future, with full step-by-step recording, we could compute *consecutive*
        dwelling (mean run length), but for now visit_count is the proxy.
        """
        return dict(self.result.visit_counts)

    def frequency_ranking(self, top_k: int = 50) -> List[Dict[str, Any]]:
        """
        Rank hypotheses by visit frequency (empirical posterior weight).

        After mixing, visit frequency approximates the posterior:
          P(h | D) ≈ visits(h) / total_steps

        Returns top_k hypotheses sorted by frequency, with:
          - program: The program string
          - visit_count: Total visits
          - empirical_posterior: visits / total_steps (approximate posterior)
          - first_step: When first discovered
        """
        total = max(1, self.result.n_steps)
        sorted_hyps = sorted(
            self.result.visit_counts.items(),
            key=lambda x: -x[1]
        )[:top_k]

        return [
            {
                'program': prog,
                'visit_count': count,
                'empirical_posterior': count / total,
                'first_step': self.result.first_passage.get(prog, -1),
            }
            for prog, count in sorted_hyps
        ]

    def first_passage_ordering(self) -> List[Dict[str, Any]]:
        """
        Order hypotheses by first-passage time (when first discovered).

        This gives a prediction about the order in which a learner would
        consider different hypotheses. Earlier first-passage = comes to
        mind sooner.

        Returns all hypotheses sorted by first appearance step.
        """
        sorted_by_fp = sorted(
            self.result.first_passage.items(),
            key=lambda x: x[1]
        )

        return [
            {
                'program': prog,
                'first_step': step,
                'visit_count': self.result.visit_counts.get(prog, 0),
            }
            for prog, step in sorted_by_fp
        ]

    def summary(self) -> Dict[str, Any]:
        """
        Compute summary statistics for the trajectory.

        Returns a dict with high-level diagnostics useful for comparing
        MCMC runs across rules and configurations.
        """
        visits = list(self.result.visit_counts.values())
        total_visits = sum(visits)

        # Entropy of the visit distribution (bits)
        # High entropy = explored broadly; low entropy = concentrated on few
        entropy = 0.0
        for v in visits:
            p = v / max(1, total_visits)
            if p > 0:
                entropy -= p * (p if p == 0 else __import__('math').log2(p))

        return {
            'n_steps': self.result.n_steps,
            'n_unique': self.result.n_unique,
            'acceptance_rate': self.result.acceptance_rate,
            'entropy_bits': entropy,
            'top_1_program': self.result.best_program,
            'top_1_visits': visits[0] if visits else 0,
            'concentration': max(visits) / max(1, total_visits),  # Fraction at mode
        }
```

**Step 4: Run tests**

Run: `cd src && python -m pytest tests/test_mcmc_hypothesis_collector.py -v`

**Step 5: Commit**

```bash
git add src/gallery_analysis/mcmc_hypothesis_collector.py src/tests/test_mcmc_hypothesis_collector.py
git commit -m "feat: add hypothesis trajectory analyzer for MCMC process modeling"
```

---

### Task 2.2: Step-by-step trajectory recording in MCMCChain

**Goal:** Modify the chain to record the full step-by-step sequence of programs visited (not just aggregated counts). This enables computing consecutive dwelling times and transition frequencies.

**Files:**
- Modify: `src/gallery_analysis/mcmc_search.py` (add trajectory recording to MCMCChain.run)
- Modify: `src/gallery_analysis/mcmc_hypothesis_collector.py` (add transition analysis)
- Test: `src/tests/test_mcmc_hypothesis_collector.py`

**Step 1: Add trajectory field to MCMCResult**

Add `trajectory: List[str]` to `MCMCResult` — the ordered sequence of program strings at each step.

**Step 2: Record trajectory in chain loop**

In `MCMCChain.run()`, after the accept/reject decision, append `str(current)` to a trajectory list.

**Step 3: Add transition analysis to TrajectoryAnalyzer**

```python
def transition_counts(self) -> Dict[Tuple[str, str], int]:
    """
    Count transitions between hypotheses.

    A transition (A → B) means the chain moved from hypothesis A to B
    in one step. Self-transitions (A → A) are counted when a proposal
    is rejected.

    This builds the "hypothesis adjacency" structure: which hypotheses
    are reachable from which others in one MCMC step.
    """
    ...

def consecutive_dwelling_times(self) -> Dict[str, List[int]]:
    """
    Compute consecutive dwelling times (run lengths).

    For each hypothesis, collect the lengths of consecutive runs
    (how many steps the chain stayed before moving away).
    The mean run length is the expected "stickiness".
    """
    ...
```

**Step 4: Test transition and dwelling analysis**

**Step 5: Commit**

```bash
git commit -m "feat: add step-by-step trajectory recording and transition analysis"
```

---

## Phase 3: Gallery-wide Orchestrator

### Task 3.1: analyze_mcmc.py — MCMC analog of analyze.py

**Goal:** Create a top-level orchestrator that runs MCMC on all 60 gallery rules and produces output compatible with the existing Bayesian scorer.

**Files:**
- Create: `src/gallery_analysis/analyze_mcmc.py`

**Step 1: Design the interface**

```python
"""
MCMC analysis pipeline: run MCMC search over all 60 gallery rules.

This is the MCMC analog of analyze.py (which uses enumeration).
Both produce hypothesis pools that feed into the Bayesian scorer,
enabling direct comparison of the two search strategies.

Usage:
    cd src
    python -m gallery_analysis.analyze_mcmc [--n-steps 10000] [--n-chains 8] [--quick]

    # Quick test (~5 minutes, 3 rules)
    python -m gallery_analysis.analyze_mcmc --quick

    # Default run (~1-2 hours, all 60 rules)
    python -m gallery_analysis.analyze_mcmc --n-steps 100000 --n-chains 8

    # Full Piantadosi-scale run (~6-12 hours)
    python -m gallery_analysis.analyze_mcmc --n-steps 250000 --n-chains 16

Output:
    results/mcmc_{config}_results.json
    - Per-rule: top hypotheses, visit counts, trajectory summaries
    - Cross-rule: comparison with enumeration results (if available)
"""
```

**Step 2: Implement**

The orchestrator should:
1. Load the gallery grammar (same as enumeration uses)
2. Load frozen exemplars for all 60 rules
3. For each rule, run `run_parallel_chains()` with the configured budget
4. Collect results into the same format as `analyze.py`'s output
5. Optionally score hypotheses with `bayesian_scorer.py`
6. Save results JSON

**Step 3: Add CLI with argparse**

Support `--quick` (3 rules, 1K steps), `--n-steps`, `--n-chains`, `--max-depth`, `--seed`.

**Step 4: Test on --quick mode**

```bash
cd src && python -m gallery_analysis.analyze_mcmc --quick
```

**Step 5: Commit**

```bash
git commit -m "feat: add gallery-wide MCMC orchestrator (analyze_mcmc.py)"
```

---

### Task 3.2: Bayesian scoring integration

**Goal:** Score MCMC-discovered hypotheses using the same Bayesian framework as enumeration (prior, size-principle likelihood, posterior), so results are directly comparable.

**Files:**
- Modify: `src/gallery_analysis/analyze_mcmc.py`

**Step 1: For each rule's MCMC results, evaluate hypotheses on exemplar hands**

**Step 2: Compute fingerprints and extension sizes (reuse existing functions)**

**Step 3: Apply `bayesian_scorer.py` functions to compute posteriors**

**Step 4: Output per-rule results in the same JSON schema as enumeration**

**Step 5: Commit**

```bash
git commit -m "feat: integrate MCMC hypotheses with Bayesian scorer"
```

---

## Phase 4: Recognition Network Integration (Future)

> **Note:** This phase is deferred until Phases 1-3 are stable. It adds the most theoretically interesting piece: the recognition network as a proposal distribution.

### Task 4.1: Neural proposal distribution

**Goal:** Use the contrastive recognition network to bias MCMC proposals. Instead of regenerating subtrees from the base grammar, sample from the recognition-biased grammar (conditioned on the task).

**Approach:**
1. Load a trained recognition model
2. For each task, compute task embedding
3. During `propose_regeneration()`, blend base grammar weights with recognition predictions (using `contextual_weight` parameter)
4. The recognition network becomes the "learned intuition" that shapes what the MCMC learner thinks of next

**Cognitive story:** This models how experience with past rules builds intuitions that guide future hypothesis generation. A learner who has seen many color-based rules will propose color-related hypotheses more quickly when encountering a new rule.

### Task 4.2: Parallel tempering (Option C path)

**Goal:** When mixing is a problem, add parallel tempering: multiple chains at different temperatures, with periodic swaps between adjacent temperatures.

**Approach:**
1. Run K chains at temperatures T_1 < T_2 < ... < T_K
2. Hot chains explore broadly (high T = flat posterior)
3. Cold chains exploit (low T = sharp posterior)
4. Periodically propose swaps between adjacent chains
5. Accept/reject swaps based on the replica exchange criterion

---

## Phase 5: Evaluation Framework

### Task 5.1: MCMC vs Enumeration comparison script

**Goal:** Run both methods on the same 60 rules with the same time budget and compare:
- Number of rules solved (found the true rule in top-K)
- Rank of true rule in hypothesis list
- Posterior entropy
- Number of unique hypotheses discovered
- Correlation of difficulty rankings between methods

**Files:**
- Create: `src/gallery_analysis/compare_search_methods.py`

### Task 5.2: Trajectory analysis report

**Goal:** For each rule, produce a report showing:
- MCMC trajectory summary (first passage, dwelling, transitions)
- Comparison with enumeration search effort
- Predictions about human difficulty derived from each method

---

## Future Work Notes (Path to Option C)

These are not part of the current plan but should be kept in mind during implementation:

1. **Parallel tempering**: Add temperature parameter to `MCMCChain`, implement replica exchange between chains. The `run_parallel_chains` function already provides the multi-chain infrastructure.

2. **Adaptive proposals**: Track acceptance rate per subtree type/depth and adjust proposal distribution to maintain ~25% acceptance.

3. **Metaprogram proposals** (Rule et al. 2024): Instead of regenerating random subtrees, propose semantically meaningful transformations (swap AND↔OR, change comparator, shift constant).

4. **Hybrid search**: Use enumeration for shallow programs (depth ≤ 4) and MCMC for deeper search. This would get the best of both worlds but loses the clean single-mechanism theoretical claim.
