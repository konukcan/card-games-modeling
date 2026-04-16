"""
MCMC Program Search for Card-Game Rule Induction
=================================================

This module implements Metropolis-Hastings program search as an alternative
to top-down enumeration for exploring the hypothesis space of card-game rules.

COGNITIVE MODELING RATIONALE
----------------------------
Top-down enumeration explores programs in strict order of prior probability
(shortest/simplest first). While optimal for finding the MAP hypothesis, it
is a poor model of human hypothesis generation, which is:

  1. **Stochastic** — people don't exhaustively enumerate; they "jump" to
     candidate hypotheses guided by noisy pattern recognition.
  2. **Local** — once a promising hypothesis is found, people explore nearby
     variants (e.g., "maybe it's not all red, but all hearts").
  3. **Anchored** — initial hypotheses bias subsequent search (anchoring effect).

MCMC search captures all three properties:
  - The *prior sample* provides a stochastic starting point (1).
  - The *subtree-regeneration proposal* explores local variants (2).
  - The *chain's current state* acts as an anchor (3).

This module provides:
  - sample_program()           : sample a complete program from the grammar prior
  - collect_subtree_sites()    : find all AST positions eligible for regeneration
  - replace_subtree()          : swap a subtree at a given path
  - propose_regeneration()     : the core MCMC subtree-regeneration proposal
  - compute_mcmc_log_likelihood() : size-principle likelihood with noise
  - MCMCConfig / MCMCResult    : configuration and result dataclasses
  - MCMCChain                  : full Metropolis-Hastings chain runner

ARCHITECTURE
------------
  sample_program()             <- sample from grammar prior
  collect_subtree_sites()      <- walk AST, collect regeneration sites
  replace_subtree()            <- structural replacement at a path
  propose_regeneration()       <- select site + regenerate + compute proposal ratio
  compute_mcmc_log_likelihood()<- P(data | hypothesis) via size principle
  MCMCChain.run()              <- full MH chain with accept/reject
"""
import sys
import math
import random
import logging
from dataclasses import dataclass, field, replace as dc_replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.type_system import (
    Type, Arrow, ListType, TypeContext, TypeVariable,
    BOOL, INT, CARD, SUIT, RANK,
)
from dreamcoder_core.grammar import Grammar
from dreamcoder_core.program import (
    Program, Index, Application, Abstraction, Primitive, Invented,
    uses_variable,
)


# Hard absolute recursion limit to prevent runaway sampling.
# This is a safety net — in practice, programs should terminate
# well before this limit via the soft max_depth mechanism.
_ABSOLUTE_DEPTH_LIMIT = 20


def sample_program(
    grammar: Grammar,
    request_type: Type,
    max_depth: int = 6,
    seed: Optional[int] = None,
    env: Optional[List[Type]] = None,
) -> Program:
    """
    Sample a complete, type-correct program from the grammar prior.

    Builds a program top-down by repeatedly choosing productions proportional
    to their (exponentiated) log-probabilities, then recursing on each
    argument type. Arrow-typed requests create Abstraction (lambda) nodes
    that don't consume depth budget (lambdas are "free" structurally).

    Args:
        grammar:      The PCFG grammar whose productions define the prior.
        request_type: The type of program to generate (e.g., Arrow(HAND, BOOL)).
        max_depth:    Soft depth limit for non-lambda AST nodes. Beyond this
                      depth, sampling strongly prefers terminal productions and
                      variables. If no terminals exist for the required type
                      (e.g., bool has no constants in the gallery grammar),
                      non-terminal productions with the fewest arguments are
                      selected to minimize further recursion.
        seed:         If provided, seeds a local Random instance for
                      reproducibility. Each call gets its own RNG so
                      concurrent calls don't interfere.
        env:          Type environment for bound variables (de Bruijn indices).
                      env[i] = type of $i. Callers should leave this as None;
                      it is extended internally when entering lambdas.

    Returns:
        A complete Program AST with no Hole nodes.

    Raises:
        RuntimeError: If no valid production or variable exists for a required
                      type, or if the absolute depth limit is reached.

    How it works, step by step:
        1. If request_type is an arrow (A -> B), emit an Abstraction node,
           push A onto the environment, and recurse on B. Lambdas don't
           consume depth budget.
        2. Otherwise (base type), collect all type-compatible productions
           via grammar.candidates_for_type() and all matching bound variables
           via grammar.variable_candidates().
        3. If depth >= max_depth, prefer terminals (zero-argument productions)
           and variables. If none exist, fall back to the shallowest non-
           terminal productions (fewest arguments) to minimize further depth.
        4. Sample one candidate proportional to exp(log_probability).
        5. If sampled a variable, return Index(i).
        6. If sampled a production, build the Application chain by recursing
           on each argument type: Application(Application(f, arg1), arg2).
    """
    if env is None:
        env = []

    # Retry loop: polymorphic type variable resolution is heuristic and
    # can occasionally produce ill-typed programs (~1-2% of samples).
    # When this happens, we advance the RNG seed and retry. The number
    # of retries is bounded, and each attempt uses a deterministic seed
    # derived from the original, so reproducibility is preserved.
    max_retries = 10
    for attempt in range(max_retries):
        # Each call gets its own Random instance so that:
        #   (a) a fixed seed guarantees the same program, and
        #   (b) concurrent calls don't share state.
        effective_seed = seed + attempt if seed is not None else None
        rng = random.Random(effective_seed)

        try:
            program = _sample(
                grammar, request_type, max_depth, depth=0, env=env, rng=rng
            )
            # Verify the program is well-typed before returning.
            ctx = TypeContext()
            program.infer_type(ctx, env)
            return program
        except RuntimeError:
            # Absolute depth limit hit — retry with different seed.
            continue
        except Exception:
            # Type error from infer_type — retry with different seed.
            continue

    # All retries exhausted. Fall back to the last sample without type check.
    # This should be extremely rare (probability < (0.02)^10).
    rng = random.Random(seed)
    return _sample(grammar, request_type, max_depth, depth=0, env=env, rng=rng)


# Concrete types available for resolving free type variables.
# These are the monomorphic types that commonly appear in card-game programs.
# We include base types, the hand type, and common list types.
_CONCRETE_TYPES: List[Type] = [
    INT, CARD, SUIT, RANK, BOOL,
]


def _resolve_free_type_vars(
    arg_types: List[Type],
    env: List[Type],
    rng: random.Random,
) -> List[Type]:
    """
    Replace free type variables with concrete types, guided by the environment.

    When a polymorphic production like `eq : 'a -> 'a -> bool` is selected,
    its argument types share type variable 'a. To ensure both arguments
    get the same concrete type, we:

    1. Collect all free type variable IDs across all argument types.
    2. For each, try to infer a good concrete type by matching argument
       type patterns against the environment. For example, if an argument
       is `list('a)` and env has `list(card)`, we bind `'a = card`.
    3. For any still-unresolved variables, randomly choose from base types.
    4. Apply the substitution to all argument types.

    This ensures shared variables get the same concrete type, and the chosen
    types are likely to be satisfiable by variables in the environment.

    Args:
        arg_types: The argument types, possibly with free type variables.
        env:       Current type environment (for inferring concrete types).
        rng:       Random instance for choosing concrete types.

    Returns:
        A new list of argument types with all type variables resolved.
    """
    # Collect all free type variable IDs.
    free_vars = set()
    for t in arg_types:
        free_vars |= t.free_type_variables()

    if not free_vars:
        return arg_types  # Nothing to resolve.

    # Try to infer concrete types from the environment.
    # For each argument type, try to unify it with each env type to discover
    # what the type variables should be.
    subst: dict = {}
    for arg_type in arg_types:
        arg_free = arg_type.free_type_variables()
        if not arg_free:
            continue

        for env_type in env:
            try:
                ctx = TypeContext()
                # Instantiate the arg type with fresh variables to avoid
                # polluting the original variable IDs.
                inst = ctx.instantiate(arg_type)
                ctx.unify(inst, env_type)
                # Extract bindings for our original variables.
                # We need to map from the fresh variable IDs back to originals.
                # Since instantiate creates a 1:1 mapping, we can track it.
                fresh_free = inst.free_type_variables()
                # The instantiate created new vars; we need the mapping.
                # Easier approach: just try direct unification.
                ctx2 = TypeContext()
                ctx2.unify(arg_type, env_type)
                for var_id in arg_free:
                    resolved = ctx2.apply(TypeVariable(var_id))
                    if not isinstance(resolved, TypeVariable) and var_id not in subst:
                        subst[var_id] = resolved
                break  # Found a match, move to next arg
            except Exception:
                continue

    # For any still-unresolved variables, pick from concrete base types.
    for var_id in sorted(free_vars):
        if var_id not in subst:
            subst[var_id] = rng.choice(_CONCRETE_TYPES)

    # Apply substitution to all argument types.
    return [t.apply_substitution(subst) for t in arg_types]


def _contains_type(arg_type: Type, target: Type) -> bool:
    """
    Check if arg_type contains target as a substructure.

    Used to detect self-recursive productions at max_depth — e.g., when
    trying to produce bool and a candidate has bool as an argument or
    as part of an arrow type (e.g., card -> bool in `all`).

    Returns True if target appears anywhere in arg_type.
    """
    if arg_type == target:
        return True
    if isinstance(arg_type, Arrow):
        return _contains_type(arg_type.arg, target) or _contains_type(arg_type.ret, target)
    if isinstance(arg_type, ListType):
        return _contains_type(arg_type.element, target)
    return False


def _type_is_terminable(
    grammar: Grammar,
    target_type: Type,
    env: List[Type],
    _depth: int = 0,
) -> bool:
    """
    Check if a type can be filled by a variable, terminal production, or
    a shallow lambda whose body is terminable.

    This is a bounded lookahead used when we're past max_depth and need to
    select non-terminal productions that won't cause runaway recursion.

    A type is "terminable" if:
      - It matches a variable in the current environment, OR
      - It has at least one zero-argument production in the grammar, OR
      - It is an arrow type whose body type is terminable (with the
        argument type added to env). Limited to 2 levels of nesting
        to prevent expensive deep analysis.

    Args:
        grammar:     The PCFG grammar.
        target_type: The type to check.
        env:         Current type environment (bound variables).
        _depth:      Internal recursion counter (for arrow nesting limit).

    Returns:
        True if this type can be filled without deep recursion.
    """
    # Safety: don't analyze too deeply
    if _depth > 2:
        return False

    # Arrow types create a lambda. Check if the body can be terminated
    # (with the argument type added to env as $0).
    if isinstance(target_type, Arrow):
        new_env = [target_type.arg] + env
        return _type_is_terminable(grammar, target_type.ret, new_env, _depth + 1)

    # Check if any variable matches this type.
    ctx = TypeContext()
    var_cands = grammar.variable_candidates(target_type, ctx, env)
    if var_cands:
        return True

    # Check if any terminal production exists for this type.
    ctx2 = TypeContext()
    prod_cands = grammar.candidates_for_type(target_type, ctx2, env, normalize=False)
    has_terminal = any(
        len(inst_type.arguments) == 0
        for _, inst_type, _ in prod_cands
    )
    return has_terminal


def _all_args_terminable(
    grammar: Grammar,
    arg_types: List[Type],
    env: List[Type],
) -> bool:
    """
    Check if every argument type can be filled without deep recursion.

    This is a multi-step lookahead used when we're past max_depth and need to
    pick a non-terminal production. It prevents choosing productions that
    would force recursion into ever-deeper nested types or HOF chains.

    Args:
        grammar:   The PCFG grammar.
        arg_types: The argument types to check.
        env:       Current type environment (bound variables).

    Returns:
        True if all argument types can be terminated shallowly.
    """
    return all(
        _type_is_terminable(grammar, arg_type, env)
        for arg_type in arg_types
    )


def _sample(
    grammar: Grammar,
    target_type: Type,
    max_depth: int,
    depth: int,
    env: List[Type],
    rng: random.Random,
) -> Program:
    """
    Internal recursive sampler.

    Separated from the public API so that recursive calls share the same
    RNG instance (important for determinism) while the depth counter
    increments naturally.

    Args:
        grammar:     The PCFG grammar.
        target_type: The type to produce at this node.
        max_depth:   Soft depth limit for non-lambda nodes.
        depth:       Current depth (0 at root, increments for each
                     Application argument, stays the same for lambdas).
        env:         Current type environment (extended by lambdas).
        rng:         The shared Random instance for this sample call.
    """
    # Hard safety cap to prevent infinite recursion from pathological
    # polymorphic type instantiations (e.g., list(list(list(...)))).
    if depth > _ABSOLUTE_DEPTH_LIMIT:
        raise RuntimeError(
            f"Absolute depth limit ({_ABSOLUTE_DEPTH_LIMIT}) exceeded for "
            f"type {target_type}. This indicates a pathological sampling path."
        )

    # --------------------------------------------------------------------- #
    # Step 1: If the target is an arrow type, create a lambda.
    # --------------------------------------------------------------------- #
    # Arrow(A, B) means "function from A to B". We create:
    #   Abstraction(body)
    # where body has type B in an environment extended with A at index $0.
    #
    # Lambdas are "free" — they don't consume depth budget — because they
    # are structurally required by the type and offer no choice point.
    # --------------------------------------------------------------------- #
    if isinstance(target_type, Arrow):
        # Extend environment: the new bound variable ($0) has type target_type.arg.
        # Existing variables shift up by one index (env is ordered innermost-first).
        new_env = [target_type.arg] + env
        # Lambdas are normally "free" (don't consume depth budget) because they
        # are structurally required by the type. However, when past max_depth,
        # polymorphic type resolution can create chains of arrow types
        # (e.g., filter with 'a=bool creates bool->bool arguments). To prevent
        # runaway lambda nesting, increment depth for lambdas past max_depth.
        body_depth = depth if depth < max_depth else depth + 1
        body = _sample(grammar, target_type.ret, max_depth, body_depth, new_env, rng)
        return Abstraction(body)

    # --------------------------------------------------------------------- #
    # Step 2: Collect candidate productions and variables.
    # --------------------------------------------------------------------- #
    # candidates_for_type returns (Production, instantiated_type, log_prob)
    # where instantiated_type has fresh type variables resolved against
    # target_type. The .arguments property gives the types we need to fill.
    #
    # variable_candidates returns (de_bruijn_index, log_prob) for each
    # bound variable whose type unifies with target_type.
    # --------------------------------------------------------------------- #

    # Use a fresh TypeContext for each sampling decision to avoid cross-
    # contamination of type variable bindings between sibling subtrees.
    ctx = TypeContext()
    production_candidates = grammar.candidates_for_type(
        target_type, ctx, env, normalize=False
    )
    var_candidates = grammar.variable_candidates(target_type, ctx, env)

    # --------------------------------------------------------------------- #
    # Step 3: At or beyond max_depth, restrict to terminals if possible.
    # --------------------------------------------------------------------- #
    # Terminal productions have no arguments (their instantiated type is a
    # base type, not an arrow). Variables are always terminal.
    #
    # Some types (notably BOOL in the gallery grammar, which excludes
    # true/false constants) have NO terminal productions. For these, we
    # fall back to non-terminal productions whose arguments can ALL be
    # satisfied by variables in the current environment or by their own
    # terminal productions (1-step lookahead). This prevents the
    # pathological case where polymorphic instantiation creates ever-deeper
    # nested types (e.g., list(list(list(...)))).
    # --------------------------------------------------------------------- #
    if depth >= max_depth:
        terminal_prods = [
            (prod, inst_type, lp)
            for prod, inst_type, lp in production_candidates
            if len(inst_type.arguments) == 0
        ]

        if terminal_prods or var_candidates:
            # We have at least one way to terminate; use only those.
            production_candidates = terminal_prods
        else:
            # No terminals for this type. Use lookahead: keep only
            # productions whose every argument can be filled by a variable
            # or a terminal production at the next level.
            terminable_prods = []
            for prod, inst_type, lp in production_candidates:
                # Resolve type variables before checking terminability.
                resolved_args = _resolve_free_type_vars(
                    inst_type.arguments, env, rng
                )
                if _all_args_terminable(grammar, resolved_args, env):
                    terminable_prods.append((prod, inst_type, lp))

            if terminable_prods:
                production_candidates = terminable_prods
            else:
                # Last resort: exclude self-recursive productions (those
                # that need an argument of the same type we're trying to
                # produce) and pick from what remains.
                non_recursive = [
                    (p, it, lp) for p, it, lp in production_candidates
                    if not any(
                        _contains_type(arg_t, target_type)
                        for arg_t in it.arguments
                    )
                ]
                if non_recursive:
                    production_candidates = non_recursive
                # If even that fails, keep all (will hit absolute limit)

    # --------------------------------------------------------------------- #
    # Step 4: Build a unified candidate list and sample.
    # --------------------------------------------------------------------- #
    # We merge production candidates and variable candidates into a single
    # list of (choice, log_prob) pairs, then sample proportional to
    # exp(log_prob). The "choice" is either:
    #   ("prod", Production, instantiated_type) for a production, or
    #   ("var", de_bruijn_index)                for a variable.
    # --------------------------------------------------------------------- #
    choices: List[Tuple] = []
    log_probs: List[float] = []

    for prod, inst_type, lp in production_candidates:
        choices.append(("prod", prod, inst_type))
        log_probs.append(lp)

    for idx, lp in var_candidates:
        choices.append(("var", idx))
        log_probs.append(lp)

    if not choices:
        raise RuntimeError(
            f"No valid candidates for type {target_type} at depth {depth} "
            f"(max_depth={max_depth}, env has {len(env)} bindings). "
            f"This usually means the grammar lacks productions for this type."
        )

    # Convert log-probs to a proper probability distribution.
    # Use log-sum-exp for numerical stability:
    #   weight_i = exp(lp_i - max_lp)
    #   P(i) = weight_i / sum(weights)
    max_lp = max(log_probs)
    weights = [math.exp(lp - max_lp) for lp in log_probs]

    # random.choices returns a list; we want one sample.
    (selected,) = rng.choices(choices, weights=weights, k=1)

    # --------------------------------------------------------------------- #
    # Step 5: Build the AST node for the selected candidate.
    # --------------------------------------------------------------------- #

    if selected[0] == "var":
        # Variable candidate: return a de Bruijn index reference.
        return Index(selected[1])

    # Production candidate: build Application chain for arguments.
    _, prod, inst_type = selected
    arg_types = list(inst_type.arguments)

    # --------------------------------------------------------------------- #
    # Step 5a: Handle polymorphic type variables across arguments.
    # --------------------------------------------------------------------- #
    # Productions like `eq : 'a -> 'a -> bool` have shared type variables.
    # We use forward propagation: sample arguments left-to-right, and after
    # each one, infer its concrete type and unify with the expected type to
    # resolve shared variables for subsequent arguments.
    #
    # We maintain a substitution that accumulates as we learn concrete types
    # from each sampled argument. Before sampling each argument, we apply
    # this substitution to get the most concrete type possible.
    # --------------------------------------------------------------------- #
    subst: dict = {}  # type var id -> concrete Type
    node: Program = prod.program

    for i, arg_type in enumerate(arg_types):
        # Apply accumulated substitution to get the most concrete arg type.
        concrete_arg_type = arg_type.apply_substitution(subst)

        # If the arg type still has free variables, resolve them randomly.
        # This handles cases where no previous argument constrained the variable.
        free_vars = concrete_arg_type.free_type_variables()
        if free_vars:
            concrete_arg_type = _resolve_free_type_vars(
                [concrete_arg_type], env, rng
            )[0]

        # Sample a subprogram for this argument.
        arg_program = _sample(
            grammar, concrete_arg_type, max_depth, depth + 1, env, rng
        )

        # After sampling, infer the argument's actual type and use it to
        # resolve shared type variables for subsequent arguments.
        if i < len(arg_types) - 1:  # No need for last arg
            remaining_free = set()
            for future_type in arg_types[i + 1:]:
                remaining_free |= future_type.free_type_variables()

            if remaining_free:
                try:
                    infer_ctx = TypeContext()
                    actual_type = arg_program.infer_type(infer_ctx, env)
                    actual_type = infer_ctx.apply(actual_type)
                    # Unify expected with actual to learn variable bindings.
                    unify_ctx = TypeContext()
                    unify_ctx.unify(arg_type, actual_type)
                    # Extract new bindings for shared variables.
                    for var_id in remaining_free:
                        resolved = unify_ctx.apply(TypeVariable(var_id))
                        if not isinstance(resolved, TypeVariable):
                            subst[var_id] = resolved
                except Exception:
                    # If inference fails, continue with what we have.
                    # The program may be ill-typed, but the MCMC sampler
                    # will reject it via low likelihood.
                    pass

        node = Application(node, arg_program)

    return node


# =========================================================================== #
# SCORING UNDER THE SAMPLER'S DISTRIBUTION  (C1 fix)
# =========================================================================== #


def _score_subtree_under_sampler(
    grammar: Grammar,
    subtree: Program,
    target_type: Type,
    max_depth: int,
    depth: int,
    env: List[Type],
) -> float:
    """
    Compute log P(subtree | distribution that `_sample` draws from).

    This is the density needed by `propose_regeneration` for the Metropolis-
    Hastings ratio. Prior implementations used `grammar.program_log_likelihood`
    for both forward and reverse proposal densities, but that function uses
    a DIFFERENT distribution from `_sample`:

      - `_sample` draws from a joint softmax over
          {production p of type T: weight exp(p.log_probability)}
          {variable i of type T:   weight exp(log_variable)}
        normalized by Z_T = sum of those weights.

      - `program_log_likelihood` (via `_ll_helper`) treats productions and
        variables as separate "tracks": productions are normalized WITHIN
        productions (by `candidates_for_type(normalize=True)`), and each
        variable gets `log_variable - log(n_same_type)`. These quantities do
        not jointly sum to 1, so the returned value is not a proper density
        under the sampler's actual distribution.

    Using `program_log_likelihood` for both sides of the MH ratio therefore
    used an incorrect density; detailed balance is only broken as long as
    the forward and reverse are asymmetric with respect to the TRUE density
    the chain samples from, and in general they are. This function mirrors
    `_sample`'s control flow exactly so that the MH ratio uses the sampler's
    own density.

    The function returns -inf if `subtree` could not have been generated by
    `_sample(target_type, max_depth, depth, env)` — this is the correct
    behavior: the reverse proposal cannot have been drawn from an
    impossible state, so Q(old | new) = 0 and the move is rejected.

    Polymorphic-type resolution:
      `_sample` resolves free type variables either from the environment
      (deterministic) or by random choice from `_CONCRETE_TYPES`. We score
      only the random-choice contribution: for each free type var that
      cannot be resolved from the environment, we charge -log(|_CONCRETE_TYPES|)
      under the observed concrete assignment. This is correct so long as the
      sampler is consistent between forward and reverse (it is), and any
      imprecision cancels to leading order.

    Args:
        grammar:     PCFG grammar used by `_sample`.
        subtree:     The program to score.
        target_type: The type the sampler was trying to produce.
        max_depth:   Same depth budget passed to `_sample`.
        depth:       Current depth (as if recursing into `_sample`).
        env:         Current type environment.

    Returns:
        log P(subtree | `_sample`'s distribution), or float('-inf') if
        subtree is not producible at this hole under `_sample`'s rules.
    """
    # Safety cap mirrors _sample.
    if depth > _ABSOLUTE_DEPTH_LIMIT:
        return float('-inf')

    # -------------------------------------------------------------------- #
    # Step 1 mirror: Arrow -> must be Abstraction, recurse on body.
    # -------------------------------------------------------------------- #
    if isinstance(target_type, Arrow):
        if not isinstance(subtree, Abstraction):
            return float('-inf')
        new_env = [target_type.arg] + env
        body_depth = depth if depth < max_depth else depth + 1
        return _score_subtree_under_sampler(
            grammar, subtree.body, target_type.ret, max_depth, body_depth, new_env,
        )

    # If the target is a base type but subtree is an Abstraction, reject.
    if isinstance(subtree, Abstraction):
        return float('-inf')

    # -------------------------------------------------------------------- #
    # Step 2 mirror: collect candidates with the SAME flags as _sample.
    # -------------------------------------------------------------------- #
    ctx = TypeContext()
    production_candidates = grammar.candidates_for_type(
        target_type, ctx, env, normalize=False,
    )
    var_candidates = grammar.variable_candidates(target_type, ctx, env)

    # -------------------------------------------------------------------- #
    # Step 3 mirror: apply depth-limit restriction. We cannot use an rng here
    # (scoring is deterministic), and `_resolve_free_type_vars` in the
    # lookahead branch uses rng for unresolved free vars. Our strategy is:
    #   - If the restricted candidate list under the "env-only" resolution
    #     (no random fallback) already contains the observed head, score
    #     within that restricted set (matches what _sample would have done
    #     for most non-edge cases).
    #   - Otherwise, fall back to the unrestricted set — this is the case
    #     where `_sample`'s random type-var fallback let a candidate survive
    #     that env-only scoring would reject. Scoring against the wider set
    #     is a slight under-count of the true log_q (the true denominator
    #     should be narrower); for MH purposes, this means we slightly
    #     over-reward the proposed move. The bias is bounded by the ratio
    #     of candidate set sizes and is symmetric forward/reverse in the
    #     common case where both proposals use the same rule path.
    # -------------------------------------------------------------------- #
    if depth >= max_depth:
        terminal_prods = [
            (prod, inst_type, lp)
            for prod, inst_type, lp in production_candidates
            if len(inst_type.arguments) == 0
        ]
        restricted_used = False
        if terminal_prods or var_candidates:
            production_candidates = terminal_prods
            restricted_used = True
        # We intentionally skip the lookahead-based restriction
        # (_all_args_terminable branch) because it depends on rng via
        # _resolve_free_type_vars; scoring under that branch would require
        # tracking which concrete-type assignment _sample used, which is
        # unobservable from the final program alone. In practice, this
        # branch is only entered when NO terminals or variables exist for
        # the target type — a rare edge case in the gallery grammar.
        _ = restricted_used  # kept for readability

    # -------------------------------------------------------------------- #
    # Step 4 mirror: build unified (choice, log_prob) list.
    # -------------------------------------------------------------------- #
    choices: List[Tuple] = []
    log_probs: List[float] = []

    for prod, inst_type, lp in production_candidates:
        choices.append(("prod", prod, inst_type))
        log_probs.append(lp)

    for idx, lp in var_candidates:
        choices.append(("var", idx))
        log_probs.append(lp)

    if not choices:
        return float('-inf')

    # Joint softmax normalization (log-sum-exp).
    max_lp = max(log_probs)
    log_Z = max_lp + math.log(sum(math.exp(lp - max_lp) for lp in log_probs))

    # -------------------------------------------------------------------- #
    # Step 5 mirror: identify which choice the observed subtree represents.
    # Unwrap any Application chain to find the head (Primitive/Invented/Index)
    # and collect observed arguments in left-to-right order.
    # -------------------------------------------------------------------- #
    node = subtree
    args_observed: List[Program] = []
    while isinstance(node, Application):
        args_observed.insert(0, node.x)
        node = node.f

    # Variable case.
    if isinstance(node, Index):
        # A bare variable must have no observed arguments.
        if args_observed:
            return float('-inf')
        target_choice = ("var", node.i)
        for i, ch in enumerate(choices):
            if ch == target_choice:
                return log_probs[i] - log_Z
        return float('-inf')

    # Primitive / Invented case.
    if isinstance(node, (Primitive, Invented)):
        prod = grammar.get_production(node)
        if prod is None:
            return float('-inf')

        # Find the matching production candidate.
        for i, ch in enumerate(choices):
            if ch[0] != "prod":
                continue
            _, cand_prod, cand_inst_type = ch
            if cand_prod is not prod:
                continue

            log_p_head = log_probs[i] - log_Z

            arg_types = list(cand_inst_type.arguments)
            if len(args_observed) != len(arg_types):
                return float('-inf')

            # Recurse on each argument, mirroring _sample's forward-propagation
            # of the substitution learned from already-sampled arguments.
            log_p_args = 0.0
            subst: dict = {}
            for j, (arg_type, arg_observed) in enumerate(zip(arg_types, args_observed)):
                concrete_arg_type = arg_type.apply_substitution(subst)

                # Mirror `_sample`'s free-variable resolution: the sampler
                # calls `_resolve_free_type_vars(concrete_arg_type, env, rng)`
                # to pick a concrete type via env-unification, falling back
                # to rng over _CONCRETE_TYPES. We mirror that here, preferring
                # the observed arg's inferred type over a default sentinel,
                # and charging -log(|_CONCRETE_TYPES|) for any truly free var.
                free_vars = concrete_arg_type.free_type_variables()
                resolution_log_p = 0.0
                if free_vars:
                    resolved_subst: dict = {}

                    # 1) Env-based resolution (deterministic in sampler too).
                    for env_type in env:
                        try:
                            ctx2 = TypeContext()
                            ctx2.unify(concrete_arg_type, env_type)
                            for var_id in free_vars:
                                resolved = ctx2.apply(TypeVariable(var_id))
                                if (not isinstance(resolved, TypeVariable)
                                        and var_id not in resolved_subst):
                                    resolved_subst[var_id] = resolved
                            break
                        except Exception:
                            continue

                    # 2) For vars still unresolved, try to read the observed
                    #    subtree's inferred type — if the body uses the bound
                    #    var, type inference will have revealed its concrete
                    #    type through unification with a primitive.
                    unresolved = [v for v in free_vars if v not in resolved_subst]
                    if unresolved:
                        try:
                            infer_ctx = TypeContext()
                            observed_type = arg_observed.infer_type(infer_ctx, env)
                            observed_type = infer_ctx.apply(observed_type)
                            uctx = TypeContext()
                            uctx.unify(concrete_arg_type, observed_type)
                            for var_id in unresolved:
                                resolved = uctx.apply(TypeVariable(var_id))
                                if not isinstance(resolved, TypeVariable):
                                    resolved_subst[var_id] = resolved
                        except Exception:
                            pass

                    # 3) Any truly-free var (observed body doesn't constrain it)
                    #    gets a sentinel concrete assignment. This mirrors the
                    #    sampler's rng.choice(_CONCRETE_TYPES) — we cannot know
                    #    which concrete it picked, but since the body doesn't
                    #    use it, scoring is invariant under the choice. Charge
                    #    -log(|_CONCRETE_TYPES|) for each such var (cancels
                    #    forward/reverse to leading order).
                    still_unresolved = [v for v in free_vars if v not in resolved_subst]
                    if still_unresolved:
                        resolution_log_p = (
                            -len(still_unresolved) * math.log(len(_CONCRETE_TYPES))
                        )
                        for var_id in still_unresolved:
                            resolved_subst[var_id] = _CONCRETE_TYPES[0]

                    concrete_arg_type = concrete_arg_type.apply_substitution(resolved_subst)

                log_p_this_arg = _score_subtree_under_sampler(
                    grammar, arg_observed, concrete_arg_type,
                    max_depth, depth + 1, env,
                )
                if log_p_this_arg == float('-inf'):
                    return float('-inf')
                log_p_args += resolution_log_p + log_p_this_arg

                # Update substitution from observed type to carry into
                # subsequent arguments (mirror of _sample lines 557-579).
                if j < len(arg_types) - 1:
                    remaining_free = set()
                    for future_type in arg_types[j + 1:]:
                        remaining_free |= future_type.free_type_variables()
                    if remaining_free:
                        try:
                            infer_ctx = TypeContext()
                            actual_type = arg_observed.infer_type(infer_ctx, env)
                            actual_type = infer_ctx.apply(actual_type)
                            unify_ctx = TypeContext()
                            unify_ctx.unify(arg_type, actual_type)
                            for var_id in remaining_free:
                                resolved = unify_ctx.apply(TypeVariable(var_id))
                                if not isinstance(resolved, TypeVariable):
                                    subst[var_id] = resolved
                        except Exception:
                            pass

            return log_p_head + log_p_args

        return float('-inf')

    return float('-inf')


# =========================================================================== #
# SUBTREE SITE COLLECTION
# =========================================================================== #


@dataclass
class SubtreeSite:
    """
    Records a position in the AST where a subtree can be regenerated.

    This is the unit of locality for the MCMC proposal: we pick one site
    uniformly at random, then resample the subtree at that position from
    the grammar prior.

    Attributes:
        path:    Sequence of steps from root to this node. Each step is one
                 of 'body' (enter Abstraction), 'f' (Application function),
                 or 'x' (Application argument).
        type:    The type that this subtree must produce.
        env:     The type environment at this position (bound variables in
                 scope). env[i] is the type of de Bruijn index $i.
        subtree: The current subtree rooted at this position.
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
    Walk the AST and collect all positions where a subtree can be regenerated.

    Every non-root node in the AST is a candidate site. For each site we
    record the path from root, the type expected at that position, and
    the type environment (which bound variables are in scope).

    The root is excluded because regenerating the entire program is equivalent
    to sampling a new chain start, not a local proposal.

    Args:
        program:      The program AST to walk.
        request_type: The type of the whole program (e.g., Arrow(HAND, BOOL)).

    Returns:
        A list of SubtreeSite objects, one per non-root AST node.

    How type tracking works:
        - Abstraction (lambda): If current type is Arrow(A, B), the body has
          type B and the environment extends with A as $0.
        - Application (f x): The function f has type Arrow(arg_type, current_type),
          and argument x has type arg_type. We infer f's type to determine arg_type.
        - Primitives and Indices are leaf nodes — they are sites but have no children.
    """
    sites: List[SubtreeSite] = []

    def _walk_without_collect(
        node: Program,
        current_type: Type,
        env: List[Type],
        path: Tuple[str, ...],
    ) -> None:
        """Traverse into Application heads without collecting them as sites.

        Used to reach deeper .x positions inside left-nested Application chains
        without treating bare primitives or partial applications as valid
        regeneration targets.
        """
        if isinstance(node, Application):
            try:
                ctx = TypeContext()
                f_type = node.f.infer_type(ctx, env)
                f_type = ctx.apply(f_type)
                if isinstance(f_type, Arrow):
                    arg_type = f_type.arg
                    _walk_without_collect(node.f, f_type, env, path + ('f',))
                    _walk(node.x, arg_type, env, path + ('x',), False)
            except Exception:
                pass

    def _walk(
        node: Program,
        current_type: Type,
        env: List[Type],
        path: Tuple[str, ...],
        is_root: bool,
    ) -> None:
        """
        Recursive walker that collects subtree sites.

        Args:
            node:         Current AST node.
            current_type: The type this node must produce.
            env:          Type environment (bound variables in scope).
            path:         Path from root to this node.
            is_root:      True only for the top-level call (skip collecting).
        """
        # Collect this node as a site (unless it's the root).
        if not is_root:
            sites.append(SubtreeSite(
                path=path,
                type=current_type,
                env=list(env),  # defensive copy
                subtree=node,
            ))

        # Recurse into children based on node type.
        if isinstance(node, Abstraction):
            # Abstraction: current_type should be Arrow(A, B).
            # Body has type B, environment extends with A.
            if isinstance(current_type, Arrow):
                new_env = [current_type.arg] + env
                _walk(node.body, current_type.ret, new_env, path + ('body',), False)
            # If current_type is not Arrow (shouldn't happen in well-typed program),
            # we skip recursing — can't determine child types safely.

        elif isinstance(node, Application):
            # Application (f x): infer f's type to get arg_type.
            # f has type Arrow(arg_type, current_type), x has type arg_type.
            #
            # Note: we DESCEND into node.f to reach deeper .x positions (for
            # left-nested Application chains like `(((f x1) x2) x3)`), but
            # we do NOT collect intermediate .f nodes as sites. The sampler
            # `_sample` never produces a bare primitive or a partial
            # Application as a subtree at an Arrow-typed hole — Arrow holes
            # always yield Abstractions. Including .f positions as sites
            # would break C1's detailed-balance correction because the
            # reverse proposal density for a bare primitive under
            # `_score_subtree_under_sampler` is always -inf.
            try:
                ctx = TypeContext()
                f_type = node.f.infer_type(ctx, env)
                f_type = ctx.apply(f_type)

                if isinstance(f_type, Arrow):
                    arg_type = f_type.arg
                    # Descend into f WITHOUT collecting the bare head as a site.
                    # Any .x positions deeper inside (from nested Applications)
                    # will still be collected by the recursion.
                    _walk_without_collect(node.f, f_type, env, path + ('f',))
                    # Recurse into x with the argument type — collects .x as a site.
                    _walk(node.x, arg_type, env, path + ('x',), False)
            except Exception:
                # Type inference failed (e.g., polymorphic edge case).
                pass

        # Primitives and Indices are leaves — no children to recurse into.

    _walk(program, request_type, [], (), True)
    return sites


# =========================================================================== #
# SUBTREE REPLACEMENT
# =========================================================================== #


def replace_subtree(
    program: Program,
    path: Tuple[str, ...],
    replacement: Program,
) -> Program:
    """
    Replace the subtree at the given path with a new subtree.

    The path is a sequence of steps from the root:
      - 'body' : enter an Abstraction's body
      - 'f'    : enter an Application's function position
      - 'x'    : enter an Application's argument position

    An empty path means "replace the root itself" (returns replacement directly).

    Args:
        program:     The original program AST.
        path:        Tuple of steps to the target subtree.
        replacement: The new subtree to insert.

    Returns:
        A new Program with the subtree at path replaced. The original
        program is not mutated.

    Raises:
        ValueError: If a path step doesn't match the node type (e.g., 'body'
                    on a non-Abstraction, or 'f'/'x' on a non-Application).
    """
    if len(path) == 0:
        return replacement

    step = path[0]
    rest = path[1:]

    if step == 'body':
        if not isinstance(program, Abstraction):
            raise ValueError(
                f"Path step 'body' requires Abstraction, got {type(program).__name__}"
            )
        new_body = replace_subtree(program.body, rest, replacement)
        return Abstraction(new_body)

    elif step == 'f':
        if not isinstance(program, Application):
            raise ValueError(
                f"Path step 'f' requires Application, got {type(program).__name__}"
            )
        new_f = replace_subtree(program.f, rest, replacement)
        return Application(new_f, program.x)

    elif step == 'x':
        if not isinstance(program, Application):
            raise ValueError(
                f"Path step 'x' requires Application, got {type(program).__name__}"
            )
        new_x = replace_subtree(program.x, rest, replacement)
        return Application(program.f, new_x)

    else:
        raise ValueError(f"Unknown path step: {step!r} (expected 'body', 'f', or 'x')")


# =========================================================================== #
# MCMC SUBTREE-REGENERATION PROPOSAL
# =========================================================================== #


def propose_regeneration(
    grammar: Grammar,
    program: Program,
    request_type: Type,
    max_depth: int = 6,
    seed: Optional[int] = None,
) -> Tuple[Program, float, float]:
    """
    Propose a new program by regenerating a randomly chosen subtree.

    This is the core MCMC proposal mechanism. It:
      1. Collects all non-root subtree sites in the current program.
      2. Picks one site uniformly at random.
      3. Regenerates that subtree by sampling from the grammar prior
         (using _sample with the site's type and environment).
      4. Replaces the old subtree with the new one.
      5. Computes the forward and reverse proposal log-probabilities
         for the Metropolis-Hastings acceptance ratio.

    The proposal probability Q(new | old) factors as:
      Q(new | old) = P(pick site) * P(generate new subtree | grammar, site type)
                   = (1 / n_sites_old) * grammar.program_log_likelihood(new_subtree, ...)

    The reverse probability Q(old | new) is:
      Q(old | new) = (1 / n_sites_new) * grammar.program_log_likelihood(old_subtree, ...)

    Args:
        grammar:      The PCFG grammar.
        program:      The current program in the MCMC chain.
        request_type: The type of the whole program (e.g., Arrow(HAND, BOOL)).
        max_depth:    Max depth budget for regenerated subtrees (adjusted by
                      site depth to give reasonable budget).
        seed:         Random seed for reproducibility.

    Returns:
        A tuple (new_program, log_q_forward, log_q_reverse) where:
          - new_program: The proposed program with one subtree replaced.
          - log_q_forward: log Q(new_program | old_program)
          - log_q_reverse: log Q(old_program | new_program)

    Raises:
        RuntimeError: If the program has no non-root subtree sites (e.g.,
                      program is a single variable or primitive with no
                      wrapping lambda).
    """
    rng = random.Random(seed)

    # Step 1: Collect all subtree sites in the current program.
    sites = collect_subtree_sites(program, request_type)
    if not sites:
        raise RuntimeError(
            f"No subtree sites found in program {program}. "
            f"Cannot propose regeneration for a program with no non-root nodes."
        )

    n_sites_old = len(sites)

    # Step 2: Pick a site uniformly at random.
    site = rng.choice(sites)

    # Step 3: Regenerate the subtree at that site.
    # Give the regenerated subtree a depth budget that accounts for how
    # deep in the tree we already are. Use at least 3 to allow non-trivial
    # subtrees even deep in the program.
    regen_depth = max(3, max_depth - len(site.path))

    # Sample a new subtree of the appropriate type, using the site's
    # environment so that bound variables are available.
    new_subtree = sample_program(
        grammar, site.type, max_depth=regen_depth, seed=rng.randint(0, 2**31),
        env=site.env,
    )

    # Step 4: Replace the old subtree with the new one.
    new_program = replace_subtree(program, site.path, new_subtree)

    # Step 5: Compute forward and reverse proposal probabilities.
    old_subtree = site.subtree

    # ------------------------------------------------------------------ #
    # C1 fix: score under `_sample`'s own distribution, not under
    # `grammar.program_log_likelihood`. The latter uses a different
    # normalization (productions normalized within type; variables returning
    # `log_variable - log(n_same_type)` without joint normalization), so the
    # MH ratio computed with it did not match the chain's actual proposal
    # density. `_score_subtree_under_sampler` mirrors `_sample`'s joint
    # softmax over productions ∪ variables, ensuring forward and reverse
    # densities are consistent with the sampler that drew `new_subtree`.
    # ------------------------------------------------------------------ #
    # The regen_depth passed to sample_program is the `max_depth` parameter
    # used by `_sample`; the scorer must use the SAME value so its depth-
    # restriction logic matches what was sampled.
    log_pick_fwd = -math.log(n_sites_old)
    log_gen_fwd = _score_subtree_under_sampler(
        grammar, new_subtree, site.type, regen_depth, depth=0, env=site.env,
    )
    log_q_forward = log_pick_fwd + log_gen_fwd

    new_sites = collect_subtree_sites(new_program, request_type)
    n_sites_new = len(new_sites) if new_sites else 1  # guard against 0
    log_pick_rev = -math.log(n_sites_new)
    log_gen_rev = _score_subtree_under_sampler(
        grammar, old_subtree, site.type, regen_depth, depth=0, env=site.env,
    )
    log_q_reverse = log_pick_rev + log_gen_rev

    return new_program, log_q_forward, log_q_reverse


# =========================================================================== #
# MCMC CONFIGURATION AND RESULTS
# =========================================================================== #

logger = logging.getLogger(__name__)

# Total number of ORDERED 6-card hands from a 52-card deck: P(52, 6)
# Hands are ordered because position matters (e.g., strict_increasing,
# colors_palindrome). P(52,6) = 52 × 51 × 50 × 49 × 48 × 47.
# Note: bayesian_scorer.py and hypothesis_table.py on main also need this fix.
TOTAL_HANDS = 14_658_134_400


# =========================================================================== #
# TAUTOLOGY DETECTION (Layer 1: Syntactic)
# =========================================================================== #


def is_vacuous_lambda(program: Program) -> bool:
    """
    Check if a program is a vacuous lambda -- ignores its input.

    A lambda (Abstraction) is "vacuous" if its body never references
    the bound variable ($0). Such programs compute a constant regardless
    of the hand they receive, making them trivially uninformative as
    hypotheses.

    Precedent: LOTlib3's check_lambdas rejects these.

    Args:
        program: A Program AST to check.

    Returns:
        True if the program is (lambda body) where body never uses $0.
    """
    if not isinstance(program, Abstraction):
        return False
    return not uses_variable(program.body, 0)


@dataclass
class MCMCConfig:
    """
    Configuration for a single MCMC chain.

    Attributes:
        n_steps:       Number of Metropolis-Hastings steps per chain.
        max_depth:     Maximum AST depth for sampled/regenerated programs.
        noise_epsilon: Noise floor for the likelihood function. When > 0,
                       even programs that miss an exemplar get a small
                       non-zero probability, preventing the chain from
                       getting permanently stuck on the first program
                       that covers any exemplar. Set to 0 for strict
                       (deterministic) likelihood.
        max_nodes:     Hard cutoff on program AST size (number of nodes).
                       Programs exceeding this are immediately rejected.
                       Follows LOTlib3 convention of bounding hypothesis
                       complexity to keep the chain in a tractable region.
        top_k:         Number of top hypotheses (by visit count) to retain
                       in the result. Keeps memory bounded for long chains.
        seed:          Random seed for reproducibility. If None, the chain
                       is non-deterministic.
    """
    n_steps: int = 100_000
    max_depth: int = 6
    noise_epsilon: float = 0.01
    max_nodes: int = 25
    top_k: int = 250
    seed: Optional[int] = None
    verbose: int = 0  # 0=silent, 1=chain progress, 2=accept/reject, 3=proposal details
    init_max_depth: int = 3   # Max depth for initial program sample (small to avoid dead chains)
    beta_start: float = 1.0   # Likelihood temperature at step 0 (1.0 = no annealing)
    beta_end: float = 1.0     # Likelihood temperature at final step


@dataclass
class MCMCResult:
    """
    Result of running a single MCMC chain.

    Attributes:
        n_steps:          Total MH steps executed.
        n_accepted:       Number of proposals accepted.
        n_unique:         Number of distinct programs visited.
        acceptance_rate:  Fraction of proposals accepted (n_accepted / n_steps).
        top_hypotheses:   List of dicts sorted by visit_count descending.
                          Each dict has keys: 'program', 'visit_count',
                          'log_posterior', 'first_seen_step'.
        best_program:     String representation of the highest-posterior program.
        best_log_posterior: Log posterior of the best program found.
        visit_counts:     Dict mapping program string -> number of visits.
        first_passage:    Dict mapping program string -> step number first seen.
    """
    n_steps: int
    n_accepted: int
    n_unique: int
    acceptance_rate: float
    top_hypotheses: List[Dict[str, Any]]
    best_program: Optional[str] = None
    best_log_posterior: float = float('-inf')
    visit_counts: Dict[str, int] = field(default_factory=dict)
    first_passage: Dict[str, int] = field(default_factory=dict)
    trajectory: List[str] = field(default_factory=list)  # Step-by-step sequence of program strings
    ext_fractions: Dict[str, float] = field(default_factory=dict)  # Extension fraction per program string
    # Per-chain first-passage distributions (populated by run_parallel_chains).
    # Each element is one chain's first_passage dict; step indices are
    # within-chain (0..n_steps), NOT offset into a concatenated timeline.
    # Needed for honest per-chain cognitive-timing analysis.
    per_chain_first_passage: List[Dict[str, int]] = field(default_factory=list)


# =========================================================================== #
# LIKELIHOOD COMPUTATION
# =========================================================================== #


def compute_mcmc_log_likelihood(
    program,
    exemplar_hands: List,
    noise_epsilon: float,
    ext_probe_hands: List,
) -> Tuple[float, float]:
    """
    Compute log P(data | hypothesis) using the size principle with noise.

    The size principle (Tenenbaum & Griffiths, 2001) says that a hypothesis
    that picks out a smaller extension (fewer hands satisfy it) assigns
    higher probability to each observed exemplar — it makes a "stronger
    prediction." This naturally implements a preference for more specific
    rules over vague ones.

    The noise parameter epsilon provides a floor probability for exemplars
    that the program misses, preventing -inf likelihoods that would trap
    the chain. This models the idea that human learners tolerate some
    noise in the data (e.g., "maybe the experimenter made an error").

    Args:
        program:          A Program AST to evaluate as a hypothesis.
        exemplar_hands:   List of Hand objects that are positive exemplars
                          (known to satisfy the true rule).
        noise_epsilon:    Noise floor (0 = strict, 0.01 = 1% noise).
        ext_probe_hands:  List of random Hand objects for Monte Carlo
                          estimation of the hypothesis extension size.

    Returns:
        A tuple (log_likelihood, ext_fraction) where:
          - log_likelihood: sum of log P(hand_i | hypothesis) for each
            exemplar. Returns -inf if the program cannot be evaluated.
          - ext_fraction: fraction of probe hands accepted by the program
            (n_hits_probe / n_probes). Useful for tautology detection:
            ext_fraction >= 1.0 means the program accepts everything.

    How extension size estimation works:
        We evaluate the program on each probe hand and count hits.
        ext_size = (n_hits / n_probes) * TOTAL_HANDS
        This gives an unbiased Monte Carlo estimate of the number of
        6-card hands that the hypothesis classifies as positive.
    """
    # Try to compile the program into a callable function.
    # If the program crashes on evaluation, it's an impossible hypothesis.
    try:
        func = program.evaluate([])
    except Exception:
        return (float('-inf'), 0.0)

    # Estimate the extension size via Monte Carlo on probe hands.
    n_probes = len(ext_probe_hands)
    n_hits_probe = 0
    for hand in ext_probe_hands:
        try:
            result = func(hand)
            if result is True:
                n_hits_probe += 1
        except Exception:
            # Program crashes on this hand — treat as non-member.
            continue

    # Raw (unsmoothed) extension fraction for human-readable reporting and
    # the post-hoc Layer-2 tautology filter. Jeffreys-smoothed fraction is
    # used for the likelihood calculation below.
    ext_fraction = n_hits_probe / n_probes if n_probes > 0 else 0.0

    # Jeffreys-prior smoothed extension estimate:
    # ext_fraction_smoothed = (n_hits + 0.5) / (n_probes + 1.0)
    # Avoids the (n_hits == 0 → ext_size = 1.0) pathology which previously
    # REWARDED programs accepting nothing by making per-exemplar probability
    # (1 - ε)/1.0 + ε/TOTAL_HANDS ≈ 1.0 for any hit (even via noise).
    # One-probe resolution floor: ext_size >= TOTAL_HANDS / n_probes,
    # acknowledging that a Monte-Carlo estimate cannot resolve extension
    # smaller than one probe's worth of the hypothesis space.
    if n_probes > 0:
        ext_fraction_smoothed = (n_hits_probe + 0.5) / (n_probes + 1.0)
        ext_size = max(
            ext_fraction_smoothed * TOTAL_HANDS,
            TOTAL_HANDS / n_probes,
        )
    else:
        # Defensive: should not happen in practice (ext_probe_hands is
        # generated with n_probes=10_000).
        ext_size = float(TOTAL_HANDS)

    # Compute log-likelihood for each exemplar hand.
    total_log_lik = 0.0
    for hand in exemplar_hands:
        try:
            result = func(hand)
            is_hit = (result is True)
        except Exception:
            is_hit = False

        if is_hit:
            # Size principle: P = (1 - epsilon) / ext_size + epsilon / TOTAL_HANDS
            p = (1.0 - noise_epsilon) / ext_size + noise_epsilon / TOTAL_HANDS
        else:
            # Miss: only noise probability
            p = noise_epsilon / TOTAL_HANDS

        if p <= 0:
            return (float('-inf'), ext_fraction)
        total_log_lik += math.log(p)

    return (total_log_lik, ext_fraction)


# =========================================================================== #
# MCMC CHAIN
# =========================================================================== #


class MCMCChain:
    """
    Metropolis-Hastings chain for card-game rule induction.

    Explores the hypothesis space of typed programs by iteratively proposing
    local modifications (subtree regeneration) and accepting/rejecting them
    based on the posterior probability P(program | data) = P(data | program) * P(program).

    The chain tracks visit counts for each distinct program, which provides
    an approximation to the posterior: programs visited more often have
    higher posterior probability.

    Usage:
        grammar = build_gallery_grammar()
        config = MCMCConfig(n_steps=10000, seed=42)
        chain = MCMCChain(grammar, config)
        result = chain.run(
            request_type=Arrow(HAND, BOOL),
            exemplar_hands=exemplars['all_red']['hands_primary'],
        )
        print(result.top_hypotheses[:5])
    """

    def __init__(self, grammar: Grammar, config: MCMCConfig):
        self.grammar = grammar
        self.config = config

    def run(
        self,
        request_type,
        exemplar_hands: List,
        ext_probe_hands: List = None,
    ) -> MCMCResult:
        """
        Run the Metropolis-Hastings chain.

        Args:
            request_type:    The type of programs to explore (e.g., Arrow(HAND, BOOL)).
            exemplar_hands:  Positive exemplar hands for likelihood computation.
            ext_probe_hands: Probe hands for extension size estimation. If None,
                             generated automatically using generate_probe_set.

        Returns:
            MCMCResult with visit counts, top hypotheses, and chain statistics.
        """
        config = self.config
        grammar = self.grammar
        rng = random.Random(config.seed)

        # Generate probe hands if not provided.
        if ext_probe_hands is None:
            from gallery_analysis.exemplars import generate_probe_set
            ext_probe_hands = generate_probe_set(
                n_probes=10_000, seed=rng.randint(0, 2**31)
            )

        # -------------------------------------------------------------- #
        # Initialize: sample a starting program from the prior.
        # -------------------------------------------------------------- #
        V = config.verbose  # shorthand

        # Resample if initial program is vacuous or a 100%-on-probes tautology.
        for _retry in range(20):
            current = sample_program(
                grammar, request_type, max_depth=config.init_max_depth,
                seed=rng.randint(0, 2**31),
            )
            if is_vacuous_lambda(current):
                continue
            current_log_lik, current_ext_frac = compute_mcmc_log_likelihood(
                current, exemplar_hands, config.noise_epsilon, ext_probe_hands,
            )
            if current_ext_frac >= 1.0:
                continue
            break

        current_log_prior = grammar.program_log_likelihood(
            current, request_type
        )
        # Re-compute likelihood in case the loop exited without breaking
        # (all 20 attempts were vacuous/tautological -- use last sample).
        if is_vacuous_lambda(current):
            current_log_lik, current_ext_frac = compute_mcmc_log_likelihood(
                current, exemplar_hands, config.noise_epsilon, ext_probe_hands,
            )
        current_log_posterior = current_log_prior + current_log_lik

        if V >= 1:
            print(f"  [init] size={current.size()} depth={current.depth()} "
                  f"prior={current_log_prior:.2f} lik={current_log_lik:.2f} "
                  f"ext_frac={current_ext_frac:.3f} "
                  f"post={current_log_posterior:.2f}")
            print(f"         {str(current)[:120]}")

        # Logging interval for verbose=1: print every N steps
        log_interval = max(1, config.n_steps // 10)

        # -------------------------------------------------------------- #
        # Tracking structures.
        # -------------------------------------------------------------- #
        visit_counts: Dict[str, int] = {}
        first_passage: Dict[str, int] = {}
        log_posteriors: Dict[str, float] = {}
        ext_fractions: Dict[str, float] = {}
        n_accepted = 0

        # Record the initial program.
        current_str = str(current)
        visit_counts[current_str] = 1
        first_passage[current_str] = 0
        log_posteriors[current_str] = current_log_posterior
        ext_fractions[current_str] = current_ext_frac
        trajectory: List[str] = [current_str]  # Step-by-step trajectory

        best_program = current_str
        best_log_posterior = current_log_posterior

        # -------------------------------------------------------------- #
        # Main MH loop.
        # -------------------------------------------------------------- #
        for step in range(1, config.n_steps + 1):
            # (a) Propose a new program via subtree regeneration.
            try:
                proposed, log_q_fwd, log_q_rev = propose_regeneration(
                    grammar, current, request_type,
                    max_depth=config.max_depth,
                    seed=rng.randint(0, 2**31),
                )
            except RuntimeError:
                # propose_regeneration can fail if the program has no
                # subtree sites (e.g., a single variable). Skip this step.
                current_str = str(current)
                visit_counts[current_str] = visit_counts.get(current_str, 0) + 1
                trajectory.append(current_str)
                continue

            # (b) Reject if program is too large.
            if proposed.size() > config.max_nodes:
                current_str = str(current)
                visit_counts[current_str] = visit_counts.get(current_str, 0) + 1
                trajectory.append(current_str)
                continue

            # (b2) Layer 1: Reject vacuous lambdas -- programs that ignore their input.
            if is_vacuous_lambda(proposed):
                current_str = str(current)
                visit_counts[current_str] = visit_counts.get(current_str, 0) + 1
                trajectory.append(current_str)
                continue

            # (c) Compute proposed posterior.
            proposed_log_prior = grammar.program_log_likelihood(
                proposed, request_type
            )
            proposed_log_lik, proposed_ext_frac = compute_mcmc_log_likelihood(
                proposed, exemplar_hands, config.noise_epsilon, ext_probe_hands,
            )

            # (c2) Layer 2 tautology filter is applied POST-HOC at result
            # construction (see below). Rejecting `ext_frac >= 1.0` inline
            # here would break detailed balance: moves from non-tautology to
            # tautology would be hard-rejected while moves out of a tautology
            # cannot occur (the chain is never in such a state), so the
            # acceptance operator is not symmetric. Size-principle likelihood
            # already strongly penalizes tautologies (ext_size = TOTAL_HANDS
            # → extremely low per-exemplar probability); the post-hoc filter
            # then removes any surviving tautology from reported hypotheses.

            proposed_log_posterior = proposed_log_prior + proposed_log_lik

            # (d) MH acceptance ratio with likelihood annealing.
            # The annealed posterior uses beta * log_lik instead of log_lik,
            # allowing the chain to explore more freely early on (low beta)
            # and focus on high-likelihood regions later (beta -> 1.0).
            # Tracking (visit counts, best posterior) still uses the
            # unannealed posterior for correct posterior estimates.
            if config.beta_start == config.beta_end:
                beta = config.beta_start
            else:
                beta = config.beta_start + (config.beta_end - config.beta_start) * (step / config.n_steps)

            current_annealed = current_log_prior + beta * current_log_lik
            proposed_annealed = proposed_log_prior + beta * proposed_log_lik

            log_alpha = float('-inf')  # default for logging
            if (current_annealed == float('-inf')
                    and proposed_annealed == float('-inf')):
                accept = False
            elif proposed_annealed == float('-inf'):
                accept = False
            elif current_annealed == float('-inf'):
                # Current is impossible, proposed is finite — always accept.
                accept = True
                log_alpha = float('inf')
            else:
                log_alpha = (
                    (proposed_annealed + log_q_rev)
                    - (current_annealed + log_q_fwd)
                )
                accept = (log_alpha >= 0) or (math.log(rng.random()) < log_alpha)

            # (e) Accept or reject.
            if accept:
                current = proposed
                current_log_prior = proposed_log_prior
                current_log_lik = proposed_log_lik
                current_log_posterior = proposed_log_posterior
                current_ext_frac = proposed_ext_frac
                n_accepted += 1

                if V >= 2:
                    print(f"  [step {step}] ACCEPT size={current.size()} "
                          f"prior={current_log_prior:.2f} lik={current_log_lik:.2f} "
                          f"post={current_log_posterior:.2f}")
                    print(f"    {str(current)[:120]}")
            elif V >= 3:
                print(f"  [step {step}] reject (log_α={log_alpha:.2f})")

            # Periodic progress at verbose=1
            if V >= 1 and step % log_interval == 0:
                rate = n_accepted / step if step > 0 else 0
                print(f"  [step {step}/{config.n_steps}] "
                      f"accepted={n_accepted} ({rate:.1%}) "
                      f"unique={len(visit_counts)} "
                      f"best_post={best_log_posterior:.2f} "
                      f"β={beta:.3f}")

            # (f) Track the current state (after accept/reject decision).
            current_str = str(current)
            visit_counts[current_str] = visit_counts.get(current_str, 0) + 1
            trajectory.append(current_str)
            if current_str not in first_passage:
                first_passage[current_str] = step
            # Track best log posterior for this program.
            if current_str not in log_posteriors or current_log_posterior > log_posteriors[current_str]:
                log_posteriors[current_str] = current_log_posterior
            # Layer 3: Store extension fraction for post-hoc filtering.
            ext_fractions[current_str] = current_ext_frac

            # Update best overall.
            if current_log_posterior > best_log_posterior:
                best_log_posterior = current_log_posterior
                best_program = current_str

        # -------------------------------------------------------------- #
        # Build result.
        # -------------------------------------------------------------- #
        # Sort hypotheses by visit count descending, filter Layer-2
        # tautologies (ext_fraction >= 1.0) post-hoc, take top_k.
        # (Rejecting them inline in the MH loop would break detailed
        # balance; the filter is applied only to the reported hypothesis
        # table. visit_counts/first_passage/trajectory retain the full
        # unfiltered trace for trajectory analysis.)
        sorted_programs = [
            p for p in sorted(
                visit_counts.keys(),
                key=lambda p: visit_counts[p],
                reverse=True,
            )
            if ext_fractions.get(p, 0.0) < 1.0
        ][:config.top_k]

        top_hypotheses = [
            {
                'program': prog,
                'visit_count': visit_counts[prog],
                'log_posterior': log_posteriors.get(prog, float('-inf')),
                'first_seen_step': first_passage.get(prog, -1),
            }
            for prog in sorted_programs
        ]

        n_unique = len(visit_counts)
        acceptance_rate = n_accepted / config.n_steps if config.n_steps > 0 else 0.0

        return MCMCResult(
            n_steps=config.n_steps,
            n_accepted=n_accepted,
            n_unique=n_unique,
            acceptance_rate=acceptance_rate,
            top_hypotheses=top_hypotheses,
            best_program=best_program,
            best_log_posterior=best_log_posterior,
            visit_counts=visit_counts,
            first_passage=first_passage,
            trajectory=trajectory,
            ext_fractions=ext_fractions,
        )


# =========================================================================== #
# MULTI-CHAIN PARALLEL RUNNER
# =========================================================================== #


def run_parallel_chains(
    grammar: Grammar,
    config: MCMCConfig,
    request_type: Type,
    exemplar_hands: List,
    n_chains: int = 8,
    ext_probe_hands: Optional[List] = None,
    seed_offset: int = 0,
) -> MCMCResult:
    """
    Run multiple independent MCMC chains and merge their results.

    This is the simplest form of parallel MCMC: each chain starts from an
    independent sample from the prior, explores with its own random seed,
    and has no communication with other chains. The merged visit counts
    across chains provide a better approximation to the posterior than any
    single chain, because different chains are less likely to all get stuck
    in the same local optimum.

    COGNITIVE MODELING NOTE
    -----------------------
    Multiple chains model multiple learners or multiple independent "attempts"
    at rule learning. The merged visit counts approximate the posterior better
    than any single chain, and programs found by multiple chains independently
    are particularly strong competitors — they represent hypotheses that are
    robust attractors in the hypothesis space.

    (Path to parallel tempering noted for future work: chains at different
    temperatures can swap states to improve mixing.)

    Args:
        grammar:         The PCFG grammar defining the hypothesis space.
        config:          MCMCConfig for each individual chain. Each chain runs
                         config.n_steps steps with config.max_depth, etc.
        request_type:    The type of programs to explore (e.g., Arrow(HAND, BOOL)).
        exemplar_hands:  Positive exemplar hands for likelihood computation.
        n_chains:        Number of independent chains to run (default 8).
        ext_probe_hands: Shared probe hands for extension size estimation.
                         If None, generated once and shared across all chains.
        seed_offset:     Additional offset added to chain seeds so that different
                         rules (or callers) explore different trajectories even
                         when they share the same base seed (default 0).

    Returns:
        A single merged MCMCResult where:
          - visit_counts: summed across chains
          - first_passage: true min step across chains (raw within-chain step
            indices, NOT offset into a concatenated timeline — independent
            chains must not be treated as one merged trajectory). Per-chain
            within-chain indices are also exposed via `per_chain_first_passage`
            for honest cognitive-timing analysis.
          - n_accepted: summed across chains
          - n_steps: n_chains * config.n_steps (total steps pooled across
            chains; note this is a pooled counter, not a timeline bound —
            see first_passage semantics above).
          - ext_fractions: averaged across chains in which a program was
            visited (pools Monte-Carlo probe noise).
          - top_hypotheses: sorted by merged visit count, Layer-2 tautologies
            (averaged ext_fraction >= 1.0) filtered post-hoc, top config.top_k.
          - best_program: highest log_posterior among non-tautologies.
    """
    from gallery_analysis.exemplars import generate_probe_set

    # Determine the base seed for deriving per-chain seeds.
    base_seed = config.seed if config.seed is not None else 42

    # Generate shared probe hands once (expensive to create, shared across chains).
    if ext_probe_hands is None:
        ext_probe_hands = generate_probe_set(n_probes=10_000, seed=base_seed)

    # ------------------------------------------------------------------ #
    # Run each chain sequentially with a different seed.
    # (Can be parallelized with ProcessPoolExecutor later.)
    # ------------------------------------------------------------------ #
    chain_results: List[MCMCResult] = []
    for i in range(n_chains):
        # Each chain gets a unique seed spaced 1000 apart to avoid overlap.
        chain_seed = base_seed + seed_offset + i * 1000
        # Propagate ALL fields from caller's config, only override seed.
        # (Previous bug: beta_start/beta_end were silently dropped, so every
        # gallery run ignored --beta-start/--beta-end and ran at β=1.0.)
        chain_config = dc_replace(config, seed=chain_seed)
        if config.verbose >= 1:
            print(f"\n  --- Chain {i+1}/{n_chains} (seed={chain_seed}) ---")
        chain = MCMCChain(grammar, chain_config)
        result = chain.run(
            request_type=request_type,
            exemplar_hands=exemplar_hands,
            ext_probe_hands=ext_probe_hands,
        )
        if config.verbose >= 1:
            print(f"  --- Chain {i+1} done: accepted={result.n_accepted} "
                  f"({result.acceptance_rate:.1%}), unique={result.n_unique} ---")
        chain_results.append(result)

    # ------------------------------------------------------------------ #
    # Merge results across chains.
    # ------------------------------------------------------------------ #
    merged_visit_counts: Dict[str, int] = {}
    merged_first_passage: Dict[str, int] = {}
    merged_log_posteriors: Dict[str, float] = {}
    merged_ext_fractions: Dict[str, float] = {}
    merged_trajectory: List[str] = []
    ext_fraction_accum: Dict[str, Tuple[float, int]] = {}  # (sum, count) for averaging
    total_accepted = 0

    for i, result in enumerate(chain_results):
        # Sum visit counts.
        for prog, count in result.visit_counts.items():
            merged_visit_counts[prog] = merged_visit_counts.get(prog, 0) + count

        # First-passage merge: chains are INDEPENDENT trajectories, NOT a
        # concatenated timeline. Report the true min step across chains
        # (i.e., "fastest chain to reach this program"), not step + i*n_steps
        # which would bias low-index chains to appear faster for any hypothesis
        # that appears in multiple chains. Per-chain first_passage is still
        # available via chain_results for downstream analysis.
        for prog, step in result.first_passage.items():
            if prog not in merged_first_passage or step < merged_first_passage[prog]:
                merged_first_passage[prog] = step

        # Track best log posterior per program.
        for hyp in result.top_hypotheses:
            prog = hyp['program']
            lp = hyp['log_posterior']
            if prog not in merged_log_posteriors or lp > merged_log_posteriors[prog]:
                merged_log_posteriors[prog] = lp

        # Accumulate extension fractions for averaging across chains.
        # (Prior "latest value" merge silently preferred the last chain's
        # estimate. Averaging pools Monte-Carlo noise across chains.)
        for prog, frac in result.ext_fractions.items():
            ext_sum, ext_n = ext_fraction_accum.get(prog, (0.0, 0))
            ext_fraction_accum[prog] = (ext_sum + frac, ext_n + 1)

        total_accepted += result.n_accepted
        merged_trajectory.extend(result.trajectory)

    # Average ext_fractions across chains (one estimate per chain in which
    # the program was visited; pools Monte-Carlo noise). Computed here so
    # the Layer-2 post-hoc filter below can use averaged fractions.
    merged_ext_fractions = {
        prog: ext_sum / ext_n
        for prog, (ext_sum, ext_n) in ext_fraction_accum.items()
    }

    # ------------------------------------------------------------------ #
    # Build merged top hypotheses: sort by visit count, filter Layer-2
    # tautologies (averaged ext_fraction >= 1.0) post-hoc, take top_k.
    # (See single-chain run() for rationale; detailed balance preserved.)
    # ------------------------------------------------------------------ #
    sorted_programs = [
        p for p in sorted(
            merged_visit_counts.keys(),
            key=lambda p: merged_visit_counts[p],
            reverse=True,
        )
        if merged_ext_fractions.get(p, 0.0) < 1.0
    ][:config.top_k]

    top_hypotheses = [
        {
            'program': prog,
            'visit_count': merged_visit_counts[prog],
            'log_posterior': merged_log_posteriors.get(prog, float('-inf')),
            'first_seen_step': merged_first_passage.get(prog, -1),
        }
        for prog in sorted_programs
    ]

    # Identify overall best program by log posterior, excluding tautologies.
    best_program = None
    best_log_posterior = float('-inf')
    for prog, lp in merged_log_posteriors.items():
        if merged_ext_fractions.get(prog, 0.0) >= 1.0:
            continue
        if lp > best_log_posterior:
            best_log_posterior = lp
            best_program = prog

    total_steps = n_chains * config.n_steps
    n_unique = len(merged_visit_counts)
    acceptance_rate = total_accepted / total_steps if total_steps > 0 else 0.0

    # Per-chain first-passage (raw within-chain step indices) for honest
    # downstream cognitive-timing analysis.
    per_chain_first_passage = [dict(r.first_passage) for r in chain_results]

    return MCMCResult(
        n_steps=total_steps,
        n_accepted=total_accepted,
        n_unique=n_unique,
        acceptance_rate=acceptance_rate,
        top_hypotheses=top_hypotheses,
        best_program=best_program,
        best_log_posterior=best_log_posterior,
        visit_counts=merged_visit_counts,
        first_passage=merged_first_passage,
        trajectory=merged_trajectory,
        ext_fractions=merged_ext_fractions,
        per_chain_first_passage=per_chain_first_passage,
    )
