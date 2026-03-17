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

This module provides the foundational building block: sampling a complete,
type-correct program from the grammar prior. This is used both for chain
initialization and for generating replacement subtrees in proposals.

ARCHITECTURE
------------
  sample_program()        <- THIS FILE: sample from grammar prior
  propose_subtree()       <- future: select subtree + regenerate
  mcmc_step()             <- future: MH accept/reject
  run_chain()             <- future: full MCMC chain
"""
import sys
import math
import random
from pathlib import Path
from typing import List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.type_system import (
    Type, Arrow, ListType, TypeContext, TypeVariable,
    BOOL, INT, CARD, SUIT, RANK,
)
from dreamcoder_core.grammar import Grammar
from dreamcoder_core.program import (
    Program, Index, Application, Abstraction,
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
