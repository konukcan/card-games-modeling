"""
Compute grammar-based log-priors for arbitrary DSL program strings.

This module computes log P(program | grammar) for any valid DSL string,
matching exactly the cost accounting used by the TopDownEnumerator.

WHY THIS EXISTS:
    The enumerator already produces log-priors for programs it generates,
    but we also need priors for *externally injected* hypotheses (e.g.,
    hand-written DSL translations of Gemini Flash rules). This function
    lets us score any valid DSL string without re-enumerating.

HOW IT WORKS:
    The key insight is that the enumerator and Grammar.program_log_likelihood()
    use different normalization strategies for multi-argument primitives.
    The enumerator normalizes each primitive at its *final return type*
    (e.g., has_color is normalized among all BOOL-returning primitives),
    while program_log_likelihood normalizes at the full curried type
    (e.g., HAND->color->BOOL). This produces different results.

    To match the enumerator exactly, we walk the program tree the same
    way the enumerator generates it:
    1. Abstraction (lambda): free — just recurse on the body with the
       argument type added to the environment.
    2. Primitive/Application chain: extract the head primitive and its
       arguments. Look up the head among candidates for the *target type*
       (the final return type after all args are applied). Sum the head's
       normalized log-prob with the costs of each argument.
    3. Index (variable): use grammar.variable_candidates() to get the
       variable's log-probability at the target type.
"""

import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Tuple

from dreamcoder_core.program import (
    parse_program, Program, Primitive, Application, Abstraction, Index,
)
from dreamcoder_core.grammar import Grammar
from dreamcoder_core.type_system import Arrow, Type, TypeContext, HAND, BOOL


def compute_log_prior(program_str: str, grammar: Grammar) -> float:
    """
    Compute the grammar-based log-prior for a DSL program string.

    The result matches the log_prob produced by TopDownEnumerator.enumerate()
    for the same program string. This is the cost-based accounting where:
    - Each primitive's cost is its type-indexed normalized -log probability
    - Each variable's cost is based on grammar.log_variable
    - Lambda wrapping is free (structural necessity for Arrow types)

    Args:
        program_str: A DSL program string. Must be a valid program that
                     type-checks as Hand -> Bool.
        grammar:     A Grammar object (typically from build_gallery_grammar()).

    Returns:
        The log-prior (float, always <= 0). More negative means less probable
        under the grammar.

    Raises:
        ValueError: If the program string cannot be parsed or doesn't
                    type-check.

    Example:
        >>> from gallery_analysis.enumerator import build_gallery_grammar
        >>> g = build_gallery_grammar()
        >>> lp = compute_log_prior("(λ has_color $0 RED)", g)
        >>> lp < 0
        True
    """
    # Build primitive lookup dict from the grammar's productions.
    # parse_program needs a dict mapping name -> Primitive object.
    prim_dict = {}
    for prod in grammar.productions:
        if isinstance(prod.program, Primitive):
            prim_dict[prod.program.name] = prod.program

    # Parse the string into a Program AST
    try:
        program = parse_program(program_str, prim_dict)
    except (ValueError, KeyError) as e:
        raise ValueError(f"Cannot parse program '{program_str}': {e}") from e

    # Compute log-prior matching the enumerator's cost accounting
    request_type = Arrow(HAND, BOOL)
    return _compute_cost(program, request_type, grammar, env=[])


def _compute_cost(
    program: Program,
    target_type: Type,
    grammar: Grammar,
    env: List[Type],
) -> float:
    """
    Recursively compute the log-prior of a program, matching the enumerator.

    This mirrors the enumerator's decomposition:
    - Arrow type -> peel lambda, recurse on body (free)
    - Base type -> decompose into head + args, sum costs
    - Variable -> look up among variable candidates

    Args:
        program:     The program AST node to score
        target_type: The type this node must produce
        grammar:     The grammar for looking up production probabilities
        env:         Type environment (types of bound variables, $0 first)

    Returns:
        Log-probability (negative float). Returns -inf if the program
        doesn't type-check at the target type.
    """
    # Case 1: Abstraction — target must be Arrow, lambda is free
    if isinstance(program, Abstraction):
        if not isinstance(target_type, Arrow):
            return float('-inf')
        # Extend env: the lambda's parameter becomes $0
        new_env = [target_type.arg] + env
        return _compute_cost(program.body, target_type.ret, grammar, new_env)

    # Case 2: Index (variable reference)
    if isinstance(program, Index):
        ctx = TypeContext()
        var_candidates = grammar.variable_candidates(target_type, ctx, env)
        for idx, log_prob in var_candidates:
            if idx == program.i:
                return log_prob
        return float('-inf')

    # Case 3: Primitive or Application chain
    # Extract the head primitive and all arguments from the application spine.
    # E.g., (has_color $0 RED) -> head=has_color, args=[$0, RED]
    head, args = _extract_head_and_args(program)

    if not isinstance(head, Primitive):
        return float('-inf')

    # Look up the head among candidates for the *target type*.
    # This matches the enumerator: has_color is found among BOOL candidates
    # even though its full type is HAND->color->BOOL.
    ctx = TypeContext()
    candidates = grammar.candidates_for_type(target_type, ctx, env, normalize=True)

    # Find the head primitive's normalized log-probability
    head_log_prob = None
    head_inst_type = None
    for prod, inst_type, log_prob in candidates:
        if prod.program.name == head.name:
            head_log_prob = log_prob
            head_inst_type = inst_type
            break

    if head_log_prob is None:
        # Head primitive can't produce this type
        return float('-inf')

    # Now compute costs for each argument.
    # The instantiated type tells us what types the arguments should be.
    # E.g., has_color: HAND -> color -> BOOL, so args should be [HAND, color]
    total_log_prob = head_log_prob
    remaining_type = head_inst_type

    for arg in args:
        if not isinstance(remaining_type, Arrow):
            # More args than the type expects
            return float('-inf')
        arg_target_type = remaining_type.arg
        remaining_type = remaining_type.ret

        arg_log_prob = _compute_cost(arg, arg_target_type, grammar, env)
        if arg_log_prob == float('-inf'):
            return float('-inf')
        total_log_prob += arg_log_prob

    return total_log_prob


def _extract_head_and_args(program: Program) -> Tuple[Program, List[Program]]:
    """
    Decompose an application spine into head and arguments.

    Application is left-associated: ((f x) y) z
    So we walk down the .f chain to find the head, collecting args along the way.

    Examples:
        has_color         -> (has_color, [])
        (has_color $0)    -> (has_color, [$0])
        (has_color $0 RED)-> (has_color, [$0, RED])

    Args:
        program: A Program node (Primitive or Application)

    Returns:
        (head_program, [arg1, arg2, ...]) where head is the leftmost leaf
    """
    args = []
    current = program
    while isinstance(current, Application):
        args.append(current.x)  # x is the argument
        current = current.f     # f is the function
    # args were collected in reverse order (outermost first)
    args.reverse()
    return current, args
