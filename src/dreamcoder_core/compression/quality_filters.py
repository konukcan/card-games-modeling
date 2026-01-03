"""
Abstraction quality filtering for compression.

These functions filter out degenerate abstractions that provide little value.
Ported from DreamCoder (OCaml) and Stitch (Rust) implementations.

PROBLEM:
    Without filtering, compression can learn useless patterns like:
    - (λ x. f x)      -- Eta-expanded wrapper, equivalent to just f
    - (λ x. x)        -- Identity function
    - (λ x. const)    -- Constant wrapper

    These provide zero or negative compression benefit but clutter the library.

SOLUTIONS IMPLEMENTED:
    1. nontrivial check (from DreamCoder OCaml)
       - Requires ≥2 primitives OR ≥1 primitive with duplicated variable uses

    2. eta-reducible check (from Stitch Rust)
       - Rejects (λ x. f x) where x not free in f

    3. single-task check (from Stitch Rust)
       - Rejects abstractions only useful in one task

REFERENCES:
    - DreamCoder: solvers/compression.ml, function 'nontrivial'
    - Stitch: src/compression.rs, src/pattern_args.rs

EXTRACTED FROM: compression.py lines 200-475
"""

import logging
from typing import Dict, List, Optional

from ..program import Program, Primitive, Application, Abstraction, Index, Invented

log_filter = logging.getLogger(__name__)


def is_nontrivial(program: Program) -> bool:
    """
    Check if an abstraction is nontrivial (worth keeping).

    Port of DreamCoder's OCaml 'nontrivial' function.

    An abstraction is trivial if it contains:
    - Only 1 primitive AND no duplicated variable references

    This catches patterns like:
    - (λ x. f x)    -> 1 primitive (f), no dup indices -> TRIVIAL
    - (λ x. x)      -> 0 primitives -> TRIVIAL
    - (λ x. const)  -> 1 primitive (const), no dup indices -> TRIVIAL

    But allows:
    - (λ x. f x x)  -> 1 primitive (f), duplicated $0 -> NONTRIVIAL
    - (λ x. f (g x)) -> 2 primitives (f, g) -> NONTRIVIAL

    Args:
        program: The abstraction body (unwrapped from Invented)

    Returns:
        True if nontrivial (should keep), False if trivial (should reject)
    """
    primitives = 0
    indices_seen = set()
    duplicated_indices = 0

    def visit(expr: Program, depth: int) -> None:
        nonlocal primitives, duplicated_indices

        if isinstance(expr, Index):
            # Adjust index for current depth to get "logical" index
            adjusted = expr.i - depth
            if adjusted >= 0:  # Only count free variables (unbound by internal lambdas)
                if adjusted in indices_seen:
                    duplicated_indices += 1
                else:
                    indices_seen.add(adjusted)

        elif isinstance(expr, Application):
            visit(expr.f, depth)
            visit(expr.x, depth)

        elif isinstance(expr, Abstraction):
            visit(expr.body, depth + 1)

        elif isinstance(expr, (Primitive, Invented)):
            primitives += 1

    # Unwrap outer lambdas to get the body
    body = program
    outer_lambdas = 0
    while isinstance(body, Abstraction):
        body = body.body
        outer_lambdas += 1

    # Visit the body
    visit(body, 0)

    # Accept if: primitives > 1 OR (primitives == 1 AND has duplicated indices)
    is_valid = primitives > 1 or (primitives == 1 and duplicated_indices > 0)

    if not is_valid:
        log_filter.debug(f"Rejected trivial abstraction: {program} "
                        f"(primitives={primitives}, dup_indices={duplicated_indices})")

    return is_valid


def is_eta_reducible(program: Program) -> bool:
    """
    Check if an abstraction is eta-reducible (equivalent to a simpler form).

    Eta reduction: (λ x. f x) → f  (when x not free in f)

    An abstraction (λ $body) is eta-reducible if:
    - body is an Application (f x)
    - x is Index(0) (the bound variable)
    - Index 0 does not appear free in f

    Args:
        program: The abstraction to check

    Returns:
        True if eta-reducible (should reject), False if not
    """
    # Must be an abstraction
    if not isinstance(program, Abstraction):
        return False

    body = program.body

    # Body must be an application
    if not isinstance(body, Application):
        return False

    # Argument must be Index(0) - the bound variable
    if not isinstance(body.x, Index) or body.x.i != 0:
        return False

    # Check that Index(0) is not free in the function part
    # If it appears in f, we can't eta-reduce
    def has_index_zero_free(expr: Program, depth: int) -> bool:
        """Check if Index(0) appears free at the given depth."""
        if isinstance(expr, Index):
            # At depth d, Index(d) refers to the original $0
            return expr.i == depth
        elif isinstance(expr, Application):
            return has_index_zero_free(expr.f, depth) or has_index_zero_free(expr.x, depth)
        elif isinstance(expr, Abstraction):
            return has_index_zero_free(expr.body, depth + 1)
        else:
            return False

    # We're inside one lambda, so check at depth 0
    # If $0 appears free in f, we cannot eta-reduce
    if has_index_zero_free(body.f, 0):
        return False

    # This is eta-reducible: (λ. f $0) where $0 not in f
    log_filter.debug(f"Rejected eta-reducible abstraction: {program}")
    return True


def is_nested_eta_reducible(program: Program) -> bool:
    """
    Check for nested eta-expansion patterns.

    Catches chains like:
    - (λ x. (λ y. f y) x)  -- double eta expansion
    - (λ x. #wrapper x) where #wrapper is eta-reducible

    Args:
        program: The abstraction to check

    Returns:
        True if contains nested eta patterns (should reject), False if not
    """
    if not isinstance(program, Abstraction):
        return False

    body = program.body

    # Check for (λ. (λ. ... $0) $0) pattern
    if isinstance(body, Application):
        # Is the function itself an abstraction applied to $0?
        if isinstance(body.f, Abstraction) and isinstance(body.x, Index) and body.x.i == 0:
            # This is (λ. (λ. inner) $0) - check if inner ends with $0
            inner = body.f.body
            if isinstance(inner, Application) and isinstance(inner.x, Index) and inner.x.i == 0:
                log_filter.debug(f"Rejected nested eta-reducible: {program}")
                return True

    return False


def is_single_task_abstraction(
    program: Program,
    match_locations: List[int],
    program_to_task: Dict[int, str]
) -> bool:
    """
    Check if an abstraction only appears in programs from a single task.

    From Stitch: abstractions should generalize across tasks, not be
    task-specific workarounds.

    Args:
        program: The abstraction (not used, but included for API consistency)
        match_locations: List of program indices where pattern was found
        program_to_task: Maps program index → task name

    Returns:
        True if single-task (should reject), False if multi-task (should keep)
    """
    if not program_to_task:
        # No task info available, can't filter
        return False

    tasks_used = set()
    for loc in match_locations:
        if loc in program_to_task:
            tasks_used.add(program_to_task[loc])

    is_single = len(tasks_used) <= 1

    if is_single and tasks_used:
        log_filter.debug(f"Rejected single-task abstraction: {program} "
                        f"(only in task: {tasks_used.pop()})")

    return is_single


def passes_abstraction_quality_checks(
    invention: "Invented",
    match_locations: Optional[List[int]] = None,
    program_to_task: Optional[Dict[int, str]] = None,
    check_nontrivial: bool = True,
    check_eta: bool = True,
    check_single_task: bool = False  # Off by default, requires task info
) -> bool:
    """
    Run all abstraction quality checks.

    This is the main entry point for filtering. Call this after creating
    an Invented abstraction but before adding to grammar.

    Args:
        invention: The Invented abstraction to check
        match_locations: Where pattern was found (for single-task check)
        program_to_task: Maps program index → task name
        check_nontrivial: Whether to run nontrivial check
        check_eta: Whether to run eta-reducible checks
        check_single_task: Whether to run single-task check

    Returns:
        True if passes all checks (should keep), False if fails any (should reject)
    """
    body = invention.body if isinstance(invention, Invented) else invention

    # Check 1: Nontrivial
    if check_nontrivial and not is_nontrivial(body):
        return False

    # Check 2: Eta-reducible (single layer)
    if check_eta and is_eta_reducible(body):
        return False

    # Check 3: Nested eta patterns
    if check_eta and is_nested_eta_reducible(body):
        return False

    # Check 4: Single-task (only if task info provided)
    if check_single_task and match_locations and program_to_task:
        if is_single_task_abstraction(body, match_locations, program_to_task):
            return False

    return True
