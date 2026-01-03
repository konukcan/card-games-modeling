"""
Program rewriting with invented abstractions.

After compression discovers an abstraction, this module rewrites all programs
to use it, enabling hierarchical abstraction discovery in subsequent rounds.

KEY INSIGHT:
    Rewriting is a SYNTACTIC operation. We look for exact pattern matches
    and replace them with calls to the invention. The target pattern is
    not shifted when entering lambdas - we're matching syntax, not semantics.

EXTRACTED FROM: compression.py lines 1220-1619
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..program import Program, Primitive, Application, Abstraction, Index, Invented


@dataclass
class RewriteResult:
    """
    Result of rewriting a program with an invention.

    FIELDS:
    -------
    program: Program
        The rewritten program.

    n_replacements: int
        Number of times the target was replaced.

    original_size: int
        Size of the original program.

    new_size: int
        Size of the rewritten program.
    """
    program: Program
    n_replacements: int
    original_size: int
    new_size: int


def rewrite_with_invention(
    program: Program,
    target: Program,
    invention: Invented,
    n_args: int
) -> Program:
    """
    Rewrite a program by replacing target subtrees with the invention.

    This is a key operation for compression: after learning an abstraction,
    we rewrite all programs to use it, enabling hierarchical abstraction
    discovery in subsequent rounds.

    Args:
        program: Program to rewrite
        target: Subtree to replace (exact syntactic pattern)
        invention: The invented abstraction to substitute
        n_args: Number of arguments the invention takes

    Returns:
        Rewritten program using the invention

    ALGORITHM:
        1. Recursively traverse the program
        2. At each node, check if it matches the target SYNTACTICALLY
        3. If match: replace with (invention arg1 arg2 ...) where args
           are the free variables of the target
        4. If no match: recurse into children

    IMPORTANT - No target shifting:
        When descending into lambdas, we do NOT shift the target.
        The target is a SYNTACTIC pattern. If we're looking for (+ $0 1),
        we want to find exactly that string, regardless of lambda depth.

        The free variables in the target tell us what arguments to pass
        to the invention. These are determined by target.free_indices()
        at depth 0, and they correspond to whatever those indices mean
        in the current program context.

    EXAMPLE:
        Program: λh. (+ (size h) (size h))
        Target: (size $0) where invention = λx.(size x) = #size_of

        In the program body, we find (size $0) twice.
        Each is replaced with (#size_of $0) where $0 = h.

        Result: λh. (+ (#size_of h) (#size_of h)) = λ(+ (#size_of $0) (#size_of $0))
    """
    return _rewrite_helper(program, target, invention, n_args)


def _rewrite_helper(
    program: Program,
    target: Program,
    invention: Invented,
    n_args: int
) -> Program:
    """
    Internal helper for rewrite_with_invention.

    Separated to allow the main function to have clean documentation
    while this handles the recursion.
    """
    # Check for exact syntactic match
    # Using == which is structural equality for frozen dataclasses
    if program == target:
        # Replace with invention applied to its arguments
        if n_args == 0:
            # Invention takes no arguments, just return it
            return invention
        else:
            # Apply invention to the free variables of the target
            # These free variables, when evaluated, will provide the
            # values that differ between uses of this pattern
            free_vars = sorted(target.free_indices())
            result: Program = invention
            for var in free_vars:
                result = Application(result, Index(var))
            return result

    # No match - recurse into structure

    if isinstance(program, (Primitive, Index)):
        # Atoms cannot contain the target
        return program

    if isinstance(program, Invented):
        # Don't rewrite inside invented abstractions
        # They're treated as atomic for rewriting purposes
        return program

    if isinstance(program, Application):
        # Rewrite both function and argument
        new_f = _rewrite_helper(program.f, target, invention, n_args)
        new_x = _rewrite_helper(program.x, target, invention, n_args)

        # Only create new Application if something changed
        if new_f is program.f and new_x is program.x:
            return program
        return Application(new_f, new_x)

    if isinstance(program, Abstraction):
        # Descend into lambda body
        # IMPORTANT: Do NOT shift the target!
        # We're doing syntactic pattern matching, not substitution.
        # The pattern (+ $0 1) should match (+ $0 1) at any depth.
        new_body = _rewrite_helper(program.body, target, invention, n_args)

        # Only create new Abstraction if body changed
        if new_body is program.body:
            return program
        return Abstraction(new_body)

    # Unknown node type (e.g., Hole) - return unchanged
    return program


def rewrite_with_invention_detailed(
    program: Program,
    target: Program,
    invention: Invented,
    n_args: int
) -> RewriteResult:
    """
    Rewrite a program and return detailed statistics.

    Same as rewrite_with_invention but tracks replacement count and sizes.
    Useful for debugging and verification.
    """
    original_size = program.size()
    n_replacements = [0]  # Use list for mutation in nested function

    def rewrite_counting(prog: Program) -> Program:
        if prog == target:
            n_replacements[0] += 1
            if n_args == 0:
                return invention
            else:
                free_vars = sorted(target.free_indices())
                result: Program = invention
                for var in free_vars:
                    result = Application(result, Index(var))
                return result

        if isinstance(prog, (Primitive, Index, Invented)):
            return prog

        if isinstance(prog, Application):
            new_f = rewrite_counting(prog.f)
            new_x = rewrite_counting(prog.x)
            if new_f is prog.f and new_x is prog.x:
                return prog
            return Application(new_f, new_x)

        if isinstance(prog, Abstraction):
            new_body = rewrite_counting(prog.body)
            if new_body is prog.body:
                return prog
            return Abstraction(new_body)

        return prog

    new_program = rewrite_counting(program)
    new_size = new_program.size()

    return RewriteResult(
        program=new_program,
        n_replacements=n_replacements[0],
        original_size=original_size,
        new_size=new_size
    )


def verify_rewrite_semantics(
    original: Program,
    rewritten: Program,
    test_inputs: List[List[Any]],
    verbose: bool = False
) -> Tuple[bool, Optional[str]]:
    """
    Verify that a rewritten program has the same behavior as the original.

    This is a CRITICAL safety check to ensure that program rewriting
    preserves semantics. If a rewrite changes behavior, we have a bug.

    Args:
        original: The original program before rewriting
        rewritten: The program after rewriting with invention(s)
        test_inputs: List of environments (input lists) to test on.
                     Each environment is a list of values for $0, $1, ...
        verbose: If True, print details of each test

    Returns:
        (success, error_message) where:
        - success: True if all tests pass
        - error_message: Description of first failure, or None if success

    USAGE:
        # For a program λh. (some_rule h):
        test_inputs = [[hand1], [hand2], [hand3]]  # Different hands
        success, err = verify_rewrite_semantics(original, rewritten, test_inputs)
        if not success:
            raise RuntimeError(f"Rewrite changed semantics: {err}")

    NOTE:
        This uses evaluation which may raise exceptions. Programs that
        raise exceptions in the original should also raise in the rewritten
        version (we verify this).
    """
    for env in test_inputs:
        try:
            orig_result = original.evaluate(env)
            orig_exception = None
        except Exception as e:
            orig_result = None
            orig_exception = type(e).__name__

        try:
            new_result = rewritten.evaluate(env)
            new_exception = None
        except Exception as e:
            new_result = None
            new_exception = type(e).__name__

        # Both should have same exception behavior
        if orig_exception != new_exception:
            return False, (
                f"Exception mismatch on input {env}: "
                f"original raised {orig_exception}, rewritten raised {new_exception}"
            )

        # If no exceptions, results should match
        if orig_exception is None:
            if orig_result != new_result:
                return False, (
                    f"Result mismatch on input {env}: "
                    f"original={orig_result}, rewritten={new_result}"
                )

        if verbose:
            status = "PASS" if orig_exception is None else f"PASS (both raised {orig_exception})"
            print(f"  Input {env}: {status}")

    return True, None


def rewrite_and_verify(
    program: Program,
    target: Program,
    invention: Invented,
    n_args: int,
    test_inputs: List[List[Any]]
) -> Tuple[Program, bool, Optional[str]]:
    """
    Rewrite a program and verify the result preserves semantics.

    Combines rewrite_with_invention and verify_rewrite_semantics.

    Args:
        program: Program to rewrite
        target: Subtree to replace
        invention: The invented abstraction
        n_args: Number of arguments the invention takes
        test_inputs: Environments to test on

    Returns:
        (rewritten_program, success, error_message)

    USAGE:
        rewritten, ok, err = rewrite_and_verify(prog, target, inv, 2, test_envs)
        if not ok:
            print(f"Warning: rewrite verification failed: {err}")
            # Could choose to skip this rewrite or proceed with caution
    """
    rewritten = rewrite_with_invention(program, target, invention, n_args)

    # Only verify if something actually changed
    if rewritten is program:
        return rewritten, True, None

    success, error = verify_rewrite_semantics(program, rewritten, test_inputs)
    return rewritten, success, error


def rewrite_frontier(
    frontier: List[Tuple[Program, float]],
    target: Program,
    invention: Invented,
    n_args: int
) -> Tuple[List[Tuple[Program, float]], Dict[str, int]]:
    """
    Rewrite all programs in a frontier with an invention.

    Args:
        frontier: List of (program, log_likelihood) pairs
        target: Subtree to replace
        invention: The invented abstraction
        n_args: Number of arguments the invention takes

    Returns:
        (rewritten_frontier, stats) where stats contains:
        - programs_changed: Number of programs that were modified
        - total_replacements: Total substitutions made
        - total_size_reduction: Sum of size reductions
    """
    rewritten_frontier = []
    stats = {
        'programs_changed': 0,
        'total_replacements': 0,
        'total_size_reduction': 0
    }

    for prog, ll in frontier:
        result = rewrite_with_invention_detailed(prog, target, invention, n_args)

        # Use rewritten program with same log-likelihood
        # NOTE: The true log-likelihood changes with the new grammar,
        # but we preserve the original value here for continuity
        rewritten_frontier.append((result.program, ll))

        if result.n_replacements > 0:
            stats['programs_changed'] += 1
            stats['total_replacements'] += result.n_replacements
            stats['total_size_reduction'] += result.original_size - result.new_size

    return rewritten_frontier, stats


def rewrite_all_frontiers(
    frontiers: List[List[Tuple[Program, float]]],
    target: Program,
    invention: Invented,
    n_args: int
) -> Tuple[List[List[Tuple[Program, float]]], Dict[str, int]]:
    """
    Rewrite all programs across all frontiers with an invention.

    Args:
        frontiers: List of frontiers, each is [(program, log_likelihood), ...]
        target: Subtree to replace
        invention: The invented abstraction
        n_args: Number of arguments the invention takes

    Returns:
        (rewritten_frontiers, aggregate_stats)
    """
    rewritten_frontiers = []
    aggregate_stats = {
        'programs_changed': 0,
        'total_replacements': 0,
        'total_size_reduction': 0,
        'frontiers_affected': 0
    }

    for frontier in frontiers:
        rewritten, stats = rewrite_frontier(frontier, target, invention, n_args)
        rewritten_frontiers.append(rewritten)

        aggregate_stats['programs_changed'] += stats['programs_changed']
        aggregate_stats['total_replacements'] += stats['total_replacements']
        aggregate_stats['total_size_reduction'] += stats['total_size_reduction']

        if stats['programs_changed'] > 0:
            aggregate_stats['frontiers_affected'] += 1

    return rewritten_frontiers, aggregate_stats
