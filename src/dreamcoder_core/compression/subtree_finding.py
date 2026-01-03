"""
Common subtree finding and abstraction creation.

This module finds exact subtree matches across programs and creates
abstractions from them.

COMPLEMENTARY TO ANTI-UNIFICATION:
- Exact matching: finds identical code
- Anti-unification: finds structural patterns with differences

EXTRACTED FROM: compression.py lines 704-900
"""

from collections import defaultdict
from typing import Dict, List, Set, Tuple

from ..program import Program, Primitive, Application, Abstraction, Index, Invented
from .data_structures import SubtreeOccurrence


def find_common_subtrees(
    programs: List[Program],
    min_size: int = 2,
    min_count: int = 2
) -> List[SubtreeOccurrence]:
    """
    Find subtrees that appear identically in multiple programs.

    This is COMPLEMENTARY to anti-unification:
    - Exact matching: finds identical code
    - Anti-unification: finds structural patterns with differences

    Args:
        programs: List of programs to analyze
        min_size: Minimum subtree size (AST nodes) to consider
        min_count: Minimum number of programs containing the subtree

    Returns:
        List of SubtreeOccurrence sorted by savings (highest first)

    ALGORITHM:
        1. For each program, enumerate all subtrees
        2. Canonicalize subtrees by string representation
        3. Count occurrences across programs (not within a program)
        4. Compute savings for each common subtree

    SAVINGS FORMULA:
        savings = (size - 1) × (count - 1)

        Intuition:
        - Original cost: size × count (each use costs full size)
        - With abstraction: size (define once) + count (use count times)
        - Net savings: size × count - size - count = (size-1) × (count-1)

    COMPARISON TO ORIGINAL:
        Original DreamCoder uses full MDL scoring which also accounts for:
        - Grammar expansion cost (adding new production)
        - Type complexity of the abstraction
        - Argument passing overhead
    """
    # Count all subtrees
    subtree_counts: Dict[str, int] = defaultdict(int)
    subtree_programs: Dict[str, Set[int]] = defaultdict(set)
    subtree_objects: Dict[str, Program] = {}

    for prog_idx, prog in enumerate(programs):
        seen_in_prog: Set[str] = set()

        for subtree in prog.subprograms():
            # Skip trivial subtrees
            if subtree.size() < min_size:
                continue

            # Skip variables (they're context-dependent)
            # A $0 in one program is not the same as $0 in another
            if isinstance(subtree, Index):
                continue

            # Canonicalize the subtree representation
            key = str(subtree)

            if key not in seen_in_prog:
                subtree_counts[key] += 1
                seen_in_prog.add(key)

            subtree_programs[key].add(prog_idx)
            subtree_objects[key] = subtree

    # Filter by minimum count and sort by potential savings
    common = []
    for key, count in subtree_counts.items():
        if count >= min_count:
            subtree = subtree_objects[key]
            size = subtree.size()

            # Savings = (size - 1) * (count - 1)
            # We save (size-1) for each use beyond the first
            # The -1 accounts for the cost of the abstraction itself
            savings = (size - 1) * (count - 1)

            if savings > 0:
                common.append(SubtreeOccurrence(
                    subtree=subtree,
                    count=count,
                    programs=[str(i) for i in subtree_programs[key]],
                    savings=savings
                ))

    # Sort by savings (highest first)
    common.sort(key=lambda x: -x.savings)
    return common


def abstract_subtree(
    subtree: Program,
    free_vars: Set[int] = None
) -> Tuple[Invented, int]:
    """
    Convert a subtree into an invented abstraction.

    If the subtree has free variables (references to outer lambdas),
    we wrap it in lambdas to abstract over them.

    Args:
        subtree: The subtree to abstract
        free_vars: Free variable indices (computed if not provided)

    Returns:
        (invented_program, n_args) where n_args is how many arguments it takes

    EXAMPLE:
        subtree = (+ $0 $2)  where $0 and $2 are free

        Step 1: Identify free vars: {0, 2}
        Step 2: Create index map: {0 → 0, 2 → 1} (renumber to consecutive)
        Step 3: Rewrite: (+ $0 $1)
        Step 4: Wrap in lambdas: λλ(+ $0 $1)

        Result: Invented(λλ(+ $0 $1)), n_args=2

    COMPARISON TO ORIGINAL:
        Original DreamCoder considers MULTIPLE ways to factor:
        - Which variables to abstract over
        - What order for arguments
        - Whether to also abstract over subexpressions

        We take a single canonical approach: abstract over all free vars.
    """
    if free_vars is None:
        free_vars = subtree.free_indices()

    # Sort free variables
    free_list = sorted(free_vars)
    n_args = len(free_list)

    if n_args == 0:
        # No free variables - just wrap as invented
        return Invented(subtree), 0

    # Need to wrap in lambdas for free variables
    # Map old indices to new ones (consecutive from 0)
    index_map = {old: new for new, old in enumerate(free_list)}

    # Rewrite the subtree with new indices
    rewritten = _reindex(subtree, index_map)

    # Wrap in n_args lambdas
    body = rewritten
    for _ in range(n_args):
        body = Abstraction(body)

    return Invented(body), n_args


def _reindex(program: Program, index_map: Dict[int, int]) -> Program:
    """
    Rewrite a program with new de Bruijn indices.

    Args:
        program: Program to rewrite
        index_map: Maps old index → new index

    IMPORTANT: When we enter an Abstraction, we must shift the map
    because de Bruijn indices are relative to binding depth.
    """
    if isinstance(program, Index):
        if program.i in index_map:
            return Index(index_map[program.i])
        return program

    if isinstance(program, (Primitive, Invented)):
        return program

    if isinstance(program, Application):
        return Application(
            _reindex(program.f, index_map),
            _reindex(program.x, index_map)
        )

    if isinstance(program, Abstraction):
        # Shift the index map for the new binding
        # Inside a lambda, all external refs increase by 1
        shifted_map = {k + 1: v + 1 for k, v in index_map.items()}
        return Abstraction(_reindex(program.body, shifted_map))

    return program
