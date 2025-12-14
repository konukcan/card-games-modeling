"""
=============================================================================
Library Learning (Compression) for DreamCoder
=============================================================================

This module implements the "sleep" phase of DreamCoder:
- Given a set of programs that solved tasks (frontiers)
- Find common subprograms that appear repeatedly
- Abstract them into new library functions (Invented)
- Add them to the grammar for future use

The key insight: if the same subprogram appears in multiple solutions,
it's worth factoring it out as a reusable abstraction.

Compression reduces total description length:
- Using an abstraction costs 1 unit
- But the abstraction itself is amortized across uses

ARCHITECTURE COMPARISON:
------------------------
ORIGINAL DREAMCODER:
    - Uses "fragment grammar" algorithm with beam search
    - Full MDL scoring: DL(grammar) + Σ DL(programs | grammar)
    - Rewrites ALL programs with new abstractions after each invention
    - Arity-aware search considers different ways to factor arguments
    - Grammar size explicitly penalized

OUR IMPLEMENTATION:
    - Two complementary approaches: exact subtree matching + anti-unification
    - Simplified savings heuristic: (size - 1) × (count - 1)
    - Greedy selection of abstractions (not beam search)
    - Programs REWRITTEN after each invention (refactor_programs=True, default)
    - Hierarchical abstraction discovery via multi-round iterative compression
    - Semantic verification for rewrite correctness
    - No explicit grammar size penalty

KEY GAPS REMAINING:
    1. Full MDL scoring instead of heuristic
    2. Arity-aware search (multiple factorizations)
    3. Beam search over candidates
    4. Corpus-guided compression using recognition model

See the implementation plan in the codebase documentation for details.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union
from collections import defaultdict, Counter
import math
import heapq

from .type_system import (
    Type, BaseType, Arrow, ListType, TypeVariable,
    TypeContext, UnificationError,
    BOOL, INT, arrow
)
from .program import (
    Program, Primitive, Application, Abstraction, Index, Invented,
    apply_args, multi_lambda
)
from .grammar import Grammar, Production


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class SubtreeOccurrence:
    """
    Tracks occurrences of a subtree across programs.

    Used for compression analysis to identify candidates for abstraction.

    FIELDS:
    -------
    subtree: Program
        The common subtree pattern found across programs.

    count: int
        Number of programs containing this subtree.
        (Not total occurrences - we count each program once)

    programs: List[str]
        Identifiers of programs containing this subtree.
        Useful for debugging and analysis.

    savings: float
        Estimated description length savings if we abstract this subtree.
        Formula: (size - 1) × (count - 1)

        COMPARISON TO ORIGINAL:
        Original DreamCoder computes full MDL change:
            ΔDL = DL(new_grammar) + Σ DL(rewritten_programs)
                - DL(old_grammar) - Σ DL(original_programs)
        Our heuristic is faster but less accurate.
    """
    subtree: Program
    count: int
    programs: List[str]
    savings: float

    def __str__(self) -> str:
        return f"{self.subtree} (count={self.count}, savings={self.savings:.2f})"


@dataclass
class CompressionResult:
    """
    Result of compression analysis.

    FIELDS:
    -------
    new_inventions: List[Invented]
        Newly discovered abstractions to add to grammar.

    old_grammar: Grammar
        Grammar before compression.

    new_grammar: Grammar
        Grammar after adding inventions (with normalized probabilities).

    total_savings: float
        Sum of savings from all inventions.
        NOTE: This is the heuristic savings, not true MDL change.

    subtree_analysis: List[SubtreeOccurrence]
        All common subtrees found (for debugging/analysis).

    rewritten_frontiers: Optional[List[List[Tuple[Program, float]]]]
        If program refactoring was enabled, contains the frontiers with
        all programs rewritten to use the new inventions.
        None if refactoring was not performed.

    rewrite_stats: Optional[Dict[str, Any]]
        Statistics about the rewriting process:
        - total_replacements: Total number of pattern replacements
        - programs_changed: Number of programs that were modified
        - size_reduction: Total reduction in AST size
        None if refactoring was not performed.
    """
    new_inventions: List[Invented]
    old_grammar: Grammar
    new_grammar: Grammar
    total_savings: float
    subtree_analysis: List[SubtreeOccurrence]
    rewritten_frontiers: Optional[List[List[Tuple[Program, float]]]] = None
    rewrite_stats: Optional[Dict[str, Any]] = None


# ============================================================================
# ANTI-UNIFICATION
# ============================================================================
"""
WHAT IS ANTI-UNIFICATION?

Anti-unification finds the "least general generalization" (LGG) of two terms.
It's the OPPOSITE of unification:
- Unification: finds most general substitution to make terms EQUAL
- Anti-unification: finds most specific pattern that COVERS both terms

EXAMPLE:
    prog1: (eq (get_color (first h)) (get_color (last h)))
    prog2: (eq (get_suit (first h)) (get_suit (last h)))

    Anti-unified pattern: (eq ($0 (first h)) ($0 (last h)))
    where $0 represents the difference (get_color vs get_suit)

WHY IT MATTERS:
    Anti-unification discovers STRUCTURAL patterns that differ in specific positions.
    This complements exact subtree matching, which only finds identical code.

COMPARISON TO ORIGINAL DREAMCODER:
    Original uses "version spaces" for more sophisticated anti-unification
    that considers type constraints during pattern discovery.
"""


def anti_unify(
    prog1: Program,
    prog2: Program,
    substitution_map: Optional[Dict[str, int]] = None,
    binding_depth: int = 0
) -> Tuple[Optional[Program], List[Tuple[Program, Program]], Dict[str, int]]:
    """
    Find the most specific generalization of two programs.

    Uses consistent variable indices when the same substitution appears
    multiple times. For example, if (get_color, get_suit) appears twice,
    both get the same Index variable.

    Args:
        prog1, prog2: Programs to anti-unify
        substitution_map: Tracks consistent variable assignment for repeated diffs
        binding_depth: How many lambdas we're inside (for correct de Bruijn indices)

    Returns:
        (pattern, substitutions, substitution_map) where:
        - pattern: Generalized program with Index nodes for differences
        - substitutions: List of unique (prog1_subterm, prog2_subterm) pairs
        - substitution_map: Maps diff pairs to their Index values

    BINDING DEPTH EXPLANATION:
        De Bruijn indices are relative to lambda depth.
        If we're inside (λ (λ ...)), bound variables use $0, $1.
        Anti-unified variables must use HIGHER indices to avoid collision.

        Example:
            prog1 = λh. eq (get_color (first h)) (get_color (last h))
            prog2 = λh. eq (get_suit (first h)) (get_suit (last h))

            Inside the lambda, $0 refers to h.
            Anti-unified variable gets $1 (binding_depth=1).

            Result pattern: λh. eq ($1 (first $0)) ($1 (last $0))

            Wrapped as abstraction: λf. λh. eq (f (first h)) (f (last h))
    """
    if substitution_map is None:
        substitution_map = {}

    # Base case: if programs are identical, return as-is
    if str(prog1) == str(prog2):
        return prog1, [], substitution_map

    # Create a key for this substitution pair
    sub_key = (str(prog1), str(prog2))

    def make_anti_var() -> Tuple[Index, List[Tuple[Program, Program]]]:
        """Create an anti-unified variable, accounting for binding depth."""
        if sub_key in substitution_map:
            # Use existing variable, shifted by current binding depth
            return Index(substitution_map[sub_key] + binding_depth), []
        else:
            # Create new variable
            new_base_idx = len(substitution_map)
            substitution_map[sub_key] = new_base_idx
            return Index(new_base_idx + binding_depth), [(prog1, prog2)]

    # If both are the same type of node, try to unify children
    if isinstance(prog1, Primitive) and isinstance(prog2, Primitive):
        # Different primitives - create a variable
        idx, subs = make_anti_var()
        return idx, subs, substitution_map

    if isinstance(prog1, Index) and isinstance(prog2, Index):
        if prog1.i == prog2.i:
            # Same variable reference
            return prog1, [], substitution_map
        else:
            # Different variables - create anti-var
            idx, subs = make_anti_var()
            return idx, subs, substitution_map

    if isinstance(prog1, Application) and isinstance(prog2, Application):
        # Try to unify both function and argument
        f_pattern, f_subs, substitution_map = anti_unify(
            prog1.f, prog2.f, substitution_map, binding_depth
        )
        x_pattern, x_subs, substitution_map = anti_unify(
            prog1.x, prog2.x, substitution_map, binding_depth
        )

        if f_pattern is not None and x_pattern is not None:
            return Application(f_pattern, x_pattern), f_subs + x_subs, substitution_map

        # Couldn't unify children - treat whole thing as a difference
        idx, subs = make_anti_var()
        return idx, subs, substitution_map

    if isinstance(prog1, Abstraction) and isinstance(prog2, Abstraction):
        # Inside abstraction, binding depth increases by 1
        # This ensures anti-unified vars don't collide with bound vars
        body_pattern, body_subs, substitution_map = anti_unify(
            prog1.body, prog2.body, substitution_map, binding_depth + 1
        )
        if body_pattern is not None:
            return Abstraction(body_pattern), body_subs, substitution_map

        idx, subs = make_anti_var()
        return idx, subs, substitution_map

    # Different node types - create a variable
    idx, subs = make_anti_var()
    return idx, subs, substitution_map


def find_anti_unified_patterns(
    programs: List[Program],
    min_uses: int = 2
) -> List[Tuple[Program, int, float]]:
    """
    Find patterns that generalize across multiple programs via anti-unification.

    For each pair of programs, computes anti-unification and tracks patterns
    that appear in multiple programs.

    Args:
        programs: List of programs to analyze
        min_uses: Minimum number of programs a pattern must cover

    Returns:
        List of (pattern, n_uses, savings) tuples sorted by savings

    COMPLEXITY:
        O(n² × m) where n = number of programs, m = average program size
        Could be expensive for large program sets.
    """
    # For each pair of programs, find anti-unification
    patterns: Dict[str, Tuple[Program, List[Tuple[Program, Program]], int]] = {}

    for i, prog1 in enumerate(programs):
        for j, prog2 in enumerate(programs):
            if i >= j:
                continue

            pattern, subs, _ = anti_unify(prog1, prog2)

            if pattern is None:
                continue

            # Only consider patterns with at least one substitution
            # (otherwise it's just an exact match)
            if len(subs) == 0:
                continue

            pattern_key = str(pattern)

            if pattern_key not in patterns:
                patterns[pattern_key] = (pattern, subs, 2)  # Found in 2 programs
            else:
                _, _, count = patterns[pattern_key]
                patterns[pattern_key] = (pattern, subs, count + 1)

    # Convert to list and compute savings
    result = []
    for pattern_key, (pattern, subs, count) in patterns.items():
        if count >= min_uses:
            # Savings: pattern_size * (count - 1) - overhead of abstraction
            pattern_size = pattern.size()
            # Each unique substitution adds one lambda and one application
            n_vars = len(subs)
            overhead = n_vars * 2  # lambda + application for each variable
            savings = pattern_size * (count - 1) - overhead
            if savings > 0:
                result.append((pattern, count, savings))

    # Sort by savings (highest first)
    result.sort(key=lambda x: -x[2])
    return result


def create_abstraction_from_pattern(
    pattern: Program,
    substitutions: List[Tuple[Program, Program]]
) -> Optional[Invented]:
    """
    Create an invented abstraction from an anti-unified pattern.

    The pattern has Index nodes where the programs differed.
    We wrap it in a lambda for each unique substitution variable.
    """
    if not substitutions:
        return None

    # Count unique variable positions in pattern (Index nodes)
    n_vars = len(set(str(s[0]) for s in substitutions))

    # Wrap in lambdas
    body = pattern
    for _ in range(n_vars):
        body = Abstraction(body)

    return Invented(body)


# ============================================================================
# COMMON SUBTREE FINDING (Exact Matches)
# ============================================================================

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


# ============================================================================
# ABSTRACTION CREATION
# ============================================================================

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


# ============================================================================
# PROGRAM REWRITING
# ============================================================================

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


# ============================================================================
# SEMANTIC VERIFICATION
# ============================================================================

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


# ============================================================================
# MAIN COMPRESSION FUNCTION
# ============================================================================

def compress_frontiers(
    grammar: Grammar,
    frontiers: List[List[Tuple[Program, float]]],
    max_inventions: int = 5,
    min_savings: float = 2.0,
    use_anti_unification: bool = True,
    refactor_programs: bool = True
) -> CompressionResult:
    """
    Compress a set of frontiers by extracting common abstractions.

    This is the main compression function. It uses two complementary approaches:
    1. Exact subtree matching - finds identical subtrees across programs
    2. Anti-unification - finds structurally similar patterns that differ only
       in specific positions (e.g., same structure but different property accessor)

    Args:
        grammar: Current grammar
        frontiers: List of frontiers, each is [(program, log_likelihood), ...]
        max_inventions: Maximum number of new abstractions to add
        min_savings: Minimum description length savings to bother
        use_anti_unification: Whether to also find patterns via anti-unification
        refactor_programs: If True, rewrite all programs to use each new invention
                          before looking for the next one. This enables finding
                          HIERARCHICAL abstractions (patterns that use patterns).
                          Default: True (matching original DreamCoder behavior).

    Returns:
        CompressionResult with new grammar, inventions, and optionally rewritten
        frontiers (if refactor_programs=True).

    ALGORITHM:
        1. Collect all programs from frontiers
        2. Find common subtrees (exact matches)
        3. Find anti-unified patterns (structural similarity)
        4. Greedily select inventions by savings:
           a. Create invention from best candidate
           b. Add to grammar
           c. If refactor_programs: rewrite ALL programs to use invention
           d. Re-analyze rewritten programs for next candidate
        5. Return final grammar and rewritten frontiers

    PROGRAM REFACTORING (refactor_programs=True):
        After selecting each invention, we rewrite ALL programs across ALL
        frontiers to use the new abstraction. This is crucial because:

        1. HIERARCHICAL ABSTRACTIONS: Patterns that use other patterns.
           Round 1 finds: #inc = λx.(+ x 1)
           After rewriting: programs contain (#inc x) instead of (+ x 1)
           Round 2 can find: #add2 = λx.(#inc (#inc x))

        2. ACCURATE SAVINGS: The savings calculation assumes programs will
           be rewritten. If we don't rewrite, we overestimate future savings.

        3. GRAMMAR UPDATES: The inside-outside algorithm should see the
           shorter, rewritten programs.

    COMPARISON TO ORIGINAL DREAMCODER:
        Original uses beam search and full MDL scoring.
        Our greedy approach is faster but may miss better solutions.
        With refactor_programs=True, we match the key behavior of rewriting
        programs after each invention.
    """
    # Collect all programs from frontiers
    all_programs = []
    for frontier in frontiers:
        for prog, _ in frontier:
            all_programs.append(prog)

    if not all_programs:
        return CompressionResult(
            new_inventions=[],
            old_grammar=grammar,
            new_grammar=grammar,
            total_savings=0.0,
            subtree_analysis=[],
            rewritten_frontiers=frontiers if refactor_programs else None,
            rewrite_stats={'total_replacements': 0, 'programs_changed': 0,
                          'total_size_reduction': 0} if refactor_programs else None
        )

    # Working copies that get updated as we add inventions
    current_frontiers = [list(f) for f in frontiers]  # Deep copy
    current_grammar = grammar

    # Track all inventions and their targets (for rewriting)
    new_inventions: List[Invented] = []
    invention_targets: List[Tuple[Program, int]] = []  # (target, n_args) pairs
    total_savings = 0.0

    # Aggregate rewrite statistics
    aggregate_rewrite_stats = {
        'total_replacements': 0,
        'programs_changed': 0,
        'total_size_reduction': 0,
        'inventions_applied': 0
    }

    # Initial subtree analysis (before any rewriting)
    initial_common = find_common_subtrees(all_programs, min_size=2, min_count=2)

    while len(new_inventions) < max_inventions:
        # Get current programs from current frontiers
        current_programs = []
        for frontier in current_frontiers:
            for prog, _ in frontier:
                current_programs.append(prog)

        # Find common subtrees in CURRENT (possibly rewritten) programs
        common = find_common_subtrees(current_programs, min_size=2, min_count=2)

        # Also find anti-unified patterns
        anti_unified_patterns = []
        if use_anti_unification and len(current_programs) >= 2:
            anti_unified_patterns = find_anti_unified_patterns(
                current_programs, min_uses=2
            )

        # Find the best candidate (highest savings)
        best_candidate = None
        best_savings = min_savings
        best_type = None  # 'exact' or 'anti'

        # Check exact subtree matches
        for occ in common:
            if occ.savings > best_savings:
                invention, n_args = abstract_subtree(occ.subtree)
                if current_grammar.get_production(invention) is None:
                    # Verify type inference works
                    try:
                        ctx = TypeContext()
                        invention.infer_type(ctx, [])
                        best_candidate = (invention, n_args, occ.subtree, occ.savings)
                        best_savings = occ.savings
                        best_type = 'exact'
                    except Exception:
                        continue

        # Check anti-unified patterns
        for pattern, count, savings in anti_unified_patterns:
            if savings > best_savings:
                free_indices = pattern.free_indices()
                n_vars = len(free_indices)
                if n_vars == 0:
                    continue

                # Create invention
                body = pattern
                for _ in range(n_vars):
                    body = Abstraction(body)
                invention = Invented(body)

                if current_grammar.get_production(invention) is None:
                    try:
                        ctx = TypeContext()
                        invention.infer_type(ctx, [])
                        best_candidate = (invention, n_vars, pattern, savings)
                        best_savings = savings
                        best_type = 'anti'
                    except Exception:
                        continue

        # If no good candidate found, we're done
        if best_candidate is None:
            break

        # Unpack the best candidate
        invention, n_args, target, savings = best_candidate

        # Add invention to grammar
        ctx = TypeContext()
        tp = invention.infer_type(ctx, [])
        log_prob = math.log(0.1)  # 10% prior probability
        current_grammar = current_grammar.with_production(
            Production(invention, tp, log_prob)
        )

        new_inventions.append(invention)
        invention_targets.append((target, n_args))
        total_savings += savings

        # Refactor programs if enabled
        if refactor_programs:
            rewritten_frontiers, stats = rewrite_all_frontiers(
                current_frontiers, target, invention, n_args
            )
            current_frontiers = rewritten_frontiers

            aggregate_rewrite_stats['total_replacements'] += stats['total_replacements']
            aggregate_rewrite_stats['programs_changed'] += stats['programs_changed']
            aggregate_rewrite_stats['total_size_reduction'] += stats['total_size_reduction']
            aggregate_rewrite_stats['inventions_applied'] += 1

    # Build final result
    return CompressionResult(
        new_inventions=new_inventions,
        old_grammar=grammar,
        new_grammar=current_grammar.normalize_probabilities(),
        total_savings=total_savings,
        subtree_analysis=initial_common,
        rewritten_frontiers=current_frontiers if refactor_programs else None,
        rewrite_stats=aggregate_rewrite_stats if refactor_programs else None
    )


# Keep the old function signature for backwards compatibility
def compress_frontiers_legacy(
    grammar: Grammar,
    frontiers: List[List[Tuple[Program, float]]],
    max_inventions: int = 5,
    min_savings: float = 2.0,
    use_anti_unification: bool = True
) -> CompressionResult:
    """
    Legacy version of compress_frontiers WITHOUT program refactoring.

    This is preserved for backwards compatibility and comparison testing.
    For new code, use compress_frontiers(refactor_programs=False) instead.
    """
    # Collect all programs from frontiers
    all_programs = []
    for frontier in frontiers:
        for prog, _ in frontier:
            all_programs.append(prog)

    if not all_programs:
        return CompressionResult(
            new_inventions=[],
            old_grammar=grammar,
            new_grammar=grammar,
            total_savings=0.0,
            subtree_analysis=[]
        )

    # Find common subtrees (exact matches)
    common = find_common_subtrees(all_programs, min_size=2, min_count=2)

    # Also find anti-unified patterns (structural similarity)
    anti_unified_patterns = []
    if use_anti_unification and len(all_programs) >= 2:
        anti_unified_patterns = find_anti_unified_patterns(all_programs, min_uses=2)

    # Greedily select inventions
    new_inventions = []
    total_savings = 0.0
    current_grammar = grammar

    # First try exact subtree matches (usually higher confidence)
    for occ in common:
        if len(new_inventions) >= max_inventions:
            break

        if occ.savings < min_savings:
            break

        # Create the invention
        invention, n_args = abstract_subtree(occ.subtree)

        # Check if this is already in the grammar
        if current_grammar.get_production(invention) is not None:
            continue

        # Add to grammar
        try:
            ctx = TypeContext()
            tp = invention.infer_type(ctx, [])
            log_prob = math.log(0.1)  # 10% prior probability
            current_grammar = current_grammar.with_production(
                Production(invention, tp, log_prob)
            )
            new_inventions.append(invention)
            total_savings += occ.savings

        except Exception as e:
            # Type inference failed - skip this invention
            continue

    # Then try anti-unified patterns (if we have room for more inventions)
    for pattern, count, savings in anti_unified_patterns:
        if len(new_inventions) >= max_inventions:
            break

        if savings < min_savings:
            break

        free_indices = pattern.free_indices()
        n_vars = len(free_indices)

        if n_vars == 0:
            continue

        body = pattern
        for _ in range(n_vars):
            body = Abstraction(body)

        invention = Invented(body)

        if current_grammar.get_production(invention) is not None:
            continue

        try:
            ctx = TypeContext()
            tp = invention.infer_type(ctx, [])
            log_prob = math.log(0.1)
            current_grammar = current_grammar.with_production(
                Production(invention, tp, log_prob)
            )
            new_inventions.append(invention)
            total_savings += savings

        except Exception as e:
            # Type inference failed - skip this invention
            continue

    return CompressionResult(
        new_inventions=new_inventions,
        old_grammar=grammar,
        new_grammar=current_grammar.normalize_probabilities(),
        total_savings=total_savings,
        subtree_analysis=common
    )


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _has_index(program: Program) -> bool:
    """Check if program contains any Index nodes."""
    if isinstance(program, Index):
        return True
    if isinstance(program, (Primitive, Invented)):
        return False
    if isinstance(program, Application):
        return _has_index(program.f) or _has_index(program.x)
    if isinstance(program, Abstraction):
        return _has_index(program.body)
    return False


def _count_max_index(program: Program) -> int:
    """Find the maximum Index value in a program."""
    if isinstance(program, Index):
        return program.i
    if isinstance(program, (Primitive, Invented)):
        return -1
    if isinstance(program, Application):
        return max(_count_max_index(program.f), _count_max_index(program.x))
    if isinstance(program, Abstraction):
        return _count_max_index(program.body)
    return -1


def compute_compression_ratio(
    programs: List[Program],
    grammar: Grammar,
    new_grammar: Grammar,
    request_type: Type
) -> Tuple[float, float]:
    """
    Compute the compression ratio achieved.

    Returns:
        (old_total_dl, new_total_dl) description lengths in bits

    NOTE: This computes DL of ORIGINAL programs with old vs new grammar.
    For true MDL comparison, we should also rewrite programs and
    measure DL of rewritten programs with new grammar.
    """
    old_dl = sum(grammar.description_length(p, request_type) for p in programs)
    new_dl = sum(new_grammar.description_length(p, request_type) for p in programs)
    return old_dl, new_dl


# ============================================================================
# ITERATIVE COMPRESSION (multiple rounds)
# ============================================================================

def iterative_compression(
    grammar: Grammar,
    frontiers: List[List[Tuple[Program, float]]],
    max_rounds: int = 3,
    max_inventions_per_round: int = 3,
    refactor_programs: bool = True
) -> CompressionResult:
    """
    Perform multiple rounds of compression.

    Each round finds new abstractions based on the current (possibly rewritten) programs.
    Between rounds, frontiers are updated to use the newly learned abstractions,
    enabling the discovery of HIERARCHICAL ABSTRACTIONS.

    Args:
        grammar: Starting grammar
        frontiers: Task solutions
        max_rounds: Maximum compression iterations
        max_inventions_per_round: Cap on new abstractions per round
        refactor_programs: If True, use rewritten programs from each round as input
                          to the next round. This enables hierarchical abstractions.
                          Default: True (matching original DreamCoder behavior).

    Returns:
        Aggregated CompressionResult with:
        - All inventions from all rounds
        - Final grammar with all inventions
        - Total savings across all rounds
        - Final rewritten frontiers (if refactor_programs=True)
        - Aggregated rewrite statistics

    WHY MULTIPLE ROUNDS?
        Round 1 finds: #inc = (λ (+ $0 1))
        Programs are rewritten: (+ (+ x 1) 1) becomes (#inc (#inc x))
        Round 2 finds: #add2 = (λ (#inc (#inc $0)))

        Hierarchical abstractions build on each other.

    ALGORITHM:
        1. For each round:
           a. Run compress_frontiers() on current frontiers
           b. Update grammar with new inventions
           c. If refactor_programs: use rewritten frontiers for next round
        2. Return aggregated results
    """
    current_grammar = grammar
    current_frontiers = [list(f) for f in frontiers]  # Deep copy
    all_inventions: List[Invented] = []
    total_savings = 0.0

    # Aggregate rewrite statistics across all rounds
    aggregate_rewrite_stats = {
        'total_replacements': 0,
        'programs_changed': 0,
        'total_size_reduction': 0,
        'inventions_applied': 0,
        'rounds_completed': 0
    }

    for round_num in range(max_rounds):
        result = compress_frontiers(
            current_grammar,
            current_frontiers,
            max_inventions=max_inventions_per_round,
            refactor_programs=refactor_programs
        )

        if not result.new_inventions:
            # No more compression opportunities
            break

        current_grammar = result.new_grammar
        all_inventions.extend(result.new_inventions)
        total_savings += result.total_savings
        aggregate_rewrite_stats['rounds_completed'] += 1

        # Update frontiers for next round (if refactoring is enabled)
        if refactor_programs and result.rewritten_frontiers is not None:
            current_frontiers = result.rewritten_frontiers

            # Aggregate rewrite stats
            if result.rewrite_stats:
                aggregate_rewrite_stats['total_replacements'] += result.rewrite_stats.get('total_replacements', 0)
                aggregate_rewrite_stats['programs_changed'] += result.rewrite_stats.get('programs_changed', 0)
                aggregate_rewrite_stats['total_size_reduction'] += result.rewrite_stats.get('total_size_reduction', 0)
                aggregate_rewrite_stats['inventions_applied'] += result.rewrite_stats.get('inventions_applied', 0)

    return CompressionResult(
        new_inventions=all_inventions,
        old_grammar=grammar,
        new_grammar=current_grammar,
        total_savings=total_savings,
        subtree_analysis=[],  # Aggregate not available across rounds
        rewritten_frontiers=current_frontiers if refactor_programs else None,
        rewrite_stats=aggregate_rewrite_stats if refactor_programs else None
    )


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def format_invention(invention: Invented) -> str:
    """Format an invention for display."""
    if invention.name:
        return f"#{invention.name}: {invention.body}"
    return f"#({invention.body})"


def compression_report(result: CompressionResult) -> str:
    """Generate a human-readable compression report."""
    lines = ["=" * 60]
    lines.append("COMPRESSION REPORT")
    lines.append("=" * 60)

    lines.append(f"\nNew inventions: {len(result.new_inventions)}")
    for i, inv in enumerate(result.new_inventions):
        lines.append(f"  {i+1}. {format_invention(inv)}")

    lines.append(f"\nTotal savings: {result.total_savings:.2f} (heuristic, not true MDL)")

    if result.subtree_analysis:
        lines.append("\nTop common subtrees:")
        for occ in result.subtree_analysis[:10]:
            lines.append(f"  {occ}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


# ============================================================================
# COMPARISON TO ORIGINAL DREAMCODER
# ============================================================================
"""
+------------------------+---------------------------+---------------------------+
| Component              | Original DreamCoder       | Our Implementation        |
+------------------------+---------------------------+---------------------------+
| Search Strategy        | Beam search over states   | Greedy selection          |
+------------------------+---------------------------+---------------------------+
| Scoring                | Full MDL:                 | Heuristic:                |
|                        | DL(G) + Σ DL(P_i|G)      | (size-1) × (count-1)      |
+------------------------+---------------------------+---------------------------+
| Program Refactoring    | Yes - rewrites all        | YES - rewrites all        |
|                        | programs after invention  | programs after invention  |
|                        |                           | (refactor_programs=True)  |
+------------------------+---------------------------+---------------------------+
| Arity Search           | Considers multiple        | Single canonical          |
|                        | factorizations            | (all free vars)           |
+------------------------+---------------------------+---------------------------+
| Grammar Size Penalty   | Yes - penalizes complex   | No - grammar can grow     |
|                        | grammars explicitly       | without penalty           |
+------------------------+---------------------------+---------------------------+
| Pattern Discovery      | Version spaces            | Exact match +             |
|                        |                           | anti-unification          |
+------------------------+---------------------------+---------------------------+
| Corpus Guidance        | Uses recognition model    | Syntax-only               |
|                        | predictions               |                           |
+------------------------+---------------------------+---------------------------+

STATUS (as of implementation):
✓ DONE: Program refactoring after compression (enables hierarchical abstractions)
  - compress_frontiers() with refactor_programs=True (default)
  - iterative_compression() passes rewritten frontiers between rounds
  - Semantic verification available via verify_rewrite_semantics()

TODO:
1. Full MDL scoring (principled abstraction selection)
2. Beam search (avoid local optima)
3. Arity-aware search (better abstractions)
4. Corpus-guided compression (recognition model integration)
"""


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("COMPREHENSIVE COMPRESSION TESTS")
    print("=" * 70)

    from .grammar import uniform_grammar

    # ========================================================================
    # Setup: Create simple primitives
    # ========================================================================
    print("\n1. SETUP: Creating primitives and grammar")
    print("-" * 50)

    add = Primitive('+', arrow(INT, INT, INT), lambda x: lambda y: x + y)
    mul = Primitive('*', arrow(INT, INT, INT), lambda x: lambda y: x * y)
    zero = Primitive('0', INT, 0)
    one = Primitive('1', INT, 1)
    two = Primitive('2', INT, 2)
    three = Primitive('3', INT, 3)

    g = uniform_grammar([add, mul, zero, one, two, three])
    print(f"  Created grammar with {len(g.productions)} productions")

    # ========================================================================
    # Test 1: Basic rewrite_with_invention
    # ========================================================================
    print("\n2. TEST: rewrite_with_invention()")
    print("-" * 50)

    # Create an invention: #inc = λx.(+ x 1)
    inc_body = Abstraction(Application(Application(add, Index(0)), one))
    inc_inv = Invented(inc_body)
    print(f"  Created invention: #inc = {inc_inv.body}")

    # Create a program that uses (+ x 1) multiple times
    # (+ (+ 5 1) 1) = (+ (+ 5 1) 1) where 5 is represented as (+ 2 3)
    five = Application(Application(add, two), three)
    add_one = Application(Application(add, five), one)  # (+ 5 1)
    add_two = Application(Application(add, add_one), one)  # (+ (+ 5 1) 1)

    print(f"  Original program: {add_two}")
    print(f"  Original value: {add_two.evaluate([])}")

    # The target pattern is (+ $0 1) - we need to create it properly
    # For programs like (+ 5 1), we want to rewrite to (#inc 5)
    target = Application(Application(add, Index(0)), one)  # (+ $0 1)
    print(f"  Target pattern: {target}")

    # Test with a program inside a lambda
    # λx. (+ (+ x 1) 1)
    inner_add = Application(Application(add, Index(0)), one)  # (+ $0 1)
    outer_add = Application(Application(add, inner_add), one)  # (+ (+ $0 1) 1)
    lambda_prog = Abstraction(outer_add)
    print(f"\n  Lambda program: {lambda_prog}")
    print(f"  Lambda(5): {lambda_prog.evaluate([])(5)}")

    rewritten = rewrite_with_invention(lambda_prog, target, inc_inv, 1)
    print(f"  Rewritten: {rewritten}")
    print(f"  Rewritten(5): {rewritten.evaluate([])(5)}")

    # Verify semantics preserved
    assert lambda_prog.evaluate([])(5) == rewritten.evaluate([])(5), "Semantics changed!"
    assert lambda_prog.evaluate([])(10) == rewritten.evaluate([])(10), "Semantics changed!"
    print("  ✓ Semantics preserved!")

    # ========================================================================
    # Test 2: rewrite_with_invention_detailed
    # ========================================================================
    print("\n3. TEST: rewrite_with_invention_detailed()")
    print("-" * 50)

    result = rewrite_with_invention_detailed(lambda_prog, target, inc_inv, 1)
    print(f"  Original size: {result.original_size}")
    print(f"  New size: {result.new_size}")
    print(f"  Replacements: {result.n_replacements}")
    print(f"  Size reduction: {result.original_size - result.new_size}")
    # Note: Only 1 replacement because (+ (+ $0 1) 1) doesn't match (+ $0 1)
    # The outer (+ ... 1) has (+ $0 1) as first arg, not $0
    assert result.n_replacements == 1, f"Expected 1 replacement, got {result.n_replacements}"
    print("  ✓ Detailed stats correct!")

    # ========================================================================
    # Test 3: verify_rewrite_semantics
    # ========================================================================
    print("\n4. TEST: verify_rewrite_semantics()")
    print("-" * 50)

    # Note: verify_rewrite_semantics compares evaluate() results directly.
    # For lambda programs, evaluate([]) returns a closure, which can't be
    # compared directly. Let's test with non-lambda programs.

    # Create ground programs (no lambdas) for verification testing
    ground_prog = Application(Application(add, two), one)  # (+ 2 1) = 3
    ground_target = Application(Application(add, two), Index(0))  # (+ 2 $0)

    # Can't really rewrite a ground program with a pattern that has free vars
    # Let's test verification with identical programs (no rewrite needed)
    success, error = verify_rewrite_semantics(
        ground_prog, ground_prog, [[]], verbose=True
    )
    assert success, f"Verification failed for identical programs: {error}"
    print("  ✓ Identical programs verify correctly")

    # Test with a program that actually changes (will create mismatch)
    wrong_prog = Application(Application(add, three), one)  # (+ 3 1) = 4
    success, error = verify_rewrite_semantics(
        ground_prog, wrong_prog, [[]]
    )
    assert not success, "Should detect semantic mismatch"
    print(f"  ✓ Correctly detected mismatch: {error}")

    # Manual verification for lambda programs (already done above, but let's be explicit)
    print("\n  Manual lambda verification:")
    for test_val in [3, 5, 10, 0, -5]:
        orig_result = lambda_prog.evaluate([])(test_val)
        new_result = rewritten.evaluate([])(test_val)
        status = "✓" if orig_result == new_result else "✗"
        print(f"    {status} f({test_val}): {orig_result} == {new_result}")
        assert orig_result == new_result, f"Lambda semantics differ at {test_val}"
    print("  ✓ Lambda semantic verification passed!")

    # ========================================================================
    # Test 4: rewrite_frontier and rewrite_all_frontiers
    # ========================================================================
    print("\n5. TEST: rewrite_frontier() and rewrite_all_frontiers()")
    print("-" * 50)

    # Create multiple lambda programs with the same pattern
    prog1 = Abstraction(Application(Application(add, Index(0)), one))  # λx.(+ x 1) - 1 match
    prog2 = Abstraction(Application(Application(add,
        Application(Application(add, Index(0)), one)), one))  # λx.(+ (+ x 1) 1) - 1 match
    prog3 = Abstraction(Application(Application(mul, Index(0)), two))  # λx.(* x 2) - no match

    frontier = [(prog1, -1.0), (prog2, -2.0), (prog3, -1.5)]
    rewritten_frontier, stats = rewrite_frontier(frontier, target, inc_inv, 1)

    print(f"  Programs changed: {stats['programs_changed']}")
    print(f"  Total replacements: {stats['total_replacements']}")
    print(f"  Size reduction: {stats['total_size_reduction']}")

    for i, ((orig, _), (new, _)) in enumerate(zip(frontier, rewritten_frontier)):
        print(f"  Program {i+1}: {orig} → {new}")

    # prog1: λx.(+ x 1) matches (+ $0 1) -> 1 replacement
    # prog2: λx.(+ (+ x 1) 1) has inner (+ $0 1) -> 1 replacement
    # prog3: no match
    assert stats['programs_changed'] == 2, f"Expected 2 programs changed, got {stats['programs_changed']}"
    assert stats['total_replacements'] == 2, f"Expected 2 total replacements, got {stats['total_replacements']}"
    print("  ✓ Frontier rewriting correct!")

    # Test rewrite_all_frontiers
    frontiers = [frontier, [(prog1, -0.5)]]
    rewritten_all, all_stats = rewrite_all_frontiers(frontiers, target, inc_inv, 1)
    print(f"\n  All frontiers stats:")
    print(f"    Programs changed: {all_stats['programs_changed']}")
    print(f"    Frontiers affected: {all_stats['frontiers_affected']}")
    print("  ✓ All frontiers rewriting correct!")

    # ========================================================================
    # Test 5: compress_frontiers with refactor_programs=True
    # ========================================================================
    print("\n6. TEST: compress_frontiers(refactor_programs=True)")
    print("-" * 50)

    # Create programs that share (+ 1 $0) pattern
    p1 = Application(Application(add, one), one)   # (+ 1 1) = 2
    p2 = Application(Application(add, one), two)   # (+ 1 2) = 3
    p3 = Application(Application(add, one), three) # (+ 1 3) = 4
    p4 = Application(Application(add, one),        # (+ 1 (+ 1 1)) = 3
            Application(Application(add, one), one))

    test_frontiers = [[(p, 0.0)] for p in [p1, p2, p3, p4]]

    print("  Original programs:")
    for i, f in enumerate(test_frontiers):
        prog = f[0][0]
        print(f"    {i+1}. {prog} = {prog.evaluate([])}")

    result = compress_frontiers(
        g, test_frontiers,
        max_inventions=3,
        min_savings=0.5,
        refactor_programs=True
    )

    print(f"\n  New inventions: {len(result.new_inventions)}")
    for inv in result.new_inventions:
        print(f"    {format_invention(inv)}")

    print(f"\n  Total savings: {result.total_savings:.2f}")

    if result.rewrite_stats:
        print(f"\n  Rewrite stats:")
        print(f"    Total replacements: {result.rewrite_stats['total_replacements']}")
        print(f"    Programs changed: {result.rewrite_stats['programs_changed']}")
        print(f"    Size reduction: {result.rewrite_stats['total_size_reduction']}")

    if result.rewritten_frontiers:
        print("\n  Rewritten programs:")
        for i, f in enumerate(result.rewritten_frontiers):
            prog = f[0][0]
            print(f"    {i+1}. {prog} = {prog.evaluate([])}")

    # Verify semantics preserved for all programs
    print("\n  Verifying semantics...")
    for orig_f, new_f in zip(test_frontiers, result.rewritten_frontiers or test_frontiers):
        orig_val = orig_f[0][0].evaluate([])
        new_val = new_f[0][0].evaluate([])
        assert orig_val == new_val, f"Semantics changed: {orig_val} vs {new_val}"
    print("  ✓ All semantics preserved!")

    # ========================================================================
    # Test 6: iterative_compression with hierarchical abstractions
    # ========================================================================
    print("\n7. TEST: iterative_compression() - Hierarchical Abstractions")
    print("-" * 50)

    # Create programs that can form hierarchical abstractions
    # (+ (+ x 1) 1) should become (#inc (#inc x)) then #add2 x

    # λx.(+ (+ x 1) 1) - add 2
    add2_prog = Abstraction(Application(Application(add,
        Application(Application(add, Index(0)), one)), one))

    # λx.(+ (+ (+ x 1) 1) 1) - add 3
    add3_prog = Abstraction(Application(Application(add,
        Application(Application(add,
            Application(Application(add, Index(0)), one)), one)), one))

    # Create frontiers with multiple instances to ensure patterns are found
    iter_frontiers = [
        [(add2_prog, -1.0)],
        [(add3_prog, -1.5)],
        [(add2_prog, -2.0)],  # Duplicate to increase count
    ]

    print("  Original programs:")
    for i, f in enumerate(iter_frontiers):
        prog = f[0][0]
        print(f"    {i+1}. {prog} evaluated at 5: {prog.evaluate([])(5)}")

    result = iterative_compression(
        g, iter_frontiers,
        max_rounds=3,
        max_inventions_per_round=2,
        refactor_programs=True
    )

    print(f"\n  Rounds completed: {result.rewrite_stats['rounds_completed'] if result.rewrite_stats else 0}")
    print(f"  Total inventions: {len(result.new_inventions)}")
    for inv in result.new_inventions:
        print(f"    {format_invention(inv)}")

    print(f"\n  Total savings: {result.total_savings:.2f}")

    if result.rewritten_frontiers:
        print("\n  Final rewritten programs:")
        for i, f in enumerate(result.rewritten_frontiers):
            prog = f[0][0]
            print(f"    {i+1}. {prog}")

    # Verify semantics
    print("\n  Verifying semantics preserved...")
    for orig_f, new_f in zip(iter_frontiers, result.rewritten_frontiers or iter_frontiers):
        for test_val in [0, 5, 10]:
            orig_val = orig_f[0][0].evaluate([])(test_val)
            new_val = new_f[0][0].evaluate([])(test_val)
            assert orig_val == new_val, f"Semantics changed at {test_val}: {orig_val} vs {new_val}"
    print("  ✓ All semantics preserved!")

    # ========================================================================
    # Test 7: Edge cases
    # ========================================================================
    print("\n8. TEST: Edge Cases")
    print("-" * 50)

    # Empty frontiers
    empty_result = compress_frontiers(g, [], refactor_programs=True)
    assert empty_result.new_inventions == [], "Empty frontiers should yield no inventions"
    assert empty_result.rewritten_frontiers == [], "Empty frontiers should return empty"
    print("  ✓ Empty frontiers handled correctly")

    # Single program (no common patterns)
    single_result = compress_frontiers(g, [[(p1, 0.0)]], refactor_programs=True)
    assert single_result.new_inventions == [], "Single program should yield no inventions"
    print("  ✓ Single program handled correctly")

    # No pattern matches
    no_match_progs = [
        [(Application(Application(add, one), two), 0.0)],
        [(Application(Application(mul, two), three), 0.0)],
    ]
    no_match_result = compress_frontiers(g, no_match_progs, min_savings=100.0)
    assert no_match_result.new_inventions == [], "High threshold should yield no inventions"
    print("  ✓ No-match case handled correctly")

    # ========================================================================
    # Test 8: Legacy function backwards compatibility
    # ========================================================================
    print("\n9. TEST: compress_frontiers_legacy() backwards compatibility")
    print("-" * 50)

    legacy_result = compress_frontiers_legacy(g, test_frontiers, max_inventions=2)
    print(f"  Legacy inventions: {len(legacy_result.new_inventions)}")
    assert legacy_result.rewritten_frontiers is None, "Legacy should not have rewritten frontiers"
    assert legacy_result.rewrite_stats is None, "Legacy should not have rewrite stats"
    print("  ✓ Legacy function works correctly")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
    print("""
Summary of tested functionality:
  1. rewrite_with_invention() - syntactic pattern replacement
  2. rewrite_with_invention_detailed() - replacement with statistics
  3. verify_rewrite_semantics() - semantic equivalence checking
  4. rewrite_frontier() - frontier-level rewriting
  5. rewrite_all_frontiers() - multi-frontier rewriting
  6. compress_frontiers(refactor_programs=True) - end-to-end compression
  7. iterative_compression() - multi-round hierarchical compression
  8. Edge cases (empty, single, no-match)
  9. Legacy backwards compatibility
""")
