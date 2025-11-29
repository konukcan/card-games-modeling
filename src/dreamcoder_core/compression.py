"""
Library Learning (Compression) for DreamCoder

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


@dataclass
class SubtreeOccurrence:
    """
    Tracks occurrences of a subtree across programs.

    For compression analysis.
    """
    subtree: Program
    count: int  # How many times it appears
    programs: List[str]  # Which programs contain it
    savings: float  # How much description length we'd save by abstracting

    def __str__(self) -> str:
        return f"{self.subtree} (count={self.count}, savings={self.savings:.2f})"


@dataclass
class CompressionResult:
    """Result of compression analysis."""
    new_inventions: List[Invented]
    old_grammar: Grammar
    new_grammar: Grammar
    total_savings: float
    subtree_analysis: List[SubtreeOccurrence]


# ============================================================================
# ANTI-UNIFICATION
# ============================================================================

def anti_unify(
    prog1: Program,
    prog2: Program,
    substitution_map: Optional[Dict[str, int]] = None,
    binding_depth: int = 0
) -> Tuple[Optional[Program], List[Tuple[Program, Program]], Dict[str, int]]:
    """
    Find the most specific generalization of two programs.

    Uses consistent variable indices when the same substitution appears multiple times.
    For example, if (get_color, get_suit) appears twice, both get the same Index.

    The binding_depth parameter tracks how many lambdas we're inside, so that
    anti-unified variables can be shifted appropriately to avoid collision with
    bound variables.

    Returns:
        (pattern, substitutions, substitution_map) where:
        - pattern is the generalized program with Index nodes for differences
        - substitutions is a list of unique (prog1_subterm, prog2_subterm) pairs
        - substitution_map maps (str(sub1), str(sub2)) to their base Index value

    Example:
        prog1 = λh. eq (get_color (first h)) (get_color (last h))
        prog2 = λh. eq (get_suit (first h)) (get_suit (last h))

        Returns: (λh. eq ($1 (first h)) ($1 (last h)), [(get_color, get_suit)], {...})

        Note: $1 is used (not $0) because $0 refers to h inside the lambda.
        When wrapped with λf, the pattern becomes:
        λf. λh. eq (f (first h)) (f (last h))
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
        idx, subs = make_anti_var()
        return idx, subs, substitution_map

    if isinstance(prog1, Index) and isinstance(prog2, Index):
        if prog1.i == prog2.i:
            return prog1, [], substitution_map
        else:
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

    Returns:
        List of (pattern, n_uses, savings) tuples sorted by savings
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

    # Sort by savings
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


def find_common_subtrees(
    programs: List[Program],
    min_size: int = 2,
    min_count: int = 2
) -> List[SubtreeOccurrence]:
    """
    Find subtrees that appear multiple times across programs.

    Args:
        programs: List of programs to analyze
        min_size: Minimum subtree size (AST nodes)
        min_count: Minimum number of occurrences

    Returns:
        List of common subtrees with occurrence counts
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
    Convert a subtree into an abstraction.

    If the subtree has free variables, wrap it in lambdas.

    Returns:
        (invented_program, n_args) where n_args is how many arguments it takes
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
    # Map old indices to new ones
    index_map = {old: new for new, old in enumerate(free_list)}

    # Rewrite the subtree with new indices
    rewritten = _reindex(subtree, index_map)

    # Wrap in n_args lambdas
    body = rewritten
    for _ in range(n_args):
        body = Abstraction(body)

    return Invented(body), n_args


def _reindex(program: Program, index_map: Dict[int, int]) -> Program:
    """Rewrite a program with new de Bruijn indices."""

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
        shifted_map = {k + 1: v + 1 for k, v in index_map.items()}
        return Abstraction(_reindex(program.body, shifted_map))

    return program


def rewrite_with_invention(
    program: Program,
    target: Program,
    invention: Invented,
    n_args: int
) -> Program:
    """
    Rewrite a program by replacing target subtrees with the invention.

    Args:
        program: Program to rewrite
        target: Subtree to replace
        invention: The invented abstraction
        n_args: Number of arguments the invention takes

    Returns:
        Rewritten program using the invention
    """
    if program == target:
        # Replace with invention (applied to its arguments)
        if n_args == 0:
            return invention
        else:
            # Need to apply to the free variables
            free_vars = sorted(target.free_indices())
            result = invention
            for var in free_vars:
                result = Application(result, Index(var))
            return result

    if isinstance(program, (Primitive, Index, Invented)):
        return program

    if isinstance(program, Application):
        return Application(
            rewrite_with_invention(program.f, target, invention, n_args),
            rewrite_with_invention(program.x, target, invention, n_args)
        )

    if isinstance(program, Abstraction):
        # Need to shift the target for the new binding scope
        shifted_target = target.shift(1, 0)
        return Abstraction(
            rewrite_with_invention(program.body, shifted_target, invention, n_args)
        )

    return program


def compress_frontiers(
    grammar: Grammar,
    frontiers: List[List[Tuple[Program, float]]],
    max_inventions: int = 5,
    min_savings: float = 2.0,
    use_anti_unification: bool = True
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

    Returns:
        CompressionResult with new grammar and inventions
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

    # First try exact subtree matches
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
            # Give it a reasonable prior (slightly favored for being useful)
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

        # Create the invention by wrapping pattern in a lambda
        # The pattern already has Index nodes where programs differed
        # Count FREE indices to determine how many lambdas needed
        # (bound indices from existing lambdas in the pattern don't count)
        free_indices = pattern.free_indices()
        n_vars = len(free_indices)

        if n_vars == 0:
            # No free variables - skip (it's an exact match, already handled)
            continue

        # Wrap in lambdas for the free (anti-unified) variables
        body = pattern
        for _ in range(n_vars):
            body = Abstraction(body)

        invention = Invented(body)

        # Check if already in grammar
        if current_grammar.get_production(invention) is not None:
            continue

        # Add to grammar
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
        (old_total_dl, new_total_dl) description lengths
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
    max_inventions_per_round: int = 3
) -> CompressionResult:
    """
    Perform multiple rounds of compression.

    Each round finds new abstractions based on the current programs.
    """
    current_grammar = grammar
    all_inventions = []
    total_savings = 0.0

    for round_num in range(max_rounds):
        result = compress_frontiers(
            current_grammar,
            frontiers,
            max_inventions=max_inventions_per_round
        )

        if not result.new_inventions:
            break

        current_grammar = result.new_grammar
        all_inventions.extend(result.new_inventions)
        total_savings += result.total_savings

        # Rewrite programs with new inventions
        # (This is optional but improves future rounds)

    return CompressionResult(
        new_inventions=all_inventions,
        old_grammar=grammar,
        new_grammar=current_grammar,
        total_savings=total_savings,
        subtree_analysis=[]  # Aggregate not available
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

    lines.append(f"\nTotal savings: {result.total_savings:.2f} bits")

    if result.subtree_analysis:
        lines.append("\nTop common subtrees:")
        for occ in result.subtree_analysis[:10]:
            lines.append(f"  {occ}")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("=== Compression Tests ===\n")

    # Create simple primitives
    add = Primitive('+', arrow(INT, INT, INT), lambda x: lambda y: x + y)
    mul = Primitive('*', arrow(INT, INT, INT), lambda x: lambda y: x * y)
    zero = Primitive('0', INT, 0)
    one = Primitive('1', INT, 1)
    two = Primitive('2', INT, 2)

    # Programs that share structure
    # (+ 1 (+ 1 x)) - add 2 to x
    # (+ 1 (+ 1 (+ 1 x))) - add 3 to x
    # These share (+ 1 _)

    from .grammar import uniform_grammar

    g = uniform_grammar([add, mul, zero, one, two])

    # Create some programs with shared structure
    p1 = Application(Application(add, one), one)  # (+ 1 1)
    p2 = Application(Application(add, one), two)  # (+ 1 2)
    p3 = Application(Application(add, one),       # (+ 1 (+ 1 1))
                     Application(Application(add, one), one))

    programs = [p1, p2, p3]

    print("Programs:")
    for p in programs:
        print(f"  {p} = {p.evaluate([])}")

    # Find common subtrees
    print("\nCommon subtrees:")
    common = find_common_subtrees(programs, min_size=2, min_count=2)
    for occ in common:
        print(f"  {occ}")

    # Compress
    print("\nCompression:")
    frontiers = [[(p, 0.0)] for p in programs]
    result = compress_frontiers(g, frontiers)
    print(compression_report(result))

    print("\n=== Compression Tests OK ===")
