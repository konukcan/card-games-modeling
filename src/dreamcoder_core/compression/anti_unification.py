"""
Anti-unification for pattern discovery.

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

EXTRACTED FROM: compression.py lines 477-700
"""

from typing import Dict, List, Optional, Tuple

from ..program import Program, Primitive, Application, Abstraction, Index, Invented


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
    # Use direct equality (uses __eq__ and __hash__) instead of string comparison
    if prog1 == prog2:
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
