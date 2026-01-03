"""
Arity-aware abstraction search.

When abstracting a subtree with free variables, there are multiple valid
ways to "cut" it - different choices of which variables to abstract over
(the "arity").

EXAMPLE:
    Pattern: (map (λ (+ $0 n)) lst)  with free vars {n, lst}

    2-arg: #f = λ n lst. (map (λ (+ $0 n)) lst)   → (#f n lst)
    1-arg: #g = λ n. (map (λ (+ $0 n)))           → (#g n lst)  -- curried
    1-arg: #h = λ lst. (map (λ (+ $0 1)) lst)     → (#h lst)    -- inline n=1

    Different arities trade off:
    - Higher arity = more general (reusable in more contexts)
    - Lower arity = simpler calls (fewer arguments)

The optimal choice depends on the CORPUS - how the pattern is actually used.
We evaluate each factorization by MDL to pick the best one.

COMPARISON TO ORIGINAL DREAMCODER:
    Original explores multiple factorizations during compression.
    We do the same: enumerate factorizations, score by MDL, pick best.

EXTRACTED FROM: compression.py lines 900-1218
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Set, Tuple

from ..program import Program, Abstraction, Invented
from ..grammar import Grammar
from ..type_system import Type, TypeContext

# Import from sibling modules
from .subtree_finding import abstract_subtree, _reindex


@dataclass
class Factorization:
    """
    A specific way to abstract a subtree.

    FIELDS:
    -------
    invention: Invented
        The abstraction with chosen arity.

    n_args: int
        Number of arguments the invention takes.

    abstracted_vars: Set[int]
        Which free variables were abstracted (became parameters).

    inlined_vars: Set[int]
        Which free variables were inlined (captured from context).
    """
    invention: Invented
    n_args: int
    abstracted_vars: Set[int]
    inlined_vars: Set[int]


def enumerate_factorizations(
    subtree: Program,
    max_args: int = 4
) -> List[Factorization]:
    """
    Enumerate different ways to abstract a subtree.

    For a subtree with free variables {0, 2, 3}, we could create:
    - 3-arg: abstract over all three
    - 2-arg: abstract over any two (3 combinations)
    - 1-arg: abstract over any one (3 combinations)
    - 0-arg: only if subtree has no free variables

    Args:
        subtree: The subtree to factor
        max_args: Maximum arity to consider (higher = slower but more options)

    Returns:
        List of Factorization objects, sorted by arity (highest first)

    ALGORITHM:
        1. Find all free variables in subtree
        2. For each subset of free vars (from all down to 1):
           a. Create abstraction over just that subset
           b. Other vars remain free (captured from context)
        3. Return all valid factorizations

    NOTE: Subsets that result in invalid programs are skipped.
    """
    free_vars = subtree.free_indices()

    if not free_vars:
        # No free variables - only one way to abstract
        return [Factorization(
            invention=Invented(subtree),
            n_args=0,
            abstracted_vars=set(),
            inlined_vars=set()
        )]

    if len(free_vars) > max_args:
        # Too many free vars - only consider full abstraction
        inv, n = abstract_subtree(subtree, free_vars)
        return [Factorization(
            invention=inv,
            n_args=n,
            abstracted_vars=free_vars,
            inlined_vars=set()
        )]

    factorizations = []

    # Consider all non-empty subsets of free variables
    for r in range(len(free_vars), 0, -1):  # From all vars down to 1
        for subset in combinations(sorted(free_vars), r):
            subset_set = set(subset)
            inlined = free_vars - subset_set

            # Create abstraction over just this subset
            inv = abstract_subtree_partial(subtree, subset_set, inlined)

            if inv is not None:
                factorizations.append(Factorization(
                    invention=inv,
                    n_args=len(subset_set),
                    abstracted_vars=subset_set,
                    inlined_vars=inlined
                ))

    return factorizations


def abstract_subtree_partial(
    subtree: Program,
    vars_to_abstract: Set[int],
    vars_to_inline: Set[int]
) -> Optional[Invented]:
    """
    Abstract over only some free variables, leaving others free.

    Args:
        subtree: The subtree to abstract
        vars_to_abstract: Variables that become lambda parameters
        vars_to_inline: Variables that remain free (captured from context)

    Returns:
        Invented abstraction, or None if invalid

    EXAMPLE:
        subtree = (+ $0 $2)  with vars_to_abstract={0}, vars_to_inline={2}

        Step 1: Reindex abstracted vars to 0, 1, ...
                Leave inlined vars as-is (they'll be captured)
        Step 2: Wrap in lambdas for abstracted vars

        Result: λ. (+ $0 $3)  -- $0 is param, $3 is captured $2 shifted by 1

    SUBTLETY:
        When we add a lambda, all free references shift up by 1.
        We must account for this when leaving variables free.
    """
    if not vars_to_abstract:
        # Must abstract over at least one variable
        return None

    all_free = subtree.free_indices()

    if not vars_to_abstract.issubset(all_free):
        return None

    # Sort abstracted variables for consistent ordering
    vars_list = sorted(vars_to_abstract)
    n_args = len(vars_list)

    # Build index map:
    # - Abstracted vars map to 0, 1, 2, ... (new lambda params)
    # - Inlined vars shift up by n_args (pushed outward by new lambdas)
    index_map = {}
    for new_idx, old_idx in enumerate(vars_list):
        index_map[old_idx] = new_idx

    for old_idx in vars_to_inline:
        # This var remains free, but shifts up by n_args due to new lambdas
        index_map[old_idx] = old_idx + n_args

    # Rewrite with new indices
    rewritten = _reindex(subtree, index_map)

    # Wrap in n_args lambdas
    body = rewritten
    for _ in range(n_args):
        body = Abstraction(body)

    return Invented(body)


def best_factorization(
    subtree: Program,
    grammar: Grammar,
    programs: List[Program],
    request_type: Type,
    grammar_weight: float = 1.0,
    max_args: int = 4
) -> Optional[Tuple[Factorization, float, List[Program], Dict[str, Any]]]:
    """
    Find the best factorization of a subtree by MDL.

    Enumerates all valid factorizations of the subtree and evaluates
    each one's MDL impact. Returns the factorization with best MDL
    improvement (if any improves).

    Args:
        subtree: The subtree to factor
        grammar: Current grammar
        programs: Current programs
        request_type: Type of the programs
        grammar_weight: MDL grammar weight
        max_args: Maximum arity to consider

    Returns:
        (best_factorization, mdl_improvement, rewritten_programs, stats)
        or None if no factorization improves MDL

    USAGE:
        Instead of always using abstract_subtree() (full abstraction),
        use best_factorization() to find the optimal arity:

        result = best_factorization(subtree, grammar, programs, tp)
        if result:
            fact, improvement, rewritten, stats = result
            # Use fact.invention instead of abstract_subtree result
    """
    # Lazy import to avoid circular dependency
    from .mdl_scoring import evaluate_invention_mdl

    factorizations = enumerate_factorizations(subtree, max_args)

    if not factorizations:
        return None

    best = None
    best_improvement = 0.0
    best_rewritten = None
    best_stats = None

    for fact in factorizations:
        # Skip if invention is already in grammar
        if grammar.get_production(fact.invention) is not None:
            continue

        # Check type inference
        try:
            ctx = TypeContext()
            fact.invention.infer_type(ctx, [])
        except Exception:
            continue

        # For partial abstraction, we need to construct the right target
        # The target is the original subtree pattern
        target = subtree

        # Evaluate MDL change
        try:
            old_mdl, new_mdl, rewritten, stats = evaluate_invention_mdl(
                grammar, programs, fact.invention, target,
                fact.n_args, request_type, grammar_weight
            )
        except Exception:
            continue

        improvement = old_mdl - new_mdl

        if improvement > best_improvement:
            best = fact
            best_improvement = improvement
            best_rewritten = rewritten
            best_stats = stats

    if best is None:
        return None

    return best, best_improvement, best_rewritten, best_stats


def rank_factorizations_by_mdl(
    subtree: Program,
    grammar: Grammar,
    programs: List[Program],
    request_type: Type,
    grammar_weight: float = 1.0,
    max_args: int = 4
) -> List[Tuple[Factorization, float, Dict[str, Any]]]:
    """
    Rank all factorizations of a subtree by MDL improvement.

    Like best_factorization but returns all valid options ranked.
    Useful for debugging or exploring alternatives.

    Returns:
        List of (factorization, mdl_improvement, stats) sorted by improvement
    """
    # Lazy import to avoid circular dependency
    from .mdl_scoring import evaluate_invention_mdl

    factorizations = enumerate_factorizations(subtree, max_args)
    ranked = []

    for fact in factorizations:
        if grammar.get_production(fact.invention) is not None:
            continue

        try:
            ctx = TypeContext()
            fact.invention.infer_type(ctx, [])

            old_mdl, new_mdl, _, stats = evaluate_invention_mdl(
                grammar, programs, fact.invention, subtree,
                fact.n_args, request_type, grammar_weight
            )

            improvement = old_mdl - new_mdl
            ranked.append((fact, improvement, stats))

        except Exception:
            continue

    ranked.sort(key=lambda x: -x[1])  # Best improvement first
    return ranked
