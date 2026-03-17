"""
Grammar factory: build any of the 7 grammar families × 3 cost structures.

This module creates Grammar objects (from src/dreamcoder_core/grammar.py)
for the grammar-comparison experiment. Each grammar variant selects a
different subset of primitives and optionally adds new ones. Each cost
structure assigns different log-probabilities to the productions.

The 7 grammars:
  G1 base               - All 64 primitives from src/dreamcoder_core/primitives.py
  G2 swap-positional     - Replace take/drop/halves with slice + shifted_match
  G3 swap-distributional - Replace count_suit/color with count_where + sorted_counts
  G4 swap-both           - G2 + G3 combined
  G5 add-both            - Base + all 5 new primitives, nothing removed
  G6 redundant           - Base + 8 cognitive shortcut primitives
  G7 minimal             - A small hand-picked subset of core primitives

The 3 cost structures:
  UNIFORM  - Equal probability per return type
  TIERED   - Tier 1/2/3 weighting (9/3/1), normalized per return type
  LOTLIB3  - Integer decay (10/n²) + terminal 5× boost, normalized per return type
"""

import math
import sys
from collections import Counter, defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Allow imports from the main src/ tree
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from dreamcoder_core.grammar import Grammar, Production
from dreamcoder_core.program import Primitive
from dreamcoder_core.type_system import (
    BOOL, INT, CARD, SUIT, RANK, HAND,
    Arrow, BaseType, ListType, TypeVariable,
    arrow,
)
from dreamcoder_core.primitives import build_primitives
from rules.cards import (
    Card, Suit, Rank, RANK_VALUES, Color,
    card_color,
)

# Import the 5 new primitive implementations
from llm.grammar_comparison.primitives.definitions import (
    prim_slice,
    prim_shifted_match,
    prim_stride,
    prim_count_where,
    prim_sorted_counts,
)


# =============================================================================
# Cost structure enum
# =============================================================================

class CostStructure(Enum):
    """The three probability-assignment strategies."""
    UNIFORM = "uniform"
    TIERED = "tiered"
    LOTLIB3 = "lotlib3"


# =============================================================================
# Grammar names
# =============================================================================

GRAMMAR_NAMES = [
    "base",
    "swap-positional",
    "swap-distributional",
    "swap-both",
    "add-both",
    "redundant",
    "minimal",
]


# =============================================================================
# Tier assignments for TIERED cost structure
# =============================================================================

# Tier 1: High-frequency, cognitively basic primitives (weight 9)
TIER1_NAMES: Set[str] = {
    "get_suit", "get_color", "rank_val", "eq",
    "all", "any",
    "CLUBS", "DIAMONDS", "HEARTS", "SPADES",
    "RED", "BLACK",
    "0", "1", "2", "3",
}

# Tier 2: Common operations (weight 3)
TIER2_NAMES: Set[str] = {
    "map", "filter", "length", "head", "last",
    "count_where",
    "lt", "gt", "not", "and", "or",
    "+", "-",
    "4", "5",
}

# Everything else is Tier 3 (weight 1):
# at, zip_with, slice, shifted_match, sorted_counts, mod, unique, reverse,
# integers 6+, and any other primitive not listed above.


# =============================================================================
# New primitive constructors (Primitive objects for the DSL)
# =============================================================================

def _make_new_positional_primitives() -> List[Primitive]:
    """Create Primitive objects for slice and shifted_match.

    These wrap the plain-Python implementations from definitions.py
    into DreamCoder Primitive objects with proper types.

    Types:
      slice:          int -> int -> list('a) -> list('a)
      shifted_match:  int -> ('a -> 'a -> bool) -> list('a) -> bool
    """
    a = TypeVariable(0)

    prims = []

    # slice: int -> int -> list(a) -> list(a)
    # Curried: slice(i)(j)(xs)
    prims.append(Primitive(
        "slice",
        arrow(INT, INT, ListType(a), ListType(a)),
        lambda i: lambda j: lambda xs: prim_slice(i, j, xs),
    ))

    # shifted_match: int -> (a -> a -> bool) -> list(a) -> bool
    # Curried: shifted_match(k)(pred)(xs)
    prims.append(Primitive(
        "shifted_match",
        arrow(INT, arrow(a, a, BOOL), ListType(a), BOOL),
        lambda k: lambda pred: lambda xs: prim_shifted_match(k, pred, xs),
    ))

    return prims


def _make_new_distributional_primitives() -> List[Primitive]:
    """Create Primitive objects for count_where and sorted_counts.

    Types:
      count_where:    ('a -> bool) -> list('a) -> int
      sorted_counts:  ('a -> 'b) -> list('a) -> list(int)
    """
    a = TypeVariable(0)
    b = TypeVariable(1)

    prims = []

    # count_where: (a -> bool) -> list(a) -> int
    prims.append(Primitive(
        "count_where",
        arrow(arrow(a, BOOL), ListType(a), INT),
        lambda pred: lambda xs: prim_count_where(pred, xs),
    ))

    # sorted_counts: (a -> b) -> list(a) -> list(int)
    prims.append(Primitive(
        "sorted_counts",
        arrow(arrow(a, b), ListType(a), ListType(INT)),
        lambda key_fn: lambda xs: prim_sorted_counts(key_fn, xs),
    ))

    return prims


def _make_stride_primitive() -> Primitive:
    """Create Primitive object for stride.

    Type: int -> list('a) -> list('a)
    """
    a = TypeVariable(0)
    return Primitive(
        "stride",
        arrow(INT, ListType(a), ListType(a)),
        lambda k: lambda xs: prim_stride(k, xs),
    )


# =============================================================================
# Cognitive shortcut primitives for G6 (redundant)
# =============================================================================

def _make_cognitive_shortcuts() -> List[Primitive]:
    """Create the 8 cognitive shortcut primitives for the redundant grammar.

    These are higher-level operations that can be expressed as compositions
    of base primitives, but humans find them cognitively natural:
      - all_same:      Do all cards share the same value of a property?
      - all_different:  Are all values of a property distinct?
      - is_sorted:      Is the hand sorted by a numeric property?
      - exactly_n:      Does exactly N cards satisfy a predicate?
      - at_least_n:     Do at least N cards satisfy a predicate?
      - n_unique:       How many distinct values of a property?
      - is_run:         Are the rank values consecutive?
      - has_pair:       Do any two cards share the same value of a property?

    Types use CARD explicitly since these are card-game-specific shortcuts.
    """
    a = TypeVariable(0)
    prims = []

    # all_same: (card -> 'a) -> list(card) -> bool
    # "All cards have the same suit" = all_same get_suit hand
    prims.append(Primitive(
        "all_same",
        arrow(arrow(CARD, a), HAND, BOOL),
        lambda f: lambda hand: len(set(f(c) for c in hand)) <= 1 if hand else True,
    ))

    # all_different: (card -> 'a) -> list(card) -> bool
    # "All cards have different ranks" = all_different get_rank hand
    prims.append(Primitive(
        "all_different",
        arrow(arrow(CARD, a), HAND, BOOL),
        lambda f: lambda hand: len(set(f(c) for c in hand)) == len(hand) if hand else True,
    ))

    # is_sorted: (card -> int) -> list(card) -> bool
    # "Cards are in ascending rank order" = is_sorted rank_val hand
    prims.append(Primitive(
        "is_sorted",
        arrow(arrow(CARD, INT), HAND, BOOL),
        lambda f: lambda hand: all(
            f(hand[i]) <= f(hand[i + 1])
            for i in range(len(hand) - 1)
        ) if len(hand) > 1 else True,
    ))

    # exactly_n: int -> (card -> bool) -> list(card) -> bool
    # "Exactly 2 red cards" = exactly_n 2 is_red hand
    prims.append(Primitive(
        "exactly_n",
        arrow(INT, arrow(CARD, BOOL), HAND, BOOL),
        lambda n: lambda pred: lambda hand: sum(1 for c in hand if pred(c)) == n,
    ))

    # at_least_n: int -> (card -> bool) -> list(card) -> bool
    # "At least 3 hearts" = at_least_n 3 is_heart hand
    prims.append(Primitive(
        "at_least_n",
        arrow(INT, arrow(CARD, BOOL), HAND, BOOL),
        lambda n: lambda pred: lambda hand: sum(1 for c in hand if pred(c)) >= n,
    ))

    # n_unique: (card -> 'a) -> list(card) -> int
    # "How many different suits?" = n_unique get_suit hand
    prims.append(Primitive(
        "n_unique",
        arrow(arrow(CARD, a), HAND, INT),
        lambda f: lambda hand: len(set(f(c) for c in hand)),
    ))

    # is_run: list(card) -> bool
    # "The rank values form a consecutive sequence" (after sorting)
    prims.append(Primitive(
        "is_run",
        arrow(HAND, BOOL),
        lambda hand: _is_run(hand),
    ))

    # has_pair: (card -> 'a) -> list(card) -> bool
    # "Two cards share the same rank" = has_pair get_rank hand
    prims.append(Primitive(
        "has_pair",
        arrow(arrow(CARD, a), HAND, BOOL),
        lambda f: lambda hand: len(set(f(c) for c in hand)) < len(hand) if hand else False,
    ))

    return prims


def _is_run(hand: list) -> bool:
    """Check whether rank values form a consecutive sequence (after sorting).

    A run means, after sorting by rank value, each consecutive pair
    differs by exactly 1. E.g., [3, 4, 5, 6] is a run.
    """
    if len(hand) <= 1:
        return True
    vals = sorted(RANK_VALUES[c.rank] for c in hand)
    return all(vals[i + 1] - vals[i] == 1 for i in range(len(vals) - 1))


# =============================================================================
# Primitive selection logic for each grammar
# =============================================================================

# Names to REMOVE for G2 (swap-positional)
_POSITIONAL_REMOVE: Set[str] = {
    "take", "drop", "first_half", "second_half",
    "adjacent_pairs", "shifted_pairs",
}

# Names to REMOVE for G3 (swap-distributional)
_DISTRIBUTIONAL_REMOVE: Set[str] = {
    "count_suit", "count_rank", "count_color",
}

# Names to KEEP for G7 (minimal)
_MINIMAL_KEEP: Set[str] = {
    "head", "at", "map", "filter", "all", "any", "zip_with",
    "length", "unique", "reverse",
    "+", "-", "mod",
    "eq", "lt", "gt", "not", "and", "or",
    "0", "1", "2", "3", "4", "5",
    "get_suit", "get_rank", "rank_val", "get_color",
    "CLUBS", "DIAMONDS", "HEARTS", "SPADES",
    "RED", "BLACK",
}


def _select_primitives(name: str) -> List[Primitive]:
    """Return the list of Primitive objects for the given grammar name.

    This function:
    1. Starts from the base primitives (build_primitives from src/)
    2. Adds/removes primitives according to the grammar variant
    3. Returns the final list

    Args:
        name: One of GRAMMAR_NAMES.

    Returns:
        List of Primitive objects for this grammar variant.

    Raises:
        ValueError: If name is not a recognized grammar.
    """
    if name not in GRAMMAR_NAMES:
        raise ValueError(
            f"Unknown grammar '{name}'. Choose from: {GRAMMAR_NAMES}"
        )

    base = build_primitives()

    if name == "base":
        return base

    elif name == "swap-positional":
        # Remove positional primitives, add slice + shifted_match
        filtered = [p for p in base if p.name not in _POSITIONAL_REMOVE]
        filtered.extend(_make_new_positional_primitives())
        return filtered

    elif name == "swap-distributional":
        # Remove distributional primitives, add count_where + sorted_counts
        filtered = [p for p in base if p.name not in _DISTRIBUTIONAL_REMOVE]
        filtered.extend(_make_new_distributional_primitives())
        return filtered

    elif name == "swap-both":
        # Combine both swaps
        remove = _POSITIONAL_REMOVE | _DISTRIBUTIONAL_REMOVE
        filtered = [p for p in base if p.name not in remove]
        filtered.extend(_make_new_positional_primitives())
        filtered.extend(_make_new_distributional_primitives())
        return filtered

    elif name == "add-both":
        # Base + all 5 new primitives, nothing removed
        extended = list(base)
        extended.extend(_make_new_positional_primitives())
        extended.extend(_make_new_distributional_primitives())
        extended.append(_make_stride_primitive())
        return extended

    elif name == "redundant":
        # Base + 8 cognitive shortcuts
        extended = list(base)
        extended.extend(_make_cognitive_shortcuts())
        return extended

    elif name == "minimal":
        # Only keep the hand-picked subset
        return [p for p in base if p.name in _MINIMAL_KEEP]

    # Should be unreachable due to the check above
    raise ValueError(f"Unhandled grammar name: {name}")  # pragma: no cover


# =============================================================================
# Cost structure assignment
# =============================================================================

def _get_return_type_key(tp) -> str:
    """Compute a string key for the return type of a primitive's type.

    This is used to group primitives by return type for per-type
    normalization. We follow the approach of canonical_type to get
    a consistent key.

    For arrow types, the return type is the final (rightmost) non-arrow type.
    For non-arrow types, the type itself is the return type.
    """
    return str(tp.returns)


def _assign_uniform(primitives: List[Primitive]) -> List[Production]:
    """Assign uniform log-probabilities (all primitives equal).

    Matches the existing uniform_grammar() function: every primitive
    (and variable) gets log(1 / (N + 1)) where N = number of primitives.
    """
    n = len(primitives) + 1  # +1 for variable
    log_p = -math.log(n)
    return [Production(p, p.tp, log_p) for p in primitives]


def _get_tier(name: str) -> int:
    """Return the tier (1, 2, or 3) for a primitive by name."""
    if name in TIER1_NAMES:
        return 1
    if name in TIER2_NAMES:
        return 2
    return 3


def _assign_tiered(primitives: List[Primitive]) -> List[Production]:
    """Assign TIERED log-probabilities.

    Within each return type group:
      - Tier 1 primitives get weight 9
      - Tier 2 primitives get weight 3
      - Tier 3 primitives get weight 1
    Weights are normalized per return-type group.

    The variable gets a fixed log-probability of log(1 / (N + 1))
    to keep it comparable across grammars.
    """
    # Tier weights
    tier_weight = {1: 9.0, 2: 3.0, 3: 1.0}

    # Group primitives by return type
    groups: Dict[str, List[Tuple[Primitive, float]]] = defaultdict(list)
    for p in primitives:
        tier = _get_tier(p.name)
        w = tier_weight[tier]
        key = _get_return_type_key(p.tp)
        groups[key].append((p, w))

    # Normalize within each group and create productions
    productions = []
    for key, members in groups.items():
        total_w = sum(w for _, w in members)
        for prim, w in members:
            log_p = math.log(w / total_w)
            productions.append(Production(prim, prim.tp, log_p))

    return productions


def _assign_lotlib3(primitives: List[Primitive]) -> List[Production]:
    """Assign LOTLIB3-style log-probabilities.

    Rules:
    1. Integer constants get weight = 10 / n² where n = the integer value.
       Special case: 0 gets weight 10 (treated as n=1).
    2. All terminal primitives (return type has no Arrow) get a 5× weight
       multiplier.
    3. Everything else gets base weight 1.0.
    4. Normalize per return type.
    """
    # Compute raw weights
    weighted: List[Tuple[Primitive, float]] = []
    for p in primitives:
        w = _lotlib3_weight(p)
        weighted.append((p, w))

    # Group by return type
    groups: Dict[str, List[Tuple[Primitive, float]]] = defaultdict(list)
    for prim, w in weighted:
        key = _get_return_type_key(prim.tp)
        groups[key].append((prim, w))

    # Normalize within each group
    productions = []
    for key, members in groups.items():
        total_w = sum(w for _, w in members)
        for prim, w in members:
            log_p = math.log(w / total_w)
            productions.append(Production(prim, prim.tp, log_p))

    return productions


def _lotlib3_weight(p: Primitive) -> float:
    """Compute the raw LOTLIB3 weight for a single primitive.

    Integer constants: 10 / n² (n = value, 0 treated as 1).
    Terminal primitives (no Arrow in return type): 5× multiplier.
    All others: base weight 1.0.
    """
    base_w = 1.0

    # Check if this is an integer constant (name is a digit string)
    if p.name.isdigit():
        n = int(p.name)
        if n == 0:
            base_w = 10.0  # 0 treated as n=1 -> 10/1
        else:
            base_w = 10.0 / (n * n)

    # Terminal multiplier: if the type has no arrow, it's a terminal
    # (constants, suit values, etc.)
    is_terminal = not isinstance(p.tp, Arrow)
    if is_terminal:
        base_w *= 5.0

    return base_w


# =============================================================================
# Main factory function
# =============================================================================

def build_grammar(name: str, cost: CostStructure) -> Grammar:
    """Build a Grammar object for the given grammar variant and cost structure.

    This is the main entry point for the grammar-comparison experiment.
    It selects the appropriate set of primitives based on the grammar name,
    then assigns log-probabilities according to the cost structure.

    Args:
        name: One of GRAMMAR_NAMES (e.g., "base", "swap-positional").
        cost: The cost structure to use for probability assignment.

    Returns:
        A Grammar object with typed, weighted productions.

    Raises:
        ValueError: If name is not a recognized grammar.

    Example:
        >>> g = build_grammar("base", CostStructure.UNIFORM)
        >>> len(g.productions)
        64
    """
    primitives = _select_primitives(name)

    if cost == CostStructure.UNIFORM:
        productions = _assign_uniform(primitives)
    elif cost == CostStructure.TIERED:
        productions = _assign_tiered(primitives)
    elif cost == CostStructure.LOTLIB3:
        productions = _assign_lotlib3(primitives)
    else:
        raise ValueError(f"Unknown cost structure: {cost}")

    # log_variable: use the same value as uniform for consistency
    n = len(primitives) + 1
    log_variable = -math.log(n)

    return Grammar(productions, log_variable=log_variable)
