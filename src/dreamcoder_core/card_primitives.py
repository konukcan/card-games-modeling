"""
Minimal Primitive Library for Card Game Rules

This module defines the TRULY ATOMIC primitives for card game rules.
The philosophy: start with basic operations, let DreamCoder discover
the compositions that form complex rules.

Hierarchy of abstraction (from most basic to composed):
1. Card accessors: get_suit, get_rank
2. Property mappings: map over lists
3. Comparisons: equality, ordering
4. List operations: first, last, length, index
5. Boolean combinators: and, or, not
6. Quantifiers: all, any, count
7. Higher-order: map, filter, fold

Key principle: Each primitive should be atomic - not decomposable
into simpler operations in our domain.
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.type_system import (
    Type, BaseType, Arrow, ListType, TypeVariable,
    BOOL, INT, CARD, SUIT, RANK, HAND, arrow,
    LIST_INT, LIST_BOOL
)
from dreamcoder_core.program import Primitive
from dreamcoder_core.grammar import Grammar, Production, uniform_grammar

# Import card domain types
from rules.cards import (
    Card, Hand, Suit, Rank, Color, AltColor1, AltColor2, Parity,
    RANK_VALUES, card_color, suit_to_altcolor1, suit_to_altcolor2, rank_parity
)


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

# We use our type system
# CARD, SUIT, RANK already defined
COLOR = BaseType('color')
LIST_SUIT = ListType(SUIT)
LIST_RANK = ListType(RANK)
LIST_COLOR = ListType(COLOR)


# ============================================================================
# LEVEL 0: CONSTANTS
# ============================================================================

def make_constants() -> List[Primitive]:
    """Constants for suits, special ranks, etc."""
    prims = []

    # Suit constants
    prims.append(Primitive('CLUBS', SUIT, Suit.CLUBS))
    prims.append(Primitive('DIAMONDS', SUIT, Suit.DIAMONDS))
    prims.append(Primitive('HEARTS', SUIT, Suit.HEARTS))
    prims.append(Primitive('SPADES', SUIT, Suit.SPADES))

    # Color constants
    prims.append(Primitive('RED', COLOR, Color.RED))
    prims.append(Primitive('BLACK', COLOR, Color.BLACK))

    # Small integer constants (for counting, indexing)
    for i in range(7):  # 0 through 6 (max hand size)
        prims.append(Primitive(str(i), INT, i))

    # Boolean constants
    prims.append(Primitive('true', BOOL, True))
    prims.append(Primitive('false', BOOL, False))

    return prims


# ============================================================================
# LEVEL 1: CARD ACCESSORS (Card -> Property)
# ============================================================================

def make_card_accessors() -> List[Primitive]:
    """Atomic operations to extract properties from a card."""
    prims = []

    # Get suit from card
    prims.append(Primitive(
        'get_suit',
        arrow(CARD, SUIT),
        lambda c: c.suit
    ))

    # Get rank from card
    prims.append(Primitive(
        'get_rank',
        arrow(CARD, RANK),
        lambda c: c.rank
    ))

    # Get rank value (numeric)
    prims.append(Primitive(
        'rank_val',
        arrow(CARD, INT),
        lambda c: RANK_VALUES[c.rank]
    ))

    # Get color from card
    prims.append(Primitive(
        'get_color',
        arrow(CARD, COLOR),
        lambda c: card_color(c)
    ))

    return prims


# ============================================================================
# LEVEL 2: LIST OPERATIONS (List a -> something)
# ============================================================================

def make_list_ops() -> List[Primitive]:
    """Basic list operations."""
    prims = []

    # Type variable for polymorphism
    a = TypeVariable(0)

    # First element
    prims.append(Primitive(
        'first',
        arrow(ListType(a), a),
        lambda xs: xs[0] if xs else None
    ))

    # Last element
    prims.append(Primitive(
        'last',
        arrow(ListType(a), a),
        lambda xs: xs[-1] if xs else None
    ))

    # Length
    prims.append(Primitive(
        'length',
        arrow(ListType(a), INT),
        lambda xs: len(xs)
    ))

    # Index (get element at position)
    prims.append(Primitive(
        'at',
        arrow(ListType(a), INT, a),
        lambda xs: lambda i: xs[i] if 0 <= i < len(xs) else None
    ))

    # Reverse
    prims.append(Primitive(
        'reverse',
        arrow(ListType(a), ListType(a)),
        lambda xs: list(reversed(xs))
    ))

    # Take first n
    prims.append(Primitive(
        'take',
        arrow(INT, ListType(a), ListType(a)),
        lambda n: lambda xs: xs[:n]
    ))

    # Drop first n
    prims.append(Primitive(
        'drop',
        arrow(INT, ListType(a), ListType(a)),
        lambda n: lambda xs: xs[n:]
    ))

    # Left half
    prims.append(Primitive(
        'left_half',
        arrow(ListType(a), ListType(a)),
        lambda xs: xs[:len(xs)//2]
    ))

    # Right half
    prims.append(Primitive(
        'right_half',
        arrow(ListType(a), ListType(a)),
        lambda xs: xs[len(xs)//2:]
    ))

    # Is empty
    prims.append(Primitive(
        'empty?',
        arrow(ListType(a), BOOL),
        lambda xs: len(xs) == 0
    ))

    return prims


# ============================================================================
# LEVEL 3: COMPARISONS
# ============================================================================

def make_comparisons() -> List[Primitive]:
    """Equality and ordering comparisons."""
    prims = []

    a = TypeVariable(0)

    # Equality (polymorphic)
    prims.append(Primitive(
        'eq',
        arrow(a, a, BOOL),
        lambda x: lambda y: x == y
    ))

    # Not equal
    prims.append(Primitive(
        'neq',
        arrow(a, a, BOOL),
        lambda x: lambda y: x != y
    ))

    # Integer comparisons
    prims.append(Primitive(
        'lt',
        arrow(INT, INT, BOOL),
        lambda x: lambda y: x < y
    ))

    prims.append(Primitive(
        'le',
        arrow(INT, INT, BOOL),
        lambda x: lambda y: x <= y
    ))

    prims.append(Primitive(
        'gt',
        arrow(INT, INT, BOOL),
        lambda x: lambda y: x > y
    ))

    prims.append(Primitive(
        'ge',
        arrow(INT, INT, BOOL),
        lambda x: lambda y: x >= y
    ))

    return prims


# ============================================================================
# LEVEL 4: BOOLEAN COMBINATORS
# ============================================================================

def make_boolean_ops() -> List[Primitive]:
    """Boolean operations."""
    prims = []

    prims.append(Primitive(
        'and',
        arrow(BOOL, BOOL, BOOL),
        lambda x: lambda y: x and y
    ))

    prims.append(Primitive(
        'or',
        arrow(BOOL, BOOL, BOOL),
        lambda x: lambda y: x or y
    ))

    prims.append(Primitive(
        'not',
        arrow(BOOL, BOOL),
        lambda x: not x
    ))

    # If-then-else
    a = TypeVariable(0)
    prims.append(Primitive(
        'if',
        arrow(BOOL, a, a, a),
        lambda cond: lambda then_val: lambda else_val: then_val if cond else else_val
    ))

    return prims


# ============================================================================
# LEVEL 5: HIGHER-ORDER FUNCTIONS
# ============================================================================

def make_higher_order() -> List[Primitive]:
    """Map, filter, fold, and quantifiers."""
    prims = []

    a = TypeVariable(0)
    b = TypeVariable(1)

    # Map
    prims.append(Primitive(
        'map',
        arrow(arrow(a, b), ListType(a), ListType(b)),
        lambda f: lambda xs: [f(x) for x in xs]
    ))

    # Filter
    prims.append(Primitive(
        'filter',
        arrow(arrow(a, BOOL), ListType(a), ListType(a)),
        lambda pred: lambda xs: [x for x in xs if pred(x)]
    ))

    # Fold (reduce) from left
    prims.append(Primitive(
        'fold',
        arrow(arrow(b, a, b), b, ListType(a), b),
        lambda f: lambda init: lambda xs: _fold_left(f, init, xs)
    ))

    # All (forall)
    prims.append(Primitive(
        'all',
        arrow(arrow(a, BOOL), ListType(a), BOOL),
        lambda pred: lambda xs: all(pred(x) for x in xs)
    ))

    # Any (exists)
    prims.append(Primitive(
        'any',
        arrow(arrow(a, BOOL), ListType(a), BOOL),
        lambda pred: lambda xs: any(pred(x) for x in xs)
    ))

    # Count
    prims.append(Primitive(
        'count',
        arrow(arrow(a, BOOL), ListType(a), INT),
        lambda pred: lambda xs: sum(1 for x in xs if pred(x))
    ))

    # Unique (deduplicate)
    prims.append(Primitive(
        'unique',
        arrow(ListType(a), ListType(a)),
        lambda xs: list(dict.fromkeys(xs))  # Preserves order
    ))

    # Unique count
    prims.append(Primitive(
        'unique_count',
        arrow(ListType(a), INT),
        lambda xs: len(set(xs))
    ))

    return prims


def _fold_left(f, init, xs):
    """Helper for fold."""
    result = init
    for x in xs:
        result = f(result)(x)
    return result


# ============================================================================
# LEVEL 6: ARITHMETIC
# ============================================================================

def make_arithmetic() -> List[Primitive]:
    """Basic arithmetic for counting and scoring."""
    prims = []

    prims.append(Primitive('+', arrow(INT, INT, INT), lambda x: lambda y: x + y))
    prims.append(Primitive('-', arrow(INT, INT, INT), lambda x: lambda y: x - y))
    prims.append(Primitive('*', arrow(INT, INT, INT), lambda x: lambda y: x * y))
    prims.append(Primitive('/', arrow(INT, INT, INT), lambda x: lambda y: x // y if y != 0 else 0))
    prims.append(Primitive('mod', arrow(INT, INT, INT), lambda x: lambda y: x % y if y != 0 else 0))
    prims.append(Primitive('abs', arrow(INT, INT), lambda x: abs(x)))

    # Sum a list
    prims.append(Primitive(
        'sum',
        arrow(ListType(INT), INT),
        lambda xs: sum(xs)
    ))

    # Max/min
    prims.append(Primitive(
        'max',
        arrow(INT, INT, INT),
        lambda x: lambda y: max(x, y)
    ))

    prims.append(Primitive(
        'min',
        arrow(INT, INT, INT),
        lambda x: lambda y: min(x, y)
    ))

    return prims


# ============================================================================
# LEVEL 7: LIST COMPARISONS (for sequences)
# ============================================================================

def make_sequence_ops() -> List[Primitive]:
    """Operations on sequences (palindrome, sorted, etc.)."""
    prims = []

    a = TypeVariable(0)
    b = TypeVariable(1)
    c = TypeVariable(2)

    # Is sorted (ascending)
    prims.append(Primitive(
        'sorted?',
        arrow(ListType(INT), BOOL),
        lambda xs: all(xs[i] <= xs[i+1] for i in range(len(xs)-1)) if len(xs) > 1 else True
    ))

    # Is palindrome
    prims.append(Primitive(
        'palindrome?',
        arrow(ListType(a), BOOL),
        lambda xs: xs == list(reversed(xs))
    ))

    # Lists equal
    prims.append(Primitive(
        'lists_eq',
        arrow(ListType(a), ListType(a), BOOL),
        lambda xs: lambda ys: xs == ys
    ))

    # Zip (pair up elements)
    prims.append(Primitive(
        'zip_with',
        arrow(arrow(a, b, c), ListType(a), ListType(b), ListType(c)),
        lambda f: lambda xs: lambda ys: [f(x)(y) for x, y in zip(xs, ys)]
    ))

    # Adjacent pairs
    prims.append(Primitive(
        'pairs',
        arrow(ListType(a), ListType(ListType(a))),
        lambda xs: [[xs[i], xs[i+1]] for i in range(len(xs)-1)] if len(xs) > 1 else []
    ))

    return prims


# ============================================================================
# BUILD COMPLETE PRIMITIVE LIBRARY
# ============================================================================

def build_minimal_primitives() -> List[Primitive]:
    """
    Build the minimal primitive library.

    These are truly atomic operations that cannot be decomposed further.
    DreamCoder should learn to compose these into more complex rules.
    """
    prims = []

    prims.extend(make_constants())
    prims.extend(make_card_accessors())
    prims.extend(make_list_ops())
    prims.extend(make_comparisons())
    prims.extend(make_boolean_ops())
    prims.extend(make_higher_order())
    prims.extend(make_arithmetic())
    prims.extend(make_sequence_ops())

    return prims


def build_card_grammar() -> Grammar:
    """Build the starting grammar for card game rule learning."""
    prims = build_minimal_primitives()
    return uniform_grammar(prims)


# ============================================================================
# PRIMITIVE SUMMARY
# ============================================================================

def print_primitive_summary():
    """Print a summary of all primitives."""
    prims = build_minimal_primitives()

    print("=" * 60)
    print("MINIMAL PRIMITIVE LIBRARY FOR CARD GAMES")
    print("=" * 60)
    print(f"\nTotal primitives: {len(prims)}\n")

    # Group by category
    categories = [
        ("Constants", make_constants()),
        ("Card Accessors", make_card_accessors()),
        ("List Operations", make_list_ops()),
        ("Comparisons", make_comparisons()),
        ("Boolean Ops", make_boolean_ops()),
        ("Higher-Order", make_higher_order()),
        ("Arithmetic", make_arithmetic()),
        ("Sequence Ops", make_sequence_ops()),
    ]

    for name, cat_prims in categories:
        print(f"\n{name} ({len(cat_prims)}):")
        for p in cat_prims:
            print(f"  {p.name}: {p.tp}")


# ============================================================================
# EXAMPLE: How rules should be composed
# ============================================================================

def example_rule_compositions():
    """
    Show how complex rules can be composed from primitives.

    This is what DreamCoder should LEARN to do.
    """
    print("\n" + "=" * 60)
    print("EXAMPLE RULE COMPOSITIONS")
    print("(What DreamCoder should learn to build)")
    print("=" * 60)

    examples = [
        ("Uniform_color",
         "all (λc. eq (get_color c) (get_color (first h))) h",
         "All cards have same color as the first card"),

        ("Sorted_by_rank",
         "sorted? (map rank_val h)",
         "Ranks are in non-decreasing order"),

        ("Ends_same_suit",
         "eq (get_suit (first h)) (get_suit (last h))",
         "First and last cards have same suit"),

        ("Has_pair_ranks",
         "lt (unique_count (map get_rank h)) (length h)",
         "Number of unique ranks < total cards"),

        ("Suits_palindrome",
         "palindrome? (map get_suit h)",
         "Suit sequence is palindrome"),

        ("Halves_copy_suits",
         "lists_eq (map get_suit (left_half h)) (map get_suit (right_half h))",
         "Left and right halves have same suit sequence"),

        ("Uniform_rank_parity",
         "le (unique_count (map (λc. mod (rank_val c) 2) h)) 1",
         "All ranks have same parity (all even or all odd)"),
    ]

    for rule_name, composition, description in examples:
        print(f"\n{rule_name}:")
        print(f"  Composition: {composition}")
        print(f"  Description: {description}")


if __name__ == "__main__":
    print_primitive_summary()
    example_rule_compositions()
