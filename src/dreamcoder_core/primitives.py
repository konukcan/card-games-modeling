#!/usr/bin/env python3
"""
Cognitive Primitive Library for Card Game Learning (v5)

Current count: 64 primitives

Philosophy:
This library is designed for COGNITIVE REALISM - it contains primitives that
reflect how humans actually think and talk about card games.

Key design principles:
1. Primitives should be "directly nameable" - expressible in short phrases
2. Include domain-specific operations that humans use naturally
3. Remove abstract combinators that have low cognitive reality
4. Use only small numeric constants (0-5) for counting, not rank thresholds
5. Keep the grammar size reasonable to maintain search tractability

Version History:
---------------
v1 (Initial):
  - Abstract combinator style following original DreamCoder
  - Included compose, flip, const, id, cons, nil, fst, snd

v2 (Cognitive Refocus):
  REMOVED: compose, flip, const, id (abstract combinators - low cognitive reality)
  REMOVED: cons, nil (list construction - not how we think about hands)
  REMOVED: fst, snd, pairs (pair operations - rarely needed)
  REMOVED: Rank constants 10-14 (face card values - too specific)
  REMOVED: Game thresholds 17, 21 (blackjack rules - too specific)
  ADDED: has_suit, has_color (direct membership queries)
  ADDED: count_suit, count_color (direct counting)
  ADDED: n_unique_suits, n_unique_ranks, n_unique_colors (diversity)
  ADDED: all_same_suit, all_same_color (gestalt perception)
  ADDED: sum_ranks, max_rank, min_rank (aggregates)

v3 (List Operations):
  ADDED: take, drop (list slicing - needed for halves operations)
  ADDED: zip_with (parallel comparison - needed for palindromes)
  ADDED: first_half, second_half, half_len (direct halves access)
  ADDED: adjacent_pairs (for sorted checks)
  REMOVED: neq (not equal) - use 'not (eq x y)' instead

v4 (Redundancy Removal):
  REMOVED: all_same_suit, all_same_color
    Reason: Redundant with (lt (n_unique_suits hand) 2)
    See experiments/run_targeted_ablation_study.py for empirical validation

v5 (Current - Gallery Analysis Extensions):
  ADDED: sort_by_rank (sorting cards by rank for straight/AP detection)
  ADDED: max_suit_count (most frequent suit count for "any suit ≥ N" rules)
  ADDED: n_repeated_ranks, n_repeated_suits (duplicate detection)
  ADDED: running_sum (constrained sequential accumulation for bracket rules)
  ADDED: suit_to_int (conventional suit ordering for monotonicity rules)
  ADDED: signum (sign function for alternation detection)
  See docs/PRIMITIVE_DESIGN_DECISIONS.md for full justification and bias analysis.
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

COLOR = BaseType('color')
LIST_SUIT = ListType(SUIT)
LIST_RANK = ListType(RANK)
LIST_COLOR = ListType(COLOR)


# ============================================================================
# LEVEL 0: CONSTANTS
# ============================================================================

def make_constants() -> List[Primitive]:
    """
    Constants for card game learning.

    Numeric constants include:
    - 0-5: Basic counting (pairs, trips, hand length, etc.)

    Note: Rank-specific constants (10-14 for face cards, 17/21 for blackjack)
    were removed to keep the grammar generalizable. Rules should use relative
    comparisons (gt, lt, eq) rather than absolute rank thresholds.
    """
    prims = []

    # Suit constants
    prims.append(Primitive('CLUBS', SUIT, Suit.CLUBS))
    prims.append(Primitive('DIAMONDS', SUIT, Suit.DIAMONDS))
    prims.append(Primitive('HEARTS', SUIT, Suit.HEARTS))
    prims.append(Primitive('SPADES', SUIT, Suit.SPADES))

    # Color constants
    prims.append(Primitive('RED', COLOR, Color.RED))
    prims.append(Primitive('BLACK', COLOR, Color.BLACK))

    # Basic counting constants (0-5)
    for i in range(6):
        prims.append(Primitive(str(i), INT, i))

    # Boolean constants
    prims.append(Primitive('true', BOOL, True))
    prims.append(Primitive('false', BOOL, False))

    return prims


# ============================================================================
# LEVEL 1: CARD ACCESSORS
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
# LEVEL 2: POSITION ACCESS
# ============================================================================

def make_position_ops() -> List[Primitive]:
    """Access elements by position - cognitively natural."""
    prims = []

    a = TypeVariable(0)

    # First element (head)
    prims.append(Primitive(
        'head',
        arrow(ListType(a), a),
        lambda xs: xs[0] if xs else None
    ))

    # Last element
    prims.append(Primitive(
        'last',
        arrow(ListType(a), a),
        lambda xs: xs[-1] if xs else None
    ))

    # Index access
    prims.append(Primitive(
        'at',
        arrow(ListType(a), INT, a),
        lambda xs: lambda i: xs[i] if 0 <= i < len(xs) else None
    ))

    # Length
    prims.append(Primitive(
        'length',
        arrow(ListType(a), INT),
        lambda xs: len(xs)
    ))

    # Reverse
    prims.append(Primitive(
        'reverse',
        arrow(ListType(a), ListType(a)),
        lambda xs: list(reversed(xs))
    ))

    return prims


# ============================================================================
# LEVEL 2b: LIST SLICING (Essential for positional rules)
# ============================================================================

def make_list_slicing() -> List[Primitive]:
    """
    List slicing primitives for positional operations.

    These enable rules that compare halves, check palindromes, etc.
    Essential for rules like:
    - Halves_copy_suits: take 3 vs drop 3
    - Suits_palindrome: zip_with eq xs (reverse xs)
    - Sorted_by_rank: adjacent pairs via zip_with
    """
    prims = []

    a = TypeVariable(0)
    b = TypeVariable(1)

    # Take first n elements - "first half" (take 3 [1,2,3,4,5,6] → [1,2,3])
    prims.append(Primitive(
        'take',
        arrow(INT, ListType(a), ListType(a)),
        lambda n: lambda xs: xs[:n] if n >= 0 else []
    ))

    # Drop first n elements - "second half" (drop 3 [1,2,3,4,5,6] → [4,5,6])
    prims.append(Primitive(
        'drop',
        arrow(INT, ListType(a), ListType(a)),
        lambda n: lambda xs: xs[n:] if n >= 0 else xs
    ))

    # Zip with function - combine two lists element-wise
    # Essential for: palindrome checks, halves comparison, sorted checks
    # (zip_with eq suits (reverse suits)) → all same forward/backward
    prims.append(Primitive(
        'zip_with',
        arrow(arrow(a, b, BOOL), ListType(a), ListType(b), LIST_BOOL),
        lambda f: lambda xs: lambda ys: [f(x)(y) for x, y in zip(xs, ys)]
    ))

    # Adjacent pairs - for checking sorted, adjacent constraints
    # Returns list of (prev, curr) pairs: [1,2,3] → [(1,2), (2,3)]
    # We represent pairs as 2-element lists for simplicity
    prims.append(Primitive(
        'adjacent_pairs',
        arrow(ListType(a), ListType(ListType(a))),
        lambda xs: [[xs[i], xs[i+1]] for i in range(len(xs)-1)] if len(xs) > 1 else []
    ))

    # Half length - divides list length by 2 (for halves operations)
    prims.append(Primitive(
        'half_len',
        arrow(ListType(a), INT),
        lambda xs: len(xs) // 2
    ))

    # First half - direct primitive (more cognitively natural than "take (half_len xs) xs")
    # For 6-card hand: first_half → [card0, card1, card2]
    prims.append(Primitive(
        'first_half',
        arrow(ListType(a), ListType(a)),
        lambda xs: xs[:len(xs) // 2]
    ))

    # Second half - direct primitive (more cognitively natural than "drop (half_len xs) xs")
    # For 6-card hand: second_half → [card3, card4, card5]
    prims.append(Primitive(
        'second_half',
        arrow(ListType(a), ListType(a)),
        lambda xs: xs[len(xs) // 2:]
    ))

    return prims


# ============================================================================
# LEVEL 3: DIRECT PROPERTY QUERIES (NEW - High cognitive reality)
# ============================================================================

def make_direct_queries() -> List[Primitive]:
    """
    Direct queries about hand properties.

    These are cognitively natural: "Does the hand have a spade?"
    Much more natural than: "any (lambda card. eq (get_suit card) SPADES) hand"
    """
    prims = []

    # Has suit? - "Does the hand have a spade?"
    prims.append(Primitive(
        'has_suit',
        arrow(HAND, SUIT, BOOL),
        lambda hand: lambda suit: any(c.suit == suit for c in hand)
    ))

    # Has color? - "Does the hand have a red card?"
    prims.append(Primitive(
        'has_color',
        arrow(HAND, COLOR, BOOL),
        lambda hand: lambda color: any(card_color(c) == color for c in hand)
    ))

    # Count suit - "How many hearts?"
    prims.append(Primitive(
        'count_suit',
        arrow(HAND, SUIT, INT),
        lambda hand: lambda suit: sum(1 for c in hand if c.suit == suit)
    ))

    # Count color - "How many red cards?"
    prims.append(Primitive(
        'count_color',
        arrow(HAND, COLOR, INT),
        lambda hand: lambda color: sum(1 for c in hand if card_color(c) == color)
    ))

    # NOTE: all_same_suit and all_same_color were removed in v4
    # They are redundant with: (lt (n_unique_suits hand) 2) and (lt (n_unique_colors hand) 2)
    # See experiments/run_targeted_ablation_study.py for empirical validation

    # Number of unique suits - "How many suits represented?"
    prims.append(Primitive(
        'n_unique_suits',
        arrow(HAND, INT),
        lambda hand: len(set(c.suit for c in hand))
    ))

    # Number of unique ranks - "How many different values?"
    prims.append(Primitive(
        'n_unique_ranks',
        arrow(HAND, INT),
        lambda hand: len(set(c.rank for c in hand))
    ))

    # Number of unique colors - "How many colors?"
    prims.append(Primitive(
        'n_unique_colors',
        arrow(HAND, INT),
        lambda hand: len(set(card_color(c) for c in hand))
    ))

    return prims


# ============================================================================
# LEVEL 4: AGGREGATE OPERATIONS (NEW - High cognitive reality)
# ============================================================================

def make_aggregates() -> List[Primitive]:
    """
    Aggregate operations over hands.

    These are cognitively natural for card games:
    "What's the total?" "What's the highest card?"
    """
    prims = []

    # Sum of rank values - "What's the total?"
    prims.append(Primitive(
        'sum_ranks',
        arrow(HAND, INT),
        lambda hand: sum(RANK_VALUES[c.rank] for c in hand)
    ))

    # Maximum rank value - "What's the highest?"
    prims.append(Primitive(
        'max_rank',
        arrow(HAND, INT),
        lambda hand: max(RANK_VALUES[c.rank] for c in hand) if hand else 0
    ))

    # Minimum rank value - "What's the lowest?"
    prims.append(Primitive(
        'min_rank',
        arrow(HAND, INT),
        lambda hand: min(RANK_VALUES[c.rank] for c in hand) if hand else 0
    ))

    return prims


# ============================================================================
# LEVEL 5: COMPARISONS
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

    # Note: 'neq' removed in v3 - use 'not (eq x y)' instead

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
# LEVEL 6: BOOLEAN OPERATIONS
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
# LEVEL 7: HIGHER-ORDER FUNCTIONS (Reduced set)
# ============================================================================

def make_higher_order() -> List[Primitive]:
    """
    Higher-order functions - kept minimal.

    Removed: fold (complex), count (replaced by count_suit/count_color)
    Kept: map, filter, all, any, unique (all cognitively meaningful)
    """
    prims = []

    a = TypeVariable(0)
    b = TypeVariable(1)

    # Map - transform each element
    prims.append(Primitive(
        'map',
        arrow(arrow(a, b), ListType(a), ListType(b)),
        lambda f: lambda xs: [f(x) for x in xs]
    ))

    # Filter - keep matching elements
    prims.append(Primitive(
        'filter',
        arrow(arrow(a, BOOL), ListType(a), ListType(a)),
        lambda pred: lambda xs: [x for x in xs if pred(x)]
    ))

    # All - every element satisfies predicate
    prims.append(Primitive(
        'all',
        arrow(arrow(a, BOOL), ListType(a), BOOL),
        lambda pred: lambda xs: all(pred(x) for x in xs)
    ))

    # Any - some element satisfies predicate
    prims.append(Primitive(
        'any',
        arrow(arrow(a, BOOL), ListType(a), BOOL),
        lambda pred: lambda xs: any(pred(x) for x in xs)
    ))

    # Unique - deduplicate (cognitively: "the different values")
    prims.append(Primitive(
        'unique',
        arrow(ListType(a), ListType(a)),
        lambda xs: list(dict.fromkeys(xs))
    ))

    return prims


# ============================================================================
# LEVEL 8: ARITHMETIC (Minimal)
# ============================================================================

def make_arithmetic() -> List[Primitive]:
    """
    Basic arithmetic - reduced set.

    Removed: * and / (rarely needed for card rules)
    Kept: +, -, mod (for sums and parity checks)
    """
    prims = []

    prims.append(Primitive('+', arrow(INT, INT, INT), lambda x: lambda y: x + y))
    prims.append(Primitive('-', arrow(INT, INT, INT), lambda x: lambda y: x - y))
    prims.append(Primitive('mod', arrow(INT, INT, INT), lambda x: lambda y: x % y if y != 0 else 0))

    # Signum: returns -1, 0, or +1 for negative, zero, positive integers
    # Enables sign-based pattern detection (e.g., zigzag = alternating signs
    # of rank differences). See docs/PRIMITIVE_DESIGN_DECISIONS.md Decision 6.
    prims.append(Primitive(
        'signum',
        arrow(INT, INT),
        lambda x: (1 if x > 0 else (-1 if x < 0 else 0))
    ))

    return prims


# ============================================================================
# LEVEL 9: GALLERY ANALYSIS EXTENSIONS (v5)
# ============================================================================

def make_gallery_extensions() -> List[Primitive]:
    """
    Primitives added to ensure all 60 gallery rules are expressible.

    Added in v5 for the Bayesian rule induction analysis. Each primitive
    is justified in docs/PRIMITIVE_DESIGN_DECISIONS.md with full discussion
    of alternatives considered and biases introduced.

    These primitives fall into four categories:
    1. Sorting: sort_by_rank (unlocks straight/AP rules)
    2. Aggregate queries: max_suit_count, n_repeated_ranks/suits (unlocks
       counting rules that need "max over" or "count of counts")
    3. Sequential: running_sum (unlocks bracket-matching rules)
    4. Mapping: suit_to_int (unlocks suit-ordering rules)
    """
    prims = []

    # --- Sort by rank ---
    # Sorts cards by rank value (ascending).
    # Unlocks 6 rules: straight5, straight5_same_suit, straight5_same_color,
    # ap_len3_step1_anywhere, ap_step1_len3_adj, ap_step2_len4_adj.
    # See Decision 1 in PRIMITIVE_DESIGN_DECISIONS.md.
    prims.append(Primitive(
        'sort_by_rank',
        arrow(ListType(CARD), ListType(CARD)),
        lambda hand: sorted(hand, key=lambda c: RANK_VALUES[c.rank])
    ))

    # --- Max suit count ---
    # Returns the count of the most frequent suit.
    # Unlocks: three_or_more_same_suit, four_any_suit_anywhere.
    # See Decision 2 in PRIMITIVE_DESIGN_DECISIONS.md.
    prims.append(Primitive(
        'max_suit_count',
        arrow(ListType(CARD), INT),
        lambda hand: max((sum(1 for c in hand if c.suit == s) for s in Suit), default=0)
    ))

    # --- N repeated ranks ---
    # Returns the number of ranks that appear more than once.
    # Unlocks: two_pairs_ranks (via ge (n_repeated_ranks $0) 2).
    # See Decision 3 in PRIMITIVE_DESIGN_DECISIONS.md.
    prims.append(Primitive(
        'n_repeated_ranks',
        arrow(ListType(CARD), INT),
        lambda hand: sum(
            1 for r in set(c.rank for c in hand)
            if sum(1 for c in hand if c.rank == r) >= 2
        )
    ))

    # --- N repeated suits ---
    # Returns the number of suits that appear more than once.
    # Parallel to n_repeated_ranks for symmetry.
    # Helps with: two_pairs_suits (via suit-counting constraints).
    prims.append(Primitive(
        'n_repeated_suits',
        arrow(ListType(CARD), INT),
        lambda hand: sum(
            1 for s in Suit
            if sum(1 for c in hand if c.suit == s) >= 2
        )
    ))

    # --- Running sum ---
    # Computes cumulative sums of a card→int mapping.
    # running_sum(f, [c1,c2,c3]) = [f(c1), f(c1)+f(c2), f(c1)+f(c2)+f(c3)]
    # Unlocks bracket-matching rules via: map each suit to +1/-1, check
    # running sum never goes negative and ends at 0.
    # See Decision 4 in PRIMITIVE_DESIGN_DECISIONS.md.
    prims.append(Primitive(
        'running_sum',
        arrow(arrow(CARD, INT), ListType(CARD), LIST_INT),
        lambda f: lambda xs: _running_sum(f, xs)
    ))

    # --- Suit to int ---
    # Maps suits to conventional bridge ordering: ♠=4, ♥=3, ♦=2, ♣=1.
    # Unlocks: suits_nonincreasing (via map suit_to_int + monotonicity check).
    # See Decision 5 in PRIMITIVE_DESIGN_DECISIONS.md.
    _SUIT_TO_INT = {Suit.SPADES: 4, Suit.HEARTS: 3, Suit.DIAMONDS: 2, Suit.CLUBS: 1}
    prims.append(Primitive(
        'suit_to_int',
        arrow(SUIT, INT),
        lambda s: _SUIT_TO_INT.get(s, 0)
    ))

    return prims


def _running_sum(f, xs):
    """Helper for running_sum primitive."""
    result = []
    total = 0
    for x in xs:
        total += f(x)
        result.append(total)
    return result


# ============================================================================
# BUILD COGNITIVE PRIMITIVE LIBRARY
# ============================================================================

def build_primitives() -> List[Primitive]:
    """
    Build the cognitively realistic primitive library v5.

    Returns 64 primitives organized by cognitive naturalness:
    - Constants (14): Suits, colors, numbers 0-5, booleans
    - Card accessors (4): get_suit, get_rank, rank_val, get_color
    - Position access (5): head, last, at, length, reverse
    - List slicing (7): take, drop, zip_with, adjacent_pairs, half_len, first_half, second_half
    - Direct queries (7): has_suit/color, count_suit/color, n_unique_suits/ranks/colors
    - Aggregates (3): sum_ranks, max_rank, min_rank
    - Comparisons (5): eq, lt, le, gt, ge
    - Boolean (4): and, or, not, if
    - Higher-order (5): map, filter, all, any, unique
    - Arithmetic (4): +, -, mod, signum
    - Gallery extensions (6): sort_by_rank, max_suit_count, n_repeated_ranks/suits,
                              running_sum, suit_to_int
    """
    prims = []

    prims.extend(make_constants())           # 14: 4 suits + 2 colors + 6 numbers (0-5) + 2 bools
    prims.extend(make_card_accessors())      # 4: get_suit, get_rank, rank_val, get_color
    prims.extend(make_position_ops())        # 5: head, last, at, length, reverse
    prims.extend(make_list_slicing())        # 7: take, drop, zip_with, adjacent_pairs, half_len, first_half, second_half
    prims.extend(make_direct_queries())      # 7: has_suit/color, count_suit/color, n_unique_suits/ranks/colors
    prims.extend(make_aggregates())          # 3: sum_ranks, max_rank, min_rank
    prims.extend(make_comparisons())         # 5: eq, lt, le, gt, ge
    prims.extend(make_boolean_ops())         # 4: and, or, not, if
    prims.extend(make_higher_order())        # 5: map, filter, all, any, unique
    prims.extend(make_arithmetic())          # 4: +, -, mod, signum
    prims.extend(make_gallery_extensions())  # 6: sort_by_rank, max_suit_count, n_repeated_ranks/suits, running_sum, suit_to_int
    # TOTAL: 64 primitives

    return prims


def build_lean_grammar() -> Grammar:
    """Build the cognitive grammar for card game learning."""
    prims = build_primitives()
    return uniform_grammar(prims)


# ============================================================================
# EXPECTED PROGRAM DEPTHS WITH NEW PRIMITIVES
# ============================================================================

def show_expected_depths():
    """
    Show expected program depths for various rules with new primitives.

    This demonstrates the dramatic improvement over v1.
    """
    print("\n" + "=" * 70)
    print("EXPECTED PROGRAM DEPTHS (v2 vs v1)")
    print("=" * 70)

    rules = [
        ("simple_first_red",
         "(λ eq RED (get_color (head $0)))",
         "same", 4, 4),
        ("simple_last_black",
         "(λ eq BLACK (get_color (last $0)))",
         "same", 4, 4),
        ("simple_has_spade",
         "(λ has_suit $0 SPADES)",
         "v1: (λ any (λ eq SPADES (get_suit $0)) $0)", 3, 5),
        ("simple_has_heart",
         "(λ has_suit $0 HEARTS)",
         "v1: (λ any (λ eq HEARTS (get_suit $0)) $0)", 3, 5),
        ("count_two_red",
         "(λ eq 2 (count_color $0 RED))",
         "v1: (λ eq 2 (count (λ eq RED (get_color $0)) $0))", 3, 6),
        ("poker_flush",
         "(λ lt (n_unique_suits $0) 2)",
         "v3: (λ all_same_suit $0) [removed in v4]", 3, 6),
        ("poker_same_color",
         "(λ lt (n_unique_colors $0) 2)",
         "v3: (λ all_same_color $0) [removed in v4]", 3, 6),
        ("simple_two_suits",
         "(λ eq 2 (n_unique_suits $0))",
         "v1: (λ eq 2 (length (unique (map get_suit $0))))", 3, 6),
        ("bj_under_21",
         "(λ le (sum_ranks $0) 21)",
         "v1: (λ le (fold + 0 (map rank_val $0)) 21)", 3, 7),
        ("bj_exactly_21",
         "(λ eq 21 (sum_ranks $0))",
         "v1: (λ eq 21 (fold + 0 (map rank_val $0)))", 3, 7),
        ("bj_stand_17",
         "(λ ge (sum_ranks $0) 17)",
         "v1: (λ ge (fold + 0 (map rank_val $0)) 17)", 3, 7),
        ("poker_high_card",
         "(λ ge (max_rank $0) 10)",
         "v1: (λ ge (fold max 0 (map rank_val $0)) 10)", 3, 7),
        ("poker_has_ace",
         "(λ any (λ eq 14 (rank_val $0)) $0)",
         "same (needs any)", 5, 5),
        ("poker_all_face",
         "(λ all (λ ge (rank_val $0) 11) $0)",
         "same (needs all)", 5, 5),
    ]

    print(f"\n{'Rule':<25} {'v2 Depth':<10} {'v1 Depth':<10} {'Improvement':<12}")
    print("-" * 60)

    for name, v2_prog, note, v2_depth, v1_depth in rules:
        improvement = v1_depth - v2_depth
        imp_str = f"-{improvement}" if improvement > 0 else "same"
        print(f"{name:<25} {v2_depth:<10} {v1_depth:<10} {imp_str:<12}")

    print("\n" + "-" * 60)
    print("Key: Lower depth = earlier in enumeration = much more likely to solve")
    print("\nDepth reduction of 2-3 typically means 100-1000x fewer programs to search!")


# ============================================================================
# DEMO / TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("COGNITIVE PRIMITIVE LIBRARY v2")
    print("=" * 70)

    prims = build_primitives()
    print(f"\nTotal primitives: {len(prims)}")

    # Group by category
    categories = [
        ("Constants", make_constants()),
        ("Card Accessors", make_card_accessors()),
        ("Position Ops", make_position_ops()),
        ("List Slicing (NEW)", make_list_slicing()),
        ("Direct Queries (NEW)", make_direct_queries()),
        ("Aggregates (NEW)", make_aggregates()),
        ("Comparisons", make_comparisons()),
        ("Boolean Ops", make_boolean_ops()),
        ("Higher-Order", make_higher_order()),
        ("Arithmetic", make_arithmetic()),
    ]

    for name, cat_prims in categories:
        print(f"\n{name} ({len(cat_prims)}):")
        for p in cat_prims:
            print(f"  {p.name}: {p.tp}")

    show_expected_depths()

    # Test that all primitives work
    print("\n" + "=" * 70)
    print("TESTING PRIMITIVES")
    print("=" * 70)

    # Create a test hand
    test_hand = [
        Card(Suit.HEARTS, Rank.ACE),
        Card(Suit.HEARTS, Rank.KING),
        Card(Suit.HEARTS, Rank.QUEEN),
        Card(Suit.HEARTS, Rank.JACK),
        Card(Suit.HEARTS, Rank.TEN),
    ]

    print(f"\nTest hand: Royal Flush in Hearts")
    print(f"  Cards: {[f'{c.rank.name} of {c.suit.name}' for c in test_hand]}")

    # Test direct queries
    print("\nDirect Query Tests:")

    # Find primitives by name
    prim_dict = {p.name: p for p in prims}

    has_suit = prim_dict['has_suit'].value
    print(f"  has_suit hand HEARTS: {has_suit(test_hand)(Suit.HEARTS)}")
    print(f"  has_suit hand SPADES: {has_suit(test_hand)(Suit.SPADES)}")

    count_suit = prim_dict['count_suit'].value
    print(f"  count_suit hand HEARTS: {count_suit(test_hand)(Suit.HEARTS)}")

    n_unique_suits = prim_dict['n_unique_suits'].value
    print(f"  n_unique_suits hand: {n_unique_suits(test_hand)}")
    # Note: all_same_suit was removed in v4 - use (lt (n_unique_suits hand) 2) instead

    # Test aggregates
    print("\nAggregate Tests:")

    sum_ranks = prim_dict['sum_ranks'].value
    print(f"  sum_ranks hand: {sum_ranks(test_hand)}")  # 14+13+12+11+10 = 60

    max_rank = prim_dict['max_rank'].value
    print(f"  max_rank hand: {max_rank(test_hand)}")  # 14 (Ace)

    min_rank = prim_dict['min_rank'].value
    print(f"  min_rank hand: {min_rank(test_hand)}")  # 10

    # Test halves primitives
    print("\nHalves Primitives Tests:")

    test_hand_6 = [
        Card(Suit.HEARTS, Rank.ACE),
        Card(Suit.HEARTS, Rank.KING),
        Card(Suit.HEARTS, Rank.QUEEN),
        Card(Suit.SPADES, Rank.JACK),
        Card(Suit.SPADES, Rank.TEN),
        Card(Suit.SPADES, Rank.NINE),
    ]
    print(f"  Test hand (6 cards): {[f'{c.rank.name} of {c.suit.name}' for c in test_hand_6]}")

    first_half = prim_dict['first_half'].value
    second_half = prim_dict['second_half'].value

    fh = first_half(test_hand_6)
    sh = second_half(test_hand_6)

    print(f"  first_half: {[f'{c.rank.name} of {c.suit.name}' for c in fh]}")
    print(f"  second_half: {[f'{c.rank.name} of {c.suit.name}' for c in sh]}")

    # Verify the halves are correct
    assert len(fh) == 3, f"Expected first_half length 3, got {len(fh)}"
    assert len(sh) == 3, f"Expected second_half length 3, got {len(sh)}"
    assert fh[0].suit == Suit.HEARTS, "First half should be hearts"
    assert sh[0].suit == Suit.SPADES, "Second half should be spades"

    print("  ✓ Halves primitives working correctly!")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
