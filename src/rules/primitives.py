"""
Primitive helper functions for catalogue rules.

This module provides simple Python helper functions used by the rule catalogue
to define the 56 experimental rules. These are NOT the typed DSL primitives
used by DreamCoder (those are in dreamcoder_core/lean_primitives.py).

These functions are designed to be readable and match human intuition about
card game rules.
"""

from typing import List, Callable, Any, TypeVar, Optional, Set
from .cards import (
    Card, Hand, Suit, Rank, Color, AltColor1, AltColor2, Parity,
    RANK_VALUES, card_color, suit_to_altcolor1, suit_to_altcolor2, rank_parity
)

# Re-export Suit for convenience (used in catalogue rules)
# This allows `from .primitives import *` to get Suit as well


T = TypeVar('T')


# ============================================================================
# CARD ACCESSORS
# ============================================================================

def get_suit(card: Card) -> Suit:
    """Get the suit of a card."""
    return card.suit


def get_rank(card: Card) -> Rank:
    """Get the rank of a card."""
    return card.rank


def get_rank_val(card: Card) -> int:
    """Get the numeric value of a card's rank (2-14)."""
    return RANK_VALUES[card.rank]


def get_color(card: Card) -> Color:
    """Get the color (RED/BLACK) of a card."""
    return card_color(card)


def get_altcolor1(card: Card) -> AltColor1:
    """Get alternative color scheme 1 (POINTY/ROUND) of a card."""
    return suit_to_altcolor1(card.suit)


def get_altcolor2(card: Card) -> AltColor2:
    """Get alternative color scheme 2 (SH/DC) of a card."""
    return suit_to_altcolor2(card.suit)


def get_parity(card: Card) -> Parity:
    """Get the parity (ODD/EVEN) of a card's rank."""
    return rank_parity(card.rank)


# ============================================================================
# POSITION ACCESSORS
# ============================================================================

def first(xs: List[T]) -> Optional[T]:
    """Get the first element of a list."""
    return xs[0] if xs else None


def last(xs: List[T]) -> Optional[T]:
    """Get the last element of a list."""
    return xs[-1] if xs else None


def nth(xs: List[T], n: int) -> Optional[T]:
    """Get the nth element of a list (0-indexed)."""
    return xs[n] if 0 <= n < len(xs) else None


# ============================================================================
# LIST OPERATIONS
# ============================================================================

def length(xs: List) -> int:
    """Get the length of a list."""
    return len(xs)


def unique(xs: List[T]) -> List[T]:
    """Get unique elements, preserving order."""
    seen = set()
    result = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


def count(pred: Callable[[T], bool], xs: List[T]) -> int:
    """Count elements satisfying a predicate."""
    return sum(1 for x in xs if pred(x))


def count_eq(value: Any, xs: List) -> int:
    """Count elements equal to value."""
    return sum(1 for x in xs if x == value)


def filter_by(pred: Callable[[T], bool], xs: List[T]) -> List[T]:
    """Filter elements by predicate."""
    return [x for x in xs if pred(x)]


def map_fn(fn: Callable[[T], Any], xs: List[T]) -> List:
    """Map a function over a list."""
    return [fn(x) for x in xs]


# ============================================================================
# HALVES OPERATIONS
# ============================================================================

def left_half(xs: List[T]) -> List[T]:
    """Get the left half of a list (first n//2 elements)."""
    n = len(xs)
    return xs[:n // 2]


def right_half(xs: List[T]) -> List[T]:
    """Get the right half of a list (last n//2 elements)."""
    n = len(xs)
    return xs[(n + 1) // 2:]  # For odd length, skip middle


def halves(xs: List[T]) -> tuple:
    """Split list into two halves."""
    return left_half(xs), right_half(xs)


# ============================================================================
# SORTED / ORDERING
# ============================================================================

def is_sorted(hand: Hand, key_fn: Optional[Callable[[Card], Any]] = None, strict: bool = False) -> bool:
    """Check if hand is sorted by key_fn.

    Args:
        hand: List of cards
        key_fn: Function to extract comparison key from card (defaults to rank value)
        strict: If True, require strictly increasing (no duplicates)
    """
    if len(hand) <= 1:
        return True

    # Default to sorting by rank value (non-decreasing)
    if key_fn is None:
        key_fn = lambda c: RANK_VALUES[c.rank]

    values = [key_fn(c) for c in hand]
    for i in range(len(values) - 1):
        if strict:
            if values[i] >= values[i + 1]:
                return False
        else:
            if values[i] > values[i + 1]:
                return False
    return True


def is_strictly_sorted(hand: Hand, key_fn: Callable[[Card], Any]) -> bool:
    """Check if hand is strictly sorted (no duplicates)."""
    return is_sorted(hand, key_fn, strict=True)


# ============================================================================
# COMPARISONS
# ============================================================================

def eq(a: Any, b: Any) -> bool:
    """Equality comparison."""
    return a == b


def lt(a: Any, b: Any) -> bool:
    """Less than comparison."""
    return a < b


def gt(a: Any, b: Any) -> bool:
    """Greater than comparison."""
    return a > b


def le(a: Any, b: Any) -> bool:
    """Less than or equal comparison."""
    return a <= b


def ge(a: Any, b: Any) -> bool:
    """Greater than or equal comparison."""
    return a >= b


# ============================================================================
# AGGREGATIONS
# ============================================================================

def sum_vals(xs: List[int]) -> int:
    """Sum of values."""
    return sum(xs)


def max_val(xs: List[int]) -> int:
    """Maximum value."""
    return max(xs) if xs else 0


def min_val(xs: List[int]) -> int:
    """Minimum value."""
    return min(xs) if xs else 0


def all_same(xs: List) -> bool:
    """Check if all elements are the same."""
    if not xs:
        return True
    first_val = xs[0]
    return all(x == first_val for x in xs)


def all_different(xs: List) -> bool:
    """Check if all elements are different."""
    return len(xs) == len(set(xs))


# ============================================================================
# COUNT OPERATIONS (for specific properties)
# ============================================================================

def count_suit(hand: Hand, suit: Suit) -> int:
    """Count cards of a specific suit."""
    return sum(1 for c in hand if c.suit == suit)


def count_rank(hand: Hand, rank: Rank) -> int:
    """Count cards of a specific rank."""
    return sum(1 for c in hand if c.rank == rank)


def count_color(hand: Hand, color: Color) -> int:
    """Count cards of a specific color."""
    return sum(1 for c in hand if card_color(c) == color)


def unique_suits(hand: Hand) -> Set[Suit]:
    """Get unique suits in hand."""
    return set(c.suit for c in hand)


def unique_ranks(hand: Hand) -> Set[Rank]:
    """Get unique ranks in hand."""
    return set(c.rank for c in hand)


def unique_colors(hand: Hand) -> Set[Color]:
    """Get unique colors in hand."""
    return set(card_color(c) for c in hand)


# ============================================================================
# HIGHER-ORDER PREDICATES
# ============================================================================

def any_fn(pred: Callable[[T], bool], xs: List[T]) -> bool:
    """Check if any element satisfies predicate."""
    return any(pred(x) for x in xs)


def all_fn(pred: Callable[[T], bool], xs: List[T]) -> bool:
    """Check if all elements satisfy predicate."""
    return all(pred(x) for x in xs)


# ============================================================================
# TERMINALS / BOUNDARY CHECKS
# ============================================================================

def terminals_equal(hand: Hand, key_fn: Callable[[Card], Any]) -> bool:
    """Check if first and last cards have equal key values."""
    if len(hand) < 2:
        return False
    return key_fn(hand[0]) == key_fn(hand[-1])


def terminals_different(hand: Hand, key_fn: Callable[[Card], Any]) -> bool:
    """Check if first and last cards have different key values."""
    if len(hand) < 2:
        return False
    return key_fn(hand[0]) != key_fn(hand[-1])


# ============================================================================
# SEQUENCE PATTERNS
# ============================================================================

def is_palindrome(xs: List) -> bool:
    """Check if list is a palindrome."""
    return xs == list(reversed(xs))


def alternating(xs: List) -> bool:
    """Check if elements alternate (no two adjacent are equal)."""
    if len(xs) <= 1:
        return True
    for i in range(len(xs) - 1):
        if xs[i] == xs[i + 1]:
            return False
    return True


def consecutive_pairs(xs: List[T]) -> List[tuple]:
    """Get consecutive pairs: [a,b,c] -> [(a,b), (b,c)]."""
    return [(xs[i], xs[i + 1]) for i in range(len(xs) - 1)]


# ============================================================================
# ARITHMETIC PROGRESSIONS
# ============================================================================

def is_arithmetic_progression(vals: List[int]) -> bool:
    """Check if values form an arithmetic progression."""
    if len(vals) <= 2:
        return True
    diff = vals[1] - vals[0]
    for i in range(2, len(vals)):
        if vals[i] - vals[i - 1] != diff:
            return False
    return True


def is_constant(vals: List[int]) -> bool:
    """Check if all values are the same (AP with diff=0)."""
    return len(set(vals)) <= 1


# ============================================================================
# EXISTENTIAL PATTERNS
# ============================================================================

def exists_ordered(hand: Hand, key_fn: Callable[[Card], Any],
                   pred1: Callable[[Any], bool], pred2: Callable[[Any], bool]) -> bool:
    """Check if there exists a card satisfying pred1 before a card satisfying pred2."""
    found_first = False
    for card in hand:
        key = key_fn(card)
        if pred1(key):
            found_first = True
        if found_first and pred2(key):
            return True
    return False


# ============================================================================
# HIGHER-ORDER PREDICATE FACTORIES
# These return predicates (Hand -> bool) for use in rule definitions
# ============================================================================

def uniform_property(key_fn: Callable[[Card], Any]) -> Callable[[Hand], bool]:
    """Create a predicate checking if all cards have the same key value.

    Args:
        key_fn: Function to extract property from card (e.g., get_color)

    Returns:
        Predicate (Hand -> bool) that returns True if all cards have same value
    """
    def predicate(hand: Hand) -> bool:
        if not hand:
            return True
        first_val = key_fn(hand[0])
        return all(key_fn(c) == first_val for c in hand)
    return predicate


def seq_palindrome(key_fn: Callable[[Card], Any]) -> Callable[[Hand], bool]:
    """Create a predicate checking if the sequence of key values is palindromic.

    Args:
        key_fn: Function to extract property from card (e.g., get_suit)

    Returns:
        Predicate (Hand -> bool) that returns True if [key(c) for c in hand] is palindrome
    """
    def predicate(hand: Hand) -> bool:
        vals = [key_fn(c) for c in hand]
        return vals == list(reversed(vals))
    return predicate


def unique_count(key_fn: Callable[[Card], Any]) -> Callable[[Hand], int]:
    """Create a function counting unique values of a property.

    Args:
        key_fn: Function to extract property from card

    Returns:
        Function (Hand -> int) returning count of unique values
    """
    def counter(hand: Hand) -> int:
        return len(set(key_fn(c) for c in hand))
    return counter


def count_equal(value: Any, key_fn: Callable[[Card], Any]) -> Callable[[Hand], int]:
    """Create a function counting cards where key equals value.

    Args:
        value: Value to match
        key_fn: Function to extract property from card

    Returns:
        Function (Hand -> int) returning count of matches
    """
    def counter(hand: Hand) -> int:
        return sum(1 for c in hand if key_fn(c) == value)
    return counter


def has_arithmetic_progression(length: int, step: Optional[int], aligned: bool) -> Callable[[Hand], bool]:
    """Create a predicate checking for arithmetic progression in ranks.

    Args:
        length: Length of AP to find (e.g., 3 for 3-term AP)
        step: Required step size (None = any step)
        aligned: If True, AP must be in consecutive positions

    Returns:
        Predicate (Hand -> bool)
    """
    def predicate(hand: Hand) -> bool:
        if len(hand) < length:
            return False

        rank_vals = [RANK_VALUES[c.rank] for c in hand]

        if aligned:
            # Check consecutive windows
            for i in range(len(rank_vals) - length + 1):
                window = rank_vals[i:i + length]
                if _is_ap(window, step):
                    return True
        else:
            # Check all combinations
            from itertools import combinations
            for combo in combinations(sorted(rank_vals), length):
                if _is_ap(list(combo), step):
                    return True
        return False

    return predicate


def _is_ap(vals: List[int], required_step: Optional[int]) -> bool:
    """Check if sorted values form an AP with optional step constraint."""
    if len(vals) < 2:
        return True
    diff = vals[1] - vals[0]
    if required_step is not None and diff != required_step:
        return False
    for i in range(2, len(vals)):
        if vals[i] - vals[i - 1] != diff:
            return False
    return True


def is_run(hand: Hand) -> bool:
    """Check if hand forms a consecutive run of ranks."""
    if len(hand) < 2:
        return True
    vals = sorted(RANK_VALUES[c.rank] for c in hand)
    for i in range(len(vals) - 1):
        if vals[i + 1] - vals[i] != 1:
            return False
    return True


def bracket_match_suits(hand: Hand) -> bool:
    """Check if spades 'open' and hearts 'close' in balanced way.

    Each ♠ pushes onto stack, each ♥ pops. Balanced if stack empty at end.
    """
    stack = 0
    for card in hand:
        if card.suit == Suit.SPADES:
            stack += 1
        elif card.suit == Suit.HEARTS:
            if stack <= 0:
                return False
            stack -= 1
    return stack == 0


def bracket_match_ranks_even_odd(hand: Hand, even_opens: bool = True) -> bool:
    """Check if ranks form balanced bracket pattern based on parity.

    Args:
        hand: Cards to check
        even_opens: If True, even ranks open/push; if False, odd ranks open

    Returns:
        True if brackets are balanced
    """
    stack = []
    for card in hand:
        val = RANK_VALUES[card.rank]
        is_even = val % 2 == 0

        if even_opens:
            opens = is_even
        else:
            opens = not is_even

        if opens:
            stack.append(val)
        else:
            # Closing - check if it matches the most recent opener
            if not stack:
                return False
            opener = stack.pop()
            # For parity matching, closer should be opener + 1 (or some rule)
            # Simplified: just check balanced structure
    return len(stack) == 0


def shifted_pairs(k: int) -> Callable[[Hand], List[tuple]]:
    """Create a function that gets shifted pairs with offset k.

    Args:
        k: Offset between paired elements

    Returns:
        Function (Hand -> List[tuple]) returning pairs (hand[i], hand[i+k])
    """
    def get_pairs(hand: Hand) -> List[tuple]:
        if k >= len(hand):
            return []
        return [(hand[i], hand[i + k]) for i in range(len(hand) - k)]
    return get_pairs


def dist(a: int, b: int) -> int:
    """Absolute difference between two values."""
    return abs(a - b)


def sum_values(hand: Hand) -> int:
    """Sum of rank values in a hand."""
    return sum(RANK_VALUES[c.rank] for c in hand)


# Suit cycle maps for the MAP family of rules
# M1: ♣→♦→♥→♠→♣ (standard CDHS order)
# M2: ♣→♥→♦→♠→♣ (alternative order)

_SUIT_ORDER = [Suit.CLUBS, Suit.DIAMONDS, Suit.HEARTS, Suit.SPADES]
_SUIT_ORDER_M2 = [Suit.CLUBS, Suit.HEARTS, Suit.DIAMONDS, Suit.SPADES]


def suit_cycle_m1(suit: Suit) -> Suit:
    """Map suit to next suit in cycle: ♣→♦→♥→♠→♣."""
    idx = _SUIT_ORDER.index(suit)
    return _SUIT_ORDER[(idx + 1) % 4]


def suit_cycle_m2(suit: Suit) -> Suit:
    """Map suit to next suit in alternative cycle: ♣→♥→♦→♠→♣."""
    idx = _SUIT_ORDER_M2.index(suit)
    return _SUIT_ORDER_M2[(idx + 1) % 4]


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Card accessors
    'get_suit', 'get_rank', 'get_rank_val', 'get_color',
    'get_altcolor1', 'get_altcolor2', 'get_parity',

    # Position
    'first', 'last', 'nth',

    # List ops
    'length', 'unique', 'count', 'count_eq', 'filter_by', 'map_fn',

    # Halves
    'left_half', 'right_half', 'halves',

    # Sorting
    'is_sorted', 'is_strictly_sorted',

    # Comparisons
    'eq', 'lt', 'gt', 'le', 'ge',

    # Aggregations
    'sum_vals', 'max_val', 'min_val', 'all_same', 'all_different',

    # Count operations
    'count_suit', 'count_rank', 'count_color',
    'unique_suits', 'unique_ranks', 'unique_colors',

    # Higher-order predicates
    'any_fn', 'all_fn',

    # Terminals
    'terminals_equal', 'terminals_different',

    # Sequences
    'is_palindrome', 'alternating', 'consecutive_pairs',

    # Arithmetic
    'is_arithmetic_progression', 'is_constant',

    # Existential
    'exists_ordered',

    # Higher-order predicate factories
    'uniform_property', 'seq_palindrome', 'unique_count', 'count_equal',
    'has_arithmetic_progression', 'is_run',

    # Bracket matching
    'bracket_match_suits', 'bracket_match_ranks_even_odd',

    # Utility
    'shifted_pairs', 'dist', 'sum_values',

    # Suit cycles for MAP rules
    'suit_cycle_m1', 'suit_cycle_m2',

    # Re-exported types (needed by catalogue rules)
    'Suit', 'Rank', 'Color', 'AltColor1', 'AltColor2', 'Parity',
    'RANK_VALUES', 'Card', 'Hand',
]
