"""
Compositional grammar primitives for card game rules.

This module implements all 5 levels of the compositional grammar:
- Level 0: Atomic primitives (property extractors, domain mappers, position selectors)
- Level 1: Basic combinators (map, filter, count, all, any, palindrome, etc.)
- Level 2: Structural combinators (halves, terminals, shiftedPairs)
- Level 3: Domain-specific operators (hasAP, bracketMatch, cycleMap)
- Level 4: Meta-combinators (halvesEqual_F, seqPalindrome_P, etc.)

Following the compositional grammar analysis in compositional_rule_grammar.tex
"""

from typing import List, Callable, Tuple, Optional, Any, Set
from dataclasses import dataclass
from enum import Enum

from .cards import (
    Card, Hand, Suit, Rank, Color, AltColor1, AltColor2, Parity,
    RANK_VALUES, suit_to_color, suit_to_altcolor1, suit_to_altcolor2, rank_parity
)


# ============================================================================
# LEVEL 0: ATOMIC PRIMITIVES
# ============================================================================

# --- Property Extractors ---

def get_suit(card: Card) -> Suit:
    """Extract suit from card."""
    return card.suit


def get_rank(card: Card) -> Rank:
    """Extract rank from card."""
    return card.rank


def get_rank_val(card: Card) -> int:
    """Extract numeric rank value (2-14)."""
    return RANK_VALUES[card.rank]


# Derived property extractors (compositions)
def get_color(card: Card) -> Color:
    """Get color of card (RED/BLACK)."""
    return suit_to_color(card.suit)


def get_parity(card: Card) -> Parity:
    """Get parity of rank (ODD/EVEN)."""
    return rank_parity(card.rank)


def get_altcolor1(card: Card) -> AltColor1:
    """Get alternative color 1 (POINTY/ROUND)."""
    return suit_to_altcolor1(card.suit)


def get_altcolor2(card: Card) -> AltColor2:
    """Get alternative color 2 (SH/DC)."""
    return suit_to_altcolor2(card.suit)


# --- Position Selectors ---

def first(hand: Hand) -> Card:
    """Get first card in hand."""
    if not hand:
        raise ValueError("Cannot get first card of empty hand")
    return hand[0]


def last(hand: Hand) -> Card:
    """Get last card in hand."""
    if not hand:
        raise ValueError("Cannot get last card of empty hand")
    return hand[-1]


def at(index: int) -> Callable[[Hand], Card]:
    """Get card at specific index (0-based)."""
    def _at(hand: Hand) -> Card:
        if index < 0 or index >= len(hand):
            raise IndexError(f"Index {index} out of range for hand of size {len(hand)}")
        return hand[index]
    return _at


def slice_hand(start: int, end: int) -> Callable[[Hand], Hand]:
    """Get subsequence of hand [start:end)."""
    def _slice(hand: Hand) -> Hand:
        return hand[start:end]
    return _slice


# --- Suit Cycle Transformations ---

# Cycle M1: C → S → H → D → C
CYCLE_M1_MAP = {
    Suit.CLUBS: Suit.SPADES,
    Suit.SPADES: Suit.HEARTS,
    Suit.HEARTS: Suit.DIAMONDS,
    Suit.DIAMONDS: Suit.CLUBS
}

# Cycle M2: C → H → S → D → C
CYCLE_M2_MAP = {
    Suit.CLUBS: Suit.HEARTS,
    Suit.HEARTS: Suit.SPADES,
    Suit.SPADES: Suit.DIAMONDS,
    Suit.DIAMONDS: Suit.CLUBS
}


def suit_cycle_m1(suit: Suit) -> Suit:
    """Apply cycle transformation M1."""
    return CYCLE_M1_MAP[suit]


def suit_cycle_m2(suit: Suit) -> Suit:
    """Apply cycle transformation M2."""
    return CYCLE_M2_MAP[suit]


# ============================================================================
# LEVEL 1: BASIC COMBINATORS
# ============================================================================

# --- List Transformations ---

def map_property(prop_fn: Callable[[Card], Any]) -> Callable[[Hand], List[Any]]:
    """Map property extractor over hand."""
    def _map(hand: Hand) -> List[Any]:
        return [prop_fn(card) for card in hand]
    return _map


def filter_cards(pred: Callable[[Card], bool]) -> Callable[[Hand], Hand]:
    """Filter hand by predicate."""
    def _filter(hand: Hand) -> Hand:
        return [card for card in hand if pred(card)]
    return _filter


def reverse_hand(hand: Hand) -> Hand:
    """Reverse order of cards."""
    return list(reversed(hand))


def sort_hand(hand: Hand, key_fn: Optional[Callable[[Card], int]] = None) -> Hand:
    """Sort hand by rank values."""
    if key_fn is None:
        key_fn = get_rank_val
    return sorted(hand, key=key_fn)


# --- Aggregators ---

def count_if(pred: Callable[[Card], bool]) -> Callable[[Hand], int]:
    """Count cards satisfying predicate."""
    def _count(hand: Hand) -> int:
        return sum(1 for card in hand if pred(card))
    return _count


def count_equal(value: Any, prop_fn: Callable[[Card], Any]) -> Callable[[Hand], int]:
    """Count cards with specific property value."""
    def _count(hand: Hand) -> int:
        return sum(1 for card in hand if prop_fn(card) == value)
    return _count


def unique_values(prop_fn: Callable[[Card], Any]) -> Callable[[Hand], Set[Any]]:
    """Get unique property values in hand."""
    def _unique(hand: Hand) -> Set[Any]:
        return set(prop_fn(card) for card in hand)
    return _unique


def unique_count(prop_fn: Callable[[Card], Any]) -> Callable[[Hand], int]:
    """Count unique property values."""
    def _count(hand: Hand) -> int:
        return len(unique_values(prop_fn)(hand))
    return _count


def sum_values(hand: Hand, val_fn: Callable[[Card], int] = get_rank_val) -> int:
    """Sum numeric values."""
    return sum(val_fn(card) for card in hand)


def max_value(hand: Hand, val_fn: Callable[[Card], int] = get_rank_val) -> int:
    """Maximum value in hand."""
    if not hand:
        raise ValueError("Cannot get max of empty hand")
    return max(val_fn(card) for card in hand)


def min_value(hand: Hand, val_fn: Callable[[Card], int] = get_rank_val) -> int:
    """Minimum value in hand."""
    if not hand:
        raise ValueError("Cannot get min of empty hand")
    return min(val_fn(card) for card in hand)


# --- Universal/Existential Quantifiers ---

def all_satisfy(pred: Callable[[Card], bool]) -> Callable[[Hand], bool]:
    """Check if all cards satisfy predicate."""
    def _all(hand: Hand) -> bool:
        return all(pred(card) for card in hand)
    return _all


def any_satisfy(pred: Callable[[Card], bool]) -> Callable[[Hand], bool]:
    """Check if any card satisfies predicate."""
    def _any(hand: Hand) -> bool:
        return any(pred(card) for card in hand)
    return _any


def none_satisfy(pred: Callable[[Card], bool]) -> Callable[[Hand], bool]:
    """Check if no cards satisfy predicate."""
    def _none(hand: Hand) -> bool:
        return not any(pred(card) for card in hand)
    return _none


# --- Sequence Predicates ---

def is_palindrome(seq: List[Any]) -> bool:
    """Check if sequence is a palindrome."""
    return seq == list(reversed(seq))


def is_sorted(hand: Hand, key_fn: Callable[[Card], int] = get_rank_val, strict: bool = False) -> bool:
    """
    Check if hand is sorted by key function.

    Args:
        hand: Hand to check
        key_fn: Function to extract sort key
        strict: If True, require strict increasing (no ties)
    """
    values = [key_fn(card) for card in hand]
    if strict:
        return all(values[i] < values[i+1] for i in range(len(values) - 1))
    else:
        return all(values[i] <= values[i+1] for i in range(len(values) - 1))


def arrays_equal(seq1: List[Any], seq2: List[Any]) -> bool:
    """Check if two sequences are element-wise equal."""
    return seq1 == seq2


# --- Pairwise Operations ---

def pairwise_adjacent(relation: Callable[[Any, Any], bool], prop_fn: Callable[[Card], Any]) -> Callable[[Hand], bool]:
    """Check if relation holds between all adjacent pairs."""
    def _check(hand: Hand) -> bool:
        if len(hand) < 2:
            return True
        values = [prop_fn(card) for card in hand]
        return all(relation(values[i], values[i+1]) for i in range(len(values) - 1))
    return _check


def pairwise_skip(k: int, relation: Callable[[Any, Any], bool], prop_fn: Callable[[Card], Any]) -> Callable[[Hand], bool]:
    """Check if relation holds between pairs separated by k positions."""
    def _check(hand: Hand) -> bool:
        if len(hand) <= k:
            return True
        values = [prop_fn(card) for card in hand]
        return all(relation(values[i], values[i+k]) for i in range(len(values) - k))
    return _check


def pairwise_all(relation: Callable[[Any, Any], bool], prop_fn: Callable[[Card], Any]) -> Callable[[Hand], bool]:
    """Check if relation holds between all pairs (not just adjacent)."""
    def _check(hand: Hand) -> bool:
        values = [prop_fn(card) for card in hand]
        for i in range(len(values)):
            for j in range(i+1, len(values)):
                if not relation(values[i], values[j]):
                    return False
        return True
    return _check


# ============================================================================
# LEVEL 2: STRUCTURAL COMBINATORS
# ============================================================================

def halves(hand: Hand) -> Tuple[Hand, Hand]:
    """
    Split hand into two equal (or nearly equal) halves.

    For even-length hands, both halves have size n/2.
    For odd-length hands, middle card is excluded, both halves have size floor(n/2).

    Examples:
        >>> h = [C1, C2, C3, C4, C5, C6]
        >>> halves(h) == ([C1, C2, C3], [C4, C5, C6])
        >>> h = [C1, C2, C3, C4, C5]
        >>> halves(h) == ([C1, C2], [C4, C5])  # C3 excluded
    """
    n = len(hand)
    k = n // 2
    left = hand[:k]
    right = hand[n-k:]
    return (left, right)


def terminals(hand: Hand) -> Tuple[Card, Card]:
    """Get first and last cards as a pair."""
    if len(hand) < 2:
        raise ValueError("Hand must have at least 2 cards for terminals")
    return (first(hand), last(hand))


def center(hand: Hand) -> Tuple[Card, ...]:
    """
    Get center card(s).

    Returns:
        - For odd-length: single center card (as 1-tuple)
        - For even-length: middle two cards (as 2-tuple)
    """
    n = len(hand)
    if n == 0:
        raise ValueError("Cannot get center of empty hand")

    if n % 2 == 1:
        # Odd: single center
        return (hand[n // 2],)
    else:
        # Even: middle two
        return (hand[n // 2 - 1], hand[n // 2])


def shifted_pairs(k: int) -> Callable[[Hand], List[Tuple[Card, Card]]]:
    """
    Extract pairs of cards separated by k positions.

    Example:
        >>> h = [C0, C1, C2, C3, C4]
        >>> shifted_pairs(2)(h) == [(C0, C2), (C1, C3), (C2, C4)]
    """
    def _pairs(hand: Hand) -> List[Tuple[Card, Card]]:
        if len(hand) <= k:
            return []
        return [(hand[i], hand[i+k]) for i in range(len(hand) - k)]
    return _pairs


def adjacent_pairs(hand: Hand) -> List[Tuple[Card, Card]]:
    """Get all adjacent pairs (k=1 case of shifted_pairs)."""
    return shifted_pairs(1)(hand)


# ============================================================================
# LEVEL 3: DOMAIN-SPECIFIC OPERATORS
# ============================================================================

def has_arithmetic_progression(
    length: int,
    step: Optional[int] = None,
    aligned: bool = False
) -> Callable[[Hand], bool]:
    """
    Check if hand contains an arithmetic progression of ranks.

    Args:
        length: Number of terms in AP (e.g., 3 for triple, 4 for quad)
        step: Step size (None = any positive step allowed)
        aligned: If True, AP must occur in consecutive positions

    Examples:
        >>> has_arithmetic_progression(3, 2, False)  # Contains ranks with diff=2 (e.g., 4,6,8)
        >>> has_arithmetic_progression(3, None, True)  # Consecutive triple with any equal step
    """
    def _check(hand: Hand) -> bool:
        rank_vals = [get_rank_val(card) for card in hand]

        if aligned:
            # Check consecutive subsequences
            for i in range(len(rank_vals) - length + 1):
                subseq = rank_vals[i:i+length]
                diffs = [subseq[j+1] - subseq[j] for j in range(len(subseq) - 1)]

                if step is None:
                    # Any constant step
                    if len(set(diffs)) == 1 and diffs[0] > 0:
                        return True
                else:
                    # Specific step
                    if all(d == step for d in diffs):
                        return True
            return False
        else:
            # Unaligned: check if set contains AP
            rank_set = set(rank_vals)

            if step is None:
                # Try all possible steps
                for start in rank_set:
                    for d in range(1, 15):  # Max possible step
                        ap = [start + i*d for i in range(length)]
                        if all(val in rank_set for val in ap):
                            return True
                return False
            else:
                # Specific step
                for start in rank_set:
                    ap = [start + i*step for i in range(length)]
                    if all(val in rank_set for val in ap):
                        return True
                return False

    return _check


def bracket_match_suits(hand: Hand) -> bool:
    """
    Check if suits form well-matched bracket sequence.

    Bracket mapping:
    - Openers: ♠ → '(', ♥ → '['
    - Closers: ♣ → ')', ♦ → ']'

    Example:
        >>> [♠, ♥, ♦, ♣]  # ( [ ] ) → True
        >>> [♠, ♣, ♥, ♦]  # ( ) [ ] → True
        >>> [♠, ♥, ♣, ♦]  # ( [ ) ] → False
    """
    opener_map = {Suit.SPADES: '(', Suit.HEARTS: '['}
    closer_map = {Suit.CLUBS: ')', Suit.DIAMONDS: ']'}
    match_map = {')': '(', ']': '['}

    stack = []
    for card in hand:
        suit = card.suit
        if suit in opener_map:
            stack.append(opener_map[suit])
        elif suit in closer_map:
            closer = closer_map[suit]
            expected = match_map[closer]
            if not stack or stack.pop() != expected:
                return False
        else:
            # Suit not in bracket mapping
            return False

    return len(stack) == 0


def bracket_match_ranks_even_odd(hand: Hand, even_opens: bool = True) -> bool:
    """
    Check if ranks form well-matched bracket sequence based on parity.

    Args:
        even_opens: If True, even ranks open and next odd closes.
                    If False, odd ranks open and next even closes.

    Example (even_opens=True):
        >>> [6, 7, 4, 5]  # 6 opens → 7 closes, 4 opens → 5 closes → True
    """
    if even_opens:
        # Even opens, odd closes
        opener_set = {2, 4, 6, 8, 10, 12}
        closer_set = {3, 5, 7, 9, 11, 13}
    else:
        # Odd opens, even closes
        opener_set = {3, 5, 7, 9, 11, 13}
        closer_set = {4, 6, 8, 10, 12, 14}

    stack = []
    for card in hand:
        val = get_rank_val(card)

        if val in opener_set:
            # Push expected closer
            stack.append(val + 1)
        elif val in closer_set:
            # Check if this is expected closer
            if not stack or stack.pop() != val:
                return False
        else:
            # Rank not in bracket mapping
            return False

    return len(stack) == 0


# ============================================================================
# LEVEL 4: META-COMBINATORS
# ============================================================================

def halves_equal(prop_fn: Callable[[Hand], Any]) -> Callable[[Hand], bool]:
    """
    Check if applying property function to both halves gives equal results.

    Example:
        >>> halves_equal(lambda h: [get_suit(c) for c in h])  # Halves have same suit sequence
    """
    def _check(hand: Hand) -> bool:
        left, right = halves(hand)
        return prop_fn(left) == prop_fn(right)
    return _check


def terminals_equal(prop_fn: Callable[[Card], Any]) -> Callable[[Hand], bool]:
    """Check if property is equal for first and last cards."""
    def _check(hand: Hand) -> bool:
        if len(hand) < 2:
            return True
        f, l = terminals(hand)
        return prop_fn(f) == prop_fn(l)
    return _check


def seq_palindrome(prop_fn: Callable[[Card], Any]) -> Callable[[Hand], bool]:
    """Check if sequence of property values is a palindrome."""
    def _check(hand: Hand) -> bool:
        seq = [prop_fn(card) for card in hand]
        return is_palindrome(seq)
    return _check


def seq_operation(
    prop_fn: Callable[[Card], Any],
    op_fn: Callable[[List[Any]], bool]
) -> Callable[[Hand], bool]:
    """
    Extract property sequence, then apply boolean operation.

    Example:
        >>> seq_operation(get_suit, lambda seq: len(set(seq)) == 2)  # Exactly 2 suits
    """
    def _check(hand: Hand) -> bool:
        seq = [prop_fn(card) for card in hand]
        return op_fn(seq)
    return _check


# ============================================================================
# HELPER PREDICATES (commonly used compositions)
# ============================================================================

def uniform_property(prop_fn: Callable[[Card], Any]) -> Callable[[Hand], bool]:
    """Check if all cards have same property value."""
    def _check(hand: Hand) -> bool:
        if not hand:
            return True
        values = [prop_fn(card) for card in hand]
        return len(set(values)) == 1
    return _check


def contains_value(value: Any, prop_fn: Callable[[Card], Any]) -> Callable[[Hand], bool]:
    """Check if any card has specified property value."""
    def _check(hand: Hand) -> bool:
        return any(prop_fn(card) == value for card in hand)
    return _check


# Binary relations (for use with pairwise combinators)
def eq(a: Any, b: Any) -> bool:
    """Equality relation."""
    return a == b


def neq(a: Any, b: Any) -> bool:
    """Inequality relation."""
    return a != b


def lt(a: int, b: int) -> bool:
    """Less than."""
    return a < b


def lte(a: int, b: int) -> bool:
    """Less than or equal."""
    return a <= b


def gt(a: int, b: int) -> bool:
    """Greater than."""
    return a > b


def gte(a: int, b: int) -> bool:
    """Greater than or equal."""
    return a >= b


if __name__ == "__main__":
    # Quick tests
    from .cards import H, D, S, C

    print("=== Primitives Tests ===\n")

    # Test halves
    hand = [H("2"), D("3"), S("4"), C("5"), H("6"), D("7")]
    left, right = halves(hand)
    print(f"Hand: {hand}")
    print(f"Halves: {left} | {right}\n")

    # Test AP detection
    has_ap_3_2 = has_arithmetic_progression(3, 2, False)
    ap_hand = [H("2"), D("4"), S("6"), C("8")]
    print(f"Hand: {ap_hand}")
    print(f"Has AP (len=3, step=2): {has_ap_3_2(ap_hand)}\n")

    # Test bracket matching
    bracket_hand = [S("2"), H("3"), D("4"), C("5")]
    print(f"Hand: {bracket_hand}")
    print(f"Bracket match: {bracket_match_suits(bracket_hand)}\n")

    # Test palindrome
    is_suit_palindrome = seq_palindrome(get_suit)
    pal_hand = [H("2"), D("3"), H("4")]
    print(f"Hand: {pal_hand}")
    print(f"Suit palindrome: {is_suit_palindrome(pal_hand)}")
