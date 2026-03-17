"""
Tests for the 5 new grammar-comparison primitives.

Written TDD-style: these tests define the expected behaviour before
the implementation exists.
"""

import sys
from pathlib import Path

# Allow importing Card objects from the main src/ tree
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import pytest
from rules.cards import Card, Suit, Rank, RANK_VALUES, H, D, S, C

# Import the primitives under test (from the grammar_comparison package)
from llm.grammar_comparison.primitives import (
    prim_slice,
    prim_shifted_match,
    prim_stride,
    prim_count_where,
    prim_sorted_counts,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def rank_val(card: Card) -> int:
    """Extract numeric rank value from a card."""
    return RANK_VALUES[card.rank]


def same_suit(a: Card, b: Card) -> bool:
    """Predicate: two cards share the same suit."""
    return a.suit == b.suit


def same_rank(a: Card, b: Card) -> bool:
    """Predicate: two cards share the same rank."""
    return a.rank == b.rank


def ascending_rank(a: Card, b: Card) -> bool:
    """Predicate: b's rank value is exactly a's + 1."""
    return RANK_VALUES[b.rank] == RANK_VALUES[a.rank] + 1


def is_red(card: Card) -> bool:
    return card.suit in {Suit.HEARTS, Suit.DIAMONDS}


def get_suit(card: Card) -> Suit:
    return card.suit


def get_rank(card: Card) -> Rank:
    return card.rank


# ---------------------------------------------------------------------------
# 1. prim_slice
# ---------------------------------------------------------------------------

class TestPrimSlice:
    """prim_slice(i, j, xs) -> xs[i:j]

    Subsumes: take (slice 0 n), drop (slice n len), first_half, second_half.
    """

    def test_basic_slice(self):
        hand = [H("2"), H("3"), H("4"), H("5"), H("6")]
        result = prim_slice(1, 3, hand)
        assert result == [H("3"), H("4")]

    def test_take_equivalent(self):
        """slice(0, n, xs) == take(n, xs)"""
        hand = [H("2"), D("3"), S("4"), C("5")]
        assert prim_slice(0, 2, hand) == [H("2"), D("3")]

    def test_drop_equivalent(self):
        """slice(n, len, xs) == drop(n, xs)"""
        hand = [H("2"), D("3"), S("4"), C("5")]
        assert prim_slice(2, 4, hand) == [S("4"), C("5")]

    def test_first_half(self):
        """slice(0, len//2, xs) == first_half(xs)"""
        hand = [H("2"), D("3"), S("4"), C("5")]
        mid = len(hand) // 2
        assert prim_slice(0, mid, hand) == [H("2"), D("3")]

    def test_second_half(self):
        """slice(len//2, len, xs) == second_half(xs)"""
        hand = [H("2"), D("3"), S("4"), C("5")]
        mid = len(hand) // 2
        assert prim_slice(mid, len(hand), hand) == [S("4"), C("5")]

    def test_empty_slice(self):
        hand = [H("2"), D("3")]
        assert prim_slice(2, 2, hand) == []

    def test_full_slice(self):
        hand = [H("A"), D("K")]
        assert prim_slice(0, 2, hand) == hand

    def test_works_with_plain_list(self):
        """Should work on any list, not just cards."""
        assert prim_slice(1, 4, [10, 20, 30, 40, 50]) == [20, 30, 40]


# ---------------------------------------------------------------------------
# 2. prim_shifted_match
# ---------------------------------------------------------------------------

class TestPrimShiftedMatch:
    """prim_shifted_match(k, pred, xs) -> all(pred(xs[i], xs[i+k]) for valid i)

    Subsumes: pairwise checks (k=1), checking every-other pair (k=2), etc.
    """

    def test_adjacent_same_suit(self):
        """k=1 with same_suit: all adjacent cards share suit."""
        hand = [H("2"), H("5"), H("9")]
        assert prim_shifted_match(1, same_suit, hand) is True

    def test_adjacent_same_suit_fails(self):
        hand = [H("2"), D("5"), H("9")]
        assert prim_shifted_match(1, same_suit, hand) is False

    def test_ascending_run(self):
        """k=1 with ascending: consecutive ranks."""
        hand = [H("3"), D("4"), S("5"), C("6")]
        assert prim_shifted_match(1, ascending_rank, hand) is True

    def test_ascending_run_fails(self):
        hand = [H("3"), D("4"), S("6")]
        assert prim_shifted_match(1, ascending_rank, hand) is False

    def test_shift_2_same_rank(self):
        """k=2: every other card has same rank (e.g., ABA pattern)."""
        hand = [H("3"), D("7"), S("3"), C("7")]
        assert prim_shifted_match(2, same_rank, hand) is True

    def test_shift_2_same_rank_fails(self):
        hand = [H("3"), D("7"), S("4"), C("7")]
        assert prim_shifted_match(2, same_rank, hand) is False

    def test_single_element_vacuously_true(self):
        """With only one element, no pairs to check -> vacuously True."""
        assert prim_shifted_match(1, same_suit, [H("A")]) is True

    def test_empty_list_vacuously_true(self):
        assert prim_shifted_match(1, same_suit, []) is True

    def test_k_equals_list_length(self):
        """k >= len means no valid pairs -> vacuously True."""
        hand = [H("2"), D("3")]
        assert prim_shifted_match(2, same_suit, hand) is True


# ---------------------------------------------------------------------------
# 3. prim_stride
# ---------------------------------------------------------------------------

class TestPrimStride:
    """prim_stride(k, xs) -> xs[::k]

    Returns every k-th element starting from index 0.
    Subsumes: odd-position / even-position extraction.
    """

    def test_stride_2(self):
        hand = [H("2"), D("3"), S("4"), C("5"), H("6")]
        assert prim_stride(2, hand) == [H("2"), S("4"), H("6")]

    def test_stride_3(self):
        hand = [H("2"), D("3"), S("4"), C("5"), H("6"), D("7")]
        assert prim_stride(3, hand) == [H("2"), C("5")]

    def test_stride_1_is_identity(self):
        hand = [H("A"), D("K")]
        assert prim_stride(1, hand) == hand

    def test_stride_larger_than_list(self):
        hand = [H("2"), D("3"), S("4")]
        assert prim_stride(10, hand) == [H("2")]

    def test_empty_list(self):
        assert prim_stride(2, []) == []

    def test_works_with_plain_list(self):
        assert prim_stride(2, [1, 2, 3, 4, 5]) == [1, 3, 5]


# ---------------------------------------------------------------------------
# 4. prim_count_where
# ---------------------------------------------------------------------------

class TestPrimCountWhere:
    """prim_count_where(pred, xs) -> number of xs where pred is True.

    Subsumes: n_red, n_high, any counting-with-filter pattern.
    """

    def test_count_red_cards(self):
        hand = [H("2"), D("3"), S("4"), C("5"), H("6")]
        assert prim_count_where(is_red, hand) == 3

    def test_count_none(self):
        hand = [S("2"), C("3")]
        assert prim_count_where(is_red, hand) == 0

    def test_count_all(self):
        hand = [H("2"), D("3"), H("4")]
        assert prim_count_where(is_red, hand) == 3

    def test_empty_list(self):
        assert prim_count_where(is_red, []) == 0

    def test_with_lambda(self):
        hand = [H("2"), D("10"), S("J"), C("Q"), H("A")]
        high = prim_count_where(lambda c: RANK_VALUES[c.rank] >= 10, hand)
        assert high == 4

    def test_works_with_plain_list(self):
        assert prim_count_where(lambda x: x > 3, [1, 2, 3, 4, 5]) == 2


# ---------------------------------------------------------------------------
# 5. prim_sorted_counts
# ---------------------------------------------------------------------------

class TestPrimSortedCounts:
    """prim_sorted_counts(key_fn, xs) -> group by key_fn, return counts descending.

    Subsumes: n_unique (len of result), most_common_count (first element),
    is_flush (result == [n]), has_pair (any count >= 2), etc.
    """

    def test_all_same_suit(self):
        """Flush: all hearts -> [5]"""
        hand = [H("2"), H("3"), H("4"), H("5"), H("6")]
        assert prim_sorted_counts(get_suit, hand) == [5]

    def test_mixed_suits(self):
        hand = [H("2"), H("3"), D("4"), D("5"), S("6")]
        result = prim_sorted_counts(get_suit, hand)
        assert result == [2, 2, 1]

    def test_all_different_ranks(self):
        hand = [H("2"), D("3"), S("4")]
        assert prim_sorted_counts(get_rank, hand) == [1, 1, 1]

    def test_pair(self):
        hand = [H("2"), D("2"), S("4")]
        result = prim_sorted_counts(get_rank, hand)
        assert result[0] == 2  # most common count is 2

    def test_three_of_a_kind(self):
        hand = [H("K"), D("K"), S("K"), C("2")]
        assert prim_sorted_counts(get_rank, hand) == [3, 1]

    def test_empty_list(self):
        assert prim_sorted_counts(get_suit, []) == []

    def test_n_unique_via_len(self):
        """len(sorted_counts) gives number of unique groups."""
        hand = [H("2"), D("3"), S("4"), C("5")]
        assert len(prim_sorted_counts(get_rank, hand)) == 4

    def test_works_with_plain_list(self):
        result = prim_sorted_counts(lambda x: x % 2, [1, 2, 3, 4, 5])
        assert result == [3, 2]  # 3 odds, 2 evens
