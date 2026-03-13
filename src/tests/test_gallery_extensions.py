"""Tests for v5 gallery extension primitives.

Each test verifies that the new primitive can express a previously-inexpressible
gallery rule. See docs/PRIMITIVE_DESIGN_DECISIONS.md for design rationale.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Card, Suit, Rank, RANK_VALUES, card_color, Color
from dreamcoder_core.primitives import build_primitives

# Build primitive dictionary once
_PRIMS = {p.name: p.value for p in build_primitives()}


# =========================================================================
# Helper: build hands from compact notation
# =========================================================================

def _hand(*specs):
    """Build a hand from (suit_char, rank_value) tuples.
    suit_char: 'S','H','D','C'; rank_value: 2-14 (14=Ace)"""
    suit_map = {'S': Suit.SPADES, 'H': Suit.HEARTS, 'D': Suit.DIAMONDS, 'C': Suit.CLUBS}
    val_to_rank = {v: r for r, v in RANK_VALUES.items()}
    return [Card(suit_map[s], val_to_rank[v]) for s, v in specs]


# =========================================================================
# sort_by_rank tests
# =========================================================================

def test_sort_by_rank_orders_cards():
    """sort_by_rank should order cards by rank value ascending."""
    hand = _hand(('H', 7), ('S', 3), ('D', 14), ('C', 5), ('H', 2), ('S', 10))
    sorted_hand = _PRIMS['sort_by_rank'](hand)
    vals = [RANK_VALUES[c.rank] for c in sorted_hand]
    assert vals == [2, 3, 5, 7, 10, 14]


def test_sort_by_rank_enables_straight_detection():
    """Can detect a 5-card straight using sort_by_rank + adjacent_pairs."""
    # 3,4,5,6,7 scattered with an extra card
    hand = _hand(('H', 5), ('S', 3), ('D', 7), ('C', 4), ('H', 6), ('S', 10))
    sorted_hand = _PRIMS['sort_by_rank'](hand)
    # Take first 5 sorted cards, check consecutive rank differences = 1
    taken = _PRIMS['take'](5)(sorted_hand)
    pairs = _PRIMS['adjacent_pairs'](taken)
    rank_val = _PRIMS['rank_val']
    head = _PRIMS['head']
    last = _PRIMS['last']
    sub = _PRIMS['-']
    all_fn = _PRIMS['all']
    eq = _PRIMS['eq']
    result = all_fn(lambda p: eq(sub(rank_val(last(p)))(rank_val(head(p))))(1))(pairs)
    assert result is True


def test_sort_enables_ap_anywhere():
    """Can detect AP-anywhere: sort, take adjacent triples, check step=1."""
    # Hand with 5,6,7 scattered among other ranks
    hand = _hand(('H', 11), ('S', 5), ('D', 7), ('C', 13), ('H', 6), ('S', 2))
    sorted_hand = _PRIMS['sort_by_rank'](hand)
    # Check if any 3 consecutive sorted cards form step-1 AP
    pairs = _PRIMS['adjacent_pairs']([RANK_VALUES[c.rank] for c in sorted_hand])
    diffs = _PRIMS['map'](lambda p: _PRIMS['-'](_PRIMS['last'](p))(_PRIMS['head'](p)))(pairs)
    diff_pairs = _PRIMS['adjacent_pairs'](diffs)
    has_ap = _PRIMS['any'](lambda dp: _PRIMS['all'](lambda x: _PRIMS['eq'](x)(1))(dp))(diff_pairs)
    assert has_ap is True


# =========================================================================
# max_suit_count tests
# =========================================================================

def test_max_suit_count_basic():
    """max_suit_count returns count of most frequent suit."""
    hand = _hand(('H', 2), ('H', 3), ('H', 4), ('S', 5), ('D', 6), ('C', 7))
    assert _PRIMS['max_suit_count'](hand) == 3  # 3 hearts


def test_max_suit_count_enables_three_or_more_same_suit():
    """three_or_more_same_suit = ge (max_suit_count hand) 3."""
    hand_yes = _hand(('S', 2), ('S', 3), ('S', 4), ('H', 5), ('D', 6), ('C', 7))
    hand_no = _hand(('S', 2), ('S', 3), ('H', 4), ('H', 5), ('D', 6), ('C', 7))
    ge = _PRIMS['ge']
    msc = _PRIMS['max_suit_count']
    assert ge(msc(hand_yes))(3) is True
    assert ge(msc(hand_no))(3) is False


# =========================================================================
# n_repeated_ranks / n_repeated_suits tests
# =========================================================================

def test_n_repeated_ranks_two_pairs():
    """n_repeated_ranks counts how many ranks appear ≥2 times."""
    # Two pairs: 3s and 7s
    hand = _hand(('H', 3), ('S', 3), ('D', 7), ('C', 7), ('H', 10), ('S', 14))
    assert _PRIMS['n_repeated_ranks'](hand) == 2


def test_n_repeated_ranks_no_pairs():
    """All distinct ranks → 0 repeated."""
    hand = _hand(('H', 2), ('S', 3), ('D', 4), ('C', 5), ('H', 6), ('S', 7))
    assert _PRIMS['n_repeated_ranks'](hand) == 0


def test_n_repeated_suits_basic():
    """n_repeated_suits counts suits appearing ≥2 times."""
    hand = _hand(('H', 2), ('H', 3), ('S', 4), ('S', 5), ('D', 6), ('C', 7))
    assert _PRIMS['n_repeated_suits'](hand) == 2  # hearts×2, spades×2


def test_n_repeated_ranks_enables_two_pairs_rule():
    """two_pairs_ranks = ge (n_repeated_ranks hand) 2."""
    hand_yes = _hand(('H', 3), ('S', 3), ('D', 7), ('C', 7), ('H', 10), ('S', 14))
    hand_no = _hand(('H', 3), ('S', 3), ('D', 7), ('C', 8), ('H', 10), ('S', 14))
    ge = _PRIMS['ge']
    nrr = _PRIMS['n_repeated_ranks']
    assert ge(nrr(hand_yes))(2) is True
    assert ge(nrr(hand_no))(2) is False


# =========================================================================
# running_sum tests
# =========================================================================

def test_running_sum_basic():
    """running_sum computes cumulative sums of a card→int mapping."""
    hand = _hand(('H', 2), ('S', 3), ('D', 4), ('C', 5), ('H', 6), ('S', 7))
    # Map each card to its rank value
    rs = _PRIMS['running_sum'](_PRIMS['rank_val'])(hand)
    assert rs == [2, 5, 9, 14, 20, 27]


def test_running_sum_bracket_matching():
    """running_sum enables bracket matching: ♠→+1, ♣→-1, check never negative and ends at 0."""
    # Valid brackets: ♠♠♣♣ (nested)
    valid = _hand(('S', 2), ('S', 3), ('C', 4), ('C', 5), ('H', 6), ('H', 7))
    # Map: ♠→+1, ♣→-1, others→0
    def bracket_a(c):
        if c.suit == Suit.SPADES: return 1
        if c.suit == Suit.CLUBS: return -1
        return 0
    rs = _PRIMS['running_sum'](bracket_a)(valid)
    # Running sum should never go negative, and... well it ends at 0 for the ♠/♣ part
    # [1, 2, 1, 0, 0, 0]
    assert rs == [1, 2, 1, 0, 0, 0]
    # Check: all ≥ 0 and last = 0
    assert all(x >= 0 for x in rs)
    assert rs[-1] == 0

    # Invalid: ♣ before ♠ (unbalanced)
    invalid = _hand(('C', 2), ('S', 3), ('S', 4), ('C', 5), ('H', 6), ('H', 7))
    rs_bad = _PRIMS['running_sum'](bracket_a)(invalid)
    # [-1, 0, 1, 0, 0, 0] — goes negative at position 0
    assert rs_bad[0] < 0


def test_running_sum_interleaved_brackets():
    """running_sum can track two independent bracket types."""
    # ♠♥♦♣ = type-A: ♠+1,♣-1; type-B: ♥+1,♦-1
    hand = _hand(('S', 2), ('H', 3), ('D', 4), ('C', 5), ('S', 6), ('C', 7))
    def type_a(c):
        if c.suit == Suit.SPADES: return 1
        if c.suit == Suit.CLUBS: return -1
        return 0
    def type_b(c):
        if c.suit == Suit.HEARTS: return 1
        if c.suit == Suit.DIAMONDS: return -1
        return 0
    rs_a = _PRIMS['running_sum'](type_a)(hand)
    rs_b = _PRIMS['running_sum'](type_b)(hand)
    # Type A: [1, 1, 1, 0, 1, 0] — balanced
    # Type B: [0, 1, 0, 0, 0, 0] — balanced
    assert rs_a[-1] == 0
    assert rs_b[-1] == 0
    assert all(x >= 0 for x in rs_a)
    assert all(x >= 0 for x in rs_b)


# =========================================================================
# suit_to_int tests
# =========================================================================

def test_suit_to_int_ordering():
    """suit_to_int maps to gallery experiment ordering: D=4, S=3, C=2, H=1."""
    sti = _PRIMS['suit_to_int']
    assert sti(Suit.DIAMONDS) == 4
    assert sti(Suit.SPADES) == 3
    assert sti(Suit.CLUBS) == 2
    assert sti(Suit.HEARTS) == 1


def test_suit_to_int_enables_monotonicity():
    """Can check suits_nonincreasing via map suit_to_int + adjacent_pairs + all ge.
    Gallery ordering: D=4, S=3, C=2, H=1."""
    # ♦♠♠♣♣♥ — nonincreasing: [4,3,3,2,2,1]
    hand = _hand(('D', 2), ('S', 3), ('S', 4), ('C', 5), ('C', 6), ('H', 7))
    sti = _PRIMS['suit_to_int']
    get_suit = _PRIMS['get_suit']
    suit_vals = _PRIMS['map'](lambda c: sti(get_suit(c)))(hand)
    pairs = _PRIMS['adjacent_pairs'](suit_vals)
    ge = _PRIMS['ge']
    head = _PRIMS['head']
    last = _PRIMS['last']
    nonincreasing = _PRIMS['all'](lambda p: ge(head(p))(last(p)))(pairs)
    assert nonincreasing is True

    # ♦♥♠♣♣♥ — NOT nonincreasing (♥=1 then ♠=3 increases)
    hand2 = _hand(('D', 2), ('H', 3), ('S', 4), ('C', 5), ('C', 6), ('H', 7))
    suit_vals2 = _PRIMS['map'](lambda c: sti(get_suit(c)))(hand2)
    pairs2 = _PRIMS['adjacent_pairs'](suit_vals2)
    nonincreasing2 = _PRIMS['all'](lambda p: ge(head(p))(last(p)))(pairs2)
    assert nonincreasing2 is False


# =========================================================================
# signum tests
# =========================================================================

def test_signum_values():
    """signum returns -1, 0, +1."""
    sig = _PRIMS['signum']
    assert sig(-42) == -1
    assert sig(0) == 0
    assert sig(17) == 1


def test_signum_enables_zigzag_detection():
    """Can detect zigzag via signum of rank differences: consecutive signs must differ."""
    # Zigzag: 3,8,2,9,4,7 — diffs: +5,-6,+7,-5,+3 — signs: +,-,+,-,+
    hand = _hand(('H', 3), ('S', 8), ('D', 2), ('C', 9), ('H', 4), ('S', 7))
    rank_val = _PRIMS['rank_val']
    ranks = _PRIMS['map'](rank_val)(hand)
    pairs = _PRIMS['adjacent_pairs'](ranks)
    diffs = _PRIMS['map'](lambda p: _PRIMS['-'](_PRIMS['last'](p))(_PRIMS['head'](p)))(pairs)
    signs = _PRIMS['map'](_PRIMS['signum'])(diffs)
    # Signs should alternate: [1, -1, 1, -1, 1]
    assert signs == [1, -1, 1, -1, 1]
    # Check alternation: adjacent sign pairs should sum to 0
    sign_pairs = _PRIMS['adjacent_pairs'](signs)
    alternating = _PRIMS['all'](
        lambda p: _PRIMS['eq'](_PRIMS['+'](_PRIMS['head'](p))(_PRIMS['last'](p)))(0)
    )(sign_pairs)
    assert alternating is True

    # Non-zigzag: 3,5,8,2,9,4 — diffs: +2,+3,-6,+7,-5 — signs: +,+,-,+,-
    hand2 = _hand(('H', 3), ('S', 5), ('D', 8), ('C', 2), ('H', 9), ('S', 4))
    ranks2 = _PRIMS['map'](rank_val)(hand2)
    diffs2 = _PRIMS['map'](lambda p: _PRIMS['-'](_PRIMS['last'](p))(_PRIMS['head'](p)))(_PRIMS['adjacent_pairs'](ranks2))
    signs2 = _PRIMS['map'](_PRIMS['signum'])(diffs2)
    sign_pairs2 = _PRIMS['adjacent_pairs'](signs2)
    alternating2 = _PRIMS['all'](
        lambda p: _PRIMS['eq'](_PRIMS['+'](_PRIMS['head'](p))(_PRIMS['last'](p)))(0)
    )(sign_pairs2)
    assert alternating2 is False
