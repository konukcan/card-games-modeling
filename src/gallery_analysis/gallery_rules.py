"""
All 61 gallery rules ported from card-games/rule-gallery/gallery-rules.js.

Each rule is a function: Hand -> bool
where Hand = List[Card] (6 cards).

These are ground truth predicates for the Bayesian rule induction analysis
(MCMC program search in gallery_analysis/). They are NOT expressed in the
DSL — they serve as reference evaluators to check whether enumerated
hypotheses match the true rule.

Rules are organized by difficulty group (1=easy, 2=medium, 3=hard).

NOTE: This file is distinct from src/rules/catalogue.py, which defines 55
rules for the DreamCoder wake-sleep experiments. That catalogue uses
CompositionNode trees and DSL primitives. This file uses raw Python
predicates. The two files serve different subsystems and should not be
confused.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import List, Dict, Callable, Tuple, Set
from rules.cards import (
    Card, Hand, Suit, Rank, RANK_VALUES,
    card_color, Color, suit_to_color
)

# Convenience: rank value lookup
def rv(card: Card) -> int:
    """Get numeric rank value (2-14) for a card."""
    return RANK_VALUES[card.rank]

# Convenience: color string for a card
def color(card: Card) -> str:
    """Get color as string 'RED' or 'BLACK'."""
    return card_color(card).value

# Convenience: suit counts
def suit_counts(hand: Hand) -> Dict[Suit, int]:
    counts = {s: 0 for s in Suit}
    for c in hand:
        counts[c.suit] += 1
    return counts

# Convenience: color counts
def color_counts(hand: Hand) -> Dict[str, int]:
    counts = {"RED": 0, "BLACK": 0}
    for c in hand:
        counts[color(c)] += 1
    return counts

# Convenience: halves
def halves(hand: Hand) -> Tuple[Hand, Hand]:
    mid = len(hand) // 2
    return hand[:mid], hand[len(hand) - mid:]


# =====================================================================
#  GROUP 1: Simple, single-feature rules
# =====================================================================

def all_red(hand: Hand) -> bool:
    """All cards are red (hearts or diamonds)."""
    return all(color(c) == "RED" for c in hand)

def all_clubs_or_hearts(hand: Hand) -> bool:
    """Every card is a club or a heart."""
    return all(c.suit in {Suit.CLUBS, Suit.HEARTS} for c in hand)

def all_same_suit(hand: Hand) -> bool:
    """All six cards share the same suit."""
    return all(c.suit == hand[0].suit for c in hand)

def all_4s_or_queens(hand: Hand) -> bool:
    """Every card is a 4 or a Queen."""
    return all(c.rank in {Rank.FOUR, Rank.QUEEN} for c in hand)

def all_4s_8s_or_9s(hand: Hand) -> bool:
    """Every card is a 4, 8, or 9."""
    return all(c.rank in {Rank.FOUR, Rank.EIGHT, Rank.NINE} for c in hand)

def triple_2s_pos234(hand: Hand) -> bool:
    """Positions 2, 3, and 4 (1-indexed) are all 2s."""
    return (len(hand) >= 4 and
            hand[1].rank == Rank.TWO and
            hand[2].rank == Rank.TWO and
            hand[3].rank == Rank.TWO)

def triple_any_pos345(hand: Hand) -> bool:
    """Positions 3, 4, and 5 (1-indexed) share the same rank."""
    return (len(hand) >= 5 and
            hand[2].rank == hand[3].rank and
            hand[3].rank == hand[4].rank)

def four_of_a_kind_adjacent(hand: Hand) -> bool:
    """Four consecutive cards share the same rank."""
    for i in range(len(hand) - 3):
        if (hand[i].rank == hand[i+1].rank ==
            hand[i+2].rank == hand[i+3].rank):
            return True
    return False

def all_but_one_same_color(hand: Hand) -> bool:
    """All cards except at most one are the same color."""
    cc = color_counts(hand)
    return min(cc["RED"], cc["BLACK"]) <= 1

def three_or_more_same_suit(hand: Hand) -> bool:
    """At least 3 cards share the same suit."""
    sc = suit_counts(hand)
    return max(sc.values()) >= 3

def all_same_color(hand: Hand) -> bool:
    """All cards share the same color (all red or all black)."""
    c0 = color(hand[0])
    return all(color(c) == c0 for c in hand)

def pair_jacks_pos45(hand: Hand) -> bool:
    """Positions 4 and 5 (1-indexed) are both Jacks."""
    return len(hand) >= 5 and hand[3].rank == Rank.JACK and hand[4].rank == Rank.JACK

def triple_3s_adjacent(hand: Hand) -> bool:
    """Three consecutive cards are all 3s."""
    for i in range(len(hand) - 2):
        if hand[i].rank == Rank.THREE and hand[i+1].rank == Rank.THREE and hand[i+2].rank == Rank.THREE:
            return True
    return False

def four_kind_adjacent_any(hand: Hand) -> bool:
    """Four positions share the same rank (any position, not necessarily adjacent)."""
    from collections import Counter
    rank_counts = Counter(c.rank for c in hand)
    return any(count >= 4 for count in rank_counts.values())

def three_clubs_adjacent(hand: Hand) -> bool:
    """Three consecutive cards are all clubs."""
    for i in range(len(hand) - 2):
        if hand[i].suit == Suit.CLUBS and hand[i+1].suit == Suit.CLUBS and hand[i+2].suit == Suit.CLUBS:
            return True
    return False

def every_other_ace(hand: Hand) -> bool:
    """Positions 1, 3, 5 (1-indexed) are all Aces."""
    return (len(hand) >= 5 and
            hand[0].rank == Rank.ACE and
            hand[2].rank == Rank.ACE and
            hand[4].rank == Rank.ACE)

def pos135_same_rank(hand: Hand) -> bool:
    """Positions 1, 3, 5 (1-indexed) share the same rank."""
    return (len(hand) >= 5 and
            hand[0].rank == hand[2].rank and
            hand[2].rank == hand[4].rank)

def left_red_right_black(hand: Hand) -> bool:
    """Left half is all red, right half is all black."""
    left, right = halves(hand)
    return (all(color(c) == "RED" for c in left) and
            all(color(c) == "BLACK" for c in right))

def some_half_red_other_black(hand: Hand) -> bool:
    """One half is all red and the other half is all black (either direction)."""
    left, right = halves(hand)
    lr = all(color(c) == "RED" for c in left) and all(color(c) == "BLACK" for c in right)
    rl = all(color(c) == "BLACK" for c in left) and all(color(c) == "RED" for c in right)
    return lr or rl

def both_halves_uniform_suit(hand: Hand) -> bool:
    """Each half has all cards of the same suit (halves may differ)."""
    left, right = halves(hand)
    return (all(c.suit == left[0].suit for c in left) and
            all(c.suit == right[0].suit for c in right))


# =====================================================================
#  GROUP 2: Multi-feature, counting, positional
# =====================================================================

def all_even(hand: Hand) -> bool:
    """Every card is an even number (2,4,6,8,10 — no face cards)."""
    return all(2 <= rv(c) <= 10 and rv(c) % 2 == 0 for c in hand)

def all_odd(hand: Hand) -> bool:
    """Every card is an odd number (3,5,7,9 — no face cards)."""
    return all(3 <= rv(c) <= 9 and rv(c) % 2 == 1 for c in hand)

def pair_5s_adjacent(hand: Hand) -> bool:
    """Two adjacent cards are both 5s."""
    for i in range(len(hand) - 1):
        if hand[i].rank == Rank.FIVE and hand[i+1].rank == Rank.FIVE:
            return True
    return False

def triple_any_adjacent(hand: Hand) -> bool:
    """Three consecutive cards share the same rank (any rank)."""
    for i in range(len(hand) - 2):
        if hand[i].rank == hand[i+1].rank == hand[i+2].rank:
            return True
    return False

def three_spades(hand: Hand) -> bool:
    """At least three of the six cards are spades."""
    return suit_counts(hand)[Suit.SPADES] >= 3

def three_any_suit_adjacent(hand: Hand) -> bool:
    """Three consecutive cards share the same suit (any suit)."""
    for i in range(len(hand) - 2):
        if hand[i].suit == hand[i+1].suit == hand[i+2].suit:
            return True
    return False

def four_hearts_adjacent(hand: Hand) -> bool:
    """Four consecutive cards are all hearts."""
    for i in range(len(hand) - 3):
        if all(hand[i+j].suit == Suit.HEARTS for j in range(4)):
            return True
    return False

def four_diamonds_anywhere(hand: Hand) -> bool:
    """At least 4 cards are diamonds."""
    return suit_counts(hand)[Suit.DIAMONDS] >= 4

def four_any_suit_adjacent(hand: Hand) -> bool:
    """Four consecutive cards share the same suit (any suit)."""
    for i in range(len(hand) - 3):
        if hand[i].suit == hand[i+1].suit == hand[i+2].suit == hand[i+3].suit:
            return True
    return False

def four_any_suit_anywhere(hand: Hand) -> bool:
    """At least 4 cards share the same suit."""
    sc = suit_counts(hand)
    return max(sc.values()) >= 4

def even_pos_red_odd_pos_black(hand: Hand) -> bool:
    """Even positions (2,4,6) are red, odd positions (1,3,5) are black (1-indexed)."""
    for i, c in enumerate(hand):
        pos = i + 1  # 1-indexed
        if pos % 2 == 0:
            if color(c) != "RED":
                return False
        else:
            if color(c) != "BLACK":
                return False
    return True

def colors_palindrome(hand: Hand) -> bool:
    """The color sequence reads the same forwards and backwards."""
    colors = [color(c) for c in hand]
    return colors == colors[::-1]

def halves_copy_colors(hand: Hand) -> bool:
    """Left and right halves have the same color sequence."""
    left, right = halves(hand)
    return [color(c) for c in left] == [color(c) for c in right]

def two_pairs_ranks(hand: Hand) -> bool:
    """The hand contains at least two pairs of matching ranks."""
    from collections import Counter
    rank_counts = Counter(c.rank for c in hand)
    pairs = sum(1 for count in rank_counts.values() if count >= 2)
    return pairs >= 2

def two_pairs_suits(hand: Hand) -> bool:
    """Two different suits each appear at least twice."""
    sc = suit_counts(hand)
    pairs = sum(1 for count in sc.values() if count >= 2)
    return pairs >= 2

def ap_len3_step1_anywhere(hand: Hand) -> bool:
    """Contains 3 cards (anywhere) forming an arithmetic progression with step 1."""
    vals = sorted(rv(c) for c in hand)
    for i in range(len(vals)):
        for j in range(i+1, len(vals)):
            for k in range(j+1, len(vals)):
                if vals[j] - vals[i] == 1 and vals[k] - vals[j] == 1:
                    return True
    return False

def halves_copy_ranks(hand: Hand) -> bool:
    """Left and right halves have the same rank sequence."""
    left, right = halves(hand)
    return [c.rank for c in left] == [c.rank for c in right]

def halves_copy_suits(hand: Hand) -> bool:
    """Left and right halves have the same suit sequence."""
    left, right = halves(hand)
    return [c.suit for c in left] == [c.suit for c in right]

def both_halves_have_pair_rank(hand: Hand) -> bool:
    """Each half contains at least one pair of matching ranks."""
    left, right = halves(hand)
    from collections import Counter
    left_pairs = any(v >= 2 for v in Counter(c.rank for c in left).values())
    right_pairs = any(v >= 2 for v in Counter(c.rank for c in right).values())
    return left_pairs and right_pairs

def both_halves_uniform_color(hand: Hand) -> bool:
    """Each half has all cards of the same color (halves may differ)."""
    left, right = halves(hand)
    return (all(color(c) == color(left[0]) for c in left) and
            all(color(c) == color(right[0]) for c in right))


# =====================================================================
#  GROUP 3: Complex structural patterns
# =====================================================================

def even_odd_pos_color_split(hand: Hand) -> bool:
    """Even positions same color, odd positions same color, colors differ."""
    evens = [hand[i] for i in range(len(hand)) if (i+1) % 2 == 0]
    odds = [hand[i] for i in range(len(hand)) if (i+1) % 2 == 1]
    if not evens or not odds:
        return False
    even_color = color(evens[0])
    odd_color = color(odds[0])
    return (even_color != odd_color and
            all(color(c) == even_color for c in evens) and
            all(color(c) == odd_color for c in odds))

def strict_increasing(hand: Hand) -> bool:
    """Rank values are in strictly increasing order."""
    vals = [rv(c) for c in hand]
    return all(vals[i] < vals[i+1] for i in range(len(vals) - 1))

def blacks_before_reds(hand: Hand) -> bool:
    """All black cards appear before all red cards."""
    seen_red = False
    for c in hand:
        if color(c) == "RED":
            seen_red = True
        elif seen_red:
            return False
    return True

def adjacent_share_rank_or_suit(hand: Hand) -> bool:
    """Every pair of adjacent cards shares either rank or suit."""
    for i in range(len(hand) - 1):
        if hand[i].rank != hand[i+1].rank and hand[i].suit != hand[i+1].suit:
            return False
    return True

def ap_step1_len3_adj_ordered(hand: Hand) -> bool:
    """Three consecutive cards form an ascending AP with step 1 (in position order)."""
    for i in range(len(hand) - 2):
        a, b, c = rv(hand[i]), rv(hand[i+1]), rv(hand[i+2])
        if b == a + 1 and c == a + 2:
            return True
    return False

def ap_step2_len4_adj_ordered(hand: Hand) -> bool:
    """Four consecutive cards form an ascending AP with step 2 (in position order)."""
    for i in range(len(hand) - 3):
        a, b, c, d = rv(hand[i]), rv(hand[i+1]), rv(hand[i+2]), rv(hand[i+3])
        if b == a + 2 and c == a + 4 and d == a + 6:
            return True
    return False

def ap_step1_len3_adj(hand: Hand) -> bool:
    """Three consecutive cards, when sorted by rank, form an AP with step 1."""
    for i in range(len(hand) - 2):
        vals = sorted([rv(hand[i]), rv(hand[i+1]), rv(hand[i+2])])
        if vals[1] - vals[0] == 1 and vals[2] - vals[1] == 1:
            return True
    return False

def ap_step2_len4_adj(hand: Hand) -> bool:
    """Four consecutive cards, when sorted by rank, form an AP with step 2."""
    for i in range(len(hand) - 3):
        vals = sorted([rv(hand[i]), rv(hand[i+1]), rv(hand[i+2]), rv(hand[i+3])])
        if all(vals[j+1] - vals[j] == 2 for j in range(3)):
            return True
    return False

def straight5(hand: Hand) -> bool:
    """Contains 5 cards with consecutive rank values (any suits)."""
    from itertools import combinations
    for combo in combinations(range(len(hand)), 5):
        vals = sorted(rv(hand[i]) for i in combo)
        if all(vals[j+1] - vals[j] == 1 for j in range(4)):
            return True
    return False

def straight5_same_suit(hand: Hand) -> bool:
    """Contains 5 cards with consecutive ranks AND the same suit."""
    from itertools import combinations
    for combo in combinations(range(len(hand)), 5):
        cards = [hand[i] for i in combo]
        if not all(c.suit == cards[0].suit for c in cards):
            continue
        vals = sorted(rv(c) for c in cards)
        if all(vals[j+1] - vals[j] == 1 for j in range(4)):
            return True
    return False

def straight5_same_color(hand: Hand) -> bool:
    """Contains 5 cards with consecutive ranks AND the same color."""
    from itertools import combinations
    for combo in combinations(range(len(hand)), 5):
        cards = [hand[i] for i in combo]
        if not all(color(c) == color(cards[0]) for c in cards):
            continue
        vals = sorted(rv(c) for c in cards)
        if all(vals[j+1] - vals[j] == 1 for j in range(4)):
            return True
    return False

def ranks_palindrome(hand: Hand) -> bool:
    """Rank values read the same forwards and backwards."""
    vals = [rv(c) for c in hand]
    return vals == vals[::-1]

def skip2_same_rank_or_suit(hand: Hand) -> bool:
    """Every card at distance 2 shares rank or suit (i and i+2)."""
    for i in range(len(hand) - 2):
        if hand[i].rank != hand[i+2].rank and hand[i].suit != hand[i+2].suit:
            return False
    return True

def no_adjacent_same_suit(hand: Hand) -> bool:
    """No two adjacent cards share the same suit."""
    for i in range(len(hand) - 1):
        if hand[i].suit == hand[i+1].suit:
            return False
    return True

def radial_increasing(hand: Hand) -> bool:
    """Ranks increase outward from center: center (3,4) < middle (2,5) < outer (1,6)."""
    if len(hand) < 6:
        return False
    outer_min = min(rv(hand[0]), rv(hand[5]))
    mid_max = max(rv(hand[1]), rv(hand[4]))
    mid_min = min(rv(hand[1]), rv(hand[4]))
    center_max = max(rv(hand[2]), rv(hand[3]))
    return center_max < mid_min and mid_max < outer_min

def zigzag_ranks(hand: Hand) -> bool:
    """Ranks alternate between local minima and maxima (zigzag pattern)."""
    vals = [rv(c) for c in hand]
    if len(vals) < 3:
        return False
    for i in range(1, len(vals) - 1):
        if i % 2 == 1:
            if not (vals[i] > vals[i-1] and vals[i] > vals[i+1]):
                return False
        else:
            if not (vals[i] < vals[i-1] and vals[i] < vals[i+1]):
                return False
    return True

def suit_brackets_no_cross(hand: Hand) -> bool:
    """Suits form non-crossing brackets — each type nests only within itself.
    ♠↔♣ (type A) and ♥↔♦ (type B). An opener of type X is only allowed
    if the stack is empty or the top of the stack is also type X.
    e.g. (())[] valid, ([]) INVALID (B nested inside A)."""
    openers = {Suit.SPADES: "A", Suit.HEARTS: "B"}
    closers = {Suit.CLUBS: "A", Suit.DIAMONDS: "B"}
    stack = []
    for c in hand:
        if c.suit in openers:
            typ = openers[c.suit]
            # New opener must match the type already on the stack (no mixing).
            if stack and stack[-1] != typ:
                return False
            stack.append(typ)
        elif c.suit in closers:
            if not stack or stack[-1] != closers[c.suit]:
                return False
            stack.pop()
        else:
            return False
    return len(stack) == 0

def suit_brackets_nested(hand: Hand) -> bool:
    """Suits form properly nested brackets (Dyck word).
    ♠↔♣ and ♥↔♦ are bracket pairs. Standard bracket matching."""
    openers = {Suit.SPADES: "A", Suit.HEARTS: "B"}
    closers = {Suit.CLUBS: "A", Suit.DIAMONDS: "B"}
    stack = []
    for c in hand:
        if c.suit in openers:
            stack.append(openers[c.suit])
        elif c.suit in closers:
            if not stack or stack[-1] != closers[c.suit]:
                return False
            stack.pop()
    return len(stack) == 0

def suit_brackets_interleaved(hand: Hand) -> bool:
    """Two bracket types tracked independently — each must balance.
    ♠ opens type A, ♣ closes type A. ♥ opens type B, ♦ closes type B.
    Both counters must stay >= 0 and end at 0."""
    count_a = 0  # ♠/♣
    count_b = 0  # ♥/♦
    for c in hand:
        if c.suit == Suit.SPADES:
            count_a += 1
        elif c.suit == Suit.CLUBS:
            count_a -= 1
        elif c.suit == Suit.HEARTS:
            count_b += 1
        elif c.suit == Suit.DIAMONDS:
            count_b -= 1
        if count_a < 0 or count_b < 0:
            return False
    return count_a == 0 and count_b == 0

def right_half_diamonds(hand: Hand) -> bool:
    """The last three cards are all diamonds."""
    _, right = halves(hand)
    return all(c.suit == Suit.DIAMONDS for c in right)

def suits_nonincreasing(hand: Hand) -> bool:
    """Suits follow D≥S≥C≥H from left to right, stepping down by at most one level.
    Ordering: D=4, S=3, C=2, H=1. Non-increasing AND no jumps (diff >= -1)."""
    suit_val = {Suit.DIAMONDS: 4, Suit.SPADES: 3, Suit.CLUBS: 2, Suit.HEARTS: 1}
    vals = [suit_val[c.suit] for c in hand]
    return all(-1 <= vals[i+1] - vals[i] <= 0 for i in range(len(vals) - 1))


# =====================================================================
#  REGISTRY: Maps rule IDs to (predicate, group, answer)
# =====================================================================

GALLERY_RULES: Dict[str, Dict] = {}

def _register(rule_id: str, group: int, answer: str, predicate: Callable[[Hand], bool]):
    GALLERY_RULES[rule_id] = {
        "id": rule_id,
        "group": group,
        "answer": answer,
        "predicate": predicate,
    }

# Group 1
_register("all_red", 1, "All cards are red (hearts or diamonds)", all_red)
_register("all_clubs_or_hearts", 1, "Every card is a club or a heart", all_clubs_or_hearts)
_register("all_same_suit", 1, "All six cards share the same suit", all_same_suit)
_register("all_4s_or_queens", 1, "Every card is a 4 or a Queen", all_4s_or_queens)
_register("all_4s_8s_or_9s", 1, "Every card is a 4, 8, or 9", all_4s_8s_or_9s)
_register("triple_2s_pos234", 1, "Positions 2, 3, and 4 are all 2s", triple_2s_pos234)
_register("triple_any_pos345", 1, "Positions 3, 4, and 5 share the same rank", triple_any_pos345)
_register("four_of_a_kind_adjacent", 1, "Four consecutive cards share the same rank", four_of_a_kind_adjacent)
_register("all_but_one_same_color", 1, "All cards except at most one are the same color", all_but_one_same_color)
_register("three_or_more_same_suit", 1, "At least 3 cards share the same suit", three_or_more_same_suit)
_register("all_same_color", 1, "All cards share the same color", all_same_color)
_register("pair_jacks_pos45", 1, "Positions 4 and 5 are both Jacks", pair_jacks_pos45)
_register("triple_3s_adjacent", 1, "Three consecutive cards are all 3s", triple_3s_adjacent)
_register("four_kind_adjacent_any", 1, "Four positions share the same rank (any position)", four_kind_adjacent_any)
_register("three_clubs_adjacent", 1, "Three consecutive clubs in a row", three_clubs_adjacent)
_register("every_other_ace", 1, "Positions 1, 3, 5 are all Aces", every_other_ace)
_register("pos135_same_rank", 1, "Positions 1, 3, 5 share the same rank", pos135_same_rank)
_register("left_red_right_black", 1, "Left half red, right half black", left_red_right_black)
_register("some_half_red_other_black", 1, "One half all red, other half all black", some_half_red_other_black)
_register("both_halves_uniform_suit", 1, "Each half has uniform suit", both_halves_uniform_suit)

# Group 2
_register("all_even", 2, "Every card is even (2,4,6,8,10)", all_even)
_register("all_odd", 2, "Every card is odd (3,5,7,9)", all_odd)
_register("pair_5s_adjacent", 2, "Two adjacent cards are both 5s", pair_5s_adjacent)
_register("triple_any_adjacent", 2, "Three consecutive cards share the same rank", triple_any_adjacent)
_register("three_spades", 2, "At least 3 cards are spades", three_spades)
_register("three_any_suit_adjacent", 2, "Three consecutive cards share the same suit", three_any_suit_adjacent)
_register("four_hearts_adjacent", 2, "Four consecutive hearts", four_hearts_adjacent)
_register("four_diamonds_anywhere", 2, "At least 4 diamonds", four_diamonds_anywhere)
_register("four_any_suit_adjacent", 2, "Four consecutive cards share the same suit", four_any_suit_adjacent)
_register("four_any_suit_anywhere", 2, "At least 4 cards share the same suit", four_any_suit_anywhere)
_register("even_pos_red_odd_pos_black", 2, "Even positions red, odd positions black", even_pos_red_odd_pos_black)
_register("colors_palindrome", 2, "Color sequence is a palindrome", colors_palindrome)
_register("halves_copy_colors", 2, "Left and right halves share color sequence", halves_copy_colors)
_register("two_pairs_ranks", 2, "At least two pairs of matching ranks", two_pairs_ranks)
_register("two_pairs_suits", 2, "Two different suits each appear at least twice", two_pairs_suits)
_register("ap_len3_step1_anywhere", 2, "Contains 3 cards forming AP with step 1", ap_len3_step1_anywhere)
_register("halves_copy_ranks", 2, "Left and right halves share rank sequence", halves_copy_ranks)
_register("halves_copy_suits", 2, "Left and right halves share suit sequence", halves_copy_suits)
_register("both_halves_have_pair_rank", 2, "Each half has at least one rank pair", both_halves_have_pair_rank)
_register("both_halves_uniform_color", 2, "Each half has uniform color", both_halves_uniform_color)
_register("right_half_diamonds", 2, "The last three cards are all diamonds", right_half_diamonds)

# Group 3
_register("even_odd_pos_color_split", 3, "Even/odd positions have opposite uniform colors", even_odd_pos_color_split)
_register("strict_increasing", 3, "Ranks strictly increase left to right", strict_increasing)
_register("blacks_before_reds", 3, "All black cards come before all red cards", blacks_before_reds)
_register("adjacent_share_rank_or_suit", 3, "Every adjacent pair shares rank or suit", adjacent_share_rank_or_suit)
_register("ap_step1_len3_adj_ordered", 3, "3 consecutive ascending cards with step 1", ap_step1_len3_adj_ordered)
_register("ap_step2_len4_adj_ordered", 3, "4 consecutive ascending cards with step 2", ap_step2_len4_adj_ordered)
_register("ap_step1_len3_adj", 3, "3 consecutive cards forming AP step 1 (any order)", ap_step1_len3_adj)
_register("ap_step2_len4_adj", 3, "4 consecutive cards forming AP step 2 (any order)", ap_step2_len4_adj)
_register("straight5", 3, "5 cards with consecutive ranks", straight5)
_register("straight5_same_suit", 3, "5-card straight flush", straight5_same_suit)
_register("straight5_same_color", 3, "5-card straight same color", straight5_same_color)
_register("ranks_palindrome", 3, "Rank values form a palindrome", ranks_palindrome)
_register("skip2_same_rank_or_suit", 3, "Every card at distance 2 shares rank or suit", skip2_same_rank_or_suit)
_register("no_adjacent_same_suit", 3, "No adjacent cards share a suit", no_adjacent_same_suit)
_register("radial_increasing", 3, "Ranks increase outward from center", radial_increasing)
_register("zigzag_ranks", 3, "Ranks alternate peaks and valleys", zigzag_ranks)
_register("suit_brackets_no_cross", 3, "Suits form non-crossing nested brackets", suit_brackets_no_cross)
_register("suit_brackets_nested", 3, "Suits form properly nested brackets", suit_brackets_nested)
_register("suit_brackets_interleaved", 3, "Two bracket types balanced independently", suit_brackets_interleaved)
_register("suits_nonincreasing", 3, "Suits follow D≥S≥C≥H, stepping down at most one level", suits_nonincreasing)
