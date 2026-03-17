# Bayesian Rule Induction Engine — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Bayesian rule induction engine that enumerates hypotheses from the DSL, computes posteriors given gallery exemplar hands, and produces difficulty predictions, competing hypothesis rankings, and test hand scoring for all 60 gallery rules.

**Architecture:** A new `gallery_analysis/` module under `card-games-modelling/src/` that imports the existing `dreamcoder_core/` enumeration, grammar, and type system. Six files: gallery rules (Python predicates), exemplar loading, wrapped enumerator with pruning, hypothesis table with fingerprinting, Bayesian scorer, and a main analysis entry point.

**Tech Stack:** Python 3, existing dreamcoder_core (grammar.py, enumeration.py, primitives.py, type_system.py, program.py), JSON for I/O, no new dependencies.

**Design doc:** `docs/plans/2026-03-10-bayesian-rule-induction-design.md`

## Execution Guidelines

- Explain code as you write it — treat this as a learning opportunity
- Start simple, build up — get basic versions working before adding optimizations
- Test and verify each step before moving on
- Present 2+ options for major decisions, wait for selection
- Keep existing structure — don't modify dreamcoder_core files

---

## Phase 1: Minimum Viable Pipeline

### Task 1: Create gallery_analysis package skeleton

**Files:**
- Create: `src/gallery_analysis/__init__.py`
- Create: `src/gallery_analysis/gallery_rules.py`
- Create: `src/tests/test_gallery_rules.py`

**Step 1: Create package directory and __init__.py**

```bash
mkdir -p src/gallery_analysis
mkdir -p src/gallery_analysis/results
```

```python
# src/gallery_analysis/__init__.py
"""
Bayesian Rule Induction Engine for the Card Gallery Experiment.

Computes posterior distributions over hypotheses given gallery exemplar hands,
using the DreamCoder DSL as the hypothesis space and the size principle for
likelihood computation.
"""
```

**Step 2: Write the first 5 gallery rule tests**

These test the simplest Group 1 rules to establish the pattern. Each rule is a function `hand -> bool` using the existing `Card`, `Hand`, `Suit`, `Rank` types from `rules/cards.py`.

```python
# src/tests/test_gallery_rules.py
"""Tests for gallery rule predicates ported from gallery-rules.js."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Card, Hand, Suit, Rank, RANK_VALUES, card_color, Color, H, D, S, C

def test_all_red_positive():
    """6 red cards should satisfy all_red."""
    from gallery_analysis.gallery_rules import all_red
    hand = [H("A"), H("K"), D("Q"), D("J"), H("10"), D("9")]
    assert all_red(hand) is True

def test_all_red_negative():
    """A hand with a black card should fail all_red."""
    from gallery_analysis.gallery_rules import all_red
    hand = [H("A"), H("K"), D("Q"), S("J"), H("10"), D("9")]
    assert all_red(hand) is False

def test_all_same_suit_positive():
    """6 spades should satisfy all_same_suit."""
    from gallery_analysis.gallery_rules import all_same_suit
    hand = [S("A"), S("K"), S("Q"), S("J"), S("10"), S("9")]
    assert all_same_suit(hand) is True

def test_all_same_suit_negative():
    """Mixed suits should fail all_same_suit."""
    from gallery_analysis.gallery_rules import all_same_suit
    hand = [S("A"), H("K"), S("Q"), S("J"), S("10"), S("9")]
    assert all_same_suit(hand) is False

def test_strict_increasing_positive():
    """Ranks in ascending order should satisfy strict_increasing."""
    from gallery_analysis.gallery_rules import strict_increasing
    hand = [H("2"), S("5"), D("7"), C("9"), H("J"), S("A")]
    assert strict_increasing(hand) is True

def test_strict_increasing_negative():
    """Non-ascending ranks should fail strict_increasing."""
    from gallery_analysis.gallery_rules import strict_increasing
    hand = [H("A"), S("K"), D("Q"), C("J"), H("10"), S("9")]
    assert strict_increasing(hand) is False
```

**Step 3: Run tests to verify they fail**

```bash
cd src && python -m pytest tests/test_gallery_rules.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'gallery_analysis'` or `ImportError`.

**Step 4: Implement the first batch of gallery rules**

Port all 60 rules from `gallery-rules.js` into Python predicates. Each is a plain function taking a `Hand` (list of `Card`) and returning `bool`. Group them by difficulty level for readability.

```python
# src/gallery_analysis/gallery_rules.py
"""
All 60 gallery rules ported from card-games/rule-gallery/gallery-rules.js.

Each rule is a function: Hand -> bool
where Hand = List[Card] (6 cards).

These are ground truth predicates for the Bayesian rule induction analysis.
They are NOT expressed in the DSL — they serve as reference evaluators to
check whether enumerated hypotheses match the true rule.

Rules are organized by difficulty group (1=easy, 2=medium, 3=hard).
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
    """Four consecutive cards share the same rank (same as four_of_a_kind_adjacent)."""
    return four_of_a_kind_adjacent(hand)

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
    """Exactly 3 cards are spades."""
    return suit_counts(hand)[Suit.SPADES] == 3

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

def right_half_diamonds(hand: Hand) -> bool:
    """The right half of the hand is all diamonds."""
    _, right = halves(hand)
    return all(c.suit == Suit.DIAMONDS for c in right)

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
    """At least two suits appear exactly twice, and all four suits present per half."""
    sc = suit_counts(hand)
    pairs = sum(1 for count in sc.values() if count == 2)
    left, right = halves(hand)
    left_suits = {c.suit for c in left}
    right_suits = {c.suit for c in right}
    return pairs >= 2 and len(left_suits) == 3 and len(right_suits) == 3

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
    """Outer ring (positions 1,6) < middle ring (2,5) < inner ring (3,4)."""
    if len(hand) < 6:
        return False
    outer_max = max(rv(hand[0]), rv(hand[5]))
    mid_min = min(rv(hand[1]), rv(hand[4]))
    mid_max = max(rv(hand[1]), rv(hand[4]))
    inner_min = min(rv(hand[2]), rv(hand[3]))
    return outer_max < mid_min and mid_max < inner_min

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
    """Suits form properly nested non-crossing brackets.
    ♠↔♣ and ♥↔♦ are bracket pairs. Each opener must close before the next pair opens."""
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

def suits_nonincreasing(hand: Hand) -> bool:
    """Suit values (♠=4,♥=3,♦=2,♣=1) are non-increasing, and all four suits present."""
    suit_val = {Suit.SPADES: 4, Suit.HEARTS: 3, Suit.DIAMONDS: 2, Suit.CLUBS: 1}
    vals = [suit_val[c.suit] for c in hand]
    all_four = len(set(c.suit for c in hand)) == 4
    nonincreasing = all(vals[i] >= vals[i+1] for i in range(len(vals) - 1))
    return all_four and nonincreasing


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
_register("four_kind_adjacent_any", 1, "Four consecutive cards share the same rank", four_kind_adjacent_any)
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
_register("three_spades", 2, "Exactly 3 cards are spades", three_spades)
_register("three_any_suit_adjacent", 2, "Three consecutive cards share the same suit", three_any_suit_adjacent)
_register("four_hearts_adjacent", 2, "Four consecutive hearts", four_hearts_adjacent)
_register("four_diamonds_anywhere", 2, "At least 4 diamonds", four_diamonds_anywhere)
_register("four_any_suit_adjacent", 2, "Four consecutive cards share the same suit", four_any_suit_adjacent)
_register("four_any_suit_anywhere", 2, "At least 4 cards share the same suit", four_any_suit_anywhere)
_register("right_half_diamonds", 2, "Right half is all diamonds", right_half_diamonds)
_register("even_pos_red_odd_pos_black", 2, "Even positions red, odd positions black", even_pos_red_odd_pos_black)
_register("colors_palindrome", 2, "Color sequence is a palindrome", colors_palindrome)
_register("halves_copy_colors", 2, "Left and right halves share color sequence", halves_copy_colors)
_register("two_pairs_ranks", 2, "At least two pairs of matching ranks", two_pairs_ranks)
_register("two_pairs_suits", 2, "At least two suits appear exactly twice, 3 suits per half", two_pairs_suits)
_register("ap_len3_step1_anywhere", 2, "Contains 3 cards forming AP with step 1", ap_len3_step1_anywhere)
_register("halves_copy_ranks", 2, "Left and right halves share rank sequence", halves_copy_ranks)
_register("halves_copy_suits", 2, "Left and right halves share suit sequence", halves_copy_suits)
_register("both_halves_have_pair_rank", 2, "Each half has at least one rank pair", both_halves_have_pair_rank)
_register("both_halves_uniform_color", 2, "Each half has uniform color", both_halves_uniform_color)

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
_register("radial_increasing", 3, "Outer < middle < inner ring ranks", radial_increasing)
_register("zigzag_ranks", 3, "Ranks alternate peaks and valleys", zigzag_ranks)
_register("suit_brackets_no_cross", 3, "Suits form non-crossing nested brackets", suit_brackets_no_cross)
_register("suit_brackets_nested", 3, "Suits form properly nested brackets", suit_brackets_nested)
_register("suit_brackets_interleaved", 3, "Two bracket types balanced independently", suit_brackets_interleaved)
_register("suits_nonincreasing", 3, "Suit values non-increasing, all four suits present", suits_nonincreasing)
```

**Step 5: Run tests to verify they pass**

```bash
cd src && python -m pytest tests/test_gallery_rules.py -v
```

Expected: All 6 tests PASS.

**Step 6: Commit**

```bash
git add src/gallery_analysis/ src/tests/test_gallery_rules.py
git commit -m "feat: port all 60 gallery rules as Python predicates with registry"
```

---

### Task 2: Exemplar loading and probe set generation

**Files:**
- Create: `src/gallery_analysis/exemplars.py`
- Create: `src/tests/test_exemplars.py`

**Step 1: Write tests for exemplar loading**

```python
# src/tests/test_exemplars.py
"""Tests for exemplar loading and probe set generation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_load_exemplars_count():
    """Should load 60 rules from frozen-exemplars.json."""
    from gallery_analysis.exemplars import load_exemplars
    exemplars = load_exemplars()
    assert len(exemplars) == 60

def test_load_exemplars_hand_size():
    """Each rule should have 6 primary hands of 6 cards each."""
    from gallery_analysis.exemplars import load_exemplars
    exemplars = load_exemplars()
    first = exemplars[list(exemplars.keys())[0]]
    assert len(first["hands_primary"]) == 6
    assert all(len(h) == 6 for h in first["hands_primary"])

def test_load_exemplars_card_type():
    """Cards should be Card objects with proper suit and rank."""
    from gallery_analysis.exemplars import load_exemplars
    from rules.cards import Card
    exemplars = load_exemplars()
    first = exemplars[list(exemplars.keys())[0]]
    card = first["hands_primary"][0][0]
    assert isinstance(card, Card)

def test_exemplars_satisfy_rules():
    """Primary hands should satisfy their corresponding rule."""
    from gallery_analysis.exemplars import load_exemplars
    from gallery_analysis.gallery_rules import GALLERY_RULES
    exemplars = load_exemplars()
    # Test a few known rules
    for rule_id in ["all_red", "strict_increasing", "all_same_suit"]:
        if rule_id in exemplars and rule_id in GALLERY_RULES:
            predicate = GALLERY_RULES[rule_id]["predicate"]
            hands = exemplars[rule_id]["hands_primary"]
            for hand in hands:
                assert predicate(hand), f"{rule_id} exemplar failed: {hand}"

def test_generate_probe_set_size():
    """Probe set should have the requested number of hands."""
    from gallery_analysis.exemplars import generate_probe_set
    probes = generate_probe_set(200, seed=42)
    assert len(probes) == 200

def test_generate_probe_set_deterministic():
    """Same seed should produce same probe set."""
    from gallery_analysis.exemplars import generate_probe_set
    probes1 = generate_probe_set(100, seed=42)
    probes2 = generate_probe_set(100, seed=42)
    for h1, h2 in zip(probes1, probes2):
        assert all(c1 == c2 for c1, c2 in zip(h1, h2))
```

**Step 2: Run tests to verify they fail**

```bash
cd src && python -m pytest tests/test_exemplars.py -v
```

**Step 3: Implement exemplars.py**

```python
# src/gallery_analysis/exemplars.py
"""
Load frozen exemplar hands from the gallery experiment and generate probe sets.

The frozen exemplars are pre-generated hands stored in
card-games/rule-gallery/frozen-exemplars.json. Each rule has 6 "primary" hands
(shown to human participants) and 6 "reserve" hands (for LLM experiments).

The probe set is a deterministic collection of random hands used for
observational equivalence fingerprinting.
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Card, Hand, Suit, Rank, STR_TO_RANK

# Path to frozen exemplars JSON (relative to this file)
FROZEN_EXEMPLARS_PATH = (
    Path(__file__).parent.parent.parent.parent /
    "card-games" / "rule-gallery" / "frozen-exemplars.json"
)


def _parse_card(card_obj: Dict[str, str]) -> Card:
    """Convert a JSON card object {suit: "HEARTS", rank: "K"} to a Card."""
    suit = Suit(card_obj["suit"])
    rank = STR_TO_RANK[card_obj["rank"]]
    return Card(suit, rank)


def _parse_hand(hand_list: List[Dict[str, str]]) -> Hand:
    """Convert a JSON hand (list of card objects) to a Hand."""
    return [_parse_card(c) for c in hand_list]


def load_exemplars(path: Path = None) -> Dict[str, Dict[str, Any]]:
    """
    Load frozen exemplar hands from JSON.

    Returns a dict keyed by rule_id, each containing:
      - "hands_primary": List of 6 Hand objects (for human experiments)
      - "hands_reserve": List of 6 Hand objects (for LLM experiments)
      - "group": int (difficulty 1-3)
      - "answer": str (human-readable rule description)
    """
    if path is None:
        path = FROZEN_EXEMPLARS_PATH

    with open(path, "r") as f:
        data = json.load(f)

    exemplars = {}
    for entry in data["catalogue"]:
        rule_id = entry["id"]
        exemplars[rule_id] = {
            "hands_primary": [_parse_hand(h) for h in entry["hands_primary"]],
            "hands_reserve": [_parse_hand(h) for h in entry["hands_reserve"]],
            "group": entry["group"],
            "answer": entry["answer"],
        }

    return exemplars


def generate_probe_set(
    n_probes: int = 200,
    hand_size: int = 6,
    seed: int = 42
) -> List[Hand]:
    """
    Generate a deterministic set of random hands for fingerprinting.

    These hands are used to compute observational equivalence fingerprints:
    two hypotheses that produce the same boolean vector on the probe set
    are treated as extensionally equivalent.

    Uses sampling without replacement (each hand has 6 distinct cards)
    to match the gallery experiment's hand generation.
    """
    rng = random.Random(seed)
    deck = [Card(suit, rank) for suit in Suit for rank in Rank]

    probes = []
    for _ in range(n_probes):
        hand = rng.sample(deck, hand_size)
        probes.append(hand)

    return probes
```

**Step 4: Run tests to verify they pass**

```bash
cd src && python -m pytest tests/test_exemplars.py -v
```

Expected: All tests PASS (assuming frozen-exemplars.json exists at the expected path).

**Step 5: Commit**

```bash
git add src/gallery_analysis/exemplars.py src/tests/test_exemplars.py
git commit -m "feat: add exemplar loading from frozen JSON and probe set generation"
```

---

### Task 3: Hypothesis table with fingerprinting and equivalence classes

**Files:**
- Create: `src/gallery_analysis/hypothesis_table.py`
- Create: `src/tests/test_hypothesis_table.py`

**Step 1: Write tests**

```python
# src/tests/test_hypothesis_table.py
"""Tests for hypothesis table, fingerprinting, and equivalence classes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Card, Suit, Rank, H, D, S, C

def _make_probe_set():
    """Small probe set for testing."""
    from gallery_analysis.exemplars import generate_probe_set
    return generate_probe_set(n_probes=50, seed=99)

def test_fingerprint_identical_functions():
    """Two identical predicates should produce the same fingerprint."""
    from gallery_analysis.hypothesis_table import compute_fingerprint
    probes = _make_probe_set()
    pred_a = lambda h: all(c.suit == Suit.SPADES for c in h)
    pred_b = lambda h: len([c for c in h if c.suit != Suit.SPADES]) == 0
    assert compute_fingerprint(pred_a, probes) == compute_fingerprint(pred_b, probes)

def test_fingerprint_different_functions():
    """Two different predicates should produce different fingerprints."""
    from gallery_analysis.hypothesis_table import compute_fingerprint
    probes = _make_probe_set()
    pred_a = lambda h: all(c.suit == Suit.SPADES for c in h)
    pred_b = lambda h: all(c.suit == Suit.HEARTS for c in h)
    assert compute_fingerprint(pred_a, probes) != compute_fingerprint(pred_b, probes)

def test_hypothesis_table_deduplication():
    """Adding equivalent hypotheses should result in one equivalence class."""
    from gallery_analysis.hypothesis_table import HypothesisTable
    probes = _make_probe_set()
    table = HypothesisTable(probes)

    pred_a = lambda h: all(c.suit == Suit.SPADES for c in h)
    pred_b = lambda h: len([c for c in h if c.suit != Suit.SPADES]) == 0

    table.add("prog_a", pred_a, log_prior=-5.0)
    table.add("prog_b", pred_b, log_prior=-7.0)

    classes = table.get_equivalence_classes()
    assert len(classes) == 1
    # Canonical should be the one with higher prior (less negative log)
    assert classes[0]["canonical_program"] == "prog_a"
    assert classes[0]["n_expressions"] == 2

def test_hit_vector_computation():
    """Hit vector should correctly track which exemplars a hypothesis covers."""
    from gallery_analysis.hypothesis_table import HypothesisTable
    probes = _make_probe_set()
    table = HypothesisTable(probes)

    # A predicate that checks if first card is a heart
    pred = lambda h: h[0].suit == Suit.HEARTS

    exemplars = [
        [H("A"), S("K"), D("Q"), C("J"), H("10"), S("9")],  # first is heart -> True
        [S("A"), H("K"), D("Q"), C("J"), H("10"), S("9")],  # first is spade -> False
    ]

    table.add("first_heart", pred, log_prior=-5.0, exemplar_hands=exemplars)
    classes = table.get_equivalence_classes()
    assert classes[0]["hit_vector"] == [True, False]
    assert classes[0]["n_hits"] == 1
```

**Step 2: Run tests to verify they fail**

```bash
cd src && python -m pytest tests/test_hypothesis_table.py -v
```

**Step 3: Implement hypothesis_table.py**

```python
# src/gallery_analysis/hypothesis_table.py
"""
Hypothesis table: fingerprinting, equivalence classes, and hit tracking.

Each enumerated program (hypothesis) is:
1. Evaluated on a probe set to get a boolean fingerprint
2. Grouped into equivalence classes by fingerprint
3. Evaluated on exemplar hands to get a hit vector

Equivalence classes track:
- The canonical (shortest/most probable) program
- All alternative expressions
- The shared fingerprint, hit vector, and extension size estimate
"""
import hashlib
import random
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Hand, Card, Suit, Rank


def compute_fingerprint(predicate: Callable[[Hand], bool], probes: List[Hand]) -> str:
    """
    Compute a fingerprint for a predicate by evaluating it on probe hands.

    Returns a hex hash of the boolean output vector. Two predicates with the
    same hash are treated as extensionally equivalent.
    """
    bits = []
    for hand in probes:
        try:
            result = predicate(hand)
            bits.append("1" if result else "0")
        except Exception:
            bits.append("E")

    bit_string = "".join(bits)
    return hashlib.sha256(bit_string.encode()).hexdigest()


def estimate_extension_size(
    predicate: Callable[[Hand], bool],
    n_samples: int = 100_000,
    hand_size: int = 6,
    seed: int = 123
) -> Tuple[int, float]:
    """
    Estimate |extension(h)| via Monte Carlo sampling.

    Returns:
        (estimated_extension_size, base_rate)
    """
    rng = random.Random(seed)
    deck = [Card(suit, rank) for suit in Suit for rank in Rank]

    hits = 0
    for _ in range(n_samples):
        hand = rng.sample(deck, hand_size)
        try:
            if predicate(hand):
                hits += 1
        except Exception:
            pass

    base_rate = hits / n_samples
    # C(52, 6) = 20,358,520
    total_hands = 20_358_520
    estimated_size = int(base_rate * total_hands)

    return estimated_size, base_rate


@dataclass
class HypothesisEntry:
    """A single enumerated program/hypothesis."""
    program_str: str
    predicate: Callable[[Hand], bool]
    log_prior: float
    fingerprint: str
    hit_vector: Optional[List[bool]] = None
    n_hits: int = 0
    n_misses: int = 0


class HypothesisTable:
    """
    Manages all enumerated hypotheses, grouped by observational equivalence.

    Usage:
        table = HypothesisTable(probe_hands)
        table.add("program_str", predicate_fn, log_prior=-5.0, exemplar_hands=hands)
        ...
        classes = table.get_equivalence_classes()
    """

    def __init__(self, probes: List[Hand]):
        self.probes = probes
        # fingerprint -> list of HypothesisEntry
        self._classes: Dict[str, List[HypothesisEntry]] = {}
        self._total_added = 0
        self._total_deduplicated = 0

    def add(
        self,
        program_str: str,
        predicate: Callable[[Hand], bool],
        log_prior: float,
        exemplar_hands: Optional[List[Hand]] = None
    ) -> bool:
        """
        Add a hypothesis to the table.

        Returns True if this is a new equivalence class, False if deduplicated.
        """
        self._total_added += 1

        fp = compute_fingerprint(predicate, self.probes)

        hit_vector = None
        n_hits = 0
        n_misses = 0
        if exemplar_hands:
            hit_vector = []
            for hand in exemplar_hands:
                try:
                    result = predicate(hand)
                    hit_vector.append(result)
                    if result:
                        n_hits += 1
                    else:
                        n_misses += 1
                except Exception:
                    hit_vector.append(False)
                    n_misses += 1

        entry = HypothesisEntry(
            program_str=program_str,
            predicate=predicate,
            log_prior=log_prior,
            fingerprint=fp,
            hit_vector=hit_vector,
            n_hits=n_hits,
            n_misses=n_misses,
        )

        is_new = fp not in self._classes
        if is_new:
            self._classes[fp] = [entry]
        else:
            self._classes[fp].append(entry)
            self._total_deduplicated += 1

        return is_new

    def get_equivalence_classes(self) -> List[Dict[str, Any]]:
        """
        Return all equivalence classes, each with canonical program and statistics.

        Classes are sorted by canonical prior (most probable first).
        """
        classes = []
        for fp, entries in self._classes.items():
            # Canonical = highest prior (least negative log_prior)
            entries_sorted = sorted(entries, key=lambda e: -e.log_prior)
            canonical = entries_sorted[0]

            import math
            summed_prior = math.log(sum(math.exp(e.log_prior) for e in entries))

            classes.append({
                "canonical_program": canonical.program_str,
                "canonical_prior": canonical.log_prior,
                "summed_prior": summed_prior,
                "n_expressions": len(entries),
                "all_programs": [e.program_str for e in entries_sorted],
                "fingerprint": fp,
                "hit_vector": canonical.hit_vector,
                "n_hits": canonical.n_hits,
                "n_misses": canonical.n_misses,
                "predicate": canonical.predicate,
            })

        classes.sort(key=lambda c: -c["canonical_prior"])
        return classes

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "total_added": self._total_added,
            "total_deduplicated": self._total_deduplicated,
            "n_equivalence_classes": len(self._classes),
        }
```

**Step 4: Run tests**

```bash
cd src && python -m pytest tests/test_hypothesis_table.py -v
```

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add src/gallery_analysis/hypothesis_table.py src/tests/test_hypothesis_table.py
git commit -m "feat: add hypothesis table with fingerprinting and equivalence classes"
```

---

### Task 4: Bayesian scorer with dual priors and noisy likelihood

**Files:**
- Create: `src/gallery_analysis/bayesian_scorer.py`
- Create: `src/tests/test_bayesian_scorer.py`

**Step 1: Write tests**

```python
# src/tests/test_bayesian_scorer.py
"""Tests for Bayesian posterior computation."""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_strict_likelihood_all_hits():
    """Hypothesis covering all 6 exemplars should have nonzero likelihood."""
    from gallery_analysis.bayesian_scorer import strict_likelihood
    ll = strict_likelihood(n_hits=6, n_exemplars=6, extension_size=1000)
    assert ll < 0  # negative log-likelihood
    assert math.isfinite(ll)

def test_strict_likelihood_any_miss():
    """Hypothesis missing any exemplar should have -inf likelihood."""
    from gallery_analysis.bayesian_scorer import strict_likelihood
    ll = strict_likelihood(n_hits=5, n_exemplars=6, extension_size=1000)
    assert ll == float('-inf')

def test_noisy_likelihood_one_miss():
    """Hypothesis with 1 miss should have finite (but low) noisy likelihood."""
    from gallery_analysis.bayesian_scorer import noisy_likelihood
    ll = noisy_likelihood(n_hits=5, n_misses=1, extension_size=1000,
                          complement_size=20_358_520 - 1000, epsilon=0.05)
    assert math.isfinite(ll)
    assert ll < 0

def test_posterior_normalization():
    """Posteriors should sum to 1."""
    from gallery_analysis.bayesian_scorer import compute_posteriors
    classes = [
        {"canonical_prior": math.log(0.3), "summed_prior": math.log(0.3),
         "extension_size": 1000, "n_hits": 6, "n_misses": 0},
        {"canonical_prior": math.log(0.7), "summed_prior": math.log(0.7),
         "extension_size": 5000, "n_hits": 6, "n_misses": 0},
    ]
    result = compute_posteriors(classes, n_exemplars=6)
    total = sum(math.exp(r["posterior_canonical_strict"]) for r in result)
    assert abs(total - 1.0) < 1e-6

def test_posterior_tighter_wins():
    """With equal priors, tighter hypothesis (smaller extension) should win."""
    from gallery_analysis.bayesian_scorer import compute_posteriors
    classes = [
        {"canonical_prior": math.log(0.5), "summed_prior": math.log(0.5),
         "extension_size": 100, "n_hits": 6, "n_misses": 0},
        {"canonical_prior": math.log(0.5), "summed_prior": math.log(0.5),
         "extension_size": 10000, "n_hits": 6, "n_misses": 0},
    ]
    result = compute_posteriors(classes, n_exemplars=6)
    assert result[0]["posterior_canonical_strict"] > result[1]["posterior_canonical_strict"]
```

**Step 2: Run tests to verify they fail**

```bash
cd src && python -m pytest tests/test_bayesian_scorer.py -v
```

**Step 3: Implement bayesian_scorer.py**

```python
# src/gallery_analysis/bayesian_scorer.py
"""
Bayesian posterior computation over equivalence classes.

Computes P(h|D) for each equivalence class using:
  P(h|D) ∝ P(D|h) × P(h)

Supports:
  - Strict likelihood: P(D|h) = (1/|ext(h)|)^n if all exemplars match, else 0
  - Noisy likelihood: allows up to 1 miss with penalty epsilon
  - Dual priors: canonical (shortest program) and summed (Solomonoff-style)
  - Test hand scoring: P(positive | data) weighted by posterior
"""
import math
from typing import Dict, List, Optional, Any, Callable

from rules.cards import Hand

TOTAL_HANDS = 20_358_520  # C(52, 6)


def strict_likelihood(
    n_hits: int,
    n_exemplars: int,
    extension_size: int
) -> float:
    """
    Strict size-principle likelihood in log-space.

    P(D|h) = (1/|ext(h)|)^n if all n exemplars are hits, else 0.

    Returns log P(D|h). Returns -inf if any exemplar is a miss.
    """
    if n_hits < n_exemplars:
        return float('-inf')
    if extension_size <= 0:
        return float('-inf')
    return -n_exemplars * math.log(extension_size)


def noisy_likelihood(
    n_hits: int,
    n_misses: int,
    extension_size: int,
    complement_size: int,
    epsilon: float = 0.05
) -> float:
    """
    Noisy likelihood allowing misses, in log-space.

    For each exemplar independently:
      P(hand_i | h) = (1-ε)/|ext(h)|   if hit
                    = ε/|complement(h)|  if miss

    Returns log P(D|h).
    """
    if extension_size <= 0 or complement_size <= 0:
        return float('-inf')

    log_hit = math.log(1 - epsilon) - math.log(extension_size)
    log_miss = math.log(epsilon) - math.log(complement_size)

    return n_hits * log_hit + n_misses * log_miss


def _log_sum_exp(log_values: List[float]) -> float:
    """Numerically stable log-sum-exp."""
    if not log_values:
        return float('-inf')
    max_val = max(log_values)
    if max_val == float('-inf'):
        return float('-inf')
    return max_val + math.log(sum(math.exp(v - max_val) for v in log_values))


def compute_posteriors(
    classes: List[Dict[str, Any]],
    n_exemplars: int = 6,
    epsilon: float = 0.05,
) -> List[Dict[str, Any]]:
    """
    Compute posteriors for all equivalence classes under four configurations:
      - canonical prior × strict likelihood
      - canonical prior × noisy likelihood
      - summed prior × strict likelihood
      - summed prior × noisy likelihood

    Each class dict must contain:
      canonical_prior, summed_prior (log-space),
      extension_size (int), n_hits, n_misses

    Returns the same list with added posterior fields (log-space, normalized).
    """
    # Compute unnormalized log-posteriors for all four configurations
    configs = [
        ("canonical_strict", "canonical_prior", "strict"),
        ("canonical_noisy", "canonical_prior", "noisy"),
        ("summed_strict", "summed_prior", "strict"),
        ("summed_noisy", "summed_prior", "noisy"),
    ]

    for config_name, prior_key, likelihood_type in configs:
        log_posteriors = []
        for cls in classes:
            prior = cls[prior_key]
            ext = cls["extension_size"]
            complement = TOTAL_HANDS - ext

            if likelihood_type == "strict":
                ll = strict_likelihood(cls["n_hits"], n_exemplars, ext)
            else:
                ll = noisy_likelihood(cls["n_hits"], cls["n_misses"], ext, complement, epsilon)

            log_posteriors.append(prior + ll)

        # Normalize
        log_z = _log_sum_exp(log_posteriors)
        for i, cls in enumerate(classes):
            cls[f"posterior_{config_name}"] = log_posteriors[i] - log_z

    return classes


def posterior_entropy(classes: List[Dict], posterior_key: str) -> float:
    """
    Compute Shannon entropy of a posterior distribution (in bits).

    Higher entropy = more uncertainty = harder rule.
    """
    entropy = 0.0
    for cls in classes:
        log_p = cls[posterior_key]
        if log_p > float('-inf'):
            p = math.exp(log_p)
            if p > 0:
                entropy -= p * math.log2(p)
    return entropy


def score_test_hand(
    hand: Hand,
    classes: List[Dict],
    posterior_key: str = "posterior_canonical_strict"
) -> Dict[str, float]:
    """
    Score a test hand's classification difficulty.

    Returns:
      p_positive: P(hand is positive | gallery data)
      agreement_with_true: fraction of posterior mass agreeing with true classification
    """
    p_positive = 0.0
    for cls in classes:
        post = math.exp(cls[posterior_key])
        pred = cls["predicate"]
        try:
            if pred(hand):
                p_positive += post
        except Exception:
            pass

    return {
        "p_positive": p_positive,
        "p_negative": 1.0 - p_positive,
        "ambiguity": 1.0 - abs(2 * p_positive - 1.0),  # 0=clear, 1=maximally ambiguous
    }
```

**Step 4: Run tests**

```bash
cd src && python -m pytest tests/test_bayesian_scorer.py -v
```

Expected: All tests PASS.

**Step 5: Commit**

```bash
git add src/gallery_analysis/bayesian_scorer.py src/tests/test_bayesian_scorer.py
git commit -m "feat: add Bayesian scorer with dual priors, noisy likelihood, test hand scoring"
```

---

### Task 5: Enumerator wrapper (basic version — no pruning yet)

**Files:**
- Create: `src/gallery_analysis/enumerator.py`
- Create: `src/tests/test_enumerator.py`

This task creates a wrapper around the existing `TopDownEnumerator` that enumerates `hand -> bool` programs and feeds them into the hypothesis table. Phase 1 uses the basic enumerator without likelihood pruning (that comes in Phase 2).

**Step 1: Write tests**

```python
# src/tests/test_enumerator.py
"""Tests for the gallery enumerator wrapper."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_enumerate_produces_programs():
    """Should enumerate at least some hand -> bool programs at low depth."""
    from gallery_analysis.enumerator import enumerate_hypotheses
    results = enumerate_hypotheses(max_depth=4, max_programs=50, timeout=30)
    assert len(results) > 0
    # Each result should be (program_str, predicate_fn, log_prior)
    for prog_str, pred_fn, log_prior in results:
        assert isinstance(prog_str, str)
        assert callable(pred_fn)
        assert isinstance(log_prior, float)

def test_enumerate_programs_are_callable():
    """Enumerated predicates should be callable on a hand."""
    from gallery_analysis.enumerator import enumerate_hypotheses
    from rules.cards import H, D, S, C
    hand = [H("A"), D("K"), S("Q"), C("J"), H("10"), D("9")]

    results = enumerate_hypotheses(max_depth=3, max_programs=20, timeout=15)
    for prog_str, pred_fn, log_prior in results:
        result = pred_fn(hand)
        assert isinstance(result, bool), f"Program {prog_str} returned {type(result)}"
```

**Step 2: Run tests to verify they fail**

```bash
cd src && python -m pytest tests/test_enumerator.py -v
```

**Step 3: Implement enumerator.py**

```python
# src/gallery_analysis/enumerator.py
"""
Gallery-specific enumerator wrapper.

Wraps dreamcoder_core's TopDownEnumerator to enumerate hand -> bool programs,
convert them to callable predicates, and feed them into the hypothesis table.

Phase 1: Basic enumeration with observational equivalence deduplication.
Phase 2 will add: likelihood pruning, constant folding, dead code elimination.
"""
import sys
import time
import math
from pathlib import Path
from typing import List, Tuple, Callable, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Hand, Card, Suit, Rank, RANK_VALUES, card_color, Color

from dreamcoder_core.type_system import BOOL, HAND, Arrow
from dreamcoder_core.primitives import build_primitives
from dreamcoder_core.grammar import uniform_grammar
from dreamcoder_core.enumeration import TopDownEnumerator
from dreamcoder_core.program import Program


def _make_evaluator(program: Program) -> Callable[[Hand], bool]:
    """
    Convert an enumerated Program AST into a callable predicate.

    The program has type hand -> bool. We evaluate it by applying it
    to a hand and checking the result.
    """
    def predicate(hand: Hand) -> bool:
        try:
            result = program.evaluate([])(hand)
            return bool(result)
        except Exception:
            return False
    return predicate


def enumerate_hypotheses(
    max_depth: int = 6,
    max_programs: int = 10000,
    max_cost: float = 50.0,
    timeout: float = 300.0,
    grammar=None,
) -> List[Tuple[str, Callable[[Hand], bool], float]]:
    """
    Enumerate hand -> bool programs from the DSL.

    Returns list of (program_string, predicate_function, log_prior) tuples.
    Programs are yielded in order of increasing cost (decreasing prior).

    Args:
        max_depth: Maximum AST depth for enumeration
        max_programs: Maximum number of complete programs to yield
        max_cost: Maximum cost (-log probability) to explore
        timeout: Wall clock timeout in seconds
        grammar: Optional grammar to use (defaults to uniform over all primitives)
    """
    if grammar is None:
        primitives = build_primitives()
        grammar = uniform_grammar(primitives)

    request_type = Arrow(HAND, BOOL)

    enumerator = TopDownEnumerator(
        grammar=grammar,
        max_depth=max_depth,
        max_programs=max_programs,
    )

    results = []
    start = time.time()

    for program, log_prob in enumerator.enumerate(
        request_type=request_type,
        max_cost=max_cost,
        timeout_seconds=timeout,
    ):
        prog_str = str(program)
        pred_fn = _make_evaluator(program)
        results.append((prog_str, pred_fn, log_prob))

        if time.time() - start > timeout:
            break

    return results
```

**Step 4: Run tests**

```bash
cd src && python -m pytest tests/test_enumerator.py -v --timeout=60
```

Expected: Tests PASS (may take a few seconds for enumeration).

**Step 5: Commit**

```bash
git add src/gallery_analysis/enumerator.py src/tests/test_enumerator.py
git commit -m "feat: add enumerator wrapper for hand->bool program enumeration"
```

---

### Task 6: Main analysis pipeline (analyze.py)

**Files:**
- Create: `src/gallery_analysis/analyze.py`

**Step 1: Implement the end-to-end pipeline**

```python
# src/gallery_analysis/analyze.py
"""
Main entry point for the Bayesian rule induction analysis.

Runs the full pipeline:
1. Load exemplar hands and gallery rules
2. Generate probe set
3. Enumerate hypotheses from DSL
4. Build hypothesis table with fingerprinting
5. Estimate extension sizes
6. Compute posteriors
7. Output per-rule JSON results

Usage:
    cd src
    python -m gallery_analysis.analyze --rules all_red strict_increasing --depth 5
    python -m gallery_analysis.analyze --all --depth 6
"""
import argparse
import json
import math
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from gallery_analysis.gallery_rules import GALLERY_RULES
from gallery_analysis.exemplars import load_exemplars, generate_probe_set
from gallery_analysis.enumerator import enumerate_hypotheses
from gallery_analysis.hypothesis_table import HypothesisTable, estimate_extension_size
from gallery_analysis.bayesian_scorer import (
    compute_posteriors, posterior_entropy, score_test_hand
)


def analyze_rule(
    rule_id: str,
    exemplar_hands,
    probe_set,
    enumerated_programs,
    top_k: int = 10,
    n_mc_samples: int = 100_000,
    epsilon: float = 0.05,
) -> dict:
    """
    Run full Bayesian analysis for a single gallery rule.

    Args:
        rule_id: Gallery rule identifier
        exemplar_hands: List of 6 Hand objects (the gallery exemplars)
        probe_set: List of probe hands for fingerprinting
        enumerated_programs: List of (prog_str, pred_fn, log_prior) from enumerator
        top_k: Number of top equivalence classes to report
        n_mc_samples: Monte Carlo samples for extension size estimation
        epsilon: Noise parameter for noisy likelihood

    Returns:
        Dict with full analysis results
    """
    true_rule = GALLERY_RULES.get(rule_id)
    if true_rule is None:
        raise ValueError(f"Unknown rule: {rule_id}")

    print(f"  Building hypothesis table for {rule_id}...")
    table = HypothesisTable(probe_set)

    # Add all enumerated programs
    for prog_str, pred_fn, log_prior in enumerated_programs:
        table.add(prog_str, pred_fn, log_prior, exemplar_hands=exemplar_hands)

    print(f"    {table.stats}")

    # Get equivalence classes
    classes = table.get_equivalence_classes()

    # Filter to hypotheses with at most 1 miss (for near-hit analysis)
    viable_classes = [c for c in classes if c["n_misses"] <= 1]

    print(f"    {len(viable_classes)} viable classes (≤1 miss) out of {len(classes)} total")

    # Estimate extension sizes for viable classes
    print(f"    Estimating extension sizes...")
    for cls in viable_classes:
        ext_size, base_rate = estimate_extension_size(
            cls["predicate"], n_samples=n_mc_samples
        )
        cls["extension_size"] = max(ext_size, 1)  # avoid zero
        cls["base_rate"] = base_rate

    # Compute posteriors
    n_exemplars = len(exemplar_hands)
    viable_classes = compute_posteriors(viable_classes, n_exemplars=n_exemplars, epsilon=epsilon)

    # Sort by canonical strict posterior (primary ranking)
    viable_classes.sort(key=lambda c: -c["posterior_canonical_strict"])

    # Find the true rule's equivalence class
    true_pred = true_rule["predicate"]
    true_fp = None
    from gallery_analysis.hypothesis_table import compute_fingerprint
    true_fp = compute_fingerprint(true_pred, probe_set)

    true_rule_rank = None
    true_rule_posterior = None
    map_is_correct = False

    for i, cls in enumerate(viable_classes):
        if cls["fingerprint"] == true_fp:
            true_rule_rank = i + 1
            true_rule_posterior = cls["posterior_canonical_strict"]
            if i == 0:
                map_is_correct = True
            break

    # Build top-K report
    top_k_report = []
    for cls in viable_classes[:top_k]:
        entry = {
            "canonical_program": cls["canonical_program"],
            "n_expressions": cls["n_expressions"],
            "extension_size": cls["extension_size"],
            "base_rate": cls["base_rate"],
            "n_hits": cls["n_hits"],
            "n_misses": cls["n_misses"],
            "canonical_prior": cls["canonical_prior"],
            "summed_prior": cls["summed_prior"],
            "multiplicity_ratio": math.exp(cls["summed_prior"] - cls["canonical_prior"]),
            "posterior_canonical_strict": cls["posterior_canonical_strict"],
            "posterior_canonical_noisy": cls["posterior_canonical_noisy"],
            "posterior_summed_strict": cls["posterior_summed_strict"],
            "posterior_summed_noisy": cls["posterior_summed_noisy"],
            "is_true_rule": cls["fingerprint"] == true_fp,
        }
        top_k_report.append(entry)

    # Compute entropy
    entropy_canonical = posterior_entropy(viable_classes, "posterior_canonical_strict")
    entropy_summed = posterior_entropy(viable_classes, "posterior_summed_strict")

    return {
        "rule_id": rule_id,
        "group": true_rule["group"],
        "answer": true_rule["answer"],
        "map_hypothesis": viable_classes[0]["canonical_program"] if viable_classes else None,
        "map_is_correct": map_is_correct,
        "true_rule_rank": true_rule_rank,
        "true_rule_posterior_canonical_strict": true_rule_posterior,
        "posterior_entropy_canonical": entropy_canonical,
        "posterior_entropy_summed": entropy_summed,
        "n_viable_classes": len(viable_classes),
        "n_total_classes": len(classes),
        "hypothesis_table_stats": table.stats,
        "top_k": top_k_report,
    }


def main():
    parser = argparse.ArgumentParser(description="Bayesian Rule Induction Analysis")
    parser.add_argument("--rules", nargs="+", help="Rule IDs to analyze")
    parser.add_argument("--all", action="store_true", help="Analyze all 60 rules")
    parser.add_argument("--depth", type=int, default=5, help="Enumeration depth (default 5)")
    parser.add_argument("--max-programs", type=int, default=10000, help="Max programs to enumerate")
    parser.add_argument("--timeout", type=float, default=300, help="Enumeration timeout in seconds")
    parser.add_argument("--n-probes", type=int, default=200, help="Probe set size")
    parser.add_argument("--n-mc-samples", type=int, default=100000, help="Monte Carlo samples")
    parser.add_argument("--epsilon", type=float, default=0.05, help="Noise parameter")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K classes to report")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    # Determine which rules to analyze
    if args.all:
        rule_ids = list(GALLERY_RULES.keys())
    elif args.rules:
        rule_ids = args.rules
    else:
        print("Specify --rules <id1> <id2> ... or --all")
        return

    # Validate rule IDs
    for rid in rule_ids:
        if rid not in GALLERY_RULES:
            print(f"Unknown rule: {rid}. Available: {list(GALLERY_RULES.keys())}")
            return

    # Setup output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load exemplars
    print("Loading exemplars...")
    exemplars = load_exemplars()

    # Generate probe set
    print(f"Generating probe set ({args.n_probes} hands)...")
    probe_set = generate_probe_set(n_probes=args.n_probes)

    # Enumerate hypotheses (shared across all rules in Phase 1)
    print(f"Enumerating hypotheses (depth={args.depth}, max={args.max_programs}, timeout={args.timeout}s)...")
    t0 = time.time()
    programs = enumerate_hypotheses(
        max_depth=args.depth,
        max_programs=args.max_programs,
        timeout=args.timeout,
    )
    enum_time = time.time() - t0
    print(f"  Enumerated {len(programs)} programs in {enum_time:.1f}s")

    # Analyze each rule
    all_results = []
    for rule_id in rule_ids:
        print(f"\nAnalyzing rule: {rule_id}")
        hands = exemplars.get(rule_id, {}).get("hands_primary")
        if hands is None:
            print(f"  WARNING: No exemplars found for {rule_id}, skipping")
            continue

        result = analyze_rule(
            rule_id=rule_id,
            exemplar_hands=hands,
            probe_set=probe_set,
            enumerated_programs=programs,
            top_k=args.top_k,
            n_mc_samples=args.n_mc_samples,
            epsilon=args.epsilon,
        )
        all_results.append(result)

        # Save individual result
        out_path = output_dir / f"{rule_id}.json"
        # Remove non-serializable fields before saving
        result_clean = {k: v for k, v in result.items()}
        for entry in result_clean.get("top_k", []):
            entry.pop("predicate", None)
        with open(out_path, "w") as f:
            json.dump(result_clean, f, indent=2)
        print(f"  Saved to {out_path}")

    # Save summary
    summary = {
        "enumeration_depth": args.depth,
        "n_programs_enumerated": len(programs),
        "enumeration_time_seconds": enum_time,
        "n_probes": args.n_probes,
        "n_mc_samples": args.n_mc_samples,
        "epsilon": args.epsilon,
        "rules_analyzed": len(all_results),
        "results": [
            {
                "rule_id": r["rule_id"],
                "group": r["group"],
                "map_is_correct": r["map_is_correct"],
                "true_rule_rank": r["true_rule_rank"],
                "posterior_entropy": r["posterior_entropy_canonical"],
                "n_viable_classes": r["n_viable_classes"],
            }
            for r in all_results
        ]
    }
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Run a sanity check on 2-3 rules**

```bash
cd src && python -m gallery_analysis.analyze --rules all_red all_same_suit strict_increasing --depth 4 --max-programs 500 --timeout 60
```

This should produce JSON results in `src/gallery_analysis/results/`. Check that:
- Programs are enumerated without errors
- Extension sizes are reasonable (all_red should have a base rate around 1/64)
- The true rule appears somewhere in the top-K (at least for simple rules)

**Step 3: Commit**

```bash
git add src/gallery_analysis/analyze.py
git commit -m "feat: add main analysis pipeline with end-to-end Bayesian rule induction"
```

---

## Phase 2: Efficiency and Full Coverage

### Task 7: Add likelihood pruning during enumeration

**Files:**
- Modify: `src/gallery_analysis/enumerator.py`
- Create: `src/tests/test_likelihood_pruning.py`

This task extends the enumerator to accept exemplar hands and prune partial programs that fail on 2+ exemplars. The implementation depends on how the existing `TopDownEnumerator` exposes partial programs during enumeration. There are two possible approaches:

**Option A: Post-filter** — Enumerate all programs, then filter by hit count. Simpler but doesn't save enumeration time.

**Option B: Custom enumerator subclass** — Override the hole-filling loop to evaluate partial programs on exemplars and prune subtrees. More complex but gives the 2-10x speedup described in the design.

Start with **Option A** (post-filter with 1-miss tolerance) to keep Phase 2 incremental. Option B can be added later if enumeration time is the bottleneck.

**Step 1: Write test**

```python
# src/tests/test_likelihood_pruning.py
"""Tests for likelihood-based filtering during/after enumeration."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import H, D, S, C

def test_filter_by_exemplar_hits():
    """Programs failing on 2+ exemplars should be excluded."""
    from gallery_analysis.enumerator import enumerate_and_filter

    # All-red exemplars
    exemplars = [
        [H("A"), H("K"), D("Q"), D("J"), H("10"), D("9")],
        [H("2"), D("3"), H("4"), D("5"), H("6"), D("7")],
        [D("A"), H("Q"), D("10"), H("8"), D("6"), H("4")],
    ]

    results = enumerate_and_filter(
        exemplar_hands=exemplars,
        max_depth=3,
        max_programs=100,
        timeout=30,
        max_misses=1,
    )

    # All returned hypotheses should have at most 1 miss
    for prog_str, pred_fn, log_prior, n_hits, n_misses in results:
        assert n_misses <= 1, f"{prog_str} has {n_misses} misses"
```

**Step 2: Implement enumerate_and_filter**

Add to `enumerator.py`:

```python
def enumerate_and_filter(
    exemplar_hands: List[Hand],
    max_depth: int = 6,
    max_programs: int = 10000,
    max_cost: float = 50.0,
    timeout: float = 300.0,
    max_misses: int = 1,
    grammar=None,
) -> List[Tuple[str, Callable, float, int, int]]:
    """
    Enumerate programs and filter by exemplar consistency.

    Returns (prog_str, pred_fn, log_prior, n_hits, n_misses) for each
    program that has at most max_misses misses on the exemplar hands.
    """
    raw = enumerate_hypotheses(max_depth, max_programs, max_cost, timeout, grammar)

    filtered = []
    for prog_str, pred_fn, log_prior in raw:
        n_hits = 0
        n_misses = 0
        for hand in exemplar_hands:
            try:
                if pred_fn(hand):
                    n_hits += 1
                else:
                    n_misses += 1
            except Exception:
                n_misses += 1

            if n_misses > max_misses:
                break

        if n_misses <= max_misses:
            filtered.append((prog_str, pred_fn, log_prior, n_hits, n_misses))

    return filtered
```

**Step 3: Run tests, commit**

```bash
cd src && python -m pytest tests/test_likelihood_pruning.py -v --timeout=60
git add src/gallery_analysis/enumerator.py src/tests/test_likelihood_pruning.py
git commit -m "feat: add post-enumeration filtering by exemplar consistency"
```

---

### Task 8: Constant folding and dead code elimination

**Files:**
- Modify: `src/gallery_analysis/enumerator.py`

This adds a check after enumeration: evaluate programs on a small sample of hands and discard those that are trivially true (always True) or trivially false (always False), since these aren't useful hypotheses.

**Step 1: Add trivial program filter to enumerate_and_filter**

```python
def _is_trivial(pred_fn: Callable, n_checks: int = 20, seed: int = 77) -> bool:
    """Check if a predicate is trivially always-True or always-False."""
    from gallery_analysis.exemplars import generate_probe_set
    test_hands = generate_probe_set(n_probes=n_checks, seed=seed)
    results = set()
    for hand in test_hands:
        try:
            results.add(pred_fn(hand))
        except Exception:
            return True  # Erroring programs are useless
        if len(results) > 1:
            return False  # Non-trivial: produces both True and False
    return True  # Only produced one value
```

Add a `filter_trivial=True` parameter to `enumerate_and_filter` that applies this check.

**Step 2: Test and commit**

```bash
cd src && python -m pytest tests/ -v --timeout=60
git add src/gallery_analysis/enumerator.py
git commit -m "feat: add trivial program filtering (always-true/always-false)"
```

---

### Task 9: Push enumeration to depth 7-8 and run on all 60 rules

**Files:**
- Modify: `src/gallery_analysis/analyze.py` (update defaults, add progress reporting)

This is primarily a scaling task. Update `analyze.py` to:
1. Default to depth 7 with higher timeout
2. Add progress bars/percentage reporting
3. Run on all 60 rules
4. Use `enumerate_and_filter` per-rule instead of shared enumeration

**Step 1: Update analyze.py defaults and per-rule enumeration**

The key change: instead of enumerating once and reusing, enumerate per-rule with exemplar filtering. This is slower total but each individual enumeration is much faster due to filtering.

Update the `main()` function in `analyze.py` to support both modes:
- `--shared-enum`: Enumerate once, filter per-rule (faster for many rules at low depth)
- `--per-rule-enum`: Enumerate per-rule with filtering (better at high depth with pruning)

Default to `--per-rule-enum` at depth 7+.

**Step 2: Run overnight on all 60 rules**

```bash
cd src
nohup caffeinate -d -i -s python -m gallery_analysis.analyze --all --depth 7 --max-programs 50000 --timeout 600 --per-rule-enum > gallery_analysis_run.out 2>&1 &
```

**Step 3: Commit results**

```bash
git add src/gallery_analysis/analyze.py
git commit -m "feat: support per-rule enumeration and depth 7-8 scaling"
```

---

## Phase 3: Analysis Outputs

### Task 10: Difficulty ranking report

**Files:**
- Create: `src/gallery_analysis/reports.py`

Reads all per-rule JSON results from `results/` and produces:
- A ranked table of all 60 rules by posterior entropy
- Comparison of canonical vs summed prior rankings
- Group-level statistics (mean entropy by difficulty group)

**Step 1: Implement reports.py**

```python
# src/gallery_analysis/reports.py
"""
Generate analysis reports from per-rule JSON results.

Reads results/*.json and produces summary tables and comparisons.
"""
import json
from pathlib import Path
from typing import List, Dict

RESULTS_DIR = Path(__file__).parent / "results"


def load_all_results() -> List[Dict]:
    """Load all per-rule JSON results."""
    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        if f.name == "summary.json":
            continue
        with open(f) as fh:
            results.append(json.load(fh))
    return results


def difficulty_ranking(results: List[Dict]) -> List[Dict]:
    """Rank rules by posterior entropy (higher = harder)."""
    ranked = sorted(results, key=lambda r: -r.get("posterior_entropy_canonical", 0))
    for i, r in enumerate(ranked):
        r["difficulty_rank"] = i + 1
    return ranked


def group_statistics(results: List[Dict]) -> Dict:
    """Compute mean entropy by difficulty group."""
    from collections import defaultdict
    groups = defaultdict(list)
    for r in results:
        groups[r["group"]].append(r.get("posterior_entropy_canonical", 0))

    stats = {}
    for group, entropies in sorted(groups.items()):
        stats[f"group_{group}"] = {
            "n_rules": len(entropies),
            "mean_entropy": sum(entropies) / len(entropies) if entropies else 0,
            "min_entropy": min(entropies) if entropies else 0,
            "max_entropy": max(entropies) if entropies else 0,
        }
    return stats


def print_difficulty_table(results: List[Dict]):
    """Print a formatted difficulty ranking table."""
    ranked = difficulty_ranking(results)
    print(f"{'Rank':>4} {'Rule ID':<35} {'Grp':>3} {'MAP?':>4} "
          f"{'TrueRank':>8} {'Entropy':>8} {'#Classes':>8}")
    print("-" * 85)
    for r in ranked:
        print(f"{r['difficulty_rank']:>4} {r['rule_id']:<35} {r['group']:>3} "
              f"{'Yes' if r.get('map_is_correct') else 'No':>4} "
              f"{r.get('true_rule_rank', '?'):>8} "
              f"{r.get('posterior_entropy_canonical', 0):>8.2f} "
              f"{r.get('n_viable_classes', 0):>8}")
```

**Step 2: Commit**

```bash
git add src/gallery_analysis/reports.py
git commit -m "feat: add difficulty ranking and group statistics reports"
```

---

### Task 11: Test hand difficulty scoring CLI

**Files:**
- Modify: `src/gallery_analysis/analyze.py`

Add a `--score-hands` mode that takes a rule ID and a set of test hands (as JSON or inline) and outputs difficulty scores.

```bash
# Score test hands for a rule (after running the main analysis)
cd src && python -m gallery_analysis.analyze --score-hands all_red --hand "AS KH QD JC 10H 9D"
```

This uses the stored posterior from the per-rule JSON and evaluates `score_test_hand()` for each provided hand.

**Step 1: Add scoring subcommand to analyze.py**

Add to the argparser:
```python
parser.add_argument("--score-hands", type=str, help="Rule ID for test hand scoring")
parser.add_argument("--hand", type=str, nargs="+", help="Test hands as card strings")
```

**Step 2: Commit**

```bash
git add src/gallery_analysis/analyze.py
git commit -m "feat: add test hand difficulty scoring CLI mode"
```

---

### Task 12: Validate against frozen exemplars (cross-check)

**Files:**
- Create: `src/tests/test_gallery_rules_vs_exemplars.py`

A comprehensive cross-validation: for every rule in the gallery, check that all 6 primary exemplar hands satisfy the Python predicate. This catches porting errors.

**Step 1: Write test**

```python
# src/tests/test_gallery_rules_vs_exemplars.py
"""Cross-validate: every exemplar hand should satisfy its rule's predicate."""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from gallery_analysis.exemplars import load_exemplars
from gallery_analysis.gallery_rules import GALLERY_RULES

@pytest.fixture(scope="module")
def exemplars():
    return load_exemplars()

def rule_ids():
    """Generate rule IDs that exist in both exemplars and gallery rules."""
    exemplars = load_exemplars()
    return [rid for rid in exemplars if rid in GALLERY_RULES]

@pytest.mark.parametrize("rule_id", rule_ids())
def test_exemplar_satisfies_rule(rule_id, exemplars):
    """Each primary exemplar hand should satisfy its rule."""
    predicate = GALLERY_RULES[rule_id]["predicate"]
    hands = exemplars[rule_id]["hands_primary"]
    for i, hand in enumerate(hands):
        assert predicate(hand), (
            f"Rule {rule_id}, exemplar {i}: predicate returned False "
            f"for hand {[str(c) for c in hand]}"
        )
```

**Step 2: Run and fix any failures**

```bash
cd src && python -m pytest tests/test_gallery_rules_vs_exemplars.py -v
```

Any failures here indicate a porting error in the Python predicate. Fix the predicate to match the JS implementation exactly.

**Step 3: Commit**

```bash
git add src/tests/test_gallery_rules_vs_exemplars.py
git commit -m "test: cross-validate all gallery rules against frozen exemplar hands"
```

---

## Summary of All Tasks

| Task | Phase | Description | Key Files |
|------|-------|-------------|-----------|
| 1 | 1 | Gallery rules + skeleton | `gallery_rules.py`, `__init__.py` |
| 2 | 1 | Exemplar loading + probes | `exemplars.py` |
| 3 | 1 | Hypothesis table + fingerprints | `hypothesis_table.py` |
| 4 | 1 | Bayesian scorer | `bayesian_scorer.py` |
| 5 | 1 | Enumerator wrapper | `enumerator.py` |
| 6 | 1 | Main pipeline | `analyze.py` |
| 7 | 2 | Likelihood pruning | `enumerator.py` (modify) |
| 8 | 2 | Trivial program filtering | `enumerator.py` (modify) |
| 9 | 2 | Depth 7-8 + all 60 rules | `analyze.py` (modify) |
| 10 | 3 | Difficulty ranking report | `reports.py` |
| 11 | 3 | Test hand scoring CLI | `analyze.py` (modify) |
| 12 | 3 | Cross-validation test suite | `test_gallery_rules_vs_exemplars.py` |
