"""
The 16 Focus Rules for Primitive Library Optimization.

These are the 16 rules selected for the transfer-and-methodology study,
organized into 8 families with 2 rules each. This file defines the exact
rule predicates and tracks which DSL primitives each rule uses.

Source: card-games/docs/transfer-and-methodology.tex

8 Families:
1. PALINDROME: suits_palindrome, colors_palindrome
2. ARITH_PROG: AP_len3_step1, AP_len3_step2
3. ADJACENCY: adj_rank_or_suit, sorted_by_rank
4. GLOBAL: uniform_color, majority_red
5. COUNT_PAIRING: has_pair_ranks, has_pair_suits
6. HALVES_BICON: halves_same_color, halves_hearts_equal
7. HALVES_COPY: halves_copy_suits, halves_copy_colors
8. HALVES_BOTH: halves_both_AP3, halves_both_adj
"""

from typing import Callable, List
from dataclasses import dataclass
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Hand, Suit, Color, Card, RANK_VALUES, card_color
from rules.primitives import (
    get_suit, get_rank, get_rank_val, get_color,
    halves, left_half, right_half,
    has_arithmetic_progression, seq_palindrome, uniform_property,
    is_sorted
)


@dataclass
class FocusRule:
    """
    A focus rule with metadata for primitive analysis.

    Attributes:
        id: Rule identifier matching transfer-and-methodology.tex
        family: One of the 8 families
        predicate: The evaluation function (Hand -> bool)
        lambda_expr: Lambda calculus expression (for reference)
        description: Plain English description

    Note: primitives_used is NOT stored here - it will be DISCOVERED
    by the optimization algorithm through program enumeration.
    """
    id: str
    family: str
    predicate: Callable[[Hand], bool]
    lambda_expr: str
    description: str

    def eval(self, hand: Hand) -> bool:
        """Evaluate rule on a hand."""
        return self.predicate(hand)


# ============================================================================
# FAMILY 1: PALINDROME
# Template: map(f, h) = reverse(map(f, h))
# ============================================================================

def suits_palindrome(hand: Hand) -> bool:
    """Suits sequence reads same forward/backward."""
    suits = [get_suit(c) for c in hand]
    return suits == list(reversed(suits))

def colors_palindrome(hand: Hand) -> bool:
    """Colors sequence reads same forward/backward."""
    colors = [get_color(c) for c in hand]
    return colors == list(reversed(colors))


# ============================================================================
# FAMILY 2: ARITH_PROG
# Template: exists subseq in ranks(h). isAP(subseq, len, step)
# ============================================================================

def ap_len3_step1(hand: Hand) -> bool:
    """Contains 3 consecutive ranks (e.g., 5-6-7)."""
    return has_arithmetic_progression(3, 1, False)(hand)

def ap_len3_step2(hand: Hand) -> bool:
    """Contains 3 ranks with step 2 (e.g., 4-6-8)."""
    return has_arithmetic_progression(3, 2, False)(hand)


# ============================================================================
# FAMILY 3: ADJACENCY
# Template: forall i. P(h[i], h[i+1])
# ============================================================================

def adj_rank_or_suit(hand: Hand) -> bool:
    """Every adjacent pair shares rank or suit."""
    for i in range(len(hand) - 1):
        c1, c2 = hand[i], hand[i+1]
        if not (get_rank(c1) == get_rank(c2) or get_suit(c1) == get_suit(c2)):
            return False
    return True

def sorted_by_rank(hand: Hand) -> bool:
    """Ranks in non-decreasing order left-to-right."""
    return is_sorted(hand, get_rank_val, strict=False)


# ============================================================================
# FAMILY 4: GLOBAL
# Template: P(h) - single predicate on whole hand
# ============================================================================

def uniform_color(hand: Hand) -> bool:
    """All cards have the same color (all red or all black)."""
    if not hand:
        return True
    first_color = get_color(hand[0])
    return all(get_color(c) == first_color for c in hand)

def majority_red(hand: Hand) -> bool:
    """More than half the cards are red."""
    if not hand:
        return False
    red_count = sum(1 for c in hand if get_color(c) == Color.RED)
    return red_count > len(hand) / 2


# ============================================================================
# FAMILY 5: COUNT_PAIRING
# Template: exists x. count(c: P(c) = x) >= 2
# ============================================================================

def has_pair_ranks(hand: Hand) -> bool:
    """At least two cards share a rank."""
    ranks_seen = set()
    for c in hand:
        r = get_rank(c)
        if r in ranks_seen:
            return True
        ranks_seen.add(r)
    return False

def has_pair_suits(hand: Hand) -> bool:
    """At least two cards share a suit."""
    suits_seen = set()
    for c in hand:
        s = get_suit(c)
        if s in suits_seen:
            return True
        suits_seen.add(s)
    return False


# ============================================================================
# FAMILY 6: HALVES_BICON
# Template: Q(L(h)) <-> Q(R(h)) - biconditional on halves
# ============================================================================

def halves_same_color(hand: Hand) -> bool:
    """Both halves are uniform in color (or both are not)."""
    left, right = halves(hand)
    left_uniform = uniform_property(get_color)(left)
    right_uniform = uniform_property(get_color)(right)
    return left_uniform == right_uniform

def halves_hearts_equal(hand: Hand) -> bool:
    """Both halves have a heart, or neither does."""
    left, right = halves(hand)
    left_has_heart = any(get_suit(c) == Suit.HEARTS for c in left)
    right_has_heart = any(get_suit(c) == Suit.HEARTS for c in right)
    return left_has_heart == right_has_heart


# ============================================================================
# FAMILY 7: HALVES_COPY
# Template: map(f, L(h)) = map(f, R(h))
# ============================================================================

def halves_copy_suits(hand: Hand) -> bool:
    """Right half mirrors left in suits."""
    left, right = halves(hand)
    return [get_suit(c) for c in left] == [get_suit(c) for c in right]

def halves_copy_colors(hand: Hand) -> bool:
    """Right half mirrors left in colors."""
    left, right = halves(hand)
    return [get_color(c) for c in left] == [get_color(c) for c in right]


# ============================================================================
# FAMILY 8: HALVES_BOTH
# Template: Q(L(h)) AND Q(R(h)) - conjunction on halves
# ============================================================================

def halves_both_AP3(hand: Hand) -> bool:
    """Both halves contain a consecutive triple (AP len=3, step=1)."""
    left, right = halves(hand)
    left_has_ap = has_arithmetic_progression(3, 1, False)(left)
    right_has_ap = has_arithmetic_progression(3, 1, False)(right)
    return left_has_ap and right_has_ap

def halves_both_adj(hand: Hand) -> bool:
    """Both halves satisfy adjacency property (neighbors share rank or suit)."""
    left, right = halves(hand)

    def satisfies_adj(h: Hand) -> bool:
        if len(h) <= 1:
            return True
        for i in range(len(h) - 1):
            c1, c2 = h[i], h[i+1]
            if not (get_rank(c1) == get_rank(c2) or get_suit(c1) == get_suit(c2)):
                return False
        return True

    return satisfies_adj(left) and satisfies_adj(right)


# ============================================================================
# NOTE ON PRIMITIVES
# ============================================================================
#
# The primitives used by each rule are NOT pre-specified here.
# Instead, the greedy optimization algorithm will DISCOVER which primitives
# are needed by:
#   1. Starting with ALL primitives from lean_primitives.py
#   2. Enumerating programs to find expressions for each rule
#   3. Trying to remove primitives and checking if rules remain expressible
#
# This lets the algorithm discover potentially better primitive sets than
# what a human might guess. See optimize_library_for_16_rules.py for the
# actual optimization algorithm.
# ============================================================================


# ============================================================================
# CREATE ALL 16 FOCUS RULES
# ============================================================================

def create_focus_rules() -> List[FocusRule]:
    """Create all 16 focus rules with full metadata."""
    return [
        # Family 1: PALINDROME
        FocusRule(
            id="suits_palindrome",
            family="PALINDROME",
            predicate=suits_palindrome,
            lambda_expr="λh. map(suit, h) = reverse(map(suit, h))",
            description="Suits sequence reads same forward/backward"
        ),
        FocusRule(
            id="colors_palindrome",
            family="PALINDROME",
            predicate=colors_palindrome,
            lambda_expr="λh. map(color, h) = reverse(map(color, h))",
            description="Colors sequence reads same forward/backward"
        ),

        # Family 2: ARITH_PROG
        FocusRule(
            id="ap_len3_step1",
            family="ARITH_PROG",
            predicate=ap_len3_step1,
            lambda_expr="λh. ∃ subseq ⊆ ranks(h). isAP(subseq, 3, 1)",
            description="Contains 3 consecutive ranks (e.g., 5-6-7)"
        ),
        FocusRule(
            id="ap_len3_step2",
            family="ARITH_PROG",
            predicate=ap_len3_step2,
            lambda_expr="λh. ∃ subseq ⊆ ranks(h). isAP(subseq, 3, 2)",
            description="Contains 3 ranks with step 2 (e.g., 4-6-8)"
        ),

        # Family 3: ADJACENCY
        FocusRule(
            id="adj_rank_or_suit",
            family="ADJACENCY",
            predicate=adj_rank_or_suit,
            lambda_expr="λh. ∀i. rank(h[i])=rank(h[i+1]) ∨ suit(h[i])=suit(h[i+1])",
            description="Every adjacent pair shares rank or suit"
        ),
        FocusRule(
            id="sorted_by_rank",
            family="ADJACENCY",
            predicate=sorted_by_rank,
            lambda_expr="λh. ∀i. rank_val(h[i]) ≤ rank_val(h[i+1])",
            description="Ranks in non-decreasing order"
        ),

        # Family 4: GLOBAL
        FocusRule(
            id="uniform_color",
            family="GLOBAL",
            predicate=uniform_color,
            lambda_expr="λh. all(λc. color(c) = color(first(h)), h)",
            description="All cards have the same color"
        ),
        FocusRule(
            id="majority_red",
            family="GLOBAL",
            predicate=majority_red,
            lambda_expr="λh. count(λc. color(c)=RED, h) > length(h)/2",
            description="More than half the cards are red"
        ),

        # Family 5: COUNT_PAIRING
        FocusRule(
            id="has_pair_ranks",
            family="COUNT_PAIRING",
            predicate=has_pair_ranks,
            lambda_expr="λh. unique_count(rank, h) < length(h)",
            description="At least two cards share a rank"
        ),
        FocusRule(
            id="has_pair_suits",
            family="COUNT_PAIRING",
            predicate=has_pair_suits,
            lambda_expr="λh. unique_count(suit, h) < length(h)",
            description="At least two cards share a suit"
        ),

        # Family 6: HALVES_BICON
        FocusRule(
            id="halves_same_color",
            family="HALVES_BICON",
            predicate=halves_same_color,
            lambda_expr="λh. uniform(color, L(h)) = uniform(color, R(h))",
            description="Both halves are uniform in color (or both not)"
        ),
        FocusRule(
            id="halves_hearts_equal",
            family="HALVES_BICON",
            predicate=halves_hearts_equal,
            lambda_expr="λh. any(λc. suit(c)=♥, L(h)) = any(λc. suit(c)=♥, R(h))",
            description="Both halves have a heart, or neither does"
        ),

        # Family 7: HALVES_COPY
        FocusRule(
            id="halves_copy_suits",
            family="HALVES_COPY",
            predicate=halves_copy_suits,
            lambda_expr="λh. map(suit, L(h)) = map(suit, R(h))",
            description="Right half mirrors left in suits"
        ),
        FocusRule(
            id="halves_copy_colors",
            family="HALVES_COPY",
            predicate=halves_copy_colors,
            lambda_expr="λh. map(color, L(h)) = map(color, R(h))",
            description="Right half mirrors left in colors"
        ),

        # Family 8: HALVES_BOTH
        FocusRule(
            id="halves_both_AP3",
            family="HALVES_BOTH",
            predicate=halves_both_AP3,
            lambda_expr="λh. has_AP(3,1)(L(h)) ∧ has_AP(3,1)(R(h))",
            description="Both halves contain a consecutive triple"
        ),
        FocusRule(
            id="halves_both_adj",
            family="HALVES_BOTH",
            predicate=halves_both_adj,
            lambda_expr="λh. adj_property(L(h)) ∧ adj_property(R(h))",
            description="Both halves satisfy adjacency property"
        ),
    ]


# ============================================================================
# EXPORTS
# ============================================================================

FOCUS_RULES = create_focus_rules()
FOCUS_RULES_DICT = {rule.id: rule for rule in FOCUS_RULES}
FOCUS_RULES_BY_FAMILY = {}
for rule in FOCUS_RULES:
    FOCUS_RULES_BY_FAMILY.setdefault(rule.family, []).append(rule)


def print_summary():
    """Print a summary of the 16 focus rules."""
    print("=" * 70)
    print("16 FOCUS RULES FOR PRIMITIVE LIBRARY OPTIMIZATION")
    print("=" * 70)
    print()
    print("Note: Primitives will be DISCOVERED by the optimization algorithm,")
    print("not pre-specified. See optimize_library_for_16_rules.py")
    print()

    for family in ["PALINDROME", "ARITH_PROG", "ADJACENCY", "GLOBAL",
                   "COUNT_PAIRING", "HALVES_BICON", "HALVES_COPY", "HALVES_BOTH"]:
        print(f"\n{family}:")
        for rule in FOCUS_RULES_BY_FAMILY[family]:
            print(f"  {rule.id}")
            print(f"    {rule.description}")
            print(f"    Lambda: {rule.lambda_expr}")

    print("\n" + "=" * 70)
    print(f"Total rules: {len(FOCUS_RULES)}")
    print(f"Total families: {len(FOCUS_RULES_BY_FAMILY)}")
    print("=" * 70)


if __name__ == "__main__":
    print_summary()
