"""
Complete catalogue of 56 card game rules.

Each rule is implemented as a predicate (Hand → bool) using the compositional primitives.
This module demonstrates the compositional structure and enables analysis of
shared subprograms across rules.
"""

from typing import Callable
from dataclasses import dataclass

from .cards import Hand, Suit, Rank, Color
from .primitives import *


@dataclass
class Rule:
    """A rule with metadata and evaluation function."""
    id: str
    name: str
    predicate: Callable[[Hand], bool]
    family: str
    description: str
    primitives_used: list  # List of primitive names for analysis

    def eval(self, hand: Hand) -> bool:
        """Evaluate rule on a hand."""
        return self.predicate(hand)


# ============================================================================
# RULE CATALOG (organized by family)
# ============================================================================

# Family A: Simple Local Properties
# ============================================================================

def rule_sorted_by_rank() -> Rule:
    """r1x: Ranks are non-decreasing left to right."""
    return Rule(
        id="Sorted_by_rank",
        name="Ranks are in non-decreasing order",
        predicate=lambda h: is_sorted(h, get_rank_val, strict=False),
        family="LOCAL",
        description="Read ranks left-to-right; they never go down (e.g., 5,7,7,9,J,…).",
        primitives_used=["is_sorted", "get_rank_val"]
    )


def rule_has_pair_ranks() -> Rule:
    """r2x: There exists a pair of equal ranks."""
    def has_pair(hand: Hand) -> bool:
        ranks_seen = set()
        for card in hand:
            r = get_rank(card)
            if r in ranks_seen:
                return True
            ranks_seen.add(r)
        return False

    return Rule(
        id="Has_pair_ranks",
        name="At least one pair (same rank)",
        predicate=has_pair,
        family="COUNT",
        description="Some two cards share the same rank, e.g., two 9s.",
        primitives_used=["get_rank", "set", "any"]
    )


def rule_uniform_color() -> Rule:
    """r3x: All cards have the same color."""
    return Rule(
        id="Uniform_color",
        name="All cards have the same color",
        predicate=uniform_property(get_color),
        family="COUNT",
        description="All cards are either black (♣/♠) or red (♦/♥).",
        primitives_used=["uniform_property", "get_color"]
    )


def rule_s_before_h() -> Rule:
    """r4x: Some spade occurs before some heart."""
    def spade_before_heart(hand: Hand) -> bool:
        seen_spade = False
        for card in hand:
            if get_suit(card) == Suit.SPADES:
                seen_spade = True
            if seen_spade and get_suit(card) == Suit.HEARTS:
                return True
        return False

    return Rule(
        id="S_before_H",
        name="Some ♠ appears before some ♥",
        predicate=spade_before_heart,
        family="LOCAL",
        description="Scan left-to-right; at least one ♠ is left of at least one ♥.",
        primitives_used=["get_suit", "positional_scan"]
    )


def rule_ap_len3_anywhere_anyk() -> Rule:
    """r5x: Any arithmetic progression of 3 ranks (any step, anywhere)."""
    return Rule(
        id="AP_len3_anywhere_anyk",
        name="There is a 3-term rank pattern with equal steps (any step)",
        predicate=has_arithmetic_progression(3, None, False),
        family="AP",
        description="Ignoring order, some three ranks differ by the same amount (e.g., 4,6,8 or 7,10,13).",
        primitives_used=["has_arithmetic_progression"]
    )


def rule_score_threshold() -> Rule:
    """r6x: Score = sum(ranks) + 10·sorted + 6·(hearts≥3) ≥ threshold."""
    def score_rule(hand: Hand, threshold: int = 50) -> bool:
        rank_sum = sum_values(hand)
        sorted_bonus = 10 if is_sorted(hand) else 0
        hearts_count = count_equal(Suit.HEARTS, get_suit)(hand)
        hearts_bonus = 6 if hearts_count >= 3 else 0
        score = rank_sum + sorted_bonus + hearts_bonus
        return score >= threshold

    return Rule(
        id="Score_threshold_Rstar",
        name="Score rule (ranks + order + hearts)",
        predicate=score_rule,
        family="SCORE",
        description="Add ranks; add 10 if non-decreasing; add 6 if ≥3 ♥; win if total ≥ threshold.",
        primitives_used=["sum_values", "is_sorted", "count_equal", "get_suit", "threshold"]
    )


# Family B: Position-based
# ============================================================================

def rule_pos3_is_jqk() -> Rule:
    """r7x: 3rd position is J, Q or K."""
    def check(hand: Hand) -> bool:
        if len(hand) < 3:
            return False
        return get_rank(hand[2]) in {Rank.JACK, Rank.QUEEN, Rank.KING}

    return Rule(
        id="Pos3_is_JQK",
        name="Card #3 is a face card (J/Q/K)",
        predicate=check,
        family="POSITION",
        description="The 3rd card (counting from the left) is J, Q, or K.",
        primitives_used=["at", "get_rank", "set_membership"]
    )


def rule_pos4_is_2_5_7() -> Rule:
    """r8x: 4th position is 2, 5 or 7."""
    def check(hand: Hand) -> bool:
        if len(hand) < 4:
            return False
        return get_rank(hand[3]) in {Rank.TWO, Rank.FIVE, Rank.SEVEN}

    return Rule(
        id="Pos4_is_2_5_7",
        name="Card #4 is 2, 5, or 7",
        predicate=check,
        family="POSITION",
        description="The 4th card (from the left) is 2, 5, or 7.",
        primitives_used=["at", "get_rank", "set_membership"]
    )


# Family C: Token-based
# ============================================================================

def rule_has_ace_of_spades() -> Rule:
    """r9x: Contains the Ace of Spades."""
    return Rule(
        id="Has_Ace_of_Spades",
        name="Contains the Ace of ♠",
        predicate=lambda h: any(get_suit(c) == Suit.SPADES and get_rank(c) == Rank.ACE for c in h),
        family="TOKEN",
        description="Somewhere in the hand there is the Ace of Spades.",
        primitives_used=["any", "get_suit", "get_rank", "eq"]
    )


def rule_has_6_of_diamonds() -> Rule:
    """r10x: Contains the 6 of Diamonds."""
    return Rule(
        id="Has_6_of_Diamonds",
        name="Contains the 6 of ♦",
        predicate=lambda h: any(get_suit(c) == Suit.DIAMONDS and get_rank(c) == Rank.SIX for c in h),
        family="TOKEN",
        description="Somewhere in the hand there is the 6 of Diamonds.",
        primitives_used=["any", "get_suit", "get_rank", "eq"]
    )


# Family D: Count-based
# ============================================================================

def rule_exactly_two_suits() -> Rule:
    """p7x: Exactly two suits appear."""
    return Rule(
        id="Exactly_two_suits",
        name="Exactly two suits appear",
        predicate=lambda h: unique_count(get_suit)(h) == 2,
        family="COUNT",
        description="Across the whole hand there are exactly two distinct suits.",
        primitives_used=["unique_count", "get_suit"]
    )


def rule_half_or_more_same_suit() -> Rule:
    """r11x: At least half of the cards share a suit."""
    def check(hand: Hand) -> bool:
        n = len(hand)
        threshold = (n + 1) // 2  # Ceiling division
        for suit in Suit:
            if count_equal(suit, get_suit)(hand) >= threshold:
                return True
        return False

    return Rule(
        id="Half_or_more_same_suit",
        name="At least half the cards share a suit",
        predicate=check,
        family="COUNT",
        description="≥ N/2 cards (rounded up) are the same suit.",
        primitives_used=["count_equal", "get_suit", "max", "threshold"]
    )


# Family E: Hierarchical (Halves) - Boolean Composition
# ============================================================================

def rule_halves_uniform_color_equal() -> Rule:
    """r12x: Halves both uniform in color or both not."""
    def check(hand: Hand) -> bool:
        left, right = halves(hand)
        left_uniform = uniform_property(get_color)(left)
        right_uniform = uniform_property(get_color)(right)
        return left_uniform == right_uniform

    return Rule(
        id="Halves_uniform_color_equal",
        name="Both halves are uniform in color (or both not)",
        predicate=check,
        family="HIER",
        description="Left and right halves either both have a single color, or both mix colors.",
        primitives_used=["halves", "uniform_property", "get_color", "eq"]
    )


def rule_halves_uniform_parity_equal() -> Rule:
    """r13x: Halves both uniform in parity (odd/even) or both not."""
    def check(hand: Hand) -> bool:
        left, right = halves(hand)
        left_uniform = uniform_property(get_parity)(left)
        right_uniform = uniform_property(get_parity)(right)
        return left_uniform == right_uniform

    return Rule(
        id="Halves_uniform_parity_equal",
        name="Both halves uniform in odd/even (or both not)",
        predicate=check,
        family="HIER",
        description="Each half either all odd or all even ranks — and the two halves match on this property.",
        primitives_used=["halves", "uniform_property", "get_parity", "eq"]
    )


def rule_halves_ap_step1_equal() -> Rule:
    """r14x: Halves both form a step-1 run (or both don't)."""
    def is_run(hand: Hand) -> bool:
        if len(hand) < 2:
            return False
        return is_sorted(hand, get_rank_val, strict=True)

    def check(hand: Hand) -> bool:
        left, right = halves(hand)
        return is_run(left) == is_run(right)

    return Rule(
        id="Halves_AP_step1_equal",
        name="Both halves are runs (+1) or both aren't",
        predicate=check,
        family="HIER",
        description="In each half, ranks strictly increase by 1 — either in both halves or in neither.",
        primitives_used=["halves", "is_sorted", "strict", "eq"]
    )


# Family F: Grammar/Language Rules
# ============================================================================

def rule_well_formed_brackets_by_suit() -> Rule:
    """r15x: Well-formed parentheses language over suits."""
    return Rule(
        id="Well_formed_brackets_by_suit",
        name="Suits form matched brackets",
        predicate=bracket_match_suits,
        family="LANG",
        description="Treat ♠/♥ as openers and ♣/♦ as closers of two bracket types; properly nested.",
        primitives_used=["bracket_match_suits"]
    )


def rule_even_opens_next_closes() -> Rule:
    """r16x: Even ranks open; must be closed by the immediate next odd rank."""
    return Rule(
        id="Even_opens_next_closes",
        name="Even opens, next odd closes",
        predicate=lambda h: bracket_match_ranks_even_odd(h, even_opens=True),
        family="LANG",
        description="Even ranks push; only the next odd of the same value closes (e.g., 6 opens; 7 closes).",
        primitives_used=["bracket_match_ranks_even_odd"]
    )


def rule_odd_opens_next_closes() -> Rule:
    """r17x: Odd ranks open; must be closed by the immediate next even rank."""
    return Rule(
        id="Odd_opens_next_closes",
        name="Odd opens, next even closes",
        predicate=lambda h: bracket_match_ranks_even_odd(h, even_opens=False),
        family="LANG",
        description="Odd ranks push; only the next even closes (e.g., 5 opens; 6 closes; Ace(14) can close 13).",
        primitives_used=["bracket_match_ranks_even_odd"]
    )


# Family G: Palindrome Rules
# ============================================================================

def rule_suits_palindrome() -> Rule:
    """r18x: Suits sequence is a palindrome."""
    return Rule(
        id="Suits_palindrome",
        name="Suits read the same forward/back",
        predicate=seq_palindrome(get_suit),
        family="PAL",
        description="The sequence of suits is palindromic.",
        primitives_used=["seq_palindrome", "get_suit"]
    )


def rule_colors_palindrome() -> Rule:
    """r19x: Colors sequence is a palindrome."""
    return Rule(
        id="Colors_palindrome",
        name="Colors read the same forward/back",
        predicate=seq_palindrome(get_color),
        family="PAL",
        description="The sequence of RED/BLACK repeats symmetrically.",
        primitives_used=["seq_palindrome", "get_color"]
    )


def rule_ranks_palindrome() -> Rule:
    """r20x: Ranks sequence is a palindrome."""
    return Rule(
        id="Ranks_palindrome",
        name="Ranks read the same forward/back",
        predicate=seq_palindrome(get_rank),
        family="PAL",
        description="The sequence of ranks is palindromic.",
        primitives_used=["seq_palindrome", "get_rank"]
    )


# Family H: Halves Copy (Sequence Match)
# ============================================================================

def rule_halves_copy_suits() -> Rule:
    """r21x: Left and right halves have identical suit sequence."""
    def suits_seq(h: Hand) -> list:
        return [get_suit(c) for c in h]

    return Rule(
        id="Halves_copy_suits",
        name="Halves have same suit sequence",
        predicate=halves_equal(suits_seq),
        family="HALVES",
        description="Split hand in half; left suits match right suits in order.",
        primitives_used=["halves_equal", "map", "get_suit", "arrays_equal"]
    )


def rule_halves_copy_colors() -> Rule:
    """r22x: Left and right halves have identical color sequence."""
    def colors_seq(h: Hand) -> list:
        return [get_color(c) for c in h]

    return Rule(
        id="Halves_copy_colors",
        name="Halves have same color sequence",
        predicate=halves_equal(colors_seq),
        family="HALVES",
        description="Split hand in half; left colors match right colors in order.",
        primitives_used=["halves_equal", "map", "get_color", "arrays_equal"]
    )


def rule_halves_copy_ranks() -> Rule:
    """r23x: Left and right halves have identical rank sequence."""
    def ranks_seq(h: Hand) -> list:
        return [get_rank(c) for c in h]

    return Rule(
        id="Halves_copy_ranks",
        name="Halves have same rank sequence",
        predicate=halves_equal(ranks_seq),
        family="HALVES",
        description="Split hand in half; left ranks match right ranks in order.",
        primitives_used=["halves_equal", "map", "get_rank", "arrays_equal"]
    )


# Family I: Shift Rules
# ============================================================================

def rule_shift_half_plus_two() -> Rule:
    """r24x: Position i+k has rank = rank_i + 2, where k = floor(n/2)."""
    def check(hand: Hand) -> bool:
        n = len(hand)
        k = n // 2
        pairs = shifted_pairs(k)(hand)
        for c1, c2 in pairs:
            if get_rank_val(c2) != get_rank_val(c1) + 2:
                return False
        return True

    return Rule(
        id="Shift_half_plus_two",
        name="Half-shift positions differ by +2 rank",
        predicate=check,
        family="SHIFT",
        description="Position i and position i+floor(n/2) differ by exactly 2 ranks.",
        primitives_used=["shifted_pairs", "get_rank_val", "eq"]
    )


def rule_shift2_plus3() -> Rule:
    """r25x: Position i+2 has rank = rank_i + 3."""
    def check(hand: Hand) -> bool:
        pairs = shifted_pairs(2)(hand)
        for c1, c2 in pairs:
            if get_rank_val(c2) != get_rank_val(c1) + 3:
                return False
        return True

    return Rule(
        id="Shift2_plus3",
        name="Skip-2 positions differ by +3 rank",
        predicate=check,
        family="SHIFT",
        description="Every pair separated by 2 positions differs by exactly 3 ranks.",
        primitives_used=["shifted_pairs", "get_rank_val", "eq"]
    )


# ============================================================================
# RULE REGISTRY
# ============================================================================

# All rules collected in a registry
ALL_RULES = [
    # Family A: Simple Local
    rule_sorted_by_rank(),
    rule_has_pair_ranks(),
    rule_uniform_color(),
    rule_s_before_h(),
    rule_ap_len3_anywhere_anyk(),
    rule_score_threshold(),

    # Family B: Position
    rule_pos3_is_jqk(),
    rule_pos4_is_2_5_7(),

    # Family C: Token
    rule_has_ace_of_spades(),
    rule_has_6_of_diamonds(),

    # Family D: Count
    rule_exactly_two_suits(),
    rule_half_or_more_same_suit(),

    # Family E: Hierarchical
    rule_halves_uniform_color_equal(),
    rule_halves_uniform_parity_equal(),
    rule_halves_ap_step1_equal(),

    # Family F: Language
    rule_well_formed_brackets_by_suit(),
    rule_even_opens_next_closes(),
    rule_odd_opens_next_closes(),

    # Family G: Palindrome
    rule_suits_palindrome(),
    rule_colors_palindrome(),
    rule_ranks_palindrome(),

    # Family H: Halves Copy
    rule_halves_copy_suits(),
    rule_halves_copy_colors(),
    rule_halves_copy_ranks(),

    # Family I: Shift
    rule_shift_half_plus_two(),
    rule_shift2_plus3(),
]

# Create lookup dictionary
RULE_DICT = {rule.id: rule for rule in ALL_RULES}


def get_rule(rule_id: str) -> Rule:
    """Get rule by ID."""
    if rule_id not in RULE_DICT:
        raise ValueError(f"Unknown rule ID: {rule_id}")
    return RULE_DICT[rule_id]


def get_rules_by_family(family: str) -> list:
    """Get all rules in a given family."""
    return [r for r in ALL_RULES if r.family == family]


if __name__ == "__main__":
    from .cards import sample_hand

    print(f"=== Rule Catalogue ===")
    print(f"Total rules loaded: {len(ALL_RULES)}\n")

    # Print summary by family
    families = {}
    for rule in ALL_RULES:
        families.setdefault(rule.family, []).append(rule)

    for family, rules in sorted(families.items()):
        print(f"{family}: {len(rules)} rules")
        for rule in rules:
            print(f"  - {rule.id}: {rule.name}")
        print()

    # Test a few rules
    print("=== Testing Rules ===\n")

    test_hand = sample_hand(6)
    print(f"Test hand: {test_hand}\n")

    for rule in ALL_RULES[:5]:
        result = rule.eval(test_hand)
        print(f"{rule.id}: {result}")
