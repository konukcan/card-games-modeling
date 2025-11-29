#!/usr/bin/env python3
"""
Pre-training Rules for Warm-Starting Neural Recognition Model

This module defines ~50 rules inspired by common card games that people
are familiar with. The purpose is to:

1. Warm-start the neural recognition model before exposure to experimental rules
2. Build up a library of useful abstractions (like "is_pair", "is_flush", etc.)
3. Give the system "prior experience" similar to what human participants have

The rules are organized by inspiration source:
- Poker: pairs, flushes, straights, etc.
- Blackjack: sum-based rules
- Rummy: runs and sets
- Solitaire: alternating colors, sequences
- Simple structural: position-based, count-based

Design criteria:
- Relatively simple (solvable within reasonable enumeration budget)
- Representative of common card game patterns
- Diverse to cover different primitive combinations
"""

import sys
from pathlib import Path
from typing import List, Callable, Any
from dataclasses import dataclass
from enum import Enum

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import (
    Card, Hand, Suit, Rank, Color,
    RANK_VALUES, card_color, sample_hand
)


# ============================================================================
# RULE DEFINITION
# ============================================================================

@dataclass
class PretrainingRule:
    """A pre-training rule definition."""
    id: str
    name: str
    family: str  # poker, blackjack, rummy, solitaire, simple
    level: int   # 1=easy, 2=medium, 3=hard
    eval: Callable[[Hand], bool]
    description: str
    expected_program: str  # Expected composition in terms of primitives


# ============================================================================
# POKER-INSPIRED RULES
# ============================================================================

def make_poker_rules() -> List[PretrainingRule]:
    """Rules inspired by poker hand rankings."""
    rules = []

    # Has a pair (two cards of same rank)
    rules.append(PretrainingRule(
        id="poker_has_pair",
        name="Has Pair",
        family="poker",
        level=1,
        eval=lambda h: len([r for r in set(c.rank for c in h)
                           if sum(1 for c in h if c.rank == r) >= 2]) > 0,
        description="Hand contains at least one pair (two cards of same rank)",
        expected_program="lt (length (unique (map get_rank hand))) (length hand)"
    ))

    # Has three of a kind
    rules.append(PretrainingRule(
        id="poker_three_of_kind",
        name="Three of a Kind",
        family="poker",
        level=2,
        eval=lambda h: any(sum(1 for c in h if c.rank == r) >= 3
                          for r in set(c.rank for c in h)),
        description="Hand contains three cards of same rank",
        expected_program="any (λr. ge (count (λc. eq (get_rank c) r) hand) 3) ranks"
    ))

    # Flush (all same suit)
    rules.append(PretrainingRule(
        id="poker_flush",
        name="Flush",
        family="poker",
        level=1,
        eval=lambda h: len(set(c.suit for c in h)) == 1 if h else False,
        description="All cards have the same suit",
        expected_program="eq 1 (length (unique (map get_suit hand)))"
    ))

    # All same color (simpler than flush)
    rules.append(PretrainingRule(
        id="poker_same_color",
        name="Same Color",
        family="poker",
        level=1,
        eval=lambda h: len(set(card_color(c) for c in h)) == 1 if h else False,
        description="All cards have the same color (red or black)",
        expected_program="eq 1 (length (unique (map get_color hand)))"
    ))

    # Has at least two suits
    rules.append(PretrainingRule(
        id="poker_two_suits",
        name="Two+ Suits",
        family="poker",
        level=1,
        eval=lambda h: len(set(c.suit for c in h)) >= 2,
        description="Hand contains at least two different suits",
        expected_program="ge (length (unique (map get_suit hand))) 2"
    ))

    # High card (has a face card: J, Q, K, or A)
    rules.append(PretrainingRule(
        id="poker_high_card",
        name="Has Face Card",
        family="poker",
        level=1,
        eval=lambda h: any(c.rank in (Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE)
                          for c in h),
        description="Hand contains at least one face card (J, Q, K, A)",
        expected_program="any (λc. ge (rank_val c) 11) hand"
    ))

    # All face cards
    rules.append(PretrainingRule(
        id="poker_all_face",
        name="All Face Cards",
        family="poker",
        level=2,
        eval=lambda h: all(c.rank in (Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE)
                          for c in h) if h else False,
        description="All cards are face cards",
        expected_program="all (λc. ge (rank_val c) 11) hand"
    ))

    # Has Ace
    rules.append(PretrainingRule(
        id="poker_has_ace",
        name="Has Ace",
        family="poker",
        level=1,
        eval=lambda h: any(c.rank == Rank.ACE for c in h),
        description="Hand contains at least one Ace",
        expected_program="any (λc. eq (rank_val c) 14) hand"
    ))

    # Straight (consecutive ranks) - simplified: just check if sorted unique ranks are consecutive
    def is_straight(h):
        if len(h) < 2:
            return True
        vals = sorted(set(RANK_VALUES[c.rank] for c in h))
        return all(vals[i+1] - vals[i] == 1 for i in range(len(vals)-1))

    rules.append(PretrainingRule(
        id="poker_straight",
        name="Straight",
        family="poker",
        level=2,
        eval=is_straight,
        description="Cards form a straight (consecutive ranks)",
        expected_program="all (λp. eq 1 (- (snd p) (fst p))) (pairs (map rank_val hand))"
    ))

    return rules


# ============================================================================
# BLACKJACK-INSPIRED RULES
# ============================================================================

def make_blackjack_rules() -> List[PretrainingRule]:
    """Rules based on blackjack scoring."""
    rules = []

    # Sum <= 21
    rules.append(PretrainingRule(
        id="bj_under_21",
        name="Under 21",
        family="blackjack",
        level=1,
        eval=lambda h: sum(min(RANK_VALUES[c.rank], 10) for c in h) <= 21,
        description="Sum of card values (face=10) is at most 21",
        expected_program="le (fold (+) 0 (map rank_val hand)) 21"
    ))

    # Sum exactly 21
    rules.append(PretrainingRule(
        id="bj_exactly_21",
        name="Exactly 21",
        family="blackjack",
        level=2,
        eval=lambda h: sum(min(RANK_VALUES[c.rank], 10) for c in h) == 21,
        description="Sum of card values equals exactly 21",
        expected_program="eq 21 (fold (+) 0 (map rank_val hand))"
    ))

    # Sum >= 17 (dealer stands)
    rules.append(PretrainingRule(
        id="bj_stand_17",
        name="Sum >= 17",
        family="blackjack",
        level=1,
        eval=lambda h: sum(RANK_VALUES[c.rank] for c in h) >= 17,
        description="Sum of rank values is at least 17",
        expected_program="ge (fold (+) 0 (map rank_val hand)) 17"
    ))

    # Sum is even
    rules.append(PretrainingRule(
        id="bj_sum_even",
        name="Even Sum",
        family="blackjack",
        level=1,
        eval=lambda h: sum(RANK_VALUES[c.rank] for c in h) % 2 == 0,
        description="Sum of rank values is even",
        expected_program="eq 0 (mod (fold (+) 0 (map rank_val hand)) 2)"
    ))

    # Sum between 15 and 20
    rules.append(PretrainingRule(
        id="bj_safe_range",
        name="Safe Range",
        family="blackjack",
        level=2,
        eval=lambda h: 15 <= sum(RANK_VALUES[c.rank] for c in h) <= 20,
        description="Sum is in the safe range 15-20",
        expected_program="and (ge sum 15) (le sum 20)"
    ))

    return rules


# ============================================================================
# RUMMY-INSPIRED RULES
# ============================================================================

def make_rummy_rules() -> List[PretrainingRule]:
    """Rules inspired by rummy/gin rummy."""
    rules = []

    # Has a run of 3 (three consecutive ranks)
    def has_run_of_3(h):
        vals = sorted(set(RANK_VALUES[c.rank] for c in h))
        for i in range(len(vals) - 2):
            if vals[i+1] == vals[i] + 1 and vals[i+2] == vals[i] + 2:
                return True
        return False

    rules.append(PretrainingRule(
        id="rummy_run_3",
        name="Run of 3",
        family="rummy",
        level=2,
        eval=has_run_of_3,
        description="Contains three consecutive ranks",
        expected_program="any (λt. and (eq (+ (at t 0) 1) (at t 1)) (eq (+ (at t 1) 1) (at t 2))) triples"
    ))

    # Has a set of 3 (three same rank)
    rules.append(PretrainingRule(
        id="rummy_set_3",
        name="Set of 3",
        family="rummy",
        level=2,
        eval=lambda h: any(sum(1 for c in h if c.rank == r) >= 3
                          for r in set(c.rank for c in h)),
        description="Contains three cards of the same rank",
        expected_program="any (λr. ge (count (λc. eq (get_rank c) r) hand) 3) ranks"
    ))

    # All different ranks
    rules.append(PretrainingRule(
        id="rummy_all_different",
        name="All Different Ranks",
        family="rummy",
        level=1,
        eval=lambda h: len(set(c.rank for c in h)) == len(h),
        description="All cards have different ranks",
        expected_program="eq (length hand) (length (unique (map get_rank hand)))"
    ))

    # Exactly 3 different ranks
    rules.append(PretrainingRule(
        id="rummy_three_ranks",
        name="Three Different Ranks",
        family="rummy",
        level=1,
        eval=lambda h: len(set(c.rank for c in h)) == 3,
        description="Hand contains exactly 3 different ranks",
        expected_program="eq 3 (length (unique (map get_rank hand)))"
    ))

    return rules


# ============================================================================
# SOLITAIRE-INSPIRED RULES
# ============================================================================

def make_solitaire_rules() -> List[PretrainingRule]:
    """Rules inspired by solitaire/patience games."""
    rules = []

    # Alternating colors
    def alternating_colors(h):
        if len(h) < 2:
            return True
        colors = [card_color(c) for c in h]
        return all(colors[i] != colors[i+1] for i in range(len(colors)-1))

    rules.append(PretrainingRule(
        id="sol_alternating",
        name="Alternating Colors",
        family="solitaire",
        level=2,
        eval=alternating_colors,
        description="Colors alternate (red-black-red or black-red-black)",
        expected_program="all (λp. neq (get_color (fst p)) (get_color (snd p))) (pairs hand)"
    ))

    # Descending ranks
    def descending_ranks(h):
        if len(h) < 2:
            return True
        vals = [RANK_VALUES[c.rank] for c in h]
        return all(vals[i] > vals[i+1] for i in range(len(vals)-1))

    rules.append(PretrainingRule(
        id="sol_descending",
        name="Descending Ranks",
        family="solitaire",
        level=2,
        eval=descending_ranks,
        description="Ranks decrease from left to right",
        expected_program="all (λp. gt (rank_val (fst p)) (rank_val (snd p))) (pairs hand)"
    ))

    # Ascending ranks
    def ascending_ranks(h):
        if len(h) < 2:
            return True
        vals = [RANK_VALUES[c.rank] for c in h]
        return all(vals[i] < vals[i+1] for i in range(len(vals)-1))

    rules.append(PretrainingRule(
        id="sol_ascending",
        name="Ascending Ranks",
        family="solitaire",
        level=2,
        eval=ascending_ranks,
        description="Ranks increase from left to right",
        expected_program="all (λp. lt (rank_val (fst p)) (rank_val (snd p))) (pairs hand)"
    ))

    # Non-decreasing ranks (sorted)
    def non_decreasing(h):
        if len(h) < 2:
            return True
        vals = [RANK_VALUES[c.rank] for c in h]
        return all(vals[i] <= vals[i+1] for i in range(len(vals)-1))

    rules.append(PretrainingRule(
        id="sol_sorted",
        name="Sorted (Non-decreasing)",
        family="solitaire",
        level=2,
        eval=non_decreasing,
        description="Ranks are in non-decreasing order",
        expected_program="all (λp. le (rank_val (fst p)) (rank_val (snd p))) (pairs hand)"
    ))

    # Same suit sequence
    def same_suit_adjacent(h):
        if len(h) < 2:
            return True
        return all(h[i].suit == h[i+1].suit for i in range(len(h)-1))

    rules.append(PretrainingRule(
        id="sol_same_suit_seq",
        name="Same Suit Sequence",
        family="solitaire",
        level=2,
        eval=same_suit_adjacent,
        description="All adjacent cards have the same suit",
        expected_program="all (λp. eq (get_suit (fst p)) (get_suit (snd p))) (pairs hand)"
    ))

    return rules


# ============================================================================
# SIMPLE STRUCTURAL RULES
# ============================================================================

def make_simple_rules() -> List[PretrainingRule]:
    """Simple structural rules about positions and counts."""
    rules = []

    # First card is red
    rules.append(PretrainingRule(
        id="simple_first_red",
        name="First is Red",
        family="simple",
        level=1,
        eval=lambda h: card_color(h[0]) == Color.RED if h else False,
        description="First card is red",
        expected_program="eq RED (get_color (head hand))"
    ))

    # Last card is black
    rules.append(PretrainingRule(
        id="simple_last_black",
        name="Last is Black",
        family="simple",
        level=1,
        eval=lambda h: card_color(h[-1]) == Color.BLACK if h else False,
        description="Last card is black",
        expected_program="eq BLACK (get_color (last hand))"
    ))

    # First and last same suit
    rules.append(PretrainingRule(
        id="simple_ends_suit",
        name="Same Suit Ends",
        family="simple",
        level=1,
        eval=lambda h: h[0].suit == h[-1].suit if len(h) >= 2 else True,
        description="First and last cards have the same suit",
        expected_program="eq (get_suit (head hand)) (get_suit (last hand))"
    ))

    # First and last same color
    rules.append(PretrainingRule(
        id="simple_ends_color",
        name="Same Color Ends",
        family="simple",
        level=1,
        eval=lambda h: card_color(h[0]) == card_color(h[-1]) if len(h) >= 2 else True,
        description="First and last cards have the same color",
        expected_program="eq (get_color (head hand)) (get_color (last hand))"
    ))

    # First and last same rank
    rules.append(PretrainingRule(
        id="simple_ends_rank",
        name="Same Rank Ends",
        family="simple",
        level=1,
        eval=lambda h: h[0].rank == h[-1].rank if len(h) >= 2 else True,
        description="First and last cards have the same rank",
        expected_program="eq (get_rank (head hand)) (get_rank (last hand))"
    ))

    # Has a spade
    rules.append(PretrainingRule(
        id="simple_has_spade",
        name="Has Spade",
        family="simple",
        level=1,
        eval=lambda h: any(c.suit == Suit.SPADES for c in h),
        description="Hand contains at least one spade",
        expected_program="any (λc. eq SPADES (get_suit c)) hand"
    ))

    # Has a heart
    rules.append(PretrainingRule(
        id="simple_has_heart",
        name="Has Heart",
        family="simple",
        level=1,
        eval=lambda h: any(c.suit == Suit.HEARTS for c in h),
        description="Hand contains at least one heart",
        expected_program="any (λc. eq HEARTS (get_suit c)) hand"
    ))

    # All even ranks
    rules.append(PretrainingRule(
        id="simple_all_even",
        name="All Even Ranks",
        family="simple",
        level=2,
        eval=lambda h: all(RANK_VALUES[c.rank] % 2 == 0 for c in h),
        description="All cards have even rank values",
        expected_program="all (λc. eq 0 (mod (rank_val c) 2)) hand"
    ))

    # All odd ranks
    rules.append(PretrainingRule(
        id="simple_all_odd",
        name="All Odd Ranks",
        family="simple",
        level=2,
        eval=lambda h: all(RANK_VALUES[c.rank] % 2 == 1 for c in h),
        description="All cards have odd rank values",
        expected_program="all (λc. eq 1 (mod (rank_val c) 2)) hand"
    ))

    # More than 3 unique ranks
    rules.append(PretrainingRule(
        id="simple_diverse_ranks",
        name="Diverse Ranks",
        family="simple",
        level=1,
        eval=lambda h: len(set(c.rank for c in h)) > 3,
        description="Hand has more than 3 different ranks",
        expected_program="gt (length (unique (map get_rank hand))) 3"
    ))

    # Exactly 2 suits
    rules.append(PretrainingRule(
        id="simple_two_suits",
        name="Exactly Two Suits",
        family="simple",
        level=1,
        eval=lambda h: len(set(c.suit for c in h)) == 2,
        description="Hand contains exactly 2 different suits",
        expected_program="eq 2 (length (unique (map get_suit hand)))"
    ))

    # Middle card is face card (for 5+ card hands)
    rules.append(PretrainingRule(
        id="simple_middle_face",
        name="Middle is Face",
        family="simple",
        level=2,
        eval=lambda h: h[len(h)//2].rank in (Rank.JACK, Rank.QUEEN, Rank.KING, Rank.ACE) if len(h) >= 3 else False,
        description="Middle card is a face card",
        expected_program="ge (rank_val (at hand (/ (length hand) 2))) 11"
    ))

    return rules


# ============================================================================
# PALINDROME AND SYMMETRY RULES
# ============================================================================

def make_symmetry_rules() -> List[PretrainingRule]:
    """Rules about palindromes and symmetry patterns."""
    rules = []

    # Suits palindrome
    rules.append(PretrainingRule(
        id="sym_suits_palindrome",
        name="Suits Palindrome",
        family="symmetry",
        level=2,
        eval=lambda h: [c.suit for c in h] == [c.suit for c in reversed(h)],
        description="Suit sequence reads the same forwards and backwards",
        expected_program="eq (map get_suit hand) (reverse (map get_suit hand))"
    ))

    # Colors palindrome
    rules.append(PretrainingRule(
        id="sym_colors_palindrome",
        name="Colors Palindrome",
        family="symmetry",
        level=2,
        eval=lambda h: [card_color(c) for c in h] == [card_color(c) for c in reversed(h)],
        description="Color sequence reads the same forwards and backwards",
        expected_program="eq (map get_color hand) (reverse (map get_color hand))"
    ))

    # Ranks palindrome
    rules.append(PretrainingRule(
        id="sym_ranks_palindrome",
        name="Ranks Palindrome",
        family="symmetry",
        level=2,
        eval=lambda h: [RANK_VALUES[c.rank] for c in h] == [RANK_VALUES[c.rank] for c in reversed(h)],
        description="Rank sequence reads the same forwards and backwards",
        expected_program="eq (map rank_val hand) (reverse (map rank_val hand))"
    ))

    # Left half equals right half (suits)
    def halves_equal_suits(h):
        mid = len(h) // 2
        left = [c.suit for c in h[:mid]]
        right = [c.suit for c in h[mid:mid+len(left)]]
        return left == right

    rules.append(PretrainingRule(
        id="sym_halves_suits",
        name="Halves Same Suits",
        family="symmetry",
        level=2,
        eval=halves_equal_suits,
        description="Left and right halves have the same suit sequence",
        expected_program="eq (map get_suit (take (/ (length hand) 2) hand)) (map get_suit (drop (/ (length hand) 2) hand))"
    ))

    return rules


# ============================================================================
# COUNTING RULES
# ============================================================================

def make_counting_rules() -> List[PretrainingRule]:
    """Rules about counting specific cards."""
    rules = []

    # Exactly 2 red cards
    rules.append(PretrainingRule(
        id="count_two_red",
        name="Two Reds",
        family="counting",
        level=1,
        eval=lambda h: sum(1 for c in h if card_color(c) == Color.RED) == 2,
        description="Exactly 2 red cards",
        expected_program="eq 2 (count (λc. eq RED (get_color c)) hand)"
    ))

    # More red than black
    rules.append(PretrainingRule(
        id="count_more_red",
        name="More Red",
        family="counting",
        level=1,
        eval=lambda h: sum(1 for c in h if card_color(c) == Color.RED) > sum(1 for c in h if card_color(c) == Color.BLACK),
        description="More red cards than black cards",
        expected_program="gt (count (λc. eq RED (get_color c)) hand) (count (λc. eq BLACK (get_color c)) hand)"
    ))

    # At least 3 of same suit
    rules.append(PretrainingRule(
        id="count_three_suit",
        name="Three of Suit",
        family="counting",
        level=2,
        eval=lambda h: any(sum(1 for c in h if c.suit == s) >= 3 for s in Suit),
        description="At least 3 cards of the same suit",
        expected_program="any (λs. ge (count (λc. eq s (get_suit c)) hand) 3) suits"
    ))

    # Majority same color
    rules.append(PretrainingRule(
        id="count_majority_color",
        name="Majority Color",
        family="counting",
        level=2,
        eval=lambda h: max(sum(1 for c in h if card_color(c) == col) for col in Color) > len(h) // 2,
        description="More than half the cards are the same color",
        expected_program="gt (count (λc. eq (get_color (head hand)) (get_color c)) hand) (/ (length hand) 2)"
    ))

    return rules


# ============================================================================
# COMPILE ALL PRE-TRAINING RULES
# ============================================================================

def get_all_pretraining_rules() -> List[PretrainingRule]:
    """Get all pre-training rules."""
    all_rules = []
    all_rules.extend(make_poker_rules())
    all_rules.extend(make_blackjack_rules())
    all_rules.extend(make_rummy_rules())
    all_rules.extend(make_solitaire_rules())
    all_rules.extend(make_simple_rules())
    all_rules.extend(make_symmetry_rules())
    all_rules.extend(make_counting_rules())
    return all_rules


def get_easy_pretraining_rules() -> List[PretrainingRule]:
    """Get only level 1 (easy) rules for initial warm-up."""
    return [r for r in get_all_pretraining_rules() if r.level == 1]


def get_rules_by_family(family: str) -> List[PretrainingRule]:
    """Get rules from a specific family."""
    return [r for r in get_all_pretraining_rules() if r.family == family]


# ============================================================================
# SUMMARY AND TEST
# ============================================================================

if __name__ == "__main__":
    all_rules = get_all_pretraining_rules()

    print("=" * 70)
    print("PRE-TRAINING RULES CATALOGUE")
    print("=" * 70)

    print(f"\nTotal rules: {len(all_rules)}")

    # By family
    families = {}
    for r in all_rules:
        families.setdefault(r.family, []).append(r)

    print("\nBy family:")
    for family, rules in sorted(families.items()):
        print(f"  {family}: {len(rules)} rules")

    # By level
    levels = {}
    for r in all_rules:
        levels.setdefault(r.level, []).append(r)

    print("\nBy difficulty level:")
    for level, rules in sorted(levels.items()):
        print(f"  Level {level}: {len(rules)} rules")

    # Test each rule
    print("\nTesting rules on random hands...")
    failed = []
    for rule in all_rules:
        try:
            # Test on a few random hands
            for _ in range(5):
                hand = sample_hand(6)
                result = rule.eval(hand)
                assert isinstance(result, bool), f"Rule {rule.id} didn't return bool"
        except Exception as e:
            failed.append((rule.id, str(e)))

    if failed:
        print(f"\n{len(failed)} rules failed:")
        for rule_id, error in failed:
            print(f"  {rule_id}: {error}")
    else:
        print(f"\nAll {len(all_rules)} rules passed basic tests!")

    # Print some examples
    print("\n" + "=" * 70)
    print("SAMPLE RULES")
    print("=" * 70)

    for family in ['poker', 'blackjack', 'simple']:
        rules = get_rules_by_family(family)[:2]
        for r in rules:
            print(f"\n{r.id} ({r.family}, level {r.level}):")
            print(f"  {r.description}")
            print(f"  Expected: {r.expected_program}")
