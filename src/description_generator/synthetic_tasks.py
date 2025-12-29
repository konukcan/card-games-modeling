#!/usr/bin/env python3
"""
Synthetic Task Generator for Description Generator Testing

This module provides a comprehensive set of procedurally generated card game rules
for testing the description generator. It generates 200+ diverse rules across
five complexity levels:

1. Atomic Rules (Level 1): Single-feature rules
2. Comparison Rules (Level 2): Comparing positions or halves
3. Counting Rules (Level 3): Count-based predicates
4. Pattern Rules (Level 4): Structural patterns
5. Compositional Rules (Level 5): AND/OR combinations

Each generated rule includes:
- Unique ID and human-readable name
- Predicate function (Hand -> bool)
- Expected primitives list (for recognition model training)
- Difficulty level (1-5)
- Methods to sample positive/negative examples

Author: Description Generator Development
"""

import random
import itertools
from dataclasses import dataclass, field
from typing import Callable, List, Set, Dict, Any, Tuple, Optional
from enum import Enum
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import (
    Card, Hand, Suit, Rank, Color,
    RANK_VALUES, card_color, sample_hand, suit_to_color,
    suit_to_altcolor1, suit_to_altcolor2, AltColor1, AltColor2,
    rank_parity, Parity
)


# ============================================================================
# SYNTHETIC RULE DATACLASS
# ============================================================================

@dataclass
class SyntheticRule:
    """
    A synthetically generated rule for testing.

    Attributes:
        id: Unique identifier (e.g., "atomic_001_all_same_suit")
        name: Human-readable name
        predicate: The evaluation function (Hand -> bool)
        expected_primitives: List of primitive names this rule uses
        level: Complexity level (1-5)
        category: Rule category for grouping
        description: Detailed explanation
    """
    id: str
    name: str
    predicate: Callable[[Hand], bool]
    expected_primitives: List[str]
    level: int
    category: str
    description: str

    def eval(self, hand: Hand) -> bool:
        """Evaluate rule on a hand."""
        return self.predicate(hand)

    def sample_positive(self, n: int = 10, max_attempts: int = 1000) -> List[Hand]:
        """
        Sample n hands that satisfy this rule.

        Args:
            n: Number of positive examples to generate
            max_attempts: Maximum sampling attempts

        Returns:
            List of hands that satisfy the rule
        """
        positives = []
        attempts = 0
        while len(positives) < n and attempts < max_attempts:
            hand = sample_hand(6)
            if self.predicate(hand):
                positives.append(hand)
            attempts += 1
        return positives

    def sample_negative(self, n: int = 10, max_attempts: int = 1000) -> List[Hand]:
        """
        Sample n hands that do NOT satisfy this rule.

        Args:
            n: Number of negative examples to generate
            max_attempts: Maximum sampling attempts

        Returns:
            List of hands that don't satisfy the rule
        """
        negatives = []
        attempts = 0
        while len(negatives) < n and attempts < max_attempts:
            hand = sample_hand(6)
            if not self.predicate(hand):
                negatives.append(hand)
            attempts += 1
        return negatives

    def sample_balanced(self, n_each: int = 10, max_attempts: int = 2000) -> Tuple[List[Hand], List[Hand]]:
        """
        Sample balanced positive and negative examples.

        Returns:
            Tuple of (positive_hands, negative_hands)
        """
        return self.sample_positive(n_each, max_attempts), self.sample_negative(n_each, max_attempts)


# ============================================================================
# PRIMITIVE HELPERS (mirroring lean_primitives.py patterns)
# ============================================================================

def get_rank_val(card: Card) -> int:
    """Get numeric rank value (2-14)."""
    return RANK_VALUES[card.rank]

def get_suit(card: Card) -> Suit:
    """Get card suit."""
    return card.suit

def get_rank(card: Card) -> Rank:
    """Get card rank."""
    return card.rank

def get_color(card: Card) -> Color:
    """Get card color."""
    return card_color(card)

def get_parity(card: Card) -> Parity:
    """Get rank parity (odd/even)."""
    return rank_parity(card.rank)

def get_altcolor1(card: Card) -> AltColor1:
    """Get alternative color 1 (pointy/round)."""
    return suit_to_altcolor1(card.suit)

def get_altcolor2(card: Card) -> AltColor2:
    """Get alternative color 2 (SH/DC)."""
    return suit_to_altcolor2(card.suit)

def halves(hand: Hand) -> Tuple[Hand, Hand]:
    """Split hand into left and right halves."""
    mid = len(hand) // 2
    return hand[:mid], hand[mid:2*mid]

def unique_count(getter: Callable, hand: Hand) -> int:
    """Count unique values of a property."""
    return len(set(getter(c) for c in hand))


# ============================================================================
# SYNTHETIC TASK GENERATOR CLASS
# ============================================================================

class SyntheticTaskGenerator:
    """
    Generates a diverse set of synthetic card game rules for testing.

    The generator creates rules across 5 complexity levels with various
    combinations of primitives to ensure comprehensive testing coverage.
    """

    # All suits and colors for parametric generation
    ALL_SUITS = list(Suit)
    ALL_COLORS = list(Color)
    ALL_PARITIES = list(Parity)
    ALL_ALTCOLOR1 = list(AltColor1)
    ALL_ALTCOLOR2 = list(AltColor2)

    def __init__(self, seed: Optional[int] = None):
        """
        Initialize the generator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)
        self.rules: List[SyntheticRule] = []
        self._rule_counter = 0

    def _next_id(self, prefix: str) -> str:
        """Generate next unique rule ID."""
        self._rule_counter += 1
        return f"{prefix}_{self._rule_counter:03d}"

    def generate_all(self) -> List[SyntheticRule]:
        """
        Generate all synthetic rules.

        Returns:
            List of 200+ synthetic rules across all levels
        """
        self.rules = []
        self._rule_counter = 0

        # Level 1: Atomic rules (single-feature)
        self._generate_atomic_rules()

        # Level 2: Comparison rules (comparing positions)
        self._generate_comparison_rules()

        # Level 3: Counting rules
        self._generate_counting_rules()

        # Level 4: Pattern rules
        self._generate_pattern_rules()

        # Level 5: Compositional rules
        self._generate_compositional_rules()

        return self.rules

    # ========================================================================
    # LEVEL 1: ATOMIC RULES (Single-feature)
    # ========================================================================

    def _generate_atomic_rules(self):
        """Generate Level 1 atomic rules."""

        # --- All Same Property ---

        # All same suit
        self.rules.append(SyntheticRule(
            id=self._next_id("atomic"),
            name="All Same Suit",
            predicate=lambda h: len(set(get_suit(c) for c in h)) == 1,
            expected_primitives=["uniform", "get_suit", "unique_count", "eq"],
            level=1,
            category="uniform",
            description="All cards have the same suit"
        ))

        # All same color
        self.rules.append(SyntheticRule(
            id=self._next_id("atomic"),
            name="All Same Color",
            predicate=lambda h: len(set(get_color(c) for c in h)) == 1,
            expected_primitives=["uniform", "get_color", "unique_count", "eq"],
            level=1,
            category="uniform",
            description="All cards have the same color (red or black)"
        ))

        # All same parity
        self.rules.append(SyntheticRule(
            id=self._next_id("atomic"),
            name="All Same Parity",
            predicate=lambda h: len(set(get_parity(c) for c in h)) == 1,
            expected_primitives=["uniform", "get_parity", "unique_count", "eq"],
            level=1,
            category="uniform",
            description="All cards have the same rank parity (all odd or all even)"
        ))

        # All same altcolor1 (pointy/round)
        self.rules.append(SyntheticRule(
            id=self._next_id("atomic"),
            name="All Same Pointy/Round",
            predicate=lambda h: len(set(get_altcolor1(c) for c in h)) == 1,
            expected_primitives=["uniform", "get_altcolor1", "unique_count", "eq"],
            level=1,
            category="uniform",
            description="All cards are pointy (spades/diamonds) or all are round (hearts/clubs)"
        ))

        # All same altcolor2 (SH/DC)
        self.rules.append(SyntheticRule(
            id=self._next_id("atomic"),
            name="All Same SH/DC",
            predicate=lambda h: len(set(get_altcolor2(c) for c in h)) == 1,
            expected_primitives=["uniform", "get_altcolor2", "unique_count", "eq"],
            level=1,
            category="uniform",
            description="All cards are SH (spades/hearts) or all are DC (diamonds/clubs)"
        ))

        # --- First Card Is X ---

        for suit in self.ALL_SUITS:
            self.rules.append(SyntheticRule(
                id=self._next_id("atomic"),
                name=f"First Card is {suit.name.capitalize()}",
                predicate=lambda h, s=suit: get_suit(h[0]) == s if h else False,
                expected_primitives=["first", "get_suit", "eq"],
                level=1,
                category="position_first",
                description=f"First card is a {suit.name.lower()}"
            ))

        for color in self.ALL_COLORS:
            self.rules.append(SyntheticRule(
                id=self._next_id("atomic"),
                name=f"First Card is {color.name.capitalize()}",
                predicate=lambda h, c=color: get_color(h[0]) == c if h else False,
                expected_primitives=["first", "get_color", "eq"],
                level=1,
                category="position_first",
                description=f"First card is {color.name.lower()}"
            ))

        for parity in self.ALL_PARITIES:
            self.rules.append(SyntheticRule(
                id=self._next_id("atomic"),
                name=f"First Card Rank is {parity.name.capitalize()}",
                predicate=lambda h, p=parity: get_parity(h[0]) == p if h else False,
                expected_primitives=["first", "get_parity", "eq"],
                level=1,
                category="position_first",
                description=f"First card has {parity.name.lower()} rank"
            ))

        # --- Last Card Is X ---

        for suit in self.ALL_SUITS:
            self.rules.append(SyntheticRule(
                id=self._next_id("atomic"),
                name=f"Last Card is {suit.name.capitalize()}",
                predicate=lambda h, s=suit: get_suit(h[-1]) == s if h else False,
                expected_primitives=["last", "get_suit", "eq"],
                level=1,
                category="position_last",
                description=f"Last card is a {suit.name.lower()}"
            ))

        for color in self.ALL_COLORS:
            self.rules.append(SyntheticRule(
                id=self._next_id("atomic"),
                name=f"Last Card is {color.name.capitalize()}",
                predicate=lambda h, c=color: get_color(h[-1]) == c if h else False,
                expected_primitives=["last", "get_color", "eq"],
                level=1,
                category="position_last",
                description=f"Last card is {color.name.lower()}"
            ))

        # --- Sum Parity ---

        self.rules.append(SyntheticRule(
            id=self._next_id("atomic"),
            name="Sum of Ranks is Even",
            predicate=lambda h: sum(get_rank_val(c) for c in h) % 2 == 0,
            expected_primitives=["fold", "add", "get_rank_val", "mod", "eq"],
            level=1,
            category="sum",
            description="Sum of all rank values is even"
        ))

        self.rules.append(SyntheticRule(
            id=self._next_id("atomic"),
            name="Sum of Ranks is Odd",
            predicate=lambda h: sum(get_rank_val(c) for c in h) % 2 == 1,
            expected_primitives=["fold", "add", "get_rank_val", "mod", "eq"],
            level=1,
            category="sum",
            description="Sum of all rank values is odd"
        ))

        # --- Has Suit/Color ---

        for suit in self.ALL_SUITS:
            self.rules.append(SyntheticRule(
                id=self._next_id("atomic"),
                name=f"Has at Least One {suit.name.capitalize()}",
                predicate=lambda h, s=suit: any(get_suit(c) == s for c in h),
                expected_primitives=["any", "get_suit", "eq"],
                level=1,
                category="has",
                description=f"Hand contains at least one {suit.name.lower()}"
            ))

        # --- No Suit/Color ---

        for suit in self.ALL_SUITS:
            self.rules.append(SyntheticRule(
                id=self._next_id("atomic"),
                name=f"No {suit.name.capitalize()}s",
                predicate=lambda h, s=suit: not any(get_suit(c) == s for c in h),
                expected_primitives=["not", "any", "get_suit", "eq"],
                level=1,
                category="no",
                description=f"Hand contains no {suit.name.lower()}s"
            ))

        for color in self.ALL_COLORS:
            self.rules.append(SyntheticRule(
                id=self._next_id("atomic"),
                name=f"No {color.name.capitalize()} Cards",
                predicate=lambda h, c=color: not any(get_color(card) == c for card in h),
                expected_primitives=["not", "any", "get_color", "eq"],
                level=1,
                category="no",
                description=f"Hand contains no {color.name.lower()} cards"
            ))

        # --- Middle Position Rules ---

        self.rules.append(SyntheticRule(
            id=self._next_id("atomic"),
            name="Middle Card is Red",
            predicate=lambda h: len(h) >= 3 and get_color(h[len(h)//2]) == Color.RED,
            expected_primitives=["at", "get_color", "eq", "div", "length"],
            level=1,
            category="position_middle",
            description="The middle card is red"
        ))

        self.rules.append(SyntheticRule(
            id=self._next_id("atomic"),
            name="Middle Card is Black",
            predicate=lambda h: len(h) >= 3 and get_color(h[len(h)//2]) == Color.BLACK,
            expected_primitives=["at", "get_color", "eq", "div", "length"],
            level=1,
            category="position_middle",
            description="The middle card is black"
        ))

        for suit in self.ALL_SUITS:
            self.rules.append(SyntheticRule(
                id=self._next_id("atomic"),
                name=f"Middle Card is {suit.name.capitalize()}",
                predicate=lambda h, s=suit: len(h) >= 3 and get_suit(h[len(h)//2]) == s,
                expected_primitives=["at", "get_suit", "eq", "div", "length"],
                level=1,
                category="position_middle",
                description=f"The middle card is a {suit.name.lower()}"
            ))

        # --- Second Position Rules ---

        for suit in self.ALL_SUITS:
            self.rules.append(SyntheticRule(
                id=self._next_id("atomic"),
                name=f"Second Card is {suit.name.capitalize()}",
                predicate=lambda h, s=suit: len(h) >= 2 and get_suit(h[1]) == s,
                expected_primitives=["at", "get_suit", "eq"],
                level=1,
                category="position_second",
                description=f"Second card is a {suit.name.lower()}"
            ))

        for color in self.ALL_COLORS:
            self.rules.append(SyntheticRule(
                id=self._next_id("atomic"),
                name=f"Second Card is {color.name.capitalize()}",
                predicate=lambda h, c=color: len(h) >= 2 and get_color(h[1]) == c,
                expected_primitives=["at", "get_color", "eq"],
                level=1,
                category="position_second",
                description=f"Second card is {color.name.lower()}"
            ))

        # --- Sum Divisibility ---

        for divisor in [3, 4, 5]:
            self.rules.append(SyntheticRule(
                id=self._next_id("atomic"),
                name=f"Sum of Ranks Divisible by {divisor}",
                predicate=lambda h, d=divisor: sum(get_rank_val(c) for c in h) % d == 0,
                expected_primitives=["fold", "add", "get_rank_val", "mod", "eq"],
                level=1,
                category="sum_divisibility",
                description=f"Sum of all rank values is divisible by {divisor}"
            ))

        # --- Range-based Rules ---

        self.rules.append(SyntheticRule(
            id=self._next_id("atomic"),
            name="All Low Cards",
            predicate=lambda h: all(get_rank_val(c) <= 7 for c in h),
            expected_primitives=["all", "get_rank_val", "le"],
            level=1,
            category="range",
            description="All cards have rank 7 or lower"
        ))

        self.rules.append(SyntheticRule(
            id=self._next_id("atomic"),
            name="All High Cards",
            predicate=lambda h: all(get_rank_val(c) >= 8 for c in h),
            expected_primitives=["all", "get_rank_val", "ge"],
            level=1,
            category="range",
            description="All cards have rank 8 or higher"
        ))

        self.rules.append(SyntheticRule(
            id=self._next_id("atomic"),
            name="Has Low Card",
            predicate=lambda h: any(get_rank_val(c) <= 5 for c in h),
            expected_primitives=["any", "get_rank_val", "le"],
            level=1,
            category="range",
            description="Hand contains at least one card with rank 5 or lower"
        ))

        self.rules.append(SyntheticRule(
            id=self._next_id("atomic"),
            name="Has High Card",
            predicate=lambda h: any(get_rank_val(c) >= 10 for c in h),
            expected_primitives=["any", "get_rank_val", "ge"],
            level=1,
            category="range",
            description="Hand contains at least one card with rank 10 or higher"
        ))

    # ========================================================================
    # LEVEL 2: COMPARISON RULES (Comparing positions or halves)
    # ========================================================================

    def _generate_comparison_rules(self):
        """Generate Level 2 comparison rules."""

        # --- First and Last Same Property ---

        self.rules.append(SyntheticRule(
            id=self._next_id("compare"),
            name="First and Last Same Suit",
            predicate=lambda h: len(h) >= 2 and get_suit(h[0]) == get_suit(h[-1]),
            expected_primitives=["first", "last", "get_suit", "eq"],
            level=2,
            category="terminals",
            description="First and last cards have the same suit"
        ))

        self.rules.append(SyntheticRule(
            id=self._next_id("compare"),
            name="First and Last Same Color",
            predicate=lambda h: len(h) >= 2 and get_color(h[0]) == get_color(h[-1]),
            expected_primitives=["first", "last", "get_color", "eq"],
            level=2,
            category="terminals",
            description="First and last cards have the same color"
        ))

        self.rules.append(SyntheticRule(
            id=self._next_id("compare"),
            name="First and Last Same Rank",
            predicate=lambda h: len(h) >= 2 and get_rank(h[0]) == get_rank(h[-1]),
            expected_primitives=["first", "last", "get_rank", "eq"],
            level=2,
            category="terminals",
            description="First and last cards have the same rank"
        ))

        self.rules.append(SyntheticRule(
            id=self._next_id("compare"),
            name="First and Last Same Parity",
            predicate=lambda h: len(h) >= 2 and get_parity(h[0]) == get_parity(h[-1]),
            expected_primitives=["first", "last", "get_parity", "eq"],
            level=2,
            category="terminals",
            description="First and last cards have the same rank parity"
        ))

        self.rules.append(SyntheticRule(
            id=self._next_id("compare"),
            name="First and Last Different Suits",
            predicate=lambda h: len(h) >= 2 and get_suit(h[0]) != get_suit(h[-1]),
            expected_primitives=["first", "last", "get_suit", "neq"],
            level=2,
            category="terminals",
            description="First and last cards have different suits"
        ))

        self.rules.append(SyntheticRule(
            id=self._next_id("compare"),
            name="First and Last Different Colors",
            predicate=lambda h: len(h) >= 2 and get_color(h[0]) != get_color(h[-1]),
            expected_primitives=["first", "last", "get_color", "neq"],
            level=2,
            category="terminals",
            description="First and last cards have different colors"
        ))

        # --- First Rank vs Last Rank ---

        self.rules.append(SyntheticRule(
            id=self._next_id("compare"),
            name="First Rank Greater Than Last",
            predicate=lambda h: len(h) >= 2 and get_rank_val(h[0]) > get_rank_val(h[-1]),
            expected_primitives=["first", "last", "get_rank_val", "gt"],
            level=2,
            category="terminals_rank",
            description="First card has higher rank than last card"
        ))

        self.rules.append(SyntheticRule(
            id=self._next_id("compare"),
            name="First Rank Less Than Last",
            predicate=lambda h: len(h) >= 2 and get_rank_val(h[0]) < get_rank_val(h[-1]),
            expected_primitives=["first", "last", "get_rank_val", "lt"],
            level=2,
            category="terminals_rank",
            description="First card has lower rank than last card"
        ))

        # --- Both Halves Have Same Property ---

        def both_halves_uniform_color(h: Hand) -> bool:
            left, right = halves(h)
            left_uniform = len(set(get_color(c) for c in left)) == 1 if left else True
            right_uniform = len(set(get_color(c) for c in right)) == 1 if right else True
            return left_uniform == right_uniform

        self.rules.append(SyntheticRule(
            id=self._next_id("compare"),
            name="Halves Uniform Color Equal",
            predicate=both_halves_uniform_color,
            expected_primitives=["halves", "uniform", "get_color", "eq"],
            level=2,
            category="halves_property",
            description="Both halves are uniformly colored, or both are not"
        ))

        def both_halves_uniform_suit(h: Hand) -> bool:
            left, right = halves(h)
            left_uniform = len(set(get_suit(c) for c in left)) == 1 if left else True
            right_uniform = len(set(get_suit(c) for c in right)) == 1 if right else True
            return left_uniform == right_uniform

        self.rules.append(SyntheticRule(
            id=self._next_id("compare"),
            name="Halves Uniform Suit Equal",
            predicate=both_halves_uniform_suit,
            expected_primitives=["halves", "uniform", "get_suit", "eq"],
            level=2,
            category="halves_property",
            description="Both halves have uniform suit, or both don't"
        ))

        # --- Halves Same Suit Set ---

        def halves_same_suit_set(h: Hand) -> bool:
            left, right = halves(h)
            return set(get_suit(c) for c in left) == set(get_suit(c) for c in right)

        self.rules.append(SyntheticRule(
            id=self._next_id("compare"),
            name="Halves Same Suit Set",
            predicate=halves_same_suit_set,
            expected_primitives=["halves", "unique", "get_suit", "eq"],
            level=2,
            category="halves_set",
            description="Both halves contain the same set of suits"
        ))

        def halves_same_color_set(h: Hand) -> bool:
            left, right = halves(h)
            return set(get_color(c) for c in left) == set(get_color(c) for c in right)

        self.rules.append(SyntheticRule(
            id=self._next_id("compare"),
            name="Halves Same Color Set",
            predicate=halves_same_color_set,
            expected_primitives=["halves", "unique", "get_color", "eq"],
            level=2,
            category="halves_set",
            description="Both halves contain the same set of colors"
        ))

        # --- Half Sum Comparisons ---

        def left_half_sum_greater(h: Hand) -> bool:
            left, right = halves(h)
            left_sum = sum(get_rank_val(c) for c in left)
            right_sum = sum(get_rank_val(c) for c in right)
            return left_sum > right_sum

        self.rules.append(SyntheticRule(
            id=self._next_id("compare"),
            name="Left Half Sum Greater",
            predicate=left_half_sum_greater,
            expected_primitives=["halves", "sum", "get_rank_val", "gt"],
            level=2,
            category="halves_sum",
            description="Sum of ranks in left half is greater than right half"
        ))

        def left_half_sum_equal(h: Hand) -> bool:
            left, right = halves(h)
            left_sum = sum(get_rank_val(c) for c in left)
            right_sum = sum(get_rank_val(c) for c in right)
            return left_sum == right_sum

        self.rules.append(SyntheticRule(
            id=self._next_id("compare"),
            name="Halves Equal Sum",
            predicate=left_half_sum_equal,
            expected_primitives=["halves", "sum", "get_rank_val", "eq"],
            level=2,
            category="halves_sum",
            description="Sum of ranks in left half equals right half"
        ))

        # --- Adjacent Card Comparisons ---

        self.rules.append(SyntheticRule(
            id=self._next_id("compare"),
            name="All Adjacent Same Color",
            predicate=lambda h: all(get_color(h[i]) == get_color(h[i+1]) for i in range(len(h)-1)),
            expected_primitives=["all", "adjacent_pairs", "get_color", "eq"],
            level=2,
            category="adjacent",
            description="All adjacent cards have the same color"
        ))

        self.rules.append(SyntheticRule(
            id=self._next_id("compare"),
            name="All Adjacent Different Colors",
            predicate=lambda h: all(get_color(h[i]) != get_color(h[i+1]) for i in range(len(h)-1)),
            expected_primitives=["all", "adjacent_pairs", "get_color", "neq"],
            level=2,
            category="adjacent",
            description="All adjacent cards have different colors (alternating)"
        ))

        # --- Position-based comparisons ---

        for i in range(3):  # positions 0, 1, 2
            for j in range(i+1, min(i+3, 6)):  # nearby positions
                self.rules.append(SyntheticRule(
                    id=self._next_id("compare"),
                    name=f"Position {i} and {j} Same Suit",
                    predicate=lambda h, a=i, b=j: len(h) > max(a, b) and get_suit(h[a]) == get_suit(h[b]),
                    expected_primitives=["at", "get_suit", "eq"],
                    level=2,
                    category="position_pair",
                    description=f"Cards at positions {i} and {j} have the same suit"
                ))

    # ========================================================================
    # LEVEL 3: COUNTING RULES
    # ========================================================================

    def _generate_counting_rules(self):
        """Generate Level 3 counting rules."""

        # --- Exactly N Unique Values ---

        for n in [1, 2, 3, 4]:
            self.rules.append(SyntheticRule(
                id=self._next_id("count"),
                name=f"Exactly {n} Unique Suit{'s' if n != 1 else ''}",
                predicate=lambda h, num=n: len(set(get_suit(c) for c in h)) == num,
                expected_primitives=["unique_count", "get_suit", "eq"],
                level=3,
                category="unique_count",
                description=f"Hand contains exactly {n} different suit{'s' if n != 1 else ''}"
            ))

        for n in [1, 2]:
            self.rules.append(SyntheticRule(
                id=self._next_id("count"),
                name=f"Exactly {n} Unique Color{'s' if n != 1 else ''}",
                predicate=lambda h, num=n: len(set(get_color(c) for c in h)) == num,
                expected_primitives=["unique_count", "get_color", "eq"],
                level=3,
                category="unique_count",
                description=f"Hand contains exactly {n} different color{'s' if n != 1 else ''}"
            ))

        for n in [2, 3, 4, 5, 6]:
            self.rules.append(SyntheticRule(
                id=self._next_id("count"),
                name=f"Exactly {n} Unique Ranks",
                predicate=lambda h, num=n: len(set(get_rank(c) for c in h)) == num,
                expected_primitives=["unique_count", "get_rank", "eq"],
                level=3,
                category="unique_count",
                description=f"Hand contains exactly {n} different ranks"
            ))

        # --- At Most / At Least N Unique Values ---

        for n in [2, 3]:
            self.rules.append(SyntheticRule(
                id=self._next_id("count"),
                name=f"At Most {n} Unique Suits",
                predicate=lambda h, num=n: len(set(get_suit(c) for c in h)) <= num,
                expected_primitives=["unique_count", "get_suit", "le"],
                level=3,
                category="unique_bound",
                description=f"Hand contains at most {n} different suits"
            ))

        for n in [2, 3, 4]:
            self.rules.append(SyntheticRule(
                id=self._next_id("count"),
                name=f"At Least {n} Unique Suits",
                predicate=lambda h, num=n: len(set(get_suit(c) for c in h)) >= num,
                expected_primitives=["unique_count", "get_suit", "ge"],
                level=3,
                category="unique_bound",
                description=f"Hand contains at least {n} different suits"
            ))

        # --- Count of Specific Suit/Color ---

        for suit in self.ALL_SUITS:
            for n in [1, 2, 3]:
                self.rules.append(SyntheticRule(
                    id=self._next_id("count"),
                    name=f"Exactly {n} {suit.name.capitalize()}{'s' if n != 1 else ''}",
                    predicate=lambda h, s=suit, num=n: sum(1 for c in h if get_suit(c) == s) == num,
                    expected_primitives=["count_equal", "get_suit", "eq"],
                    level=3,
                    category="count_specific",
                    description=f"Hand contains exactly {n} {suit.name.lower()}{'s' if n != 1 else ''}"
                ))

        for color in self.ALL_COLORS:
            for n in [1, 2, 3, 4]:
                self.rules.append(SyntheticRule(
                    id=self._next_id("count"),
                    name=f"Exactly {n} {color.name.capitalize()} Card{'s' if n != 1 else ''}",
                    predicate=lambda h, c=color, num=n: sum(1 for card in h if get_color(card) == c) == num,
                    expected_primitives=["count_equal", "get_color", "eq"],
                    level=3,
                    category="count_specific",
                    description=f"Hand contains exactly {n} {color.name.lower()} card{'s' if n != 1 else ''}"
                ))

        # --- More/Less Comparisons ---

        self.rules.append(SyntheticRule(
            id=self._next_id("count"),
            name="More Red Than Black",
            predicate=lambda h: sum(1 for c in h if get_color(c) == Color.RED) > sum(1 for c in h if get_color(c) == Color.BLACK),
            expected_primitives=["count_equal", "get_color", "gt"],
            level=3,
            category="count_compare",
            description="More red cards than black cards"
        ))

        self.rules.append(SyntheticRule(
            id=self._next_id("count"),
            name="More Black Than Red",
            predicate=lambda h: sum(1 for c in h if get_color(c) == Color.BLACK) > sum(1 for c in h if get_color(c) == Color.RED),
            expected_primitives=["count_equal", "get_color", "gt"],
            level=3,
            category="count_compare",
            description="More black cards than red cards"
        ))

        self.rules.append(SyntheticRule(
            id=self._next_id("count"),
            name="Equal Red and Black",
            predicate=lambda h: sum(1 for c in h if get_color(c) == Color.RED) == sum(1 for c in h if get_color(c) == Color.BLACK),
            expected_primitives=["count_equal", "get_color", "eq"],
            level=3,
            category="count_compare",
            description="Equal number of red and black cards"
        ))

        # --- Majority Rules ---

        self.rules.append(SyntheticRule(
            id=self._next_id("count"),
            name="Majority Same Suit",
            predicate=lambda h: max(sum(1 for c in h if get_suit(c) == s) for s in Suit) > len(h) // 2,
            expected_primitives=["max_count", "get_suit", "length", "div", "gt"],
            level=3,
            category="majority",
            description="More than half the cards share the same suit"
        ))

        self.rules.append(SyntheticRule(
            id=self._next_id("count"),
            name="Majority Same Color",
            predicate=lambda h: max(sum(1 for c in h if get_color(c) == col) for col in Color) > len(h) // 2,
            expected_primitives=["max_count", "get_color", "length", "div", "gt"],
            level=3,
            category="majority",
            description="More than half the cards share the same color"
        ))

        # --- Count Parity Constraints ---

        for n in [1, 2, 3]:
            self.rules.append(SyntheticRule(
                id=self._next_id("count"),
                name=f"Exactly {n} Odd Rank{'s' if n != 1 else ''}",
                predicate=lambda h, num=n: sum(1 for c in h if get_rank_val(c) % 2 == 1) == num,
                expected_primitives=["count", "filter", "is_odd", "get_rank_val", "eq"],
                level=3,
                category="count_parity",
                description=f"Exactly {n} card{'s' if n != 1 else ''} with odd rank"
            ))

        for n in [1, 2, 3]:
            self.rules.append(SyntheticRule(
                id=self._next_id("count"),
                name=f"Exactly {n} Even Rank{'s' if n != 1 else ''}",
                predicate=lambda h, num=n: sum(1 for c in h if get_rank_val(c) % 2 == 0) == num,
                expected_primitives=["count", "filter", "is_even", "get_rank_val", "eq"],
                level=3,
                category="count_parity",
                description=f"Exactly {n} card{'s' if n != 1 else ''} with even rank"
            ))

    # ========================================================================
    # LEVEL 4: PATTERN RULES
    # ========================================================================

    def _generate_pattern_rules(self):
        """Generate Level 4 pattern rules."""

        # --- Sorted Patterns ---

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Sorted by Rank (Non-decreasing)",
            predicate=lambda h: all(get_rank_val(h[i]) <= get_rank_val(h[i+1]) for i in range(len(h)-1)),
            expected_primitives=["is_sorted", "get_rank_val"],
            level=4,
            category="sorted",
            description="Ranks are in non-decreasing order left to right"
        ))

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Sorted by Rank (Strictly Increasing)",
            predicate=lambda h: all(get_rank_val(h[i]) < get_rank_val(h[i+1]) for i in range(len(h)-1)),
            expected_primitives=["is_sorted_strict", "get_rank_val"],
            level=4,
            category="sorted",
            description="Ranks are in strictly increasing order left to right"
        ))

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Sorted by Rank (Non-increasing)",
            predicate=lambda h: all(get_rank_val(h[i]) >= get_rank_val(h[i+1]) for i in range(len(h)-1)),
            expected_primitives=["is_sorted_desc", "get_rank_val"],
            level=4,
            category="sorted",
            description="Ranks are in non-increasing order left to right"
        ))

        # --- Alternating Patterns ---

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Alternating Colors",
            predicate=lambda h: all(get_color(h[i]) != get_color(h[i+1]) for i in range(len(h)-1)),
            expected_primitives=["all", "adjacent_pairs", "get_color", "neq"],
            level=4,
            category="alternating",
            description="Colors alternate (red-black-red-black... or black-red-black-red...)"
        ))

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Alternating Parities",
            predicate=lambda h: all(get_parity(h[i]) != get_parity(h[i+1]) for i in range(len(h)-1)),
            expected_primitives=["all", "adjacent_pairs", "get_parity", "neq"],
            level=4,
            category="alternating",
            description="Rank parities alternate (odd-even-odd-even... or even-odd-even-odd...)"
        ))

        # --- Pair/Triple Detection ---

        def has_pair(h: Hand) -> bool:
            ranks = [get_rank(c) for c in h]
            return len(ranks) != len(set(ranks))

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Has Pair (Same Rank)",
            predicate=has_pair,
            expected_primitives=["unique_count", "get_rank", "length", "lt"],
            level=4,
            category="duplicates",
            description="At least two cards share the same rank"
        ))

        def has_triple(h: Hand) -> bool:
            from collections import Counter
            rank_counts = Counter(get_rank(c) for c in h)
            return any(count >= 3 for count in rank_counts.values())

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Has Triple (Three of a Kind)",
            predicate=has_triple,
            expected_primitives=["any", "count_equal", "get_rank", "ge"],
            level=4,
            category="duplicates",
            description="At least three cards share the same rank"
        ))

        def has_suit_pair(h: Hand) -> bool:
            """At least two cards of same suit in adjacent positions."""
            return any(get_suit(h[i]) == get_suit(h[i+1]) for i in range(len(h)-1))

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Has Adjacent Suit Pair",
            predicate=has_suit_pair,
            expected_primitives=["any", "adjacent_pairs", "get_suit", "eq"],
            level=4,
            category="duplicates",
            description="At least two adjacent cards share the same suit"
        ))

        # --- Palindrome Patterns ---

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Suits Palindrome",
            predicate=lambda h: [get_suit(c) for c in h] == [get_suit(c) for c in reversed(h)],
            expected_primitives=["seq_palindrome", "get_suit", "map", "reverse", "eq"],
            level=4,
            category="palindrome",
            description="Suit sequence reads the same forwards and backwards"
        ))

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Colors Palindrome",
            predicate=lambda h: [get_color(c) for c in h] == [get_color(c) for c in reversed(h)],
            expected_primitives=["seq_palindrome", "get_color", "map", "reverse", "eq"],
            level=4,
            category="palindrome",
            description="Color sequence reads the same forwards and backwards"
        ))

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Ranks Palindrome",
            predicate=lambda h: [get_rank(c) for c in h] == [get_rank(c) for c in reversed(h)],
            expected_primitives=["seq_palindrome", "get_rank", "map", "reverse", "eq"],
            level=4,
            category="palindrome",
            description="Rank sequence reads the same forwards and backwards"
        ))

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Parities Palindrome",
            predicate=lambda h: [get_parity(c) for c in h] == [get_parity(c) for c in reversed(h)],
            expected_primitives=["seq_palindrome", "get_parity", "map", "reverse", "eq"],
            level=4,
            category="palindrome",
            description="Parity sequence reads the same forwards and backwards"
        ))

        # --- Halves Copy Patterns ---

        def halves_copy_suits(h: Hand) -> bool:
            left, right = halves(h)
            return [get_suit(c) for c in left] == [get_suit(c) for c in right]

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Halves Copy Suits",
            predicate=halves_copy_suits,
            expected_primitives=["halves_equal", "get_suit", "map", "eq"],
            level=4,
            category="halves_copy",
            description="Left and right halves have the same suit sequence"
        ))

        def halves_copy_colors(h: Hand) -> bool:
            left, right = halves(h)
            return [get_color(c) for c in left] == [get_color(c) for c in right]

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Halves Copy Colors",
            predicate=halves_copy_colors,
            expected_primitives=["halves_equal", "get_color", "map", "eq"],
            level=4,
            category="halves_copy",
            description="Left and right halves have the same color sequence"
        ))

        def halves_copy_ranks(h: Hand) -> bool:
            left, right = halves(h)
            return [get_rank(c) for c in left] == [get_rank(c) for c in right]

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Halves Copy Ranks",
            predicate=halves_copy_ranks,
            expected_primitives=["halves_equal", "get_rank", "map", "eq"],
            level=4,
            category="halves_copy",
            description="Left and right halves have the same rank sequence"
        ))

        # --- Run/Consecutive Patterns ---

        def has_run_of_3(h: Hand) -> bool:
            vals = sorted(set(get_rank_val(c) for c in h))
            for i in range(len(vals) - 2):
                if vals[i+1] == vals[i] + 1 and vals[i+2] == vals[i] + 2:
                    return True
            return False

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Has Run of 3",
            predicate=has_run_of_3,
            expected_primitives=["has_AP", "get_rank_val", "unique", "sort"],
            level=4,
            category="runs",
            description="Contains at least 3 consecutive ranks (ignoring position)"
        ))

        def has_run_of_4(h: Hand) -> bool:
            vals = sorted(set(get_rank_val(c) for c in h))
            for i in range(len(vals) - 3):
                if all(vals[i+j+1] == vals[i] + j + 1 for j in range(3)):
                    return True
            return False

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Has Run of 4",
            predicate=has_run_of_4,
            expected_primitives=["has_AP", "get_rank_val", "unique", "sort"],
            level=4,
            category="runs",
            description="Contains at least 4 consecutive ranks (ignoring position)"
        ))

        # --- Adjacent Constraint Patterns ---

        def adj_rank_diff_le_2(h: Hand) -> bool:
            return all(abs(get_rank_val(h[i]) - get_rank_val(h[i+1])) <= 2 for i in range(len(h)-1))

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Adjacent Rank Gap <= 2",
            predicate=adj_rank_diff_le_2,
            expected_primitives=["all", "adjacent_pairs", "get_rank_val", "abs", "diff", "le"],
            level=4,
            category="adjacent_constraint",
            description="All adjacent cards differ by at most 2 in rank"
        ))

        def adj_rank_diff_le_3(h: Hand) -> bool:
            return all(abs(get_rank_val(h[i]) - get_rank_val(h[i+1])) <= 3 for i in range(len(h)-1))

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Adjacent Rank Gap <= 3",
            predicate=adj_rank_diff_le_3,
            expected_primitives=["all", "adjacent_pairs", "get_rank_val", "abs", "diff", "le"],
            level=4,
            category="adjacent_constraint",
            description="All adjacent cards differ by at most 3 in rank"
        ))

        def adj_same_rank_or_suit(h: Hand) -> bool:
            return all(get_rank(h[i]) == get_rank(h[i+1]) or get_suit(h[i]) == get_suit(h[i+1])
                      for i in range(len(h)-1))

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Adjacent Same Rank or Suit",
            predicate=adj_same_rank_or_suit,
            expected_primitives=["all", "adjacent_pairs", "get_rank", "get_suit", "eq", "or"],
            level=4,
            category="adjacent_constraint",
            description="All adjacent cards share either rank or suit"
        ))

        # --- V-Shape and Wave Patterns ---

        def v_shape_ranks(h: Hand) -> bool:
            if len(h) < 3:
                return True
            vals = [get_rank_val(c) for c in h]
            mid = len(vals) // 2
            desc = all(vals[i] >= vals[i+1] for i in range(mid))
            asc = all(vals[i] <= vals[i+1] for i in range(mid, len(vals)-1))
            return desc and asc

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="V-Shape Ranks",
            predicate=v_shape_ranks,
            expected_primitives=["and", "all", "get_rank_val", "ge", "le", "halves"],
            level=4,
            category="shape",
            description="Ranks descend to middle then ascend (V shape)"
        ))

        def inverted_v_shape(h: Hand) -> bool:
            if len(h) < 3:
                return True
            vals = [get_rank_val(c) for c in h]
            mid = len(vals) // 2
            asc = all(vals[i] <= vals[i+1] for i in range(mid))
            desc = all(vals[i] >= vals[i+1] for i in range(mid, len(vals)-1))
            return asc and desc

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Inverted V-Shape Ranks",
            predicate=inverted_v_shape,
            expected_primitives=["and", "all", "get_rank_val", "ge", "le", "halves"],
            level=4,
            category="shape",
            description="Ranks ascend to middle then descend (inverted V)"
        ))

        # --- Periodic Patterns ---

        def period_2_colors(h: Hand) -> bool:
            if len(h) < 2:
                return True
            colors = [get_color(c) for c in h]
            return all(colors[i] == colors[i % 2] for i in range(len(colors)))

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Period-2 Colors",
            predicate=period_2_colors,
            expected_primitives=["all", "get_color", "mod", "eq"],
            level=4,
            category="periodic",
            description="Colors repeat with period 2 (ABABAB...)"
        ))

        def period_2_suits(h: Hand) -> bool:
            if len(h) < 2:
                return True
            suits = [get_suit(c) for c in h]
            return all(suits[i] == suits[i % 2] for i in range(len(suits)))

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Period-2 Suits",
            predicate=period_2_suits,
            expected_primitives=["all", "get_suit", "mod", "eq"],
            level=4,
            category="periodic",
            description="Suits repeat with period 2 (ABABAB...)"
        ))

        # --- Skip Patterns ---

        def skip1_same_suit(h: Hand) -> bool:
            if len(h) < 3:
                return True
            return all(get_suit(h[i]) == get_suit(h[i+2]) for i in range(len(h)-2))

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Skip-1 Same Suit",
            predicate=skip1_same_suit,
            expected_primitives=["all", "shifted_pairs", "get_suit", "eq"],
            level=4,
            category="skip",
            description="Cards at positions i and i+2 have same suit"
        ))

        def skip1_same_color(h: Hand) -> bool:
            if len(h) < 3:
                return True
            return all(get_color(h[i]) == get_color(h[i+2]) for i in range(len(h)-2))

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Skip-1 Same Color",
            predicate=skip1_same_color,
            expected_primitives=["all", "shifted_pairs", "get_color", "eq"],
            level=4,
            category="skip",
            description="Cards at positions i and i+2 have same color"
        ))

        # --- Spread Patterns ---

        def max_rank_spread_ge_8(h: Hand) -> bool:
            vals = [get_rank_val(c) for c in h]
            return max(vals) - min(vals) >= 8

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Rank Spread >= 8",
            predicate=max_rank_spread_ge_8,
            expected_primitives=["max", "min", "get_rank_val", "sub", "ge"],
            level=4,
            category="spread",
            description="Difference between highest and lowest rank is at least 8"
        ))

        def max_rank_spread_le_5(h: Hand) -> bool:
            vals = [get_rank_val(c) for c in h]
            return max(vals) - min(vals) <= 5

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Rank Spread <= 5",
            predicate=max_rank_spread_le_5,
            expected_primitives=["max", "min", "get_rank_val", "sub", "le"],
            level=4,
            category="spread",
            description="Difference between highest and lowest rank is at most 5"
        ))

        # --- Monotonicity Patterns ---

        def non_strictly_monotonic(h: Hand) -> bool:
            vals = [get_rank_val(c) for c in h]
            non_dec = all(vals[i] <= vals[i+1] for i in range(len(vals)-1))
            non_inc = all(vals[i] >= vals[i+1] for i in range(len(vals)-1))
            return non_dec or non_inc

        self.rules.append(SyntheticRule(
            id=self._next_id("pattern"),
            name="Monotonic Ranks",
            predicate=non_strictly_monotonic,
            expected_primitives=["or", "is_sorted", "is_sorted_desc", "get_rank_val"],
            level=4,
            category="monotonic",
            description="Ranks are either non-decreasing or non-increasing"
        ))

    # ========================================================================
    # LEVEL 5: COMPOSITIONAL RULES
    # ========================================================================

    def _generate_compositional_rules(self):
        """Generate Level 5 compositional rules (AND/OR combinations)."""

        # --- AND Combinations ---

        # Uniform color AND sorted
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="All Same Color AND Sorted",
            predicate=lambda h: (len(set(get_color(c) for c in h)) == 1 and
                                all(get_rank_val(h[i]) <= get_rank_val(h[i+1]) for i in range(len(h)-1))),
            expected_primitives=["and", "uniform", "get_color", "is_sorted", "get_rank_val"],
            level=5,
            category="and_combination",
            description="All cards same color AND ranks are sorted"
        ))

        # Has pair AND alternating colors
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Has Pair AND Alternating Colors",
            predicate=lambda h: (len([get_rank(c) for c in h]) != len(set(get_rank(c) for c in h)) and
                                all(get_color(h[i]) != get_color(h[i+1]) for i in range(len(h)-1))),
            expected_primitives=["and", "has_pair", "get_rank", "alternating", "get_color"],
            level=5,
            category="and_combination",
            description="Has a pair AND colors alternate"
        ))

        # First is red AND last is black
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="First Red AND Last Black",
            predicate=lambda h: len(h) >= 2 and get_color(h[0]) == Color.RED and get_color(h[-1]) == Color.BLACK,
            expected_primitives=["and", "first", "last", "get_color", "eq"],
            level=5,
            category="and_combination",
            description="First card is red AND last card is black"
        ))

        # First is black AND last is red
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="First Black AND Last Red",
            predicate=lambda h: len(h) >= 2 and get_color(h[0]) == Color.BLACK and get_color(h[-1]) == Color.RED,
            expected_primitives=["and", "first", "last", "get_color", "eq"],
            level=5,
            category="and_combination",
            description="First card is black AND last card is red"
        ))

        # Exactly 2 suits AND has pair
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Exactly 2 Suits AND Has Pair",
            predicate=lambda h: (len(set(get_suit(c) for c in h)) == 2 and
                                len([get_rank(c) for c in h]) != len(set(get_rank(c) for c in h))),
            expected_primitives=["and", "unique_count", "get_suit", "eq", "has_pair", "get_rank"],
            level=5,
            category="and_combination",
            description="Exactly 2 suits AND has at least one pair"
        ))

        # Sum even AND first red
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Sum Even AND First Red",
            predicate=lambda h: sum(get_rank_val(c) for c in h) % 2 == 0 and get_color(h[0]) == Color.RED if h else False,
            expected_primitives=["and", "sum", "get_rank_val", "mod", "eq", "first", "get_color"],
            level=5,
            category="and_combination",
            description="Sum of ranks is even AND first card is red"
        ))

        # Halves same color set AND sorted
        def halves_same_color_and_sorted(h: Hand) -> bool:
            left, right = halves(h)
            same_set = set(get_color(c) for c in left) == set(get_color(c) for c in right)
            sorted_ = all(get_rank_val(h[i]) <= get_rank_val(h[i+1]) for i in range(len(h)-1))
            return same_set and sorted_

        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Halves Same Colors AND Sorted",
            predicate=halves_same_color_and_sorted,
            expected_primitives=["and", "halves", "unique", "get_color", "eq", "is_sorted", "get_rank_val"],
            level=5,
            category="and_combination",
            description="Both halves have same color set AND ranks are sorted"
        ))

        # --- OR Combinations ---

        # All same suit OR all same color
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="All Same Suit OR All Same Color",
            predicate=lambda h: len(set(get_suit(c) for c in h)) == 1 or len(set(get_color(c) for c in h)) == 1,
            expected_primitives=["or", "uniform", "get_suit", "get_color"],
            level=5,
            category="or_combination",
            description="All cards same suit OR all cards same color"
        ))

        # Has pair OR sorted
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Has Pair OR Sorted",
            predicate=lambda h: (len([get_rank(c) for c in h]) != len(set(get_rank(c) for c in h)) or
                                all(get_rank_val(h[i]) <= get_rank_val(h[i+1]) for i in range(len(h)-1))),
            expected_primitives=["or", "has_pair", "get_rank", "is_sorted", "get_rank_val"],
            level=5,
            category="or_combination",
            description="Has a pair OR ranks are sorted"
        ))

        # First red OR last red
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="First Red OR Last Red",
            predicate=lambda h: len(h) >= 1 and (get_color(h[0]) == Color.RED or get_color(h[-1]) == Color.RED),
            expected_primitives=["or", "first", "last", "get_color", "eq"],
            level=5,
            category="or_combination",
            description="First card is red OR last card is red"
        ))

        # Palindrome suits OR palindrome colors
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Palindrome Suits OR Palindrome Colors",
            predicate=lambda h: ([get_suit(c) for c in h] == [get_suit(c) for c in reversed(h)] or
                                [get_color(c) for c in h] == [get_color(c) for c in reversed(h)]),
            expected_primitives=["or", "seq_palindrome", "get_suit", "get_color", "map", "reverse", "eq"],
            level=5,
            category="or_combination",
            description="Suits form palindrome OR colors form palindrome"
        ))

        # Sum even OR has spade
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Sum Even OR Has Spade",
            predicate=lambda h: sum(get_rank_val(c) for c in h) % 2 == 0 or any(get_suit(c) == Suit.SPADES for c in h),
            expected_primitives=["or", "sum", "get_rank_val", "mod", "eq", "any", "get_suit"],
            level=5,
            category="or_combination",
            description="Sum of ranks is even OR has at least one spade"
        ))

        # --- Conditional Rules (If X then Y) ---

        # If first is red, then last is black
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="If First Red Then Last Black",
            predicate=lambda h: len(h) < 2 or get_color(h[0]) != Color.RED or get_color(h[-1]) == Color.BLACK,
            expected_primitives=["implies", "first", "last", "get_color", "eq"],
            level=5,
            category="conditional",
            description="If first card is red, then last card must be black"
        ))

        # If has spade, then has heart
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="If Has Spade Then Has Heart",
            predicate=lambda h: not any(get_suit(c) == Suit.SPADES for c in h) or any(get_suit(c) == Suit.HEARTS for c in h),
            expected_primitives=["implies", "any", "get_suit", "eq"],
            level=5,
            category="conditional",
            description="If hand has a spade, it must also have a heart"
        ))

        # If sorted, then has pair
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="If Sorted Then Has Pair",
            predicate=lambda h: (not all(get_rank_val(h[i]) <= get_rank_val(h[i+1]) for i in range(len(h)-1)) or
                                len([get_rank(c) for c in h]) != len(set(get_rank(c) for c in h))),
            expected_primitives=["implies", "is_sorted", "get_rank_val", "has_pair", "get_rank"],
            level=5,
            category="conditional",
            description="If ranks are sorted, then must have at least one pair"
        ))

        # If uniform color, then not sorted
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="If Uniform Color Then Not Sorted",
            predicate=lambda h: (len(set(get_color(c) for c in h)) != 1 or
                                not all(get_rank_val(h[i]) <= get_rank_val(h[i+1]) for i in range(len(h)-1))),
            expected_primitives=["implies", "uniform", "get_color", "not", "is_sorted", "get_rank_val"],
            level=5,
            category="conditional",
            description="If all cards same color, then ranks must not be sorted"
        ))

        # --- Complex Nested Combinations ---

        # (First red AND last black) OR (first black AND last red)
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Opposite Terminal Colors",
            predicate=lambda h: len(h) >= 2 and get_color(h[0]) != get_color(h[-1]),
            expected_primitives=["or", "and", "first", "last", "get_color", "eq", "neq"],
            level=5,
            category="complex",
            description="First and last cards have opposite colors"
        ))

        # All same color AND (sorted OR has pair)
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Same Color AND (Sorted OR Pair)",
            predicate=lambda h: (len(set(get_color(c) for c in h)) == 1 and
                                (all(get_rank_val(h[i]) <= get_rank_val(h[i+1]) for i in range(len(h)-1)) or
                                 len([get_rank(c) for c in h]) != len(set(get_rank(c) for c in h)))),
            expected_primitives=["and", "or", "uniform", "get_color", "is_sorted", "get_rank_val", "has_pair", "get_rank"],
            level=5,
            category="complex",
            description="All same color AND (sorted OR has pair)"
        ))

        # (Has spade AND has heart) OR (has diamond AND has club)
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Has Both of a Color Pair",
            predicate=lambda h: ((any(get_suit(c) == Suit.SPADES for c in h) and any(get_suit(c) == Suit.HEARTS for c in h)) or
                                (any(get_suit(c) == Suit.DIAMONDS for c in h) and any(get_suit(c) == Suit.CLUBS for c in h))),
            expected_primitives=["or", "and", "any", "get_suit", "eq"],
            level=5,
            category="complex",
            description="Has (spade AND heart) OR (diamond AND club)"
        ))

        # More red than black AND first is red
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="More Red AND First Red",
            predicate=lambda h: (sum(1 for c in h if get_color(c) == Color.RED) >
                                sum(1 for c in h if get_color(c) == Color.BLACK) and
                                get_color(h[0]) == Color.RED if h else False),
            expected_primitives=["and", "count_equal", "get_color", "gt", "first", "eq"],
            level=5,
            category="complex",
            description="More red cards than black AND first card is red"
        ))

        # --- XOR-like Combinations ---

        # Exactly one of: all same suit, all same color
        def xor_same_suit_color(h: Hand) -> bool:
            same_suit = len(set(get_suit(c) for c in h)) == 1
            same_color = len(set(get_color(c) for c in h)) == 1
            return same_suit != same_color

        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Same Suit XOR Same Color",
            predicate=xor_same_suit_color,
            expected_primitives=["xor", "uniform", "get_suit", "get_color"],
            level=5,
            category="xor",
            description="All same suit XOR all same color (exactly one, not both)"
        ))

        # Exactly one of: sorted, has pair
        def xor_sorted_pair(h: Hand) -> bool:
            sorted_ = all(get_rank_val(h[i]) <= get_rank_val(h[i+1]) for i in range(len(h)-1))
            has_pair_ = len([get_rank(c) for c in h]) != len(set(get_rank(c) for c in h))
            return sorted_ != has_pair_

        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Sorted XOR Has Pair",
            predicate=xor_sorted_pair,
            expected_primitives=["xor", "is_sorted", "get_rank_val", "has_pair", "get_rank"],
            level=5,
            category="xor",
            description="Sorted XOR has pair (exactly one, not both)"
        ))

        # --- More AND Combinations ---

        # Alternating colors AND first is spade
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Alternating Colors AND First Spade",
            predicate=lambda h: (all(get_color(h[i]) != get_color(h[i+1]) for i in range(len(h)-1)) and
                                get_suit(h[0]) == Suit.SPADES if h else False),
            expected_primitives=["and", "all", "adjacent_pairs", "get_color", "neq", "first", "get_suit", "eq"],
            level=5,
            category="and_combination",
            description="Colors alternate AND first card is a spade"
        ))

        # Has run of 3 AND uniform color
        def has_run_3_and_uniform(h: Hand) -> bool:
            vals = sorted(set(get_rank_val(c) for c in h))
            has_run = any(vals[i+1] == vals[i] + 1 and vals[i+2] == vals[i] + 2 for i in range(len(vals) - 2)) if len(vals) >= 3 else False
            uniform = len(set(get_color(c) for c in h)) == 1
            return has_run and uniform

        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Has Run of 3 AND Uniform Color",
            predicate=has_run_3_and_uniform,
            expected_primitives=["and", "has_run", "get_rank_val", "uniform", "get_color"],
            level=5,
            category="and_combination",
            description="Contains 3 consecutive ranks AND all cards same color"
        ))

        # Palindrome colors AND has heart
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Palindrome Colors AND Has Heart",
            predicate=lambda h: ([get_color(c) for c in h] == [get_color(c) for c in reversed(h)] and
                                any(get_suit(c) == Suit.HEARTS for c in h)),
            expected_primitives=["and", "seq_palindrome", "get_color", "any", "get_suit", "eq"],
            level=5,
            category="and_combination",
            description="Color sequence is palindrome AND has at least one heart"
        ))

        # More black than red AND last is black
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="More Black AND Last Black",
            predicate=lambda h: (sum(1 for c in h if get_color(c) == Color.BLACK) >
                                sum(1 for c in h if get_color(c) == Color.RED) and
                                get_color(h[-1]) == Color.BLACK if h else False),
            expected_primitives=["and", "count_equal", "get_color", "gt", "last", "eq"],
            level=5,
            category="and_combination",
            description="More black cards than red AND last card is black"
        ))

        # At least 3 suits AND sorted
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="At Least 3 Suits AND Sorted",
            predicate=lambda h: (len(set(get_suit(c) for c in h)) >= 3 and
                                all(get_rank_val(h[i]) <= get_rank_val(h[i+1]) for i in range(len(h)-1))),
            expected_primitives=["and", "unique_count", "get_suit", "ge", "is_sorted", "get_rank_val"],
            level=5,
            category="and_combination",
            description="At least 3 different suits AND ranks are sorted"
        ))

        # --- More OR Combinations ---

        # Has triple OR all same color
        def has_triple_or_uniform(h: Hand) -> bool:
            from collections import Counter
            rank_counts = Counter(get_rank(c) for c in h)
            has_trip = any(count >= 3 for count in rank_counts.values())
            uniform = len(set(get_color(c) for c in h)) == 1
            return has_trip or uniform

        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Has Triple OR All Same Color",
            predicate=has_triple_or_uniform,
            expected_primitives=["or", "any", "count_equal", "get_rank", "ge", "uniform", "get_color"],
            level=5,
            category="or_combination",
            description="Has three of a kind OR all cards same color"
        ))

        # First black OR last black
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="First Black OR Last Black",
            predicate=lambda h: len(h) >= 1 and (get_color(h[0]) == Color.BLACK or get_color(h[-1]) == Color.BLACK),
            expected_primitives=["or", "first", "last", "get_color", "eq"],
            level=5,
            category="or_combination",
            description="First card is black OR last card is black"
        ))

        # Sum even OR alternating colors
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Sum Even OR Alternating Colors",
            predicate=lambda h: (sum(get_rank_val(c) for c in h) % 2 == 0 or
                                all(get_color(h[i]) != get_color(h[i+1]) for i in range(len(h)-1))),
            expected_primitives=["or", "sum", "get_rank_val", "mod", "eq", "all", "adjacent_pairs", "get_color", "neq"],
            level=5,
            category="or_combination",
            description="Sum of ranks is even OR colors alternate"
        ))

        # More unique suits than colors OR has pair
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="More Suit Variety OR Has Pair",
            predicate=lambda h: (len(set(get_suit(c) for c in h)) > len(set(get_color(c) for c in h)) or
                                len([get_rank(c) for c in h]) != len(set(get_rank(c) for c in h))),
            expected_primitives=["or", "unique_count", "get_suit", "get_color", "gt", "has_pair", "get_rank"],
            level=5,
            category="or_combination",
            description="More unique suits than colors OR has a pair"
        ))

        # --- More Conditional Rules ---

        # If all same color, then sum is even
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="If Uniform Color Then Sum Even",
            predicate=lambda h: len(set(get_color(c) for c in h)) != 1 or sum(get_rank_val(c) for c in h) % 2 == 0,
            expected_primitives=["implies", "uniform", "get_color", "sum", "get_rank_val", "mod", "eq"],
            level=5,
            category="conditional",
            description="If all cards same color, then sum of ranks must be even"
        ))

        # If has pair, then not palindrome suits
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="If Has Pair Then Not Palindrome Suits",
            predicate=lambda h: (len([get_rank(c) for c in h]) == len(set(get_rank(c) for c in h)) or
                                [get_suit(c) for c in h] != [get_suit(c) for c in reversed(h)]),
            expected_primitives=["implies", "has_pair", "get_rank", "not", "seq_palindrome", "get_suit"],
            level=5,
            category="conditional",
            description="If hand has a pair, then suits must not form palindrome"
        ))

        # If first is heart, then last is diamond
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="If First Heart Then Last Diamond",
            predicate=lambda h: len(h) < 2 or get_suit(h[0]) != Suit.HEARTS or get_suit(h[-1]) == Suit.DIAMONDS,
            expected_primitives=["implies", "first", "last", "get_suit", "eq"],
            level=5,
            category="conditional",
            description="If first card is heart, then last card must be diamond"
        ))

        # If more red than black, then first is red
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="If More Red Then First Red",
            predicate=lambda h: (sum(1 for c in h if get_color(c) == Color.RED) <=
                                sum(1 for c in h if get_color(c) == Color.BLACK) or
                                get_color(h[0]) == Color.RED if h else True),
            expected_primitives=["implies", "count_equal", "get_color", "gt", "first", "eq"],
            level=5,
            category="conditional",
            description="If more red than black, then first card must be red"
        ))

        # --- Triple Combinations ---

        # All same color AND sorted AND has pair
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Same Color AND Sorted AND Pair",
            predicate=lambda h: (len(set(get_color(c) for c in h)) == 1 and
                                all(get_rank_val(h[i]) <= get_rank_val(h[i+1]) for i in range(len(h)-1)) and
                                len([get_rank(c) for c in h]) != len(set(get_rank(c) for c in h))),
            expected_primitives=["and", "uniform", "get_color", "is_sorted", "get_rank_val", "has_pair", "get_rank"],
            level=5,
            category="triple",
            description="All same color AND sorted AND has at least one pair"
        ))

        # First red AND last black AND has spade
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="First Red AND Last Black AND Has Spade",
            predicate=lambda h: (len(h) >= 2 and get_color(h[0]) == Color.RED and
                                get_color(h[-1]) == Color.BLACK and
                                any(get_suit(c) == Suit.SPADES for c in h)),
            expected_primitives=["and", "first", "last", "get_color", "eq", "any", "get_suit"],
            level=5,
            category="triple",
            description="First is red AND last is black AND has at least one spade"
        ))

        # --- Negation Combinations ---

        # Not (all same suit)
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Not All Same Suit",
            predicate=lambda h: len(set(get_suit(c) for c in h)) > 1,
            expected_primitives=["not", "uniform", "get_suit"],
            level=5,
            category="negation",
            description="Not all cards have the same suit (at least 2 different suits)"
        ))

        # Not sorted
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Not Sorted",
            predicate=lambda h: not all(get_rank_val(h[i]) <= get_rank_val(h[i+1]) for i in range(len(h)-1)),
            expected_primitives=["not", "is_sorted", "get_rank_val"],
            level=5,
            category="negation",
            description="Ranks are NOT in non-decreasing order"
        ))

        # Not palindrome colors
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Not Palindrome Colors",
            predicate=lambda h: [get_color(c) for c in h] != [get_color(c) for c in reversed(h)],
            expected_primitives=["not", "seq_palindrome", "get_color"],
            level=5,
            category="negation",
            description="Color sequence is NOT a palindrome"
        ))

        # --- Biconditional (Iff) ---

        # All same suit IFF all same color
        # Note: same suit implies same color, so this is really about when neither holds
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Same Suit IFF Same Color",
            predicate=lambda h: (len(set(get_suit(c) for c in h)) == 1) == (len(set(get_color(c) for c in h)) == 1),
            expected_primitives=["iff", "uniform", "get_suit", "get_color"],
            level=5,
            category="biconditional",
            description="All same suit if and only if all same color"
        ))

        # Has pair IFF more than 4 unique ranks
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Has Pair IFF <= 5 Unique Ranks",
            predicate=lambda h: (len([get_rank(c) for c in h]) != len(set(get_rank(c) for c in h))) == (len(set(get_rank(c) for c in h)) <= 5),
            expected_primitives=["iff", "has_pair", "get_rank", "unique_count", "le"],
            level=5,
            category="biconditional",
            description="Has pair if and only if at most 5 unique ranks"
        ))

        # --- Additional Diverse Compositions ---

        # Neither all same suit nor all same color
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Neither Same Suit Nor Same Color",
            predicate=lambda h: len(set(get_suit(c) for c in h)) > 1 and len(set(get_color(c) for c in h)) > 1,
            expected_primitives=["and", "not", "uniform", "get_suit", "get_color"],
            level=5,
            category="negation",
            description="Not all same suit AND not all same color"
        ))

        # Both have heart AND have club
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Has Heart AND Has Club",
            predicate=lambda h: (any(get_suit(c) == Suit.HEARTS for c in h) and
                                any(get_suit(c) == Suit.CLUBS for c in h)),
            expected_primitives=["and", "any", "get_suit", "eq"],
            level=5,
            category="and_combination",
            description="Has at least one heart AND at least one club"
        ))

        # Has diamond AND has spade
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Has Diamond AND Has Spade",
            predicate=lambda h: (any(get_suit(c) == Suit.DIAMONDS for c in h) and
                                any(get_suit(c) == Suit.SPADES for c in h)),
            expected_primitives=["and", "any", "get_suit", "eq"],
            level=5,
            category="and_combination",
            description="Has at least one diamond AND at least one spade"
        ))

        # All 4 suits present
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="All Four Suits Present",
            predicate=lambda h: len(set(get_suit(c) for c in h)) == 4,
            expected_primitives=["unique_count", "get_suit", "eq"],
            level=5,
            category="and_combination",
            description="Hand contains cards from all four suits"
        ))

        # Exactly one suit missing
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Exactly One Suit Missing",
            predicate=lambda h: len(set(get_suit(c) for c in h)) == 3,
            expected_primitives=["unique_count", "get_suit", "eq"],
            level=5,
            category="and_combination",
            description="Exactly three suits are present (one is missing)"
        ))

        # Has all even OR has all odd ranks
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="All Even OR All Odd Ranks",
            predicate=lambda h: (all(get_rank_val(c) % 2 == 0 for c in h) or
                                all(get_rank_val(c) % 2 == 1 for c in h)),
            expected_primitives=["or", "all", "get_rank_val", "mod", "eq"],
            level=5,
            category="or_combination",
            description="All cards have even ranks OR all cards have odd ranks"
        ))

        # First two same suit AND last two same suit
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="First Pair AND Last Pair Same Suit",
            predicate=lambda h: (len(h) >= 4 and get_suit(h[0]) == get_suit(h[1]) and
                                get_suit(h[-2]) == get_suit(h[-1])),
            expected_primitives=["and", "at", "get_suit", "eq"],
            level=5,
            category="and_combination",
            description="First two cards share suit AND last two cards share suit"
        ))

        # First two same color OR last two same color
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="First Pair OR Last Pair Same Color",
            predicate=lambda h: (len(h) >= 4 and (get_color(h[0]) == get_color(h[1]) or
                                get_color(h[-2]) == get_color(h[-1]))),
            expected_primitives=["or", "at", "get_color", "eq"],
            level=5,
            category="or_combination",
            description="First two cards share color OR last two cards share color"
        ))

        # No pair AND not sorted
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="No Pair AND Not Sorted",
            predicate=lambda h: (len([get_rank(c) for c in h]) == len(set(get_rank(c) for c in h)) and
                                not all(get_rank_val(h[i]) <= get_rank_val(h[i+1]) for i in range(len(h)-1))),
            expected_primitives=["and", "not", "has_pair", "get_rank", "is_sorted", "get_rank_val"],
            level=5,
            category="negation",
            description="No pairs AND ranks are not sorted"
        ))

        # Sum divisible by 3 AND has heart
        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Sum Div 3 AND Has Heart",
            predicate=lambda h: (sum(get_rank_val(c) for c in h) % 3 == 0 and
                                any(get_suit(c) == Suit.HEARTS for c in h)),
            expected_primitives=["and", "sum", "get_rank_val", "mod", "eq", "any", "get_suit"],
            level=5,
            category="and_combination",
            description="Sum of ranks divisible by 3 AND has at least one heart"
        ))

        # Halves copy colors AND first is black
        def halves_copy_and_first_black(h: Hand) -> bool:
            left, right = halves(h)
            copy = [get_color(c) for c in left] == [get_color(c) for c in right]
            return copy and get_color(h[0]) == Color.BLACK if h else False

        self.rules.append(SyntheticRule(
            id=self._next_id("comp"),
            name="Halves Copy Colors AND First Black",
            predicate=halves_copy_and_first_black,
            expected_primitives=["and", "halves_equal", "get_color", "first", "eq"],
            level=5,
            category="complex",
            description="Left and right halves have same color sequence AND first card is black"
        ))

    # ========================================================================
    # SUMMARY AND STATISTICS
    # ========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics of generated rules."""
        if not self.rules:
            return {"error": "No rules generated. Call generate_all() first."}

        by_level = {}
        by_category = {}

        for rule in self.rules:
            by_level.setdefault(rule.level, []).append(rule)
            by_category.setdefault(rule.category, []).append(rule)

        return {
            "total_rules": len(self.rules),
            "by_level": {level: len(rules) for level, rules in sorted(by_level.items())},
            "by_category": {cat: len(rules) for cat, rules in sorted(by_category.items())},
            "level_details": {
                1: "Atomic (single-feature)",
                2: "Comparison (positions/halves)",
                3: "Counting (count-based predicates)",
                4: "Pattern (structural patterns)",
                5: "Compositional (AND/OR combinations)"
            }
        }

    def get_rules_by_level(self, level: int) -> List[SyntheticRule]:
        """Get all rules at a specific complexity level."""
        return [r for r in self.rules if r.level == level]

    def get_rules_by_category(self, category: str) -> List[SyntheticRule]:
        """Get all rules in a specific category."""
        return [r for r in self.rules if r.category == category]

    def get_rule_by_id(self, rule_id: str) -> Optional[SyntheticRule]:
        """Get a specific rule by ID."""
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        return None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def generate_synthetic_rules(seed: Optional[int] = 42) -> List[SyntheticRule]:
    """
    Generate all synthetic rules with default settings.

    Args:
        seed: Random seed for reproducibility

    Returns:
        List of 200+ synthetic rules
    """
    generator = SyntheticTaskGenerator(seed=seed)
    return generator.generate_all()


def get_rule_dict(rules: List[SyntheticRule]) -> Dict[str, SyntheticRule]:
    """Convert list of rules to dictionary keyed by ID."""
    return {rule.id: rule for rule in rules}


# ============================================================================
# MAIN: TEST AND DISPLAY
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SYNTHETIC TASK GENERATOR")
    print("=" * 70)

    # Generate all rules
    generator = SyntheticTaskGenerator(seed=42)
    rules = generator.generate_all()

    # Print summary
    summary = generator.get_summary()
    print(f"\nTotal rules generated: {summary['total_rules']}")

    print("\nRules by level:")
    for level, count in summary['by_level'].items():
        print(f"  Level {level} ({summary['level_details'][level]}): {count}")

    print("\nRules by category:")
    for cat, count in sorted(summary['by_category'].items()):
        print(f"  {cat}: {count}")

    # Test a few rules
    print("\n" + "=" * 70)
    print("TESTING SAMPLE RULES")
    print("=" * 70)

    # Test one rule from each level
    for level in range(1, 6):
        level_rules = generator.get_rules_by_level(level)
        if level_rules:
            rule = level_rules[0]
            print(f"\nLevel {level}: {rule.name}")
            print(f"  ID: {rule.id}")
            print(f"  Category: {rule.category}")
            print(f"  Description: {rule.description}")
            print(f"  Primitives: {rule.expected_primitives}")

            # Sample hands
            positives = rule.sample_positive(3)
            negatives = rule.sample_negative(3)

            print(f"  Positive examples: {len(positives)}")
            print(f"  Negative examples: {len(negatives)}")

    # Verify all rules work
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    failed = []
    for rule in rules:
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
        print(f"\nAll {len(rules)} rules passed verification!")

    print(f"\nRules saved in: synthetic_tasks.py")
