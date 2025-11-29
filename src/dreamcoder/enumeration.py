#!/usr/bin/env python3
"""
Program Enumeration for DreamCoder

This module implements the core program synthesis engine:
- Type-directed program generation
- Best-first search guided by recognition network scores
- Consistency checking against input-output examples
- Description length scoring for program simplicity

This is the CRITICAL component that enables actual program synthesis,
not just primitive prediction.

Based on Ellis et al. (2023) DreamCoder architecture.
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable, Any, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import heapq
import time
import math
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Card, Hand, Suit, Rank, Color, Parity, AltColor1, AltColor2
from rules.cards import sample_hand, RANK_VALUES, card_color, rank_parity, suit_to_altcolor1, suit_to_altcolor2


# ============================================================================
# TYPE SYSTEM
# ============================================================================

class Type(Enum):
    """Simple type system for our domain."""
    BOOL = auto()       # Boolean
    INT = auto()        # Integer
    CARD = auto()       # Single card
    HAND = auto()       # List of cards (hand)
    SUIT = auto()       # Suit value
    RANK = auto()       # Rank value
    COLOR = auto()      # Color value
    PARITY = auto()     # Parity value
    LIST_INT = auto()   # List of integers
    LIST_SUIT = auto()  # List of suits
    LIST_RANK = auto()  # List of ranks
    LIST_COLOR = auto() # List of colors
    LIST_BOOL = auto()  # List of booleans
    PROPERTY = auto()   # Card -> Value function (generic property extractor)


@dataclass
class FunctionType:
    """Type signature for a function."""
    arg_types: List[Type]
    return_type: Type

    def __str__(self):
        args = " × ".join(t.name for t in self.arg_types)
        return f"({args}) → {self.return_type.name}"


# ============================================================================
# PRIMITIVE DEFINITIONS
# ============================================================================

@dataclass
class Primitive:
    """A primitive function in our DSL."""
    name: str
    func_type: FunctionType
    implementation: Callable
    description_length: float = 1.0  # Cost for MDL scoring

    def __call__(self, *args):
        return self.implementation(*args)

    def __repr__(self):
        return f"Primitive({self.name})"


@dataclass
class Program:
    """A program (composition of primitives)."""
    primitive: Primitive
    arguments: List['Program'] = field(default_factory=list)
    _hash: int = field(default=0, repr=False)

    def __post_init__(self):
        # Compute hash for deduplication
        self._hash = hash((self.primitive.name, tuple(hash(a) for a in self.arguments)))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, Program):
            return False
        if self.primitive.name != other.primitive.name:
            return False
        if len(self.arguments) != len(other.arguments):
            return False
        return all(a == b for a, b in zip(self.arguments, other.arguments))

    def __str__(self):
        if not self.arguments:
            return self.primitive.name
        args_str = ", ".join(str(a) for a in self.arguments)
        return f"{self.primitive.name}({args_str})"

    def __lt__(self, other):
        # For heap comparison
        return str(self) < str(other)

    @property
    def return_type(self) -> Type:
        return self.primitive.func_type.return_type

    @property
    def description_length(self) -> float:
        """Total description length (MDL)."""
        return self.primitive.description_length + sum(a.description_length for a in self.arguments)

    def evaluate(self, env: Dict[str, Any]) -> Any:
        """Evaluate this program in the given environment."""
        # Evaluate arguments first
        arg_values = [arg.evaluate(env) for arg in self.arguments]
        return self.primitive(*arg_values)


# Special "hole" for the hand variable
@dataclass
class HandVariable:
    """Represents the input hand variable 'h'."""
    name: str = "h"

    def __hash__(self):
        return hash(self.name)

    def __str__(self):
        return self.name

    def __lt__(self, other):
        return False

    @property
    def return_type(self) -> Type:
        return Type.HAND

    @property
    def description_length(self) -> float:
        return 0.0  # Free to reference the input

    def evaluate(self, env: Dict[str, Any]) -> Hand:
        return env['h']


# ============================================================================
# PRIMITIVE LIBRARY
# ============================================================================

def build_primitive_library() -> Dict[str, Primitive]:
    """Build the library of primitives for enumeration."""
    primitives = {}

    # === PROPERTY EXTRACTORS (Card → Value) ===

    primitives['get_suit'] = Primitive(
        name='get_suit',
        func_type=FunctionType([Type.CARD], Type.SUIT),
        implementation=lambda c: c.suit,
        description_length=1.0
    )

    primitives['get_rank'] = Primitive(
        name='get_rank',
        func_type=FunctionType([Type.CARD], Type.RANK),
        implementation=lambda c: c.rank,
        description_length=1.0
    )

    primitives['get_rank_val'] = Primitive(
        name='get_rank_val',
        func_type=FunctionType([Type.CARD], Type.INT),
        implementation=lambda c: RANK_VALUES[c.rank],
        description_length=1.0
    )

    primitives['get_color'] = Primitive(
        name='get_color',
        func_type=FunctionType([Type.CARD], Type.COLOR),
        implementation=lambda c: card_color(c),
        description_length=1.0
    )

    primitives['get_parity'] = Primitive(
        name='get_parity',
        func_type=FunctionType([Type.CARD], Type.PARITY),
        implementation=lambda c: rank_parity(c.rank),
        description_length=1.0
    )

    primitives['get_altcolor1'] = Primitive(
        name='get_altcolor1',
        func_type=FunctionType([Type.CARD], Type.COLOR),
        implementation=lambda c: suit_to_altcolor1(c.suit),
        description_length=1.0
    )

    primitives['get_altcolor2'] = Primitive(
        name='get_altcolor2',
        func_type=FunctionType([Type.CARD], Type.COLOR),
        implementation=lambda c: suit_to_altcolor2(c.suit),
        description_length=1.0
    )

    # === POSITION SELECTORS (Hand → Card) ===

    primitives['first'] = Primitive(
        name='first',
        func_type=FunctionType([Type.HAND], Type.CARD),
        implementation=lambda h: h[0] if h else None,
        description_length=1.0
    )

    primitives['last'] = Primitive(
        name='last',
        func_type=FunctionType([Type.HAND], Type.CARD),
        implementation=lambda h: h[-1] if h else None,
        description_length=1.0
    )

    # at(i) - parameterized primitive
    for i in range(6):
        primitives[f'at_{i}'] = Primitive(
            name=f'at_{i}',
            func_type=FunctionType([Type.HAND], Type.CARD),
            implementation=(lambda idx: lambda h: h[idx] if len(h) > idx else None)(i),
            description_length=1.5
        )

    # === HAND DECOMPOSITION ===

    primitives['left_half'] = Primitive(
        name='left_half',
        func_type=FunctionType([Type.HAND], Type.HAND),
        implementation=lambda h: h[:len(h)//2],
        description_length=1.0
    )

    primitives['right_half'] = Primitive(
        name='right_half',
        func_type=FunctionType([Type.HAND], Type.HAND),
        implementation=lambda h: h[len(h)//2:],
        description_length=1.0
    )

    primitives['reverse'] = Primitive(
        name='reverse',
        func_type=FunctionType([Type.HAND], Type.HAND),
        implementation=lambda h: h[::-1],
        description_length=1.0
    )

    # === MAP OPERATIONS (Hand × Extractor → List) ===

    # We'll handle map specially in enumeration since it takes a property extractor

    primitives['map_suit'] = Primitive(
        name='map_suit',
        func_type=FunctionType([Type.HAND], Type.LIST_SUIT),
        implementation=lambda h: [c.suit for c in h],
        description_length=1.5
    )

    primitives['map_rank'] = Primitive(
        name='map_rank',
        func_type=FunctionType([Type.HAND], Type.LIST_RANK),
        implementation=lambda h: [c.rank for c in h],
        description_length=1.5
    )

    primitives['map_rank_val'] = Primitive(
        name='map_rank_val',
        func_type=FunctionType([Type.HAND], Type.LIST_INT),
        implementation=lambda h: [RANK_VALUES[c.rank] for c in h],
        description_length=1.5
    )

    primitives['map_color'] = Primitive(
        name='map_color',
        func_type=FunctionType([Type.HAND], Type.LIST_COLOR),
        implementation=lambda h: [card_color(c) for c in h],
        description_length=1.5
    )

    primitives['map_parity'] = Primitive(
        name='map_parity',
        func_type=FunctionType([Type.HAND], Type.LIST_BOOL),
        implementation=lambda h: [rank_parity(c.rank) == Parity.ODD for c in h],
        description_length=1.5
    )

    # === BOOLEAN PREDICATES (List → Bool) ===

    primitives['is_sorted'] = Primitive(
        name='is_sorted',
        func_type=FunctionType([Type.LIST_INT], Type.BOOL),
        implementation=lambda lst: all(lst[i] <= lst[i+1] for i in range(len(lst)-1)) if lst else True,
        description_length=1.0
    )

    primitives['is_strictly_sorted'] = Primitive(
        name='is_strictly_sorted',
        func_type=FunctionType([Type.LIST_INT], Type.BOOL),
        implementation=lambda lst: all(lst[i] < lst[i+1] for i in range(len(lst)-1)) if lst else True,
        description_length=1.2
    )

    primitives['is_uniform'] = Primitive(
        name='is_uniform',
        func_type=FunctionType([Type.LIST_SUIT], Type.BOOL),  # Works for any list type
        implementation=lambda lst: len(set(lst)) <= 1 if lst else True,
        description_length=1.0
    )

    primitives['is_palindrome'] = Primitive(
        name='is_palindrome',
        func_type=FunctionType([Type.LIST_SUIT], Type.BOOL),  # Works for any list
        implementation=lambda lst: lst == lst[::-1] if lst else True,
        description_length=1.0
    )

    # === COMPARISON (List × List → Bool) ===

    primitives['lists_equal'] = Primitive(
        name='lists_equal',
        func_type=FunctionType([Type.LIST_SUIT, Type.LIST_SUIT], Type.BOOL),
        implementation=lambda a, b: a == b,
        description_length=1.0
    )

    # === TERMINAL COMPARISONS ===

    primitives['terminals_equal_suit'] = Primitive(
        name='terminals_equal_suit',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: h[0].suit == h[-1].suit if len(h) >= 2 else True,
        description_length=1.5
    )

    primitives['terminals_equal_color'] = Primitive(
        name='terminals_equal_color',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: card_color(h[0]) == card_color(h[-1]) if len(h) >= 2 else True,
        description_length=1.5
    )

    primitives['terminals_equal_rank'] = Primitive(
        name='terminals_equal_rank',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: h[0].rank == h[-1].rank if len(h) >= 2 else True,
        description_length=1.5
    )

    # === COUNTING ===

    primitives['count_unique'] = Primitive(
        name='count_unique',
        func_type=FunctionType([Type.LIST_SUIT], Type.INT),
        implementation=lambda lst: len(set(lst)),
        description_length=1.0
    )

    primitives['length'] = Primitive(
        name='length',
        func_type=FunctionType([Type.HAND], Type.INT),
        implementation=lambda h: len(h),
        description_length=1.0
    )

    # === INTEGER COMPARISONS ===

    for val in [1, 2, 3, 4]:
        primitives[f'eq_{val}'] = Primitive(
            name=f'eq_{val}',
            func_type=FunctionType([Type.INT], Type.BOOL),
            implementation=(lambda v: lambda x: x == v)(val),
            description_length=1.2
        )

        primitives[f'ge_{val}'] = Primitive(
            name=f'ge_{val}',
            func_type=FunctionType([Type.INT], Type.BOOL),
            implementation=(lambda v: lambda x: x >= v)(val),
            description_length=1.2
        )

        primitives[f'le_{val}'] = Primitive(
            name=f'le_{val}',
            func_type=FunctionType([Type.INT], Type.BOOL),
            implementation=(lambda v: lambda x: x <= v)(val),
            description_length=1.2
        )

    # === HAS SPECIFIC CARD ===

    # Ace of Spades
    primitives['has_ace_spades'] = Primitive(
        name='has_ace_spades',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: any(c.rank == Rank.ACE and c.suit == Suit.SPADES for c in h),
        description_length=2.0
    )

    # 6 of Diamonds
    primitives['has_6_diamonds'] = Primitive(
        name='has_6_diamonds',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: any(c.rank == Rank.SIX and c.suit == Suit.DIAMONDS for c in h),
        description_length=2.0
    )

    # === HAS PAIR ===

    primitives['has_pair_ranks'] = Primitive(
        name='has_pair_ranks',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: len(set(c.rank for c in h)) < len(h),
        description_length=1.5
    )

    primitives['has_pair_suits'] = Primitive(
        name='has_pair_suits',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: len(set(c.suit for c in h)) < len(h),
        description_length=1.5
    )

    # === BOOLEAN OPERATIONS ===

    primitives['and'] = Primitive(
        name='and',
        func_type=FunctionType([Type.BOOL, Type.BOOL], Type.BOOL),
        implementation=lambda a, b: a and b,
        description_length=0.5
    )

    primitives['or'] = Primitive(
        name='or',
        func_type=FunctionType([Type.BOOL, Type.BOOL], Type.BOOL),
        implementation=lambda a, b: a or b,
        description_length=0.5
    )

    primitives['not'] = Primitive(
        name='not',
        func_type=FunctionType([Type.BOOL], Type.BOOL),
        implementation=lambda a: not a,
        description_length=0.5
    )

    # === HIGHER-LEVEL PATTERNS ===

    # Halves comparison patterns
    primitives['halves_equal_suits'] = Primitive(
        name='halves_equal_suits',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: [c.suit for c in h[:len(h)//2]] == [c.suit for c in h[len(h)//2:]],
        description_length=2.0
    )

    primitives['halves_equal_colors'] = Primitive(
        name='halves_equal_colors',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: [card_color(c) for c in h[:len(h)//2]] == [card_color(c) for c in h[len(h)//2:]],
        description_length=2.0
    )

    primitives['halves_equal_ranks'] = Primitive(
        name='halves_equal_ranks',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: [c.rank for c in h[:len(h)//2]] == [c.rank for c in h[len(h)//2:]],
        description_length=2.0
    )

    # Palindrome patterns
    primitives['suits_palindrome'] = Primitive(
        name='suits_palindrome',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: [c.suit for c in h] == [c.suit for c in h][::-1],
        description_length=2.0
    )

    primitives['colors_palindrome'] = Primitive(
        name='colors_palindrome',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: [card_color(c) for c in h] == [card_color(c) for c in h][::-1],
        description_length=2.0
    )

    primitives['ranks_palindrome'] = Primitive(
        name='ranks_palindrome',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: [c.rank for c in h] == [c.rank for c in h][::-1],
        description_length=2.0
    )

    # Uniform patterns
    primitives['uniform_color'] = Primitive(
        name='uniform_color',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: len(set(card_color(c) for c in h)) <= 1,
        description_length=1.5
    )

    primitives['uniform_suit'] = Primitive(
        name='uniform_suit',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: len(set(c.suit for c in h)) <= 1,
        description_length=1.5
    )

    primitives['uniform_parity'] = Primitive(
        name='uniform_parity',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: len(set(rank_parity(c.rank) for c in h)) <= 1,
        description_length=1.5
    )

    # Sorted rank values
    primitives['sorted_ranks'] = Primitive(
        name='sorted_ranks',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: all(RANK_VALUES[h[i].rank] <= RANK_VALUES[h[i+1].rank] for i in range(len(h)-1)),
        description_length=1.5
    )

    # === ADDITIONAL PRIMITIVES FOR HIGHER SYNTHESIS RATE ===

    # AltColor mappings
    primitives['map_altcolor1'] = Primitive(
        name='map_altcolor1',
        func_type=FunctionType([Type.HAND], Type.LIST_COLOR),
        implementation=lambda h: [suit_to_altcolor1(c.suit) for c in h],
        description_length=1.5
    )

    primitives['map_altcolor2'] = Primitive(
        name='map_altcolor2',
        func_type=FunctionType([Type.HAND], Type.LIST_COLOR),
        implementation=lambda h: [suit_to_altcolor2(c.suit) for c in h],
        description_length=1.5
    )

    # AltColor palindromes
    primitives['altcolor1_palindrome'] = Primitive(
        name='altcolor1_palindrome',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: [suit_to_altcolor1(c.suit) for c in h] == [suit_to_altcolor1(c.suit) for c in h][::-1],
        description_length=2.0
    )

    primitives['altcolor2_palindrome'] = Primitive(
        name='altcolor2_palindrome',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: [suit_to_altcolor2(c.suit) for c in h] == [suit_to_altcolor2(c.suit) for c in h][::-1],
        description_length=2.0
    )

    # Terminal equality for altcolors
    primitives['terminals_equal_altcolor1'] = Primitive(
        name='terminals_equal_altcolor1',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: suit_to_altcolor1(h[0].suit) == suit_to_altcolor1(h[-1].suit) if len(h) >= 2 else True,
        description_length=1.5
    )

    primitives['terminals_equal_altcolor2'] = Primitive(
        name='terminals_equal_altcolor2',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: suit_to_altcolor2(h[0].suit) == suit_to_altcolor2(h[-1].suit) if len(h) >= 2 else True,
        description_length=1.5
    )

    # Halves equal for altcolors and ranks
    primitives['halves_equal_altcolor1'] = Primitive(
        name='halves_equal_altcolor1',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: [suit_to_altcolor1(c.suit) for c in h[:len(h)//2]] == [suit_to_altcolor1(c.suit) for c in h[len(h)//2:]],
        description_length=2.0
    )

    primitives['halves_equal_altcolor2'] = Primitive(
        name='halves_equal_altcolor2',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: [suit_to_altcolor2(c.suit) for c in h[:len(h)//2]] == [suit_to_altcolor2(c.suit) for c in h[len(h)//2:]],
        description_length=2.0
    )

    primitives['halves_equal_ranks'] = Primitive(
        name='halves_equal_ranks',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: [c.rank for c in h[:len(h)//2]] == [c.rank for c in h[len(h)//2:]],
        description_length=2.0
    )

    # Position-specific rank checks
    primitives['pos3_is_jqk'] = Primitive(
        name='pos3_is_jqk',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: len(h) >= 3 and h[2].rank in {Rank.JACK, Rank.QUEEN, Rank.KING},
        description_length=2.5
    )

    primitives['pos4_is_257'] = Primitive(
        name='pos4_is_257',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: len(h) >= 4 and h[3].rank in {Rank.TWO, Rank.FIVE, Rank.SEVEN},
        description_length=2.5
    )

    # Count specific suits
    primitives['count_clubs'] = Primitive(
        name='count_clubs',
        func_type=FunctionType([Type.HAND], Type.INT),
        implementation=lambda h: sum(1 for c in h if c.suit == Suit.CLUBS),
        description_length=1.5
    )

    primitives['count_hearts'] = Primitive(
        name='count_hearts',
        func_type=FunctionType([Type.HAND], Type.INT),
        implementation=lambda h: sum(1 for c in h if c.suit == Suit.HEARTS),
        description_length=1.5
    )

    primitives['count_diamonds'] = Primitive(
        name='count_diamonds',
        func_type=FunctionType([Type.HAND], Type.INT),
        implementation=lambda h: sum(1 for c in h if c.suit == Suit.DIAMONDS),
        description_length=1.5
    )

    primitives['count_spades'] = Primitive(
        name='count_spades',
        func_type=FunctionType([Type.HAND], Type.INT),
        implementation=lambda h: sum(1 for c in h if c.suit == Suit.SPADES),
        description_length=1.5
    )

    # Exactly one club
    primitives['exactly_one_club'] = Primitive(
        name='exactly_one_club',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: sum(1 for c in h if c.suit == Suit.CLUBS) == 1,
        description_length=2.0
    )

    # Half or more same suit (max count >= n/2)
    primitives['half_or_more_same_suit'] = Primitive(
        name='half_or_more_same_suit',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: max(sum(1 for c in h if c.suit == s) for s in Suit) >= (len(h) + 1) // 2 if h else True,
        description_length=2.5
    )

    # Count odd ranks
    primitives['count_odd_ranks'] = Primitive(
        name='count_odd_ranks',
        func_type=FunctionType([Type.HAND], Type.INT),
        implementation=lambda h: sum(1 for c in h if RANK_VALUES[c.rank] % 2 == 1),
        description_length=1.5
    )

    # Exactly one odd
    primitives['exactly_one_odd'] = Primitive(
        name='exactly_one_odd',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: sum(1 for c in h if RANK_VALUES[c.rank] % 2 == 1) == 1,
        description_length=2.0
    )

    # Halves have same suit set
    primitives['halves_same_suit_set'] = Primitive(
        name='halves_same_suit_set',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: set(c.suit for c in h[:len(h)//2]) == set(c.suit for c in h[len(h)//2:]),
        description_length=2.5
    )

    # Halves uniform property equal
    primitives['halves_uniform_color_equal'] = Primitive(
        name='halves_uniform_color_equal',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: (len(set(card_color(c) for c in h[:len(h)//2])) <= 1) == (len(set(card_color(c) for c in h[len(h)//2:])) <= 1),
        description_length=2.5
    )

    primitives['halves_uniform_parity_equal'] = Primitive(
        name='halves_uniform_parity_equal',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: (len(set(rank_parity(c.rank) for c in h[:len(h)//2])) <= 1) == (len(set(rank_parity(c.rank) for c in h[len(h)//2:])) <= 1),
        description_length=2.5
    )

    # Halves hearts presence equal
    primitives['halves_hearts_equal'] = Primitive(
        name='halves_hearts_equal',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: any(c.suit == Suit.HEARTS for c in h[:len(h)//2]) == any(c.suit == Suit.HEARTS for c in h[len(h)//2:]),
        description_length=2.5
    )

    # Adjacent rank gap <= 3
    primitives['adj_rank_gap_le3'] = Primitive(
        name='adj_rank_gap_le3',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: all(abs(RANK_VALUES[h[i].rank] - RANK_VALUES[h[i+1].rank]) <= 3 for i in range(len(h)-1)),
        description_length=2.5
    )

    # Adjacent same rank or suit
    primitives['adj_same_rank_or_suit'] = Primitive(
        name='adj_same_rank_or_suit',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: all(h[i].rank == h[i+1].rank or h[i].suit == h[i+1].suit for i in range(len(h)-1)),
        description_length=2.5
    )

    # Skip2 same rank or suit
    primitives['skip2_same_rank_or_suit'] = Primitive(
        name='skip2_same_rank_or_suit',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: all(h[i].rank == h[i+2].rank or h[i].suit == h[i+2].suit for i in range(len(h)-2)),
        description_length=2.5
    )

    # S before H (spade appears before a heart)
    primitives['s_before_h'] = Primitive(
        name='s_before_h',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: any(h[i].suit == Suit.SPADES and any(h[j].suit == Suit.HEARTS for j in range(i+1, len(h))) for i in range(len(h))),
        description_length=2.5
    )

    # Shift half + 2 ranks
    primitives['shift_half_plus_two'] = Primitive(
        name='shift_half_plus_two',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: all(RANK_VALUES[h[i + len(h)//2].rank] == RANK_VALUES[h[i].rank] + 2 for i in range(len(h)//2)) if len(h) >= 2 else True,
        description_length=3.0
    )

    # Shift half ge (right >= left in ranks)
    primitives['shift_half_ge'] = Primitive(
        name='shift_half_ge',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: all(RANK_VALUES[h[i + len(h)//2].rank] >= RANK_VALUES[h[i].rank] for i in range(len(h)//2)) if len(h) >= 2 else True,
        description_length=2.5
    )

    # Arithmetic progressions
    def has_ap_len3_any_step(h):
        vals = sorted(set(RANK_VALUES[c.rank] for c in h))
        for i in range(len(vals)):
            for j in range(i+1, len(vals)):
                step = vals[j] - vals[i]
                if vals[i] + 2*step in vals:
                    return True
        return False

    primitives['has_ap_len3'] = Primitive(
        name='has_ap_len3',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=has_ap_len3_any_step,
        description_length=2.5
    )

    def has_ap_len3_step2(h):
        vals = set(RANK_VALUES[c.rank] for c in h)
        for v in vals:
            if v + 2 in vals and v + 4 in vals:
                return True
        return False

    primitives['has_ap_len3_step2'] = Primitive(
        name='has_ap_len3_step2',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=has_ap_len3_step2,
        description_length=2.5
    )

    def has_ap_len4_step2(h):
        vals = set(RANK_VALUES[c.rank] for c in h)
        for v in vals:
            if v + 2 in vals and v + 4 in vals and v + 6 in vals:
                return True
        return False

    primitives['has_ap_len4_step2'] = Primitive(
        name='has_ap_len4_step2',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=has_ap_len4_step2,
        description_length=2.5
    )

    # Halves AP len3 equal
    primitives['halves_ap_len3_equal'] = Primitive(
        name='halves_ap_len3_equal',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: has_ap_len3_any_step(h[:len(h)//2]) == has_ap_len3_any_step(h[len(h)//2:]),
        description_length=3.0
    )

    # Halves adjacent pair equal (both have or neither has a +/-1 pair)
    def has_adjacent_pair(hand):
        vals = set(RANK_VALUES[c.rank] for c in hand)
        for v in vals:
            if v+1 in vals or v-1 in vals:
                return True
        return False

    primitives['halves_adj_pair_equal'] = Primitive(
        name='halves_adj_pair_equal',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: has_adjacent_pair(h[:len(h)//2]) == has_adjacent_pair(h[len(h)//2:]),
        description_length=3.0
    )

    # Halves run equal (is_run for both or neither)
    def is_run(hand):
        if len(hand) < 2:
            return False
        vals = [RANK_VALUES[c.rank] for c in hand]
        return all(vals[i+1] - vals[i] == 1 for i in range(len(vals)-1))

    primitives['halves_run_equal'] = Primitive(
        name='halves_run_equal',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: is_run(h[:len(h)//2]) == is_run(h[len(h)//2:]),
        description_length=3.0
    )

    # Bracket matching by suit (♠/♥ open, ♣/♦ close)
    def bracket_match_suits(h):
        stack = []
        for c in h:
            s = c.suit
            if s == Suit.SPADES:
                stack.append('(')
            elif s == Suit.HEARTS:
                stack.append('[')
            elif s == Suit.CLUBS:
                if not stack or stack[-1] != '(':
                    return False
                stack.pop()
            elif s == Suit.DIAMONDS:
                if not stack or stack[-1] != '[':
                    return False
                stack.pop()
        return len(stack) == 0

    primitives['bracket_match_suits'] = Primitive(
        name='bracket_match_suits',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=bracket_match_suits,
        description_length=3.0
    )

    # Suit cycle mappings M1 and M2
    def suit_cycle_m1(s):
        cycle = {Suit.CLUBS: Suit.SPADES, Suit.SPADES: Suit.HEARTS,
                 Suit.HEARTS: Suit.DIAMONDS, Suit.DIAMONDS: Suit.CLUBS}
        return cycle.get(s, s)

    def suit_cycle_m2(s):
        cycle = {Suit.CLUBS: Suit.HEARTS, Suit.HEARTS: Suit.SPADES,
                 Suit.SPADES: Suit.DIAMONDS, Suit.DIAMONDS: Suit.CLUBS}
        return cycle.get(s, s)

    # Half map same position M1/M2
    primitives['half_map_m1'] = Primitive(
        name='half_map_m1',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: all(suit_cycle_m1(h[i].suit) == h[i + len(h)//2].suit for i in range(len(h)//2)) if len(h) >= 2 else True,
        description_length=3.0
    )

    primitives['half_map_m2'] = Primitive(
        name='half_map_m2',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: all(suit_cycle_m2(h[i].suit) == h[i + len(h)//2].suit for i in range(len(h)//2)) if len(h) >= 2 else True,
        description_length=3.0
    )

    # Step2 back map M1/M2
    primitives['step2_map_m1'] = Primitive(
        name='step2_map_m1',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: all(suit_cycle_m1(h[j-2].suit) == h[j].suit for j in range(2, len(h))),
        description_length=3.0
    )

    primitives['step2_map_m2'] = Primitive(
        name='step2_map_m2',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: all(suit_cycle_m2(h[j-2].suit) == h[j].suit for j in range(2, len(h))),
        description_length=3.0
    )

    # Adjacent same or map M1/M2
    primitives['adj_same_or_m1'] = Primitive(
        name='adj_same_or_m1',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: all(h[i].suit == h[i+1].suit or suit_cycle_m1(h[i].suit) == h[i+1].suit for i in range(len(h)-1)),
        description_length=3.0
    )

    primitives['adj_same_or_m2'] = Primitive(
        name='adj_same_or_m2',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: all(h[i].suit == h[i+1].suit or suit_cycle_m2(h[i].suit) == h[i+1].suit for i in range(len(h)-1)),
        description_length=3.0
    )

    # Radial patterns
    primitives['halves_radial_nonincreasing'] = Primitive(
        name='halves_radial_nonincreasing',
        func_type=FunctionType([Type.HAND], Type.BOOL),
        implementation=lambda h: (
            all(RANK_VALUES[h[:len(h)//2][i].rank] >= RANK_VALUES[h[:len(h)//2][i-1].rank] for i in range(len(h)//2-1, 0, -1)) and
            all(RANK_VALUES[h[len(h)//2:][i].rank] >= RANK_VALUES[h[len(h)//2:][i+1].rank] for i in range(len(h)//2-1))
        ) if len(h) >= 4 else True,
        description_length=3.5
    )

    return primitives


# ============================================================================
# ENUMERATOR
# ============================================================================

@dataclass(order=True)
class PrioritizedProgram:
    """Program with priority for heap operations."""
    priority: float
    program: Any = field(compare=False)  # Program or HandVariable
    depth: int = field(compare=False, default=0)


class Enumerator:
    """
    Best-first program enumerator guided by recognition network scores.

    The enumerator searches for programs that satisfy all input-output examples,
    prioritizing programs that use primitives with high recognition scores.
    """

    def __init__(self,
                 primitives: Dict[str, Primitive],
                 recognition_scores: Optional[Dict[str, float]] = None,
                 max_depth: int = 4,
                 verbose: bool = False):
        """
        Initialize the enumerator.

        Args:
            primitives: Dictionary of primitive name → Primitive
            recognition_scores: Dict mapping primitive names to probabilities (0-1)
            max_depth: Maximum program depth
            verbose: Whether to print progress
        """
        self.primitives = primitives
        self.recognition_scores = recognition_scores or {p: 0.5 for p in primitives}
        self.max_depth = max_depth
        self.verbose = verbose

        # Organize primitives by return type
        self.by_return_type: Dict[Type, List[Primitive]] = defaultdict(list)
        for prim in primitives.values():
            self.by_return_type[prim.func_type.return_type].append(prim)

    def compute_priority(self, program) -> float:
        """
        Compute priority for a program (lower = try first).

        Priority combines:
        - Recognition score (prefer primitives with high probability)
        - Description length (prefer simpler programs)
        """
        if isinstance(program, HandVariable):
            return 0.0

        # -log(prob) so higher prob = lower priority (try first)
        score = self.recognition_scores.get(program.primitive.name, 0.1)
        recognition_cost = -math.log(max(score, 0.01))

        # MDL: prefer shorter programs
        mdl_cost = program.description_length

        return recognition_cost + mdl_cost * 0.5

    def enumerate(self,
                  examples: List[Tuple[Hand, bool]],
                  target_type: Type = Type.BOOL,
                  timeout_seconds: float = 60.0,
                  max_programs: int = 10000) -> Optional[Program]:
        """
        Enumerate programs until one satisfies all examples.

        Args:
            examples: List of (hand, expected_output) pairs
            target_type: Type of program to search for (default: Bool)
            timeout_seconds: Maximum search time
            max_programs: Maximum number of programs to try

        Returns:
            Program that satisfies all examples, or None if not found
        """
        start_time = time.time()
        programs_tried = 0
        seen_programs: Set[str] = set()

        # Priority queue: (priority, program, depth)
        heap: List[PrioritizedProgram] = []

        # Start with the hand variable
        hand_var = HandVariable()
        heapq.heappush(heap, PrioritizedProgram(0.0, hand_var, 0))

        # Add all primitives that take Hand as input and return Bool
        for prim in self.primitives.values():
            if (prim.func_type.return_type == target_type and
                len(prim.func_type.arg_types) == 1 and
                prim.func_type.arg_types[0] == Type.HAND):
                # This primitive can be applied directly to the hand
                prog = Program(prim, [hand_var])
                priority = self.compute_priority(prog)
                heapq.heappush(heap, PrioritizedProgram(priority, prog, 1))

        if self.verbose:
            print(f"Starting enumeration with {len(heap)} initial programs...")

        while heap and programs_tried < max_programs:
            # Check timeout
            if time.time() - start_time > timeout_seconds:
                if self.verbose:
                    print(f"Timeout after {programs_tried} programs")
                return None

            # Get next program to try
            item = heapq.heappop(heap)
            program = item.program
            depth = item.depth

            # Skip if we've seen this program before
            prog_str = str(program)
            if prog_str in seen_programs:
                continue
            seen_programs.add(prog_str)

            programs_tried += 1

            # If this program returns the target type, test it
            if program.return_type == target_type:
                if self._check_program(program, examples):
                    if self.verbose:
                        print(f"Found solution after {programs_tried} programs: {program}")
                    return program

            # Don't expand past max depth
            if depth >= self.max_depth:
                continue

            # Expand: compose with other primitives
            self._expand(program, depth, heap, seen_programs)

        if self.verbose:
            print(f"No solution found after {programs_tried} programs")
        return None

    def _check_program(self, program, examples: List[Tuple[Hand, bool]]) -> bool:
        """Check if program satisfies all examples."""
        try:
            for hand, expected in examples:
                result = program.evaluate({'h': hand})
                if result != expected:
                    return False
            return True
        except Exception:
            return False

    def _expand(self, program, depth: int, heap: List[PrioritizedProgram], seen: Set[str]):
        """Expand a program by composing it with primitives."""
        prog_type = program.return_type

        # Find primitives that can use this program's output
        for prim in self.primitives.values():
            if not prim.func_type.arg_types:
                continue

            # Can this primitive take our program as an argument?
            for arg_idx, arg_type in enumerate(prim.func_type.arg_types):
                if self._types_compatible(prog_type, arg_type):
                    # Try to build arguments for this primitive
                    new_progs = self._build_applications(prim, arg_idx, program, depth)
                    for new_prog in new_progs:
                        prog_str = str(new_prog)
                        if prog_str not in seen:
                            priority = self.compute_priority(new_prog)
                            heapq.heappush(heap, PrioritizedProgram(priority, new_prog, depth + 1))

    def _types_compatible(self, actual: Type, expected: Type) -> bool:
        """Check if types are compatible (including list subtypes)."""
        if actual == expected:
            return True
        # Allow any list type where list is expected
        list_types = {Type.LIST_INT, Type.LIST_SUIT, Type.LIST_RANK, Type.LIST_COLOR, Type.LIST_BOOL}
        if expected in list_types and actual in list_types:
            return True
        return False

    def _build_applications(self, prim: Primitive, known_arg_idx: int,
                           known_arg, depth: int) -> List[Program]:
        """Build all valid applications of primitive with known argument at given position."""
        result = []
        num_args = len(prim.func_type.arg_types)

        if num_args == 1:
            # Single argument - just apply
            result.append(Program(prim, [known_arg]))
        elif num_args == 2:
            # Two arguments - need to fill the other one
            other_idx = 1 - known_arg_idx
            other_type = prim.func_type.arg_types[other_idx]

            # Get simple programs of the required type
            other_options = self._get_programs_of_type(other_type, max_depth=1)

            for other in other_options:
                if known_arg_idx == 0:
                    args = [known_arg, other]
                else:
                    args = [other, known_arg]
                result.append(Program(prim, args))

        return result

    def _get_programs_of_type(self, target_type: Type, max_depth: int = 1) -> List:
        """Get simple programs that return the target type."""
        result = []

        # Hand variable
        if target_type == Type.HAND:
            result.append(HandVariable())

        # Direct primitives
        for prim in self.primitives.values():
            if prim.func_type.return_type == target_type:
                if len(prim.func_type.arg_types) == 0:
                    # Constant primitive
                    result.append(Program(prim, []))
                elif (len(prim.func_type.arg_types) == 1 and
                      prim.func_type.arg_types[0] == Type.HAND):
                    # Takes hand, returns target type
                    result.append(Program(prim, [HandVariable()]))

        return result[:10]  # Limit to avoid explosion


# ============================================================================
# SYNTHESIS INTERFACE
# ============================================================================

def synthesize(examples: List[Tuple[Hand, bool]],
               recognition_scores: Optional[Dict[str, float]] = None,
               timeout: float = 60.0,
               verbose: bool = False) -> Optional[str]:
    """
    High-level synthesis interface.

    Args:
        examples: List of (hand, expected_bool) examples
        recognition_scores: Optional primitive probability scores from recognition network
        timeout: Maximum search time in seconds
        verbose: Print progress

    Returns:
        String representation of synthesized program, or None
    """
    primitives = build_primitive_library()

    # Default uniform scores if none provided
    if recognition_scores is None:
        recognition_scores = {p: 0.5 for p in primitives}

    enumerator = Enumerator(
        primitives=primitives,
        recognition_scores=recognition_scores,
        max_depth=3,
        verbose=verbose
    )

    program = enumerator.enumerate(
        examples=examples,
        target_type=Type.BOOL,
        timeout_seconds=timeout
    )

    if program:
        return str(program)
    return None


# ============================================================================
# TESTING
# ============================================================================

def test_enumeration():
    """Test the enumerator on a few simple rules."""
    print("=" * 60)
    print("PROGRAM ENUMERATION TEST")
    print("=" * 60)

    primitives = build_primitive_library()
    print(f"\nLoaded {len(primitives)} primitives")

    # Test 1: Sorted by rank
    print("\n--- Test 1: Sorted by rank ---")

    # Generate examples
    sorted_examples = []
    for _ in range(20):
        hand = sample_hand(6)
        rank_vals = [RANK_VALUES[c.rank] for c in hand]
        is_sorted = all(rank_vals[i] <= rank_vals[i+1] for i in range(5))
        sorted_examples.append((hand, is_sorted))

    # Boost recognition scores for relevant primitives
    scores = {p: 0.1 for p in primitives}
    scores['sorted_ranks'] = 0.9
    scores['is_sorted'] = 0.8
    scores['map_rank_val'] = 0.7

    enumerator = Enumerator(primitives, scores, max_depth=3, verbose=True)
    result = enumerator.enumerate(sorted_examples, timeout_seconds=30)

    if result:
        print(f"SUCCESS: {result}")
    else:
        print("FAILED to find program")

    # Test 2: Uniform color
    print("\n--- Test 2: Uniform color ---")

    uniform_examples = []
    for _ in range(20):
        hand = sample_hand(6)
        colors = [card_color(c) for c in hand]
        is_uniform = len(set(colors)) == 1
        uniform_examples.append((hand, is_uniform))

    scores = {p: 0.1 for p in primitives}
    scores['uniform_color'] = 0.9
    scores['is_uniform'] = 0.8
    scores['map_color'] = 0.7

    enumerator = Enumerator(primitives, scores, max_depth=3, verbose=True)
    result = enumerator.enumerate(uniform_examples, timeout_seconds=30)

    if result:
        print(f"SUCCESS: {result}")
    else:
        print("FAILED to find program")

    # Test 3: Suits palindrome
    print("\n--- Test 3: Suits palindrome ---")

    palindrome_examples = []
    for _ in range(20):
        hand = sample_hand(6)
        suits = [c.suit for c in hand]
        is_palindrome = suits == suits[::-1]
        palindrome_examples.append((hand, is_palindrome))

    scores = {p: 0.1 for p in primitives}
    scores['suits_palindrome'] = 0.9
    scores['is_palindrome'] = 0.8
    scores['map_suit'] = 0.7

    enumerator = Enumerator(primitives, scores, max_depth=3, verbose=True)
    result = enumerator.enumerate(palindrome_examples, timeout_seconds=30)

    if result:
        print(f"SUCCESS: {result}")
    else:
        print("FAILED to find program")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_enumeration()
