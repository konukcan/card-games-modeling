#!/usr/bin/env python3
"""
Rule Dependency Tree Generator - Interactive Tree Visualization

Creates an actual tree graph showing dependencies between:
- Base primitives (leaves at bottom)
- Intermediate abstractions (internal nodes)
- Rules (leaves at top)

Each node is clickable to expand/collapse its children.
Leaf nodes show the lambda expression.
"""

import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.catalogue import ALL_RULES, Rule


# ============================================================================
# NODE DEFINITIONS
# ============================================================================

@dataclass
class TreeNode:
    """A node in the dependency tree."""
    id: str
    name: str
    node_type: str  # 'primitive', 'abstraction', 'rule'
    depth: int  # Construction depth from base primitives
    definition: str = ""  # Lambda expression
    description: str = ""
    children: List[str] = field(default_factory=list)  # IDs of child nodes
    parents: List[str] = field(default_factory=list)  # IDs of parent nodes
    family: str = ""  # For rules only


# ============================================================================
# BASE PRIMITIVES (76 total)
# ============================================================================

BASE_PRIMITIVES = {
    # Constants
    'CLUBS': ('constant', 'Suit.CLUBS'),
    'DIAMONDS': ('constant', 'Suit.DIAMONDS'),
    'HEARTS': ('constant', 'Suit.HEARTS'),
    'SPADES': ('constant', 'Suit.SPADES'),
    'RED': ('constant', 'Color.RED'),
    'BLACK': ('constant', 'Color.BLACK'),
    'true': ('constant', 'True'),
    'false': ('constant', 'False'),
    '0': ('constant', '0'),
    '1': ('constant', '1'),
    '2': ('constant', '2'),
    '3': ('constant', '3'),
    '4': ('constant', '4'),
    '5': ('constant', '5'),
    '10': ('constant', '10'),
    '11': ('constant', '11'),
    '12': ('constant', '12'),
    '13': ('constant', '13'),
    '14': ('constant', '14'),
    '17': ('constant', '17'),
    '21': ('constant', '21'),

    # Card accessors
    'get_suit': ('accessor', 'λc. c.suit'),
    'get_rank': ('accessor', 'λc. c.rank'),
    'rank_val': ('accessor', 'λc. RANK_VALUES[c.rank]'),
    'get_color': ('accessor', 'λc. card_color(c)'),

    # Position ops
    'head': ('position', 'λxs. xs[0]'),
    'last': ('position', 'λxs. xs[-1]'),
    'at': ('position', 'λxs.λi. xs[i]'),
    'length': ('position', 'λxs. len(xs)'),
    'reverse': ('position', 'λxs. reversed(xs)'),

    # List ops
    'take': ('list', 'λn.λxs. xs[:n]'),
    'drop': ('list', 'λn.λxs. xs[n:]'),
    'zip_with': ('list', 'λf.λxs.λys. [f(x)(y) for x,y in zip(xs,ys)]'),
    'adjacent_pairs': ('list', 'λxs. [[xs[i],xs[i+1]] for i in range(len(xs)-1)]'),
    'half_len': ('list', 'λxs. len(xs)//2'),
    'cons': ('list', 'λx.λxs. [x] + xs'),
    'empty': ('list', '[]'),
    'tail': ('list', 'λxs. xs[1:]'),
    'is_empty': ('list', 'λxs. len(xs) == 0'),

    # Direct queries
    'has_suit': ('query', 'λh.λs. any(c.suit == s for c in h)'),
    'has_color': ('query', 'λh.λc. any(card_color(x) == c for x in h)'),
    'count_suit': ('query', 'λh.λs. sum(1 for c in h if c.suit == s)'),
    'count_color': ('query', 'λh.λc. sum(1 for x in h if card_color(x) == c)'),
    'all_same_suit': ('query', 'λh. len(set(c.suit for c in h)) == 1'),
    'all_same_color': ('query', 'λh. len(set(card_color(c) for c in h)) == 1'),
    'n_unique_suits': ('query', 'λh. len(set(c.suit for c in h))'),
    'n_unique_ranks': ('query', 'λh. len(set(c.rank for c in h))'),
    'n_unique_colors': ('query', 'λh. len(set(card_color(c) for c in h))'),

    # Aggregates
    'sum_ranks': ('aggregate', 'λh. sum(RANK_VALUES[c.rank] for c in h)'),
    'max_rank': ('aggregate', 'λh. max(RANK_VALUES[c.rank] for c in h)'),
    'min_rank': ('aggregate', 'λh. min(RANK_VALUES[c.rank] for c in h)'),

    # Comparisons
    'eq': ('comparison', 'λx.λy. x == y'),
    'neq': ('comparison', 'λx.λy. x != y'),
    'lt': ('comparison', 'λx.λy. x < y'),
    'le': ('comparison', 'λx.λy. x <= y'),
    'gt': ('comparison', 'λx.λy. x > y'),
    'ge': ('comparison', 'λx.λy. x >= y'),

    # Boolean
    'and': ('boolean', 'λx.λy. x and y'),
    'or': ('boolean', 'λx.λy. x or y'),
    'not': ('boolean', 'λx. not x'),
    'if': ('boolean', 'λc.λt.λe. t if c else e'),

    # Higher-order
    'map': ('higher', 'λf.λxs. [f(x) for x in xs]'),
    'filter': ('higher', 'λp.λxs. [x for x in xs if p(x)]'),
    'all': ('higher', 'λp.λxs. all(p(x) for x in xs)'),
    'any': ('higher', 'λp.λxs. any(p(x) for x in xs)'),
    'unique': ('higher', 'λxs. list(dict.fromkeys(xs))'),
    'fold': ('higher', 'λf.λz.λxs. reduce(f, xs, z)'),
    'foldr': ('higher', 'λf.λz.λxs. reduce(f, reversed(xs), z)'),

    # State threading (Y&P)
    'pair': ('state', 'λx.λy. [x, y]'),
    'fst': ('state', 'λp. p[0]'),
    'snd': ('state', 'λp. p[1]'),

    # Type bridging
    'all_true': ('bridge', 'λxs. all(xs)'),
    'any_true': ('bridge', 'λxs. any(xs)'),

    # Arithmetic
    '+': ('arithmetic', 'λx.λy. x + y'),
    '-': ('arithmetic', 'λx.λy. x - y'),
    'mod': ('arithmetic', 'λx.λy. x % y'),
}


# ============================================================================
# INTERMEDIATE ABSTRACTIONS
# ============================================================================

ABSTRACTIONS = {
    # Level 1: Simple compositions (depth 2-4)
    'get_parity': {
        'definition': 'λc. mod (rank_val c) 2',
        'depth': 3,
        'children': ['rank_val', 'mod', '2'],
        'description': 'Parity of card rank (0=even, 1=odd)'
    },
    'first_half': {
        'definition': 'λh. take (half_len h) h',
        'depth': 4,
        'children': ['take', 'half_len'],
        'description': 'First half of a hand'
    },
    'second_half': {
        'definition': 'λh. drop (half_len h) h',
        'depth': 4,
        'children': ['drop', 'half_len'],
        'description': 'Second half of a hand'
    },
    'abs': {
        'definition': 'λx. if (lt x 0) (- 0 x) x',
        'depth': 5,
        'children': ['if', 'lt', '-', '0'],
        'description': 'Absolute value'
    },

    # Level 2: Pattern abstractions (depth 5-7)
    'get_altcolor1': {
        'definition': 'λc. if (or (eq (get_suit c) SPADES) (eq (get_suit c) DIAMONDS)) 0 1',
        'depth': 6,
        'children': ['if', 'or', 'eq', 'get_suit', 'SPADES', 'DIAMONDS', '0', '1'],
        'description': 'Pointy (♠♦) vs Round (♥♣)'
    },
    'get_altcolor2': {
        'definition': 'λc. if (or (eq (get_suit c) SPADES) (eq (get_suit c) HEARTS)) 0 1',
        'depth': 6,
        'children': ['if', 'or', 'eq', 'get_suit', 'SPADES', 'HEARTS', '0', '1'],
        'description': 'SH vs DC grouping'
    },
    'terminals_equal_by': {
        'definition': 'λf.λh. eq (f (head h)) (f (last h))',
        'depth': 5,
        'children': ['eq', 'head', 'last'],
        'description': 'First and last share property f'
    },
    'uniform_by': {
        'definition': 'λf.λh. eq 1 (length (unique (map f h)))',
        'depth': 6,
        'children': ['eq', '1', 'length', 'unique', 'map'],
        'description': 'All elements have same value under f'
    },
    'lists_equal': {
        'definition': 'λxs.λys. all_true (zip_with eq xs ys)',
        'depth': 5,
        'children': ['all_true', 'zip_with', 'eq'],
        'description': 'Element-wise list equality'
    },
    'is_palindrome_by': {
        'definition': 'λf.λh. all_true (zip_with eq (map f h) (reverse (map f h)))',
        'depth': 7,
        'children': ['all_true', 'zip_with', 'eq', 'map', 'reverse'],
        'description': 'Sequence of f(card) is palindrome'
    },
    'is_sorted_by': {
        'definition': 'λf.λh. all_true (zip_with le (map f h) (drop 1 (map f h)))',
        'depth': 8,
        'children': ['all_true', 'zip_with', 'le', 'map', 'drop', '1'],
        'description': 'Sequence sorted by property f'
    },
    'shifted_pairs': {
        'definition': 'λk.λh. zip_with pair (take (- (length h) k) h) (drop k h)',
        'depth': 8,
        'children': ['zip_with', 'pair', 'take', 'drop', 'length', '-'],
        'description': 'Pairs of elements offset by k'
    },

    # Level 3: Halves-based (depth 8-10)
    'halves_equal_by': {
        'definition': 'λf.λh. lists_equal (map f (first_half h)) (map f (second_half h))',
        'depth': 9,
        'children': ['lists_equal', 'map', 'first_half', 'second_half'],
        'description': 'Halves have same sequence under f'
    },
    'halves_property_equal': {
        'definition': 'λP.λh. eq (P (first_half h)) (P (second_half h))',
        'depth': 7,
        'children': ['eq', 'first_half', 'second_half'],
        'description': 'Boolean property P same for both halves'
    },
    'halves_set_equal_by': {
        'definition': 'λf.λh. eq (unique (map f (first_half h))) (unique (map f (second_half h)))',
        'depth': 10,
        'children': ['eq', 'unique', 'map', 'first_half', 'second_half'],
        'description': 'Halves have same set under f'
    },
    'suit_cycle_m1': {
        'definition': 'λs. case s: ♣→♠, ♠→♥, ♥→♦, ♦→♣',
        'depth': 8,
        'children': ['if', 'eq', 'CLUBS', 'SPADES', 'HEARTS', 'DIAMONDS'],
        'description': 'Suit cycle M1: ♣→♠→♥→♦→♣'
    },
    'suit_cycle_m2': {
        'definition': 'λs. case s: ♣→♥, ♥→♠, ♠→♦, ♦→♣',
        'depth': 8,
        'children': ['if', 'eq', 'CLUBS', 'SPADES', 'HEARTS', 'DIAMONDS'],
        'description': 'Suit cycle M2: ♣→♥→♠→♦→♣'
    },

    # Level 4: Complex (depth 10+)
    'bracket_match': {
        'definition': 'λopen.λclose.λh. snd (fold (λst.λc. pair (adjust_count st c) (check_valid st c)) (pair 0 true) h)',
        'depth': 12,
        'children': ['fold', 'pair', 'fst', 'snd', 'if', 'and', 'ge', '0', '1', '-', '+'],
        'description': 'Bracket matching via fold+state'
    },
    'has_AP': {
        'definition': 'λlen.λstep.λh. exists subset of ranks forming arithmetic progression',
        'depth': 12,
        'children': ['fold', 'map', 'rank_val', 'unique', 'filter', '-', 'eq'],
        'description': 'Has arithmetic progression of given length'
    },
}


# ============================================================================
# RULE DEFINITIONS WITH DEPENDENCIES
# ============================================================================

RULE_DEPS = {
    # LOCAL family
    'Sorted_by_rank': {
        'family': 'LOCAL',
        'definition': 'λh. is_sorted_by rank_val h',
        'depth': 9,
        'children': ['is_sorted_by', 'rank_val'],
        'description': 'Ranks in non-decreasing order'
    },
    'S_before_H': {
        'family': 'LOCAL',
        'definition': 'λh. fold (track_spade_then_heart) (pair false false) h',
        'depth': 10,
        'children': ['fold', 'pair', 'get_suit', 'eq', 'SPADES', 'HEARTS', 'or', 'and'],
        'description': 'Some ♠ appears before some ♥'
    },
    'Ends_same_suit': {
        'family': 'LOCAL',
        'definition': 'λh. terminals_equal_by get_suit h',
        'depth': 6,
        'children': ['terminals_equal_by', 'get_suit'],
        'description': 'First and last share suit'
    },
    'Ends_same_color': {
        'family': 'LOCAL',
        'definition': 'λh. terminals_equal_by get_color h',
        'depth': 6,
        'children': ['terminals_equal_by', 'get_color'],
        'description': 'First and last share color'
    },

    # COUNT family
    'Has_pair_ranks': {
        'family': 'COUNT',
        'definition': 'λh. lt (length (unique (map get_rank h))) (length h)',
        'depth': 6,
        'children': ['lt', 'length', 'unique', 'map', 'get_rank'],
        'description': 'At least one pair (same rank)'
    },
    'Uniform_color': {
        'family': 'COUNT',
        'definition': 'λh. all_same_color h',
        'depth': 2,
        'children': ['all_same_color'],
        'description': 'All cards same color'
    },
    'Exactly_two_suits': {
        'family': 'COUNT',
        'definition': 'λh. eq 2 (n_unique_suits h)',
        'depth': 3,
        'children': ['eq', '2', 'n_unique_suits'],
        'description': 'Exactly two suits appear'
    },
    'Half_or_more_same_suit': {
        'family': 'COUNT',
        'definition': 'λh. any (λs. ge (count_suit h s) (half_len h)) [♣,♦,♥,♠]',
        'depth': 7,
        'children': ['any', 'ge', 'count_suit', 'half_len', 'CLUBS', 'DIAMONDS', 'HEARTS', 'SPADES'],
        'description': '≥ half cards share a suit'
    },
    'At_most_three_suits': {
        'family': 'COUNT',
        'definition': 'λh. le (n_unique_suits h) 3',
        'depth': 3,
        'children': ['le', 'n_unique_suits', '3'],
        'description': 'At most 3 suits'
    },
    'Exactly_one_club': {
        'family': 'COUNT',
        'definition': 'λh. eq 1 (count_suit h CLUBS)',
        'depth': 4,
        'children': ['eq', '1', 'count_suit', 'CLUBS'],
        'description': 'Exactly one ♣'
    },

    # POSITION family
    'Pos3_is_JQK': {
        'family': 'POSITION',
        'definition': 'λh. or (eq (rank_val (at h 2)) 11) (or (eq ... 12) (eq ... 13))',
        'depth': 7,
        'children': ['or', 'eq', 'rank_val', 'at', '2', '11', '12', '13'],
        'description': 'Card #3 is J/Q/K'
    },
    'Pos4_is_2_5_7': {
        'family': 'POSITION',
        'definition': 'λh. or (eq (rank_val (at h 3)) 2) (or ... 5) (or ... 7))',
        'depth': 7,
        'children': ['or', 'eq', 'rank_val', 'at', '3', '2', '5'],
        'description': 'Card #4 is 2/5/7'
    },

    # TOKEN family
    'Has_Ace_of_Spades': {
        'family': 'TOKEN',
        'definition': 'λh. any (λc. and (eq (get_suit c) SPADES) (eq (rank_val c) 14)) h',
        'depth': 7,
        'children': ['any', 'and', 'eq', 'get_suit', 'rank_val', 'SPADES', '14'],
        'description': 'Contains A♠'
    },
    'Has_6_of_Diamonds': {
        'family': 'TOKEN',
        'definition': 'λh. any (λc. and (eq (get_suit c) DIAMONDS) (eq (rank_val c) 6)) h',
        'depth': 7,
        'children': ['any', 'and', 'eq', 'get_suit', 'rank_val', 'DIAMONDS'],
        'description': 'Contains 6♦'
    },

    # PARITY family
    'Only_one_odd_rank': {
        'family': 'PARITY',
        'definition': 'λh. eq 1 (length (filter (λc. eq 1 (get_parity c)) h))',
        'depth': 7,
        'children': ['eq', '1', 'length', 'filter', 'get_parity'],
        'description': 'Exactly one odd rank'
    },
    'Uniform_rank_parity': {
        'family': 'PARITY',
        'definition': 'λh. uniform_by get_parity h',
        'depth': 8,
        'children': ['uniform_by', 'get_parity'],
        'description': 'All ranks same parity'
    },

    # PAL family
    'Suits_palindrome': {
        'family': 'PAL',
        'definition': 'λh. is_palindrome_by get_suit h',
        'depth': 8,
        'children': ['is_palindrome_by', 'get_suit'],
        'description': 'Suits form palindrome'
    },
    'Colors_palindrome': {
        'family': 'PAL',
        'definition': 'λh. is_palindrome_by get_color h',
        'depth': 8,
        'children': ['is_palindrome_by', 'get_color'],
        'description': 'Colors form palindrome'
    },
    'Ranks_palindrome': {
        'family': 'PAL',
        'definition': 'λh. is_palindrome_by get_rank h',
        'depth': 8,
        'children': ['is_palindrome_by', 'get_rank'],
        'description': 'Ranks form palindrome'
    },

    # ALTCLR family
    'AltColor1_palindrome': {
        'family': 'ALTCLR',
        'definition': 'λh. is_palindrome_by get_altcolor1 h',
        'depth': 12,
        'children': ['is_palindrome_by', 'get_altcolor1'],
        'description': 'Pointy/Round palindrome'
    },
    'AltColor2_palindrome': {
        'family': 'ALTCLR',
        'definition': 'λh. is_palindrome_by get_altcolor2 h',
        'depth': 12,
        'children': ['is_palindrome_by', 'get_altcolor2'],
        'description': 'SH/DC palindrome'
    },
    'Ends_same_altcolor1': {
        'family': 'ALTCLR',
        'definition': 'λh. terminals_equal_by get_altcolor1 h',
        'depth': 10,
        'children': ['terminals_equal_by', 'get_altcolor1'],
        'description': 'First/last same Pointy/Round'
    },

    # HIER family
    'Halves_uniform_color_equal': {
        'family': 'HIER',
        'definition': 'λh. halves_property_equal (uniform_by get_color) h',
        'depth': 12,
        'children': ['halves_property_equal', 'uniform_by', 'get_color'],
        'description': 'Both halves uniform color (or both not)'
    },
    'Halves_uniform_parity_equal': {
        'family': 'HIER',
        'definition': 'λh. halves_property_equal (uniform_by get_parity) h',
        'depth': 13,
        'children': ['halves_property_equal', 'uniform_by', 'get_parity'],
        'description': 'Both halves uniform parity (or both not)'
    },
    'Halves_AP_step1_equal': {
        'family': 'HIER',
        'definition': 'λh. halves_property_equal is_run h',
        'depth': 15,
        'children': ['halves_property_equal', 'is_sorted_by', 'rank_val', '-', '1'],
        'description': 'Both halves are runs (or both not)'
    },
    'Halves_hearts_presence_equal': {
        'family': 'HIER',
        'definition': 'λh. eq (has_suit (first_half h) HEARTS) (has_suit (second_half h) HEARTS)',
        'depth': 10,
        'children': ['eq', 'has_suit', 'first_half', 'second_half', 'HEARTS'],
        'description': 'Both halves have ♥ (or neither)'
    },
    'Halves_AP_len3_any_equal': {
        'family': 'HIER',
        'definition': 'λh. halves_property_equal (has_AP 3 any) h',
        'depth': 16,
        'children': ['halves_property_equal', 'has_AP', 'first_half', 'second_half'],
        'description': 'Both halves have 3-AP (or both not)'
    },
    'Halves_AP_len2_step1_equal': {
        'family': 'HIER',
        'definition': 'λh. halves_property_equal has_adjacent_pair h',
        'depth': 14,
        'children': ['halves_property_equal', 'first_half', 'second_half', 'any', 'abs', '-', '1'],
        'description': 'Both halves have ±1 pair (or both not)'
    },

    # COPY family
    'Halves_copy_suits': {
        'family': 'COPY',
        'definition': 'λh. halves_equal_by get_suit h',
        'depth': 10,
        'children': ['halves_equal_by', 'get_suit'],
        'description': 'Halves have same suit sequence'
    },
    'Halves_copy_colors': {
        'family': 'COPY',
        'definition': 'λh. halves_equal_by get_color h',
        'depth': 10,
        'children': ['halves_equal_by', 'get_color'],
        'description': 'Halves have same color sequence'
    },
    'Halves_copy_ranks': {
        'family': 'COPY',
        'definition': 'λh. halves_equal_by get_rank h',
        'depth': 10,
        'children': ['halves_equal_by', 'get_rank'],
        'description': 'Halves have same rank sequence'
    },
    'Halves_copy_altcolor1': {
        'family': 'COPY',
        'definition': 'λh. halves_equal_by get_altcolor1 h',
        'depth': 14,
        'children': ['halves_equal_by', 'get_altcolor1'],
        'description': 'Halves copy Pointy/Round'
    },
    'Halves_copy_altcolor2': {
        'family': 'COPY',
        'definition': 'λh. halves_equal_by get_altcolor2 h',
        'depth': 14,
        'children': ['halves_equal_by', 'get_altcolor2'],
        'description': 'Halves copy SH/DC'
    },
    'Halves_same_suit_set': {
        'family': 'COPY',
        'definition': 'λh. halves_set_equal_by get_suit h',
        'depth': 11,
        'children': ['halves_set_equal_by', 'get_suit'],
        'description': 'Halves have same suit set'
    },

    # SHIFT family
    'Shift_half_plus_two': {
        'family': 'SHIFT',
        'definition': 'λh. all (λp. eq (- (rank_val (snd p)) (rank_val (fst p))) 2) (shifted_pairs (half_len h) h)',
        'depth': 12,
        'children': ['all', 'shifted_pairs', 'half_len', 'rank_val', '-', 'eq', '2', 'fst', 'snd'],
        'description': 'Half-shift positions +2 rank'
    },
    'Shift2_plus3': {
        'family': 'SHIFT',
        'definition': 'λh. all (λp. eq diff 3) (shifted_pairs 2 h)',
        'depth': 10,
        'children': ['all', 'shifted_pairs', 'rank_val', '-', 'eq', '2', '3'],
        'description': 'Skip-2 positions +3 rank'
    },
    'Shift_half_ge': {
        'family': 'SHIFT',
        'definition': 'λh. all (λp. ge (rank_val (snd p)) (rank_val (fst p))) (shifted_pairs (half_len h) h)',
        'depth': 11,
        'children': ['all', 'shifted_pairs', 'half_len', 'rank_val', 'ge', 'fst', 'snd'],
        'description': 'Right half ≥ left half'
    },

    # MAP family
    'Half_map_samepos_M1': {
        'family': 'MAP',
        'definition': 'λh. all (λp. eq (suit_cycle_m1 (get_suit (fst p))) (get_suit (snd p))) (shifted_pairs k h)',
        'depth': 14,
        'children': ['all', 'shifted_pairs', 'suit_cycle_m1', 'get_suit', 'eq', 'fst', 'snd', 'half_len'],
        'description': 'Right = M1(left) suits'
    },
    'Half_map_samepos_M2': {
        'family': 'MAP',
        'definition': 'λh. all (λp. eq (suit_cycle_m2 (get_suit (fst p))) (get_suit (snd p))) (shifted_pairs k h)',
        'depth': 14,
        'children': ['all', 'shifted_pairs', 'suit_cycle_m2', 'get_suit', 'eq', 'fst', 'snd', 'half_len'],
        'description': 'Right = M2(left) suits'
    },
    'Step2_back_map_M1': {
        'family': 'MAP',
        'definition': 'λh. all (λp. eq (suit_cycle_m1 ...) ...) (shifted_pairs 2 h)',
        'depth': 12,
        'children': ['all', 'shifted_pairs', 'suit_cycle_m1', 'get_suit', 'eq', '2'],
        'description': 'suit[j] = M1(suit[j-2])'
    },
    'Step2_back_map_M2': {
        'family': 'MAP',
        'definition': 'λh. all (λp. eq (suit_cycle_m2 ...) ...) (shifted_pairs 2 h)',
        'depth': 12,
        'children': ['all', 'shifted_pairs', 'suit_cycle_m2', 'get_suit', 'eq', '2'],
        'description': 'suit[j] = M2(suit[j-2])'
    },
    'Adj_same_or_map_M1': {
        'family': 'MAP',
        'definition': 'λh. all (λp. or (eq suits) (eq (suit_cycle_m1 ...) ...)) (adjacent_pairs h)',
        'depth': 12,
        'children': ['all', 'adjacent_pairs', 'suit_cycle_m1', 'get_suit', 'eq', 'or'],
        'description': 'Adjacent: same or M1'
    },
    'Adj_same_or_map_M2': {
        'family': 'MAP',
        'definition': 'λh. all (λp. or (eq suits) (eq (suit_cycle_m2 ...) ...)) (adjacent_pairs h)',
        'depth': 12,
        'children': ['all', 'adjacent_pairs', 'suit_cycle_m2', 'get_suit', 'eq', 'or'],
        'description': 'Adjacent: same or M2'
    },

    # ADJ family
    'Adj_same_rank_or_suit': {
        'family': 'ADJ',
        'definition': 'λh. all (λp. or (eq ranks) (eq suits)) (adjacent_pairs h)',
        'depth': 8,
        'children': ['all', 'adjacent_pairs', 'get_rank', 'get_suit', 'eq', 'or', 'fst', 'snd'],
        'description': 'Neighbors share rank or suit'
    },
    'Skip2_same_rank_or_suit': {
        'family': 'ADJ',
        'definition': 'λh. all (λp. or ...) (shifted_pairs 2 h)',
        'depth': 10,
        'children': ['all', 'shifted_pairs', 'get_rank', 'get_suit', 'eq', 'or', '2'],
        'description': 'i and i+2 share rank or suit'
    },
    'Adj_rank_gap_le3': {
        'family': 'ADJ',
        'definition': 'λh. all (λp. le (abs (- r1 r2)) 3) (adjacent_pairs h)',
        'depth': 10,
        'children': ['all', 'adjacent_pairs', 'abs', '-', 'rank_val', 'le', '3'],
        'description': 'Neighbors differ by ≤3'
    },

    # SCORE family
    'Score_threshold_Rstar': {
        'family': 'SCORE',
        'definition': 'λh. ge (+ (sum_ranks h) (+ sorted_bonus hearts_bonus)) 50',
        'depth': 15,
        'children': ['ge', '+', 'sum_ranks', 'if', 'is_sorted_by', 'count_suit', 'HEARTS', '10', '3'],
        'description': 'Score ≥ 50 (ranks+bonuses)'
    },
    'Half_sum_diff_geN': {
        'family': 'SCORE',
        'definition': 'λh. ge (- (sum_ranks (first_half h)) (sum_ranks (second_half h))) (length h)',
        'depth': 11,
        'children': ['ge', '-', 'sum_ranks', 'first_half', 'second_half', 'length'],
        'description': 'Left - Right ≥ N'
    },
    'Half_sum_one_side_ge_2x_other': {
        'family': 'SCORE',
        'definition': 'λh. or (ge L (* 2 R)) (ge R (* 2 L))',
        'depth': 13,
        'children': ['or', 'ge', '+', 'sum_ranks', 'first_half', 'second_half'],
        'description': 'One half ≥ 2× other'
    },

    # AP family
    'AP_len3_anywhere_anyk': {
        'family': 'AP',
        'definition': 'λh. has_AP 3 any_step h',
        'depth': 13,
        'children': ['has_AP'],
        'description': '3-term AP anywhere'
    },
    'AP_len3_step2_anywhere': {
        'family': 'AP',
        'definition': 'λh. has_AP 3 step_2 h',
        'depth': 13,
        'children': ['has_AP', '2'],
        'description': '3-term AP step 2'
    },
    'AP_len4_step2_anywhere': {
        'family': 'AP',
        'definition': 'λh. has_AP 4 step_2 h',
        'depth': 14,
        'children': ['has_AP', '4', '2'],
        'description': '4-term AP step 2'
    },

    # LANG family
    'Well_formed_brackets_by_suit': {
        'family': 'LANG',
        'definition': 'λh. bracket_match {♠:"(", ♥:"["} {♣:")", ♦:"]"} h',
        'depth': 14,
        'children': ['bracket_match', 'get_suit', 'SPADES', 'HEARTS', 'CLUBS', 'DIAMONDS'],
        'description': 'Suits form matched brackets'
    },
    'Even_opens_next_closes': {
        'family': 'LANG',
        'definition': 'λh. bracket_match_parity even_opens h',
        'depth': 16,
        'children': ['bracket_match', 'get_parity', 'rank_val', 'mod', '2'],
        'description': 'Even opens, next odd closes'
    },
    'Odd_opens_next_closes': {
        'family': 'LANG',
        'definition': 'λh. bracket_match_parity odd_opens h',
        'depth': 16,
        'children': ['bracket_match', 'get_parity', 'rank_val', 'mod', '2'],
        'description': 'Odd opens, next even closes'
    },

    # CENTER family
    'Halves_radial_nonincreasing': {
        'family': 'CENTER',
        'definition': 'λh. and (sorted_desc (reverse (first_half h))) (sorted_desc (second_half h))',
        'depth': 14,
        'children': ['and', 'is_sorted_by', 'reverse', 'first_half', 'second_half', 'rank_val'],
        'description': 'Ranks decrease outward from center'
    },
    'Global_radial_no_dominance': {
        'family': 'CENTER',
        'definition': 'λh. forall i j. dist(j)>dist(i) → rank(j)≤rank(i)',
        'depth': 18,
        'children': ['all', 'fold', 'if', 'and', 'gt', 'le', 'abs', '-', 'rank_val', 'length', '2'],
        'description': 'Farther cards never outrank nearer'
    },
}


# ============================================================================
# BUILD TREE STRUCTURE
# ============================================================================

def build_tree() -> Dict[str, TreeNode]:
    """Build the complete tree of all nodes."""
    nodes = {}

    # Add primitive nodes
    for prim_name, (category, definition) in BASE_PRIMITIVES.items():
        nodes[prim_name] = TreeNode(
            id=prim_name,
            name=prim_name,
            node_type='primitive',
            depth=0,
            definition=definition,
            description=f'{category} primitive',
            children=[],
            parents=[]
        )

    # Add abstraction nodes
    for abs_name, abs_data in ABSTRACTIONS.items():
        nodes[abs_name] = TreeNode(
            id=abs_name,
            name=abs_name,
            node_type='abstraction',
            depth=abs_data['depth'],
            definition=abs_data['definition'],
            description=abs_data['description'],
            children=abs_data['children'],
            parents=[]
        )

    # Add rule nodes
    for rule_name, rule_data in RULE_DEPS.items():
        nodes[rule_name] = TreeNode(
            id=rule_name,
            name=rule_name,
            node_type='rule',
            depth=rule_data['depth'],
            definition=rule_data['definition'],
            description=rule_data['description'],
            family=rule_data['family'],
            children=rule_data['children'],
            parents=[]
        )

    # Build parent links
    for node_id, node in nodes.items():
        for child_id in node.children:
            if child_id in nodes:
                nodes[child_id].parents.append(node_id)

    return nodes


def get_tree_json(nodes: Dict[str, TreeNode]) -> str:
    """Convert tree to JSON for JavaScript."""
    data = {}
    for node_id, node in nodes.items():
        data[node_id] = {
            'id': node.id,
            'name': node.name,
            'type': node.node_type,
            'depth': node.depth,
            'definition': node.definition,
            'description': node.description,
            'family': node.family,
            'children': node.children,
            'parents': node.parents
        }
    return json.dumps(data)


# ============================================================================
# HTML GENERATION
# ============================================================================

def generate_search_tree_data() -> str:
    """
    Generate a COMPLETE search tree showing program synthesis as hole-filling.

    The tree starts with λh. [?:bool] and branches into ALL possible ways to fill
    holes, including intermediate steps and dead-ends, covering all 57 rules.
    """

    # Complete search tree with all rules and intermediate compositions
    search_tree = {
        "id": "root",
        "label": "λh. [?:bool]",
        "type": "hole",
        "description": "Root: A function from Hand to Bool (57 rules to discover)",
        "children": [
            # ================================================================
            # BRANCH 1: Direct hand predicates (depth 2)
            # ================================================================
            {
                "id": "direct_pred",
                "label": "[?:Hand→bool] h",
                "type": "hole",
                "description": "Apply a direct hand predicate",
                "children": [
                    {"id": "r_uniform_color", "label": "all_same_color h", "type": "rule", "rule": "Uniform_color", "depth": 2, "family": "COUNT"},
                    {
                        "id": "dead_all_same_suit",
                        "label": "all_same_suit h",
                        "type": "dead_end",
                        "description": "Not in our rule set (would be depth 2)",
                        "depth": 2
                    }
                ]
            },

            # ================================================================
            # BRANCH 2: Comparisons [?:cmp] [?:int] [?:int]
            # ================================================================
            {
                "id": "comparison",
                "label": "[?:cmp] [?:int] [?:int]",
                "type": "hole",
                "description": "Compare two values",
                "children": [
                    # --- eq comparisons ---
                    {
                        "id": "eq_branch",
                        "label": "eq [?:int] [?:int]",
                        "type": "hole",
                        "description": "Equality comparison",
                        "children": [
                            # eq N (query h)
                            {
                                "id": "eq_const_query",
                                "label": "eq [N] ([?:Hand→int] h)",
                                "type": "hole",
                                "description": "Compare constant to hand query result",
                                "children": [
                                    {"id": "r_exactly_two_suits", "label": "eq 2 (n_unique_suits h)", "type": "rule", "rule": "Exactly_two_suits", "depth": 3, "family": "COUNT"},
                                    {"id": "r_exactly_one_club", "label": "eq 1 (count_suit h ♣)", "type": "rule", "rule": "Exactly_one_club", "depth": 4, "family": "COUNT"},
                                    {
                                        "id": "eq_1_filter_len",
                                        "label": "eq 1 (length (filter [?:pred] h))",
                                        "type": "hole",
                                        "description": "Exactly one card satisfies predicate",
                                        "children": [
                                            {"id": "r_only_one_odd", "label": "eq 1 (length (filter (λc. eq 1 (mod (rank_val c) 2)) h))", "type": "rule", "rule": "Only_one_odd_rank", "depth": 7, "family": "PARITY"}
                                        ]
                                    },
                                    {
                                        "id": "eq_1_unique_len",
                                        "label": "eq 1 (length (unique (map [?:f] h)))",
                                        "type": "hole",
                                        "description": "All cards have same value under f (uniform pattern)",
                                        "children": [
                                            {"id": "r_uniform_parity", "label": "eq 1 (length (unique (map get_parity h)))", "type": "rule", "rule": "Uniform_rank_parity", "depth": 8, "family": "PARITY"}
                                        ]
                                    }
                                ]
                            },
                            # eq (f (head h)) (f (last h)) - terminal comparisons
                            {
                                "id": "eq_terminals",
                                "label": "eq ([?:f] (head h)) ([?:f] (last h))",
                                "type": "hole",
                                "description": "Compare first and last card properties",
                                "children": [
                                    {"id": "r_ends_same_suit", "label": "eq (get_suit (head h)) (get_suit (last h))", "type": "rule", "rule": "Ends_same_suit", "depth": 5, "family": "LOCAL"},
                                    {"id": "r_ends_same_color", "label": "eq (get_color (head h)) (get_color (last h))", "type": "rule", "rule": "Ends_same_color", "depth": 5, "family": "LOCAL"},
                                    {"id": "r_ends_same_altcolor1", "label": "eq (get_altcolor1 (head h)) (get_altcolor1 (last h))", "type": "rule", "rule": "Ends_same_altcolor1", "depth": 10, "family": "ALTCLR", "requires": ["get_altcolor1"]}
                                ]
                            },
                            # eq (P (first_half h)) (P (second_half h)) - halves property comparison
                            {
                                "id": "eq_halves_prop",
                                "label": "eq ([?:P] (first_half h)) ([?:P] (second_half h))",
                                "type": "abstraction",
                                "description": "Compare halves by some property P",
                                "requires": ["first_half", "second_half"],
                                "children": [
                                    {"id": "r_halves_hearts", "label": "eq (has_suit (first_half h) ♥) (has_suit (second_half h) ♥)", "type": "rule", "rule": "Halves_hearts_presence_equal", "depth": 10, "family": "HIER"},
                                    {"id": "r_halves_uniform_color", "label": "eq (uniform get_color (first_half h)) (uniform get_color (second_half h))", "type": "rule", "rule": "Halves_uniform_color_equal", "depth": 12, "family": "HIER", "requires": ["uniform_by"]},
                                    {"id": "r_halves_uniform_parity", "label": "eq (uniform get_parity (first_half h)) (uniform get_parity (second_half h))", "type": "rule", "rule": "Halves_uniform_parity_equal", "depth": 13, "family": "HIER", "requires": ["uniform_by", "get_parity"]},
                                    {"id": "r_halves_ap_step1", "label": "eq (is_run (first_half h)) (is_run (second_half h))", "type": "rule", "rule": "Halves_AP_step1_equal", "depth": 15, "family": "HIER", "requires": ["is_sorted_by"]},
                                    {"id": "r_halves_ap_len3", "label": "eq (has_AP 3 (first_half h)) (has_AP 3 (second_half h))", "type": "rule", "rule": "Halves_AP_len3_any_equal", "depth": 16, "family": "HIER", "requires": ["has_AP"]},
                                    {"id": "r_halves_ap_len2", "label": "eq (has_adj_pair (first_half h)) (has_adj_pair (second_half h))", "type": "rule", "rule": "Halves_AP_len2_step1_equal", "depth": 14, "family": "HIER"}
                                ]
                            },
                            # eq (unique sets of halves)
                            {
                                "id": "eq_halves_set",
                                "label": "eq (unique (map [?:f] (first_half h))) (unique (map [?:f] (second_half h)))",
                                "type": "abstraction",
                                "description": "Compare halves by set of values",
                                "requires": ["first_half", "second_half", "halves_set_equal_by"],
                                "children": [
                                    {"id": "r_halves_same_suit_set", "label": "eq (unique (map get_suit (first_half h))) (unique (map get_suit (second_half h)))", "type": "rule", "rule": "Halves_same_suit_set", "depth": 11, "family": "COPY"}
                                ]
                            }
                        ]
                    },
                    # --- le/lt/ge/gt comparisons ---
                    {
                        "id": "ordering_branch",
                        "label": "[le|lt|ge|gt] [?:int] [?:int]",
                        "type": "hole",
                        "description": "Ordering comparisons",
                        "children": [
                            {"id": "r_at_most_three", "label": "le (n_unique_suits h) 3", "type": "rule", "rule": "At_most_three_suits", "depth": 3, "family": "COUNT"},
                            {"id": "r_has_pair", "label": "lt (length (unique (map get_rank h))) (length h)", "type": "rule", "rule": "Has_pair_ranks", "depth": 6, "family": "COUNT"},
                            {
                                "id": "ge_count_half",
                                "label": "ge (count_suit h [?:s]) (half_len h)",
                                "type": "hole",
                                "description": "At least half the cards have suit s",
                                "children": [
                                    {"id": "r_half_same_suit", "label": "any (λs. ge (count_suit h s) (half_len h)) [♣,♦,♥,♠]", "type": "rule", "rule": "Half_or_more_same_suit", "depth": 7, "family": "COUNT"}
                                ]
                            },
                            # Score threshold comparisons
                            {
                                "id": "ge_score",
                                "label": "ge [?:score_expr] [?:threshold]",
                                "type": "hole",
                                "description": "Score threshold comparisons",
                                "children": [
                                    {"id": "r_score_rstar", "label": "ge (sum_ranks h + bonuses) 50", "type": "rule", "rule": "Score_threshold_Rstar", "depth": 15, "family": "SCORE"},
                                    {"id": "r_half_sum_diff", "label": "ge (sum_ranks (first_half h) - sum_ranks (second_half h)) (length h)", "type": "rule", "rule": "Half_sum_diff_geN", "depth": 11, "family": "SCORE", "requires": ["first_half", "second_half"]},
                                    {"id": "r_half_2x", "label": "or (ge L (* 2 R)) (ge R (* 2 L))", "type": "rule", "rule": "Half_sum_one_side_ge_2x_other", "depth": 13, "family": "SCORE", "requires": ["first_half", "second_half"]}
                                ]
                            }
                        ]
                    }
                ]
            },

            # ================================================================
            # BRANCH 3: all_true [?:list(bool)] - Type bridge pattern
            # ================================================================
            {
                "id": "all_true_branch",
                "label": "all_true [?:list(bool)]",
                "type": "hole",
                "description": "All elements of boolean list are true (type bridge)",
                "children": [
                    {
                        "id": "all_true_zipwith",
                        "label": "all_true (zip_with [?:cmp] [?:list] [?:list])",
                        "type": "hole",
                        "description": "Element-wise list comparison",
                        "children": [
                            # Palindrome pattern
                            {
                                "id": "palindrome_pattern",
                                "label": "all_true (zip_with eq (map [?:f] h) (reverse (map [?:f] h)))",
                                "type": "abstraction",
                                "description": "Palindrome check: sequence equals its reverse",
                                "requires": ["is_palindrome_by"],
                                "children": [
                                    {"id": "r_suits_pal", "label": "... eq (map get_suit h) (reverse (map get_suit h))", "type": "rule", "rule": "Suits_palindrome", "depth": 8, "family": "PAL"},
                                    {"id": "r_colors_pal", "label": "... eq (map get_color h) (reverse (map get_color h))", "type": "rule", "rule": "Colors_palindrome", "depth": 8, "family": "PAL"},
                                    {"id": "r_ranks_pal", "label": "... eq (map get_rank h) (reverse (map get_rank h))", "type": "rule", "rule": "Ranks_palindrome", "depth": 8, "family": "PAL"},
                                    {"id": "r_altcolor1_pal", "label": "... eq (map get_altcolor1 h) (reverse ...)", "type": "rule", "rule": "AltColor1_palindrome", "depth": 12, "family": "ALTCLR", "requires": ["get_altcolor1"]},
                                    {"id": "r_altcolor2_pal", "label": "... eq (map get_altcolor2 h) (reverse ...)", "type": "rule", "rule": "AltColor2_palindrome", "depth": 12, "family": "ALTCLR", "requires": ["get_altcolor2"]}
                                ]
                            },
                            # Sorted pattern
                            {
                                "id": "sorted_pattern",
                                "label": "all_true (zip_with le (map [?:f] h) (drop 1 (map [?:f] h)))",
                                "type": "abstraction",
                                "description": "Sorted check: each element <= next",
                                "requires": ["is_sorted_by"],
                                "children": [
                                    {"id": "r_sorted_rank", "label": "... le (map rank_val h) (drop 1 (map rank_val h))", "type": "rule", "rule": "Sorted_by_rank", "depth": 9, "family": "LOCAL"}
                                ]
                            },
                            # Halves copy pattern
                            {
                                "id": "halves_copy_pattern",
                                "label": "all_true (zip_with eq (map [?:f] (first_half h)) (map [?:f] (second_half h)))",
                                "type": "abstraction",
                                "description": "Halves have identical sequence under f",
                                "requires": ["first_half", "second_half", "lists_equal", "halves_equal_by"],
                                "children": [
                                    {"id": "r_halves_copy_suits", "label": "... eq (map get_suit (first_half h)) (map get_suit (second_half h))", "type": "rule", "rule": "Halves_copy_suits", "depth": 10, "family": "COPY"},
                                    {"id": "r_halves_copy_colors", "label": "... eq (map get_color (first_half h)) (map get_color (second_half h))", "type": "rule", "rule": "Halves_copy_colors", "depth": 10, "family": "COPY"},
                                    {"id": "r_halves_copy_ranks", "label": "... eq (map get_rank (first_half h)) (map get_rank (second_half h))", "type": "rule", "rule": "Halves_copy_ranks", "depth": 10, "family": "COPY"},
                                    {"id": "r_halves_copy_altcolor1", "label": "... eq (map get_altcolor1 (first_half h)) ...", "type": "rule", "rule": "Halves_copy_altcolor1", "depth": 14, "family": "COPY", "requires": ["get_altcolor1"]},
                                    {"id": "r_halves_copy_altcolor2", "label": "... eq (map get_altcolor2 (first_half h)) ...", "type": "rule", "rule": "Halves_copy_altcolor2", "depth": 14, "family": "COPY", "requires": ["get_altcolor2"]}
                                ]
                            }
                        ]
                    }
                ]
            },

            # ================================================================
            # BRANCH 4: all [?:pred] [?:list] - Universal quantification
            # ================================================================
            {
                "id": "all_branch",
                "label": "all [?:a→bool] [?:list(a)]",
                "type": "hole",
                "description": "All elements satisfy predicate",
                "children": [
                    # all pred (adjacent_pairs h)
                    {
                        "id": "all_adjacent",
                        "label": "all [?:pair→bool] (adjacent_pairs h)",
                        "type": "hole",
                        "description": "All adjacent pairs satisfy predicate",
                        "children": [
                            {"id": "r_adj_rank_suit", "label": "all (λp. or (eq ranks) (eq suits)) (adjacent_pairs h)", "type": "rule", "rule": "Adj_same_rank_or_suit", "depth": 8, "family": "ADJ"},
                            {"id": "r_adj_gap_le3", "label": "all (λp. le (abs (- r1 r2)) 3) (adjacent_pairs h)", "type": "rule", "rule": "Adj_rank_gap_le3", "depth": 10, "family": "ADJ"},
                            {
                                "id": "adj_map_pattern",
                                "label": "all (λp. or (eq suits) (eq (cycle (suit left)) (suit right))) (adjacent_pairs h)",
                                "type": "abstraction",
                                "description": "Adjacent cards: same suit or mapped by cycle",
                                "requires": ["suit_cycle_m1", "suit_cycle_m2"],
                                "children": [
                                    {"id": "r_adj_same_or_m1", "label": "... or eq (suit_cycle_m1 ...) ...", "type": "rule", "rule": "Adj_same_or_map_M1", "depth": 12, "family": "MAP"},
                                    {"id": "r_adj_same_or_m2", "label": "... or eq (suit_cycle_m2 ...) ...", "type": "rule", "rule": "Adj_same_or_map_M2", "depth": 12, "family": "MAP"}
                                ]
                            }
                        ]
                    },
                    # all pred (shifted_pairs k h)
                    {
                        "id": "all_shifted",
                        "label": "all [?:pair→bool] (shifted_pairs [?:k] h)",
                        "type": "abstraction",
                        "description": "All k-offset pairs satisfy predicate",
                        "requires": ["shifted_pairs"],
                        "children": [
                            # Shift with k = 2
                            {
                                "id": "shift_k2",
                                "label": "all [?:pred] (shifted_pairs 2 h)",
                                "type": "hole",
                                "description": "Skip-2 pairs",
                                "children": [
                                    {"id": "r_skip2_rank_suit", "label": "all (λp. or (eq ranks) (eq suits)) (shifted_pairs 2 h)", "type": "rule", "rule": "Skip2_same_rank_or_suit", "depth": 10, "family": "ADJ"},
                                    {"id": "r_shift2_plus3", "label": "all (λp. eq (diff ranks) 3) (shifted_pairs 2 h)", "type": "rule", "rule": "Shift2_plus3", "depth": 10, "family": "SHIFT"},
                                    {"id": "r_step2_map_m1", "label": "all (λp. eq (suit_cycle_m1 ...) ...) (shifted_pairs 2 h)", "type": "rule", "rule": "Step2_back_map_M1", "depth": 12, "family": "MAP", "requires": ["suit_cycle_m1"]},
                                    {"id": "r_step2_map_m2", "label": "all (λp. eq (suit_cycle_m2 ...) ...) (shifted_pairs 2 h)", "type": "rule", "rule": "Step2_back_map_M2", "depth": 12, "family": "MAP", "requires": ["suit_cycle_m2"]}
                                ]
                            },
                            # Shift with k = half_len h
                            {
                                "id": "shift_half",
                                "label": "all [?:pred] (shifted_pairs (half_len h) h)",
                                "type": "abstraction",
                                "description": "Half-offset pairs (position i vs i+n/2)",
                                "requires": ["half_len"],
                                "children": [
                                    {"id": "r_shift_half_ge", "label": "all (λp. ge (rank right) (rank left)) (shifted_pairs (half_len h) h)", "type": "rule", "rule": "Shift_half_ge", "depth": 11, "family": "SHIFT"},
                                    {"id": "r_shift_half_plus2", "label": "all (λp. eq (diff ranks) 2) (shifted_pairs (half_len h) h)", "type": "rule", "rule": "Shift_half_plus_two", "depth": 12, "family": "SHIFT"},
                                    {"id": "r_half_map_m1", "label": "all (λp. eq (suit_cycle_m1 (suit left)) (suit right)) ...", "type": "rule", "rule": "Half_map_samepos_M1", "depth": 14, "family": "MAP", "requires": ["suit_cycle_m1"]},
                                    {"id": "r_half_map_m2", "label": "all (λp. eq (suit_cycle_m2 (suit left)) (suit right)) ...", "type": "rule", "rule": "Half_map_samepos_M2", "depth": 14, "family": "MAP", "requires": ["suit_cycle_m2"]}
                                ]
                            }
                        ]
                    }
                ]
            },

            # ================================================================
            # BRANCH 5: any [?:pred] h - Existential quantification
            # ================================================================
            {
                "id": "any_branch",
                "label": "any [?:Card→bool] h",
                "type": "hole",
                "description": "Some card satisfies predicate",
                "children": [
                    # Specific card check
                    {
                        "id": "any_specific",
                        "label": "any (λc. and (eq (get_suit c) [?:s]) (eq (rank_val c) [?:r])) h",
                        "type": "hole",
                        "description": "Contains specific card (suit + rank)",
                        "children": [
                            {"id": "r_ace_spades", "label": "any (λc. and (eq (get_suit c) ♠) (eq (rank_val c) 14)) h", "type": "rule", "rule": "Has_Ace_of_Spades", "depth": 7, "family": "TOKEN"},
                            {"id": "r_6_diamonds", "label": "any (λc. and (eq (get_suit c) ♦) (eq (rank_val c) 6)) h", "type": "rule", "rule": "Has_6_of_Diamonds", "depth": 7, "family": "TOKEN"}
                        ]
                    },
                    # Suit presence check with threshold
                    {
                        "id": "any_suit_count",
                        "label": "any (λs. ge (count_suit h s) [?:n]) [suits]",
                        "type": "hole",
                        "description": "Some suit has at least n cards",
                        "children": [
                            {"id": "r_half_same_suit_alt", "label": "any (λs. ge (count_suit h s) (half_len h)) [♣,♦,♥,♠]", "type": "rule", "rule": "Half_or_more_same_suit", "depth": 7, "family": "COUNT", "note": "Alternative path to same rule"}
                        ]
                    }
                ]
            },

            # ================================================================
            # BRANCH 6: snd (fold ...) - Stateful iteration
            # ================================================================
            {
                "id": "fold_branch",
                "label": "snd (fold [?:state→Card→state] (pair [?:init] [?:init]) h)",
                "type": "abstraction",
                "description": "Stateful fold with pair state",
                "requires": ["fold", "pair", "fst", "snd"],
                "children": [
                    # Bracket matching
                    {
                        "id": "bracket_pattern",
                        "label": "snd (fold (λst.λc. pair (adjust) (check)) (pair 0 true) h)",
                        "type": "abstraction",
                        "description": "Bracket matching via counter + validity flag",
                        "requires": ["bracket_match"],
                        "children": [
                            {"id": "r_brackets_suit", "label": "bracket_match {♠,♥:open} {♣,♦:close} h", "type": "rule", "rule": "Well_formed_brackets_by_suit", "depth": 14, "family": "LANG"},
                            {"id": "r_even_opens", "label": "bracket_match even_rank_opens h", "type": "rule", "rule": "Even_opens_next_closes", "depth": 16, "family": "LANG"},
                            {"id": "r_odd_opens", "label": "bracket_match odd_rank_opens h", "type": "rule", "rule": "Odd_opens_next_closes", "depth": 16, "family": "LANG"}
                        ]
                    },
                    # Ordering tracking
                    {
                        "id": "order_track",
                        "label": "snd (fold (λst.λc. track_order) (pair false false) h)",
                        "type": "hole",
                        "description": "Track ordering between card types",
                        "children": [
                            {"id": "r_s_before_h", "label": "snd (fold track_spade_heart (pair false false) h)", "type": "rule", "rule": "S_before_H", "depth": 10, "family": "LOCAL"}
                        ]
                    }
                ]
            },

            # ================================================================
            # BRANCH 7: or [?:bool] [?:bool] - Disjunction
            # ================================================================
            {
                "id": "or_branch",
                "label": "or [?:bool] [?:bool]",
                "type": "hole",
                "description": "Disjunction of conditions",
                "children": [
                    # Position-based checks
                    {
                        "id": "pos_check_or",
                        "label": "or (eq (rank_val (at h [?:i])) [?:r]) (or ...)",
                        "type": "hole",
                        "description": "Card at position has one of several ranks",
                        "children": [
                            {"id": "r_pos3_jqk", "label": "or (eq (rank (at h 2)) 11) (or (eq ... 12) (eq ... 13))", "type": "rule", "rule": "Pos3_is_JQK", "depth": 7, "family": "POSITION"},
                            {"id": "r_pos4_257", "label": "or (eq (rank (at h 3)) 2) (or (eq ... 5) (eq ... 7))", "type": "rule", "rule": "Pos4_is_2_5_7", "depth": 7, "family": "POSITION"}
                        ]
                    },
                    # Half-based disjunction
                    {
                        "id": "or_halves",
                        "label": "or (ge L (* 2 R)) (ge R (* 2 L))",
                        "type": "abstraction",
                        "description": "One half dominates the other",
                        "requires": ["first_half", "second_half"],
                        "children": [
                            {"id": "r_half_2x_alt", "label": "or (ge (sum left) (* 2 (sum right))) ...", "type": "rule", "rule": "Half_sum_one_side_ge_2x_other", "depth": 13, "family": "SCORE", "note": "Alternative path"}
                        ]
                    }
                ]
            },

            # ================================================================
            # BRANCH 8: and [?:bool] [?:bool] - Conjunction
            # ================================================================
            {
                "id": "and_branch",
                "label": "and [?:bool] [?:bool]",
                "type": "hole",
                "description": "Conjunction of conditions",
                "children": [
                    # Radial patterns
                    {
                        "id": "radial_pattern",
                        "label": "and (sorted_desc (reverse (first_half h))) (sorted_desc (second_half h))",
                        "type": "abstraction",
                        "description": "Radial pattern: center is highest",
                        "requires": ["first_half", "second_half", "is_sorted_by"],
                        "children": [
                            {"id": "r_radial_nonincr", "label": "and (sorted_desc (reverse (first_half h))) (sorted_desc (second_half h))", "type": "rule", "rule": "Halves_radial_nonincreasing", "depth": 14, "family": "CENTER"},
                            {"id": "r_radial_no_dom", "label": "forall i j. dist(j)>dist(i) → rank(j)≤rank(i)", "type": "rule", "rule": "Global_radial_no_dominance", "depth": 18, "family": "CENTER"}
                        ]
                    }
                ]
            },

            # ================================================================
            # BRANCH 9: Arithmetic progression detection
            # ================================================================
            {
                "id": "ap_branch",
                "label": "has_AP [?:len] [?:step] h",
                "type": "abstraction",
                "description": "Contains arithmetic progression in ranks",
                "requires": ["has_AP"],
                "children": [
                    {"id": "r_ap_len3_any", "label": "has_AP 3 any h", "type": "rule", "rule": "AP_len3_anywhere_anyk", "depth": 13, "family": "AP"},
                    {"id": "r_ap_len3_step2", "label": "has_AP 3 2 h", "type": "rule", "rule": "AP_len3_step2_anywhere", "depth": 13, "family": "AP"},
                    {"id": "r_ap_len4_step2", "label": "has_AP 4 2 h", "type": "rule", "rule": "AP_len4_step2_anywhere", "depth": 14, "family": "AP"}
                ]
            },

            # ================================================================
            # DEAD-END BRANCHES (intermediate compositions that don't lead to rules)
            # ================================================================
            {
                "id": "dead_ends",
                "label": "Other intermediate compositions (no rules)",
                "type": "dead_end",
                "description": "Compositions explored during search that don't yield rules",
                "children": [
                    {
                        "id": "dead_not",
                        "label": "not [?:bool]",
                        "type": "dead_end",
                        "description": "Negation - no rules use top-level negation",
                        "depth": "varies"
                    },
                    {
                        "id": "dead_if",
                        "label": "if [?:bool] [?:bool] [?:bool]",
                        "type": "dead_end",
                        "description": "Conditional - used internally but not at top level",
                        "depth": "varies"
                    },
                    {
                        "id": "dead_length_direct",
                        "label": "eq (length h) [?:n]",
                        "type": "dead_end",
                        "description": "Direct length check - not a rule (hand size is fixed)",
                        "depth": 2
                    },
                    {
                        "id": "dead_head_rank",
                        "label": "eq (rank_val (head h)) [?:n]",
                        "type": "dead_end",
                        "description": "First card rank equals n - not in rule set",
                        "depth": 4
                    },
                    {
                        "id": "dead_sum_eq",
                        "label": "eq (sum_ranks h) [?:n]",
                        "type": "dead_end",
                        "description": "Exact sum - only threshold comparisons are rules",
                        "depth": 3
                    },
                    {
                        "id": "dead_mod_0",
                        "label": "eq 0 (mod (sum_ranks h) [?:n])",
                        "type": "dead_end",
                        "description": "Sum divisible by n - not in current rule set",
                        "depth": 5
                    },
                    {
                        "id": "dead_filter_len_gt",
                        "label": "gt (length (filter [?:pred] h)) [?:n]",
                        "type": "dead_end",
                        "description": "More than n cards satisfy pred",
                        "depth": "varies"
                    },
                    {
                        "id": "dead_reverse_eq",
                        "label": "eq h (reverse h)",
                        "type": "dead_end",
                        "description": "Hand equals its reverse - too strong (palindrome by all properties)",
                        "depth": 3
                    }
                ]
            }
        ]
    }

    return json.dumps(search_tree)


def generate_html_report(nodes: Dict[str, TreeNode], output_path: str):
    """Generate an interactive tree visualization with two tabs."""

    tree_json = get_tree_json(nodes)
    search_tree_json = generate_search_tree_data()

    # Count statistics
    n_primitives = sum(1 for n in nodes.values() if n.node_type == 'primitive')
    n_abstractions = sum(1 for n in nodes.values() if n.node_type == 'abstraction')
    n_rules = sum(1 for n in nodes.values() if n.node_type == 'rule')

    html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Rule Dependency & Search Tree - Interactive</title>
    <style>
        :root {
            --bg-dark: #1a1a2e;
            --bg-card: #252540;
            --bg-hover: #2d2d50;
            --text-primary: #e0e0e0;
            --text-secondary: #a0a0a0;
            --accent-blue: #7aa2f7;
            --accent-green: #9ece6a;
            --accent-yellow: #e0af68;
            --accent-red: #f7768e;
            --accent-purple: #bb9af7;
            --accent-cyan: #7dcfff;
            --accent-orange: #ff9e64;
        }

        * {
            box-sizing: border-box;
        }

        body {
            font-family: 'SF Mono', 'Consolas', 'Monaco', monospace;
            background: var(--bg-dark);
            color: var(--text-primary);
            margin: 0;
            padding: 20px;
            line-height: 1.5;
        }

        h1 {
            color: var(--accent-purple);
            text-align: center;
            margin-bottom: 10px;
        }

        .subtitle {
            text-align: center;
            color: var(--text-secondary);
            margin-bottom: 20px;
        }

        .stats {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-bottom: 20px;
        }

        .stat {
            background: var(--bg-card);
            padding: 10px 20px;
            border-radius: 8px;
            text-align: center;
        }

        .stat .value {
            font-size: 1.5em;
            font-weight: bold;
            color: var(--accent-green);
        }

        .stat .label {
            color: var(--text-secondary);
            font-size: 0.85em;
        }

        .controls {
            display: flex;
            justify-content: center;
            gap: 15px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }

        .controls button {
            background: var(--bg-card);
            color: var(--text-primary);
            border: 1px solid var(--accent-blue);
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-family: inherit;
            font-size: 0.9em;
        }

        .controls button:hover {
            background: var(--accent-blue);
            color: var(--bg-dark);
        }

        .controls input {
            background: var(--bg-card);
            color: var(--text-primary);
            border: 1px solid var(--accent-purple);
            padding: 8px 12px;
            border-radius: 6px;
            width: 250px;
            font-family: inherit;
        }

        .tree-container {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 20px;
            overflow: auto;
            max-height: 70vh;
        }

        .tree-view {
            padding-left: 0;
        }

        .tree-node {
            list-style: none;
            margin: 2px 0;
        }

        .node-content {
            display: flex;
            align-items: center;
            padding: 6px 10px;
            border-radius: 6px;
            cursor: pointer;
            transition: background 0.15s;
        }

        .node-content:hover {
            background: var(--bg-hover);
        }

        .node-content.selected {
            background: var(--bg-hover);
            border-left: 3px solid var(--accent-cyan);
        }

        .toggle-icon {
            width: 20px;
            color: var(--text-secondary);
            font-size: 0.8em;
            flex-shrink: 0;
        }

        .node-icon {
            width: 24px;
            text-align: center;
            margin-right: 8px;
            flex-shrink: 0;
        }

        .node-name {
            flex-grow: 1;
            font-weight: 500;
        }

        .node-badge {
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 0.75em;
            margin-left: 8px;
        }

        .node-depth {
            color: var(--accent-yellow);
            font-size: 0.8em;
            margin-left: 10px;
        }

        /* Node type colors */
        .node-primitive .node-name { color: var(--accent-green); }
        .node-abstraction .node-name { color: var(--accent-cyan); }
        .node-rule .node-name { color: var(--accent-purple); }

        .node-primitive .node-icon { color: var(--accent-green); }
        .node-abstraction .node-icon { color: var(--accent-cyan); }
        .node-rule .node-icon { color: var(--accent-purple); }

        /* Family badges */
        .badge-LOCAL { background: #3d5a80; }
        .badge-COUNT { background: #2a9d8f; }
        .badge-POSITION { background: #e76f51; }
        .badge-TOKEN { background: #f4a261; }
        .badge-PARITY { background: #264653; }
        .badge-PAL { background: #e9c46a; color: #1a1a2e; }
        .badge-ALTCLR { background: #2d6a4f; }
        .badge-HIER { background: #9b2226; }
        .badge-COPY { background: #005f73; }
        .badge-SHIFT { background: #ae2012; }
        .badge-MAP { background: #0a9396; }
        .badge-ADJ { background: #94d2bd; color: #1a1a2e; }
        .badge-SCORE { background: #ee9b00; color: #1a1a2e; }
        .badge-AP { background: #bb3e03; }
        .badge-LANG { background: #ca6702; }
        .badge-CENTER { background: #001219; }

        .children {
            padding-left: 24px;
            border-left: 1px dashed var(--text-secondary);
            margin-left: 10px;
            display: none;
        }

        .children.expanded {
            display: block;
        }

        /* Detail panel */
        .detail-panel {
            position: fixed;
            right: 20px;
            top: 20px;
            width: 400px;
            background: var(--bg-card);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.4);
            display: none;
            max-height: 80vh;
            overflow-y: auto;
        }

        .detail-panel.visible {
            display: block;
        }

        .detail-panel h3 {
            color: var(--accent-cyan);
            margin-top: 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .detail-panel .close-btn {
            position: absolute;
            right: 15px;
            top: 15px;
            background: none;
            border: none;
            color: var(--text-secondary);
            font-size: 1.5em;
            cursor: pointer;
        }

        .detail-panel .close-btn:hover {
            color: var(--accent-red);
        }

        .detail-section {
            margin: 15px 0;
        }

        .detail-section .label {
            color: var(--text-secondary);
            font-size: 0.85em;
            margin-bottom: 5px;
        }

        .detail-section .value {
            background: #1a1a2e;
            padding: 10px;
            border-radius: 6px;
            font-size: 0.9em;
            overflow-x: auto;
        }

        .detail-section .value.lambda {
            color: var(--accent-yellow);
            font-family: 'SF Mono', monospace;
        }

        .dep-list {
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
        }

        .dep-tag {
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            cursor: pointer;
        }

        .dep-tag.primitive {
            background: var(--accent-green);
            color: var(--bg-dark);
        }

        .dep-tag.abstraction {
            background: var(--accent-cyan);
            color: var(--bg-dark);
        }

        .dep-tag.rule {
            background: var(--accent-purple);
            color: var(--bg-dark);
        }

        .dep-tag:hover {
            opacity: 0.8;
        }

        /* Legend */
        .legend {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }

        .legend-item {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 0.85em;
        }

        .legend-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
        }

        .legend-dot.primitive { background: var(--accent-green); }
        .legend-dot.abstraction { background: var(--accent-cyan); }
        .legend-dot.rule { background: var(--accent-purple); }

        /* Root selector */
        .root-selector {
            display: flex;
            justify-content: center;
            gap: 10px;
            margin-bottom: 15px;
            flex-wrap: wrap;
        }

        .root-btn {
            background: var(--bg-card);
            color: var(--text-primary);
            border: 1px solid transparent;
            padding: 6px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85em;
        }

        .root-btn:hover {
            border-color: var(--accent-purple);
        }

        .root-btn.active {
            background: var(--accent-purple);
            color: var(--bg-dark);
        }

        /* Tab Navigation */
        .tab-nav {
            display: flex;
            justify-content: center;
            gap: 0;
            margin-bottom: 20px;
            border-bottom: 2px solid var(--bg-card);
        }

        .tab-btn {
            background: transparent;
            color: var(--text-secondary);
            border: none;
            padding: 12px 30px;
            cursor: pointer;
            font-family: inherit;
            font-size: 1em;
            font-weight: 500;
            border-bottom: 3px solid transparent;
            transition: all 0.2s;
        }

        .tab-btn:hover {
            color: var(--text-primary);
            background: var(--bg-card);
        }

        .tab-btn.active {
            color: var(--accent-cyan);
            border-bottom-color: var(--accent-cyan);
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
        }

        /* Search Tree Specific Styles */
        .search-tree-container {
            background: var(--bg-card);
            border-radius: 12px;
            padding: 20px;
            overflow: auto;
            max-height: 70vh;
        }

        .search-node {
            list-style: none;
            margin: 4px 0;
        }

        .search-node-content {
            display: flex;
            align-items: flex-start;
            padding: 8px 12px;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.15s;
            gap: 10px;
        }

        .search-node-content:hover {
            background: var(--bg-hover);
        }

        .search-toggle {
            width: 20px;
            color: var(--text-secondary);
            font-size: 0.9em;
            flex-shrink: 0;
            padding-top: 2px;
        }

        .search-node-label {
            flex-grow: 1;
            font-family: 'SF Mono', monospace;
            font-size: 0.95em;
        }

        .search-node-info {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .search-node-desc {
            color: var(--text-secondary);
            font-size: 0.8em;
            font-style: italic;
        }

        /* Search node types */
        .search-node-hole .search-node-label {
            color: var(--accent-yellow);
        }

        .search-node-abstraction .search-node-label {
            color: var(--accent-cyan);
        }

        .search-node-rule .search-node-label {
            color: var(--accent-green);
            font-weight: 600;
        }

        .search-node-rule .search-node-content {
            background: rgba(158, 206, 106, 0.1);
            border-left: 3px solid var(--accent-green);
        }

        .search-node-dead_end .search-node-label {
            color: var(--accent-red);
            opacity: 0.7;
        }

        .search-node-dead_end .search-node-content {
            background: rgba(247, 118, 142, 0.08);
            border-left: 2px dashed var(--accent-red);
        }

        .search-rule-name {
            background: var(--accent-green);
            color: var(--bg-dark);
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75em;
            font-weight: 600;
        }

        .search-depth-badge {
            background: var(--accent-orange);
            color: var(--bg-dark);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.7em;
            margin-left: 8px;
        }

        .search-requires {
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
            margin-top: 4px;
        }

        .search-requires-tag {
            background: var(--accent-purple);
            color: var(--bg-dark);
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.7em;
        }

        .search-children {
            padding-left: 28px;
            border-left: 2px dashed var(--text-secondary);
            margin-left: 10px;
            display: none;
        }

        .search-children.expanded {
            display: block;
        }

        /* Hole placeholder styling */
        .hole-placeholder {
            background: rgba(224, 175, 104, 0.2);
            border: 1px dashed var(--accent-yellow);
            padding: 1px 4px;
            border-radius: 3px;
        }

        .search-legend {
            display: flex;
            justify-content: center;
            gap: 25px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }

        .search-legend-item {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.85em;
        }

        .search-legend-sample {
            padding: 4px 10px;
            border-radius: 4px;
            font-family: 'SF Mono', monospace;
            font-size: 0.85em;
        }

        .search-legend-sample.hole {
            color: var(--accent-yellow);
            background: rgba(224, 175, 104, 0.15);
            border: 1px dashed var(--accent-yellow);
        }

        .search-legend-sample.abstraction {
            color: var(--accent-cyan);
            background: rgba(125, 207, 255, 0.15);
        }

        .search-legend-sample.rule {
            color: var(--accent-green);
            background: rgba(158, 206, 106, 0.15);
            border-left: 3px solid var(--accent-green);
        }

        .search-intro {
            text-align: center;
            color: var(--text-secondary);
            margin-bottom: 20px;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
            line-height: 1.6;
        }

        .search-intro code {
            background: var(--bg-hover);
            padding: 2px 6px;
            border-radius: 4px;
            color: var(--accent-yellow);
        }
    </style>
</head>
<body>
    <h1>🌳 Rule Dependency & Search Tree</h1>
    <p class="subtitle">Interactive visualization of program synthesis structure</p>

    <!-- Tab Navigation -->
    <div class="tab-nav">
        <button class="tab-btn active" onclick="switchTab('dependency')">Dependency Tree</button>
        <button class="tab-btn" onclick="switchTab('search')">Search Tree</button>
    </div>

    <!-- Tab 1: Dependency Tree -->
    <div id="dependency-tab" class="tab-content active">
        <div class="stats">
            <div class="stat">
                <div class="value">""" + str(n_primitives) + """</div>
                <div class="label">Primitives</div>
            </div>
            <div class="stat">
                <div class="value">""" + str(n_abstractions) + """</div>
                <div class="label">Abstractions</div>
            </div>
            <div class="stat">
                <div class="value">""" + str(n_rules) + """</div>
                <div class="label">Rules</div>
            </div>
        </div>

        <div class="legend">
            <div class="legend-item"><div class="legend-dot primitive"></div> Primitive (depth 0)</div>
            <div class="legend-item"><div class="legend-dot abstraction"></div> Abstraction (depth 2-12)</div>
            <div class="legend-item"><div class="legend-dot rule"></div> Rule (target)</div>
        </div>

        <div class="controls">
            <input type="text" id="search" placeholder="Search nodes..." onkeyup="searchNodes()">
            <button onclick="expandAll()">Expand All</button>
            <button onclick="collapseAll()">Collapse All</button>
            <button onclick="showRulesOnly()">Show Rules</button>
            <button onclick="showAbstractionsOnly()">Show Abstractions</button>
            <button onclick="showAll()">Show All</button>
        </div>

        <div class="root-selector" id="rootSelector">
            <span style="color: var(--text-secondary); margin-right: 10px;">View from:</span>
            <button class="root-btn active" onclick="setView('rules')">Rules (top-down)</button>
            <button class="root-btn" onclick="setView('families')">By Family</button>
            <button class="root-btn" onclick="setView('abstractions')">Key Abstractions</button>
        </div>

        <div class="tree-container">
            <ul class="tree-view" id="treeView"></ul>
        </div>
    </div>

    <!-- Tab 2: Search Tree -->
    <div id="search-tab" class="tab-content">
        <p class="search-intro">
            This tree shows program synthesis as <strong>hole-filling</strong>. Starting from the root
            <code>λh. [?:bool]</code> (a function from Hand to Bool), each branch represents a way to
            fill a typed hole <code>[?:type]</code> with a primitive, abstraction, or further holes.
            Click on nodes to expand and explore paths to complete rule programs at the leaves.
        </p>

        <div class="search-legend">
            <div class="search-legend-item">
                <span class="search-legend-sample hole">[?:type]</span>
                <span>Unfilled hole</span>
            </div>
            <div class="search-legend-item">
                <span class="search-legend-sample abstraction">abstraction</span>
                <span>Requires learned abstraction</span>
            </div>
            <div class="search-legend-item">
                <span class="search-legend-sample rule">Complete rule</span>
                <span>Leaf = solved task (57 total)</span>
            </div>
            <div class="search-legend-item">
                <span class="search-legend-sample" style="color: var(--accent-red); opacity: 0.7; border-left: 2px dashed var(--accent-red);">dead end</span>
                <span>No rule at this path</span>
            </div>
        </div>

        <div class="controls">
            <button onclick="expandSearchAll()">Expand All</button>
            <button onclick="collapseSearchAll()">Collapse All</button>
            <button onclick="expandSearchToDepth(2)">Expand 2 Levels</button>
            <button onclick="expandSearchToDepth(3)">Expand 3 Levels</button>
        </div>

        <div class="search-tree-container">
            <ul class="search-tree-view" id="searchTreeView"></ul>
        </div>
    </div>

    <div class="detail-panel" id="detailPanel">
        <button class="close-btn" onclick="closeDetail()">×</button>
        <h3 id="detailTitle"></h3>
        <div class="detail-section">
            <div class="label">Type</div>
            <div class="value" id="detailType"></div>
        </div>
        <div class="detail-section">
            <div class="label">Description</div>
            <div class="value" id="detailDesc"></div>
        </div>
        <div class="detail-section">
            <div class="label">Lambda Expression</div>
            <div class="value lambda" id="detailLambda"></div>
        </div>
        <div class="detail-section">
            <div class="label">Depth</div>
            <div class="value" id="detailDepth"></div>
        </div>
        <div class="detail-section" id="childrenSection">
            <div class="label">Dependencies (children)</div>
            <div class="dep-list" id="detailChildren"></div>
        </div>
        <div class="detail-section" id="parentsSection">
            <div class="label">Used by (parents)</div>
            <div class="dep-list" id="detailParents"></div>
        </div>
    </div>

    <script>
        const treeData = """ + tree_json + """;

        let currentView = 'rules';

        function getIcon(type) {
            switch(type) {
                case 'primitive': return '◆';
                case 'abstraction': return '◇';
                case 'rule': return '▣';
                default: return '•';
            }
        }

        function createNodeElement(nodeId, expanded = false) {
            const node = treeData[nodeId];
            if (!node) return null;

            const li = document.createElement('li');
            li.className = 'tree-node';
            li.dataset.nodeId = nodeId;
            li.dataset.type = node.type;

            const content = document.createElement('div');
            content.className = `node-content node-${node.type}`;

            // Toggle icon
            const toggle = document.createElement('span');
            toggle.className = 'toggle-icon';
            if (node.children && node.children.length > 0) {
                toggle.textContent = expanded ? '▼' : '▶';
                toggle.onclick = (e) => {
                    e.stopPropagation();
                    toggleNode(li);
                };
            }
            content.appendChild(toggle);

            // Node icon
            const icon = document.createElement('span');
            icon.className = 'node-icon';
            icon.textContent = getIcon(node.type);
            content.appendChild(icon);

            // Node name
            const name = document.createElement('span');
            name.className = 'node-name';
            name.textContent = node.name;
            content.appendChild(name);

            // Family badge for rules
            if (node.family) {
                const badge = document.createElement('span');
                badge.className = `node-badge badge-${node.family}`;
                badge.textContent = node.family;
                content.appendChild(badge);
            }

            // Depth indicator
            if (node.depth > 0) {
                const depth = document.createElement('span');
                depth.className = 'node-depth';
                depth.textContent = `d=${node.depth}`;
                content.appendChild(depth);
            }

            content.onclick = () => showDetail(nodeId);
            li.appendChild(content);

            // Children container
            if (node.children && node.children.length > 0) {
                const childrenUl = document.createElement('ul');
                childrenUl.className = 'children' + (expanded ? ' expanded' : '');

                // Sort children: rules first, then abstractions, then primitives
                const sortedChildren = [...node.children].sort((a, b) => {
                    const typeOrder = { 'rule': 0, 'abstraction': 1, 'primitive': 2 };
                    const aType = treeData[a]?.type || 'primitive';
                    const bType = treeData[b]?.type || 'primitive';
                    return typeOrder[aType] - typeOrder[bType];
                });

                for (const childId of sortedChildren) {
                    if (treeData[childId]) {
                        const childEl = createNodeElement(childId, false);
                        if (childEl) childrenUl.appendChild(childEl);
                    }
                }
                li.appendChild(childrenUl);
            }

            return li;
        }

        function toggleNode(li) {
            const children = li.querySelector('.children');
            const toggle = li.querySelector('.toggle-icon');
            if (children) {
                children.classList.toggle('expanded');
                toggle.textContent = children.classList.contains('expanded') ? '▼' : '▶';
            }
        }

        function showDetail(nodeId) {
            const node = treeData[nodeId];
            if (!node) return;

            // Highlight selected node
            document.querySelectorAll('.node-content').forEach(el => el.classList.remove('selected'));
            const nodeEl = document.querySelector(`[data-node-id="${nodeId}"] > .node-content`);
            if (nodeEl) nodeEl.classList.add('selected');

            document.getElementById('detailTitle').textContent = node.name;
            document.getElementById('detailType').textContent = node.type.charAt(0).toUpperCase() + node.type.slice(1) + (node.family ? ` (${node.family})` : '');
            document.getElementById('detailDesc').textContent = node.description || '-';
            document.getElementById('detailLambda').textContent = node.definition || '-';
            document.getElementById('detailDepth').textContent = node.depth;

            // Children
            const childrenEl = document.getElementById('detailChildren');
            childrenEl.innerHTML = '';
            if (node.children && node.children.length > 0) {
                document.getElementById('childrenSection').style.display = 'block';
                for (const childId of node.children) {
                    const child = treeData[childId];
                    if (child) {
                        const tag = document.createElement('span');
                        tag.className = `dep-tag ${child.type}`;
                        tag.textContent = childId;
                        tag.onclick = () => {
                            showDetail(childId);
                            expandToNode(childId);
                        };
                        childrenEl.appendChild(tag);
                    }
                }
            } else {
                document.getElementById('childrenSection').style.display = 'none';
            }

            // Parents
            const parentsEl = document.getElementById('detailParents');
            parentsEl.innerHTML = '';
            if (node.parents && node.parents.length > 0) {
                document.getElementById('parentsSection').style.display = 'block';
                for (const parentId of node.parents) {
                    const parent = treeData[parentId];
                    if (parent) {
                        const tag = document.createElement('span');
                        tag.className = `dep-tag ${parent.type}`;
                        tag.textContent = parentId;
                        tag.onclick = () => {
                            showDetail(parentId);
                            expandToNode(parentId);
                        };
                        parentsEl.appendChild(tag);
                    }
                }
            } else {
                document.getElementById('parentsSection').style.display = 'none';
            }

            document.getElementById('detailPanel').classList.add('visible');
        }

        function closeDetail() {
            document.getElementById('detailPanel').classList.remove('visible');
            document.querySelectorAll('.node-content').forEach(el => el.classList.remove('selected'));
        }

        function expandToNode(nodeId) {
            const nodeEl = document.querySelector(`[data-node-id="${nodeId}"]`);
            if (nodeEl) {
                // Expand all parents
                let parent = nodeEl.parentElement;
                while (parent) {
                    if (parent.classList.contains('children')) {
                        parent.classList.add('expanded');
                        const toggle = parent.previousElementSibling?.querySelector('.toggle-icon');
                        if (toggle) toggle.textContent = '▼';
                    }
                    parent = parent.parentElement;
                }
                nodeEl.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
        }

        function expandAll() {
            document.querySelectorAll('.children').forEach(el => el.classList.add('expanded'));
            document.querySelectorAll('.toggle-icon').forEach(el => {
                if (el.textContent === '▶') el.textContent = '▼';
            });
        }

        function collapseAll() {
            document.querySelectorAll('.children').forEach(el => el.classList.remove('expanded'));
            document.querySelectorAll('.toggle-icon').forEach(el => {
                if (el.textContent === '▼') el.textContent = '▶';
            });
        }

        function showRulesOnly() {
            document.querySelectorAll('.tree-node').forEach(el => {
                el.style.display = el.dataset.type === 'rule' ? '' : 'none';
            });
        }

        function showAbstractionsOnly() {
            document.querySelectorAll('.tree-node').forEach(el => {
                el.style.display = (el.dataset.type === 'rule' || el.dataset.type === 'abstraction') ? '' : 'none';
            });
        }

        function showAll() {
            document.querySelectorAll('.tree-node').forEach(el => {
                el.style.display = '';
            });
        }

        function searchNodes() {
            const query = document.getElementById('search').value.toLowerCase();
            document.querySelectorAll('.tree-node').forEach(el => {
                const nodeId = el.dataset.nodeId;
                const node = treeData[nodeId];
                const match = nodeId.toLowerCase().includes(query) ||
                              (node.description && node.description.toLowerCase().includes(query)) ||
                              (node.family && node.family.toLowerCase().includes(query));
                el.style.display = match || query === '' ? '' : 'none';
                if (match && query !== '') {
                    expandToNode(nodeId);
                }
            });
        }

        function setView(view) {
            currentView = view;
            document.querySelectorAll('.root-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            renderTree();
        }

        function renderTree() {
            const treeView = document.getElementById('treeView');
            treeView.innerHTML = '';

            if (currentView === 'rules') {
                // Group rules by depth
                const rulesByDepth = {};
                for (const [id, node] of Object.entries(treeData)) {
                    if (node.type === 'rule') {
                        const bucket = node.depth <= 6 ? 'trivial' : node.depth <= 8 ? 'easy' : node.depth <= 10 ? 'medium' : node.depth <= 14 ? 'hard' : 'extreme';
                        if (!rulesByDepth[bucket]) rulesByDepth[bucket] = [];
                        rulesByDepth[bucket].push(id);
                    }
                }

                const buckets = [
                    { key: 'trivial', name: 'Trivial (depth ≤6)', color: '#9ece6a' },
                    { key: 'easy', name: 'Easy (depth 7-8)', color: '#7dcfff' },
                    { key: 'medium', name: 'Medium (depth 9-10)', color: '#e0af68' },
                    { key: 'hard', name: 'Hard (depth 11-14)', color: '#f7768e' },
                    { key: 'extreme', name: 'Extreme (depth 15+)', color: '#bb9af7' }
                ];

                for (const bucket of buckets) {
                    if (rulesByDepth[bucket.key]?.length > 0) {
                        const li = document.createElement('li');
                        li.className = 'tree-node';

                        const content = document.createElement('div');
                        content.className = 'node-content';
                        content.innerHTML = `
                            <span class="toggle-icon">▶</span>
                            <span class="node-icon" style="color: ${bucket.color}">📁</span>
                            <span class="node-name" style="color: ${bucket.color}">${bucket.name}</span>
                            <span class="node-badge" style="background: ${bucket.color}; color: #1a1a2e">${rulesByDepth[bucket.key].length}</span>
                        `;
                        content.onclick = () => toggleNode(li);
                        li.appendChild(content);

                        const childrenUl = document.createElement('ul');
                        childrenUl.className = 'children';

                        for (const ruleId of rulesByDepth[bucket.key].sort()) {
                            const ruleEl = createNodeElement(ruleId, false);
                            if (ruleEl) childrenUl.appendChild(ruleEl);
                        }
                        li.appendChild(childrenUl);
                        treeView.appendChild(li);
                    }
                }
            } else if (currentView === 'families') {
                // Group rules by family
                const families = {};
                for (const [id, node] of Object.entries(treeData)) {
                    if (node.type === 'rule' && node.family) {
                        if (!families[node.family]) families[node.family] = [];
                        families[node.family].push(id);
                    }
                }

                for (const family of Object.keys(families).sort()) {
                    const li = document.createElement('li');
                    li.className = 'tree-node';

                    const content = document.createElement('div');
                    content.className = 'node-content';
                    content.innerHTML = `
                        <span class="toggle-icon">▶</span>
                        <span class="node-icon" style="color: var(--accent-purple)">📂</span>
                        <span class="node-name" style="color: var(--accent-purple)">${family}</span>
                        <span class="node-badge badge-${family}">${families[family].length}</span>
                    `;
                    content.onclick = () => toggleNode(li);
                    li.appendChild(content);

                    const childrenUl = document.createElement('ul');
                    childrenUl.className = 'children';

                    for (const ruleId of families[family].sort()) {
                        const ruleEl = createNodeElement(ruleId, false);
                        if (ruleEl) childrenUl.appendChild(ruleEl);
                    }
                    li.appendChild(childrenUl);
                    treeView.appendChild(li);
                }
            } else if (currentView === 'abstractions') {
                // Show key abstractions with their dependents
                const keyAbstractions = ['first_half', 'second_half', 'is_palindrome_by', 'halves_equal_by', 'halves_property_equal', 'shifted_pairs', 'suit_cycle_m1', 'suit_cycle_m2', 'bracket_match', 'has_AP'];

                for (const absId of keyAbstractions) {
                    const node = treeData[absId];
                    if (node) {
                        const li = document.createElement('li');
                        li.className = 'tree-node';
                        li.dataset.nodeId = absId;
                        li.dataset.type = 'abstraction';

                        const content = document.createElement('div');
                        content.className = 'node-content node-abstraction';
                        content.innerHTML = `
                            <span class="toggle-icon">${node.parents.length > 0 ? '▶' : ''}</span>
                            <span class="node-icon">◇</span>
                            <span class="node-name">${absId}</span>
                            <span class="node-depth">d=${node.depth}</span>
                            <span class="node-badge" style="background: var(--accent-cyan); color: var(--bg-dark)">${node.parents.length} rules</span>
                        `;
                        content.onclick = () => {
                            showDetail(absId);
                            if (node.parents.length > 0) toggleNode(li);
                        };
                        li.appendChild(content);

                        if (node.parents.length > 0) {
                            const childrenUl = document.createElement('ul');
                            childrenUl.className = 'children';

                            for (const parentId of node.parents.sort()) {
                                const parentNode = treeData[parentId];
                                if (parentNode && parentNode.type === 'rule') {
                                    const parentLi = document.createElement('li');
                                    parentLi.className = 'tree-node';
                                    parentLi.dataset.nodeId = parentId;
                                    parentLi.dataset.type = 'rule';

                                    const parentContent = document.createElement('div');
                                    parentContent.className = 'node-content node-rule';
                                    parentContent.innerHTML = `
                                        <span class="toggle-icon"></span>
                                        <span class="node-icon">▣</span>
                                        <span class="node-name">${parentId}</span>
                                        <span class="node-badge badge-${parentNode.family}">${parentNode.family}</span>
                                    `;
                                    parentContent.onclick = () => showDetail(parentId);
                                    parentLi.appendChild(parentContent);
                                    childrenUl.appendChild(parentLi);
                                }
                            }
                            li.appendChild(childrenUl);
                        }
                        treeView.appendChild(li);
                    }
                }
            }
        }

        // Initial render
        renderTree();

        // =====================================================================
        // TAB SWITCHING
        // =====================================================================

        function switchTab(tabName) {
            // Update tab buttons
            document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');

            // Update tab content
            document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
            document.getElementById(tabName + '-tab').classList.add('active');

            // Close detail panel when switching tabs
            closeDetail();

            // Render search tree on first view
            if (tabName === 'search' && !searchTreeRendered) {
                renderSearchTree();
                searchTreeRendered = true;
            }
        }

        // =====================================================================
        // SEARCH TREE RENDERING
        // =====================================================================

        const searchTreeData = """ + search_tree_json + """;
        let searchTreeRendered = false;

        function createSearchNodeElement(node, depth = 0) {
            const li = document.createElement('li');
            li.className = `search-node search-node-${node.type}`;
            li.dataset.depth = depth;

            const content = document.createElement('div');
            content.className = 'search-node-content';

            // Toggle icon
            const toggle = document.createElement('span');
            toggle.className = 'search-toggle';
            if (node.children && node.children.length > 0) {
                toggle.textContent = '▶';
                toggle.onclick = (e) => {
                    e.stopPropagation();
                    toggleSearchNode(li);
                };
            }
            content.appendChild(toggle);

            // Node info container
            const infoDiv = document.createElement('div');
            infoDiv.className = 'search-node-info';

            // Label (the program expression)
            const label = document.createElement('div');
            label.className = 'search-node-label';

            // Highlight holes in the label
            let labelText = node.label;
            labelText = labelText.replace(/\\[\\?:[^\\]]+\\]/g, '<span class="hole-placeholder">$&</span>');
            label.innerHTML = labelText;

            // Add rule badge if it's a rule
            if (node.type === 'rule' && node.rule) {
                const ruleBadge = document.createElement('span');
                ruleBadge.className = 'search-rule-name';
                ruleBadge.textContent = node.rule;
                label.appendChild(document.createTextNode(' '));
                label.appendChild(ruleBadge);

                // Add family badge for rules
                if (node.family) {
                    const familyBadge = document.createElement('span');
                    familyBadge.className = 'node-badge badge-' + node.family;
                    familyBadge.textContent = node.family;
                    familyBadge.style.marginLeft = '5px';
                    label.appendChild(familyBadge);
                }
            }

            // Add depth badge if available
            if (node.depth !== undefined && node.depth !== 'varies') {
                const depthBadge = document.createElement('span');
                depthBadge.className = 'search-depth-badge';
                depthBadge.textContent = 'd=' + node.depth;
                label.appendChild(depthBadge);
            }

            infoDiv.appendChild(label);

            // Description
            if (node.description) {
                const desc = document.createElement('div');
                desc.className = 'search-node-desc';
                desc.textContent = node.description;
                infoDiv.appendChild(desc);
            }

            // Required abstractions
            if (node.requires && node.requires.length > 0) {
                const reqDiv = document.createElement('div');
                reqDiv.className = 'search-requires';
                for (const req of node.requires) {
                    const tag = document.createElement('span');
                    tag.className = 'search-requires-tag';
                    tag.textContent = req;
                    reqDiv.appendChild(tag);
                }
                infoDiv.appendChild(reqDiv);
            }

            content.appendChild(infoDiv);
            li.appendChild(content);

            // Children
            if (node.children && node.children.length > 0) {
                const childrenUl = document.createElement('ul');
                childrenUl.className = 'search-children';

                for (const child of node.children) {
                    const childEl = createSearchNodeElement(child, depth + 1);
                    childrenUl.appendChild(childEl);
                }
                li.appendChild(childrenUl);
            }

            return li;
        }

        function toggleSearchNode(li) {
            const children = li.querySelector('.search-children');
            const toggle = li.querySelector('.search-toggle');
            if (children) {
                children.classList.toggle('expanded');
                toggle.textContent = children.classList.contains('expanded') ? '▼' : '▶';
            }
        }

        function renderSearchTree() {
            const container = document.getElementById('searchTreeView');
            container.innerHTML = '';

            const rootEl = createSearchNodeElement(searchTreeData, 0);
            container.appendChild(rootEl);

            // Expand root by default
            const rootChildren = rootEl.querySelector('.search-children');
            const rootToggle = rootEl.querySelector('.search-toggle');
            if (rootChildren) {
                rootChildren.classList.add('expanded');
                rootToggle.textContent = '▼';
            }
        }

        function expandSearchAll() {
            document.querySelectorAll('.search-children').forEach(el => el.classList.add('expanded'));
            document.querySelectorAll('.search-toggle').forEach(el => {
                if (el.textContent === '▶') el.textContent = '▼';
            });
        }

        function collapseSearchAll() {
            document.querySelectorAll('.search-children').forEach(el => el.classList.remove('expanded'));
            document.querySelectorAll('.search-toggle').forEach(el => {
                if (el.textContent === '▼') el.textContent = '▶';
            });
        }

        function expandSearchToDepth(maxDepth) {
            // First collapse all
            collapseSearchAll();

            // Then expand up to maxDepth
            document.querySelectorAll('.search-node').forEach(node => {
                const depth = parseInt(node.dataset.depth);
                if (depth < maxDepth) {
                    const children = node.querySelector(':scope > .search-children');
                    const toggle = node.querySelector(':scope > .search-node-content > .search-toggle');
                    if (children) {
                        children.classList.add('expanded');
                        if (toggle) toggle.textContent = '▼';
                    }
                }
            });
        }
    </script>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Interactive tree saved to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("Building dependency tree...")
    nodes = build_tree()

    output_path = Path(__file__).parent.parent / 'results' / 'rule_dependency_tree.html'
    output_path.parent.mkdir(exist_ok=True)

    generate_html_report(nodes, str(output_path))

    # Print summary
    print("\n" + "=" * 60)
    print("DEPENDENCY TREE SUMMARY")
    print("=" * 60)

    n_prims = sum(1 for n in nodes.values() if n.node_type == 'primitive')
    n_abs = sum(1 for n in nodes.values() if n.node_type == 'abstraction')
    n_rules = sum(1 for n in nodes.values() if n.node_type == 'rule')

    print(f"Primitives: {n_prims}")
    print(f"Abstractions: {n_abs}")
    print(f"Rules: {n_rules}")

    # Most connected abstractions
    print("\nMost-used abstractions (by parent count):")
    abs_by_parents = [(n.name, len(n.parents)) for n in nodes.values() if n.node_type == 'abstraction']
    for name, count in sorted(abs_by_parents, key=lambda x: -x[1])[:10]:
        print(f"  {name}: {count} dependents")
