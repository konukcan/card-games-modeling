#!/usr/bin/env python3
"""
Write missing true-rule DSL programs for all 60 gallery rules.

For each gallery rule:
1. Check if any existing injected hypothesis is semantically equivalent
   (matches on 500+ random hands with at least some True results)
2. If not, add a hand-written DSL program with source="true_rule"
3. Verify every true-rule DSL against exemplar hands before saving

Usage:
    python -m gallery_analysis.write_true_rules          # dry-run: show what's missing
    python -m gallery_analysis.write_true_rules --save   # write updated JSON
"""

import sys
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Card, Hand, Suit, Rank, RANK_VALUES
from gallery_analysis.gallery_rules import GALLERY_RULES
from gallery_analysis.exemplars import load_exemplars
from dreamcoder_core.program import parse_program, Primitive
from dreamcoder_core.primitives import build_primitives
from gallery_analysis.enumerator import build_gallery_grammar


# =========================================================================
# Setup
# =========================================================================

DATA_PATH = Path(__file__).parent / "data" / "injected_hypotheses.json"

FULL_DECK = [Card(s, r) for s in Suit for r in Rank]


def random_hands(n: int = 500, seed: int = 12345) -> List[Hand]:
    """Generate n random 6-card hands deterministically."""
    rng = random.Random(seed)
    return [rng.sample(FULL_DECK, 6) for _ in range(n)]


def build_prim_dict():
    """Build primitive lookup dictionary from gallery grammar + full primitives."""
    grammar = build_gallery_grammar()
    d = {}
    for prod in grammar.productions:
        if isinstance(prod.program, Primitive):
            d[prod.program.name] = prod.program
    all_prims = build_primitives()
    for p in all_prims:
        if p.name not in d:
            d[p.name] = p
    return d


def make_evaluator(dsl_program: str, prim_dict: dict):
    """Parse a DSL program and return a callable predicate."""
    prog = parse_program(dsl_program, prim_dict)

    def predicate(hand: Hand) -> bool:
        try:
            result = prog.evaluate([])(hand)
            return bool(result)
        except Exception:
            return False
    return predicate


# =========================================================================
# Hand-written DSL programs for all 60 gallery rules
# =========================================================================
#
# DE BRUIJN INDEX SCOPING RULES:
#   (lambda BODY) where $0 = bound variable
#   In HOFs like (all PRED LIST):
#     PRED = (lambda BODY) creates a new scope where $0 = element
#     LIST is evaluated in the OUTER scope (same as the enclosing lambda)
#   Nested HOFs: each (lambda ...) adds one index layer
#     Outer lambda: $0 = hand
#     (all (lambda ...) $0)  -- $0 here is the hand (outer scope)
#       Inside pred lambda: $0 = element, $1 = hand
#       (any (lambda ...) $1) -- $1 here is hand (one layer up)
#         Inside inner pred lambda: $0 = inner element, $1 = outer element, $2 = hand

TRUE_RULE_DSL: Dict[str, str] = {
    # =====================================================================
    # GROUP 1: Simple, single-feature rules
    # =====================================================================

    # All cards are red (hearts or diamonds)
    "all_red": "(λ all (λ eq (get_color $0) RED) $0)",

    # Every card is a club or a heart
    "all_clubs_or_hearts": "(λ all (λ or (eq (get_suit $0) CLUBS) (eq (get_suit $0) HEARTS)) $0)",

    # All six cards share the same suit
    "all_same_suit": "(λ eq (n_unique_suits $0) 1)",

    # Every card is a 4 or a Queen (rank_val 4 or 12 = 5+5+2)
    "all_4s_or_queens": "(λ all (λ or (eq (rank_val $0) 4) (eq (rank_val $0) (+ 5 (+ 5 2)))) $0)",

    # Every card is a 4, 8, or 9
    "all_4s_8s_or_9s": "(λ all (λ or (eq (rank_val $0) 4) (or (eq (rank_val $0) (+ 5 3)) (eq (rank_val $0) (+ 5 4)))) $0)",

    # Positions 2, 3, and 4 (1-indexed = indices 1,2,3) are all 2s
    "triple_2s_pos234": "(λ and (eq (rank_val (at $0 1)) 2) (and (eq (rank_val (at $0 2)) 2) (eq (rank_val (at $0 3)) 2)))",

    # Positions 3, 4, and 5 (1-indexed = indices 2,3,4) share the same rank
    "triple_any_pos345": "(λ and (eq (get_rank (at $0 2)) (get_rank (at $0 3))) (eq (get_rank (at $0 3)) (get_rank (at $0 4))))",

    # Four consecutive cards share the same rank (check windows 0-3, 1-4, 2-5)
    "four_of_a_kind_adjacent":
        "(λ or (and (eq (get_rank (at $0 0)) (get_rank (at $0 1))) "
        "(and (eq (get_rank (at $0 1)) (get_rank (at $0 2))) (eq (get_rank (at $0 2)) (get_rank (at $0 3))))) "
        "(or (and (eq (get_rank (at $0 1)) (get_rank (at $0 2))) "
        "(and (eq (get_rank (at $0 2)) (get_rank (at $0 3))) (eq (get_rank (at $0 3)) (get_rank (at $0 4))))) "
        "(and (eq (get_rank (at $0 2)) (get_rank (at $0 3))) "
        "(and (eq (get_rank (at $0 3)) (get_rank (at $0 4))) (eq (get_rank (at $0 4)) (get_rank (at $0 5)))))))",

    # All cards except at most one are the same color: min(red_count, black_count) <= 1
    "all_but_one_same_color":
        "(λ le (if (le (count_color $0 RED) (count_color $0 BLACK)) (count_color $0 RED) (count_color $0 BLACK)) 1)",

    # At least 3 cards share the same suit
    "three_or_more_same_suit": "(λ ge (max_suit_count $0) 3)",

    # All cards share the same color
    "all_same_color": "(λ eq (n_unique_colors $0) 1)",

    # Positions 4 and 5 (1-indexed = indices 3,4) are both Jacks (rank_val 11 = 5+5+1)
    "pair_jacks_pos45": "(λ and (eq (rank_val (at $0 3)) (+ 5 (+ 5 1))) (eq (rank_val (at $0 4)) (+ 5 (+ 5 1))))",

    # Three consecutive cards are all 3s (windows 0-2, 1-3, 2-4, 3-5)
    "triple_3s_adjacent":
        "(λ or (and (eq (rank_val (at $0 0)) 3) (and (eq (rank_val (at $0 1)) 3) (eq (rank_val (at $0 2)) 3))) "
        "(or (and (eq (rank_val (at $0 1)) 3) (and (eq (rank_val (at $0 2)) 3) (eq (rank_val (at $0 3)) 3))) "
        "(or (and (eq (rank_val (at $0 2)) 3) (and (eq (rank_val (at $0 3)) 3) (eq (rank_val (at $0 4)) 3))) "
        "(and (eq (rank_val (at $0 3)) 3) (and (eq (rank_val (at $0 4)) 3) (eq (rank_val (at $0 5)) 3))))))",

    # Four cards share the same rank (any position, not necessarily adjacent)
    # any card c: count of cards with same rank >= 4
    # Scoping: outer λ $0=hand; any's pred λ $0=card_c, $1=hand;
    #   filter's pred λ $0=card_d, $1=card_c; filter's list = $1 (hand in any's scope)
    "four_kind_adjacent_any":
        "(λ any (λ ge (length (filter (λ eq (get_rank $0) (get_rank $1)) $1)) 4) $0)",

    # Three consecutive cards are all clubs (windows 0-2, 1-3, 2-4, 3-5)
    "three_clubs_adjacent":
        "(λ or (and (eq (get_suit (at $0 0)) CLUBS) (and (eq (get_suit (at $0 1)) CLUBS) (eq (get_suit (at $0 2)) CLUBS))) "
        "(or (and (eq (get_suit (at $0 1)) CLUBS) (and (eq (get_suit (at $0 2)) CLUBS) (eq (get_suit (at $0 3)) CLUBS))) "
        "(or (and (eq (get_suit (at $0 2)) CLUBS) (and (eq (get_suit (at $0 3)) CLUBS) (eq (get_suit (at $0 4)) CLUBS))) "
        "(and (eq (get_suit (at $0 3)) CLUBS) (and (eq (get_suit (at $0 4)) CLUBS) (eq (get_suit (at $0 5)) CLUBS))))))",

    # Positions 1, 3, 5 (1-indexed = indices 0,2,4) are all Aces (rank_val 14 = 5+5+4)
    "every_other_ace":
        "(λ and (eq (rank_val (at $0 0)) (+ 5 (+ 5 4))) "
        "(and (eq (rank_val (at $0 2)) (+ 5 (+ 5 4))) (eq (rank_val (at $0 4)) (+ 5 (+ 5 4)))))",

    # Positions 1, 3, 5 (1-indexed = indices 0,2,4) share the same rank
    "pos135_same_rank":
        "(λ and (eq (get_rank (at $0 0)) (get_rank (at $0 2))) (eq (get_rank (at $0 2)) (get_rank (at $0 4))))",

    # Left half is all red, right half is all black
    "left_red_right_black":
        "(λ and (all (λ eq (get_color $0) RED) (first_half $1)) (all (λ eq (get_color $0) BLACK) (second_half $1)))",

    # One half is all red and the other half is all black (either direction)
    "some_half_red_other_black":
        "(λ or (and (all (λ eq (get_color $0) RED) (first_half $1)) (all (λ eq (get_color $0) BLACK) (second_half $1))) "
        "(and (all (λ eq (get_color $0) BLACK) (first_half $1)) (all (λ eq (get_color $0) RED) (second_half $1))))",

    # Each half has all cards of the same suit (halves may differ)
    "both_halves_uniform_suit":
        "(λ and (eq (get_suit (at (first_half $0) 0)) (get_suit (at (first_half $0) 1))) "
        "(and (eq (get_suit (at (first_half $0) 1)) (get_suit (at (first_half $0) 2))) "
        "(and (eq (get_suit (at (second_half $0) 0)) (get_suit (at (second_half $0) 1))) "
        "(eq (get_suit (at (second_half $0) 1)) (get_suit (at (second_half $0) 2))))))",

    # =====================================================================
    # GROUP 2: Multi-feature, counting, positional
    # =====================================================================

    # Every card is an even number (2,4,6,8,10). rank_val % 2 == 0 AND rank_val <= 10
    "all_even": "(λ all (λ and (eq (mod (rank_val $0) 2) 0) (le (rank_val $0) (+ 5 5))) $0)",

    # Every card is odd (3,5,7,9). rank_val % 2 == 1 AND 3 <= rank_val <= 9
    "all_odd": "(λ all (λ and (eq (mod (rank_val $0) 2) 1) (and (ge (rank_val $0) 3) (le (rank_val $0) (+ 5 4)))) $0)",

    # Two adjacent cards are both 5s (windows 0-1, 1-2, 2-3, 3-4, 4-5)
    "pair_5s_adjacent":
        "(λ or (and (eq (rank_val (at $0 0)) 5) (eq (rank_val (at $0 1)) 5)) "
        "(or (and (eq (rank_val (at $0 1)) 5) (eq (rank_val (at $0 2)) 5)) "
        "(or (and (eq (rank_val (at $0 2)) 5) (eq (rank_val (at $0 3)) 5)) "
        "(or (and (eq (rank_val (at $0 3)) 5) (eq (rank_val (at $0 4)) 5)) "
        "(and (eq (rank_val (at $0 4)) 5) (eq (rank_val (at $0 5)) 5))))))",

    # Three consecutive cards share the same rank (any rank)
    "triple_any_adjacent":
        "(λ or (and (eq (get_rank (at $0 0)) (get_rank (at $0 1))) (eq (get_rank (at $0 1)) (get_rank (at $0 2)))) "
        "(or (and (eq (get_rank (at $0 1)) (get_rank (at $0 2))) (eq (get_rank (at $0 2)) (get_rank (at $0 3)))) "
        "(or (and (eq (get_rank (at $0 2)) (get_rank (at $0 3))) (eq (get_rank (at $0 3)) (get_rank (at $0 4)))) "
        "(and (eq (get_rank (at $0 3)) (get_rank (at $0 4))) (eq (get_rank (at $0 4)) (get_rank (at $0 5)))))))",

    # At least three of the six cards are spades
    "three_spades": "(λ ge (count_suit $0 SPADES) 3)",

    # Three consecutive cards share the same suit (any suit)
    "three_any_suit_adjacent":
        "(λ or (and (eq (get_suit (at $0 0)) (get_suit (at $0 1))) (eq (get_suit (at $0 1)) (get_suit (at $0 2)))) "
        "(or (and (eq (get_suit (at $0 1)) (get_suit (at $0 2))) (eq (get_suit (at $0 2)) (get_suit (at $0 3)))) "
        "(or (and (eq (get_suit (at $0 2)) (get_suit (at $0 3))) (eq (get_suit (at $0 3)) (get_suit (at $0 4)))) "
        "(and (eq (get_suit (at $0 3)) (get_suit (at $0 4))) (eq (get_suit (at $0 4)) (get_suit (at $0 5)))))))",

    # Four consecutive cards are all hearts (windows 0-3, 1-4, 2-5)
    "four_hearts_adjacent":
        "(λ or (and (eq (get_suit (at $0 0)) HEARTS) (and (eq (get_suit (at $0 1)) HEARTS) "
        "(and (eq (get_suit (at $0 2)) HEARTS) (eq (get_suit (at $0 3)) HEARTS)))) "
        "(or (and (eq (get_suit (at $0 1)) HEARTS) (and (eq (get_suit (at $0 2)) HEARTS) "
        "(and (eq (get_suit (at $0 3)) HEARTS) (eq (get_suit (at $0 4)) HEARTS)))) "
        "(and (eq (get_suit (at $0 2)) HEARTS) (and (eq (get_suit (at $0 3)) HEARTS) "
        "(and (eq (get_suit (at $0 4)) HEARTS) (eq (get_suit (at $0 5)) HEARTS))))))",

    # At least 4 cards are diamonds
    "four_diamonds_anywhere": "(λ ge (count_suit $0 DIAMONDS) 4)",

    # Four consecutive cards share the same suit (any suit)
    "four_any_suit_adjacent":
        "(λ or (and (eq (get_suit (at $0 0)) (get_suit (at $0 1))) "
        "(and (eq (get_suit (at $0 1)) (get_suit (at $0 2))) (eq (get_suit (at $0 2)) (get_suit (at $0 3))))) "
        "(or (and (eq (get_suit (at $0 1)) (get_suit (at $0 2))) "
        "(and (eq (get_suit (at $0 2)) (get_suit (at $0 3))) (eq (get_suit (at $0 3)) (get_suit (at $0 4))))) "
        "(and (eq (get_suit (at $0 2)) (get_suit (at $0 3))) "
        "(and (eq (get_suit (at $0 3)) (get_suit (at $0 4))) (eq (get_suit (at $0 4)) (get_suit (at $0 5)))))))",

    # At least 4 cards share the same suit
    "four_any_suit_anywhere": "(λ ge (max_suit_count $0) 4)",

    # Even positions (2,4,6=indices 1,3,5) red, odd positions (1,3,5=indices 0,2,4) black
    "even_pos_red_odd_pos_black":
        "(λ and (eq (get_color (at $0 0)) BLACK) "
        "(and (eq (get_color (at $0 1)) RED) "
        "(and (eq (get_color (at $0 2)) BLACK) "
        "(and (eq (get_color (at $0 3)) RED) "
        "(and (eq (get_color (at $0 4)) BLACK) "
        "(eq (get_color (at $0 5)) RED))))))",

    # Color sequence is a palindrome: c[0]==c[5], c[1]==c[4], c[2]==c[3]
    "colors_palindrome":
        "(λ and (eq (get_color (at $0 0)) (get_color (at $0 5))) "
        "(and (eq (get_color (at $0 1)) (get_color (at $0 4))) "
        "(eq (get_color (at $0 2)) (get_color (at $0 3)))))",

    # Left and right halves have the same color sequence
    "halves_copy_colors":
        "(λ and (eq (get_color (at $0 0)) (get_color (at $0 3))) "
        "(and (eq (get_color (at $0 1)) (get_color (at $0 4))) "
        "(eq (get_color (at $0 2)) (get_color (at $0 5)))))",

    # At least two pairs of matching ranks
    "two_pairs_ranks": "(λ ge (n_repeated_ranks $0) 2)",

    # Two different suits each appear at least twice
    "two_pairs_suits": "(λ ge (n_repeated_suits $0) 2)",

    # Contains 3 cards (anywhere) forming AP with step 1
    # = some card c has rank c+1 and c+2 both present in hand
    # Scoping: outer λ $0=hand; any's pred λ $0=card_c, $1=hand;
    #   inner any's pred λ $0=card_d, $1=card_c, $2=hand
    "ap_len3_step1_anywhere":
        "(λ any (λ and (any (λ eq (rank_val $0) (+ (rank_val $1) 1)) $1) "
        "(any (λ eq (rank_val $0) (+ (rank_val $1) 2)) $1)) $0)",

    # Left and right halves have the same rank sequence
    "halves_copy_ranks":
        "(λ and (eq (rank_val (at $0 0)) (rank_val (at $0 3))) "
        "(and (eq (rank_val (at $0 1)) (rank_val (at $0 4))) "
        "(eq (rank_val (at $0 2)) (rank_val (at $0 5)))))",

    # Left and right halves have the same suit sequence
    "halves_copy_suits":
        "(λ and (eq (get_suit (at $0 0)) (get_suit (at $0 3))) "
        "(and (eq (get_suit (at $0 1)) (get_suit (at $0 4))) "
        "(eq (get_suit (at $0 2)) (get_suit (at $0 5)))))",

    # Each half contains at least one pair of matching ranks
    "both_halves_have_pair_rank":
        "(λ and (or (eq (get_rank (at $0 0)) (get_rank (at $0 1))) "
        "(or (eq (get_rank (at $0 0)) (get_rank (at $0 2))) (eq (get_rank (at $0 1)) (get_rank (at $0 2))))) "
        "(or (eq (get_rank (at $0 3)) (get_rank (at $0 4))) "
        "(or (eq (get_rank (at $0 3)) (get_rank (at $0 5))) (eq (get_rank (at $0 4)) (get_rank (at $0 5))))))",

    # Each half has all cards of the same color (halves may differ)
    "both_halves_uniform_color":
        "(λ and (and (eq (get_color (at $0 0)) (get_color (at $0 1))) (eq (get_color (at $0 1)) (get_color (at $0 2)))) "
        "(and (eq (get_color (at $0 3)) (get_color (at $0 4))) (eq (get_color (at $0 4)) (get_color (at $0 5)))))",

    # =====================================================================
    # GROUP 3: Complex structural patterns
    # =====================================================================

    # Even/odd positions have opposite uniform colors
    "even_odd_pos_color_split":
        "(λ and (and (eq (get_color (at $0 0)) (get_color (at $0 2))) (eq (get_color (at $0 2)) (get_color (at $0 4)))) "
        "(and (and (eq (get_color (at $0 1)) (get_color (at $0 3))) (eq (get_color (at $0 3)) (get_color (at $0 5)))) "
        "(not (eq (get_color (at $0 0)) (get_color (at $0 1))))))",

    # Ranks strictly increase left to right
    "strict_increasing":
        "(λ and (lt (rank_val (at $0 0)) (rank_val (at $0 1))) "
        "(and (lt (rank_val (at $0 1)) (rank_val (at $0 2))) "
        "(and (lt (rank_val (at $0 2)) (rank_val (at $0 3))) "
        "(and (lt (rank_val (at $0 3)) (rank_val (at $0 4))) "
        "(lt (rank_val (at $0 4)) (rank_val (at $0 5)))))))",

    # All black cards appear before all red cards
    # = no adjacent pair has (RED, BLACK)
    "blacks_before_reds":
        "(λ all (λ not (and (eq (get_color (head $0)) RED) (eq (get_color (last $0)) BLACK))) (adjacent_pairs $0))",

    # Every pair of adjacent cards shares either rank or suit
    "adjacent_share_rank_or_suit":
        "(λ all (λ or (eq (get_rank (head $0)) (get_rank (last $0))) (eq (get_suit (head $0)) (get_suit (last $0)))) (adjacent_pairs $0))",

    # Three consecutive ascending cards with step 1 (in position order)
    "ap_step1_len3_adj_ordered":
        "(λ or (and (eq (rank_val (at $0 1)) (+ (rank_val (at $0 0)) 1)) (eq (rank_val (at $0 2)) (+ (rank_val (at $0 0)) 2))) "
        "(or (and (eq (rank_val (at $0 2)) (+ (rank_val (at $0 1)) 1)) (eq (rank_val (at $0 3)) (+ (rank_val (at $0 1)) 2))) "
        "(or (and (eq (rank_val (at $0 3)) (+ (rank_val (at $0 2)) 1)) (eq (rank_val (at $0 4)) (+ (rank_val (at $0 2)) 2))) "
        "(and (eq (rank_val (at $0 4)) (+ (rank_val (at $0 3)) 1)) (eq (rank_val (at $0 5)) (+ (rank_val (at $0 3)) 2))))))",

    # Four consecutive ascending cards with step 2 (in position order)
    # v[i+1]=v[i]+2, v[i+2]=v[i]+4, v[i+3]=v[i]+6 (6 = 5+1)
    "ap_step2_len4_adj_ordered":
        "(λ or (and (eq (rank_val (at $0 1)) (+ (rank_val (at $0 0)) 2)) "
        "(and (eq (rank_val (at $0 2)) (+ (rank_val (at $0 0)) 4)) (eq (rank_val (at $0 3)) (+ (rank_val (at $0 0)) (+ 5 1))))) "
        "(or (and (eq (rank_val (at $0 2)) (+ (rank_val (at $0 1)) 2)) "
        "(and (eq (rank_val (at $0 3)) (+ (rank_val (at $0 1)) 4)) (eq (rank_val (at $0 4)) (+ (rank_val (at $0 1)) (+ 5 1))))) "
        "(and (eq (rank_val (at $0 3)) (+ (rank_val (at $0 2)) 2)) "
        "(and (eq (rank_val (at $0 4)) (+ (rank_val (at $0 2)) 4)) (eq (rank_val (at $0 5)) (+ (rank_val (at $0 2)) (+ 5 1)))))))",

    # Three consecutive cards, when sorted by rank, form AP with step 1
    # For each window: any card is the minimum, and min+1 and min+2 are present
    # Scoping: outer λ $0=hand; any's λ $0=card_c, $1=hand
    "ap_step1_len3_adj":
        "(λ or "
        # Window 0,1,2: any card in window has +1 and +2 in window
        "(any (λ and (any (λ eq (rank_val $0) (+ (rank_val $1) 1)) (take 3 $1)) "
        "(any (λ eq (rank_val $0) (+ (rank_val $1) 2)) (take 3 $1))) (take 3 $0)) "
        # Window 1,2,3
        "(or (any (λ and (any (λ eq (rank_val $0) (+ (rank_val $1) 1)) (take 3 (drop 1 $1))) "
        "(any (λ eq (rank_val $0) (+ (rank_val $1) 2)) (take 3 (drop 1 $1)))) (take 3 (drop 1 $0))) "
        # Window 2,3,4
        "(or (any (λ and (any (λ eq (rank_val $0) (+ (rank_val $1) 1)) (take 3 (drop 2 $1))) "
        "(any (λ eq (rank_val $0) (+ (rank_val $1) 2)) (take 3 (drop 2 $1)))) (take 3 (drop 2 $0))) "
        # Window 3,4,5
        "(any (λ and (any (λ eq (rank_val $0) (+ (rank_val $1) 1)) (drop 3 $1)) "
        "(any (λ eq (rank_val $0) (+ (rank_val $1) 2)) (drop 3 $1))) (drop 3 $0)))))",

    # Four consecutive cards, when sorted by rank, form AP with step 2
    # For each window: any card is the minimum, and min+2, min+4, min+6 present
    "ap_step2_len4_adj":
        "(λ or "
        # Window 0,1,2,3
        "(any (λ and (any (λ eq (rank_val $0) (+ (rank_val $1) 2)) (take 4 $1)) "
        "(and (any (λ eq (rank_val $0) (+ (rank_val $1) 4)) (take 4 $1)) "
        "(any (λ eq (rank_val $0) (+ (rank_val $1) (+ 5 1))) (take 4 $1)))) (take 4 $0)) "
        # Window 1,2,3,4
        "(or (any (λ and (any (λ eq (rank_val $0) (+ (rank_val $1) 2)) (take 4 (drop 1 $1))) "
        "(and (any (λ eq (rank_val $0) (+ (rank_val $1) 4)) (take 4 (drop 1 $1))) "
        "(any (λ eq (rank_val $0) (+ (rank_val $1) (+ 5 1))) (take 4 (drop 1 $1))))) (take 4 (drop 1 $0))) "
        # Window 2,3,4,5
        "(any (λ and (any (λ eq (rank_val $0) (+ (rank_val $1) 2)) (drop 2 $1)) "
        "(and (any (λ eq (rank_val $0) (+ (rank_val $1) 4)) (drop 2 $1)) "
        "(any (λ eq (rank_val $0) (+ (rank_val $1) (+ 5 1))) (drop 2 $1)))) (drop 2 $0))))",

    # Contains 5 cards with consecutive rank values (any suits)
    # = some card has rank+1, rank+2, rank+3, rank+4 all present
    "straight5":
        "(λ any (λ and (any (λ eq (rank_val $0) (+ (rank_val $1) 1)) $1) "
        "(and (any (λ eq (rank_val $0) (+ (rank_val $1) 2)) $1) "
        "(and (any (λ eq (rank_val $0) (+ (rank_val $1) 3)) $1) "
        "(any (λ eq (rank_val $0) (+ (rank_val $1) 4)) $1)))) $0)",

    # 5-card straight flush (consecutive ranks AND same suit)
    # Scoping: outer λ $0=hand; any's pred λ $0=card_c, $1=hand;
    #   inner any's pred λ $0=card_d, $1=card_c; inner any's list = $1 (hand)
    "straight5_same_suit":
        "(λ any (λ and "
        "(any (λ and (eq (get_suit $0) (get_suit $1)) (eq (rank_val $0) (+ (rank_val $1) 1))) $1) "
        "(and (any (λ and (eq (get_suit $0) (get_suit $1)) (eq (rank_val $0) (+ (rank_val $1) 2))) $1) "
        "(and (any (λ and (eq (get_suit $0) (get_suit $1)) (eq (rank_val $0) (+ (rank_val $1) 3))) $1) "
        "(any (λ and (eq (get_suit $0) (get_suit $1)) (eq (rank_val $0) (+ (rank_val $1) 4))) $1)))) $0)",

    # 5-card straight same color
    "straight5_same_color":
        "(λ any (λ and "
        "(any (λ and (eq (get_color $0) (get_color $1)) (eq (rank_val $0) (+ (rank_val $1) 1))) $1) "
        "(and (any (λ and (eq (get_color $0) (get_color $1)) (eq (rank_val $0) (+ (rank_val $1) 2))) $1) "
        "(and (any (λ and (eq (get_color $0) (get_color $1)) (eq (rank_val $0) (+ (rank_val $1) 3))) $1) "
        "(any (λ and (eq (get_color $0) (get_color $1)) (eq (rank_val $0) (+ (rank_val $1) 4))) $1)))) $0)",

    # Rank values form a palindrome
    "ranks_palindrome":
        "(λ and (eq (rank_val (at $0 0)) (rank_val (at $0 5))) "
        "(and (eq (rank_val (at $0 1)) (rank_val (at $0 4))) "
        "(eq (rank_val (at $0 2)) (rank_val (at $0 3)))))",

    # Every card at distance 2 shares rank or suit
    "skip2_same_rank_or_suit":
        "(λ and (or (eq (get_rank (at $0 0)) (get_rank (at $0 2))) (eq (get_suit (at $0 0)) (get_suit (at $0 2)))) "
        "(and (or (eq (get_rank (at $0 1)) (get_rank (at $0 3))) (eq (get_suit (at $0 1)) (get_suit (at $0 3)))) "
        "(and (or (eq (get_rank (at $0 2)) (get_rank (at $0 4))) (eq (get_suit (at $0 2)) (get_suit (at $0 4)))) "
        "(or (eq (get_rank (at $0 3)) (get_rank (at $0 5))) (eq (get_suit (at $0 3)) (get_suit (at $0 5)))))))",

    # No two adjacent cards share the same suit
    "no_adjacent_same_suit":
        "(λ all (λ not (eq (get_suit (head $0)) (get_suit (last $0)))) (adjacent_pairs $0))",

    # Ranks increase outward from center: max(center) < min(middle) < min(outer)
    # outer = (0,5), middle = (1,4), inner = (2,3)
    "radial_increasing":
        "(λ and (lt (if (ge (rank_val (at $0 0)) (rank_val (at $0 5))) (rank_val (at $0 0)) (rank_val (at $0 5))) "
        "(if (le (rank_val (at $0 1)) (rank_val (at $0 4))) (rank_val (at $0 1)) (rank_val (at $0 4)))) "
        "(lt (if (ge (rank_val (at $0 1)) (rank_val (at $0 4))) (rank_val (at $0 1)) (rank_val (at $0 4))) "
        "(if (le (rank_val (at $0 2)) (rank_val (at $0 3))) (rank_val (at $0 2)) (rank_val (at $0 3)))))",

    # Zigzag: v[1]>v[0], v[1]>v[2], v[2]<v[3], v[3]>v[4], v[4]<v[5]
    "zigzag_ranks":
        "(λ and (gt (rank_val (at $0 1)) (rank_val (at $0 0))) "
        "(and (gt (rank_val (at $0 1)) (rank_val (at $0 2))) "
        "(and (lt (rank_val (at $0 2)) (rank_val (at $0 3))) "
        "(and (gt (rank_val (at $0 3)) (rank_val (at $0 4))) "
        "(lt (rank_val (at $0 4)) (rank_val (at $0 5)))))))",

    # Suits form non-crossing nested brackets (= properly nested Dyck word)
    # S opens A, C closes A, H opens B, D closes B
    # NOTE: suit_brackets_no_cross and suit_brackets_nested are IDENTICAL
    # (verified exhaustively on all 4^6 = 4096 suit patterns). Both use
    # stack-based matching. Stack discipline cannot be fully expressed with
    # running_sum alone (would need a stack primitive). We use the interleaved
    # (independent counter) approximation, which is a superset (70 vs 40
    # valid patterns out of 4096). This means our DSL program accepts some
    # hands the true rule rejects, giving it a slightly worse likelihood.
    "suit_brackets_no_cross":
        "(λ and "
        "(and (all (λ ge $0 0) (running_sum (λ if (eq (get_suit $0) SPADES) 1 (if (eq (get_suit $0) CLUBS) (- 0 1) 0)) $0)) "
        "(eq (last (running_sum (λ if (eq (get_suit $0) SPADES) 1 (if (eq (get_suit $0) CLUBS) (- 0 1) 0)) $0)) 0)) "
        "(and (all (λ ge $0 0) (running_sum (λ if (eq (get_suit $0) HEARTS) 1 (if (eq (get_suit $0) DIAMONDS) (- 0 1) 0)) $0)) "
        "(eq (last (running_sum (λ if (eq (get_suit $0) HEARTS) 1 (if (eq (get_suit $0) DIAMONDS) (- 0 1) 0)) $0)) 0)))",

    # Suits form properly nested brackets (Dyck word) — identical to no_cross
    # Same DSL as no_cross (see note above about stack discipline limitation)
    "suit_brackets_nested":
        "(λ and "
        "(and (all (λ ge $0 0) (running_sum (λ if (eq (get_suit $0) SPADES) 1 (if (eq (get_suit $0) CLUBS) (- 0 1) 0)) $0)) "
        "(eq (last (running_sum (λ if (eq (get_suit $0) SPADES) 1 (if (eq (get_suit $0) CLUBS) (- 0 1) 0)) $0)) 0)) "
        "(and (all (λ ge $0 0) (running_sum (λ if (eq (get_suit $0) HEARTS) 1 (if (eq (get_suit $0) DIAMONDS) (- 0 1) 0)) $0)) "
        "(eq (last (running_sum (λ if (eq (get_suit $0) HEARTS) 1 (if (eq (get_suit $0) DIAMONDS) (- 0 1) 0)) $0)) 0)))",

    # Two bracket types tracked independently (counters, not stack)
    # Balanced: running_sum >= 0 and ends at 0 for both types (NO non-crossing check)
    "suit_brackets_interleaved":
        "(λ and "
        "(and (all (λ ge $0 0) (running_sum (λ if (eq (get_suit $0) SPADES) 1 (if (eq (get_suit $0) CLUBS) (- 0 1) 0)) $0)) "
        "(eq (last (running_sum (λ if (eq (get_suit $0) SPADES) 1 (if (eq (get_suit $0) CLUBS) (- 0 1) 0)) $0)) 0)) "
        "(and (all (λ ge $0 0) (running_sum (λ if (eq (get_suit $0) HEARTS) 1 (if (eq (get_suit $0) DIAMONDS) (- 0 1) 0)) $0)) "
        "(eq (last (running_sum (λ if (eq (get_suit $0) HEARTS) 1 (if (eq (get_suit $0) DIAMONDS) (- 0 1) 0)) $0)) 0)))",

    # Suits follow D>=S>=C>=H from left to right, stepping down at most one level
    # D=4, S=3, C=2, H=1. Adjacent pairs: next <= curr AND next >= curr-1
    "suits_nonincreasing":
        "(λ all (λ and (le (suit_to_int (get_suit (last $0))) (suit_to_int (get_suit (head $0)))) "
        "(ge (suit_to_int (get_suit (last $0))) (- (suit_to_int (get_suit (head $0))) 1))) "
        "(adjacent_pairs $0))",
}


# =========================================================================
# Main logic
# =========================================================================

def find_matching_llm_hypothesis(
    rule_id: str,
    true_pred,
    hypotheses: list,
    prim_dict: dict,
    test_hands: List[Hand],
    exemplar_hands: List[Hand],
) -> Optional[dict]:
    """
    Check if any existing LLM hypothesis matches the true rule on all test hands
    AND all exemplar hands.

    To avoid false positives from rules that are almost never True on random
    hands (producing all-False fingerprints), we require that:
    1. The fingerprints match exactly on all test hands
    2. The true rule has at least 3 True results on the test hands
    3. The hypothesis also returns True on ALL exemplar hands

    If the true rule is too rare (< 3 True), we skip LLM matching entirely
    and always use the hand-written DSL instead.
    """
    true_fp = tuple(true_pred(h) for h in test_hands)
    n_true = sum(true_fp)

    if n_true < 3:
        return None

    for hyp in hypotheses:
        if hyp["source"] != "llm_foil":
            continue
        try:
            pred = make_evaluator(hyp["dsl_program"], prim_dict)
            hyp_fp = tuple(pred(h) for h in test_hands)
            if hyp_fp != true_fp:
                continue
            # Also verify on exemplar hands (all must return True)
            if not all(pred(h) for h in exemplar_hands):
                continue
            return hyp
        except Exception:
            continue
    return None


def verify_on_exemplars(
    rule_id: str,
    dsl_program: str,
    prim_dict: dict,
    exemplars: dict,
) -> bool:
    """Verify that DSL program returns True on all exemplar hands for the rule."""
    if rule_id not in exemplars:
        print(f"  WARNING: No exemplars for {rule_id}")
        return True

    rule_exemplars = exemplars[rule_id]
    pred = make_evaluator(dsl_program, prim_dict)

    for label in ["hands_primary", "hands_reserve"]:
        for i, hand in enumerate(rule_exemplars.get(label, [])):
            result = pred(hand)
            if not result:
                print(f"  FAIL: {rule_id} {label}[{i}] returned {result}")
                return False
    return True


def verify_on_random_hands(
    rule_id: str,
    dsl_program: str,
    true_pred,
    prim_dict: dict,
    test_hands: List[Hand],
) -> Tuple[bool, int]:
    """Verify DSL program matches true rule on random hands."""
    pred = make_evaluator(dsl_program, prim_dict)
    mismatches = 0
    for hand in test_hands:
        try:
            dsl_result = pred(hand)
        except Exception:
            dsl_result = False
        true_result = true_pred(hand)
        if bool(dsl_result) != bool(true_result):
            mismatches += 1
    return mismatches == 0, mismatches


# Rules where the DSL is an APPROXIMATION of the true rule because the DSL
# lacks the expressiveness (e.g., stack operations) to represent it exactly.
# These rules use the interleaved (counter) DSL instead of the true stack-based
# semantics. The DSL accepts a superset of hands compared to the true rule.
APPROXIMATE_RULES = {
    "suit_brackets_no_cross",
    "suit_brackets_nested",
}


def main():
    parser = argparse.ArgumentParser(description="Write true-rule DSL programs")
    parser.add_argument("--save", action="store_true",
                        help="Save updated injected_hypotheses.json")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed output")
    args = parser.parse_args()

    print("=" * 70)
    print("TRUE RULE DSL WRITER")
    print("=" * 70)

    # Load existing data
    with open(DATA_PATH) as f:
        hypotheses = json.load(f)
    print(f"\nLoaded {len(hypotheses)} existing hypotheses")

    prim_dict = build_prim_dict()
    test_hands = random_hands(500)
    exemplars = load_exemplars()
    print(f"Generated {len(test_hands)} test hands")
    print(f"Loaded exemplars for {len(exemplars)} rules")

    already_covered = []
    newly_written = []
    failed = []

    print(f"\nChecking {len(GALLERY_RULES)} gallery rules...\n")

    for rule_id, rule_info in sorted(GALLERY_RULES.items()):
        true_pred = rule_info["predicate"]
        group = rule_info["group"]

        # Gather exemplar hands for this rule
        rule_exemplars = exemplars.get(rule_id, {})
        exemplar_hands = (
            rule_exemplars.get("hands_primary", []) +
            rule_exemplars.get("hands_reserve", [])
        )

        # Step 1: Check if any existing LLM hypothesis matches
        matching_hyp = find_matching_llm_hypothesis(
            rule_id, true_pred, hypotheses, prim_dict, test_hands, exemplar_hands
        )

        if matching_hyp:
            matching_hyp["true_for_rule"] = rule_id
            already_covered.append(rule_id)
            if args.verbose:
                print(f"  [LLM MATCH] {rule_id} (group {group}) <- {matching_hyp['id']}")
            continue

        # Step 2: Use hand-written DSL
        if rule_id not in TRUE_RULE_DSL:
            print(f"  [MISSING] {rule_id} (group {group}) - no hand-written DSL!")
            failed.append(rule_id)
            continue

        dsl_program = TRUE_RULE_DSL[rule_id]

        # Step 3: Verify the DSL program parses
        try:
            parse_program(dsl_program, prim_dict)
        except Exception as e:
            print(f"  [PARSE ERROR] {rule_id}: {e}")
            failed.append(rule_id)
            continue

        # Step 4: Verify on exemplar hands
        exemplar_ok = verify_on_exemplars(rule_id, dsl_program, prim_dict, exemplars)
        if not exemplar_ok:
            print(f"  [EXEMPLAR FAIL] {rule_id} (group {group})")
            failed.append(rule_id)
            continue

        # Step 5: Verify on random hands (skip for approximate rules)
        if rule_id in APPROXIMATE_RULES:
            if args.verbose:
                _, n_mismatch = verify_on_random_hands(
                    rule_id, dsl_program, true_pred, prim_dict, test_hands
                )
                print(f"  [APPROX] {rule_id} (group {group}) - {n_mismatch}/500 mismatches (expected)")
        else:
            match_ok, n_mismatch = verify_on_random_hands(
                rule_id, dsl_program, true_pred, prim_dict, test_hands
            )
            if not match_ok:
                print(f"  [MISMATCH] {rule_id} (group {group}) - {n_mismatch}/500 mismatches")
                failed.append(rule_id)
                continue

        # Step 6: Add to hypotheses
        source = "true_rule_approximate" if rule_id in APPROXIMATE_RULES else "true_rule"
        entry = {
            "id": f"true__{rule_id}",
            "source": source,
            "true_for_rule": rule_id,
            "dsl_program": dsl_program,
            "origin": {
                "hypothesis_text": rule_info["answer"],
                "source_model": "hand_written",
                "original_rule_id": rule_id,
            }
        }
        hypotheses.append(entry)
        newly_written.append(rule_id)
        if args.verbose:
            print(f"  [WRITTEN] {rule_id} (group {group})")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total gallery rules:    {len(GALLERY_RULES)}")
    print(f"  LLM matches found:      {len(already_covered)}")
    print(f"  Hand-written DSL added:  {len(newly_written)}")
    print(f"  Failed/missing:          {len(failed)}")
    print(f"  Total covered:           {len(already_covered) + len(newly_written)}/{len(GALLERY_RULES)}")

    if already_covered:
        print(f"\n  LLM matches: {', '.join(sorted(already_covered))}")
    if failed:
        print(f"\n  FAILED: {', '.join(sorted(failed))}")

    if args.save:
        with open(DATA_PATH, "w") as f:
            json.dump(hypotheses, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {len(hypotheses)} hypotheses to {DATA_PATH}")
    else:
        print(f"\nDry run - use --save to write {len(hypotheses)} hypotheses")


if __name__ == "__main__":
    main()
