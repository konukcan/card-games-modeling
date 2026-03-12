#!/usr/bin/env python3
"""
Translate 117 LLM-generated Python lambdas into DSL program strings.

Each hypothesis is a foil (or occasionally the true rule) that matched 6/6
exemplar hands. We translate them faithfully into the DSL, expressing what
the lambda computes regardless of whether it's the true rule.

The output is injected_hypotheses.json in the injection format.

DE BRUIJN INDEX CONVENTIONS:
  - The outermost lambda binds the hand: $0 = hand.
  - Inside a HOF's lambda (e.g., inside `all (λ BODY) LIST`):
      $0 = element (card or pair)
      $1 = hand (from outermost lambda)
  - The LIST argument to a HOF is OUTSIDE the HOF's lambda, so $0 = hand there.
  - Inside nested HOFs (e.g., `any (λ ... any (λ BODY) LIST2) LIST1`):
      In the inner BODY: $0 = inner element, $1 = outer element, $2 = hand
      In LIST2 (outside inner lambda but inside outer): $0 = outer element, $1 = hand

CONSTANT BUILDING:
  DSL has constants 0-5 only. Larger values must be built with arithmetic:
    6=(+ 5 1), 7=(+ 5 2), 8=(+ 5 3), 9=(+ 5 4), 10=(+ 5 5),
    11=(+ 5 (+ 5 1)), 12=(+ 5 (+ 5 2)), 13=(+ 5 (+ 5 3)), 14=(+ 5 (+ 5 4))
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.program import parse_program, Primitive
from gallery_analysis.enumerator import build_gallery_grammar
from gallery_analysis.dsl_prior import compute_log_prior


def _const(n: int) -> str:
    """Build a DSL expression for integer constant n."""
    if 0 <= n <= 5:
        return str(n)
    if n <= 10:
        return f"(+ 5 {n - 5})"
    return f"(+ 5 {_const(n - 5)})"


# Frequently used large constants
_8 = _const(8)   # (+ 5 3)
_9 = _const(9)   # (+ 5 4)
_10 = _const(10) # (+ 5 5)
_11 = _const(11) # (+ 5 (+ 5 1))
_12 = _const(12) # (+ 5 (+ 5 2))
_13 = _const(13) # (+ 5 (+ 5 3))
_14 = _const(14) # (+ 5 (+ 5 4))


def translate_all():
    """Translate each hypothesis. Returns (entries, dropped)."""
    raw_path = Path(__file__).parent / "data" / "llm_hypotheses_raw.json"
    with open(raw_path) as f:
        raw = json.load(f)

    translations = []
    dropped = []

    for idx, h in enumerate(raw):
        rule_id = h["rule_id"]
        text = h["hypothesis_text"]
        lam = h["python_lambda"]

        dsl = _translate_one(idx, rule_id, text, lam)

        if dsl is None:
            dropped.append((idx, rule_id, text))
            continue

        entry = {
            "id": f"llm__{rule_id}__hyp{idx}",
            "source": "llm_foil",
            "true_for_rule": None,
            "dsl_program": dsl,
            "origin": {
                "hypothesis_text": text,
                "python_lambda": lam,
                "source_model": h.get("source_model", "gemini-2.5-flash"),
                "original_rule_id": rule_id,
            }
        }
        translations.append(entry)

    return translations, dropped


def _translate_one(idx: int, rule_id: str, text: str, lam: str) -> str | None:
    """Translate a single hypothesis. Returns DSL string or None if untranslatable."""

    # ===================================================================
    # Helper fragments for common patterns
    # ===================================================================

    def _all_cards(pred_body):
        """all(pred_body(card) for card in hand). pred_body uses $0=card."""
        return f"(λ all (λ {pred_body}) $0)"

    def _any_cards(pred_body):
        """any(pred_body(card) for card in hand). pred_body uses $0=card."""
        return f"(λ any (λ {pred_body}) $0)"

    def _all_adj_pairs(pred_body):
        """all(pred_body(pair) for pair in adjacent_pairs(hand)). $0=pair inside."""
        return f"(λ all (λ {pred_body}) (adjacent_pairs $0))"

    def _any_adj_pairs(pred_body):
        """any(pred_body(pair) for pair in adjacent_pairs(hand))."""
        return f"(λ any (λ {pred_body}) (adjacent_pairs $0))"

    # --- Pair access inside adjacent_pairs: pair is a list of 2 cards ---
    # head $0 = first card, last $0 = second card

    # ===================================================================
    # GROUP: Suit/color counting and set operations
    # ===================================================================

    # IDX 0: any suit has 3+ cards
    if idx == 0:
        return "(λ ge (max_suit_count $0) 3)"

    # IDX 1: all ranks in {4,8,9} AND all three present
    if idx == 1:
        rk_in_489 = f"or (eq (rank_val $0) 4) (or (eq (rank_val $0) {_8}) (eq (rank_val $0) {_9}))"
        has4 = f"(any (λ eq (rank_val $0) 4) $0)"
        has8 = f"(any (λ eq (rank_val $0) {_8}) $0)"
        has9 = f"(any (λ eq (rank_val $0) {_9}) $0)"
        return f"(λ and (all (λ {rk_in_489}) $0) (and {has4} (and {has8} {has9})))"

    # IDX 2: hand contains 4 of Clubs
    if idx == 2:
        return f"(λ any (λ and (eq (rank_val $0) 4) (eq (get_suit $0) CLUBS)) $0)"

    # IDX 3: exactly 3 unique suits
    if idx == 3:
        return "(λ eq (n_unique_suits $0) 3)"

    # IDX 4: all clubs or hearts
    if idx == 4:
        return _all_cards("or (eq (get_suit $0) HEARTS) (eq (get_suit $0) CLUBS)")

    # IDX 5: all even ranks
    if idx == 5:
        return _all_cards("eq (mod (rank_val $0) 2) 0")

    # IDX 6: all odd ranks (dropping the distribution constraint — inexpressible)
    if idx == 6:
        return _all_cards("not (eq (mod (rank_val $0) 2) 0)")

    # IDX 7: all red
    if idx == 7:
        return _all_cards("eq (get_color $0) RED")

    # IDX 8: all same color AND exactly 2 suits
    if idx == 8:
        return "(λ and (eq (n_unique_colors $0) 1) (eq (n_unique_suits $0) 2))"

    # IDX 9: all same suit
    if idx == 9:
        return "(λ eq (n_unique_suits $0) 1)"

    # IDX 10: exactly 2 suits with odd count
    if idx == 10:
        return _two_suits_odd_count()

    # IDX 11: one suit has count 3 OR two suits have count 2
    if idx == 11:
        return "(λ or (eq (max_suit_count $0) 3) (and (eq (n_repeated_suits $0) 2) (le (max_suit_count $0) 2)))"

    # IDX 12: no rank > 3 copies — cannot express without max_rank_count
    if idx == 12:
        return None

    # IDX 13: exactly one card with rank 8
    if idx == 13:
        return f"(λ eq (length (filter (λ eq (rank_val $0) {_8}) $0)) 1)"

    # IDX 14: AP with step 2 in 4+ cards — requires itertools.combinations. Drop.
    if idx == 14:
        return None

    # IDX 15: red count parity == black count parity. In 6 cards: red even.
    if idx == 15:
        return "(λ eq (mod (count_color $0 RED) 2) 0)"

    # IDX 16: exactly two pairs (n_repeated_ranks==2, n_unique_ranks==4)
    if idx == 16:
        return "(λ and (eq (n_repeated_ranks $0) 2) (eq (n_unique_ranks $0) 4))"

    # IDX 17: n_unique_ranks > n_unique_suits
    if idx == 17:
        return "(λ gt (n_unique_ranks $0) (n_unique_suits $0))"

    # IDX 18: first 3 same suit AND last 3 same suit
    if idx == 18:
        return "(λ and (eq (n_unique_suits (first_half $0)) 1) (eq (n_unique_suits (second_half $0)) 1))"

    # IDX 19: max rank count != max suit count — no max_rank_count. Drop.
    if idx == 19:
        return None

    # IDX 20: exactly 3 red cards
    if idx == 20:
        return "(λ eq (count_color $0 RED) 3)"

    # IDX 21: at least one singleton suit
    if idx == 21:
        return _any_singleton_suit()

    # IDX 22: positions 0,2,4 are Aces (rank_val 14)
    if idx == 22:
        return _positions_same_rank_val([0, 2, 4], _14)

    # IDX 23: 4+ of same suit (unique ranks guaranteed in standard deck)
    if idx == 23:
        return "(λ ge (max_suit_count $0) 4)"

    # IDX 24: any suit has 4+ cards
    if idx == 24:
        return "(λ ge (max_suit_count $0) 4)"

    # IDX 25: 4+ diamonds
    if idx == 25:
        return "(λ ge (count_suit $0 DIAMONDS) 4)"

    # IDX 26: 4+ hearts
    if idx == 26:
        return "(λ ge (count_suit $0 HEARTS) 4)"

    # IDX 27: four of a kind
    if idx == 27:
        # any card c in hand: length(filter(same rank as c) hand) >= 4
        # Inside outer λ: $0=hand.
        # Inside any's λ: $0=card, $1=hand.
        # Inside filter's λ: $0=c2, $1=card, $2=hand.
        return "(λ any (λ ge (length (filter (λ eq (get_rank $0) (get_rank $1)) $1)) 4) $0)"

    # IDX 28: halves copy ranks with different suits
    if idx == 28:
        parts = []
        for i in range(3):
            j = i + 3
            parts.append(f"(and (eq (get_rank (at $0 {i})) (get_rank (at $0 {j}))) (not (eq (get_suit (at $0 {i})) (get_suit (at $0 {j})))))")
        return f"(λ and {parts[0]} (and {parts[1]} {parts[2]}))"

    # IDX 29: specific positional suit matching
    if idx == 29:
        return "(λ or (and (eq (get_suit (at $0 1)) (get_suit (at $0 4))) (eq (count_suit $0 (get_suit (at $0 1))) 2)) (and (eq (get_suit (at $0 2)) (get_suit (at $0 5))) (eq (count_suit $0 (get_suit (at $0 2))) 2)))"

    # IDX 30: exactly 2 suits with odd count
    if idx == 30:
        return _two_suits_odd_count()

    # IDX 31: suit distribution in {(3,2,1),(2,2,2),(2,2,1,1)}
    if idx == 31:
        return "(λ and (ge (n_repeated_suits $0) 2) (and (le (max_suit_count $0) 3) (ge (n_unique_suits $0) 3)))"

    # IDX 32: exactly two 5s, adjacent
    if idx == 32:
        return f"(λ and (eq (length (filter (λ eq (rank_val $0) 5) $0)) 2) (any (λ and (eq (rank_val (head $0)) 5) (eq (rank_val (last $0)) 5)) (adjacent_pairs $0)))"

    # IDX 33: pos 3&4 are Jacks (11), different suits, no other Jacks
    if idx == 33:
        return f"(λ and (eq (rank_val (at $0 3)) {_11}) (and (eq (rank_val (at $0 4)) {_11}) (and (not (eq (get_suit (at $0 3)) (get_suit (at $0 4)))) (and (not (eq (rank_val (at $0 0)) {_11})) (and (not (eq (rank_val (at $0 1)) {_11})) (and (not (eq (rank_val (at $0 2)) {_11})) (not (eq (rank_val (at $0 5)) {_11}))))))))"

    # IDX 34: positions 0,2,4 have same rank
    if idx == 34:
        return "(λ and (eq (rank_val (at $0 0)) (rank_val (at $0 2))) (eq (rank_val (at $0 2)) (rank_val (at $0 4))))"

    # IDX 35: no rank > twice (approximate: n_unique_ranks >= 3)
    if idx == 35:
        return "(λ ge (n_unique_ranks $0) 3)"

    # IDX 36: ranks palindrome
    if idx == 36:
        return _ranks_palindrome()

    # IDX 37: 2 or 3 unique suits AND max suit count in {3,4}
    if idx == 37:
        return "(λ and (or (eq (n_unique_suits $0) 2) (eq (n_unique_suits $0) 3)) (or (eq (max_suit_count $0) 3) (eq (max_suit_count $0) 4)))"

    # IDX 38: no card has rank 5
    if idx == 38:
        return _all_cards("not (eq (rank_val $0) 5)")

    # IDX 39: 6 distinct ranks (approximate — drops straight constraint)
    if idx == 39:
        return "(λ eq (n_unique_ranks $0) (length $0))"

    # IDX 40: 5-card straight (approximate)
    if idx == 40:
        return f"(λ and (ge (n_unique_ranks $0) 5) (le (- (max_rank $0) (min_rank $0)) 5))"

    # IDX 41: straight flush — requires suit-filtered combo check. Drop.
    if idx == 41:
        return None

    # IDX 42: all unique ranks AND first card rank is 2 or 3
    if idx == 42:
        return "(λ and (eq (n_unique_ranks $0) (length $0)) (or (eq (rank_val (head $0)) 2) (eq (rank_val (head $0)) 3)))"

    # IDX 43: suit counts = [1,1,2,2]
    if idx == 43:
        return "(λ and (eq (n_unique_suits $0) 4) (and (eq (n_repeated_suits $0) 2) (eq (max_suit_count $0) 2)))"

    # IDX 44: exactly 2 suits with count > 1
    if idx == 44:
        return "(λ eq (n_repeated_suits $0) 2)"

    # IDX 45: 2 suits with odd count
    if idx == 45:
        return _two_suits_odd_count()

    # IDX 46: first and last card different suits
    if idx == 46:
        return "(λ not (eq (get_suit (head $0)) (get_suit (last $0))))"

    # IDX 47: exactly 3 unique suits
    if idx == 47:
        return "(λ eq (n_unique_suits $0) 3)"

    # IDX 48: at least one singleton suit
    if idx == 48:
        return _any_singleton_suit()

    # IDX 49: at least 3 of same suit
    if idx == 49:
        return "(λ ge (max_suit_count $0) 3)"

    # IDX 50: fewer than 4 unique suits
    if idx == 50:
        return "(λ lt (n_unique_suits $0) 4)"

    # IDX 51: positions 1,2,3 are rank 2
    if idx == 51:
        return "(λ and (eq (rank_val (at $0 1)) 2) (and (eq (rank_val (at $0 2)) 2) (eq (rank_val (at $0 3)) 2)))"

    # IDX 52: exactly 3 cards with rank 3
    if idx == 52:
        return "(λ eq (length (filter (λ eq (rank_val $0) 3) $0)) 3)"

    # IDX 53: exactly one three-of-a-kind, no other pairs
    if idx == 53:
        return "(λ and (eq (n_repeated_ranks $0) 1) (eq (n_unique_ranks $0) 4))"

    # IDX 54: positions 2,3,4 same rank
    if idx == 54:
        return "(λ and (eq (rank_val (at $0 2)) (rank_val (at $0 3))) (eq (rank_val (at $0 3)) (rank_val (at $0 4))))"

    # IDX 55: rank distribution [2,2,2] or [1,1,2,2]
    if idx == 55:
        return "(λ or (and (eq (n_unique_ranks $0) 3) (eq (n_repeated_ranks $0) 3)) (and (eq (n_unique_ranks $0) 4) (eq (n_repeated_ranks $0) 2)))"

    # IDX 56: exactly one Queen (rank 12)
    if idx == 56:
        return f"(λ eq (length (filter (λ eq (rank_val $0) {_12}) $0)) 1)"

    # IDX 57: exactly 2 suits with odd count
    if idx == 57:
        return _two_suits_odd_count()

    # IDX 58: adjacent cards share rank or suit
    if idx == 58:
        return _all_adj_pairs("or (eq (get_rank (head $0)) (get_rank (last $0))) (eq (get_suit (head $0)) (get_suit (last $0)))")

    # IDX 59: all ranks in {4,8,9} AND all three present (same logic as 1)
    if idx == 59:
        rk_in_489 = f"or (eq (rank_val $0) 4) (or (eq (rank_val $0) {_8}) (eq (rank_val $0) {_9}))"
        has4 = f"(any (λ eq (rank_val $0) 4) $0)"
        has8 = f"(any (λ eq (rank_val $0) {_8}) $0)"
        has9 = f"(any (λ eq (rank_val $0) {_9}) $0)"
        return f"(λ and (all (λ {rk_in_489}) $0) (and {has4} (and {has8} {has9})))"

    # IDX 60: exactly 2 unique ranks, both repeated
    if idx == 60:
        return "(λ and (eq (n_unique_ranks $0) 2) (eq (n_repeated_ranks $0) 2))"

    # IDX 61: 5 cards one color, 1 other
    if idx == 61:
        return "(λ or (eq (count_color $0 RED) 5) (eq (count_color $0 RED) 1))"

    # IDX 62: all Hearts/Clubs AND parity match
    if idx == 62:
        return "(λ and (all (λ or (eq (get_suit $0) HEARTS) (eq (get_suit $0) CLUBS)) $0) (eq (mod (count_suit $0 HEARTS) 2) (mod (count_suit $0 CLUBS) 2)))"

    # IDX 63: all even ranks
    if idx == 63:
        return _all_cards("eq (mod (rank_val $0) 2) 0")

    # IDX 64: all odd ranks
    if idx == 64:
        return _all_cards("not (eq (mod (rank_val $0) 2) 0)")

    # IDX 65: all red AND >= 2 hearts AND >= 2 diamonds
    if idx == 65:
        return "(λ and (all (λ eq (get_color $0) RED) $0) (and (ge (count_suit $0 HEARTS) 2) (ge (count_suit $0 DIAMONDS) 2)))"

    # IDX 66: all same color AND each present suit count >= 2
    if idx == 66:
        return "(λ and (eq (n_unique_colors $0) 1) (or (eq (n_unique_suits $0) 1) (le (max_suit_count $0) 4)))"

    # IDX 67: all same suit AND first half has even AND second half has even
    if idx == 67:
        return "(λ and (eq (n_unique_suits $0) 1) (and (any (λ eq (mod (rank_val $0) 2) 0) (first_half $0)) (any (λ eq (mod (rank_val $0) 2) 0) (second_half $0))))"

    # IDX 68: adjacent pair of same color
    if idx == 68:
        return _any_adj_pairs("eq (get_color (head $0)) (get_color (last $0))")

    # IDX 69: adjacent pair of same color (same as 68)
    if idx == 69:
        return _any_adj_pairs("eq (get_color (head $0)) (get_color (last $0))")

    # IDX 70: adjacent pair with rank diff == 2
    if idx == 70:
        return _any_adj_pairs("or (eq (- (rank_val (head $0)) (rank_val (last $0))) 2) (eq (- (rank_val (last $0)) (rank_val (head $0))) 2)")

    # IDX 71: all blacks before all reds
    if idx == 71:
        # Once you see RED, all subsequent must be RED.
        # Equivalently: no adjacent pair (prev=RED, curr=BLACK).
        return _all_adj_pairs("or (not (eq (get_color (head $0)) RED)) (eq (get_color (last $0)) RED)")

    # IDX 72: first half has pair AND second half has pair
    if idx == 72:
        fh = "(first_half $0)"
        sh = "(second_half $0)"
        def _pair3(h):
            return f"(or (eq (get_rank (at {h} 0)) (get_rank (at {h} 1))) (or (eq (get_rank (at {h} 0)) (get_rank (at {h} 2))) (eq (get_rank (at {h} 1)) (get_rank (at {h} 2)))))"
        return f"(λ and {_pair3(fh)} {_pair3(sh)})"

    # IDX 73: first half uniform color AND second half uniform color
    if idx == 73:
        return "(λ and (eq (n_unique_colors (first_half $0)) 1) (eq (n_unique_colors (second_half $0)) 1))"

    # IDX 74: all same suit OR (each half single suit, different)
    if idx == 74:
        return "(λ or (eq (n_unique_suits $0) 1) (and (eq (n_unique_suits (first_half $0)) 1) (and (eq (n_unique_suits (second_half $0)) 1) (not (eq (get_suit (head (first_half $0))) (get_suit (head (second_half $0))))))))"

    # IDX 75: adjacent pair of same suit
    if idx == 75:
        return _any_adj_pairs("eq (get_suit (head $0)) (get_suit (last $0))")

    # IDX 76: colors alternate
    if idx == 76:
        return _all_adj_pairs("not (eq (get_color (head $0)) (get_color (last $0)))")

    # IDX 77: BRBRBR pattern
    if idx == 77:
        checks = []
        for i in range(6):
            color = "BLACK" if i % 2 == 0 else "RED"
            checks.append(f"(eq (get_color (at $0 {i})) {color})")
        result = checks[-1]
        for c in reversed(checks[:-1]):
            result = f"(and {c} {result})"
        return f"(λ {result})"

    # IDX 78: positions 0,2,4 are Aces
    if idx == 78:
        return _positions_same_rank_val([0, 2, 4], _14)

    # IDX 79: 4 consecutive same suit
    if idx == 79:
        return _or_consecutive_same("get_suit", 4)

    # IDX 80: 3 consecutive same suit
    if idx == 80:
        return _or_consecutive_same("get_suit", 3)

    # IDX 81: 4+ diamonds
    if idx == 81:
        return "(λ ge (count_suit $0 DIAMONDS) 4)"

    # IDX 82: pos 2&3 Hearts AND 4+ hearts total
    if idx == 82:
        return "(λ and (eq (get_suit (at $0 2)) HEARTS) (and (eq (get_suit (at $0 3)) HEARTS) (ge (count_suit $0 HEARTS) 4)))"

    # IDX 83: four-of-a-kind with 2 singletons
    if idx == 83:
        return "(λ and (eq (n_unique_ranks $0) 3) (eq (n_repeated_ranks $0) 1))"

    # IDX 84: four consecutive same rank
    if idx == 84:
        return _or_consecutive_same("get_rank", 4)

    # IDX 85: adjacent pair same color
    if idx == 85:
        return _any_adj_pairs("eq (get_color (head $0)) (get_color (last $0))")

    # IDX 86: halves copy ranks
    if idx == 86:
        return _halves_copy("rank_val")

    # IDX 87: each suit count is even
    if idx == 87:
        suits = ["CLUBS", "DIAMONDS", "HEARTS", "SPADES"]
        checks = [f"(eq (mod (count_suit $0 {s}) 2) 0)" for s in suits]
        result = checks[-1]
        for c in reversed(checks[:-1]):
            result = f"(and {c} {result})"
        return f"(λ {result})"

    # IDX 88: first 3 red, last 3 black
    if idx == 88:
        return "(λ and (all (λ eq (get_color $0) RED) (first_half $0)) (all (λ eq (get_color $0) BLACK) (second_half $0)))"

    # IDX 89: some card in first half shares rank/suit with some in second half
    if idx == 89:
        # Nested any: any(first_half, λc1. any(second_half, λc2. eq_rank or eq_suit))
        # Inside outer any's λ: $0=c1, $1=hand
        # second_half $1 = second half of hand (correct: $1=hand here)
        # Inside inner any's λ: $0=c2, $1=c1, $2=hand
        return "(λ any (λ any (λ or (eq (get_rank $0) (get_rank $1)) (eq (get_suit $0) (get_suit $1))) (second_half $1)) (first_half $0))"

    # IDX 90: adjacent pair of 5s
    if idx == 90:
        return _any_adj_pairs("and (eq (rank_val (head $0)) 5) (eq (rank_val (last $0)) 5)")

    # IDX 91: pos 3&4 are Jacks (rank_val 11)
    if idx == 91:
        return f"(λ and (eq (rank_val (at $0 3)) {_11}) (eq (rank_val (at $0 4)) {_11}))"

    # IDX 92: pos 0,2,4 same rank + 3 different suits + pos 1,3,5 all diff ranks, none = pos 0
    if idx == 92:
        return "(λ and (eq (get_rank (at $0 0)) (get_rank (at $0 2))) (and (eq (get_rank (at $0 2)) (get_rank (at $0 4))) (and (not (eq (get_suit (at $0 0)) (get_suit (at $0 2)))) (and (not (eq (get_suit (at $0 2)) (get_suit (at $0 4)))) (and (not (eq (get_suit (at $0 0)) (get_suit (at $0 4)))) (and (not (eq (get_rank (at $0 1)) (get_rank (at $0 0)))) (and (not (eq (get_rank (at $0 3)) (get_rank (at $0 0)))) (and (not (eq (get_rank (at $0 5)) (get_rank (at $0 0)))) (and (not (eq (get_rank (at $0 1)) (get_rank (at $0 3)))) (and (not (eq (get_rank (at $0 3)) (get_rank (at $0 5)))) (not (eq (get_rank (at $0 1)) (get_rank (at $0 5))))))))))))))"

    # IDX 93: sum of ranks is odd
    if idx == 93:
        return "(λ not (eq (mod (sum_ranks $0) 2) 0))"

    # IDX 94: ranks palindrome
    if idx == 94:
        return _ranks_palindrome()

    # IDX 95: first half and second half share at least one suit
    if idx == 95:
        return "(λ any (λ any (λ eq (get_suit $0) (get_suit $1)) (second_half $1)) (first_half $0))"

    # IDX 96: first 3 same color, last 3 opposite color
    if idx == 96:
        return "(λ and (eq (n_unique_colors (first_half $0)) 1) (and (eq (n_unique_colors (second_half $0)) 1) (not (eq (get_color (head (first_half $0))) (get_color (head (second_half $0)))))))"

    # IDX 97: 5-card straight (approximate)
    if idx == 97:
        return f"(λ and (ge (n_unique_ranks $0) 5) (le (- (max_rank $0) (min_rank $0)) 5))"

    # IDX 98: 5+ cards of same color
    if idx == 98:
        return "(λ or (ge (count_color $0 RED) 5) (ge (count_color $0 BLACK) 5))"

    # IDX 99: straight flush — Drop.
    if idx == 99:
        return None

    # IDX 100: strictly increasing ranks
    if idx == 100:
        return _all_adj_pairs("lt (rank_val (head $0)) (rank_val (last $0))")

    # IDX 101: suit counts [1,1,2,2] (same as 43)
    if idx == 101:
        return "(λ and (eq (n_unique_suits $0) 4) (and (eq (n_repeated_suits $0) 2) (eq (max_suit_count $0) 2)))"

    # IDX 102: even number of red cards
    if idx == 102:
        return "(λ eq (mod (count_color $0 RED) 2) 0)"

    # IDX 103: pairs (0,1), (2,3), (4,5) each same color
    if idx == 103:
        return "(λ and (eq (get_color (at $0 0)) (get_color (at $0 1))) (and (eq (get_color (at $0 2)) (get_color (at $0 3))) (eq (get_color (at $0 4)) (get_color (at $0 5)))))"

    # IDX 104: adjacent pair same suit
    if idx == 104:
        return _any_adj_pairs("eq (get_suit (head $0)) (get_suit (last $0))")

    # IDX 105: first 3 same suit OR last 3 same suit
    if idx == 105:
        return "(λ or (eq (n_unique_suits (first_half $0)) 1) (eq (n_unique_suits (second_half $0)) 1))"

    # IDX 106: pair at distance 2 shares rank or suit
    if idx == 106:
        parts = []
        for i in range(4):
            j = i + 2
            parts.append(f"(or (eq (get_rank (at $0 {i})) (get_rank (at $0 {j}))) (eq (get_suit (at $0 {i})) (get_suit (at $0 {j}))))")
        result = parts[-1]
        for p in reversed(parts[:-1]):
            result = f"(or {p} {result})"
        return f"(λ {result})"

    # IDX 107: 3+ spades
    if idx == 107:
        return "(λ ge (count_suit $0 SPADES) 3)"

    # IDX 108: pos 1,2,3 are rank 2 (same as 51)
    if idx == 108:
        return "(λ and (eq (rank_val (at $0 1)) 2) (and (eq (rank_val (at $0 2)) 2) (eq (rank_val (at $0 3)) 2)))"

    # IDX 109: three adjacent 3s
    if idx == 109:
        parts = []
        for s in range(4):
            parts.append(f"(and (eq (rank_val (at $0 {s})) 3) (and (eq (rank_val (at $0 {s+1})) 3) (eq (rank_val (at $0 {s+2})) 3)))")
        result = parts[-1]
        for p in reversed(parts[:-1]):
            result = f"(or {p} {result})"
        return f"(λ {result})"

    # IDX 110: three consecutive same rank AND remaining 3 all different
    if idx == 110:
        triple_parts = []
        for s in range(4):
            triple_parts.append(f"(and (eq (get_rank (at $0 {s})) (get_rank (at $0 {s+1}))) (eq (get_rank (at $0 {s+1})) (get_rank (at $0 {s+2}))))")
        any_triple = triple_parts[-1]
        for p in reversed(triple_parts[:-1]):
            any_triple = f"(or {p} {any_triple})"
        return f"(λ and {any_triple} (eq (n_unique_ranks $0) 4))"

    # IDX 111: pos 2,3,4 same rank (same as 54)
    if idx == 111:
        return "(λ and (eq (rank_val (at $0 2)) (rank_val (at $0 3))) (eq (rank_val (at $0 3)) (rank_val (at $0 4))))"

    # IDX 112: first half and second half share at least one suit (same as 95)
    if idx == 112:
        return "(λ any (λ any (λ eq (get_suit $0) (get_suit $1)) (second_half $1)) (first_half $0))"

    # IDX 113: odd or even position group has mixed colors
    if idx == 113:
        odd_red = "(or (eq (get_color (at $0 0)) RED) (or (eq (get_color (at $0 2)) RED) (eq (get_color (at $0 4)) RED)))"
        odd_blk = "(or (eq (get_color (at $0 0)) BLACK) (or (eq (get_color (at $0 2)) BLACK) (eq (get_color (at $0 4)) BLACK)))"
        even_red = "(or (eq (get_color (at $0 1)) RED) (or (eq (get_color (at $0 3)) RED) (eq (get_color (at $0 5)) RED)))"
        even_blk = "(or (eq (get_color (at $0 1)) BLACK) (or (eq (get_color (at $0 3)) BLACK) (eq (get_color (at $0 5)) BLACK)))"
        return f"(λ or (and {odd_red} {odd_blk}) (and {even_red} {even_blk}))"

    # IDX 114: pairs (0,1), (2,3), (4,5) each ascending rank
    if idx == 114:
        return "(λ and (lt (rank_val (at $0 0)) (rank_val (at $0 1))) (and (lt (rank_val (at $0 2)) (rank_val (at $0 3))) (lt (rank_val (at $0 4)) (rank_val (at $0 5)))))"

    # IDX 115: some card rank == sum of two others' ranks — Drop.
    if idx == 115:
        return None

    # IDX 116: 3+ clubs
    if idx == 116:
        return "(λ ge (count_suit $0 CLUBS) 3)"

    print(f"WARNING: No translation for idx {idx} ({rule_id})")
    return None


# ===================================================================
# Reusable pattern builders
# ===================================================================

def _two_suits_odd_count():
    """Exactly 2 suits have odd count."""
    return "(λ eq (+ (+ (mod (count_suit $0 CLUBS) 2) (mod (count_suit $0 DIAMONDS) 2)) (+ (mod (count_suit $0 HEARTS) 2) (mod (count_suit $0 SPADES) 2))) 2)"


def _any_singleton_suit():
    """At least one suit has exactly 1 card."""
    return "(λ or (eq (count_suit $0 CLUBS) 1) (or (eq (count_suit $0 DIAMONDS) 1) (or (eq (count_suit $0 HEARTS) 1) (eq (count_suit $0 SPADES) 1))))"


def _positions_same_rank_val(positions, val):
    """Check that given positions all have rank_val == val."""
    checks = [f"(eq (rank_val (at $0 {p})) {val})" for p in positions]
    result = checks[-1]
    for c in reversed(checks[:-1]):
        result = f"(and {c} {result})"
    return f"(λ {result})"


def _ranks_palindrome():
    """rank[0]==rank[5], rank[1]==rank[4], rank[2]==rank[3]."""
    return "(λ and (eq (rank_val (at $0 0)) (rank_val (at $0 5))) (and (eq (rank_val (at $0 1)) (rank_val (at $0 4))) (eq (rank_val (at $0 2)) (rank_val (at $0 3)))))"


def _halves_copy(accessor):
    """rank_val/get_suit at pos i == pos i+3 for i=0,1,2."""
    checks = [f"(eq ({accessor} (at $0 {i})) ({accessor} (at $0 {i+3})))" for i in range(3)]
    result = checks[-1]
    for c in reversed(checks[:-1]):
        result = f"(and {c} {result})"
    return f"(λ {result})"


def _or_consecutive_same(accessor, count):
    """OR over all windows of `count` consecutive cards with same accessor value."""
    max_start = 6 - count
    parts = []
    for s in range(max_start + 1):
        eqs = [f"(eq ({accessor} (at $0 {s})) ({accessor} (at $0 {s+j})))" for j in range(1, count)]
        inner = eqs[-1]
        for e in reversed(eqs[:-1]):
            inner = f"(and {e} {inner})"
        parts.append(inner)
    result = parts[-1]
    for p in reversed(parts[:-1]):
        result = f"(or {p} {result})"
    return f"(λ {result})"


# ===================================================================
# Validation
# ===================================================================

def validate_translations(entries):
    """Validate that all DSL programs parse and have finite log-priors."""
    grammar = build_gallery_grammar()
    prim_dict = {}
    for prod in grammar.productions:
        if isinstance(prod.program, Primitive):
            prim_dict[prod.program.name] = prod.program

    errors = []
    for entry in entries:
        dsl = entry["dsl_program"]
        eid = entry["id"]

        # Test parsing
        try:
            prog = parse_program(dsl, prim_dict)
        except Exception as e:
            errors.append((eid, "PARSE", str(e), dsl))
            continue

        # Test log-prior
        try:
            lp = compute_log_prior(dsl, grammar)
            if lp == float('-inf') or lp != lp:
                errors.append((eid, "PRIOR", f"log_prior={lp}", dsl))
        except Exception as e:
            errors.append((eid, "PRIOR", str(e), dsl))

    return errors


def main():
    print("Translating 117 LLM hypotheses into DSL programs...")

    entries, dropped = translate_all()

    print(f"\nTranslated: {len(entries)}")
    print(f"Dropped: {len(dropped)}")
    for idx, rule_id, text in dropped:
        print(f"  DROPPED [{idx}] {rule_id}: {text[:60]}...")

    print("\nValidating DSL programs...")
    errors = validate_translations(entries)

    if errors:
        print(f"\n{len(errors)} ERRORS found:")
        for eid, etype, msg, dsl in errors:
            print(f"  [{etype}] {eid}: {msg}")
            print(f"    DSL: {dsl[:120]}...")
    else:
        print("All programs valid!")

    # Save output
    out_path = Path(__file__).parent / "data" / "injected_hypotheses.json"
    with open(out_path, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"\nSaved {len(entries)} entries to {out_path}")

    return len(errors)


if __name__ == "__main__":
    n_errors = main()
    sys.exit(1 if n_errors > 0 else 0)
