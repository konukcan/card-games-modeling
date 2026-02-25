#!/usr/bin/env python3
"""
Phase 2b: Template-Based Rule Generation

Extracts structural templates from the 8+ existing rule families,
identifies variable slots, and enumerates all valid filler combinations
to generate new candidate rules.

Each template is a higher-order function with "slots" that accept
different property extractors, comparison operators, or predicates.
By systematically filling these slots, we generate new rules that
are structurally similar to existing ones but test different features.

Usage:
    cd card-games-modelling/src
    python3 template_generation.py [--samples 50000] [--hand-size 6]
"""

import sys
import csv
import math
import random
import argparse
from pathlib import Path
from typing import Callable, List, Dict, Any, Tuple, Optional, Set

sys.path.insert(0, str(Path(__file__).parent))

from rules.cards import (
    Card, Hand, Suit, Rank, Color, RANK_VALUES,
    card_color, sample_hand, suit_to_altcolor1, suit_to_altcolor2, rank_parity
)

# ============================================================================
# Property extractors — the main "slot fillers"
# ============================================================================
# These are functions Card → Value that we plug into templates.
# Each returns a hashable value so we can compare, palindrome-check, etc.

PROPERTY_EXTRACTORS = {
    "suit":     lambda c: c.suit,
    "rank":     lambda c: c.rank,
    "rank_val": lambda c: RANK_VALUES[c.rank],
    "color":    lambda c: card_color(c),
    "parity":   lambda c: RANK_VALUES[c.rank] % 2,  # 0=even, 1=odd
    "face":     lambda c: c.rank in (Rank.JACK, Rank.QUEEN, Rank.KING),
    "high":     lambda c: RANK_VALUES[c.rank] >= 10,  # 10, J, Q, K, A
    "mod3":     lambda c: RANK_VALUES[c.rank] % 3,
}

# Boolean properties of a sub-hand (List[Card] → bool)
BOOLEAN_PROPERTIES = {
    "uniform_color": lambda h: len(set(card_color(c) for c in h)) <= 1 if h else True,
    "uniform_suit": lambda h: len(set(c.suit for c in h)) <= 1 if h else True,
    "uniform_parity": lambda h: len(set(RANK_VALUES[c.rank] % 2 for c in h)) <= 1 if h else True,
    "has_face": lambda h: any(c.rank in (Rank.JACK, Rank.QUEEN, Rank.KING) for c in h),
    "has_ace": lambda h: any(c.rank == Rank.ACE for c in h),
    "has_heart": lambda h: any(c.suit == Suit.HEARTS for c in h),
    "has_spade": lambda h: any(c.suit == Suit.SPADES for c in h),
    "has_club": lambda h: any(c.suit == Suit.CLUBS for c in h),
    "has_diamond": lambda h: any(c.suit == Suit.DIAMONDS for c in h),
    "has_pair_ranks": lambda h: len(h) != len(set(c.rank for c in h)),
    "has_pair_suits": lambda h: any(
        sum(1 for c in h if c.suit == s) >= 2 for s in Suit
    ),
    "is_sorted_asc": lambda h: all(
        RANK_VALUES[h[i].rank] <= RANK_VALUES[h[i+1].rank]
        for i in range(len(h)-1)
    ) if len(h) > 1 else True,
    "is_sorted_desc": lambda h: all(
        RANK_VALUES[h[i].rank] >= RANK_VALUES[h[i+1].rank]
        for i in range(len(h)-1)
    ) if len(h) > 1 else True,
    "has_consec3": lambda h: _has_consec_n(h, 3),
    "has_consec2": lambda h: _has_consec_n(h, 2),
    "all_face": lambda h: all(c.rank in (Rank.JACK, Rank.QUEEN, Rank.KING) for c in h) if h else True,
    "all_red": lambda h: all(card_color(c) == Color.RED for c in h) if h else True,
    "all_black": lambda h: all(card_color(c) == Color.BLACK for c in h) if h else True,
    "monotone_inc": lambda h: all(
        RANK_VALUES[h[i].rank] < RANK_VALUES[h[i+1].rank]
        for i in range(len(h)-1)
    ) if len(h) > 1 else True,
}

def _has_consec_n(hand, n):
    """Check if hand contains n consecutive ranks."""
    vals = set(RANK_VALUES[c.rank] for c in hand)
    for v in vals:
        if all((v + k) in vals for k in range(n)):
            return True
    return False


def halves(hand):
    """Split hand into left and right halves."""
    n = len(hand)
    k = n // 2
    return hand[:k], hand[n - k:]


# ============================================================================
# Template definitions — each is a function that takes slot fillers
# and returns a (name, predicate) tuple
# ============================================================================

def gen_ends_rules():
    """
    ENDS template: λh. f(first(h)) OP f(last(h))

    Compare a property of the first card to the same property of the last card.
    We vary the property extractor f and the comparison operator OP.
    """
    rules = []

    # f(first) == f(last)
    for prop_name, prop_fn in PROPERTY_EXTRACTORS.items():
        rule_id = f"Ends_same_{prop_name}"

        def make_pred(fn):
            def pred(hand):
                if len(hand) < 2:
                    return False
                return fn(hand[0]) == fn(hand[-1])
            return pred

        rules.append({
            "id": rule_id,
            "family": "ENDS",
            "template": f"λh. {prop_name}(first(h)) = {prop_name}(last(h))",
            "predicate": make_pred(prop_fn),
            "description": f"First and last cards have the same {prop_name}",
        })

    # f(first) != f(last) — a few interesting ones
    for prop_name in ["suit", "color", "parity"]:
        prop_fn = PROPERTY_EXTRACTORS[prop_name]
        rule_id = f"Ends_diff_{prop_name}"

        def make_pred_diff(fn):
            def pred(hand):
                if len(hand) < 2:
                    return False
                return fn(hand[0]) != fn(hand[-1])
            return pred

        rules.append({
            "id": rule_id,
            "family": "ENDS",
            "template": f"λh. {prop_name}(first(h)) ≠ {prop_name}(last(h))",
            "predicate": make_pred_diff(prop_fn),
            "description": f"First and last cards have different {prop_name}",
        })

    return rules


def gen_palindrome_rules():
    """
    PALINDROME template: λh. map(f, h) == reverse(map(f, h))

    Extract a sequence of property values from the hand, check if it
    reads the same forwards and backwards.
    """
    rules = []

    for prop_name, prop_fn in PROPERTY_EXTRACTORS.items():
        rule_id = f"Palindrome_{prop_name}"

        def make_pred(fn):
            def pred(hand):
                seq = [fn(c) for c in hand]
                return seq == list(reversed(seq))
            return pred

        rules.append({
            "id": rule_id,
            "family": "PALINDROME",
            "template": f"λh. map({prop_name}, h) = reverse(map({prop_name}, h))",
            "predicate": make_pred(prop_fn),
            "description": f"The sequence of {prop_name} values is a palindrome",
        })

    return rules


def gen_halves_copy_rules():
    """
    HALVES_COPY template: λh. map(f, L(h)) == map(f, R(h))

    Both halves have identical property sequences.
    """
    rules = []

    for prop_name, prop_fn in PROPERTY_EXTRACTORS.items():
        rule_id = f"Halves_copy_{prop_name}"

        def make_pred(fn):
            def pred(hand):
                L, R = halves(hand)
                return [fn(c) for c in L] == [fn(c) for c in R]
            return pred

        rules.append({
            "id": rule_id,
            "family": "HALVES_COPY",
            "template": f"λh. map({prop_name}, L(h)) = map({prop_name}, R(h))",
            "predicate": make_pred(prop_fn),
            "description": f"Right half copies left half in {prop_name}",
        })

    # Set variant: same SET of values (ignoring order/count)
    for prop_name in ["suit", "rank", "color", "parity"]:
        prop_fn = PROPERTY_EXTRACTORS[prop_name]
        rule_id = f"Halves_same_set_{prop_name}"

        def make_pred_set(fn):
            def pred(hand):
                L, R = halves(hand)
                return set(fn(c) for c in L) == set(fn(c) for c in R)
            return pred

        rules.append({
            "id": rule_id,
            "family": "HALVES_COPY",
            "template": f"λh. set(map({prop_name}, L(h))) = set(map({prop_name}, R(h)))",
            "predicate": make_pred_set(prop_fn),
            "description": f"Both halves contain the same set of {prop_name} values",
        })

    return rules


def gen_halves_property_rules():
    """
    HALVES_PROPERTY template: λh. P(L(h)) ↔ P(R(h))

    A boolean property holds equally in both halves (both true or both false).
    This is the biconditional template.
    """
    rules = []

    for prop_name, prop_fn in BOOLEAN_PROPERTIES.items():
        rule_id = f"Halves_bicon_{prop_name}"

        def make_pred(fn):
            def pred(hand):
                L, R = halves(hand)
                return fn(L) == fn(R)
            return pred

        rules.append({
            "id": rule_id,
            "family": "HALVES_PROPERTY",
            "template": f"λh. {prop_name}(L(h)) ↔ {prop_name}(R(h))",
            "predicate": make_pred(prop_fn),
            "description": f"Both halves satisfy {prop_name}, or neither does",
        })

    # "Both" variant: P(L) AND P(R) (not biconditional)
    for prop_name in ["has_face", "has_pair_ranks", "has_consec3", "has_consec2",
                       "has_heart", "has_spade", "is_sorted_asc", "monotone_inc"]:
        prop_fn = BOOLEAN_PROPERTIES[prop_name]
        rule_id = f"Halves_both_{prop_name}"

        def make_pred_both(fn):
            def pred(hand):
                L, R = halves(hand)
                return fn(L) and fn(R)
            return pred

        rules.append({
            "id": rule_id,
            "family": "HALVES_PROPERTY",
            "template": f"λh. {prop_name}(L(h)) ∧ {prop_name}(R(h))",
            "predicate": make_pred_both(prop_fn),
            "description": f"Both halves satisfy {prop_name}",
        })

    return rules


def gen_adjacent_rules():
    """
    ADJACENT template: λh. ∀ consecutive pairs (h[i], h[i+1]). R(pair)

    Every adjacent pair must satisfy a relation.
    Also generates skip-2 variants.
    """
    rules = []

    # Adjacent pair relations
    pair_relations = {
        "same_suit": lambda a, b: a.suit == b.suit,
        "same_color": lambda a, b: card_color(a) == card_color(b),
        "same_parity": lambda a, b: (RANK_VALUES[a.rank] % 2) == (RANK_VALUES[b.rank] % 2),
        "rank_diff_le2": lambda a, b: abs(RANK_VALUES[a.rank] - RANK_VALUES[b.rank]) <= 2,
        "rank_diff_le4": lambda a, b: abs(RANK_VALUES[a.rank] - RANK_VALUES[b.rank]) <= 4,
        "rank_inc": lambda a, b: RANK_VALUES[a.rank] < RANK_VALUES[b.rank],
        "rank_nondec": lambda a, b: RANK_VALUES[a.rank] <= RANK_VALUES[b.rank],
        "same_rank_or_suit": lambda a, b: a.rank == b.rank or a.suit == b.suit,
        "same_rank_or_color": lambda a, b: a.rank == b.rank or card_color(a) == card_color(b),
        "diff_suit": lambda a, b: a.suit != b.suit,
        "alt_color": lambda a, b: card_color(a) != card_color(b),
    }

    for offset in [1, 2]:
        offset_name = "adj" if offset == 1 else "skip2"
        for rel_name, rel_fn in pair_relations.items():
            rule_id = f"{offset_name}_{rel_name}"

            def make_pred(fn, k):
                def pred(hand):
                    for i in range(len(hand) - k):
                        if not fn(hand[i], hand[i + k]):
                            return False
                    return True
                return pred

            rules.append({
                "id": rule_id,
                "family": "ADJACENT",
                "template": f"λh. ∀i. {rel_name}(h[i], h[i+{offset}])",
                "predicate": make_pred(rel_fn, offset),
                "description": f"Every pair at offset {offset} satisfies {rel_name}",
            })

    return rules


def gen_count_rules():
    """
    COUNT template: λh. op(count(h, X), k)

    Count something about the hand and compare to a threshold.
    """
    rules = []

    # Count of specific suits
    for suit_name, suit_val in [("clubs", Suit.CLUBS), ("diamonds", Suit.DIAMONDS),
                                 ("hearts", Suit.HEARTS), ("spades", Suit.SPADES)]:
        for k in [0, 1, 2, 3]:
            for op_name, op_fn in [("eq", lambda c, k: c == k),
                                    ("ge", lambda c, k: c >= k),
                                    ("le", lambda c, k: c <= k)]:
                rule_id = f"count_{suit_name}_{op_name}_{k}"

                def make_pred(sv, opf, kv):
                    def pred(hand):
                        cnt = sum(1 for c in hand if c.suit == sv)
                        return opf(cnt, kv)
                    return pred

                rules.append({
                    "id": rule_id,
                    "family": "COUNT",
                    "template": f"λh. count(suit={suit_name}) {op_name} {k}",
                    "predicate": make_pred(suit_val, op_fn, k),
                    "description": f"Number of {suit_name} is {op_name} {k}",
                })

    # Count of unique suits/ranks
    for target, target_fn in [("unique_suits", lambda h: len(set(c.suit for c in h))),
                               ("unique_ranks", lambda h: len(set(c.rank for c in h))),
                               ("unique_colors", lambda h: len(set(card_color(c) for c in h)))]:
        for k in [1, 2, 3, 4]:
            for op_name, op_fn in [("eq", lambda c, k: c == k),
                                    ("le", lambda c, k: c <= k),
                                    ("ge", lambda c, k: c >= k)]:
                rule_id = f"{target}_{op_name}_{k}"

                def make_pred_uc(tf, opf, kv):
                    def pred(hand):
                        return opf(tf(hand), kv)
                    return pred

                rules.append({
                    "id": rule_id,
                    "family": "COUNT",
                    "template": f"λh. {target}(h) {op_name} {k}",
                    "predicate": make_pred_uc(target_fn, op_fn, k),
                    "description": f"Number of {target} is {op_name} {k}",
                })

    # Count face cards / high cards / odd ranks
    special_counts = {
        "face_cards": lambda c: c.rank in (Rank.JACK, Rank.QUEEN, Rank.KING),
        "aces": lambda c: c.rank == Rank.ACE,
        "odd_ranks": lambda c: RANK_VALUES[c.rank] % 2 == 1,
        "even_ranks": lambda c: RANK_VALUES[c.rank] % 2 == 0,
        "high_cards": lambda c: RANK_VALUES[c.rank] >= 10,
    }

    for count_name, count_fn in special_counts.items():
        for k in [0, 1, 2, 3]:
            for op_name, op_fn in [("eq", lambda c, k: c == k),
                                    ("ge", lambda c, k: c >= k)]:
                rule_id = f"count_{count_name}_{op_name}_{k}"

                def make_pred_sc(cf, opf, kv):
                    def pred(hand):
                        cnt = sum(1 for c in hand if cf(c))
                        return opf(cnt, kv)
                    return pred

                rules.append({
                    "id": rule_id,
                    "family": "COUNT",
                    "template": f"λh. count({count_name}) {op_name} {k}",
                    "predicate": make_pred_sc(count_fn, op_fn, k),
                    "description": f"Number of {count_name} is {op_name} {k}",
                })

    # Number of rank pairs
    for k in [0, 1, 2, 3]:
        for op_name, op_fn in [("eq", lambda c, k: c == k), ("ge", lambda c, k: c >= k)]:
            rule_id = f"rank_pairs_{op_name}_{k}"

            def make_pred_rp(opf, kv):
                def pred(hand):
                    from collections import Counter
                    cnt = Counter(c.rank for c in hand)
                    pairs = sum(1 for v in cnt.values() if v >= 2)
                    return opf(pairs, kv)
                return pred

            rules.append({
                "id": rule_id,
                "family": "COUNT",
                "template": f"λh. rank_pairs(h) {op_name} {k}",
                "predicate": make_pred_rp(op_fn, k),
                "description": f"Number of rank pairs is {op_name} {k}",
            })

    return rules


def gen_shift_rules():
    """
    SHIFT template: λh. ∀i. rel(h[i], h[i + k])

    Position-wise comparison between elements at fixed offset.
    """
    rules = []

    shift_relations = {
        "rank_plus1": lambda a, b: RANK_VALUES[a.rank] + 1 == RANK_VALUES[b.rank],
        "rank_plus2": lambda a, b: RANK_VALUES[a.rank] + 2 == RANK_VALUES[b.rank],
        "rank_plus3": lambda a, b: RANK_VALUES[a.rank] + 3 == RANK_VALUES[b.rank],
        "rank_ge": lambda a, b: RANK_VALUES[b.rank] >= RANK_VALUES[a.rank],
        "rank_gt": lambda a, b: RANK_VALUES[b.rank] > RANK_VALUES[a.rank],
        "same_suit": lambda a, b: a.suit == b.suit,
        "same_color": lambda a, b: card_color(a) == card_color(b),
        "rank_eq": lambda a, b: a.rank == b.rank,
    }

    for offset_name, offset in [("half", None), ("2", 2), ("3", 3)]:
        for rel_name, rel_fn in shift_relations.items():
            rule_id = f"shift_{offset_name}_{rel_name}"

            def make_pred(fn, k_fixed):
                def pred(hand):
                    k = len(hand) // 2 if k_fixed is None else k_fixed
                    for i in range(len(hand) - k):
                        if not fn(hand[i], hand[i + k]):
                            return False
                    return True
                return pred

            rules.append({
                "id": rule_id,
                "family": "SHIFT",
                "template": f"λh. ∀i. {rel_name}(h[i], h[i+{offset_name}])",
                "predicate": make_pred(rel_fn, offset),
                "description": f"At offset {offset_name}: {rel_name}",
            })

    return rules


def gen_global_rules():
    """
    GLOBAL template: λh. P(h) — whole-hand properties

    Various properties that apply to the entire hand.
    """
    rules = []

    # Range rules: max_rank - min_rank comparisons
    for threshold in [2, 3, 4, 5, 6]:
        rule_id = f"rank_range_le_{threshold}"

        def make_pred(t):
            def pred(hand):
                vals = [RANK_VALUES[c.rank] for c in hand]
                return max(vals) - min(vals) <= t
            return pred

        rules.append({
            "id": rule_id,
            "family": "GLOBAL",
            "template": f"λh. max_rank(h) - min_rank(h) ≤ {threshold}",
            "predicate": make_pred(threshold),
            "description": f"Rank range (max - min) is at most {threshold}",
        })

    # Sum threshold rules
    for threshold in [30, 40, 50, 60]:
        rule_id = f"sum_ranks_ge_{threshold}"

        def make_pred_sum(t):
            def pred(hand):
                return sum(RANK_VALUES[c.rank] for c in hand) >= t
            return pred

        rules.append({
            "id": rule_id,
            "family": "GLOBAL",
            "template": f"λh. sum_ranks(h) ≥ {threshold}",
            "predicate": make_pred_sum(threshold),
            "description": f"Sum of rank values is at least {threshold}",
        })

    # All-but-one variants
    for prop_name, prop_fn, desc in [
        ("color", lambda c: card_color(c), "same color"),
        ("suit", lambda c: c.suit, "same suit"),
        ("parity", lambda c: RANK_VALUES[c.rank] % 2, "same parity"),
    ]:
        rule_id = f"all_but_one_same_{prop_name}"

        def make_pred_abo(fn):
            def pred(hand):
                from collections import Counter
                vals = Counter(fn(c) for c in hand)
                return min(vals.values()) <= 1
            return pred

        rules.append({
            "id": rule_id,
            "family": "GLOBAL",
            "template": f"λh. all_but_one_same_{prop_name}(h)",
            "predicate": make_pred_abo(prop_fn),
            "description": f"All but at most one card share the {desc}",
        })

    return rules


# ============================================================================
# Main: generate all template rules, compute base rates, filter
# ============================================================================

def make_deck(deck_size=52):
    """Create a deck of cards."""
    from rules.cards import RANK_VALUES as RV
    all_ranks = list(Rank)
    if deck_size == 52:
        ranks = all_ranks
    elif deck_size == 32:
        ranks = [r for r in all_ranks if RV[r] >= 7]
    elif deck_size == 28:
        ranks = [r for r in all_ranks if RV[r] >= 8]
    else:
        raise ValueError(f"Unsupported deck size: {deck_size}")
    return [Card(suit, rank) for suit in Suit for rank in ranks]


def compute_base_rate(pred, deck, hand_size, n_samples):
    """Estimate P(rule) via Monte Carlo."""
    n_true = 0
    for _ in range(n_samples):
        hand = random.sample(deck, hand_size)
        try:
            if pred(hand):
                n_true += 1
        except Exception:
            pass
    p_hat = n_true / n_samples
    se = math.sqrt(p_hat * (1 - p_hat) / n_samples) if n_samples > 0 else 0
    return p_hat, se


def main():
    parser = argparse.ArgumentParser(description="Phase 2b: Template-Based Rule Generation")
    parser.add_argument("--samples", type=int, default=50_000,
                        help="Monte Carlo samples per rule (default: 50,000)")
    parser.add_argument("--hand-size", type=int, default=6)
    parser.add_argument("--deck-size", type=int, default=52)
    parser.add_argument("--min-rate", type=float, default=0.03,
                        help="Minimum base rate (default: 3%%)")
    parser.add_argument("--max-rate", type=float, default=0.50,
                        help="Maximum base rate (default: 50%%)")
    parser.add_argument("--output", type=str, default="template_candidates.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    deck = make_deck(args.deck_size)

    print(f"=== Phase 2b: Template-Based Rule Generation ===")
    print(f"  Deck: {args.deck_size}, Hand: {args.hand_size}, Samples: {args.samples:,}")
    print(f"  Base rate filter: [{args.min_rate*100:.0f}%, {args.max_rate*100:.0f}%]")
    print()

    # 1. Generate all template rules
    # Each generator returns a list of dicts with keys: id, family, template, predicate, description
    generators = [
        ("ENDS", gen_ends_rules),
        ("PALINDROME", gen_palindrome_rules),
        ("HALVES_COPY", gen_halves_copy_rules),
        ("HALVES_PROPERTY", gen_halves_property_rules),
        ("ADJACENT", gen_adjacent_rules),
        ("COUNT", gen_count_rules),
        ("SHIFT", gen_shift_rules),
        ("GLOBAL", gen_global_rules),
    ]

    all_candidates = []
    for family_name, gen_fn in generators:
        candidates = gen_fn()
        all_candidates.extend(candidates)
        print(f"  {family_name}: generated {len(candidates)} candidates")

    print(f"\nTotal raw candidates: {len(all_candidates)}")

    # 2. Compute base rates and filter
    print(f"\nComputing base rates...")
    passed = []
    too_low = 0
    too_high = 0

    for i, rule in enumerate(all_candidates):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(all_candidates)} ({len(passed)} passed so far)")

        p_hat, se = compute_base_rate(rule["predicate"], deck, args.hand_size, args.samples)

        if p_hat < args.min_rate:
            too_low += 1
            continue
        if p_hat > args.max_rate:
            too_high += 1
            continue

        passed.append({
            "id": rule["id"],
            "family": rule["family"],
            "template": rule["template"],
            "description": rule["description"],
            "base_rate": round(p_hat, 6),
            "base_rate_pct": round(p_hat * 100, 2),
            "base_rate_se": round(se, 6),
        })

    print(f"\n=== Filtering Results ===")
    print(f"  Total candidates: {len(all_candidates)}")
    print(f"  Too low (< {args.min_rate*100:.0f}%): {too_low}")
    print(f"  Too high (> {args.max_rate*100:.0f}%): {too_high}")
    print(f"  Passed: {len(passed)}")

    # 3. Remove extensionally equivalent rules (same base rate ± SE)
    # Sort by base rate for dedup
    passed.sort(key=lambda r: r["base_rate"])

    # 4. Write CSV
    output_path = Path(__file__).parent / args.output
    fieldnames = ["id", "family", "base_rate", "base_rate_pct", "base_rate_se",
                  "template", "description"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(passed)

    print(f"\nWrote {len(passed)} candidates to {output_path}")

    # 5. Print summary by family
    print(f"\n{'='*80}")
    print("CANDIDATES BY FAMILY")
    print(f"{'='*80}")

    from collections import Counter
    family_counts = Counter(r["family"] for r in passed)
    for family, count in sorted(family_counts.items()):
        family_rules = [r for r in passed if r["family"] == family]
        rates = [r["base_rate"] for r in family_rules]
        print(f"  {family:<20} {count:>4} rules  [{min(rates)*100:>5.1f}% - {max(rates)*100:>5.1f}%]")

    # 6. Print top candidates (most interesting base rates near 10-30%)
    print(f"\n{'='*80}")
    print("TOP 30 CANDIDATES (sorted by base rate, targeting 10-30%)")
    print(f"{'='*80}")

    sweet = [r for r in passed if 0.08 <= r["base_rate"] <= 0.35]
    sweet.sort(key=lambda r: r["base_rate"])

    for r in sweet[:30]:
        print(f"  {r['base_rate_pct']:>6.2f}%  {r['id']:<40} {r['family']}")


if __name__ == "__main__":
    main()
