#!/usr/bin/env python3
"""
Phase 3: LLM-Assisted Rule Discovery

Implements 18 novel rules brainstormed by an LLM agent, targeting structural
families NOT covered by template generation (Phase 2b) or grammar enumeration
(Phase 2a). Each rule is a predicate (Hand → bool) defined over 6-card hands.

New structural families introduced:
  ZIGZAG       — rank direction alternates up/down
  GRAPH        — connectivity via shared rank/suit
  SANDWICH     — outer vs inner positional constraints
  CROSS_FEAT   — cross-feature equality counts
  EXTREMAL     — argmax/argmin positional constraints
  STRIDE       — properties at strided positions (1,3,5 or 2,4,6)
  DISTRIBUTION — suit/rank distribution constraints
  CONDITIONAL  — conditional monotonicity (almost-sorted)

Usage:
    cd card-games-modelling/src
    python3 llm_rule_candidates.py [--samples 50000] [--seed 42]
"""

import sys
import csv
import math
import random
import argparse
from pathlib import Path
from typing import List, Callable, Set
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

from rules.cards import Card, Hand, Suit, Rank, RANK_VALUES, card_color, Color


# ============================================================================
# Helper: make a deck and sample hands (same as other Phase 2 scripts)
# ============================================================================

def make_deck(deck_size=52):
    """Create a standard deck of the specified size."""
    all_ranks = list(Rank)
    if deck_size == 52:
        ranks = all_ranks
    elif deck_size == 32:
        ranks = [r for r in all_ranks if RANK_VALUES[r] >= 7]
    else:
        raise ValueError(f"Unsupported deck size: {deck_size}")
    return [Card(suit, rank) for suit in Suit for rank in ranks]


def sample_hands(deck, hand_size, n):
    """Pre-generate n random hands (without replacement within each hand)."""
    return [tuple(random.sample(deck, hand_size)) for _ in range(n)]


# ============================================================================
# Rule definitions — 18 novel rules across 8 new structural families
# ============================================================================
# Each rule is a dict with:
#   id:          unique string identifier
#   family:      structural family name
#   description: human-readable explanation
#   predicate:   Callable[[Hand], bool]

def define_llm_rules() -> List[dict]:
    """
    Return the list of 18 LLM-brainstormed rules.

    Each rule targets a structural pattern not well-covered by the existing
    catalogue (55 rules), template generation (111 candidates), or grammar
    enumeration (604 candidates).
    """
    rules = []

    # ── FAMILY: ZIGZAG ──────────────────────────────────────────────────
    # Rules about alternating rank direction (up-down-up-down)

    def zigzag_ranks(hand: Hand) -> bool:
        """Ranks alternate: each consecutive pair switches direction.
        e.g., 3→7→2→9→4→8 alternates up-down-up-down-up.
        Requires all 5 consecutive differences to alternate sign."""
        vals = [RANK_VALUES[c.rank] for c in hand]
        diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
        # Need all diffs non-zero and alternating sign
        for d in diffs:
            if d == 0:
                return False
        for i in range(len(diffs)-1):
            # Adjacent diffs must have opposite signs
            if (diffs[i] > 0) == (diffs[i+1] > 0):
                return False
        return True

    rules.append({
        "id": "zigzag_ranks",
        "family": "ZIGZAG",
        "description": "Ranks alternate direction: up-down-up-down-up (or down-up-down-up-down)",
        "predicate": zigzag_ranks,
    })

    def convex_or_concave(hand: Hand) -> bool:
        """Ranks form a mountain (rise then fall) or valley (fall then rise).
        Must have exactly one direction change, and it must be interior
        (not at the very start or end)."""
        vals = [RANK_VALUES[c.rank] for c in hand]
        diffs = [vals[i+1] - vals[i] for i in range(len(vals)-1)]
        # Filter out zero diffs (ties break the pattern)
        nonzero = [d for d in diffs if d != 0]
        if len(nonzero) < 2:
            return False
        # Count sign changes
        changes = 0
        for i in range(len(nonzero)-1):
            if (nonzero[i] > 0) != (nonzero[i+1] > 0):
                changes += 1
        return changes == 1

    rules.append({
        "id": "convex_or_concave",
        "family": "ZIGZAG",
        "description": "Ranks form a mountain (up then down) or valley (down then up)",
        "predicate": convex_or_concave,
    })

    # ── FAMILY: GRAPH ───────────────────────────────────────────────────
    # Treat cards as nodes; edges connect cards sharing rank or suit

    def connected_graph(hand: Hand) -> bool:
        """The 6 cards form a connected graph where edges link cards that
        share a rank or a suit. Uses union-find for efficiency."""
        n = len(hand)
        # parent array for union-find
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i+1, n):
                if hand[i].rank == hand[j].rank or hand[i].suit == hand[j].suit:
                    union(i, j)

        # Check if all nodes share the same root
        roots = set(find(i) for i in range(n))
        return len(roots) == 1

    rules.append({
        "id": "connected_graph",
        "family": "GRAPH",
        "description": "Cards form a connected graph via shared rank or suit",
        "predicate": connected_graph,
    })

    def color_neighbor(hand: Hand) -> bool:
        """Every card has at least one adjacent card (by position) with the
        same color (red/black)."""
        colors = [card_color(c) for c in hand]
        for i in range(len(colors)):
            has_same = False
            if i > 0 and colors[i] == colors[i-1]:
                has_same = True
            if i < len(colors)-1 and colors[i] == colors[i+1]:
                has_same = True
            if not has_same:
                return False
        return True

    rules.append({
        "id": "color_neighbor",
        "family": "GRAPH",
        "description": "Every card has an adjacent card of the same color",
        "predicate": color_neighbor,
    })

    # ── FAMILY: SANDWICH ────────────────────────────────────────────────
    # Outer vs inner positional constraints

    def sandwich_suits(hand: Hand) -> bool:
        """Positions 1 and 6 share a suit, AND positions 2 and 5 share a suit.
        (A sandwich: outer layers match, middle layers match.)"""
        return (hand[0].suit == hand[5].suit and
                hand[1].suit == hand[4].suit)

    rules.append({
        "id": "sandwich_suits",
        "family": "SANDWICH",
        "description": "Pos 1&6 share suit, AND pos 2&5 share suit (sandwich mirror)",
        "predicate": sandwich_suits,
    })

    def bridge_suit(hand: Hand) -> bool:
        """Endpoints (pos 1 and 6) share a suit, and no interior card
        (pos 2-5) has that same suit."""
        endpoint_suit = hand[0].suit
        if hand[5].suit != endpoint_suit:
            return False
        for c in hand[1:5]:
            if c.suit == endpoint_suit:
                return False
        return True

    rules.append({
        "id": "bridge_suit",
        "family": "SANDWICH",
        "description": "Endpoints share a suit that no interior card has",
        "predicate": bridge_suit,
    })

    def frame_higher(hand: Hand) -> bool:
        """Edge cards (pos 1, 2, 5, 6) all outrank interior cards (pos 3, 4).
        The 'frame' is strictly higher than the 'interior'."""
        edge_vals = [RANK_VALUES[hand[i].rank] for i in [0, 1, 4, 5]]
        interior_vals = [RANK_VALUES[hand[i].rank] for i in [2, 3]]
        return min(edge_vals) > max(interior_vals)

    rules.append({
        "id": "frame_higher",
        "family": "SANDWICH",
        "description": "Edge cards (1,2,5,6) all outrank interior cards (3,4)",
        "predicate": frame_higher,
    })

    # ── FAMILY: CROSS_FEAT ──────────────────────────────────────────────
    # Cross-feature equality counts

    def suit_change_eq_rank_increase(hand: Hand) -> bool:
        """Count of suit transitions equals count of rank increases.
        A suit transition: hand[i].suit != hand[i+1].suit.
        A rank increase: RANK_VALUES[hand[i+1]] > RANK_VALUES[hand[i]]."""
        suit_changes = 0
        rank_increases = 0
        for i in range(len(hand)-1):
            if hand[i].suit != hand[i+1].suit:
                suit_changes += 1
            if RANK_VALUES[hand[i+1].rank] > RANK_VALUES[hand[i].rank]:
                rank_increases += 1
        return suit_changes == rank_increases

    rules.append({
        "id": "suit_change_eq_rank_increase",
        "family": "CROSS_FEAT",
        "description": "Number of suit transitions = number of rank increases",
        "predicate": suit_change_eq_rank_increase,
    })

    # ── FAMILY: EXTREMAL ────────────────────────────────────────────────
    # Argmax/argmin positional constraints

    def extremes_same_suit(hand: Hand) -> bool:
        """The highest-ranked card and the lowest-ranked card share a suit.
        (If ties, use the first occurrence for highest, last for lowest.)"""
        vals = [RANK_VALUES[c.rank] for c in hand]
        max_val = max(vals)
        min_val = min(vals)
        # First card with max rank
        max_card = next(c for c, v in zip(hand, vals) if v == max_val)
        # Last card with min rank
        min_card = None
        for c, v in zip(hand, vals):
            if v == min_val:
                min_card = c
        return max_card.suit == min_card.suit

    rules.append({
        "id": "extremes_same_suit",
        "family": "EXTREMAL",
        "description": "Highest and lowest ranked cards share a suit",
        "predicate": extremes_same_suit,
    })

    def suit_respects_order(hand: Hand) -> bool:
        """For each suit present, the cards of that suit appear in ascending
        rank order within the hand (by position). e.g., if hearts appear at
        positions 1, 3, 5, their ranks must be increasing."""
        from collections import defaultdict
        suit_positions = defaultdict(list)
        for i, c in enumerate(hand):
            suit_positions[c.suit].append(RANK_VALUES[c.rank])
        for suit, vals in suit_positions.items():
            if len(vals) >= 2:
                # Check if strictly increasing
                for j in range(len(vals)-1):
                    if vals[j] >= vals[j+1]:
                        return False
        return True

    rules.append({
        "id": "suit_respects_order",
        "family": "EXTREMAL",
        "description": "Same-suit cards appear in ascending rank order by position",
        "predicate": suit_respects_order,
    })

    def max_rank_at_edge(hand: Hand) -> bool:
        """The highest-ranked card is at position 1 or 6 (the edges).
        Ties: True if ANY card with the max rank is at position 1 or 6."""
        vals = [RANK_VALUES[c.rank] for c in hand]
        max_val = max(vals)
        return vals[0] == max_val or vals[5] == max_val

    rules.append({
        "id": "max_rank_at_edge",
        "family": "EXTREMAL",
        "description": "Highest-ranked card is at position 1 or 6",
        "predicate": max_rank_at_edge,
    })

    def min_rank_at_center(hand: Hand) -> bool:
        """The lowest-ranked card is at position 3 or 4 (the center).
        Ties: True if ANY card with the min rank is at center."""
        vals = [RANK_VALUES[c.rank] for c in hand]
        min_val = min(vals)
        return vals[2] == min_val or vals[3] == min_val

    rules.append({
        "id": "min_rank_at_center",
        "family": "EXTREMAL",
        "description": "Lowest-ranked card is at position 3 or 4",
        "predicate": min_rank_at_center,
    })

    # ── FAMILY: STRIDE ──────────────────────────────────────────────────
    # Properties at strided positions (evens vs odds)

    def interleaved_colors(hand: Hand) -> bool:
        """Even-indexed positions (0,2,4) share one color, odd-indexed
        positions (1,3,5) share another color, and the two colors differ."""
        even_colors = set(card_color(hand[i]) for i in [0, 2, 4])
        odd_colors = set(card_color(hand[i]) for i in [1, 3, 5])
        if len(even_colors) != 1 or len(odd_colors) != 1:
            return False
        return even_colors != odd_colors  # The two groups must differ

    rules.append({
        "id": "interleaved_colors",
        "family": "STRIDE",
        "description": "Even positions share one color, odd positions share the other",
        "predicate": interleaved_colors,
    })

    def skip_same_suit(hand: Hand) -> bool:
        """Positions 1, 3, 5 (0-indexed: 0, 2, 4) all share the same suit."""
        return (hand[0].suit == hand[2].suit == hand[4].suit)

    rules.append({
        "id": "skip_same_suit",
        "family": "STRIDE",
        "description": "Positions 1, 3, 5 share the same suit",
        "predicate": skip_same_suit,
    })

    # ── FAMILY: DISTRIBUTION ────────────────────────────────────────────
    # Suit/rank distribution constraints

    def rainbow(hand: Hand) -> bool:
        """All four suits are represented in the hand."""
        suits = set(c.suit for c in hand)
        return len(suits) == 4

    rules.append({
        "id": "rainbow",
        "family": "DISTRIBUTION",
        "description": "All four suits are present in the hand",
        "predicate": rainbow,
    })

    def suit_run_3(hand: Hand) -> bool:
        """Three consecutive cards (by position) share the same suit."""
        for i in range(len(hand) - 2):
            if hand[i].suit == hand[i+1].suit == hand[i+2].suit:
                return True
        return False

    rules.append({
        "id": "suit_run_3",
        "family": "DISTRIBUTION",
        "description": "Three consecutive cards share the same suit",
        "predicate": suit_run_3,
    })

    def pair_at_distance_3(hand: Hand) -> bool:
        """There exist positions i and i+3 with the same rank.
        In a 6-card hand, this means pos (0,3), (1,4), or (2,5)."""
        for i in range(len(hand) - 3):
            if hand[i].rank == hand[i+3].rank:
                return True
        return False

    rules.append({
        "id": "pair_at_distance_3",
        "family": "DISTRIBUTION",
        "description": "Same rank appears at positions exactly 3 apart",
        "predicate": pair_at_distance_3,
    })

    # ── FAMILY: CONDITIONAL ─────────────────────────────────────────────
    # Conditional monotonicity

    def at_most_one_descent(hand: Hand) -> bool:
        """Ranks decrease at most once across the hand (almost sorted).
        A 'descent' is where hand[i+1] < hand[i]. We allow at most 1."""
        vals = [RANK_VALUES[c.rank] for c in hand]
        descents = sum(1 for i in range(len(vals)-1) if vals[i+1] < vals[i])
        return descents <= 1

    rules.append({
        "id": "at_most_one_descent",
        "family": "CONDITIONAL",
        "description": "Ranks decrease at most once (almost sorted ascending)",
        "predicate": at_most_one_descent,
    })

    return rules


# ============================================================================
# Base rate computation and fingerprinting
# ============================================================================

def compute_base_rate(predicate: Callable, hands: list):
    """
    Compute base rate of a rule over pre-sampled hands.

    Returns (base_rate, se, n_valid) where:
      - base_rate: fraction of hands where rule is True
      - se: standard error of the estimate
      - n_valid: number of hands successfully evaluated
    """
    n_true = 0
    n_valid = 0

    for hand in hands:
        try:
            result = predicate(hand)
            if isinstance(result, bool):
                n_valid += 1
                if result:
                    n_true += 1
        except Exception:
            pass  # Skip hands that cause errors

    if n_valid < 100:
        return None, None, n_valid

    p_hat = n_true / n_valid
    se = math.sqrt(p_hat * (1 - p_hat) / n_valid) if 0 < p_hat < 1 else 0.0
    return p_hat, se, n_valid


def compute_fingerprint(predicate: Callable, hands: list) -> str:
    """
    Compute boolean fingerprint over a fixed set of hands.
    Two rules with the same fingerprint are extensionally equivalent.
    """
    bits = []
    for hand in hands:
        try:
            result = predicate(hand)
            if result is True:
                bits.append('1')
            elif result is False:
                bits.append('0')
            else:
                bits.append('?')
        except Exception:
            bits.append('?')
    return ''.join(bits)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 3: LLM-Assisted Rule Discovery")
    parser.add_argument("--samples", type=int, default=50_000,
                        help="Monte Carlo samples for base rate (default: 50,000)")
    parser.add_argument("--fingerprint-size", type=int, default=1000,
                        help="Number of hands for fingerprinting (default: 1000)")
    parser.add_argument("--hand-size", type=int, default=6)
    parser.add_argument("--deck-size", type=int, default=52)
    parser.add_argument("--min-rate", type=float, default=0.03,
                        help="Minimum base rate filter (default: 3%%)")
    parser.add_argument("--max-rate", type=float, default=0.50,
                        help="Maximum base rate filter (default: 50%%)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="llm_candidates.csv")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"{'='*80}")
    print(f"Phase 3: LLM-Assisted Rule Discovery")
    print(f"{'='*80}")
    print(f"  Deck: {args.deck_size}, Hand: {args.hand_size}")
    print(f"  Monte Carlo samples: {args.samples:,}")
    print(f"  Fingerprint size: {args.fingerprint_size}")
    print(f"  Base rate filter: [{args.min_rate*100:.0f}%, {args.max_rate*100:.0f}%]")
    print()

    # 1. Generate hands for evaluation and fingerprinting
    print("1. Generating sample hands...")
    deck = make_deck(args.deck_size)
    eval_hands = sample_hands(deck, args.hand_size, args.samples)
    fp_hands = sample_hands(deck, args.hand_size, args.fingerprint_size)
    print(f"   {len(eval_hands):,} evaluation hands, {len(fp_hands)} fingerprint hands")

    # 2. Load LLM-brainstormed rules
    print("\n2. Loading LLM-brainstormed rules...")
    all_rules = define_llm_rules()
    print(f"   {len(all_rules)} rules defined across {len(set(r['family'] for r in all_rules))} families")

    # 3. Load existing fingerprints for deduplication
    # We compute fingerprints for existing catalogue rules to check for overlap
    print("\n3. Loading catalogue rules for dedup...")
    from rules.catalogue import create_all_rules
    cat_rules = create_all_rules()
    cat_fingerprints: Set[str] = set()
    for rule in cat_rules:
        fp = compute_fingerprint(rule.predicate, fp_hands)
        cat_fingerprints.add(fp)
    print(f"   {len(cat_fingerprints)} unique catalogue fingerprints")

    # 4. Evaluate each LLM rule
    print(f"\n4. Evaluating rules...")
    print(f"   {'ID':<30} {'Family':<15} {'Rate':>7} {'SE':>7} {'Status'}")
    print(f"   {'-'*30} {'-'*15} {'-'*7} {'-'*7} {'-'*20}")

    candidates = []
    n_filtered_low = 0
    n_filtered_high = 0
    n_duplicate = 0
    n_error = 0

    for rule in all_rules:
        rule_id = rule["id"]
        family = rule["family"]
        predicate = rule["predicate"]

        # Compute base rate on the full sample
        p_hat, se, n_valid = compute_base_rate(predicate, eval_hands)

        if p_hat is None:
            print(f"   {rule_id:<30} {family:<15} {'ERROR':>7} {'':>7} Too few valid evaluations ({n_valid})")
            n_error += 1
            continue

        rate_pct = p_hat * 100
        se_pct = se * 100

        # Filter by base rate
        if p_hat < args.min_rate:
            print(f"   {rule_id:<30} {family:<15} {rate_pct:>6.1f}% {se_pct:>6.2f}% FILTERED (too low)")
            n_filtered_low += 1
            continue
        if p_hat > args.max_rate:
            print(f"   {rule_id:<30} {family:<15} {rate_pct:>6.1f}% {se_pct:>6.2f}% FILTERED (too high)")
            n_filtered_high += 1
            continue

        # Fingerprint dedup against catalogue
        fp = compute_fingerprint(predicate, fp_hands)
        if fp in cat_fingerprints:
            print(f"   {rule_id:<30} {family:<15} {rate_pct:>6.1f}% {se_pct:>6.2f}% DUPLICATE (matches catalogue)")
            n_duplicate += 1
            continue

        # Passed all filters!
        print(f"   {rule_id:<30} {family:<15} {rate_pct:>6.1f}% {se_pct:>6.2f}% ★ CANDIDATE")
        candidates.append({
            "id": rule_id,
            "source": "llm",
            "family": family,
            "base_rate": round(p_hat, 6),
            "base_rate_pct": round(rate_pct, 2),
            "base_rate_se": round(se, 6),
            "description": rule["description"],
        })

    # 5. Summary
    print(f"\n{'='*80}")
    print(f"PHASE 3 RESULTS")
    print(f"{'='*80}")
    print(f"  Total rules defined: {len(all_rules)}")
    print(f"  Filtered (rate < {args.min_rate*100:.0f}%): {n_filtered_low}")
    print(f"  Filtered (rate > {args.max_rate*100:.0f}%): {n_filtered_high}")
    print(f"  Duplicates (match catalogue): {n_duplicate}")
    print(f"  Errors: {n_error}")
    print(f"  ★ Candidates: {len(candidates)}")

    # 6. Write CSV
    if candidates:
        output_path = Path(__file__).parent / args.output
        fieldnames = ["id", "source", "family", "base_rate", "base_rate_pct",
                      "base_rate_se", "description"]

        # Sort by base rate
        candidates.sort(key=lambda r: float(r["base_rate"]))

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(candidates)

        print(f"\n  Wrote {len(candidates)} candidates to {output_path.name}")

        # Print candidates table
        print(f"\n{'='*80}")
        print(f"CANDIDATE DETAILS (sorted by base rate)")
        print(f"{'='*80}")
        for c in candidates:
            print(f"  {c['base_rate_pct']:>5.1f}%  {c['id']:<30} [{c['family']:<12}] {c['description']}")
    else:
        print("\n  No candidates survived filtering. All rules were too common/rare or duplicates.")

    # 7. Family distribution
    print(f"\n{'='*80}")
    print(f"FAMILY DISTRIBUTION")
    print(f"{'='*80}")
    family_counts = Counter(c["family"] for c in candidates)
    for family, count in sorted(family_counts.items()):
        print(f"  {family:<15} {count} candidates")


if __name__ == "__main__":
    main()
