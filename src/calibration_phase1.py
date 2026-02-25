#!/usr/bin/env python3
"""
Phase 1: Calibration — compute base rates and compositional depth for all rules.

This script reads the 45-rule catalogue, samples hands via Monte Carlo,
and produces a CSV with:
  - token: rule token (e.g., r1x)
  - rule_id: rule identifier
  - family: rule family
  - base_rate_6: P(rule) for 6-card hands (our experiment uses 6)
  - base_rate_6_se: standard error of the base rate estimate
  - tree_size: number of nodes in the CompositionNode tree
  - tree_depth: depth of the CompositionNode tree
  - n_primitives: number of unique primitives used
  - level: compositional depth annotation from catalogue

How it works:
  - For each rule, we sample N_SAMPLES random 6-card hands (without replacement
    from a 52-card deck) and compute the fraction that satisfy the rule.
  - The CompositionNode tree is walked to compute tree size (total nodes) and
    tree depth (longest root-to-leaf path), which serve as proxies for
    Description Length under a uniform grammar.

Usage:
    cd card-games-modelling/src
    python3 calibration_phase1.py [--samples 100000] [--hand-size 6] [--deck-size 52]
"""

import sys
import csv
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict

# ── Setup path so we can import the catalogue ──
# We add the parent directory (src/) to sys.path so Python finds
# `rules.catalogue`, `rules.cards`, etc.
sys.path.insert(0, str(Path(__file__).parent))

from rules.catalogue import create_all_rules, CompositionNode
from rules.cards import Card, Suit, Rank, Hand, sample_hand


# ============================================================================
# CompositionNode metrics — walk the tree to measure complexity
# ============================================================================

def tree_size(node: CompositionNode) -> int:
    """
    Count total number of nodes in a CompositionNode tree.

    Each node counts as 1 plus the sizes of all its children.
    This is a proxy for "program size" — how many primitives
    and combinators are needed to express the rule.

    Example:
        C("is_sorted", C("map", C("get_rank_val")))
        has 3 nodes: is_sorted → map → get_rank_val.
    """
    return 1 + sum(tree_size(child) for child in node.args)


def tree_depth(node: CompositionNode) -> int:
    """
    Compute the depth (longest root-to-leaf path) of a CompositionNode tree.

    A single node (leaf) has depth 1.
    This measures the "nesting level" of the rule — deeper rules
    require more compositional reasoning to understand.
    """
    if not node.args:
        return 1
    return 1 + max(tree_depth(child) for child in node.args)


# ============================================================================
# Deck generation for different deck sizes
# ============================================================================

def make_deck(deck_size: int = 52) -> list:
    """
    Create a deck of cards for the specified size.

    How deck sizes work:
      - 52: Standard deck (all 13 ranks × 4 suits)
      - 32: Piquet deck (ranks 7-A only, i.e., 7,8,9,10,J,Q,K,A × 4 suits)
      - 28: Euchre-style (ranks 8-A only)

    The experiment supports all three; 32-card is our current default.
    """
    all_ranks = list(Rank)  # TWO through ACE

    from rules.cards import RANK_VALUES

    if deck_size == 52:
        ranks = all_ranks
    elif deck_size == 32:
        # Piquet: 7-A = ranks with value >= 7
        ranks = [r for r in all_ranks if RANK_VALUES[r] >= 7]
    elif deck_size == 28:
        # Euchre: 8-A = ranks with value >= 8
        ranks = [r for r in all_ranks if RANK_VALUES[r] >= 8]
    else:
        raise ValueError(f"Unsupported deck size: {deck_size}")

    deck = [Card(suit, rank) for suit in Suit for rank in ranks]
    assert len(deck) == deck_size, f"Expected {deck_size} cards, got {len(deck)}"
    return deck


def sample_hand_from_deck(deck: list, hand_size: int) -> Hand:
    """
    Draw a hand of `hand_size` cards from `deck` without replacement.

    This matches the experiment's sampling: participants see hands drawn
    from a shuffled deck, never with duplicate cards.
    """
    return random.sample(deck, hand_size)


# ============================================================================
# Monte Carlo base rate estimation
# ============================================================================

def compute_base_rates(rules, deck, hand_size: int, n_samples: int):
    """
    Estimate P(rule) for each rule via Monte Carlo sampling.

    For each rule, we draw `n_samples` random hands and compute:
      - p_hat: fraction of hands satisfying the rule
      - se: standard error = sqrt(p_hat * (1 - p_hat) / n_samples)

    Returns a dict: rule_id → (p_hat, se, n_true)

    Why Monte Carlo? The exact combinatorial count C(deck, hand_size)
    is huge (52-choose-6 ≈ 20 million), and many rules involve ordering,
    so Monte Carlo with 100K samples gives <0.2% SE for typical rules.
    """
    results = {}

    for rule in rules:
        n_true = 0
        for _ in range(n_samples):
            hand = sample_hand_from_deck(deck, hand_size)
            try:
                if rule.eval(hand):
                    n_true += 1
            except Exception as e:
                # Some rules may error on edge cases; count as False
                pass

        p_hat = n_true / n_samples
        # Standard error of a proportion
        se = math.sqrt(p_hat * (1 - p_hat) / n_samples) if n_samples > 0 else 0

        results[rule.id] = (p_hat, se, n_true)

    return results


# ============================================================================
# Main — produce calibration table
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 1 Calibration: base rates & compositional depth")
    parser.add_argument("--samples", type=int, default=100_000,
                        help="Monte Carlo samples per rule (default: 100,000)")
    parser.add_argument("--hand-size", type=int, default=6,
                        help="Hand size (default: 6)")
    parser.add_argument("--deck-size", type=int, default=52,
                        help="Deck size: 52, 32, or 28 (default: 52)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: calibration_phase1_<deck>.csv)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    random.seed(args.seed)

    if args.output is None:
        args.output = f"calibration_phase1_deck{args.deck_size}.csv"

    print(f"=== Phase 1 Calibration ===")
    print(f"  Samples: {args.samples:,}")
    print(f"  Hand size: {args.hand_size}")
    print(f"  Deck size: {args.deck_size}")
    print(f"  Output: {args.output}")
    print()

    # 1. Load all rules from the catalogue
    # create_all_rules() returns a list of Rule dataclass objects,
    # each with .id, .token, .predicate, .family, .composition, etc.
    rules = create_all_rules()
    print(f"Loaded {len(rules)} rules from catalogue.")

    # 2. Build the deck
    deck = make_deck(args.deck_size)
    print(f"Deck: {len(deck)} cards ({args.deck_size}-card deck)")
    print()

    # 3. Compute base rates via Monte Carlo
    print(f"Computing base rates ({args.samples:,} samples per rule)...")
    base_rates = compute_base_rates(rules, deck, args.hand_size, args.samples)

    # 4. Compute tree metrics for each rule
    # These come from the CompositionNode tree attached to each rule.
    tree_metrics = {}
    for rule in rules:
        ts = tree_size(rule.composition)
        td = tree_depth(rule.composition)
        tree_metrics[rule.id] = (ts, td)

    # 5. Build output table
    rows = []
    for rule in rules:
        p_hat, se, n_true = base_rates[rule.id]
        ts, td = tree_metrics[rule.id]

        rows.append({
            "token": rule.token,
            "rule_id": rule.id,
            "family": rule.family,
            "base_rate": round(p_hat, 6),
            "base_rate_se": round(se, 6),
            "base_rate_pct": round(p_hat * 100, 2),
            "n_true": n_true,
            "tree_size": ts,
            "tree_depth": td,
            "n_primitives": len(rule.primitives_used),
            "level": rule.level,
            "name": rule.name,
        })

    # Sort by base rate for easy reading
    rows.sort(key=lambda r: r["base_rate"])

    # 6. Write CSV
    output_path = Path(__file__).parent / args.output
    fieldnames = ["token", "rule_id", "family", "base_rate", "base_rate_se",
                  "base_rate_pct", "n_true", "tree_size", "tree_depth",
                  "n_primitives", "level", "name"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} rules to {output_path}")

    # 7. Print summary table to console
    print(f"\n{'='*110}")
    print(f"{'Token':<8} {'Rule ID':<35} {'Family':<12} {'Base%':>7} {'±SE%':>6} "
          f"{'TrSz':>5} {'TrDp':>5} {'#Prim':>6} {'Lvl':>4}")
    print(f"{'='*110}")

    for r in rows:
        print(f"{r['token']:<8} {r['rule_id']:<35} {r['family']:<12} "
              f"{r['base_rate_pct']:>6.2f}% {r['base_rate_se']*100:>5.2f}% "
              f"{r['tree_size']:>5} {r['tree_depth']:>5} {r['n_primitives']:>6} {r['level']:>4}")

    # 8. Print summary statistics
    base_rates_list = [r["base_rate"] for r in rows]
    tree_sizes = [r["tree_size"] for r in rows]
    tree_depths = [r["tree_depth"] for r in rows]

    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"  Base rate range: [{min(base_rates_list):.4f}, {max(base_rates_list):.4f}]")
    print(f"  Base rate mean:  {sum(base_rates_list)/len(base_rates_list):.4f}")
    print(f"  Base rate median: {sorted(base_rates_list)[len(base_rates_list)//2]:.4f}")
    print(f"  Tree size range: [{min(tree_sizes)}, {max(tree_sizes)}]")
    print(f"  Tree depth range: [{min(tree_depths)}, {max(tree_depths)}]")

    # 9. Flag desiderata violations
    print(f"\n{'='*60}")
    print("DESIDERATA VIOLATIONS")
    print(f"{'='*60}")

    a1_violations = [r for r in rows if r["base_rate"] < 0.01]
    a2_violations = [r for r in rows if r["base_rate"] > 0.60]

    if a1_violations:
        print(f"\n  A1 (base rate < 1%): {len(a1_violations)} rules")
        for r in a1_violations:
            print(f"    {r['token']} ({r['rule_id']}): {r['base_rate_pct']:.2f}%")
    else:
        print(f"\n  A1 (base rate < 1%): NONE — all rules pass")

    if a2_violations:
        print(f"\n  A2 (base rate > 60%): {len(a2_violations)} rules")
        for r in a2_violations:
            print(f"    {r['token']} ({r['rule_id']}): {r['base_rate_pct']:.2f}%")
    else:
        print(f"\n  A2 (base rate > 60%): NONE — all rules pass")

    # Moderate difficulty sweet spot (5-50%)
    sweet_spot = [r for r in rows if 0.05 <= r["base_rate"] <= 0.50]
    print(f"\n  D2 sweet spot (5-50% base rate): {len(sweet_spot)}/{len(rows)} rules")

    print(f"\n{'='*60}")
    print(f"Phase 1 complete. Output: {output_path}")


if __name__ == "__main__":
    main()
