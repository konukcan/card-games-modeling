#!/usr/bin/env python3
"""
Phase 2a: Grammar Enumeration — Discover New Rules via PCFG Search

This script uses DreamCoder's TopDownEnumerator to systematically enumerate
all programs of type HAND → BOOL from the cognitive grammar, then filters
them by base rate to find candidate rules.

The grammar contains 57 primitives (card accessors, list operations,
comparisons, boolean combinators, etc.). The enumerator explores programs
in order of increasing cost (= -log probability under the uniform grammar),
which means shorter/simpler programs are found first.

Each enumerated program is evaluated on a sample of random hands to compute
its base rate. Programs with base rates in the [3%, 50%] sweet spot are
retained as candidate rules.

Key trade-off: This is computationally expensive. With 57 primitives and
max_depth=8, the search space is enormous. We use aggressive timeouts and
program limits to keep it tractable.

Usage:
    cd card-games-modelling/src
    python3 grammar_enumeration.py [--max-cost 30] [--timeout 3600] [--samples 10000]
"""

import sys
import csv
import math
import time
import random
import argparse
import traceback
from pathlib import Path
from typing import List, Set, Tuple, Optional

sys.path.insert(0, str(Path(__file__).parent))

from dreamcoder_core.type_system import HAND, BOOL, arrow
from dreamcoder_core.primitives import build_lean_grammar, build_primitives
from dreamcoder_core.enumeration import TopDownEnumerator
from dreamcoder_core.grammar import Grammar
from rules.cards import Card, Hand, Suit, Rank, RANK_VALUES, sample_hand


# ============================================================================
# Hand sampling
# ============================================================================

def make_deck(deck_size=52):
    """Create a deck of cards for the specified size."""
    all_ranks = list(Rank)
    if deck_size == 52:
        ranks = all_ranks
    elif deck_size == 32:
        ranks = [r for r in all_ranks if RANK_VALUES[r] >= 7]
    elif deck_size == 28:
        ranks = [r for r in all_ranks if RANK_VALUES[r] >= 8]
    else:
        raise ValueError(f"Unsupported deck size: {deck_size}")
    return [Card(suit, rank) for suit in Suit for rank in ranks]


def sample_hands(deck, hand_size, n):
    """Pre-generate n random hands for base rate estimation."""
    return [tuple(random.sample(deck, hand_size)) for _ in range(n)]


# ============================================================================
# Program evaluation
# ============================================================================

def evaluate_program(program, hand):
    """
    Evaluate a HAND→BOOL program on a hand.

    The program is a lambda: when evaluated in the empty environment [],
    it produces a function. We then call that function on the hand.
    Returns True/False if the program successfully evaluates to a bool,
    or None if evaluation fails (type error, runtime error, etc.).
    """
    try:
        fn = program.evaluate([])
        result = fn(hand)
        # Ensure the result is actually a boolean
        if isinstance(result, bool):
            return result
        # Some programs might return truthy/falsy values — coerce carefully
        if result is None:
            return None
        return bool(result)
    except (ValueError, TypeError, ZeroDivisionError, IndexError,
            KeyError, AttributeError, RecursionError, StopIteration):
        return None
    except Exception:
        return None


def compute_base_rate(program, hands):
    """
    Estimate P(rule) by evaluating the program on pre-sampled hands.

    Returns (base_rate, se, n_valid) where:
    - base_rate: fraction of valid evaluations that returned True
    - se: standard error of the estimate
    - n_valid: number of hands where the program evaluated successfully
    """
    n_true = 0
    n_valid = 0

    for hand in hands:
        result = evaluate_program(program, hand)
        if result is not None:
            n_valid += 1
            if result:
                n_true += 1

    if n_valid < 10:
        # Too few valid evaluations — unreliable estimate
        return None, None, n_valid

    p_hat = n_true / n_valid
    se = math.sqrt(p_hat * (1 - p_hat) / n_valid)
    return p_hat, se, n_valid


# ============================================================================
# Deduplication — detect extensionally equivalent programs
# ============================================================================

def compute_fingerprint(program, fingerprint_hands):
    """
    Compute a boolean fingerprint: the pattern of True/False results
    over a fixed set of hands. Two programs with the same fingerprint
    are likely extensionally equivalent (same rule, different syntax).
    """
    bits = []
    for hand in fingerprint_hands:
        result = evaluate_program(program, hand)
        if result is True:
            bits.append('1')
        elif result is False:
            bits.append('0')
        else:
            bits.append('?')
    return ''.join(bits)


# ============================================================================
# Main enumeration loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 2a: Grammar Enumeration")
    parser.add_argument("--max-cost", type=float, default=30.0,
                        help="Maximum program cost to explore (default: 30)")
    parser.add_argument("--max-programs", type=int, default=500_000,
                        help="Max programs to enumerate (default: 500,000)")
    parser.add_argument("--timeout", type=float, default=3600,
                        help="Enumeration timeout in seconds (default: 3600 = 1 hour)")
    parser.add_argument("--samples", type=int, default=10_000,
                        help="Monte Carlo samples for base rate (default: 10,000)")
    parser.add_argument("--fingerprint-size", type=int, default=500,
                        help="Number of hands for dedup fingerprints (default: 500)")
    parser.add_argument("--hand-size", type=int, default=6)
    parser.add_argument("--deck-size", type=int, default=52)
    parser.add_argument("--min-rate", type=float, default=0.03,
                        help="Minimum base rate (default: 3%%)")
    parser.add_argument("--max-rate", type=float, default=0.50,
                        help="Maximum base rate (default: 50%%)")
    parser.add_argument("--max-depth", type=int, default=6,
                        help="Max AST depth (default: 6)")
    parser.add_argument("--output", type=str, default="grammar_candidates.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1,
                        help="Verbosity: 0=quiet, 1=normal, 2=detailed")
    args = parser.parse_args()

    random.seed(args.seed)

    print(f"{'='*80}")
    print(f"Phase 2a: Grammar Enumeration")
    print(f"{'='*80}")
    print(f"  Max cost: {args.max_cost}")
    print(f"  Max programs: {args.max_programs:,}")
    print(f"  Timeout: {args.timeout:.0f}s ({args.timeout/60:.1f} min)")
    print(f"  Max AST depth: {args.max_depth}")
    print(f"  Base rate filter: [{args.min_rate*100:.0f}%, {args.max_rate*100:.0f}%]")
    print(f"  Deck: {args.deck_size}, Hand: {args.hand_size}")
    print(f"  Monte Carlo samples: {args.samples:,}")
    print(f"  Fingerprint size: {args.fingerprint_size}")
    print()

    # 1. Build the grammar
    # The uniform grammar assigns equal probability to all 57 primitives,
    # which means cost = log(57) ≈ 4.04 per primitive usage.
    print("Building grammar...")
    grammar = build_lean_grammar()
    n_prods = len(grammar.productions)
    print(f"  Grammar has {n_prods} productions")
    print(f"  Cost per primitive ≈ {-math.log(1.0/n_prods):.2f} nats")
    print()

    # 2. Pre-sample hands for evaluation
    print("Pre-sampling hands...")
    deck = make_deck(args.deck_size)
    eval_hands = sample_hands(deck, args.hand_size, args.samples)
    fp_hands = sample_hands(deck, args.hand_size, args.fingerprint_size)
    print(f"  {len(eval_hands):,} evaluation hands, {len(fp_hands)} fingerprint hands")
    print()

    # 3. Set up the enumerator
    # We use TopDownEnumerator with memoization for efficiency.
    # The request type is HAND → BOOL: we want programs that take a hand
    # and return a boolean (true/false rule).
    request_type = arrow(HAND, BOOL)
    print(f"Request type: {request_type}")

    enumerator = TopDownEnumerator(
        grammar=grammar,
        max_depth=args.max_depth,
        max_programs=args.max_programs
    )

    # 4. Enumerate and evaluate
    print(f"\nStarting enumeration...")
    start_time = time.time()

    n_enumerated = 0
    n_valid_programs = 0  # Programs that evaluate to bool on ≥10 hands
    n_trivial = 0         # Always True or always False
    n_too_low = 0
    n_too_high = 0
    n_duplicate = 0
    candidates = []

    # Track fingerprints for deduplication
    seen_fingerprints: Set[str] = set()

    # Progress reporting
    last_report_time = start_time
    report_interval = 30  # seconds

    try:
        for program, log_prob in enumerator.enumerate_memoized(
            request_type=request_type,
            max_cost=args.max_cost,
            timeout_seconds=args.timeout,
            depth_limit=args.max_depth
        ):
            n_enumerated += 1
            cost = -log_prob  # Cost = -log_prob (positive)

            # Progress reporting
            now = time.time()
            if now - last_report_time >= report_interval:
                elapsed = now - start_time
                rate = n_enumerated / elapsed if elapsed > 0 else 0
                print(f"  [{elapsed:.0f}s] Enumerated: {n_enumerated:,} | "
                      f"Valid: {n_valid_programs} | Candidates: {len(candidates)} | "
                      f"Rate: {rate:.0f} prog/s | Cost: {cost:.1f}")
                last_report_time = now

            # Verbose per-program logging
            if args.verbose >= 2 and n_enumerated <= 50:
                print(f"    #{n_enumerated}: cost={cost:.2f} | {program}")

            # Evaluate on a quick sample first (fast reject)
            # Use a small subset for the initial check
            quick_hands = eval_hands[:100]
            quick_true = 0
            quick_valid = 0
            for hand in quick_hands:
                r = evaluate_program(program, hand)
                if r is not None:
                    quick_valid += 1
                    if r:
                        quick_true += 1

            # Skip if <50% of hands produce valid results
            if quick_valid < 50:
                continue

            n_valid_programs += 1

            # Quick base rate check — reject obvious trivials
            quick_rate = quick_true / quick_valid if quick_valid > 0 else 0
            if quick_rate == 0.0 or quick_rate == 1.0:
                n_trivial += 1
                continue

            # Quick rate bounds check (with tolerance for sampling noise)
            if quick_rate < args.min_rate * 0.5 or quick_rate > args.max_rate * 1.5:
                if quick_rate < args.min_rate * 0.5:
                    n_too_low += 1
                else:
                    n_too_high += 1
                continue

            # Fingerprint check for deduplication
            fp = compute_fingerprint(program, fp_hands)
            if fp in seen_fingerprints:
                n_duplicate += 1
                continue
            seen_fingerprints.add(fp)

            # Full base rate computation on all samples
            p_hat, se, n_valid = compute_base_rate(program, eval_hands)
            if p_hat is None:
                continue

            # Final rate filter
            if p_hat < args.min_rate:
                n_too_low += 1
                continue
            if p_hat > args.max_rate:
                n_too_high += 1
                continue

            # This is a candidate!
            candidates.append({
                "program": str(program),
                "cost": round(cost, 4),
                "dl_bits": round(cost / math.log(2), 4),  # Convert nats to bits
                "base_rate": round(p_hat, 6),
                "base_rate_pct": round(p_hat * 100, 2),
                "base_rate_se": round(se, 6),
                "n_valid": n_valid,
            })

            if args.verbose >= 1:
                print(f"  ★ CANDIDATE #{len(candidates)}: rate={p_hat*100:.1f}% "
                      f"cost={cost:.1f} DL={cost/math.log(2):.1f}bits | {program}")

    except KeyboardInterrupt:
        print(f"\n[Interrupted by user]")
    except Exception as e:
        print(f"\n[Error during enumeration: {e}]")
        traceback.print_exc()

    # 5. Report results
    elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"ENUMERATION COMPLETE")
    print(f"{'='*80}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Programs enumerated: {n_enumerated:,}")
    print(f"  Rate: {n_enumerated/elapsed:.0f} programs/second")
    print(f"  Valid programs (evaluate to bool): {n_valid_programs:,}")
    print(f"  Trivial (always T/F): {n_trivial:,}")
    print(f"  Too low (<{args.min_rate*100:.0f}%): {n_too_low:,}")
    print(f"  Too high (>{args.max_rate*100:.0f}%): {n_too_high:,}")
    print(f"  Duplicates (same fingerprint): {n_duplicate:,}")
    print(f"  ★ Candidates: {len(candidates)}")

    # 6. Write CSV
    if candidates:
        output_path = Path(__file__).parent / args.output
        fieldnames = ["program", "cost", "dl_bits", "base_rate", "base_rate_pct",
                      "base_rate_se", "n_valid"]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            # Sort by cost (simplest programs first)
            for row in sorted(candidates, key=lambda r: r["cost"]):
                writer.writerow(row)

        print(f"\nWrote {len(candidates)} candidates to {output_path}")

        # Print summary
        print(f"\n{'='*80}")
        print(f"CANDIDATE SUMMARY (sorted by cost)")
        print(f"{'='*80}")
        for c in sorted(candidates, key=lambda r: r["cost"])[:30]:
            print(f"  DL={c['dl_bits']:>6.1f}bits  rate={c['base_rate_pct']:>5.1f}%  {c['program']}")
    else:
        print("\nNo candidates found. Try increasing --max-cost or --timeout.")


if __name__ == "__main__":
    main()
