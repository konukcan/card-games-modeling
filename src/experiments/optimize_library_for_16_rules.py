#!/usr/bin/env python3
"""
Primitive Library Analysis for 16 Focus Rules.

This script analyzes which primitives from the lean_primitives.py library are
ESSENTIAL for expressing the 16 focus rules. It implements a Stitch-style
compression analysis:

1. Start with the 57 base primitives from lean_primitives.py (defines expressibility)
2. Find the shortest program for each rule
3. Identify which primitives are used by ≥2 rules (ESSENTIAL - reusable)
4. Identify which primitives are used by only 1 rule (SPECIALIZED)
5. Compute compression metrics

Key insight: We do NOT add rule-specific primitives. The base library defines
what's expressible. We're discovering which parts of it are actually used.

The reuse constraint (≥2 rules) is Stitch's key innovation: primitives that
are only used once don't help with compression - they just add to library cost.

Usage:
    python optimize_library_for_16_rules.py [--verbose] [--max-cost COST] [--timeout SECONDS]
"""

import sys
import math
import time
import argparse
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dreamcoder_core.lean_primitives import build_lean_primitives
from dreamcoder_core.grammar import uniform_grammar, Grammar, Production
from dreamcoder_core.program import Primitive, Program
from dreamcoder_core.enumeration import TopDownEnumerator, EnumerationResult
from dreamcoder_core.type_system import arrow, HAND, BOOL

from experiments.focus_rules_16 import FOCUS_RULES, FocusRule

from rules.cards import Card, Hand, Suit, Rank, Color, RANK_VALUES, card_color


# NOTE: Extended primitives REMOVED
# The previous version added 18 rule-specific primitives (suits_palindrome,
# has_ap_3_1, etc.) that essentially hardcoded the rules as single primitives.
# This defeated the purpose of program synthesis.
#
# The correct approach: Use ONLY the base primitives from lean_primitives.py.
# These define what's expressible. The analysis discovers which ones are
# actually used and reused across rules.


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AnalysisConfig:
    """Configuration for the library analysis."""
    max_cost: float = 30.0       # Maximum cost for program search
    timeout_per_rule: float = 30.0  # Timeout per rule (seconds)
    max_programs: int = 100000   # Max programs to enumerate per rule
    min_reuse: int = 2           # Minimum rules using a primitive to count as "essential"
    verbose: int = 1             # Verbosity level (0=quiet, 1=normal, 2=detailed, 3=debug)


# Alias for backward compatibility
OptimizationConfig = AnalysisConfig


# ============================================================================
# TEST EXAMPLES GENERATION
# ============================================================================

def generate_test_hands(n_examples: int = 100) -> List[Hand]:
    """
    Generate diverse test hands for rule evaluation.

    Key insight: Random hands don't cover all rule behaviors well.
    Some rules are rare (like suits_palindrome) or always true (like has_pair_suits
    on 6-card hands). We need structured examples.

    IMPORTANT: Structured hands are added FIRST to ensure coverage,
    then random hands fill the remaining slots.
    """
    import random
    random.seed(42)  # Reproducibility

    hands = []
    all_cards = [Card(s, r) for s in Suit for r in Rank]

    # =========================================================================
    # STRUCTURED HANDS FIRST (ensure coverage for rare rules)
    # =========================================================================

    # Guaranteed adjacent-matching hands (adj_rank_or_suit = True)
    # Ensure each consecutive pair shares rank OR suit
    for _ in range(5):
        hand = []
        suit_cycle = list(Suit)
        random.shuffle(suit_cycle)
        rank_cycle = list(Rank)[:6]
        random.shuffle(rank_cycle)

        # Start with first card
        hand.append(Card(suit_cycle[0], rank_cycle[0]))

        # Build chain where each card shares rank or suit with previous
        for i in range(1, 6):
            prev = hand[-1]
            if random.random() < 0.5:
                # Same suit, different rank
                hand.append(Card(prev.suit, rank_cycle[i]))
            else:
                # Different suit, same rank
                hand.append(Card(suit_cycle[i % 4], prev.rank))
        hands.append(hand)

    # Halves copy suits: first 3 suits = last 3 suits (guaranteed)
    for _ in range(5):
        suits = [random.choice(list(Suit)) for _ in range(3)]
        suits = suits + suits  # Exact copy
        ranks = random.sample(list(Rank), 6)
        hands.append([Card(suits[i], ranks[i]) for i in range(6)])

    # Halves both adjacent: both halves satisfy adjacency
    # Each half (3 cards) has consecutive pairs sharing rank or suit
    for _ in range(5):
        left_half = []
        right_half = []

        # Build left half with adjacency
        left_half.append(Card(random.choice(list(Suit)), random.choice(list(Rank))))
        for _ in range(2):
            prev = left_half[-1]
            if random.random() < 0.5:
                left_half.append(Card(prev.suit, random.choice([r for r in Rank if r != prev.rank])))
            else:
                left_half.append(Card(random.choice([s for s in Suit if s != prev.suit]), prev.rank))

        # Build right half with adjacency
        right_half.append(Card(random.choice(list(Suit)), random.choice(list(Rank))))
        for _ in range(2):
            prev = right_half[-1]
            if random.random() < 0.5:
                right_half.append(Card(prev.suit, random.choice([r for r in Rank if r != prev.rank])))
            else:
                right_half.append(Card(random.choice([s for s in Suit if s != prev.suit]), prev.rank))

        hands.append(left_half + right_half)

    # Halves both AP3: both halves have arithmetic progression
    # For 3-card halves, need 3 cards with consecutive ranks
    for _ in range(5):
        # Left half: 3 consecutive ranks
        left_start = random.randint(0, len(list(Rank)) - 3)
        left_ranks = list(Rank)[left_start:left_start + 3]
        left_suits = [random.choice(list(Suit)) for _ in range(3)]
        left_half = [Card(left_suits[i], left_ranks[i]) for i in range(3)]

        # Right half: 3 consecutive ranks
        right_start = random.randint(0, len(list(Rank)) - 3)
        right_ranks = list(Rank)[right_start:right_start + 3]
        right_suits = [random.choice(list(Suit)) for _ in range(3)]
        right_half = [Card(right_suits[i], right_ranks[i]) for i in range(3)]

        hands.append(left_half + right_half)

    # Palindrome-friendly hands (suits: ABCCBA pattern)
    for _ in range(5):
        suits = [random.choice(list(Suit)) for _ in range(3)]
        suits = suits + suits[::-1]  # Mirror
        ranks = random.sample(list(Rank), 6)
        hands.append([Card(suits[i], ranks[i]) for i in range(6)])

    # Non-palindrome hands with distinct suits
    for _ in range(5):
        suits = random.sample(list(Suit), 4) + random.sample(list(Suit), 2)
        random.shuffle(suits)
        ranks = random.sample(list(Rank), 6)
        hands.append([Card(suits[i], ranks[i]) for i in range(6)])

    # Uniform color hands (all red or all black)
    red_suits = [Suit.HEARTS, Suit.DIAMONDS]
    black_suits = [Suit.CLUBS, Suit.SPADES]
    for _ in range(5):
        color_suits = random.choice([red_suits, black_suits])
        suits = [random.choice(color_suits) for _ in range(6)]
        ranks = random.sample(list(Rank), 6)
        hands.append([Card(suits[i], ranks[i]) for i in range(6)])

    # Sorted hands (by rank)
    for _ in range(5):
        ranks = sorted(random.sample(list(Rank), 6), key=lambda r: list(Rank).index(r))
        suits = [random.choice(list(Suit)) for _ in range(6)]
        hands.append([Card(suits[i], ranks[i]) for i in range(6)])

    # Arithmetic progression hands (consecutive ranks)
    for _ in range(5):
        start_idx = random.randint(0, len(list(Rank)) - 6)
        ranks = list(Rank)[start_idx:start_idx + 6]
        suits = [random.choice(list(Suit)) for _ in range(6)]
        hands.append([Card(suits[i], ranks[i]) for i in range(6)])

    # Halves-matching hands (first 3 suits = last 3 suits)
    for _ in range(5):
        first_suits = [random.choice(list(Suit)) for _ in range(3)]
        suits = first_suits + first_suits  # Same pattern
        ranks = random.sample(list(Rank), 6)
        hands.append([Card(suits[i], ranks[i]) for i in range(6)])

    # Adjacent-matching hands (consecutive cards share rank or suit)
    for _ in range(5):
        hand = []
        prev_suit = random.choice(list(Suit))
        prev_rank = random.choice(list(Rank))
        hand.append(Card(prev_suit, prev_rank))
        for _ in range(5):
            if random.random() < 0.5:
                # Match suit
                new_rank = random.choice([r for r in Rank if r != prev_rank])
                hand.append(Card(prev_suit, new_rank))
                prev_rank = new_rank
            else:
                # Match rank
                new_suit = random.choice([s for s in Suit if s != prev_suit])
                hand.append(Card(new_suit, prev_rank))
                prev_suit = new_suit
        hands.append(hand)

    # Pair hands (at least two cards with same rank)
    for _ in range(5):
        pair_rank = random.choice(list(Rank))
        other_ranks = random.sample([r for r in Rank if r != pair_rank], 4)
        ranks = [pair_rank, pair_rank] + other_ranks
        random.shuffle(ranks)
        suits = [random.choice(list(Suit)) for _ in range(6)]
        hands.append([Card(suits[i], ranks[i]) for i in range(6)])

    # No-pair hands (all distinct ranks) - need smaller hand or specific ranks
    # Note: With 6 cards, always have pair_suits (pigeonhole), but can have no pair_ranks
    for _ in range(5):
        ranks = random.sample(list(Rank), 6)  # All distinct
        suits = [random.choice(list(Suit)) for _ in range(6)]
        hands.append([Card(suits[i], ranks[i]) for i in range(6)])

    # Fill remaining slots with random hands
    while len(hands) < n_examples:
        hand = random.sample(all_cards, 6)
        hands.append(hand)

    return hands[:n_examples]


def evaluate_rule_on_hands(rule: FocusRule, hands: List[Hand]) -> List[Tuple[Hand, bool]]:
    """Evaluate a rule on test hands to get (input, output) examples."""
    return [(hand, rule.predicate(hand)) for hand in hands]


# ============================================================================
# PROGRAM EVALUATION
# ============================================================================

def evaluate_program(program: Program, hand: Hand) -> Any:
    """
    Evaluate a synthesized program on a hand.

    Programs are lambda expressions: λh. body
    We evaluate by providing the hand as the argument.
    """
    try:
        fn = program.evaluate([])
        return fn(hand)
    except Exception as e:
        return None


def program_matches_rule(program: Program, rule: FocusRule, test_hands: List[Hand]) -> bool:
    """
    Check if a program matches a rule on all test hands.

    A program matches if:
    - It doesn't raise exceptions on any test hand
    - Its output equals the rule's output on all test hands
    """
    for hand in test_hands:
        try:
            expected = rule.predicate(hand)
            actual = evaluate_program(program, hand)
            if actual != expected:
                return False
        except Exception:
            return False
    return True


# ============================================================================
# CORE FUNCTIONS: find_shortest_program, description_length, is_essential
# ============================================================================

def find_shortest_program(
    rule: FocusRule,
    grammar: Grammar,
    test_hands: List[Hand],
    config: OptimizationConfig
) -> Optional[Tuple[Program, float, int]]:
    """
    Find the shortest program that implements a rule.

    Uses enumeration to search through the program space.

    Returns:
        Tuple of (program, cost, programs_searched) if found, None otherwise
    """
    request_type = arrow(HAND, BOOL)  # All rules are Hand -> Bool

    enumerator = TopDownEnumerator(
        grammar,
        max_depth=8,
        max_programs=config.max_programs
    )

    start_time = time.time()
    programs_searched = 0

    for program, log_prob in enumerator.enumerate(
        request_type,
        max_cost=config.max_cost,
        timeout_seconds=config.timeout_per_rule
    ):
        programs_searched += 1

        # Check if this program matches the rule
        if program_matches_rule(program, rule, test_hands):
            cost = -log_prob  # Cost = negative log probability
            return (program, cost, programs_searched)

        # Timeout check
        if time.time() - start_time > config.timeout_per_rule:
            break

    return None


def description_length(cost: float) -> float:
    """
    Convert cost (negative log probability in nats) to MDL in bits.

    MDL = -log₂(P(program)) = cost / log(2)
    """
    return cost / math.log(2)


def is_essential(
    primitive: Primitive,
    rules: List[FocusRule],
    remaining_primitives: List[Primitive],
    test_hands: List[Hand],
    config: OptimizationConfig
) -> Tuple[bool, Optional[str]]:
    """
    Check if removing a primitive makes any rule inexpressible.

    Returns:
        (is_essential, blocking_rule_id)
        - is_essential: True if some rule cannot be expressed without this primitive
        - blocking_rule_id: The ID of the first rule that cannot be expressed (if any)
    """
    # Build grammar without this primitive
    remaining_prims = [p for p in remaining_primitives if p.name != primitive.name]
    test_grammar = uniform_grammar(remaining_prims)

    # Check each rule
    for rule in rules:
        result = find_shortest_program(rule, test_grammar, test_hands, config)
        if result is None:
            return (True, rule.id)

    return (False, None)


def extract_primitives_from_program(program: Program) -> Set[str]:
    """Extract all primitive names used in a program."""
    prims = set()

    def traverse(p):
        if isinstance(p, Primitive):
            prims.add(p.name)
        elif hasattr(p, 'func'):  # Application
            traverse(p.func)
            traverse(p.arg)
        elif hasattr(p, 'body'):  # Abstraction
            traverse(p.body)

    traverse(program)
    return prims


# ============================================================================
# COMPRESSION ANALYSIS (Stitch-style)
# ============================================================================

@dataclass
class PrimitiveUsageStats:
    """Statistics about primitive usage across rules."""
    name: str
    rules_using: List[str]  # Rule IDs that use this primitive
    total_occurrences: int  # Total times it appears (counting duplicates)
    is_essential: bool      # True if used by >= min_reuse rules
    is_specialized: bool    # True if used by exactly 1 rule


@dataclass
class CompressionAnalysis:
    """Results of Stitch-style compression analysis."""
    programs_by_rule: Dict[str, Tuple[Program, float]]  # rule_id -> (program, cost)
    primitive_stats: Dict[str, PrimitiveUsageStats]     # prim_name -> stats
    essential_primitives: List[str]     # Used by >= min_reuse rules
    specialized_primitives: List[str]   # Used by exactly 1 rule
    unused_primitives: List[str]        # Not used by any rule
    total_program_mdl: float            # Sum of program MDLs
    essential_library_cost: float       # Cost of essential primitives only
    full_library_cost: float            # Cost of all used primitives


def count_rules_using_primitive(prim_name: str, programs_by_rule: Dict[str, Tuple[Program, float]]) -> List[str]:
    """Return list of rule IDs that use this primitive."""
    rules = []
    for rule_id, (program, _) in programs_by_rule.items():
        prims_in_prog = extract_primitives_from_program(program)
        if prim_name in prims_in_prog:
            rules.append(rule_id)
    return rules


def compute_compression_analysis(
    rules: List[FocusRule],
    primitives: List[Primitive],
    test_hands: List[Hand],
    config: AnalysisConfig
) -> CompressionAnalysis:
    """
    Analyze which primitives are essential vs specialized.

    This is the core Stitch-style analysis:
    - ESSENTIAL: Used by >= min_reuse rules (default 2). These provide compression.
    - SPECIALIZED: Used by exactly 1 rule. These add library cost without reuse benefit.
    - UNUSED: Not used by any rule.
    """
    grammar = uniform_grammar(primitives)

    print("\n" + "=" * 70)
    print("PHASE 1: FINDING SHORTEST PROGRAMS")
    print("=" * 70)
    print(f"Grammar size: {len(primitives)} primitives")
    print(f"Max cost: {config.max_cost}, Timeout per rule: {config.timeout_per_rule}s")
    print()

    # Find shortest program for each rule
    programs_by_rule = {}
    failed_rules = []

    for i, rule in enumerate(rules, 1):
        print(f"  [{i:2}/{len(rules)}] {rule.id}...", end=" ", flush=True)

        result = find_shortest_program(rule, grammar, test_hands, config)

        if result is None:
            print("FAILED (no solution found)")
            failed_rules.append(rule.id)
            continue

        program, cost, n_searched = result
        mdl = description_length(cost)
        programs_by_rule[rule.id] = (program, cost)

        print(f"OK (MDL={mdl:.1f} bits, {n_searched} progs)")
        if config.verbose >= 2:
            print(f"       Program: {program}")

    if failed_rules:
        print(f"\nWARNING: {len(failed_rules)} rules could not be solved: {failed_rules}")

    # Phase 2: Analyze primitive usage
    print("\n" + "=" * 70)
    print("PHASE 2: PRIMITIVE USAGE ANALYSIS")
    print("=" * 70)

    primitive_stats = {}
    essential_primitives = []
    specialized_primitives = []
    unused_primitives = []

    for prim in primitives:
        rules_using = count_rules_using_primitive(prim.name, programs_by_rule)
        n_rules = len(rules_using)

        # Count total occurrences (including duplicates within programs)
        total_occ = 0
        for rule_id, (program, _) in programs_by_rule.items():
            prims_in_prog = extract_primitives_from_program(program)
            if prim.name in prims_in_prog:
                total_occ += 1  # Count once per rule for now

        is_essential = n_rules >= config.min_reuse
        is_specialized = n_rules == 1

        primitive_stats[prim.name] = PrimitiveUsageStats(
            name=prim.name,
            rules_using=rules_using,
            total_occurrences=total_occ,
            is_essential=is_essential,
            is_specialized=is_specialized
        )

        if n_rules == 0:
            unused_primitives.append(prim.name)
        elif is_essential:
            essential_primitives.append(prim.name)
        else:
            specialized_primitives.append(prim.name)

    # Compute costs
    total_program_mdl = sum(description_length(cost) for _, cost in programs_by_rule.values())
    essential_library_cost = len(essential_primitives)
    full_library_cost = len(essential_primitives) + len(specialized_primitives)

    return CompressionAnalysis(
        programs_by_rule=programs_by_rule,
        primitive_stats=primitive_stats,
        essential_primitives=essential_primitives,
        specialized_primitives=specialized_primitives,
        unused_primitives=unused_primitives,
        total_program_mdl=total_program_mdl,
        essential_library_cost=essential_library_cost,
        full_library_cost=full_library_cost
    )


def print_compression_report(analysis: CompressionAnalysis, config: AnalysisConfig):
    """Print a detailed compression analysis report."""
    print("\n" + "=" * 70)
    print("COMPRESSION ANALYSIS REPORT")
    print("=" * 70)

    # Summary statistics
    n_rules = len(analysis.programs_by_rule)
    print(f"\nRules analyzed: {n_rules}")
    print(f"Total program MDL: {analysis.total_program_mdl:.1f} bits")
    print(f"Average MDL per rule: {analysis.total_program_mdl / n_rules:.1f} bits")

    # Primitive breakdown
    print(f"\n{'Category':<20} {'Count':<10} {'Description'}")
    print("-" * 60)
    print(f"{'ESSENTIAL':<20} {len(analysis.essential_primitives):<10} Used by >= {config.min_reuse} rules")
    print(f"{'SPECIALIZED':<20} {len(analysis.specialized_primitives):<10} Used by exactly 1 rule")
    print(f"{'UNUSED':<20} {len(analysis.unused_primitives):<10} Not used by any rule")

    # Essential primitives (the good ones - provide compression)
    print("\n" + "-" * 70)
    print("ESSENTIAL PRIMITIVES (provide compression through reuse)")
    print("-" * 70)

    # Sort by number of rules using
    essential_sorted = sorted(
        [(name, analysis.primitive_stats[name]) for name in analysis.essential_primitives],
        key=lambda x: -len(x[1].rules_using)
    )

    print(f"\n{'Primitive':<20} {'Rules':<8} {'Which rules'}")
    print("-" * 60)
    for name, stats in essential_sorted:
        rules_str = ", ".join(stats.rules_using[:5])
        if len(stats.rules_using) > 5:
            rules_str += f"... (+{len(stats.rules_using)-5} more)"
        print(f"{name:<20} {len(stats.rules_using):<8} {rules_str}")

    # Specialized primitives (used by only 1 rule)
    if analysis.specialized_primitives:
        print("\n" + "-" * 70)
        print("SPECIALIZED PRIMITIVES (no reuse benefit - consider removing)")
        print("-" * 70)

        print(f"\n{'Primitive':<20} {'Used by rule'}")
        print("-" * 40)
        for name in sorted(analysis.specialized_primitives):
            stats = analysis.primitive_stats[name]
            print(f"{name:<20} {stats.rules_using[0]}")

    # Compression metrics
    print("\n" + "-" * 70)
    print("COMPRESSION METRICS")
    print("-" * 70)

    print(f"\nStitch-style MDL breakdown:")
    print(f"  Program descriptions: {analysis.total_program_mdl:.1f} bits")
    print(f"  Essential library cost: {analysis.essential_library_cost} primitives")
    print(f"  Full library cost: {analysis.full_library_cost} primitives")

    # Compute compression ratio
    if analysis.full_library_cost > 0:
        compression = analysis.essential_library_cost / analysis.full_library_cost
        print(f"\n  Compression ratio (essential/full): {compression:.1%}")
        print(f"  Potential savings: {analysis.full_library_cost - analysis.essential_library_cost} primitives")

    # Rules by complexity
    print("\n" + "-" * 70)
    print("RULES BY COMPLEXITY (MDL)")
    print("-" * 70)

    rules_by_mdl = sorted(
        [(rule_id, description_length(cost)) for rule_id, (_, cost) in analysis.programs_by_rule.items()],
        key=lambda x: -x[1]
    )

    print(f"\n{'Rule':<30} {'MDL (bits)':<12} {'Primitives used'}")
    print("-" * 70)
    for rule_id, mdl in rules_by_mdl:
        program, _ = analysis.programs_by_rule[rule_id]
        prims = extract_primitives_from_program(program)
        prims_str = ", ".join(sorted(prims)[:5])
        if len(prims) > 5:
            prims_str += f"... (+{len(prims)-5})"
        print(f"{rule_id:<30} {mdl:<12.1f} {prims_str}")


def run_library_analysis(
    rules: List[FocusRule],
    primitives: List[Primitive],
    test_hands: List[Hand],
    config: AnalysisConfig
) -> CompressionAnalysis:
    """
    Main analysis function - replaces the old greedy optimization.

    This function:
    1. Finds shortest programs using the FULL base library
    2. Analyzes which primitives are essential (reused) vs specialized
    3. Reports compression metrics

    It does NOT try to remove primitives - it just discovers which ones matter.
    """
    print("\n" + "=" * 70)
    print("PRIMITIVE LIBRARY ANALYSIS")
    print("Using base library from lean_primitives.py")
    print("=" * 70)

    analysis = compute_compression_analysis(rules, primitives, test_hands, config)
    print_compression_report(analysis, config)

    return analysis


# For backward compatibility
def run_greedy_optimization(
    rules: List[FocusRule],
    primitives: List[Primitive],
    test_hands: List[Hand],
    config: OptimizationConfig
) -> CompressionAnalysis:
    """Backward-compatible wrapper - now calls run_library_analysis."""
    return run_library_analysis(rules, primitives, test_hands, config)


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Analyze primitive library usage for 16 focus rules (Stitch-style compression analysis)"
    )
    parser.add_argument("--verbose", "-v", type=int, default=1, help="Verbosity level (0-3)")
    parser.add_argument("--max-cost", type=float, default=25.0, help="Maximum cost for program search")
    parser.add_argument("--timeout", type=float, default=20.0, help="Timeout per rule (seconds)")
    parser.add_argument("--min-reuse", type=int, default=2, help="Minimum rules using a primitive to count as essential")
    parser.add_argument("--quick", action="store_true", help="Quick test with reduced parameters")
    args = parser.parse_args()

    # Configuration
    if args.quick:
        config = AnalysisConfig(
            max_cost=15.0,
            timeout_per_rule=10.0,
            max_programs=10000,
            min_reuse=args.min_reuse,
            verbose=args.verbose
        )
    else:
        config = AnalysisConfig(
            max_cost=args.max_cost,
            timeout_per_rule=args.timeout,
            min_reuse=args.min_reuse,
            verbose=args.verbose
        )

    print("=" * 70)
    print("PRIMITIVE LIBRARY ANALYSIS FOR 16 FOCUS RULES")
    print("(Stitch-style compression analysis)")
    print("=" * 70)
    print()
    print("This analysis uses ONLY the base primitives from lean_primitives.py.")
    print("No rule-specific primitives are added.")
    print()
    print(f"Reuse threshold: A primitive is 'essential' if used by >= {config.min_reuse} rules")
    print("=" * 70)

    # Load primitives - ONLY from lean_primitives.py
    print("\nLoading primitives...")
    primitives = build_lean_primitives()
    print(f"  Base primitives (lean_primitives.py): {len(primitives)}")

    # Load rules
    print(f"\nLoaded {len(FOCUS_RULES)} focus rules from 8 families:")
    for family in ["PALINDROME", "ARITH_PROG", "ADJACENCY", "GLOBAL",
                   "COUNT_PAIRING", "HALVES_BICON", "HALVES_COPY", "HALVES_BOTH"]:
        rules_in_family = [r.id for r in FOCUS_RULES if r.family == family]
        print(f"  {family}: {', '.join(rules_in_family)}")

    # Generate test hands
    print("\nGenerating test hands...")
    test_hands = generate_test_hands(n_examples=100)
    print(f"Generated {len(test_hands)} test hands")

    # Show balance for each rule
    print("\nRule balance on test hands:")
    for rule in FOCUS_RULES:
        true_count = sum(1 for h in test_hands if rule.predicate(h))
        false_count = len(test_hands) - true_count
        print(f"  {rule.id}: {true_count} True, {false_count} False")

    # Run analysis
    print(f"\nConfiguration:")
    print(f"  Max cost: {config.max_cost}")
    print(f"  Timeout per rule: {config.timeout_per_rule}s")
    print(f"  Min reuse for essential: {config.min_reuse}")

    start_time = time.time()
    analysis = run_library_analysis(FOCUS_RULES, primitives, test_hands, config)
    elapsed = time.time() - start_time

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"\nAnalysis completed in {elapsed:.1f} seconds")
    print(f"\nKey findings:")
    print(f"  - {len(analysis.programs_by_rule)}/{len(FOCUS_RULES)} rules successfully expressed")
    print(f"  - {len(analysis.essential_primitives)} essential primitives (used by >= {config.min_reuse} rules)")
    print(f"  - {len(analysis.specialized_primitives)} specialized primitives (used by 1 rule)")
    print(f"  - {len(analysis.unused_primitives)} unused primitives")

    if analysis.essential_primitives:
        print(f"\nEssential primitives: {', '.join(sorted(analysis.essential_primitives))}")

    if analysis.specialized_primitives:
        print(f"\nSpecialized primitives: {', '.join(sorted(analysis.specialized_primitives))}")

    return analysis


if __name__ == "__main__":
    main()
