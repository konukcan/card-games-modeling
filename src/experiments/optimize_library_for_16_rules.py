#!/usr/bin/env python3
"""
Greedy Primitive Library Optimization for 16 Focus Rules.

This script implements the greedy search algorithm from the plan:
1. Start with ALL primitives from lean_primitives.py
2. For each rule, find the shortest program that implements it
3. Try removing primitives one at a time
4. Keep the removal if all rules remain expressible and total MDL doesn't increase

The goal is to find the minimal primitive set that can express all 16 focus rules
while minimizing the total description length (MDL).

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
from dreamcoder_core.type_system import (
    arrow, HAND, BOOL, SUIT, INT, CARD,
    TypeVariable, ListType, BaseType
)

from experiments.focus_rules_16 import FOCUS_RULES, FocusRule

from rules.cards import Card, Hand, Suit, Rank, Color, RANK_VALUES, card_color

# Import helper functions from rules/primitives for extended primitives
from rules.primitives import (
    seq_palindrome, has_arithmetic_progression, uniform_property,
    is_sorted, halves, left_half, right_half
)


# ============================================================================
# EXTENDED PRIMITIVES FOR 16 FOCUS RULES
# ============================================================================

def build_extended_primitives() -> List[Primitive]:
    """
    Build extended primitives that enable the 16 focus rules to be expressed.

    These are domain-specific primitives that directly implement common patterns
    in the focus rules. The greedy algorithm will discover which are essential
    and which can be removed.

    Philosophy: Per user guidance, we explore BEYOND the pre-selected primitives
    to let the algorithm discover what's truly needed.
    """
    prims = []

    # Type imports
    a = TypeVariable(0)
    LIST_SUIT = ListType(SUIT)
    LIST_COLOR = ListType(BaseType('color'))
    COLOR = BaseType('color')

    # =========================================================================
    # 1. PALINDROME PRIMITIVES (for PALINDROME family)
    # =========================================================================

    # Check if suits form palindrome
    prims.append(Primitive(
        'suits_palindrome',
        arrow(HAND, BOOL),
        lambda h: seq_palindrome(lambda c: c.suit)(h)
    ))

    # Check if colors form palindrome
    prims.append(Primitive(
        'colors_palindrome',
        arrow(HAND, BOOL),
        lambda h: seq_palindrome(lambda c: card_color(c))(h)
    ))

    # =========================================================================
    # 2. ARITHMETIC PROGRESSION PRIMITIVES (for ARITH_PROG family)
    # =========================================================================

    # AP with length 3, step 1 (consecutive like 5-6-7)
    prims.append(Primitive(
        'has_ap_3_1',
        arrow(HAND, BOOL),
        lambda h: has_arithmetic_progression(3, 1, False)(h)
    ))

    # AP with length 3, step 2 (skip like 4-6-8)
    prims.append(Primitive(
        'has_ap_3_2',
        arrow(HAND, BOOL),
        lambda h: has_arithmetic_progression(3, 2, False)(h)
    ))

    # =========================================================================
    # 3. ADJACENCY PRIMITIVES (for ADJACENCY family)
    # =========================================================================

    # Every adjacent pair shares rank or suit
    def adj_rank_or_suit_fn(h):
        if len(h) <= 1:
            return True
        for i in range(len(h) - 1):
            c1, c2 = h[i], h[i+1]
            if not (c1.rank == c2.rank or c1.suit == c2.suit):
                return False
        return True

    prims.append(Primitive(
        'adj_rank_or_suit',
        arrow(HAND, BOOL),
        adj_rank_or_suit_fn
    ))

    # Sorted by rank (non-decreasing)
    prims.append(Primitive(
        'sorted_by_rank',
        arrow(HAND, BOOL),
        lambda h: is_sorted(h, lambda c: RANK_VALUES[c.rank], strict=False)
    ))

    # =========================================================================
    # 4. GLOBAL PRIMITIVES (for GLOBAL family)
    # =========================================================================

    # All same color
    prims.append(Primitive(
        'all_same_color',
        arrow(HAND, BOOL),
        lambda h: uniform_property(lambda c: card_color(c))(h)
    ))

    # Majority red (more than half are red)
    def majority_red_fn(h):
        if not h:
            return False
        red_count = sum(1 for c in h if card_color(c) == Color.RED)
        return red_count > len(h) / 2

    prims.append(Primitive(
        'majority_red',
        arrow(HAND, BOOL),
        majority_red_fn
    ))

    # =========================================================================
    # 5. COUNT/PAIRING PRIMITIVES (for COUNT_PAIRING family)
    # =========================================================================

    # Has pair of ranks
    def has_pair_ranks_fn(h):
        ranks = [c.rank for c in h]
        return len(ranks) != len(set(ranks))

    prims.append(Primitive(
        'has_pair_ranks',
        arrow(HAND, BOOL),
        has_pair_ranks_fn
    ))

    # Has pair of suits
    def has_pair_suits_fn(h):
        suits = [c.suit for c in h]
        return len(suits) != len(set(suits))

    prims.append(Primitive(
        'has_pair_suits',
        arrow(HAND, BOOL),
        has_pair_suits_fn
    ))

    # =========================================================================
    # 6. HALVES BICONDITIONAL PRIMITIVES (for HALVES_BICON family)
    # =========================================================================

    # Both halves uniform in color (or both not uniform)
    def halves_same_color_fn(h):
        left, right = halves(h)
        left_uniform = uniform_property(lambda c: card_color(c))(left)
        right_uniform = uniform_property(lambda c: card_color(c))(right)
        return left_uniform == right_uniform

    prims.append(Primitive(
        'halves_same_color',
        arrow(HAND, BOOL),
        halves_same_color_fn
    ))

    # Both halves have heart, or neither does
    def halves_hearts_equal_fn(h):
        left, right = halves(h)
        left_has = any(c.suit == Suit.HEARTS for c in left)
        right_has = any(c.suit == Suit.HEARTS for c in right)
        return left_has == right_has

    prims.append(Primitive(
        'halves_hearts_equal',
        arrow(HAND, BOOL),
        halves_hearts_equal_fn
    ))

    # =========================================================================
    # 7. HALVES COPY PRIMITIVES (for HALVES_COPY family)
    # =========================================================================

    # Right half mirrors left in suits
    def halves_copy_suits_fn(h):
        left, right = halves(h)
        return [c.suit for c in left] == [c.suit for c in right]

    prims.append(Primitive(
        'halves_copy_suits',
        arrow(HAND, BOOL),
        halves_copy_suits_fn
    ))

    # Right half mirrors left in colors
    def halves_copy_colors_fn(h):
        left, right = halves(h)
        return [card_color(c) for c in left] == [card_color(c) for c in right]

    prims.append(Primitive(
        'halves_copy_colors',
        arrow(HAND, BOOL),
        halves_copy_colors_fn
    ))

    # =========================================================================
    # 8. HALVES BOTH PRIMITIVES (for HALVES_BOTH family)
    # =========================================================================

    # Both halves have consecutive triple
    def halves_both_ap3_fn(h):
        left, right = halves(h)
        left_has = has_arithmetic_progression(3, 1, False)(left)
        right_has = has_arithmetic_progression(3, 1, False)(right)
        return left_has and right_has

    prims.append(Primitive(
        'halves_both_ap3',
        arrow(HAND, BOOL),
        halves_both_ap3_fn
    ))

    # Both halves satisfy adjacency
    def halves_both_adj_fn(h):
        left, right = halves(h)
        return adj_rank_or_suit_fn(left) and adj_rank_or_suit_fn(right)

    prims.append(Primitive(
        'halves_both_adj',
        arrow(HAND, BOOL),
        halves_both_adj_fn
    ))

    # =========================================================================
    # 9. UTILITY PRIMITIVES (may help compose solutions)
    # =========================================================================

    # List equality (polymorphic)
    prims.append(Primitive(
        'list_eq',
        arrow(ListType(a), ListType(a), BOOL),
        lambda xs: lambda ys: xs == ys
    ))

    # All same in list
    def all_same_fn(xs):
        if not xs:
            return True
        first = xs[0]
        return all(x == first for x in xs)

    prims.append(Primitive(
        'all_same',
        arrow(ListType(a), BOOL),
        all_same_fn
    ))

    return prims


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class OptimizationConfig:
    """Configuration for the optimization algorithm."""
    max_cost: float = 30.0       # Maximum cost for program search
    timeout_per_rule: float = 30.0  # Timeout per rule (seconds)
    max_programs: int = 100000   # Max programs to enumerate per rule
    lambda_weight: float = 1.0   # Weight for library size in cost function
    verbose: int = 1             # Verbosity level (0=quiet, 1=normal, 2=detailed, 3=debug)


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


# ============================================================================
# MAIN OPTIMIZATION LOOP
# ============================================================================

@dataclass
class OptimizationState:
    """Current state of the optimization."""
    primitives: List[Primitive]
    programs: Dict[str, Program]  # rule_id -> program
    costs: Dict[str, float]       # rule_id -> cost
    mdls: Dict[str, float]        # rule_id -> MDL in bits
    total_mdl: float
    library_cost: float
    combined_score: float


def compute_baseline(
    rules: List[FocusRule],
    primitives: List[Primitive],
    test_hands: List[Hand],
    config: OptimizationConfig
) -> OptimizationState:
    """
    Compute baseline: find shortest program for each rule with full library.
    """
    grammar = uniform_grammar(primitives)

    programs = {}
    costs = {}
    mdls = {}

    print(f"\nFinding shortest programs for {len(rules)} rules...")
    print(f"Grammar size: {len(primitives)} primitives")
    print(f"Max cost: {config.max_cost}, Timeout per rule: {config.timeout_per_rule}s")
    print()

    # Pre-check: How many True/False for each rule?
    print("Rule balance on test hands:")
    for rule in rules:
        true_count = sum(1 for h in test_hands if rule.predicate(h))
        false_count = len(test_hands) - true_count
        print(f"  {rule.id}: {true_count} True, {false_count} False")
    print()

    for i, rule in enumerate(rules, 1):
        print(f"  [{i:2}/{len(rules)}] {rule.id}...", end=" ", flush=True)

        result = find_shortest_program(rule, grammar, test_hands, config)

        if result is None:
            print(f"FAILED (no solution found)")
            continue

        program, cost, n_searched = result
        mdl = description_length(cost)

        programs[rule.id] = program
        costs[rule.id] = cost
        mdls[rule.id] = mdl

        # Show the program found
        print(f"OK (MDL={mdl:.1f} bits, {n_searched} progs)")
        if config.verbose >= 2:
            print(f"       Program: {program}")

    total_mdl = sum(mdls.values())
    library_cost = len(primitives) * config.lambda_weight
    combined_score = total_mdl + library_cost

    return OptimizationState(
        primitives=primitives,
        programs=programs,
        costs=costs,
        mdls=mdls,
        total_mdl=total_mdl,
        library_cost=library_cost,
        combined_score=combined_score
    )


def rank_primitives_by_removability(
    state: OptimizationState,
    rules: List[FocusRule],
    test_hands: List[Hand],
    config: OptimizationConfig
) -> List[Tuple[Primitive, int, bool]]:
    """
    Rank primitives by how likely they are to be removable.

    Returns list of (primitive, usage_count, is_essential) sorted by removability.
    """
    print("\n" + "=" * 70)
    print("PHASE 2: PRIMITIVE USAGE ANALYSIS")
    print("=" * 70)

    # Count usage across all found programs
    usage_counts = defaultdict(int)

    for rule_id, program in state.programs.items():
        # Extract primitive names from the program
        prims_in_prog = extract_primitives_from_program(program)
        for prim_name in prims_in_prog:
            usage_counts[prim_name] += 1

    # Determine essentiality (expensive - only for low-usage primitives initially)
    ranked = []

    print(f"\n{'Primitive':<20} {'Used by':<10} {'Notes':<30}")
    print("-" * 60)

    for prim in state.primitives:
        count = usage_counts.get(prim.name, 0)

        # Only check essentiality for primitives used by 0-2 rules
        # (High-usage primitives are unlikely to be removable)
        if count <= 2:
            is_ess, blocking = is_essential(prim, rules, state.primitives, test_hands, config)
            ess_str = f"Essential for {blocking}" if is_ess else "Removable?"
        else:
            is_ess = False  # Assume not essential, will check during removal
            ess_str = "Check on removal"

        ranked.append((prim, count, is_ess))
        print(f"{prim.name:<20} {count:<10} {ess_str:<30}")

    # Sort by: essential last, then by usage count (ascending)
    ranked.sort(key=lambda x: (x[2], x[1]))

    return ranked


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


def run_greedy_optimization(
    rules: List[FocusRule],
    primitives: List[Primitive],
    test_hands: List[Hand],
    config: OptimizationConfig
) -> OptimizationState:
    """
    Main greedy optimization loop.

    Iteratively tries to remove primitives while maintaining expressibility.
    """
    # Phase 1: Compute baseline
    print("\n" + "=" * 70)
    print("PHASE 1: BASELINE COMPUTATION")
    print("=" * 70)

    state = compute_baseline(rules, primitives, test_hands, config)

    print("\n" + "-" * 70)
    print("BASELINE SUMMARY")
    print("-" * 70)
    print(f"Rules solved: {len(state.programs)}/{len(rules)}")
    print(f"Total MDL: {state.total_mdl:.1f} bits")
    print(f"Library cost: {state.library_cost:.1f} ({len(state.primitives)} × {config.lambda_weight})")
    print(f"Combined score: {state.combined_score:.1f}")

    if len(state.programs) < len(rules):
        print("\nWARNING: Not all rules could be solved! Cannot proceed with optimization.")
        return state

    # Phase 2: Rank primitives by removability
    ranked = rank_primitives_by_removability(state, rules, test_hands, config)

    # Phase 3: Greedy removal
    print("\n" + "=" * 70)
    print("PHASE 3: GREEDY REMOVAL")
    print("=" * 70)

    current_primitives = list(state.primitives)
    current_programs = dict(state.programs)
    current_costs = dict(state.costs)
    current_mdls = dict(state.mdls)
    baseline_score = state.combined_score

    removed_primitives = []
    iteration = 0

    for prim, usage_count, is_essential in ranked:
        if is_essential:
            print(f"\nSkipping {prim.name} (marked essential)")
            continue

        iteration += 1
        print(f"\nIteration {iteration}: Testing removal of '{prim.name}' (used by {usage_count} rules)")

        # Try removing this primitive
        test_primitives = [p for p in current_primitives if p.name != prim.name]
        test_grammar = uniform_grammar(test_primitives)

        # Check if all rules are still expressible
        new_programs = {}
        new_costs = {}
        new_mdls = {}
        all_expressible = True

        for rule in rules:
            result = find_shortest_program(rule, test_grammar, test_hands, config)
            if result is None:
                print(f"  - Rule '{rule.id}' becomes inexpressible")
                all_expressible = False
                break

            program, cost, _ = result
            new_programs[rule.id] = program
            new_costs[rule.id] = cost
            new_mdls[rule.id] = description_length(cost)

        if not all_expressible:
            print(f"  DECISION: KEEP {prim.name} (essential for expressibility)")
            continue

        # Compute new score
        new_total_mdl = sum(new_mdls.values())
        new_library_cost = len(test_primitives) * config.lambda_weight
        new_score = new_total_mdl + new_library_cost

        mdl_change = new_total_mdl - sum(current_mdls.values())
        lib_change = new_library_cost - (len(current_primitives) * config.lambda_weight)
        score_change = new_score - baseline_score

        print(f"  - All {len(rules)} rules still expressible: YES")
        print(f"  - New total MDL: {new_total_mdl:.1f} bits ({mdl_change:+.1f})")
        print(f"  - New library cost: {new_library_cost:.1f} ({lib_change:+.1f})")
        print(f"  - New combined score: {new_score:.1f} ({score_change:+.1f})")

        # Accept if score doesn't increase (or increases only slightly)
        if new_score <= baseline_score * 1.01:  # Allow 1% increase
            print(f"  DECISION: REMOVE {prim.name} (score acceptable)")
            current_primitives = test_primitives
            current_programs = new_programs
            current_costs = new_costs
            current_mdls = new_mdls
            removed_primitives.append(prim.name)
            baseline_score = new_score
        else:
            print(f"  DECISION: KEEP {prim.name} (score increase too large)")

    # Final state
    final_state = OptimizationState(
        primitives=current_primitives,
        programs=current_programs,
        costs=current_costs,
        mdls=current_mdls,
        total_mdl=sum(current_mdls.values()),
        library_cost=len(current_primitives) * config.lambda_weight,
        combined_score=baseline_score
    )

    # Print final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Starting primitives: {len(primitives)}")
    print(f"Final primitives: {len(final_state.primitives)}")
    print(f"Primitives removed ({len(removed_primitives)}): {removed_primitives}")
    print(f"Final MDL: {final_state.total_mdl:.1f} bits")
    print(f"Final combined score: {final_state.combined_score:.1f}")

    # Improvement summary
    initial_score = state.combined_score
    improvement = initial_score - final_state.combined_score
    print(f"\nImprovement: {improvement:.1f} ({100 * improvement / initial_score:.1f}%)")

    print("\nOPTIMIZED LIBRARY:")
    for prim in sorted(final_state.primitives, key=lambda p: p.name):
        print(f"  {prim.name}")

    return final_state


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Optimize primitive library for 16 focus rules")
    parser.add_argument("--verbose", "-v", type=int, default=1, help="Verbosity level (0-3)")
    parser.add_argument("--max-cost", type=float, default=25.0, help="Maximum cost for program search")
    parser.add_argument("--timeout", type=float, default=20.0, help="Timeout per rule (seconds)")
    parser.add_argument("--lambda-weight", type=float, default=1.0, help="Library size weight")
    parser.add_argument("--quick", action="store_true", help="Quick test with reduced parameters")
    args = parser.parse_args()

    # Configuration
    if args.quick:
        config = OptimizationConfig(
            max_cost=15.0,
            timeout_per_rule=10.0,
            max_programs=10000,
            lambda_weight=args.lambda_weight,
            verbose=args.verbose
        )
    else:
        config = OptimizationConfig(
            max_cost=args.max_cost,
            timeout_per_rule=args.timeout,
            lambda_weight=args.lambda_weight,
            verbose=args.verbose
        )

    print("=" * 70)
    print("GREEDY PRIMITIVE LIBRARY OPTIMIZATION")
    print("Target: 16 focus rules (8 families × 2 rules)")
    print("=" * 70)

    # Load primitives
    print("\nLoading primitives...")
    lean_primitives = build_lean_primitives()
    print(f"  Base primitives (lean_primitives.py): {len(lean_primitives)}")

    extended_primitives = build_extended_primitives()
    print(f"  Extended primitives (for 16 focus rules): {len(extended_primitives)}")

    # Combine both sets
    primitives = lean_primitives + extended_primitives
    print(f"  Total primitives: {len(primitives)}")

    # Show extended primitives
    print("\nExtended primitives added:")
    for p in extended_primitives:
        print(f"  - {p.name}")

    # Load rules
    print(f"\nLoaded {len(FOCUS_RULES)} focus rules from 8 families")

    # Generate test hands
    print("\nGenerating test hands...")
    test_hands = generate_test_hands(n_examples=100)  # Need enough for all structured hands + random
    print(f"Generated {len(test_hands)} test hands")

    # Run optimization
    print(f"\nConfiguration:")
    print(f"  Max cost: {config.max_cost}")
    print(f"  Timeout per rule: {config.timeout_per_rule}s")
    print(f"  Lambda weight: {config.lambda_weight}")

    start_time = time.time()
    final_state = run_greedy_optimization(FOCUS_RULES, primitives, test_hands, config)
    elapsed = time.time() - start_time

    print(f"\nTotal optimization time: {elapsed:.1f} seconds")

    return final_state


if __name__ == "__main__":
    main()
