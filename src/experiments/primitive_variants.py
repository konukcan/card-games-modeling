#!/usr/bin/env python3
"""
Primitive Library Variants for Factorial Experiment

This module provides three variants of the primitive library:

1. LEAN (67 primitives): Current "cognitive realism" library
   - High-level direct queries (all_same_suit, count_color, etc.)
   - Domain aggregates (sum_ranks, max_rank, min_rank)
   - No fold (replaced by aggregates)

2. LEAN_PLUS_FOLD (78 primitives): Lean + DreamCoder primitives
   - Everything in Lean
   - Plus: fold, foldr, cons, empty, tail, is_empty
   - Plus: all_true, any_true (boolean aggregation)
   - Plus: pair, fst, snd (tuple operations from Yang & Piantadosi)

3. MINIMAL (48 primitives): Forces abstraction learning
   - Removes direct queries (has_suit, all_same_suit, etc.)
   - Removes aggregates (sum_ranks, max_rank, min_rank)
   - Removes convenience slicing (first_half, second_half, etc.)
   - Model must LEARN these patterns through compression

Hypothesis:
- Lean: Fast initial learning, less room for abstraction discovery
- Lean+Fold: Universal primitives enable more expressive programs
- Minimal: Slower start, but richer library learning potential
"""

import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.type_system import (
    Type, BaseType, Arrow, ListType, TypeVariable,
    BOOL, INT, CARD, SUIT, RANK, HAND, arrow,
    LIST_INT, LIST_BOOL
)
from dreamcoder_core.program import Primitive
from dreamcoder_core.grammar import Grammar, uniform_grammar

# Import from lean primitives
from dreamcoder_core.lean_primitives import (
    make_constants, make_card_accessors, make_position_ops,
    make_list_slicing, make_direct_queries, make_aggregates,
    make_comparisons, make_boolean_ops, make_higher_order, make_arithmetic,
    COLOR
)

# Import card domain
from rules.cards import Card, Suit, Rank, RANK_VALUES, card_color


# ============================================================================
# VARIANT 1: LEAN (Current - 67 primitives)
# ============================================================================

def build_lean_primitives() -> List[Primitive]:
    """
    Build the standard lean primitive library (67 primitives).

    This is the "cognitive realism" library with:
    - Direct queries: all_same_suit, count_color, etc.
    - Domain aggregates: sum_ranks, max_rank, min_rank
    - No fold (replaced by aggregates)
    """
    prims = []
    prims.extend(make_constants())       # 21
    prims.extend(make_card_accessors())  # 4
    prims.extend(make_position_ops())    # 5
    prims.extend(make_list_slicing())    # 7
    prims.extend(make_direct_queries())  # 9
    prims.extend(make_aggregates())      # 3
    prims.extend(make_comparisons())     # 6
    prims.extend(make_boolean_ops())     # 4
    prims.extend(make_higher_order())    # 5
    prims.extend(make_arithmetic())      # 3
    return prims


def build_lean_grammar() -> Grammar:
    """Build grammar from lean primitives."""
    return uniform_grammar(build_lean_primitives())


# ============================================================================
# VARIANT 2: LEAN + FOLD (78 primitives)
# ============================================================================

def make_fold_primitives() -> List[Primitive]:
    """
    DreamCoder's universal list primitives.

    These enable expressing any list computation:
    - fold: left-to-right iteration with accumulator
    - foldr: right-to-left iteration
    - cons/empty/tail: list construction
    """
    prims = []

    a = TypeVariable(0)
    b = TypeVariable(1)

    # Fold left: (acc -> elem -> acc) -> acc -> list -> acc
    def safe_fold(f):
        def fold_with_init(init):
            def fold_over_list(xs):
                acc = init
                for x in xs:
                    try:
                        acc = f(acc)(x)
                    except Exception:
                        return acc
                return acc
            return fold_over_list
        return fold_with_init

    prims.append(Primitive(
        'fold',
        arrow(arrow(b, a, b), b, ListType(a), b),
        safe_fold
    ))

    # Fold right: (elem -> acc -> acc) -> acc -> list -> acc
    def safe_foldr(f):
        def foldr_with_init(init):
            def foldr_over_list(xs):
                acc = init
                for x in reversed(xs):
                    try:
                        acc = f(x)(acc)
                    except Exception:
                        return acc
                return acc
            return foldr_over_list
        return foldr_with_init

    prims.append(Primitive(
        'foldr',
        arrow(arrow(a, b, b), b, ListType(a), b),
        safe_foldr
    ))

    # Cons: prepend element to list
    prims.append(Primitive(
        'cons',
        arrow(a, ListType(a), ListType(a)),
        lambda x: lambda xs: [x] + list(xs)
    ))

    # Empty list
    prims.append(Primitive(
        'empty',
        ListType(a),
        []
    ))

    # Tail: all but first element
    prims.append(Primitive(
        'tail',
        arrow(ListType(a), ListType(a)),
        lambda xs: xs[1:] if len(xs) > 0 else []
    ))

    # Is empty?
    prims.append(Primitive(
        'is_empty',
        arrow(ListType(a), BOOL),
        lambda xs: len(xs) == 0
    ))

    return prims


def make_bool_aggregators() -> List[Primitive]:
    """
    Boolean list aggregators.

    Equivalent to (fold and true xs) and (fold or false xs),
    but provided as direct primitives for efficiency.
    """
    prims = []

    prims.append(Primitive(
        'all_true',
        arrow(LIST_BOOL, BOOL),
        lambda xs: all(xs)
    ))

    prims.append(Primitive(
        'any_true',
        arrow(LIST_BOOL, BOOL),
        lambda xs: any(xs)
    ))

    return prims


def make_pair_primitives() -> List[Primitive]:
    """
    Tuple operations from Yang & Piantadosi (2022).

    For state threading in recursive computations.
    We represent pairs as 2-element lists.
    """
    prims = []

    a = TypeVariable(0)

    prims.append(Primitive(
        'pair',
        arrow(a, a, ListType(a)),
        lambda x: lambda y: [x, y]
    ))

    prims.append(Primitive(
        'fst',
        arrow(ListType(a), a),
        lambda xs: xs[0] if len(xs) > 0 else None
    ))

    prims.append(Primitive(
        'snd',
        arrow(ListType(a), a),
        lambda xs: xs[1] if len(xs) > 1 else None
    ))

    return prims


def build_lean_plus_fold_primitives() -> List[Primitive]:
    """
    Build lean + fold primitive library (78 primitives).

    Adds to lean:
    - fold, foldr (universal iteration)
    - cons, empty, tail, is_empty (list construction)
    - all_true, any_true (boolean aggregation)
    - pair, fst, snd (tuple operations)
    """
    prims = build_lean_primitives()  # 67
    prims.extend(make_fold_primitives())  # +6 = 73
    prims.extend(make_bool_aggregators())  # +2 = 75
    prims.extend(make_pair_primitives())  # +3 = 78
    return prims


def build_lean_plus_fold_grammar() -> Grammar:
    """Build grammar from lean+fold primitives."""
    return uniform_grammar(build_lean_plus_fold_primitives())


# ============================================================================
# VARIANT 3: MINIMAL (48 primitives)
# ============================================================================

def make_minimal_list_slicing() -> List[Primitive]:
    """
    Minimal list slicing - only essential operations.

    Removes: first_half, second_half, half_len, adjacent_pairs
    Keeps: take, drop, zip_with (composable building blocks)
    """
    prims = []

    a = TypeVariable(0)
    b = TypeVariable(1)

    # Take first n elements
    prims.append(Primitive(
        'take',
        arrow(INT, ListType(a), ListType(a)),
        lambda n: lambda xs: xs[:n] if n >= 0 else []
    ))

    # Drop first n elements
    prims.append(Primitive(
        'drop',
        arrow(INT, ListType(a), ListType(a)),
        lambda n: lambda xs: xs[n:] if n >= 0 else xs
    ))

    # Zip with function
    prims.append(Primitive(
        'zip_with',
        arrow(arrow(a, b, BOOL), ListType(a), ListType(b), LIST_BOOL),
        lambda f: lambda xs: lambda ys: [f(x)(y) for x, y in zip(xs, ys)]
    ))

    return prims


def make_minimal_higher_order() -> List[Primitive]:
    """
    Minimal higher-order functions.

    Removes: unique (can be composed)
    Keeps: map, filter, all, any (essential)
    """
    prims = []

    a = TypeVariable(0)
    b = TypeVariable(1)

    prims.append(Primitive(
        'map',
        arrow(arrow(a, b), ListType(a), ListType(b)),
        lambda f: lambda xs: [f(x) for x in xs]
    ))

    prims.append(Primitive(
        'filter',
        arrow(arrow(a, BOOL), ListType(a), ListType(a)),
        lambda pred: lambda xs: [x for x in xs if pred(x)]
    ))

    prims.append(Primitive(
        'all',
        arrow(arrow(a, BOOL), ListType(a), BOOL),
        lambda pred: lambda xs: all(pred(x) for x in xs)
    ))

    prims.append(Primitive(
        'any',
        arrow(arrow(a, BOOL), ListType(a), BOOL),
        lambda pred: lambda xs: any(pred(x) for x in xs)
    ))

    return prims


def build_minimal_primitives() -> List[Primitive]:
    """
    Build minimal primitive library (48 primitives).

    Forces the model to LEARN patterns through compression:
    - No direct queries (has_suit, all_same_suit, etc.)
    - No aggregates (sum_ranks, max_rank, min_rank)
    - No convenience slicing (first_half, second_half, etc.)

    The model must discover abstractions like:
    - all_same_suit = (eq 1 (length (unique (map get_suit hand))))
    - sum_ranks = (fold + 0 (map rank_val hand))
    """
    prims = []
    prims.extend(make_constants())           # 21
    prims.extend(make_card_accessors())      # 4
    prims.extend(make_position_ops())        # 5
    prims.extend(make_minimal_list_slicing())  # 3 (instead of 7)
    # NO direct_queries (0 instead of 9)
    # NO aggregates (0 instead of 3)
    prims.extend(make_comparisons())         # 6
    prims.extend(make_boolean_ops())         # 4
    prims.extend(make_minimal_higher_order())  # 4 (instead of 5)
    prims.extend(make_arithmetic())          # 3
    # Add fold for minimal - it's essential for sum, max, etc.
    prims.extend(make_fold_primitives())     # 6
    return prims  # Total: 56


def build_minimal_grammar() -> Grammar:
    """Build grammar from minimal primitives."""
    return uniform_grammar(build_minimal_primitives())


# ============================================================================
# VARIANT FACTORY
# ============================================================================

PRIMITIVE_VARIANTS = {
    'lean': {
        'builder': build_lean_primitives,
        'grammar_builder': build_lean_grammar,
        'description': 'Cognitive realism library (67 primitives)',
        'expected_count': 67,
    },
    'lean_plus_fold': {
        'builder': build_lean_plus_fold_primitives,
        'grammar_builder': build_lean_plus_fold_grammar,
        'description': 'Lean + DreamCoder fold/cons (78 primitives)',
        'expected_count': 78,
    },
    'minimal': {
        'builder': build_minimal_primitives,
        'grammar_builder': build_minimal_grammar,
        'description': 'Minimal library, forces abstraction learning (56 primitives)',
        'expected_count': 56,
    },
}


def get_primitive_variant(variant: str) -> Dict[str, Any]:
    """Get a primitive variant by name."""
    if variant not in PRIMITIVE_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Choose from: {list(PRIMITIVE_VARIANTS.keys())}")
    return PRIMITIVE_VARIANTS[variant]


def build_grammar_for_variant(variant: str) -> Grammar:
    """Build grammar for a specific primitive variant."""
    info = get_primitive_variant(variant)
    return info['grammar_builder']()


# ============================================================================
# VALIDATION
# ============================================================================

def validate_all_variants():
    """Validate all primitive variants work correctly."""
    print("=" * 70)
    print("VALIDATING PRIMITIVE VARIANTS")
    print("=" * 70)

    from rules.cards import sample_hand, Card, Suit, Rank

    test_hand = [
        Card(Suit.HEARTS, Rank.ACE),
        Card(Suit.HEARTS, Rank.KING),
        Card(Suit.HEARTS, Rank.QUEEN),
        Card(Suit.SPADES, Rank.JACK),
        Card(Suit.SPADES, Rank.TEN),
    ]

    all_passed = True

    for variant_name, info in PRIMITIVE_VARIANTS.items():
        print(f"\n{variant_name.upper()}:")

        # Build primitives
        prims = info['builder']()
        actual_count = len(prims)
        expected_count = info['expected_count']

        print(f"  Primitives: {actual_count} (expected {expected_count})")

        if actual_count != expected_count:
            print(f"  ❌ COUNT MISMATCH!")
            all_passed = False

        # Build grammar
        grammar = info['grammar_builder']()
        print(f"  Grammar size: {len(grammar)}")

        # Test a few primitives work
        prim_dict = {p.name: p for p in prims}

        # Test basic operations that should exist in all variants
        tests_passed = 0
        tests_total = 0

        # Test get_suit
        if 'get_suit' in prim_dict:
            tests_total += 1
            try:
                result = prim_dict['get_suit'].value(test_hand[0])
                if result == Suit.HEARTS:
                    tests_passed += 1
            except Exception as e:
                print(f"  ❌ get_suit failed: {e}")

        # Test map
        if 'map' in prim_dict:
            tests_total += 1
            try:
                get_suit = prim_dict['get_suit'].value
                map_fn = prim_dict['map'].value
                result = map_fn(get_suit)(test_hand)
                if len(result) == 5:
                    tests_passed += 1
            except Exception as e:
                print(f"  ❌ map failed: {e}")

        # Test variant-specific primitives
        if variant_name == 'lean':
            if 'all_same_suit' in prim_dict:
                tests_total += 1
                try:
                    result = prim_dict['all_same_suit'].value(test_hand)
                    if result == False:  # Mixed suits
                        tests_passed += 1
                except Exception as e:
                    print(f"  ❌ all_same_suit failed: {e}")

        if variant_name == 'lean_plus_fold':
            if 'fold' in prim_dict:
                tests_total += 1
                try:
                    fold = prim_dict['fold'].value
                    add = prim_dict['+'].value
                    result = fold(add)(0)([1, 2, 3, 4, 5])
                    if result == 15:
                        tests_passed += 1
                except Exception as e:
                    print(f"  ❌ fold failed: {e}")

        if variant_name == 'minimal':
            if 'fold' in prim_dict and 'all_same_suit' not in prim_dict:
                tests_passed += 1
                tests_total += 1

        print(f"  Tests: {tests_passed}/{tests_total} passed")

        if tests_passed < tests_total:
            all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL VARIANTS VALIDATED SUCCESSFULLY")
    else:
        print("❌ SOME VARIANTS FAILED VALIDATION")
    print("=" * 70)

    return all_passed


if __name__ == "__main__":
    validate_all_variants()
