#!/usr/bin/env python3
"""
Extended Primitive Library - DreamCoder + Yang & Piantadosi Primitives ONLY

This module extends lean_primitives.py with primitives that are found in:
1. Ellis et al. DreamCoder (PLDI 2021)
2. Yang & Piantadosi (Cognition 2022)

NO ad-hoc convenience primitives. The system should LEARN patterns like
palindromes, sorted sequences, etc. - not be given them.

DreamCoder primitives added:
- fold/foldr: Universal list iteration
- cons: List construction
- empty: Empty list
- fix: Recursion (Y combinator) - NOT ADDED (too complex for enumeration)

Yang & Piantadosi primitives added:
- pair, fst, snd: Tuple operations for state threading

Type-gap bridging (necessary, not convenience):
- all_true: list(bool) -> bool (this is just `and` folded over a list)
- any_true: list(bool) -> bool (this is just `or` folded over a list)

These are NOT convenience - they're the natural way to aggregate boolean lists,
equivalent to (fold and true xs) and (fold or false xs).
"""

import sys
from pathlib import Path
from typing import List

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.type_system import (
    Type, BaseType, Arrow, ListType, TypeVariable,
    BOOL, INT, CARD, SUIT, RANK, HAND, arrow,
    LIST_INT, LIST_BOOL
)
from dreamcoder_core.program import Primitive
from dreamcoder_core.grammar import Grammar, uniform_grammar
from dreamcoder_core.lean_primitives import build_lean_primitives

# Import card domain types
from rules.cards import (
    Card, Suit, Rank, RANK_VALUES, card_color,
    suit_to_altcolor1, suit_to_altcolor2, rank_parity
)


# ============================================================================
# DREAMCODER PRIMITIVES: fold, cons, empty
# From Ellis et al. "DreamCoder: Bootstrapping Inductive Program Synthesis"
# ============================================================================

def make_dreamcoder_primitives() -> List[Primitive]:
    """
    Core DreamCoder list primitives.

    These are the fundamental building blocks for list processing:
    - fold: (acc -> elem -> acc) -> acc -> list -> acc
    - foldr: (elem -> acc -> acc) -> acc -> list -> acc
    - cons: elem -> list -> list
    - empty: list (polymorphic empty list)
    - car/cdr: Already have head/tail in lean_primitives
    """
    prims = []

    a = TypeVariable(0)
    b = TypeVariable(1)

    # Fold left (foldl): Process list left-to-right with accumulator
    # Type: (acc -> elem -> acc) -> acc -> list(elem) -> acc
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

    # Fold right: Process list right-to-left
    # Type: (elem -> acc -> acc) -> acc -> list(elem) -> acc
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

    # Cons: Prepend element to list
    # Type: a -> list(a) -> list(a)
    prims.append(Primitive(
        'cons',
        arrow(a, ListType(a), ListType(a)),
        lambda x: lambda xs: [x] + list(xs)
    ))

    # Empty list (polymorphic)
    # Type: list(a)
    prims.append(Primitive(
        'empty',
        ListType(a),
        []
    ))

    # Tail (cdr) - explicitly named for DreamCoder compatibility
    # We have 'drop 1' but explicit tail is clearer
    prims.append(Primitive(
        'tail',
        arrow(ListType(a), ListType(a)),
        lambda xs: xs[1:] if len(xs) > 0 else []
    ))

    # Is empty check
    prims.append(Primitive(
        'is_empty',
        arrow(ListType(a), BOOL),
        lambda xs: len(xs) == 0
    ))

    return prims


# ============================================================================
# YANG & PIANTADOSI PRIMITIVES: pair, fst, snd
# From "A Rational Constructivist Model of Concept Learning" (Cognition 2022)
# ============================================================================

def make_yp_primitives() -> List[Primitive]:
    """
    Yang & Piantadosi primitives for state threading.

    Essential for computations that need to track multiple values,
    like bracket matching (counter + validity flag).

    We represent pairs as 2-element lists since our type system
    doesn't have native tuples.
    """
    prims = []

    a = TypeVariable(0)
    b = TypeVariable(1)

    # Pair constructor: Creates a 2-element list [a, b]
    # Type: a -> b -> list(a)  (simplified - treats as homogeneous)
    prims.append(Primitive(
        'pair',
        arrow(a, a, ListType(a)),
        lambda x: lambda y: [x, y]
    ))

    # First element of pair/list
    # Type: list(a) -> a
    prims.append(Primitive(
        'fst',
        arrow(ListType(a), a),
        lambda xs: xs[0] if len(xs) > 0 else None
    ))

    # Second element of pair/list
    # Type: list(a) -> a
    prims.append(Primitive(
        'snd',
        arrow(ListType(a), a),
        lambda xs: xs[1] if len(xs) > 1 else None
    ))

    return prims


# ============================================================================
# TYPE-GAP BRIDGING: all_true, any_true
# These are NOT convenience - they're (fold and true) and (fold or false)
# ============================================================================

def make_bool_aggregators() -> List[Primitive]:
    """
    Boolean list aggregators.

    These bridge the type gap: zip_with returns list(bool), but rules need bool.

    Mathematically:
    - all_true xs = fold (λacc x. and acc x) true xs
    - any_true xs = fold (λacc x. or acc x) false xs

    We include them as primitives because:
    1. They're the canonical way to aggregate boolean lists
    2. Expressing them via fold adds 3-4 depth unnecessarily
    3. DreamCoder's list library includes similar aggregators
    """
    prims = []

    # all_true: list(bool) -> bool
    # Equivalent to: fold and true xs
    prims.append(Primitive(
        'all_true',
        arrow(LIST_BOOL, BOOL),
        lambda xs: all(xs)
    ))

    # any_true: list(bool) -> bool
    # Equivalent to: fold or false xs
    prims.append(Primitive(
        'any_true',
        arrow(LIST_BOOL, BOOL),
        lambda xs: any(xs)
    ))

    return prims


# ============================================================================
# BUILD EXTENDED PRIMITIVE LIBRARY
# ============================================================================

def build_extended_primitives() -> List[Primitive]:
    """
    Build the extended primitive library.

    Includes:
    - All lean primitives (base library): ~65 primitives
    - DreamCoder primitives: fold, foldr, cons, empty, tail, is_empty (6)
    - Y&P primitives: pair, fst, snd (3)
    - Type-gap bridging: all_true, any_true (2)

    Total: ~76 primitives

    NOT included (these should be LEARNED):
    - is_palindrome_by
    - is_sorted_by
    - first_half, second_half
    - Any other convenience/pattern primitives
    """
    prims = []

    # Base primitives from lean library
    prims.extend(build_lean_primitives())

    # DreamCoder primitives
    prims.extend(make_dreamcoder_primitives())

    # Yang & Piantadosi primitives
    prims.extend(make_yp_primitives())

    # Type-gap bridging
    prims.extend(make_bool_aggregators())

    return prims


def build_extended_grammar() -> Grammar:
    """Build grammar with extended primitives."""
    prims = build_extended_primitives()
    return uniform_grammar(prims)


# ============================================================================
# SUMMARY
# ============================================================================

def print_primitive_summary():
    """Print summary of primitives added."""
    print("=" * 70)
    print("EXTENDED PRIMITIVE LIBRARY")
    print("DreamCoder + Yang & Piantadosi primitives ONLY")
    print("=" * 70)

    lean_prims = build_lean_primitives()
    dc_prims = make_dreamcoder_primitives()
    yp_prims = make_yp_primitives()
    bool_prims = make_bool_aggregators()

    print(f"\nBase (lean_primitives.py): {len(lean_prims)} primitives")

    print(f"\nDreamCoder additions ({len(dc_prims)}):")
    for p in dc_prims:
        print(f"  {p.name}: {p.tp}")

    print(f"\nYang & Piantadosi additions ({len(yp_prims)}):")
    for p in yp_prims:
        print(f"  {p.name}: {p.tp}")

    print(f"\nType-gap bridging ({len(bool_prims)}):")
    for p in bool_prims:
        print(f"  {p.name}: {p.tp}")

    total = len(lean_prims) + len(dc_prims) + len(yp_prims) + len(bool_prims)
    print(f"\n{'=' * 70}")
    print(f"TOTAL: {total} primitives")
    print("=" * 70)


# ============================================================================
# TESTS
# ============================================================================

def test_extended_primitives():
    """Test the extended primitives."""
    print("\n" + "=" * 70)
    print("TESTING EXTENDED PRIMITIVES")
    print("=" * 70)

    prims = build_extended_primitives()
    prim_dict = {p.name: p for p in prims}

    # Test hand
    test_hand = [
        Card(Suit.HEARTS, Rank.TWO),
        Card(Suit.SPADES, Rank.FOUR),
        Card(Suit.HEARTS, Rank.SIX),
    ]

    print(f"\nTest hand: {[f'{c.rank.name} of {c.suit.name}' for c in test_hand]}")

    # Test fold
    print("\n1. Testing fold:")
    fold = prim_dict['fold'].value
    sum_result = fold(lambda acc: lambda x: acc + RANK_VALUES[x.rank])(0)(test_hand)
    print(f"   fold (+rank) 0 hand = {sum_result}")
    assert sum_result == 2 + 4 + 6, f"Expected 12, got {sum_result}"
    print("   ✓ fold works")

    # Test pair/fst/snd
    print("\n2. Testing pair/fst/snd:")
    pair = prim_dict['pair'].value
    fst = prim_dict['fst'].value
    snd = prim_dict['snd'].value
    p = pair(1)(2)
    print(f"   pair 1 2 = {p}")
    print(f"   fst (pair 1 2) = {fst(p)}")
    print(f"   snd (pair 1 2) = {snd(p)}")
    assert fst(p) == 1 and snd(p) == 2
    print("   ✓ pair/fst/snd work")

    # Test all_true/any_true
    print("\n3. Testing all_true/any_true:")
    all_true = prim_dict['all_true'].value
    any_true = prim_dict['any_true'].value
    print(f"   all_true [T,T,T] = {all_true([True, True, True])}")
    print(f"   all_true [T,F,T] = {all_true([True, False, True])}")
    print(f"   any_true [F,F,T] = {any_true([False, False, True])}")
    assert all_true([True, True, True]) == True
    assert all_true([True, False, True]) == False
    assert any_true([False, False, True]) == True
    print("   ✓ all_true/any_true work")

    # Test cons/tail
    print("\n4. Testing cons/tail:")
    cons = prim_dict['cons'].value
    tail = prim_dict['tail'].value
    result = cons(1)([2, 3])
    print(f"   cons 1 [2,3] = {result}")
    assert result == [1, 2, 3]
    result2 = tail([1, 2, 3])
    print(f"   tail [1,2,3] = {result2}")
    assert result2 == [2, 3]
    print("   ✓ cons/tail work")

    # Test fold with pair state (bracket counting pattern)
    print("\n5. Testing fold with pair state:")
    def count_step(state):
        def process(card):
            count = state[0]
            return [count + 1, state[1]]
        return process

    result = fold(count_step)([0, True])(test_hand)
    print(f"   fold (count_step) [0,True] hand = {result}")
    assert result[0] == 3
    print("   ✓ fold with pair state works")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    print_primitive_summary()
    test_extended_primitives()
