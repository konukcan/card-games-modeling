#!/usr/bin/env python3
"""
Extended Primitive Library for Full Lambda Calculus Experiment

This module extends lean_primitives.py with the FULL set of primitives needed
to test whether lambda calculus can solve our 56 rules:

1. DreamCoder primitives: fold, cons, car, cdr, empty, fix (recursion)
2. Yang & Piantadosi primitives: pair, fst, snd
3. Type-gap bridging: all_true, any_true (list(bool) -> bool)
4. Convenience: first_half, second_half, is_sorted_by

This is an EXPERIMENTAL library for overnight testing.
The goal is to push the lambda paradigm to its limits.

NOTE: This does NOT add domain-specific primitives like is_palindrome or
bracket_match - those would make the problem trivial. We want to see what
the basic lambda calculus machinery can handle.
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.type_system import (
    Type, BaseType, Arrow, ListType, TypeVariable,
    BOOL, INT, CARD, SUIT, RANK, HAND, arrow,
    LIST_INT, LIST_BOOL
)
from dreamcoder_core.program import Primitive
from dreamcoder_core.grammar import Grammar, Production, uniform_grammar
from dreamcoder_core.lean_primitives import (
    build_lean_primitives,
    make_constants,
    make_card_accessors,
    make_position_ops,
    make_list_slicing,
    make_direct_queries,
    make_aggregates,
    make_comparisons,
    make_boolean_ops,
    make_higher_order,
    make_arithmetic,
    COLOR, LIST_SUIT, LIST_RANK, LIST_COLOR
)

# Import card domain types
from rules.cards import (
    Card, Hand, Suit, Rank, Color, AltColor1, AltColor2, Parity,
    RANK_VALUES, card_color, suit_to_altcolor1, suit_to_altcolor2, rank_parity
)


# ============================================================================
# TYPE DEFINITIONS FOR NEW PRIMITIVES
# ============================================================================

# Pair type constructor: pair(a, b)
class PairType(Type):
    """Pair type for (a, b) tuples."""
    def __init__(self, first_type: Type, second_type: Type):
        self.first = first_type
        self.second = second_type

    def __str__(self) -> str:
        return f"pair({self.first}, {self.second})"

    def __eq__(self, other) -> bool:
        return isinstance(other, PairType) and self.first == other.first and self.second == other.second

    def __hash__(self) -> int:
        return hash(('pair', self.first, self.second))

    def free_type_variables(self):
        return self.first.free_type_variables() | self.second.free_type_variables()

    def apply_substitution(self, subst):
        return PairType(
            self.first.apply_substitution(subst),
            self.second.apply_substitution(subst)
        )


# ============================================================================
# LEVEL 9: FOLD AND LIST CONSTRUCTION (DreamCoder core)
# ============================================================================

def make_fold_primitives() -> List[Primitive]:
    """
    Fold and list construction primitives from DreamCoder.

    These are ESSENTIAL for stateful iteration:
    - fold: The universal list iterator
    - cons: Construct a list
    - car/cdr: Destructure a list (aka head/tail but we already have those)
    - empty: Empty list (polymorphic)

    fold type: (b -> a -> b) -> b -> list(a) -> b

    With fold, we can express:
    - sum: fold (+) 0 xs
    - all: fold (λacc x. and acc (pred x)) true xs
    - bracket matching: fold (λstate card. ...) (0, true) xs
    """
    prims = []

    a = TypeVariable(0)
    b = TypeVariable(1)

    # Fold left: (acc -> elem -> acc) -> acc -> list(elem) -> acc
    # This is the most important primitive for stateful iteration
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

    # Fold right: (elem -> acc -> acc) -> acc -> list(elem) -> acc
    # Sometimes more natural for certain computations
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

    # Empty list (polymorphic)
    # Note: We use a function that returns [] to handle polymorphism
    prims.append(Primitive(
        'empty',
        ListType(a),
        []
    ))

    # Tail (cdr) - we have this as drop 1, but explicit is clearer
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
# LEVEL 10: PAIR PRIMITIVES (Yang & Piantadosi style)
# ============================================================================

def make_pair_primitives() -> List[Primitive]:
    """
    Pair primitives for threading state through fold.

    Essential for bracket matching and other stateful computations:
    - pair: Construct a pair
    - fst: Get first element
    - snd: Get second element

    With pairs, we can thread (counter, is_valid) through a fold:
    fold (λstate card. if is_opener
                        (pair (+ 1 (fst state)) (snd state))
                        (pair (- 1 (fst state)) (and (snd state) (> (fst state) 0))))
         (pair 0 true)
         hand
    """
    prims = []

    a = TypeVariable(0)
    b = TypeVariable(1)

    # Since we can't easily add a new type to the type system,
    # we'll represent pairs as 2-element lists (a compromise)
    # This works because we already have ListType

    # However, for proper typing, let's use a tuple representation
    # and make fst/snd work on any 2-element list

    # Pair constructor: (a, b) represented as [a, b]
    # We'll treat list(any) as our pair type
    prims.append(Primitive(
        'pair',
        arrow(a, b, ListType(a)),  # Simplified: returns list of first type
        lambda x: lambda y: [x, y]
    ))

    # First element of pair (or list)
    prims.append(Primitive(
        'fst',
        arrow(ListType(a), a),
        lambda xs: xs[0] if len(xs) > 0 else None
    ))

    # Second element of pair (or list)
    prims.append(Primitive(
        'snd',
        arrow(ListType(a), a),
        lambda xs: xs[1] if len(xs) > 1 else None
    ))

    # Third element (for tuples with 3 elements)
    prims.append(Primitive(
        'thd',
        arrow(ListType(a), a),
        lambda xs: xs[2] if len(xs) > 2 else None
    ))

    # Triple constructor
    prims.append(Primitive(
        'triple',
        arrow(a, a, a, ListType(a)),
        lambda x: lambda y: lambda z: [x, y, z]
    ))

    return prims


# ============================================================================
# LEVEL 11: TYPE GAP BRIDGING (all_true, any_true)
# ============================================================================

def make_bool_aggregators() -> List[Primitive]:
    """
    Boolean list aggregators to bridge the type gap.

    The problem: zip_with returns list(bool) but we need bool.
    Solution: all_true and any_true aggregate list(bool) to bool.

    With these, we can express:
    - Palindrome: all_true (zip_with eq xs (reverse xs))
    - Sorted: all_true (zip_with le xs (tail xs))
    - Halves equal: all_true (zip_with eq (take 3 xs) (drop 3 xs))
    """
    prims = []

    # all_true: list(bool) -> bool
    # Returns true iff all elements are true
    prims.append(Primitive(
        'all_true',
        arrow(LIST_BOOL, BOOL),
        lambda xs: all(xs)
    ))

    # any_true: list(bool) -> bool
    # Returns true iff at least one element is true
    prims.append(Primitive(
        'any_true',
        arrow(LIST_BOOL, BOOL),
        lambda xs: any(xs)
    ))

    # none_true: list(bool) -> bool
    # Returns true iff no elements are true
    prims.append(Primitive(
        'none_true',
        arrow(LIST_BOOL, BOOL),
        lambda xs: not any(xs)
    ))

    # count_true: list(bool) -> int
    # Count how many elements are true
    prims.append(Primitive(
        'count_true',
        arrow(LIST_BOOL, INT),
        lambda xs: sum(1 for x in xs if x)
    ))

    return prims


# ============================================================================
# LEVEL 12: CONVENIENCE HALVES OPERATIONS
# ============================================================================

def make_halves_convenience() -> List[Primitive]:
    """
    Convenience primitives for halves operations.

    Instead of (take (half_len xs) xs), we can use (first_half xs).
    This reduces compositional depth significantly.
    """
    prims = []

    a = TypeVariable(0)

    # First half of list
    prims.append(Primitive(
        'first_half',
        arrow(ListType(a), ListType(a)),
        lambda xs: xs[:len(xs)//2]
    ))

    # Second half of list
    prims.append(Primitive(
        'second_half',
        arrow(ListType(a), ListType(a)),
        lambda xs: xs[len(xs)//2:]
    ))

    # Check if two lists are equal (polymorphic)
    prims.append(Primitive(
        'list_eq',
        arrow(ListType(a), ListType(a), BOOL),
        lambda xs: lambda ys: xs == ys
    ))

    return prims


# ============================================================================
# LEVEL 13: SORTED AND SEQUENCE CHECKS
# ============================================================================

def make_sequence_checks() -> List[Primitive]:
    """
    Sequence checking primitives.

    These check structural properties of sequences:
    - is_sorted_by: Check if list is sorted by a key function
    - is_palindrome_by: Check if list is palindromic by a property
    """
    prims = []

    a = TypeVariable(0)
    b = TypeVariable(1)

    # is_sorted_by: (a -> int) -> list(a) -> bool
    # Check if list is sorted (non-decreasing) by key function
    def is_sorted_by(key_fn):
        def check(xs):
            if len(xs) <= 1:
                return True
            keys = [key_fn(x) for x in xs]
            for i in range(len(keys) - 1):
                if keys[i] > keys[i+1]:
                    return False
            return True
        return check

    prims.append(Primitive(
        'is_sorted_by',
        arrow(arrow(a, INT), ListType(a), BOOL),
        is_sorted_by
    ))

    # is_strictly_sorted_by: Check strictly increasing
    def is_strictly_sorted_by(key_fn):
        def check(xs):
            if len(xs) <= 1:
                return True
            keys = [key_fn(x) for x in xs]
            for i in range(len(keys) - 1):
                if keys[i] >= keys[i+1]:
                    return False
            return True
        return check

    prims.append(Primitive(
        'is_strictly_sorted_by',
        arrow(arrow(a, INT), ListType(a), BOOL),
        is_strictly_sorted_by
    ))

    # is_palindrome_by: (a -> b) -> list(a) -> bool
    # Check if the sequence of property values is palindromic
    def is_palindrome_by(prop_fn):
        def check(xs):
            props = [prop_fn(x) for x in xs]
            return props == list(reversed(props))
        return check

    prims.append(Primitive(
        'is_palindrome_by',
        arrow(arrow(a, b), ListType(a), BOOL),
        is_palindrome_by
    ))

    return prims


# ============================================================================
# LEVEL 14: ADDITIONAL ARITHMETIC
# ============================================================================

def make_extended_arithmetic() -> List[Primitive]:
    """
    Extended arithmetic for more complex calculations.
    """
    prims = []

    # Absolute value
    prims.append(Primitive(
        'abs',
        arrow(INT, INT),
        lambda x: abs(x)
    ))

    # Maximum of two integers
    prims.append(Primitive(
        'max2',
        arrow(INT, INT, INT),
        lambda x: lambda y: max(x, y)
    ))

    # Minimum of two integers
    prims.append(Primitive(
        'min2',
        arrow(INT, INT, INT),
        lambda x: lambda y: min(x, y)
    ))

    # Multiplication (sometimes needed for threshold checks)
    prims.append(Primitive(
        '*',
        arrow(INT, INT, INT),
        lambda x: lambda y: x * y
    ))

    # Integer division
    prims.append(Primitive(
        '//',
        arrow(INT, INT, INT),
        lambda x: lambda y: x // y if y != 0 else 0
    ))

    return prims


# ============================================================================
# LEVEL 15: SUIT CYCLE MAPPINGS (for MAP family rules)
# ============================================================================

def make_suit_cycles() -> List[Primitive]:
    """
    Suit cycle mappings for the MAP family rules.

    M1: ♣→♠→♥→♦→♣ (CLUBS -> SPADES -> HEARTS -> DIAMONDS -> CLUBS)
    M2: ♣→♥→♠→♦→♣ (CLUBS -> HEARTS -> SPADES -> DIAMONDS -> CLUBS)
    """
    prims = []

    # Cycle M1: CLUBS -> SPADES -> HEARTS -> DIAMONDS -> CLUBS
    M1 = {
        Suit.CLUBS: Suit.SPADES,
        Suit.SPADES: Suit.HEARTS,
        Suit.HEARTS: Suit.DIAMONDS,
        Suit.DIAMONDS: Suit.CLUBS
    }

    prims.append(Primitive(
        'cycle_m1',
        arrow(SUIT, SUIT),
        lambda s: M1.get(s, s)
    ))

    # Cycle M2: CLUBS -> HEARTS -> SPADES -> DIAMONDS -> CLUBS
    M2 = {
        Suit.CLUBS: Suit.HEARTS,
        Suit.HEARTS: Suit.SPADES,
        Suit.SPADES: Suit.DIAMONDS,
        Suit.DIAMONDS: Suit.CLUBS
    }

    prims.append(Primitive(
        'cycle_m2',
        arrow(SUIT, SUIT),
        lambda s: M2.get(s, s)
    ))

    return prims


# ============================================================================
# LEVEL 16: ALTCOLOR ACCESSORS
# ============================================================================

def make_altcolor_accessors() -> List[Primitive]:
    """
    Alternative color groupings for ALTCLR family rules.
    """
    prims = []

    # AltColor1: Pointy (♠♦) vs Round (♥♣)
    ALTCOLOR1 = BaseType('altcolor1')

    prims.append(Primitive(
        'get_altcolor1',
        arrow(CARD, ALTCOLOR1),
        lambda c: suit_to_altcolor1(c.suit)
    ))

    # AltColor2: SH (♠♥) vs DC (♦♣)
    ALTCOLOR2 = BaseType('altcolor2')

    prims.append(Primitive(
        'get_altcolor2',
        arrow(CARD, ALTCOLOR2),
        lambda c: suit_to_altcolor2(c.suit)
    ))

    # Parity accessor
    PARITY = BaseType('parity')

    prims.append(Primitive(
        'get_parity',
        arrow(CARD, PARITY),
        lambda c: rank_parity(c.rank)
    ))

    # Is odd rank
    prims.append(Primitive(
        'is_odd_rank',
        arrow(CARD, BOOL),
        lambda c: RANK_VALUES[c.rank] % 2 == 1
    ))

    # Is even rank
    prims.append(Primitive(
        'is_even_rank',
        arrow(CARD, BOOL),
        lambda c: RANK_VALUES[c.rank] % 2 == 0
    ))

    return prims


# ============================================================================
# BUILD EXTENDED PRIMITIVE LIBRARY
# ============================================================================

def build_extended_primitives() -> List[Primitive]:
    """
    Build the extended primitive library with all additions.

    This includes:
    - All lean primitives (base library)
    - Fold and list construction (DreamCoder)
    - Pair primitives (Y&P)
    - Boolean aggregators (type gap bridging)
    - Convenience halves operations
    - Sequence checks
    - Extended arithmetic
    - Suit cycles
    - Altcolor accessors
    """
    prims = []

    # Base primitives from lean library
    prims.extend(build_lean_primitives())           # ~63 primitives

    # New extensions
    prims.extend(make_fold_primitives())            # 6 primitives
    prims.extend(make_pair_primitives())            # 5 primitives
    prims.extend(make_bool_aggregators())           # 4 primitives
    prims.extend(make_halves_convenience())         # 3 primitives
    prims.extend(make_sequence_checks())            # 3 primitives
    prims.extend(make_extended_arithmetic())        # 5 primitives
    prims.extend(make_suit_cycles())                # 2 primitives
    prims.extend(make_altcolor_accessors())         # 5 primitives

    return prims


def build_extended_grammar() -> Grammar:
    """Build grammar with extended primitives."""
    prims = build_extended_primitives()
    return uniform_grammar(prims)


# ============================================================================
# PRIMITIVE SUMMARY
# ============================================================================

def print_primitive_summary():
    """Print summary of all primitives by category."""
    print("=" * 70)
    print("EXTENDED PRIMITIVE LIBRARY SUMMARY")
    print("=" * 70)

    categories = [
        ("Lean Base Library", build_lean_primitives()),
        ("Fold & List Construction (DreamCoder)", make_fold_primitives()),
        ("Pair Primitives (Y&P)", make_pair_primitives()),
        ("Boolean Aggregators (Type Gap)", make_bool_aggregators()),
        ("Halves Convenience", make_halves_convenience()),
        ("Sequence Checks", make_sequence_checks()),
        ("Extended Arithmetic", make_extended_arithmetic()),
        ("Suit Cycles", make_suit_cycles()),
        ("AltColor Accessors", make_altcolor_accessors()),
    ]

    total = 0
    for name, prims in categories:
        print(f"\n{name} ({len(prims)} primitives):")
        for p in prims:
            print(f"  {p.name}: {p.tp}")
        total += len(prims)

    print(f"\n{'=' * 70}")
    print(f"TOTAL: {total} primitives")
    print("=" * 70)


# ============================================================================
# EXPECTED DEPTHS WITH EXTENDED PRIMITIVES
# ============================================================================

def show_expected_depths():
    """Show expected depths for various rule families with extended primitives."""
    print("\n" + "=" * 70)
    print("EXPECTED DEPTHS WITH EXTENDED PRIMITIVES")
    print("=" * 70)

    examples = [
        # PAL family - previously impossible
        ("Suits_palindrome",
         "(λ is_palindrome_by get_suit $0)", 3,
         "all_true (zip_with eq (map get_suit $0) (reverse (map get_suit $0)))"),
        ("Colors_palindrome",
         "(λ is_palindrome_by get_color $0)", 3,
         "Same pattern"),

        # COPY family - previously depth 7+
        ("Halves_copy_suits",
         "(λ list_eq (map get_suit (first_half $0)) (map get_suit (second_half $0)))", 5,
         "Or: all_true (zip_with eq ...)"),

        # Sorted - previously impossible
        ("Sorted_by_rank",
         "(λ is_sorted_by rank_val $0)", 3,
         "Or: all_true (zip_with le (map rank_val $0) (tail (map rank_val $0)))"),

        # HIER family - halves comparison
        ("Halves_uniform_color_equal",
         "(λ eq (all_same_color (first_half $0)) (all_same_color (second_half $0)))", 5,
         "With convenience primitives"),

        # SHIFT family
        ("Shift2_plus3",
         "(λ all_true (zip_with (λ x y. eq (+ 3 (rank_val x)) (rank_val y)) $0 (drop 2 $0)))", 8,
         "Still deep but expressible"),

        # ADJ family
        ("Adj_same_rank_or_suit",
         "(λ all (λ pair. or (eq (get_rank (fst pair)) (get_rank (snd pair))) " +
         "(eq (get_suit (fst pair)) (get_suit (snd pair)))) (adjacent_pairs $0))", 10,
         "Complex but expressible"),

        # LANG family - bracket matching (hardest)
        ("Well_formed_brackets_by_suit",
         "(λ and (eq 0 (snd (fold ...))) (fst (fold ...)))", 12,
         "Requires fold + pair for state"),
    ]

    print(f"\n{'Rule':<30} {'Depth':<8} {'Status':<15}")
    print("-" * 60)

    for name, program, depth, note in examples:
        status = "✓ Easy" if depth <= 4 else ("✓ Feasible" if depth <= 8 else "⚠ Challenging")
        print(f"{name:<30} {depth:<8} {status:<15}")

    print("\nKey:")
    print("  ✓ Easy (depth ≤ 4): High probability of finding")
    print("  ✓ Feasible (depth 5-8): Moderate search time")
    print("  ⚠ Challenging (depth 9+): May need longer search")


# ============================================================================
# TESTS
# ============================================================================

def test_extended_primitives():
    """Test all extended primitives work correctly."""
    print("\n" + "=" * 70)
    print("TESTING EXTENDED PRIMITIVES")
    print("=" * 70)

    # Create test hand
    test_hand = [
        Card(Suit.HEARTS, Rank.TWO),
        Card(Suit.SPADES, Rank.FOUR),
        Card(Suit.HEARTS, Rank.SIX),
        Card(Suit.DIAMONDS, Rank.EIGHT),
        Card(Suit.CLUBS, Rank.TEN),
        Card(Suit.HEARTS, Rank.QUEEN),
    ]

    prims = build_extended_primitives()
    prim_dict = {p.name: p for p in prims}

    print(f"\nTest hand: {[f'{c.rank.name} of {c.suit.name}' for c in test_hand]}")

    # Test fold
    print("\n1. Testing fold:")
    fold = prim_dict['fold'].value
    add = prim_dict['+'].value
    sum_result = fold(lambda acc: lambda x: acc + RANK_VALUES[x.rank])(0)(test_hand)
    print(f"   fold (+rank) 0 hand = {sum_result}")  # 2+4+6+8+10+12 = 42
    assert sum_result == 42, f"Expected 42, got {sum_result}"
    print("   ✓ fold works")

    # Test pairs
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

    # Test all_true
    print("\n3. Testing all_true/any_true:")
    all_true = prim_dict['all_true'].value
    any_true = prim_dict['any_true'].value
    print(f"   all_true [True, True, True] = {all_true([True, True, True])}")
    print(f"   all_true [True, False, True] = {all_true([True, False, True])}")
    print(f"   any_true [False, False, True] = {any_true([False, False, True])}")
    assert all_true([True, True, True]) == True
    assert all_true([True, False, True]) == False
    assert any_true([False, False, True]) == True
    print("   ✓ all_true/any_true work")

    # Test first_half/second_half
    print("\n4. Testing first_half/second_half:")
    first_half = prim_dict['first_half'].value
    second_half = prim_dict['second_half'].value
    fh = first_half(test_hand)
    sh = second_half(test_hand)
    print(f"   first_half hand = {len(fh)} cards")
    print(f"   second_half hand = {len(sh)} cards")
    assert len(fh) == 3 and len(sh) == 3
    print("   ✓ first_half/second_half work")

    # Test is_sorted_by
    print("\n5. Testing is_sorted_by:")
    is_sorted_by = prim_dict['is_sorted_by'].value
    rank_val = prim_dict['rank_val'].value
    sorted_result = is_sorted_by(rank_val)(test_hand)
    print(f"   is_sorted_by rank_val hand = {sorted_result}")
    # Our test hand is [2,4,6,8,10,12] which IS sorted
    assert sorted_result == True
    print("   ✓ is_sorted_by works")

    # Test is_palindrome_by
    print("\n6. Testing is_palindrome_by:")
    is_palindrome_by = prim_dict['is_palindrome_by'].value
    get_suit = prim_dict['get_suit'].value
    palindrome_result = is_palindrome_by(get_suit)(test_hand)
    print(f"   is_palindrome_by get_suit hand = {palindrome_result}")
    # [H, S, H, D, C, H] - not a palindrome
    assert palindrome_result == False

    # Make a palindrome hand
    palindrome_hand = [
        Card(Suit.HEARTS, Rank.TWO),
        Card(Suit.SPADES, Rank.THREE),
        Card(Suit.HEARTS, Rank.FOUR),
        Card(Suit.HEARTS, Rank.FIVE),
        Card(Suit.SPADES, Rank.SIX),
        Card(Suit.HEARTS, Rank.SEVEN),
    ]
    palindrome_result2 = is_palindrome_by(get_suit)(palindrome_hand)
    print(f"   is_palindrome_by get_suit [H,S,H,H,S,H] = {palindrome_result2}")
    assert palindrome_result2 == True
    print("   ✓ is_palindrome_by works")

    # Test suit cycles
    print("\n7. Testing suit cycles:")
    cycle_m1 = prim_dict['cycle_m1'].value
    cycle_m2 = prim_dict['cycle_m2'].value
    print(f"   cycle_m1 CLUBS = {cycle_m1(Suit.CLUBS)}")  # SPADES
    print(f"   cycle_m1 SPADES = {cycle_m1(Suit.SPADES)}")  # HEARTS
    print(f"   cycle_m2 CLUBS = {cycle_m2(Suit.CLUBS)}")  # HEARTS
    assert cycle_m1(Suit.CLUBS) == Suit.SPADES
    assert cycle_m2(Suit.CLUBS) == Suit.HEARTS
    print("   ✓ suit cycles work")

    # Test cons/tail
    print("\n8. Testing cons/tail:")
    cons = prim_dict['cons'].value
    tail = prim_dict['tail'].value
    result = cons(1)([2, 3, 4])
    print(f"   cons 1 [2,3,4] = {result}")
    assert result == [1, 2, 3, 4]
    result2 = tail([1, 2, 3, 4])
    print(f"   tail [1,2,3,4] = {result2}")
    assert result2 == [2, 3, 4]
    print("   ✓ cons/tail work")

    # Test zip_with + all_true (the key combination)
    print("\n9. Testing zip_with + all_true (palindrome pattern):")
    zip_with = prim_dict['zip_with'].value
    eq = prim_dict['eq'].value
    reverse = prim_dict['reverse'].value
    map_fn = prim_dict['map'].value

    suits = map_fn(get_suit)(palindrome_hand)
    reversed_suits = reverse(suits)
    zipped = zip_with(eq)(suits)(reversed_suits)
    all_match = all_true(zipped)
    print(f"   suits = {[s.name for s in suits]}")
    print(f"   reversed = {[s.name for s in reversed_suits]}")
    print(f"   zip_with eq suits (reverse suits) = {zipped}")
    print(f"   all_true (...) = {all_match}")
    assert all_match == True
    print("   ✓ zip_with + all_true work together")

    print("\n" + "=" * 70)
    print("ALL EXTENDED PRIMITIVE TESTS PASSED!")
    print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print_primitive_summary()
    show_expected_depths()
    test_extended_primitives()
