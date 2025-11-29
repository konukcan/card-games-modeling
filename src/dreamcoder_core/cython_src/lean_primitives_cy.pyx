# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
"""
Cython-Native Lean Primitive Library for Card Game Learning

This module creates primitives using Cython-native types from program_cy.pyx,
enabling full Cython speedup throughout the enumeration pipeline.

Key difference from lean_primitives.py:
- Uses Primitive from program_cy (Cython extension type)
- Uses Type classes from type_system_cy (Cython extension types)
- All primitives are Cython-native for maximum performance
"""

import sys
from pathlib import Path

# Import Cython-native types
from .type_system_cy import (
    Type, BaseType, Arrow, ListType, TypeVariable,
    BOOL, INT, CARD, SUIT, RANK, HAND, arrow
)
from .program_cy import Primitive
from .grammar_cy import Grammar, Production, uniform_grammar

# Import card domain types (these are Python objects, but that's fine for values)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from rules.cards import (
    Card, Hand, Suit, Rank, Color,
    RANK_VALUES, card_color
)


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

COLOR = BaseType('color')
LIST_SUIT = ListType(SUIT)
LIST_RANK = ListType(RANK)
LIST_COLOR = ListType(COLOR)
LIST_INT = ListType(INT)
LIST_BOOL = ListType(BOOL)


# ============================================================================
# HELPER FUNCTIONS (defined at module level to avoid lambda issues)
# ============================================================================

# Identity
def _id(x):
    return x

# Constant (K combinator)
def _const(x):
    def inner(y):
        return x
    return inner

# Composition
def _compose(f):
    def inner1(g):
        def inner2(x):
            return f(g(x))
        return inner2
    return inner1

# Flip
def _flip(f):
    def inner1(x):
        def inner2(y):
            return f(y)(x)
        return inner2
    return inner1

# Card accessors
def _get_suit(c):
    return c.suit

def _get_rank(c):
    return c.rank

def _rank_val(c):
    return RANK_VALUES[c.rank]

def _get_color(c):
    return card_color(c)

# List operations
def _is_empty(xs):
    return len(xs) == 0

def _cons(x):
    def inner(xs):
        return [x] + list(xs)
    return inner

def _head(xs):
    return xs[0] if xs else None

def _tail(xs):
    return xs[1:] if xs else []

def _last(xs):
    return xs[-1] if xs else None

def _length(xs):
    return len(xs)

def _at(xs):
    def inner(i):
        return xs[i] if 0 <= i < len(xs) else None
    return inner

def _reverse(xs):
    return list(reversed(xs))

def _take(n):
    def inner(xs):
        return xs[:n]
    return inner

def _drop(n):
    def inner(xs):
        return xs[n:]
    return inner

# Comparisons
def _eq(x):
    def inner(y):
        return x == y
    return inner

def _neq(x):
    def inner(y):
        return x != y
    return inner

def _lt(x):
    def inner(y):
        return x < y
    return inner

def _le(x):
    def inner(y):
        return x <= y
    return inner

def _gt(x):
    def inner(y):
        return x > y
    return inner

def _ge(x):
    def inner(y):
        return x >= y
    return inner

# Boolean operations
def _and(x):
    def inner(y):
        return x and y
    return inner

def _or(x):
    def inner(y):
        return x or y
    return inner

def _not(x):
    return not x

def _if(cond):
    def inner1(then_val):
        def inner2(else_val):
            return then_val if cond else else_val
        return inner2
    return inner1

# Higher-order functions
def _map(f):
    def inner(xs):
        return [f(x) for x in xs]
    return inner

def _filter(pred):
    def inner(xs):
        return [x for x in xs if pred(x)]
    return inner

def _fold(f):
    def inner1(init):
        def inner2(xs):
            result = init
            for x in xs:
                result = f(result)(x)
            return result
        return inner2
    return inner1

def _all(pred):
    def inner(xs):
        return all(pred(x) for x in xs)
    return inner

def _any(pred):
    def inner(xs):
        return any(pred(x) for x in xs)
    return inner

def _count(pred):
    def inner(xs):
        return sum(1 for x in xs if pred(x))
    return inner

def _unique(xs):
    return list(dict.fromkeys(xs))

# Arithmetic
def _add(x):
    def inner(y):
        return x + y
    return inner

def _sub(x):
    def inner(y):
        return x - y
    return inner

def _mul(x):
    def inner(y):
        return x * y
    return inner

def _div(x):
    def inner(y):
        return x // y if y != 0 else 0
    return inner

def _mod(x):
    def inner(y):
        return x % y if y != 0 else 0
    return inner

# Pair operations
def _pair(x):
    def inner(y):
        return [x, y]
    return inner

def _fst(p):
    return p[0] if len(p) >= 1 else None

def _snd(p):
    return p[1] if len(p) >= 2 else None

def _pairs(xs):
    return [[xs[i], xs[i+1]] for i in range(len(xs)-1)] if len(xs) > 1 else []

def _zip_with(f):
    def inner1(xs):
        def inner2(ys):
            return [f(x)(y) for x, y in zip(xs, ys)]
        return inner2
    return inner1


# ============================================================================
# PRIMITIVE CONSTRUCTION
# ============================================================================

def make_constants():
    """Essential constants only."""
    cdef list prims = []

    # Suit constants
    prims.append(Primitive('CLUBS', SUIT, Suit.CLUBS))
    prims.append(Primitive('DIAMONDS', SUIT, Suit.DIAMONDS))
    prims.append(Primitive('HEARTS', SUIT, Suit.HEARTS))
    prims.append(Primitive('SPADES', SUIT, Suit.SPADES))

    # Color constants
    prims.append(Primitive('RED', COLOR, Color.RED))
    prims.append(Primitive('BLACK', COLOR, Color.BLACK))

    # Minimal integer constants (0-4)
    cdef int i
    for i in range(5):
        prims.append(Primitive(str(i), INT, i))

    # Boolean constants
    prims.append(Primitive('true', BOOL, True))
    prims.append(Primitive('false', BOOL, False))

    return prims


def make_combinators():
    """Lambda calculus combinators."""
    cdef list prims = []

    a = TypeVariable(0)
    b = TypeVariable(1)
    c = TypeVariable(2)

    # Identity
    prims.append(Primitive('id', arrow(a, a), _id))

    # Constant (K combinator)
    prims.append(Primitive('const', arrow(a, b, a), _const))

    # Composition
    prims.append(Primitive('compose', arrow(arrow(b, c), arrow(a, b), a, c), _compose))

    # Flip
    prims.append(Primitive('flip', arrow(arrow(a, b, c), b, a, c), _flip))

    return prims


def make_card_accessors():
    """Card property accessors."""
    cdef list prims = []

    prims.append(Primitive('get_suit', arrow(CARD, SUIT), _get_suit))
    prims.append(Primitive('get_rank', arrow(CARD, RANK), _get_rank))
    prims.append(Primitive('rank_val', arrow(CARD, INT), _rank_val))
    prims.append(Primitive('get_color', arrow(CARD, COLOR), _get_color))

    return prims


def make_list_ops():
    """Core list operations."""
    cdef list prims = []

    a = TypeVariable(0)

    prims.append(Primitive('empty?', arrow(ListType(a), BOOL), _is_empty))
    prims.append(Primitive('nil', ListType(a), []))
    prims.append(Primitive('cons', arrow(a, ListType(a), ListType(a)), _cons))
    prims.append(Primitive('head', arrow(ListType(a), a), _head))
    prims.append(Primitive('tail', arrow(ListType(a), ListType(a)), _tail))
    prims.append(Primitive('last', arrow(ListType(a), a), _last))
    prims.append(Primitive('length', arrow(ListType(a), INT), _length))
    prims.append(Primitive('at', arrow(ListType(a), INT, a), _at))
    prims.append(Primitive('reverse', arrow(ListType(a), ListType(a)), _reverse))
    prims.append(Primitive('take', arrow(INT, ListType(a), ListType(a)), _take))
    prims.append(Primitive('drop', arrow(INT, ListType(a), ListType(a)), _drop))

    return prims


def make_comparisons():
    """Equality and ordering comparisons."""
    cdef list prims = []

    a = TypeVariable(0)

    prims.append(Primitive('eq', arrow(a, a, BOOL), _eq))
    prims.append(Primitive('neq', arrow(a, a, BOOL), _neq))
    prims.append(Primitive('lt', arrow(INT, INT, BOOL), _lt))
    prims.append(Primitive('le', arrow(INT, INT, BOOL), _le))
    prims.append(Primitive('gt', arrow(INT, INT, BOOL), _gt))
    prims.append(Primitive('ge', arrow(INT, INT, BOOL), _ge))

    return prims


def make_boolean_ops():
    """Boolean operations."""
    cdef list prims = []

    a = TypeVariable(0)

    prims.append(Primitive('and', arrow(BOOL, BOOL, BOOL), _and))
    prims.append(Primitive('or', arrow(BOOL, BOOL, BOOL), _or))
    prims.append(Primitive('not', arrow(BOOL, BOOL), _not))
    prims.append(Primitive('if', arrow(BOOL, a, a, a), _if))

    return prims


def make_higher_order():
    """Map, filter, fold, and quantifiers."""
    cdef list prims = []

    a = TypeVariable(0)
    b = TypeVariable(1)

    prims.append(Primitive('map', arrow(arrow(a, b), ListType(a), ListType(b)), _map))
    prims.append(Primitive('filter', arrow(arrow(a, BOOL), ListType(a), ListType(a)), _filter))
    prims.append(Primitive('fold', arrow(arrow(b, a, b), b, ListType(a), b), _fold))
    prims.append(Primitive('all', arrow(arrow(a, BOOL), ListType(a), BOOL), _all))
    prims.append(Primitive('any', arrow(arrow(a, BOOL), ListType(a), BOOL), _any))
    prims.append(Primitive('count', arrow(arrow(a, BOOL), ListType(a), INT), _count))
    prims.append(Primitive('unique', arrow(ListType(a), ListType(a)), _unique))

    return prims


def make_arithmetic():
    """Basic arithmetic."""
    cdef list prims = []

    prims.append(Primitive('+', arrow(INT, INT, INT), _add))
    prims.append(Primitive('-', arrow(INT, INT, INT), _sub))
    prims.append(Primitive('*', arrow(INT, INT, INT), _mul))
    prims.append(Primitive('/', arrow(INT, INT, INT), _div))
    prims.append(Primitive('mod', arrow(INT, INT, INT), _mod))

    return prims


def make_pair_ops():
    """Pair operations for adjacent element patterns."""
    cdef list prims = []

    a = TypeVariable(0)
    b = TypeVariable(1)
    c = TypeVariable(2)

    prims.append(Primitive('pair', arrow(a, b, ListType(a)), _pair))
    prims.append(Primitive('fst', arrow(ListType(a), a), _fst))
    prims.append(Primitive('snd', arrow(ListType(a), a), _snd))
    prims.append(Primitive('pairs', arrow(ListType(a), ListType(ListType(a))), _pairs))
    prims.append(Primitive('zip_with', arrow(arrow(a, b, c), ListType(a), ListType(b), ListType(c)), _zip_with))

    return prims


# ============================================================================
# BUILD LEAN PRIMITIVE LIBRARY (CYTHON-NATIVE)
# ============================================================================

def build_lean_primitives_cy():
    """
    Build the Cython-native lean primitive library.

    Returns a list of Cython Primitive objects.
    """
    cdef list prims = []

    prims.extend(make_constants())      # ~13 primitives
    prims.extend(make_combinators())    # 4 primitives
    prims.extend(make_card_accessors()) # 4 primitives
    prims.extend(make_list_ops())       # 11 primitives
    prims.extend(make_comparisons())    # 6 primitives
    prims.extend(make_boolean_ops())    # 4 primitives
    prims.extend(make_higher_order())   # 7 primitives
    prims.extend(make_arithmetic())     # 5 primitives
    prims.extend(make_pair_ops())       # 5 primitives

    return prims


def build_lean_grammar_cy():
    """Build the Cython-native lean grammar."""
    prims = build_lean_primitives_cy()
    return uniform_grammar(prims)


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=== Cython-Native Lean Primitives Test ===\n")

    prims = build_lean_primitives_cy()
    print(f"Total primitives: {len(prims)}")

    grammar = build_lean_grammar_cy()
    print(f"Grammar productions: {len(grammar)}")

    # Test a few primitives
    print("\nSample primitives:")
    for p in prims[:5]:
        print(f"  {p.name}: {p.tp}")

    print("\n=== Test OK ===")
