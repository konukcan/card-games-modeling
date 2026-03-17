"""
Tests for the Python-to-DSL AST converter.

Covers all major patterns that appear in Phase 0 Python-freeform translations:
    - all/any with generator expressions
    - len(set(...)) for unique counts
    - sum(1 for c in hand if pred) for counting
    - hand[0], hand[-1] indexing
    - card.suit, card.rank, RANK_VALUES[card.rank] accessors
    - Comparisons: ==, >=, <, !=
    - Membership: card.suit in (Suit.X, Suit.Y)
    - Boolean operators: and, or, not
    - Arithmetic: +, -, %
    - len(hand)
"""

import sys
from pathlib import Path

# Allow importing from src/ and llm/
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pytest
from dreamcoder_core.program import Primitive, Application, Abstraction, Index

from llm.grammar_comparison.translation.python_parser import (
    python_to_ast,
    _prim,
    _int_prim,
    _apply2,
    PRIMITIVE_REGISTRY,
)


# ---------------------------------------------------------------------------
# Helpers for readable assertions
# ---------------------------------------------------------------------------

def _str(prog):
    """Get the string representation of a Program for debugging."""
    return str(prog)


# ---------------------------------------------------------------------------
# 1. all/any with generator expressions
# ---------------------------------------------------------------------------

class TestAllAnyPatterns:
    """all(expr for card in hand) and any(expr for card in hand) patterns."""

    def test_all_even_ranks(self):
        """all(RANK_VALUES[card.rank] % 2 == 0 for card in hand)

        Expected DSL:
            (lambda. (all (lambda. (eq (mod (rank_val $0) 2) 0)) $0))

        The outer lambda binds `hand` as $0.
        Inside `all`, the inner lambda binds `card` as $0 and `hand` shifts
        to $1. But since `hand` is used as the iterable argument to `all`
        (not inside the inner lambda body), it stays as $0 in the outer scope.
        """
        code = "rule = lambda hand: all(RANK_VALUES[card.rank] % 2 == 0 for card in hand)"
        prog = python_to_ast(code)

        # Top level should be Abstraction
        assert isinstance(prog, Abstraction)

        # Body: (all (lambda. ...) $0)
        body = prog.body
        assert isinstance(body, Application)

        # The argument to all should be $0 (hand)
        assert body.x == Index(0), f"Expected $0, got {body.x}"

        # The function part: (all (lambda. ...))
        all_app = body.f
        assert isinstance(all_app, Application)
        assert all_app.f == _prim('all')

        # The inner lambda
        inner_lambda = all_app.x
        assert isinstance(inner_lambda, Abstraction)

        # Inner body: (eq (mod (rank_val $0) 2) 0)
        inner_body = inner_lambda.body
        # Should be (eq X 0) where X = (mod (rank_val $0) 2)
        assert isinstance(inner_body, Application)

    def test_all_same_suit_as_first(self):
        """all(card.suit == hand[0].suit for card in hand)"""
        code = "rule = lambda hand: all(card.suit == hand[0].suit for card in hand)"
        prog = python_to_ast(code)

        assert isinstance(prog, Abstraction)
        body = prog.body

        # (all (lambda. ...) $0)
        assert isinstance(body, Application)
        assert body.x == Index(0)

    def test_any_is_hearts(self):
        """any(card.suit == Suit.HEARTS for card in hand)"""
        code = "rule = lambda hand: any(card.suit == Suit.HEARTS for card in hand)"
        prog = python_to_ast(code)

        assert isinstance(prog, Abstraction)
        body = prog.body

        # (any (lambda. ...) $0)
        all_app = body.f
        assert isinstance(all_app, Application)
        assert all_app.f == _prim('any')

    def test_all_red_suits(self):
        """all(card.suit in (Suit.HEARTS, Suit.DIAMONDS) for card in hand)

        The inner body should be:
            (or (eq (get_suit $0) HEARTS) (eq (get_suit $0) DIAMONDS))
        """
        code = "rule = lambda hand: all(card.suit in (Suit.HEARTS, Suit.DIAMONDS) for card in hand)"
        prog = python_to_ast(code)

        assert isinstance(prog, Abstraction)
        body = prog.body

        # Get inner lambda body
        inner_lambda = body.f.x
        assert isinstance(inner_lambda, Abstraction)

        # The body should use 'or' and 'eq'
        inner_body = inner_lambda.body
        prog_str = str(inner_body)
        assert 'or' in prog_str
        assert 'eq' in prog_str
        assert 'HEARTS' in prog_str
        assert 'DIAMONDS' in prog_str


# ---------------------------------------------------------------------------
# 2. len(set(...)) patterns
# ---------------------------------------------------------------------------

class TestLenSetPatterns:
    """len(set(expr for c in hand)) -> length(unique(map(lambda. expr, hand)))"""

    def test_num_unique_suits(self):
        """len(set(card.suit for card in hand)) == 1

        Expected DSL:
            (lambda. (eq (length (unique (map (lambda. (get_suit $0)) $0))) 1))
        """
        code = "rule = lambda hand: len(set(card.suit for card in hand)) == 1"
        prog = python_to_ast(code)

        assert isinstance(prog, Abstraction)
        body = prog.body

        # Should be (eq X 1) where X = (length (unique (map ...)))
        assert isinstance(body, Application)

        prog_str = str(prog)
        assert 'length' in prog_str
        assert 'unique' in prog_str
        assert 'map' in prog_str
        assert 'get_suit' in prog_str

    def test_num_unique_ranks(self):
        """len(set(card.rank for card in hand))"""
        code = "rule = lambda hand: len(set(card.rank for card in hand)) >= 3"
        prog = python_to_ast(code)

        prog_str = str(prog)
        assert 'length' in prog_str
        assert 'unique' in prog_str
        assert 'get_rank' in prog_str


# ---------------------------------------------------------------------------
# 3. Indexing: hand[0], hand[-1]
# ---------------------------------------------------------------------------

class TestIndexingPatterns:
    """hand[0] -> head, hand[-1] -> last, hand[n] -> at"""

    def test_hand_first_card(self):
        """hand[0].suit -> (get_suit (head $0))"""
        code = "rule = lambda hand: hand[0].suit == Suit.HEARTS"
        prog = python_to_ast(code)

        assert isinstance(prog, Abstraction)
        prog_str = str(prog)
        assert 'head' in prog_str
        assert 'get_suit' in prog_str
        assert 'HEARTS' in prog_str

    def test_hand_last_card(self):
        """hand[-1].suit -> (get_suit (last $0))"""
        code = "rule = lambda hand: hand[-1].suit == Suit.SPADES"
        prog = python_to_ast(code)

        prog_str = str(prog)
        assert 'last' in prog_str
        assert 'get_suit' in prog_str
        assert 'SPADES' in prog_str

    def test_first_equals_last_suit(self):
        """hand[0].suit == hand[-1].suit

        Expected DSL:
            (lambda. (eq (get_suit (head $0)) (get_suit (last $0))))
        """
        code = "rule = lambda hand: hand[0].suit == hand[-1].suit"
        prog = python_to_ast(code)

        assert isinstance(prog, Abstraction)
        body = prog.body

        # (eq (get_suit (head $0)) (get_suit (last $0)))
        assert isinstance(body, Application)
        prog_str = str(body)
        assert 'head' in prog_str
        assert 'last' in prog_str
        assert 'eq' in prog_str

    def test_hand_at_index(self):
        """hand[2] -> (at $0 2)"""
        code = "rule = lambda hand: hand[2].suit == Suit.CLUBS"
        prog = python_to_ast(code)

        prog_str = str(prog)
        assert 'at' in prog_str


# ---------------------------------------------------------------------------
# 4. Card accessors: card.suit, card.rank, RANK_VALUES[card.rank]
# ---------------------------------------------------------------------------

class TestCardAccessors:
    """Test card property accessors are correctly translated."""

    def test_rank_val(self):
        """RANK_VALUES[card.rank] inside all()"""
        code = "rule = lambda hand: all(RANK_VALUES[card.rank] > 5 for card in hand)"
        prog = python_to_ast(code)

        prog_str = str(prog)
        assert 'rank_val' in prog_str
        assert 'gt' in prog_str


# ---------------------------------------------------------------------------
# 5. Comparison operators
# ---------------------------------------------------------------------------

class TestComparisons:
    """Test ==, >=, <=, >, <, != operators."""

    def test_equality(self):
        code = "rule = lambda hand: len(hand) == 5"
        prog = python_to_ast(code)
        prog_str = str(prog)
        assert 'eq' in prog_str
        assert 'length' in prog_str

    def test_greater_equal(self):
        code = "rule = lambda hand: len(hand) >= 3"
        prog = python_to_ast(code)
        prog_str = str(prog)
        assert 'ge' in prog_str

    def test_less_than(self):
        code = "rule = lambda hand: len(hand) < 4"
        prog = python_to_ast(code)
        prog_str = str(prog)
        assert 'lt' in prog_str

    def test_not_equal(self):
        """!= should become not(eq(...))"""
        code = "rule = lambda hand: hand[0].suit != Suit.HEARTS"
        prog = python_to_ast(code)
        prog_str = str(prog)
        assert 'not' in prog_str
        assert 'eq' in prog_str


# ---------------------------------------------------------------------------
# 6. Membership: card.suit in (Suit.X, Suit.Y)
# ---------------------------------------------------------------------------

class TestMembership:
    """Test `x in (A, B, ...)` -> `(or (eq x A) (eq x B) ...)`"""

    def test_suit_in_two_suits(self):
        """card.suit in (Suit.HEARTS, Suit.DIAMONDS)"""
        code = "rule = lambda hand: all(card.suit in (Suit.HEARTS, Suit.DIAMONDS) for card in hand)"
        prog = python_to_ast(code)

        prog_str = str(prog)
        assert 'or' in prog_str
        assert 'HEARTS' in prog_str
        assert 'DIAMONDS' in prog_str

    def test_suit_in_three_suits(self):
        """card.suit in (Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS)"""
        code = "rule = lambda hand: all(card.suit in (Suit.HEARTS, Suit.DIAMONDS, Suit.CLUBS) for card in hand)"
        prog = python_to_ast(code)

        prog_str = str(prog)
        # Should have nested or's
        assert 'or' in prog_str
        assert 'HEARTS' in prog_str
        assert 'CLUBS' in prog_str


# ---------------------------------------------------------------------------
# 7. Boolean operators
# ---------------------------------------------------------------------------

class TestBooleanOperators:
    """Test and, or, not operators."""

    def test_and(self):
        """hand[0].suit == Suit.HEARTS and hand[-1].suit == Suit.SPADES"""
        code = "rule = lambda hand: hand[0].suit == Suit.HEARTS and hand[-1].suit == Suit.SPADES"
        prog = python_to_ast(code)

        assert isinstance(prog, Abstraction)
        prog_str = str(prog)
        assert 'and' in prog_str
        assert 'HEARTS' in prog_str
        assert 'SPADES' in prog_str

    def test_or(self):
        """hand[0].suit == Suit.HEARTS or hand[0].suit == Suit.DIAMONDS"""
        code = "rule = lambda hand: hand[0].suit == Suit.HEARTS or hand[0].suit == Suit.DIAMONDS"
        prog = python_to_ast(code)

        prog_str = str(prog)
        assert 'or' in prog_str

    def test_not(self):
        """not any(card.suit == Suit.HEARTS for card in hand)"""
        code = "rule = lambda hand: not any(card.suit == Suit.HEARTS for card in hand)"
        prog = python_to_ast(code)

        prog_str = str(prog)
        assert 'not' in prog_str
        assert 'any' in prog_str


# ---------------------------------------------------------------------------
# 8. Counting: sum(1 for c in hand if pred)
# ---------------------------------------------------------------------------

class TestCountingPatterns:
    """sum(1 for c in hand if pred) -> length(filter(lambda. pred, hand))"""

    def test_count_hearts(self):
        """sum(1 for card in hand if card.suit == Suit.HEARTS) >= 3

        Expected DSL:
            (lambda. (ge (length (filter (lambda. (eq (get_suit $0) HEARTS)) $0)) 3))
        """
        code = "rule = lambda hand: sum(1 for card in hand if card.suit == Suit.HEARTS) >= 3"
        prog = python_to_ast(code)

        assert isinstance(prog, Abstraction)
        prog_str = str(prog)
        assert 'ge' in prog_str
        assert 'length' in prog_str
        assert 'filter' in prog_str
        assert 'HEARTS' in prog_str

    def test_count_high_cards(self):
        """sum(1 for card in hand if RANK_VALUES[card.rank] >= 10)"""
        code = "rule = lambda hand: sum(1 for card in hand if RANK_VALUES[card.rank] >= 10) >= 2"
        prog = python_to_ast(code)

        prog_str = str(prog)
        assert 'filter' in prog_str
        assert 'rank_val' in prog_str


# ---------------------------------------------------------------------------
# 9. Arithmetic
# ---------------------------------------------------------------------------

class TestArithmetic:
    """Test +, -, % operators."""

    def test_modulo(self):
        """RANK_VALUES[card.rank] % 2 == 0"""
        code = "rule = lambda hand: all(RANK_VALUES[card.rank] % 2 == 0 for card in hand)"
        prog = python_to_ast(code)

        prog_str = str(prog)
        assert 'mod' in prog_str


# ---------------------------------------------------------------------------
# 10. len(hand) standalone
# ---------------------------------------------------------------------------

class TestLenHand:
    """len(hand) -> length($0)"""

    def test_len_hand(self):
        code = "rule = lambda hand: len(hand) == 5"
        prog = python_to_ast(code)

        assert isinstance(prog, Abstraction)
        body = prog.body

        # Should be (eq (length $0) 5)
        # The eq application should contain length
        prog_str = str(prog)
        assert 'length' in prog_str
        assert 'eq' in prog_str


# ---------------------------------------------------------------------------
# 11. De Bruijn index correctness
# ---------------------------------------------------------------------------

class TestDeBruijnIndices:
    """Verify correct de Bruijn index management across nesting levels."""

    def test_outer_hand_is_zero(self):
        """In `lambda hand: len(hand)`, hand should be $0."""
        code = "rule = lambda hand: len(hand)"
        prog = python_to_ast(code)

        # Abstraction body should be (length $0)
        body = prog.body
        assert isinstance(body, Application)
        assert body.f == _prim('length')
        assert body.x == Index(0)

    def test_inner_card_is_zero_hand_shifts_to_one(self):
        """In `all(card.suit ... for card in hand)`:
        - hand is $0 in the outer scope (argument to all)
        - card is $0 inside the inner lambda
        - hand would be $1 if referenced inside the inner lambda
        """
        code = "rule = lambda hand: all(card.suit == hand[0].suit for card in hand)"
        prog = python_to_ast(code)

        # Outer: Abstraction(body)
        # body: Application(Application(all, inner_lambda), $0)
        body = prog.body
        assert body.x == Index(0), "hand should be $0 as argument to all"

        # inner_lambda body should reference $0 for card and $1 for hand
        inner_lambda = body.f.x
        assert isinstance(inner_lambda, Abstraction)

        # Inside the inner lambda, `hand` references become $1
        inner_str = str(inner_lambda.body)
        assert '$1' in inner_str, (
            f"hand should be $1 inside inner lambda, got: {inner_str}"
        )
        assert '$0' in inner_str, (
            f"card should be $0 inside inner lambda, got: {inner_str}"
        )


# ---------------------------------------------------------------------------
# 12. Bare lambda (no assignment)
# ---------------------------------------------------------------------------

class TestBareInput:
    """Test that bare lambda expressions (without `rule = ...`) work."""

    def test_bare_lambda(self):
        code = "lambda hand: len(hand) == 5"
        prog = python_to_ast(code)
        assert isinstance(prog, Abstraction)
        assert 'length' in str(prog)

    def test_no_lambda_raises(self):
        with pytest.raises(ValueError, match="No lambda"):
            python_to_ast("x = 42")


# ---------------------------------------------------------------------------
# 13. Full integration: real Phase 0 patterns
# ---------------------------------------------------------------------------

class TestRealPatterns:
    """Test with patterns from actual Phase 0 Python-freeform translations."""

    def test_all_even(self):
        """All card rank values are even."""
        code = "rule = lambda hand: all(RANK_VALUES[card.rank] % 2 == 0 for card in hand)"
        prog = python_to_ast(code)
        assert isinstance(prog, Abstraction)
        # Should parse without error and produce valid AST
        prog_str = str(prog)
        assert 'all' in prog_str
        assert 'mod' in prog_str
        assert 'eq' in prog_str

    def test_flush(self):
        """All cards same suit: len(set(card.suit for card in hand)) == 1"""
        code = "rule = lambda hand: len(set(card.suit for card in hand)) == 1"
        prog = python_to_ast(code)
        prog_str = str(prog)
        assert 'length' in prog_str
        assert 'unique' in prog_str

    def test_count_hearts_ge_3(self):
        """At least 3 hearts."""
        code = "rule = lambda hand: sum(1 for card in hand if card.suit == Suit.HEARTS) >= 3"
        prog = python_to_ast(code)
        prog_str = str(prog)
        assert 'filter' in prog_str
        assert 'HEARTS' in prog_str

    def test_first_equals_last(self):
        """First and last card have same suit."""
        code = "rule = lambda hand: hand[0].suit == hand[-1].suit"
        prog = python_to_ast(code)
        prog_str = str(prog)
        assert 'head' in prog_str
        assert 'last' in prog_str

    def test_all_red_suits(self):
        """All cards red (hearts or diamonds)."""
        code = "rule = lambda hand: all(card.suit in (Suit.HEARTS, Suit.DIAMONDS) for card in hand)"
        prog = python_to_ast(code)
        prog_str = str(prog)
        assert 'all' in prog_str
        assert 'or' in prog_str
