"""
Python-to-DSL AST converter for card game rules.

Converts Python-freeform lambda expressions (from Phase 0 translations) into
DSL Program ASTs using Python's built-in `ast` module.

HOW IT WORKS:
    1. Parse the Python source code into a Python AST using `ast.parse`.
    2. Extract the lambda node from assignments like `rule = lambda hand: ...`
       or bare `lambda hand: ...` expressions.
    3. Walk the Python AST with pattern-matching rules that produce DSL
       Program nodes (Primitive, Application, Abstraction, Index).

DE BRUIJN INDEX TRACKING:
    The outer `lambda hand:` becomes the top-level Abstraction with `hand`
    mapped to $0. Inside `all(expr for card in hand)`, `card` becomes $0
    in the inner Abstraction and `hand` shifts to $1. Each nested lambda
    shifts outer variables up by 1.

SUPPORTED PATTERNS:
    - all/any(expr for card in hand)
    - len(set(expr for c in hand))  ->  length(unique(map(...)))
    - sum(1 for c in hand if pred)  ->  length(filter(...))
    - card.suit / card.rank / RANK_VALUES[card.rank]
    - hand[0] / hand[-1] / hand[n]
    - Comparisons: ==, !=, >=, <=, >, <
    - Boolean: and, or, not
    - Arithmetic: +, -, %
    - Suit constants: Suit.HEARTS, Suit.DIAMONDS, etc.
    - Membership: card.suit in (Suit.X, Suit.Y)
    - len(hand)
"""

import ast
import sys
from pathlib import Path
from typing import Dict, Optional

# Add src/ to path so we can import Program classes
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

from dreamcoder_core.program import Primitive, Application, Abstraction, Index, Program
from dreamcoder_core.type_system import (
    BOOL, INT, CARD, SUIT, RANK, HAND, arrow, ListType, TypeVariable, BaseType
)


# ---------------------------------------------------------------------------
# Primitive registry
# ---------------------------------------------------------------------------
# We create lightweight Primitive objects for AST construction. These only need
# name and type — value is None because we only care about the AST structure,
# not evaluation. The actual semantics come from src/dreamcoder_core/primitives.py.

COLOR = BaseType('color')

_a = TypeVariable(0)
_b = TypeVariable(1)

# A registry mapping primitive names to Primitive AST nodes.
# This is the single source of truth for what primitives the parser can emit.
PRIMITIVE_REGISTRY: Dict[str, Primitive] = {}


def _reg(name: str, tp, value=None) -> Primitive:
    """Register a primitive and return it."""
    p = Primitive(name, tp, value)
    PRIMITIVE_REGISTRY[name] = p
    return p


# Constants
_reg('CLUBS', SUIT)
_reg('DIAMONDS', SUIT)
_reg('HEARTS', SUIT)
_reg('SPADES', SUIT)
_reg('RED', COLOR)
_reg('BLACK', COLOR)
_reg('true', BOOL, True)
_reg('false', BOOL, False)
for i in range(6):
    _reg(str(i), INT, i)

# Card accessors
_reg('get_suit', arrow(CARD, SUIT))
_reg('get_rank', arrow(CARD, RANK))
_reg('rank_val', arrow(CARD, INT))
_reg('get_color', arrow(CARD, COLOR))

# Position access
_reg('head', arrow(ListType(_a), _a))
_reg('last', arrow(ListType(_a), _a))
_reg('at', arrow(ListType(_a), INT, _a))
_reg('length', arrow(ListType(_a), INT))

# Comparisons
_reg('eq', arrow(_a, _a, BOOL))
_reg('lt', arrow(INT, INT, BOOL))
_reg('le', arrow(INT, INT, BOOL))
_reg('gt', arrow(INT, INT, BOOL))
_reg('ge', arrow(INT, INT, BOOL))

# Boolean
_reg('and', arrow(BOOL, BOOL, BOOL))
_reg('or', arrow(BOOL, BOOL, BOOL))
_reg('not', arrow(BOOL, BOOL))

# Higher-order
_reg('map', arrow(arrow(_a, _b), ListType(_a), ListType(_b)))
_reg('filter', arrow(arrow(_a, BOOL), ListType(_a), ListType(_a)))
_reg('all', arrow(arrow(_a, BOOL), ListType(_a), BOOL))
_reg('any', arrow(arrow(_a, BOOL), ListType(_a), BOOL))
_reg('unique', arrow(ListType(_a), ListType(_a)))

# Arithmetic
_reg('+', arrow(INT, INT, INT))
_reg('-', arrow(INT, INT, INT))
_reg('mod', arrow(INT, INT, INT))


def _prim(name: str) -> Primitive:
    """Look up a primitive by name."""
    if name not in PRIMITIVE_REGISTRY:
        raise ValueError(f"Unknown primitive: {name!r}")
    return PRIMITIVE_REGISTRY[name]


def _int_prim(n: int) -> Primitive:
    """Return a Primitive for an integer literal.

    Small integers (0-5) use the pre-registered primitives.
    Larger integers get created on-the-fly with INT type.
    """
    name = str(n)
    if name in PRIMITIVE_REGISTRY:
        return PRIMITIVE_REGISTRY[name]
    # For integers outside 0-5, create a new primitive
    return Primitive(name, INT, n)


def _apply2(f: Program, x: Program, y: Program) -> Application:
    """Build a curried two-argument application: ((f x) y)."""
    return Application(Application(f, x), y)


# ---------------------------------------------------------------------------
# Suit name mapping
# ---------------------------------------------------------------------------

SUIT_MAP = {
    'HEARTS': 'HEARTS',
    'DIAMONDS': 'DIAMONDS',
    'CLUBS': 'CLUBS',
    'SPADES': 'SPADES',
}


# ---------------------------------------------------------------------------
# AST conversion
# ---------------------------------------------------------------------------

class _Converter:
    """Walks a Python AST and produces DSL Program nodes.

    Maintains an environment mapping Python variable names to de Bruijn
    indices. When we enter a new lambda/comprehension, the new variable
    becomes $0 and all existing bindings shift up by 1.

    Attributes:
        env: Maps Python variable names to their current de Bruijn index.
    """

    def __init__(self):
        self.env: Dict[str, int] = {}

    def _push_var(self, name: str) -> Dict[str, int]:
        """Push a new variable binding.

        The new variable gets index 0, and all existing variables shift up
        by 1. Returns the old environment so it can be restored later.

        This implements the de Bruijn index convention: each nested lambda
        shifts outer variables up by 1.
        """
        old_env = dict(self.env)
        # Shift all existing bindings up by 1
        self.env = {k: v + 1 for k, v in self.env.items()}
        # New variable is $0
        self.env[name] = 0
        return old_env

    def _pop_var(self, old_env: Dict[str, int]) -> None:
        """Restore the environment after leaving a scope."""
        self.env = old_env

    def convert(self, node: ast.AST) -> Program:
        """Main dispatch: convert a Python AST node to a DSL Program node.

        Pattern-matches on the Python AST node type and delegates to
        specialised handlers.
        """
        # --- Generator expressions: all(...), any(...), len(set(...)), etc.
        # These are handled by their enclosing Call node, not here.

        # --- Variable reference ---
        if isinstance(node, ast.Name):
            return self._convert_name(node)

        # --- Integer literal ---
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return _int_prim(node.value)

        # --- Boolean literal ---
        if isinstance(node, ast.Constant) and isinstance(node.value, bool):
            return _prim('true') if node.value else _prim('false')

        # --- Attribute access: card.suit, card.rank, Suit.HEARTS, etc. ---
        if isinstance(node, ast.Attribute):
            return self._convert_attribute(node)

        # --- Subscript: hand[0], hand[-1], RANK_VALUES[card.rank] ---
        if isinstance(node, ast.Subscript):
            return self._convert_subscript(node)

        # --- Comparison: x == y, x >= y, etc. ---
        if isinstance(node, ast.Compare):
            return self._convert_compare(node)

        # --- Boolean operators: x and y, x or y ---
        if isinstance(node, ast.BoolOp):
            return self._convert_boolop(node)

        # --- Unary operator: not x ---
        if isinstance(node, ast.UnaryOp):
            return self._convert_unaryop(node)

        # --- Binary operator: x + y, x - y, x % y ---
        if isinstance(node, ast.BinOp):
            return self._convert_binop(node)

        # --- Function call: all(...), any(...), len(...), sum(...), set(...) ---
        if isinstance(node, ast.Call):
            return self._convert_call(node)

        # --- IfExp (ternary): x if cond else y ---
        if isinstance(node, ast.IfExp):
            return self._convert_ifexp(node)

        raise NotImplementedError(f"Unsupported pattern: {ast.dump(node)}")

    # ------------------------------------------------------------------
    # Name references
    # ------------------------------------------------------------------

    def _convert_name(self, node: ast.Name) -> Program:
        """Convert a variable reference.

        Looks up the variable in the current environment to get its
        de Bruijn index. Raises an error for unknown variables (which
        likely means a pattern we haven't handled).
        """
        name = node.id
        if name in self.env:
            return Index(self.env[name])
        # Could be a boolean
        if name == 'True':
            return _prim('true')
        if name == 'False':
            return _prim('false')
        raise NotImplementedError(
            f"Unsupported variable reference: {name!r} "
            f"(env: {self.env})"
        )

    # ------------------------------------------------------------------
    # Attribute access
    # ------------------------------------------------------------------

    def _convert_attribute(self, node: ast.Attribute) -> Program:
        """Convert attribute access patterns.

        Handles:
            - card.suit  -> Application(get_suit, $card)
            - card.rank  -> Application(get_rank, $card)
            - Suit.HEARTS -> Primitive('HEARTS')
            - Color.RED   -> Primitive('RED')
        """
        attr = node.attr

        # Suit.X or Color.X constants
        if isinstance(node.value, ast.Name):
            obj_name = node.value.id
            if obj_name == 'Suit' and attr in SUIT_MAP:
                return _prim(SUIT_MAP[attr])
            if obj_name == 'Color':
                if attr in ('RED', 'BLACK'):
                    return _prim(attr)

        # card.suit -> Application(get_suit, <card>)
        if attr == 'suit':
            card_node = self.convert(node.value)
            return Application(_prim('get_suit'), card_node)

        # card.rank -> Application(get_rank, <card>)
        if attr == 'rank':
            card_node = self.convert(node.value)
            return Application(_prim('get_rank'), card_node)

        raise NotImplementedError(f"Unsupported attribute: {ast.dump(node)}")

    # ------------------------------------------------------------------
    # Subscript
    # ------------------------------------------------------------------

    def _convert_subscript(self, node: ast.Subscript) -> Program:
        """Convert subscript access patterns.

        Handles:
            - hand[0]   -> Application(head, $hand)
            - hand[-1]  -> Application(last, $hand)
            - hand[n]   -> Application(Application(at, $hand), n)
            - RANK_VALUES[card.rank] -> Application(rank_val, $card)
        """
        # RANK_VALUES[card.rank] -> rank_val(card)
        if (isinstance(node.value, ast.Name)
                and node.value.id == 'RANK_VALUES'):
            # The subscript should be card.rank
            inner = node.slice
            if isinstance(inner, ast.Attribute) and inner.attr == 'rank':
                card_node = self.convert(inner.value)
                return Application(_prim('rank_val'), card_node)

        # hand[0] -> head(hand)
        # hand[-1] -> last(hand)
        # hand[n] -> at(hand, n)
        hand_node = self.convert(node.value)
        idx = node.slice

        if isinstance(idx, ast.Constant) and isinstance(idx.value, int):
            n = idx.value
            if n == 0:
                return Application(_prim('head'), hand_node)
            elif n == -1:
                return Application(_prim('last'), hand_node)
            else:
                return _apply2(_prim('at'), hand_node, _int_prim(n))

        if isinstance(idx, ast.UnaryOp) and isinstance(idx.op, ast.USub):
            # hand[-n] for n > 1 — not commonly needed, but handle gracefully
            if isinstance(idx.operand, ast.Constant):
                n = idx.operand.value
                if n == 1:
                    return Application(_prim('last'), hand_node)

        raise NotImplementedError(f"Unsupported subscript: {ast.dump(node)}")

    # ------------------------------------------------------------------
    # Comparisons
    # ------------------------------------------------------------------

    def _convert_compare(self, node: ast.Compare) -> Program:
        """Convert comparison expressions.

        Handles:
            - x == y  -> (eq x' y')
            - x != y  -> (not (eq x' y'))
            - x >= y  -> (ge x' y')
            - x <= y  -> (le x' y')
            - x > y   -> (gt x' y')
            - x < y   -> (lt x' y')
            - x in (A, B, ...)  -> (or (eq x' A') (eq x' B') ...)

        For chained comparisons like `a == b == c`, only the first
        comparison is handled; chained comparisons are rare in card rules.
        """
        left = node.left
        ops = node.ops
        comparators = node.comparators

        # Single comparison
        if len(ops) == 1:
            op = ops[0]
            right = comparators[0]

            # Handle `x in (A, B, ...)` membership test
            if isinstance(op, ast.In):
                return self._convert_membership(left, right)

            # Handle `x not in (A, B, ...)` membership test
            if isinstance(op, ast.NotIn):
                inner = self._convert_membership(left, right)
                return Application(_prim('not'), inner)

            left_prog = self.convert(left)
            right_prog = self.convert(right)

            if isinstance(op, ast.Eq):
                return _apply2(_prim('eq'), left_prog, right_prog)
            elif isinstance(op, ast.NotEq):
                return Application(_prim('not'),
                                   _apply2(_prim('eq'), left_prog, right_prog))
            elif isinstance(op, ast.GtE):
                return _apply2(_prim('ge'), left_prog, right_prog)
            elif isinstance(op, ast.LtE):
                return _apply2(_prim('le'), left_prog, right_prog)
            elif isinstance(op, ast.Gt):
                return _apply2(_prim('gt'), left_prog, right_prog)
            elif isinstance(op, ast.Lt):
                return _apply2(_prim('lt'), left_prog, right_prog)

        raise NotImplementedError(f"Unsupported comparison: {ast.dump(node)}")

    def _convert_membership(self, elem: ast.AST, container: ast.AST) -> Program:
        """Convert `x in (A, B, ...)` to `(or (eq x A) (eq x B) ...)`.

        The container must be a Tuple literal (the common pattern in card rules).
        """
        if not isinstance(container, ast.Tuple):
            raise NotImplementedError(
                f"Unsupported 'in' container: {ast.dump(container)}"
            )

        elem_prog = self.convert(elem)
        # Build (or (eq elem A) (or (eq elem B) ...)) right-associatively
        comparisons = [
            _apply2(_prim('eq'), elem_prog, self.convert(elt))
            for elt in container.elts
        ]

        if len(comparisons) == 0:
            return _prim('false')
        if len(comparisons) == 1:
            return comparisons[0]

        # Chain with `or`: fold right
        result = comparisons[-1]
        for comp in reversed(comparisons[:-1]):
            result = _apply2(_prim('or'), comp, result)
        return result

    # ------------------------------------------------------------------
    # Boolean operators
    # ------------------------------------------------------------------

    def _convert_boolop(self, node: ast.BoolOp) -> Program:
        """Convert `x and y` / `x or y` to DSL.

        Handles arbitrary chains: `a and b and c` becomes
        `(and a' (and b' c'))` (right-associated).
        """
        values = node.values
        progs = [self.convert(v) for v in values]

        if isinstance(node.op, ast.And):
            prim_name = 'and'
        elif isinstance(node.op, ast.Or):
            prim_name = 'or'
        else:
            raise NotImplementedError(f"Unsupported BoolOp: {ast.dump(node)}")

        # Right-fold: a and b and c -> (and a (and b c))
        result = progs[-1]
        for p in reversed(progs[:-1]):
            result = _apply2(_prim(prim_name), p, result)
        return result

    # ------------------------------------------------------------------
    # Unary operators
    # ------------------------------------------------------------------

    def _convert_unaryop(self, node: ast.UnaryOp) -> Program:
        """Convert unary operators.

        Handles:
            - not x -> (not x')
            - -n    -> negative integer literal
        """
        if isinstance(node.op, ast.Not):
            operand = self.convert(node.operand)
            return Application(_prim('not'), operand)

        if isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, int):
                return _int_prim(-node.operand.value)

        raise NotImplementedError(f"Unsupported unary op: {ast.dump(node)}")

    # ------------------------------------------------------------------
    # Binary operators
    # ------------------------------------------------------------------

    def _convert_binop(self, node: ast.BinOp) -> Program:
        """Convert binary arithmetic operators.

        Handles:
            - x + y  -> (+ x' y')
            - x - y  -> (- x' y')
            - x % y  -> (mod x' y')
        """
        left = self.convert(node.left)
        right = self.convert(node.right)

        if isinstance(node.op, ast.Add):
            return _apply2(_prim('+'), left, right)
        elif isinstance(node.op, ast.Sub):
            return _apply2(_prim('-'), left, right)
        elif isinstance(node.op, ast.Mod):
            return _apply2(_prim('mod'), left, right)

        raise NotImplementedError(f"Unsupported binop: {ast.dump(node)}")

    # ------------------------------------------------------------------
    # Function calls
    # ------------------------------------------------------------------

    def _convert_call(self, node: ast.Call) -> Program:
        """Convert function call patterns.

        This is the most complex handler because Python uses calls for many
        patterns that map to different DSL constructs:

            all(expr for c in hand) -> (all (lambda. expr') $hand)
            any(expr for c in hand) -> (any (lambda. expr') $hand)
            len(set(expr for c in hand)) -> (length (unique (map (lambda. expr') $hand)))
            sum(1 for c in hand if pred) -> (length (filter (lambda. pred') $hand))
            len(hand) -> (length $hand)
        """
        func = node.func

        # Get function name for builtins
        func_name = None
        if isinstance(func, ast.Name):
            func_name = func.id

        # --- all(expr for card in hand) ---
        if func_name == 'all' and len(node.args) == 1:
            return self._convert_quantifier('all', node.args[0])

        # --- any(expr for card in hand) ---
        if func_name == 'any' and len(node.args) == 1:
            return self._convert_quantifier('any', node.args[0])

        # --- len(...) ---
        if func_name == 'len' and len(node.args) == 1:
            return self._convert_len_call(node.args[0])

        # --- sum(1 for c in hand if pred) ---
        if func_name == 'sum' and len(node.args) == 1:
            return self._convert_sum_call(node.args[0])

        # --- set(expr for c in hand) --- standalone set call
        if func_name == 'set' and len(node.args) == 1:
            return self._convert_set_call(node.args[0])

        raise NotImplementedError(f"Unsupported call: {ast.dump(node)}")

    def _convert_quantifier(self, quant_name: str, arg: ast.AST) -> Program:
        """Convert all/any with generator expression.

        Pattern: all(expr for card in hand)
        Result:  (all (lambda. expr') $hand)

        The generator variable (`card`) becomes $0 in the inner lambda,
        and the iterable (`hand`) is the argument.
        """
        if not isinstance(arg, ast.GeneratorExp):
            raise NotImplementedError(
                f"Expected generator in {quant_name}(), got: {ast.dump(arg)}"
            )

        gen = arg
        if len(gen.generators) != 1:
            raise NotImplementedError(
                f"Expected single generator in {quant_name}(), "
                f"got {len(gen.generators)}"
            )

        comp = gen.generators[0]
        var_name = comp.target.id  # e.g., 'card'
        iterable = self.convert(comp.iter)  # e.g., $hand

        # Push the generator variable as $0
        old_env = self._push_var(var_name)

        # Handle optional `if` clause in generator
        if comp.ifs:
            # all(expr for card in hand if pred)
            # -> all(lambda. and pred' expr') hand
            # But more naturally: all(lambda. if pred then expr else True/False) hand
            # For `all`: if there's an `if`, it means: all(expr for card in hand if pred)
            # which is equivalent to: all(lambda c. (not pred(c)) or expr(c), hand)
            # Actually, the simplest interpretation:
            # all(expr for c in hand if pred) = all(lambda c. pred(c) implies expr(c), hand)
            # But typically these patterns mean: for the filtered subset, expr holds.
            # The most common usage is: all(True for c in hand if pred) which is really
            # just checking that `any` items satisfy pred (but that's weird).
            # More commonly: all(expr for c in hand if something) - filter then check.
            # Let's handle the `if` by combining with `and`:
            # all(expr if pred) -> all(lambda. and pred' expr') is WRONG
            # Actually all(expr for c in hand if pred) means:
            #   "for every c in hand where pred(c), expr(c) holds"
            # which is logically: all(lambda c. not pred(c) or expr(c), hand)
            # But in practice, the common pattern is:
            #   sum(1 for c in hand if pred) which is handled separately
            # For all/any with if, raise for now
            if len(comp.ifs) > 0:
                # Common pattern: all(True for ... if pred) is really checking
                # that all items satisfy pred
                body = gen.elt
                if (isinstance(body, ast.Constant) and body.value is True):
                    # all(True for c in hand if pred) -> all(lambda. pred', hand)
                    pred_prog = self.convert(comp.ifs[0])
                    self._pop_var(old_env)
                    return _apply2(_prim(quant_name),
                                   Abstraction(pred_prog), iterable)
                # Otherwise, combine: not pred or expr (material implication)
                pred_prog = self.convert(comp.ifs[0])
                body_prog = self.convert(body)
                combined = _apply2(_prim('or'),
                                   Application(_prim('not'), pred_prog),
                                   body_prog)
                self._pop_var(old_env)
                return _apply2(_prim(quant_name),
                               Abstraction(combined), iterable)

        body_prog = self.convert(gen.elt)
        self._pop_var(old_env)

        return _apply2(_prim(quant_name), Abstraction(body_prog), iterable)

    def _convert_len_call(self, arg: ast.AST) -> Program:
        """Convert len(...) calls.

        Handles:
            - len(hand)  ->  (length $hand)
            - len(set(expr for c in hand))  ->  (length (unique (map (lambda. expr') $hand)))
            - len([x for x in hand if pred])  ->  length(filter(lambda. pred', $hand))
        """
        # len(set(...)) -> length(unique(map(lambda. ..., hand)))
        if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
            if arg.func.id == 'set' and len(arg.args) == 1:
                set_inner = arg.args[0]
                return self._convert_len_set(set_inner)

        # len(hand) or len(some_expression)
        inner = self.convert(arg)
        return Application(_prim('length'), inner)

    def _convert_len_set(self, inner: ast.AST) -> Program:
        """Convert len(set(expr for c in hand)) pattern.

        Result: (length (unique (map (lambda. expr') $hand)))

        This is the standard DSL way to express "number of unique X values".
        """
        if not isinstance(inner, ast.GeneratorExp):
            raise NotImplementedError(
                f"Expected generator in set(), got: {ast.dump(inner)}"
            )

        gen = inner
        if len(gen.generators) != 1:
            raise NotImplementedError(
                f"Expected single generator in set(), "
                f"got {len(gen.generators)}"
            )

        comp = gen.generators[0]
        var_name = comp.target.id
        iterable = self.convert(comp.iter)

        # Push generator variable
        old_env = self._push_var(var_name)
        body_prog = self.convert(gen.elt)
        self._pop_var(old_env)

        # Build: (length (unique (map (lambda. body) iterable)))
        map_expr = _apply2(_prim('map'), Abstraction(body_prog), iterable)
        unique_expr = Application(_prim('unique'), map_expr)
        return Application(_prim('length'), unique_expr)

    def _convert_sum_call(self, arg: ast.AST) -> Program:
        """Convert sum(...) calls.

        Handles:
            - sum(1 for c in hand if pred) -> (length (filter (lambda. pred') $hand))

        The pattern `sum(1 for c in hand if pred)` counts how many cards
        satisfy pred. This is equivalent to length(filter(pred, hand)).
        """
        if not isinstance(arg, ast.GeneratorExp):
            raise NotImplementedError(
                f"Expected generator in sum(), got: {ast.dump(arg)}"
            )

        gen = arg
        if len(gen.generators) != 1:
            raise NotImplementedError(
                f"Expected single generator in sum(), "
                f"got {len(gen.generators)}"
            )

        comp = gen.generators[0]

        # Check for sum(1 for c in hand if pred) pattern
        # The element should be 1 (or any constant) and there should be an if clause
        if (isinstance(gen.elt, ast.Constant) and gen.elt.value == 1
                and comp.ifs):
            var_name = comp.target.id
            iterable = self.convert(comp.iter)

            old_env = self._push_var(var_name)
            pred_prog = self.convert(comp.ifs[0])
            self._pop_var(old_env)

            # (length (filter (lambda. pred') iterable))
            filter_expr = _apply2(_prim('filter'),
                                  Abstraction(pred_prog), iterable)
            return Application(_prim('length'), filter_expr)

        # sum(expr for c in hand) without if -> map then sum
        # This is less common; raise for now
        raise NotImplementedError(
            f"Unsupported sum() pattern: {ast.dump(ast.Call(func=ast.Name(id='sum'), args=[arg], keywords=[]))}"
        )

    def _convert_set_call(self, arg: ast.AST) -> Program:
        """Convert standalone set(...) calls.

        Handles:
            - set(expr for c in hand) -> (unique (map (lambda. expr') $hand))
        """
        if not isinstance(arg, ast.GeneratorExp):
            raise NotImplementedError(
                f"Expected generator in set(), got: {ast.dump(arg)}"
            )

        gen = arg
        if len(gen.generators) != 1:
            raise NotImplementedError("Expected single generator in set()")

        comp = gen.generators[0]
        var_name = comp.target.id
        iterable = self.convert(comp.iter)

        old_env = self._push_var(var_name)
        body_prog = self.convert(gen.elt)
        self._pop_var(old_env)

        map_expr = _apply2(_prim('map'), Abstraction(body_prog), iterable)
        return Application(_prim('unique'), map_expr)

    def _convert_ifexp(self, node: ast.IfExp) -> Program:
        """Convert ternary if expression: x if cond else y.

        Result: (if cond' x' y')  using the DSL 'if' primitive.
        """
        cond = self.convert(node.test)
        then_val = self.convert(node.body)
        else_val = self.convert(node.orelse)

        if_prim = _prim('if') if 'if' in PRIMITIVE_REGISTRY else None
        if if_prim is None:
            # Fallback: register if primitive
            if_prim = _reg('if', arrow(BOOL, _a, _a, _a))

        return Application(Application(Application(if_prim, cond), then_val), else_val)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def python_to_ast(code: str) -> Program:
    """Convert a Python-freeform rule lambda to a DSL Program AST.

    Accepts code in these forms:
        - "rule = lambda hand: ..."
        - "lambda hand: ..."
        - "f = lambda hand: ..."

    The outer lambda becomes a top-level Abstraction, with the parameter
    (`hand`) mapped to de Bruijn index $0.

    Args:
        code: Python source code containing a lambda expression.

    Returns:
        A Program AST representing the rule.

    Raises:
        ValueError: If no lambda is found in the source code.
        NotImplementedError: If the lambda body contains unsupported patterns.
    """
    tree = ast.parse(code.strip(), mode='exec')

    # Find the lambda node
    lambda_node = _find_lambda(tree)
    if lambda_node is None:
        raise ValueError(f"No lambda found in: {code!r}")

    # Extract parameter name (should be 'hand')
    if len(lambda_node.args.args) != 1:
        raise ValueError(
            f"Expected single-argument lambda, got "
            f"{len(lambda_node.args.args)} args"
        )
    param_name = lambda_node.args.args[0].arg  # e.g., 'hand'

    # Convert the lambda body with param_name as $0
    converter = _Converter()
    old_env = converter._push_var(param_name)
    body_prog = converter.convert(lambda_node.body)
    converter._pop_var(old_env)

    # Wrap in top-level Abstraction
    return Abstraction(body_prog)


def _find_lambda(tree: ast.AST) -> Optional[ast.Lambda]:
    """Find the first lambda node in an AST.

    Searches assignment targets (`rule = lambda ...`) and expression
    statements (`lambda ...`).
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Lambda):
            return node
    return None
