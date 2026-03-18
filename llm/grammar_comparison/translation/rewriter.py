"""
Mechanical AST rewriter for grammar-specific translations.

PURPOSE:
    Given a Program AST built from Base grammar primitives, rewrite it to use
    only the primitives available in a target grammar. This is a deterministic
    tree transformation -- not search.

APPROACH:
    We subclass ProgramTransformer (from src/dreamcoder_core/program.py) to
    walk the AST and apply rewriting rules at each node. Simple substitutions
    replace a single primitive (e.g., first_half -> slice 0 3). Pattern-based
    rewrites match multi-node Application patterns (e.g., the adjacent_pairs
    idiom -> shifted_match).

SUPPORTED GRAMMARS:
    - "base":               Identity (no changes)
    - "swap-positional":    Replace positional prims with slice/shifted_match
    - "swap-distributional": Replace count_suit/count_color with count_where
    - "swap-both":          Both positional and distributional rewrites
    - "add-both":           Identity (superset of base)
    - "redundant":          Compression -- detect patterns and replace with shortcuts
    - "minimal":            Decompose most primitives; InexpressibleError only for
                            primitives that truly require fold/reduce (sum_ranks,
                            max_rank, min_rank, sort_by_rank, max_suit_count,
                            n_repeated_ranks, n_repeated_suits, running_sum,
                            suit_to_int, signum).

BIDIRECTIONAL REWRITING:
    - Expansion: base -> swap-*, minimal (decompose compound primitives)
    - Compression: base -> redundant (detect patterns, replace with shortcuts)

    For all grammars except minimal, the rewriter NEVER fails. For minimal,
    InexpressibleError is raised only for primitives that genuinely require
    constructs not in the minimal set (fold/reduce/sort).
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Set

# Allow imports from the main src/ tree
_SRC_DIR = str(Path(__file__).parent.parent.parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dreamcoder_core.program import (
    Program, Primitive, Application, Abstraction, Index,
    ProgramTransformer, collect_primitive_names,
)
from llm.grammar_comparison.grammars.grammar_factory import (
    _select_primitives, _MINIMAL_KEEP,
    _make_new_positional_primitives,
    _make_new_distributional_primitives,
    _make_cognitive_shortcuts,
    GRAMMAR_NAMES,
)


# =============================================================================
# Exception
# =============================================================================

class InexpressibleError(Exception):
    """Raised when a primitive cannot be expressed in the target grammar.

    Only raised for the minimal grammar, and only for primitives that
    genuinely require constructs not in the minimal set (fold/reduce/sort).

    Attributes:
        primitive_name: The name of the primitive that has no rewriting rule.
        target_grammar: The grammar that was being targeted.
    """

    def __init__(self, primitive_name: str, target_grammar: str):
        self.primitive_name = primitive_name
        self.target_grammar = target_grammar
        super().__init__(
            f"Primitive '{primitive_name}' cannot be expressed in "
            f"grammar '{target_grammar}'"
        )


# Primitives that genuinely cannot be mechanically decomposed into the minimal
# set because they require fold/reduce, sorting, or domain-specific aggregation.
MINIMAL_INEXPRESSIBLE: Set[str] = {
    "sum_ranks",        # Needs fold/reduce
    "max_rank",         # Needs fold/reduce
    "min_rank",         # Needs fold/reduce
    "sort_by_rank",     # Needs sort (not in minimal)
    "max_suit_count",   # Needs nested aggregation
    "n_repeated_ranks", # Needs count-of-counts
    "n_repeated_suits", # Needs count-of-counts
    "running_sum",      # Needs fold with accumulation
    "suit_to_int",      # Domain-specific mapping constant
    "signum",           # Could be done with if/lt/gt but if is not in minimal
}


# =============================================================================
# Helpers: build Primitive lookup tables from grammar factory
# =============================================================================

def _get_target_primitive_names(grammar_name: str) -> Set[str]:
    """Return the set of primitive names available in a target grammar."""
    prims = _select_primitives(grammar_name)
    return {p.name for p in prims}


def _make_prim_dict(primitives) -> Dict[str, Primitive]:
    """Build a name -> Primitive dictionary from a list of Primitive objects."""
    return {p.name: p for p in primitives}


# Cache Primitive objects for the new primitives so we can embed them in ASTs.
_NEW_POSITIONAL = _make_prim_dict(_make_new_positional_primitives())
_NEW_DISTRIBUTIONAL = _make_prim_dict(_make_new_distributional_primitives())
_COGNITIVE_SHORTCUTS = _make_prim_dict(_make_cognitive_shortcuts())

# We also need integer-constant Primitives for building slice arguments.
# Pull them from the base grammar.
_BASE_PRIMS = _make_prim_dict(_select_primitives("base"))


def _int_prim(n: int) -> Primitive:
    """Return the Primitive for integer constant n (from base grammar)."""
    name = str(n)
    if name in _BASE_PRIMS:
        return _BASE_PRIMS[name]
    # Fallback: build a literal int primitive
    from dreamcoder_core.type_system import INT
    return Primitive(name, INT, n)


def _prim(name: str) -> Primitive:
    """Return a base-grammar Primitive by name."""
    return _BASE_PRIMS[name]


# =============================================================================
# Pattern matching helpers
# =============================================================================

def _uncurry(program: Program):
    """Flatten a left-associative Application chain into (func, [arg1, arg2, ...]).

    Example:
        ((f a) b) c  ->  (f, [a, b, c])

    This is useful for matching curried calls like (count_suit hand HEARTS).
    """
    args = []
    while isinstance(program, Application):
        args.append(program.x)
        program = program.f
    args.reverse()
    return program, args


def _curry(func: Program, args) -> Program:
    """Build a left-associative Application chain: func arg1 arg2 ...

    This is the inverse of _uncurry.
    """
    result = func
    for arg in args:
        result = Application(result, arg)
    return result


def _is_prim(prog: Program, name: str) -> bool:
    """Check if a program node is a Primitive with the given name."""
    return isinstance(prog, Primitive) and prog.name == name


def _is_app_of(prog: Program, prim_name: str, nargs: int):
    """Check if prog is an application of prim_name with nargs arguments.

    Returns (True, [args]) or (False, []).
    """
    func, args = _uncurry(prog)
    if _is_prim(func, prim_name) and len(args) == nargs:
        return True, args
    return False, []


def _is_int_literal(prog: Program, value: int) -> bool:
    """Check if a program node is an integer literal with the given value."""
    return isinstance(prog, Primitive) and prog.name == str(value)


# =============================================================================
# Pair predicate extraction (for adjacent_pairs rewriting)
# =============================================================================

class _PairPredicateRewriter(ProgramTransformer):
    """Rewrite a pair-lambda body by replacing (head $0)/(last $0) with new indices.

    Given a lambda body that uses (head $0) and (last $0) to access pair elements,
    this transformer replaces:
      - (head $0) -> $1  (first element of the pair)
      - (last $0) -> $0  (second element of the pair)

    The result is suitable for wrapping in two lambdas to create a 2-arg predicate.
    """

    def __init__(self, pair_index: int = 0):
        """Args:
            pair_index: The de Bruijn index of the pair variable in the original lambda.
        """
        self._pair_index = pair_index

    def transform_application(self, program: Application) -> Program:
        func, args = _uncurry(program)

        # Match (head $pair_index) -> $1
        if isinstance(func, Primitive) and func.name == "head" and len(args) == 1:
            if isinstance(args[0], Index) and args[0].i == self._pair_index:
                return Index(1)

        # Match (last $pair_index) -> $0
        if isinstance(func, Primitive) and func.name == "last" and len(args) == 1:
            if isinstance(args[0], Index) and args[0].i == self._pair_index:
                return Index(0)

        # Default: recurse
        new_f = self.transform(program.f)
        new_x = self.transform(program.x)
        if new_f is program.f and new_x is program.x:
            return program
        return Application(new_f, new_x)


def _extract_pair_predicate(pred: Program) -> Optional[Program]:
    """Extract a two-argument predicate from a pair-lambda.

    The adjacent_pairs idiom uses predicates like:
        (lambda pair. f (ACCESSOR (head pair)) (ACCESSOR (last pair)))

    where ACCESSOR might be get_rank, get_color, rank_val, etc., or even
    a more complex expression. The key pattern is that (head $0) and (last $0)
    are the only references to the pair variable $0.

    We transform this by replacing (head $0) with $1 and (last $0) with $0,
    then wrapping in two abstractions to create a curried 2-arg predicate.

    Also handles compound predicates like:
        (lambda pair. or (eq (get_rank (head pair)) (get_rank (last pair)))
                        (eq (get_suit (head pair)) (get_suit (last pair))))

    Returns None if the pattern cannot be extracted.
    """
    if not isinstance(pred, Abstraction):
        return None

    body = pred.body

    # Check that the body only references $0 through (head $0) and (last $0).
    if _has_bare_index(body, 0):
        return None

    # Use the rewriter to replace (head $0) -> $1, (last $0) -> $0
    rewriter = _PairPredicateRewriter(pair_index=0)
    new_body = rewriter.transform(body)

    # Wrap in two abstractions: (lambda a. lambda b. new_body)
    return Abstraction(Abstraction(new_body))


def _has_bare_index(prog: Program, index: int) -> bool:
    """Check if prog contains a bare (non-head/last) reference to $index.

    A 'bare' reference means $index appears directly, NOT as the argument
    of (head $index) or (last $index). This is used to verify that a
    pair-lambda only accesses pair elements through head/last.
    """
    if isinstance(prog, Index):
        return prog.i == index

    if isinstance(prog, Primitive):
        return False

    if isinstance(prog, Application):
        func, args = _uncurry(prog)
        # (head $index) and (last $index) are OK -- not bare
        if isinstance(func, Primitive) and func.name in ("head", "last"):
            if len(args) == 1 and isinstance(args[0], Index) and args[0].i == index:
                return False
        # Check all sub-expressions
        return _has_bare_index(prog.f, index) or _has_bare_index(prog.x, index)

    if isinstance(prog, Abstraction):
        # Inside a lambda, the index shifts up by 1
        return _has_bare_index(prog.body, index + 1)

    return False


# =============================================================================
# Positional rewriter (for swap-positional and swap-both)
# =============================================================================

class _PositionalRewriter(ProgramTransformer):
    """Rewrite positional primitives to use slice / shifted_match.

    Rewriting rules:
      - first_half arg         -> slice 0 3 arg
      - second_half arg        -> slice 3 6 arg
      - take n arg             -> slice 0 n arg
      - drop n arg             -> slice n 6 arg
      - all/any pred (adjacent_pairs hand)  -> shifted_match 1 pred' hand
      - all/any pred (shifted_pairs hand)   -> shifted_match 2 pred' hand
      - filter pred (adjacent_pairs hand)   -> filter id (zip_with pred' (slice 0 5 hand) (slice 1 6 hand))
      - standalone adjacent_pairs hand      -> zip_with dummy (slice 0 5 hand) (slice 1 6 hand)
    """

    _POSITIONAL_NAMES = {
        "first_half", "second_half", "take", "drop",
        "adjacent_pairs", "shifted_pairs",
    }

    def __init__(self):
        self._slice = _NEW_POSITIONAL["slice"]
        self._shifted_match = _NEW_POSITIONAL["shifted_match"]

    def transform_application(self, program: Application) -> Program:
        """Pattern-match on the original tree, then recurse into sub-parts."""
        func, args = _uncurry(program)

        if isinstance(func, Primitive):
            # first_half arg -> slice 0 3 arg
            if func.name == "first_half" and len(args) == 1:
                return _curry(self._slice, [
                    _int_prim(0), _int_prim(3), self.transform(args[0])
                ])

            # second_half arg -> slice 3 6 arg
            if func.name == "second_half" and len(args) == 1:
                return _curry(self._slice, [
                    _int_prim(3), _int_prim(6), self.transform(args[0])
                ])

            # take n arg -> slice 0 n arg
            if func.name == "take" and len(args) == 2:
                return _curry(self._slice, [
                    _int_prim(0), self.transform(args[0]), self.transform(args[1])
                ])

            # drop n arg -> slice n 6 arg
            if func.name == "drop" and len(args) == 2:
                return _curry(self._slice, [
                    self.transform(args[0]), _int_prim(6), self.transform(args[1])
                ])

            # all/any pred (adjacent_pairs/shifted_pairs hand) -> shifted_match
            if func.name in ("all", "any") and len(args) == 2:
                list_arg = args[1]
                if isinstance(list_arg, Application):
                    lf, la = _uncurry(list_arg)
                    if isinstance(lf, Primitive) and lf.name in ("adjacent_pairs", "shifted_pairs") and len(la) == 1:
                        offset = 1 if lf.name == "adjacent_pairs" else 2
                        pred_arg = args[0]
                        hand_arg = la[0]
                        extracted = _extract_pair_predicate(pred_arg)
                        if extracted is not None:
                            return _curry(
                                self._shifted_match,
                                [_int_prim(offset), extracted,
                                 self.transform(hand_arg)]
                            )

            # filter pred (adjacent_pairs hand)
            # -> filter (lambda b. b) (zip_with pred' (slice 0 5 hand) (slice 1 6 hand))
            if func.name == "filter" and len(args) == 2:
                list_arg = args[1]
                if isinstance(list_arg, Application):
                    lf, la = _uncurry(list_arg)
                    if isinstance(lf, Primitive) and lf.name == "adjacent_pairs" and len(la) == 1:
                        pred_arg = args[0]
                        hand_arg = la[0]
                        extracted = _extract_pair_predicate(pred_arg)
                        if extracted is not None:
                            hand_t = self.transform(hand_arg)
                            left = _curry(self._slice, [_int_prim(0), _int_prim(5), hand_t])
                            right = _curry(self._slice, [_int_prim(1), _int_prim(6), hand_t])
                            bool_list = _curry(_prim("zip_with"), [extracted, left, right])
                            identity_pred = Abstraction(Index(0))
                            return _curry(_prim("filter"), [identity_pred, bool_list])

            # Standalone adjacent_pairs hand -> zip_with dummy (slice 0 5 hand) (slice 1 6 hand)
            if func.name == "adjacent_pairs" and len(args) == 1:
                hand_t = self.transform(args[0])
                left = _curry(self._slice, [_int_prim(0), _int_prim(5), hand_t])
                right = _curry(self._slice, [_int_prim(1), _int_prim(6), hand_t])
                true_pred = Abstraction(Abstraction(
                    _curry(_prim("eq"), [_int_prim(0), _int_prim(0)])
                ))
                return _curry(_prim("zip_with"), [true_pred, left, right])

            if func.name == "shifted_pairs" and len(args) == 1:
                hand_t = self.transform(args[0])
                left = _curry(self._slice, [_int_prim(0), _int_prim(4), hand_t])
                right = _curry(self._slice, [_int_prim(2), _int_prim(6), hand_t])
                true_pred = Abstraction(Abstraction(
                    _curry(_prim("eq"), [_int_prim(0), _int_prim(0)])
                ))
                return _curry(_prim("zip_with"), [true_pred, left, right])

        # No positional pattern matched -> default: recurse into children
        new_f = self.transform(program.f)
        new_x = self.transform(program.x)
        if new_f is program.f and new_x is program.x:
            return program
        return Application(new_f, new_x)

    def transform_primitive(self, program: Primitive) -> Program:
        """Pass through all primitives."""
        return program


# =============================================================================
# Distributional rewriter (for swap-distributional and swap-both)
# =============================================================================

class _DistributionalRewriter(ProgramTransformer):
    """Rewrite distributional primitives to use count_where.

    Rewriting rules:
      - count_suit hand S  -> count_where (lambda c. eq (get_suit c) S) hand
      - count_color hand C -> count_where (lambda c. eq (get_color c) C) hand
    """

    def __init__(self):
        self._count_where = _NEW_DISTRIBUTIONAL["count_where"]

    def transform_application(self, program: Application) -> Program:
        func, args = _uncurry(program)

        if isinstance(func, Primitive):
            if func.name == "count_suit" and len(args) == 2:
                hand_arg = self.transform(args[0])
                suit_val = self.transform(args[1])
                return self._build_count_where("get_suit", suit_val, hand_arg)

            if func.name == "count_color" and len(args) == 2:
                hand_arg = self.transform(args[0])
                color_val = self.transform(args[1])
                return self._build_count_where("get_color", color_val, hand_arg)

        new_f = self.transform(program.f)
        new_x = self.transform(program.x)
        if new_f is program.f and new_x is program.x:
            return program
        return Application(new_f, new_x)

    def _build_count_where(self, accessor_name: str, value: Program,
                           hand: Program) -> Program:
        """Build: count_where (lambda eq (accessor $0) value) hand"""
        accessor = _BASE_PRIMS[accessor_name]
        eq_prim = _BASE_PRIMS["eq"]
        shifted_value = value.shift(1, 0)
        lambda_body = Application(
            Application(eq_prim, Application(accessor, Index(0))),
            shifted_value
        )
        predicate = Abstraction(lambda_body)
        return Application(Application(self._count_where, predicate), hand)


# =============================================================================
# Minimal grammar rewriter
# =============================================================================

class _MinimalRewriter(ProgramTransformer):
    """Rewrite for the minimal grammar.

    Decomposes compound primitives into minimal-set equivalents.
    Raises InexpressibleError only for primitives requiring fold/reduce/sort.
    """

    def transform_application(self, program: Application) -> Program:
        func, args = _uncurry(program)

        if isinstance(func, Primitive):
            name = func.name

            if name == "last" and len(args) == 1:
                hand = self.transform(args[0])
                return _curry(_prim("at"), [
                    hand,
                    _curry(_prim("-"), [
                        Application(_prim("length"), hand),
                        _int_prim(1)
                    ])
                ])

            if name == "le" and len(args) == 2:
                x = self.transform(args[0])
                y = self.transform(args[1])
                return _curry(_prim("or"), [
                    _curry(_prim("lt"), [x, y]),
                    _curry(_prim("eq"), [x, y])
                ])

            if name == "ge" and len(args) == 2:
                x = self.transform(args[0])
                y = self.transform(args[1])
                return _curry(_prim("or"), [
                    _curry(_prim("gt"), [x, y]),
                    _curry(_prim("eq"), [x, y])
                ])

            if name == "has_suit" and len(args) == 2:
                hand = self.transform(args[0])
                suit_val = self.transform(args[1])
                return self._build_any_eq("get_suit", suit_val, hand)

            if name == "has_color" and len(args) == 2:
                hand = self.transform(args[0])
                color_val = self.transform(args[1])
                return self._build_any_eq("get_color", color_val, hand)

            if name == "count_suit" and len(args) == 2:
                hand = self.transform(args[0])
                suit_val = self.transform(args[1])
                return self._build_length_filter("get_suit", suit_val, hand)

            if name == "count_color" and len(args) == 2:
                hand = self.transform(args[0])
                color_val = self.transform(args[1])
                return self._build_length_filter("get_color", color_val, hand)

            if name == "n_unique_suits" and len(args) == 1:
                return self._build_n_unique("get_suit", self.transform(args[0]))

            if name == "n_unique_ranks" and len(args) == 1:
                return self._build_n_unique("get_rank", self.transform(args[0]))

            if name == "n_unique_colors" and len(args) == 1:
                return self._build_n_unique("get_color", self.transform(args[0]))

            if name == "half_len" and len(args) == 1:
                return _int_prim(3)

            if name in ("take", "drop", "first_half", "second_half",
                        "adjacent_pairs", "shifted_pairs"):
                raise InexpressibleError(name, "minimal")

            if name in MINIMAL_INEXPRESSIBLE:
                raise InexpressibleError(name, "minimal")

            if name == "if":
                raise InexpressibleError(name, "minimal")

        new_f = self.transform(program.f)
        new_x = self.transform(program.x)
        if new_f is program.f and new_x is program.x:
            return program
        return Application(new_f, new_x)

    def transform_primitive(self, program: Primitive) -> Program:
        name = program.name
        if name in _MINIMAL_KEEP:
            return program

        if name == "true":
            return _curry(_prim("eq"), [_int_prim(0), _int_prim(0)])

        if name == "false":
            return Application(_prim("not"),
                               _curry(_prim("eq"), [_int_prim(0), _int_prim(0)]))

        if name == "le":
            body = _curry(_prim("or"), [
                _curry(_prim("lt"), [Index(1), Index(0)]),
                _curry(_prim("eq"), [Index(1), Index(0)])
            ])
            return Abstraction(Abstraction(body))

        if name == "ge":
            body = _curry(_prim("or"), [
                _curry(_prim("gt"), [Index(1), Index(0)]),
                _curry(_prim("eq"), [Index(1), Index(0)])
            ])
            return Abstraction(Abstraction(body))

        if name == "last":
            body = _curry(_prim("at"), [
                Index(0),
                _curry(_prim("-"), [
                    Application(_prim("length"), Index(0)),
                    _int_prim(1)
                ])
            ])
            return Abstraction(body)

        if name == "n_unique_suits":
            return Abstraction(self._build_n_unique("get_suit", Index(0)))
        if name == "n_unique_ranks":
            return Abstraction(self._build_n_unique("get_rank", Index(0)))
        if name == "n_unique_colors":
            return Abstraction(self._build_n_unique("get_color", Index(0)))

        if name == "has_suit":
            pred_body = _curry(_prim("eq"), [
                Application(_prim("get_suit"), Index(0)), Index(1)])
            return Abstraction(Abstraction(
                _curry(_prim("any"), [Abstraction(pred_body), Index(1)])))

        if name == "has_color":
            pred_body = _curry(_prim("eq"), [
                Application(_prim("get_color"), Index(0)), Index(1)])
            return Abstraction(Abstraction(
                _curry(_prim("any"), [Abstraction(pred_body), Index(1)])))

        if name == "count_suit":
            pred_body = _curry(_prim("eq"), [
                Application(_prim("get_suit"), Index(0)), Index(1)])
            return Abstraction(Abstraction(Application(_prim("length"),
                _curry(_prim("filter"), [Abstraction(pred_body), Index(1)]))))

        if name == "count_color":
            pred_body = _curry(_prim("eq"), [
                Application(_prim("get_color"), Index(0)), Index(1)])
            return Abstraction(Abstraction(Application(_prim("length"),
                _curry(_prim("filter"), [Abstraction(pred_body), Index(1)]))))

        if name == "half_len":
            return _int_prim(3)

        if name in MINIMAL_INEXPRESSIBLE or name in (
            "take", "drop", "first_half", "second_half",
            "adjacent_pairs", "shifted_pairs", "if",
        ):
            raise InexpressibleError(name, "minimal")

        raise InexpressibleError(name, "minimal")

    @staticmethod
    def _build_any_eq(accessor_name: str, value: Program, hand: Program) -> Program:
        shifted_value = value.shift(1, 0)
        lambda_body = _curry(_prim("eq"), [
            Application(_prim(accessor_name), Index(0)), shifted_value])
        return _curry(_prim("any"), [Abstraction(lambda_body), hand])

    @staticmethod
    def _build_length_filter(accessor_name: str, value: Program, hand: Program) -> Program:
        shifted_value = value.shift(1, 0)
        lambda_body = _curry(_prim("eq"), [
            Application(_prim(accessor_name), Index(0)), shifted_value])
        return Application(_prim("length"),
            _curry(_prim("filter"), [Abstraction(lambda_body), hand]))

    @staticmethod
    def _build_n_unique(accessor_name: str, hand: Program) -> Program:
        return Application(_prim("length"),
            Application(_prim("unique"),
                _curry(_prim("map"), [_prim(accessor_name), hand])))


# =============================================================================
# Compression rewriter for redundant grammar (G6)
# =============================================================================

class _RedundantCompressor(ProgramTransformer):
    """Detect base-grammar patterns and compress them into cognitive shortcuts.

    Compression runs bottom-up: recurse into children first, then try to
    compress the resulting tree.

    Patterns detected:
      1. n_unique(KEY, hand):      length(unique(map(KEY)(hand)))
      2. all_different(KEY, hand): eq(n_unique_expr)(length(hand))
      3. all_same(KEY, hand):      lt/le/eq(n_unique_expr)(2/1)
      4. exactly_n(n, pred, hand): eq(length(filter(pred)(hand)))(n)
      5. at_least_n(n, pred, hand): ge(length(filter(pred)(hand)))(n)
    """

    def transform_application(self, program: Application) -> Program:
        new_f = self.transform(program.f)
        new_x = self.transform(program.x)
        if new_f is not program.f or new_x is not program.x:
            program = Application(new_f, new_x)

        result = self._try_compress(program)
        return result if result is not None else program

    def _try_compress(self, program: Program) -> Optional[Program]:
        # Try more specific patterns first
        for matcher in (self._match_all_different, self._match_all_same,
                        self._match_n_unique, self._match_exactly_n,
                        self._match_at_least_n):
            result = matcher(program)
            if result is not None:
                return result
        return None

    def _match_n_unique(self, prog: Program) -> Optional[Program]:
        extracted = self._extract_n_unique_pattern(prog)
        if extracted is not None:
            key_fn, hand = extracted
            return _curry(_COGNITIVE_SHORTCUTS["n_unique"], [key_fn, hand])
        return None

    def _match_all_different(self, prog: Program) -> Optional[Program]:
        func, args = _uncurry(prog)
        if not _is_prim(func, "eq") or len(args) != 2:
            return None

        for a, b in [(args[0], args[1]), (args[1], args[0])]:
            n_uniq_match = self._extract_n_unique_pattern(a)
            if n_uniq_match is None:
                n_uniq_match = self._extract_n_unique_shortcut(a)
            if n_uniq_match is None:
                continue
            key_fn, hand_a = n_uniq_match

            ok, len_args = _is_app_of(b, "length", 1)
            if not ok:
                continue
            if str(len_args[0]) == str(hand_a):
                return _curry(_COGNITIVE_SHORTCUTS["all_different"], [key_fn, hand_a])

        return None

    def _match_all_same(self, prog: Program) -> Optional[Program]:
        func, args = _uncurry(prog)

        def _try_extract(expr):
            for extractor in (self._extract_n_unique_pattern,
                              self._extract_compound_n_unique,
                              self._extract_n_unique_shortcut):
                m = extractor(expr)
                if m:
                    return m
            return None

        if _is_prim(func, "lt") and len(args) == 2 and _is_int_literal(args[1], 2):
            match = _try_extract(args[0])
            if match:
                return _curry(_COGNITIVE_SHORTCUTS["all_same"], [match[0], match[1]])

        if _is_prim(func, "le") and len(args) == 2 and _is_int_literal(args[1], 1):
            match = _try_extract(args[0])
            if match:
                return _curry(_COGNITIVE_SHORTCUTS["all_same"], [match[0], match[1]])

        if _is_prim(func, "eq") and len(args) == 2:
            for a, b in [(args[0], args[1]), (args[1], args[0])]:
                if not _is_int_literal(b, 1):
                    continue
                match = _try_extract(a)
                if match:
                    return _curry(_COGNITIVE_SHORTCUTS["all_same"], [match[0], match[1]])

        return None

    def _match_exactly_n(self, prog: Program) -> Optional[Program]:
        func, args = _uncurry(prog)
        if not _is_prim(func, "eq") or len(args) != 2:
            return None

        for a, b in [(args[0], args[1]), (args[1], args[0])]:
            lf_match = self._extract_length_filter(a)
            if lf_match is not None and isinstance(b, Primitive) and b.name.isdigit():
                pred, hand = lf_match
                return _curry(_COGNITIVE_SHORTCUTS["exactly_n"], [b, pred, hand])

        return None

    def _match_at_least_n(self, prog: Program) -> Optional[Program]:
        func, args = _uncurry(prog)
        if not _is_prim(func, "ge") or len(args) != 2:
            return None

        lf_match = self._extract_length_filter(args[0])
        if lf_match is not None and isinstance(args[1], Primitive) and args[1].name.isdigit():
            pred, hand = lf_match
            return _curry(_COGNITIVE_SHORTCUTS["at_least_n"], [args[1], pred, hand])

        return None

    # --- Extraction helpers ---

    @staticmethod
    def _extract_n_unique_pattern(prog: Program):
        """Extract (key_fn, hand) from length(unique(map(key_fn)(hand)))."""
        ok, length_args = _is_app_of(prog, "length", 1)
        if not ok:
            return None
        ok, unique_args = _is_app_of(length_args[0], "unique", 1)
        if not ok:
            return None
        ok, map_args = _is_app_of(unique_args[0], "map", 2)
        if not ok:
            return None
        if not isinstance(map_args[0], Primitive):
            return None
        return map_args[0], map_args[1]

    @staticmethod
    def _extract_compound_n_unique(prog: Program):
        """Extract (key_fn, hand) from n_unique_suits/ranks/colors(hand)."""
        _MAP = {"n_unique_suits": "get_suit", "n_unique_ranks": "get_rank",
                "n_unique_colors": "get_color"}
        func, args = _uncurry(prog)
        if isinstance(func, Primitive) and func.name in _MAP and len(args) == 1:
            return _prim(_MAP[func.name]), args[0]
        return None

    @staticmethod
    def _extract_n_unique_shortcut(prog: Program):
        """Extract (key_fn, hand) from n_unique(key_fn, hand) (cognitive shortcut)."""
        ok, n_unique_args = _is_app_of(prog, "n_unique", 2)
        if ok and isinstance(n_unique_args[0], Primitive):
            return n_unique_args[0], n_unique_args[1]
        return None

    @staticmethod
    def _extract_length_filter(prog: Program):
        """Extract (pred, hand) from length(filter(pred)(hand))."""
        ok, length_args = _is_app_of(prog, "length", 1)
        if not ok:
            return None
        ok, filter_args = _is_app_of(length_args[0], "filter", 2)
        if not ok:
            return None
        return filter_args[0], filter_args[1]


# =============================================================================
# Public API
# =============================================================================

def rewrite_ast(program: Program, target_grammar: str) -> Program:
    """Rewrite a Base grammar AST to use only the target grammar's primitives.

    For non-minimal grammars (base, swap-*, add-both, redundant), the rewriter
    NEVER raises InexpressibleError. For the minimal grammar, InexpressibleError
    is raised only for primitives that genuinely require fold/reduce/sort.

    Args:
        program: A Program AST built from base-grammar primitives.
        target_grammar: One of the 7 grammar names from GRAMMAR_NAMES.

    Returns:
        A new Program AST using only primitives from the target grammar.

    Raises:
        InexpressibleError: Only for "minimal" grammar, for fold/reduce prims.
        ValueError: If target_grammar is not recognized.
    """
    if target_grammar not in GRAMMAR_NAMES:
        raise ValueError(
            f"Unknown grammar '{target_grammar}'. "
            f"Choose from: {GRAMMAR_NAMES}"
        )

    if target_grammar in ("base", "add-both"):
        return program

    if target_grammar == "redundant":
        return _RedundantCompressor().transform(program)

    if target_grammar == "swap-positional":
        return _PositionalRewriter().transform(program)

    if target_grammar == "swap-distributional":
        return _DistributionalRewriter().transform(program)

    if target_grammar == "swap-both":
        result = _PositionalRewriter().transform(program)
        result = _DistributionalRewriter().transform(result)
        return result

    if target_grammar == "minimal":
        return _MinimalRewriter().transform(program)

    raise ValueError(f"Unhandled grammar: {target_grammar}")  # pragma: no cover
