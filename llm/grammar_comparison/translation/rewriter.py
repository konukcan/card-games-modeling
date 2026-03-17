"""
Mechanical AST rewriter for grammar-specific translations.

PURPOSE:
    Given a Program AST built from Base grammar primitives, rewrite it to use
    only the primitives available in a target grammar. This is a deterministic
    tree transformation — not search.

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
    - "redundant":          Identity (superset of base)
    - "minimal":            Keep only minimal-set primitives; raise if missing
"""

import sys
from pathlib import Path
from typing import Dict, Optional, Set

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
    GRAMMAR_NAMES,
)


# =============================================================================
# Exception
# =============================================================================

class InexpressibleError(Exception):
    """Raised when a primitive cannot be expressed in the target grammar.

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
      - all pred (adjacent_pairs hand)  -> shifted_match 1 pred' hand
        where pred' wraps the pair-predicate into a two-argument predicate
      - standalone adjacent_pairs       -> InexpressibleError

    IMPORTANT: We pattern-match on the ORIGINAL AST nodes before recursing,
    because the primitives we want to rewrite (first_half, take, etc.) are
    themselves the function heads. If we recurse first, transform_primitive
    would see them as bare primitives and could raise prematurely.
    """

    # Names of positional primitives that this rewriter handles.
    _POSITIONAL_NAMES = {
        "first_half", "second_half", "take", "drop",
        "adjacent_pairs", "shifted_pairs",
    }

    def __init__(self):
        self._slice = _NEW_POSITIONAL["slice"]
        self._shifted_match = _NEW_POSITIONAL["shifted_match"]

    def transform_application(self, program: Application) -> Program:
        """Pattern-match on the original tree, then recurse into sub-parts.

        We uncurry the original application to see the full call pattern
        (e.g., first_half applied to its argument) BEFORE transforming
        children. Only the non-matched sub-trees get recursed into.
        """
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

            # all/any pred (adjacent_pairs hand)
            # -> shifted_match 1 pred' hand
            if func.name in ("all", "any") and len(args) == 2:
                list_arg = args[1]
                if isinstance(list_arg, Application):
                    lf, la = _uncurry(list_arg)
                    if (isinstance(lf, Primitive)
                            and lf.name == "adjacent_pairs"
                            and len(la) == 1):
                        pred_arg = args[0]
                        hand_arg = la[0]
                        extracted = _extract_pair_predicate(pred_arg)
                        if extracted is not None:
                            return _curry(
                                self._shifted_match,
                                [_int_prim(1), extracted,
                                 self.transform(hand_arg)]
                            )
                        raise InexpressibleError(
                            "adjacent_pairs", "swap-positional"
                        )

            # Standalone adjacent_pairs (not inside all/any)
            if func.name == "adjacent_pairs":
                raise InexpressibleError("adjacent_pairs", "swap-positional")

            # shifted_pairs is also removed
            if func.name == "shifted_pairs":
                raise InexpressibleError("shifted_pairs", "swap-positional")

        # No positional pattern matched -> default: recurse into children
        new_f = self.transform(program.f)
        new_x = self.transform(program.x)
        if new_f is program.f and new_x is program.x:
            return program
        return Application(new_f, new_x)

    def transform_primitive(self, program: Primitive) -> Program:
        """Pass through all primitives.

        Positional primitives that appear bare (not as the head of an
        Application) could in theory be passed as higher-order arguments.
        But since transform_application handles all applied cases first,
        we only reach here for truly bare references. We leave them as-is
        rather than raising, because they might appear inside a lambda body
        that will be pattern-matched at a higher level.
        """
        return program


def _extract_pair_predicate(pred: Program) -> Optional[Program]:
    """Try to extract a two-argument predicate from a pair-lambda.

    The adjacent_pairs idiom uses predicates like:
        (lambda pair. f (head pair) (last pair))
    or equivalently:
        (lambda pair. f (at pair 0) (at pair 1))

    We want to extract f as a curried 2-argument function:
        (lambda a. lambda b. f a b)

    For simple cases where the lambda body is:
        (pred (head $0) (last $0))  ->  pred

    Returns None if the pattern cannot be extracted.
    """
    if not isinstance(pred, Abstraction):
        return None

    body = pred.body
    # Pattern: (func (head $0) (last $0))
    # Uncurried: func applied to [head $0, last $0]
    func, args = _uncurry(body)

    if len(args) == 2:
        # Check arg[0] is (head $0) and arg[1] is (last $0)
        a0_f, a0_args = _uncurry(args[0])
        a1_f, a1_args = _uncurry(args[1])

        is_head_dollar0 = (
            isinstance(a0_f, Primitive) and a0_f.name == "head"
            and len(a0_args) == 1 and isinstance(a0_args[0], Index) and a0_args[0].i == 0
        )
        is_last_dollar0 = (
            isinstance(a1_f, Primitive) and a1_f.name == "last"
            and len(a1_args) == 1 and isinstance(a1_args[0], Index) and a1_args[0].i == 0
        )

        if is_head_dollar0 and is_last_dollar0:
            # The predicate body is: func (head $0) (last $0)
            # We need to return: (lambda a. lambda b. func a b)
            # = Abstraction(Abstraction(Application(Application(func_shifted, $1), $0)))
            # But func might reference $0 from the outer lambda, so we need to shift.
            # Actually, func should NOT reference $0 (it's a combinator like eq or lt).
            # If func has no free indices, we can just wrap it.
            free = func.free_indices(0)
            if not free:
                # func is closed -> wrap in two lambdas
                # (lambda. lambda. func $1 $0)
                inner_body = Application(
                    Application(func, Index(1)),
                    Index(0)
                )
                return Abstraction(Abstraction(inner_body))

    return None


# =============================================================================
# Distributional rewriter (for swap-distributional and swap-both)
# =============================================================================

class _DistributionalRewriter(ProgramTransformer):
    """Rewrite distributional primitives to use count_where.

    Rewriting rules:
      - count_suit hand S  -> count_where (lambda c. eq (get_suit c) S) hand
      - count_color hand C -> count_where (lambda c. eq (get_color c) C) hand

    Note: count_rank does not exist in the base grammar, so no rule needed.

    Like _PositionalRewriter, we pattern-match on the ORIGINAL tree before
    recursing so that count_suit/count_color are caught as application heads
    rather than triggering transform_primitive prematurely.
    """

    def __init__(self):
        self._count_where = _NEW_DISTRIBUTIONAL["count_where"]

    def transform_application(self, program: Application) -> Program:
        """Pattern-match distributional calls before recursing."""
        func, args = _uncurry(program)

        if isinstance(func, Primitive):
            # count_suit hand S -> count_where (λ eq (get_suit $0) S) hand
            # Base type: count_suit : HAND -> SUIT -> INT
            # Curried call: ((count_suit hand) S), so args = [hand, S]
            if func.name == "count_suit" and len(args) == 2:
                hand_arg = self.transform(args[0])
                suit_val = self.transform(args[1])
                return self._build_count_where("get_suit", suit_val, hand_arg)

            # count_color hand C -> count_where (λ eq (get_color $0) C) hand
            if func.name == "count_color" and len(args) == 2:
                hand_arg = self.transform(args[0])
                color_val = self.transform(args[1])
                return self._build_count_where("get_color", color_val, hand_arg)

        # No distributional pattern matched -> default: recurse into children
        new_f = self.transform(program.f)
        new_x = self.transform(program.x)
        if new_f is program.f and new_x is program.x:
            return program
        return Application(new_f, new_x)

    def _build_count_where(self, accessor_name: str, value: Program,
                           hand: Program) -> Program:
        """Build: count_where (λ eq (accessor $0) value) hand

        The lambda body is: eq (accessor $0) value
        Since value might contain free variables, we need to shift it
        when placing it inside the lambda body.

        Args:
            accessor_name: "get_suit" or "get_color"
            value: The suit/color constant (e.g., HEARTS, RED)
            hand: The hand argument
        """
        accessor = _BASE_PRIMS[accessor_name]
        eq_prim = _BASE_PRIMS["eq"]

        # Inside the lambda, $0 refers to the card.
        # value needs to be shifted up by 1 since we're inside a new lambda.
        shifted_value = value.shift(1, 0)

        # Build: eq (accessor $0) shifted_value
        lambda_body = Application(
            Application(eq_prim, Application(accessor, Index(0))),
            shifted_value
        )
        predicate = Abstraction(lambda_body)

        # count_where predicate hand
        return Application(Application(self._count_where, predicate), hand)


# =============================================================================
# Minimal grammar rewriter
# =============================================================================

class _MinimalRewriter(ProgramTransformer):
    """Rewrite for the minimal grammar.

    The minimal grammar keeps only primitives in _MINIMAL_KEEP.
    Some compound primitives can be decomposed:
      - first_half hand  -> take 3 hand   (take IS in minimal via head/at etc.
        -- actually take is NOT in minimal, check below)
      - Actually, minimal keeps: head, at, map, filter, all, any, zip_with,
        length, unique, reverse, +, -, mod, eq, lt, gt, not, and, or,
        0-5, get_suit, get_rank, rank_val, get_color, CLUBS, DIAMONDS,
        HEARTS, SPADES, RED, BLACK

    Primitives NOT in minimal that we might encounter:
      - first_half, second_half, take, drop -> not in minimal
      - adjacent_pairs, shifted_pairs -> not in minimal
      - count_suit, count_color -> not in minimal
      - has_suit, has_color -> not in minimal
      - n_unique_suits, n_unique_ranks, n_unique_colors -> not in minimal
      - all_same_suit, all_same_color -> not in minimal
      - sum_ranks, max_rank, min_rank -> not in minimal
      - half_len -> not in minimal
      - le, ge -> not in minimal
      - last -> IS in minimal? No, 'last' is not in _MINIMAL_KEEP.
        Wait: let me check. _MINIMAL_KEEP has 'head' but not 'last'.
        Actually looking again: it does NOT have 'last'.

    For the minimal grammar, any primitive not in _MINIMAL_KEEP raises
    InexpressibleError, since mechanical decomposition of arbitrary
    compound primitives is beyond scope.
    """

    def transform_primitive(self, program: Primitive) -> Program:
        if program.name not in _MINIMAL_KEEP:
            raise InexpressibleError(program.name, "minimal")
        return program


# =============================================================================
# Public API
# =============================================================================

def rewrite_ast(program: Program, target_grammar: str) -> Program:
    """Rewrite a Base grammar AST to use only the target grammar's primitives.

    This applies deterministic tree transformations to replace base-grammar
    primitives with their equivalents in the target grammar. The rewriting
    is purely structural — no search or enumeration is involved.

    Args:
        program: A Program AST built from base-grammar primitives
            (typically parsed from an s-expression).
        target_grammar: One of the 7 grammar names from GRAMMAR_NAMES.

    Returns:
        A new Program AST using only primitives from the target grammar.

    Raises:
        InexpressibleError: If a primitive in the program has no rewriting
            rule for the target grammar (e.g., adjacent_pairs used standalone
            in swap-positional).
        ValueError: If target_grammar is not a recognized grammar name.
    """
    if target_grammar not in GRAMMAR_NAMES:
        raise ValueError(
            f"Unknown grammar '{target_grammar}'. "
            f"Choose from: {GRAMMAR_NAMES}"
        )

    # Identity grammars: base, add-both, redundant
    # These are supersets of (or equal to) the base grammar,
    # so no rewriting is needed.
    if target_grammar in ("base", "add-both", "redundant"):
        return program

    # Apply the appropriate rewriter(s)
    if target_grammar == "swap-positional":
        return _PositionalRewriter().transform(program)

    elif target_grammar == "swap-distributional":
        return _DistributionalRewriter().transform(program)

    elif target_grammar == "swap-both":
        # Apply both rewrites: positional first, then distributional
        result = _PositionalRewriter().transform(program)
        result = _DistributionalRewriter().transform(result)
        return result

    elif target_grammar == "minimal":
        return _MinimalRewriter().transform(program)

    # Should be unreachable
    raise ValueError(f"Unhandled grammar: {target_grammar}")  # pragma: no cover
