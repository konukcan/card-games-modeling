"""
Gallery-specific enumerator wrapper.

Wraps dreamcoder_core's TopDownEnumerator to enumerate hand -> bool programs,
convert them to callable predicates, and optionally filter by exemplar consistency.

The main entry point is enumerate_hypotheses() which yields programs in
order of increasing cost (decreasing prior probability).

SYNTACTIC PRUNING
-----------------
Before the downstream trivial filter (which evaluates each program on 360
exemplar hands), we apply fast syntactic checks to reject programs that are
guaranteed to be trivial or redundant. This matters because the trivial
filter is the pipeline bottleneck (~0.7ms per program × 360 evaluations).

The following patterns are rejected:

  CATEGORY 1 — Removed from grammar (never enumerated):
    - Boolean constants `true`/`false`: In a hand→bool program, every use of
      true/false is either trivially constant or redundant with a simpler
      expression. See analysis in explore_efficiency.py.

  CATEGORY 2 — Rejected by syntactic string checks (is_syntactically_redundant):
    - reverse(reverse(X)): identity, always equals X
    - unique(unique(X)): idempotent, always equals unique(X)
    - not(not(X)): identity, always equals X
    - and(X, X) / or(X, X): idempotent, always equals X
    - take(0, X): always empty list → downstream operations degenerate
    - Pure constant arithmetic in comparisons: e.g., (lt (+ 2 3) 4),
      evaluates to a constant with no hand dependence (100% trivial rate)
    - Constant-vs-constant comparisons: e.g., (lt 3 2), (eq 1 1)
      — no hand dependence at all (100% trivial rate)

Empirical impact (depth 5, 100K programs):
  - Removing true/false: eliminates ~33% of enumerated programs
  - Syntactic pruning: eliminates additional ~10-15%
  - Combined: ~45% fewer programs entering the trivial filter
"""
import sys
import re
import time
import math
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Generator, Dict

sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Hand, Card, Suit, Rank, RANK_VALUES, card_color, Color

from dreamcoder_core.type_system import BOOL, HAND, Arrow
from dreamcoder_core.primitives import build_primitives
from dreamcoder_core.grammar import uniform_grammar
from dreamcoder_core.enumeration import TopDownEnumerator
from dreamcoder_core.program import Program


# =========================================================================
# Gallery-specific grammar (no boolean constants)
# =========================================================================

def build_gallery_primitives() -> list:
    """
    Build the primitive library for gallery analysis, excluding true/false.

    In a hand→bool program, boolean constants are never needed:
    - (and X false) → false, (or X true) → true — trivially constant
    - (and X true) → X, (or X false) → X — redundant identity wrappers
    - (if true X Y) → X, (if false X Y) → Y — redundant
    - (if cond true false) → cond — redundant
    - (λ true), (λ false) — trivially constant programs

    Every non-trivial use of true/false is expressible more simply without it.
    Removing them eliminates ~37% of trivial programs at the source.
    """
    all_prims = build_primitives()
    return [p for p in all_prims if p.name not in ('true', 'false')]


def build_gallery_grammar():
    """Build a grammar for gallery analysis (no boolean constants)."""
    prims = build_gallery_primitives()
    return uniform_grammar(prims)


# =========================================================================
# List→list composition chain detection (Option B)
# =========================================================================
#
# The grammar allows list→list transforms (reverse, first_half, second_half,
# sort_by_rank, unique, take, drop, filter) to compose freely. At depth 5-6,
# this creates hundreds of "deeply wrapped shallow predicates" like:
#
#   (has_color (first_half (reverse (first_half (sort_by_rank (reverse $0))))) RED)
#
# These carry <0.01% posterior mass but consume significant enumeration time.
# We limit consecutive list→list compositions to MAX_LIST_CHAIN (default 2),
# which allows useful patterns like `first_half (sort_by_rank $0)` but blocks
# 3+ layers of wrapping.
#
# This filter runs on the program string BEFORE _make_evaluator() and the
# trivial filter, so the expensive downstream steps are avoided entirely.
# It does not prune at enumeration time (which would require fragmenting the
# memoization cache by chain depth), but achieves >95% of the efficiency gain
# since the per-program enumeration cost is ~microseconds while the downstream
# evaluation cost is ~2ms per program.

# Unary list→list primitives: each takes a list and returns a list of the
# same element type, without any non-list arguments.
_UNARY_LIST_TRANSFORMS = frozenset({
    'reverse', 'first_half', 'second_half', 'unique', 'sort_by_rank',
})

# Binary list→list primitives: take extra args + a list, return a list.
# In program strings these appear as e.g. "take 3 (reverse $0)" or
# "filter (λ ...) (first_half $0)". The list arg is the last positional arg.
_BINARY_LIST_TRANSFORMS = frozenset({
    'take', 'drop', 'filter',
})

_ALL_LIST_TRANSFORMS = _UNARY_LIST_TRANSFORMS | _BINARY_LIST_TRANSFORMS

# Default maximum consecutive list→list compositions allowed.
# K=2 allows "first_half (sort_by_rank $0)" but blocks 3+ layers.
# None of the 60 true gallery rules need more than 2 consecutive transforms.
MAX_LIST_CHAIN = 2

# Pre-compiled regex: matches "(transform_name " at the start of a nested
# list→list application. Used by _max_list_chain_depth().
_RE_LIST_TRANSFORM = re.compile(
    r'\((?:' + '|'.join(re.escape(t) for t in sorted(_ALL_LIST_TRANSFORMS)) + r') '
)


def _max_list_chain_depth(prog_str: str) -> int:
    """
    Find the maximum consecutive list→list composition chain in a program.

    Scans the program string for nested patterns like:
        (first_half (reverse (sort_by_rank $0)))
    and returns the chain length (3 in this example).

    This works by finding each list transform and walking inward to count
    how many consecutive transforms follow.

    For efficiency, we only look at unary list transforms for chain counting
    since they are the main source of the composition explosion. Binary
    transforms like (take 3 (reverse $0)) count as 1 link in the chain
    when their list argument is another transform.
    """
    max_chain = 0

    # Find all positions where a list transform starts
    for match in _RE_LIST_TRANSFORM.finditer(prog_str):
        chain = 1
        # Walk forward from after this match to see if the argument is
        # another list transform. We need to find the list argument,
        # which for unary transforms is immediately after the name,
        # and for binary transforms is after their first argument(s).
        pos = match.end()

        # For unary transforms, the next token should be the list arg.
        # For binary transforms (take, drop, filter), we need to skip
        # past the non-list argument(s) to find the list arg.
        transform_name = prog_str[match.start()+1:match.end()-1]

        if transform_name in _BINARY_LIST_TRANSFORMS:
            # Skip past the first argument(s) to find the list argument.
            # For "take N (list_expr)" / "drop N (list_expr)": skip the int
            # For "filter (λ ...) (list_expr)": skip the lambda
            # We find the list arg by counting balanced parens.
            depth = 0
            # Skip first argument
            while pos < len(prog_str):
                if prog_str[pos] == '(':
                    depth += 1
                elif prog_str[pos] == ')':
                    depth -= 1
                    if depth == 0:
                        pos += 1
                        break
                elif depth == 0 and prog_str[pos] == ' ':
                    # Simple (non-parenthesized) first arg like a digit
                    pos += 1
                    break
                pos += 1
            # Now pos should be at the start of the list argument
            # (possibly after a space)
            while pos < len(prog_str) and prog_str[pos] == ' ':
                pos += 1

        # Now check if the argument at `pos` is another list transform
        while pos < len(prog_str):
            # Check if we're looking at "(transform_name "
            found_next = False
            if prog_str[pos] == '(':
                for t in _ALL_LIST_TRANSFORMS:
                    prefix = '(' + t + ' '
                    if prog_str[pos:pos+len(prefix)] == prefix:
                        chain += 1
                        pos += len(prefix)
                        # If this is a binary transform, skip its first arg
                        if t in _BINARY_LIST_TRANSFORMS:
                            depth = 0
                            while pos < len(prog_str):
                                if prog_str[pos] == '(':
                                    depth += 1
                                elif prog_str[pos] == ')':
                                    depth -= 1
                                    if depth == 0:
                                        pos += 1
                                        break
                                elif depth == 0 and prog_str[pos] == ' ':
                                    pos += 1
                                    break
                                pos += 1
                            while pos < len(prog_str) and prog_str[pos] == ' ':
                                pos += 1
                        found_next = True
                        break
            if not found_next:
                break

        max_chain = max(max_chain, chain)

    return max_chain


def exceeds_list_chain_limit(prog_str: str, max_chain: int = MAX_LIST_CHAIN) -> bool:
    """
    Check if a program exceeds the maximum allowed list→list composition chain.

    Returns True if the program should be rejected.

    Args:
        prog_str: The program string to check.
        max_chain: Maximum allowed chain length (default MAX_LIST_CHAIN=2).
    """
    return _max_list_chain_depth(prog_str) > max_chain


# =========================================================================
# Syntactic redundancy checks
# =========================================================================

# Pre-compiled regexes for performance (these run on every enumerated program)
#
# NOTE on program string format:
#   The enumerator uses curried application with minimal parenthesization.
#   - Lambdas: (λ body)
#   - Application: (f arg1 arg2) — NO separate parens around each application
#   - Nested: (λ lt (+ 0 0) 0) not (λ (lt (+ 0 0) 0))
#   So "lt 0 0" appears without a leading paren on "lt".

# Matches: "lt 3 2)", "eq 1 1)", etc. — comparison of two bare digit constants
# The trailing ) ensures we're at the end of an expression, not mid-expression
_RE_CONST_VS_CONST = re.compile(
    r'(?:lt|le|gt|ge|eq) \d+ \d+\)'
)
# Matches: "lt (+ 0 0) 0" or "lt 0 (+ 0 0)" — comparison with constant arithmetic
_RE_CONST_ARITH_CMP = re.compile(
    r'(?:lt|le|gt|ge|eq) \((?:\+|-|mod) \d+ \d+\) \d+'
    r'|(?:lt|le|gt|ge|eq) \d+ \((?:\+|-|mod) \d+ \d+\)'
)


def is_syntactically_redundant(prog_str: str, max_list_chain: int = MAX_LIST_CHAIN) -> bool:
    """
    Fast syntactic check for programs guaranteed to be trivial or redundant.

    These checks operate on the program string representation and catch
    patterns that are provably constant or equivalent to shorter programs.
    The goal is to avoid the expensive exemplar-based trivial filter
    (~0.7ms per program) for programs we can reject by inspection.

    Also enforces the list→list composition chain limit (Option B).

    Args:
        prog_str: The program string to check.
        max_list_chain: Maximum allowed consecutive list→list transforms.
            Set to None to disable chain checking.

    Returns True if the program should be discarded.
    """
    # Option B: Reject programs with excessive list→list composition chains
    if max_list_chain is not None and exceeds_list_chain_limit(prog_str, max_list_chain):
        return True
    # Identity compositions: f(f⁻¹(X)) = X or f(f(X)) = f(X)
    if 'reverse (reverse' in prog_str:
        return True
    if 'unique (unique' in prog_str:
        return True

    # Double negation: not(not(X)) = X
    # Format is "not (not " — the outer not is applied via curried application
    if 'not (not ' in prog_str:
        return True

    # take 0 always produces empty list — downstream ops degenerate
    if 'take 0 ' in prog_str:
        return True

    # Pure constant arithmetic in comparisons: "lt (+ 2 3) 4" — no hand dependence
    if _RE_CONST_ARITH_CMP.search(prog_str):
        return True

    # Constant-vs-constant comparisons: "lt 3 2", "eq 1 1" — no hand dependence
    # We match when BOTH args to a comparison are bare digits.
    # Safe because (eq (n_unique_suits $0) 2) has a nested expr, not a bare digit.
    if _RE_CONST_VS_CONST.search(prog_str):
        return True

    # Idempotent boolean: and(X, X) = X, or(X, X) = X
    # This regex matches when both arguments to and/or are identical
    m = re.search(r'\(and (.+?) \1\)', prog_str)
    if m:
        return True
    m = re.search(r'\(or (.+?) \1\)', prog_str)
    if m:
        return True

    return False


# =========================================================================
# Evaluator
# =========================================================================

def _make_evaluator(program: Program) -> Callable[[Hand], bool]:
    """
    Convert an enumerated Program AST into a callable predicate.

    The program has type hand -> bool. We evaluate it by calling
    program.evaluate([]) which returns a closure, then apply that
    closure to a hand.
    """
    def predicate(hand: Hand) -> bool:
        try:
            result = program.evaluate([])(hand)
            return bool(result)
        except Exception:
            return False
    return predicate


# =========================================================================
# Main enumeration entry point
# =========================================================================

def enumerate_hypotheses(
    max_depth: int = 6,
    max_programs: int = 10000,
    max_cost: float = 50.0,
    timeout: float = 300.0,
    grammar=None,
    syntactic_filter: bool = True,
    max_list_chain: int = MAX_LIST_CHAIN,
) -> List[Tuple[str, Callable[[Hand], bool], float]]:
    """
    Enumerate hand -> bool programs from the DSL.

    Returns list of (program_string, predicate_function, log_prior) tuples.
    Programs are yielded in order of increasing cost (decreasing prior).

    By default, uses the gallery-specific grammar (no boolean constants)
    and applies syntactic pruning to reject provably trivial programs.

    Args:
        max_depth: Maximum AST depth for enumeration
        max_programs: Maximum number of complete programs to yield
        max_cost: Maximum cost (-log probability) to explore
        timeout: Wall clock timeout in seconds
        grammar: Optional grammar override. If None, uses gallery grammar
                 (no true/false). Pass a custom grammar to override.
        syntactic_filter: If True (default), reject syntactically redundant
                         programs before adding them to results.
        max_list_chain: Maximum consecutive list→list transforms allowed.
                       Set to None to disable. Default MAX_LIST_CHAIN (2).
    """
    if grammar is None:
        grammar = build_gallery_grammar()

    request_type = Arrow(HAND, BOOL)

    enumerator = TopDownEnumerator(
        grammar=grammar,
        max_depth=max_depth,
        max_programs=max_programs,
    )

    results = []
    n_syntactic_rejected = 0
    start = time.time()

    for program, log_prob in enumerator.enumerate(
        request_type=request_type,
        max_cost=max_cost,
        timeout_seconds=timeout,
    ):
        prog_str = str(program)

        # Fast syntactic check — skip before creating the evaluator
        if syntactic_filter and is_syntactically_redundant(prog_str, max_list_chain):
            n_syntactic_rejected += 1
            continue

        pred_fn = _make_evaluator(program)
        results.append((prog_str, pred_fn, log_prob))

        if time.time() - start > timeout:
            break

    return results


def enumerate_hypotheses_with_stats(
    max_depth: int = 6,
    max_programs: int = 10000,
    max_cost: float = 50.0,
    timeout: float = 300.0,
    grammar=None,
    syntactic_filter: bool = True,
    max_list_chain: int = MAX_LIST_CHAIN,
) -> Tuple[List[Tuple[str, Callable[[Hand], bool], float]], Dict[str, int]]:
    """
    Like enumerate_hypotheses but also returns enumeration statistics.

    Returns:
        (programs, stats) where stats includes:
        - total_yielded: programs yielded by the enumerator (before syntactic filter)
        - syntactic_rejected: programs rejected by syntactic filter
        - list_chain_rejected: programs rejected by list→list chain limit
        - accepted: programs in the final results list
    """
    if grammar is None:
        grammar = build_gallery_grammar()

    request_type = Arrow(HAND, BOOL)

    enumerator = TopDownEnumerator(
        grammar=grammar,
        max_depth=max_depth,
        max_programs=max_programs,
    )

    results = []
    n_total = 0
    n_rejected = 0
    n_chain_rejected = 0
    start = time.time()

    for program, log_prob in enumerator.enumerate(
        request_type=request_type,
        max_cost=max_cost,
        timeout_seconds=timeout,
    ):
        n_total += 1
        prog_str = str(program)

        # Check list chain limit separately for stats
        if syntactic_filter and max_list_chain is not None and exceeds_list_chain_limit(prog_str, max_list_chain):
            n_chain_rejected += 1
            n_rejected += 1
            continue

        if syntactic_filter and is_syntactically_redundant(prog_str, max_list_chain=None):
            n_rejected += 1
            continue

        pred_fn = _make_evaluator(program)
        results.append((prog_str, pred_fn, log_prob))

        if time.time() - start > timeout:
            break

    stats = {
        "total_yielded": n_total,
        "syntactic_rejected": n_rejected,
        "list_chain_rejected": n_chain_rejected,
        "accepted": len(results),
    }
    return results, stats
