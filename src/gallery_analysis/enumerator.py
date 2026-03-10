"""
Gallery-specific enumerator wrapper.

Wraps dreamcoder_core's TopDownEnumerator to enumerate hand -> bool programs,
convert them to callable predicates, and optionally filter by exemplar consistency.

The main entry point is enumerate_hypotheses() which yields programs in
order of increasing cost (decreasing prior probability).
"""
import sys
import time
import math
from pathlib import Path
from typing import List, Tuple, Callable, Optional, Generator

sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Hand, Card, Suit, Rank, RANK_VALUES, card_color, Color

from dreamcoder_core.type_system import BOOL, HAND, Arrow
from dreamcoder_core.primitives import build_primitives
from dreamcoder_core.grammar import uniform_grammar
from dreamcoder_core.enumeration import TopDownEnumerator
from dreamcoder_core.program import Program


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


def enumerate_hypotheses(
    max_depth: int = 6,
    max_programs: int = 10000,
    max_cost: float = 50.0,
    timeout: float = 300.0,
    grammar=None,
) -> List[Tuple[str, Callable[[Hand], bool], float]]:
    """
    Enumerate hand -> bool programs from the DSL.

    Returns list of (program_string, predicate_function, log_prior) tuples.
    Programs are yielded in order of increasing cost (decreasing prior).

    Args:
        max_depth: Maximum AST depth for enumeration
        max_programs: Maximum number of complete programs to yield
        max_cost: Maximum cost (-log probability) to explore
        timeout: Wall clock timeout in seconds
        grammar: Optional grammar to use (defaults to uniform over all primitives)
    """
    if grammar is None:
        primitives = build_primitives()
        grammar = uniform_grammar(primitives)

    request_type = Arrow(HAND, BOOL)

    enumerator = TopDownEnumerator(
        grammar=grammar,
        max_depth=max_depth,
        max_programs=max_programs,
    )

    results = []
    start = time.time()

    for program, log_prob in enumerator.enumerate(
        request_type=request_type,
        max_cost=max_cost,
        timeout_seconds=timeout,
    ):
        prog_str = str(program)
        pred_fn = _make_evaluator(program)
        results.append((prog_str, pred_fn, log_prob))

        if time.time() - start > timeout:
            break

    return results
