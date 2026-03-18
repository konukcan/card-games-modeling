"""
Compute log-probabilities of hypothesis programs under different grammars.

PURPOSE:
    This module connects the s-expression parser (which converts hypothesis
    strings into Program ASTs) with the grammar factory (which builds Grammar
    objects with weighted productions) to produce log-probability scores.

    These scores are the core metric for the grammar-comparison experiment:
    a hypothesis that is more "natural" under a grammar will have a higher
    (less negative) log-probability.

HOW IT WORKS:
    1. Parse the hypothesis s-expression into a Program AST using
       parse_hypothesis_sexpr() from the translation layer.
    2. Score the Program under a Grammar using grammar.program_log_likelihood(),
       which computes log P(program | grammar, request_type).
    3. The request type is arrow(HAND, BOOL) because every hypothesis is a
       function from a hand (list of cards) to a boolean.

USAGE:
    from llm.grammar_comparison.evaluation.compute_costs import (
        score_hypothesis,
        score_all_hypotheses,
    )
    from llm.grammar_comparison.grammars.grammar_factory import (
        build_grammar, CostStructure,
    )

    grammar = build_grammar("base", CostStructure.UNIFORM)
    ll = score_hypothesis("(λ all (λ eq (get_suit $0) CLUBS) $0)", grammar)
    # ll is a negative float (log-probability)

    results = score_all_hypotheses("base", CostStructure.UNIFORM, limit=10)
    # list of dicts with rule_id, rank, log_prob, etc.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup: allow importing from the main src/ tree
# ---------------------------------------------------------------------------
_SRC_DIR = str(Path(__file__).parent.parent.parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dreamcoder_core.grammar import Grammar
from dreamcoder_core.program import Program
from dreamcoder_core.type_system import HAND, BOOL, arrow

from llm.grammar_comparison.translation.sexpr_parser import parse_hypothesis_sexpr
from llm.grammar_comparison.translation.python_parser import python_to_ast
from llm.grammar_comparison.grammars.grammar_factory import (
    build_grammar,
    CostStructure,
)
from llm.grammar_comparison.data_loader import load_phase1b_hypotheses

logger = logging.getLogger(__name__)

# The request type for all hypotheses: a function from a hand to a boolean.
# Every hypothesis is λ hand. body, where hand : list(card) and body : bool.
REQUEST_TYPE = arrow(HAND, BOOL)



def parse_hypothesis(hypothesis: Dict) -> Optional[Program]:
    """Parse a hypothesis dict into a Program AST, trying available representations.

    Attempts parsing in priority order:
        1. dsl_code (s-expression) via parse_hypothesis_sexpr()
        2. python_code (Python lambda) via python_to_ast()

    This fallback ensures that hypotheses without s-expression translations
    (19 of 271 in Phase 1b) can still be scored, as long as they have a
    Python representation.

    Args:
        hypothesis: A dict with optional 'dsl_code' and 'python_code' keys.

    Returns:
        A Program AST, or None if neither representation can be parsed.
    """
    dsl_code = hypothesis.get("dsl_code")
    python_code = hypothesis.get("python_code")

    # Priority 1: s-expression
    if dsl_code is not None:
        try:
            return parse_hypothesis_sexpr(dsl_code)
        except (ValueError, Exception) as e:
            logger.debug("S-expr parse error for %r: %s", dsl_code, e)
            return None

    # Priority 2: Python lambda
    if python_code is not None:
        try:
            return python_to_ast(python_code)
        except (ValueError, NotImplementedError, Exception) as e:
            logger.debug("Python parse error for %r: %s", python_code, e)
            return None

    # Neither representation available
    return None


def score_hypothesis(sexpr: str, grammar: Grammar) -> float:
    """Score a single hypothesis s-expression under a grammar.

    Parses the s-expression into a Program AST, then computes
    log P(program | grammar, request_type) where request_type = HAND -> BOOL.

    Args:
        sexpr: An s-expression string, e.g.
               "(λ all (λ eq (get_suit $0) CLUBS) $0)"
        grammar: A Grammar object (from grammar_factory.build_grammar).

    Returns:
        The log-probability (a negative float, or -inf if the program
        cannot be parsed or is inexpressible under this grammar).

    Why this function returns -inf for failures:
        - If the s-expression has a syntax error, parse_hypothesis_sexpr
          raises ValueError.
        - If the program uses a primitive not in the grammar, the grammar's
          program_log_likelihood returns -inf.
        - Both cases mean "this grammar cannot generate this program."
    """
    try:
        program = parse_hypothesis_sexpr(sexpr)
    except (ValueError, Exception) as e:
        # Parsing failed — this hypothesis is inexpressible as a valid AST
        logger.debug("Parse error for %r: %s", sexpr, e)
        return float('-inf')

    try:
        ll = grammar.program_log_likelihood(program, REQUEST_TYPE)
    except Exception as e:
        # Scoring failed — e.g. type mismatch during unification
        logger.debug("Scoring error for %r: %s", sexpr, e)
        return float('-inf')

    return ll


def score_program(program: Program, grammar: Grammar) -> float:
    """Score a pre-parsed Program AST under a grammar.

    Like score_hypothesis() but takes an already-parsed Program instead
    of an s-expression string. Useful when the Program was obtained via
    parse_hypothesis() (which may have used the Python fallback path).

    Args:
        program: A Program AST node.
        grammar: A Grammar object.

    Returns:
        The log-probability, or -inf on error.
    """
    try:
        return grammar.program_log_likelihood(program, REQUEST_TYPE)
    except Exception as e:
        logger.debug("Scoring error for program %s: %s", program, e)
        return float('-inf')


def score_all_hypotheses(
    grammar_name: str,
    cost_structure: CostStructure,
    limit: int = 0,
) -> List[Dict]:
    """Score all Phase 1b hypotheses under a specific grammar + cost structure.

    This is the batch entry point for the grammar-comparison pipeline.
    It loads hypotheses, builds the grammar, and scores each one.

    Args:
        grammar_name: One of the 7 grammar names (e.g. "base",
                      "swap-positional"). See grammar_factory.GRAMMAR_NAMES.
        cost_structure: The cost structure to use (UNIFORM, TIERED, or LOTLIB3).
        limit: If > 0, only process the first `limit` hypotheses.
               Useful for quick testing.

    Returns:
        A list of dicts, one per hypothesis, each containing:
          - rule_id        (str): The card-game rule this hypothesis targets.
          - rank           (int): Confidence rank from the LLM (1 = most confident).
          - confidence     (str): HIGH / MEDIUM / LOW label.
          - nl_description (str): Natural-language description.
          - log_prob     (float): Log-probability under this grammar (-inf if
                                  inexpressible or missing DSL code).
          - grammar_name   (str): Which grammar was used.
          - cost_structure  (str): Which cost structure was used.

    How it works:
        1. load_phase1b_hypotheses() returns a flat list of hypothesis dicts
           that already have 'dsl_code' (s-expression) fields from cross-
           referencing with injected_hypotheses.json.
        2. build_grammar() constructs a Grammar with the chosen primitives
           and log-probability assignments.
        3. For each hypothesis, parse_hypothesis() tries the s-expression
           first and falls back to python_to_ast() if dsl_code is missing.
           Hypotheses with neither representation get -inf.
    """
    # Load hypotheses from Phase 1b data files
    hypotheses = load_phase1b_hypotheses()

    # Apply limit if requested
    if limit > 0:
        hypotheses = hypotheses[:limit]

    # Build the grammar for scoring
    grammar = build_grammar(grammar_name, cost_structure)

    results = []
    for hyp in hypotheses:
        # Try to parse the hypothesis using the fallback chain:
        # dsl_code (s-expression) first, then python_code if needed.
        program = parse_hypothesis(hyp)

        if program is not None:
            log_prob = score_program(program, grammar)
        else:
            # Neither representation could be parsed — cannot score
            log_prob = float('-inf')

        results.append({
            "rule_id": hyp["rule_id"],
            "rank": hyp["rank"],
            "confidence": hyp["confidence"],
            "nl_description": hyp["nl_description"],
            "log_prob": log_prob,
            "grammar_name": grammar_name,
            "cost_structure": cost_structure.value,
        })

    return results
