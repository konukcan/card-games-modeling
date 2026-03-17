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
from typing import Dict, List

# ---------------------------------------------------------------------------
# Path setup: allow importing from the main src/ tree
# ---------------------------------------------------------------------------
_SRC_DIR = str(Path(__file__).parent.parent.parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dreamcoder_core.grammar import Grammar
from dreamcoder_core.type_system import HAND, BOOL, arrow

from llm.grammar_comparison.translation.sexpr_parser import parse_hypothesis_sexpr
from llm.grammar_comparison.grammars.grammar_factory import (
    build_grammar,
    CostStructure,
)
from llm.grammar_comparison.data_loader import load_phase1b_hypotheses

logger = logging.getLogger(__name__)

# The request type for all hypotheses: a function from a hand to a boolean.
# Every hypothesis is λ hand. body, where hand : list(card) and body : bool.
REQUEST_TYPE = arrow(HAND, BOOL)


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
        3. For each hypothesis with a dsl_code, score_hypothesis() computes
           the log-probability. Hypotheses without dsl_code get -inf.
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
        dsl_code = hyp.get("dsl_code")

        if dsl_code:
            log_prob = score_hypothesis(dsl_code, grammar)
        else:
            # No DSL translation available — cannot score
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
