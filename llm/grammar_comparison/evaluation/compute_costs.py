"""
Compute log-probabilities and posteriors of hypothesis programs under grammars.

PURPOSE:
    This module connects the s-expression parser (which converts hypothesis
    strings into Program ASTs) with the grammar factory (which builds Grammar
    objects with weighted productions) to produce log-probability scores.

    The full posterior for each hypothesis is:
        log_posterior = log_prior + log_likelihood

    where:
        - log_prior: grammar PCFG score (how naturally the grammar generates
          the program)
        - log_likelihood: size principle score, -n * log(|ext(h)|), rewarding
          specific hypotheses consistent with observed exemplar hands

HOW IT WORKS:
    1. Parse the hypothesis s-expression into a Program AST using
       parse_hypothesis_sexpr() from the translation layer.
    2. Compute a fingerprint on 200 probe hands (for extension caching and
       the correct_rank metric).
    3. Rewrite the AST for the target grammar using rewrite_ast().
    4. Score the rewritten Program under the Grammar using
       grammar.program_log_likelihood() to get log_prior.
    5. Compute the extension size via estimate_extension() using the
       hypothesis's exemplar_hands to get log_likelihood.
    6. log_posterior = log_prior + log_likelihood.

USAGE:
    from llm.grammar_comparison.evaluation.compute_costs import (
        score_hypothesis,
        score_all_hypotheses,
    )
    from llm.grammar_comparison.grammars.grammar_factory import (
        build_grammar, CostStructure,
    )

    grammar = build_grammar("base", CostStructure.UNIFORM)
    ll = score_hypothesis("(\u03bb all (\u03bb eq (get_suit $0) CLUBS) $0)", grammar)
    # ll is a negative float (log-probability / prior only)

    results = score_all_hypotheses("base", CostStructure.UNIFORM, limit=10)
    # list of dicts with rule_id, rank, log_prior, log_likelihood, log_posterior, etc.
"""

import logging
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
from llm.grammar_comparison.translation.rewriter import rewrite_ast, InexpressibleError
from llm.grammar_comparison.translation.verification import (
    compute_ast_fingerprint,
    load_probe_hands,
)
from llm.grammar_comparison.evaluation.extension import estimate_extension
from llm.grammar_comparison.grammars.grammar_factory import (
    build_grammar,
    CostStructure,
)
from llm.grammar_comparison.data_loader import load_phase1b_hypotheses

logger = logging.getLogger(__name__)

REQUEST_TYPE = arrow(HAND, BOOL)


def parse_hypothesis(hypothesis: Dict) -> Optional[Program]:
    """Parse a hypothesis dict into a Program AST, trying available representations."""
    dsl_code = hypothesis.get("dsl_code")
    python_code = hypothesis.get("python_code")

    if dsl_code is not None:
        try:
            return parse_hypothesis_sexpr(dsl_code)
        except (ValueError, Exception) as e:
            logger.debug("S-expr parse error for %r: %s", dsl_code, e)
            return None

    if python_code is not None:
        try:
            return python_to_ast(python_code)
        except (ValueError, NotImplementedError, Exception) as e:
            logger.debug("Python parse error for %r: %s", python_code, e)
            return None

    return None


def score_hypothesis(sexpr: str, grammar: Grammar) -> float:
    """Score a single hypothesis s-expression under a grammar (prior only).

    Backward-compatible convenience function returning only the log_prior.
    """
    try:
        program = parse_hypothesis_sexpr(sexpr)
    except (ValueError, Exception) as e:
        logger.debug("Parse error for %r: %s", sexpr, e)
        return float('-inf')

    try:
        ll = grammar.program_log_likelihood(program, REQUEST_TYPE)
    except Exception as e:
        logger.debug("Scoring error for %r: %s", sexpr, e)
        return float('-inf')

    return ll


def score_program(program: Program, grammar: Grammar) -> float:
    """Score a pre-parsed Program AST under a grammar."""
    try:
        return grammar.program_log_likelihood(program, REQUEST_TYPE)
    except Exception as e:
        logger.debug("Scoring error for program %s: %s", program, e)
        return float('-inf')


def _fingerprint_to_string(fp: Tuple[Optional[bool], ...]) -> str:
    """Convert a fingerprint tuple to a compact string: True->'1', False->'0', None->'X'."""
    return ''.join(
        '1' if v is True else ('0' if v is False else 'X')
        for v in fp
    )


def _make_predicate(program: Program):
    """Turn a Program AST into a callable predicate, or None if evaluation fails."""
    try:
        func = program.evaluate([])
    except Exception:
        return None

    def predicate(hand):
        result = func(hand)
        return bool(result)

    return predicate


def precompute_hypothesis_data(
    limit: int = 0,
) -> List[Dict]:
    """Parse all hypotheses, compute fingerprints and extensions ONCE.

    Returns a list of dicts with parsed programs, fingerprints, and
    extension results that can be reused across multiple grammar evaluations.
    This avoids recomputing 1M-sample Monte Carlo for each grammar.
    """
    hypotheses = load_phase1b_hypotheses()
    if limit > 0:
        hypotheses = hypotheses[:limit]

    try:
        probes = load_probe_hands()
    except FileNotFoundError:
        logger.warning("Probe hands file not found; fingerprints will be empty.")
        probes = []

    precomputed = []
    for hyp in hypotheses:
        program = parse_hypothesis(hyp)

        if program is None:
            precomputed.append({
                **hyp,
                "_program": None,
                "_fingerprint": "",
                "_log_likelihood": float('-inf'),
                "_base_rate": 0.0,
                "_exemplars_consistent": False,
            })
            continue

        # Fingerprint (grammar-independent)
        if probes:
            fp_tuple = compute_ast_fingerprint(program, probes)
            fingerprint = _fingerprint_to_string(fp_tuple)
        else:
            fingerprint = ""

        # Extension and likelihood (grammar-independent)
        exemplar_hands = hyp.get("exemplar_hands", [])
        predicate = _make_predicate(program)

        if predicate is not None:
            ext_result = estimate_extension(predicate, exemplar_hands)
            log_likelihood = ext_result.log_likelihood
            base_rate = ext_result.base_rate
            exemplars_consistent = ext_result.exemplars_consistent
        else:
            log_likelihood = float('-inf')
            base_rate = 0.0
            exemplars_consistent = False

        precomputed.append({
            **hyp,
            "_program": program,
            "_fingerprint": fingerprint,
            "_log_likelihood": log_likelihood,
            "_base_rate": base_rate,
            "_exemplars_consistent": exemplars_consistent,
        })

    return precomputed


def score_all_hypotheses(
    grammar_name: str,
    cost_structure: CostStructure,
    limit: int = 0,
    precomputed: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Score all Phase 1b hypotheses under a grammar with posterior = prior + likelihood.

    If precomputed is provided, skips parsing/fingerprinting/extension estimation
    (which are grammar-independent) and only computes the grammar-specific prior.
    This makes subsequent grammar evaluations near-instant.

    Returns a list of dicts with: rule_id, rank, confidence, nl_description,
    log_prior, log_likelihood, log_posterior, base_rate, fingerprint,
    exemplars_consistent, grammar_name, cost_structure.
    """
    if precomputed is None:
        precomputed = precompute_hypothesis_data(limit)

    grammar = build_grammar(grammar_name, cost_structure)

    results = []
    for hyp in precomputed:
        program = hyp.get("_program")

        if program is None:
            results.append({
                "rule_id": hyp["rule_id"],
                "rank": hyp["rank"],
                "confidence": hyp["confidence"],
                "nl_description": hyp["nl_description"],
                "log_prior": float('-inf'),
                "log_likelihood": float('-inf'),
                "log_posterior": float('-inf'),
                "base_rate": 0.0,
                "fingerprint": "",
                "exemplars_consistent": False,
                "grammar_name": grammar_name,
                "cost_structure": cost_structure.value,
            })
            continue

        fingerprint = hyp["_fingerprint"]
        log_likelihood = hyp["_log_likelihood"]
        base_rate = hyp["_base_rate"]
        exemplars_consistent = hyp["_exemplars_consistent"]

        try:
            rewritten = rewrite_ast(program, grammar_name)
            log_prior = score_program(rewritten, grammar)
        except InexpressibleError:
            log_prior = float('-inf')

        if math.isinf(log_prior) or math.isinf(log_likelihood):
            log_posterior = float('-inf')
        else:
            log_posterior = log_prior + log_likelihood

        results.append({
            "rule_id": hyp["rule_id"],
            "rank": hyp["rank"],
            "confidence": hyp["confidence"],
            "nl_description": hyp["nl_description"],
            "log_prior": log_prior,
            "log_likelihood": log_likelihood,
            "log_posterior": log_posterior,
            "base_rate": base_rate,
            "fingerprint": fingerprint,
            "exemplars_consistent": exemplars_consistent,
            "grammar_name": grammar_name,
            "cost_structure": cost_structure.value,
        })

    return results
