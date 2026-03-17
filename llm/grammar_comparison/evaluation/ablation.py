"""Ablation analysis: leave-one-out, leave-one-in, and cross-validation.

Stage 2 of the grammar comparison framework. These functions perform
fine-grained ablation analysis by testing what happens when individual
primitives are added to or removed from a grammar.

Since scoring a hypothesis under a grammar takes microseconds, we can
afford to test many configurations.

Functions
---------
leave_one_out
    Remove each primitive one at a time from a grammar and measure impact.
leave_one_in
    Add each candidate primitive one at a time to a grammar and measure impact.
cross_validate
    k-fold cross-validation over hypotheses for a given grammar.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# Allow imports from the main src/ tree
_SRC_DIR = str(Path(__file__).parent.parent.parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dreamcoder_core.grammar import Grammar, Production

from llm.grammar_comparison.grammars.grammar_factory import (
    build_grammar,
    CostStructure,
    GRAMMAR_NAMES,
)
from llm.grammar_comparison.evaluation.compute_costs import (
    score_hypothesis,
    score_all_hypotheses,
)
from llm.grammar_comparison.evaluation.metrics import (
    spearman_rank_correlation,
    weighted_log_probability,
    top1_accuracy,
    expressibility,
)
from llm.grammar_comparison.data_loader import load_phase1b_hypotheses


# ---------------------------------------------------------------------------
# Metric dispatch
# ---------------------------------------------------------------------------

# Maps metric name -> callable that takes a scored list and returns a float.
METRIC_FUNCTIONS: Dict[str, Callable[[List[Dict]], float]] = {
    "spearman": spearman_rank_correlation,
    "weighted_log_prob": weighted_log_probability,
    "top1": top1_accuracy,
    "expressibility": expressibility,
}


def _get_metric_fn(metric: str) -> Callable[[List[Dict]], float]:
    """Look up the metric function by name.

    Args:
        metric: One of "spearman", "weighted_log_prob", "top1", "expressibility".

    Returns:
        The corresponding metric function.

    Raises:
        ValueError: If the metric name is not recognized.
    """
    if metric not in METRIC_FUNCTIONS:
        raise ValueError(
            f"Unknown metric '{metric}'. Choose from: {list(METRIC_FUNCTIONS.keys())}"
        )
    return METRIC_FUNCTIONS[metric]


# ---------------------------------------------------------------------------
# Hypothesis loading and scoring helpers
# ---------------------------------------------------------------------------

def _load_hypotheses(limit: int = 0) -> List[Dict]:
    """Load Phase 1b hypotheses (cached within a single Python session).

    Args:
        limit: If > 0, only return the first `limit` hypotheses.

    Returns:
        List of hypothesis dicts with rule_id, rank, dsl_code, etc.
    """
    hypotheses = load_phase1b_hypotheses()
    if limit > 0:
        hypotheses = hypotheses[:limit]
    return hypotheses


def _score_hypotheses_with_grammar(
    hypotheses: List[Dict],
    grammar: Grammar,
    grammar_name: str = "",
    cost_structure_name: str = "",
) -> List[Dict]:
    """Score a list of hypotheses under a given grammar.

    This avoids re-loading hypothesis data from disk (unlike
    score_all_hypotheses which loads data each time).

    Args:
        hypotheses: List of hypothesis dicts (must have 'dsl_code' field).
        grammar: The Grammar object to score against.
        grammar_name: Label for the grammar (stored in results).
        cost_structure_name: Label for the cost structure (stored in results).

    Returns:
        List of scored dicts with rule_id, rank, log_prob, etc.
    """
    results = []
    for hyp in hypotheses:
        dsl_code = hyp.get("dsl_code")

        if dsl_code:
            log_prob = score_hypothesis(dsl_code, grammar)
        else:
            log_prob = float('-inf')

        results.append({
            "rule_id": hyp["rule_id"],
            "rank": hyp["rank"],
            "confidence": hyp.get("confidence", ""),
            "nl_description": hyp.get("nl_description", ""),
            "log_prob": log_prob,
            "grammar_name": grammar_name,
            "cost_structure": cost_structure_name,
        })

    return results


def _build_grammar_without(
    base_grammar: Grammar,
    removed_name: str,
) -> Grammar:
    """Build a new Grammar with one production removed.

    Creates a new Grammar from the base grammar's productions, excluding
    the production whose primitive name matches `removed_name`. The
    log_variable is preserved from the base grammar.

    Args:
        base_grammar: The original Grammar object.
        removed_name: The name of the primitive to remove (e.g. "map").

    Returns:
        A new Grammar object without the specified primitive.
    """
    filtered_productions = [
        p for p in base_grammar.productions
        if str(p.program) != removed_name
    ]
    return Grammar(filtered_productions, base_grammar.log_variable)


def _build_grammar_with_addition(
    base_grammar: Grammar,
    new_production: Production,
) -> Grammar:
    """Build a new Grammar with one additional production.

    Creates a new Grammar from the base grammar's productions plus the
    given new production. The log_variable is preserved from the base.

    Args:
        base_grammar: The original Grammar object.
        new_production: The Production to add.

    Returns:
        A new Grammar object with the additional primitive.
    """
    extended_productions = list(base_grammar.productions) + [new_production]
    return Grammar(extended_productions, base_grammar.log_variable)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def leave_one_out(
    base_grammar: str,
    cost_structure: CostStructure = CostStructure.UNIFORM,
    metric: str = "spearman",
    limit: int = 0,
) -> List[Dict]:
    """Remove each primitive one at a time and measure impact on a metric.

    For each primitive in the base grammar, this function:
    1. Builds a new grammar WITHOUT that primitive.
    2. Scores all hypotheses under the modified grammar.
    3. Computes the specified metric.
    4. Records the delta (change from baseline).

    Results are sorted by absolute delta (most impactful removal first).

    Args:
        base_grammar: Name of the grammar to ablate (e.g. "base", "minimal").
                      Must be one of GRAMMAR_NAMES.
        cost_structure: The cost structure for probability assignment.
        metric: The evaluation metric to compute. One of:
                "spearman", "weighted_log_prob", "top1", "expressibility".
        limit: If > 0, only use the first `limit` hypotheses (for speed).

    Returns:
        List of dicts sorted by absolute delta (descending), each containing:
          - removed (str): Name of the removed primitive.
          - metric_value (float): Metric value with this primitive removed.
          - baseline_value (float): Metric value with the full grammar.
          - delta (float): metric_value - baseline_value.
    """
    metric_fn = _get_metric_fn(metric)

    # Load hypotheses once
    hypotheses = _load_hypotheses(limit)

    # Build the full grammar and compute baseline metric
    grammar = build_grammar(base_grammar, cost_structure)
    baseline_scored = _score_hypotheses_with_grammar(
        hypotheses, grammar, base_grammar, cost_structure.value
    )
    baseline_value = metric_fn(baseline_scored)

    # For each primitive, remove it and compute the metric
    results = []
    for production in grammar.productions:
        prim_name = str(production.program)

        # Build grammar without this primitive
        modified_grammar = _build_grammar_without(grammar, prim_name)

        # Score hypotheses under modified grammar
        scored = _score_hypotheses_with_grammar(
            hypotheses, modified_grammar, base_grammar, cost_structure.value
        )

        # Compute metric
        metric_value = metric_fn(scored)
        delta = metric_value - baseline_value

        results.append({
            "removed": prim_name,
            "metric_value": metric_value,
            "baseline_value": baseline_value,
            "delta": delta,
        })

    # Sort by absolute delta descending (most impactful first)
    results.sort(key=lambda r: abs(r["delta"]), reverse=True)

    return results


def leave_one_in(
    base_grammar: str,
    candidates: Optional[List[str]] = None,
    cost_structure: CostStructure = CostStructure.UNIFORM,
    metric: str = "spearman",
    limit: int = 0,
) -> List[Dict]:
    """Add one candidate primitive at a time and measure impact on a metric.

    Starting from a base grammar (e.g., "minimal"), this function adds
    one candidate primitive at a time and measures how each addition
    affects the evaluation metric.

    If candidates is None, all primitives from the "base" grammar that
    are NOT in the base_grammar are used as candidates.

    Args:
        base_grammar: Name of the starting grammar (e.g. "minimal").
        candidates: Optional list of primitive names to try adding.
                    If None, uses primitives in "base" but not in base_grammar.
        cost_structure: The cost structure for probability assignment.
        metric: The evaluation metric. One of:
                "spearman", "weighted_log_prob", "top1", "expressibility".
        limit: If > 0, only use the first `limit` hypotheses (for speed).

    Returns:
        List of dicts sorted by absolute delta (descending), each containing:
          - added (str): Name of the added primitive.
          - metric_value (float): Metric value with this primitive added.
          - baseline_value (float): Metric value with the base grammar alone.
          - delta (float): metric_value - baseline_value.
    """
    metric_fn = _get_metric_fn(metric)

    # Load hypotheses once
    hypotheses = _load_hypotheses(limit)

    # Build the base grammar and compute baseline metric
    grammar = build_grammar(base_grammar, cost_structure)
    baseline_scored = _score_hypotheses_with_grammar(
        hypotheses, grammar, base_grammar, cost_structure.value
    )
    baseline_value = metric_fn(baseline_scored)

    # Determine candidate primitives to try adding
    base_prim_names = {str(p.program) for p in grammar.productions}

    if candidates is not None:
        # Use the "base" grammar to find Production objects for the requested names
        full_grammar = build_grammar("base", cost_structure)
        candidate_productions = [
            p for p in full_grammar.productions
            if str(p.program) in set(candidates)
        ]
    else:
        # Use all primitives from "base" that aren't in the base_grammar
        full_grammar = build_grammar("base", cost_structure)
        candidate_productions = [
            p for p in full_grammar.productions
            if str(p.program) not in base_prim_names
        ]

    # For each candidate, add it and compute the metric
    results = []
    for production in candidate_productions:
        prim_name = str(production.program)

        # Build grammar with this additional primitive
        modified_grammar = _build_grammar_with_addition(grammar, production)

        # Score hypotheses under modified grammar
        scored = _score_hypotheses_with_grammar(
            hypotheses, modified_grammar, base_grammar, cost_structure.value
        )

        # Compute metric
        metric_value = metric_fn(scored)
        delta = metric_value - baseline_value

        results.append({
            "added": prim_name,
            "metric_value": metric_value,
            "baseline_value": baseline_value,
            "delta": delta,
        })

    # Sort by absolute delta descending (most beneficial first)
    results.sort(key=lambda r: abs(r["delta"]), reverse=True)

    return results


def cross_validate(
    grammar_name: str,
    cost_structure: CostStructure = CostStructure.UNIFORM,
    k: int = 5,
    metric: str = "spearman",
    limit: int = 0,
) -> Tuple[float, float]:
    """k-fold cross-validation of a grammar's scoring metric over hypotheses.

    Splits the hypotheses into k folds. For each fold, scores the held-out
    hypotheses under the grammar and computes the evaluation metric. Returns
    the mean and standard deviation of the metric across folds.

    This helps assess how stable a grammar's metric is across different
    subsets of hypotheses, guarding against overfitting to particular rules.

    Args:
        grammar_name: Name of the grammar to evaluate (e.g. "base", "minimal").
        cost_structure: The cost structure for probability assignment.
        k: Number of folds (default 5).
        metric: The evaluation metric. One of:
                "spearman", "weighted_log_prob", "top1", "expressibility".
        limit: If > 0, only use the first `limit` hypotheses (for speed).

    Returns:
        Tuple of (mean, std) of the metric across k folds.
        std is always >= 0.
    """
    metric_fn = _get_metric_fn(metric)

    # Load hypotheses once
    hypotheses = _load_hypotheses(limit)

    # Build the grammar once
    grammar = build_grammar(grammar_name, cost_structure)

    # Score all hypotheses once (scoring is cheap; we just split the results)
    all_scored = _score_hypotheses_with_grammar(
        hypotheses, grammar, grammar_name, cost_structure.value
    )

    # If fewer hypotheses than folds, reduce k
    n = len(all_scored)
    if n == 0:
        return (0.0, 0.0)
    effective_k = min(k, n)

    # Split into k folds
    fold_size = n // effective_k
    fold_metrics: List[float] = []

    for i in range(effective_k):
        start = i * fold_size
        # Last fold gets any remainder
        if i == effective_k - 1:
            end = n
        else:
            end = start + fold_size

        # Held-out fold
        fold_scored = all_scored[start:end]

        # Compute metric on this fold
        fold_value = metric_fn(fold_scored)
        fold_metrics.append(fold_value)

    # Compute mean and standard deviation
    mean = sum(fold_metrics) / len(fold_metrics)
    variance = sum((v - mean) ** 2 for v in fold_metrics) / len(fold_metrics)
    std = math.sqrt(variance)

    return (mean, std)
