"""Evaluation metrics for comparing grammar scoring against LLM rankings.

Each metric takes a list of scored hypothesis dicts (with at minimum
``rule_id``, ``rank``, and ``log_prob`` keys) and returns a single float
summary statistic.

Metrics
-------
spearman_rank_correlation
    Average Spearman rho between LLM rank and grammar log-prob per rule.
weighted_log_probability
    Rank-weighted sum of grammar log-probs across all hypotheses.
top1_accuracy
    Fraction of rules where the LLM's rank-1 pick equals the grammar's
    highest-log-prob hypothesis.
expressibility
    Fraction of hypotheses that received a finite grammar log-prob.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List

from scipy.stats import spearmanr


def spearman_rank_correlation(scored: List[Dict]) -> float:
    """Average Spearman rho between LLM rank and grammar log_prob, per rule.

    For each rule we compute the Spearman correlation between the ordinal
    LLM rank (1 = best) and the grammar's ``log_prob``.  Because rank 1
    should correspond to the *highest* (least negative) log-prob, we expect
    a **negative** rho when grammar and LLM agree.

    Rules with fewer than 2 hypotheses or with all-identical log_probs are
    skipped.  Returns 0.0 when no valid rules remain.
    """
    # Group hypotheses by rule_id
    by_rule: Dict[str, List[Dict]] = defaultdict(list)
    for h in scored:
        by_rule[h["rule_id"]].append(h)

    rhos: List[float] = []
    for rule_id, hyps in by_rule.items():
        # Need at least 2 hypotheses to compute a correlation
        if len(hyps) < 2:
            continue

        ranks = [h["rank"] for h in hyps]
        log_probs = [h["log_prob"] for h in hyps]

        # Skip if all log_probs are identical (no variance → undefined corr)
        if len(set(log_probs)) < 2:
            continue

        rho, _ = spearmanr(ranks, log_probs)

        # spearmanr can return nan for degenerate inputs; skip those
        if math.isnan(rho):
            continue

        rhos.append(rho)

    if not rhos:
        return 0.0

    return sum(rhos) / len(rhos)


def weighted_log_probability(scored: List[Dict]) -> float:
    """Rank-weighted sum of grammar log-probs.

    Each hypothesis is weighted by ``(6 - rank)``, so rank 1 → weight 5,
    rank 5 → weight 1.  Hypotheses with ``-inf`` log_prob are skipped
    entirely (they contribute 0 to the sum).

    Returns 0.0 for an empty input list.
    """
    total = 0.0
    for h in scored:
        lp = h["log_prob"]
        if math.isinf(lp) and lp < 0:
            continue
        weight = 6 - h["rank"]
        total += weight * lp
    return total


def top1_accuracy(scored: List[Dict]) -> float:
    """Fraction of rules where LLM rank-1 matches the grammar's top pick.

    For each rule we identify:
      (a) the hypothesis with ``rank == 1`` (LLM's favourite), and
      (b) the hypothesis with the highest ``log_prob`` (grammar's favourite).

    If both point to the same hypothesis (matched by rank), the rule counts
    as correct.  Returns 0.0 if there are no rules.
    """
    by_rule: Dict[str, List[Dict]] = defaultdict(list)
    for h in scored:
        by_rule[h["rule_id"]].append(h)

    if not by_rule:
        return 0.0

    correct = 0
    total = 0
    for rule_id, hyps in by_rule.items():
        # Find LLM's rank-1 hypothesis
        llm_top = None
        for h in hyps:
            if h["rank"] == 1:
                llm_top = h
                break
        if llm_top is None:
            # No rank-1 hypothesis for this rule; skip it
            continue

        # Find grammar's highest-log_prob hypothesis
        grammar_top = max(hyps, key=lambda h: h["log_prob"])

        total += 1
        if llm_top["rank"] == grammar_top["rank"]:
            correct += 1

    if total == 0:
        return 0.0

    return correct / total


def expressibility(scored: List[Dict]) -> float:
    """Fraction of hypotheses with a finite (non ``-inf``) grammar log-prob.

    Returns 0.0 for an empty input list.
    """
    if not scored:
        return 0.0

    finite_count = sum(
        1 for h in scored if not (math.isinf(h["log_prob"]) and h["log_prob"] < 0)
    )
    return finite_count / len(scored)
