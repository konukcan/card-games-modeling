"""Evaluation metrics for comparing grammar scoring against LLM rankings.

Each metric takes a list of scored hypothesis dicts (with at minimum
``rule_id``, ``rank``, and ``log_posterior`` keys) and returns a single float
summary statistic.  For backward compatibility, ``log_prob`` is accepted as a
fallback when ``log_posterior`` is absent.

Metrics
-------
spearman_agreement
    Average *negated* Spearman rho between LLM rank and grammar log-posterior
    per rule.  Positive values indicate agreement.
weighted_log_probability
    Rank-weighted sum of grammar log-posteriors across all hypotheses.
top1_accuracy_corrected
    Chance-corrected fraction of rules where the LLM's rank-1 pick equals the
    grammar's highest-log-posterior hypothesis.
expressibility
    Fraction of hypotheses that received a finite grammar log-prob.
correct_rank
    Average rank of the ground-truth hypothesis (by log_posterior).
    Lower = better (1.0 means always ranked first).
rule_difficulty_correlation
    Spearman correlation between the grammar's intrinsic cost (log_prior of
    rank-1 hypothesis) and human-rated rule difficulty groups.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional

from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_log_posterior(h: Dict) -> float:
    """Return log_posterior from a hypothesis dict, falling back to log_prob."""
    if "log_posterior" in h:
        return h["log_posterior"]
    return h["log_prob"]


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def spearman_agreement(scored: List[Dict]) -> float:
    """Average *negated* Spearman rho between LLM rank and grammar log-posterior.

    For each rule we compute the Spearman correlation between the ordinal
    LLM rank (1 = best) and the grammar's log-posterior.  Because rank 1
    should correspond to the *highest* (least negative) log-posterior, the
    raw rho is **negative** when grammar and LLM agree.  We negate the
    result so that **positive = agreement**.

    Rules with fewer than 2 hypotheses or with all-identical log-posteriors
    are skipped.  Returns 0.0 when no valid rules remain.
    """
    by_rule: Dict[str, List[Dict]] = defaultdict(list)
    for h in scored:
        by_rule[h["rule_id"]].append(h)

    rhos: List[float] = []
    for rule_id, hyps in by_rule.items():
        if len(hyps) < 2:
            continue

        ranks = [h["rank"] for h in hyps]
        log_posts = [_get_log_posterior(h) for h in hyps]

        if len(set(log_posts)) < 2:
            continue

        rho, _ = spearmanr(ranks, log_posts)

        if math.isnan(rho):
            continue

        rhos.append(rho)

    if not rhos:
        return 0.0

    # Negate so positive = agreement
    return -(sum(rhos) / len(rhos))


def weighted_log_probability(scored: List[Dict]) -> float:
    """Rank-weighted sum of grammar log-posteriors.

    Each hypothesis is weighted by ``(6 - rank)``, so rank 1 -> weight 5,
    rank 5 -> weight 1.  Hypotheses with ``-inf`` log-posterior are skipped
    entirely (they contribute 0 to the sum).

    Returns 0.0 for an empty input list.
    """
    total = 0.0
    for h in scored:
        lp = _get_log_posterior(h)
        if math.isinf(lp) and lp < 0:
            continue
        weight = 6 - h["rank"]
        total += weight * lp
    return total


def top1_accuracy_corrected(scored: List[Dict]) -> float:
    """Chance-corrected top-1 accuracy.

    For each rule we check whether the LLM's rank-1 hypothesis matches the
    grammar's highest-log-posterior hypothesis.  Raw accuracy is then
    chance-corrected per rule: ``(hit - 1/k) / (1 - 1/k)`` where *k* is
    the number of hypotheses for that rule.

    The result is averaged across rules.  Can be negative (worse than
    chance).  Returns 0.0 if there are no rules.
    """
    by_rule: Dict[str, List[Dict]] = defaultdict(list)
    for h in scored:
        by_rule[h["rule_id"]].append(h)

    if not by_rule:
        return 0.0

    corrected_scores: List[float] = []
    for rule_id, hyps in by_rule.items():
        # Find LLM's rank-1 hypothesis
        llm_top = None
        for h in hyps:
            if h["rank"] == 1:
                llm_top = h
                break
        if llm_top is None:
            continue

        # Find grammar's highest-log-posterior hypothesis
        grammar_top = max(hyps, key=lambda h: _get_log_posterior(h))

        k = len(hyps)
        hit = 1.0 if llm_top["rank"] == grammar_top["rank"] else 0.0

        if k <= 1:
            # Only one hypothesis — trivially correct, chance = 1.0
            # Corrected score is 0.0 (no information)
            corrected_scores.append(0.0)
        else:
            chance = 1.0 / k
            corrected_scores.append((hit - chance) / (1.0 - chance))

    if not corrected_scores:
        return 0.0

    return sum(corrected_scores) / len(corrected_scores)


def expressibility(scored: List[Dict]) -> float:
    """Fraction of hypotheses with a finite (non ``-inf``) grammar log-prob.

    Uses ``log_posterior`` (with ``log_prob`` fallback) to determine finiteness.
    Returns 0.0 for an empty input list.
    """
    if not scored:
        return 0.0

    finite_count = sum(
        1 for h in scored
        if not (math.isinf(_get_log_posterior(h)) and _get_log_posterior(h) < 0)
    )
    return finite_count / len(scored)


def correct_rank(
    scored: List[Dict],
    ground_truth_fingerprints: Dict[str, str],
) -> float:
    """Average rank of the ground-truth hypothesis per rule.

    For each rule, hypotheses are ranked by ``log_posterior`` (highest = rank 1).
    The ground-truth hypothesis is identified by matching its ``fingerprint``
    field against *ground_truth_fingerprints[rule_id]*.

    Rules are skipped when:
    - The rule has no entry in *ground_truth_fingerprints*.
    - No hypothesis carries a ``fingerprint`` field.
    - The ground-truth fingerprint is not found among hypotheses.

    Returns ``float('inf')`` if no rules can be evaluated.
    Lower is better (1.0 = correct hypothesis always ranked first).
    """
    by_rule: Dict[str, List[Dict]] = defaultdict(list)
    for h in scored:
        by_rule[h["rule_id"]].append(h)

    ranks: List[float] = []
    for rule_id, hyps in by_rule.items():
        if rule_id not in ground_truth_fingerprints:
            continue

        gt_fp = ground_truth_fingerprints[rule_id]

        # Check that at least one hypothesis has a fingerprint
        if not any("fingerprint" in h for h in hyps):
            continue

        # Sort by log_posterior descending (highest = rank 1)
        sorted_hyps = sorted(
            hyps, key=lambda h: _get_log_posterior(h), reverse=True
        )

        # Find ground-truth hypothesis
        found = False
        for position, h in enumerate(sorted_hyps, start=1):
            if h.get("fingerprint") == gt_fp:
                ranks.append(float(position))
                found = True
                break

        # If ground-truth fingerprint not found among hypotheses, skip rule

    if not ranks:
        return float("inf")

    return sum(ranks) / len(ranks)


def rule_difficulty_correlation(
    scored: List[Dict],
    rule_groups: Dict[str, int],
) -> float:
    """Spearman correlation between grammar cost and rule difficulty.

    For each rule, finds the rank-1 (LLM's best) hypothesis and reads its
    ``log_prior``.  The *rule_groups* dict maps rule_id to a difficulty
    group (1 = easy, 2 = medium, 3 = hard).

    We expect **positive** correlation: harder rules (higher group) should
    have more negative log_prior (higher cost in the grammar).

    Wait — more negative log_prior means *lower* numerical value.  So if
    hard rules have more negative log_prior, that's a negative correlation
    between log_prior and group.  We negate the Spearman rho so that
    positive = "grammar agrees with difficulty ordering".

    Returns 0.0 when fewer than 2 rules can be evaluated.
    """
    by_rule: Dict[str, List[Dict]] = defaultdict(list)
    for h in scored:
        by_rule[h["rule_id"]].append(h)

    log_priors: List[float] = []
    groups: List[int] = []

    for rule_id, hyps in by_rule.items():
        if rule_id not in rule_groups:
            continue

        # Find rank-1 hypothesis
        rank1 = None
        for h in hyps:
            if h["rank"] == 1:
                rank1 = h
                break
        if rank1 is None:
            continue

        lp = rank1.get("log_prior")
        if lp is None:
            continue

        log_priors.append(lp)
        groups.append(rule_groups[rule_id])

    if len(log_priors) < 2:
        return 0.0

    if len(set(log_priors)) < 2 or len(set(groups)) < 2:
        return 0.0

    rho, _ = spearmanr(log_priors, groups)

    if math.isnan(rho):
        return 0.0

    # Negate: harder rules → more negative log_prior → negative raw rho
    # We want positive = agreement, so negate.
    return -rho


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

def spearman_rank_correlation(scored: List[Dict]) -> float:
    """Deprecated alias — returns the *raw* (non-negated) Spearman rho.

    Prefer :func:`spearman_agreement` which negates so positive = agreement.
    """
    return -spearman_agreement(scored)


def top1_accuracy(scored: List[Dict]) -> float:
    """Deprecated alias — returns uncorrected top-1 accuracy.

    Prefer :func:`top1_accuracy_corrected` which applies chance correction.
    """
    by_rule: Dict[str, List[Dict]] = defaultdict(list)
    for h in scored:
        by_rule[h["rule_id"]].append(h)

    if not by_rule:
        return 0.0

    correct = 0
    total = 0
    for rule_id, hyps in by_rule.items():
        llm_top = None
        for h in hyps:
            if h["rank"] == 1:
                llm_top = h
                break
        if llm_top is None:
            continue

        grammar_top = max(hyps, key=lambda h: _get_log_posterior(h))

        total += 1
        if llm_top["rank"] == grammar_top["rank"]:
            correct += 1

    if total == 0:
        return 0.0

    return correct / total
