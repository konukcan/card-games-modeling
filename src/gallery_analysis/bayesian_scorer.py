"""
Bayesian scorer: compute posterior probabilities over hypotheses given data.

BAYESIAN MODEL
--------------
For each gallery rule r with exemplar hands D = {d_1, ..., d_n}:

  P(h | D) ∝ P(D | h) × P(h)

where:
  P(h)    = grammar prior, from the PCFG log probability of program h
  P(D|h)  = likelihood of observing these exemplar hands under hypothesis h

LIKELIHOOD: SIZE PRINCIPLE (Tenenbaum & Griffiths, 2001)
--------------------------------------------------------
The size principle says: if a learner assumes exemplars are sampled
uniformly from the extension of the true rule, then:

  P(D | h) = (1 / |ext(h)|)^n    if all d_i ∈ ext(h)
           = 0                     if any d_i ∉ ext(h)

where |ext(h)| is the number of 6-card hands satisfying hypothesis h,
and n is the number of observed exemplars.

This implements a "suspicious coincidence" effect: if a hypothesis has
a small extension (is very specific) and all exemplars happen to fall
in that extension, the hypothesis is strongly supported. A hypothesis
with a large extension (very permissive) gets weaker support because
the same exemplars could easily arise by chance.

NOISY LIKELIHOOD VARIANT
-------------------------
The strict size principle assigns zero likelihood when any exemplar
violates the hypothesis. A noisy variant allows for exceptions:

  P(D | h) = ∏_i [ (1-ε)/|ext(h)| + ε/|total_hands| ]   if d_i ∈ ext(h)
           = ∏_i [ ε/|total_hands| ]                       if d_i ∉ ext(h)

where ε is a noise parameter (probability of observing an exemplar
outside the true rule's extension). This prevents complete washout
from a single noisy exemplar.

PRIOR OPTIONS
-------------
Two prior formulations are supported:

1. Program prior: P(h) based on the single canonical (shortest) program
   in the equivalence class. This is the standard DreamCoder prior.

2. Summed prior: P(h) = Σ_j exp(log_prior_j) summed over all programs
   in the equivalence class. This gives credit for multiple ways to
   express the same hypothesis (more "natural" hypotheses that can be
   stated in many ways get higher prior).

EXTENSION SIZE ESTIMATION
-------------------------
|ext(h)| is estimated via Monte Carlo: sample N random 6-card hands
from the full deck (C(52,6) = 20,358,520 possible hands), count how
many satisfy h, and scale up. See hypothesis_table.estimate_extension_size().
"""
import math
from typing import Callable, Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Hand
from gallery_analysis.hypothesis_table import HypothesisTable, estimate_extension_size


# Total number of possible 6-card hands (ordered, without replacement):
# P(52, 6) = 52! / (52-6)! = 14,658,134,400
# Ordered because card position matters for many rules (e.g. ranks_palindrome,
# blacks_before_reds). Previously C(52,6) = 20,358,520 which undercounted by 6!.
# The error cancelled in posterior computations but affected reported extension sizes.
TOTAL_HANDS = 14_658_134_400


@dataclass
class ScoredHypothesis:
    """A hypothesis with its Bayesian scores computed."""
    canonical_program: str
    n_expressions: int
    # Prior
    log_prior_canonical: float   # log P(h) from shortest program
    log_prior_summed: float      # log Σ exp(log_prior_j) over equivalence class
    # Likelihood
    hit_vector: List[bool]       # which exemplars this hypothesis covers
    n_hits: int                  # number of exemplars covered
    n_exemplars: int             # total exemplars for the rule
    extension_size: int          # estimated |ext(h)|
    base_rate: float             # extension_size / TOTAL_HANDS
    log_likelihood_strict: float # log P(D|h) under strict size principle
    log_likelihood_noisy: float  # log P(D|h) under noisy variant
    # Posteriors (log, unnormalized — normalized later across all hypotheses)
    log_posterior_strict: float  # log_prior + log_likelihood_strict
    log_posterior_noisy: float   # log_prior + log_likelihood_noisy
    # Metadata
    fingerprint: str
    all_programs: List[str]


def compute_log_likelihood_strict(
    n_hits: int,
    n_exemplars: int,
    extension_size: int,
) -> float:
    """
    Strict size principle likelihood.

    P(D|h) = (1/|ext(h)|)^n if all exemplars are hits, else 0.

    If extension_size is 0 (hypothesis is empty), returns -inf.

    Args:
        n_hits: Number of exemplars that satisfy the hypothesis.
        n_exemplars: Total number of exemplars for the rule.
        extension_size: Estimated |ext(h)| — number of hands satisfying h.

    Returns:
        Log-likelihood (log P(D|h)). Returns -inf if any exemplar misses.
    """
    if n_hits < n_exemplars:
        return float('-inf')
    if extension_size <= 0:
        return float('-inf')
    # P(D|h) = (1/|ext(h)|)^n
    return -n_exemplars * math.log(extension_size)


def compute_log_likelihood_noisy(
    n_hits: int,
    n_exemplars: int,
    extension_size: int,
    epsilon: float = 0.01,
) -> float:
    """
    Noisy size principle likelihood.

    For each exemplar d_i:
      If d_i ∈ ext(h): P(d_i|h) = (1-ε)/|ext(h)| + ε/|total|
      If d_i ∉ ext(h): P(d_i|h) = ε/|total|

    This prevents complete washout from a single noisy exemplar.

    Args:
        n_hits: Number of exemplars satisfying the hypothesis.
        n_exemplars: Total number of exemplars.
        extension_size: Estimated |ext(h)|.
        epsilon: Noise parameter (default 0.01 = 1% chance of noise).

    Returns:
        Log-likelihood under the noisy model.
    """
    if extension_size <= 0:
        # Treat as maximally permissive (uniform over all hands)
        return -n_exemplars * math.log(TOTAL_HANDS)

    n_misses = n_exemplars - n_hits

    # Log probability for a hit
    p_hit = (1 - epsilon) / extension_size + epsilon / TOTAL_HANDS
    # Log probability for a miss
    p_miss = epsilon / TOTAL_HANDS

    log_lik = 0.0
    if n_hits > 0:
        log_lik += n_hits * math.log(p_hit)
    if n_misses > 0:
        log_lik += n_misses * math.log(p_miss)

    return log_lik


# ---------------------------------------------------------------------------
# Base-rate-direct variants (Finding 6, Round 1 review)
# ---------------------------------------------------------------------------
# The int-based likelihood functions above accept `extension_size: int`.
# Upstream, `hypothesis_table.estimate_extension_size()` computes
# `extension_size = int(base_rate * TOTAL_HANDS)`, which floors the precise
# float value. That rounding breaks the exact cancellation of TOTAL_HANDS in
# the posterior normalization and can silently collapse very-small base
# rates to zero (where the formulas fall back to `-inf` / uniform).
#
# The `*_from_base_rate` variants below compute the likelihood from the
# unrounded `base_rate` directly, keeping TOTAL_HANDS as a (now-exact)
# multiplicative factor inside the logs. They are mathematically identical
# to the int-based versions whenever `base_rate * TOTAL_HANDS >= 1`, and
# strictly more faithful when it doesn't.
# ---------------------------------------------------------------------------

def compute_log_likelihood_strict_from_base_rate(
    n_hits: int,
    n_exemplars: int,
    base_rate: float,
) -> float:
    """Strict size-principle likelihood scored from base_rate (no int rounding)."""
    if n_hits < n_exemplars:
        return float('-inf')
    if base_rate <= 0.0:
        return float('-inf')
    # Equivalent to `-n_exemplars * log(base_rate * TOTAL_HANDS)` without
    # the upstream int() floor on `base_rate * TOTAL_HANDS`.
    return -n_exemplars * (math.log(base_rate) + math.log(TOTAL_HANDS))


def compute_log_likelihood_noisy_from_base_rate(
    n_hits: int,
    n_exemplars: int,
    base_rate: float,
    epsilon: float = 0.01,
) -> float:
    """Noisy size-principle likelihood scored from base_rate (no int rounding)."""
    if base_rate <= 0.0:
        # MC estimator found zero hits — treat as maximally permissive
        # (uniform over TOTAL_HANDS), matching the int-based fallback
        # semantics so that base_rate=0 hypotheses get a consistent score.
        return -n_exemplars * math.log(TOTAL_HANDS)
    n_misses = n_exemplars - n_hits
    ext_precise = base_rate * TOTAL_HANDS  # float; not rounded
    p_hit = (1 - epsilon) / ext_precise + epsilon / TOTAL_HANDS
    p_miss = epsilon / TOTAL_HANDS
    log_lik = 0.0
    if n_hits > 0:
        log_lik += n_hits * math.log(p_hit)
    if n_misses > 0:
        log_lik += n_misses * math.log(p_miss)
    return log_lik


def score_hypotheses(
    equivalence_classes: List[Dict[str, Any]],
    n_exemplars: int,
    epsilon: float = 0.01,
    extension_samples: int = 100_000,
    extension_seed: int = 123,
    prior_mode: str = "summed",
) -> List[ScoredHypothesis]:
    """
    Score all equivalence classes for a single rule.

    For each equivalence class, computes:
    - Extension size via Monte Carlo sampling
    - Strict and noisy log-likelihoods
    - Log-posterior (prior + likelihood)

    Args:
        equivalence_classes: Output of HypothesisTable.get_equivalence_classes().
            Each dict has: canonical_program, canonical_prior, summed_prior,
            n_expressions, all_programs, fingerprint, hit_vector, n_hits,
            n_misses, predicate.
        n_exemplars: Number of exemplar hands for this rule.
        epsilon: Noise parameter for noisy likelihood.
        extension_samples: Number of Monte Carlo samples for |ext(h)|.
        extension_seed: Random seed for reproducibility.
        prior_mode: "canonical" or "summed" — which prior to use for posterior.

    Returns:
        List of ScoredHypothesis, sorted by log_posterior_noisy (descending).
    """
    scored = []

    for cls in equivalence_classes:
        # Estimate extension size
        ext_size, base_rate = estimate_extension_size(
            cls["predicate"],
            n_samples=extension_samples,
            seed=extension_seed,
        )

        n_hits = cls["n_hits"]

        # Compute likelihoods. Score from base_rate directly so that the
        # int(base_rate * TOTAL_HANDS) rounding in ext_size does not break
        # the exact P(52,6)/C(52,6) cancellation (Finding 6).
        log_lik_strict = compute_log_likelihood_strict_from_base_rate(
            n_hits, n_exemplars, base_rate
        )
        log_lik_noisy = compute_log_likelihood_noisy_from_base_rate(
            n_hits, n_exemplars, base_rate, epsilon
        )

        # Select prior
        log_prior_canonical = cls["canonical_prior"]
        log_prior_summed = cls["summed_prior"]

        if prior_mode == "canonical":
            log_prior = log_prior_canonical
        else:
            log_prior = log_prior_summed

        # Posterior (unnormalized)
        log_post_strict = log_prior + log_lik_strict
        log_post_noisy = log_prior + log_lik_noisy

        scored.append(ScoredHypothesis(
            canonical_program=cls["canonical_program"],
            n_expressions=cls["n_expressions"],
            log_prior_canonical=log_prior_canonical,
            log_prior_summed=log_prior_summed,
            hit_vector=cls["hit_vector"],
            n_hits=n_hits,
            n_exemplars=n_exemplars,
            extension_size=ext_size,
            base_rate=base_rate,
            log_likelihood_strict=log_lik_strict,
            log_likelihood_noisy=log_lik_noisy,
            log_posterior_strict=log_post_strict,
            log_posterior_noisy=log_post_noisy,
            fingerprint=cls["fingerprint"],
            all_programs=cls["all_programs"],
        ))

    # Sort by noisy posterior (descending = best first)
    scored.sort(key=lambda s: -s.log_posterior_noisy)
    return scored


def normalize_posteriors(
    scored: List[ScoredHypothesis],
    mode: str = "noisy",
) -> List[Tuple[ScoredHypothesis, float]]:
    """
    Normalize log-posteriors to proper probabilities using log-sum-exp.

    Args:
        scored: List of ScoredHypothesis (from score_hypotheses).
        mode: "strict" or "noisy" — which posterior to normalize.

    Returns:
        List of (ScoredHypothesis, normalized_probability) tuples,
        sorted by probability (descending).
    """
    if not scored:
        return []

    # Extract the relevant log-posteriors
    if mode == "strict":
        log_posts = [s.log_posterior_strict for s in scored]
    else:
        log_posts = [s.log_posterior_noisy for s in scored]

    # Log-sum-exp for numerical stability
    # Filter out -inf values first
    finite_posts = [lp for lp in log_posts if lp > float('-inf')]
    if not finite_posts:
        # All hypotheses have -inf posterior
        return [(s, 0.0) for s in scored]

    max_lp = max(finite_posts)
    log_normalizer = max_lp + math.log(
        sum(math.exp(lp - max_lp) for lp in log_posts if lp > float('-inf'))
    )

    # Compute normalized probabilities
    result = []
    for s, lp in zip(scored, log_posts):
        if lp > float('-inf'):
            prob = math.exp(lp - log_normalizer)
        else:
            prob = 0.0
        result.append((s, prob))

    # Sort by probability descending
    result.sort(key=lambda x: -x[1])
    return result


def compute_rule_difficulty(
    normalized: List[Tuple[ScoredHypothesis, float]],
    true_rule_program: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compute difficulty metrics for a rule based on its posterior distribution.

    Difficulty is related to how "easy" it is to identify the true rule:
    - If one hypothesis dominates the posterior → easy rule
    - If many hypotheses compete → hard rule (uncertainty)
    - If the true rule ranks low → hard rule (misleading evidence)

    Args:
        normalized: Output of normalize_posteriors().
        true_rule_program: Optional program string of the true rule
            (for computing rank metrics).

    Returns:
        Dict with difficulty metrics:
        - posterior_entropy: Shannon entropy of the posterior (higher = harder)
        - top1_probability: Probability of the most likely hypothesis
        - top5_probability: Cumulative probability of top 5 hypotheses
        - n_effective_hypotheses: exp(entropy), effective number of competing hypotheses
        - true_rule_rank: Rank of the true rule (if provided), None otherwise
        - true_rule_probability: Posterior probability of the true rule
    """
    if not normalized:
        return {
            "posterior_entropy": 0.0,
            "top1_probability": 0.0,
            "top5_probability": 0.0,
            "n_effective_hypotheses": 0,
            "true_rule_rank": None,
            "true_rule_probability": 0.0,
        }

    probs = [p for _, p in normalized]

    # Shannon entropy
    entropy = -sum(p * math.log(p) for p in probs if p > 0)

    # Top-k probabilities
    top1 = probs[0] if probs else 0.0
    top5 = sum(probs[:5])

    # Effective number of hypotheses
    n_effective = math.exp(entropy) if entropy > 0 else 1.0

    # True rule rank (if provided)
    true_rank = None
    true_prob = 0.0
    if true_rule_program:
        for i, (sh, p) in enumerate(normalized):
            if sh.canonical_program == true_rule_program:
                true_rank = i + 1
                true_prob = p
                break
            # Also check all_programs in the equivalence class
            if true_rule_program in sh.all_programs:
                true_rank = i + 1
                true_prob = p
                break

    return {
        "posterior_entropy": entropy,
        "top1_probability": top1,
        "top5_probability": top5,
        "n_effective_hypotheses": n_effective,
        "true_rule_rank": true_rank,
        "true_rule_probability": true_prob,
    }
