"""
Extension size estimation for hypotheses using Monte Carlo sampling.

Estimates how many of the C(52, 6) = 20,358,520 possible 6-card hands
satisfy a given predicate (hypothesis). This is used for the size principle
likelihood: P(data | h) = (1 / |ext(h)|)^n, which rewards specific
hypotheses that are consistent with observed data.

The sampling uses adaptive escalation: starts with 1M samples, escalates
to 10M or 100M if the base rate is extreme (< 0.1% or > 99.9%).
"""

import sys
import os
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
import random
import math

# Add src/ to path so we can import cards
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
from rules.cards import Card, Suit, Rank, Hand

TOTAL_HANDS = 20_358_520  # C(52, 6)


@dataclass
class ExtensionResult:
    """Result of estimating the extension size of a hypothesis.

    Attributes:
        extension_size: Estimated number of 6-card hands satisfying the predicate.
        base_rate: Fraction of sampled hands that satisfied the predicate (hits / n_samples).
        n_samples: Number of Monte Carlo samples used (1M, 10M, or 100M).
        log_likelihood: Size principle log-likelihood: -n * log(extension_size).
            Always negative for valid predicates (since extension_size > 1).
            Set to -inf when exemplars are inconsistent.
        exemplars_consistent: Whether all exemplar hands satisfy the predicate.
    """
    extension_size: int
    base_rate: float
    n_samples: int
    log_likelihood: float
    exemplars_consistent: bool


def _make_deck() -> List[Card]:
    """Create a standard 52-card deck.

    Returns:
        List of all 52 Card objects (4 suits x 13 ranks).
    """
    return [Card(suit, rank) for suit in Suit for rank in Rank]


# Module-level cache for sampled hands.
# Key: (n_samples, seed) -> List of hands.
# This avoids re-sampling when estimate_extension is called multiple times
# with the same parameters (e.g., when evaluating many hypotheses).
_hand_cache: Dict[Tuple[int, int], List[List[Card]]] = {}


def _sample_hands(n: int, seed: int = 42) -> List[List[Card]]:
    """Sample n random 6-card hands (without replacement within each hand).

    Results are cached at module level so repeated calls with the same
    n and seed return instantly without re-sampling.

    Args:
        n: Number of hands to sample.
        seed: Random seed for reproducibility.

    Returns:
        List of n hands, each a list of 6 Cards.
    """
    cache_key = (n, seed)
    if cache_key in _hand_cache:
        return _hand_cache[cache_key]

    deck = _make_deck()
    rng = random.Random(seed)
    hands = [rng.sample(deck, 6) for _ in range(n)]
    _hand_cache[cache_key] = hands
    return hands


def estimate_extension(
    predicate: Callable[[List[Card]], bool],
    exemplar_hands: List[List[Card]],
    n_exemplars_expected: int = 6,
    seed: int = 42,
) -> ExtensionResult:
    """Estimate the extension size of a hypothesis and compute size principle likelihood.

    The extension of a hypothesis h is the set of all 6-card hands that satisfy it.
    We estimate |ext(h)| via Monte Carlo sampling, then compute the size principle
    log-likelihood: log P(data | h) = -n * log(|ext(h)|), where n is the number of
    observed exemplar hands.

    Adaptive escalation:
        - Start with 1,000,000 samples
        - If base_rate < 0.001, escalate to 10,000,000
        - If still < 0.001, escalate to 100,000,000

    This ensures rare or near-universal predicates get enough samples for
    a reliable estimate.

    Args:
        predicate: A function that takes a list of Cards (a hand) and returns
            True if the hand satisfies the hypothesis.
        exemplar_hands: The observed hands that the hypothesis must explain.
            If any exemplar fails the predicate, the hypothesis is rejected.
        n_exemplars_expected: Expected number of exemplars (unused, for documentation).
        seed: Random seed for reproducibility.

    Returns:
        ExtensionResult with estimated extension size, base rate, sample count,
        log-likelihood, and exemplar consistency flag.
    """
    # Step 1: Check exemplar consistency.
    # Every observed hand must satisfy the predicate for it to be a valid hypothesis.
    for hand in exemplar_hands:
        try:
            if not predicate(hand):
                return ExtensionResult(
                    extension_size=0,
                    base_rate=0.0,
                    n_samples=0,
                    log_likelihood=-math.inf,
                    exemplars_consistent=False,
                )
        except Exception:
            return ExtensionResult(
                extension_size=0,
                base_rate=0.0,
                n_samples=0,
                log_likelihood=-math.inf,
                exemplars_consistent=False,
            )

    # Step 2: Adaptive Monte Carlo sampling.
    # Start with 1M samples, escalate if base rate is extreme.
    sample_tiers = [1_000_000, 10_000_000, 100_000_000]

    hits = 0
    n_samples = 0
    base_rate = 0.0

    for tier in sample_tiers:
        hands = _sample_hands(tier, seed=seed)
        hits = 0
        for hand in hands:
            try:
                if predicate(hand):
                    hits += 1
            except Exception:
                # Predicate errors on this hand -- treat as non-hit
                pass

        n_samples = tier
        base_rate = hits / n_samples

        # If base rate is not extremely low, we have enough precision -- stop.
        # We only escalate for very rare predicates (base_rate < 0.001) where
        # we need more samples to get a reliable estimate. Near-universal
        # predicates (base_rate > 0.999) are already precisely estimated:
        # they have huge extensions, so the exact value doesn't materially
        # affect the size principle likelihood ranking.
        if base_rate >= 0.001:
            break

    # Step 3: Compute extension size and log-likelihood.
    extension_size = int(base_rate * TOTAL_HANDS)

    # log P(data | h) = -n * log(|ext(h)|)
    # This is always negative since extension_size >= 1 for any valid hypothesis.
    # If extension_size == 0 (predicate accepts no sampled hands), likelihood is -inf.
    n = len(exemplar_hands)
    if extension_size > 0:
        log_likelihood = -n * math.log(extension_size)
    else:
        log_likelihood = -math.inf

    return ExtensionResult(
        extension_size=extension_size,
        base_rate=base_rate,
        n_samples=n_samples,
        log_likelihood=log_likelihood,
        exemplars_consistent=True,
    )


def build_extension_cache(
    hypotheses: List[dict],
    predicate_key: str = "predicate",
    exemplar_hands: List[List[Card]] = None,
    probe_hands: List[List[Card]] = None,
    n_probes: int = 200,
    seed: int = 42,
) -> Dict[str, ExtensionResult]:
    """Build a cache of extension results, deduplicating by fingerprint.

    Many hypotheses are semantically equivalent (they accept exactly the same hands).
    To avoid redundant computation, we fingerprint each hypothesis by evaluating it
    on a fixed set of 200 probe hands. Hypotheses with the same fingerprint
    (same True/False pattern on all probes) share the same extension result.

    Args:
        hypotheses: List of hypothesis dicts. Each must have a callable under
            predicate_key and optionally a 'fingerprint' field.
        predicate_key: Key in each hypothesis dict for the predicate callable.
        exemplar_hands: The observed hands for likelihood computation.
            Defaults to empty list.
        probe_hands: Fixed set of hands for fingerprinting. If None, 200 hands
            are sampled with seed 0 (separate from the estimation seed).
        n_probes: Number of probe hands to use for fingerprinting.
        seed: Random seed for extension estimation.

    Returns:
        Dict mapping fingerprint string -> ExtensionResult.
    """
    if exemplar_hands is None:
        exemplar_hands = []

    # Generate probe hands for fingerprinting (use a distinct seed from estimation).
    if probe_hands is None:
        probe_hands = _sample_hands(n_probes, seed=0)
    else:
        probe_hands = probe_hands[:n_probes]

    cache: Dict[str, ExtensionResult] = {}

    for hyp in hypotheses:
        pred = hyp[predicate_key]

        # Compute fingerprint: evaluate predicate on each probe hand.
        # The fingerprint is a string of '1' and '0' characters.
        bits = []
        for hand in probe_hands:
            try:
                bits.append('1' if pred(hand) else '0')
            except Exception:
                bits.append('0')
        fingerprint = ''.join(bits)

        # If we've already computed extension for this fingerprint, skip.
        if fingerprint in cache:
            continue

        # Compute extension for this unique fingerprint.
        result = estimate_extension(pred, exemplar_hands, seed=seed)
        cache[fingerprint] = result

    return cache
