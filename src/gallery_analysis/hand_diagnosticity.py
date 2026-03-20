"""
Hand diagnosticity: rate how diagnostic (easy/hard to classify) candidate hands
are for a given rule, based on the Bayesian posterior over hypotheses.

Given a rule and its 6 exemplar hands, the posterior distribution tells us which
hypotheses the ideal learner considers plausible. For a new candidate hand, we
compute the posterior predictive probability — the weighted vote across all
plausible hypotheses on whether that hand satisfies the rule.

Hands where the posterior is confident (near 0 or 1) are "diagnostic" — easy to
classify. Hands near 0.5 are "ambiguous" — the hypotheses disagree.

Usage:
    from gallery_analysis.hand_diagnosticity import (
        rate_hand, rate_hand_set, generate_diagnostic_spectrum,
        compute_posteriors_for_rule,
    )
"""
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Card, Hand, Suit, Rank
from gallery_analysis.bayesian_scorer import (
    compute_log_likelihood_noisy,
    TOTAL_HANDS,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class DiagnosticityReport:
    """Rating for a single candidate hand against a rule's posterior."""
    hand: Hand
    rule_id: str
    p_accept: float                    # posterior predictive P(hand ∈ rule | data)
    confidence: float                  # |p_accept - 0.5| × 2, in [0, 1]
    ground_truth: bool                 # does hand actually satisfy the rule?
    correct_prediction: bool           # posterior agrees with ground truth?
    top_hypotheses_votes: List[Dict]   # top 5 hypotheses: {program, prob, accepts_hand}


@dataclass
class DiagnosticSpectrum:
    """Summary of diagnosticity across many random candidate hands for one rule."""
    rule_id: str
    group: int
    n_candidates: int
    # Distribution statistics
    mean_p_accept: float
    std_p_accept: float
    mean_confidence: float
    fraction_high_confidence: float    # confidence > 0.8
    fraction_ambiguous: float          # confidence < 0.2
    accuracy: float                    # fraction of correct predictions
    # Binned distribution
    p_accept_histogram: Dict[str, int]  # bins like "0.0-0.1", "0.1-0.2", etc.
    # Selected representative hands
    easy_accept_hands: List[DiagnosticityReport]   # high confidence, accept
    easy_reject_hands: List[DiagnosticityReport]    # high confidence, reject
    ambiguous_hands: List[DiagnosticityReport]      # low confidence
    # Ground-truth-split histograms: bin_label → {"true_accept": count, "true_reject": count}
    gt_histogram: Dict[str, Dict[str, int]] = field(default_factory=dict)
    balanced_gt_histogram: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # Per-hand summaries: lightweight (p_accept, ground_truth) pairs for all hands
    hand_summaries: List[Dict[str, Any]] = field(default_factory=list)
    balanced_hand_summaries: List[Dict[str, Any]] = field(default_factory=list)
    # Balanced sampling (accept + reject hands in equal numbers)
    balanced_reports: List[DiagnosticityReport] = field(default_factory=list)
    balanced_n: int = 0


# ---------------------------------------------------------------------------
# Core: compute posteriors for a rule
# ---------------------------------------------------------------------------

def compute_posteriors_for_rule(
    equiv_classes: List[Dict[str, Any]],
    extensions: List[Tuple[int, float]],
    exemplar_hands: List[Hand],
    epsilon: float = 0.01,
    prior_mode: str = "summed",
    mass_threshold: float = 0.001,
    grammar=None,
    likelihood_exponent: float = 1.0,
) -> List[Tuple[float, int, List[bool]]]:
    """
    Compute normalized posteriors for all equivalence classes given exemplar hands.

    This mirrors the scoring logic in analyze.py/depth_mass_analysis.py but
    returns posteriors in the format needed for hand classification, with an
    optional mass threshold to prune low-mass hypotheses for efficiency.

    Args:
        equiv_classes: Shared equivalence classes from build_hypothesis_pool().
        extensions: Parallel list of (ext_size, base_rate) tuples.
        exemplar_hands: The rule's exemplar hands (typically 6).
        epsilon: Noise parameter for noisy likelihood.
        prior_mode: "canonical" or "summed".
        mass_threshold: Drop hypotheses with posterior mass below this fraction.
            Default 0.001 (0.1%) — typically keeps ~50-200 of ~4,135 classes,
            giving ~20× speedup when rating many hands.
        grammar: Optional grammar object for re-scoring priors. When provided,
            priors are recomputed under this grammar (e.g. weighted 4-tier)
            instead of using the stored uniform priors.
        likelihood_exponent: Exponent k on P(D|h)^k. k>1 inflates size principle.

    Returns:
        List of (probability, cls_idx, hit_vector) tuples, sorted by probability
        descending. Only includes hypotheses above mass_threshold.
    """
    # Lazy import to avoid circular dependency
    if grammar is not None:
        from gallery_analysis.analyze import _recompute_class_prior

    n_exemplars = len(exemplar_hands)

    # Score each equivalence class
    scored = []  # (log_posterior, cls_idx, hit_vector)
    for i, (cls, (ext_size, base_rate)) in enumerate(zip(equiv_classes, extensions)):
        pred = cls["predicate"]

        # Compute hit vector: which exemplars does this hypothesis accept?
        hit_vector = []
        n_hits = 0
        for hand in exemplar_hands:
            try:
                result = pred(hand)
                hit_vector.append(result)
                if result:
                    n_hits += 1
            except Exception:
                hit_vector.append(False)

        # Likelihood (noisy size principle)
        log_lik = compute_log_likelihood_noisy(n_hits, n_exemplars, ext_size, epsilon)

        # Prior: recompute under provided grammar, or use stored prior
        if grammar is not None and prior_mode == "canonical":
            # Canonical under new grammar: use only the single cheapest program
            from gallery_analysis.dsl_prior import compute_log_prior
            try:
                log_prior = compute_log_prior(cls["canonical_program"], grammar)
            except Exception:
                log_prior = float('-inf')
        elif grammar is not None:
            # Summed under new grammar: log-sum-exp across all programs
            log_prior = _recompute_class_prior(cls, grammar)
        elif prior_mode == "canonical":
            log_prior = cls["canonical_prior"]
        else:
            log_prior = cls["summed_prior"]

        log_post = log_prior + likelihood_exponent * log_lik
        scored.append((log_post, i, hit_vector))

    # Normalize to get P(h_j | D)
    scored.sort(key=lambda x: -x[0])
    max_lp = scored[0][0]
    log_norm = max_lp + math.log(sum(math.exp(s[0] - max_lp) for s in scored))

    posteriors = []
    for log_post, cls_idx, hit_vec in scored:
        prob = math.exp(log_post - log_norm)
        if prob >= mass_threshold:
            posteriors.append((prob, cls_idx, hit_vec))

    # Renormalize after pruning so posteriors sum to 1.0.
    # Without this, p_accept in rate_hand would be biased downward
    # by the missing tail mass.
    if posteriors:
        total_mass = sum(p for p, _, _ in posteriors)
        if total_mass > 0 and total_mass < 1.0:
            posteriors = [(p / total_mass, idx, hv) for p, idx, hv in posteriors]

    return posteriors


# ---------------------------------------------------------------------------
# Core: rate a single hand
# ---------------------------------------------------------------------------

def rate_hand(
    rule_id: str,
    new_hand: Hand,
    posteriors: List[Tuple[float, int, List[bool]]],
    equiv_classes: List[Dict[str, Any]],
    ground_truth_pred: Callable[[Hand], bool],
    n_top: int = 5,
) -> DiagnosticityReport:
    """
    Rate how diagnostic a single candidate hand is for a given rule.

    Computes the posterior predictive probability P(hand ∈ rule | data) by
    taking a weighted vote across all posterior hypotheses.

    Args:
        rule_id: Gallery rule identifier.
        new_hand: The candidate hand to classify.
        posteriors: [(probability, cls_idx, hit_vector), ...] from scoring.
        equiv_classes: The shared equivalence classes.
        ground_truth_pred: The true rule's predicate for computing ground truth.
        n_top: Number of top hypotheses to include in the vote report.

    Returns:
        DiagnosticityReport with p_accept, confidence, ground truth, and top votes.
    """
    # Posterior predictive: weighted vote
    p_accept = 0.0
    top_votes = []

    for prob, cls_idx, _ in posteriors:
        pred = equiv_classes[cls_idx]["predicate"]
        try:
            accepts = bool(pred(new_hand))
        except Exception:
            accepts = False

        if accepts:
            p_accept += prob

        if len(top_votes) < n_top:
            top_votes.append({
                "program": equiv_classes[cls_idx]["canonical_program"],
                "prob": prob,
                "accepts_hand": accepts,
            })

    # Confidence: how far from 0.5 (scaled to [0, 1])
    confidence = abs(p_accept - 0.5) * 2.0

    # Ground truth
    try:
        ground_truth = bool(ground_truth_pred(new_hand))
    except Exception:
        ground_truth = False

    # Does the posterior prediction match ground truth?
    # p_accept > 0.5 → predict accept; p_accept <= 0.5 → predict reject
    predicted_accept = p_accept > 0.5
    correct_prediction = (predicted_accept == ground_truth)

    return DiagnosticityReport(
        hand=new_hand,
        rule_id=rule_id,
        p_accept=p_accept,
        confidence=confidence,
        ground_truth=ground_truth,
        correct_prediction=correct_prediction,
        top_hypotheses_votes=top_votes,
    )


# ---------------------------------------------------------------------------
# Batch rating
# ---------------------------------------------------------------------------

def rate_hand_set(
    rule_id: str,
    hands: List[Hand],
    posteriors: List[Tuple[float, int, List[bool]]],
    equiv_classes: List[Dict[str, Any]],
    ground_truth_pred: Callable[[Hand], bool],
    n_top: int = 5,
) -> List[DiagnosticityReport]:
    """
    Rate multiple candidate hands at once.

    This is a convenience wrapper around rate_hand — each hand is rated
    independently against the same posterior.

    Args:
        rule_id: Gallery rule identifier.
        hands: List of candidate hands to rate.
        posteriors: [(probability, cls_idx, hit_vector), ...] from scoring.
        equiv_classes: The shared equivalence classes.
        ground_truth_pred: The true rule's predicate.
        n_top: Number of top hypotheses per hand.

    Returns:
        List of DiagnosticityReport, one per hand.
    """
    return [
        rate_hand(rule_id, hand, posteriors, equiv_classes, ground_truth_pred, n_top)
        for hand in hands
    ]


# ---------------------------------------------------------------------------
# Diagnostic spectrum
# ---------------------------------------------------------------------------

def generate_diagnostic_spectrum(
    rule_id: str,
    posteriors: List[Tuple[float, int, List[bool]]],
    equiv_classes: List[Dict[str, Any]],
    ground_truth_pred: Callable[[Hand], bool],
    n_candidates: int = 10_000,
    seed: int = 42,
    group: int = 0,
    n_representative: int = 5,
    balanced_n: int = 0,
    verbose: int = 0,
) -> DiagnosticSpectrum:
    """
    Sample random hands and rate them to produce a diagnosticity spectrum.

    The spectrum shows the distribution of classification confidence across
    random hands, revealing how many hands are easy vs. ambiguous for this rule.

    Args:
        rule_id: Gallery rule identifier.
        posteriors: Pre-computed posteriors for this rule.
        equiv_classes: Shared equivalence classes.
        ground_truth_pred: The true rule's predicate.
        n_candidates: Number of random hands to sample and rate.
        seed: Random seed for reproducibility.
        group: Difficulty group (1=easy, 2=medium, 3=hard) for metadata.
        n_representative: Number of representative hands per category.
        balanced_n: When > 0, additionally generate this many accept + this many
            reject hands via rejection sampling, then rate them. Stored in
            ``balanced_reports`` on the returned spectrum.
        verbose: Verbosity level (0=silent, 2=progress for balanced sampling).

    Returns:
        DiagnosticSpectrum with distribution statistics and representative hands.
    """
    # Generate random candidate hands
    rng = random.Random(seed)
    deck = [Card(suit, rank) for suit in Suit for rank in Rank]

    candidate_hands = [rng.sample(deck, 6) for _ in range(n_candidates)]

    # Rate all candidates
    reports = rate_hand_set(
        rule_id, candidate_hands, posteriors, equiv_classes, ground_truth_pred,
    )

    # Compute statistics
    p_accepts = [r.p_accept for r in reports]
    confidences = [r.confidence for r in reports]
    corrects = [r.correct_prediction for r in reports]

    mean_p = sum(p_accepts) / len(p_accepts)
    std_p = math.sqrt(sum((p - mean_p) ** 2 for p in p_accepts) / len(p_accepts))
    mean_conf = sum(confidences) / len(confidences)
    n_high_conf = sum(1 for c in confidences if c > 0.8)
    n_ambiguous = sum(1 for c in confidences if c < 0.2)
    n_correct = sum(corrects)

    # Histogram: 10 bins for p_accept
    bin_edges = [i / 10 for i in range(11)]
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(10)]
    histogram = {label: 0 for label in bin_labels}

    for p in p_accepts:
        # Find which bin this falls into
        bin_idx = min(int(p * 10), 9)  # clamp to [0, 9]
        histogram[bin_labels[bin_idx]] += 1

    # Select representative hands
    # Sort reports into categories
    easy_accept = sorted(
        [r for r in reports if r.confidence > 0.8 and r.p_accept > 0.5],
        key=lambda r: -r.confidence,
    )[:n_representative]

    easy_reject = sorted(
        [r for r in reports if r.confidence > 0.8 and r.p_accept <= 0.5],
        key=lambda r: -r.confidence,
    )[:n_representative]

    ambiguous = sorted(
        [r for r in reports if r.confidence < 0.2],
        key=lambda r: r.confidence,
    )[:n_representative]

    # --- Balanced sampling (rejection sampling for accept + reject hands) ---
    balanced_reports: List[DiagnosticityReport] = []
    actual_balanced_n = 0

    if balanced_n > 0:
        MAX_ATTEMPTS = 1_000_000
        balanced_rng = random.Random(seed + 999)  # separate seed stream
        accept_hands: List[Hand] = []
        reject_hands: List[Hand] = []
        attempts = 0

        if verbose >= 2:
            print(f"    Balanced sampling: targeting {balanced_n} accept + "
                  f"{balanced_n} reject hands...", flush=True)

        while (len(accept_hands) < balanced_n or len(reject_hands) < balanced_n) \
                and attempts < MAX_ATTEMPTS:
            hand = balanced_rng.sample(deck, 6)
            attempts += 1
            try:
                accepts = bool(ground_truth_pred(hand))
            except Exception:
                continue

            if accepts and len(accept_hands) < balanced_n:
                accept_hands.append(hand)
            elif not accepts and len(reject_hands) < balanced_n:
                reject_hands.append(hand)

            # Progress reporting every 100k attempts
            if verbose >= 2 and attempts % 100_000 == 0:
                print(f"      {attempts:,} attempts: "
                      f"{len(accept_hands)}/{balanced_n} accept, "
                      f"{len(reject_hands)}/{balanced_n} reject", flush=True)

        if verbose >= 2:
            print(f"      Done: {len(accept_hands)} accept + {len(reject_hands)} reject "
                  f"in {attempts:,} attempts", flush=True)

        # Rate all balanced hands
        balanced_hands = accept_hands + reject_hands
        balanced_reports = rate_hand_set(
            rule_id, balanced_hands, posteriors, equiv_classes, ground_truth_pred,
        )
        actual_balanced_n = balanced_n

    # --- Ground-truth-split histogram (uniform sampling) ---
    gt_histogram: Dict[str, Dict[str, int]] = {
        label: {"true_accept": 0, "true_reject": 0} for label in bin_labels
    }
    for r in reports:
        bin_idx = min(int(r.p_accept * 10), 9)
        key = "true_accept" if r.ground_truth else "true_reject"
        gt_histogram[bin_labels[bin_idx]][key] += 1

    # --- Ground-truth-split histogram (balanced sampling) ---
    balanced_gt_histogram: Dict[str, Dict[str, int]] = {}
    if balanced_reports:
        balanced_gt_histogram = {
            label: {"true_accept": 0, "true_reject": 0} for label in bin_labels
        }
        for r in balanced_reports:
            bin_idx = min(int(r.p_accept * 10), 9)
            key = "true_accept" if r.ground_truth else "true_reject"
            balanced_gt_histogram[bin_labels[bin_idx]][key] += 1

    # Per-hand summaries: lightweight (p_accept, ground_truth) for all hands.
    hand_summaries = [
        {"p_accept": round(r.p_accept, 6), "ground_truth": r.ground_truth}
        for r in reports
    ]
    balanced_hand_summaries = [
        {"p_accept": round(r.p_accept, 6), "ground_truth": r.ground_truth}
        for r in balanced_reports
    ]

    return DiagnosticSpectrum(
        rule_id=rule_id,
        group=group,
        n_candidates=n_candidates,
        mean_p_accept=mean_p,
        std_p_accept=std_p,
        mean_confidence=mean_conf,
        fraction_high_confidence=n_high_conf / n_candidates,
        fraction_ambiguous=n_ambiguous / n_candidates,
        accuracy=n_correct / n_candidates,
        p_accept_histogram=histogram,
        gt_histogram=gt_histogram,
        balanced_gt_histogram=balanced_gt_histogram,
        hand_summaries=hand_summaries,
        balanced_hand_summaries=balanced_hand_summaries,
        easy_accept_hands=easy_accept,
        easy_reject_hands=easy_reject,
        ambiguous_hands=ambiguous,
        balanced_reports=balanced_reports,
        balanced_n=actual_balanced_n,
    )
