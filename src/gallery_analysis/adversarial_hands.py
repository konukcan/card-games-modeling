"""
Adversarial hand generation for Bayesian rule learning.

Given a posterior P(rule | exemplars), find hands that are MAXIMALLY DIAGNOSTIC
under the *surviving* (mass-thresholded, renormalized) posterior — i.e., hands
that would split that posterior most evenly when used as a follow-up query.

Methods
-------
1. ``find_most_diagnostic_hands`` — Posterior predictive entropy maximization
   under the surviving posterior (BALD-on-survivors, Houlsby et al. 2011).

   For each candidate hand h, given posteriors ``q(h_i) := p(h_i | D, h_i ∈ S)``:
       p_S(h)  := P(accept | D, h, S) = Σ_{i∈S} q(h_i) · I[h_i(h)]
       H_S(h) := -p_S log p_S - (1-p_S) log(1-p_S)

   ``H_S`` peaks at p_S = 0.5 — maximally divisive among the surviving
   hypotheses. NOTE: this is *NOT* the exact mutual information for the full
   pre-pruning posterior. The discarded mass m_drop = 1 − retained_mass bounds
   the TV-distance between p_S and the true posterior predictive p_full
   (|p_S − p_full| ≤ m_drop). When retained_mass is well below 1 the entropy
   score is an APPROXIMATION; the predictive-probability error is bounded by
   m_drop, but the resulting entropy can move in either direction depending
   on which side of 0.5 the perturbation lands. (Round 2 finding #2.)

   Computational cost: O(n_candidates × n_hypotheses_after_pruning).

2. ``find_most_adversarial_hands`` — Confident-but-wrong probes. Given a
   ground-truth rule predicate, find hands where the posterior predictive
   strongly disagrees with the true rule (false positives where p_accept > τ
   but true rule rejects, or false negatives where p_accept < 1−τ but true
   rule accepts). These hands surface the worst learner mistakes *visible
   under uniform-MC sampling*; not a worst-case guarantee for rare hands.

Diversity
---------
A naive top-k by entropy returns near-duplicate hands. We optionally apply a
greedy *exact-multiset* dedup on the sorted (rank, suit) tuple. This collapses
order variants and exact re-draws, NOTHING ELSE — suit-isomorphic hands
(e.g., ``2♣ 3♣`` vs ``2♠ 3♠``) survive as distinct. We deliberately do NOT
claim "structural variety"; this is the cheapest possible dedup and is named
``dedup_exact`` accordingly. (Round 1 review finding #4.)
"""
from __future__ import annotations

import math
import random
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rules.cards import Card, Hand, Suit, Rank


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AdversarialHand:
    """A hand selected by adversarial / diagnostic search.

    Fields
    ------
    hand
        The 6-card hand.
    p_accept
        Posterior predictive ``P(rule accepts | data)`` UNDER THE SURVIVING
        (mass-thresholded, renormalized) POSTERIOR. Not exact unless
        ``retained_mass ≈ 1`` for the parent posterior.
    entropy_bits
        Binary entropy of ``p_accept`` in bits.
    retained_mass
        Pre-renormalization mass kept by the parent posterior (in [0, 1]).
        ``1 - retained_mass`` upper-bounds TV error vs the full posterior
        predictive. Carried on every hand for downstream warnings.
    splitting_minority
        Top-mass hypotheses on the side OPPOSITE the predictive majority —
        i.e., the hypotheses that actually drove ``p_accept`` away from
        {0, 1}. (Round 1 finding #3.)
    splitting_majority
        Top-mass hypotheses on the predictive-majority side, included for
        symmetry / interpretability.
    """
    hand: Hand
    p_accept: float
    entropy_bits: float
    retained_mass: float = 1.0
    ground_truth: Optional[bool] = None
    correct_prediction: Optional[bool] = None
    score: float = 0.0
    score_kind: str = "entropy"      # "entropy" | "false_positive" | "false_negative"
    splitting_minority: List[Dict[str, Any]] = field(default_factory=list)
    splitting_majority: List[Dict[str, Any]] = field(default_factory=list)
    # Back-compat field: union of minority + majority. The two search
    # functions in this module populate it explicitly after annotation
    # (search for "splitting_hypotheses ="). It is NOT auto-populated on
    # construction — direct ``AdversarialHand(...)`` instantiation leaves it
    # as the default empty list. Callers building hands by hand must set it
    # themselves if they want the legacy view.
    splitting_hypotheses: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Errors / sentinels
# ---------------------------------------------------------------------------

class EmptyPosteriorError(ValueError):
    """Raised when the surviving posterior has no hypotheses to rank against.

    This is a hard failure rather than silent zero-output: an empty posterior
    means every class fell below ``mass_threshold`` and ``p_accept`` would be
    a meaningless constant 0.0 for every candidate hand. Round 1 finding #2.
    """


# ---------------------------------------------------------------------------
# Binary entropy helpers
# ---------------------------------------------------------------------------

def _binary_entropy_bits(p: float) -> float:
    """Shannon entropy of Bernoulli(p) in bits. Returns 0 at p in {0, 1}."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _hand_signature(hand: Hand) -> Tuple[Tuple[int, int], ...]:
    """Exact-multiset signature: sorted ``(rank, suit)`` tuple.

    Two hands collapse iff they contain the EXACT same six cards (in any
    order). Suit-isomorphic or rank-shifted hands stay distinct. Named
    accordingly (``_dedup_exact_hands`` below). Round 1 finding #4.
    """
    return tuple(sorted((c.rank.value, c.suit.value) for c in hand))


# ---------------------------------------------------------------------------
# Posterior predictive
# ---------------------------------------------------------------------------

def _evaluate_posterior_predictive(
    hand: Hand,
    posteriors: Sequence[Tuple[float, int, Sequence[bool]]],
    equiv_classes: Sequence[Dict[str, Any]],
) -> Tuple[float, List[Tuple[float, int, bool]]]:
    """Compute ``p_accept`` and per-hypothesis ``(prob, cls_idx, accepts)``.

    ``posteriors`` is the surviving (renormalized) posterior in the same shape
    as ``hand_diagnosticity.compute_posteriors_for_rule`` returns.

    Caller is responsible for guaranteeing ``posteriors`` is non-empty —
    callers in this module raise ``EmptyPosteriorError`` before reaching here.
    """
    p_accept = 0.0
    decisions: List[Tuple[float, int, bool]] = []
    for prob, cls_idx, _ in posteriors:
        pred = equiv_classes[cls_idx]["predicate"]
        try:
            accepts = bool(pred(hand))
        except Exception:
            accepts = False
        if accepts:
            p_accept += prob
        decisions.append((prob, cls_idx, accepts))
    return p_accept, decisions


def _build_splitter_annotations(
    decisions: Sequence[Tuple[float, int, bool]],
    p_accept: float,
    equiv_classes: Sequence[Dict[str, Any]],
    n_top: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return ``(minority_top, majority_top)`` blocks of length ≤ ``n_top``.

    Majority side = whichever vote (accept / reject) carries > 0.5 of the
    surviving mass. Minority side = the OTHER vote. For p_accept exactly
    0.5 the accept side is treated as minority (so that the reported
    "minority" block is the underdog under a small perturbation). Each block
    is sorted by mass descending and truncated to ``n_top``.

    Round 1 finding #3: the previous implementation sorted *all* decisions
    by mass and took top-N, which mostly surfaces majority voters. The
    correct interpretability story is: the minority is what kept p away
    from {0, 1}.
    """
    if not decisions:
        return [], []
    accept_side = [(p, idx, True) for p, idx, accepts in decisions if accepts]
    reject_side = [(p, idx, False) for p, idx, accepts in decisions if not accepts]
    # p_accept is the sum of accept-side probs in this surviving posterior.
    if p_accept > 0.5:
        majority, minority = accept_side, reject_side
    else:
        # ties or reject majority: minority is accept (underdog under tiny perturbation).
        majority, minority = reject_side, accept_side

    def _to_dicts(side, k):
        side_sorted = sorted(side, key=lambda d: -d[0])[:k]
        return [
            {
                "canonical_program": str(equiv_classes[idx]["canonical_program"]),
                "prob": prob,
                "accepts_hand": accepts,
                "side": "minority" if (side is minority) else "majority",
            }
            for prob, idx, accepts in side_sorted
        ]

    return _to_dicts(minority, n_top), _to_dicts(majority, n_top)


# ---------------------------------------------------------------------------
# Most-diagnostic (BALD-on-survivors)
# ---------------------------------------------------------------------------

def find_most_diagnostic_hands(
    posteriors: Sequence[Tuple[float, int, Sequence[bool]]],
    equiv_classes: Sequence[Dict[str, Any]],
    n_candidates: int = 50_000,
    top_k: int = 100,
    seed: int = 12345,
    diversity: bool = True,
    n_top_splitters: int = 5,
    ground_truth_pred: Optional[Callable[[Hand], bool]] = None,
    retained_mass: float = 1.0,
    min_retained_mass: float = 0.5,
) -> List[AdversarialHand]:
    """Return ``top_k`` hands ranked by surviving-posterior predictive entropy.

    Args:
        posteriors: Surviving posterior over equivalence classes from
            ``compute_posteriors_for_rule``. Must be non-empty
            (``EmptyPosteriorError`` otherwise).
        equiv_classes: Shared equivalence classes (for predicate access).
        n_candidates: Number of random hands to sample and score.
        top_k: How many best hands to return.
        seed: Random seed for candidate generation. Caller should derive a
            per-rule seed (e.g., ``zlib.crc32(rule_id.encode())`` — must be
            a STABLE hash, NOT Python's process-randomized ``hash()``) to
            avoid the same candidate set being reused across rules.
        diversity: If True, dedup top-k by *exact-multiset* hand signature
            (sorted (rank, suit)). NOT structural / suit-isomorphic dedup —
            see ``_hand_signature`` docstring.
        n_top_splitters: Number of top-mass hypotheses to attach FROM EACH
            SIDE (minority and majority) for interpretability.
        ground_truth_pred: Optional true-rule predicate. If supplied, the
            returned hands carry ``ground_truth`` and ``correct_prediction``.
        retained_mass: Pre-renormalization mass kept by ``posteriors``. If
            below ``min_retained_mass`` a warning is emitted (the entropy
            score is only a surviving-posterior proxy, not exact BALD).
        min_retained_mass: Threshold below which a UserWarning is emitted.

    Raises:
        EmptyPosteriorError: if ``posteriors`` is empty.

    Returns:
        List of ``AdversarialHand``, sorted by entropy descending.

    Notes:
        - Cost: O(n_candidates × |posteriors|).
        - Entropy peaks at p_accept = 0.5; ties broken by closeness to 0.5.
        - The score is exact MI for the SURVIVING posterior under
          deterministic-label predicates. It is NOT the MI under the full
          (pre-pruning) posterior unless ``retained_mass == 1.0``.
    """
    if not posteriors:
        raise EmptyPosteriorError(
            "find_most_diagnostic_hands: posterior is empty (every class fell "
            "below mass_threshold). Re-run compute_posteriors_for_rule with "
            "mass_threshold=0.0 or check upstream pool quality."
        )
    if retained_mass < min_retained_mass:
        warnings.warn(
            f"find_most_diagnostic_hands: retained_mass={retained_mass:.3f} "
            f"< min_retained_mass={min_retained_mass:.3f}; entropy scores are "
            "an approximation under surviving-posterior conditioning. The "
            "predictive-probability p_accept has TV-error bounded by "
            f"{1.0 - retained_mass:.3f} vs the full posterior; the resulting "
            "binary entropy can move in EITHER direction depending on which "
            "side of 0.5 that perturbation lands.",
            UserWarning,
            stacklevel=2,
        )

    rng = random.Random(seed)
    deck = [Card(s, r) for s in Suit for r in Rank]

    scored: List[AdversarialHand] = []
    for _ in range(n_candidates):
        hand = rng.sample(deck, 6)
        p_accept, _ = _evaluate_posterior_predictive(
            hand, posteriors, equiv_classes
        )
        ent = _binary_entropy_bits(p_accept)
        scored.append(AdversarialHand(
            hand=hand,
            p_accept=p_accept,
            entropy_bits=ent,
            retained_mass=retained_mass,
            score=ent,
            score_kind="entropy",
        ))

    # Rank by entropy desc, then by closeness to 0.5 (tiebreaker)
    scored.sort(key=lambda h: (-h.entropy_bits, abs(h.p_accept - 0.5)))

    if diversity:
        scored = _dedup_exact_hands(scored)

    out = scored[:top_k]

    # Annotate splitters (minority side first) and ground truth.
    for h in out:
        _, decisions = _evaluate_posterior_predictive(
            h.hand, posteriors, equiv_classes
        )
        minority, majority = _build_splitter_annotations(
            decisions, h.p_accept, equiv_classes, n_top_splitters
        )
        h.splitting_minority = minority
        h.splitting_majority = majority
        h.splitting_hypotheses = minority + majority  # legacy field
        if ground_truth_pred is not None:
            try:
                h.ground_truth = bool(ground_truth_pred(h.hand))
            except Exception:
                h.ground_truth = None
            if h.ground_truth is not None:
                predicted_accept = h.p_accept > 0.5
                h.correct_prediction = (predicted_accept == h.ground_truth)
    return out


# ---------------------------------------------------------------------------
# Adversarial (confident-but-wrong)
# ---------------------------------------------------------------------------

def find_most_adversarial_hands(
    posteriors: Sequence[Tuple[float, int, Sequence[bool]]],
    equiv_classes: Sequence[Dict[str, Any]],
    rule_predicate: Callable[[Hand], bool],
    n_candidates: int = 50_000,
    top_k: int = 50,
    confidence_threshold: float = 0.8,
    seed: int = 23456,
    diversity: bool = True,
    n_top_splitters: int = 5,
    retained_mass: float = 1.0,
    min_retained_mass: float = 0.5,
) -> Dict[str, List[AdversarialHand]]:
    """Find hands where the learner is *confidently wrong* about the true rule.

    Returns:
        ``{"false_positives": [...], "false_negatives": [...]}``
        - false_positives: hands where p_accept ≥ τ but rule_predicate rejects.
        - false_negatives: hands where p_accept ≤ 1−τ but rule_predicate accepts.

    Each list is sorted by ``score = |p_accept − truth|`` (more confidently
    wrong = higher score), capped at ``top_k`` per category.

    Confidence threshold τ defaults to 0.8 (80% confident wrong). This is a
    *probe* threshold, not a calibrated decision boundary — vary it in the
    driver to characterize the FP/FN curve. Round 1 finding #4.

    Raises:
        EmptyPosteriorError: if ``posteriors`` is empty.
    """
    if not posteriors:
        raise EmptyPosteriorError(
            "find_most_adversarial_hands: posterior is empty (every class "
            "fell below mass_threshold). Re-run compute_posteriors_for_rule "
            "with mass_threshold=0.0 or check upstream pool quality."
        )
    if retained_mass < min_retained_mass:
        warnings.warn(
            f"find_most_adversarial_hands: retained_mass={retained_mass:.3f} "
            f"< min_retained_mass={min_retained_mass:.3f}; FP/FN designations "
            "use surviving-posterior p_accept, which is an approximation. The "
            "predictive-probability error is bounded by "
            f"{1.0 - retained_mass:.3f}; the τ-threshold classification can "
            "flip in either direction near the boundary.",
            UserWarning,
            stacklevel=2,
        )

    rng = random.Random(seed)
    deck = [Card(s, r) for s in Suit for r in Rank]

    fps: List[AdversarialHand] = []
    fns: List[AdversarialHand] = []

    for _ in range(n_candidates):
        hand = rng.sample(deck, 6)
        try:
            truth = bool(rule_predicate(hand))
        except Exception:
            continue
        p_accept, _ = _evaluate_posterior_predictive(
            hand, posteriors, equiv_classes
        )
        ent = _binary_entropy_bits(p_accept)

        # False positive: model accepts confidently, true rule rejects
        if (not truth) and p_accept >= confidence_threshold:
            fps.append(AdversarialHand(
                hand=hand,
                p_accept=p_accept,
                entropy_bits=ent,
                retained_mass=retained_mass,
                ground_truth=False,
                correct_prediction=False,
                score=p_accept,
                score_kind="false_positive",
            ))
        # False negative: model rejects confidently, true rule accepts
        if truth and p_accept <= (1.0 - confidence_threshold):
            fns.append(AdversarialHand(
                hand=hand,
                p_accept=p_accept,
                entropy_bits=ent,
                retained_mass=retained_mass,
                ground_truth=True,
                correct_prediction=False,
                score=1.0 - p_accept,
                score_kind="false_negative",
            ))

    fps.sort(key=lambda h: -h.score)
    fns.sort(key=lambda h: -h.score)

    if diversity:
        fps = _dedup_exact_hands(fps)
        fns = _dedup_exact_hands(fns)

    fps = fps[:top_k]
    fns = fns[:top_k]

    # Annotate splitters (minority side first).
    for hands_list in (fps, fns):
        for h in hands_list:
            _, decisions = _evaluate_posterior_predictive(
                h.hand, posteriors, equiv_classes
            )
            minority, majority = _build_splitter_annotations(
                decisions, h.p_accept, equiv_classes, n_top_splitters
            )
            h.splitting_minority = minority
            h.splitting_majority = majority
            h.splitting_hypotheses = minority + majority

    return {"false_positives": fps, "false_negatives": fns}


# ---------------------------------------------------------------------------
# Diversity filter (exact-multiset only)
# ---------------------------------------------------------------------------

def _dedup_exact_hands(
    hands: Sequence[AdversarialHand],
) -> List[AdversarialHand]:
    """Greedy keep-first-by-exact-multiset dedup.

    The signature is the sorted ``(rank, suit)`` tuple, so two hands collapse
    iff they hold the EXACT same six cards. Suit-isomorphic and structurally
    similar hands survive separately. We deliberately do not implement coarser
    dedup here — that would be a separate, justified canonicalization choice.
    """
    seen = set()
    out: List[AdversarialHand] = []
    for h in hands:
        sig = _hand_signature(h.hand)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(h)
    return out


# Back-compat alias for tests / external callers using the old name.
_dedup_by_signature = _dedup_exact_hands


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def adversarial_hand_to_dict(h: AdversarialHand) -> Dict[str, Any]:
    """JSON-serializable view of an AdversarialHand."""
    return {
        "hand": [(c.rank.value, c.suit.value) for c in h.hand],
        "hand_str": " ".join(str(c) for c in h.hand),
        "p_accept": h.p_accept,
        "entropy_bits": h.entropy_bits,
        "retained_mass": h.retained_mass,
        "ground_truth": h.ground_truth,
        "correct_prediction": h.correct_prediction,
        "score": h.score,
        "score_kind": h.score_kind,
        "splitting_minority": h.splitting_minority,
        "splitting_majority": h.splitting_majority,
        "splitting_hypotheses": h.splitting_hypotheses,  # legacy
    }
