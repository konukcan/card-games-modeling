"""
Adversarial hand generation for Bayesian rule learning.

Given a posterior P(rule | exemplars), find hands that are MAXIMALLY DIAGNOSTIC
for the ideal Bayesian learner — i.e., hands that would split the posterior
most evenly when used as a follow-up query.

Methods
-------
1. ``find_most_diagnostic_hands`` — Posterior predictive entropy maximization
   (BALD proxy, Houlsby et al. 2011, "Bayesian Active Learning by Disagreement").

   For each candidate hand h:
       p(h) := P(accept | h, D) = Σ_i posterior(h_i) · I[h_i(h)]
       H(h) := -p log p - (1-p) log(1-p)

   H is maximized when p(h) = 0.5 — the candidate is maximally divisive.

   This is the *binary* BALD score; for the binary-classification posterior
   predictive (each hypothesis assigns 0/1) it equals the mutual information
   I(h; rule | D) up to a constant. Computational cost:
       O(n_candidates × n_hypotheses_after_pruning)

2. ``find_most_adversarial_hands`` — Confident-but-wrong probes. Given a
   ground-truth rule predicate, find hands where the posterior predictive
   strongly disagrees with the true rule (false positives where p_accept > τ
   but true rule rejects, or false negatives where p_accept < 1−τ but true
   rule accepts). These hands surface the worst learner mistakes.

Diversity
---------
A naive top-k by entropy returns near-duplicate hands. We optionally apply a
greedy diversity filter on a coarse hand signature (sorted (rank, suit) tuple)
to ensure the returned set has structural variety.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from rules.cards import Card, Hand, Suit, Rank


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AdversarialHand:
    """A hand selected by adversarial / diagnostic search."""
    hand: Hand
    p_accept: float                  # posterior predictive P(rule accepts | data)
    entropy_bits: float              # binary entropy of p_accept, in bits
    ground_truth: Optional[bool] = None
    correct_prediction: Optional[bool] = None
    # For ranking; depends on selection method:
    score: float = 0.0
    score_kind: str = "entropy"      # "entropy" | "false_positive" | "false_negative"
    # Top hypotheses that drive the disagreement (canonical_program, prob, accepts)
    splitting_hypotheses: List[Dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Binary entropy helpers
# ---------------------------------------------------------------------------

def _binary_entropy_bits(p: float) -> float:
    """Shannon entropy of Bernoulli(p) in bits. Returns 0 at p in {0, 1}."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _hand_signature(hand: Hand) -> Tuple[Tuple[int, int], ...]:
    """Coarse signature for diversity filtering.

    Sorts the hand by (rank, suit) so two hands with the same multiset of
    cards (regardless of order) share a signature. NOT a perfect equivalence
    class — just a cheap dedup against trivially-similar hands.
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
    """Compute p_accept and per-hypothesis (prob, cls_idx, accepts) decisions.

    posteriors: list of (prob, cls_idx, hit_vec) — same shape as
        ``hand_diagnosticity.compute_posteriors_for_rule`` returns.
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


# ---------------------------------------------------------------------------
# Most-diagnostic (BALD)
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
) -> List[AdversarialHand]:
    """Return ``top_k`` hands ranked by posterior predictive entropy (BALD).

    Args:
        posteriors: Posterior over equivalence classes from
            ``compute_posteriors_for_rule``.
        equiv_classes: Shared equivalence classes (for predicate access).
        n_candidates: Number of random hands to sample and score.
        top_k: How many best hands to return.
        seed: Random seed for candidate generation.
        diversity: If True, dedup top-k by hand signature so we don't return
            many near-identical hands. Increases ``n_candidates`` budget needed
            to reach ``top_k`` distinct returns.
        n_top_splitters: Number of top-mass hypotheses (sorted by absolute
            disagreement contribution) to attach to each returned hand for
            interpretability.
        ground_truth_pred: Optional true-rule predicate. If supplied, the
            returned hands carry ``ground_truth`` and ``correct_prediction``.

    Returns:
        List of ``AdversarialHand``, sorted by entropy descending.

    Notes:
        - Cost: O(n_candidates × |posteriors|). With 50k candidates and ~300
          surviving classes (typical after mass_threshold pruning), this is
          ~15M predicate evaluations — order of seconds in pure Python.
        - Entropy peaks at p_accept = 0.5; ties broken by closeness to 0.5.
        - The BALD reduction to binary entropy assumes each hypothesis
          deterministically labels the hand 0/1 (true here, since predicates
          are deterministic).
    """
    rng = random.Random(seed)
    deck = [Card(s, r) for s in Suit for r in Rank]

    scored: List[AdversarialHand] = []
    for _ in range(n_candidates):
        hand = rng.sample(deck, 6)
        p_accept, decisions = _evaluate_posterior_predictive(
            hand, posteriors, equiv_classes
        )
        ent = _binary_entropy_bits(p_accept)
        scored.append(AdversarialHand(
            hand=hand,
            p_accept=p_accept,
            entropy_bits=ent,
            score=ent,
            score_kind="entropy",
        ))

    # Rank by entropy desc, then by closeness to 0.5 (tiebreaker)
    scored.sort(key=lambda h: (-h.entropy_bits, abs(h.p_accept - 0.5)))

    if diversity:
        scored = _dedup_by_signature(scored)

    out = scored[:top_k]

    # Annotate top splitters and ground truth
    for h in out:
        _, decisions = _evaluate_posterior_predictive(
            h.hand, posteriors, equiv_classes
        )
        # Top splitters: hypotheses with the largest probability mass that
        # ALSO disagree with the majority decision. Use prob as proxy for
        # contribution since each hypothesis votes once.
        decisions.sort(key=lambda d: -d[0])
        h.splitting_hypotheses = [
            {
                "canonical_program": str(equiv_classes[idx]["canonical_program"]),
                "prob": prob,
                "accepts_hand": accepts,
            }
            for prob, idx, accepts in decisions[:n_top_splitters]
        ]
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
) -> Dict[str, List[AdversarialHand]]:
    """Find hands where the learner is *confidently wrong* about the true rule.

    Returns:
        ``{"false_positives": [...], "false_negatives": [...]}``
        - false_positives: hands where p_accept > τ but rule_predicate says reject.
        - false_negatives: hands where p_accept < 1−τ but rule_predicate says accept.

    Each list is sorted by ``score = |p_accept - rule_predicate(hand)|``
    (more confidently wrong = higher score), capped at ``top_k`` per category.

    Confidence threshold τ defaults to 0.8 (80% confident wrong).
    """
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
                ground_truth=True,
                correct_prediction=False,
                score=1.0 - p_accept,
                score_kind="false_negative",
            ))

    fps.sort(key=lambda h: -h.score)
    fns.sort(key=lambda h: -h.score)

    if diversity:
        fps = _dedup_by_signature(fps)
        fns = _dedup_by_signature(fns)

    fps = fps[:top_k]
    fns = fns[:top_k]

    # Annotate top splitters
    for hands_list in (fps, fns):
        for h in hands_list:
            _, decisions = _evaluate_posterior_predictive(
                h.hand, posteriors, equiv_classes
            )
            decisions.sort(key=lambda d: -d[0])
            h.splitting_hypotheses = [
                {
                    "canonical_program": str(equiv_classes[idx]["canonical_program"]),
                    "prob": prob,
                    "accepts_hand": accepts,
                }
                for prob, idx, accepts in decisions[:n_top_splitters]
            ]

    return {"false_positives": fps, "false_negatives": fns}


# ---------------------------------------------------------------------------
# Diversity filter
# ---------------------------------------------------------------------------

def _dedup_by_signature(
    hands: Sequence[AdversarialHand],
) -> List[AdversarialHand]:
    """Greedy keep-first-by-signature dedup.

    Preserves input order. The signature is the sorted (rank, suit) tuple,
    so two hands with the same multiset collapse to one. Stricter dedup
    (e.g., on suit-multiset class) would be more aggressive but lose
    interesting structure.
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
        "ground_truth": h.ground_truth,
        "correct_prediction": h.correct_prediction,
        "score": h.score,
        "score_kind": h.score_kind,
        "splitting_hypotheses": h.splitting_hypotheses,
    }
