"""Tests for adversarial_hands module."""
import math
import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Card, Suit, Rank
from gallery_analysis.adversarial_hands import (
    find_most_diagnostic_hands,
    find_most_adversarial_hands,
    _binary_entropy_bits,
    _hand_signature,
    _dedup_by_signature,
    AdversarialHand,
    adversarial_hand_to_dict,
)


def _hand(*spec):
    """Tiny hand-builder: _hand((Rank.TWO, Suit.SPADES), ...)."""
    return [Card(s, r) for r, s in spec]


def test_binary_entropy_extremes():
    assert _binary_entropy_bits(0.0) == 0.0
    assert _binary_entropy_bits(1.0) == 0.0
    # H(0.5) = 1 bit
    assert abs(_binary_entropy_bits(0.5) - 1.0) < 1e-9
    # Symmetric
    assert _binary_entropy_bits(0.3) == pytest.approx(_binary_entropy_bits(0.7))
    # Monotone toward 0.5 from each side
    assert _binary_entropy_bits(0.1) < _binary_entropy_bits(0.3) < _binary_entropy_bits(0.5)


def test_hand_signature_canonical():
    h1 = _hand((Rank.TWO, Suit.SPADES), (Rank.ACE, Suit.HEARTS))
    h2 = _hand((Rank.ACE, Suit.HEARTS), (Rank.TWO, Suit.SPADES))
    assert _hand_signature(h1) == _hand_signature(h2)


def test_dedup_by_signature_keeps_first():
    h1 = _hand((Rank.TWO, Suit.SPADES))
    h2 = _hand((Rank.TWO, Suit.SPADES))  # same signature
    h3 = _hand((Rank.THREE, Suit.SPADES))
    objs = [
        AdversarialHand(hand=h1, p_accept=0.5, entropy_bits=1.0, score=1.0),
        AdversarialHand(hand=h2, p_accept=0.5, entropy_bits=0.99, score=0.99),
        AdversarialHand(hand=h3, p_accept=0.5, entropy_bits=0.5, score=0.5),
    ]
    out = _dedup_by_signature(objs)
    assert len(out) == 2
    assert out[0].entropy_bits == 1.0  # kept first
    assert out[1].hand == h3


def _toy_classes_and_posterior():
    """Tiny three-class example for end-to-end smoke test.

    Classes:
      0 — accepts hands containing at least one SPADE
      1 — accepts hands containing at least one HEART
      2 — accepts hands with at least 4 distinct suits
    Posterior: equal weight on classes 0 and 1, near-zero on 2.
    """
    def has_spade(hand):
        return any(c.suit == Suit.SPADES for c in hand)

    def has_heart(hand):
        return any(c.suit == Suit.HEARTS for c in hand)

    def has_4_suits(hand):
        return len({c.suit for c in hand}) >= 4

    classes = [
        {"predicate": has_spade, "canonical_program": "(λ has_suit $0 SPADES)"},
        {"predicate": has_heart, "canonical_program": "(λ has_suit $0 HEARTS)"},
        {"predicate": has_4_suits, "canonical_program": "(λ ge (n_unique_suits $0) 4)"},
    ]
    posterior = [(0.495, 0, []), (0.495, 1, []), (0.010, 2, [])]
    return classes, posterior


def test_find_most_diagnostic_hands_returns_top_k():
    classes, posterior = _toy_classes_and_posterior()
    out = find_most_diagnostic_hands(
        posterior, classes, n_candidates=2_000, top_k=10, seed=7,
        diversity=False,
    )
    assert len(out) == 10
    # Sorted by entropy descending
    for a, b in zip(out[:-1], out[1:]):
        assert a.entropy_bits >= b.entropy_bits - 1e-9


def test_find_most_diagnostic_hands_p_accept_near_half():
    """Top-entropy hands for the toy posterior should split ≈0.5.

    With 0.495+0.495 mass on disjoint-but-overlapping predicates, hands that
    contain exactly one of {spade, heart} (and not the other) get
    p_accept ≈ 0.495. The very top should be near that.
    """
    classes, posterior = _toy_classes_and_posterior()
    out = find_most_diagnostic_hands(
        posterior, classes, n_candidates=5_000, top_k=20, seed=11,
        diversity=False,
    )
    # Top hand's entropy should correspond to p_accept in [0.4, 0.6]
    top = out[0]
    assert 0.4 <= top.p_accept <= 0.6, f"got p_accept={top.p_accept}"


def test_find_most_diagnostic_diversity_filter():
    classes, posterior = _toy_classes_and_posterior()
    no_div = find_most_diagnostic_hands(
        posterior, classes, n_candidates=2_000, top_k=50, seed=13,
        diversity=False,
    )
    with_div = find_most_diagnostic_hands(
        posterior, classes, n_candidates=2_000, top_k=50, seed=13,
        diversity=True,
    )
    sigs_no_div = {_hand_signature(h.hand) for h in no_div}
    sigs_with_div = {_hand_signature(h.hand) for h in with_div}
    # Diversity-filtered set has unique signatures by construction
    assert len(sigs_with_div) == len(with_div)
    # Without diversity, returned set may have duplicates (n_candidates=2k vs top_k=50)
    # so sigs_no_div ≤ no_div in size — sanity check:
    assert len(sigs_no_div) <= len(no_div)


def test_find_most_adversarial_false_positive():
    """If the model believes 'has_spade' but truth is 'all_red', spade-bearing
    hands without red cards should appear as false positives."""
    def has_spade(hand):
        return any(c.suit == Suit.SPADES for c in hand)

    def all_red(hand):
        return all(c.suit in {Suit.HEARTS, Suit.DIAMONDS} for c in hand)

    classes = [{"predicate": has_spade, "canonical_program": "(λ has_suit $0 SPADES)"}]
    posterior = [(1.0, 0, [])]

    out = find_most_adversarial_hands(
        posterior, classes, rule_predicate=all_red,
        n_candidates=3_000, top_k=20,
        confidence_threshold=0.8, seed=17, diversity=False,
    )
    # Expect plenty of hands the model accepts (has_spade=True) where truth
    # rejects (not all-red).
    assert len(out["false_positives"]) > 0
    for h in out["false_positives"]:
        assert h.p_accept >= 0.8
        assert h.ground_truth is False


def test_find_most_adversarial_false_negative():
    """Model says 'has_spade' (rejects spadeless hands) but truth is 'all_red'
    — all-red hands have no spade, model rejects with p_accept=0, true rule accepts.
    """
    def has_spade(hand):
        return any(c.suit == Suit.SPADES for c in hand)

    def all_red(hand):
        return all(c.suit in {Suit.HEARTS, Suit.DIAMONDS} for c in hand)

    classes = [{"predicate": has_spade, "canonical_program": "(λ has_suit $0 SPADES)"}]
    posterior = [(1.0, 0, [])]

    out = find_most_adversarial_hands(
        posterior, classes, rule_predicate=all_red,
        n_candidates=10_000, top_k=20,
        confidence_threshold=0.8, seed=19, diversity=False,
    )
    # All-red hands exist (~prob (26/52)^6 ≈ 1.6%, ~160 in 10k); each gives p_accept=0
    # ground_truth=True, so falls into false_negatives.
    assert len(out["false_negatives"]) > 0
    for h in out["false_negatives"]:
        assert h.p_accept <= 0.2
        assert h.ground_truth is True


def test_to_dict_serializable():
    classes, posterior = _toy_classes_and_posterior()
    out = find_most_diagnostic_hands(
        posterior, classes, n_candidates=200, top_k=3, seed=5,
    )
    d = adversarial_hand_to_dict(out[0])
    import json
    json.dumps(d)  # raises if not serializable
    assert d["score_kind"] == "entropy"
    assert "hand" in d and len(d["hand"]) == 6
