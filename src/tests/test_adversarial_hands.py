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
    _dedup_exact_hands,
    _build_splitter_annotations,
    AdversarialHand,
    EmptyPosteriorError,
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
    # Round 2: new fields are surfaced.
    assert "retained_mass" in d
    assert "splitting_minority" in d
    assert "splitting_majority" in d


# ---------------------------------------------------------------------------
# Round 2 (Night 2) — failure-mode coverage requested by Codex review.
# ---------------------------------------------------------------------------

def test_empty_posterior_raises_diagnostic():
    """Empty posterior must hard-fail, not return arbitrary zero-entropy hands.

    Round 1 finding #2: previous code emitted ranked hands with p_accept=0.0
    when the surviving posterior was empty. We now refuse to rank.
    """
    with pytest.raises(EmptyPosteriorError):
        find_most_diagnostic_hands(
            posteriors=[], equiv_classes=[],
            n_candidates=10, top_k=5, seed=0,
        )


def test_empty_posterior_raises_adversarial():
    """Same hard-fail in find_most_adversarial_hands."""
    def truth(_h):
        return False

    with pytest.raises(EmptyPosteriorError):
        find_most_adversarial_hands(
            posteriors=[], equiv_classes=[], rule_predicate=truth,
            n_candidates=10, top_k=5, seed=0,
        )


def test_low_retained_mass_warns_diagnostic():
    """retained_mass below floor emits UserWarning.

    Round 1 finding #1: the BALD score on a pruned posterior is only a
    surviving-posterior proxy. When mass is mostly thrown away we must say so.
    """
    classes, posterior = _toy_classes_and_posterior()
    with pytest.warns(UserWarning, match="retained_mass"):
        find_most_diagnostic_hands(
            posterior, classes, n_candidates=200, top_k=3, seed=0,
            retained_mass=0.10, min_retained_mass=0.50,
        )


def test_low_retained_mass_warns_adversarial():
    classes, posterior = _toy_classes_and_posterior()

    def truth(h):
        return any(c.suit == Suit.SPADES for c in h)

    with pytest.warns(UserWarning, match="retained_mass"):
        find_most_adversarial_hands(
            posterior, classes, rule_predicate=truth,
            n_candidates=200, top_k=3,
            retained_mass=0.10, min_retained_mass=0.50,
            confidence_threshold=0.6, seed=0,
        )


def test_full_retained_mass_does_not_warn():
    """At retained_mass=1.0 (default), no warning is emitted."""
    classes, posterior = _toy_classes_and_posterior()
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)  # any UserWarning fails the test
        find_most_diagnostic_hands(
            posterior, classes, n_candidates=200, top_k=3, seed=0,
        )


def test_retained_mass_carried_on_each_hand():
    classes, posterior = _toy_classes_and_posterior()
    out = find_most_diagnostic_hands(
        posterior, classes, n_candidates=200, top_k=5, seed=0,
        retained_mass=0.83,
    )
    assert all(abs(h.retained_mass - 0.83) < 1e-12 for h in out)


def test_splitter_annotation_separates_minority_from_majority():
    """When p_accept ≈ 0.5 with a clear minority/majority by mass, the
    annotation must surface BOTH sides correctly.

    Round 1 finding #3: previous implementation just sorted all decisions by
    mass and took the top, often missing the actual splitter.
    """
    # Construct a synthetic decisions list where:
    # - majority side (reject) carries 0.55 mass via two classes (0.30 + 0.25)
    # - minority side (accept) carries 0.45 mass via two classes (0.40 + 0.05)
    # p_accept = 0.45, so reject is majority.
    classes = [
        {"predicate": lambda h: True, "canonical_program": "ACCEPT_BIG"},     # 0.40, accepts
        {"predicate": lambda h: False, "canonical_program": "REJECT_BIG"},    # 0.30, rejects
        {"predicate": lambda h: False, "canonical_program": "REJECT_MID"},    # 0.25, rejects
        {"predicate": lambda h: True, "canonical_program": "ACCEPT_SMALL"},   # 0.05, accepts
    ]
    decisions = [
        (0.40, 0, True),
        (0.30, 1, False),
        (0.25, 2, False),
        (0.05, 3, True),
    ]
    p_accept = 0.45
    minority, majority = _build_splitter_annotations(
        decisions, p_accept, classes, n_top=2,
    )
    assert [m["canonical_program"] for m in minority] == ["ACCEPT_BIG", "ACCEPT_SMALL"]
    assert [m["canonical_program"] for m in majority] == ["REJECT_BIG", "REJECT_MID"]
    assert all(m["side"] == "minority" for m in minority)
    assert all(m["side"] == "majority" for m in majority)
    assert all(m["accepts_hand"] is True for m in minority)
    assert all(m["accepts_hand"] is False for m in majority)


def test_splitter_annotation_exact_tie_treats_accept_as_minority():
    """Round 2 finding #4: pin the documented tie convention.

    At p_accept == 0.5 (exact tie), the helper treats reject as majority and
    accept as minority — so the minority block surfaces the underdog
    (accepting hypotheses) under any small downward perturbation. Locking
    this in a test prevents silent regressions on the boundary case.
    """
    classes = [
        {"predicate": lambda h: True, "canonical_program": "ACCEPT_A"},
        {"predicate": lambda h: True, "canonical_program": "ACCEPT_B"},
        {"predicate": lambda h: False, "canonical_program": "REJECT_A"},
        {"predicate": lambda h: False, "canonical_program": "REJECT_B"},
    ]
    decisions = [
        (0.30, 0, True),
        (0.20, 1, True),
        (0.30, 2, False),
        (0.20, 3, False),
    ]
    p_accept = 0.50
    minority, majority = _build_splitter_annotations(
        decisions, p_accept, classes, n_top=2,
    )
    assert all(m["accepts_hand"] is True for m in minority), \
        "tie convention: accept side should be labelled minority"
    assert all(m["accepts_hand"] is False for m in majority), \
        "tie convention: reject side should be labelled majority"
    assert [m["canonical_program"] for m in minority] == ["ACCEPT_A", "ACCEPT_B"]
    assert [m["canonical_program"] for m in majority] == ["REJECT_A", "REJECT_B"]


def test_splitter_annotation_handles_unanimous():
    """Unanimous accept → minority block is empty; majority block gets all."""
    classes = [
        {"predicate": lambda h: True, "canonical_program": "A"},
        {"predicate": lambda h: True, "canonical_program": "B"},
    ]
    decisions = [(0.7, 0, True), (0.3, 1, True)]
    minority, majority = _build_splitter_annotations(decisions, 1.0, classes, n_top=3)
    assert minority == []
    assert [m["canonical_program"] for m in majority] == ["A", "B"]


def test_p_accept_exactly_half_entropy_one_bit():
    """Sanity: a hand with p_accept exactly 0.5 has H = 1.0 bit."""
    assert _binary_entropy_bits(0.5) == pytest.approx(1.0, abs=1e-12)


def test_threshold_sensitivity_monotonic_in_tau():
    """Higher τ ⇒ fewer FPs (each FP requires a stricter accept-confidence)."""
    def has_spade(hand):
        return any(c.suit == Suit.SPADES for c in hand)

    def all_red(hand):
        return all(c.suit in {Suit.HEARTS, Suit.DIAMONDS} for c in hand)

    classes = [{"predicate": has_spade, "canonical_program": "(λ has_suit $0 SPADES)"}]
    posterior = [(1.0, 0, [])]

    out_lo = find_most_adversarial_hands(
        posterior, classes, rule_predicate=all_red,
        n_candidates=2_000, top_k=10_000,  # collect all, no truncation
        confidence_threshold=0.5, seed=21, diversity=False,
    )
    out_hi = find_most_adversarial_hands(
        posterior, classes, rule_predicate=all_red,
        n_candidates=2_000, top_k=10_000,
        confidence_threshold=0.95, seed=21, diversity=False,
    )
    # τ=0.5 is the loosest possible; τ=0.95 is stricter.
    assert len(out_lo["false_positives"]) >= len(out_hi["false_positives"])


def test_adversarial_full_retained_mass_silent():
    classes, posterior = _toy_classes_and_posterior()

    def truth(h):
        return False

    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error", UserWarning)
        find_most_adversarial_hands(
            posterior, classes, rule_predicate=truth,
            n_candidates=200, top_k=3,
            confidence_threshold=0.5, seed=0,
        )


def test_dedup_alias_back_compat():
    """The old ``_dedup_by_signature`` name still resolves (back-compat)."""
    assert _dedup_by_signature is _dedup_exact_hands
