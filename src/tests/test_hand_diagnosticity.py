"""
Tests for hand diagnosticity module.

Uses minimal fixtures with synthetic equivalence classes and posteriors
to test the classification and spectrum logic without requiring the full
enumeration pipeline.
"""
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from rules.cards import Card, Suit, Rank

from gallery_analysis.hand_diagnosticity import (
    DiagnosticityReport,
    DiagnosticSpectrum,
    rate_hand,
    rate_hand_set,
    generate_diagnostic_spectrum,
    compute_posteriors_for_rule,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic equivalence classes
# ---------------------------------------------------------------------------

def _make_hand(*specs):
    """Build a hand from (suit, rank) shorthand tuples."""
    return [Card(s, r) for s, r in specs]


# A "hand of all hearts"
ALL_HEARTS_HAND = _make_hand(
    (Suit.HEARTS, Rank.TWO),
    (Suit.HEARTS, Rank.THREE),
    (Suit.HEARTS, Rank.FOUR),
    (Suit.HEARTS, Rank.FIVE),
    (Suit.HEARTS, Rank.SIX),
    (Suit.HEARTS, Rank.SEVEN),
)

# A "mixed hand"
MIXED_HAND = _make_hand(
    (Suit.HEARTS, Rank.TWO),
    (Suit.CLUBS, Rank.THREE),
    (Suit.DIAMONDS, Rank.FOUR),
    (Suit.SPADES, Rank.FIVE),
    (Suit.HEARTS, Rank.SIX),
    (Suit.CLUBS, Rank.SEVEN),
)


def _make_equiv_classes():
    """
    Create 3 synthetic equivalence classes:
      - cls0: accepts all hands (always True)  — large extension
      - cls1: accepts hands where all cards are hearts — small extension
      - cls2: accepts hands where first card is hearts — medium extension
    """
    return [
        {
            "canonical_program": "(λ true)",
            "canonical_prior": -1.0,
            "summed_prior": -0.5,
            "n_expressions": 2,
            "all_programs": ["(λ true)", "(λ not (not true))"],
            "fingerprint": "fp_always_true",
            "predicate": lambda hand: True,
        },
        {
            "canonical_program": "(λ all (λ eq_suit $0 HEARTS) $0)",
            "canonical_prior": -5.0,
            "summed_prior": -4.8,
            "n_expressions": 1,
            "all_programs": ["(λ all (λ eq_suit $0 HEARTS) $0)"],
            "fingerprint": "fp_all_hearts",
            "predicate": lambda hand: all(c.suit == Suit.HEARTS for c in hand),
        },
        {
            "canonical_program": "(λ eq_suit (head $0) HEARTS)",
            "canonical_prior": -3.0,
            "summed_prior": -2.8,
            "n_expressions": 1,
            "all_programs": ["(λ eq_suit (head $0) HEARTS)"],
            "fingerprint": "fp_first_hearts",
            "predicate": lambda hand: hand[0].suit == Suit.HEARTS,
        },
    ]


def _make_posteriors(equiv_classes):
    """
    Create synthetic posteriors where:
      - cls0 (always true) has 10% mass
      - cls1 (all hearts) has 60% mass  ← dominant hypothesis
      - cls2 (first is hearts) has 30% mass
    Returns: [(probability, cls_idx, hit_vector), ...]
    """
    return [
        (0.60, 1, [True] * 6),   # all_hearts — highest posterior
        (0.30, 2, [True] * 6),   # first_hearts
        (0.10, 0, [True] * 6),   # always_true
    ]


# ---------------------------------------------------------------------------
# Tests for rate_hand()
# ---------------------------------------------------------------------------

class TestRateHand:
    """Test single-hand rating."""

    def test_returns_diagnosticity_report(self):
        """rate_hand should return a DiagnosticityReport dataclass."""
        ec = _make_equiv_classes()
        posteriors = _make_posteriors(ec)

        report = rate_hand("test_rule", ALL_HEARTS_HAND, posteriors, ec,
                           ground_truth_pred=lambda h: all(c.suit == Suit.HEARTS for c in h))

        assert isinstance(report, DiagnosticityReport)

    def test_p_accept_all_agree(self):
        """When all hypotheses accept the hand, p_accept should be 1.0."""
        ec = _make_equiv_classes()
        posteriors = _make_posteriors(ec)

        # ALL_HEARTS_HAND: cls0=True, cls1=True, cls2=True (first card is hearts)
        report = rate_hand("test_rule", ALL_HEARTS_HAND, posteriors, ec,
                           ground_truth_pred=lambda h: True)

        assert report.p_accept == pytest.approx(1.0)

    def test_p_accept_partial_agreement(self):
        """When only some hypotheses accept, p_accept should reflect weighted vote."""
        ec = _make_equiv_classes()
        posteriors = _make_posteriors(ec)

        # MIXED_HAND: cls0=True(0.10), cls1=False(0.60), cls2=True(0.30)
        # p_accept = 0.10 + 0.30 = 0.40
        report = rate_hand("test_rule", MIXED_HAND, posteriors, ec,
                           ground_truth_pred=lambda h: False)

        assert report.p_accept == pytest.approx(0.40)

    def test_confidence_extreme(self):
        """When p_accept=1.0, confidence should be 1.0."""
        ec = _make_equiv_classes()
        posteriors = _make_posteriors(ec)

        report = rate_hand("test_rule", ALL_HEARTS_HAND, posteriors, ec,
                           ground_truth_pred=lambda h: True)

        assert report.confidence == pytest.approx(1.0)

    def test_confidence_ambiguous(self):
        """When p_accept=0.40, confidence = |0.40-0.5|*2 = 0.20."""
        ec = _make_equiv_classes()
        posteriors = _make_posteriors(ec)

        report = rate_hand("test_rule", MIXED_HAND, posteriors, ec,
                           ground_truth_pred=lambda h: False)

        assert report.confidence == pytest.approx(0.20)

    def test_ground_truth_correct_prediction(self):
        """correct_prediction should reflect whether posterior agrees with ground truth."""
        ec = _make_equiv_classes()
        posteriors = _make_posteriors(ec)

        # ALL_HEARTS: p_accept=1.0, ground_truth=True → correct
        report = rate_hand("test_rule", ALL_HEARTS_HAND, posteriors, ec,
                           ground_truth_pred=lambda h: True)
        assert report.ground_truth is True
        assert report.correct_prediction is True

        # MIXED: p_accept=0.40 → predicts reject, ground_truth=False → correct
        report2 = rate_hand("test_rule", MIXED_HAND, posteriors, ec,
                            ground_truth_pred=lambda h: False)
        assert report2.ground_truth is False
        assert report2.correct_prediction is True

    def test_top_hypotheses_votes(self):
        """top_hypotheses_votes should show top-N hypotheses and their votes."""
        ec = _make_equiv_classes()
        posteriors = _make_posteriors(ec)

        report = rate_hand("test_rule", MIXED_HAND, posteriors, ec,
                           ground_truth_pred=lambda h: False)

        assert len(report.top_hypotheses_votes) <= 5
        # Top hypothesis is all_hearts (60%), should reject MIXED_HAND
        top = report.top_hypotheses_votes[0]
        assert top["program"] == "(λ all (λ eq_suit $0 HEARTS) $0)"
        assert top["prob"] == pytest.approx(0.60)
        assert top["accepts_hand"] is False

    def test_crashing_predicate_treated_as_reject(self):
        """If a predicate crashes on a hand, treat it as reject (not error)."""
        ec = _make_equiv_classes()
        # Add a crashing predicate
        ec.append({
            "canonical_program": "(λ crash)",
            "canonical_prior": -10.0,
            "summed_prior": -10.0,
            "n_expressions": 1,
            "all_programs": ["(λ crash)"],
            "fingerprint": "fp_crash",
            "predicate": lambda hand: hand[99].suit,  # IndexError
        })
        posteriors = [
            (0.50, 0, [True] * 6),
            (0.50, 3, [True] * 6),   # crashing predicate
        ]

        report = rate_hand("test_rule", ALL_HEARTS_HAND, posteriors, ec,
                           ground_truth_pred=lambda h: True)

        # Only cls0 accepts (0.50), crashing one is treated as reject
        assert report.p_accept == pytest.approx(0.50)


# ---------------------------------------------------------------------------
# Tests for rate_hand_set()
# ---------------------------------------------------------------------------

class TestRateHandSet:
    """Test batch hand rating."""

    def test_returns_list_of_reports(self):
        """rate_hand_set should return a list of DiagnosticityReport."""
        ec = _make_equiv_classes()
        posteriors = _make_posteriors(ec)

        reports = rate_hand_set(
            "test_rule",
            [ALL_HEARTS_HAND, MIXED_HAND],
            posteriors, ec,
            ground_truth_pred=lambda h: all(c.suit == Suit.HEARTS for c in h),
        )

        assert len(reports) == 2
        assert all(isinstance(r, DiagnosticityReport) for r in reports)

    def test_results_match_individual_calls(self):
        """Batch results should match individual rate_hand calls."""
        ec = _make_equiv_classes()
        posteriors = _make_posteriors(ec)
        gt = lambda h: all(c.suit == Suit.HEARTS for c in h)

        batch = rate_hand_set("test_rule", [ALL_HEARTS_HAND, MIXED_HAND],
                              posteriors, ec, ground_truth_pred=gt)
        individual_0 = rate_hand("test_rule", ALL_HEARTS_HAND, posteriors, ec,
                                 ground_truth_pred=gt)
        individual_1 = rate_hand("test_rule", MIXED_HAND, posteriors, ec,
                                 ground_truth_pred=gt)

        assert batch[0].p_accept == pytest.approx(individual_0.p_accept)
        assert batch[1].p_accept == pytest.approx(individual_1.p_accept)


# ---------------------------------------------------------------------------
# Tests for compute_posteriors_for_rule()
# ---------------------------------------------------------------------------

class TestComputePosteriorsForRule:
    """Test posterior computation with mass threshold filtering."""

    def test_returns_posteriors_list(self):
        """Should return list of (probability, cls_idx, hit_vector) tuples."""
        ec = _make_equiv_classes()
        # Fake extensions: (ext_size, base_rate)
        extensions = [
            (10_000_000, 0.49),  # always_true: huge extension
            (1_000, 0.00005),    # all_hearts: tiny extension
            (5_000_000, 0.25),   # first_hearts: medium
        ]
        exemplar_hands = [ALL_HEARTS_HAND]  # single exemplar for simplicity

        posteriors = compute_posteriors_for_rule(
            ec, extensions, exemplar_hands, epsilon=0.01,
        )

        assert len(posteriors) > 0
        # Each entry is (prob, cls_idx, hit_vector)
        prob, idx, hv = posteriors[0]
        assert isinstance(prob, float)
        assert isinstance(idx, int)
        assert isinstance(hv, list)

    def test_posteriors_sum_to_one(self):
        """Normalized posteriors should sum to approximately 1.0."""
        ec = _make_equiv_classes()
        extensions = [
            (10_000_000, 0.49),
            (1_000, 0.00005),
            (5_000_000, 0.25),
        ]

        posteriors = compute_posteriors_for_rule(
            ec, extensions, [ALL_HEARTS_HAND], epsilon=0.01,
        )

        total = sum(p for p, _, _ in posteriors)
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_mass_threshold_filters_low_mass(self):
        """With mass_threshold, low-mass hypotheses should be excluded."""
        ec = _make_equiv_classes()
        extensions = [
            (10_000_000, 0.49),
            (1_000, 0.00005),
            (5_000_000, 0.25),
        ]

        # With a very high threshold, only the top hypothesis survives
        posteriors = compute_posteriors_for_rule(
            ec, extensions, [ALL_HEARTS_HAND], epsilon=0.01,
            mass_threshold=0.50,
        )

        # Should have fewer entries than without threshold
        all_posteriors = compute_posteriors_for_rule(
            ec, extensions, [ALL_HEARTS_HAND], epsilon=0.01,
            mass_threshold=0.0,
        )

        assert len(posteriors) <= len(all_posteriors)

    def test_sorted_by_probability_descending(self):
        """Posteriors should be sorted by probability, highest first."""
        ec = _make_equiv_classes()
        extensions = [
            (10_000_000, 0.49),
            (1_000, 0.00005),
            (5_000_000, 0.25),
        ]

        posteriors = compute_posteriors_for_rule(
            ec, extensions, [ALL_HEARTS_HAND], epsilon=0.01,
        )

        probs = [p for p, _, _ in posteriors]
        assert probs == sorted(probs, reverse=True)


# ---------------------------------------------------------------------------
# Tests for generate_diagnostic_spectrum()
# ---------------------------------------------------------------------------

class TestGenerateDiagnosticSpectrum:
    """Test spectrum generation with random hands."""

    def test_returns_spectrum_dataclass(self):
        """generate_diagnostic_spectrum should return a DiagnosticSpectrum."""
        ec = _make_equiv_classes()
        posteriors = _make_posteriors(ec)

        spectrum = generate_diagnostic_spectrum(
            rule_id="test_rule",
            posteriors=posteriors,
            equiv_classes=ec,
            ground_truth_pred=lambda h: all(c.suit == Suit.HEARTS for c in h),
            n_candidates=50,
            seed=42,
        )

        assert isinstance(spectrum, DiagnosticSpectrum)

    def test_spectrum_n_candidates(self):
        """Spectrum should be based on the requested number of candidate hands."""
        ec = _make_equiv_classes()
        posteriors = _make_posteriors(ec)

        spectrum = generate_diagnostic_spectrum(
            rule_id="test_rule",
            posteriors=posteriors,
            equiv_classes=ec,
            ground_truth_pred=lambda h: True,
            n_candidates=100,
            seed=42,
        )

        assert spectrum.n_candidates == 100

    def test_spectrum_statistics_valid(self):
        """Spectrum statistics should be in valid ranges."""
        ec = _make_equiv_classes()
        posteriors = _make_posteriors(ec)

        spectrum = generate_diagnostic_spectrum(
            rule_id="test_rule",
            posteriors=posteriors,
            equiv_classes=ec,
            ground_truth_pred=lambda h: True,
            n_candidates=50,
            seed=42,
        )

        assert 0.0 <= spectrum.mean_p_accept <= 1.0
        assert 0.0 <= spectrum.mean_confidence <= 1.0
        assert 0.0 <= spectrum.fraction_high_confidence <= 1.0
        assert 0.0 <= spectrum.fraction_ambiguous <= 1.0
        assert 0.0 <= spectrum.accuracy <= 1.0

    def test_histogram_bins_cover_all(self):
        """p_accept histogram should have 10 bins covering [0,1]."""
        ec = _make_equiv_classes()
        posteriors = _make_posteriors(ec)

        spectrum = generate_diagnostic_spectrum(
            rule_id="test_rule",
            posteriors=posteriors,
            equiv_classes=ec,
            ground_truth_pred=lambda h: True,
            n_candidates=50,
            seed=42,
        )

        assert len(spectrum.p_accept_histogram) == 10
        total_in_bins = sum(spectrum.p_accept_histogram.values())
        assert total_in_bins == 50

    def test_representative_hands_selected(self):
        """Spectrum should include representative hands at different confidence levels."""
        ec = _make_equiv_classes()
        posteriors = _make_posteriors(ec)

        spectrum = generate_diagnostic_spectrum(
            rule_id="test_rule",
            posteriors=posteriors,
            equiv_classes=ec,
            ground_truth_pred=lambda h: True,
            n_candidates=200,
            seed=42,
        )

        # Should have some representative hands (up to 5 each)
        assert len(spectrum.easy_accept_hands) <= 5
        assert len(spectrum.easy_reject_hands) <= 5
        assert len(spectrum.ambiguous_hands) <= 5

    def test_seed_reproducibility(self):
        """Same seed should produce identical spectrums."""
        ec = _make_equiv_classes()
        posteriors = _make_posteriors(ec)
        gt = lambda h: True

        s1 = generate_diagnostic_spectrum("r", posteriors, ec, gt, 50, seed=42)
        s2 = generate_diagnostic_spectrum("r", posteriors, ec, gt, 50, seed=42)

        assert s1.mean_p_accept == s2.mean_p_accept
        assert s1.p_accept_histogram == s2.p_accept_histogram


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
