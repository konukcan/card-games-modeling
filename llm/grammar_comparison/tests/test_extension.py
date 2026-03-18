"""
Tests for extension size estimation and size principle likelihood.

These tests verify:
- Base rate estimates for known predicates (all_even, constant_true)
- Exemplar consistency checking
- Adaptive escalation for rare predicates
- Log-likelihood sign and ordering (specific > vague)
- Fingerprint-based caching in build_extension_cache
"""

import sys
import os
import math
import pytest

# Add project paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from rules.cards import Card, Suit, Rank, RANK_VALUES, Hand
from grammar_comparison.evaluation.extension import (
    TOTAL_HANDS,
    ExtensionResult,
    _make_deck,
    _sample_hands,
    estimate_extension,
    build_extension_cache,
)


# ---------------------------------------------------------------------------
# Helper predicates
# ---------------------------------------------------------------------------

def all_even(hand):
    """True if every card in the hand has an even rank value (2,4,6,8,10,Q)."""
    return all(RANK_VALUES[c.rank] % 2 == 0 for c in hand)


def constant_true(hand):
    """Always returns True -- accepts every hand."""
    return True


def constant_false(hand):
    """Always returns False -- accepts no hand."""
    return False


def all_hearts(hand):
    """True if every card is a heart. Very rare: (13/52)^6-ish by sampling."""
    return all(c.suit == Suit.HEARTS for c in hand)


def has_ace(hand):
    """True if at least one card is an ace. Common predicate."""
    return any(c.rank == Rank.ACE for c in hand)


# ---------------------------------------------------------------------------
# Exemplar hands for testing
# ---------------------------------------------------------------------------

def _make_all_even_hand():
    """A hand of 6 even-ranked cards (guaranteed to satisfy all_even)."""
    return [
        Card(Suit.HEARTS, Rank.TWO),
        Card(Suit.DIAMONDS, Rank.FOUR),
        Card(Suit.CLUBS, Rank.SIX),
        Card(Suit.SPADES, Rank.EIGHT),
        Card(Suit.HEARTS, Rank.TEN),
        Card(Suit.DIAMONDS, Rank.QUEEN),
    ]


def _make_mixed_hand():
    """A hand with both odd and even ranks (fails all_even)."""
    return [
        Card(Suit.HEARTS, Rank.TWO),
        Card(Suit.DIAMONDS, Rank.THREE),  # odd
        Card(Suit.CLUBS, Rank.SIX),
        Card(Suit.SPADES, Rank.EIGHT),
        Card(Suit.HEARTS, Rank.TEN),
        Card(Suit.DIAMONDS, Rank.QUEEN),
    ]


def _make_all_hearts_hand():
    """A hand of 6 hearts."""
    return [
        Card(Suit.HEARTS, Rank.TWO),
        Card(Suit.HEARTS, Rank.FOUR),
        Card(Suit.HEARTS, Rank.SIX),
        Card(Suit.HEARTS, Rank.EIGHT),
        Card(Suit.HEARTS, Rank.TEN),
        Card(Suit.HEARTS, Rank.QUEEN),
    ]


# ===========================================================================
# Tests
# ===========================================================================

class TestMakeDeck:
    def test_deck_has_52_cards(self):
        deck = _make_deck()
        assert len(deck) == 52

    def test_deck_has_unique_cards(self):
        deck = _make_deck()
        assert len(set(deck)) == 52


class TestSampleHands:
    def test_correct_count(self):
        hands = _sample_hands(100, seed=99)
        assert len(hands) == 100

    def test_each_hand_has_6_cards(self):
        hands = _sample_hands(50, seed=99)
        for hand in hands:
            assert len(hand) == 6

    def test_no_duplicates_within_hand(self):
        """Each hand is sampled without replacement, so no duplicate cards."""
        hands = _sample_hands(200, seed=99)
        for hand in hands:
            assert len(set(hand)) == 6

    def test_caching(self):
        """Calling with the same n and seed returns the exact same object."""
        a = _sample_hands(100, seed=77)
        b = _sample_hands(100, seed=77)
        assert a is b  # same object, not just equal

    def test_different_seeds_differ(self):
        a = _sample_hands(100, seed=1)
        b = _sample_hands(100, seed=2)
        assert a != b


class TestEstimateExtension:
    """Tests for the core estimate_extension function."""

    def test_all_even_base_rate(self):
        """all_even should have a base rate around 1.8% (6/13)^6 ~ 0.018."""
        exemplars = [_make_all_even_hand()]
        result = estimate_extension(all_even, exemplars)

        # Theoretical: (6/13)^6 = 0.0177. Allow generous margin for sampling.
        assert 0.010 < result.base_rate < 0.030, (
            f"Expected base_rate ~0.018, got {result.base_rate}"
        )
        # Extension size should be roughly 0.018 * 20M ~ 360K
        assert 200_000 < result.extension_size < 600_000

    def test_constant_true_base_rate(self):
        """constant_true accepts every hand, so base_rate should be ~1.0."""
        exemplars = [_make_all_even_hand()]
        result = estimate_extension(constant_true, exemplars)

        assert result.base_rate > 0.999
        assert result.extension_size > 20_000_000

    def test_exemplar_inconsistency(self):
        """If an exemplar fails the predicate, log_likelihood should be -inf."""
        # _make_mixed_hand has odd cards, so all_even returns False on it
        exemplars = [_make_all_even_hand(), _make_mixed_hand()]
        result = estimate_extension(all_even, exemplars)

        assert result.log_likelihood == -math.inf
        assert result.exemplars_consistent is False
        assert result.extension_size == 0

    def test_adaptive_escalation_rare_predicate(self):
        """A very rare predicate (all_hearts) should trigger escalation to > 1M samples."""
        exemplars = [_make_all_hearts_hand()]
        result = estimate_extension(all_hearts, exemplars)

        # all_hearts base rate ~ (13/52 choose 6) / C(52,6) ~ 0.000084
        # This is < 0.001, so it should escalate beyond 1M samples.
        assert result.n_samples > 1_000_000, (
            f"Expected escalation for rare predicate, got n_samples={result.n_samples}"
        )
        assert result.base_rate < 0.001

    def test_log_likelihood_is_negative(self):
        """For any valid predicate, log_likelihood = -n * log(ext_size) should be negative."""
        exemplars = [_make_all_even_hand()]
        result = estimate_extension(all_even, exemplars)

        assert result.log_likelihood < 0
        assert math.isfinite(result.log_likelihood)

    def test_specific_hypothesis_has_higher_likelihood(self):
        """A more specific hypothesis should have a LESS negative (higher) log_likelihood.

        all_even is more specific than has_ace, so it should be rewarded more
        by the size principle (its extension is smaller, so each consistent
        observation is more surprising and thus more informative).
        """
        exemplars_even = [_make_all_even_hand()]
        # Make exemplars that satisfy has_ace
        exemplars_ace = [[
            Card(Suit.HEARTS, Rank.ACE),
            Card(Suit.DIAMONDS, Rank.FOUR),
            Card(Suit.CLUBS, Rank.SIX),
            Card(Suit.SPADES, Rank.EIGHT),
            Card(Suit.HEARTS, Rank.TEN),
            Card(Suit.DIAMONDS, Rank.QUEEN),
        ]]

        result_specific = estimate_extension(all_even, exemplars_even)
        result_vague = estimate_extension(has_ace, exemplars_ace)

        # all_even ext ~ 360K, has_ace ext ~ 8M
        # log_likelihood_specific = -1 * log(360K) ~ -12.8
        # log_likelihood_vague = -1 * log(8M) ~ -15.9
        # specific > vague (less negative)
        assert result_specific.log_likelihood > result_vague.log_likelihood, (
            f"Specific ({result_specific.log_likelihood:.2f}) should be > "
            f"vague ({result_vague.log_likelihood:.2f})"
        )

    def test_more_exemplars_amplify_difference(self):
        """With more exemplars, the likelihood gap between specific and vague widens.

        log_likelihood = -n * log(ext_size), so doubling n doubles the gap.
        """
        hand1 = _make_all_even_hand()
        hand2 = [
            Card(Suit.CLUBS, Rank.TWO),
            Card(Suit.SPADES, Rank.FOUR),
            Card(Suit.HEARTS, Rank.SIX),
            Card(Suit.DIAMONDS, Rank.EIGHT),
            Card(Suit.CLUBS, Rank.TEN),
            Card(Suit.SPADES, Rank.QUEEN),
        ]

        result_1 = estimate_extension(all_even, [hand1])
        result_2 = estimate_extension(all_even, [hand1, hand2])

        # With 2 exemplars, log_likelihood should be roughly 2x as negative
        assert result_2.log_likelihood < result_1.log_likelihood
        ratio = result_2.log_likelihood / result_1.log_likelihood
        assert 1.8 < ratio < 2.2, f"Expected ratio ~2.0, got {ratio:.2f}"

    def test_predicate_that_errors(self):
        """A predicate that raises an exception on exemplars should be marked inconsistent."""
        def broken_predicate(hand):
            raise ValueError("oops")

        exemplars = [_make_all_even_hand()]
        result = estimate_extension(broken_predicate, exemplars)

        assert result.exemplars_consistent is False
        assert result.log_likelihood == -math.inf


class TestBuildExtensionCache:
    """Tests for fingerprint-based caching."""

    def test_cache_deduplicates(self):
        """Two predicates with the same behavior should share a fingerprint."""
        # Both always return True
        def pred_a(hand):
            return True

        def pred_b(hand):
            return True

        hypotheses = [
            {"predicate": pred_a, "name": "a"},
            {"predicate": pred_b, "name": "b"},
        ]
        exemplars = [_make_all_even_hand()]
        cache = build_extension_cache(hypotheses, exemplar_hands=exemplars)

        # Both should map to the same fingerprint (all 1s), so cache has 1 entry
        assert len(cache) == 1

    def test_cache_separates_different_predicates(self):
        """Predicates with different behavior get different fingerprints."""
        hypotheses = [
            {"predicate": all_even, "name": "all_even"},
            {"predicate": has_ace, "name": "has_ace"},
        ]
        exemplars = [_make_all_even_hand()]
        cache = build_extension_cache(hypotheses, exemplar_hands=exemplars)

        # These predicates differ on most hands, so fingerprints should differ
        assert len(cache) == 2

    def test_cache_values_are_extension_results(self):
        """Each cache value should be a proper ExtensionResult."""
        hypotheses = [{"predicate": constant_true, "name": "true"}]
        exemplars = [_make_all_even_hand()]
        cache = build_extension_cache(hypotheses, exemplar_hands=exemplars)

        for fp, result in cache.items():
            assert isinstance(fp, str)
            assert isinstance(result, ExtensionResult)
            assert result.base_rate > 0
            assert result.n_samples > 0
