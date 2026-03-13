"""Tests for LLM translation verification."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from rules.cards import Card, Suit, Rank


class TestVerifyTranslation:
    def test_matching_translation_returns_no_disagreements(self):
        """Identical predicates should produce zero disagreements."""
        from gallery_analysis.translate_hypotheses import verify_translation

        python_fn = lambda hand: True
        dsl_fn = lambda hand: True

        disagreements = verify_translation(python_fn, dsl_fn, n_test=100, seed=99)
        assert len(disagreements) == 0

    def test_mismatched_translation_returns_disagreements(self):
        """Different predicates should return disagreeing hands."""
        from gallery_analysis.translate_hypotheses import verify_translation

        python_fn = lambda hand: True
        dsl_fn = lambda hand: False

        disagreements = verify_translation(python_fn, dsl_fn, n_test=100, seed=99)
        assert len(disagreements) == 100

    def test_partial_mismatch(self):
        """Predicates that differ on some hands should return those hands."""
        from gallery_analysis.translate_hypotheses import verify_translation

        python_fn = lambda hand: hand[0].suit == Suit.HEARTS
        dsl_fn = lambda hand: hand[0].suit == Suit.SPADES

        disagreements = verify_translation(python_fn, dsl_fn, n_test=500, seed=99)
        assert 0 < len(disagreements) < 500
