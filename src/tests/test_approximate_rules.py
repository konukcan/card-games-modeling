"""Tests for approximate true rule flagging."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


class TestApproximateRuleFlag:
    def test_score_rule_includes_approximate_flag(self):
        """score_rule should return true_rule_approximate when source is approximate."""
        from gallery_analysis.analyze import score_rule
        from rules.cards import Card, Suit, Rank

        # Minimal equiv class marked as approximate true rule
        hand = [Card(Suit.HEARTS, r) for r in [Rank.TWO, Rank.THREE, Rank.FOUR,
                                                 Rank.FIVE, Rank.SIX, Rank.SEVEN]]
        equiv = [{
            "canonical_program": "(λ true)",
            "canonical_prior": -1.0,
            "summed_prior": -0.5,
            "n_expressions": 1,
            "all_programs": ["(λ true)"],
            "fingerprint": "fp_approx",
            "predicate": lambda h: True,
            "source": "true_rule_approximate",
        }]
        extensions = [(10_000_000, 0.49)]

        result = score_rule(
            "test_rule", [hand], equiv, extensions,
            true_rule_fingerprint="fp_approx",
        )

        assert result["true_rule_approximate"] is True

    def test_exact_true_rule_not_flagged(self):
        """Exact true rules should have true_rule_approximate=False."""
        from gallery_analysis.analyze import score_rule
        from rules.cards import Card, Suit, Rank

        hand = [Card(Suit.HEARTS, r) for r in [Rank.TWO, Rank.THREE, Rank.FOUR,
                                                 Rank.FIVE, Rank.SIX, Rank.SEVEN]]
        equiv = [{
            "canonical_program": "(λ true)",
            "canonical_prior": -1.0,
            "summed_prior": -0.5,
            "n_expressions": 1,
            "all_programs": ["(λ true)"],
            "fingerprint": "fp_exact",
            "predicate": lambda h: True,
            "source": "merged",
        }]
        extensions = [(10_000_000, 0.49)]

        result = score_rule(
            "test_rule", [hand], equiv, extensions,
            true_rule_fingerprint="fp_exact",
        )

        assert result["true_rule_approximate"] is False

    def test_no_true_rule_gives_none(self):
        """When no true rule fingerprint provided, flag should be None."""
        from gallery_analysis.analyze import score_rule
        from rules.cards import Card, Suit, Rank

        hand = [Card(Suit.HEARTS, r) for r in [Rank.TWO, Rank.THREE, Rank.FOUR,
                                                 Rank.FIVE, Rank.SIX, Rank.SEVEN]]
        equiv = [{
            "canonical_program": "(λ true)",
            "canonical_prior": -1.0,
            "summed_prior": -0.5,
            "n_expressions": 1,
            "all_programs": ["(λ true)"],
            "fingerprint": "fp_x",
            "predicate": lambda h: True,
        }]
        extensions = [(10_000_000, 0.49)]

        result = score_rule("test_rule", [hand], equiv, extensions)

        assert result["true_rule_approximate"] is None
