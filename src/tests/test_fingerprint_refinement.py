"""Tests for rare fingerprint refinement."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from rules.cards import Card, Suit, Rank
from gallery_analysis.exemplars import generate_probe_set


def _make_hand(*specs):
    return [Card(s, r) for s, r in specs]


class TestRefineRareClasses:
    def test_splits_colliding_rare_predicates(self):
        """Two rare predicates that collide on 10 probes should be split
        when refinement probes distinguish them."""
        from gallery_analysis.hypothesis_table import (
            compute_fingerprint, refine_rare_classes,
        )

        # Create two predicates that are both False on all normal probes
        # but differ on specific hands
        pred_a = lambda h: (h[0].suit == Suit.SPADES and h[0].rank == Rank.ACE
                            and h[1].suit == Suit.SPADES and h[1].rank == Rank.KING)
        pred_b = lambda h: (h[0].suit == Suit.HEARTS and h[0].rank == Rank.ACE
                            and h[1].suit == Suit.HEARTS and h[1].rank == Rank.KING)

        # Use very few probes so they collide
        small_probes = generate_probe_set(10, seed=42)
        fp_a = compute_fingerprint(pred_a, small_probes)
        fp_b = compute_fingerprint(pred_b, small_probes)

        # They should collide on 10 probes (both all-False)
        assert fp_a == fp_b, "Test setup: predicates should collide on 10 probes"

        # Create a fake equivalence class with both predicates merged
        equiv_classes = [{
            "canonical_program": "pred_a",
            "canonical_prior": -5.0,
            "summed_prior": -4.3,
            "n_expressions": 2,
            "all_programs": ["pred_a", "pred_b"],
            "fingerprint": fp_a,
            "predicate": pred_a,
            "_all_predicates": [pred_a, pred_b],
            "_all_priors": [-5.0, -5.0],
        }]

        refined = refine_rare_classes(
            equiv_classes, small_probes,
            hit_threshold=5,
            n_refinement_probes=2000,
            refinement_seed=4242,
        )

        # After refinement, should have 2 classes (split)
        assert len(refined) == 2

    def test_non_rare_classes_unchanged(self):
        """Classes that hit many probes should not be touched."""
        from gallery_analysis.hypothesis_table import refine_rare_classes

        probes = generate_probe_set(100, seed=42)

        equiv_classes = [{
            "canonical_program": "(λ true)",
            "canonical_prior": -1.0,
            "summed_prior": -0.5,
            "n_expressions": 1,
            "all_programs": ["(λ true)"],
            "fingerprint": "fp_common",
            "predicate": lambda h: True,
        }]

        refined = refine_rare_classes(
            equiv_classes, probes,
            hit_threshold=5,
            n_refinement_probes=2000,
            refinement_seed=4242,
        )

        assert len(refined) == 1
        assert refined[0]["fingerprint"] == "fp_common"

    def test_single_program_class_unchanged(self):
        """Rare classes with only 1 program can't be split."""
        from gallery_analysis.hypothesis_table import (
            compute_fingerprint, refine_rare_classes,
        )

        probes = generate_probe_set(100, seed=42)
        pred = lambda h: False
        fp = compute_fingerprint(pred, probes)

        equiv_classes = [{
            "canonical_program": "(λ false)",
            "canonical_prior": -2.0,
            "summed_prior": -2.0,
            "n_expressions": 1,
            "all_programs": ["(λ false)"],
            "fingerprint": fp,
            "predicate": pred,
        }]

        refined = refine_rare_classes(
            equiv_classes, probes,
            hit_threshold=5,
            n_refinement_probes=2000,
            refinement_seed=4242,
        )

        assert len(refined) == 1
