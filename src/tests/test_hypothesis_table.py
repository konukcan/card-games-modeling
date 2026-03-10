"""Tests for hypothesis table, fingerprinting, and equivalence classes."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Card, Suit, Rank, H, D, S, C

def _make_probe_set():
    """Small probe set for testing."""
    from gallery_analysis.exemplars import generate_probe_set
    return generate_probe_set(n_probes=50, seed=99)

def test_fingerprint_identical_functions():
    """Two identical predicates should produce the same fingerprint."""
    from gallery_analysis.hypothesis_table import compute_fingerprint
    probes = _make_probe_set()
    pred_a = lambda h: all(c.suit == Suit.SPADES for c in h)
    pred_b = lambda h: len([c for c in h if c.suit != Suit.SPADES]) == 0
    assert compute_fingerprint(pred_a, probes) == compute_fingerprint(pred_b, probes)

def test_fingerprint_different_functions():
    """Two different predicates should produce different fingerprints."""
    from gallery_analysis.hypothesis_table import compute_fingerprint
    probes = _make_probe_set()
    # Use predicates that produce different True/False patterns on random hands
    pred_a = lambda h: h[0].suit == Suit.SPADES  # ~25% True
    pred_b = lambda h: h[0].suit == Suit.HEARTS  # ~25% True, but different hands
    assert compute_fingerprint(pred_a, probes) != compute_fingerprint(pred_b, probes)

def test_hypothesis_table_deduplication():
    """Adding equivalent hypotheses should result in one equivalence class."""
    from gallery_analysis.hypothesis_table import HypothesisTable
    probes = _make_probe_set()
    table = HypothesisTable(probes)

    pred_a = lambda h: all(c.suit == Suit.SPADES for c in h)
    pred_b = lambda h: len([c for c in h if c.suit != Suit.SPADES]) == 0

    table.add("prog_a", pred_a, log_prior=-5.0)
    table.add("prog_b", pred_b, log_prior=-7.0)

    classes = table.get_equivalence_classes()
    assert len(classes) == 1
    # Canonical should be the one with higher prior (less negative log)
    assert classes[0]["canonical_program"] == "prog_a"
    assert classes[0]["n_expressions"] == 2

def test_hit_vector_computation():
    """Hit vector should correctly track which exemplars a hypothesis covers."""
    from gallery_analysis.hypothesis_table import HypothesisTable
    probes = _make_probe_set()
    table = HypothesisTable(probes)

    # A predicate that checks if first card is a heart
    pred = lambda h: h[0].suit == Suit.HEARTS

    exemplars = [
        [H("A"), S("K"), D("Q"), C("J"), H("10"), S("9")],  # first is heart -> True
        [S("A"), H("K"), D("Q"), C("J"), H("10"), S("9")],  # first is spade -> False
    ]

    table.add("first_heart", pred, log_prior=-5.0, exemplar_hands=exemplars)
    classes = table.get_equivalence_classes()
    assert classes[0]["hit_vector"] == [True, False]
    assert classes[0]["n_hits"] == 1
