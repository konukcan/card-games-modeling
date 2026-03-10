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


# =========================================================================
# Trivial filter tests
# =========================================================================

def _make_diverse_hands():
    """Create a small set of diverse hands for trivial filter testing."""
    return [
        [H("A"), H("K"), D("Q"), D("J"), H("10"), D("9")],  # all red
        [S("A"), S("K"), S("Q"), S("J"), S("10"), S("9")],   # all spades
        [H("2"), S("3"), D("4"), C("5"), H("6"), S("7")],    # mixed everything
        [C("A"), C("K"), C("Q"), C("J"), C("10"), C("9")],   # all clubs
    ]

def test_is_trivial_always_true():
    """A predicate that's True on all hands is trivial."""
    from gallery_analysis.hypothesis_table import is_trivial
    hands = _make_diverse_hands()
    pred = lambda h: True
    assert is_trivial(pred, hands) is True

def test_is_trivial_always_false():
    """A predicate that's False on all hands is trivial."""
    from gallery_analysis.hypothesis_table import is_trivial
    hands = _make_diverse_hands()
    pred = lambda h: False
    assert is_trivial(pred, hands) is True

def test_is_trivial_non_trivial():
    """A predicate that varies across hands is non-trivial."""
    from gallery_analysis.hypothesis_table import is_trivial
    hands = _make_diverse_hands()
    # "all red" — True on first hand, False on second (spades)
    pred = lambda h: all(h[i].suit in {Suit.HEARTS, Suit.DIAMONDS} for i in range(len(h)))
    assert is_trivial(pred, hands) is False

def test_is_trivial_rare_but_meaningful():
    """A rare predicate that fires on at least one curated hand is non-trivial."""
    from gallery_analysis.hypothesis_table import is_trivial
    hands = _make_diverse_hands()
    # "all spades" — False on most, True on the second hand
    pred = lambda h: all(c.suit == Suit.SPADES for c in h)
    assert is_trivial(pred, hands) is False

def test_filter_trivial_removes_constants():
    """filter_trivial should remove always-true and always-false programs."""
    from gallery_analysis.hypothesis_table import filter_trivial
    hands = _make_diverse_hands()

    programs = [
        ("true", lambda h: True, -1.0),
        ("false", lambda h: False, -1.0),
        ("all_red", lambda h: all(c.suit in {Suit.HEARTS, Suit.DIAMONDS} for c in h), -5.0),
        ("all_spades", lambda h: all(c.suit == Suit.SPADES for c in h), -8.0),
    ]

    survivors, stats = filter_trivial(programs, hands)
    assert stats["trivial_true"] == 1
    assert stats["trivial_false"] == 1
    assert stats["survivors"] == 2
    survivor_names = [s[0] for s in survivors]
    assert "all_red" in survivor_names
    assert "all_spades" in survivor_names


# =========================================================================
# Syntactic filter tests
# =========================================================================

def test_syntactic_filter_rejects_reverse_reverse():
    """reverse(reverse(X)) = X, should be rejected."""
    from gallery_analysis.enumerator import is_syntactically_redundant
    assert is_syntactically_redundant("(λ all ((λ eq (get_suit $0) HEARTS)) (reverse (reverse $0)))") is True

def test_syntactic_filter_rejects_unique_unique():
    """unique(unique(X)) = unique(X), should be rejected."""
    from gallery_analysis.enumerator import is_syntactically_redundant
    assert is_syntactically_redundant("(λ all ((λ eq (get_suit $0) HEARTS)) (unique (unique $0)))") is True

def test_syntactic_filter_rejects_double_negation():
    """not(not(X)) = X, should be rejected."""
    from gallery_analysis.enumerator import is_syntactically_redundant
    # Actual enumerator format: "not (not " without leading paren on outer not
    assert is_syntactically_redundant("(λ not (not true))") is True
    assert is_syntactically_redundant("(λ not (not (has_suit $0 HEARTS)))") is True

def test_syntactic_filter_rejects_take_0():
    """take 0 X = [], should be rejected."""
    from gallery_analysis.enumerator import is_syntactically_redundant
    assert is_syntactically_redundant("(λ has_color (take 0 $0) RED)") is True

def test_syntactic_filter_rejects_const_arithmetic():
    """(lt (+ 2 3) 4) has no hand dependence, should be rejected."""
    from gallery_analysis.enumerator import is_syntactically_redundant
    # Actual format: "lt (+ 0 0) 0" — curried application, no extra parens
    assert is_syntactically_redundant("(λ lt (+ 2 3) 4)") is True
    assert is_syntactically_redundant("(λ lt (+ 0 0) 0)") is True

def test_syntactic_filter_rejects_const_vs_const():
    """(lt 3 2) has no hand dependence, should be rejected."""
    from gallery_analysis.enumerator import is_syntactically_redundant
    # Actual format: "lt 3 2)" — curried application
    assert is_syntactically_redundant("(λ lt 3 2)") is True
    assert is_syntactically_redundant("(λ lt 0 0)") is True
    assert is_syntactically_redundant("(λ eq 1 1)") is True

def test_syntactic_filter_accepts_meaningful_program():
    """A meaningful program should NOT be rejected."""
    from gallery_analysis.enumerator import is_syntactically_redundant
    assert is_syntactically_redundant("(λ eq (n_unique_suits $0) 1)") is False
    assert is_syntactically_redundant("(λ all ((λ eq (get_suit $0) HEARTS)) $0)") is False
    assert is_syntactically_redundant("(λ lt (count_suit $0 HEARTS) 3)") is False

def test_gallery_grammar_has_no_bool_constants():
    """Gallery grammar should exclude true and false."""
    from gallery_analysis.enumerator import build_gallery_primitives
    prims = build_gallery_primitives()
    names = {p.name for p in prims}
    assert "true" not in names
    assert "false" not in names
    # But other primitives should still be present
    assert "and" in names
    assert "or" in names
    assert "not" in names
    assert "eq" in names
