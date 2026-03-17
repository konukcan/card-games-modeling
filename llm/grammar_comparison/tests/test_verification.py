"""
Tests for fingerprint verification of translation pipeline.

Verifies that:
    - Probe hands load correctly (200 hands of Card objects).
    - AST fingerprints produce expected boolean tuples.
    - Python fingerprints produce expected boolean tuples.
    - Dual-path verification agrees for correct translations.
    - Error handling works for malformed inputs.

Written TDD-style: each test documents expected behaviour.
"""

import sys
from pathlib import Path

# Allow importing from src/ and llm/
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import pytest
from rules.cards import Card, Suit, Rank, RANK_VALUES, Hand, H, D, S, C

from llm.grammar_comparison.translation.verification import (
    load_probe_hands,
    compute_ast_fingerprint,
    compute_python_fingerprint,
    verify_dual_path,
    verify_rewrite_preserves_semantics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def probes() -> list:
    """Load the 200 probe hands once for all tests in this module."""
    return load_probe_hands()


@pytest.fixture
def small_probes() -> list:
    """A small set of hand-crafted probe hands for deterministic tests."""
    return [
        # Hand 1: all hearts, low ranks
        [H("2"), H("3"), H("4"), H("5"), H("6"), H("7")],
        # Hand 2: all spades, high ranks
        [S("9"), S("10"), S("J"), S("Q"), S("K"), S("A")],
        # Hand 3: mixed suits and ranks
        [H("A"), D("K"), S("Q"), C("J"), H("10"), D("9")],
        # Hand 4: all same rank (pair-heavy)
        [H("5"), D("5"), S("5"), C("5"), H("6"), D("6")],
    ]


# ---------------------------------------------------------------------------
# 1. load_probe_hands
# ---------------------------------------------------------------------------

class TestLoadProbeHands:
    """Tests for loading and parsing the probe hand set."""

    def test_returns_200_hands(self, probes):
        """The probe set should contain exactly 200 hands."""
        assert len(probes) == 200

    def test_each_hand_has_6_cards(self, probes):
        """Each hand should have 6 cards (the standard hand size)."""
        for i, hand in enumerate(probes):
            assert len(hand) == 6, f"Hand {i} has {len(hand)} cards, expected 6"

    def test_cards_are_card_objects(self, probes):
        """Every element in every hand should be a Card instance."""
        for hand in probes:
            for card in hand:
                assert isinstance(card, Card), f"Expected Card, got {type(card)}"

    def test_cards_have_valid_suits(self, probes):
        """All cards should have valid Suit enum values."""
        for hand in probes:
            for card in hand:
                assert isinstance(card.suit, Suit)

    def test_cards_have_valid_ranks(self, probes):
        """All cards should have valid Rank enum values."""
        for hand in probes:
            for card in hand:
                assert isinstance(card.rank, Rank)

    def test_file_not_found_raises(self, tmp_path):
        """Loading from a nonexistent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_probe_hands(tmp_path / "nonexistent.json")


# ---------------------------------------------------------------------------
# 2. compute_ast_fingerprint
# ---------------------------------------------------------------------------

class TestComputeAstFingerprint:
    """Tests for computing fingerprints from Program ASTs."""

    def test_constant_true_program(self, small_probes):
        """A program that always returns True should give all-True fingerprint."""
        from llm.grammar_comparison.translation.sexpr_parser import (
            parse_hypothesis_sexpr,
        )
        # (λ true) — ignores hand, returns True
        program = parse_hypothesis_sexpr("(λ true)")
        fp = compute_ast_fingerprint(program, small_probes)
        assert len(fp) == len(small_probes)
        assert all(v is True for v in fp)

    def test_constant_false_program(self, small_probes):
        """A program that always returns False should give all-False fingerprint."""
        from llm.grammar_comparison.translation.sexpr_parser import (
            parse_hypothesis_sexpr,
        )
        program = parse_hypothesis_sexpr("(λ false)")
        fp = compute_ast_fingerprint(program, small_probes)
        assert len(fp) == len(small_probes)
        assert all(v is False for v in fp)

    def test_fingerprint_length_matches_probes(self, probes):
        """Fingerprint length should equal the number of probe hands."""
        from llm.grammar_comparison.translation.sexpr_parser import (
            parse_hypothesis_sexpr,
        )
        program = parse_hypothesis_sexpr("(λ true)")
        fp = compute_ast_fingerprint(program, probes)
        assert len(fp) == 200

    def test_all_even_rule(self, small_probes):
        """Test a rule checking if all cards have even rank values.

        (λ all (λ eq (mod (rank_val $0) 2) 0) $0)
        This checks: all cards in hand have rank_val % 2 == 0.

        Hand 1: [2,3,4,5,6,7] -> 3,5,7 are odd -> False
        Hand 2: [9,10,J,Q,K,A] -> 9,11,13 are odd -> False
        Hand 3: [A,K,Q,J,10,9] -> odd ranks present -> False
        Hand 4: [5,5,5,5,6,6] -> 5 is odd -> False
        """
        from llm.grammar_comparison.translation.sexpr_parser import (
            parse_hypothesis_sexpr,
        )
        program = parse_hypothesis_sexpr(
            "(λ all (λ eq (mod (rank_val $0) 2) 0) $0)"
        )
        fp = compute_ast_fingerprint(program, small_probes)
        assert len(fp) == 4
        # All hands contain odd-valued cards, so all should be False
        assert all(v is False for v in fp)


# ---------------------------------------------------------------------------
# 3. compute_python_fingerprint
# ---------------------------------------------------------------------------

class TestComputePythonFingerprint:
    """Tests for computing fingerprints from Python code strings."""

    def test_constant_true(self, small_probes):
        """Python rule returning True for all hands."""
        code = "rule = lambda hand: True"
        fp = compute_python_fingerprint(code, small_probes)
        assert len(fp) == len(small_probes)
        assert all(v is True for v in fp)

    def test_constant_false(self, small_probes):
        """Python rule returning False for all hands."""
        code = "rule = lambda hand: False"
        fp = compute_python_fingerprint(code, small_probes)
        assert all(v is False for v in fp)

    def test_def_form(self, small_probes):
        """Python rule using def syntax (not lambda)."""
        code = """
def rule(hand):
    return True
"""
        fp = compute_python_fingerprint(code, small_probes)
        assert all(v is True for v in fp)

    def test_all_hearts_rule(self, small_probes):
        """Python rule checking if all cards are hearts.

        Hand 1: all hearts -> True
        Hand 2: all spades -> False
        Hand 3: mixed -> False
        Hand 4: mixed -> False
        """
        code = "rule = lambda hand: all(c.suit == Suit.HEARTS for c in hand)"
        fp = compute_python_fingerprint(code, small_probes)
        assert fp == (True, False, False, False)

    def test_missing_rule_raises(self, small_probes):
        """Code that doesn't define 'rule' should raise ValueError."""
        with pytest.raises(ValueError, match="must define 'rule'"):
            compute_python_fingerprint("x = 42", small_probes)

    def test_syntax_error_raises(self, small_probes):
        """Code with syntax errors should raise ValueError."""
        with pytest.raises(ValueError, match="Failed to compile"):
            compute_python_fingerprint("def rule(hand: !!!!", small_probes)

    def test_runtime_error_gives_none(self, small_probes):
        """A rule that crashes on a specific hand should produce None for that hand."""
        # Division by zero on every hand
        code = "rule = lambda hand: 1 / 0"
        fp = compute_python_fingerprint(code, small_probes)
        assert all(v is None for v in fp)

    def test_uses_rank_values(self, small_probes):
        """Python code should have access to RANK_VALUES from namespace."""
        code = "rule = lambda hand: all(RANK_VALUES[c.rank] >= 9 for c in hand)"
        fp = compute_python_fingerprint(code, small_probes)
        # Hand 1: ranks 2-7, all < 9 -> False
        # Hand 2: ranks 9,10,11,12,13,14, all >= 9 -> True
        # Hand 3: 14,13,12,11,10,9, all >= 9 -> True
        # Hand 4: 5,5,5,5,6,6 -> False
        assert fp == (False, True, True, False)


# ---------------------------------------------------------------------------
# 4. verify_dual_path
# ---------------------------------------------------------------------------

class TestVerifyDualPath:
    """Tests for dual-path (AST vs Python) fingerprint comparison."""

    def test_matching_constant_true(self, small_probes):
        """Both paths produce True for all hands -> match."""
        sexpr = "(λ true)"
        python_code = "rule = lambda hand: True"
        match, details = verify_dual_path(sexpr, python_code, small_probes)
        assert match is True
        assert "mismatches" not in details

    def test_matching_all_hearts(self, small_probes):
        """Both paths check 'all hearts' -> should agree."""
        sexpr = "(λ all (λ eq (get_suit $0) HEARTS) $0)"
        python_code = "rule = lambda hand: all(c.suit == Suit.HEARTS for c in hand)"
        match, details = verify_dual_path(sexpr, python_code, small_probes)
        assert match is True

    def test_mismatch_detected(self, small_probes):
        """When s-expr and Python implement different rules, mismatch is caught."""
        sexpr = "(λ true)"  # always True
        python_code = "rule = lambda hand: False"  # always False
        match, details = verify_dual_path(sexpr, python_code, small_probes)
        assert match is False
        assert "mismatches" in details
        assert details["n_mismatches"] == len(small_probes)

    def test_invalid_sexpr_returns_error(self, small_probes):
        """An unparseable s-expression should return match=False with error."""
        match, details = verify_dual_path(
            "(λ unknown_primitive $0)",
            "rule = lambda hand: True",
            small_probes,
        )
        assert match is False
        assert "error" in details

    def test_invalid_python_returns_error(self, small_probes):
        """Invalid Python code should return match=False with error."""
        match, details = verify_dual_path(
            "(λ true)",
            "not valid python!!!",
            small_probes,
        )
        assert match is False
        assert "error" in details


# ---------------------------------------------------------------------------
# 5. verify_rewrite_preserves_semantics
# ---------------------------------------------------------------------------

class TestVerifyRewritePreservesSemantics:
    """Tests for rewrite verification (graceful when rewriter not available)."""

    def test_missing_rewriter_handled_gracefully(self, small_probes):
        """When the rewriter module doesn't exist, should return False + error."""
        match, details = verify_rewrite_preserves_semantics(
            "(λ true)", "some_grammar", small_probes
        )
        # If rewriter is not available, we expect a graceful failure
        # (not a crash)
        assert isinstance(match, bool)
        assert isinstance(details, dict)
        if not match:
            assert "error" in details

    def test_invalid_sexpr_handled(self, small_probes):
        """Bad s-expression should return match=False with error."""
        match, details = verify_rewrite_preserves_semantics(
            "(λ nonexistent_prim $0)", "some_grammar", small_probes
        )
        assert match is False
        assert "error" in details


# ---------------------------------------------------------------------------
# 6. Integration: full probe set
# ---------------------------------------------------------------------------

class TestIntegrationWithFullProbes:
    """Integration tests using the actual 200-hand probe set."""

    def test_dual_path_agrees_on_full_probes(self, probes):
        """Dual-path should agree for 'all hearts' on the full 200-hand probe set."""
        sexpr = "(λ all (λ eq (get_suit $0) HEARTS) $0)"
        python_code = "rule = lambda hand: all(c.suit == Suit.HEARTS for c in hand)"
        match, details = verify_dual_path(sexpr, python_code, probes)
        assert match is True

    def test_fingerprint_is_not_trivially_all_same(self, probes):
        """A non-trivial rule should produce a mix of True and False on 200 hands."""
        from llm.grammar_comparison.translation.sexpr_parser import (
            parse_hypothesis_sexpr,
        )
        # "any card has rank_val equal to 2" — should produce a mix
        # (only integers 0-5 are in the primitive registry)
        program = parse_hypothesis_sexpr(
            "(λ any (λ eq (rank_val $0) 2) $0)"
        )
        fp = compute_ast_fingerprint(program, probes)
        n_true = sum(1 for v in fp if v is True)
        n_false = sum(1 for v in fp if v is False)
        # With 200 random hands of 6 cards, both True and False should appear
        assert n_true > 0, "Expected at least some True values"
        assert n_false > 0, "Expected at least some False values"
