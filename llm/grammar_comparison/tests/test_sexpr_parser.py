"""
Tests for the Phase 1b s-expression parser.

Verifies that hypothesis s-expressions from injected_hypotheses.json can be
parsed into Program AST objects.

Written TDD-style: tests define expected behaviour and serve as documentation
of what the parser must handle.
"""

import sys
from pathlib import Path

# Allow importing from the main src/ tree
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import pytest
from dreamcoder_core.program import Abstraction, Application, Primitive, Index

from llm.grammar_comparison.translation.sexpr_parser import (
    get_primitive_registry,
    parse_hypothesis_sexpr,
)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestPrimitiveRegistry:
    """Verify that the registry contains all names used in Phase 1b hypotheses."""

    def test_registry_has_suit_constants(self):
        reg = get_primitive_registry()
        for name in ("CLUBS", "DIAMONDS", "HEARTS", "SPADES"):
            assert name in reg, f"Missing suit constant: {name}"

    def test_registry_has_color_constants(self):
        reg = get_primitive_registry()
        for name in ("RED", "BLACK"):
            assert name in reg, f"Missing color constant: {name}"

    def test_registry_has_integers_0_to_5(self):
        reg = get_primitive_registry()
        for i in range(6):
            assert str(i) in reg, f"Missing integer constant: {i}"

    def test_registry_has_booleans(self):
        reg = get_primitive_registry()
        assert "true" in reg
        assert "false" in reg

    def test_registry_has_card_accessors(self):
        reg = get_primitive_registry()
        for name in ("get_suit", "get_rank", "rank_val", "get_color"):
            assert name in reg, f"Missing card accessor: {name}"

    def test_registry_has_comparisons(self):
        reg = get_primitive_registry()
        for name in ("eq", "gt", "lt", "le", "ge"):
            assert name in reg, f"Missing comparison: {name}"

    def test_registry_has_boolean_ops(self):
        reg = get_primitive_registry()
        for name in ("not", "and", "or"):
            assert name in reg, f"Missing boolean op: {name}"

    def test_registry_has_higher_order(self):
        reg = get_primitive_registry()
        for name in ("all", "any", "map", "filter", "unique"):
            assert name in reg, f"Missing HOF: {name}"

    def test_registry_has_arithmetic(self):
        reg = get_primitive_registry()
        for name in ("+", "-", "mod", "*"):
            assert name in reg, f"Missing arithmetic op: {name}"

    def test_registry_has_list_ops(self):
        reg = get_primitive_registry()
        for name in ("head", "last", "at", "take", "drop", "length",
                      "reverse", "first_half", "second_half",
                      "adjacent_pairs", "zip_with"):
            assert name in reg, f"Missing list op: {name}"

    def test_registry_has_direct_queries(self):
        reg = get_primitive_registry()
        for name in ("count_suit", "count_color",
                      "n_unique_suits", "n_unique_ranks", "n_unique_colors"):
            assert name in reg, f"Missing direct query: {name}"

    def test_registry_has_aggregates_and_aliases(self):
        """The LLM uses sum_vals/max_val/min_val; DSL has sum_ranks/max_rank/min_rank."""
        reg = get_primitive_registry()
        # Canonical names
        for name in ("sum_ranks", "max_rank", "min_rank"):
            assert name in reg, f"Missing aggregate: {name}"
        # Aliases used by the LLM
        for name in ("sum_vals", "max_val", "min_val"):
            assert name in reg, f"Missing aggregate alias: {name}"

    def test_registry_has_n_cards_alias(self):
        """The LLM uses 'n_cards'; DSL has 'length'."""
        reg = get_primitive_registry()
        assert "n_cards" in reg
        # Should point to the same Primitive as 'length'
        assert reg["n_cards"] is reg["length"]

    def test_registry_at_least_64_entries(self):
        """The base grammar has 64 primitives; with aliases we should have more."""
        reg = get_primitive_registry()
        assert len(reg) >= 64


# ---------------------------------------------------------------------------
# Simple parsing tests
# ---------------------------------------------------------------------------

class TestSimpleParsing:
    """Test parsing of simple, common hypothesis patterns."""

    def test_all_suits_clubs(self):
        """All cards are clubs: (λ all (λ eq (get_suit $0) CLUBS) $0)"""
        prog = parse_hypothesis_sexpr("(λ all (λ eq (get_suit $0) CLUBS) $0)")
        assert isinstance(prog, Abstraction)
        assert prog.size() > 3

    def test_parity_check(self):
        """All rank values are even: (λ all (λ eq (mod (rank_val $0) 2) 0) $0)"""
        prog = parse_hypothesis_sexpr(
            "(λ all (λ eq (mod (rank_val $0) 2) 0) $0)"
        )
        assert isinstance(prog, Abstraction)

    def test_first_card_red(self):
        """First card is red: (λ eq RED (get_color (head $0)))"""
        prog = parse_hypothesis_sexpr("(λ eq RED (get_color (head $0)))")
        assert isinstance(prog, Abstraction)

    def test_unique_suits_check(self):
        """Exactly 2 unique suits: (λ eq 2 (n_unique_suits $0))"""
        prog = parse_hypothesis_sexpr("(λ eq 2 (n_unique_suits $0))")
        assert isinstance(prog, Abstraction)
        assert prog.size() > 3

    def test_boolean_conjunction(self):
        """Conjunction: (λ and (gt (n_unique_suits $0) 1) (lt (n_unique_ranks $0) 4))"""
        prog = parse_hypothesis_sexpr(
            "(λ and (gt (n_unique_suits $0) 1) (lt (n_unique_ranks $0) 4))"
        )
        assert isinstance(prog, Abstraction)


# ---------------------------------------------------------------------------
# Complex / nested parsing tests
# ---------------------------------------------------------------------------

class TestComplexParsing:
    """Test parsing of more complex hypothesis patterns with nesting."""

    def test_nested_lambda_with_filter(self):
        """Filter + nested lambda: (λ gt (length (filter (λ eq (get_color $0) RED) $0)) 2)"""
        prog = parse_hypothesis_sexpr(
            "(λ gt (length (filter (λ eq (get_color $0) RED) $0)) 2)"
        )
        assert isinstance(prog, Abstraction)
        assert prog.size() > 5

    def test_adjacent_pairs(self):
        """Using adjacent_pairs: (λ all (λ lt (rank_val (head $0)) (rank_val (last $0))) (adjacent_pairs $0))"""
        prog = parse_hypothesis_sexpr(
            "(λ all (λ lt (rank_val (head $0)) (rank_val (last $0))) (adjacent_pairs $0))"
        )
        assert isinstance(prog, Abstraction)

    def test_map_with_lambda(self):
        """Map over hand: (λ all (λ eq $0 0) (map (λ mod (rank_val $0) 2) $0))"""
        prog = parse_hypothesis_sexpr(
            "(λ all (λ eq $0 0) (map (λ mod (rank_val $0) 2) $0))"
        )
        assert isinstance(prog, Abstraction)

    def test_zip_with(self):
        """Zip_with for halves comparison: (λ all (λ $0) (zip_with eq (map get_suit (first_half $0)) (map get_suit (second_half $0))))"""
        prog = parse_hypothesis_sexpr(
            "(λ all (λ $0) (zip_with eq (map get_suit (first_half $0)) (map get_suit (second_half $0))))"
        )
        assert isinstance(prog, Abstraction)

    def test_deeply_nested(self):
        """Deeply nested: (λ and (all (λ eq (mod (rank_val $0) 2) 0) $0) (gt (n_unique_suits $0) 2))"""
        prog = parse_hypothesis_sexpr(
            "(λ and (all (λ eq (mod (rank_val $0) 2) 0) $0) (gt (n_unique_suits $0) 2))"
        )
        assert isinstance(prog, Abstraction)


# ---------------------------------------------------------------------------
# Alias resolution tests
# ---------------------------------------------------------------------------

class TestAliasResolution:
    """Test that LLM-specific names are resolved via aliases."""

    def test_sum_vals_alias(self):
        """sum_vals should parse as sum_ranks."""
        prog = parse_hypothesis_sexpr("(λ gt (sum_vals $0) 3)")
        assert isinstance(prog, Abstraction)

    def test_max_val_alias(self):
        """max_val should parse as max_rank."""
        prog = parse_hypothesis_sexpr("(λ gt (max_val $0) 3)")
        assert isinstance(prog, Abstraction)

    def test_min_val_alias(self):
        """min_val should parse as min_rank."""
        prog = parse_hypothesis_sexpr("(λ lt (min_val $0) 3)")
        assert isinstance(prog, Abstraction)

    def test_n_cards_alias(self):
        """n_cards should parse as length."""
        prog = parse_hypothesis_sexpr("(λ eq (n_cards $0) 5)")
        assert isinstance(prog, Abstraction)

    def test_multiply(self):
        """Multiplication (*) should be available."""
        prog = parse_hypothesis_sexpr("(λ eq (* 2 3) (n_cards $0))")
        assert isinstance(prog, Abstraction)


# ---------------------------------------------------------------------------
# Edge cases and error handling
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_bare_lambda_identity(self):
        """Minimal hypothesis: (λ $0)"""
        prog = parse_hypothesis_sexpr("(λ $0)")
        assert isinstance(prog, Abstraction)
        assert isinstance(prog.body, Index)

    def test_whitespace_handling(self):
        """Extra whitespace should be tolerated."""
        prog = parse_hypothesis_sexpr("  (λ  all (λ eq (get_suit $0) CLUBS)  $0)  ")
        assert isinstance(prog, Abstraction)

    def test_unrecognised_token_raises_error(self):
        """An unknown primitive name should raise ValueError."""
        with pytest.raises(ValueError, match="Unexpected token"):
            parse_hypothesis_sexpr("(λ nonexistent_primitive $0)")

    def test_empty_string_raises_error(self):
        """Empty input should raise ValueError."""
        with pytest.raises(ValueError):
            parse_hypothesis_sexpr("")

    def test_str_roundtrip_preserves_structure(self):
        """Parsing then stringifying should give a valid representation."""
        sexpr = "(λ all (λ eq (get_suit $0) CLUBS) $0)"
        prog = parse_hypothesis_sexpr(sexpr)
        # The str() output should be parseable again
        reparsed = parse_hypothesis_sexpr(str(prog))
        assert prog == reparsed
