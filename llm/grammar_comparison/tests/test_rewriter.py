"""
Tests for the mechanical AST rewriter.

Verifies that rewrite_ast correctly transforms Base grammar ASTs into
equivalent ASTs using each target grammar's primitives.

Written TDD-style: tests define expected rewriting behavior.
"""

import sys
from pathlib import Path

# Allow importing from the main src/ tree
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import pytest
from dreamcoder_core.program import (
    Abstraction, Application, Primitive, Index,
    collect_primitive_names,
)

from llm.grammar_comparison.translation.sexpr_parser import parse_hypothesis_sexpr
from llm.grammar_comparison.translation.rewriter import (
    rewrite_ast,
    InexpressibleError,
)


# ---------------------------------------------------------------------------
# Helper: parse and rewrite in one step
# ---------------------------------------------------------------------------

def _rewrite(sexpr: str, target: str):
    """Parse a base-grammar s-expression and rewrite to target grammar."""
    prog = parse_hypothesis_sexpr(sexpr)
    return rewrite_ast(prog, target)


# ===========================================================================
# Identity grammars: base, add-both, redundant
# ===========================================================================

class TestIdentityGrammars:
    """base, add-both, and redundant should return the program unchanged."""

    @pytest.mark.parametrize("grammar", ["base", "add-both", "redundant"])
    def test_simple_rule_unchanged(self, grammar):
        sexpr = "(λ all (λ eq (get_suit $0) HEARTS) $0)"
        prog = parse_hypothesis_sexpr(sexpr)
        result = rewrite_ast(prog, grammar)
        # Identity: should be the exact same object
        assert result is prog

    @pytest.mark.parametrize("grammar", ["base", "add-both", "redundant"])
    def test_str_preserved(self, grammar):
        sexpr = "(λ eq (length $0) 3)"
        prog = parse_hypothesis_sexpr(sexpr)
        result = rewrite_ast(prog, grammar)
        assert str(result) == str(prog)


# ===========================================================================
# swap-positional grammar
# ===========================================================================

class TestSwapPositional:
    """Verify positional primitive rewrites to slice / shifted_match."""

    def test_first_half_becomes_slice_0_3(self):
        # (λ all pred (first_half $0))
        # first_half $0 -> slice 0 3 $0
        sexpr = "(λ all (λ eq (get_suit $0) HEARTS) (first_half $0))"
        result = _rewrite(sexpr, "swap-positional")
        result_str = str(result)
        # Should contain 'slice' instead of 'first_half'
        assert "slice" in result_str
        assert "first_half" not in result_str
        # The primitives used should include slice
        prims = collect_primitive_names(result)
        assert "slice" in prims
        assert "first_half" not in prims

    def test_second_half_becomes_slice_3_6(self):
        sexpr = "(λ all (λ eq (get_suit $0) HEARTS) (second_half $0))"
        result = _rewrite(sexpr, "swap-positional")
        result_str = str(result)
        assert "slice" in result_str
        assert "second_half" not in result_str
        prims = collect_primitive_names(result)
        assert "slice" in prims

    def test_take_becomes_slice_0_n(self):
        # take 2 $0 -> slice 0 2 $0
        sexpr = "(λ length (take 2 $0))"
        result = _rewrite(sexpr, "swap-positional")
        result_str = str(result)
        assert "slice" in result_str
        assert "take" not in collect_primitive_names(result)

    def test_drop_becomes_slice_n_6(self):
        # drop 2 $0 -> slice 2 6 $0
        sexpr = "(λ length (drop 2 $0))"
        result = _rewrite(sexpr, "swap-positional")
        result_str = str(result)
        assert "slice" in result_str
        assert "drop" not in collect_primitive_names(result)

    def test_non_positional_prims_unchanged(self):
        # A rule using no positional primitives should be untouched
        sexpr = "(λ eq (length $0) 3)"
        prog = parse_hypothesis_sexpr(sexpr)
        result = rewrite_ast(prog, "swap-positional")
        assert str(result) == str(prog)

    def test_standalone_adjacent_pairs_raises(self):
        # adjacent_pairs outside of all/any -> InexpressibleError
        sexpr = "(λ length (adjacent_pairs $0))"
        with pytest.raises(InexpressibleError, match="adjacent_pairs"):
            _rewrite(sexpr, "swap-positional")


# ===========================================================================
# swap-distributional grammar
# ===========================================================================

class TestSwapDistributional:
    """Verify distributional primitive rewrites to count_where."""

    def test_count_suit_becomes_count_where(self):
        # count_suit $0 HEARTS -> count_where (λ eq (get_suit $0) HEARTS) $0
        sexpr = "(λ eq 2 (count_suit $0 HEARTS))"
        result = _rewrite(sexpr, "swap-distributional")
        result_str = str(result)
        assert "count_where" in result_str
        assert "count_suit" not in result_str
        prims = collect_primitive_names(result)
        assert "count_where" in prims
        assert "count_suit" not in prims

    def test_count_color_becomes_count_where(self):
        # count_color $0 RED -> count_where (λ eq (get_color $0) RED) $0
        sexpr = "(λ eq 2 (count_color $0 RED))"
        result = _rewrite(sexpr, "swap-distributional")
        result_str = str(result)
        assert "count_where" in result_str
        assert "count_color" not in result_str
        prims = collect_primitive_names(result)
        assert "count_where" in prims
        assert "count_color" not in prims

    def test_count_where_body_has_correct_structure(self):
        """The generated lambda should be (λ eq (get_suit $0) HEARTS)."""
        sexpr = "(λ eq 2 (count_suit $0 HEARTS))"
        result = _rewrite(sexpr, "swap-distributional")
        # Find the count_where application and inspect its predicate
        # Result structure: (λ eq 2 (count_where (λ eq (get_suit $0) HEARTS) $0))
        # Let's check the string representation includes the expected pattern
        result_str = str(result)
        assert "get_suit" in result_str
        assert "eq" in result_str

    def test_non_distributional_prims_unchanged(self):
        sexpr = "(λ eq (length $0) 3)"
        prog = parse_hypothesis_sexpr(sexpr)
        result = rewrite_ast(prog, "swap-distributional")
        assert str(result) == str(prog)


# ===========================================================================
# swap-both grammar
# ===========================================================================

class TestSwapBoth:
    """Verify that swap-both applies both positional and distributional rewrites."""

    def test_both_positional_and_distributional(self):
        # A rule using both first_half and count_suit
        # This tests that both rewriters are applied
        sexpr = "(λ eq (count_suit (first_half $0) HEARTS) 2)"
        result = _rewrite(sexpr, "swap-both")
        prims = collect_primitive_names(result)
        # Positional primitives should be gone
        assert "first_half" not in prims
        # Distributional primitives should be gone
        assert "count_suit" not in prims
        # New primitives should be present
        assert "slice" in prims
        assert "count_where" in prims

    def test_only_positional_in_swap_both(self):
        sexpr = "(λ all (λ eq (get_suit $0) HEARTS) (first_half $0))"
        result = _rewrite(sexpr, "swap-both")
        prims = collect_primitive_names(result)
        assert "first_half" not in prims
        assert "slice" in prims

    def test_only_distributional_in_swap_both(self):
        sexpr = "(λ eq 2 (count_suit $0 HEARTS))"
        result = _rewrite(sexpr, "swap-both")
        prims = collect_primitive_names(result)
        assert "count_suit" not in prims
        assert "count_where" in prims


# ===========================================================================
# minimal grammar
# ===========================================================================

class TestMinimal:
    """Verify that minimal raises InexpressibleError for non-minimal primitives."""

    def test_minimal_keeps_basic_rule(self):
        # All primitives in this rule are in _MINIMAL_KEEP
        sexpr = "(λ all (λ eq (get_suit $0) HEARTS) $0)"
        result = _rewrite(sexpr, "minimal")
        assert str(result) == str(parse_hypothesis_sexpr(sexpr))

    def test_minimal_raises_for_first_half(self):
        sexpr = "(λ all (λ eq (get_suit $0) HEARTS) (first_half $0))"
        with pytest.raises(InexpressibleError, match="first_half"):
            _rewrite(sexpr, "minimal")

    def test_minimal_raises_for_count_suit(self):
        sexpr = "(λ eq 2 (count_suit $0 HEARTS))"
        with pytest.raises(InexpressibleError, match="count_suit"):
            _rewrite(sexpr, "minimal")

    def test_minimal_raises_for_has_suit(self):
        sexpr = "(λ has_suit $0 HEARTS)"
        with pytest.raises(InexpressibleError, match="has_suit"):
            _rewrite(sexpr, "minimal")

    def test_minimal_allows_arithmetic(self):
        # + and - are in minimal
        sexpr = "(λ eq (+ (length $0) 1) 4)"
        result = _rewrite(sexpr, "minimal")
        # Should pass without error
        assert "+" in collect_primitive_names(result)

    def test_minimal_raises_for_le(self):
        # le is NOT in minimal (only eq, lt, gt)
        sexpr = "(λ le (length $0) 3)"
        with pytest.raises(InexpressibleError, match="le"):
            _rewrite(sexpr, "minimal")


# ===========================================================================
# Error handling
# ===========================================================================

class TestErrorHandling:
    """Verify error cases."""

    def test_unknown_grammar_raises_value_error(self):
        sexpr = "(λ eq (length $0) 3)"
        prog = parse_hypothesis_sexpr(sexpr)
        with pytest.raises(ValueError, match="Unknown grammar"):
            rewrite_ast(prog, "nonexistent-grammar")

    def test_inexpressible_error_attributes(self):
        err = InexpressibleError("foo", "minimal")
        assert err.primitive_name == "foo"
        assert err.target_grammar == "minimal"
        assert "foo" in str(err)
        assert "minimal" in str(err)


# ===========================================================================
# AST size / structure checks
# ===========================================================================

class TestASTStructure:
    """Verify that rewrites produce well-formed ASTs with expected sizes."""

    def test_identity_preserves_size(self):
        sexpr = "(λ all (λ eq (get_suit $0) HEARTS) $0)"
        prog = parse_hypothesis_sexpr(sexpr)
        for grammar in ("base", "add-both", "redundant"):
            result = rewrite_ast(prog, grammar)
            assert result.size() == prog.size()

    def test_positional_rewrite_increases_size(self):
        """first_half (1 node) -> slice 0 3 (3 nodes), so size increases."""
        sexpr = "(λ all (λ eq (get_suit $0) HEARTS) (first_half $0))"
        prog = parse_hypothesis_sexpr(sexpr)
        result = _rewrite(sexpr, "swap-positional")
        # slice 0 3 replaces first_half: +2 primitives, +2 applications
        # So size should increase
        assert result.size() > prog.size()

    def test_distributional_rewrite_changes_size(self):
        """count_suit hand S -> count_where (λ ...) hand increases size."""
        sexpr = "(λ eq 2 (count_suit $0 HEARTS))"
        prog = parse_hypothesis_sexpr(sexpr)
        result = _rewrite(sexpr, "swap-distributional")
        # The count_where version has a lambda with eq, get_suit, $0, etc.
        # Size will increase
        assert result.size() > prog.size()

    def test_rewritten_ast_is_valid_program(self):
        """Rewritten ASTs should still be structurally valid Programs."""
        sexpr = "(λ eq 2 (count_suit $0 HEARTS))"
        result = _rewrite(sexpr, "swap-distributional")
        # Should be able to get str, size, depth without errors
        assert isinstance(str(result), str)
        assert result.size() > 0
        assert result.depth() > 0
