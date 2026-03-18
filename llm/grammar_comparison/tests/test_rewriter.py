"""
Tests for the mechanical AST rewriter.

Verifies that rewrite_ast correctly transforms Base grammar ASTs into
equivalent ASTs using each target grammar's primitives.

Key invariants:
  - For non-minimal grammars, the rewriter NEVER raises InexpressibleError.
  - For minimal, InexpressibleError is raised ONLY for primitives that
    genuinely require fold/reduce/sort (see MINIMAL_INEXPRESSIBLE).
  - Redundant grammar applies compression: detect base-grammar patterns
    and replace them with cognitive shortcuts.
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
    MINIMAL_INEXPRESSIBLE,
)
from llm.grammar_comparison.grammars.grammar_factory import (
    _MINIMAL_KEEP, _POSITIONAL_REMOVE, _DISTRIBUTIONAL_REMOVE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rewrite(sexpr: str, target: str):
    """Parse a base-grammar s-expression and rewrite to target grammar."""
    prog = parse_hypothesis_sexpr(sexpr)
    return rewrite_ast(prog, target)


# ===========================================================================
# Identity grammars: base, add-both
# ===========================================================================

class TestIdentityGrammars:
    @pytest.mark.parametrize("grammar", ["base", "add-both"])
    def test_simple_rule_unchanged(self, grammar):
        sexpr = "(λ all (λ eq (get_suit $0) HEARTS) $0)"
        prog = parse_hypothesis_sexpr(sexpr)
        result = rewrite_ast(prog, grammar)
        assert result is prog

    @pytest.mark.parametrize("grammar", ["base", "add-both"])
    def test_str_preserved(self, grammar):
        sexpr = "(λ eq (length $0) 3)"
        prog = parse_hypothesis_sexpr(sexpr)
        result = rewrite_ast(prog, grammar)
        assert str(result) == str(prog)


# ===========================================================================
# swap-positional grammar
# ===========================================================================

class TestSwapPositional:
    def test_first_half_becomes_slice_0_3(self):
        sexpr = "(λ all (λ eq (get_suit $0) HEARTS) (first_half $0))"
        result = _rewrite(sexpr, "swap-positional")
        prims = collect_primitive_names(result)
        assert "slice" in prims
        assert "first_half" not in prims

    def test_second_half_becomes_slice_3_6(self):
        result = _rewrite("(λ all (λ eq (get_suit $0) HEARTS) (second_half $0))", "swap-positional")
        prims = collect_primitive_names(result)
        assert "slice" in prims
        assert "second_half" not in prims

    def test_take_becomes_slice_0_n(self):
        result = _rewrite("(λ length (take 2 $0))", "swap-positional")
        prims = collect_primitive_names(result)
        assert "take" not in prims
        assert "slice" in prims

    def test_drop_becomes_slice_n_6(self):
        result = _rewrite("(λ length (drop 2 $0))", "swap-positional")
        prims = collect_primitive_names(result)
        assert "drop" not in prims
        assert "slice" in prims

    def test_non_positional_prims_unchanged(self):
        sexpr = "(λ eq (length $0) 3)"
        prog = parse_hypothesis_sexpr(sexpr)
        result = rewrite_ast(prog, "swap-positional")
        assert str(result) == str(prog)

    def test_adjacent_pairs_in_all_with_simple_pred(self):
        """all + adjacent_pairs with simple eq comparator -> shifted_match."""
        sexpr = "(λ all (λ eq (get_color (head $0)) (get_color (last $0))) (adjacent_pairs $0))"
        result = _rewrite(sexpr, "swap-positional")
        prims = collect_primitive_names(result)
        assert "shifted_match" in prims
        assert "adjacent_pairs" not in prims

    def test_adjacent_pairs_in_all_with_compound_pred(self):
        """all + adjacent_pairs with or-compound predicate -> shifted_match."""
        sexpr = "(λ all (λ or (eq (get_rank (head $0)) (get_rank (last $0))) (eq (get_suit (head $0)) (get_suit (last $0)))) (adjacent_pairs $0))"
        result = _rewrite(sexpr, "swap-positional")
        prims = collect_primitive_names(result)
        assert "shifted_match" in prims
        assert "adjacent_pairs" not in prims

    def test_adjacent_pairs_in_any(self):
        sexpr = "(λ any (λ eq (get_rank (head $0)) (get_rank (last $0))) (adjacent_pairs $0))"
        result = _rewrite(sexpr, "swap-positional")
        prims = collect_primitive_names(result)
        assert "shifted_match" in prims
        assert "adjacent_pairs" not in prims

    def test_adjacent_pairs_in_filter(self):
        """filter + adjacent_pairs -> zip_with + filter decomposition."""
        sexpr = "(λ length (filter (λ eq (get_suit (head $0)) (get_suit (last $0))) (adjacent_pairs $0)))"
        result = _rewrite(sexpr, "swap-positional")
        prims = collect_primitive_names(result)
        assert "adjacent_pairs" not in prims

    def test_standalone_adjacent_pairs_decomposed(self):
        sexpr = "(λ length (adjacent_pairs $0))"
        result = _rewrite(sexpr, "swap-positional")
        prims = collect_primitive_names(result)
        assert "adjacent_pairs" not in prims


# ===========================================================================
# swap-distributional grammar
# ===========================================================================

class TestSwapDistributional:
    def test_count_suit_becomes_count_where(self):
        result = _rewrite("(λ eq 2 (count_suit $0 HEARTS))", "swap-distributional")
        prims = collect_primitive_names(result)
        assert "count_where" in prims
        assert "count_suit" not in prims

    def test_count_color_becomes_count_where(self):
        result = _rewrite("(λ eq 2 (count_color $0 RED))", "swap-distributional")
        prims = collect_primitive_names(result)
        assert "count_where" in prims
        assert "count_color" not in prims

    def test_non_distributional_prims_unchanged(self):
        sexpr = "(λ eq (length $0) 3)"
        prog = parse_hypothesis_sexpr(sexpr)
        result = rewrite_ast(prog, "swap-distributional")
        assert str(result) == str(prog)


# ===========================================================================
# swap-both grammar
# ===========================================================================

class TestSwapBoth:
    def test_both_positional_and_distributional(self):
        result = _rewrite("(λ eq (count_suit (first_half $0) HEARTS) 2)", "swap-both")
        prims = collect_primitive_names(result)
        assert "first_half" not in prims
        assert "count_suit" not in prims
        assert "slice" in prims
        assert "count_where" in prims

    def test_only_positional_in_swap_both(self):
        result = _rewrite("(λ all (λ eq (get_suit $0) HEARTS) (first_half $0))", "swap-both")
        prims = collect_primitive_names(result)
        assert "first_half" not in prims
        assert "slice" in prims


# ===========================================================================
# minimal grammar: decomposition
# ===========================================================================

class TestMinimalDecomposition:
    def test_minimal_keeps_basic_rule(self):
        sexpr = "(λ all (λ eq (get_suit $0) HEARTS) $0)"
        result = _rewrite(sexpr, "minimal")
        assert str(result) == str(parse_hypothesis_sexpr(sexpr))

    def test_le_decomposed(self):
        result = _rewrite("(λ le (length $0) 3)", "minimal")
        prims = collect_primitive_names(result)
        assert "le" not in prims
        assert {"or", "lt", "eq"} <= prims

    def test_ge_decomposed(self):
        result = _rewrite("(λ ge (length $0) 3)", "minimal")
        prims = collect_primitive_names(result)
        assert "ge" not in prims
        assert {"or", "gt", "eq"} <= prims

    def test_has_suit_decomposed(self):
        result = _rewrite("(λ has_suit $0 HEARTS)", "minimal")
        prims = collect_primitive_names(result)
        assert "has_suit" not in prims
        assert {"any", "get_suit", "eq"} <= prims

    def test_has_color_decomposed(self):
        result = _rewrite("(λ has_color $0 RED)", "minimal")
        prims = collect_primitive_names(result)
        assert "has_color" not in prims
        assert "any" in prims

    def test_count_suit_decomposed(self):
        result = _rewrite("(λ eq 2 (count_suit $0 HEARTS))", "minimal")
        prims = collect_primitive_names(result)
        assert "count_suit" not in prims
        assert {"length", "filter", "get_suit"} <= prims

    def test_count_color_decomposed(self):
        result = _rewrite("(λ eq 2 (count_color $0 RED))", "minimal")
        prims = collect_primitive_names(result)
        assert "count_color" not in prims
        assert {"length", "filter"} <= prims

    def test_n_unique_suits_decomposed(self):
        result = _rewrite("(λ eq 2 (n_unique_suits $0))", "minimal")
        prims = collect_primitive_names(result)
        assert "n_unique_suits" not in prims
        assert {"length", "unique", "map", "get_suit"} <= prims

    def test_n_unique_ranks_decomposed(self):
        result = _rewrite("(λ eq 2 (n_unique_ranks $0))", "minimal")
        prims = collect_primitive_names(result)
        assert "n_unique_ranks" not in prims
        assert "get_rank" in prims

    def test_n_unique_colors_decomposed(self):
        result = _rewrite("(λ eq 2 (n_unique_colors $0))", "minimal")
        assert "n_unique_colors" not in collect_primitive_names(result)

    def test_last_decomposed(self):
        result = _rewrite("(λ eq (get_suit (last $0)) HEARTS)", "minimal")
        prims = collect_primitive_names(result)
        assert "last" not in prims
        assert {"at", "-", "length"} <= prims

    def test_all_minimal_prims_in_result(self):
        """After decomposition, all primitives should be in _MINIMAL_KEEP."""
        result = _rewrite("(λ and (ge (n_unique_suits $0) 2) (le (n_unique_colors $0) 1))", "minimal")
        prims = collect_primitive_names(result)
        for p in prims:
            assert p in _MINIMAL_KEEP, f"'{p}' not in minimal set"


# ===========================================================================
# minimal grammar: inexpressible
# ===========================================================================

class TestMinimalInexpressible:
    def test_sum_ranks_raises(self):
        with pytest.raises(InexpressibleError, match="sum_ranks"):
            _rewrite("(λ sum_ranks $0)", "minimal")

    def test_max_rank_raises(self):
        with pytest.raises(InexpressibleError, match="max_rank"):
            _rewrite("(λ max_rank $0)", "minimal")

    def test_min_rank_raises(self):
        with pytest.raises(InexpressibleError, match="min_rank"):
            _rewrite("(λ min_rank $0)", "minimal")

    def test_first_half_raises(self):
        with pytest.raises(InexpressibleError, match="first_half"):
            _rewrite("(λ all (λ eq (get_suit $0) HEARTS) (first_half $0))", "minimal")

    def test_adjacent_pairs_raises(self):
        with pytest.raises(InexpressibleError, match="adjacent_pairs"):
            _rewrite("(λ all (λ eq (head $0) (last $0)) (adjacent_pairs $0))", "minimal")

    def test_sort_by_rank_raises(self):
        with pytest.raises(InexpressibleError, match="sort_by_rank"):
            _rewrite("(λ length (sort_by_rank $0))", "minimal")

    def test_all_inexpressible_prims_fail(self):
        """Every primitive in MINIMAL_INEXPRESSIBLE should fail standalone."""
        for prim_name in MINIMAL_INEXPRESSIBLE:
            with pytest.raises(InexpressibleError, match=prim_name):
                _rewrite(f"(λ {prim_name} $0)", "minimal")


# ===========================================================================
# redundant grammar: compression
# ===========================================================================

class TestRedundantCompression:
    def test_preserves_simple_rule(self):
        sexpr = "(λ all (λ eq (get_suit $0) HEARTS) $0)"
        prog = parse_hypothesis_sexpr(sexpr)
        assert rewrite_ast(prog, "redundant") is prog

    def test_n_unique_compressed(self):
        result = _rewrite("(λ eq 2 (length (unique (map get_suit $0))))", "redundant")
        assert "n_unique" in collect_primitive_names(result)

    def test_all_different_compressed(self):
        result = _rewrite("(λ eq (length (unique (map get_rank $0))) (length $0))", "redundant")
        assert "all_different" in collect_primitive_names(result)

    def test_all_same_lt_2_composed(self):
        result = _rewrite("(λ lt (length (unique (map get_suit $0))) 2)", "redundant")
        assert "all_same" in collect_primitive_names(result)

    def test_all_same_compound_lt_2(self):
        result = _rewrite("(λ lt (n_unique_suits $0) 2)", "redundant")
        assert "all_same" in collect_primitive_names(result)

    def test_all_same_compound_le_1(self):
        result = _rewrite("(λ le (n_unique_colors $0) 1)", "redundant")
        assert "all_same" in collect_primitive_names(result)

    def test_all_same_compound_eq_1(self):
        result = _rewrite("(λ eq (n_unique_colors $0) 1)", "redundant")
        assert "all_same" in collect_primitive_names(result)

    def test_exactly_n_compressed(self):
        result = _rewrite("(λ eq (length (filter (λ eq (get_suit $0) HEARTS) $0)) 3)", "redundant")
        assert "exactly_n" in collect_primitive_names(result)

    def test_at_least_n_compressed(self):
        result = _rewrite("(λ ge (length (filter (λ eq (get_suit $0) HEARTS) $0)) 3)", "redundant")
        assert "at_least_n" in collect_primitive_names(result)

    def test_no_compression_for_non_pattern(self):
        sexpr = "(λ eq (length $0) 3)"
        prog = parse_hypothesis_sexpr(sexpr)
        assert rewrite_ast(prog, "redundant") is prog


# ===========================================================================
# Bulk expressibility tests
# ===========================================================================

class TestBulkExpressibility:
    @pytest.fixture(scope="class")
    def all_hypotheses(self):
        from llm.grammar_comparison.data_loader import load_phase1b_hypotheses
        hyps = load_phase1b_hypotheses()
        return [h for h in hyps if h["dsl_code"]]

    def _count_expressible(self, hypotheses, grammar):
        success = 0
        for h in hypotheses:
            try:
                prog = parse_hypothesis_sexpr(h["dsl_code"])
                rewrite_ast(prog, grammar)
                success += 1
            except (InexpressibleError, Exception):
                pass
        return success

    @pytest.mark.parametrize("grammar", [
        "swap-positional", "swap-distributional", "swap-both", "redundant"
    ])
    def test_non_minimal_100_percent(self, all_hypotheses, grammar):
        """Non-minimal grammars should rewrite ALL hypotheses without error."""
        count = self._count_expressible(all_hypotheses, grammar)
        assert count == len(all_hypotheses), (
            f"Expected 100% for {grammar}, got {count}/{len(all_hypotheses)}"
        )

    def test_minimal_at_least_70_percent(self, all_hypotheses):
        """Minimal should handle at least 70% (fold/reduce prims cause failures)."""
        count = self._count_expressible(all_hypotheses, "minimal")
        pct = count / len(all_hypotheses) * 100
        assert pct >= 70.0, f"Got {pct:.1f}% ({count}/{len(all_hypotheses)})"

    def test_redundant_compresses_some(self, all_hypotheses):
        """Redundant grammar should compress at least some patterns."""
        shortcuts = {"all_same", "all_different", "n_unique", "has_pair",
                     "exactly_n", "at_least_n", "is_sorted", "is_run"}
        compressed = sum(
            1 for h in all_hypotheses
            if (prog := parse_hypothesis_sexpr(h["dsl_code"])) and
            collect_primitive_names(rewrite_ast(prog, "redundant")) & shortcuts
        )
        assert compressed > 0


# ===========================================================================
# Swap completeness: verify no removed prims remain
# ===========================================================================

class TestSwapCompleteness:
    @pytest.fixture(scope="class")
    def all_hypotheses(self):
        from llm.grammar_comparison.data_loader import load_phase1b_hypotheses
        hyps = load_phase1b_hypotheses()
        return [h for h in hyps if h["dsl_code"]]

    def test_swap_positional_removes_all(self, all_hypotheses):
        for h in all_hypotheses:
            prog = parse_hypothesis_sexpr(h["dsl_code"])
            result = rewrite_ast(prog, "swap-positional")
            remaining = collect_primitive_names(result) & _POSITIONAL_REMOVE
            assert not remaining, f"Rule {h['rule_id']}: {remaining} remain"

    def test_swap_distributional_removes_all(self, all_hypotheses):
        for h in all_hypotheses:
            prog = parse_hypothesis_sexpr(h["dsl_code"])
            result = rewrite_ast(prog, "swap-distributional")
            remaining = collect_primitive_names(result) & _DISTRIBUTIONAL_REMOVE
            assert not remaining, f"Rule {h['rule_id']}: {remaining} remain"

    def test_swap_both_removes_all(self, all_hypotheses):
        all_removed = _POSITIONAL_REMOVE | _DISTRIBUTIONAL_REMOVE
        for h in all_hypotheses:
            prog = parse_hypothesis_sexpr(h["dsl_code"])
            result = rewrite_ast(prog, "swap-both")
            remaining = collect_primitive_names(result) & all_removed
            assert not remaining, f"Rule {h['rule_id']}: {remaining} remain"


# ===========================================================================
# Error handling
# ===========================================================================

class TestErrorHandling:
    def test_unknown_grammar_raises_value_error(self):
        prog = parse_hypothesis_sexpr("(λ eq (length $0) 3)")
        with pytest.raises(ValueError, match="Unknown grammar"):
            rewrite_ast(prog, "nonexistent-grammar")

    def test_inexpressible_error_attributes(self):
        err = InexpressibleError("foo", "minimal")
        assert err.primitive_name == "foo"
        assert err.target_grammar == "minimal"


# ===========================================================================
# AST structure checks
# ===========================================================================

class TestASTStructure:
    def test_identity_preserves_size(self):
        sexpr = "(λ all (λ eq (get_suit $0) HEARTS) $0)"
        prog = parse_hypothesis_sexpr(sexpr)
        for g in ("base", "add-both"):
            assert rewrite_ast(prog, g).size() == prog.size()

    def test_positional_rewrite_increases_size(self):
        sexpr = "(λ all (λ eq (get_suit $0) HEARTS) (first_half $0))"
        prog = parse_hypothesis_sexpr(sexpr)
        assert _rewrite(sexpr, "swap-positional").size() > prog.size()

    def test_distributional_rewrite_increases_size(self):
        sexpr = "(λ eq 2 (count_suit $0 HEARTS))"
        prog = parse_hypothesis_sexpr(sexpr)
        assert _rewrite(sexpr, "swap-distributional").size() > prog.size()

    def test_minimal_decomposition_increases_size(self):
        sexpr = "(λ eq 2 (n_unique_suits $0))"
        prog = parse_hypothesis_sexpr(sexpr)
        assert _rewrite(sexpr, "minimal").size() > prog.size()

    def test_redundant_compression_may_decrease_size(self):
        sexpr = "(λ eq 2 (length (unique (map get_suit $0))))"
        prog = parse_hypothesis_sexpr(sexpr)
        assert _rewrite(sexpr, "redundant").size() <= prog.size()
