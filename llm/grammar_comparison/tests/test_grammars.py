"""
Tests for grammar_factory: 7 grammar families × 3 cost structures.

Written TDD-style: these tests define the expected behaviour of the
grammar factory before the full implementation exists.
"""

import sys
from pathlib import Path

# Allow importing from the main src/ tree
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import math
import pytest

from llm.grammar_comparison.grammars.grammar_factory import (
    build_grammar,
    CostStructure,
    GRAMMAR_NAMES,
)
from dreamcoder_core.grammar import Grammar, Production


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prim_names(grammar: Grammar) -> set:
    """Return the set of primitive names in a grammar."""
    return {p.program.name for p in grammar.productions}


# ---------------------------------------------------------------------------
# 1. All 7 grammars are buildable
# ---------------------------------------------------------------------------

class TestAllGrammarsBuildable:
    """Every (name, cost) pair should produce a valid Grammar."""

    @pytest.mark.parametrize("name", GRAMMAR_NAMES)
    def test_builds_with_uniform(self, name):
        g = build_grammar(name, CostStructure.UNIFORM)
        assert isinstance(g, Grammar)
        assert len(g.productions) > 0

    @pytest.mark.parametrize("name", GRAMMAR_NAMES)
    def test_builds_with_tiered(self, name):
        g = build_grammar(name, CostStructure.TIERED)
        assert isinstance(g, Grammar)
        assert len(g.productions) > 0

    @pytest.mark.parametrize("name", GRAMMAR_NAMES)
    def test_builds_with_lotlib3(self, name):
        g = build_grammar(name, CostStructure.LOTLIB3)
        assert isinstance(g, Grammar)
        assert len(g.productions) > 0


# ---------------------------------------------------------------------------
# 2. build_grammar returns Grammar with productions
# ---------------------------------------------------------------------------

class TestGrammarStructure:
    """Basic structural checks on returned Grammar objects."""

    def test_returns_grammar_instance(self):
        g = build_grammar("base", CostStructure.UNIFORM)
        assert isinstance(g, Grammar)

    def test_productions_are_production_objects(self):
        g = build_grammar("base", CostStructure.UNIFORM)
        for p in g.productions:
            assert isinstance(p, Production)

    def test_all_log_probs_are_negative(self):
        """Log-probabilities should be negative (probabilities < 1)."""
        g = build_grammar("base", CostStructure.UNIFORM)
        for p in g.productions:
            assert p.log_probability < 0, f"{p.program.name} has non-negative log_prob"


# ---------------------------------------------------------------------------
# 3. swap-positional: has slice/shifted_match, lacks take/drop/etc.
# ---------------------------------------------------------------------------

class TestSwapPositional:
    """G2 should add slice and shifted_match, remove positional primitives."""

    def setup_method(self):
        self.g = build_grammar("swap-positional", CostStructure.UNIFORM)
        self.names = _prim_names(self.g)

    def test_has_slice(self):
        assert "slice" in self.names

    def test_has_shifted_match(self):
        assert "shifted_match" in self.names

    def test_lacks_take(self):
        assert "take" not in self.names

    def test_lacks_drop(self):
        assert "drop" not in self.names

    def test_lacks_first_half(self):
        assert "first_half" not in self.names

    def test_lacks_second_half(self):
        assert "second_half" not in self.names

    def test_lacks_adjacent_pairs(self):
        assert "adjacent_pairs" not in self.names

    def test_lacks_shifted_pairs(self):
        # shifted_pairs may not be in base either, but should not appear
        assert "shifted_pairs" not in self.names


# ---------------------------------------------------------------------------
# 4. swap-distributional: has count_where/sorted_counts, lacks count_suit etc.
# ---------------------------------------------------------------------------

class TestSwapDistributional:
    """G3 should add count_where and sorted_counts, remove count_suit/rank/color."""

    def setup_method(self):
        self.g = build_grammar("swap-distributional", CostStructure.UNIFORM)
        self.names = _prim_names(self.g)

    def test_has_count_where(self):
        assert "count_where" in self.names

    def test_has_sorted_counts(self):
        assert "sorted_counts" in self.names

    def test_lacks_count_suit(self):
        assert "count_suit" not in self.names

    def test_lacks_count_rank(self):
        # count_rank is not in base, but let's confirm
        assert "count_rank" not in self.names

    def test_lacks_count_color(self):
        assert "count_color" not in self.names


# ---------------------------------------------------------------------------
# 5. minimal has fewer productions than base
# ---------------------------------------------------------------------------

class TestMinimal:
    """G7 (minimal) should have strictly fewer primitives than G1 (base)."""

    def test_fewer_than_base(self):
        base = build_grammar("base", CostStructure.UNIFORM)
        minimal = build_grammar("minimal", CostStructure.UNIFORM)
        assert len(minimal.productions) < len(base.productions), (
            f"minimal ({len(minimal.productions)}) should be < "
            f"base ({len(base.productions)})"
        )

    def test_has_expected_core_primitives(self):
        """Minimal should still contain core primitives like head, map, eq."""
        g = build_grammar("minimal", CostStructure.UNIFORM)
        names = _prim_names(g)
        for expected in ["head", "map", "filter", "eq", "get_suit", "get_rank"]:
            assert expected in names, f"minimal grammar missing {expected}"


# ---------------------------------------------------------------------------
# 6. redundant has more productions than base
# ---------------------------------------------------------------------------

class TestRedundant:
    """G6 (redundant) should have strictly more primitives than G1 (base)."""

    def test_more_than_base(self):
        base = build_grammar("base", CostStructure.UNIFORM)
        redundant = build_grammar("redundant", CostStructure.UNIFORM)
        assert len(redundant.productions) > len(base.productions), (
            f"redundant ({len(redundant.productions)}) should be > "
            f"base ({len(base.productions)})"
        )

    def test_has_cognitive_shortcuts(self):
        """Redundant should include the 8 cognitive shortcut primitives."""
        g = build_grammar("redundant", CostStructure.UNIFORM)
        names = _prim_names(g)
        shortcuts = [
            "all_same", "all_different", "is_sorted", "exactly_n",
            "at_least_n", "n_unique", "is_run", "has_pair",
        ]
        for s in shortcuts:
            assert s in names, f"redundant grammar missing shortcut '{s}'"


# ---------------------------------------------------------------------------
# 7. All 3 cost structures produce different log-probabilities
# ---------------------------------------------------------------------------

class TestCostStructures:
    """Different cost structures should produce different probability assignments."""

    def test_uniform_all_same_log_prob(self):
        """Under UNIFORM, all productions should have the same log-prob."""
        g = build_grammar("base", CostStructure.UNIFORM)
        log_probs = {p.log_probability for p in g.productions}
        # uniform_grammar gives all the same log_prob
        assert len(log_probs) == 1, (
            f"UNIFORM should have 1 unique log_prob, got {len(log_probs)}"
        )

    def test_tiered_has_multiple_log_probs(self):
        """Under TIERED, different tiers should produce different log-probs."""
        g = build_grammar("base", CostStructure.TIERED)
        log_probs = {round(p.log_probability, 6) for p in g.productions}
        assert len(log_probs) > 1, (
            "TIERED should have multiple different log_probs"
        )

    def test_lotlib3_has_multiple_log_probs(self):
        """Under LOTLIB3, different primitives should get different weights."""
        g = build_grammar("base", CostStructure.LOTLIB3)
        log_probs = {round(p.log_probability, 6) for p in g.productions}
        assert len(log_probs) > 1, (
            "LOTLIB3 should have multiple different log_probs"
        )


# ---------------------------------------------------------------------------
# 8. swap-both combines both swaps
# ---------------------------------------------------------------------------

class TestSwapBoth:
    """G4 = G2 + G3 combined."""

    def setup_method(self):
        self.g = build_grammar("swap-both", CostStructure.UNIFORM)
        self.names = _prim_names(self.g)

    def test_has_slice(self):
        assert "slice" in self.names

    def test_has_shifted_match(self):
        assert "shifted_match" in self.names

    def test_has_count_where(self):
        assert "count_where" in self.names

    def test_has_sorted_counts(self):
        assert "sorted_counts" in self.names

    def test_lacks_take(self):
        assert "take" not in self.names

    def test_lacks_count_suit(self):
        assert "count_suit" not in self.names


# ---------------------------------------------------------------------------
# 9. add-both adds all 5 new primitives without removing anything
# ---------------------------------------------------------------------------

class TestAddBoth:
    """G5 = base + all 5 new prims, nothing removed."""

    def setup_method(self):
        self.g = build_grammar("add-both", CostStructure.UNIFORM)
        self.names = _prim_names(self.g)

    def test_has_all_new_primitives(self):
        for name in ["slice", "shifted_match", "stride", "count_where", "sorted_counts"]:
            assert name in self.names, f"add-both missing '{name}'"

    def test_still_has_original_positional(self):
        """Nothing removed, so take/drop should still be present."""
        for name in ["take", "drop", "first_half", "second_half"]:
            assert name in self.names, f"add-both should still have '{name}'"

    def test_more_than_base(self):
        base = build_grammar("base", CostStructure.UNIFORM)
        assert len(self.g.productions) > len(base.productions)


# ---------------------------------------------------------------------------
# 10. Invalid grammar name raises error
# ---------------------------------------------------------------------------

class TestInvalidInput:
    def test_unknown_name_raises(self):
        with pytest.raises((ValueError, KeyError)):
            build_grammar("nonexistent-grammar", CostStructure.UNIFORM)
