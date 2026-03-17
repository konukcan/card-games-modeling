"""Tests for ablation analysis: leave-one-out, leave-one-in, cross-validation.

Written TDD-style: these tests define the expected behaviour of the ablation
framework before verifying the implementation.
"""

import sys
from pathlib import Path

# Allow importing from the main src/ tree
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

import pytest

from llm.grammar_comparison.evaluation.ablation import (
    leave_one_out,
    leave_one_in,
    cross_validate,
)
from llm.grammar_comparison.grammars.grammar_factory import CostStructure


# ---------------------------------------------------------------------------
# 1. leave_one_out
# ---------------------------------------------------------------------------

class TestLeaveOneOut:
    """Removing one primitive at a time from a grammar."""

    def test_returns_list(self):
        """leave_one_out should return a list."""
        result = leave_one_out("minimal", limit=5)
        assert isinstance(result, list)

    def test_one_entry_per_primitive(self):
        """Should have one entry for each primitive in the base grammar."""
        from llm.grammar_comparison.grammars.grammar_factory import build_grammar
        grammar = build_grammar("minimal", CostStructure.UNIFORM)
        n_prims = len(grammar.productions)

        result = leave_one_out("minimal", limit=5)
        assert len(result) == n_prims

    def test_dict_keys(self):
        """Each entry should have the required dict keys."""
        result = leave_one_out("minimal", limit=5)
        assert len(result) > 0
        entry = result[0]
        assert "removed" in entry
        assert "metric_value" in entry
        assert "baseline_value" in entry
        assert "delta" in entry

    def test_sorted_by_delta(self):
        """Results should be sorted by delta (most impactful first).

        "Most impactful" means the largest absolute change. For metrics
        where higher is better (like expressibility), removing a useful
        primitive makes delta negative, so we sort by absolute delta
        descending.
        """
        result = leave_one_out("minimal", limit=5)
        deltas = [abs(r["delta"]) for r in result]
        assert deltas == sorted(deltas, reverse=True)

    def test_delta_is_difference(self):
        """delta should equal metric_value - baseline_value."""
        result = leave_one_out("minimal", limit=5)
        for entry in result:
            assert abs(entry["delta"] - (entry["metric_value"] - entry["baseline_value"])) < 1e-10

    def test_cost_structure_parameter(self):
        """Should accept a cost_structure parameter."""
        result = leave_one_out("minimal", cost_structure=CostStructure.TIERED, limit=5)
        assert isinstance(result, list)

    def test_metric_parameter(self):
        """Should accept different metric names."""
        result = leave_one_out("minimal", metric="expressibility", limit=5)
        assert isinstance(result, list)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# 2. leave_one_in
# ---------------------------------------------------------------------------

class TestLeaveOneIn:
    """Adding one candidate primitive at a time to a grammar."""

    def test_returns_list(self):
        """leave_one_in should return a list."""
        result = leave_one_in("minimal", limit=5)
        assert isinstance(result, list)

    def test_entries_for_candidates(self):
        """Should have one entry for each candidate primitive added."""
        result = leave_one_in("minimal", limit=5)
        assert len(result) > 0  # base has more prims than minimal

    def test_dict_keys(self):
        """Each entry should have the required dict keys."""
        result = leave_one_in("minimal", limit=5)
        assert len(result) > 0
        entry = result[0]
        assert "added" in entry
        assert "metric_value" in entry
        assert "baseline_value" in entry
        assert "delta" in entry

    def test_sorted_by_delta(self):
        """Results should be sorted by delta (most beneficial first).

        Most beneficial addition = largest positive delta (improves the metric).
        We sort by absolute delta descending.
        """
        result = leave_one_in("minimal", limit=5)
        deltas = [abs(r["delta"]) for r in result]
        assert deltas == sorted(deltas, reverse=True)

    def test_explicit_candidates(self):
        """Should accept an explicit list of candidate primitive names."""
        result = leave_one_in("minimal", candidates=["take", "drop"], limit=5)
        assert len(result) == 2
        added_names = {r["added"] for r in result}
        assert added_names == {"take", "drop"}

    def test_delta_is_difference(self):
        """delta should equal metric_value - baseline_value."""
        result = leave_one_in("minimal", limit=5)
        for entry in result:
            assert abs(entry["delta"] - (entry["metric_value"] - entry["baseline_value"])) < 1e-10


# ---------------------------------------------------------------------------
# 3. cross_validate
# ---------------------------------------------------------------------------

class TestCrossValidate:
    """k-fold cross-validation over hypotheses."""

    def test_returns_tuple(self):
        """cross_validate should return a (mean, std) tuple."""
        mean, std = cross_validate("minimal", limit=5)
        assert isinstance(mean, float)
        assert isinstance(std, float)

    def test_std_non_negative(self):
        """Standard deviation should be >= 0."""
        _, std = cross_validate("minimal", limit=5)
        assert std >= 0.0

    def test_k_parameter(self):
        """Should accept a k parameter for number of folds."""
        mean, std = cross_validate("minimal", k=3, limit=5)
        assert isinstance(mean, float)
        assert isinstance(std, float)

    def test_metric_parameter(self):
        """Should accept different metric names."""
        mean, std = cross_validate("minimal", metric="expressibility", limit=5)
        assert isinstance(mean, float)
