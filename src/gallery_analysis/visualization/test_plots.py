"""Tests for the plot library.

Each test loads real data from the depth6_injected.json results and verifies
that the corresponding plot function returns a valid Altair chart object.
"""

import sys
from pathlib import Path

import pytest
import altair as alt

# Ensure src/ is on the path so imports resolve.
SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gallery_analysis.visualization.data import BayesianResults, load_results
from gallery_analysis.visualization.plots import (
    difficulty_strip,
    difficulty_scatter,
    true_rule_recovery,
    equiv_class_bars,
    posterior_bars,
    prior_vs_likelihood,
    diagnosticity_bars,
)

# Path to the real results JSON produced by analyze.py.
RESULTS_JSON = (
    Path(__file__).resolve().parent.parent / "results" / "depth6_injected.json"
)

# Altair chart types that any of our functions may return.
CHART_TYPES = (alt.Chart, alt.LayerChart, alt.HConcatChart, alt.VConcatChart)


@pytest.fixture(scope="module")
def results() -> BayesianResults:
    """Load results once for the whole test module."""
    return load_results(RESULTS_JSON)


# ── Summary-level plots (take difficulty_df) ─────────────────────────


def test_difficulty_strip_returns_chart(results: BayesianResults):
    chart = difficulty_strip(results.difficulty_df)
    assert isinstance(chart, CHART_TYPES)


def test_difficulty_scatter_returns_chart(results: BayesianResults):
    chart = difficulty_scatter(results.difficulty_df)
    assert isinstance(chart, CHART_TYPES)


def test_true_rule_recovery_returns_chart(results: BayesianResults):
    chart = true_rule_recovery(results.difficulty_df)
    assert isinstance(chart, CHART_TYPES)


def test_equiv_class_bars_returns_chart(results: BayesianResults):
    chart = equiv_class_bars(results.difficulty_df)
    assert isinstance(chart, CHART_TYPES)


# ── Per-rule plots (take filtered hypotheses_df / diagnosticity_df) ──


def test_posterior_bars_returns_chart(results: BayesianResults):
    # Pick the first rule_id that has hypotheses.
    rule_id = results.hypotheses_df["rule_id"].iloc[0]
    hyp_df = results.hypotheses_df[results.hypotheses_df["rule_id"] == rule_id]
    chart = posterior_bars(hyp_df, rule_id)
    assert isinstance(chart, CHART_TYPES)


def test_prior_vs_likelihood_returns_chart(results: BayesianResults):
    rule_id = results.hypotheses_df["rule_id"].iloc[0]
    hyp_df = results.hypotheses_df[results.hypotheses_df["rule_id"] == rule_id]
    chart = prior_vs_likelihood(hyp_df)
    assert isinstance(chart, CHART_TYPES)


def test_diagnosticity_bars_returns_chart(results: BayesianResults):
    diag_df = results.diagnosticity_df
    if len(diag_df) == 0:
        pytest.skip("No diagnosticity data available")
    rule_id = diag_df["rule_id"].iloc[0]
    subset = diag_df[diag_df["rule_id"] == rule_id]
    chart = diagnosticity_bars(subset)
    assert isinstance(chart, CHART_TYPES)


# ── Run as script ─────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
