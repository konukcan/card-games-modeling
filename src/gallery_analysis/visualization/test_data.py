"""Tests for the data loading module.

Loads the real depth6_injected.json results and verifies that
load_results() produces correctly shaped DataFrames.
"""

import sys
from pathlib import Path

import pytest
import pandas as pd

# Ensure src/ is on the path so imports resolve.
SRC_DIR = Path(__file__).resolve().parent.parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gallery_analysis.visualization.data import BayesianResults, load_results

# Path to the real results JSON produced by analyze.py.
RESULTS_JSON = (
    Path(__file__).resolve().parent.parent / "results" / "depth6_injected.json"
)


@pytest.fixture(scope="module")
def results() -> BayesianResults:
    """Load results once for the whole test module."""
    return load_results(RESULTS_JSON)


# ── BayesianResults type ──────────────────────────────────────────────


def test_returns_dataclass(results: BayesianResults):
    assert isinstance(results, BayesianResults)


# ── difficulty_df ─────────────────────────────────────────────────────


def test_difficulty_df_row_count(results: BayesianResults):
    assert len(results.difficulty_df) == 60


def test_difficulty_df_columns(results: BayesianResults):
    expected = {
        "rule_id",
        "group",
        "group_label",
        "answer",
        "posterior_entropy",
        "top1_probability",
        "top5_probability",
        "n_effective",
        "n_with_all_hits",
        "true_rule_rank",
        "true_rule_posterior_mass",
    }
    assert expected.issubset(set(results.difficulty_df.columns))


def test_group_labels_correct(results: BayesianResults):
    labels = set(results.difficulty_df["group_label"].unique())
    assert labels == {"Easy", "Medium", "Hard"}


def test_group_label_mapping(results: BayesianResults):
    """Each group integer maps to the correct label."""
    df = results.difficulty_df
    for _, row in df.iterrows():
        expected = {1: "Easy", 2: "Medium", 3: "Hard"}[row["group"]]
        assert row["group_label"] == expected


# ── hypotheses_df ─────────────────────────────────────────────────────


def test_hypotheses_df_columns(results: BayesianResults):
    expected = {
        "rule_id",
        "rank",
        "program",
        "probability",
        "n_expressions",
        "extension_size",
        "log_prior",
        "log_likelihood",
        "is_true_rule",
    }
    assert expected.issubset(set(results.hypotheses_df.columns))


def test_hypotheses_df_max_rows(results: BayesianResults):
    # 60 rules * 10 hypotheses each = 600 max
    assert len(results.hypotheses_df) <= 600


def test_hypotheses_rank_starts_at_one(results: BayesianResults):
    assert results.hypotheses_df["rank"].min() == 1


def test_hypotheses_is_true_rule_dtype(results: BayesianResults):
    assert results.hypotheses_df["is_true_rule"].dtype == bool


# ── diagnosticity_df ──────────────────────────────────────────────────


def test_diagnosticity_df_is_dataframe(results: BayesianResults):
    assert isinstance(results.diagnosticity_df, pd.DataFrame)


def test_diagnosticity_df_columns_if_nonempty(results: BayesianResults):
    df = results.diagnosticity_df
    if len(df) > 0:
        expected = {"rule_id", "hand_idx", "agreement_rate", "diagnostic"}
        assert expected.issubset(set(df.columns))


# ── pipeline_stats and config ─────────────────────────────────────────


def test_pipeline_stats_is_dict(results: BayesianResults):
    assert isinstance(results.pipeline_stats, dict)
    assert len(results.pipeline_stats) > 0


def test_config_is_dict(results: BayesianResults):
    assert isinstance(results.config, dict)
    assert "max_depth" in results.config


# ── Run as script ─────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
