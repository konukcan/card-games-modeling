"""Data loading and normalization for Bayesian rule-induction results.

Reads the JSON output of ``gallery_analysis/analyze.py`` and reshapes it into
three tidy pandas DataFrames packaged in a :class:`BayesianResults` dataclass.

Usage::

    from gallery_analysis.visualization.data import load_results

    results = load_results("gallery_analysis/results/depth6_injected.json")
    results.difficulty_df   # 60-row rule-level summary
    results.hypotheses_df   # up to 600 rows (top-10 hypotheses per rule)
    results.diagnosticity_df  # exemplar diagnosticity (may be empty)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

# Import the canonical group-label mapping from the shared theme module.
# Kept as a local import so the module can also be used standalone with a
# fallback when the shared package is not on sys.path.
try:
    from shared.theme import GROUP_LABELS
except ImportError:
    GROUP_LABELS = {1: "Easy", 2: "Medium", 3: "Hard"}


# ── Output dataclass ─────────────────────────────────────────────────


@dataclass
class BayesianResults:
    """Container for normalized Bayesian analysis results.

    Attributes
    ----------
    difficulty_df : pd.DataFrame
        One row per rule (60 rows).  Columns include rule_id, group,
        group_label, answer, and all scalar difficulty metrics.
    hypotheses_df : pd.DataFrame
        Top-k hypotheses per rule (up to 600 rows).  Includes a ``rank``
        column (1-based) and an ``is_true_rule`` boolean flag.
    diagnosticity_df : pd.DataFrame
        Exemplar diagnosticity per rule.  May be empty if no rules have
        diagnosticity data.
    pipeline_stats : dict
        Raw pipeline statistics from the JSON (enumeration, filtering, etc.).
    config : dict
        Run configuration (max_depth, max_programs, etc.).
    """

    difficulty_df: pd.DataFrame
    hypotheses_df: pd.DataFrame
    diagnosticity_df: pd.DataFrame
    pipeline_stats: Dict[str, Any]
    config: Dict[str, Any]


# ── Public API ────────────────────────────────────────────────────────


def load_results(path: Union[str, Path]) -> BayesianResults:
    """Load and normalize a Bayesian analysis results JSON.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file produced by ``analyze.py --output``.

    Returns
    -------
    BayesianResults
        Normalized DataFrames and raw metadata dicts.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If the JSON is missing expected top-level keys.
    """
    path = Path(path)
    with open(path) as f:
        raw: Dict[str, Any] = json.load(f)

    # Unpack the four top-level sections.
    pipeline_stats: Dict[str, Any] = raw["pipeline_stats"]
    config: Dict[str, Any] = raw["config"]
    rule_details: Dict[str, Any] = raw["rule_details"]

    # Build each DataFrame from rule_details.
    difficulty_df = _build_difficulty_df(rule_details)
    hypotheses_df = _build_hypotheses_df(rule_details)
    diagnosticity_df = _build_diagnosticity_df(rule_details)

    return BayesianResults(
        difficulty_df=difficulty_df,
        hypotheses_df=hypotheses_df,
        diagnosticity_df=diagnosticity_df,
        pipeline_stats=pipeline_stats,
        config=config,
    )


# ── Internal helpers ──────────────────────────────────────────────────


def _build_difficulty_df(rule_details: Dict[str, Any]) -> pd.DataFrame:
    """Extract one row per rule with scalar difficulty metrics.

    Columns: rule_id, group, group_label, answer, posterior_entropy,
    top1_probability, top5_probability, n_effective, n_with_all_hits,
    true_rule_rank, true_rule_posterior_mass.
    """
    rows: List[Dict[str, Any]] = []
    for rule_id, detail in rule_details.items():
        diff = detail.get("difficulty", {})
        rows.append(
            {
                "rule_id": detail["rule_id"],
                "group": detail["group"],
                "group_label": GROUP_LABELS[detail["group"]],
                "answer": detail["answer"],
                # Difficulty metrics (from the nested "difficulty" dict).
                "posterior_entropy": diff.get("posterior_entropy"),
                "top1_probability": diff.get("top1_probability"),
                "top5_probability": diff.get("top5_probability"),
                "n_effective": diff.get("n_effective_hypotheses"),
                # Counts from the rule-level fields.
                "n_with_all_hits": detail.get("n_with_all_hits"),
                # True-rule diagnostics (may be null).
                "true_rule_rank": detail.get("true_rule_rank"),
                "true_rule_posterior_mass": detail.get("true_rule_posterior_mass"),
            }
        )
    return pd.DataFrame(rows)


def _build_hypotheses_df(rule_details: Dict[str, Any]) -> pd.DataFrame:
    """Flatten top_hypotheses into a long DataFrame with a rank column.

    Each hypothesis gets a 1-based ``rank`` within its rule and an
    ``is_true_rule`` boolean indicating whether its program string
    matches the rule's ``true_rule_program``.
    """
    rows: List[Dict[str, Any]] = []
    for rule_id, detail in rule_details.items():
        true_program: Optional[str] = detail.get("true_rule_program")
        for rank, hyp in enumerate(detail.get("top_hypotheses", []), start=1):
            rows.append(
                {
                    "rule_id": detail["rule_id"],
                    "rank": rank,
                    "program": hyp["program"],
                    "probability": hyp["probability"],
                    "n_expressions": hyp["n_expressions"],
                    "extension_size": hyp["extension_size"],
                    "log_prior": hyp["log_prior"],
                    "log_likelihood": hyp["log_likelihood"],
                    "is_true_rule": (
                        hyp["program"] == true_program
                        if true_program is not None
                        else False
                    ),
                }
            )
    return pd.DataFrame(rows)


def _build_diagnosticity_df(rule_details: Dict[str, Any]) -> pd.DataFrame:
    """Build a long DataFrame of exemplar diagnosticity entries.

    Returns an empty DataFrame (with correct column names) when no rule
    has diagnosticity data.
    """
    rows: List[Dict[str, Any]] = []
    for rule_id, detail in rule_details.items():
        diag_list = detail.get("exemplar_diagnosticity")
        if not diag_list:
            continue
        for entry in diag_list:
            rows.append(
                {
                    "rule_id": detail["rule_id"],
                    "hand_idx": entry["hand_idx"],
                    "agreement_rate": entry["agreement_rate"],
                    "diagnostic": entry["diagnostic"],
                }
            )
    if not rows:
        return pd.DataFrame(columns=["rule_id", "hand_idx", "agreement_rate", "diagnostic"])
    return pd.DataFrame(rows)
