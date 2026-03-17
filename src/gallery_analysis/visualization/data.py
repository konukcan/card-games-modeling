"""Data loading and normalization for Bayesian rule-induction results.

Reads the JSON output of ``gallery_analysis/analyze.py`` and reshapes it into
three tidy pandas DataFrames packaged in a :class:`BayesianResults` dataclass.

Also provides :func:`load_depth_decomposition` for loading the depth
decomposition analysis JSON into a :class:`DepthDecompositionResults` dataclass.

Usage::

    from gallery_analysis.visualization.data import load_results

    results = load_results("gallery_analysis/results/depth6_injected.json")
    results.difficulty_df   # 60-row rule-level summary
    results.hypotheses_df   # up to 600 rows (top-10 hypotheses per rule)
    results.diagnosticity_df  # exemplar diagnosticity (may be empty)

    from gallery_analysis.visualization.data import load_depth_decomposition

    dd = load_depth_decomposition("gallery_analysis/results/depth_decomposition_data.json")
    dd.depth_population_df  # one row per depth (global counts)
    dd.depth_rule_df        # one row per (rule, depth) pair
    dd.rule_summary_df      # one row per rule with true_rule_depth
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


def _program_depth(program: str) -> int:
    """Compute AST depth from an S-expression program string.

    Depth is defined as the maximum parenthesis nesting level.
    For example:
        ``(λ not (has_color $0 BLACK))`` → depth 2
        ``(λ not (has_color (take 5 $0) BLACK))`` → depth 3
    """
    max_depth = 0
    current = 0
    for ch in program:
        if ch == "(":
            current += 1
            if current > max_depth:
                max_depth = current
        elif ch == ")":
            current -= 1
    return max_depth


def _build_hypotheses_df(rule_details: Dict[str, Any]) -> pd.DataFrame:
    """Flatten top_hypotheses into a long DataFrame with a rank column.

    Each hypothesis gets a 1-based ``rank`` within its rule and an
    ``is_true_rule`` boolean indicating whether its program string
    matches the rule's ``true_rule_program``.  A ``program_depth``
    column is computed from the max parenthesis nesting of the
    S-expression.
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
                    "program_depth": _program_depth(hyp["program"]),
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


# ══════════════════════════════════════════════════════════════════════
# Depth Decomposition
# ══════════════════════════════════════════════════════════════════════


@dataclass
class DepthDecompositionResults:
    """Container for depth decomposition analysis results.

    Attributes
    ----------
    depth_population_df : pd.DataFrame
        One row per AST depth.  Columns: depth (int), count (int),
        prior_min, prior_max, prior_mean (floats).
    depth_rule_df : pd.DataFrame
        One row per (rule, depth) pair — only depths with at least one
        equivalence class.  Columns: rule_id, group, group_label, depth,
        n_total, n_all_hits, n_any_hits, posterior_mass,
        posterior_mass_allhit_only, mean_log_prior_allhit,
        mean_log_lik_allhit, mean_ext_size_allhit.
    rule_summary_df : pd.DataFrame
        One row per rule.  Columns: rule_id, group, group_label,
        true_rule_depth, true_rule_rank, true_rule_mass.
    metadata : dict
        Raw metadata dict from the JSON.
    """

    depth_population_df: pd.DataFrame
    depth_rule_df: pd.DataFrame
    rule_summary_df: pd.DataFrame
    metadata: Dict[str, Any]


def load_depth_decomposition(path: Union[str, Path]) -> DepthDecompositionResults:
    """Load and normalize a depth decomposition JSON.

    Supports two JSON formats:

    **Old format** (has a ``metadata`` key):
        Uses ``metadata.depth_population`` for population counts and
        ``rule.depth_decomposition`` with detailed per-depth stats.

    **New format** (has a ``provenance`` key, no ``metadata``):
        Uses ``rule.depth_mass`` (depth → posterior mass) and derives
        population counts by aggregating across rules.

    Parameters
    ----------
    path : str or Path
        Path to ``depth_decomposition_data.json``.

    Returns
    -------
    DepthDecompositionResults
    """
    path = Path(path)
    with open(path) as f:
        raw: Dict[str, Any] = json.load(f)

    rules = raw["rules"]

    # Detect format: old format has "metadata", new format has "provenance".
    if "metadata" in raw:
        return _load_depth_decomposition_old(raw, rules)
    else:
        return _load_depth_decomposition_new(raw, rules)


def _load_depth_decomposition_old(
    raw: Dict[str, Any], rules: Dict[str, Any]
) -> DepthDecompositionResults:
    """Load depth decomposition from the old JSON format (has ``metadata``)."""
    metadata = raw["metadata"]

    # ── depth_population_df ───────────────────────────────────────────
    pop_rows: List[Dict[str, Any]] = []
    prior_ranges = metadata.get("depth_prior_ranges", {})
    for depth_str, count in metadata["depth_population"].items():
        d = int(depth_str)
        pr = prior_ranges.get(depth_str, {})
        pop_rows.append({
            "depth": d,
            "count": count,
            "prior_min": pr.get("min"),
            "prior_max": pr.get("max"),
            "prior_mean": pr.get("mean"),
        })
    depth_population_df = pd.DataFrame(pop_rows).sort_values("depth")

    # ── depth_rule_df ─────────────────────────────────────────────────
    dr_rows: List[Dict[str, Any]] = []
    for rule_id, rule in rules.items():
        for depth_str, dd in rule["depth_decomposition"].items():
            dr_rows.append({
                "rule_id": rule["rule_id"],
                "group": rule["group"],
                "group_label": rule["group_label"],
                "depth": int(depth_str),
                "n_total": dd["n_total"],
                "n_all_hits": dd["n_all_hits"],
                "n_any_hits": dd["n_any_hits"],
                "posterior_mass": dd["posterior_mass"],
                "posterior_mass_allhit_only": dd["posterior_mass_allhit_only"],
                "mean_log_prior_allhit": dd["mean_log_prior_allhit"],
                "mean_log_lik_allhit": dd["mean_log_lik_allhit"],
                "mean_ext_size_allhit": dd["mean_ext_size_allhit"],
            })
    depth_rule_df = pd.DataFrame(dr_rows)

    # ── rule_summary_df ───────────────────────────────────────────────
    rs_rows: List[Dict[str, Any]] = []
    for rule_id, rule in rules.items():
        rs_rows.append({
            "rule_id": rule["rule_id"],
            "group": rule["group"],
            "group_label": rule["group_label"],
            "true_rule_depth": rule["true_rule_depth"],
            "true_rule_rank": rule["true_rule_rank"],
            "true_rule_mass": rule["true_rule_mass"],
        })
    rule_summary_df = pd.DataFrame(rs_rows)

    return DepthDecompositionResults(
        depth_population_df=depth_population_df,
        depth_rule_df=depth_rule_df,
        rule_summary_df=rule_summary_df,
        metadata=metadata,
    )


def _load_depth_decomposition_new(
    raw: Dict[str, Any], rules: Dict[str, Any]
) -> DepthDecompositionResults:
    """Load depth decomposition from the new JSON format (has ``provenance``).

    The new format stores ``depth_mass`` (depth → posterior mass) per rule
    instead of the richer ``depth_decomposition`` dict.  Fields that don't
    exist in the new format (n_total, n_all_hits, etc.) are filled with 0.

    ``true_rule_depth`` is derived as the depth carrying the highest mass.
    ``group_label`` is derived from the group number via GROUP_LABELS.
    """
    metadata = raw.get("provenance", {})

    # ── depth_rule_df ─────────────────────────────────────────────────
    # One row per (rule, depth) pair.  Only depth_mass is available in
    # the new format; fill missing detail columns with 0.
    dr_rows: List[Dict[str, Any]] = []
    for rule_id, rule in rules.items():
        group = rule["group"]
        group_label = GROUP_LABELS.get(group, f"Group {group}")
        for depth_str, mass in rule.get("depth_mass", {}).items():
            dr_rows.append({
                "rule_id": rule_id,
                "group": group,
                "group_label": group_label,
                "depth": int(depth_str),
                "n_total": 0,
                "n_all_hits": 0,
                "n_any_hits": 0,
                "posterior_mass": mass,
                "posterior_mass_allhit_only": 0.0,
                "mean_log_prior_allhit": 0.0,
                "mean_log_lik_allhit": 0.0,
                "mean_ext_size_allhit": 0.0,
            })
    depth_rule_df = pd.DataFrame(dr_rows)

    # ── depth_population_df ───────────────────────────────────────────
    # Derive population by counting how many rules have non-negligible
    # mass (> 1e-12) at each depth.  Prior range stats are unavailable
    # in the new format, so they are set to None.
    depth_counts: Dict[int, int] = {}
    for rule_id, rule in rules.items():
        for depth_str, mass in rule.get("depth_mass", {}).items():
            d = int(depth_str)
            if mass > 1e-12:
                depth_counts[d] = depth_counts.get(d, 0) + 1
    pop_rows: List[Dict[str, Any]] = []
    for d, count in sorted(depth_counts.items()):
        pop_rows.append({
            "depth": d,
            "count": count,
            "prior_min": None,
            "prior_max": None,
            "prior_mean": None,
        })
    depth_population_df = pd.DataFrame(pop_rows).sort_values("depth")

    # ── rule_summary_df ───────────────────────────────────────────────
    # Derive true_rule_depth as the depth with the highest mass.
    rs_rows: List[Dict[str, Any]] = []
    for rule_id, rule in rules.items():
        group = rule["group"]
        group_label = GROUP_LABELS.get(group, f"Group {group}")
        # Find depth with highest mass to use as true_rule_depth.
        depth_mass = rule.get("depth_mass", {})
        if depth_mass:
            best_depth_str = max(depth_mass, key=lambda k: depth_mass[k])
            true_rule_depth = int(best_depth_str)
        else:
            true_rule_depth = None
        rs_rows.append({
            "rule_id": rule_id,
            "group": group,
            "group_label": group_label,
            "true_rule_depth": true_rule_depth,
            "true_rule_rank": rule.get("true_rule_rank"),
            "true_rule_mass": rule.get("true_rule_mass"),
        })
    rule_summary_df = pd.DataFrame(rs_rows)

    return DepthDecompositionResults(
        depth_population_df=depth_population_df,
        depth_rule_df=depth_rule_df,
        rule_summary_df=rule_summary_df,
        metadata=metadata,
    )


# ══════════════════════════════════════════════════════════════════════
# Diagnosticity Spectrums
# ══════════════════════════════════════════════════════════════════════


@dataclass
class DiagnosticityResults:
    """Container for diagnosticity spectrum analysis results.

    Attributes
    ----------
    spectrum_df : pd.DataFrame
        One row per rule with scalar metrics: rule_id, group,
        mean_p_accept, std_p_accept, mean_confidence,
        fraction_high_confidence, fraction_ambiguous, accuracy.
    histogram_data : dict
        Dict keyed by rule_id → list of dicts with ``bin`` (str) and
        ``count`` (int) for the P(accept) histogram.
    representative_hands : dict
        Dict keyed by rule_id → dict with keys ``easy_accept``,
        ``easy_reject``, ``ambiguous``, each a list of hand dicts
        containing card data and metrics.
    config : dict
        Run configuration from the JSON.
    """

    spectrum_df: pd.DataFrame
    histogram_data: Dict[str, List[Dict[str, Any]]]
    representative_hands: Dict[str, Dict[str, List[Dict[str, Any]]]]
    config: Dict[str, Any]


def load_diagnosticity_spectrums(path: Union[str, Path]) -> DiagnosticityResults:
    """Load and normalize a diagnosticity spectrum JSON.

    Parameters
    ----------
    path : str or Path
        Path to a ``diagnosticity_*.json`` file.

    Returns
    -------
    DiagnosticityResults
    """
    path = Path(path)
    with open(path) as f:
        raw: Dict[str, Any] = json.load(f)

    config = raw.get("config", {})
    spectrums = raw["spectrums"]

    # ── spectrum_df ───────────────────────────────────────────────────
    # One row per rule with scalar metrics.
    rows: List[Dict[str, Any]] = []
    for rule_id, spec in spectrums.items():
        rows.append({
            "rule_id": spec["rule_id"],
            "group": spec["group"],
            "group_label": GROUP_LABELS[spec["group"]],
            "n_candidates": spec["n_candidates"],
            "mean_p_accept": spec["mean_p_accept"],
            "std_p_accept": spec["std_p_accept"],
            "mean_confidence": spec["mean_confidence"],
            "fraction_high_confidence": spec["fraction_high_confidence"],
            "fraction_ambiguous": spec["fraction_ambiguous"],
            "accuracy": spec["accuracy"],
        })
    spectrum_df = pd.DataFrame(rows)

    # ── histogram_data ────────────────────────────────────────────────
    # Convert the histogram dict to a list of {bin, count} for Altair.
    histogram_data: Dict[str, List[Dict[str, Any]]] = {}
    for rule_id, spec in spectrums.items():
        hist_list = []
        for bin_label, count in spec["p_accept_histogram"].items():
            hist_list.append({"bin": bin_label, "count": count})
        histogram_data[rule_id] = hist_list

    # ── representative_hands ──────────────────────────────────────────
    # Extract the three categories of representative hands per rule.
    representative_hands: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
    for rule_id, spec in spectrums.items():
        representative_hands[rule_id] = {
            "easy_accept": spec.get("easy_accept_hands", []),
            "easy_reject": spec.get("easy_reject_hands", []),
            "ambiguous": spec.get("ambiguous_hands", []),
        }

    return DiagnosticityResults(
        spectrum_df=spectrum_df,
        histogram_data=histogram_data,
        representative_hands=representative_hands,
        config=config,
    )


def build_calibration_df(
    diag_results: DiagnosticityResults,
    difficulty_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build calibration data from representative hands.

    Pools representative hands (easy_accept, easy_reject, ambiguous) across
    all rules, bins them by P(accept), and computes the observed acceptance
    rate within each (bin, difficulty-group) combination.

    Parameters
    ----------
    diag_results : DiagnosticityResults
        Loaded diagnosticity results containing ``representative_hands``
        and ``spectrum_df``.
    difficulty_df : pd.DataFrame
        Rule-level difficulty DataFrame (needs ``rule_id`` and
        ``group_label`` columns).

    Returns
    -------
    pd.DataFrame
        Columns: ``bin_center``, ``observed_rate``, ``group_label``,
        ``n_hands``.  Empty DataFrame (with those columns) when no
        representative hands are available.
    """
    import numpy as np

    # Map rule_id -> group_label from difficulty_df.
    group_map = difficulty_df.set_index("rule_id")["group_label"].to_dict()

    # Collect all representative hands across all rules.
    rows: List[Dict[str, Any]] = []
    for rule_id, categories in diag_results.representative_hands.items():
        label = group_map.get(rule_id)
        if label is None:
            continue
        for cat_name in ("easy_accept", "easy_reject", "ambiguous"):
            for hand in categories.get(cat_name, []):
                p_accept = hand.get("p_accept")
                gt = hand.get("ground_truth")
                if p_accept is None or gt is None:
                    continue
                rows.append({
                    "p_accept": p_accept,
                    "ground_truth": bool(gt),
                    "group_label": label,
                })

    empty = pd.DataFrame(columns=["bin_center", "observed_rate", "group_label", "n_hands"])
    if not rows:
        return empty

    hands_df = pd.DataFrame(rows)

    # Bin P(accept) into 10 equal-width bins [0, 0.1), [0.1, 0.2), ..., [0.9, 1.0].
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    hands_df["bin_idx"] = np.clip(
        np.digitize(hands_df["p_accept"], bins) - 1, 0, 9
    )
    hands_df["bin_center"] = hands_df["bin_idx"].map(
        lambda i: bin_centers[i]
    )

    # Aggregate: observed acceptance rate per (bin, group).
    agg = (
        hands_df.groupby(["bin_center", "group_label"])
        .agg(
            observed_rate=("ground_truth", "mean"),
            n_hands=("ground_truth", "count"),
        )
        .reset_index()
    )

    return agg
