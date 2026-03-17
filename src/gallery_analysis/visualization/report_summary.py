"""Summary report generator for the Bayesian rule-induction visualization.

Wires together the data layer, Altair plot functions, and the summary.html
Jinja2 template to produce a self-contained HTML index page.

Usage::

    from gallery_analysis.visualization.data import load_results
    from gallery_analysis.visualization.report_summary import generate_summary

    results = load_results("gallery_analysis/results/depth6_injected.json")
    path = generate_summary(results, Path("output"))
"""

from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

# Local visualization imports — follow the same try/except pattern used
# by the sibling modules (data.py, plots.py, cards.py).
try:
    from gallery_analysis.visualization.data import (
        BayesianResults,
        DepthDecompositionResults,
        DiagnosticityResults,
        build_calibration_df,
    )
    from gallery_analysis.visualization.plots import (
        difficulty_strip,
        difficulty_scatter,
        true_rule_recovery,
        equiv_class_bars,
        depth_population,
        depth_vs_difficulty,
        depth_posterior_heatmap,
        depth_prior_range,
        diagnosticity_overview_scatter,
        entropy_vs_accuracy,
        calibration_plot,
    )
except ImportError:
    # Fallback for direct execution: add parent packages to sys.path.
    import sys

    _this_dir = Path(__file__).resolve().parent
    _src_dir = _this_dir.parent.parent
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))

    from gallery_analysis.visualization.data import (
        BayesianResults,
        DepthDecompositionResults,
        DiagnosticityResults,
        build_calibration_df,
    )
    from gallery_analysis.visualization.plots import (
        difficulty_strip,
        difficulty_scatter,
        true_rule_recovery,
        equiv_class_bars,
        depth_population,
        depth_vs_difficulty,
        depth_posterior_heatmap,
        depth_prior_range,
        diagnosticity_overview_scatter,
        entropy_vs_accuracy,
        calibration_plot,
    )


# ── Template directory ───────────────────────────────────────────────

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


# ── Public API ───────────────────────────────────────────────────────


def generate_summary(
    results: BayesianResults,
    output_dir: Path,
    depth_results: DepthDecompositionResults | None = None,
    diag_results: DiagnosticityResults | None = None,
    variant_info: dict | None = None,
) -> Path:
    """Generate the summary index.html from Bayesian analysis results.

    Builds four Altair charts (strip, scatter, recovery, equivalence-class),
    plus optional depth decomposition and diagnosticity charts, serializes
    them as Vega-Lite JSON specs, and renders the summary.html template.

    Parameters
    ----------
    results : BayesianResults
        Loaded and normalized results from :func:`data.load_results`.
    output_dir : Path
        Directory to write ``index.html`` into.  Created if it does not exist.
    depth_results : DepthDecompositionResults, optional
        Depth decomposition data.
    diag_results : DiagnosticityResults, optional
        Diagnosticity spectrum data.
    variant_info : dict, optional
        Variant metadata for the dropdown switcher.  Keys:
        ``variant_name``, ``variant_label``, ``all_variants`` (list of
        dicts with ``name``, ``label``, ``path``, ``has_diag``).

    Returns
    -------
    Path
        Absolute path to the generated ``index.html``.
    """
    # Set up Jinja2 environment with the templates directory.
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=False,
    )

    # Register a custom filter for formatting large integers with commas.
    # Example: 1234567 -> "1,234,567"
    env.filters["format_number"] = _format_number

    template = env.get_template("summary.html")

    # ── Build Altair chart specs as JSON strings ─────────────────────
    df = results.difficulty_df

    chart_strip = json.dumps(difficulty_strip(df).to_dict())
    chart_scatter = json.dumps(difficulty_scatter(df).to_dict())
    chart_recovery = json.dumps(true_rule_recovery(df).to_dict())
    chart_equiv = json.dumps(equiv_class_bars(df).to_dict())

    # ── Depth decomposition charts (optional) ────────────────────────
    depth_charts = {}
    if depth_results is not None:
        depth_charts["chart_depth_pop"] = json.dumps(
            depth_population(depth_results.depth_population_df).to_dict()
        )
        depth_charts["chart_depth_prior"] = json.dumps(
            depth_prior_range(depth_results.depth_population_df).to_dict()
        )
        # Merge rule_summary_df with difficulty_df to get entropy + depth.
        merged = depth_results.rule_summary_df.merge(
            df[["rule_id", "posterior_entropy"]], on="rule_id", how="left"
        )
        depth_charts["chart_depth_vs_diff"] = json.dumps(
            depth_vs_difficulty(merged).to_dict()
        )
        depth_charts["chart_depth_heatmap"] = json.dumps(
            depth_posterior_heatmap(
                depth_results.depth_rule_df, depth_results.rule_summary_df
            ).to_dict()
        )

    # ── Diagnosticity charts (optional) ─────────────────────────────
    diag_charts = {}
    if diag_results is not None:
        diag_charts["chart_diag_overview"] = json.dumps(
            diagnosticity_overview_scatter(diag_results.spectrum_df).to_dict()
        )
        # Entropy vs accuracy scatter — merge diagnosticity metrics
        # with difficulty metrics.
        diag_merged = diag_results.spectrum_df.merge(
            df[["rule_id", "posterior_entropy"]], on="rule_id", how="inner"
        )
        if len(diag_merged) > 0:
            diag_charts["chart_accuracy"] = json.dumps(
                entropy_vs_accuracy(diag_merged).to_dict()
            )
        # Calibration plot from representative hands.
        cal_df = build_calibration_df(diag_results, df)
        if len(cal_df) > 0:
            diag_charts["chart_calibration"] = json.dumps(
                calibration_plot(cal_df).to_dict()
            )

    # ── Build the rules list for the index table ─────────────────────
    # Optionally enrich with true_rule_depth if depth data is available.
    if depth_results is not None:
        depth_map = depth_results.rule_summary_df.set_index("rule_id")[
            "true_rule_depth"
        ].to_dict()
        df = df.copy()
        df["true_rule_depth"] = df["rule_id"].map(depth_map)

    # Optionally enrich with diagnosticity metrics.
    if diag_results is not None:
        diag_map = diag_results.spectrum_df.set_index("rule_id")[
            ["mean_confidence", "fraction_ambiguous", "accuracy"]
        ].to_dict("index")
        if "true_rule_depth" not in df.columns:
            df = df.copy()
        for col in ("mean_confidence", "fraction_ambiguous", "accuracy"):
            df[col] = df["rule_id"].map(
                lambda rid, c=col: diag_map.get(rid, {}).get(c)
            )

    rules = (
        df.sort_values("posterior_entropy", ascending=False)
        .to_dict("records")
    )

    # ── Render the template ──────────────────────────────────────────
    variant_ctx = {}
    if variant_info:
        variant_ctx = {
            "variant_name": variant_info.get("variant_name", ""),
            "variant_label": variant_info.get("variant_label", ""),
            "all_variants": variant_info.get("all_variants", []),
        }

    html = template.render(
        stats=results.pipeline_stats,
        chart_strip=chart_strip,
        chart_scatter=chart_scatter,
        chart_recovery=chart_recovery,
        chart_equiv=chart_equiv,
        has_depth=depth_results is not None,
        has_diag=diag_results is not None,
        rules=rules,
        **depth_charts,
        **diag_charts,
        **variant_ctx,
    )

    # ── Write to disk ────────────────────────────────────────────────
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "index.html"
    out_path.write_text(html, encoding="utf-8")

    return out_path


# ── Private helpers ──────────────────────────────────────────────────


def _format_number(value: object) -> str:
    """Jinja2 filter: format an integer with comma separators.

    Handles ints, floats (truncated to int), and strings that look numeric.
    Non-numeric values are returned unchanged.

    Examples:
        1234567  -> "1,234,567"
        42       -> "42"
        "hello"  -> "hello"
    """
    try:
        return f"{int(value):,}"
    except (ValueError, TypeError):
        return str(value)
