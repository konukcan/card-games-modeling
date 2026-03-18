"""Pure Altair chart functions for Bayesian rule-induction analysis.

Each function takes a pandas DataFrame (or a subset thereof) and returns an
Altair chart object.  No I/O, no side effects — all rendering decisions are
deferred to the caller.

Summary-level plots consume ``difficulty_df`` (one row per rule).
Per-rule plots consume a filtered slice of ``hypotheses_df`` or
``diagnosticity_df``.

Usage::

    from gallery_analysis.visualization.data import load_results
    from gallery_analysis.visualization.plots import difficulty_scatter

    results = load_results("gallery_analysis/results/depth6_injected.json")
    chart = difficulty_scatter(results.difficulty_df)
    chart.save("difficulty_scatter.html")
"""

from __future__ import annotations

import altair as alt
import pandas as pd

# Import shared theme helpers.  Fallback gracefully if the shared package
# is not on sys.path (mirrors the pattern in data.py).
try:
    from shared.theme import difficulty_color_scale, DIFFICULTY_COLORS
except ImportError:
    # Minimal fallback so the module can be imported standalone.
    DIFFICULTY_COLORS = {"Easy": "#4A90D9", "Medium": "#D4A029", "Hard": "#C44E52"}

    def difficulty_color_scale() -> alt.Scale:
        return alt.Scale(
            domain=["Easy", "Medium", "Hard"],
            range=[DIFFICULTY_COLORS[k] for k in ("Easy", "Medium", "Hard")],
        )


# ── Highlight color for true-rule markers ─────────────────────────────

_TRUE_RULE_COLOR = "#2CA02C"  # distinguishable green


# ══════════════════════════════════════════════════════════════════════
# Summary-level charts (take difficulty_df)
# ══════════════════════════════════════════════════════════════════════


def difficulty_scatter(df: pd.DataFrame) -> alt.LayerChart:
    """Scatter of posterior entropy vs true-rule posterior mass with trend line.

    Shows how much probability the correct rule captures as a function of
    overall posterior uncertainty.  Colored by difficulty group.  A loess
    trend line shows the empirical relationship — outliers above the trend
    recover the true rule better than expected; below, worse.

    Parameters
    ----------
    df : pd.DataFrame
        ``difficulty_df`` from :func:`data.load_results`.
    """
    import numpy as np
    from scipy.stats import spearmanr

    # Filter out rules with zero or missing true-rule mass.
    plot_df = df.dropna(subset=["true_rule_posterior_mass"]).copy()
    plot_df = plot_df[plot_df["true_rule_posterior_mass"] > 0].copy()

    # Spearman rank correlation between entropy and log10(mass).
    plot_df["log10_mass"] = np.log10(plot_df["true_rule_posterior_mass"])
    rho, pval = spearmanr(plot_df["posterior_entropy"], plot_df["log10_mass"])

    # Floor/ceiling for log scale with padding.
    min_mass = plot_df["true_rule_posterior_mass"].min()
    max_entropy = plot_df["posterior_entropy"].max()
    floor = min_mass * 0.05   # ~1.3 decades below the smallest value
    x_pad = max_entropy * 0.05

    points = (
        alt.Chart(plot_df)
        .mark_circle(size=60, opacity=0.8)
        .encode(
            x=alt.X(
                "posterior_entropy:Q",
                title="Posterior Entropy (bits)",
                scale=alt.Scale(domain=[-x_pad, max_entropy + x_pad]),
            ),
            y=alt.Y(
                "true_rule_posterior_mass:Q",
                title="True Rule Posterior Mass (log scale)",
                scale=alt.Scale(type="log", domain=[floor, 2]),
            ),
            color=alt.Color(
                "group_label:N",
                title="Difficulty",
                scale=difficulty_color_scale(),
            ),
            tooltip=[
                alt.Tooltip("rule_id:N", title="Rule"),
                alt.Tooltip("posterior_entropy:Q", title="Entropy", format=".3f"),
                alt.Tooltip("true_rule_posterior_mass:Q", title="True Rule Mass", format=".2e"),
                alt.Tooltip("true_rule_rank:Q", title="True Rule Rank"),
            ],
        )
    )

    # Loess trend line on log-transformed mass.
    trend = (
        alt.Chart(plot_df)
        .transform_loess(
            "posterior_entropy", "log10_mass",
            bandwidth=0.5,
        )
        .transform_calculate(
            mass_fitted="pow(10, datum.log10_mass)"
        )
        .mark_line(strokeDash=[6, 3], color="#666", strokeWidth=1.5)
        .encode(
            x=alt.X("posterior_entropy:Q"),
            y=alt.Y("mass_fitted:Q", scale=alt.Scale(type="log", domain=[floor, 2])),
        )
    )

    # Annotation: Spearman rho and p-value.
    pval_str = f"p < 0.001" if pval < 0.001 else f"p = {pval:.3f}"
    annotation = (
        alt.Chart(pd.DataFrame({
            "x": [max_entropy * 0.95],
            "y": [1.0],
            "text": [f"Spearman ρ = {rho:.2f}, {pval_str}"],
        }))
        .mark_text(align="right", fontSize=11, color="#555")
        .encode(x="x:Q", y="y:Q", text="text:N")
    )

    return (
        (points + trend + annotation)
        .properties(
            width=500,
            height=400,
            title="Entropy vs True Rule Posterior Mass",
        )
    )


# ══════════════════════════════════════════════════════════════════════
# Depth decomposition charts (take DataFrames from load_depth_decomposition)
# ══════════════════════════════════════════════════════════════════════


def depth_posterior_heatmap(depth_rule_df: pd.DataFrame,
                           rule_summary_df: pd.DataFrame) -> alt.Chart:
    """Heatmap of posterior mass by rule × depth.

    Rules on y-axis (sorted by true-rule depth then rule_id), depths on
    x-axis, cell color = log10(posterior mass).  Shows where each rule's
    posterior concentrates across depths.

    Parameters
    ----------
    depth_rule_df : pd.DataFrame
        ``depth_rule_df`` from :func:`data.load_depth_decomposition`.
    rule_summary_df : pd.DataFrame
        ``rule_summary_df`` for sorting order.
    """
    import numpy as np

    # Filter to depths 1-6 (where meaningful mass exists) and positive mass.
    plot_df = depth_rule_df[
        (depth_rule_df["depth"] <= 6) & (depth_rule_df["posterior_mass"] > 0)
    ].copy()
    plot_df["log10_mass"] = np.log10(plot_df["posterior_mass"])

    # Sort rules by true_rule_depth (ascending) then rule_id.
    sort_order = (
        rule_summary_df
        .sort_values(["true_rule_depth", "rule_id"], ascending=[True, True])
        ["rule_id"].tolist()
    )

    return (
        alt.Chart(plot_df)
        .mark_rect()
        .encode(
            x=alt.X("depth:O", title="AST Depth"),
            y=alt.Y(
                "rule_id:N",
                title="Rule",
                sort=sort_order,
                axis=alt.Axis(labelLimit=200),
            ),
            color=alt.Color(
                "log10_mass:Q",
                title="log₁₀(mass)",
                scale=alt.Scale(scheme="viridis"),
            ),
            tooltip=[
                alt.Tooltip("rule_id:N", title="Rule"),
                alt.Tooltip("depth:O", title="Depth"),
                alt.Tooltip("posterior_mass:Q", title="Posterior Mass", format=".2e"),
                alt.Tooltip("n_all_hits:Q", title="All-Hit Classes"),
                alt.Tooltip("n_total:Q", title="Total Classes"),
            ],
        )
        .properties(
            width=350,
            height=800,
            title="Posterior Mass by Rule × Depth",
        )
    )


# ══════════════════════════════════════════════════════════════════════
# Diagnosticity spectrum charts
# ══════════════════════════════════════════════════════════════════════


def p_accept_histogram(histogram_data: list, rule_id: str) -> alt.Chart:
    """Horizontal bar chart of P(accept) distribution across 10 bins.

    Shows how random hands distribute across P(accept) bins for a single
    rule.  Hands clustered near 0 and 1 indicate a confident model; mass
    in the middle indicates ambiguity.

    Parameters
    ----------
    histogram_data : list of dict
        List of ``{"bin": "0.0-0.1", "count": 9902}`` entries for one rule.
    rule_id : str
        Rule identifier (used in chart title).
    """
    df = pd.DataFrame(histogram_data)

    # Ensure bins are ordered correctly (0.0-0.1 at bottom, 0.9-1.0 at top).
    bin_order = [
        "0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5",
        "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0",
    ]

    return (
        alt.Chart(df)
        .mark_bar(color="#5B8DB8")
        .encode(
            x=alt.X("count:Q", title="Number of Hands"),
            y=alt.Y(
                "bin:N",
                title="P(accept)",
                sort=bin_order,
            ),
            tooltip=[
                alt.Tooltip("bin:N", title="P(accept) bin"),
                alt.Tooltip("count:Q", title="Count", format=","),
            ],
        )
        .properties(
            width=400,
            height=250,
            title=f"P(accept) Distribution — {rule_id}",
        )
    )


def p_accept_ground_truth(
    gt_hist: dict,
    rule_id: str,
    title: str = "",
    show_legend: bool = True,
) -> alt.Chart:
    """Stacked bar chart of P(accept) bins split by ground truth.

    Each bin is split into green (true accept) and red (true reject)
    segments, revealing whether the model is overly permissive or
    restrictive at each confidence level.

    Parameters
    ----------
    gt_hist : dict
        Mapping of bin label (e.g. "0.0-0.1") to
        ``{"true_accept": count, "true_reject": count}``.
    rule_id : str
        Rule identifier (used in chart title if *title* is empty).
    title : str, optional
        Override chart title.  Defaults to
        "P(accept) Ground Truth -- <rule_id>".
    """
    if not gt_hist:
        # Return an empty chart when data is missing.
        return alt.Chart(pd.DataFrame({"x": []})).mark_point()

    bin_order = [
        "0.0-0.1", "0.1-0.2", "0.2-0.3", "0.3-0.4", "0.4-0.5",
        "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0",
    ]

    # Build long-form DataFrame: two rows per bin (Accept + Reject).
    rows = []
    for bin_label in bin_order:
        counts = gt_hist.get(bin_label, {"true_accept": 0, "true_reject": 0})
        rows.append({
            "bin": bin_label,
            "component": "Accept",
            "count": counts.get("true_accept", 0),
        })
        rows.append({
            "bin": bin_label,
            "component": "Reject",
            "count": counts.get("true_reject", 0),
        })
    df = pd.DataFrame(rows)

    gt_scale = alt.Scale(
        domain=["Accept", "Reject"],
        range=["#2CA02C", "#C44E52"],
    )

    chart_title = title or f"P(accept) Ground Truth — {rule_id}"

    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("count:Q", title="Number of Hands", stack="zero"),
            y=alt.Y("bin:N", title="P(accept)", sort=bin_order),
            color=alt.Color(
                "component:N",
                title="Ground Truth",
                scale=gt_scale,
                legend=alt.Legend() if show_legend else None,
            ),
            tooltip=[
                alt.Tooltip("bin:N", title="P(accept) bin"),
                alt.Tooltip("component:N", title="Ground Truth"),
                alt.Tooltip("count:Q", title="Count", format=","),
            ],
        )
        .properties(
            width=300,
            height=250,
            title=chart_title,
        )
    )


# ══════════════════════════════════════════════════════════════════════
# Per-rule charts (take a filtered slice of hypotheses_df / diagnosticity_df)
# ══════════════════════════════════════════════════════════════════════


def posterior_decomposition(hyp_df: pd.DataFrame, rule_id: str) -> alt.Chart:
    """Stacked horizontal bars showing posterior = prior + likelihood contributions.

    Each hypothesis gets one bar with total width = posterior probability,
    split into prior contribution (blue) and likelihood contribution (amber).
    The true rule bar gets a green border.  Y-axis labels are natural language
    translations of the DSL programs.

    Parameters
    ----------
    hyp_df : pd.DataFrame
        Filtered slice of ``hypotheses_df`` for a single rule.
    rule_id : str
        The rule identifier (used in the chart title).
    """
    from gallery_analysis.visualization.dsl_translator import translate_dsl

    plot_df = hyp_df.copy()

    # Compute prior/likelihood share of each hypothesis's posterior.
    # |log_prior| and |log_likelihood| measure how much each factor
    # "contributes" to the unnormalised log-posterior.  We turn those into
    # proportional widths of the posterior bar.
    abs_lp = plot_df["log_prior"].abs()
    abs_ll = plot_df["log_likelihood"].abs()
    denom = abs_lp + abs_ll
    # Guard against division by zero (both logs are 0 -- extremely unlikely
    # but handle gracefully by splitting 50/50).
    denom = denom.replace(0, 1)

    prior_share = abs_lp / denom
    likelihood_share = 1 - prior_share

    plot_df["prior_width"] = plot_df["probability"] * prior_share
    plot_df["likelihood_width"] = plot_df["probability"] * likelihood_share

    # Two-line label: NL translation + DSL program on separate lines.
    # Vega-Lite renders \n in axis labels as line breaks.
    labels: list[str] = []
    seen: dict[str, int] = {}
    for _, row in plot_df.iterrows():
        nl = translate_dsl(row["program"])[:55]
        dsl = row["program"][:65]
        lbl = nl + "\n" + dsl
        if lbl in seen:
            seen[lbl] += 1
            lbl = nl + " (" + str(seen[lbl]) + ")\n" + dsl
        else:
            seen[lbl] = 1
        labels.append(lbl)
    plot_df["program_label"] = labels

    # Build long-form DataFrame: two rows per hypothesis (Prior + Likelihood).
    rows = []
    for _, row in plot_df.iterrows():
        base = {
            "program_label": row["program_label"],
            "is_true_rule": row["is_true_rule"],
            "probability": row["probability"],
        }
        rows.append({
            **base,
            "component": "Prior",
            "start": 0,
            "end": row["prior_width"],
        })
        rows.append({
            **base,
            "component": "Likelihood",
            "start": row["prior_width"],
            "end": row["probability"],
        })
    long_df = pd.DataFrame(rows)

    # Sort order: highest posterior at top.
    sorted_labels = (
        plot_df.sort_values("probability", ascending=False)["program_label"].tolist()
    )

    component_scale = alt.Scale(
        domain=["Prior", "Likelihood"],
        range=["#4A90D9", "#D4A029"],
    )

    bars = (
        alt.Chart(long_df)
        .mark_bar()
        .encode(
            x=alt.X("start:Q", title="Posterior Probability"),
            x2=alt.X2("end:Q"),
            y=alt.Y(
                "program_label:N",
                title=None,
                sort=sorted_labels,
                axis=alt.Axis(
                    labelLimit=450,
                    labelExpr="split(datum.label, '\\n')",
                ),
            ),
            color=alt.Color(
                "component:N",
                title="Component",
                scale=component_scale,
            ),
            stroke=alt.condition(
                alt.datum.is_true_rule,
                alt.value(_TRUE_RULE_COLOR),
                alt.value("transparent"),
            ),
            strokeWidth=alt.condition(
                alt.datum.is_true_rule,
                alt.value(2),
                alt.value(0),
            ),
            tooltip=[
                alt.Tooltip("program_label:N", title="Hypothesis"),
                alt.Tooltip("component:N", title="Component"),
                alt.Tooltip("probability:Q", title="Posterior", format=".4f"),
                alt.Tooltip("start:Q", title="Segment Start", format=".4f"),
                alt.Tooltip("end:Q", title="Segment End", format=".4f"),
            ],
        )
    )

    # Text labels showing exact posterior at the end of each bar.
    # Use one row per hypothesis (not per component) to avoid duplicate labels.
    label_df = long_df[long_df["component"] == "Likelihood"].copy()
    # Ensure tiny bars still get a visible label position.
    label_df["label_x"] = label_df["end"].clip(lower=0.002)

    text = (
        alt.Chart(label_df)
        .mark_text(align="left", dx=4, fontSize=10)
        .encode(
            x=alt.X("label_x:Q"),
            y=alt.Y("program_label:N", sort=sorted_labels),
            text=alt.Text("probability:Q", format=".4f"),
        )
    )

    n_hyps = len(plot_df)
    return (
        (bars + text)
        .properties(
            width=550,
            height=max(300, n_hyps * 50),
            title=f"Posterior Decomposition — {rule_id}",
        )
    )


def diagnosticity_bars(diag_df: pd.DataFrame) -> alt.LayerChart:
    """Bar chart of exemplar agreement rate with a diagnostic threshold line.

    Hand index on the x-axis, agreement rate on the y-axis.  Diagnostic
    hands colored in red, others in grey.  A horizontal rule at 0.90
    marks the diagnostic threshold.

    Parameters
    ----------
    diag_df : pd.DataFrame
        Filtered slice of ``diagnosticity_df`` for a single rule.
    """
    threshold = 0.90

    # Convert 0-indexed to 1-indexed for display
    plot_df = diag_df.copy()
    plot_df["hand_num"] = plot_df["hand_idx"] + 1

    bars = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("hand_num:O", title="Hand"),
            y=alt.Y("agreement_rate:Q", title="Agreement Rate", scale=alt.Scale(domain=[0, 1])),
            color=alt.condition(
                alt.datum.diagnostic,
                alt.value("#C44E52"),  # red for diagnostic hands
                alt.value("#B0B0B0"),  # grey for non-diagnostic
            ),
            tooltip=[
                alt.Tooltip("hand_num:O", title="Hand"),
                alt.Tooltip("agreement_rate:Q", title="Agreement", format=".3f"),
                alt.Tooltip("diagnostic:N", title="Diagnostic?"),
            ],
        )
    )

    rule_line = (
        alt.Chart(pd.DataFrame({"y": [threshold]}))
        .mark_rule(strokeDash=[4, 4], color="black")
        .encode(y="y:Q")
    )

    return (
        (bars + rule_line)
        .properties(
            width=500,
            height=300,
            title="Exemplar Diagnosticity",
        )
    )


def calibration_plot(cal_df: pd.DataFrame) -> alt.LayerChart:
    """Calibration curves per rule with a diagonal reference line.

    Each rule gets a thin colored line (colored by difficulty group)
    connecting its P(accept) bins to observed acceptance rates.  Points
    are sized by the number of hands in each bin.  A dashed diagonal
    from (0,0) to (1,1) represents perfect calibration.

    Parameters
    ----------
    cal_df : pd.DataFrame
        Output of :func:`data.build_calibration_df`.  Must contain columns
        ``bin_center``, ``observed_rate``, ``group_label``, ``rule_id``,
        ``n_hands``.
    """
    # Diagonal reference line (perfect calibration).
    diag_data = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    diagonal = (
        alt.Chart(diag_data)
        .mark_line(strokeDash=[4, 4], color="#999", strokeWidth=1)
        .encode(x="x:Q", y="y:Q")
    )

    # One line per rule, colored by difficulty group.
    lines = (
        alt.Chart(cal_df)
        .mark_line(opacity=0.4, strokeWidth=1)
        .encode(
            x=alt.X(
                "bin_center:Q",
                title="Predicted P(accept)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            y=alt.Y(
                "observed_rate:Q",
                title="Observed Acceptance Rate",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "group_label:N",
                title="Difficulty",
                scale=difficulty_color_scale(),
            ),
            detail=alt.Detail("rule_id:N"),
        )
    )

    # Points at each (rule, bin) with size = n_hands.
    points = (
        alt.Chart(cal_df)
        .mark_circle(opacity=0.6)
        .encode(
            x=alt.X("bin_center:Q"),
            y=alt.Y("observed_rate:Q"),
            color=alt.Color(
                "group_label:N",
                title="Difficulty",
                scale=difficulty_color_scale(),
            ),
            size=alt.Size(
                "n_hands:Q",
                title="N hands",
                scale=alt.Scale(range=[15, 150]),
            ),
            tooltip=[
                alt.Tooltip("rule_id:N", title="Rule"),
                alt.Tooltip("group_label:N", title="Group"),
                alt.Tooltip("bin_center:Q", title="Bin Center", format=".2f"),
                alt.Tooltip("observed_rate:Q", title="Obs. Rate", format=".2f"),
                alt.Tooltip("n_hands:Q", title="N Hands"),
            ],
        )
    )

    return (
        (diagonal + lines + points)
        .properties(
            width=450,
            height=350,
            title="Calibration: P(accept) vs Observed Acceptance",
        )
    )


def entropy_vs_accuracy(merged_df: pd.DataFrame) -> alt.Chart:
    """Scatter of posterior entropy vs weighted-vote classification accuracy.

    X-axis is posterior entropy (Bayesian difficulty measure), y-axis is
    accuracy — the fraction of random test hands correctly classified by
    the posterior majority vote.  Rules in the bottom-right are both hard
    to identify AND poorly classified on novel hands.

    Parameters
    ----------
    merged_df : pd.DataFrame
        Must contain columns: rule_id, posterior_entropy,
        accuracy, group_label.
    """
    return (
        alt.Chart(merged_df)
        .mark_circle(size=80, opacity=0.8)
        .encode(
            x=alt.X(
                "posterior_entropy:Q",
                title="Posterior Entropy (bits)",
            ),
            y=alt.Y(
                "accuracy:Q",
                title="Classification Accuracy",
                scale=alt.Scale(domain=[0, 1.05]),
            ),
            color=alt.Color(
                "group_label:N",
                title="Difficulty",
                scale=difficulty_color_scale(),
            ),
            tooltip=[
                alt.Tooltip("rule_id:N", title="Rule"),
                alt.Tooltip("posterior_entropy:Q", title="Entropy", format=".3f"),
                alt.Tooltip("accuracy:Q", title="Accuracy", format=".1%"),
                alt.Tooltip("group_label:N", title="Group"),
            ],
        )
        .properties(
            width=450,
            height=350,
            title="Entropy vs Classification Accuracy",
        )
    )
