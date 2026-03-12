"""Pure Altair chart functions for Bayesian rule-induction analysis.

Each function takes a pandas DataFrame (or a subset thereof) and returns an
Altair chart object.  No I/O, no side effects — all rendering decisions are
deferred to the caller.

Summary-level plots consume ``difficulty_df`` (one row per rule).
Per-rule plots consume a filtered slice of ``hypotheses_df`` or
``diagnosticity_df``.

Usage::

    from gallery_analysis.visualization.data import load_results
    from gallery_analysis.visualization.plots import difficulty_strip

    results = load_results("gallery_analysis/results/depth6_injected.json")
    chart = difficulty_strip(results.difficulty_df)
    chart.save("difficulty_strip.html")
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


def difficulty_strip(df: pd.DataFrame) -> alt.Chart:
    """Strip/dot plot of posterior entropy per rule.

    Rules on the y-axis sorted by entropy (hardest at top), entropy on the
    x-axis, colored by difficulty group.  Tooltips show rule_id, answer,
    entropy, N_eff, and top-1 probability.

    Parameters
    ----------
    df : pd.DataFrame
        ``difficulty_df`` from :func:`data.load_results`.
    """
    # Sort order: highest entropy at top.
    sorted_rules = (
        df.sort_values("posterior_entropy", ascending=False)["rule_id"].tolist()
    )

    return (
        alt.Chart(df)
        .mark_circle(size=60)
        .encode(
            x=alt.X("posterior_entropy:Q", title="Posterior Entropy (bits)"),
            y=alt.Y(
                "rule_id:N",
                title="Rule",
                sort=sorted_rules,
                axis=alt.Axis(labelLimit=200),
            ),
            color=alt.Color(
                "group_label:N",
                title="Difficulty",
                scale=difficulty_color_scale(),
            ),
            tooltip=[
                alt.Tooltip("rule_id:N", title="Rule"),
                alt.Tooltip("answer:N", title="Answer"),
                alt.Tooltip("posterior_entropy:Q", title="Entropy", format=".3f"),
                alt.Tooltip("n_effective:Q", title="N_eff", format=".1f"),
                alt.Tooltip("top1_probability:Q", title="Top-1 %", format=".3f"),
            ],
        )
        .properties(
            width=500,
            height=800,
            title="Rule Difficulty by Posterior Entropy",
        )
    )


def difficulty_scatter(df: pd.DataFrame) -> alt.Chart:
    """Scatter of posterior entropy vs top-1 probability.

    Points sized by effective number of hypotheses and colored by difficulty
    group.

    Parameters
    ----------
    df : pd.DataFrame
        ``difficulty_df`` from :func:`data.load_results`.
    """
    return (
        alt.Chart(df)
        .mark_circle()
        .encode(
            x=alt.X("posterior_entropy:Q", title="Posterior Entropy (bits)"),
            y=alt.Y("top1_probability:Q", title="Top-1 Probability"),
            size=alt.Size(
                "n_effective:Q",
                title="N_eff",
                scale=alt.Scale(range=[30, 400]),
            ),
            color=alt.Color(
                "group_label:N",
                title="Difficulty",
                scale=difficulty_color_scale(),
            ),
            tooltip=[
                alt.Tooltip("rule_id:N", title="Rule"),
                alt.Tooltip("posterior_entropy:Q", title="Entropy", format=".3f"),
                alt.Tooltip("top1_probability:Q", title="Top-1 %", format=".3f"),
                alt.Tooltip("n_effective:Q", title="N_eff", format=".1f"),
            ],
        )
        .properties(
            width=500,
            height=400,
            title="Entropy vs Top-1 Probability",
        )
    )


def true_rule_recovery(df: pd.DataFrame) -> alt.Chart:
    """Dot plot of true-rule posterior mass per rule (log scale).

    Rules on the y-axis sorted by posterior mass (highest at top), colored by
    difficulty group.  Rules with null ``true_rule_posterior_mass`` are
    excluded.

    Parameters
    ----------
    df : pd.DataFrame
        ``difficulty_df`` from :func:`data.load_results`.
    """
    # Filter out rules where the true rule was not found.
    plot_df = df.dropna(subset=["true_rule_posterior_mass"]).copy()

    # Sort order: highest mass at top.
    sorted_rules = (
        plot_df.sort_values("true_rule_posterior_mass", ascending=False)["rule_id"]
        .tolist()
    )

    return (
        alt.Chart(plot_df)
        .mark_circle(size=60)
        .encode(
            x=alt.X(
                "true_rule_posterior_mass:Q",
                title="True-Rule Posterior Mass",
                scale=alt.Scale(type="log"),
            ),
            y=alt.Y(
                "rule_id:N",
                title="Rule",
                sort=sorted_rules,
                axis=alt.Axis(labelLimit=200),
            ),
            color=alt.Color(
                "group_label:N",
                title="Difficulty",
                scale=difficulty_color_scale(),
            ),
            tooltip=[
                alt.Tooltip("rule_id:N", title="Rule"),
                alt.Tooltip("answer:N", title="Answer"),
                alt.Tooltip(
                    "true_rule_posterior_mass:Q",
                    title="Posterior Mass",
                    format=".2e",
                ),
                alt.Tooltip("true_rule_rank:Q", title="True-Rule Rank"),
            ],
        )
        .properties(
            width=500,
            height=600,
            title="True-Rule Recovery (Posterior Mass)",
        )
    )


def equiv_class_bars(df: pd.DataFrame) -> alt.Chart:
    """Bar chart of the number of hypotheses with all-hits per rule.

    Rules sorted by count (descending), colored by difficulty group.

    Parameters
    ----------
    df : pd.DataFrame
        ``difficulty_df`` from :func:`data.load_results`.
    """
    sorted_rules = (
        df.sort_values("n_with_all_hits", ascending=False)["rule_id"].tolist()
    )

    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("n_with_all_hits:Q", title="Programs with All Hits"),
            y=alt.Y(
                "rule_id:N",
                title="Rule",
                sort=sorted_rules,
                axis=alt.Axis(labelLimit=200),
            ),
            color=alt.Color(
                "group_label:N",
                title="Difficulty",
                scale=difficulty_color_scale(),
            ),
            tooltip=[
                alt.Tooltip("rule_id:N", title="Rule"),
                alt.Tooltip("n_with_all_hits:Q", title="All-hits count"),
                alt.Tooltip("group_label:N", title="Difficulty"),
            ],
        )
        .properties(
            width=500,
            height=800,
            title="Equivalence-Class Size (Programs with All Hits)",
        )
    )


# ══════════════════════════════════════════════════════════════════════
# Per-rule charts (take a filtered slice of hypotheses_df / diagnosticity_df)
# ══════════════════════════════════════════════════════════════════════


def posterior_bars(hyp_df: pd.DataFrame, rule_id: str) -> alt.Chart:
    """Horizontal bar chart of top hypotheses for a single rule.

    Hypotheses on the y-axis (truncated program text), posterior probability
    on the x-axis.  The true rule is highlighted in green.

    Parameters
    ----------
    hyp_df : pd.DataFrame
        Filtered slice of ``hypotheses_df`` for a single rule.
    rule_id : str
        The rule identifier (used in the chart title).
    """
    # Truncate long program strings for the y-axis label.
    plot_df = hyp_df.copy()
    plot_df["program_short"] = plot_df["program"].str[:60]

    # Sort by probability descending (highest at top).
    sorted_programs = (
        plot_df.sort_values("probability", ascending=False)["program_short"].tolist()
    )

    return (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(
            x=alt.X("probability:Q", title="Posterior Probability"),
            y=alt.Y(
                "program_short:N",
                title=None,
                sort=sorted_programs,
                axis=alt.Axis(labelLimit=400),
            ),
            color=alt.condition(
                alt.datum.is_true_rule,
                alt.value(_TRUE_RULE_COLOR),
                alt.value("#7F7F7F"),
            ),
            tooltip=[
                alt.Tooltip("program:N", title="Full Program"),
                alt.Tooltip("probability:Q", title="P(h|D)", format=".4f"),
                alt.Tooltip("log_prior:Q", title="Log Prior", format=".2f"),
                alt.Tooltip("log_likelihood:Q", title="Log Lik.", format=".2f"),
                alt.Tooltip("is_true_rule:N", title="True Rule?"),
            ],
        )
        .properties(
            width=500,
            height=max(200, len(plot_df) * 25),
            title=f"Posterior Distribution — {rule_id}",
        )
    )


def prior_vs_likelihood(hyp_df: pd.DataFrame) -> alt.Chart:
    """Scatter of log-prior vs log-likelihood for a single rule's hypotheses.

    Points sized by posterior probability.

    Parameters
    ----------
    hyp_df : pd.DataFrame
        Filtered slice of ``hypotheses_df`` for a single rule.
    """
    return (
        alt.Chart(hyp_df)
        .mark_circle()
        .encode(
            x=alt.X("log_prior:Q", title="Log Prior"),
            y=alt.Y("log_likelihood:Q", title="Log Likelihood"),
            size=alt.Size(
                "probability:Q",
                title="Posterior",
                scale=alt.Scale(range=[20, 300]),
            ),
            color=alt.condition(
                alt.datum.is_true_rule,
                alt.value(_TRUE_RULE_COLOR),
                alt.value("#7F7F7F"),
            ),
            tooltip=[
                alt.Tooltip("program:N", title="Program"),
                alt.Tooltip("log_prior:Q", title="Log Prior", format=".2f"),
                alt.Tooltip("log_likelihood:Q", title="Log Lik.", format=".2f"),
                alt.Tooltip("probability:Q", title="Posterior", format=".4f"),
            ],
        )
        .properties(
            width=450,
            height=400,
            title="Prior vs Likelihood Trade-off",
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
