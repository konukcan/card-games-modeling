"""
Altair chart functions for the LLM hypothesis analysis reports.

Every function is pure: takes data, returns an Altair Chart object.
No file I/O, no side effects.

Charts use encoding conventions from theme.py:
  Color = difficulty group, Shape = translator, Dash = prompt variant
"""

import altair as alt
import pandas as pd
from typing import Dict, List
from collections import defaultdict

from .theme import (
    DIFFICULTY_COLORS,
    DIFFICULTY_DOMAIN,
    DIFFICULTY_RANGE,
    THRESHOLDS,
    difficulty_color_scale,
    continuous_color_scale,
    GROUP_LABELS,
)


def translator_leaderboard_chart(
    leaderboard: Dict[str, Dict],
) -> alt.LayerChart:
    """Grouped bar chart: 4 translators x 3 metrics with threshold lines."""
    rows = []
    for translator, stats in leaderboard.items():
        rows.append({"Translator": translator, "Metric": "Syntax Rate", "Value": stats["syntax_rate"]})
        rows.append({"Translator": translator, "Metric": "Semantic 6/6", "Value": stats["semantic_6_6_rate"]})
        if stats.get("cross_gen_agreement") is not None:
            rows.append({"Translator": translator, "Metric": "Cross-Gen Agreement", "Value": stats["cross_gen_agreement"]})

    df = pd.DataFrame(rows)

    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Translator:N", title=None),
            y=alt.Y("Value:Q", title="Rate", scale=alt.Scale(domain=[0, 1]),
                     axis=alt.Axis(format=".0%")),
            color=alt.Color("Metric:N", title="Metric",
                            scale=alt.Scale(
                                domain=["Syntax Rate", "Semantic 6/6", "Cross-Gen Agreement"],
                                range=["#5B8DB8", "#7AB648", "#E8963E"],
                            )),
            xOffset="Metric:N",
            tooltip=["Translator:N", "Metric:N", alt.Tooltip("Value:Q", format=".1%")],
        )
        .properties(title="Translator Leaderboard", width=500, height=300)
    )

    threshold_data = pd.DataFrame([
        {"Metric": "Syntax Rate",         "Threshold": THRESHOLDS["syntax"]},
        {"Metric": "Semantic 6/6",        "Threshold": THRESHOLDS["semantic"]},
        {"Metric": "Cross-Gen Agreement", "Threshold": THRESHOLDS["cross_gen"]},
    ])

    rules = (
        alt.Chart(threshold_data)
        .mark_rule(strokeDash=[5, 5], opacity=0.7)
        .encode(
            y="Threshold:Q",
            color=alt.Color("Metric:N", scale=alt.Scale(
                domain=["Syntax Rate", "Semantic 6/6", "Cross-Gen Agreement"],
                range=["#5B8DB8", "#7AB648", "#E8963E"],
            )),
        )
    )

    return bars + rules


def hands_comparison_chart(
    condition_comparison: Dict[str, Dict],
) -> alt.Chart:
    """Paired bar chart: with-hands vs no-hands on syntax and semantic rates."""
    rows = []
    for condition, stats in condition_comparison.items():
        rows.append({"Condition": condition, "Metric": "Syntax Rate", "Value": stats["syntax_rate"]})
        rows.append({"Condition": condition, "Metric": "Semantic 6/6", "Value": stats["semantic_6_6_rate"]})

    df = pd.DataFrame(rows)

    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Condition:N", title=None),
            y=alt.Y("Value:Q", title="Rate", scale=alt.Scale(domain=[0, 1]),
                     axis=alt.Axis(format=".0%")),
            color=alt.Color("Metric:N", title="Metric",
                            scale=alt.Scale(domain=["Syntax Rate", "Semantic 6/6"],
                                          range=["#5B8DB8", "#7AB648"])),
            xOffset="Metric:N",
            tooltip=["Condition:N", "Metric:N", alt.Tooltip("Value:Q", format=".1%")],
        )
        .properties(title="With/Without Hands Comparison", width=300, height=300)
    )


def error_taxonomy_chart(
    error_taxonomy: Dict[str, int],
) -> alt.Chart:
    """Horizontal bar chart of error categories, sorted by frequency."""
    df = pd.DataFrame([
        {"Category": cat, "Count": count}
        for cat, count in error_taxonomy.items()
    ])

    return (
        alt.Chart(df)
        .mark_bar(color="#888888")
        .encode(
            y=alt.Y("Category:N", title=None,
                    sort=alt.EncodingSortField("Count", order="descending")),
            x=alt.X("Count:Q", title="Number of Failures"),
            tooltip=["Category:N", "Count:Q"],
        )
        .properties(title="Error Taxonomy", width=400,
                    height=max(150, len(error_taxonomy) * 30))
    )


def per_rule_heatmap(
    results: List[Dict],
) -> alt.Chart:
    """Heatmap: rules (rows) x translators (columns), colored by semantic match rate."""
    agg = defaultdict(list)
    rule_difficulty = {}
    for r in results:
        key = (r["rule_id"], r["translator"])
        match_rate = r.get("semantics", {}).get("match_rate", 0.0)
        agg[key].append(match_rate)
        rule_difficulty[r["rule_id"]] = r.get("difficulty", "Unknown")

    rows = []
    for (rule_id, translator), rates in agg.items():
        rows.append({
            "Rule": rule_id,
            "Translator": translator,
            "Match Rate": sum(rates) / len(rates),
            "Difficulty": rule_difficulty.get(rule_id, "Unknown"),
        })

    df = pd.DataFrame(rows)

    difficulty_order = {"Easy": 0, "Medium": 1, "Hard": 2}
    df["_sort"] = df["Difficulty"].map(difficulty_order).fillna(3)
    sort_order = (
        df[["Rule", "_sort"]]
        .drop_duplicates()
        .sort_values(["_sort", "Rule"])["Rule"]
        .tolist()
    )

    return (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("Translator:N", title=None),
            y=alt.Y("Rule:N", title=None, sort=sort_order),
            color=alt.Color("Match Rate:Q", title="Semantic Match Rate",
                            scale=continuous_color_scale(),
                            legend=alt.Legend(format=".0%")),
            tooltip=["Rule:N", "Translator:N", "Difficulty:N",
                     alt.Tooltip("Match Rate:Q", format=".1%")],
        )
        .properties(title="Per-Rule Translation Success", width=200,
                    height=max(400, len(sort_order) * 12))
    )
