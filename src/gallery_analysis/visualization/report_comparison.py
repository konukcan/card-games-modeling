"""Variant comparison dashboard generator.

Loads all 10 v2 result JSON files, extracts per-rule metrics, and
produces a standalone HTML page with:

- Chart B: Distribution shift panels (density curves) for each factor
  (grammar, prior mode, injection, likelihood) — 4 panels for entropy,
  4 panels for true-rule posterior mass.
- Chart C: Effect-size heatmap (60 rules × 10 variants) with tab
  switcher for three metrics.

Usage::

    from gallery_analysis.visualization.report_comparison import generate_comparison_page
    generate_comparison_page("gallery_analysis/results", "/tmp/comparison.html")
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import altair as alt
import pandas as pd
from jinja2 import Environment, FileSystemLoader

# Ensure src/ is on sys.path so sibling package imports work.
_src_dir = Path(__file__).resolve().parent.parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

try:
    from shared.theme import register_theme
except ImportError:
    def register_theme() -> None:
        pass


# ── Variant file definitions ────────────────────────────────────────
# Each tuple: (filename, grammar, prior_mode, inject, likelihood)
# 'inject' is "inject" / "noinject"; 'likelihood' is "noisy" / "strict".

VARIANT_FILES = [
    ("v2_weighted_canonical_inject.json",   "weighted", "canonical", "inject",   "noisy"),
    ("v2_weighted_summed_inject.json",      "weighted", "summed",    "inject",   "noisy"),
    ("v2_weighted_canonical_noinject.json",  "weighted", "canonical", "noinject", "noisy"),
    ("v2_weighted_summed_noinject.json",     "weighted", "summed",    "noinject", "noisy"),
    ("v2_uniform_canonical_inject.json",     "uniform",  "canonical", "inject",   "noisy"),
    ("v2_uniform_summed_inject.json",        "uniform",  "summed",    "inject",   "noisy"),
    ("v2_uniform_canonical_noinject.json",   "uniform",  "canonical", "noinject", "noisy"),
    ("v2_uniform_summed_noinject.json",      "uniform",  "summed",    "noinject", "noisy"),
    ("v2_weighted_canonical_strict.json",    "weighted", "canonical", "inject",   "strict"),
    ("v2_weighted_summed_strict.json",       "weighted", "summed",    "inject",   "strict"),
]


# ── Data loading ────────────────────────────────────────────────────

def _load_variant(results_dir: Path, filename: str, grammar: str,
                  prior_mode: str, inject: str, likelihood: str) -> list[dict]:
    """Load one variant JSON and return a list of per-rule row dicts.

    Each dict has: rule_id, variant, grammar, prior_mode, inject,
    likelihood, posterior_entropy, true_rule_posterior_mass, true_rule_rank.
    """
    path = results_dir / filename
    if not path.exists():
        return []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # Build a short human-readable variant label.
    variant_label = f"{grammar}-{prior_mode}-{inject}"
    if likelihood == "strict":
        variant_label = f"{grammar}-{prior_mode}-strict"

    rows: list[dict] = []
    for rule_id, rule_data in data.get("rule_details", {}).items():
        difficulty = rule_data.get("difficulty", {})
        entropy = difficulty.get("posterior_entropy")
        mass = rule_data.get("true_rule_posterior_mass")
        rank = rule_data.get("true_rule_rank")

        rows.append({
            "rule_id": rule_id,
            "variant": variant_label,
            "grammar": grammar,
            "prior_mode": prior_mode,
            "inject": inject,
            "likelihood": likelihood,
            "posterior_entropy": entropy,
            "true_rule_posterior_mass": mass,
            "true_rule_rank": rank,
        })

    return rows


def load_all_variants(results_dir: str | Path) -> pd.DataFrame:
    """Load all 10 v2 variants into a single DataFrame.

    Returns a DataFrame with columns: rule_id, variant, grammar,
    prior_mode, inject, likelihood, posterior_entropy,
    true_rule_posterior_mass, true_rule_rank, log10_mass.
    """
    results_dir = Path(results_dir)
    all_rows: list[dict] = []
    for filename, grammar, prior_mode, inject, likelihood in VARIANT_FILES:
        rows = _load_variant(results_dir, filename, grammar, prior_mode,
                             inject, likelihood)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)

    # Add log10 of posterior mass (handling zeros / very small values).
    # Clamp at -10 so density plots stay readable.
    df["log10_mass"] = df["true_rule_posterior_mass"].apply(
        lambda x: max(math.log10(x), -10) if x and x > 0 else -10
    )

    return df


# ── Chart builders ──────────────────────────────────────────────────

# Colours for two-level factors.
_COLORS_2 = ["#4A90D9", "#E67E22"]


def _density_panel(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    title: str,
    x_title: str,
) -> alt.Chart:
    """Build a single density-overlay panel comparing two factor levels.

    Uses Altair's transform_density to produce smooth KDE curves rendered
    as semi-transparent filled areas, one per level of *group_col*.
    """
    levels = sorted(df[group_col].unique())

    chart = (
        alt.Chart(df)
        .transform_density(
            value_col,
            as_=[value_col, "density"],
            groupby=[group_col],
            extent=[float(df[value_col].min() - 0.1),
                    float(df[value_col].max() + 0.1)],
        )
        .mark_area(opacity=0.4, interpolate="monotone")
        .encode(
            x=alt.X(f"{value_col}:Q", title=x_title),
            y=alt.Y("density:Q", title="Density"),
            color=alt.Color(
                f"{group_col}:N",
                scale=alt.Scale(domain=levels, range=_COLORS_2[:len(levels)]),
                legend=alt.Legend(title=group_col.replace("_", " ").title()),
            ),
        )
        .properties(width=280, height=180, title=title)
    )
    return chart


def build_density_charts(
    df: pd.DataFrame,
    metric: str,
    x_title: str,
) -> alt.Chart:
    """Build the 4-panel density comparison for one metric.

    Panels: Grammar, Prior Mode, Injection, Likelihood.
    For Likelihood, only weighted-grammar variants are used (since strict
    only exists for weighted).

    Returns a 2×2 concatenated chart.
    """
    panels: list[alt.Chart] = []

    # Panel 1: Grammar (weighted vs uniform).
    # Exclude strict likelihood (only exists for weighted) to keep the
    # comparison balanced.
    df_grammar = df[df["likelihood"] != "strict"]
    panels.append(_density_panel(
        df_grammar, metric, "grammar",
        title="Grammar Effect", x_title=x_title,
    ))

    # Panel 2: Prior mode (canonical vs summed).
    df_prior = df[df["likelihood"] != "strict"]
    panels.append(_density_panel(
        df_prior, metric, "prior_mode",
        title="Prior Mode Effect", x_title=x_title,
    ))

    # Panel 3: Injection (inject vs noinject).
    # Exclude strict variants (they don't have a noinject pair).
    df_inject = df[df["likelihood"] != "strict"]
    panels.append(_density_panel(
        df_inject, metric, "inject",
        title="Injection Effect", x_title=x_title,
    ))

    # Panel 4: Likelihood (noisy vs strict) — weighted only.
    # Compare inject variants: weighted-canonical-inject (noisy) vs
    # weighted-canonical-strict, and weighted-summed-inject (noisy) vs
    # weighted-summed-strict.
    df_likelihood = df[
        (df["grammar"] == "weighted") &
        ((df["likelihood"] == "strict") | (df["inject"] == "inject"))
    ]
    panels.append(_density_panel(
        df_likelihood, metric, "likelihood",
        title="Likelihood Effect (weighted only)", x_title=x_title,
    ))

    # Arrange 2×2.
    row1 = alt.hconcat(panels[0], panels[1]).resolve_scale(color="independent")
    row2 = alt.hconcat(panels[2], panels[3]).resolve_scale(color="independent")
    combined = alt.vconcat(row1, row2)
    return combined


def build_heatmap(df: pd.DataFrame, metric: str, title: str,
                  color_title: str, scheme: str = "redblue",
                  reverse: bool = False) -> alt.Chart:
    """Build a 60-rule × 10-variant heatmap for one metric.

    Rules (Y axis) sorted by the reference variant's value of *metric*.
    Variants (X axis) ordered consistently.

    Parameters
    ----------
    df : DataFrame with columns rule_id, variant, and *metric*.
    metric : column name to map to cell colour.
    title : chart title.
    color_title : legend title for the colour scale.
    scheme : Vega colour scheme name.
    reverse : if True, reverse the colour scale direction.
    """
    # Sort rules by reference variant (weighted-canonical-inject) value.
    ref_variant = "weighted-canonical-inject"
    ref_df = df[df["variant"] == ref_variant].set_index("rule_id")
    if ref_df.empty:
        # Fallback: use first variant alphabetically.
        first = sorted(df["variant"].unique())[0]
        ref_df = df[df["variant"] == first].set_index("rule_id")

    rule_order = (
        ref_df
        .sort_values(metric, ascending=True)
        .index.tolist()
    )

    variant_order = sorted(df["variant"].unique())

    chart = (
        alt.Chart(df)
        .mark_rect()
        .encode(
            x=alt.X("variant:N", title="Variant",
                     sort=variant_order,
                     axis=alt.Axis(labelAngle=-45, labelFontSize=9)),
            y=alt.Y("rule_id:N", title="Rule",
                     sort=rule_order,
                     axis=alt.Axis(labelFontSize=8)),
            color=alt.Color(
                f"{metric}:Q",
                title=color_title,
                scale=alt.Scale(scheme=scheme, reverse=reverse),
            ),
            tooltip=[
                alt.Tooltip("rule_id:N", title="Rule"),
                alt.Tooltip("variant:N", title="Variant"),
                alt.Tooltip(f"{metric}:Q", title=color_title, format=".3f"),
            ],
        )
        .properties(
            width=500,
            height=900,
            title=title,
        )
    )
    return chart


# ── Page generation ─────────────────────────────────────────────────

def generate_comparison_page(
    results_dir: str | Path,
    output_path: str | Path,
) -> Path:
    """Generate the full comparison dashboard HTML page.

    Parameters
    ----------
    results_dir : directory containing the v2_*.json result files.
    output_path : where to write the HTML file.

    Returns
    -------
    Path to the written HTML file.
    """
    register_theme()
    alt.data_transformers.disable_max_rows()

    results_dir = Path(results_dir)
    output_path = Path(output_path)

    # 1. Load all variants.
    df = load_all_variants(results_dir)
    n_rules = df["rule_id"].nunique()
    n_variants = df["variant"].nunique()
    print(f"Loaded {len(df)} rows: {n_rules} rules × {n_variants} variants")

    # 2. Build density charts.
    chart_dist_entropy = build_density_charts(
        df, "posterior_entropy", x_title="Posterior Entropy (bits)",
    )
    chart_dist_mass = build_density_charts(
        df, "log10_mass", x_title="log₁₀(True Rule Posterior Mass)",
    )

    # 3. Build heatmaps (one per metric, switched via JS tabs).
    heatmap_entropy = build_heatmap(
        df, "posterior_entropy",
        title="Posterior Entropy by Rule × Variant",
        color_title="Entropy (bits)",
        scheme="redblue", reverse=True,
    )
    heatmap_mass = build_heatmap(
        df, "log10_mass",
        title="log₁₀(True Rule Mass) by Rule × Variant",
        color_title="log₁₀(mass)",
        scheme="redblue", reverse=False,
    )
    heatmap_rank = build_heatmap(
        df, "true_rule_rank",
        title="True Rule Rank by Rule × Variant",
        color_title="Rank",
        scheme="redblue", reverse=True,
    )

    # 4. Convert charts to Vega-Lite JSON specs.
    spec_dist_entropy = chart_dist_entropy.to_json(indent=None)
    spec_dist_mass = chart_dist_mass.to_json(indent=None)
    spec_heatmap_entropy = heatmap_entropy.to_json(indent=None)
    spec_heatmap_mass = heatmap_mass.to_json(indent=None)
    spec_heatmap_rank = heatmap_rank.to_json(indent=None)

    # 5. Render template.
    templates_dir = Path(__file__).resolve().parent / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=False,
    )
    template = env.get_template("comparison.html")

    html = template.render(
        n_rules=n_rules,
        n_variants=n_variants,
        chart_dist_entropy=spec_dist_entropy,
        chart_dist_mass=spec_dist_mass,
        chart_heatmap_entropy=spec_heatmap_entropy,
        chart_heatmap_mass=spec_heatmap_mass,
        chart_heatmap_rank=spec_heatmap_rank,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Comparison page written to: {output_path}")
    return output_path
