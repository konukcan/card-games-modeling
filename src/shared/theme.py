"""
Shared visual design system for all analysis reports.

This module defines the color palette, Altair theme configuration, and scale
helpers that both the LLM analysis and Bayesian rule-induction visualizations
share.  Module-specific constants (e.g. translator shapes, variant dashes)
live in their respective wrapper modules.

Encoding discipline (when applicable):
  - Color  = difficulty group (always)

Colors:
  - Easy (group 1):   blue family   (#4A90D9 base)
  - Medium (group 2): amber family  (#D4A029 base)
  - Hard (group 3):   red family    (#C44E52 base)

Continuous scales use viridis (perceptually uniform, colorblind-safe).
Diverging scales use blue-white-red (0.5 = maximally ambiguous).
"""

import altair as alt

# ── Difficulty color palette ────────────────────────────────────────────

DIFFICULTY_COLORS = {
    "Easy":   "#4A90D9",
    "Medium": "#D4A029",
    "Hard":   "#C44E52",
}

DIFFICULTY_COLORS_LIGHT = {
    "Easy":   "#A3C4E9",
    "Medium": "#E8CC7A",
    "Hard":   "#D99A9C",
}

DIFFICULTY_COLORS_DARK = {
    "Easy":   "#2B5F9E",
    "Medium": "#9E7518",
    "Hard":   "#8E2E32",
}

DIFFICULTY_DOMAIN = ["Easy", "Medium", "Hard"]
DIFFICULTY_RANGE = [DIFFICULTY_COLORS[d] for d in DIFFICULTY_DOMAIN]

# ── Group label mapping ─────────────────────────────────────────────────

GROUP_LABELS = {1: "Easy", 2: "Medium", 3: "Hard"}

# ── Thresholds ──────────────────────────────────────────────────────────

THRESHOLDS = {
    "syntax":    0.95,
    "semantic":  0.80,
    "cross_gen": 0.70,
}

# ── Altair scale helpers ────────────────────────────────────────────────


def difficulty_color_scale() -> alt.Scale:
    """Categorical color scale mapping Easy/Medium/Hard to the palette."""
    return alt.Scale(domain=DIFFICULTY_DOMAIN, range=DIFFICULTY_RANGE)


def continuous_color_scale() -> alt.Scale:
    """Perceptually uniform continuous scale (viridis)."""
    return alt.Scale(scheme="viridis")


def diverging_color_scale(midpoint: float = 0.5) -> alt.Scale:
    """Blue-white-red diverging scale; *midpoint* marks neutral."""
    return alt.Scale(
        domain=[0.0, midpoint, 1.0],
        range=["#4A90D9", "#FFFFFF", "#C44E52"],
    )


# ── Shared Altair theme ─────────────────────────────────────────────────

def _base_theme():
    """Base Altair theme configuration used by all reports."""
    return {
        "config": {
            "background": "#FFFFFF",
            "font": "system-ui, -apple-system, sans-serif",
            "axis": {
                "labelFontSize": 12,
                "titleFontSize": 13,
                "titleFontWeight": "normal",
            },
            "header": {
                "labelFontSize": 13,
                "titleFontSize": 14,
            },
            "legend": {
                "labelFontSize": 12,
                "titleFontSize": 13,
            },
            "title": {
                "fontSize": 16,
                "fontWeight": "bold",
                "anchor": "start",
            },
            "view": {
                "continuousWidth": 600,
                "continuousHeight": 400,
            },
        }
    }


def register_theme(name: str = "shared_analysis") -> None:
    """Register and enable the shared Altair theme.

    Parameters
    ----------
    name : str
        Theme name to register under.  Defaults to ``"shared_analysis"``.
        The LLM wrapper passes ``"llm_analysis"`` to keep backward
        compatibility.
    """
    alt.themes.register(name, _base_theme)
    alt.themes.enable(name)
