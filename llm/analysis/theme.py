"""
Visual design system for the LLM hypothesis analysis reports.

This is a thin wrapper around :mod:`src.shared.theme` that re-exports all
shared constants and adds LLM-specific mappings (translator shapes, prompt
variant dashes, model colors).

Encoding discipline:
  - Color  = difficulty group (always)
  - Shape  = translator or model (always)
  - Dash   = prompt variant (always)
"""

import sys
from pathlib import Path

# Ensure the worktree / project root is on sys.path so that
# ``from src.shared.theme import ...`` resolves correctly.
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ── Re-export everything from the shared theme ──────────────────────────

from src.shared.theme import (                       # noqa: E402, F401
    DIFFICULTY_COLORS,
    DIFFICULTY_COLORS_LIGHT,
    DIFFICULTY_COLORS_DARK,
    DIFFICULTY_DOMAIN,
    DIFFICULTY_RANGE,
    GROUP_LABELS,
    THRESHOLDS,
    difficulty_color_scale,
    continuous_color_scale,
    diverging_color_scale,
    register_theme as _register_shared_theme,
)

import altair as alt                                  # noqa: E402

# ── LLM-specific constants ──────────────────────────────────────────────

TRANSLATOR_SHAPES = {
    "claude-opus":   "circle",
    "claude-sonnet": "square",
    "gemini-flash":  "triangle-up",
    "qwen-coder":    "diamond",
}

TRANSLATOR_DOMAIN = list(TRANSLATOR_SHAPES.keys())
TRANSLATOR_RANGE = list(TRANSLATOR_SHAPES.values())

VARIANT_DASHES = {
    "baseline":         [1, 0],
    "feature-primed":   [5, 5],
    "minimal":          [2, 2],
    "cognitive":        [5, 2, 2, 2],
    "numeric-prob":     [8, 4],
}

MODEL_COLORS = {
    "gemini-2.5-flash": "#666666",
    "gemini-2.5-pro":   "#333333",
}

# ── LLM-specific scale helper ───────────────────────────────────────────


def translator_shape_scale() -> alt.Scale:
    """Categorical shape scale mapping translator names to point shapes."""
    return alt.Scale(domain=TRANSLATOR_DOMAIN, range=TRANSLATOR_RANGE)


# ── Theme registration (backward-compatible name) ───────────────────────


def register_theme() -> None:
    """Register and enable the Altair theme as ``"llm_analysis"``."""
    _register_shared_theme(name="llm_analysis")
