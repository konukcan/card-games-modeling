"""Tests for the visual design system."""

import sys
from pathlib import Path

# Put the project root on sys.path so both llm.analysis and src.shared resolve.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm.analysis.theme import (
    DIFFICULTY_COLORS,
    TRANSLATOR_SHAPES,
    DIFFICULTY_DOMAIN,
    difficulty_color_scale,
    translator_shape_scale,
    continuous_color_scale,
    diverging_color_scale,
    register_theme,
)

# Also verify that the shared module is independently importable.
from src.shared.theme import (
    DIFFICULTY_COLORS as SHARED_COLORS,
    GROUP_LABELS,
    THRESHOLDS,
    register_theme as shared_register,
)


def test_difficulty_colors_has_three_groups():
    assert len(DIFFICULTY_COLORS) == 3
    assert "Easy" in DIFFICULTY_COLORS
    assert "Medium" in DIFFICULTY_COLORS
    assert "Hard" in DIFFICULTY_COLORS


def test_shared_colors_match_llm_colors():
    """The LLM wrapper re-exports the exact same objects from shared."""
    assert DIFFICULTY_COLORS is SHARED_COLORS


def test_translator_shapes_has_four_models():
    assert len(TRANSLATOR_SHAPES) == 4
    assert "claude-opus" in TRANSLATOR_SHAPES
    assert "qwen-coder" in TRANSLATOR_SHAPES


def test_difficulty_color_scale_returns_altair_scale():
    import altair as alt
    scale = difficulty_color_scale()
    assert isinstance(scale, alt.Scale)


def test_shared_group_labels():
    assert GROUP_LABELS == {1: "Easy", 2: "Medium", 3: "Hard"}


def test_shared_thresholds():
    assert "syntax" in THRESHOLDS
    assert "semantic" in THRESHOLDS
    assert "cross_gen" in THRESHOLDS


def test_register_theme_does_not_crash():
    register_theme()


def test_shared_register_theme_does_not_crash():
    shared_register()


if __name__ == "__main__":
    test_difficulty_colors_has_three_groups()
    test_shared_colors_match_llm_colors()
    test_translator_shapes_has_four_models()
    test_difficulty_color_scale_returns_altair_scale()
    test_shared_group_labels()
    test_shared_thresholds()
    test_register_theme_does_not_crash()
    test_shared_register_theme_does_not_crash()
    print("All tests passed!")
