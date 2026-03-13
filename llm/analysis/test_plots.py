"""Tests for the Altair chart functions."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import altair as alt
from llm.analysis.theme import register_theme
from llm.analysis.plots import (
    translator_leaderboard_chart,
    hands_comparison_chart,
    error_taxonomy_chart,
    per_rule_heatmap,
)

register_theme()


def _make_leaderboard_data():
    return {
        "claude-opus": {"total_translations": 100, "syntax_rate": 0.98,
                        "semantic_6_6_rate": 0.85, "cross_gen_agreement": 0.75},
        "gemini-flash": {"total_translations": 100, "syntax_rate": 0.92,
                         "semantic_6_6_rate": 0.70, "cross_gen_agreement": 0.60},
    }

def _make_condition_data():
    return {
        "with-hands": {"total": 200, "syntax_rate": 0.96, "semantic_6_6_rate": 0.82},
        "no-hands":   {"total": 200, "syntax_rate": 0.94, "semantic_6_6_rate": 0.72},
    }

def _make_error_data():
    return {"syntax_failure": 15, "semantic_mismatch": 25, "vague_hypothesis": 10}

def _make_results_for_heatmap():
    results = []
    for rule_id in ["all_red", "all_even"]:
        for translator in ["claude-opus", "gemini-flash"]:
            results.append({
                "rule_id": rule_id, "translator": translator,
                "difficulty": "Easy",
                "semantics": {"match_rate": 1.0},
            })
    return results


def test_translator_leaderboard_returns_chart():
    chart = translator_leaderboard_chart(_make_leaderboard_data())
    assert isinstance(chart, (alt.Chart, alt.LayerChart, alt.VConcatChart, alt.HConcatChart))

def test_hands_comparison_returns_chart():
    chart = hands_comparison_chart(_make_condition_data())
    assert isinstance(chart, (alt.Chart, alt.LayerChart, alt.VConcatChart, alt.HConcatChart))

def test_error_taxonomy_returns_chart():
    chart = error_taxonomy_chart(_make_error_data())
    assert isinstance(chart, (alt.Chart, alt.LayerChart))

def test_per_rule_heatmap_returns_chart():
    chart = per_rule_heatmap(_make_results_for_heatmap())
    assert isinstance(chart, (alt.Chart, alt.LayerChart))

def test_charts_produce_valid_vegalite():
    chart = translator_leaderboard_chart(_make_leaderboard_data())
    spec = chart.to_dict()
    assert "$schema" in spec


if __name__ == "__main__":
    test_translator_leaderboard_returns_chart()
    test_hands_comparison_returns_chart()
    test_error_taxonomy_returns_chart()
    test_per_rule_heatmap_returns_chart()
    test_charts_produce_valid_vegalite()
    print("All tests passed!")
