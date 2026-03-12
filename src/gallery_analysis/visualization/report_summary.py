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
    from gallery_analysis.visualization.data import BayesianResults
    from gallery_analysis.visualization.plots import (
        difficulty_strip,
        difficulty_scatter,
        true_rule_recovery,
        equiv_class_bars,
    )
except ImportError:
    # Fallback for direct execution: add parent packages to sys.path.
    import sys

    _this_dir = Path(__file__).resolve().parent
    _src_dir = _this_dir.parent.parent
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))

    from gallery_analysis.visualization.data import BayesianResults
    from gallery_analysis.visualization.plots import (
        difficulty_strip,
        difficulty_scatter,
        true_rule_recovery,
        equiv_class_bars,
    )


# ── Template directory ───────────────────────────────────────────────

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


# ── Public API ───────────────────────────────────────────────────────


def generate_summary(results: BayesianResults, output_dir: Path) -> Path:
    """Generate the summary index.html from Bayesian analysis results.

    Builds four Altair charts (strip, scatter, recovery, equivalence-class),
    serializes them as Vega-Lite JSON specs, and renders the summary.html
    Jinja2 template with pipeline statistics and a sortable rule table.

    Parameters
    ----------
    results : BayesianResults
        Loaded and normalized results from :func:`data.load_results`.
    output_dir : Path
        Directory to write ``index.html`` into.  Created if it does not exist.

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
    # Each plot function returns an Altair Chart; .to_dict() gives the
    # Vega-Lite spec as a Python dict; json.dumps() serializes it for
    # safe embedding in the HTML template.
    df = results.difficulty_df

    chart_strip = json.dumps(difficulty_strip(df).to_dict())
    chart_scatter = json.dumps(difficulty_scatter(df).to_dict())
    chart_recovery = json.dumps(true_rule_recovery(df).to_dict())
    chart_equiv = json.dumps(equiv_class_bars(df).to_dict())

    # ── Build the rules list for the index table ─────────────────────
    # Sort by posterior entropy descending (hardest rules first) so the
    # default table order matches the strip plot.
    rules = (
        df.sort_values("posterior_entropy", ascending=False)
        .to_dict("records")
    )

    # ── Render the template ──────────────────────────────────────────
    html = template.render(
        stats=results.pipeline_stats,
        chart_strip=chart_strip,
        chart_scatter=chart_scatter,
        chart_recovery=chart_recovery,
        chart_equiv=chart_equiv,
        rules=rules,
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
