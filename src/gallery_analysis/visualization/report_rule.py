"""Per-rule detail page generator for the Bayesian visualization pipeline.

Produces one HTML page per rule, combining card exemplars, posterior charts,
prior-vs-likelihood scatter, optional diagnosticity bars, and a hypotheses
table.

Usage::

    from gallery_analysis.visualization.data import load_results
    from gallery_analysis.visualization.cards import load_exemplars, get_rule_hands, hands_to_json
    from gallery_analysis.visualization.report_rule import generate_rule_page

    results = load_results("results.json")
    exemplars = load_exemplars(Path("frozen-exemplars.json"))
    generate_rule_page("all_red", results, exemplars, "../../stim", cards_js, Path("output/rules"))
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader

# Local visualization imports — follow the same try/except pattern used
# by the sibling modules (data.py, plots.py, cards.py).
try:
    from gallery_analysis.visualization.data import BayesianResults
    from gallery_analysis.visualization.plots import (
        posterior_bars,
        prior_vs_likelihood,
        diagnosticity_bars,
    )
    from gallery_analysis.visualization.cards import get_rule_hands, hands_to_json
except ImportError:
    # Fallback for direct execution: add parent packages to sys.path.
    import sys

    _this_dir = Path(__file__).resolve().parent
    _src_dir = _this_dir.parent.parent
    if str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))

    from gallery_analysis.visualization.data import BayesianResults
    from gallery_analysis.visualization.plots import (
        posterior_bars,
        prior_vs_likelihood,
        diagnosticity_bars,
    )
    from gallery_analysis.visualization.cards import get_rule_hands, hands_to_json


# ── Template directory ───────────────────────────────────────────────

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


# ── Public API ───────────────────────────────────────────────────────


def generate_rule_page(
    rule_id: str,
    results: BayesianResults,
    exemplars: Dict[str, Any],
    card_images_path: str,
    cards_js: str,
    output_dir: Path,
    prev_rule: Optional[str] = None,
    next_rule: Optional[str] = None,
) -> Path:
    """Generate a detail HTML page for a single rule.

    Combines card exemplar images, Altair charts (posterior bars,
    prior-vs-likelihood scatter, optional diagnosticity), and a
    hypotheses table into a self-contained HTML file.

    Parameters
    ----------
    rule_id : str
        Identifier for the rule (e.g. "all_red").
    results : BayesianResults
        Full normalized results from :func:`data.load_results`.
    exemplars : Dict[str, Any]
        Exemplar catalogue from :func:`cards.load_exemplars`.
    card_images_path : str
        Relative path from the output rule HTML to the card images
        directory (e.g. "../../stim").
    cards_js : str
        JavaScript source code for the card renderer, inlined into
        the HTML page.
    output_dir : Path
        Directory to write ``<rule_id>.html`` into.  Created if needed.
    prev_rule : str, optional
        Rule ID of the previous rule (for navigation links).
    next_rule : str, optional
        Rule ID of the next rule (for navigation links).

    Returns
    -------
    Path
        Absolute path to the generated HTML file.
    """
    # ── Set up Jinja2 ────────────────────────────────────────────────
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=False,
    )
    template = env.get_template("rule_detail.html")

    # ── Extract rule metadata ────────────────────────────────────────
    # Filter difficulty_df for this rule and convert the single row to
    # a dict for easy template access.
    rule_row = results.difficulty_df[
        results.difficulty_df["rule_id"] == rule_id
    ]
    difficulty = rule_row.iloc[0].to_dict()

    # ── Filter hypotheses and diagnosticity ──────────────────────────
    rule_hyps = results.hypotheses_df[
        results.hypotheses_df["rule_id"] == rule_id
    ]
    rule_diag = results.diagnosticity_df[
        results.diagnosticity_df["rule_id"] == rule_id
    ]

    # ── Card hands ───────────────────────────────────────────────────
    hands = get_rule_hands(exemplars, rule_id)
    hands_json = hands_to_json(hands, card_images_path)

    # ── Build Altair chart specs ─────────────────────────────────────
    # Posterior bar chart (always present).
    chart_posterior = json.dumps(posterior_bars(rule_hyps, rule_id).to_dict())

    # Prior vs likelihood scatter (always present).
    chart_prior_lik = json.dumps(prior_vs_likelihood(rule_hyps).to_dict())

    # Diagnosticity bars (only if data is available for this rule).
    has_diagnosticity = len(rule_diag) > 0
    chart_diag = (
        json.dumps(diagnosticity_bars(rule_diag).to_dict())
        if has_diagnosticity
        else None
    )

    # ── Build hypotheses list ────────────────────────────────────────
    # Sorted by rank (ascending) so rank-1 appears first in the table.
    hypotheses = (
        rule_hyps.sort_values("rank", ascending=True)
        .to_dict("records")
    )

    # ── Render template ──────────────────────────────────────────────
    html = template.render(
        rule_id=rule_id,
        group_label=difficulty["group_label"],
        answer=difficulty["answer"],
        difficulty=difficulty,
        prev_rule=prev_rule,
        next_rule=next_rule,
        hands_json=hands_json,
        cards_js=cards_js,
        chart_posterior=chart_posterior,
        chart_prior_lik=chart_prior_lik,
        has_diagnosticity=has_diagnosticity,
        chart_diag=chart_diag,
        hypotheses=hypotheses,
    )

    # ── Write to disk ────────────────────────────────────────────────
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{rule_id}.html"
    out_path.write_text(html, encoding="utf-8")

    return out_path
