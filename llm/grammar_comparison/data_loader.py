"""Load Phase 1b LLM-generated hypotheses and cross-reference with DSL translations.

This module reads the Phase 1b judge-verified hypothesis JSON files produced by
the LLM annotation pipeline and optionally enriches each hypothesis with its
s-expression DSL translation from the injected_hypotheses.json file.

Typical usage
-------------
    from llm.grammar_comparison.data_loader import load_phase1b_hypotheses

    hypotheses = load_phase1b_hypotheses()          # defaults: dsl-constrained, passed_only
    hypotheses = load_phase1b_hypotheses(passed_only=False)   # include failures too
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import List, Optional

# ---------------------------------------------------------------------------
# Path setup: allow importing from the main src/ tree
# ---------------------------------------------------------------------------
_SRC_DIR = str(Path(__file__).resolve().parents[2] / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from rules.cards import Card, Suit, Rank


# Default paths relative to this file's location within the repo.
# llm/grammar_comparison/data_loader.py  =>  llm/results/phase1b/
_DEFAULT_PHASE1B_DIR = Path(__file__).resolve().parents[1] / "results" / "phase1b"

# src/gallery_analysis/data/injected_hypotheses.json
_DEFAULT_INJECTED_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "gallery_analysis"
    / "data"
    / "injected_hypotheses.json"
)


def _build_dsl_lookup(injected_path: Path) -> dict[tuple[str, str], str]:
    """Build a lookup from (rule_id, hypothesis_text) -> dsl_program.

    The injected_hypotheses.json contains both ground-truth entries (id starts
    with "true__") and LLM foil entries (id starts with "llm__").  We only
    care about the LLM entries here, since those are the ones that correspond
    to Phase 1b hypotheses.

    Parameters
    ----------
    injected_path : Path
        Path to injected_hypotheses.json.

    Returns
    -------
    dict
        Mapping of (rule_id, normalised_hypothesis_text) -> dsl_program string.
    """
    with open(injected_path, "r") as f:
        data = json.load(f)

    lookup: dict[tuple[str, str], str] = {}
    for entry in data:
        # Only LLM foil entries have origin.original_rule_id and hypothesis_text
        # Phase 1b entries start with "phase1b__", earlier ones with "llm__"
        if not (entry["id"].startswith("llm__") or entry["id"].startswith("phase1b__")):
            continue
        origin = entry.get("origin", {})
        rule_id = origin.get("original_rule_id")
        hyp_text = origin.get("hypothesis_text")
        dsl = entry.get("dsl_program")
        if rule_id and hyp_text and dsl:
            # Normalise whitespace for robust matching
            key = (rule_id, _normalise(hyp_text))
            lookup[key] = dsl

    return lookup


def _normalise(text: str) -> str:
    """Lowercase and collapse whitespace for fuzzy text matching."""
    return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------
# Suit and rank symbol mappings for parsing hand strings like "2♠ 5♥ K♣"
# ---------------------------------------------------------------------------

# Maps Unicode suit symbols and single-letter abbreviations to Suit enum values.
_SUIT_SYMBOL: dict[str, Suit] = {
    "♠": Suit.SPADES,
    "♥": Suit.HEARTS,
    "♦": Suit.DIAMONDS,
    "♣": Suit.CLUBS,
    "S": Suit.SPADES,
    "H": Suit.HEARTS,
    "D": Suit.DIAMONDS,
    "C": Suit.CLUBS,
}

# Maps rank strings (as they appear in hand notation) to Rank enum values.
# Includes both numeric ranks (2-10) and face-card abbreviations (J, Q, K, A).
_RANK_SYMBOL: dict[str, Rank] = {r.value: r for r in Rank}


def _parse_hand_string(hand_str: str) -> List[Card]:
    """Convert a hand string like ``"2♠ 5♥ K♣ 3♦ 7♠ J♥"`` into a list of Card objects.

    Each token in the space-separated string is a card where the last character
    is the suit symbol (Unicode or letter) and the preceding characters are the
    rank (2-10, J, Q, K, A).

    Parameters
    ----------
    hand_str : str
        Space-separated card tokens, e.g. ``"2♠ 5♥ K♣ 3♦ 7♠ J♥"``.

    Returns
    -------
    List[Card]
        Parsed Card objects in the same order as the input string.

    Raises
    ------
    ValueError
        If a token cannot be parsed into a valid rank + suit.
    """
    cards: List[Card] = []
    for token in hand_str.split():
        # The last character is always the suit symbol.
        suit_char = token[-1]
        rank_str = token[:-1]

        suit = _SUIT_SYMBOL.get(suit_char)
        if suit is None:
            raise ValueError(f"Unknown suit symbol '{suit_char}' in token '{token}'")

        rank = _RANK_SYMBOL.get(rank_str)
        if rank is None:
            raise ValueError(f"Unknown rank '{rank_str}' in token '{token}'")

        cards.append(Card(suit, rank))
    return cards


def load_phase1b_hypotheses(
    *,
    phase1b_dir: Optional[Path] = None,
    injected_path: Optional[Path] = _DEFAULT_INJECTED_PATH,
    format_filter: str = "dsl-constrained",
    passed_only: bool = True,
) -> list[dict]:
    """Load Phase 1b hypotheses as a flat list of hypothesis dicts.

    Each hypothesis dict contains:
        - rule_id       (str): The card-game rule this hypothesis was generated for.
        - rank          (int): Confidence rank assigned by the LLM (1 = most confident).
        - confidence    (str): HIGH / MEDIUM / LOW label.
        - nl_description(str): Natural-language description of the hypothesis.
        - dsl_code      (str | None): S-expression DSL translation, if available.
        - python_code   (str): Python lambda from the LLM (the 'code' field).
        - judge_verdict (str): "PASS" or "FAIL" from the code judge.
        - source_model  (str): Which LLM produced this hypothesis (e.g. "gemini-pro").
        - exemplar_hands (List[List[Card]]): The 6 example hands shown to the LLM,
          parsed into Card objects.  All hypotheses for the same rule share the
          same exemplar hands.

    Parameters
    ----------
    phase1b_dir : Path, optional
        Directory containing Phase 1b JSON files.  Defaults to
        ``llm/results/phase1b/`` relative to the repo root.
    injected_path : Path, optional
        Path to ``injected_hypotheses.json`` for DSL cross-referencing.
        Pass ``None`` to skip DSL enrichment entirely.  Defaults to the
        standard location under ``src/gallery_analysis/data/``.
    format_filter : str
        Only load files matching this format slug in the filename
        (e.g. "dsl-constrained", "python-freeform", "webppl").
        Default: "dsl-constrained".
    passed_only : bool
        If True (default), only include hypotheses where the judge verdict
        was PASS (i.e. ``passed`` is True).

    Returns
    -------
    list[dict]
        Flat list of hypothesis records, sorted by (rule_id, rank).
    """
    # Resolve directories
    phase1b_dir = Path(phase1b_dir) if phase1b_dir else _DEFAULT_PHASE1B_DIR
    if not phase1b_dir.exists():
        raise FileNotFoundError(f"Phase 1b directory not found: {phase1b_dir}")

    # Build DSL lookup table (empty dict if no injected file provided)
    dsl_lookup: dict[tuple[str, str], str] = {}
    if injected_path is not None:
        injected_path = Path(injected_path)
        if injected_path.exists():
            dsl_lookup = _build_dsl_lookup(injected_path)

    # Iterate over Phase 1b JSON files matching the requested format
    # Filename pattern: {model}__{format}__{rule_id}.json
    results: list[dict] = []

    for json_file in sorted(phase1b_dir.glob("*.json")):
        # Parse filename to extract format slug
        parts = json_file.stem.split("__")
        if len(parts) < 3:
            continue  # skip unexpected filenames
        file_format = parts[1]
        if file_format != format_filter:
            continue

        with open(json_file, "r") as f:
            data = json.load(f)

        rule_id = data["rule_id"]
        source_model = data.get("source_model", parts[0])

        # Parse exemplar hands (per-file, shared across all hypotheses for this rule).
        # The hands_shown field contains space-separated card strings like "2♠ 5♥ K♣".
        exemplar_hands: List[List[Card]] = []
        for hand_str in data.get("hands_shown", []):
            exemplar_hands.append(_parse_hand_string(hand_str))

        for hyp in data.get("hypotheses", []):
            # Apply passed filter
            if passed_only and not hyp.get("passed", False):
                continue

            # Extract judge verdict string
            verdict_obj = hyp.get("judge_verdict", {})
            verdict_str = verdict_obj.get("verdict", "UNKNOWN") if isinstance(verdict_obj, dict) else str(verdict_obj)

            # Cross-reference DSL code via normalised hypothesis text
            nl_desc = hyp.get("nl_description", "")
            lookup_key = (rule_id, _normalise(nl_desc))
            dsl_code = dsl_lookup.get(lookup_key)

            results.append({
                "rule_id": rule_id,
                "rank": hyp["rank"],
                "confidence": hyp.get("confidence", "UNKNOWN"),
                "nl_description": nl_desc,
                "dsl_code": dsl_code,
                "python_code": hyp.get("code", ""),
                "judge_verdict": verdict_str,
                "source_model": source_model,
                "exemplar_hands": exemplar_hands,
            })

    # Sort by rule_id then rank for deterministic ordering
    results.sort(key=lambda r: (r["rule_id"], r["rank"]))
    return results
