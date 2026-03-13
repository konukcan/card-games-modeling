"""Card rendering utilities for the gallery visualization pipeline.

Maps card rank+suit to PNG filenames, loads frozen exemplars,
and serializes hand data as JSON for the JavaScript renderer.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


# Mapping from short rank codes to the word form used in PNG filenames.
# Card images follow the pattern: {WORD}_OF_{SUIT}.png
RANK_WORDS: Dict[str, str] = {
    "2": "TWO",
    "3": "THREE",
    "4": "FOUR",
    "5": "FIVE",
    "6": "SIX",
    "7": "SEVEN",
    "8": "EIGHT",
    "9": "NINE",
    "10": "TEN",
    "J": "JACK",
    "Q": "QUEEN",
    "K": "KING",
    "A": "ACE",
}


def rank_to_filename(rank: str, suit: str) -> str:
    """Convert a card's rank and suit to its PNG filename.

    Args:
        rank: Short rank code (e.g. "2", "J", "A", "10").
        suit: Uppercase suit name (e.g. "HEARTS", "SPADES").

    Returns:
        Filename like "TWO_OF_HEARTS.png".

    Raises:
        KeyError: If rank is not recognised.
    """
    word = RANK_WORDS[rank]
    return f"{word}_OF_{suit}.png"


def load_exemplars(path: Path) -> Dict[str, Any]:
    """Load frozen-exemplars.json and return a dict keyed by rule id.

    The input JSON has the structure::

        {
            "metadata": {...},
            "catalogue": [
                {"id": "all_red", "group": 1, "hands_primary": [...], ...},
                ...
            ]
        }

    Args:
        path: Path to frozen-exemplars.json.

    Returns:
        Dict mapping rule_id strings to their catalogue entry dicts.
    """
    with open(path, "r") as f:
        data = json.load(f)

    # Build a lookup dict keyed by the rule's "id" field
    return {entry["id"]: entry for entry in data["catalogue"]}


def get_rule_hands(
    exemplars: Dict[str, Any], rule_id: str
) -> List[List[Dict[str, str]]]:
    """Get the 6 primary hands for a given rule.

    Args:
        exemplars: Dict returned by load_exemplars().
        rule_id: Rule identifier (e.g. "all_red").

    Returns:
        List of 6 hands, each hand being a list of card dicts
        with "suit" and "rank" keys.

    Raises:
        KeyError: If rule_id is not found.
    """
    return exemplars[rule_id]["hands_primary"]


def hands_to_json(hands: List[List[Dict[str, str]]], card_images_path: str) -> str:
    """Serialize hands as JSON with image_path for each card.

    Each card dict gets an added "image_path" field pointing to the
    card's PNG relative to the given base path.

    Args:
        hands: List of hands (each a list of card dicts with suit/rank).
        card_images_path: Base path prefix for card images
            (e.g. "../../stim").

    Returns:
        JSON string with structure::

            {"hands": [[{"suit": "...", "rank": "...", "image_path": "..."}, ...], ...]}
    """
    enriched_hands = []
    for hand in hands:
        enriched_cards = []
        for card in hand:
            filename = rank_to_filename(card["rank"], card["suit"])
            enriched_cards.append(
                {
                    "suit": card["suit"],
                    "rank": card["rank"],
                    "image_path": f"{card_images_path}/{filename}",
                }
            )
        enriched_hands.append(enriched_cards)

    return json.dumps({"hands": enriched_hands})


def test_hands_to_json(
    representative_hands: Dict[str, List[Dict[str, Any]]],
    card_images_path: str,
) -> str:
    """Serialize representative test hands as JSON with image paths.

    Takes a dict with keys ``easy_accept``, ``easy_reject``,
    ``ambiguous``, each containing a list of hand entries with card
    data and metrics.  Each card gets an ``image_path`` field.

    Args:
        representative_hands: Dict from DiagnosticityResults.representative_hands
            for a single rule.
        card_images_path: Base path prefix for card images.

    Returns:
        JSON string with structure::

            {
              "easy_accept": [{"hand": [{card with image_path}, ...], "p_accept": ..., ...}],
              "easy_reject": [...],
              "ambiguous": [...]
            }
    """
    result: Dict[str, List[Dict[str, Any]]] = {}
    for category in ("easy_accept", "easy_reject", "ambiguous"):
        entries = representative_hands.get(category, [])
        enriched_entries = []
        for entry in entries:
            enriched_cards = []
            for card in entry["hand"]:
                filename = rank_to_filename(card["rank"], card["suit"])
                enriched_cards.append({
                    "suit": card["suit"],
                    "rank": card["rank"],
                    "image_path": f"{card_images_path}/{filename}",
                })
            enriched_entry = {
                "hand": enriched_cards,
                "p_accept": entry.get("p_accept"),
                "confidence": entry.get("confidence"),
                "ground_truth": entry.get("ground_truth"),
                "correct_prediction": entry.get("correct_prediction"),
            }
            enriched_entries.append(enriched_entry)
        result[category] = enriched_entries
    return json.dumps(result)
