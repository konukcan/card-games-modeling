"""
Load frozen exemplar hands from the gallery experiment and generate probe sets.

The frozen exemplars are pre-generated hands stored in
card-games/rule-gallery/frozen-exemplars.json. Each rule has 6 "primary" hands
(shown to human participants) and 6 "reserve" hands (for LLM experiments).

The probe set is a deterministic collection of random hands used for
observational equivalence fingerprinting.
"""
import json
import random
from pathlib import Path
from typing import Dict, List, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Card, Hand, Suit, Rank, STR_TO_RANK

# Path to frozen exemplars JSON
# Use git's commondir to find the real repo root (works in worktrees too)
def _find_frozen_exemplars() -> Path:
    """Find frozen-exemplars.json relative to the git repo root."""
    import subprocess
    try:
        # git rev-parse --git-common-dir gives the main .git directory
        result = subprocess.run(
            ["git", "rev-parse", "--git-common-dir"],
            capture_output=True, text=True, check=True,
            cwd=str(Path(__file__).parent)
        )
        git_common = Path(result.stdout.strip())
        if not git_common.is_absolute():
            git_common = (Path(__file__).parent / git_common).resolve()
        # .git's parent is the repo root (card-games-modelling/)
        repo_root = git_common.parent
        # card-games is a sibling of card-games-modelling
        return repo_root.parent / "card-games" / "rule-gallery" / "frozen-exemplars.json"
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback: try relative path from file
        return (
            Path(__file__).parent.parent.parent.parent /
            "card-games" / "rule-gallery" / "frozen-exemplars.json"
        )

FROZEN_EXEMPLARS_PATH = _find_frozen_exemplars()


def _parse_card(card_obj: Dict[str, str]) -> Card:
    """Convert a JSON card object {suit: "HEARTS", rank: "K"} to a Card."""
    suit = Suit(card_obj["suit"])
    rank = STR_TO_RANK[card_obj["rank"]]
    return Card(suit, rank)


def _parse_hand(hand_list: List[Dict[str, str]]) -> Hand:
    """Convert a JSON hand (list of card objects) to a Hand."""
    return [_parse_card(c) for c in hand_list]


def load_exemplars(path: Path = None) -> Dict[str, Dict[str, Any]]:
    """
    Load frozen exemplar hands from JSON.

    Returns a dict keyed by rule_id, each containing:
      - "hands_primary": List of 6 Hand objects (for human experiments)
      - "hands_reserve": List of 6 Hand objects (for LLM experiments)
      - "group": int (difficulty 1-3)
      - "answer": str (human-readable rule description)
    """
    if path is None:
        path = FROZEN_EXEMPLARS_PATH

    with open(path, "r") as f:
        data = json.load(f)

    exemplars = {}
    for entry in data["catalogue"]:
        rule_id = entry["id"]
        exemplars[rule_id] = {
            "hands_primary": [_parse_hand(h) for h in entry["hands_primary"]],
            "hands_reserve": [_parse_hand(h) for h in entry["hands_reserve"]],
            "group": entry["group"],
            "answer": entry["answer"],
        }

    return exemplars


def generate_probe_set(
    n_probes: int = 200,
    hand_size: int = 6,
    seed: int = 42
) -> List[Hand]:
    """
    Generate a deterministic set of random hands for fingerprinting.

    These hands are used to compute observational equivalence fingerprints:
    two hypotheses that produce the same boolean vector on the probe set
    are treated as extensionally equivalent.

    Uses sampling without replacement (each hand has 6 distinct cards)
    to match the gallery experiment's hand generation.
    """
    rng = random.Random(seed)
    deck = [Card(suit, rank) for suit in Suit for rank in Rank]

    probes = []
    for _ in range(n_probes):
        hand = rng.sample(deck, hand_size)
        probes.append(hand)

    return probes
