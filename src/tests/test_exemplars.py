"""Tests for exemplar loading and probe set generation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_load_exemplars_count():
    """Should load 60 rules from frozen-exemplars.json."""
    from gallery_analysis.exemplars import load_exemplars
    exemplars = load_exemplars()
    assert len(exemplars) == 60

def test_load_exemplars_hand_size():
    """Each rule should have 6 primary hands of 6 cards each."""
    from gallery_analysis.exemplars import load_exemplars
    exemplars = load_exemplars()
    first = exemplars[list(exemplars.keys())[0]]
    assert len(first["hands_primary"]) == 6
    assert all(len(h) == 6 for h in first["hands_primary"])

def test_load_exemplars_card_type():
    """Cards should be Card objects with proper suit and rank."""
    from gallery_analysis.exemplars import load_exemplars
    from rules.cards import Card
    exemplars = load_exemplars()
    first = exemplars[list(exemplars.keys())[0]]
    card = first["hands_primary"][0][0]
    assert isinstance(card, Card)

def test_exemplars_satisfy_rules():
    """Primary hands should satisfy their corresponding rule."""
    from gallery_analysis.exemplars import load_exemplars
    from gallery_analysis.gallery_rules import GALLERY_RULES
    exemplars = load_exemplars()
    # Test a few known rules
    for rule_id in ["all_red", "strict_increasing", "all_same_suit"]:
        if rule_id in exemplars and rule_id in GALLERY_RULES:
            predicate = GALLERY_RULES[rule_id]["predicate"]
            hands = exemplars[rule_id]["hands_primary"]
            for hand in hands:
                assert predicate(hand), f"{rule_id} exemplar failed: {hand}"

def test_generate_probe_set_size():
    """Probe set should have the requested number of hands."""
    from gallery_analysis.exemplars import generate_probe_set
    probes = generate_probe_set(200, seed=42)
    assert len(probes) == 200

def test_generate_probe_set_deterministic():
    """Same seed should produce same probe set."""
    from gallery_analysis.exemplars import generate_probe_set
    probes1 = generate_probe_set(100, seed=42)
    probes2 = generate_probe_set(100, seed=42)
    for h1, h2 in zip(probes1, probes2):
        assert all(c1 == c2 for c1, c2 in zip(h1, h2))
