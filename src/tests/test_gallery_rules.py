"""Tests for gallery rule predicates ported from gallery-rules.js."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import Card, Hand, Suit, Rank, RANK_VALUES, card_color, Color, H, D, S, C

def test_all_red_positive():
    """6 red cards should satisfy all_red."""
    from gallery_analysis.gallery_rules import all_red
    hand = [H("A"), H("K"), D("Q"), D("J"), H("10"), D("9")]
    assert all_red(hand) is True

def test_all_red_negative():
    """A hand with a black card should fail all_red."""
    from gallery_analysis.gallery_rules import all_red
    hand = [H("A"), H("K"), D("Q"), S("J"), H("10"), D("9")]
    assert all_red(hand) is False

def test_all_same_suit_positive():
    """6 spades should satisfy all_same_suit."""
    from gallery_analysis.gallery_rules import all_same_suit
    hand = [S("A"), S("K"), S("Q"), S("J"), S("10"), S("9")]
    assert all_same_suit(hand) is True

def test_all_same_suit_negative():
    """Mixed suits should fail all_same_suit."""
    from gallery_analysis.gallery_rules import all_same_suit
    hand = [S("A"), H("K"), S("Q"), S("J"), S("10"), S("9")]
    assert all_same_suit(hand) is False

def test_strict_increasing_positive():
    """Ranks in ascending order should satisfy strict_increasing."""
    from gallery_analysis.gallery_rules import strict_increasing
    hand = [H("2"), S("5"), D("7"), C("9"), H("J"), S("A")]
    assert strict_increasing(hand) is True

def test_strict_increasing_negative():
    """Non-ascending ranks should fail strict_increasing."""
    from gallery_analysis.gallery_rules import strict_increasing
    hand = [H("A"), S("K"), D("Q"), C("J"), H("10"), S("9")]
    assert strict_increasing(hand) is False
