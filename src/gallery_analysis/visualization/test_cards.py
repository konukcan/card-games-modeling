"""Tests for card rendering utilities."""

import json
import unittest
from pathlib import Path

from cards import (
    rank_to_filename,
    load_exemplars,
    get_rule_hands,
    hands_to_json,
)

# Real frozen-exemplars path for integration tests
FROZEN_EXEMPLARS = Path(
    "/Users/cankonuk/Documents/self-explanations-project"
    "/card-games/rule-gallery/frozen-exemplars.json"
)


class TestRankToFilename(unittest.TestCase):
    """Test rank_to_filename mapping."""

    def test_two_of_hearts(self):
        self.assertEqual(rank_to_filename("2", "HEARTS"), "TWO_OF_HEARTS.png")

    def test_jack_of_clubs(self):
        self.assertEqual(rank_to_filename("J", "CLUBS"), "JACK_OF_CLUBS.png")

    def test_ace_of_spades(self):
        self.assertEqual(rank_to_filename("A", "SPADES"), "ACE_OF_SPADES.png")

    def test_ten_of_diamonds(self):
        self.assertEqual(rank_to_filename("10", "DIAMONDS"), "TEN_OF_DIAMONDS.png")


class TestLoadExemplars(unittest.TestCase):
    """Test loading frozen-exemplars.json."""

    def setUp(self):
        if not FROZEN_EXEMPLARS.exists():
            self.skipTest("frozen-exemplars.json not found")
        self.exemplars = load_exemplars(FROZEN_EXEMPLARS)

    def test_returns_dict_with_60_rules(self):
        self.assertIsInstance(self.exemplars, dict)
        self.assertEqual(len(self.exemplars), 60)

    def test_all_red_key_exists(self):
        self.assertIn("all_red", self.exemplars)


class TestGetRuleHands(unittest.TestCase):
    """Test extracting primary hands for a rule."""

    def setUp(self):
        if not FROZEN_EXEMPLARS.exists():
            self.skipTest("frozen-exemplars.json not found")
        self.exemplars = load_exemplars(FROZEN_EXEMPLARS)

    def test_returns_6_hands(self):
        hands = get_rule_hands(self.exemplars, "all_red")
        self.assertEqual(len(hands), 6)

    def test_each_hand_has_6_cards(self):
        hands = get_rule_hands(self.exemplars, "all_red")
        for hand in hands:
            self.assertEqual(len(hand), 6)

    def test_each_card_has_suit_and_rank(self):
        hands = get_rule_hands(self.exemplars, "all_red")
        for hand in hands:
            for card in hand:
                self.assertIn("suit", card)
                self.assertIn("rank", card)


class TestHandsToJson(unittest.TestCase):
    """Test JSON serialization with image paths."""

    def setUp(self):
        if not FROZEN_EXEMPLARS.exists():
            self.skipTest("frozen-exemplars.json not found")
        self.exemplars = load_exemplars(FROZEN_EXEMPLARS)

    def test_returns_valid_json(self):
        hands = get_rule_hands(self.exemplars, "all_red")
        result = hands_to_json(hands, "../../stim")
        data = json.loads(result)
        self.assertIn("hands", data)

    def test_each_card_has_image_path(self):
        hands = get_rule_hands(self.exemplars, "all_red")
        result = hands_to_json(hands, "../../stim")
        data = json.loads(result)
        for hand in data["hands"]:
            for card in hand:
                self.assertIn("image_path", card)
                self.assertTrue(card["image_path"].startswith("../../stim/"))
                self.assertTrue(card["image_path"].endswith(".png"))


if __name__ == "__main__":
    unittest.main()
