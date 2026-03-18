"""Tests for Phase 1b hypothesis data loader.

Tests verify:
- Basic loading and filtering behaviour (format, passed_only)
- DSL cross-referencing via injected_hypotheses.json
- Exemplar hands parsing from hands_shown field
- Hand string -> Card object conversion
"""

import json
import sys
import os
import tempfile
from pathlib import Path

import pytest

# Add project paths so that both llm.grammar_comparison and rules.cards resolve.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from llm.grammar_comparison.data_loader import (
    load_phase1b_hypotheses,
    _parse_hand_string,
)
from rules.cards import Card, Suit, Rank


# ---------------------------------------------------------------------------
# Fixtures: create minimal Phase 1b JSON and injected-hypotheses JSON
# ---------------------------------------------------------------------------

@pytest.fixture
def phase1b_dir(tmp_path):
    """Create a temporary directory with sample Phase 1b JSON files."""
    # File 1: dsl-constrained format, 3 hypotheses (2 passed, 1 failed)
    file1 = {
        "rule_id": "all_even",
        "format": "dsl-constrained",
        "source_model": "gemini-pro",
        "hands_shown": [
            "2♥ 6♥ 10♠ 4♥ 6♦ 6♠",
            "8♥ 6♦ 6♠ 2♦ 10♠ 4♠",
            "2♦ 10♠ 10♥ 2♠ 4♥ 6♠",
            "2♦ 2♣ 2♥ 4♠ 6♣ 10♠",
            "4♠ 6♥ 8♥ 8♣ 10♦ 2♠",
            "10♠ 4♣ 4♠ 10♣ 2♠ 4♦",
        ],
        "hypotheses": [
            {
                "rank": 1,
                "nl_description": "All cards have an even rank.",
                "confidence": "HIGH",
                "code": "rule = lambda hand: all(lambda c: eq(mod(rank_val(c))(2))(0))(hand)",
                "judge_verdict": {"verdict": "PASS", "explanation": "correct"},
                "passed": True,
            },
            {
                "rank": 2,
                "nl_description": "Sum of ranks is even.",
                "confidence": "MEDIUM",
                "code": "rule = lambda hand: eq(mod(sum_ranks(hand))(2))(0)",
                "judge_verdict": {"verdict": "PASS", "explanation": "ok"},
                "passed": True,
            },
            {
                "rank": 3,
                "nl_description": "Hand has a pair.",
                "confidence": "LOW",
                "code": "rule = lambda hand: has_pair(hand)",
                "judge_verdict": {"verdict": "FAIL", "explanation": "wrong"},
                "passed": False,
            },
        ],
    }
    (tmp_path / "gemini-pro__dsl-constrained__all_even.json").write_text(
        json.dumps(file1)
    )

    # File 2: python-freeform format (should be skipped by default)
    file2 = {
        "rule_id": "all_odd",
        "format": "python-freeform",
        "source_model": "gemini-pro",
        "hands_shown": [
            "3♠ 5♥ 7♣ 9♦ J♠ A♥",
        ],
        "hypotheses": [
            {
                "rank": 1,
                "nl_description": "All cards odd.",
                "confidence": "HIGH",
                "code": "rule = lambda hand: all(c.rank % 2 == 1 for c in hand)",
                "judge_verdict": {"verdict": "PASS", "explanation": "ok"},
                "passed": True,
            },
        ],
    }
    (tmp_path / "gemini-pro__python-freeform__all_odd.json").write_text(
        json.dumps(file2)
    )

    return tmp_path


@pytest.fixture
def injected_file(tmp_path):
    """Create a temporary injected_hypotheses.json with DSL translations."""
    data = [
        # Ground-truth entry (should be ignored)
        {
            "id": "true__all_even",
            "source": "catalogue",
            "dsl_program": "(lambda all (lambda eq (mod (rank_val $0) 2) 0) $0)",
            "origin": {},
        },
        # LLM hypothesis matching rank 1 of all_even
        {
            "id": "llm__all_even__hyp0",
            "source": "llm_foil",
            "dsl_program": "(lambda all (lambda eq (mod (rank_val $0) 2) 0) $0)",
            "origin": {
                "hypothesis_text": "All cards have an even rank.",
                "python_lambda": "rule = lambda hand: ...",
                "source_model": "gemini-2.5-flash",
                "original_rule_id": "all_even",
            },
        },
    ]
    path = tmp_path / "injected_hypotheses.json"
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# Tests: _parse_hand_string
# ---------------------------------------------------------------------------


class TestParseHandString:
    """Tests for the _parse_hand_string helper function."""

    def test_unicode_suit_symbols(self):
        """Parse a hand using Unicode suit symbols."""
        cards = _parse_hand_string("2♠ 5♥ K♣ 3♦ 7♠ J♥")
        assert len(cards) == 6
        assert cards[0] == Card(Suit.SPADES, Rank.TWO)
        assert cards[1] == Card(Suit.HEARTS, Rank.FIVE)
        assert cards[2] == Card(Suit.CLUBS, Rank.KING)
        assert cards[3] == Card(Suit.DIAMONDS, Rank.THREE)
        assert cards[4] == Card(Suit.SPADES, Rank.SEVEN)
        assert cards[5] == Card(Suit.HEARTS, Rank.JACK)

    def test_ten_rank(self):
        """10 is a multi-character rank that must parse correctly."""
        cards = _parse_hand_string("10♠ 10♥ 10♣ 10♦ 2♠ 3♥")
        assert cards[0].rank == Rank.TEN
        assert cards[0].suit == Suit.SPADES
        assert all(c.rank == Rank.TEN for c in cards[:4])

    def test_letter_suit_abbreviations(self):
        """Handle single-letter suit abbreviations (S, H, D, C)."""
        cards = _parse_hand_string("AS KH QD JC")
        assert len(cards) == 4
        assert cards[0] == Card(Suit.SPADES, Rank.ACE)
        assert cards[1] == Card(Suit.HEARTS, Rank.KING)
        assert cards[2] == Card(Suit.DIAMONDS, Rank.QUEEN)
        assert cards[3] == Card(Suit.CLUBS, Rank.JACK)

    def test_all_card_objects_have_suit_and_rank(self):
        """Every parsed card must have valid suit and rank attributes."""
        cards = _parse_hand_string("2♥ 6♥ 10♠ 4♥ 6♦ 6♠")
        for card in cards:
            assert isinstance(card, Card)
            assert isinstance(card.suit, Suit)
            assert isinstance(card.rank, Rank)

    def test_invalid_suit_raises(self):
        """Unknown suit symbol should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown suit"):
            _parse_hand_string("2X")

    def test_invalid_rank_raises(self):
        """Unknown rank string should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown rank"):
            _parse_hand_string("Z♠")


# ---------------------------------------------------------------------------
# Tests: load_phase1b_hypotheses (basic behaviour)
# ---------------------------------------------------------------------------


class TestLoadPhase1bDefaults:
    """Test default behaviour: dsl-constrained format, passed_only=True."""

    def test_returns_list_of_dicts(self, phase1b_dir, injected_file):
        result = load_phase1b_hypotheses(
            phase1b_dir=phase1b_dir, injected_path=injected_file
        )
        assert isinstance(result, list)
        assert all(isinstance(r, dict) for r in result)

    def test_filters_to_dsl_constrained_only(self, phase1b_dir, injected_file):
        """By default only dsl-constrained files are loaded."""
        result = load_phase1b_hypotheses(
            phase1b_dir=phase1b_dir, injected_path=injected_file
        )
        rule_ids = {r["rule_id"] for r in result}
        assert "all_odd" not in rule_ids
        assert "all_even" in rule_ids

    def test_filters_passed_only(self, phase1b_dir, injected_file):
        """By default only passed hypotheses are included."""
        result = load_phase1b_hypotheses(
            phase1b_dir=phase1b_dir, injected_path=injected_file
        )
        assert len(result) == 2
        assert all(r["judge_verdict"] == "PASS" for r in result)

    def test_required_keys_present(self, phase1b_dir, injected_file):
        """Each dict must have all required keys including exemplar_hands."""
        required = {
            "rule_id", "rank", "confidence", "nl_description",
            "dsl_code", "python_code", "judge_verdict", "source_model",
            "exemplar_hands",
        }
        result = load_phase1b_hypotheses(
            phase1b_dir=phase1b_dir, injected_path=injected_file
        )
        for r in result:
            assert required.issubset(r.keys()), f"Missing keys: {required - r.keys()}"

    def test_dsl_code_populated_when_match_exists(self, phase1b_dir, injected_file):
        """Rank 1 of all_even matches injected hyp => dsl_code should be set."""
        result = load_phase1b_hypotheses(
            phase1b_dir=phase1b_dir, injected_path=injected_file
        )
        rank1 = [r for r in result if r["rank"] == 1][0]
        assert rank1["dsl_code"] is not None
        assert "all" in rank1["dsl_code"]

    def test_dsl_code_none_when_no_match(self, phase1b_dir, injected_file):
        """Rank 2 of all_even has no injected match => dsl_code should be None."""
        result = load_phase1b_hypotheses(
            phase1b_dir=phase1b_dir, injected_path=injected_file
        )
        rank2 = [r for r in result if r["rank"] == 2][0]
        assert rank2["dsl_code"] is None

    def test_python_code_from_phase1b(self, phase1b_dir, injected_file):
        """python_code should come from the Phase 1b 'code' field."""
        result = load_phase1b_hypotheses(
            phase1b_dir=phase1b_dir, injected_path=injected_file
        )
        rank1 = [r for r in result if r["rank"] == 1][0]
        assert "lambda hand" in rank1["python_code"]


# ---------------------------------------------------------------------------
# Tests: exemplar_hands in loaded hypotheses
# ---------------------------------------------------------------------------


class TestExemplarHands:
    """Test that exemplar hands are correctly parsed and included."""

    def test_exemplar_hands_field_exists(self, phase1b_dir, injected_file):
        """Every loaded hypothesis must have an exemplar_hands field."""
        result = load_phase1b_hypotheses(
            phase1b_dir=phase1b_dir, injected_path=injected_file
        )
        for r in result:
            assert "exemplar_hands" in r

    def test_exemplar_hands_has_six_hands(self, phase1b_dir, injected_file):
        """The all_even fixture has 6 hands_shown entries."""
        result = load_phase1b_hypotheses(
            phase1b_dir=phase1b_dir, injected_path=injected_file
        )
        for r in result:
            assert len(r["exemplar_hands"]) == 6, (
                f"Expected 6 exemplar hands for {r['rule_id']}, got {len(r['exemplar_hands'])}"
            )

    def test_each_hand_has_six_cards(self, phase1b_dir, injected_file):
        """Each exemplar hand should contain 6 Card objects."""
        result = load_phase1b_hypotheses(
            phase1b_dir=phase1b_dir, injected_path=injected_file
        )
        for r in result:
            for i, hand in enumerate(r["exemplar_hands"]):
                assert len(hand) == 6, (
                    f"Hand {i} for {r['rule_id']} has {len(hand)} cards, expected 6"
                )

    def test_cards_have_valid_suit_and_rank(self, phase1b_dir, injected_file):
        """Every card in every exemplar hand must have valid Suit and Rank."""
        result = load_phase1b_hypotheses(
            phase1b_dir=phase1b_dir, injected_path=injected_file
        )
        for r in result:
            for hand in r["exemplar_hands"]:
                for card in hand:
                    assert isinstance(card, Card)
                    assert isinstance(card.suit, Suit)
                    assert isinstance(card.rank, Rank)

    def test_known_rule_has_expected_first_hand(self, phase1b_dir, injected_file):
        """Verify the first hand of all_even matches the fixture data: '2H 6H 10S 4H 6D 6S'."""
        result = load_phase1b_hypotheses(
            phase1b_dir=phase1b_dir, injected_path=injected_file
        )
        rank1 = [r for r in result if r["rule_id"] == "all_even" and r["rank"] == 1][0]
        first_hand = rank1["exemplar_hands"][0]

        # "2♥ 6♥ 10♠ 4♥ 6♦ 6♠"
        assert first_hand[0] == Card(Suit.HEARTS, Rank.TWO)
        assert first_hand[1] == Card(Suit.HEARTS, Rank.SIX)
        assert first_hand[2] == Card(Suit.SPADES, Rank.TEN)
        assert first_hand[3] == Card(Suit.HEARTS, Rank.FOUR)
        assert first_hand[4] == Card(Suit.DIAMONDS, Rank.SIX)
        assert first_hand[5] == Card(Suit.SPADES, Rank.SIX)

    def test_all_hypotheses_same_rule_share_hands(self, phase1b_dir, injected_file):
        """All hypotheses for the same rule should reference the same exemplar hands list."""
        result = load_phase1b_hypotheses(
            phase1b_dir=phase1b_dir, injected_path=injected_file
        )
        all_even = [r for r in result if r["rule_id"] == "all_even"]
        assert len(all_even) == 2  # ranks 1 and 2 (passed_only)

        # The lists should be identical (same object since they come from same parse)
        assert all_even[0]["exemplar_hands"] == all_even[1]["exemplar_hands"]

    def test_missing_hands_shown_gives_empty_list(self, tmp_path, injected_file):
        """If hands_shown is absent from JSON, exemplar_hands should be an empty list."""
        file_data = {
            "rule_id": "no_hands",
            "format": "dsl-constrained",
            "source_model": "gemini-pro",
            # No hands_shown field
            "hypotheses": [
                {
                    "rank": 1,
                    "nl_description": "Test.",
                    "confidence": "HIGH",
                    "code": "rule = lambda hand: True",
                    "judge_verdict": {"verdict": "PASS", "explanation": "ok"},
                    "passed": True,
                },
            ],
        }
        (tmp_path / "gemini-pro__dsl-constrained__no_hands.json").write_text(
            json.dumps(file_data)
        )
        result = load_phase1b_hypotheses(
            phase1b_dir=tmp_path, injected_path=injected_file
        )
        no_hands = [r for r in result if r["rule_id"] == "no_hands"]
        assert len(no_hands) == 1
        assert no_hands[0]["exemplar_hands"] == []


# ---------------------------------------------------------------------------
# Tests: loading options
# ---------------------------------------------------------------------------


class TestLoadPhase1bOptions:
    """Test non-default loading options."""

    def test_passed_only_false_includes_failures(self, phase1b_dir, injected_file):
        result = load_phase1b_hypotheses(
            phase1b_dir=phase1b_dir,
            injected_path=injected_file,
            passed_only=False,
        )
        verdicts = [r["judge_verdict"] for r in result]
        assert "FAIL" in verdicts
        assert len(result) == 3

    def test_format_filter_python_freeform(self, phase1b_dir, injected_file):
        result = load_phase1b_hypotheses(
            phase1b_dir=phase1b_dir,
            injected_path=injected_file,
            format_filter="python-freeform",
        )
        assert len(result) == 1
        assert result[0]["rule_id"] == "all_odd"

    def test_no_injected_file_still_works(self, phase1b_dir):
        """If injected_path is None, dsl_code should always be None."""
        result = load_phase1b_hypotheses(
            phase1b_dir=phase1b_dir,
            injected_path=None,
        )
        assert len(result) == 2
        assert all(r["dsl_code"] is None for r in result)


# ---------------------------------------------------------------------------
# Tests: real data smoke tests
# ---------------------------------------------------------------------------


class TestLoadPhase1bRealData:
    """Smoke tests against the actual project data (skipped if files missing)."""

    REAL_PHASE1B = Path(__file__).resolve().parents[2] / "results" / "phase1b"
    REAL_INJECTED = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "gallery_analysis"
        / "data"
        / "injected_hypotheses.json"
    )

    @pytest.mark.skipif(
        not REAL_PHASE1B.exists(), reason="Real Phase 1b data not found"
    )
    def test_loads_real_data_without_error(self):
        injected = self.REAL_INJECTED if self.REAL_INJECTED.exists() else None
        result = load_phase1b_hypotheses(
            phase1b_dir=self.REAL_PHASE1B,
            injected_path=injected,
        )
        assert len(result) > 0
        for r in result:
            assert r["rule_id"]
            assert 1 <= r["rank"] <= 5

    @pytest.mark.skipif(
        not REAL_PHASE1B.exists() or not REAL_INJECTED.exists(),
        reason="Real data not found",
    )
    def test_some_dsl_codes_populated(self):
        result = load_phase1b_hypotheses(
            phase1b_dir=self.REAL_PHASE1B,
            injected_path=self.REAL_INJECTED,
        )
        dsl_hits = [r for r in result if r["dsl_code"] is not None]
        assert len(dsl_hits) > 0, "No DSL codes matched -- check cross-referencing logic"

    @pytest.mark.skipif(
        not REAL_PHASE1B.exists(), reason="Real Phase 1b data not found"
    )
    def test_real_data_has_exemplar_hands(self):
        """All real Phase 1b files should have hands_shown with 6 hands of 6 cards."""
        result = load_phase1b_hypotheses(
            phase1b_dir=self.REAL_PHASE1B,
            injected_path=None,
        )
        for r in result:
            assert len(r["exemplar_hands"]) == 6, (
                f"{r['rule_id']} has {len(r['exemplar_hands'])} exemplar hands, expected 6"
            )
            for hand in r["exemplar_hands"]:
                assert len(hand) == 6, (
                    f"{r['rule_id']} has a hand with {len(hand)} cards, expected 6"
                )
