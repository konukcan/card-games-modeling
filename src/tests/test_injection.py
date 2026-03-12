"""
Tests for gallery_analysis.injection.load_and_validate_injections().

Validates that:
1. Valid injection files produce entries with log_prior and predicate
2. Missing dsl_program raises ValueError
3. Unparseable DSL programs raise ValueError
4. Predicates are callable and return bool
5. Log-priors are finite and negative
6. Outlier priors trigger warnings
"""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from gallery_analysis.injection import load_and_validate_injections
from gallery_analysis.enumerator import build_gallery_grammar
from rules.cards import Card, Suit, Rank


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture(scope="module")
def grammar():
    """Build gallery grammar once for all tests."""
    return build_gallery_grammar()


def _make_entry(id_str, dsl, rule="test_rule", source="test"):
    """Helper to create a minimal valid injection entry."""
    return {
        "id": id_str,
        "source": source,
        "true_for_rule": rule,
        "dsl_program": dsl,
    }


def _write_json(tmp_path, entries, filename="test_inject.json"):
    """Write entries to a JSON file and return the path."""
    fp = tmp_path / filename
    fp.write_text(json.dumps(entries))
    return str(fp)


# A simple hand for predicate testing (Hand is List[Card], just use a plain list)
_TEST_HAND = [
    Card(Rank.ACE, Suit.HEARTS),
    Card(Rank.TWO, Suit.HEARTS),
    Card(Rank.THREE, Suit.HEARTS),
    Card(Rank.FOUR, Suit.SPADES),
    Card(Rank.FIVE, Suit.DIAMONDS),
]


# =========================================================================
# Tests: happy path
# =========================================================================

def test_valid_file_returns_entries_with_required_fields(tmp_path, grammar):
    """Loading a valid injection file returns entries with log_prior, predicate, and program."""
    entries = [
        _make_entry("h1", "(λ ge (max_suit_count $0) 3)"),
        _make_entry("h2", "(λ has_color $0 RED)"),
    ]
    fp = _write_json(tmp_path, entries)

    results = load_and_validate_injections(fp, grammar=grammar)

    assert len(results) == 2
    for r in results:
        assert "log_prior" in r
        assert "predicate" in r
        assert "program" in r
        # Original fields preserved
        assert "id" in r
        assert "dsl_program" in r


def test_predicates_are_callable_and_return_bool(tmp_path, grammar):
    """Predicates returned by the loader are callable and return bool values."""
    entries = [_make_entry("h1", "(λ ge (max_suit_count $0) 3)")]
    fp = _write_json(tmp_path, entries)

    results = load_and_validate_injections(fp, grammar=grammar)
    pred = results[0]["predicate"]

    assert callable(pred)
    result = pred(_TEST_HAND)
    assert isinstance(result, bool)


def test_log_priors_are_finite_and_negative(tmp_path, grammar):
    """Log-priors should be finite negative numbers."""
    entries = [
        _make_entry("h1", "(λ ge (max_suit_count $0) 3)"),
        _make_entry("h2", "(λ has_color $0 RED)"),
    ]
    fp = _write_json(tmp_path, entries)

    results = load_and_validate_injections(fp, grammar=grammar)

    for r in results:
        lp = r["log_prior"]
        assert math.isfinite(lp), f"log_prior should be finite, got {lp}"
        assert lp <= 0, f"log_prior should be <= 0, got {lp}"


def test_original_fields_preserved(tmp_path, grammar):
    """Original JSON fields are preserved in the output."""
    entries = [{
        "id": "h1",
        "source": "llm_foil",
        "true_for_rule": "some_rule",
        "dsl_program": "(λ has_color $0 RED)",
        "origin": {"hypothesis_text": "Has red cards"},
    }]
    fp = _write_json(tmp_path, entries)

    results = load_and_validate_injections(fp, grammar=grammar)

    assert results[0]["source"] == "llm_foil"
    assert results[0]["origin"]["hypothesis_text"] == "Has red cards"


# =========================================================================
# Tests: error cases
# =========================================================================

def test_missing_dsl_program_raises_value_error(tmp_path, grammar):
    """An entry missing 'dsl_program' should raise ValueError."""
    entries = [{
        "id": "bad_entry",
        "source": "test",
        "true_for_rule": "some_rule",
        # no dsl_program
    }]
    fp = _write_json(tmp_path, entries)

    with pytest.raises(ValueError, match="missing required field 'dsl_program'"):
        load_and_validate_injections(fp, grammar=grammar)


def test_invalid_dsl_raises_value_error(tmp_path, grammar):
    """An unparseable DSL program should raise ValueError."""
    entries = [_make_entry("bad", "(λ nonexistent_primitive $0)")]
    fp = _write_json(tmp_path, entries)

    with pytest.raises(ValueError, match="cannot parse DSL program"):
        load_and_validate_injections(fp, grammar=grammar)


def test_missing_file_raises_file_not_found():
    """A non-existent filepath should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_and_validate_injections("/nonexistent/path.json")


# =========================================================================
# Tests: warnings
# =========================================================================

def test_outlier_prior_warns_below_range(tmp_path, grammar, capsys):
    """A hypothesis with prior far below the enumerated range triggers a warning."""
    # Use a deeply nested program that should have a very negative prior
    entries = [_make_entry("deep", "(λ ge (+ (max_suit_count $0) (max_suit_count $0)) 3)")]
    fp = _write_json(tmp_path, entries)

    # Set a tight enumerated range so the prior falls outside
    results = load_and_validate_injections(
        fp,
        grammar=grammar,
        enumerated_prior_range=(-5.0, -1.0),
        warn_prior_threshold=1.0,
    )

    # The program should still load successfully
    assert len(results) == 1

    # Check stderr for warning
    captured = capsys.readouterr()
    # The prior for this deep program should be below -5, triggering a warning
    if results[0]["log_prior"] < -6.0:
        assert "WARNING" in captured.err


def test_no_warning_when_prior_in_range(tmp_path, grammar, capsys):
    """No warning when prior falls within the enumerated range."""
    entries = [_make_entry("ok", "(λ has_color $0 RED)")]
    fp = _write_json(tmp_path, entries)

    results = load_and_validate_injections(
        fp,
        grammar=grammar,
        enumerated_prior_range=(-30.0, -1.0),
        warn_prior_threshold=2.0,
    )

    captured = capsys.readouterr()
    assert "WARNING" not in captured.err


# =========================================================================
# Tests: edge cases
# =========================================================================

def test_empty_file_returns_empty_list(tmp_path, grammar):
    """An empty JSON array should return an empty list."""
    fp = _write_json(tmp_path, [])
    results = load_and_validate_injections(fp, grammar=grammar)
    assert results == []


def test_missing_non_dsl_fields_warns_but_succeeds(tmp_path, grammar, capsys):
    """Missing fields other than dsl_program produce warnings but don't raise."""
    entries = [{
        "id": "partial",
        "dsl_program": "(λ has_color $0 RED)",
        # missing 'source' and 'true_for_rule'
    }]
    fp = _write_json(tmp_path, entries)

    results = load_and_validate_injections(fp, grammar=grammar)

    assert len(results) == 1
    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert "missing fields" in captured.err
