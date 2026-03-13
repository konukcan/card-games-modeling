"""
Tests for gallery_analysis.injection module.

Tests for load_and_validate_injections():
1. Valid injection files produce entries with log_prior and predicate
2. Missing dsl_program raises ValueError
3. Unparseable DSL programs raise ValueError
4. Predicates are callable and return bool
5. Log-priors are finite and negative
6. Outlier priors trigger warnings

Tests for merge_injected():
7. Novel injected hypotheses create new equivalence classes
8. Duplicate injected hypotheses merge into existing classes
9. Merge preserves unmatched existing classes
10. Summed prior is updated correctly on merge
"""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from gallery_analysis.injection import load_and_validate_injections, merge_injected
from gallery_analysis.enumerator import build_gallery_grammar
from gallery_analysis.hypothesis_table import compute_fingerprint
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


# =========================================================================
# Tests: merge_injected
# =========================================================================

# Probe hands used for fingerprinting in merge tests.
# We use a small fixed set so tests are deterministic.
_PROBE_HANDS = [
    # Hand 1: all hearts
    [Card(Rank.ACE, Suit.HEARTS), Card(Rank.TWO, Suit.HEARTS),
     Card(Rank.THREE, Suit.HEARTS), Card(Rank.FOUR, Suit.HEARTS),
     Card(Rank.FIVE, Suit.HEARTS), Card(Rank.SIX, Suit.HEARTS)],
    # Hand 2: mixed suits
    [Card(Rank.ACE, Suit.HEARTS), Card(Rank.TWO, Suit.SPADES),
     Card(Rank.THREE, Suit.DIAMONDS), Card(Rank.FOUR, Suit.CLUBS),
     Card(Rank.FIVE, Suit.HEARTS), Card(Rank.SIX, Suit.SPADES)],
    # Hand 3: all spades
    [Card(Rank.ACE, Suit.SPADES), Card(Rank.TWO, Suit.SPADES),
     Card(Rank.THREE, Suit.SPADES), Card(Rank.FOUR, Suit.SPADES),
     Card(Rank.FIVE, Suit.SPADES), Card(Rank.SIX, Suit.SPADES)],
]


def _make_equiv_class(predicate, program_str, log_prior, probes):
    """Helper to build a synthetic equivalence class dict."""
    fp = compute_fingerprint(predicate, probes)
    return {
        "canonical_program": program_str,
        "canonical_prior": log_prior,
        "summed_prior": log_prior,
        "n_expressions": 1,
        "all_programs": [program_str],
        "fingerprint": fp,
        "predicate": predicate,
    }


def _make_injected(id_str, predicate, log_prior, dsl_str="(injected)",
                   true_for_rule=None):
    """Helper to build a synthetic injected hypothesis dict."""
    return {
        "id": id_str,
        "source": "llm",
        "true_for_rule": true_for_rule,
        "dsl_program": dsl_str,
        "predicate": predicate,
        "log_prior": log_prior,
    }


def test_merge_novel_hypothesis():
    """An injected hypothesis with a novel fingerprint creates a new class."""
    # always-True predicate as existing class
    always_true = lambda hand: True
    ec = _make_equiv_class(always_true, "(always-true)", -2.0, _PROBE_HANDS)

    # always-False predicate as injected (different fingerprint)
    always_false = lambda hand: False
    inj = _make_injected("novel_1", always_false, -3.0)

    result = merge_injected([ec], [inj], _PROBE_HANDS)

    # Should now have 2 classes: original + novel
    assert len(result) == 2
    # The novel class should be at the end
    novel = result[1]
    assert novel["source"] == "injected"
    assert novel["injection_ids"] == ["novel_1"]
    assert novel["canonical_prior"] == -3.0
    assert novel["n_expressions"] == 1


def test_merge_duplicate_hypothesis():
    """An injected hypothesis matching an existing fingerprint merges."""
    # Both predicates are always-True -> same fingerprint
    always_true_1 = lambda hand: True
    always_true_2 = lambda hand: True
    ec = _make_equiv_class(always_true_1, "(always-true-1)", -2.0, _PROBE_HANDS)

    inj = _make_injected("dup_1", always_true_2, -3.0,
                         dsl_str="(always-true-2)", true_for_rule="rule_X")

    result = merge_injected([ec], [inj], _PROBE_HANDS)

    # Should still be 1 class (merged, not duplicated)
    assert len(result) == 1
    merged_class = result[0]
    assert merged_class["source"] == "merged"
    assert merged_class["n_expressions"] == 2
    assert "(always-true-2)" in merged_class["all_programs"]
    assert merged_class["injection_ids"] == ["dup_1"]
    assert merged_class["true_for_rule"] == "rule_X"
    assert merged_class["true_for_rules"] == ["rule_X"]


def test_merge_multiple_true_rules_same_fingerprint():
    """Multiple true rules merging into the same class are all tracked."""
    # All three predicates are always-True -> same fingerprint
    always_true = lambda hand: True
    ec = _make_equiv_class(always_true, "(always-true)", -2.0, _PROBE_HANDS)

    inj_a = _make_injected("true_A", lambda hand: True, -3.0,
                           dsl_str="(true-A)", true_for_rule="rule_A")
    inj_b = _make_injected("true_B", lambda hand: True, -4.0,
                           dsl_str="(true-B)", true_for_rule="rule_B")
    inj_c = _make_injected("true_C", lambda hand: True, -5.0,
                           dsl_str="(true-C)", true_for_rule="rule_C")

    result = merge_injected([ec], [inj_a, inj_b, inj_c], _PROBE_HANDS)

    assert len(result) == 1
    merged_class = result[0]
    # All three true rules should be tracked
    assert set(merged_class["true_for_rules"]) == {"rule_A", "rule_B", "rule_C"}
    assert len(merged_class["true_for_rules"]) == 3
    # Last-wins backward compat field
    assert merged_class["true_for_rule"] == "rule_C"


def test_merge_preserves_existing_classes():
    """Merge doesn't modify classes that don't match any injection."""
    always_true = lambda hand: True
    always_false = lambda hand: False
    ec_true = _make_equiv_class(always_true, "(true)", -2.0, _PROBE_HANDS)
    ec_false = _make_equiv_class(always_false, "(false)", -4.0, _PROBE_HANDS)

    # Inject something that only matches always_true
    inj = _make_injected("merge_1", lambda hand: True, -5.0)

    original_classes = [ec_true, ec_false]
    result = merge_injected(original_classes, [inj], _PROBE_HANDS)

    # Original list should be untouched (deep copy)
    assert len(original_classes) == 2
    assert original_classes[0]["n_expressions"] == 1  # not modified

    # The always_false class should be unchanged in result
    false_class = result[1]
    assert false_class["n_expressions"] == 1
    assert false_class["canonical_program"] == "(false)"
    assert "injection_ids" not in false_class
    assert "source" not in false_class  # original had no source field


def test_merge_updates_summed_prior():
    """When merging, summed_prior is updated correctly via log-sum-exp."""
    always_true = lambda hand: True
    ec = _make_equiv_class(always_true, "(true)", -2.0, _PROBE_HANDS)
    # Override summed_prior to a known value
    ec["summed_prior"] = -2.0

    inj = _make_injected("sp_1", lambda hand: True, -3.0)

    result = merge_injected([ec], [inj], _PROBE_HANDS)

    # Expected: log(exp(-2) + exp(-3))
    #         = max(-2,-3) + log(1 + exp(-1))
    #         = -2 + log(1 + 0.3679)
    #         ≈ -2 + 0.3133 = -1.6867
    expected = -2.0 + math.log(1 + math.exp(-1.0))
    assert math.isclose(result[0]["summed_prior"], expected, rel_tol=1e-9)
