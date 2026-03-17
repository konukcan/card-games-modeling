"""Tests for Phase 1b hypothesis data loader.

TDD tests written first, then implementation follows.
"""

import json
import tempfile
from pathlib import Path

import pytest

from llm.grammar_comparison.data_loader import load_phase1b_hypotheses


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
            "dsl_program": "(λ all (λ eq (mod (rank_val $0) 2) 0) $0)",
            "origin": {},
        },
        # LLM hypothesis matching rank 1 of all_even
        {
            "id": "llm__all_even__hyp0",
            "source": "llm_foil",
            "dsl_program": "(λ all (λ eq (mod (rank_val $0) 2) 0) $0)",
            "origin": {
                "hypothesis_text": "All cards have an even rank.",
                "python_lambda": "rule = lambda hand: ...",
                "source_model": "gemini-2.5-flash",
                "original_rule_id": "all_even",
            },
        },
        # LLM hypothesis for a different rule (no Phase 1b match in fixture)
        {
            "id": "llm__all_red__hyp1",
            "source": "llm_foil",
            "dsl_program": "(λ all (λ eq (color $0) RED) $0)",
            "origin": {
                "hypothesis_text": "All cards are red.",
                "python_lambda": "rule = lambda hand: ...",
                "source_model": "gemini-2.5-flash",
                "original_rule_id": "all_red",
            },
        },
    ]
    path = tmp_path / "injected_hypotheses.json"
    path.write_text(json.dumps(data))
    return path


# ---------------------------------------------------------------------------
# Tests
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
        # all_odd comes from python-freeform => should be absent
        rule_ids = {r["rule_id"] for r in result}
        assert "all_odd" not in rule_ids
        assert "all_even" in rule_ids

    def test_filters_passed_only(self, phase1b_dir, injected_file):
        """By default only passed hypotheses are included."""
        result = load_phase1b_hypotheses(
            phase1b_dir=phase1b_dir, injected_path=injected_file
        )
        # all_even has 2 passed, 1 failed => expect 2
        assert len(result) == 2
        assert all(r["judge_verdict"] == "PASS" for r in result)

    def test_required_keys_present(self, phase1b_dir, injected_file):
        """Each dict must have all required keys."""
        required = {
            "rule_id", "rank", "confidence", "nl_description",
            "dsl_code", "python_code", "judge_verdict", "source_model",
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
        # 2 passed + 1 failed from dsl-constrained
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
        assert len(result) == 2  # passed_only from dsl-constrained
        assert all(r["dsl_code"] is None for r in result)


class TestLoadPhase1bRealData:
    """Smoke tests against the actual project data (skipped if files missing)."""

    # tests/ -> grammar_comparison/ -> llm/ -> llm/results/phase1b/
    REAL_PHASE1B = Path(__file__).resolve().parents[2] / "results" / "phase1b"
    # tests/ -> grammar_comparison/ -> llm/ -> card-games-modelling/
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
        # Should return a non-empty list
        assert len(result) > 0
        # Spot check: every entry has a rule_id and rank
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
        # We expect at least some matches
        assert len(dsl_hits) > 0, "No DSL codes matched — check cross-referencing logic"
