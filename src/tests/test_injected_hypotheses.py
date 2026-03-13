"""
Tests for injected LLM hypotheses (DSL translations).

Validates that:
1. Every dsl_program parses without error
2. Every hypothesis has a finite (non -inf) log-prior
3. We have a reasonable number of hypotheses
"""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.program import parse_program, Primitive
from gallery_analysis.enumerator import build_gallery_grammar
from gallery_analysis.dsl_prior import compute_log_prior


DATA_PATH = Path(__file__).parent.parent / "gallery_analysis" / "data" / "injected_hypotheses.json"


@pytest.fixture(scope="module")
def hypotheses():
    """Load the injected hypotheses JSON."""
    with open(DATA_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def grammar():
    """Build the gallery grammar (shared across tests)."""
    return build_gallery_grammar()


@pytest.fixture(scope="module")
def prim_dict(grammar):
    """Build the primitive lookup dictionary."""
    d = {}
    for prod in grammar.productions:
        if isinstance(prod.program, Primitive):
            d[prod.program.name] = prod.program
    return d


def test_count(hypotheses):
    """Check we have a reasonable number of hypotheses (100+)."""
    assert len(hypotheses) >= 100, (
        f"Expected at least 100 hypotheses, got {len(hypotheses)}"
    )


def test_all_injected_parse(hypotheses, prim_dict):
    """Every dsl_program must parse without error."""
    failures = []
    for entry in hypotheses:
        dsl = entry["dsl_program"]
        eid = entry["id"]
        try:
            parse_program(dsl, prim_dict)
        except Exception as e:
            failures.append(f"{eid}: {e}")
    assert not failures, (
        f"{len(failures)} parse failures:\n" + "\n".join(failures[:10])
    )


def test_all_injected_have_priors(hypotheses, grammar):
    """Every injected hypothesis must have a finite log-prior."""
    failures = []
    for entry in hypotheses:
        dsl = entry["dsl_program"]
        eid = entry["id"]
        try:
            lp = compute_log_prior(dsl, grammar)
            if not math.isfinite(lp):
                failures.append(f"{eid}: log_prior={lp}")
        except Exception as e:
            failures.append(f"{eid}: {e}")
    assert not failures, (
        f"{len(failures)} prior failures:\n" + "\n".join(failures[:10])
    )


def test_all_have_required_fields(hypotheses):
    """Every entry must have the required injection format fields."""
    required = {"id", "source", "true_for_rule", "dsl_program", "origin"}
    for entry in hypotheses:
        missing = required - set(entry.keys())
        assert not missing, f"{entry.get('id', '??')}: missing fields {missing}"


def test_unique_ids(hypotheses):
    """All hypothesis IDs must be unique."""
    ids = [e["id"] for e in hypotheses]
    assert len(ids) == len(set(ids)), "Duplicate IDs found"


def test_no_bare_large_constants(hypotheses):
    """No DSL program should contain bare integer constants > 5."""
    import re
    failures = []
    for entry in hypotheses:
        dsl = entry["dsl_program"]
        # Find tokens that look like bare integers (not preceded by $)
        tokens = re.findall(r'(?<!\$)\b(\d+)\b', dsl)
        for t in tokens:
            n = int(t)
            if n > 5:
                failures.append(f"{entry['id']}: bare constant {n}")
                break
    assert not failures, (
        f"{len(failures)} entries with bare constants > 5:\n"
        + "\n".join(failures[:10])
    )
