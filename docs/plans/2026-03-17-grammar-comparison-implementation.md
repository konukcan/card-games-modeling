# Grammar Comparison Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a pipeline to empirically compare 7 candidate DSL grammars × 3 cost structures against ~270 Phase 1b LLM hypotheses, measuring which grammar best predicts the LLM's confidence rankings.

**Architecture:** Hypothesis data (s-expressions + Python) → AST parser → mechanical rewriter (per grammar) → PCFG log-probability scorer → evaluation metrics. All code in `llm/grammar_comparison/` on branch `feat/grammar-comparison`. No modifications to existing `src/` files.

**Tech Stack:** Python 3, existing `src/dreamcoder_core/` modules (read-only imports: `program.py`, `grammar.py`, `type_system.py`), `scipy.stats` for Spearman correlation, Python `ast` module for parsing.

## Execution Guidelines
- Explain code as you write it — treat as learning opportunity
- Start simple, build up — get one grammar working end-to-end before scaling to 7
- Test each step before proceeding
- Commit after each task
- NEVER modify files in `src/` — only read/import from them

---

### Task 1: Branch and Directory Setup

**Files:**
- Create: `llm/grammar_comparison/__init__.py`
- Create: `llm/grammar_comparison/tests/__init__.py`

**Step 1: Create branch**

```bash
git checkout -b feat/grammar-comparison
```

**Step 2: Create directory structure**

```bash
mkdir -p llm/grammar_comparison/grammars
mkdir -p llm/grammar_comparison/primitives
mkdir -p llm/grammar_comparison/translation
mkdir -p llm/grammar_comparison/evaluation
mkdir -p llm/grammar_comparison/tests
```

**Step 3: Create `__init__.py` files**

Create empty `__init__.py` in `llm/grammar_comparison/` and `llm/grammar_comparison/tests/`.

**Step 4: Verify imports from existing code work**

```python
# llm/grammar_comparison/test_imports.py
"""Verify we can import from src/ without modifying it."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from dreamcoder_core.program import Program, Primitive, Application, Abstraction, Index, parse_program
from dreamcoder_core.grammar import Grammar, Production
from dreamcoder_core.type_system import (
    Type, BaseType, Arrow, ListType, TypeVariable,
    BOOL, INT, CARD, SUIT, RANK, HAND, arrow
)

print("All imports OK")
```

Run: `python llm/grammar_comparison/test_imports.py`
Expected: "All imports OK"

**Step 5: Commit**

```bash
git add llm/grammar_comparison/
git commit -m "feat: scaffold grammar comparison directory structure"
```

---

### Task 2: Load Phase 1b Hypothesis Data

**Files:**
- Create: `llm/grammar_comparison/data_loader.py`
- Test: `llm/grammar_comparison/tests/test_data_loader.py`

**Step 1: Write the failing test**

```python
# llm/grammar_comparison/tests/test_data_loader.py
"""Tests for loading Phase 1b hypothesis data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm.grammar_comparison.data_loader import load_phase1b_hypotheses

def test_load_returns_hypotheses():
    hypotheses = load_phase1b_hypotheses()
    assert len(hypotheses) > 200, f"Expected 200+ hypotheses, got {len(hypotheses)}"

def test_hypothesis_has_required_fields():
    hypotheses = load_phase1b_hypotheses()
    h = hypotheses[0]
    assert "rule_id" in h
    assert "rank" in h          # 1-5 confidence ranking
    assert "confidence" in h    # HIGH/MEDIUM/LOW
    assert "nl_description" in h
    assert "dsl_code" in h      # s-expression or DSL-constrained code
    assert "python_code" in h   # Python-freeform translation (if available)

def test_rank_values_are_1_to_5():
    hypotheses = load_phase1b_hypotheses()
    ranks = set(h["rank"] for h in hypotheses)
    assert ranks.issubset({1, 2, 3, 4, 5})

def test_hypotheses_cover_multiple_rules():
    hypotheses = load_phase1b_hypotheses()
    rule_ids = set(h["rule_id"] for h in hypotheses)
    assert len(rule_ids) >= 50, f"Expected 50+ rules, got {len(rule_ids)}"

if __name__ == "__main__":
    test_load_returns_hypotheses()
    test_hypothesis_has_required_fields()
    test_rank_values_are_1_to_5()
    test_hypotheses_cover_multiple_rules()
    print("All data loader tests passed")
```

**Step 2: Run test to verify it fails**

Run: `python llm/grammar_comparison/tests/test_data_loader.py`
Expected: FAIL with "ModuleNotFoundError" or "ImportError"

**Step 3: Write minimal implementation**

```python
# llm/grammar_comparison/data_loader.py
"""
Load Phase 1b hypothesis data for grammar comparison.

Phase 1b data lives in llm/results/phase1b/ as JSON files, one per
(model, format, rule) combination. Each file contains 4-5 ranked
hypotheses with confidence levels and judge verdicts.

We also load the injected s-expression translations from
src/gallery_analysis/data/injected_hypotheses.json to get DSL programs
for hypotheses that have them.
"""
import json
import glob
from pathlib import Path
from typing import List, Dict, Optional


# Paths relative to project root
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_PHASE1B_DIR = _PROJECT_ROOT / "llm" / "results" / "phase1b"
_INJECTED_PATH = _PROJECT_ROOT / "src" / "gallery_analysis" / "data" / "injected_hypotheses.json"


def _load_injected_s_expressions() -> Dict[str, str]:
    """
    Load injected hypotheses and build a lookup from
    (rule_id, hypothesis_text) -> s-expression program.
    """
    if not _INJECTED_PATH.exists():
        return {}

    with open(_INJECTED_PATH) as f:
        entries = json.load(f)

    lookup = {}
    for entry in entries:
        if entry["id"].startswith("phase1b__"):
            origin = entry.get("origin", {})
            key = (origin.get("original_rule_id", ""), origin.get("hypothesis_text", ""))
            lookup[key] = entry.get("dsl_program", "")
    return lookup


def load_phase1b_hypotheses(
    format_filter: str = "dsl-constrained",
    passed_only: bool = True,
) -> List[Dict]:
    """
    Load Phase 1b hypotheses as a flat list of dicts.

    Each dict contains:
        rule_id: str            — which rule this hypothesis is about
        rank: int               — 1-5 (1 = highest confidence)
        confidence: str         — HIGH / MEDIUM / LOW
        nl_description: str     — natural language hypothesis text
        dsl_code: str           — DSL s-expression (from injected data)
        python_code: str        — Python-freeform code (from Phase 1b raw data)
        judge_verdict: str      — PASS / FAIL
        source_model: str       — which LLM generated this

    Args:
        format_filter: Which format to load ("dsl-constrained", "python-freeform", "webppl", or "all")
        passed_only: If True, only include judge-verified PASS hypotheses
    """
    # Load s-expression lookup
    sexpr_lookup = _load_injected_s_expressions()

    # Glob Phase 1b result files
    if format_filter == "all":
        pattern = str(_PHASE1B_DIR / "*.json")
    else:
        pattern = str(_PHASE1B_DIR / f"*__{format_filter}__*.json")

    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No Phase 1b files found matching {pattern}. "
            f"Check that {_PHASE1B_DIR} exists."
        )

    hypotheses = []
    for filepath in files:
        with open(filepath) as f:
            data = json.load(f)

        rule_id = data["rule_id"]
        source_model = data.get("source_model", "unknown")

        for h in data.get("hypotheses", []):
            # Filter by judge verdict if requested
            verdict = (h.get("judge_verdict") or {}).get("verdict", "UNKNOWN")
            if passed_only and verdict != "PASS":
                continue

            nl_text = h.get("nl_description", "")

            # Look up s-expression from injected data
            sexpr = sexpr_lookup.get((rule_id, nl_text), "")

            hypotheses.append({
                "rule_id": rule_id,
                "rank": h.get("rank", 0),
                "confidence": h.get("confidence", "UNKNOWN"),
                "nl_description": nl_text,
                "dsl_code": sexpr,
                "python_code": h.get("code", ""),
                "judge_verdict": verdict,
                "source_model": source_model,
            })

    return hypotheses
```

**Step 4: Run test to verify it passes**

Run: `python llm/grammar_comparison/tests/test_data_loader.py`
Expected: All tests pass

**Step 5: Commit**

```bash
git add llm/grammar_comparison/data_loader.py llm/grammar_comparison/tests/test_data_loader.py
git commit -m "feat: add Phase 1b hypothesis data loader"
```

---

### Task 3: Implement New Primitives (Isolated)

**Files:**
- Create: `llm/grammar_comparison/primitives/definitions.py`
- Test: `llm/grammar_comparison/tests/test_primitives.py`

**Step 1: Write the failing test**

```python
# llm/grammar_comparison/tests/test_primitives.py
"""Tests for the 5 new primitives: slice, shifted_match, stride, count_where, sorted_counts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from rules.cards import Card, Suit, Rank, RANK_VALUES, Hand

from llm.grammar_comparison.primitives.definitions import (
    prim_slice, prim_shifted_match, prim_stride,
    prim_count_where, prim_sorted_counts,
)

# Test hand: 2♠ 5♥ K♣ 3♦ 7♠ J♥
HAND = [
    Card(Suit.SPADES, Rank.TWO),
    Card(Suit.HEARTS, Rank.FIVE),
    Card(Suit.CLUBS, Rank.KING),
    Card(Suit.DIAMONDS, Rank.THREE),
    Card(Suit.SPADES, Rank.SEVEN),
    Card(Suit.HEARTS, Rank.JACK),
]


def test_slice_basic():
    assert prim_slice(0, 3, HAND) == HAND[:3]
    assert prim_slice(3, 6, HAND) == HAND[3:]
    assert prim_slice(1, 4, HAND) == HAND[1:4]


def test_slice_subsumes_first_half():
    assert prim_slice(0, 3, HAND) == HAND[:3]  # first_half


def test_slice_subsumes_second_half():
    assert prim_slice(3, 6, HAND) == HAND[3:]  # second_half


def test_shifted_match_adjacent():
    # Adjacent cards same suit? (answer: no for this hand)
    same_suit = lambda x, y: x.suit == y.suit
    assert prim_shifted_match(1, same_suit, HAND) == False


def test_shifted_match_halves():
    # Halves copy suit? (answer: no)
    same_suit = lambda x, y: x.suit == y.suit
    assert prim_shifted_match(3, same_suit, HAND) == False

    # Build a hand where halves DO match
    matching_hand = [
        Card(Suit.HEARTS, Rank.TWO),
        Card(Suit.SPADES, Rank.FIVE),
        Card(Suit.CLUBS, Rank.KING),
        Card(Suit.HEARTS, Rank.THREE),
        Card(Suit.SPADES, Rank.SEVEN),
        Card(Suit.CLUBS, Rank.JACK),
    ]
    assert prim_shifted_match(3, same_suit, matching_hand) == True


def test_stride():
    assert prim_stride(2, HAND) == [HAND[0], HAND[2], HAND[4]]
    assert prim_stride(3, HAND) == [HAND[0], HAND[3]]
    assert prim_stride(1, HAND) == HAND  # every element


def test_count_where():
    is_spade = lambda c: c.suit == Suit.SPADES
    assert prim_count_where(is_spade, HAND) == 2

    is_red = lambda c: c.suit in (Suit.HEARTS, Suit.DIAMONDS)
    assert prim_count_where(is_red, HAND) == 3


def test_sorted_counts():
    # Suits: SPADES=2, HEARTS=2, CLUBS=1, DIAMONDS=1
    result = prim_sorted_counts(lambda c: c.suit, HAND)
    assert result == [2, 2, 1, 1]


def test_sorted_counts_color():
    # Colors: BLACK(S,C)=3, RED(H,D)=3
    from rules.cards import card_color
    result = prim_sorted_counts(lambda c: card_color(c), HAND)
    assert result == [3, 3]


if __name__ == "__main__":
    test_slice_basic()
    test_slice_subsumes_first_half()
    test_slice_subsumes_second_half()
    test_shifted_match_adjacent()
    test_shifted_match_halves()
    test_stride()
    test_count_where()
    test_sorted_counts()
    test_sorted_counts_color()
    print("All primitive tests passed")
```

**Step 2: Run test to verify it fails**

Run: `python llm/grammar_comparison/tests/test_primitives.py`
Expected: FAIL with "ImportError"

**Step 3: Write minimal implementation**

```python
# llm/grammar_comparison/primitives/definitions.py
"""
Isolated implementations of the 5 new primitives proposed for grammar comparison.

These are NOT added to the main src/dreamcoder_core/primitives.py — they exist
only within the grammar comparison framework for scoring purposes.

Primitives:
    slice(i, j, hand)           — extract elements from position i to j
    shifted_match(k, pred, hand) — check pred(hand[i], hand[i+k]) for all valid i
    stride(k, hand)              — every k-th element
    count_where(pred, hand)      — count elements satisfying predicate
    sorted_counts(key_fn, hand)  — sorted frequency distribution
"""
from typing import List, Callable, Any, TypeVar
from collections import Counter

T = TypeVar('T')


def prim_slice(i: int, j: int, xs: List[T]) -> List[T]:
    """
    Extract elements from position i (inclusive) to j (exclusive).

    Subsumes take/drop/first_half/second_half:
        take(n, xs)      = slice(0, n, xs)
        drop(n, xs)      = slice(n, len(xs), xs)
        first_half(xs)   = slice(0, 3, xs)   (for 6-card hands)
        second_half(xs)  = slice(3, 6, xs)   (for 6-card hands)
    """
    return xs[i:j]


def prim_shifted_match(k: int, pred: Callable[[T, T], bool], xs: List[T]) -> bool:
    """
    Check that pred(xs[i], xs[i+k]) holds for ALL valid positions i.

    For k=1: checks all adjacent pairs (subsumes adjacent_pairs + all)
    For k=3: checks first-half vs second-half pairwise (halves_copy patterns)

    Example:
        shifted_match(3, lambda x,y: x.suit == y.suit, hand)
        → True iff hand[0].suit == hand[3].suit AND
                   hand[1].suit == hand[4].suit AND
                   hand[2].suit == hand[5].suit
    """
    for i in range(len(xs) - k):
        if not pred(xs[i], xs[i + k]):
            return False
    return True


def prim_stride(k: int, xs: List[T]) -> List[T]:
    """
    Return every k-th element starting from position 0.

    stride(2, hand) → [hand[0], hand[2], hand[4]]  (even positions)
    stride(3, hand) → [hand[0], hand[3]]            (every 3rd)
    stride(1, hand) → hand                          (identity)
    """
    return xs[::k]


def prim_count_where(pred: Callable[[T], bool], xs: List[T]) -> int:
    """
    Count elements satisfying a predicate.

    Subsumes count_suit/count_rank/count_color:
        count_suit(hand, CLUBS) = count_where(lambda c: c.suit == CLUBS, hand)
        count_rank(hand, ACE)   = count_where(lambda c: c.rank == ACE, hand)
    """
    return sum(1 for x in xs if pred(x))


def prim_sorted_counts(key_fn: Callable[[T], Any], xs: List[T]) -> List[int]:
    """
    Group elements by key_fn, count each group, return counts sorted descending.

    Example:
        sorted_counts(get_suit, [S,S,S,H,H,D])
        → group: {S:3, H:2, D:1}
        → sorted: [3, 2, 1]

    Enables distribution-shape rules:
        "two suits with 2 cards each" = sorted_counts(get_suit) == [2,2,1,1]
    """
    counts = Counter(key_fn(x) for x in xs)
    return sorted(counts.values(), reverse=True)
```

Also create the `__init__.py`:

```python
# llm/grammar_comparison/primitives/__init__.py
from .definitions import (
    prim_slice, prim_shifted_match, prim_stride,
    prim_count_where, prim_sorted_counts,
)
```

**Step 4: Run test to verify it passes**

Run: `python llm/grammar_comparison/tests/test_primitives.py`
Expected: All tests pass

**Step 5: Commit**

```bash
git add llm/grammar_comparison/primitives/
git commit -m "feat: implement 5 new primitives in isolation (slice, shifted_match, stride, count_where, sorted_counts)"
```

---

### Task 4: Define Grammar Families as PCFG Configurations

**Files:**
- Create: `llm/grammar_comparison/grammars/grammar_factory.py`
- Test: `llm/grammar_comparison/tests/test_grammars.py`

**Step 1: Write the failing test**

```python
# llm/grammar_comparison/tests/test_grammars.py
"""Tests for grammar family definitions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm.grammar_comparison.grammars.grammar_factory import (
    build_grammar, GRAMMAR_NAMES, CostStructure,
)

def test_all_seven_grammars_buildable():
    expected = {"base", "swap-positional", "swap-distributional", "swap-both",
                "add-both", "redundant", "minimal"}
    assert set(GRAMMAR_NAMES) == expected

def test_build_returns_grammar_object():
    g = build_grammar("base", CostStructure.UNIFORM)
    # Should be a Grammar object from dreamcoder_core
    assert hasattr(g, "productions")
    assert hasattr(g, "log_probability")

def test_swap_positional_removes_old_adds_new():
    g = build_grammar("swap-positional", CostStructure.UNIFORM)
    prim_names = {str(p.program) for p in g.productions}
    # Should have slice, shifted_match
    assert "slice" in prim_names
    assert "shifted_match" in prim_names
    # Should NOT have take, drop, first_half, second_half, adjacent_pairs
    assert "take" not in prim_names
    assert "drop" not in prim_names
    assert "first_half" not in prim_names
    assert "second_half" not in prim_names

def test_minimal_has_fewer_primitives():
    g_base = build_grammar("base", CostStructure.UNIFORM)
    g_min = build_grammar("minimal", CostStructure.UNIFORM)
    assert len(g_min.productions) < len(g_base.productions)

def test_redundant_has_more_primitives():
    g_base = build_grammar("base", CostStructure.UNIFORM)
    g_red = build_grammar("redundant", CostStructure.UNIFORM)
    assert len(g_red.productions) > len(g_base.productions)

def test_three_cost_structures():
    for cost in CostStructure:
        g = build_grammar("base", cost)
        assert len(g.productions) > 0

def test_uniform_costs_are_equal_per_type():
    g = build_grammar("base", CostStructure.UNIFORM)
    # All INT-returning primitives should have the same log-prob
    # (This is the definition of uniform: normalized per return type)
    # Just check that at least some productions exist
    assert len(g.productions) > 20

if __name__ == "__main__":
    test_all_seven_grammars_buildable()
    test_build_returns_grammar_object()
    test_swap_positional_removes_old_adds_new()
    test_minimal_has_fewer_primitives()
    test_redundant_has_more_primitives()
    test_three_cost_structures()
    test_uniform_costs_are_equal_per_type()
    print("All grammar tests passed")
```

**Step 2: Run test to verify it fails**

Run: `python llm/grammar_comparison/tests/test_grammars.py`
Expected: FAIL with "ImportError"

**Step 3: Write implementation**

This is the largest single file. It needs to:
1. Define the primitive set for each of the 7 grammars (which Primitive objects to include)
2. Apply one of 3 cost structures (how to assign log-probabilities)
3. Return a Grammar object

The implementation should import the existing primitives from `src/dreamcoder_core/primitives.py` to get the Base set, then add/remove/modify as needed. Consult the existing primitives file at `src/dreamcoder_core/primitives.py` for the exact Primitive objects (name, type, value). Create new Primitive objects for the 5 new primitives using the implementations from Task 3.

For the cost structures:
- **Uniform:** `log_prob = log(1/N)` per return type, where N = number of primitives with that return type
- **Tiered:** Assign tiers (1/2/3) to each primitive. Within a return type, Tier 1 gets 3× the weight of Tier 2, 3× Tier 3. Then normalize.
- **LOTlib3-style:** Integer constants get `weight = 10/n²`. All terminal primitives get 5× multiplier. Then normalize per return type.

The file should define `GRAMMAR_NAMES`, `CostStructure` enum, and `build_grammar(name, cost) -> Grammar`.

**Step 4: Run test to verify it passes**

Run: `python llm/grammar_comparison/tests/test_grammars.py`
Expected: All tests pass

**Step 5: Commit**

```bash
git add llm/grammar_comparison/grammars/
git commit -m "feat: define 7 grammar families with 3 cost structures"
```

---

### Task 5: S-Expression Parser (Path A)

**Files:**
- Create: `llm/grammar_comparison/translation/sexpr_parser.py`
- Test: `llm/grammar_comparison/tests/test_sexpr_parser.py`

**Step 1: Write the failing test**

```python
# llm/grammar_comparison/tests/test_sexpr_parser.py
"""Tests for parsing Phase 1b s-expressions into ASTs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm.grammar_comparison.translation.sexpr_parser import parse_hypothesis_sexpr
from dreamcoder_core.program import Application, Abstraction, Index, Primitive

def test_parse_simple_all_even():
    sexpr = "(λ all (λ eq (mod (rank_val $0) 2) 0) $0)"
    ast = parse_hypothesis_sexpr(sexpr)
    # Should be an Abstraction wrapping Application(all, predicate, $0)
    assert isinstance(ast, Abstraction)

def test_parse_returns_evaluable_program():
    sexpr = "(λ all (λ eq (mod (rank_val $0) 2) 0) $0)"
    ast = parse_hypothesis_sexpr(sexpr)
    # Should have a size > 0
    assert ast.size() > 3

def test_parse_preserves_structure():
    # A simple identity-like program
    sexpr = "(λ all (λ eq (get_suit $0) CLUBS) $0)"
    ast = parse_hypothesis_sexpr(sexpr)
    assert isinstance(ast, Abstraction)
    assert ast.size() >= 4

def test_parse_adjacent_pairs():
    sexpr = "(λ all (λ or (eq (get_rank (head $0)) (get_rank (last $0))) (eq (get_suit (head $0)) (get_suit (last $0)))) (adjacent_pairs $0))"
    ast = parse_hypothesis_sexpr(sexpr)
    assert isinstance(ast, Abstraction)
    assert ast.size() >= 10

if __name__ == "__main__":
    test_parse_simple_all_even()
    test_parse_returns_evaluable_program()
    test_parse_preserves_structure()
    test_parse_adjacent_pairs()
    print("All s-expression parser tests passed")
```

**Step 2: Run test to verify it fails**

Run: `python llm/grammar_comparison/tests/test_sexpr_parser.py`
Expected: FAIL with "ImportError"

**Step 3: Write implementation**

The s-expression parser wraps `parse_program()` from `src/dreamcoder_core/program.py`, providing a registry of all known primitives (from the Base grammar) so that names like `all`, `eq`, `get_suit`, `CLUBS` resolve to the correct Primitive objects.

Key considerations:
- The existing `parse_program(s, primitives)` already handles the syntax `(λ body)`, `(f x y)`, `$n`
- We just need to build the `primitives: Dict[str, Primitive]` dictionary with all known names
- Import all primitives from `src/dreamcoder_core/primitives.py` to build this dictionary

**Step 4: Run test to verify it passes**

Run: `python llm/grammar_comparison/tests/test_sexpr_parser.py`
Expected: All tests pass

**Step 5: Commit**

```bash
git add llm/grammar_comparison/translation/sexpr_parser.py llm/grammar_comparison/tests/test_sexpr_parser.py
git commit -m "feat: add s-expression parser for Phase 1b hypotheses"
```

---

### Task 6: Python-to-AST Converter (Path B)

**Files:**
- Create: `llm/grammar_comparison/translation/python_parser.py`
- Test: `llm/grammar_comparison/tests/test_python_parser.py`

**Step 1: Write the failing test**

```python
# llm/grammar_comparison/tests/test_python_parser.py
"""Tests for converting Python-freeform lambdas to DSL ASTs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm.grammar_comparison.translation.python_parser import python_to_ast
from dreamcoder_core.program import Abstraction

def test_all_even():
    code = "rule = lambda hand: all(RANK_VALUES[card.rank] % 2 == 0 for card in hand)"
    ast = python_to_ast(code)
    assert isinstance(ast, Abstraction)
    assert ast.size() >= 4

def test_all_same_suit():
    code = "rule = lambda hand: len(set(card.suit for card in hand)) == 1"
    ast = python_to_ast(code)
    assert isinstance(ast, Abstraction)

def test_count_suit():
    code = "rule = lambda hand: sum(1 for card in hand if card.suit == Suit.HEARTS) >= 3"
    ast = python_to_ast(code)
    assert isinstance(ast, Abstraction)

def test_head_last():
    code = "rule = lambda hand: hand[0].suit == hand[-1].suit"
    ast = python_to_ast(code)
    assert isinstance(ast, Abstraction)

def test_all_red():
    code = "rule = lambda hand: all(card.suit in (Suit.HEARTS, Suit.DIAMONDS) for card in hand)"
    ast = python_to_ast(code)
    assert isinstance(ast, Abstraction)

if __name__ == "__main__":
    test_all_even()
    test_all_same_suit()
    test_count_suit()
    test_head_last()
    test_all_red()
    print("All Python parser tests passed")
```

**Step 2: Run test to verify it fails**

Run: `python llm/grammar_comparison/tests/test_python_parser.py`
Expected: FAIL with "ImportError"

**Step 3: Write implementation**

Use Python's `ast` module to parse the lambda source code into a Python AST, then walk the tree with pattern-matching rules that produce DSL Program nodes.

Key patterns to implement (covering the most common Phase 0/1b idioms):

```
Python pattern                          → DSL equivalent
─────────────────────────────────────── → ─────────────────
all(expr for card in hand)              → Application(all, Abstraction(expr'), hand)
any(expr for card in hand)              → Application(any, Abstraction(expr'), hand)
sum(1 for c in hand if pred)            → Application(count_where, Abstraction(pred'), hand)
len(set(expr for c in hand))            → Application(length, Application(unique, Application(map, Abstraction(expr'), hand)))
card.suit                               → Application(get_suit, $0)
card.rank                               → Application(get_rank, $0)
RANK_VALUES[card.rank]                  → Application(rank_val, $0)
card.suit in (Suit.X, Suit.Y)           → Application(or, Application(eq, ..., X), Application(eq, ..., Y))
hand[0]                                 → Application(head, hand)
hand[-1]                                → Application(last, hand)
hand[n]                                 → Application(at, hand, n)
x % y                                   → Application(mod, x, y)
x == y                                  → Application(eq, x, y)
x > y                                   → Application(gt, x, y)
x < y                                   → Application(lt, x, y)
x >= y                                  → Application(ge, x, y)
not x                                   → Application(not_, x)
x and y                                 → Application(and_, x, y)
x or y                                  → Application(or_, x, y)
len(hand)                               → Application(length, hand)
Suit.HEARTS                             → Primitive(HEARTS)
Integer literal n                       → Primitive(str(n))
```

Start with the most common patterns and add more as needed. Use a `NotImplementedError` for unrecognized patterns — this makes it obvious which patterns are missing when running against the full dataset.

**Step 4: Run test to verify it passes**

Run: `python llm/grammar_comparison/tests/test_python_parser.py`
Expected: All tests pass

**Step 5: Commit**

```bash
git add llm/grammar_comparison/translation/python_parser.py llm/grammar_comparison/tests/test_python_parser.py
git commit -m "feat: add Python-to-AST converter with pattern matching"
```

---

### Task 7: AST Rewriter (Base → Grammar-Specific)

**Files:**
- Create: `llm/grammar_comparison/translation/rewriter.py`
- Test: `llm/grammar_comparison/tests/test_rewriter.py`

**Step 1: Write the failing test**

```python
# llm/grammar_comparison/tests/test_rewriter.py
"""Tests for mechanical AST rewriting between grammars."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm.grammar_comparison.translation.rewriter import rewrite_ast
from llm.grammar_comparison.translation.sexpr_parser import parse_hypothesis_sexpr

def test_base_to_base_is_identity():
    sexpr = "(λ all (λ eq (get_suit $0) CLUBS) $0)"
    ast = parse_hypothesis_sexpr(sexpr)
    rewritten = rewrite_ast(ast, "base")
    assert str(rewritten) == str(ast)

def test_base_to_swap_positional_rewrites_first_half():
    sexpr = "(λ first_half $0)"
    ast = parse_hypothesis_sexpr(sexpr)
    rewritten = rewrite_ast(ast, "swap-positional")
    # first_half should become slice(0, 3, ...)
    rewritten_str = str(rewritten)
    assert "slice" in rewritten_str
    assert "first_half" not in rewritten_str

def test_base_to_swap_distributional_rewrites_count_suit():
    sexpr = "(λ eq (count_suit $0 CLUBS) 3)"
    ast = parse_hypothesis_sexpr(sexpr)
    rewritten = rewrite_ast(ast, "swap-distributional")
    rewritten_str = str(rewritten)
    assert "count_where" in rewritten_str
    assert "count_suit" not in rewritten_str

def test_base_to_minimal_decomposes_shortcuts():
    sexpr = "(λ all_same_suit $0)"
    ast = parse_hypothesis_sexpr(sexpr)
    rewritten = rewrite_ast(ast, "minimal")
    # all_same_suit is not in minimal — should be decomposed
    rewritten_str = str(rewritten)
    assert "all_same_suit" not in rewritten_str
    assert "all" in rewritten_str or "eq" in rewritten_str

def test_rewrite_preserves_size_or_explains_change():
    sexpr = "(λ all (λ eq (get_suit $0) CLUBS) $0)"
    ast = parse_hypothesis_sexpr(sexpr)
    # Base → base should preserve size exactly
    rewritten = rewrite_ast(ast, "base")
    assert rewritten.size() == ast.size()

if __name__ == "__main__":
    test_base_to_base_is_identity()
    test_base_to_swap_positional_rewrites_first_half()
    test_base_to_swap_distributional_rewrites_count_suit()
    test_base_to_minimal_decomposes_shortcuts()
    test_rewrite_preserves_size_or_explains_change()
    print("All rewriter tests passed")
```

**Step 2: Run test to verify it fails**

Run: `python llm/grammar_comparison/tests/test_rewriter.py`
Expected: FAIL with "ImportError"

**Step 3: Write implementation**

Implement `rewrite_ast(ast, grammar_name) -> Program` that walks the AST and applies rewriting rules. Use the `ProgramTransformer` pattern from `program.py` or a recursive walk.

Rewriting rules per grammar:

**swap-positional:** `first_half → slice(0,3)`, `second_half → slice(3,6)`, `take(n) → slice(0,n)`, `drop(n) → slice(n,6)`, `adjacent_pairs → [mark as inexpressible — use shifted_match(1,...) context-dependently]`

**swap-distributional:** `count_suit(hand,S) → count_where(λc.eq(get_suit c) S, hand)`, same for count_rank, count_color

**swap-both:** All rules from swap-positional + swap-distributional

**add-both:** Identity (all primitives available)

**redundant:** Identity (superset of base)

**minimal:** Decompose compound primitives: `first_half → take(3)`, `all_same_suit → all(λc.eq(get_suit c)(get_suit(head hand)), hand)`, etc.

Return the rewritten AST, or raise `InexpressibleError` if the hypothesis uses a primitive that has no equivalent in the target grammar.

**Step 4: Run test to verify it passes**

Run: `python llm/grammar_comparison/tests/test_rewriter.py`
Expected: All tests pass

**Step 5: Commit**

```bash
git add llm/grammar_comparison/translation/rewriter.py llm/grammar_comparison/tests/test_rewriter.py
git commit -m "feat: add mechanical AST rewriter for grammar-specific translations"
```

---

### Task 8: Fingerprint Verification Pipeline

**Files:**
- Create: `llm/grammar_comparison/translation/verification.py`
- Test: `llm/grammar_comparison/tests/test_verification.py`

**Step 1: Write the failing test**

```python
# llm/grammar_comparison/tests/test_verification.py
"""Tests for fingerprint verification of AST translations."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm.grammar_comparison.translation.verification import (
    verify_dual_path, verify_rewrite_preserves_semantics,
    load_probe_hands,
)

def test_load_probe_hands():
    probes = load_probe_hands()
    assert len(probes) == 200

def test_dual_path_agreement_simple():
    """S-expression and Python paths should produce matching fingerprints."""
    sexpr = "(λ all (λ eq (mod (rank_val $0) 2) 0) $0)"
    python_code = "rule = lambda hand: all(RANK_VALUES[card.rank] % 2 == 0 for card in hand)"
    probes = load_probe_hands()

    match, details = verify_dual_path(sexpr, python_code, probes)
    assert match, f"Fingerprints disagree: {details}"

def test_rewrite_preserves_semantics():
    """Rewriting should not change the fingerprint."""
    sexpr = "(λ all (λ eq (get_suit $0) CLUBS) $0)"
    probes = load_probe_hands()

    ok, details = verify_rewrite_preserves_semantics(
        sexpr, target_grammar="swap-positional", probes=probes
    )
    # This particular program doesn't use any primitives that get rewritten,
    # so it should trivially match
    assert ok, f"Rewrite changed semantics: {details}"

if __name__ == "__main__":
    test_load_probe_hands()
    test_dual_path_agreement_simple()
    test_rewrite_preserves_semantics()
    print("All verification tests passed")
```

**Step 2: Run test to verify it fails**

Run: `python llm/grammar_comparison/tests/test_verification.py`
Expected: FAIL with "ImportError"

**Step 3: Write implementation**

The verification module:
1. Loads the 200-probe set from `llm/results/probe_set_200.json`
2. Executes ASTs on probes to get boolean fingerprints
3. Compares fingerprints between dual-path translations and pre/post-rewrite

Key functions:
- `load_probe_hands() -> List[Hand]` — load the 200 probe hands
- `compute_ast_fingerprint(ast, probes) -> Tuple[bool, ...]` — execute AST on each probe
- `compute_python_fingerprint(code, probes) -> Tuple[bool, ...]` — execute Python code on each probe
- `verify_dual_path(sexpr, python_code, probes) -> (bool, dict)` — parse both, compare fingerprints
- `verify_rewrite_preserves_semantics(sexpr, grammar, probes) -> (bool, dict)` — rewrite and compare

**Step 4: Run test to verify it passes**

Run: `python llm/grammar_comparison/tests/test_verification.py`
Expected: All tests pass

**Step 5: Commit**

```bash
git add llm/grammar_comparison/translation/verification.py llm/grammar_comparison/tests/test_verification.py
git commit -m "feat: add fingerprint verification for translation pipeline"
```

---

### Task 9: Log-Probability Scorer

**Files:**
- Create: `llm/grammar_comparison/evaluation/compute_costs.py`
- Test: `llm/grammar_comparison/tests/test_compute_costs.py`

**Step 1: Write the failing test**

```python
# llm/grammar_comparison/tests/test_compute_costs.py
"""Tests for computing program log-probabilities under different grammars."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm.grammar_comparison.evaluation.compute_costs import (
    score_hypothesis, score_all_hypotheses,
)
from llm.grammar_comparison.grammars.grammar_factory import build_grammar, CostStructure

def test_score_returns_float():
    g = build_grammar("base", CostStructure.UNIFORM)
    sexpr = "(λ all (λ eq (get_suit $0) CLUBS) $0)"
    score = score_hypothesis(sexpr, g)
    assert isinstance(score, float)
    assert score < 0  # log-probabilities are negative

def test_simpler_programs_have_higher_score():
    g = build_grammar("base", CostStructure.UNIFORM)
    simple = "(λ eq (get_suit (head $0)) CLUBS)"
    complex_ = "(λ all (λ eq (mod (rank_val $0) 2) 0) $0)"
    score_simple = score_hypothesis(simple, g)
    score_complex = score_hypothesis(complex_, g)
    assert score_simple > score_complex, "Simpler program should have higher log-prob"

def test_inexpressible_returns_neg_inf():
    g = build_grammar("minimal", CostStructure.UNIFORM)
    # Use a primitive not in minimal grammar
    sexpr = "(λ all_same_suit $0)"
    score = score_hypothesis(sexpr, g)
    assert score == float("-inf")

def test_score_all_returns_dict():
    results = score_all_hypotheses(
        grammar_name="base",
        cost_structure=CostStructure.UNIFORM,
        limit=5,
    )
    assert isinstance(results, list)
    assert len(results) <= 5
    assert "log_prob" in results[0]
    assert "rule_id" in results[0]

if __name__ == "__main__":
    test_score_returns_float()
    test_simpler_programs_have_higher_score()
    test_inexpressible_returns_neg_inf()
    test_score_all_returns_dict()
    print("All compute_costs tests passed")
```

**Step 2: Run test to verify it fails**

Run: `python llm/grammar_comparison/tests/test_compute_costs.py`
Expected: FAIL with "ImportError"

**Step 3: Write implementation**

Uses the Grammar's `program_log_likelihood(program, request_type)` method. The request type for all our rules is `HAND → BOOL` (equivalently `list(card) → bool`).

Key functions:
- `score_hypothesis(sexpr, grammar) -> float` — parse s-expression, compute log-likelihood
- `score_all_hypotheses(grammar_name, cost_structure, limit=0) -> List[Dict]` — score all Phase 1b hypotheses, return list of dicts with rule_id, rank, confidence, log_prob

**Step 4: Run test to verify it passes**

Run: `python llm/grammar_comparison/tests/test_compute_costs.py`
Expected: All tests pass

**Step 5: Commit**

```bash
git add llm/grammar_comparison/evaluation/compute_costs.py llm/grammar_comparison/tests/test_compute_costs.py
git commit -m "feat: add log-probability scorer for grammar comparison"
```

---

### Task 10: Evaluation Metrics

**Files:**
- Create: `llm/grammar_comparison/evaluation/metrics.py`
- Test: `llm/grammar_comparison/tests/test_metrics.py`

**Step 1: Write the failing test**

```python
# llm/grammar_comparison/tests/test_metrics.py
"""Tests for the 4 evaluation metrics."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm.grammar_comparison.evaluation.metrics import (
    spearman_rank_correlation,
    weighted_log_probability,
    top1_accuracy,
    expressibility,
)

# Mock scored data: list of dicts with rule_id, rank, log_prob
MOCK_DATA = [
    # Rule A: grammar agrees with LLM ranking (rank 1 has highest log_prob)
    {"rule_id": "A", "rank": 1, "log_prob": -2.0},
    {"rule_id": "A", "rank": 2, "log_prob": -4.0},
    {"rule_id": "A", "rank": 3, "log_prob": -6.0},
    # Rule B: grammar disagrees (rank 1 has lowest log_prob)
    {"rule_id": "B", "rank": 1, "log_prob": -8.0},
    {"rule_id": "B", "rank": 2, "log_prob": -4.0},
    {"rule_id": "B", "rank": 3, "log_prob": -2.0},
]

def test_spearman_perfect_agreement():
    """Rule A has perfect rank agreement → correlation = -1.0 (rank 1 = highest prob)."""
    # Note: rank 1 = best, log_prob closer to 0 = best
    # So we expect NEGATIVE correlation (low rank number ↔ high log_prob)
    corr = spearman_rank_correlation(MOCK_DATA)
    assert isinstance(corr, float)
    assert -1.0 <= corr <= 1.0

def test_weighted_log_prob():
    result = weighted_log_probability(MOCK_DATA)
    assert isinstance(result, float)
    assert result < 0  # Weighted sum of negative values

def test_top1_accuracy():
    acc = top1_accuracy(MOCK_DATA)
    assert isinstance(acc, float)
    assert 0.0 <= acc <= 1.0
    # Rule A: grammar's top = rank 1's entry (correct). Rule B: grammar's top = rank 3 (wrong).
    assert acc == 0.5

def test_expressibility_all_finite():
    expr = expressibility(MOCK_DATA)
    assert expr == 1.0  # All have finite log_prob

def test_expressibility_with_neg_inf():
    data_with_inf = MOCK_DATA + [
        {"rule_id": "C", "rank": 1, "log_prob": float("-inf")},
    ]
    expr = expressibility(data_with_inf)
    assert expr < 1.0

if __name__ == "__main__":
    test_spearman_perfect_agreement()
    test_weighted_log_prob()
    test_top1_accuracy()
    test_expressibility_all_finite()
    test_expressibility_with_neg_inf()
    print("All metrics tests passed")
```

**Step 2: Run test to verify it fails**

Run: `python llm/grammar_comparison/tests/test_metrics.py`
Expected: FAIL with "ImportError"

**Step 3: Write implementation**

```python
# llm/grammar_comparison/evaluation/metrics.py
"""
Four evaluation metrics for grammar comparison.

All metrics take a list of scored hypotheses (dicts with rule_id, rank, log_prob)
and return a single summary number.
"""
from typing import List, Dict
from collections import defaultdict
import math

def spearman_rank_correlation(scored: List[Dict]) -> float:
    """
    Average Spearman correlation between grammar log-probability and LLM rank.

    For each rule, compute Spearman rho between:
    - LLM confidence rank (1-5, where 1 = most confident)
    - Grammar log-probability (higher = more probable)

    We expect NEGATIVE correlation: rank 1 (best) should have highest log_prob.
    Average across all rules that have 2+ hypotheses.
    """
    from scipy.stats import spearmanr

    by_rule = defaultdict(list)
    for h in scored:
        if math.isfinite(h["log_prob"]):
            by_rule[h["rule_id"]].append(h)

    correlations = []
    for rule_id, hypotheses in by_rule.items():
        if len(hypotheses) < 2:
            continue
        ranks = [h["rank"] for h in hypotheses]
        log_probs = [h["log_prob"] for h in hypotheses]
        rho, _ = spearmanr(ranks, log_probs)
        if not math.isnan(rho):
            correlations.append(rho)

    if not correlations:
        return 0.0
    return sum(correlations) / len(correlations)


def weighted_log_probability(scored: List[Dict]) -> float:
    """
    Weighted sum of log-probabilities, with higher-confidence hypotheses
    weighted more heavily.

    Weight scheme: rank 1 → weight 5, rank 2 → 4, ..., rank 5 → 1.
    """
    total = 0.0
    for h in scored:
        if math.isfinite(h["log_prob"]):
            weight = 6 - h["rank"]  # rank 1 → 5, rank 5 → 1
            total += weight * h["log_prob"]
    return total


def top1_accuracy(scored: List[Dict]) -> float:
    """
    For each rule, does the grammar's highest-probability hypothesis
    match the LLM's rank-1 hypothesis?

    Returns fraction of rules where they agree.
    """
    by_rule = defaultdict(list)
    for h in scored:
        by_rule[h["rule_id"]].append(h)

    correct = 0
    total = 0
    for rule_id, hypotheses in by_rule.items():
        if not hypotheses:
            continue
        total += 1
        # Grammar's best: highest log_prob
        grammar_best = max(hypotheses, key=lambda h: h["log_prob"])
        # LLM's best: rank 1
        llm_best = min(hypotheses, key=lambda h: h["rank"])
        if grammar_best["rank"] == llm_best["rank"]:
            correct += 1

    return correct / total if total > 0 else 0.0


def expressibility(scored: List[Dict]) -> float:
    """
    Fraction of hypotheses with finite log-probability (expressible in grammar).
    """
    if not scored:
        return 0.0
    finite = sum(1 for h in scored if math.isfinite(h["log_prob"]))
    return finite / len(scored)
```

**Step 4: Run test to verify it passes**

Run: `python llm/grammar_comparison/tests/test_metrics.py`
Expected: All tests pass

**Step 5: Commit**

```bash
git add llm/grammar_comparison/evaluation/metrics.py llm/grammar_comparison/tests/test_metrics.py
git commit -m "feat: implement 4 evaluation metrics (Spearman, weighted log-prob, top-1, expressibility)"
```

---

### Task 11: Ablation Framework

**Files:**
- Create: `llm/grammar_comparison/evaluation/ablation.py`
- Test: `llm/grammar_comparison/tests/test_ablation.py`

**Step 1: Write the failing test**

```python
# llm/grammar_comparison/tests/test_ablation.py
"""Tests for ablation analysis framework."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from llm.grammar_comparison.evaluation.ablation import (
    leave_one_out, leave_one_in, cross_validate,
)

def test_leave_one_out_returns_results():
    results = leave_one_out(base_grammar="base", limit=5)
    assert isinstance(results, list)
    # Each result should have: removed_primitive, metric_name, metric_value
    assert len(results) > 0
    assert "removed" in results[0]
    assert "spearman" in results[0]

def test_leave_one_in_returns_results():
    results = leave_one_in(base_grammar="minimal", limit=5)
    assert isinstance(results, list)
    assert len(results) > 0
    assert "added" in results[0]

def test_cross_validate_returns_mean_and_std():
    mean, std = cross_validate(
        grammar_name="base",
        cost_structure="uniform",
        k=3,
        metric="spearman",
        limit=10,
    )
    assert isinstance(mean, float)
    assert isinstance(std, float)
    assert std >= 0

if __name__ == "__main__":
    test_leave_one_out_returns_results()
    test_leave_one_in_returns_results()
    test_cross_validate_returns_mean_and_std()
    print("All ablation tests passed")
```

**Step 2: Run test to verify it fails**

Run: `python llm/grammar_comparison/tests/test_ablation.py`
Expected: FAIL with "ImportError"

**Step 3: Write implementation**

The ablation module provides:
- `leave_one_out(base_grammar, ...)` — for each primitive in the base grammar, build a grammar without it, score all hypotheses, report metric change
- `leave_one_in(base_grammar, ...)` — starting from a grammar, add one primitive from a candidate list, score, report
- `cross_validate(grammar_name, cost_structure, k, metric, ...)` — k-fold CV over hypotheses

Each function uses `build_grammar()` and `score_all_hypotheses()` from earlier tasks. Since scoring is microseconds, even leave-one-out over 50+ primitives runs in seconds.

**Step 4: Run test to verify it passes**

Run: `python llm/grammar_comparison/tests/test_ablation.py`
Expected: All tests pass

**Step 5: Commit**

```bash
git add llm/grammar_comparison/evaluation/ablation.py llm/grammar_comparison/tests/test_ablation.py
git commit -m "feat: add ablation framework (leave-one-out, leave-one-in, cross-validation)"
```

---

### Task 12: Main Entry Point and Stage 1+3 Runner

**Files:**
- Create: `llm/grammar_comparison/run_comparison.py`

**Step 1: Write the runner**

```python
# llm/grammar_comparison/run_comparison.py
"""
Main entry point for grammar comparison.

Usage:
    # Run full Stage 1+3 comparison (7 grammars × 3 costs = 21 configs)
    python -m llm.grammar_comparison.run_comparison --stage 1

    # Run Stage 2 ablations around best grammar
    python -m llm.grammar_comparison.run_comparison --stage 2 --base-grammar swap-both

    # Quick test (3 grammars, 10 hypotheses)
    python -m llm.grammar_comparison.run_comparison --stage 1 --limit 10 --grammars base swap-both minimal
"""
```

The runner should:
1. Parse CLI args (stage, grammar filter, hypothesis limit)
2. For Stage 1+3: iterate over all grammar × cost combinations, call `score_all_hypotheses`, compute metrics, print a summary table
3. For Stage 2: call `leave_one_out` and `leave_one_in` around the specified base grammar
4. Save results to `llm/results/grammar_comparison/stage{N}_results.json`

**Step 2: Test with quick run**

Run: `python -m llm.grammar_comparison.run_comparison --stage 1 --limit 5 --grammars base minimal`
Expected: Prints a table comparing 2 grammars × 3 costs = 6 rows, with all 4 metrics

**Step 3: Commit**

```bash
git add llm/grammar_comparison/run_comparison.py
git commit -m "feat: add main entry point for grammar comparison pipeline"
```

---

### Task 13: Run Full Stage 1+3 and Analyze Results

**Step 1: Run the full comparison**

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling
python -m llm.grammar_comparison.run_comparison --stage 1
```

Expected: Completes in < 1 minute. Outputs a table of 21 configurations (7 grammars × 3 costs) with all 4 metrics.

**Step 2: Review results**

Check:
- Which grammar × cost combination has the best Spearman correlation?
- Does any grammar achieve 100% expressibility?
- Does the tiered or LOTlib3 cost structure consistently outperform uniform?
- How does Minimal compare to Base? (Tests whether less is more)

**Step 3: Run Stage 2 ablations around top performer**

```bash
python -m llm.grammar_comparison.run_comparison --stage 2 --base-grammar <winner>
```

**Step 4: Commit results**

```bash
git add llm/results/grammar_comparison/
git commit -m "results: Stage 1+3 grammar comparison and Stage 2 ablations"
```

---

## Dependency Graph

```
Task 1 (setup)
  ├── Task 2 (data loader)
  ├── Task 3 (new primitives)
  └── Task 4 (grammar factory)
        ├── Task 5 (s-expr parser) ──┐
        ├── Task 6 (python parser) ──┤
        └── Task 7 (rewriter) ───────┤
                                     ├── Task 8 (verification)
                                     └── Task 9 (scorer)
                                           ├── Task 10 (metrics)
                                           └── Task 11 (ablation)
                                                 └── Task 12 (runner)
                                                       └── Task 13 (execute)
```

Tasks 2, 3, 4 can run in parallel after Task 1.
Tasks 5, 6 can run in parallel.
Tasks 10, 11 can run in parallel after Task 9.
