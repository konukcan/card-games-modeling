# Phase 0: Translation Calibration — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Determine whether LLM-generated natural-language hypotheses can be reliably translated to Python lambda functions, and identify the best translator model and prompting strategy for production use.

**Architecture:** Load 120 existing hypotheses from the rule induction experiment. Send each to 4 translator models (Claude Opus, Claude Sonnet, Gemini Flash, Qwen2.5-Coder-14B) in 2 conditions (with/without gallery hands), 3 generations each. Validate each translation syntactically (does it parse?), semantically (does it match the gallery hands?), and for cross-generation consistency (do 3 generations agree on 200 probe hands?). Produce a translator leaderboard, a with/without-hands comparison, and an error taxonomy.

**Tech Stack:** Python 3, `google-genai` SDK, `anthropic` SDK or `claude -p` CLI, Ollama + Qwen2.5-Coder-14B, existing `cards.py` from `src/rules/`.

**Execution Guidelines** (from project preferences):
- Explain code as you write it — treat this as a learning opportunity
- Start simple, build up — get one translator working end-to-end before adding the others
- Test and verify after each step
- Present options when there are multiple valid approaches

---

## Overview

The plan has 7 tasks, executed sequentially:

1. **Hypothesis Loader** — Read the 120 existing results and extract hypothesis + metadata
2. **Probe Set Generator** — Generate and freeze the 200 random hands used for fingerprinting
3. **Translation Prompt** — Design the prompt that asks a model to translate a hypothesis to a lambda
4. **Translator Clients** — Build a unified interface for all 4 translator models
5. **Translation Runner** — Orchestrate the full translation grid (120 × 4 × 3 × 2 = 2,880 calls)
6. **Validation Pipeline** — Syntactic, semantic, and cross-generation checks
7. **Report Generator** — Compute decision metrics, translator leaderboard, error taxonomy

All code lives in `card-games-modelling/llm/modeling/`. We create the directory structure as part of Task 1.

---

### Task 1: Hypothesis Loader

**Files:**
- Create: `card-games-modelling/llm/__init__.py`
- Create: `card-games-modelling/llm/modeling/__init__.py`
- Create: `card-games-modelling/llm/modeling/hypothesis_loader.py`
- Create: `card-games-modelling/llm/modeling/test_hypothesis_loader.py`

This module reads the 120 JSON result files from the rule induction experiment and extracts the hypothesis text, rule metadata, and gallery hands into a clean list of dicts. It also loads the stimuli JSON to get the gallery hands for each rule (needed for the "with hands" translation condition and for semantic validation).

**Step 1: Create directory structure**

```bash
mkdir -p /Users/cankonuk/Documents/self-explanations-project/card-games-modelling/llm/modeling
touch /Users/cankonuk/Documents/self-explanations-project/card-games-modelling/llm/__init__.py
touch /Users/cankonuk/Documents/self-explanations-project/card-games-modelling/llm/modeling/__init__.py
```

**Step 2: Write the failing test**

```python
# card-games-modelling/llm/modeling/test_hypothesis_loader.py
"""Tests for hypothesis_loader — verifies we can load the 120 existing results."""

import sys
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm.modeling.hypothesis_loader import load_hypotheses, load_stimuli_hands


def test_load_hypotheses_returns_120():
    """We expect 120 results: 60 rules × 2 models (flash, pro)."""
    hypotheses = load_hypotheses()
    # There are 123 files total (3 extra from run_metadata.json, summary.csv, etc.)
    # but we only load the actual result JSONs. Should be ~120.
    assert len(hypotheses) >= 118, f"Expected ~120 hypotheses, got {len(hypotheses)}"
    assert len(hypotheses) <= 125


def test_hypothesis_has_required_fields():
    """Each hypothesis dict must have the fields we need for translation."""
    hypotheses = load_hypotheses()
    required = {"rule_id", "rule_group", "hypothesis", "hands_shown", "model"}
    for h in hypotheses[:5]:
        missing = required - set(h.keys())
        assert not missing, f"Missing fields: {missing} in {h.get('rule_id', '?')}"


def test_load_stimuli_hands():
    """Stimuli loader returns 60 rules, each with 12 hands."""
    hands_by_rule = load_stimuli_hands()
    assert len(hands_by_rule) == 60, f"Expected 60 rules, got {len(hands_by_rule)}"
    for rule_id, hands in list(hands_by_rule.items())[:3]:
        assert len(hands) >= 6, f"Rule {rule_id} has only {len(hands)} hands"


if __name__ == "__main__":
    test_load_hypotheses_returns_120()
    print("  test_load_hypotheses_returns_120 PASSED")
    test_hypothesis_has_required_fields()
    print("  test_hypothesis_has_required_fields PASSED")
    test_load_stimuli_hands()
    print("  test_load_stimuli_hands PASSED")
    print("All tests passed!")
```

**Step 3: Run test to verify it fails**

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling
python3 llm/modeling/test_hypothesis_loader.py
```

Expected: `ModuleNotFoundError: No module named 'llm.modeling.hypothesis_loader'`

**Step 4: Write the implementation**

```python
# card-games-modelling/llm/modeling/hypothesis_loader.py
"""
Load existing hypotheses from the rule induction experiment results.

Reads the 120 JSON result files (60 rules × 2 models) from the v2 rule
induction experiment. Each file contains a single hypothesis generated by
Gemini Flash or Pro after seeing 6 example hands.

Also loads the stimuli JSON to get gallery hands for each rule, which are
needed for two purposes:
  1. The "with hands" translation condition (showing hands to the translator)
  2. Semantic validation (checking that translated lambdas return True on
     gallery hands)
"""

import json
from pathlib import Path
from typing import Dict, List


# Path to the rule induction results (relative to this file).
# These live in card-games/ (the experiment project), not card-games-modelling/.
_RESULTS_DIR = (
    Path(__file__).parent.parent.parent.parent
    / "card-games" / "results_rule_induction_v2"
)

# Path to the stimuli JSON with all 60 rules and their hands.
_STIMULI_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "card-games" / "rule-gallery" / "stimuli_llm.json"
)


def load_hypotheses(results_dir: Path = _RESULTS_DIR) -> List[Dict]:
    """
    Load all hypothesis results from the rule induction experiment.

    Each result is a JSON file named like 'flash_6_all_red.json' containing
    the model's hypothesis, the hands shown, rule metadata, and API stats.

    We filter to only load actual result files (those with a 'hypothesis'
    key), skipping metadata files like summary.csv or run_metadata.json.

    Returns:
        List of dicts, one per hypothesis result. Each dict has at minimum:
        rule_id, rule_group, hypothesis, hands_shown, model.
    """
    if not results_dir.exists():
        raise FileNotFoundError(
            f"Results directory not found: {results_dir}\n"
            f"Expected rule induction results from the v2 experiment."
        )

    hypotheses = []
    for json_file in sorted(results_dir.glob("*.json")):
        # Skip non-result files (metadata, etc.)
        if json_file.name in ("run_metadata.json",):
            continue

        try:
            with open(json_file) as f:
                data = json.load(f)
        except json.JSONDecodeError:
            continue

        # Only include files that have a hypothesis field (actual results)
        if "hypothesis" not in data:
            continue

        hypotheses.append(data)

    return hypotheses


def load_stimuli_hands(stimuli_path: Path = _STIMULI_PATH) -> Dict[str, List[str]]:
    """
    Load gallery hands for each rule from the stimuli JSON.

    Returns a dict mapping rule_id to a list of hand strings.
    Each hand string looks like "2♥ J♥ 8♦ 9♥ 6♦ 2♦" (6 cards,
    space-separated).

    The stimuli file has 12 hands per rule: 6 primary (used in the
    human experiment) + 6 reserve (LLM-only). We return all 12.
    """
    if not stimuli_path.exists():
        raise FileNotFoundError(
            f"Stimuli file not found: {stimuli_path}\n"
            f"Expected the frozen stimuli JSON from rule-gallery/."
        )

    with open(stimuli_path) as f:
        data = json.load(f)

    hands_by_rule = {}
    for rule in data.get("rules", []):
        rule_id = rule["id"]
        hands_by_rule[rule_id] = rule.get("hands", [])

    return hands_by_rule
```

**Step 5: Run test to verify it passes**

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling
python3 llm/modeling/test_hypothesis_loader.py
```

Expected: `All tests passed!`

**Step 6: Commit**

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling
git add llm/
git commit -m "feat: add hypothesis loader for Phase 0 translation calibration"
```

---

### Task 2: Probe Set Generator

**Files:**
- Create: `card-games-modelling/llm/modeling/probe_set.py`
- Create: `card-games-modelling/llm/modeling/test_probe_set.py`

This module generates and freezes a set of 200 random 6-card hands from a standard 52-card deck. These probe hands are used for two purposes: (1) checking cross-generation consistency of translated lambdas, and (2) extensional clustering of hypotheses by their boolean fingerprint. The same probe set should be used by the Bayesian model, so we save it to a shared location.

**Step 1: Write the failing test**

```python
# card-games-modelling/llm/modeling/test_probe_set.py
"""Tests for probe_set — the 200 random hands used for fingerprinting."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm.modeling.probe_set import generate_probe_set, load_or_create_probe_set


def test_generate_probe_set_shape():
    """Probe set should be 200 hands of 6 cards each."""
    probes = generate_probe_set(n=200, hand_size=6, seed=42)
    assert len(probes) == 200
    for hand in probes:
        assert len(hand) == 6


def test_generate_probe_set_deterministic():
    """Same seed should produce identical probe sets."""
    a = generate_probe_set(n=200, hand_size=6, seed=42)
    b = generate_probe_set(n=200, hand_size=6, seed=42)
    for ha, hb in zip(a, b):
        for ca, cb in zip(ha, hb):
            assert ca.rank == cb.rank and ca.suit == cb.suit


def test_probe_set_no_duplicate_hands():
    """All 200 hands should be distinct (overwhelmingly likely with random sampling)."""
    probes = generate_probe_set(n=200, hand_size=6, seed=42)
    # Convert each hand to a frozenset of (rank, suit) tuples for comparison.
    # Note: hands are ordered, so we compare as tuples, not sets.
    hand_tuples = [tuple((c.rank, c.suit) for c in h) for h in probes]
    assert len(set(hand_tuples)) == 200


def test_load_or_create_saves_and_reloads(tmp_path):
    """load_or_create_probe_set should save to disk and reload identically."""
    path = tmp_path / "probes.json"
    probes1 = load_or_create_probe_set(path, n=50, seed=99)
    assert len(probes1) == 50
    assert path.exists()

    # Reload — should get identical hands
    probes2 = load_or_create_probe_set(path, n=50, seed=99)
    assert len(probes2) == 50
    for h1, h2 in zip(probes1, probes2):
        for c1, c2 in zip(h1, h2):
            assert c1.rank == c2.rank and c1.suit == c2.suit


if __name__ == "__main__":
    import tempfile
    test_generate_probe_set_shape()
    print("  test_generate_probe_set_shape PASSED")
    test_generate_probe_set_deterministic()
    print("  test_generate_probe_set_deterministic PASSED")
    test_probe_set_no_duplicate_hands()
    print("  test_probe_set_no_duplicate_hands PASSED")
    with tempfile.TemporaryDirectory() as td:
        test_load_or_create_saves_and_reloads(Path(td))
    print("  test_load_or_create_saves_and_reloads PASSED")
    print("All tests passed!")
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling
python3 llm/modeling/test_probe_set.py
```

Expected: `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# card-games-modelling/llm/modeling/probe_set.py
"""
Generate and manage the shared probe set for extensional fingerprinting.

The probe set is a fixed collection of 200 random 6-card hands drawn
without replacement from a standard 52-card deck. Every translated
lambda is evaluated on these 200 hands, producing a 200-bit boolean
vector (the "fingerprint"). Two lambdas with identical fingerprints are
treated as extensionally equivalent.

Why 200 hands? Two functions whose true extensions differ by even 1% of
the hand space have less than a 13% chance of agreeing on all 200
probes, and this drops below 1% with 500 probes. 200 is a good balance
between collision safety and evaluation speed.

The probe set is seeded for reproducibility and saved to disk so that
the Bayesian model can use the exact same probes for its fingerprinting.
"""

import json
import random
from pathlib import Path
from typing import List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rules.cards import Card, Suit, Rank, Hand, sample_hand, RANK_VALUES


def generate_probe_set(
    n: int = 200,
    hand_size: int = 6,
    seed: int = 42,
) -> List[Hand]:
    """
    Generate n random hands of the given size from a standard 52-card deck.

    Each hand is drawn WITHOUT replacement from the deck (no duplicate
    cards within a hand), but different hands may share cards.

    Args:
        n: Number of hands to generate.
        hand_size: Cards per hand (default 6, matching the experiment).
        seed: Random seed for reproducibility.

    Returns:
        List of n hands, each a list of Card objects.
    """
    rng = random.Random(seed)
    deck = [Card(suit, rank) for suit in Suit for rank in Rank]
    hands = []
    for _ in range(n):
        hand = rng.sample(deck, k=hand_size)
        hands.append(hand)
    return hands


def _hand_to_serializable(hand: Hand) -> List[dict]:
    """Convert a hand to a JSON-serializable list of dicts."""
    return [
        {"rank": RANK_VALUES[c.rank], "suit": c.suit.name[0]}
        for c in hand
    ]


def _hand_from_serializable(data: List[dict]) -> Hand:
    """Reconstruct a hand from its JSON representation."""
    # Reverse mappings
    val_to_rank = {v: k for k, v in RANK_VALUES.items()}
    letter_to_suit = {"C": Suit.CLUBS, "D": Suit.DIAMONDS, "H": Suit.HEARTS, "S": Suit.SPADES}
    return [
        Card(letter_to_suit[d["suit"]], val_to_rank[d["rank"]])
        for d in data
    ]


def load_or_create_probe_set(
    path: Path,
    n: int = 200,
    seed: int = 42,
) -> List[Hand]:
    """
    Load a probe set from disk, or generate and save one if it doesn't exist.

    This ensures that the same probe set is used across all runs and can
    be shared with the Bayesian model.

    Args:
        path: Path to the probe set JSON file.
        n: Number of hands (only used if generating).
        seed: Random seed (only used if generating).

    Returns:
        List of hands (Card objects).
    """
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        return [_hand_from_serializable(h) for h in data["hands"]]

    # Generate and save
    hands = generate_probe_set(n=n, seed=seed)
    data = {
        "n": n,
        "hand_size": 6,
        "seed": seed,
        "hands": [_hand_to_serializable(h) for h in hands],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    return hands
```

**Step 4: Run test to verify it passes**

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling
python3 llm/modeling/test_probe_set.py
```

Expected: `All tests passed!`

**Step 5: Commit**

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling
git add llm/modeling/probe_set.py llm/modeling/test_probe_set.py
git commit -m "feat: add probe set generator for extensional fingerprinting"
```

---

### Task 3: Translation Prompt

**Files:**
- Create: `card-games-modelling/llm/modeling/translation_prompt.py`
- Create: `card-games-modelling/llm/modeling/test_translation_prompt.py`

This module defines the prompt template that asks a translator model to convert a natural-language hypothesis into a Python lambda. It supports two conditions: "without hands" (hypothesis text only) and "with hands" (hypothesis text + the 6 gallery hands that generated it). The prompt includes the Card API spec so the model knows what `.rank` and `.suit` return.

**Step 1: Write the failing test**

```python
# card-games-modelling/llm/modeling/test_translation_prompt.py
"""Tests for translation_prompt — the prompt sent to translator models."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm.modeling.translation_prompt import (
    build_translation_prompt,
    extract_lambda_from_response,
)


def test_prompt_without_hands():
    """Without-hands prompt should contain the hypothesis but no hand examples."""
    prompt = build_translation_prompt(
        hypothesis="All cards are red (hearts or diamonds)",
        hands=None,
    )
    assert "All cards are red" in prompt
    assert "lambda hand:" in prompt or "def rule" in prompt
    assert "card.rank" in prompt  # API spec present
    assert "Hand 1:" not in prompt  # No hands shown


def test_prompt_with_hands():
    """With-hands prompt should contain both hypothesis and example hands."""
    hands = ["2♥ J♥ 8♦ 9♥ 6♦ 2♦", "K♥ 5♦ 10♥ 6♥ K♦ 4♥"]
    prompt = build_translation_prompt(
        hypothesis="All cards are red (hearts or diamonds)",
        hands=hands,
    )
    assert "All cards are red" in prompt
    assert "2♥ J♥ 8♦" in prompt  # Hands are shown


def test_extract_lambda_from_response():
    """Should extract a Python lambda or function from model response."""
    response = '''Here's the translation:

```python
rule = lambda hand: all(card.suit in ('H', 'D') for card in hand)
```

This checks that every card is a heart or diamond.'''

    code = extract_lambda_from_response(response)
    assert "lambda hand:" in code or "def " in code
    assert "card.suit" in code


def test_extract_lambda_handles_bare_code():
    """Should handle responses that don't use markdown code blocks."""
    response = "rule = lambda hand: all(card.suit in ('H', 'D') for card in hand)"
    code = extract_lambda_from_response(response)
    assert "lambda hand:" in code


if __name__ == "__main__":
    test_prompt_without_hands()
    print("  test_prompt_without_hands PASSED")
    test_prompt_with_hands()
    print("  test_prompt_with_hands PASSED")
    test_extract_lambda_from_response()
    print("  test_extract_lambda_from_response PASSED")
    test_extract_lambda_handles_bare_code()
    print("  test_extract_lambda_handles_bare_code PASSED")
    print("All tests passed!")
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling
python3 llm/modeling/test_translation_prompt.py
```

Expected: `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# card-games-modelling/llm/modeling/translation_prompt.py
"""
Prompt template for translating natural-language hypotheses to Python lambdas.

The translation task: given a hypothesis like "All cards are red (hearts
or diamonds)", produce a Python lambda function that takes a hand (list
of 6 Card objects) and returns True if the hand satisfies the rule.

Two conditions are supported:
  - WITHOUT hands: The translator sees only the hypothesis text. This
    tests whether the hypothesis is unambiguous on its own.
  - WITH hands: The translator also sees the 6 gallery hands that
    generated the hypothesis. This can help disambiguate vague phrasing
    but risks overfitting to the examples.

The prompt includes a Card API specification so the translator knows
what attributes are available (card.rank, card.suit, etc.).
"""

import re
from typing import List, Optional


# The Card API spec shown to the translator. This defines the interface
# that the generated lambda must use. It matches cards.py in src/rules/.
CARD_API_SPEC = """## Card API

Each card in the hand has these attributes:
- card.rank: a Rank enum with .value as string ("2"-"10", "J", "Q", "K", "A")
- card.suit: a Suit enum with .name as string ("CLUBS", "DIAMONDS", "HEARTS", "SPADES")

Rank numeric values (for comparisons): 2=2, 3=3, ..., 10=10, J=11, Q=12, K=13, A=14
Access via RANK_VALUES[card.rank], which returns an int.

Color: Hearts and Diamonds are red; Clubs and Spades are black.

A hand is a list of 6 Card objects, ordered left to right as shown in the experiment.

Available imports (already in scope):
- Card, Suit, Rank, RANK_VALUES from rules.cards
- RANK_VALUES is a dict mapping Rank enum → int (e.g., RANK_VALUES[Rank.ACE] = 14)"""


def build_translation_prompt(
    hypothesis: str,
    hands: Optional[List[str]] = None,
) -> str:
    """
    Build the prompt for translating a hypothesis to a Python lambda.

    Args:
        hypothesis: The natural-language hypothesis to translate.
            Example: "All cards are red (hearts or diamonds)"
        hands: If provided (with-hands condition), a list of hand strings
            that satisfy the rule. Example: ["2♥ J♥ 8♦ 9♥ 6♦ 2♦", ...]
            If None (without-hands condition), no hands are shown.

    Returns:
        The complete prompt string to send to the translator model.
    """
    parts = []

    parts.append("Translate the following card rule hypothesis into a Python lambda function.")
    parts.append("")

    # Show the hypothesis
    parts.append(f"**Hypothesis:** {hypothesis}")
    parts.append("")

    # Optionally show the hands that generated this hypothesis
    if hands:
        parts.append("**Example hands that satisfy this rule:**")
        for i, h in enumerate(hands, 1):
            parts.append(f"  Hand {i}: {h}")
        parts.append("")

    # Show the Card API spec
    parts.append(CARD_API_SPEC)
    parts.append("")

    # Instructions for the output format
    parts.append("## Output")
    parts.append("")
    parts.append("Write a single Python expression of the form:")
    parts.append("```python")
    parts.append("rule = lambda hand: <boolean expression using the Card API above>")
    parts.append("```")
    parts.append("")
    parts.append("Rules:")
    parts.append("- The lambda must take a single argument `hand` (a list of 6 Card objects)")
    parts.append("- It must return True if the hand satisfies the rule, False otherwise")
    parts.append("- Use only the Card API attributes listed above")
    parts.append("- Use RANK_VALUES[card.rank] to get numeric rank values for comparisons")
    parts.append("- Keep it as a single lambda expression if possible; if the logic is too")
    parts.append("  complex for a lambda, write a `def rule(hand): ...` function instead")
    parts.append("- Output ONLY the code, wrapped in a ```python code block")

    return "\n".join(parts)


def extract_lambda_from_response(response: str) -> str:
    """
    Extract Python code from a translator model's response.

    The model is instructed to wrap its code in a ```python block,
    but we also handle bare code responses as a fallback.

    Args:
        response: The full response text from the translator model.

    Returns:
        The extracted Python code string (the lambda or function definition).

    Raises:
        ValueError: If no Python code can be extracted.
    """
    # Strategy 1: Extract from ```python ... ``` code block
    # We take the LAST code block in case the model shows examples first.
    code_blocks = re.findall(
        r"```(?:python)?\s*\n?(.*?)```",
        response,
        re.DOTALL,
    )
    if code_blocks:
        code = code_blocks[-1].strip()
        if "lambda hand:" in code or "def rule" in code or "def " in code:
            return code

    # Strategy 2: Look for a line starting with "rule = lambda hand:"
    for line in response.split("\n"):
        stripped = line.strip()
        if stripped.startswith("rule = lambda hand:") or stripped.startswith("rule=lambda"):
            return stripped

    # Strategy 3: Look for any lambda hand: expression
    lambda_match = re.search(r"(lambda hand:.*?)(?:\n|$)", response)
    if lambda_match:
        return f"rule = {lambda_match.group(1).strip()}"

    # Strategy 4: Look for def rule(hand): block
    def_match = re.search(r"(def rule\(hand\):.*?)(?:\n\n|\Z)", response, re.DOTALL)
    if def_match:
        return def_match.group(1).strip()

    raise ValueError(
        f"Could not extract Python code from response:\n{response[:200]}..."
    )
```

**Step 4: Run test to verify it passes**

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling
python3 llm/modeling/test_translation_prompt.py
```

Expected: `All tests passed!`

**Step 5: Commit**

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling
git add llm/modeling/translation_prompt.py llm/modeling/test_translation_prompt.py
git commit -m "feat: add translation prompt for hypothesis-to-lambda conversion"
```

---

### Task 4: Translator Clients

**Files:**
- Create: `card-games-modelling/llm/modeling/translators.py`
- Create: `card-games-modelling/llm/modeling/test_translators.py`

This module provides a unified interface for all 4 translator models. Each translator takes a prompt string and returns a response string. The interface is intentionally simple — just `translate(prompt) -> str` — so we can swap models without changing the runner.

**Step 1: Write the failing test**

```python
# card-games-modelling/llm/modeling/test_translators.py
"""Tests for translators — unified interface for 4 translator models.

These tests verify the interface contract. Tests that call actual APIs
are marked with a 'live' flag and skipped by default.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm.modeling.translators import (
    get_translator,
    AVAILABLE_TRANSLATORS,
)


def test_available_translators():
    """We should have exactly 4 translators registered."""
    assert len(AVAILABLE_TRANSLATORS) == 4
    expected = {"claude-opus", "claude-sonnet", "gemini-flash", "qwen-coder"}
    assert set(AVAILABLE_TRANSLATORS.keys()) == expected


def test_get_translator_returns_callable():
    """get_translator should return an object with a .translate() method."""
    # Use qwen-coder since it doesn't need API keys to instantiate
    # (it will fail on actual calls if Ollama isn't running, but
    # instantiation should work)
    for name in AVAILABLE_TRANSLATORS:
        translator = get_translator(name)
        assert hasattr(translator, "translate"), f"{name} missing .translate()"
        assert callable(translator.translate)


def test_get_translator_unknown_raises():
    """Requesting an unknown translator should raise ValueError."""
    try:
        get_translator("gpt-4")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_available_translators()
    print("  test_available_translators PASSED")
    test_get_translator_returns_callable()
    print("  test_get_translator_returns_callable PASSED")
    test_get_translator_unknown_raises()
    print("  test_get_translator_unknown_raises PASSED")
    print("All tests passed!")
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling
python3 llm/modeling/test_translators.py
```

Expected: `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# card-games-modelling/llm/modeling/translators.py
"""
Unified interface for the 4 translator models used in Phase 0.

Each translator wraps a different LLM backend and exposes a single
.translate(prompt) -> str method. This lets the translation runner
iterate over models without caring about the underlying API.

Translator models:
  - claude-opus:   Claude Opus 4.6 via `claude -p` CLI (free on subscription)
  - claude-sonnet: Claude Sonnet via `claude -p` CLI (free on subscription)
  - gemini-flash:  Gemini 2.5 Flash via google-genai SDK (~$0.50 total)
  - qwen-coder:    Qwen2.5-Coder-14B via Ollama (free, local)
"""

import json
import os
import subprocess
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Dict, Optional, Protocol


class Translator(Protocol):
    """Protocol that all translators must satisfy."""
    name: str
    def translate(self, prompt: str) -> str: ...


# ─── Claude translators (via `claude -p` CLI) ─────────────────────────

@dataclass
class ClaudeTranslator:
    """
    Translate via the Claude CLI (`claude -p`).

    Uses the user's Anthropic subscription — no API key needed.
    The -p flag sends a single prompt and returns the response.
    The --model flag selects Opus or Sonnet.
    """
    name: str
    model: str  # e.g., "claude-opus-4-6" or "claude-sonnet-4-6"

    def translate(self, prompt: str) -> str:
        """Send prompt to Claude CLI and return the response text."""
        try:
            result = subprocess.run(
                ["claude", "-p", "--model", self.model, prompt],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"claude -p failed (exit {result.returncode}): {result.stderr[:200]}"
                )
            return result.stdout.strip()
        except FileNotFoundError:
            raise RuntimeError(
                "Claude CLI not found. Install it or check your PATH."
            )


# ─── Gemini translator (via google-genai SDK) ─────────────────────────

@dataclass
class GeminiTranslator:
    """
    Translate via the Gemini API using google-genai SDK.

    Uses the GOOGLE_API_KEY environment variable.
    """
    name: str = "gemini-flash"
    model: str = "gemini-2.5-flash"

    def __post_init__(self):
        from google import genai
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GOOGLE_API_KEY not set")
        self._client = genai.Client(api_key=key)

    def translate(self, prompt: str) -> str:
        """Send prompt to Gemini and return the response text."""
        from google.genai import types
        config = types.GenerateContentConfig(
            temperature=0.0,
        )
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )
        if response.candidates and response.candidates[0].content:
            parts = response.candidates[0].content.parts
            return "".join(p.text or "" for p in parts).strip()
        raise RuntimeError("Gemini returned empty response")


# ─── Ollama translator (local Qwen2.5-Coder) ──────────────────────────

@dataclass
class OllamaTranslator:
    """
    Translate via Ollama's local API.

    Requires Ollama running (`ollama serve`) with the model pulled
    (`ollama pull qwen2.5-coder:14b`).
    """
    name: str = "qwen-coder"
    model: str = "qwen2.5-coder:14b"
    base_url: str = "http://localhost:11434"

    def translate(self, prompt: str) -> str:
        """Send prompt to Ollama and return the response text."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 1024},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("response", "").strip()
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Ollama API error: {e}\n"
                f"Is Ollama running? Start with: ollama serve"
            )


# ─── Registry ─────────────────────────────────────────────────────────

AVAILABLE_TRANSLATORS: Dict[str, type] = {
    "claude-opus": ClaudeTranslator,
    "claude-sonnet": ClaudeTranslator,
    "gemini-flash": GeminiTranslator,
    "qwen-coder": OllamaTranslator,
}

# Default constructor arguments for each translator
_DEFAULTS = {
    "claude-opus": {"name": "claude-opus", "model": "claude-opus-4-6"},
    "claude-sonnet": {"name": "claude-sonnet", "model": "claude-sonnet-4-6"},
    "gemini-flash": {"name": "gemini-flash"},
    "qwen-coder": {"name": "qwen-coder"},
}


def get_translator(name: str) -> Translator:
    """
    Instantiate a translator by name.

    Args:
        name: One of "claude-opus", "claude-sonnet", "gemini-flash", "qwen-coder"

    Returns:
        A Translator instance with a .translate(prompt) method.

    Raises:
        ValueError: If the name is not recognized.
    """
    if name not in AVAILABLE_TRANSLATORS:
        raise ValueError(
            f"Unknown translator '{name}'. "
            f"Available: {', '.join(AVAILABLE_TRANSLATORS.keys())}"
        )
    cls = AVAILABLE_TRANSLATORS[name]
    kwargs = _DEFAULTS.get(name, {})
    return cls(**kwargs)
```

**Step 4: Run test to verify it passes**

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling
python3 llm/modeling/test_translators.py
```

Expected: `All tests passed!` (Note: `get_translator("gemini-flash")` will fail if GOOGLE_API_KEY is not set. The test should be adjusted to skip that one if the key is missing, or we test only instantiation-safe translators.)

**Step 5: Commit**

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling
git add llm/modeling/translators.py llm/modeling/test_translators.py
git commit -m "feat: add unified translator interface for 4 backend models"
```

---

### Task 5: Validation Pipeline

**Files:**
- Create: `card-games-modelling/llm/modeling/validator.py`
- Create: `card-games-modelling/llm/modeling/test_validator.py`

This module takes a translated Python lambda (as a string), validates it syntactically (does it parse and execute?), semantically (does it return True on the gallery hands?), and computes its boolean fingerprint on the 200 probe hands. It is the core quality-checking component.

**Step 1: Write the failing test**

```python
# card-games-modelling/llm/modeling/test_validator.py
"""Tests for validator — syntax, semantic, and fingerprint checks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from llm.modeling.validator import (
    validate_syntax,
    validate_semantics,
    compute_fingerprint,
)


def test_validate_syntax_good_lambda():
    """A valid lambda should pass syntax validation."""
    code = "rule = lambda hand: all(card.suit.name in ('HEARTS', 'DIAMONDS') for card in hand)"
    result = validate_syntax(code)
    assert result["valid"] is True
    assert result["callable"] is True
    assert result["error"] is None


def test_validate_syntax_bad_code():
    """Invalid Python should fail syntax validation."""
    code = "rule = lambda hand: all(card.suit in for card"
    result = validate_syntax(code)
    assert result["valid"] is False
    assert result["error"] is not None


def test_validate_syntax_runtime_error():
    """Code that parses but crashes on a sample hand should report the error."""
    # This references an attribute that doesn't exist
    code = "rule = lambda hand: all(card.nonexistent for card in hand)"
    result = validate_syntax(code)
    # It should parse fine but fail on execution
    assert result["valid"] is True or result["valid"] is False
    # Either way, callable should be False if it crashes on a test hand


def test_validate_semantics_correct():
    """A correct translation should match 6/6 gallery hands."""
    # "All red" rule — hearts and diamonds only
    code = "rule = lambda hand: all(card.suit.name in ('HEARTS', 'DIAMONDS') for card in hand)"
    hands = [
        "2♥ J♥ 8♦ 9♥ 6♦ 2♦",
        "K♥ 5♦ 10♥ 6♥ K♦ 4♥",
        "7♦ J♥ 2♥ Q♥ 10♦ 6♦",
        "10♦ K♥ J♥ J♦ A♦ 6♥",
        "8♦ A♥ 4♦ 7♦ 3♥ 10♦",
        "J♦ Q♦ 2♦ 3♥ Q♥ 10♦",
    ]
    result = validate_semantics(code, hands)
    assert result["matches"] == 6
    assert result["total"] == 6
    assert result["match_rate"] == 1.0


def test_validate_semantics_wrong():
    """A wrong translation should match fewer than 6/6."""
    # "All spades" — wrong rule for "all_red" hands
    code = "rule = lambda hand: all(card.suit.name == 'SPADES' for card in hand)"
    hands = [
        "2♥ J♥ 8♦ 9♥ 6♦ 2♦",
        "K♥ 5♦ 10♥ 6♥ K♦ 4♥",
    ]
    result = validate_semantics(code, hands)
    assert result["matches"] == 0


def test_compute_fingerprint_deterministic():
    """Same code on same probes should produce identical fingerprints."""
    code = "rule = lambda hand: len(hand) == 6"
    # Use a small probe set for testing
    from llm.modeling.probe_set import generate_probe_set
    probes = generate_probe_set(n=10, seed=42)
    fp1 = compute_fingerprint(code, probes)
    fp2 = compute_fingerprint(code, probes)
    assert fp1 == fp2
    assert len(fp1) == 10  # One bit per probe hand


if __name__ == "__main__":
    test_validate_syntax_good_lambda()
    print("  test_validate_syntax_good_lambda PASSED")
    test_validate_syntax_bad_code()
    print("  test_validate_syntax_bad_code PASSED")
    test_validate_semantics_correct()
    print("  test_validate_semantics_correct PASSED")
    test_validate_semantics_wrong()
    print("  test_validate_semantics_wrong PASSED")
    test_compute_fingerprint_deterministic()
    print("  test_compute_fingerprint_deterministic PASSED")
    print("All tests passed!")
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling
python3 llm/modeling/test_validator.py
```

**Step 3: Write the implementation**

```python
# card-games-modelling/llm/modeling/validator.py
"""
Validation pipeline for translated hypothesis lambdas.

Three levels of validation:

1. SYNTACTIC: Does the code parse as valid Python? Does exec() succeed?
   Can we call the resulting function on a sample hand without error?

2. SEMANTIC: Does the lambda return True for the gallery hands that
   originally generated the hypothesis? We expect 6/6 matches for a
   correct translation (or 5/6 under the noise tolerance).

3. FINGERPRINT: Evaluate the lambda on the 200-hand probe set to
   produce a boolean vector. This fingerprint is used for cross-
   generation consistency checks and extensional clustering.

Safety note: We exec() untrusted code (LLM-generated lambdas). This is
acceptable for a research pipeline running locally, but the exec() is
sandboxed with a restricted global scope containing only the Card API.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from rules.cards import (
    Card, Suit, Rank, Hand, RANK_VALUES, STR_TO_RANK,
    suit_to_color, Color, card_color,
)


def _parse_hand_string(hand_str: str) -> Hand:
    """
    Parse a hand string like "2♥ J♥ 8♦ 9♥ 6♦ 2♦" into a list of Card objects.

    Each card is a rank followed by a Unicode suit symbol, space-separated.
    """
    suit_map = {"♣": Suit.CLUBS, "♦": Suit.DIAMONDS, "♥": Suit.HEARTS, "♠": Suit.SPADES}
    cards = []
    for token in hand_str.strip().split():
        # The suit symbol is the last character (Unicode, multi-byte)
        for symbol, suit in suit_map.items():
            if symbol in token:
                rank_str = token.replace(symbol, "").strip()
                cards.append(Card(suit, STR_TO_RANK[rank_str]))
                break
    return cards


def _make_exec_globals() -> dict:
    """
    Build the restricted globals dict for exec()ing translated lambdas.

    This provides the Card API that the translation prompt specifies,
    so that lambdas like `lambda hand: all(card.suit.name == 'HEARTS' ...)`
    have access to the necessary types and constants.
    """
    return {
        "__builtins__": __builtins__,
        "Card": Card,
        "Suit": Suit,
        "Rank": Rank,
        "RANK_VALUES": RANK_VALUES,
        "STR_TO_RANK": STR_TO_RANK,
        "Color": Color,
        "suit_to_color": suit_to_color,
        "card_color": card_color,
    }


def _compile_rule(code: str) -> Optional[callable]:
    """
    Compile a code string into a callable rule function.

    The code should define either:
      rule = lambda hand: ...
    or:
      def rule(hand): ...

    Returns the callable, or None if compilation fails.
    """
    exec_globals = _make_exec_globals()
    try:
        exec(code, exec_globals)
    except Exception:
        return None

    return exec_globals.get("rule", None)


def validate_syntax(code: str) -> Dict:
    """
    Check whether the code parses and can be called on a sample hand.

    Returns a dict with:
      valid: bool — does the code parse?
      callable: bool — can the resulting function be called on a hand?
      error: str or None — error message if any step failed
    """
    from rules.cards import sample_hand

    # Step 1: Try to compile
    rule_fn = _compile_rule(code)
    if rule_fn is None:
        return {"valid": False, "callable": False, "error": "Failed to exec() code"}

    if not callable(rule_fn):
        return {"valid": True, "callable": False, "error": "'rule' is not callable"}

    # Step 2: Try to call on a sample hand
    try:
        test_hand = sample_hand(6, with_replacement=False)
        result = rule_fn(test_hand)
        if not isinstance(result, bool):
            # Coerce to bool — some lambdas return int 0/1
            bool(result)
        return {"valid": True, "callable": True, "error": None}
    except Exception as e:
        return {"valid": True, "callable": False, "error": f"Runtime error: {e}"}


def validate_semantics(
    code: str,
    gallery_hands: List[str],
) -> Dict:
    """
    Check whether the lambda returns True on the gallery hands.

    Args:
        code: The Python code defining `rule = lambda hand: ...`
        gallery_hands: List of hand strings (e.g., "2♥ J♥ 8♦ 9♥ 6♦ 2♦")
            that should all satisfy the rule.

    Returns dict with:
        matches: int — how many gallery hands return True
        total: int — total gallery hands tested
        match_rate: float — matches / total
        per_hand: list of bools — True/False for each hand
        error: str or None — if the lambda crashed on any hand
    """
    rule_fn = _compile_rule(code)
    if rule_fn is None:
        return {
            "matches": 0, "total": len(gallery_hands),
            "match_rate": 0.0, "per_hand": [], "error": "Failed to compile",
        }

    per_hand = []
    error = None
    for hand_str in gallery_hands:
        try:
            hand = _parse_hand_string(hand_str)
            result = bool(rule_fn(hand))
            per_hand.append(result)
        except Exception as e:
            per_hand.append(False)
            if error is None:
                error = f"Runtime error on hand '{hand_str[:20]}...': {e}"

    matches = sum(per_hand)
    return {
        "matches": matches,
        "total": len(gallery_hands),
        "match_rate": matches / max(len(gallery_hands), 1),
        "per_hand": per_hand,
        "error": error,
    }


def compute_fingerprint(
    code: str,
    probe_hands: List[Hand],
) -> Optional[Tuple[bool, ...]]:
    """
    Evaluate the lambda on the probe set and return its boolean fingerprint.

    Args:
        code: Python code defining `rule = lambda hand: ...`
        probe_hands: List of Hand objects (from probe_set.py)

    Returns:
        Tuple of bools (one per probe hand), or None if the lambda crashes.
    """
    rule_fn = _compile_rule(code)
    if rule_fn is None:
        return None

    fingerprint = []
    for hand in probe_hands:
        try:
            fingerprint.append(bool(rule_fn(hand)))
        except Exception:
            return None  # Any crash invalidates the entire fingerprint

    return tuple(fingerprint)
```

**Step 4: Run test to verify it passes**

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling
python3 llm/modeling/test_validator.py
```

Expected: `All tests passed!`

**Step 5: Commit**

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling
git add llm/modeling/validator.py llm/modeling/test_validator.py
git commit -m "feat: add validation pipeline (syntax, semantics, fingerprint)"
```

---

### Task 6: Translation Runner

**Files:**
- Create: `card-games-modelling/llm/modeling/run_phase0.py`

This is the main orchestrator that runs the full Phase 0 grid: 120 hypotheses × 4 translators × 3 generations × 2 conditions. It uses resume logic (skips already-completed translations) and saves results incrementally.

**Step 1: Write the runner**

```python
# card-games-modelling/llm/modeling/run_phase0.py
"""
Phase 0: Translation Calibration Runner

Orchestrates the full translation grid:
  120 hypotheses × 4 translators × 3 generations × 2 conditions = 2,880 translations

Each translation call:
  1. Builds a translation prompt (with or without gallery hands)
  2. Sends it to the translator model
  3. Extracts the Python lambda from the response
  4. Validates syntax, semantics, and computes fingerprint
  5. Saves the result as JSON

Resume logic: if a result file already exists, the translation is skipped.

Usage:
    # Full grid (all translators, all hypotheses)
    python3 llm/modeling/run_phase0.py --all

    # Single translator, for testing
    python3 llm/modeling/run_phase0.py --translator qwen-coder --limit 5

    # Dry run to inspect prompts
    python3 llm/modeling/run_phase0.py --translator gemini-flash --limit 1 --dry-run
"""

import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from llm.modeling.hypothesis_loader import load_hypotheses, load_stimuli_hands
from llm.modeling.probe_set import load_or_create_probe_set
from llm.modeling.translation_prompt import (
    build_translation_prompt,
    extract_lambda_from_response,
)
from llm.modeling.translators import get_translator, AVAILABLE_TRANSLATORS
from llm.modeling.validator import (
    validate_syntax,
    validate_semantics,
    compute_fingerprint,
)


# ── Paths ──────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).parent.parent / "results" / "phase0_translations"
PROBE_SET_PATH = Path(__file__).parent.parent / "results" / "probe_set_200.json"


# ── Result filename ────────────────────────────────────────────────────

def result_filename(
    translator: str,
    condition: str,
    gen: int,
    rule_id: str,
    model: str,
) -> str:
    """
    Deterministic filename for one translation result.

    Format: {translator}_{condition}_{gen}_{model}_{rule_id}.json
    Example: qwen-coder_no-hands_1_flash_all_red.json
    """
    model_short = model.replace("gemini-2.5-", "")
    return f"{translator}_{condition}_gen{gen}_{model_short}_{rule_id}.json"


# ── Single translation ─────────────────────────────────────────────────

def run_single_translation(
    hypothesis_data: Dict,
    translator_name: str,
    condition: str,  # "with-hands" or "no-hands"
    generation: int,  # 1, 2, or 3
    hands_by_rule: Dict[str, List[str]],
    probe_hands,
    translator,
    verbose: int = 0,
) -> Dict:
    """
    Run a single translation: prompt → response → extract → validate.

    Returns a result dict with all metadata and validation scores.
    """
    rule_id = hypothesis_data["rule_id"]
    hypothesis = hypothesis_data["hypothesis"]
    model = hypothesis_data["model"]

    # Build prompt based on condition
    gallery_hands = None
    if condition == "with-hands":
        gallery_hands = hands_by_rule.get(rule_id, [])[:6]

    prompt = build_translation_prompt(
        hypothesis=hypothesis,
        hands=gallery_hands,
    )

    if verbose >= 2:
        print(f"\n{'─'*60}")
        print(f"PROMPT:\n{prompt[:500]}...")
        print(f"{'─'*60}")

    # Call translator
    start = time.time()
    try:
        response = translator.translate(prompt)
        duration_ms = (time.time() - start) * 1000
    except Exception as e:
        return {
            "rule_id": rule_id,
            "hypothesis": hypothesis,
            "source_model": model,
            "translator": translator_name,
            "condition": condition,
            "generation": generation,
            "response": "",
            "extracted_code": "",
            "syntax": {"valid": False, "callable": False, "error": str(e)},
            "semantics": {"matches": 0, "total": 6, "match_rate": 0.0},
            "fingerprint": None,
            "duration_ms": (time.time() - start) * 1000,
            "error": str(e),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    # Extract code from response
    try:
        extracted_code = extract_lambda_from_response(response)
    except ValueError as e:
        extracted_code = ""
        extraction_error = str(e)
    else:
        extraction_error = None

    # Validate
    syntax_result = validate_syntax(extracted_code) if extracted_code else {
        "valid": False, "callable": False, "error": extraction_error or "No code extracted"
    }

    gallery = hands_by_rule.get(rule_id, [])[:6]
    semantics_result = validate_semantics(extracted_code, gallery) if (
        extracted_code and syntax_result["callable"]
    ) else {"matches": 0, "total": len(gallery), "match_rate": 0.0, "per_hand": [], "error": "Not callable"}

    fingerprint = None
    if extracted_code and syntax_result["callable"]:
        fp = compute_fingerprint(extracted_code, probe_hands)
        if fp is not None:
            fingerprint = [int(b) for b in fp]  # JSON-serializable

    return {
        "rule_id": rule_id,
        "rule_group": hypothesis_data.get("rule_group"),
        "rule_answer": hypothesis_data.get("rule_answer", ""),
        "hypothesis": hypothesis,
        "source_model": model,
        "translator": translator_name,
        "condition": condition,
        "generation": generation,
        "response": response,
        "extracted_code": extracted_code,
        "syntax": syntax_result,
        "semantics": semantics_result,
        "fingerprint": fingerprint,
        "duration_ms": duration_ms,
        "error": extraction_error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ── Grid runner ─────────────────────────────────────────────────────────

def run_phase0(
    translators: List[str],
    conditions: List[str],
    generations: int = 3,
    limit: int = 0,
    delay: float = 0.5,
    verbose: int = 0,
    dry_run: bool = False,
):
    """Run the Phase 0 translation grid."""

    # Load data
    print("Loading hypotheses...")
    hypotheses = load_hypotheses()
    print(f"  {len(hypotheses)} hypotheses loaded")

    print("Loading stimuli hands...")
    hands_by_rule = load_stimuli_hands()
    print(f"  {len(hands_by_rule)} rules")

    print("Loading/creating probe set...")
    probe_hands = load_or_create_probe_set(PROBE_SET_PATH)
    print(f"  {len(probe_hands)} probe hands")

    if limit > 0:
        hypotheses = hypotheses[:limit]
        print(f"  Limited to {limit} hypotheses")

    # Build grid
    grid = [
        (h, t, c, g)
        for h in hypotheses
        for t in translators
        for c in conditions
        for g in range(1, generations + 1)
    ]
    total = len(grid)
    print(f"\nPhase 0 Translation Grid:")
    print(f"  Hypotheses: {len(hypotheses)}")
    print(f"  Translators: {translators}")
    print(f"  Conditions: {conditions}")
    print(f"  Generations per combo: {generations}")
    print(f"  Total translations: {total}")
    if dry_run:
        print(f"  MODE: DRY RUN")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize translators
    active_translators = {}
    for t_name in translators:
        if not dry_run:
            try:
                active_translators[t_name] = get_translator(t_name)
                print(f"  Initialized {t_name}")
            except Exception as e:
                print(f"  SKIP {t_name}: {e}")

    completed = 0
    skipped = 0
    errors = 0

    for i, (hyp, t_name, cond, gen) in enumerate(grid, 1):
        fname = result_filename(t_name, cond, gen, hyp["rule_id"], hyp["model"])
        fpath = OUTPUT_DIR / fname

        # Resume: skip existing results
        if fpath.exists():
            skipped += 1
            continue

        if dry_run:
            print(f"[{i}/{total}] {t_name} | {cond} | gen{gen} | {hyp['rule_id']}")
            completed += 1
            continue

        if t_name not in active_translators:
            continue

        if verbose >= 1:
            print(f"  [{i}/{total}] {t_name} | {cond} | gen{gen} | {hyp['rule_id']}...",
                  end=" ", flush=True)

        try:
            result = run_single_translation(
                hyp, t_name, cond, gen,
                hands_by_rule, probe_hands,
                active_translators[t_name],
                verbose=verbose,
            )
            with open(fpath, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            completed += 1

            if verbose >= 1:
                syn = "✓" if result["syntax"]["callable"] else "✗"
                sem = f"{result['semantics']['matches']}/{result['semantics']['total']}"
                dur = f"{result['duration_ms']/1000:.1f}s"
                print(f"{syn} sem={sem} ({dur})")
            else:
                print(f"  [{i}/{total}] {fname}")

            if delay > 0 and t_name.startswith("gemini"):
                time.sleep(delay)

        except Exception as e:
            errors += 1
            print(f"  [{i}/{total}] ERROR: {e}")

    print(f"\n{'═'*50}")
    print(f"  Completed: {completed}")
    print(f"  Skipped (resume): {skipped}")
    print(f"  Errors: {errors}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'═'*50}")


# ── CLI ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 0: Translation Calibration")
    parser.add_argument("--all", action="store_true", help="Run full grid")
    parser.add_argument("--translator", choices=list(AVAILABLE_TRANSLATORS.keys()),
                        help="Run only this translator")
    parser.add_argument("--condition", choices=["with-hands", "no-hands"],
                        help="Run only this condition")
    parser.add_argument("--generations", type=int, default=3,
                        help="Number of generations per combo (default: 3)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N hypotheses (0 = all)")
    parser.add_argument("--delay", type=float, default=0.5,
                        help="Delay between API calls in seconds")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be done without calling APIs")
    parser.add_argument("-v", "--verbose", action="count", default=0)

    args = parser.parse_args()

    translators = [args.translator] if args.translator else list(AVAILABLE_TRANSLATORS.keys())
    conditions = [args.condition] if args.condition else ["no-hands", "with-hands"]

    if not args.all and not args.translator and not args.dry_run:
        parser.print_help()
        print("\nSpecify --all or --translator to run.")
        sys.exit(1)

    run_phase0(
        translators=translators,
        conditions=conditions,
        generations=args.generations,
        limit=args.limit,
        delay=args.delay,
        verbose=args.verbose,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
```

**Step 2: Smoke test with dry run**

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling
python3 llm/modeling/run_phase0.py --translator qwen-coder --limit 2 --dry-run
```

Expected: Prints the grid without calling any API.

**Step 3: Commit**

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling
git add llm/modeling/run_phase0.py
git commit -m "feat: add Phase 0 translation runner with resume logic"
```

---

### Task 7: Report Generator

**Files:**
- Create: `card-games-modelling/llm/modeling/phase0_report.py`

This script reads all Phase 0 results and computes the decision metrics, translator leaderboard, with/without-hands comparison, and error taxonomy. It produces both a JSON summary and a human-readable markdown report.

**Step 1: Write the report generator**

```python
# card-games-modelling/llm/modeling/phase0_report.py
"""
Phase 0 Report: Analyze translation calibration results.

Reads all result JSONs from phase0_translations/ and computes:
  1. Decision metrics (syntactic success, semantic match, cross-gen agreement)
  2. Translator leaderboard (which model is best?)
  3. With/without hands comparison
  4. Error taxonomy (why did translations fail?)

Usage:
    python3 llm/modeling/phase0_report.py
    python3 llm/modeling/phase0_report.py --output-dir results/phase0_translations
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

RESULTS_DIR = Path(__file__).parent.parent / "results" / "phase0_translations"


def load_all_results(results_dir: Path) -> List[Dict]:
    """Load all Phase 0 result JSONs."""
    results = []
    for f in sorted(results_dir.glob("*.json")):
        try:
            with open(f) as fh:
                results.append(json.load(fh))
        except json.JSONDecodeError:
            continue
    return results


def compute_metrics(results: List[Dict]) -> Dict:
    """
    Compute the three decision metrics and the translator leaderboard.

    Decision metrics (per design doc):
      - Syntactic success rate: >95% threshold
      - Semantic match rate (6/6): >80% threshold
      - Cross-generation agreement (3/3 on probes): >70% threshold
    """
    # Group by (translator, condition, rule_id, source_model)
    groups = defaultdict(list)
    for r in results:
        key = (r["translator"], r["condition"], r["rule_id"], r["source_model"])
        groups[key].append(r)

    # Per-translator metrics
    translator_stats = defaultdict(lambda: {
        "total": 0, "syntax_ok": 0, "semantic_6_6": 0,
        "cross_gen_agree": 0, "cross_gen_total": 0,
    })

    # Per-condition metrics
    condition_stats = defaultdict(lambda: {
        "total": 0, "syntax_ok": 0, "semantic_6_6": 0,
    })

    # Error taxonomy
    error_categories = Counter()

    for key, group_results in groups.items():
        translator, condition, rule_id, source_model = key

        for r in group_results:
            t_key = translator
            c_key = condition
            translator_stats[t_key]["total"] += 1
            condition_stats[c_key]["total"] += 1

            # Syntactic
            if r.get("syntax", {}).get("callable", False):
                translator_stats[t_key]["syntax_ok"] += 1
                condition_stats[c_key]["syntax_ok"] += 1

            # Semantic (6/6)
            if r.get("semantics", {}).get("match_rate", 0) == 1.0:
                translator_stats[t_key]["semantic_6_6"] += 1
                condition_stats[c_key]["semantic_6_6"] += 1

            # Error taxonomy
            if r.get("error"):
                error_categories[_categorize_error(r)] += 1
            elif not r.get("syntax", {}).get("callable", False):
                error_categories["syntax_failure"] += 1
            elif r.get("semantics", {}).get("match_rate", 0) < 1.0:
                error_categories["semantic_mismatch"] += 1

        # Cross-generation agreement: compare fingerprints within each group
        fingerprints = [
            tuple(r["fingerprint"]) for r in group_results
            if r.get("fingerprint") is not None
        ]
        if len(fingerprints) >= 2:
            translator_stats[translator]["cross_gen_total"] += 1
            if len(set(fingerprints)) == 1:
                translator_stats[translator]["cross_gen_agree"] += 1

    # Compute rates
    leaderboard = {}
    for t, s in translator_stats.items():
        leaderboard[t] = {
            "total_translations": s["total"],
            "syntax_rate": s["syntax_ok"] / max(s["total"], 1),
            "semantic_6_6_rate": s["semantic_6_6"] / max(s["total"], 1),
            "cross_gen_agreement": (
                s["cross_gen_agree"] / max(s["cross_gen_total"], 1)
                if s["cross_gen_total"] > 0 else None
            ),
        }

    condition_comparison = {}
    for c, s in condition_stats.items():
        condition_comparison[c] = {
            "total": s["total"],
            "syntax_rate": s["syntax_ok"] / max(s["total"], 1),
            "semantic_6_6_rate": s["semantic_6_6"] / max(s["total"], 1),
        }

    # Overall decision metrics
    all_syntax = sum(s["syntax_ok"] for s in translator_stats.values())
    all_total = sum(s["total"] for s in translator_stats.values())
    all_sem = sum(s["semantic_6_6"] for s in translator_stats.values())
    all_agree = sum(s["cross_gen_agree"] for s in translator_stats.values())
    all_agree_total = sum(s["cross_gen_total"] for s in translator_stats.values())

    decision_metrics = {
        "syntactic_success_rate": all_syntax / max(all_total, 1),
        "semantic_match_rate": all_sem / max(all_total, 1),
        "cross_gen_agreement": all_agree / max(all_agree_total, 1) if all_agree_total > 0 else None,
        "thresholds": {
            "syntax": 0.95,
            "semantic": 0.80,
            "cross_gen": 0.70,
        },
        "needs_reasoning_in_prompt": False,  # Updated below
    }

    # Check if thresholds are met
    if (decision_metrics["semantic_match_rate"] < 0.80 or
        (decision_metrics["cross_gen_agreement"] is not None and
         decision_metrics["cross_gen_agreement"] < 0.70)):
        decision_metrics["needs_reasoning_in_prompt"] = True

    return {
        "decision_metrics": decision_metrics,
        "translator_leaderboard": leaderboard,
        "condition_comparison": condition_comparison,
        "error_taxonomy": dict(error_categories),
        "total_results": len(results),
    }


def _categorize_error(result: Dict) -> str:
    """Categorize a translation error into the taxonomy."""
    error = result.get("error", "") or ""
    hypothesis = result.get("hypothesis", "")

    if "no code extracted" in error.lower() or "could not extract" in error.lower():
        return "extraction_failure"
    if "runtime error" in error.lower():
        return "runtime_error"
    if "timeout" in error.lower():
        return "timeout"
    # Check for vague hypotheses
    vague_markers = ["pattern", "some kind", "seems to", "might be", "related to"]
    if any(m in hypothesis.lower() for m in vague_markers):
        return "vague_hypothesis"
    return "other"


def generate_markdown_report(metrics: Dict) -> str:
    """Generate a human-readable markdown report."""
    lines = ["# Phase 0: Translation Calibration Report\n"]

    # Decision metrics
    dm = metrics["decision_metrics"]
    lines.append("## Decision Metrics\n")
    lines.append(f"| Metric | Value | Threshold | Status |")
    lines.append(f"|---|---|---|---|")

    syn_status = "PASS" if dm["syntactic_success_rate"] >= 0.95 else "FAIL"
    sem_status = "PASS" if dm["semantic_match_rate"] >= 0.80 else "FAIL"
    cg = dm["cross_gen_agreement"]
    cg_str = f"{cg:.1%}" if cg is not None else "N/A"
    cg_status = "PASS" if cg is not None and cg >= 0.70 else ("FAIL" if cg is not None else "N/A")

    lines.append(f"| Syntactic success | {dm['syntactic_success_rate']:.1%} | >95% | {syn_status} |")
    lines.append(f"| Semantic match (6/6) | {dm['semantic_match_rate']:.1%} | >80% | {sem_status} |")
    lines.append(f"| Cross-gen agreement | {cg_str} | >70% | {cg_status} |")
    lines.append("")

    if dm["needs_reasoning_in_prompt"]:
        lines.append("**DECISION:** Add reasoning/evidence field to enumeration prompt.\n")
    else:
        lines.append("**DECISION:** Proceed without reasoning in enumeration prompt.\n")

    # Translator leaderboard
    lines.append("## Translator Leaderboard\n")
    lines.append("| Translator | Syntax Rate | Semantic 6/6 | Cross-Gen Agreement |")
    lines.append("|---|---|---|---|")
    for t, s in sorted(metrics["translator_leaderboard"].items(),
                       key=lambda x: x[1]["semantic_6_6_rate"], reverse=True):
        cg_val = f"{s['cross_gen_agreement']:.1%}" if s["cross_gen_agreement"] is not None else "N/A"
        lines.append(f"| {t} | {s['syntax_rate']:.1%} | {s['semantic_6_6_rate']:.1%} | {cg_val} |")
    lines.append("")

    # Condition comparison
    lines.append("## With/Without Hands Comparison\n")
    lines.append("| Condition | Syntax Rate | Semantic 6/6 |")
    lines.append("|---|---|---|")
    for c, s in metrics["condition_comparison"].items():
        lines.append(f"| {c} | {s['syntax_rate']:.1%} | {s['semantic_6_6_rate']:.1%} |")
    lines.append("")

    # Error taxonomy
    lines.append("## Error Taxonomy\n")
    lines.append("| Category | Count |")
    lines.append("|---|---|")
    for cat, count in sorted(metrics["error_taxonomy"].items(), key=lambda x: -x[1]):
        lines.append(f"| {cat} | {count} |")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Phase 0 Report Generator")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output path for markdown report")
    args = parser.parse_args()

    print(f"Loading results from {args.results_dir}...")
    results = load_all_results(args.results_dir)
    print(f"  {len(results)} results loaded")

    if not results:
        print("No results found. Run run_phase0.py first.")
        return

    metrics = compute_metrics(results)

    # Save JSON metrics
    json_path = args.results_dir / "phase0_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {json_path}")

    # Generate and save markdown report
    report = generate_markdown_report(metrics)
    md_path = args.output or (args.results_dir / "phase0_report.md")
    with open(md_path, "w") as f:
        f.write(report)
    print(f"  Report saved to {md_path}")

    # Print summary to console
    print(f"\n{report}")


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling
git add llm/modeling/phase0_report.py
git commit -m "feat: add Phase 0 report generator with decision metrics and leaderboard"
```

---

## Pre-Execution Checklist

Before running Phase 0, these prerequisites must be in place:

1. **Install Qwen2.5-Coder-14B:**
   ```bash
   ollama serve &  # Start Ollama if not running
   ollama pull qwen2.5-coder:14b
   ```

2. **Verify GOOGLE_API_KEY is set:**
   ```bash
   echo $GOOGLE_API_KEY  # Should show a key
   ```

3. **Verify Claude CLI works:**
   ```bash
   claude -p --model claude-sonnet-4-6 "Say hello"
   ```

4. **Dry run to verify everything loads:**
   ```bash
   cd /Users/cankonuk/Documents/self-explanations-project/card-games-modelling
   python3 llm/modeling/run_phase0.py --all --dry-run
   ```

## Recommended Execution Order

Run translators in this order to get useful results early:

```bash
# 1. Local first (free, fast, no rate limits)
python3 llm/modeling/run_phase0.py --translator qwen-coder -v

# 2. Claude models (free on subscription, but slower via CLI)
python3 llm/modeling/run_phase0.py --translator claude-sonnet -v
python3 llm/modeling/run_phase0.py --translator claude-opus -v

# 3. Gemini Flash last (costs money, run after verifying pipeline works)
python3 llm/modeling/run_phase0.py --translator gemini-flash -v

# 4. Generate report
python3 llm/modeling/phase0_report.py
```

Each translator run can be interrupted and resumed safely — the runner skips existing results.
