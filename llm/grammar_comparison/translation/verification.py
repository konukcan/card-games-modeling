"""
Fingerprint verification for the translation pipeline.

PURPOSE:
    Verify that AST translations are semantically correct by computing
    boolean fingerprints (the pattern of True/False results over a set of
    probe hands) and checking they match between different translation paths.

HOW IT WORKS:
    1. Load 200 probe hands from llm/results/probe_set_200.json.
    2. Evaluate a Program AST on every probe hand -> tuple of bools.
    3. Evaluate a Python lambda string on every probe hand -> tuple of bools.
    4. Compare the two fingerprints to verify semantic equivalence.

    Two programs with the same fingerprint are (very likely) extensionally
    equivalent — they implement the same rule, just expressed differently.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup: allow importing from the main src/ tree
# ---------------------------------------------------------------------------
_SRC_DIR = str(Path(__file__).parent.parent.parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dreamcoder_core.program import Program
from rules.cards import Card, Hand, Suit, Rank, RANK_VALUES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rank integer -> Rank enum reverse mapping
# ---------------------------------------------------------------------------
# The probe JSON stores ranks as integers (2–14).  We need to convert back
# to Rank enum values so that Card objects are compatible with the DSL
# primitives (which operate on Card.rank : Rank).
_INT_TO_RANK: Dict[int, Rank] = {v: k for k, v in RANK_VALUES.items()}

# Suit letter -> Suit enum mapping
_LETTER_TO_SUIT: Dict[str, Suit] = {
    "S": Suit.SPADES,
    "H": Suit.HEARTS,
    "D": Suit.DIAMONDS,
    "C": Suit.CLUBS,
}


# ============================================================================
# 1. Load probe hands
# ============================================================================

_PROBE_PATH = Path(__file__).parent.parent.parent / "results" / "probe_set_200.json"


def load_probe_hands(path: Optional[Path] = None) -> List[Hand]:
    """Load the 200 probe hands from llm/results/probe_set_200.json.

    Each hand in the JSON is a list of {rank: int, suit: str} objects.
    We parse each into a list of Card objects.

    Args:
        path: Override path to probe JSON (default: llm/results/probe_set_200.json).

    Returns:
        List of 200 hands, where each hand is a list of Card objects.

    Raises:
        FileNotFoundError: If the probe file doesn't exist.
        KeyError/ValueError: If the JSON format is unexpected.
    """
    probe_file = path or _PROBE_PATH
    with open(probe_file) as f:
        data = json.load(f)

    hands: List[Hand] = []
    for hand_data in data["hands"]:
        hand: Hand = []
        for card_obj in hand_data:
            rank_int = card_obj["rank"]
            suit_letter = card_obj["suit"]
            rank = _INT_TO_RANK[rank_int]
            suit = _LETTER_TO_SUIT[suit_letter]
            hand.append(Card(suit, rank))
        hands.append(hand)

    return hands


# ============================================================================
# 2. Compute AST fingerprint
# ============================================================================

def compute_ast_fingerprint(
    program: Program,
    probes: List[Hand],
) -> Tuple[Optional[bool], ...]:
    """Evaluate a Program AST on each probe hand and return a bool fingerprint.

    The program is expected to be a lambda (hand -> bool).  We call
    program.evaluate([]) to get the closure, then apply it to each hand.

    Args:
        program: A parsed Program AST (typically an Abstraction).
        probes:  List of probe hands.

    Returns:
        Tuple of (True, False, or None) — one entry per probe hand.
        None indicates an evaluation error on that hand.
    """
    # Get the closure once (program is λ hand. body)
    try:
        func = program.evaluate([])
    except Exception as exc:
        logger.warning("Failed to evaluate program to closure: %s", exc)
        return tuple(None for _ in probes)

    results: List[Optional[bool]] = []
    for hand in probes:
        try:
            result = func(hand)
            if isinstance(result, bool):
                results.append(result)
            elif result is None:
                results.append(None)
            else:
                # Coerce truthy/falsy to bool
                results.append(bool(result))
        except Exception:
            results.append(None)

    return tuple(results)


# ============================================================================
# 3. Compute Python fingerprint
# ============================================================================

def _build_python_namespace() -> Dict[str, Any]:
    """Build the namespace that Python lambda code expects.

    This provides Card, Suit, Rank, RANK_VALUES, and any other symbols
    that the Python translations typically reference.

    Returns:
        Dict usable as globals/locals for exec().
    """
    from rules.cards import (
        Card, Suit, Rank, RANK_VALUES, Color,
        suit_to_color, card_color,
    )
    return {
        "Card": Card,
        "Suit": Suit,
        "Rank": Rank,
        "RANK_VALUES": RANK_VALUES,
        "Color": Color,
        "suit_to_color": suit_to_color,
        "card_color": card_color,
        # Builtins that Python lambdas use
        "len": len,
        "sum": sum,
        "all": all,
        "any": any,
        "set": set,
        "sorted": sorted,
        "min": min,
        "max": max,
        "abs": abs,
        "range": range,
        "list": list,
        "True": True,
        "False": False,
    }


def compute_python_fingerprint(
    code: str,
    probes: List[Hand],
) -> Tuple[Optional[bool], ...]:
    """Execute a Python lambda/def string on each probe hand.

    The code should define a callable named ``rule``.  Two forms are
    supported:

        # Form 1 — assignment to a lambda
        rule = lambda hand: all(RANK_VALUES[c.rank] > 5 for c in hand)

        # Form 2 — def statement
        def rule(hand):
            return all(RANK_VALUES[c.rank] > 5 for c in hand)

    Args:
        code:   Python source that defines ``rule``.
        probes: List of probe hands.

    Returns:
        Tuple of (True, False, or None) — one entry per probe hand.
        None indicates an evaluation error on that hand.

    Raises:
        ValueError: If the code fails to compile or doesn't define ``rule``.
    """
    namespace = _build_python_namespace()

    try:
        exec(code, namespace)  # noqa: S102  — intentional exec
    except Exception as exc:
        raise ValueError(f"Failed to compile Python code: {exc}") from exc

    if "rule" not in namespace:
        raise ValueError(
            "Python code must define 'rule'; "
            f"available names: {sorted(k for k in namespace if not k.startswith('_'))}"
        )

    rule_fn = namespace["rule"]

    results: List[Optional[bool]] = []
    for hand in probes:
        try:
            result = rule_fn(hand)
            if isinstance(result, bool):
                results.append(result)
            elif result is None:
                results.append(None)
            else:
                results.append(bool(result))
        except Exception:
            results.append(None)

    return tuple(results)


# ============================================================================
# 4. Dual-path verification
# ============================================================================

def verify_dual_path(
    sexpr: str,
    python_code: str,
    probes: List[Hand],
) -> Tuple[bool, dict]:
    """Verify that an s-expression and Python code produce the same fingerprint.

    This is the primary correctness check: if both translation paths yield
    the same boolean fingerprint, the translations are semantically equivalent.

    Args:
        sexpr:       S-expression string (e.g. "(λ all (λ ge (rank_val $0) 5) $0)").
        python_code: Python source defining ``rule``.
        probes:      List of probe hands.

    Returns:
        (match, details) where:
            match:   True if fingerprints are identical.
            details: Dict with keys 'ast_fp', 'python_fp', and optionally
                     'mismatches' (list of {index, ast, python} dicts).
    """
    from llm.grammar_comparison.translation.sexpr_parser import parse_hypothesis_sexpr

    # --- AST path ---
    try:
        program = parse_hypothesis_sexpr(sexpr)
    except Exception as exc:
        return False, {"error": f"sexpr parse failed: {exc}"}

    ast_fp = compute_ast_fingerprint(program, probes)

    # --- Python path ---
    try:
        py_fp = compute_python_fingerprint(python_code, probes)
    except ValueError as exc:
        return False, {"error": f"python code failed: {exc}"}

    # --- Compare ---
    match = ast_fp == py_fp
    details: dict = {
        "ast_fp_len": len(ast_fp),
        "python_fp_len": len(py_fp),
        "n_probes": len(probes),
    }

    if not match:
        mismatches = []
        for i, (a, p) in enumerate(zip(ast_fp, py_fp)):
            if a != p:
                mismatches.append({"index": i, "ast": a, "python": p})
        details["mismatches"] = mismatches
        details["n_mismatches"] = len(mismatches)

    return match, details


# ============================================================================
# 5. Rewrite-preserves-semantics verification
# ============================================================================

def verify_rewrite_preserves_semantics(
    sexpr: str,
    target_grammar: str,
    probes: List[Hand],
) -> Tuple[bool, dict]:
    """Verify that rewriting an s-expression to a target grammar preserves semantics.

    Parses the original s-expression, rewrites it, and compares fingerprints.

    Args:
        sexpr:          Original s-expression string.
        target_grammar: Name of the target grammar to rewrite into.
        probes:         List of probe hands.

    Returns:
        (match, details) where:
            match:   True if fingerprints are identical before and after rewrite.
            details: Dict with diagnostic information.
    """
    from llm.grammar_comparison.translation.sexpr_parser import parse_hypothesis_sexpr

    # --- Parse original ---
    try:
        original_program = parse_hypothesis_sexpr(sexpr)
    except Exception as exc:
        return False, {"error": f"sexpr parse failed: {exc}"}

    original_fp = compute_ast_fingerprint(original_program, probes)

    # --- Rewrite ---
    # The rewriter module may not exist yet; handle gracefully.
    try:
        from llm.grammar_comparison.translation.rewriter import rewrite  # type: ignore
    except ImportError:
        return False, {
            "error": "rewriter module not available yet",
            "original_fp_len": len(original_fp),
        }

    try:
        rewritten_program = rewrite(original_program, target_grammar)
    except Exception as exc:
        return False, {"error": f"rewrite failed: {exc}"}

    rewritten_fp = compute_ast_fingerprint(rewritten_program, probes)

    # --- Compare ---
    match = original_fp == rewritten_fp
    details: dict = {
        "n_probes": len(probes),
        "original_fp_len": len(original_fp),
        "rewritten_fp_len": len(rewritten_fp),
    }

    if not match:
        mismatches = []
        for i, (o, r) in enumerate(zip(original_fp, rewritten_fp)):
            if o != r:
                mismatches.append({"index": i, "original": o, "rewritten": r})
        details["mismatches"] = mismatches
        details["n_mismatches"] = len(mismatches)

    return match, details
