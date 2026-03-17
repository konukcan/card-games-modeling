"""
S-expression parser for Phase 1b LLM-generated hypotheses.

PURPOSE:
    Parse hypothesis s-expressions like:
        (λ all (λ eq (mod (rank_val $0) 2) 0) $0)
    into Program AST objects from src/dreamcoder_core/program.py.

HOW IT WORKS:
    The existing parse_program(s, primitives) function already handles the full
    s-expression syntax (lambdas, application, de Bruijn indices).  All we need
    is a complete primitives dictionary mapping every name used in the hypotheses
    to its Primitive object.

    _build_primitive_registry() constructs that dictionary by calling
    build_primitives() from the main DSL and adding aliases for names that the
    LLM hypotheses use but that differ from the canonical primitive names
    (e.g. "sum_vals" -> the same Primitive as "sum_ranks").
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Path setup: allow importing from the main src/ tree
# ---------------------------------------------------------------------------
_SRC_DIR = str(Path(__file__).parent.parent.parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dreamcoder_core.program import Primitive, Program, parse_program
from dreamcoder_core.primitives import build_primitives
from dreamcoder_core.type_system import INT, arrow

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Primitive registry (singleton — built once, cached)
# ---------------------------------------------------------------------------

_REGISTRY: Optional[Dict[str, Primitive]] = None


def _build_primitive_registry() -> Dict[str, Primitive]:
    """Build a dictionary mapping every known primitive name to its Primitive.

    This includes:
    1. All 64 primitives from the canonical DSL (build_primitives()).
    2. Aliases for names that appear in Phase 1b hypotheses but use different
       names than the canonical DSL (e.g. "sum_vals" -> "sum_ranks").
    3. A multiplication primitive ("*") which the LLM sometimes generates
       but is not in the base grammar.

    Returns:
        Dict mapping name strings to Primitive objects.
    """
    # Start with all canonical primitives keyed by name
    prims = build_primitives()
    registry: Dict[str, Primitive] = {p.name: p for p in prims}

    # ------------------------------------------------------------------
    # Aliases: names used in LLM hypotheses that map to existing prims
    # ------------------------------------------------------------------
    # The LLM sometimes uses alternative names for the same operations.
    # We map them to the canonical Primitive object so parse_program can
    # resolve them.
    _aliases = {
        # Aggregate aliases: LLM uses "val" variants, DSL uses "rank" variants
        "sum_vals": "sum_ranks",
        "max_val": "max_rank",
        "min_val": "min_rank",

        # Length alias: LLM uses "n_cards", DSL uses "length"
        "n_cards": "length",
    }

    for alias, canonical in _aliases.items():
        if canonical in registry:
            registry[alias] = registry[canonical]
        else:
            logger.warning(
                "Alias target %r not found in registry; skipping alias %r",
                canonical, alias,
            )

    # ------------------------------------------------------------------
    # Extra primitives not in the base grammar but used by the LLM
    # ------------------------------------------------------------------
    # Multiplication: the LLM occasionally generates (λ ... (* x y) ...)
    if "*" not in registry:
        registry["*"] = Primitive(
            "*", arrow(INT, INT, INT), lambda x: lambda y: x * y
        )

    return registry


def get_primitive_registry() -> Dict[str, Primitive]:
    """Return the cached primitive registry, building it on first call.

    The registry is a module-level singleton so we don't re-create 64+
    Primitive objects on every parse call.

    Returns:
        Dict mapping primitive name strings to Primitive objects.
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_primitive_registry()
    return _REGISTRY


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_hypothesis_sexpr(sexpr: str) -> Program:
    """Parse a Phase 1b hypothesis s-expression into a Program AST.

    This is a thin wrapper around the existing parse_program() that supplies
    the full primitive registry automatically.

    Args:
        sexpr: An s-expression string, e.g.
               "(λ all (λ eq (get_suit $0) CLUBS) $0)"

    Returns:
        A Program AST (typically an Abstraction since all hypotheses are
        λ hand. ...).

    Raises:
        ValueError: If the s-expression has syntax errors or uses names
            that are not in the primitive registry.  The error message
            includes the unrecognised token for debugging.
    """
    registry = get_primitive_registry()
    return parse_program(sexpr.strip(), registry)
