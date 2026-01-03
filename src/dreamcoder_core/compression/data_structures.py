"""
Data structures for compression (library learning).

This module contains the core dataclasses used throughout the compression system:
- SubtreeOccurrence: Tracks common subtrees found across programs
- CompressionResult: Final result of compression analysis
- CompressionState: State for beam search exploration

EXTRACTED FROM: compression.py lines 64-200
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# Import from sibling modules (one level up)
from ..program import Program, Invented
from ..grammar import Grammar


@dataclass
class SubtreeOccurrence:
    """
    Tracks occurrences of a subtree across programs.

    Used for compression analysis to identify candidates for abstraction.

    FIELDS:
    -------
    subtree: Program
        The common subtree pattern found across programs.

    count: int
        Number of programs containing this subtree.
        (Not total occurrences - we count each program once)

    programs: List[str]
        Identifiers of programs containing this subtree.
        Useful for debugging and analysis.

    savings: float
        Estimated description length savings if we abstract this subtree.
        Formula: (size - 1) × (count - 1)

        COMPARISON TO ORIGINAL:
        Original DreamCoder computes full MDL change:
            ΔDL = DL(new_grammar) + Σ DL(rewritten_programs)
                - DL(old_grammar) - Σ DL(original_programs)
        Our heuristic is faster but less accurate.
    """
    subtree: Program
    count: int
    programs: List[str]
    savings: float

    def __str__(self) -> str:
        return f"{self.subtree} (count={self.count}, savings={self.savings:.2f})"


@dataclass
class CompressionResult:
    """
    Result of compression analysis.

    FIELDS:
    -------
    new_inventions: List[Invented]
        Newly discovered abstractions to add to grammar.

    old_grammar: Grammar
        Grammar before compression.

    new_grammar: Grammar
        Grammar after adding inventions (with normalized probabilities).

    total_savings: float
        Sum of savings from all inventions.
        NOTE: This is the heuristic savings, not true MDL change.

    subtree_analysis: List[SubtreeOccurrence]
        All common subtrees found (for debugging/analysis).

    rewritten_frontiers: Optional[List[List[Tuple[Program, float]]]]
        If program refactoring was enabled, contains the frontiers with
        all programs rewritten to use the new inventions.
        None if refactoring was not performed.

    rewrite_stats: Optional[Dict[str, Any]]
        Statistics about the rewriting process:
        - total_replacements: Total number of pattern replacements
        - programs_changed: Number of programs that were modified
        - size_reduction: Total reduction in AST size
        None if refactoring was not performed.
    """
    new_inventions: List[Invented]
    old_grammar: Grammar
    new_grammar: Grammar
    total_savings: float
    subtree_analysis: List[SubtreeOccurrence]
    rewritten_frontiers: Optional[List[List[Tuple[Program, float]]]] = None
    rewrite_stats: Optional[Dict[str, Any]] = None


@dataclass
class CompressionState:
    """
    State in beam search over compression choices.

    Used by beam_search_compression() to explore multiple paths through
    the space of possible abstractions, avoiding local optima.

    FIELDS:
    -------
    grammar: Grammar
        Current grammar (with inventions added so far).

    programs: List[Program]
        Current programs (possibly rewritten with inventions).

    inventions: List[Invented]
        Inventions added so far in this search path.

    targets: List[Tuple[Program, int]]
        (target, n_args) pairs for each invention, needed for rewriting.

    mdl: float
        Current MDL score (lower is better).

    history: List[str]
        Human-readable log of decisions (for debugging).
    """
    grammar: Grammar
    programs: List[Program]
    inventions: List[Invented]
    targets: List[Tuple[Program, int]]
    mdl: float
    history: List[str] = field(default_factory=list)

    def __lt__(self, other: 'CompressionState') -> bool:
        """For heap operations: lower MDL is better."""
        return self.mdl < other.mdl

    def __hash__(self) -> int:
        """Hash based on invention set for deduplication."""
        return hash(tuple(str(inv) for inv in self.inventions))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompressionState):
            return False
        return set(str(inv) for inv in self.inventions) == set(str(inv) for inv in other.inventions)
