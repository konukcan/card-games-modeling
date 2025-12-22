"""
Unified Task Definition
=======================

This module provides the canonical Task class used across all DreamCoder components.
All modules should import Task from here rather than defining their own.

HISTORY:
    Previously, Task was defined in multiple places:
    - wake_sleep.py (basic: name, request_type, examples)
    - contrastive_dreaming.py (+ family, difficulty_level, near_miss_pairs)
    - dreamcoder_original.py (+ family, difficulty_level)
    - run_recognition_dream_experiment.py (+ holdout, rule_fn)

    This unified definition is a superset of all variants.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

from .type_system import Type


__all__ = ['Task']


@dataclass
class Task:
    """
    A learning task defined by input-output examples.

    For card game rules:
    - Input: a hand (list of cards)
    - Output: True/False (does the hand satisfy the rule?)

    Attributes:
        name: Unique identifier for this task
        request_type: Type signature of the program (e.g., hand -> bool)
        examples: Training examples as [(input, output), ...]
        holdout: Held-out examples for verification (optional)
        family: Task family/category for grouping (optional)
        difficulty_level: Numeric difficulty rating (optional)
        near_miss_pairs: Indices of contrastive example pairs (optional)
        rule_fn: The underlying rule function for generating examples (optional)
    """
    # Required fields
    name: str
    request_type: Type
    examples: List[Tuple[Any, Any]]

    # Optional fields with defaults
    holdout: List[Tuple[Any, Any]] = field(default_factory=list)
    family: str = ""
    difficulty_level: int = 0
    near_miss_pairs: List[Tuple[int, int]] = field(default_factory=list)
    rule_fn: Optional[Callable] = None

    def __hash__(self) -> int:
        """Hash by name for use in sets/dicts."""
        return hash(self.name)

    def __eq__(self, other) -> bool:
        """Equality by name."""
        if not isinstance(other, Task):
            return False
        return self.name == other.name

    def __str__(self) -> str:
        """Human-readable representation."""
        holdout_str = f", {len(self.holdout)} holdout" if self.holdout else ""
        family_str = f", family={self.family}" if self.family else ""
        return f"Task({self.name}, {len(self.examples)} examples{holdout_str}{family_str})"

    def __repr__(self) -> str:
        return self.__str__()
