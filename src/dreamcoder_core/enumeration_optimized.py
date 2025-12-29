"""
Optimized Program Enumeration for DreamCoder

This module provides an optimized version of enumeration with:
1. Multiprocessing support for parallel task enumeration
2. Early pruning for all-or-nothing mode
3. Configurable likelihood modes (all-or-nothing vs relaxed)

Based on Ellis et al. (2021) DreamCoder implementation.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union
from enum import Enum
import heapq
import math
import time
import multiprocessing as mp
from collections import defaultdict
from functools import partial

from .type_system import (
    Type, BaseType, Arrow, ListType, TypeVariable,
    TypeContext, UnificationError,
    BOOL, INT, CARD, SUIT, RANK, HAND, arrow, type_arity
)
from .program import (
    Program, Primitive, Application, Abstraction, Index, Invented,
    apply_args
)
from .grammar import Grammar, Production


# ============================================================================
# LIKELIHOOD MODES
# ============================================================================

class LikelihoodMode(Enum):
    """
    How to score programs that don't solve all examples.

    ALL_OR_NOTHING: Standard DreamCoder - only 100% correct programs are kept
    RELAXED: Keep top-k programs by accuracy, even if not perfect
    """
    ALL_OR_NOTHING = "all_or_nothing"
    RELAXED = "relaxed"


@dataclass
class LikelihoodConfig:
    """Configuration for likelihood computation."""
    mode: LikelihoodMode = LikelihoodMode.ALL_OR_NOTHING

    # For RELAXED mode: minimum accuracy to be considered
    min_accuracy: float = 0.5

    # For RELAXED mode: how many partial solutions to keep per task
    max_partial_solutions: int = 10

    def __post_init__(self):
        if isinstance(self.mode, str):
            self.mode = LikelihoodMode(self.mode)


def compute_log_likelihood(
    correct: int,
    total: int,
    config: LikelihoodConfig
) -> Tuple[float, bool]:
    """
    Compute log-likelihood for a program based on how many examples it got right.

    Args:
        correct: Number of correct examples
        total: Total number of examples
        config: Likelihood configuration

    Returns:
        (log_likelihood, should_add_to_frontier)
    """
    if correct == total:
        # Perfect solution
        return 0.0, True

    if config.mode == LikelihoodMode.ALL_OR_NOTHING:
        # Only perfect solutions count
        return float('-inf'), False

    # RELAXED mode: compute partial likelihood
    accuracy = correct / total

    if accuracy < config.min_accuracy:
        # Below minimum threshold
        return float('-inf'), False

    # Log-likelihood based on accuracy
    # Using log(accuracy) as a simple likelihood model
    # This gives 0 for perfect, negative for imperfect
    log_likelihood = math.log(accuracy + 1e-10)

    return log_likelihood, True


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class EnumerationResult:
    """Result of enumerating programs for a task."""
    program: Program
    log_probability: float  # Grammar probability
    log_likelihood: float   # How well it fits the examples
    description_length: float  # = -log_probability
    programs_enumerated: int  # How many programs were tried
    time_seconds: float  # Wall clock time
    accuracy: float = 1.0  # Fraction of examples correct


@dataclass
class TaskFrontier:
    """
    Top-k solutions for a single task.

    Supports both perfect and partial solutions depending on likelihood mode.
    """
    task_name: str
    request_type: Type
    entries: List[EnumerationResult] = field(default_factory=list)
    max_size: int = 10

    # Tracking
    total_programs_searched: int = 0
    total_time: float = 0.0

    # Hash-based deduplication for O(1) duplicate checking
    _seen_hashes: Set[int] = field(default_factory=set)

    def add(self, result: EnumerationResult) -> bool:
        """
        Add a result if it improves the frontier.
        Returns True if added.
        """
        # Check for duplicates using hash (O(1) instead of O(n) string comparison)
        prog_hash = hash(result.program)
        if prog_hash in self._seen_hashes:
            return False

        self._seen_hashes.add(prog_hash)
        self.entries.append(result)

        # Sort by posterior (higher is better = less negative)
        # posterior = log_prob + log_likelihood
        self.entries.sort(key=lambda e: -(e.log_probability + e.log_likelihood))

        # Keep only top-k
        if len(self.entries) > self.max_size:
            removed = self.entries.pop()
            return result != removed
        return True

    @property
    def best(self) -> Optional[EnumerationResult]:
        return self.entries[0] if self.entries else None

    @property
    def solved(self) -> bool:
        """Task is solved if we have at least one perfect solution."""
        return any(e.log_likelihood == 0.0 for e in self.entries)

    @property
    def has_partial(self) -> bool:
        """Check if we have any partial (non-perfect) solutions."""
        return any(e.log_likelihood < 0.0 and e.log_likelihood > float('-inf')
                   for e in self.entries)

    @property
    def n_solutions(self) -> int:
        return len(self.entries)

    @property
    def n_perfect_solutions(self) -> int:
        """Count only perfect solutions."""
        return sum(1 for e in self.entries if e.log_likelihood == 0.0)

    def perfect_solutions(self) -> List[EnumerationResult]:
        """Get only perfect solutions."""
        return [e for e in self.entries if e.log_likelihood == 0.0]

    def partial_solutions(self) -> List[EnumerationResult]:
        """Get only partial solutions (non-perfect but kept)."""
        return [e for e in self.entries
                if e.log_likelihood < 0.0 and e.log_likelihood > float('-inf')]

    def all_programs(self) -> List[Program]:
        """Get all programs in this frontier."""
        return [e.program for e in self.entries]


# ============================================================================
# ENUMERATION WITH EARLY PRUNING
# ============================================================================

def enumerate_for_task_optimized(
    grammar: Grammar,
    task_name: str,
    examples: List[Tuple[Any, Any]],
    request_type: Type,
    eval_fn: Callable[[Program, Any], Any],
    max_depth: int = 8,
    max_programs: int = 100000,
    timeout_seconds: float = 180.0,
    keep_top_k: int = 5,
    likelihood_config: LikelihoodConfig = None
) -> TaskFrontier:
    """
    Enumerate programs to solve a task with optimizations.

    Key optimizations:
    1. Early pruning: In ALL_OR_NOTHING mode, stop evaluating as soon as one example fails
    2. Configurable likelihood: Support both strict and relaxed modes

    Args:
        grammar: The PCFG
        task_name: Name of the task
        examples: List of (input, output) pairs
        request_type: Type of programs to enumerate
        eval_fn: Function to evaluate a program on an input
        max_depth: Maximum AST depth
        max_programs: Max programs to try
        timeout_seconds: Timeout
        keep_top_k: Number of solutions to keep
        likelihood_config: Likelihood computation settings

    Returns:
        TaskFrontier with best programs found
    """
    if likelihood_config is None:
        likelihood_config = LikelihoodConfig()

    frontier = TaskFrontier(
        task_name=task_name,
        request_type=request_type,
        max_size=keep_top_k
    )

    start_time = time.time()
    programs_tried = 0

    # Import here to avoid circular imports
    from .enumeration import enumerate_simple

    for program, log_prob in enumerate_simple(grammar, request_type, max_depth=max_depth):
        programs_tried += 1

        # Check budget and timeout
        if programs_tried > max_programs:
            break
        if time.time() - start_time > timeout_seconds:
            break

        # Evaluate program on examples
        try:
            correct = 0
            total = len(examples)

            if likelihood_config.mode == LikelihoodMode.ALL_OR_NOTHING:
                # EARLY PRUNING: Stop as soon as one example fails
                all_correct = True
                for inp, expected in examples:
                    result = eval_fn(program, inp)
                    if result == expected:
                        correct += 1
                    else:
                        all_correct = False
                        break  # Early exit!

                if all_correct:
                    # Need to count remaining examples
                    correct = total
            else:
                # RELAXED mode: evaluate all examples to get accuracy
                for inp, expected in examples:
                    result = eval_fn(program, inp)
                    if result == expected:
                        correct += 1

            # Compute likelihood
            log_likelihood, should_add = compute_log_likelihood(
                correct, total, likelihood_config
            )

            if should_add:
                accuracy = correct / total
                entry = EnumerationResult(
                    program=program,
                    log_probability=log_prob,
                    log_likelihood=log_likelihood,
                    description_length=-log_prob / math.log(2),
                    programs_enumerated=programs_tried,
                    time_seconds=time.time() - start_time,
                    accuracy=accuracy
                )
                frontier.add(entry)

                # In ALL_OR_NOTHING mode, stop after finding enough perfect solutions
                if (likelihood_config.mode == LikelihoodMode.ALL_OR_NOTHING and
                    frontier.n_perfect_solutions >= keep_top_k):
                    break

        except Exception:
            # Program crashed - skip it
            pass

    frontier.total_programs_searched = programs_tried
    frontier.total_time = time.time() - start_time

    return frontier


# ============================================================================
# MULTIPROCESSING SUPPORT
# ============================================================================

def _worker_enumerate_task(args: Tuple) -> Dict:
    """
    Worker function for multiprocessing.

    Args is a tuple of:
        (task_dict, grammar_pickle, eval_fn_name, config_dict)

    Returns a dict with results.
    """
    import pickle

    (task_name, examples, request_type_str,
     grammar_data, max_depth, max_programs,
     timeout_seconds, keep_top_k, likelihood_config_dict) = args

    # Reconstruct objects from serializable forms
    # Note: Grammar and types need special handling
    # For now, we pass the grammar directly (works if using fork)

    try:
        # Import grammar and type reconstruction
        from .grammar import Grammar
        from .lean_primitives import build_lean_grammar
        from .type_system import arrow, HAND, BOOL

        # Rebuild grammar (in practice, might want to pass pickled version)
        grammar = build_lean_grammar()

        # Rebuild request type
        request_type = arrow(HAND, BOOL)  # Assuming standard card game type

        # Rebuild likelihood config
        likelihood_config = LikelihoodConfig(**likelihood_config_dict)

        # Create eval function
        from .program import Program
        def eval_fn(program: Program, hand):
            fn = program.evaluate([])
            return fn(hand)

        # Run enumeration
        frontier = enumerate_for_task_optimized(
            grammar=grammar,
            task_name=task_name,
            examples=examples,
            request_type=request_type,
            eval_fn=eval_fn,
            max_depth=max_depth,
            max_programs=max_programs,
            timeout_seconds=timeout_seconds,
            keep_top_k=keep_top_k,
            likelihood_config=likelihood_config
        )

        # Convert to serializable format
        return {
            'task_name': task_name,
            'solved': frontier.solved,
            'n_solutions': frontier.n_solutions,
            'programs_searched': frontier.total_programs_searched,
            'time': frontier.total_time,
            'entries': [
                {
                    'program': str(e.program),
                    'log_probability': e.log_probability,
                    'log_likelihood': e.log_likelihood,
                    'accuracy': e.accuracy,
                    'programs_enumerated': e.programs_enumerated
                }
                for e in frontier.entries
            ]
        }
    except Exception as e:
        return {
            'task_name': task_name,
            'error': str(e),
            'solved': False,
            'n_solutions': 0
        }


def enumerate_tasks_parallel(
    grammar: Grammar,
    tasks: List[Dict],  # List of {name, examples, request_type}
    eval_fn: Callable,
    max_depth: int = 8,
    max_programs: int = 100000,
    timeout_seconds: float = 180.0,
    keep_top_k: int = 5,
    likelihood_config: LikelihoodConfig = None,
    n_workers: int = None,
    verbose: bool = True
) -> List[TaskFrontier]:
    """
    Enumerate programs for multiple tasks in parallel.

    Args:
        grammar: The PCFG
        tasks: List of task dicts with 'name', 'examples', 'request_type'
        eval_fn: Evaluation function
        max_depth: Maximum AST depth
        max_programs: Max programs per task
        timeout_seconds: Timeout per task
        keep_top_k: Solutions to keep per task
        likelihood_config: Likelihood settings
        n_workers: Number of parallel workers (default: CPU count)
        verbose: Print progress

    Returns:
        List of TaskFrontier, one per task
    """
    if likelihood_config is None:
        likelihood_config = LikelihoodConfig()

    if n_workers is None:
        n_workers = mp.cpu_count()

    # Prepare arguments for workers
    # Note: In practice, you'd want to pickle the grammar more efficiently
    worker_args = []
    for task in tasks:
        args = (
            task['name'],
            task['examples'],
            str(task['request_type']),
            None,  # grammar_data - handled in worker
            max_depth,
            max_programs,
            timeout_seconds,
            keep_top_k,
            {'mode': likelihood_config.mode.value,
             'min_accuracy': likelihood_config.min_accuracy,
             'max_partial_solutions': likelihood_config.max_partial_solutions}
        )
        worker_args.append(args)

    if verbose:
        print(f"Starting parallel enumeration with {n_workers} workers...")
        print(f"Tasks: {len(tasks)}, Budget: {max_programs}, Depth: {max_depth}")

    # Run in parallel
    start_time = time.time()

    # Use 'spawn' context on macOS to avoid fork issues
    # For simpler cases, you might use a ProcessPoolExecutor
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_worker_enumerate_task, worker_args)

    if verbose:
        elapsed = time.time() - start_time
        solved = sum(1 for r in results if r.get('solved', False))
        print(f"Parallel enumeration complete: {solved}/{len(tasks)} solved in {elapsed:.1f}s")

    # Convert results back to TaskFrontier objects
    # Note: Programs are strings here, would need reconstruction for full use
    frontiers = []
    for result in results:
        frontier = TaskFrontier(
            task_name=result['task_name'],
            request_type=None,  # Would need reconstruction
            max_size=keep_top_k
        )
        frontier.total_programs_searched = result.get('programs_searched', 0)
        frontier.total_time = result.get('time', 0.0)
        # Note: entries would need program reconstruction
        frontiers.append(frontier)

    return frontiers


# ============================================================================
# SIMPLIFIED PARALLEL ENUMERATION (Thread-based for easier debugging)
# ============================================================================

def enumerate_tasks_sequential_optimized(
    grammar: Grammar,
    tasks: List,  # List of Task objects or dicts
    eval_fn: Callable[[Program, Any], Any],
    max_depth: int = 8,
    max_programs: int = 100000,
    timeout_seconds: float = 180.0,
    keep_top_k: int = 5,
    likelihood_config: LikelihoodConfig = None,
    verbose: bool = True,
    log_fn: Callable[[str], None] = None
) -> Dict[str, TaskFrontier]:
    """
    Enumerate programs for multiple tasks sequentially with optimizations.

    This is a drop-in replacement for the current enumeration loop in
    run_overnight_pretraining.py but with early pruning.

    Args:
        grammar: The PCFG
        tasks: List of Task objects
        eval_fn: Evaluation function
        max_depth: Maximum AST depth
        max_programs: Max programs per task
        timeout_seconds: Timeout per task
        keep_top_k: Solutions to keep per task
        likelihood_config: Likelihood settings
        verbose: Print progress
        log_fn: Custom logging function

    Returns:
        Dict mapping task_name -> TaskFrontier
    """
    if likelihood_config is None:
        likelihood_config = LikelihoodConfig()

    if log_fn is None:
        log_fn = print if verbose else lambda x: None

    frontiers = {}
    total_start = time.time()

    for i, task in enumerate(tasks):
        # Handle both Task objects and dicts
        if hasattr(task, 'name'):
            task_name = task.name
            examples = task.examples
            request_type = task.request_type
        else:
            task_name = task['name']
            examples = task['examples']
            request_type = task['request_type']

        frontier = enumerate_for_task_optimized(
            grammar=grammar,
            task_name=task_name,
            examples=examples,
            request_type=request_type,
            eval_fn=eval_fn,
            max_depth=max_depth,
            max_programs=max_programs,
            timeout_seconds=timeout_seconds,
            keep_top_k=keep_top_k,
            likelihood_config=likelihood_config
        )

        frontiers[task_name] = frontier

        if frontier.solved and verbose:
            log_fn(f"  SOLVED: {task_name} ({frontier.total_programs_searched:,} programs)")
        elif frontier.has_partial and verbose:
            best = frontier.best
            if best:
                log_fn(f"  PARTIAL: {task_name} (accuracy={best.accuracy:.1%})")

    if verbose:
        total_time = time.time() - total_start
        solved = sum(1 for f in frontiers.values() if f.solved)
        total_programs = sum(f.total_programs_searched for f in frontiers.values())
        log_fn(f"Enumeration complete: {solved}/{len(tasks)} solved, "
               f"{total_programs:,} programs in {total_time:.1f}s")

    return frontiers


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'LikelihoodMode',
    'LikelihoodConfig',
    'compute_log_likelihood',
    'EnumerationResult',
    'TaskFrontier',
    'enumerate_for_task_optimized',
    'enumerate_tasks_parallel',
    'enumerate_tasks_sequential_optimized',
]
