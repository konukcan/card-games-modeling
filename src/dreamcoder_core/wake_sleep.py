"""
Wake-Sleep Learning Loop for DreamCoder

This module implements the full DreamCoder learning cycle:

1. WAKE (Enumeration): Search for programs that solve tasks
2. SLEEP (Compression): Extract common abstractions into library
3. SLEEP (Recognition): Train neural network to guide search (optional)
4. Iterate until convergence or budget exhausted

Key metrics for cognitive realism:
- Learning curve: success rate vs. iterations
- Transfer effects: how learning rule X affects learning rule Y
- Library growth: what abstractions are discovered
- Enumeration effort: programs searched per solution
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Set, Tuple, Union
from collections import defaultdict
import math
import time
import json
from pathlib import Path
import copy

from .type_system import (
    Type, BaseType, Arrow, ListType, TypeVariable,
    TypeContext, UnificationError,
    BOOL, INT, CARD, SUIT, RANK, HAND, arrow
)
from .program import (
    Program, Primitive, Application, Abstraction, Index, Invented,
    apply_args
)
from .grammar import Grammar, Production, uniform_grammar
from .enumeration import (
    Enumerator, Frontier, EnumerationResult,
    TopDownEnumerator, enumerate_for_task
)
# NOTE: enumerate_simple is deprecated - use TopDownEnumerator instead
from .compression import (
    compress_frontiers, CompressionResult, compression_report,
    find_common_subtrees
)
from .task import Task


@dataclass
class IterationResult:
    """Result of one wake-sleep iteration."""
    iteration: int
    tasks_solved: int
    total_tasks: int
    new_inventions: List[Invented]
    programs_enumerated: int
    time_seconds: float
    frontiers: Dict[str, Frontier]

    @property
    def success_rate(self) -> float:
        return self.tasks_solved / self.total_tasks if self.total_tasks > 0 else 0.0


@dataclass
class LearningMetrics:
    """
    Comprehensive metrics for cognitive analysis.

    Tracks how difficult each task was to learn.
    """
    task_name: str
    solved: bool
    iteration_solved: Optional[int]  # Which iteration found a solution
    programs_enumerated: int  # Total programs tried
    time_to_solve: float  # Seconds to find solution
    solution: Optional[Program]
    solution_size: int  # AST size of solution
    description_length: float  # Bits

    def to_dict(self) -> Dict:
        return {
            'task_name': self.task_name,
            'solved': self.solved,
            'iteration_solved': self.iteration_solved,
            'programs_enumerated': self.programs_enumerated,
            'time_to_solve': self.time_to_solve,
            'solution': str(self.solution) if self.solution else None,
            'solution_size': self.solution_size,
            'description_length': self.description_length
        }


@dataclass
class DreamCoderResult:
    """Complete result of a DreamCoder run."""
    initial_grammar: Grammar
    final_grammar: Grammar
    iterations: List[IterationResult]
    task_metrics: Dict[str, LearningMetrics]
    library_growth: List[int]  # Number of primitives after each iteration
    total_time: float

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = ["=" * 60]
        lines.append("DREAMCODER LEARNING SUMMARY")
        lines.append("=" * 60)

        lines.append(f"\nIterations: {len(self.iterations)}")
        lines.append(f"Total time: {self.total_time:.2f}s")

        lines.append(f"\nInitial primitives: {len(self.initial_grammar)}")
        lines.append(f"Final primitives: {len(self.final_grammar)}")

        solved = sum(1 for m in self.task_metrics.values() if m.solved)
        total = len(self.task_metrics)
        lines.append(f"\nTasks solved: {solved}/{total} ({100*solved/total:.1f}%)")

        lines.append("\nLearning curve:")
        for it in self.iterations:
            lines.append(f"  Iter {it.iteration}: {it.tasks_solved}/{it.total_tasks} "
                        f"({100*it.success_rate:.1f}%), "
                        f"{len(it.new_inventions)} new abstractions")

        lines.append("\nLibrary growth: " + " -> ".join(map(str, self.library_growth)))

        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)

    def to_json(self) -> str:
        """Export results to JSON for analysis."""
        return json.dumps({
            'iterations': len(self.iterations),
            'total_time': self.total_time,
            'initial_primitives': len(self.initial_grammar),
            'final_primitives': len(self.final_grammar),
            'library_growth': self.library_growth,
            'task_metrics': {k: v.to_dict() for k, v in self.task_metrics.items()},
            'learning_curve': [
                {
                    'iteration': it.iteration,
                    'tasks_solved': it.tasks_solved,
                    'success_rate': it.success_rate,
                    'new_inventions': len(it.new_inventions)
                }
                for it in self.iterations
            ]
        }, indent=2)


class DreamCoder:
    """
    The DreamCoder learning system.

    Implements wake-sleep learning with:
    - Wake: Enumerate programs to solve tasks
    - Sleep: Compress to find abstractions
    - Sleep: (Optional) Train recognition network
    """

    def __init__(
        self,
        grammar: Grammar,
        tasks: List[Task],
        eval_fn: Callable[[Program, Any], Any],
        max_iterations: int = 10,
        enumeration_timeout: float = 30.0,
        enumeration_budget: int = 10000,
        max_depth: int = 6,
        compress_every: int = 1,
        verbose: bool = True
    ):
        """
        Initialize DreamCoder.

        Args:
            grammar: Initial grammar (primitives)
            tasks: Tasks to learn
            eval_fn: How to evaluate programs on inputs
            max_iterations: Maximum wake-sleep iterations
            enumeration_timeout: Timeout per task (seconds)
            enumeration_budget: Max programs per task
            max_depth: Max AST depth
            compress_every: How often to run compression
            verbose: Print progress
        """
        self.initial_grammar = grammar
        self.grammar = copy.deepcopy(grammar)
        self.tasks = tasks
        self.eval_fn = eval_fn
        self.max_iterations = max_iterations
        self.enumeration_timeout = enumeration_timeout
        self.enumeration_budget = enumeration_budget
        self.max_depth = max_depth
        self.compress_every = compress_every
        self.verbose = verbose

        # Tracking
        self.frontiers: Dict[str, Frontier] = {}
        self.task_metrics: Dict[str, LearningMetrics] = {}
        self.iterations: List[IterationResult] = []
        self.library_growth: List[int] = [len(grammar)]

    def run(self) -> DreamCoderResult:
        """
        Run the full wake-sleep learning loop.

        Returns:
            DreamCoderResult with all metrics
        """
        start_time = time.time()

        # Initialize metrics
        for task in self.tasks:
            self.task_metrics[task.name] = LearningMetrics(
                task_name=task.name,
                solved=False,
                iteration_solved=None,
                programs_enumerated=0,
                time_to_solve=0.0,
                solution=None,
                solution_size=0,
                description_length=float('inf')
            )

        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"ITERATION {iteration + 1}/{self.max_iterations}")
                print(f"{'='*60}")
                print(f"Grammar: {len(self.grammar)} productions")

            iter_result = self._run_iteration(iteration)
            self.iterations.append(iter_result)
            self.library_growth.append(len(self.grammar))

            if self.verbose:
                print(f"\nSolved: {iter_result.tasks_solved}/{iter_result.total_tasks}")
                print(f"New abstractions: {len(iter_result.new_inventions)}")
                for inv in iter_result.new_inventions:
                    print(f"  {inv}")

            # Early stopping if all tasks solved
            if iter_result.tasks_solved == iter_result.total_tasks:
                if self.verbose:
                    print("\nAll tasks solved! Stopping early.")
                break

        return DreamCoderResult(
            initial_grammar=self.initial_grammar,
            final_grammar=self.grammar,
            iterations=self.iterations,
            task_metrics=self.task_metrics,
            library_growth=self.library_growth,
            total_time=time.time() - start_time
        )

    def _run_iteration(self, iteration: int) -> IterationResult:
        """Run one wake-sleep iteration."""
        iter_start = time.time()
        tasks_solved = 0
        total_programs = 0
        new_frontiers = {}

        # WAKE PHASE: Enumerate programs for unsolved tasks
        for task in self.tasks:
            if self.verbose:
                print(f"\n  [{task.name}]", end=" ", flush=True)

            # Skip if already solved with a perfect solution
            if task.name in self.frontiers and not self.frontiers[task.name].empty:
                best = self.frontiers[task.name].best
                if best and best.log_likelihood == 0.0:
                    tasks_solved += 1
                    new_frontiers[task.name] = self.frontiers[task.name]
                    if self.verbose:
                        print("(already solved)")
                    continue

            # Enumerate
            frontier = self._enumerate_task(task)
            new_frontiers[task.name] = frontier

            if not frontier.empty:
                tasks_solved += 1
                best = frontier.best
                if self.verbose:
                    print(f"SOLVED: {best.program} ({best.programs_enumerated} programs)")

                # Update metrics
                metrics = self.task_metrics[task.name]
                if not metrics.solved:
                    metrics.solved = True
                    metrics.iteration_solved = iteration
                    metrics.solution = best.program
                    metrics.solution_size = best.program.size()
                    metrics.description_length = best.description_length
                    metrics.time_to_solve = best.time_seconds
                metrics.programs_enumerated += best.programs_enumerated
            else:
                if self.verbose:
                    print("(unsolved)")
                metrics = self.task_metrics[task.name]
                metrics.programs_enumerated += self.enumeration_budget

            total_programs += frontier.best.programs_enumerated if frontier.best else self.enumeration_budget

        self.frontiers = new_frontiers

        # SLEEP PHASE: Compression
        new_inventions = []
        if iteration % self.compress_every == 0:
            new_inventions = self._compress()

        return IterationResult(
            iteration=iteration,
            tasks_solved=tasks_solved,
            total_tasks=len(self.tasks),
            new_inventions=new_inventions,
            programs_enumerated=total_programs,
            time_seconds=time.time() - iter_start,
            frontiers=new_frontiers
        )

    def _enumerate_task(self, task: Task) -> Frontier:
        """Enumerate programs for a single task."""
        frontier = Frontier(task_name=task.name, request_type=task.request_type)

        start_time = time.time()
        programs_tried = 0

        # Use TopDownEnumerator (replaces deprecated enumerate_simple)
        enumerator = TopDownEnumerator(
            self.grammar,
            max_depth=self.max_depth,
            max_programs=self.enumeration_budget
        )

        for program, log_prob in enumerator.enumerate(
            task.request_type,
            timeout_seconds=self.enumeration_timeout
        ):
            programs_tried += 1

            if programs_tried > self.enumeration_budget:
                break
            if time.time() - start_time > self.enumeration_timeout:
                break

            # Evaluate on examples
            try:
                correct = 0
                for inp, expected in task.examples:
                    result = self.eval_fn(program, inp)
                    if result == expected:
                        correct += 1

                if correct == len(task.examples):
                    # Perfect solution!
                    result = EnumerationResult(
                        program=program,
                        log_probability=log_prob,
                        log_likelihood=0.0,
                        description_length=-log_prob / math.log(2),
                        programs_enumerated=programs_tried,
                        time_seconds=time.time() - start_time
                    )
                    frontier.add(result)
                    break  # Found a solution, stop

            except Exception as e:
                # Program crashed - skip
                pass

        return frontier

    def _compress(self) -> List[Invented]:
        """Run compression to find new abstractions."""
        # Collect programs from frontiers
        all_frontiers = []
        for name, frontier in self.frontiers.items():
            if not frontier.empty:
                all_frontiers.append([
                    (e.program, e.log_likelihood)
                    for e in frontier.entries
                ])

        if not all_frontiers:
            return []

        # Compress
        result = compress_frontiers(
            self.grammar,
            all_frontiers,
            max_inventions=3,
            min_savings=2.0
        )

        if result.new_inventions:
            self.grammar = result.new_grammar

        return result.new_inventions


# ============================================================================
# EXPERIMENT HELPERS
# ============================================================================

def run_experiment(
    grammar: Grammar,
    tasks: List[Task],
    eval_fn: Callable[[Program, Any], Any],
    n_runs: int = 1,
    task_order: str = 'fixed',  # 'fixed', 'random', 'shuffled'
    **kwargs
) -> List[DreamCoderResult]:
    """
    Run multiple DreamCoder experiments.

    Args:
        grammar: Initial grammar
        tasks: Tasks to learn
        eval_fn: Evaluation function
        n_runs: Number of runs
        task_order: How to order tasks
        **kwargs: Passed to DreamCoder

    Returns:
        List of results from each run
    """
    import random

    results = []
    for run in range(n_runs):
        # Optionally shuffle tasks
        if task_order == 'random':
            run_tasks = list(tasks)
            random.shuffle(run_tasks)
        else:
            run_tasks = tasks

        dc = DreamCoder(grammar, run_tasks, eval_fn, **kwargs)
        result = dc.run()
        results.append(result)

    return results


def analyze_transfer(results: List[DreamCoderResult]) -> Dict:
    """
    Analyze transfer effects across runs.

    Returns statistics on how task order affects learning.
    """
    # For each task, collect when it was solved across runs
    task_iterations = defaultdict(list)

    for result in results:
        for name, metrics in result.task_metrics.items():
            if metrics.solved:
                task_iterations[name].append(metrics.iteration_solved)
            else:
                task_iterations[name].append(float('inf'))

    # Compute statistics
    analysis = {}
    for task, iterations in task_iterations.items():
        finite = [i for i in iterations if i != float('inf')]
        analysis[task] = {
            'always_solved': len(finite) == len(iterations),
            'solve_rate': len(finite) / len(iterations),
            'avg_iteration': sum(finite) / len(finite) if finite else float('inf'),
            'min_iteration': min(iterations),
            'max_iteration': max(finite) if finite else float('inf')
        }

    return analysis


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("=== Wake-Sleep Tests ===\n")

    # Create simple primitives for arithmetic
    add = Primitive('+', arrow(INT, INT, INT), lambda x: lambda y: x + y)
    mul = Primitive('*', arrow(INT, INT, INT), lambda x: lambda y: x * y)
    zero = Primitive('0', INT, 0)
    one = Primitive('1', INT, 1)
    two = Primitive('2', INT, 2)

    g = uniform_grammar([add, mul, zero, one, two])

    # Create tasks
    # Task 1: double the input (x + x)
    task1 = Task(
        name="double",
        request_type=arrow(INT, INT),
        examples=[(1, 2), (2, 4), (3, 6)]
    )

    # Task 2: square the input (x * x)
    task2 = Task(
        name="square",
        request_type=arrow(INT, INT),
        examples=[(1, 1), (2, 4), (3, 9)]
    )

    # Task 3: add one
    task3 = Task(
        name="add_one",
        request_type=arrow(INT, INT),
        examples=[(0, 1), (1, 2), (2, 3)]
    )

    def eval_fn(program: Program, inp: Any) -> Any:
        fn = program.evaluate([])
        return fn(inp)

    # Run DreamCoder
    dc = DreamCoder(
        grammar=g,
        tasks=[task1, task2, task3],
        eval_fn=eval_fn,
        max_iterations=3,
        enumeration_timeout=10.0,
        enumeration_budget=1000,
        verbose=True
    )

    result = dc.run()
    print(result.summary())

    print("\n=== Wake-Sleep Tests OK ===")
