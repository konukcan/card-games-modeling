#!/usr/bin/env python3
"""
Parallel Primitive Experiment

Tests the hypothesis that low accessibility of parallel comparison structures
is the main barrier to solving rules.

This script:
1. Adds ad-hoc "parallel combinator" primitives to the grammar
2. Runs enumeration with the same 5M program budget
3. Compares results to the baseline (no parallel primitives)

Primitives added:
- halves_equal_by: λf. λhand. eq (f (first_half hand)) (f (second_half hand))
- ends_equal_by: λf. λlist. eq (f (head list)) (f (last list))
- is_palindrome_by: λf. λlist. eq (map f list) (reverse (map f list))
"""

import sys
import os
import time
import json
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dreamcoder_core.lean_primitives import build_lean_grammar, build_lean_primitives
from dreamcoder_core.grammar import Grammar, Production, uniform_grammar
from dreamcoder_core.program import Primitive, Program
from dreamcoder_core.enumeration import TopDownEnumerator
from dreamcoder_core.type_system import (
    Type, BaseType, Arrow, ListType, TypeVariable,
    BOOL, INT, CARD, SUIT, RANK, HAND, arrow
)
from rules.cards import (
    Card, Hand, Suit, Rank, Color, RANK_VALUES, card_color, sample_hand
)
from rules.catalogue import create_all_rules, Rule

# Type definitions
COLOR = BaseType('color')
LIST_SUIT = ListType(SUIT)
LIST_RANK = ListType(RANK)
LIST_COLOR = ListType(COLOR)


def make_parallel_primitives() -> List[Primitive]:
    """
    Create parallel comparison primitives that make structural patterns
    like `eq (f left) (f right)` directly accessible.

    These should dramatically reduce the search space for rules that
    compare properties of two halves or two ends.
    """
    prims = []

    a = TypeVariable(0)  # Generic type for function result

    # halves_equal_by: Check if applying f to both halves gives same result
    # Type: (Hand → a) → Hand → Bool
    # Usage: halves_equal_by all_same_color hand
    # Equivalent to: eq (all_same_color (first_half hand)) (all_same_color (second_half hand))
    def halves_equal_by_impl(f):
        def apply_to_hand(hand):
            first_h = hand[:len(hand)//2]
            second_h = hand[len(hand)//2:]
            return f(first_h) == f(second_h)
        return apply_to_hand

    prims.append(Primitive(
        'halves_equal_by',
        arrow(arrow(HAND, a), HAND, BOOL),
        halves_equal_by_impl
    ))

    # ends_equal_by: Check if applying f to first and last element gives same result
    # Type: (Card → a) → Hand → Bool
    # Usage: ends_equal_by get_suit hand
    # Equivalent to: eq (get_suit (head hand)) (get_suit (last hand))
    def ends_equal_by_impl(f):
        def apply_to_hand(hand):
            if len(hand) < 2:
                return True
            return f(hand[0]) == f(hand[-1])
        return apply_to_hand

    prims.append(Primitive(
        'ends_equal_by',
        arrow(arrow(CARD, a), HAND, BOOL),
        ends_equal_by_impl
    ))

    # is_palindrome_by: Check if list is a palindrome when transformed by f
    # Type: (Card → a) → Hand → Bool
    # Usage: is_palindrome_by get_suit hand
    # Equivalent to: eq (map get_suit hand) (reverse (map get_suit hand))
    def is_palindrome_by_impl(f):
        def apply_to_hand(hand):
            transformed = [f(c) for c in hand]
            return transformed == list(reversed(transformed))
        return apply_to_hand

    prims.append(Primitive(
        'is_palindrome_by',
        arrow(arrow(CARD, a), HAND, BOOL),
        is_palindrome_by_impl
    ))

    # all_adjacent_satisfy: Check if all adjacent pairs satisfy a relation
    # Type: (Card → Card → Bool) → Hand → Bool
    # Usage: all_adjacent_satisfy (λa b. le (rank_val a) (rank_val b)) hand
    # This helps with sorted checks
    def all_adjacent_satisfy_impl(pred):
        def apply_to_hand(hand):
            if len(hand) < 2:
                return True
            for i in range(len(hand) - 1):
                if not pred(hand[i])(hand[i+1]):
                    return False
            return True
        return apply_to_hand

    prims.append(Primitive(
        'all_adjacent_satisfy',
        arrow(arrow(CARD, CARD, BOOL), HAND, BOOL),
        all_adjacent_satisfy_impl
    ))

    return prims


def build_extended_grammar() -> Grammar:
    """Build grammar with parallel comparison primitives added."""
    # Start with base primitives
    prims = build_lean_primitives()

    # Add parallel comparison primitives
    parallel_prims = make_parallel_primitives()
    prims.extend(parallel_prims)

    print(f"  Base primitives: {len(prims) - len(parallel_prims)}")
    print(f"  Parallel primitives added: {len(parallel_prims)}")
    print(f"  New primitives: {[p.name for p in parallel_prims]}")

    return uniform_grammar(prims)


@dataclass
class TaskResult:
    """Result for a single task."""
    task_name: str
    solved: bool
    programs_enumerated: int
    solution: Optional[str] = None
    solution_depth: Optional[int] = None
    time_to_solve: Optional[float] = None


class ParallelPrimitiveExperiment:
    """
    Run enumeration experiment with parallel comparison primitives.
    """

    def __init__(
        self,
        time_limit_minutes: float = 120.0,
        max_programs_per_task: int = 5_000_000,
        max_depth: int = 8,
        n_examples: int = 20,
        results_dir: str = "results_parallel_exp"
    ):
        self.time_limit_seconds = time_limit_minutes * 60
        self.max_programs_per_task = max_programs_per_task
        self.max_depth = max_depth
        self.n_examples = n_examples
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        self.grammar = None
        self.enumerator = None
        self.tasks = []
        self.results = {}
        self.total_programs = 0
        self.start_time = 0.0

    def setup(self):
        """Initialize grammar and tasks."""
        print("Building extended grammar (with parallel primitives)...", flush=True)
        self.grammar = build_extended_grammar()
        print(f"  Total grammar size: {len(list(self.grammar.primitives()))} primitives")

        print("\nCreating enumerator...")
        self.enumerator = TopDownEnumerator(
            grammar=self.grammar,
            max_programs=self.max_programs_per_task,
            max_depth=self.max_depth
        )

        print("\nCreating tasks...")
        all_rules = create_all_rules()
        print(f"  Total rules: {len(all_rules)}")

        # Create examples with balanced sampling
        random.seed(42)

        min_per_class = 5
        max_attempts = 100000

        for rule in all_rules:
            positives = []
            negatives = []
            target = self.n_examples // 2
            attempts = 0

            while (len(positives) < target or len(negatives) < target) and attempts < max_attempts:
                hand = sample_hand(6)
                try:
                    result = rule.predicate(hand)
                    if result and len(positives) < target:
                        positives.append((hand, True))
                    elif not result and len(negatives) < target:
                        negatives.append((hand, False))
                except Exception:
                    pass
                attempts += 1

            is_valid = len(positives) >= min_per_class and len(negatives) >= min_per_class

            if not is_valid:
                print(f"  WARNING: {rule.id} - got {len(positives)} positive, {len(negatives)} negative (SKIPPING)")
                continue

            examples = positives[:target] + negatives[:target]
            random.shuffle(examples)
            self.tasks.append((rule.id, rule, examples))

        print(f"  Created {len(self.tasks)} tasks with {self.n_examples} examples each")

    def evaluate_program(self, program: Program, examples: List[Tuple[Hand, bool]]) -> bool:
        """Check if program correctly classifies all examples."""
        try:
            fn = program.evaluate([])
            for hand, expected in examples:
                result = fn(hand)
                if result != expected:
                    return False
            return True
        except Exception:
            return False

    def verify_on_holdout(self, program: Program, rule: Rule, n_holdout: int = 100) -> bool:
        """Verify a solution on balanced holdout examples."""
        try:
            fn = program.evaluate([])

            target_per_class = n_holdout // 2
            positives = []
            negatives = []
            max_attempts = 50000
            attempts = 0

            while (len(positives) < target_per_class or len(negatives) < target_per_class) and attempts < max_attempts:
                hand = sample_hand(6)
                try:
                    expected = rule.predicate(hand)
                    if expected and len(positives) < target_per_class:
                        positives.append((hand, True))
                    elif not expected and len(negatives) < target_per_class:
                        negatives.append((hand, False))
                except Exception:
                    pass
                attempts += 1

            min_per_class = 5
            if len(positives) < min_per_class or len(negatives) < min_per_class:
                return False

            holdout = positives + negatives
            for hand, expected in holdout:
                result = fn(hand)
                if result != expected:
                    return False
            return True
        except Exception:
            return False

    def _save_incremental_results(self):
        """Save results after each task (FIX 3: incremental saves)."""
        elapsed = time.time() - self.start_time
        solved_tasks = [name for name, r in self.results.items() if r.solved]

        output = {
            "status": "in_progress",
            "elapsed_seconds": elapsed,
            "elapsed_minutes": elapsed / 60,
            "total_programs_enumerated": self.total_programs,
            "tasks_completed": len(self.results),
            "tasks_total": len(self.tasks),
            "tasks_solved": len(solved_tasks),
            "parallel_primitives_added": [
                "halves_equal_by",
                "ends_equal_by",
                "is_palindrome_by",
                "all_adjacent_satisfy"
            ],
            "results": {
                name: {
                    "solved": r.solved,
                    "programs": r.programs_enumerated,
                    "depth": r.solution_depth,
                    "solution": r.solution,
                    "time": r.time_to_solve
                }
                for name, r in self.results.items()
            }
        }

        incremental_file = self.results_dir / "incremental_results.json"
        with open(incremental_file, 'w') as f:
            json.dump(output, f, indent=2)

    def run(self) -> Dict[str, Any]:
        """Run the experiment."""
        self.setup()

        self.results = {}
        self.total_programs = 0
        self.start_time = time.time()

        target_type = arrow(HAND, BOOL)

        print("\n" + "=" * 70, flush=True)
        print("PARALLEL PRIMITIVE EXPERIMENT", flush=True)
        print("=" * 70, flush=True)
        print(f"Time limit: {self.time_limit_seconds / 60:.1f} minutes", flush=True)
        print(f"Max programs per task: {self.max_programs_per_task:,}", flush=True)
        print(f"Max depth: {self.max_depth}", flush=True)
        print(flush=True)

        for task_idx, (task_name, rule, examples) in enumerate(self.tasks):
            elapsed = time.time() - self.start_time
            if elapsed > self.time_limit_seconds:
                print(f"\nTime limit reached ({elapsed/60:.1f} min)")
                break

            task_start = time.time()
            solved = False
            solution_str = None
            solution_depth = None
            programs_checked = 0

            # Enumerate programs for this task (FIX 4: use memoized enumeration for speed)
            timed_out = False
            for program, cost in self.enumerator.enumerate_memoized(target_type):
                programs_checked += 1
                self.total_programs += 1

                # FIX 1: Check timeout every 1000 programs
                if programs_checked % 1000 == 0:
                    elapsed = time.time() - self.start_time
                    if elapsed > self.time_limit_seconds:
                        timed_out = True
                        break

                if self.evaluate_program(program, examples):
                    # Verify on holdout
                    if self.verify_on_holdout(program, rule):
                        solved = True
                        solution_str = str(program)
                        solution_depth = program.depth()
                        break

                if programs_checked >= self.max_programs_per_task:
                    break

            task_time = time.time() - task_start

            self.results[task_name] = TaskResult(
                task_name=task_name,
                solved=solved,
                programs_enumerated=programs_checked,
                solution=solution_str,
                solution_depth=solution_depth,
                time_to_solve=task_time if solved else None
            )

            status = f"SOLVED at {programs_checked:,} programs (depth {solution_depth})" if solved else f"Unsolved after {programs_checked:,} programs"
            # FIX 2: Flush stdout immediately
            print(f"  [{task_idx+1}/{len(self.tasks)}] {task_name}: {status}", flush=True)

            # FIX 3: Save incremental results after each task
            self._save_incremental_results()

            # Check if we timed out during enumeration
            if timed_out:
                print(f"\nTime limit reached during task {task_name} ({time.time() - self.start_time:.1f}s)", flush=True)
                break

        # Generate summary
        total_time = time.time() - self.start_time
        solved_tasks = [name for name, r in self.results.items() if r.solved]

        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Total programs: {self.total_programs:,}")
        print(f"Tasks solved: {len(solved_tasks)}/{len(self.tasks)}")

        print("\nSolved tasks (by difficulty):")
        solved_results = [(name, self.results[name]) for name in solved_tasks]
        solved_results.sort(key=lambda x: x[1].programs_enumerated)
        for name, result in solved_results:
            print(f"  {name}: {result.programs_enumerated:,} programs (depth {result.solution_depth})")
            print(f"    Solution: {result.solution}")

        # Save results
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        results_file = self.results_dir / f"parallel_exp_results_{timestamp}.json"

        output = {
            "summary": {
                "total_time_seconds": total_time,
                "total_time_minutes": total_time / 60,
                "total_programs_enumerated": self.total_programs,
                "tasks_solved": len(solved_tasks),
                "tasks_total": len(self.tasks),
            },
            "parallel_primitives_added": [
                "halves_equal_by",
                "ends_equal_by",
                "is_palindrome_by",
                "all_adjacent_satisfy"
            ],
            "solved_tasks": [
                {
                    "name": name,
                    "programs": r.programs_enumerated,
                    "depth": r.solution_depth,
                    "solution": r.solution,
                    "time": r.time_to_solve
                }
                for name, r in sorted(solved_results, key=lambda x: x[1].programs_enumerated)
            ],
            "unsolved_tasks": [
                {
                    "name": name,
                    "programs_tried": r.programs_enumerated
                }
                for name, r in self.results.items() if not r.solved
            ]
        }

        with open(results_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"\nResults saved to: {results_file}")

        return output


def main():
    """Run the parallel primitive experiment."""
    import argparse

    parser = argparse.ArgumentParser(description="Parallel Primitive Experiment")
    parser.add_argument("--time-limit", type=float, default=120.0,
                       help="Time limit in minutes (default: 120)")
    parser.add_argument("--max-programs", type=int, default=5_000_000,
                       help="Max programs per task (default: 5,000,000)")
    parser.add_argument("--max-depth", type=int, default=8,
                       help="Max program depth (default: 8)")

    args = parser.parse_args()

    experiment = ParallelPrimitiveExperiment(
        time_limit_minutes=args.time_limit,
        max_programs_per_task=args.max_programs,
        max_depth=args.max_depth
    )

    results = experiment.run()

    # Print comparison hint
    print("\n" + "=" * 70)
    print("COMPARISON WITH BASELINE")
    print("=" * 70)
    print("Baseline (no parallel primitives): 8/45 solved in 5M programs")
    print(f"With parallel primitives: {results['summary']['tasks_solved']}/45 solved")
    print("\nHypothesis: If parallel structure accessibility was the barrier,")
    print("we should see significantly more tasks solved with the new primitives.")


if __name__ == "__main__":
    main()
