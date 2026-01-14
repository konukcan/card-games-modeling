#!/usr/bin/env python3
"""
Deep Enumeration Test - Track Programs to Solution

This script enumerates programs WITHOUT the full wake-sleep loop to measure
exactly how many programs need to be enumerated to solve each task.

Key features:
- Single iteration, no recognition model guidance (pure enumeration)
- Tracks exact program count when each solution is found
- Stops when half tasks solved OR time limit reached
- Checkpoint/resume support for long runs

Usage:
    # Fresh run (30 min default)
    python deep_enumeration_test.py

    # Resume from checkpoint
    python deep_enumeration_test.py --resume

    # Custom time limit (in minutes)
    python deep_enumeration_test.py --time-limit 60

    # Keep running even after hitting half (for deeper analysis)
    python deep_enumeration_test.py --no-early-stop
"""

import sys
import os
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dreamcoder_core.primitives import build_lean_grammar
from dreamcoder_core.enumeration import TopDownEnumerator
from dreamcoder_core.type_system import HAND, BOOL, arrow
from dreamcoder_core.program import Program
from rules.cards import sample_hand, Hand
from rules.catalogue import create_all_rules, Rule


@dataclass
class TaskResult:
    """Result for a single task."""
    task_name: str
    solved: bool
    programs_enumerated: int
    solution: Optional[str] = None
    solution_depth: Optional[int] = None
    time_to_solve: Optional[float] = None


@dataclass
class CheckpointState:
    """State that can be saved and restored."""
    timestamp: str
    total_programs_enumerated: int
    total_time_seconds: float
    tasks_solved: int
    tasks_total: int
    results: Dict[str, Dict]  # task_name -> TaskResult as dict
    current_task_idx: int
    current_task_programs: int  # Programs enumerated on current task before checkpoint


class DeepEnumerationTest:
    """
    Deep enumeration test with checkpointing.

    Enumerates programs for all tasks, tracking exactly when each is solved.
    """

    def __init__(
        self,
        time_limit_minutes: float = 30.0,
        stop_at_half: bool = True,
        max_programs_per_task: int = 500_000,
        max_depth: int = 8,
        n_examples: int = 20,
        checkpoint_interval: int = 10_000,  # Save every N programs
        results_dir: str = "results_deep_enum"
    ):
        self.time_limit_seconds = time_limit_minutes * 60
        self.stop_at_half = stop_at_half
        self.max_programs_per_task = max_programs_per_task
        self.max_depth = max_depth
        self.n_examples = n_examples
        self.checkpoint_interval = checkpoint_interval
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # State
        self.grammar = None
        self.enumerator = None
        self.tasks: List[Tuple[str, Rule, List[Tuple[Hand, bool]]]] = []
        self.results: Dict[str, TaskResult] = {}
        self.total_programs = 0
        self.start_time = 0.0
        self.current_task_idx = 0
        self.current_task_programs = 0

        # Checkpoint file
        self.checkpoint_file = self.results_dir / "checkpoint.json"

    def setup(self):
        """Initialize grammar and tasks."""
        print("Building grammar...")
        self.grammar = build_lean_grammar()
        print(f"  Grammar size: {len(list(self.grammar.primitives()))} primitives")

        print("\nCreating enumerator...")
        self.enumerator = TopDownEnumerator(
            grammar=self.grammar,
            max_programs=self.max_programs_per_task,
            max_depth=self.max_depth
        )

        print("\nCreating tasks...")
        all_rules = create_all_rules()
        print(f"  Total rules: {len(all_rules)}")

        # Create examples for each task with STRICT balanced sampling
        # Using protocol from run_experimental_rules.py (best practice)
        import random
        random.seed(42)

        skipped_rules = []
        min_per_class = 5  # Minimum examples needed per class
        max_attempts = 100000  # High attempt count for rare rules

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
                    pass  # Skip hands that cause errors
                attempts += 1

            # VALIDATE: Check if we got enough of each class
            is_valid = len(positives) >= min_per_class and len(negatives) >= min_per_class

            if not is_valid:
                print(f"  WARNING: {rule.id} - got {len(positives)} positive, {len(negatives)} negative (SKIPPING)")
                skipped_rules.append(rule.id)
                continue

            if len(positives) < target or len(negatives) < target:
                print(f"  WARNING: {rule.id} - got {len(positives)} positive, {len(negatives)} negative (using anyway)")

            # Combine ONLY what we have of each class (guaranteed balanced)
            examples = positives[:target] + negatives[:target]
            random.shuffle(examples)

            self.tasks.append((rule.id, rule, examples))

        print(f"  Created {len(self.tasks)} tasks with balanced examples")
        if skipped_rules:
            print(f"  Skipped {len(skipped_rules)} rules due to extreme class imbalance: {skipped_rules}")

    def save_checkpoint(self):
        """Save current state to checkpoint file."""
        state = CheckpointState(
            timestamp=datetime.now().isoformat(),
            total_programs_enumerated=self.total_programs,
            total_time_seconds=time.time() - self.start_time,
            tasks_solved=sum(1 for r in self.results.values() if r.solved),
            tasks_total=len(self.tasks),
            results={name: asdict(result) for name, result in self.results.items()},
            current_task_idx=self.current_task_idx,
            current_task_programs=self.current_task_programs
        )

        with open(self.checkpoint_file, 'w') as f:
            json.dump(asdict(state), f, indent=2, default=str)

    def load_checkpoint(self) -> bool:
        """Load state from checkpoint file. Returns True if loaded."""
        if not self.checkpoint_file.exists():
            print("No checkpoint found.")
            return False

        print(f"Loading checkpoint from {self.checkpoint_file}...")
        with open(self.checkpoint_file, 'r') as f:
            data = json.load(f)

        # Restore results
        for name, result_dict in data['results'].items():
            self.results[name] = TaskResult(**result_dict)

        self.total_programs = data['total_programs_enumerated']
        self.current_task_idx = data['current_task_idx']
        self.current_task_programs = data['current_task_programs']

        print(f"  Restored: {data['tasks_solved']}/{data['tasks_total']} solved")
        print(f"  Programs enumerated: {self.total_programs:,}")
        print(f"  Resuming from task {self.current_task_idx}")

        return True

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
        """
        Verify a solution on BALANCED holdout examples.

        This filters out spurious solutions that happen to fit the training examples
        but don't actually implement the rule correctly.

        CRITICAL: Uses balanced sampling (50% positive, 50% negative) to catch
        solutions like `(λ false)` that would pass random holdout for rare rules.
        """
        import random

        try:
            fn = program.evaluate([])

            # Generate BALANCED holdout examples
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

            # Require minimum examples of each class for valid verification
            min_per_class = 5
            if len(positives) < min_per_class or len(negatives) < min_per_class:
                # Can't verify properly - rule has extreme class imbalance
                # Be conservative: return False (don't accept as solution)
                return False

            # Test on balanced holdout
            holdout = positives + negatives
            for hand, expected in holdout:
                result = fn(hand)
                if result != expected:
                    return False
            return True
        except Exception:
            return False

    def run(self, resume: bool = False) -> Dict[str, Any]:
        """
        Run the deep enumeration test.

        Args:
            resume: If True, try to resume from checkpoint

        Returns:
            Dictionary with results and statistics
        """
        self.setup()

        if resume:
            self.load_checkpoint()
        else:
            # Initialize fresh results
            self.results = {}
            self.total_programs = 0
            self.current_task_idx = 0
            self.current_task_programs = 0

        self.start_time = time.time()
        half_target = len(self.tasks) // 2 + 1  # 23 for 45 tasks

        print("\n" + "=" * 70)
        print("DEEP ENUMERATION TEST")
        print("=" * 70)
        print(f"Time limit: {self.time_limit_seconds / 60:.1f} minutes")
        print(f"Stop at half: {self.stop_at_half} ({half_target} tasks)")
        print(f"Max programs per task: {self.max_programs_per_task:,}")
        print(f"Max depth: {self.max_depth}")
        print()

        request_type = arrow(HAND, BOOL)
        last_checkpoint_programs = self.total_programs

        # Process each task starting from current_task_idx
        for task_idx in range(self.current_task_idx, len(self.tasks)):
            task_name, rule, examples = self.tasks[task_idx]
            self.current_task_idx = task_idx

            # Check stop conditions
            elapsed = time.time() - self.start_time
            if elapsed > self.time_limit_seconds:
                print(f"\nTime limit reached ({elapsed/60:.1f} min)")
                break

            solved_count = sum(1 for r in self.results.values() if r.solved)
            if self.stop_at_half and solved_count >= half_target:
                print(f"\nReached half target ({solved_count} solved)")
                break

            # Skip if already solved (from checkpoint)
            if task_name in self.results and self.results[task_name].solved:
                print(f"  [{task_idx+1}/{len(self.tasks)}] {task_name}: Already solved (skipping)")
                continue

            print(f"  [{task_idx+1}/{len(self.tasks)}] {task_name}:", end=" ", flush=True)

            task_start = time.time()
            task_programs = self.current_task_programs  # Resume from checkpoint
            self.current_task_programs = 0  # Reset for next task

            solved = False
            solution = None
            solution_depth = None

            # Enumerate programs for this task
            for program, log_prob in self.enumerator.enumerate_memoized(
                request_type,
                max_cost=100.0,  # High cost to explore deeply
                timeout_seconds=self.time_limit_seconds - elapsed,
                depth_limit=self.max_depth
            ):
                task_programs += 1
                self.total_programs += 1

                # Check if solution (fits training examples)
                if self.evaluate_program(program, examples):
                    # HOLDOUT VERIFICATION: Verify on fresh examples
                    if self.verify_on_holdout(program, rule, n_holdout=100):
                        solved = True
                        solution = str(program)
                        solution_depth = program.depth()
                        break
                    # else: spurious solution, keep searching

                # Budget check
                if task_programs >= self.max_programs_per_task:
                    break

                # Periodic checkpoint
                if self.total_programs - last_checkpoint_programs >= self.checkpoint_interval:
                    self.current_task_programs = task_programs
                    self.save_checkpoint()
                    last_checkpoint_programs = self.total_programs

            task_time = time.time() - task_start

            # Record result
            self.results[task_name] = TaskResult(
                task_name=task_name,
                solved=solved,
                programs_enumerated=task_programs,
                solution=solution,
                solution_depth=solution_depth,
                time_to_solve=task_time if solved else None
            )

            if solved:
                print(f"SOLVED at {task_programs:,} programs (depth {solution_depth})")
            else:
                print(f"Unsolved after {task_programs:,} programs")

            # Save checkpoint after each task
            self.save_checkpoint()

        # Final save
        self.save_checkpoint()

        return self._compile_results()

    def _compile_results(self) -> Dict[str, Any]:
        """Compile final results."""
        elapsed = time.time() - self.start_time

        solved_tasks = [r for r in self.results.values() if r.solved]
        unsolved_tasks = [r for r in self.results.values() if not r.solved]

        # Statistics for solved tasks
        if solved_tasks:
            programs_to_solve = [r.programs_enumerated for r in solved_tasks]
            avg_programs = sum(programs_to_solve) / len(programs_to_solve)
            median_programs = sorted(programs_to_solve)[len(programs_to_solve) // 2]
            max_programs = max(programs_to_solve)
            min_programs = min(programs_to_solve)
        else:
            avg_programs = median_programs = max_programs = min_programs = 0

        # Sort solved by programs required
        solved_by_difficulty = sorted(solved_tasks, key=lambda r: r.programs_enumerated)

        results = {
            'summary': {
                'total_time_seconds': elapsed,
                'total_time_minutes': elapsed / 60,
                'total_programs_enumerated': self.total_programs,
                'tasks_solved': len(solved_tasks),
                'tasks_total': len(self.results),
                'tasks_remaining': len(self.tasks) - len(self.results),
            },
            'solve_statistics': {
                'avg_programs_to_solve': avg_programs,
                'median_programs_to_solve': median_programs,
                'min_programs_to_solve': min_programs,
                'max_programs_to_solve': max_programs,
            },
            'solved_tasks': [
                {
                    'name': r.task_name,
                    'programs': r.programs_enumerated,
                    'depth': r.solution_depth,
                    'solution': r.solution,
                    'time': r.time_to_solve
                }
                for r in solved_by_difficulty
            ],
            'unsolved_tasks': [
                {
                    'name': r.task_name,
                    'programs_tried': r.programs_enumerated
                }
                for r in unsolved_tasks
            ],
            'estimation': self._estimate_unsolved(solved_tasks, unsolved_tasks)
        }

        # Save final results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.results_dir / f"deep_enum_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Total programs: {self.total_programs:,}")
        print(f"Tasks solved: {len(solved_tasks)}/{len(self.results)}")
        print()

        if solved_tasks:
            print("Solved tasks (by difficulty):")
            for r in solved_by_difficulty[:10]:
                print(f"  {r.task_name}: {r.programs_enumerated:,} programs (depth {r.solution_depth})")
            if len(solved_by_difficulty) > 10:
                print(f"  ... and {len(solved_by_difficulty) - 10} more")

        print(f"\nResults saved to: {results_file}")

        return results

    def _estimate_unsolved(
        self,
        solved: List[TaskResult],
        unsolved: List[TaskResult]
    ) -> Dict[str, Any]:
        """
        Estimate programs needed for unsolved tasks based on solved patterns.

        This uses the distribution of programs-to-solve for solved tasks
        to estimate what it might take for unsolved tasks.
        """
        if not solved:
            return {'note': 'No solved tasks to base estimation on'}

        programs_to_solve = sorted([r.programs_enumerated for r in solved])

        # Compute percentiles
        def percentile(data, p):
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            return data[f] + (data[c] - data[f]) * (k - f)

        p50 = percentile(programs_to_solve, 50)
        p75 = percentile(programs_to_solve, 75)
        p90 = percentile(programs_to_solve, 90)
        p99 = percentile(programs_to_solve, 99) if len(programs_to_solve) >= 10 else max(programs_to_solve)

        # For each unsolved task, estimate based on how many programs we've already tried
        estimates = []
        for r in unsolved:
            tried = r.programs_enumerated
            # If we've already tried more than p99, it's likely very hard
            if tried > p99:
                estimate = "Very hard (>p99 already tried)"
            elif tried > p90:
                estimate = f"Hard (p90-p99 range), try {int(p99 - tried):,} more"
            elif tried > p75:
                estimate = f"Medium-hard, try {int(p90 - tried):,} more"
            elif tried > p50:
                estimate = f"Medium, try {int(p75 - tried):,} more"
            else:
                estimate = f"Possibly solvable soon, try {int(p50 - tried):,} more"

            estimates.append({
                'task': r.task_name,
                'tried': tried,
                'estimate': estimate
            })

        return {
            'percentiles': {
                'p50': int(p50),
                'p75': int(p75),
                'p90': int(p90),
                'p99': int(p99)
            },
            'interpretation': (
                f"Based on {len(solved)} solved tasks: "
                f"50% solved by {int(p50):,} programs, "
                f"90% by {int(p90):,} programs"
            ),
            'task_estimates': estimates
        }


def main():
    parser = argparse.ArgumentParser(description="Deep enumeration test")
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--time-limit', type=float, default=30.0, help='Time limit in minutes')
    parser.add_argument('--no-early-stop', action='store_true', help='Continue past half solved')
    parser.add_argument('--max-depth', type=int, default=8, help='Max program depth')
    parser.add_argument('--max-programs', type=int, default=500_000, help='Max programs per task')

    args = parser.parse_args()

    test = DeepEnumerationTest(
        time_limit_minutes=args.time_limit,
        stop_at_half=not args.no_early_stop,
        max_depth=args.max_depth,
        max_programs_per_task=args.max_programs
    )

    results = test.run(resume=args.resume)

    # Print estimation for next 20 unsolved
    if 'estimation' in results and 'task_estimates' in results['estimation']:
        print("\n" + "=" * 70)
        print("ESTIMATION FOR UNSOLVED TASKS")
        print("=" * 70)
        for est in results['estimation']['task_estimates'][:20]:
            print(f"  {est['task']}: {est['estimate']}")

    return results


if __name__ == "__main__":
    main()
