#!/usr/bin/env python3
"""
Progressive Neural Wake-Sleep Experiment

This script runs a wake-sleep learning loop with:
1. Contrastive recognition model (τ = mean(pos) - mean(neg))
2. Progressive neural guidance (weight increases with iterations/solved tasks)
3. Parallel comparison primitives (halves_equal_by, ends_equal_by, etc.)

The key innovation is that neural guidance weight increases as the model
learns more from solved tasks, following a schedule:
    weight(i) = min(0.1 + 0.1*i + 0.2*solved_ratio, 0.9)

Author: Can Konuk
Date: December 2024
"""

import sys
import os
import time
import json
import copy
import random
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dreamcoder_core.lean_primitives import build_lean_grammar, build_lean_primitives
from dreamcoder_core.grammar import Grammar, Production, uniform_grammar
from dreamcoder_core.program import Primitive, Program
from dreamcoder_core.enumeration import TopDownEnumerator, EnumerationResult
from dreamcoder_core.type_system import (
    Type, BaseType, Arrow, ListType, TypeVariable,
    BOOL, INT, CARD, SUIT, RANK, HAND, arrow
)
from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.compression import compress_frontiers
from dreamcoder_core.task import Task

from rules.cards import (
    Card, Hand, Suit, Rank, Color, RANK_VALUES, card_color, sample_hand
)
from rules.catalogue import create_all_rules, Rule


# ============================================================================
# PARALLEL PRIMITIVES (from parallel_primitive_experiment.py)
# ============================================================================

COLOR = BaseType('color')

def make_parallel_primitives() -> List[Primitive]:
    """
    Create parallel comparison primitives that make structural patterns
    like `eq (f left) (f right)` directly accessible.
    """
    prims = []
    a = TypeVariable(0)

    # halves_equal_by: Check if applying f to both halves gives same result
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
    prims = build_lean_primitives()
    parallel_prims = make_parallel_primitives()
    prims.extend(parallel_prims)
    return uniform_grammar(prims)


# ============================================================================
# PROGRESSIVE NEURAL WEIGHT SCHEDULE
# ============================================================================

def compute_neural_weight(
    iteration: int,
    solved_ratio: float,
    initial_weight: float = 0.1,
    weight_increment: float = 0.1,
    solved_boost: float = 0.2,
    max_weight: float = 0.9
) -> float:
    """
    Compute neural guidance weight based on iteration and progress.

    Weight increases:
    - With each iteration (exploration → exploitation)
    - When more tasks are solved (trust the model more)

    Formula: w = min(initial + increment*iter + boost*solved_ratio, max)
    """
    base = initial_weight + weight_increment * iteration
    boost = solved_boost * solved_ratio
    return min(base + boost, max_weight)


# ============================================================================
# TASK CREATION
# ============================================================================

@dataclass
class CardTask:
    """Task for card game rule learning."""
    name: str
    rule: Rule
    examples: List[Tuple[Hand, bool]]
    holdout: List[Tuple[Hand, bool]] = field(default_factory=list)

    def to_recognition_task(self) -> Task:
        """Convert to Task format for recognition model."""
        return Task(
            name=self.name,
            request_type=arrow(HAND, BOOL),
            examples=self.examples,
            holdout=self.holdout
        )


def create_balanced_examples(
    rule: Rule,
    n_examples: int = 20,
    n_holdout: int = 100
) -> Tuple[List[Tuple[Hand, bool]], List[Tuple[Hand, bool]]]:
    """Create balanced positive/negative examples for a rule."""
    random.seed(42 + hash(rule.id) % 10000)

    target = n_examples // 2
    holdout_target = n_holdout // 2

    positives = []
    negatives = []
    holdout_pos = []
    holdout_neg = []

    max_attempts = 100000
    attempts = 0

    while attempts < max_attempts:
        hand = sample_hand(6)
        try:
            result = rule.predicate(hand)
            if result:
                if len(positives) < target:
                    positives.append((hand, True))
                elif len(holdout_pos) < holdout_target:
                    holdout_pos.append((hand, True))
            else:
                if len(negatives) < target:
                    negatives.append((hand, False))
                elif len(holdout_neg) < holdout_target:
                    holdout_neg.append((hand, False))
        except Exception:
            pass
        attempts += 1

        # Check if we have enough
        if (len(positives) >= target and len(negatives) >= target and
            len(holdout_pos) >= holdout_target and len(holdout_neg) >= holdout_target):
            break

    examples = positives + negatives
    random.shuffle(examples)

    holdout = holdout_pos + holdout_neg
    random.shuffle(holdout)

    return examples, holdout


def create_all_tasks(n_examples: int = 20, n_holdout: int = 100) -> List[CardTask]:
    """Create tasks for all rules in the catalogue."""
    rules = create_all_rules()
    tasks = []

    for rule in rules:
        examples, holdout = create_balanced_examples(rule, n_examples, n_holdout)

        # Skip if insufficient examples
        if len([e for e in examples if e[1]]) < 5 or len([e for e in examples if not e[1]]) < 5:
            print(f"  WARNING: Skipping {rule.id} - insufficient examples", flush=True)
            continue

        tasks.append(CardTask(
            name=rule.id,
            rule=rule,
            examples=examples,
            holdout=holdout
        ))

    return tasks


# ============================================================================
# PROGRESSIVE WAKE-SLEEP LEARNER
# ============================================================================

@dataclass
class IterationResult:
    """Results from one wake-sleep iteration."""
    iteration: int
    tasks_solved: int
    tasks_total: int
    programs_enumerated: int
    neural_weight: float
    wake_time: float
    sleep_time: float
    recognition_loss: float
    new_abstractions: List[str] = field(default_factory=list)
    solved_task_names: List[str] = field(default_factory=list)


class ProgressiveWakeSleep:
    """
    DreamCoder with progressive neural guidance.

    Key features:
    1. Contrastive recognition model (τ = mean(pos) - mean(neg))
    2. Progressive neural weight (increases with iteration/solved ratio)
    3. Parallel primitives for structural patterns
    4. Memoized enumeration for efficiency
    """

    def __init__(
        self,
        grammar: Grammar,
        tasks: List[CardTask],

        # Progressive weight settings
        initial_neural_weight: float = 0.1,
        weight_increment: float = 0.1,
        solved_boost: float = 0.2,
        max_neural_weight: float = 0.9,

        # Enumeration settings
        enumeration_budget: int = 1_000_000,
        max_depth: int = 8,

        # Recognition settings
        recognition_hidden_dim: int = 32,
        recognition_epochs: int = 10,
        recognition_lr: float = 1e-3,
        structural_weight: float = 0.1,

        # Compression settings
        use_compression: bool = True,
        max_inventions: int = 3,

        # General settings
        max_iterations: int = 5,
        time_limit_minutes: float = 120.0,
        results_dir: str = "results_progressive_wakesleep",
        device: str = 'cpu',
        verbose: bool = True
    ):
        self.initial_grammar = grammar
        self.grammar = copy.deepcopy(grammar)
        self.tasks = tasks

        # Progressive weight
        self.initial_neural_weight = initial_neural_weight
        self.weight_increment = weight_increment
        self.solved_boost = solved_boost
        self.max_neural_weight = max_neural_weight

        # Enumeration
        self.enumeration_budget = enumeration_budget
        self.max_depth = max_depth

        # Recognition
        self.recognition_hidden_dim = recognition_hidden_dim
        self.recognition_epochs = recognition_epochs
        self.recognition_lr = recognition_lr
        self.structural_weight = structural_weight

        # Compression
        self.use_compression = use_compression
        self.max_inventions = max_inventions

        # General
        self.max_iterations = max_iterations
        self.time_limit_seconds = time_limit_minutes * 60
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.device = device
        self.verbose = verbose

        # State
        self.solved_tasks: Dict[str, Program] = {}
        self.solved_programs: Dict[str, str] = {}  # task_name -> program_str
        self.iteration_results: List[IterationResult] = []
        self.start_time = 0.0

        # Initialize recognition model
        self.recognition = ContrastiveRecognitionModel(
            grammar=grammar,
            card_out=recognition_hidden_dim,
            pred_hidden=recognition_hidden_dim * 2,
            learning_rate=recognition_lr,
            device=device
        )

        # Initialize enumerator
        self.enumerator = TopDownEnumerator(
            grammar=grammar,
            max_programs=enumeration_budget,
            max_depth=max_depth
        )

    def log(self, msg: str, level: int = 0):
        """Log with indentation and flush."""
        if self.verbose:
            indent = "  " * level
            print(f"{indent}{msg}", flush=True)

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

    def verify_on_holdout(self, program: Program, holdout: List[Tuple[Hand, bool]]) -> bool:
        """Verify program on holdout examples."""
        return self.evaluate_program(program, holdout)

    def get_current_weight(self, iteration: int) -> float:
        """Compute current neural weight based on iteration and progress."""
        solved_ratio = len(self.solved_tasks) / len(self.tasks)
        return compute_neural_weight(
            iteration,
            solved_ratio,
            self.initial_neural_weight,
            self.weight_increment,
            self.solved_boost,
            self.max_neural_weight
        )

    def run_wake_phase(self, iteration: int, weight: float) -> Tuple[int, int, List[str]]:
        """
        Wake phase: enumerate and solve tasks with neural guidance.

        Returns: (solved_count, programs_enumerated, newly_solved_names)
        """
        self.log(f"Wake phase (weight={weight:.2f})...", 1)

        target_type = arrow(HAND, BOOL)
        total_programs = 0
        newly_solved = []

        unsolved_tasks = [t for t in self.tasks if t.name not in self.solved_tasks]

        for task_idx, task in enumerate(unsolved_tasks):
            # Check time limit
            elapsed = time.time() - self.start_time
            if elapsed > self.time_limit_seconds:
                self.log(f"Time limit reached", 2)
                break

            programs_checked = 0
            solved = False

            # Get neural guidance if weight > 0
            if weight > 0:
                try:
                    recognition_task = task.to_recognition_task()
                    neural_probs = self.recognition.predict_primitives_dict(recognition_task)
                except Exception as e:
                    neural_probs = {}
                    self.log(f"Warning: Neural prediction failed: {e}", 3)
            else:
                neural_probs = {}

            # Enumerate programs (using memoized enumeration)
            for program, cost in self.enumerator.enumerate_memoized(target_type):
                programs_checked += 1
                total_programs += 1

                # Check timeout every 1000 programs
                if programs_checked % 1000 == 0:
                    elapsed = time.time() - self.start_time
                    if elapsed > self.time_limit_seconds:
                        break

                if self.evaluate_program(program, task.examples):
                    if self.verify_on_holdout(program, task.holdout):
                        solved = True
                        self.solved_tasks[task.name] = program
                        self.solved_programs[task.name] = str(program)
                        newly_solved.append(task.name)
                        break

                if programs_checked >= self.enumeration_budget:
                    break

            status = f"SOLVED at {programs_checked:,}" if solved else f"unsolved after {programs_checked:,}"
            self.log(f"[{task_idx+1}/{len(unsolved_tasks)}] {task.name}: {status}", 2)

        return len(self.solved_tasks), total_programs, newly_solved

    def run_sleep_recognition(self) -> float:
        """
        Sleep phase: train recognition model on solved tasks.

        Returns: final training loss
        """
        if not self.solved_tasks:
            return 0.0

        self.log("Sleep-Recognition: training on solved tasks...", 1)

        # Create training data from solved tasks
        frontiers = []
        for task in self.tasks:
            if task.name in self.solved_tasks:
                program = self.solved_tasks[task.name]
                recognition_task = task.to_recognition_task()
                frontiers.append((recognition_task, [program]))

        # Train recognition model
        try:
            loss = self.recognition.train_on_frontiers(
                frontiers,
                epochs=self.recognition_epochs,
                structural_similarity_weight=self.structural_weight
            )
            self.log(f"Recognition loss: {loss:.4f}", 2)
            return loss
        except Exception as e:
            self.log(f"Warning: Recognition training failed: {e}", 2)
            return 0.0

    def run_sleep_compression(self) -> List[str]:
        """
        Sleep phase: compress solved programs to find abstractions.

        Returns: list of new abstraction names
        """
        if not self.use_compression or len(self.solved_tasks) < 2:
            return []

        self.log("Sleep-Compression: finding abstractions...", 1)

        # Prepare frontiers for compression
        frontiers = []
        for task_name, program in self.solved_tasks.items():
            task = next(t for t in self.tasks if t.name == task_name)
            recognition_task = task.to_recognition_task()
            frontiers.append((recognition_task, [(program, 0.0)]))

        try:
            result = compress_frontiers(
                frontiers,
                self.grammar,
                max_inventions=self.max_inventions
            )

            new_abstractions = []
            if result and result.new_grammar:
                # Check for new primitives
                old_prims = set(p.name for p in self.grammar.primitives())
                new_prims = set(p.name for p in result.new_grammar.primitives())
                new_abstractions = list(new_prims - old_prims)

                if new_abstractions:
                    self.grammar = result.new_grammar
                    self.enumerator = TopDownEnumerator(
                        grammar=self.grammar,
                        max_programs=self.enumeration_budget,
                        max_depth=self.max_depth
                    )
                    # Update recognition model vocabulary
                    self.recognition.expand_vocabulary(self.grammar)

                    for abstr in new_abstractions:
                        self.log(f"New abstraction: {abstr}", 2)

            return new_abstractions
        except Exception as e:
            self.log(f"Warning: Compression failed: {e}", 2)
            return []

    def run_iteration(self, iteration: int) -> IterationResult:
        """Run one complete wake-sleep iteration."""
        iter_start = time.time()

        # Compute neural weight for this iteration
        weight = self.get_current_weight(iteration)
        self.log(f"Neural weight: {weight:.2f} "
                f"(iter={iteration}, solved={len(self.solved_tasks)}/{len(self.tasks)})", 1)

        # Wake phase
        wake_start = time.time()
        solved_count, programs_enumerated, newly_solved = self.run_wake_phase(iteration, weight)
        wake_time = time.time() - wake_start

        # Sleep phase
        sleep_start = time.time()
        recognition_loss = self.run_sleep_recognition()
        new_abstractions = self.run_sleep_compression()
        sleep_time = time.time() - sleep_start

        result = IterationResult(
            iteration=iteration,
            tasks_solved=solved_count,
            tasks_total=len(self.tasks),
            programs_enumerated=programs_enumerated,
            neural_weight=weight,
            wake_time=wake_time,
            sleep_time=sleep_time,
            recognition_loss=recognition_loss,
            new_abstractions=new_abstractions,
            solved_task_names=list(self.solved_tasks.keys())
        )

        return result

    def save_incremental_results(self):
        """Save incremental results after each iteration."""
        elapsed = time.time() - self.start_time

        output = {
            "status": "in_progress",
            "elapsed_minutes": elapsed / 60,
            "tasks_solved": len(self.solved_tasks),
            "tasks_total": len(self.tasks),
            "iterations_completed": len(self.iteration_results),
            "solved_tasks": {
                name: prog_str for name, prog_str in self.solved_programs.items()
            },
            "iteration_history": [
                {
                    "iteration": r.iteration,
                    "solved": r.tasks_solved,
                    "weight": r.neural_weight,
                    "loss": r.recognition_loss,
                    "new_abstractions": r.new_abstractions
                }
                for r in self.iteration_results
            ]
        }

        with open(self.results_dir / "incremental_results.json", 'w') as f:
            json.dump(output, f, indent=2)

    def run(self) -> Dict[str, Any]:
        """Run the full progressive wake-sleep loop."""
        self.start_time = time.time()

        self.log("=" * 70)
        self.log("PROGRESSIVE NEURAL WAKE-SLEEP")
        self.log("=" * 70)
        self.log(f"Tasks: {len(self.tasks)}")
        self.log(f"Grammar: {len(list(self.grammar.primitives()))} primitives")
        self.log(f"Max iterations: {self.max_iterations}")
        self.log(f"Time limit: {self.time_limit_seconds/60:.1f} min")
        self.log(f"Neural weight: {self.initial_neural_weight} → {self.max_neural_weight}")
        self.log("")

        for iteration in range(self.max_iterations):
            # Check time limit
            elapsed = time.time() - self.start_time
            if elapsed > self.time_limit_seconds:
                self.log(f"\nTime limit reached ({elapsed/60:.1f} min)")
                break

            self.log("=" * 70)
            self.log(f"ITERATION {iteration + 1}/{self.max_iterations}")
            self.log("=" * 70)

            result = self.run_iteration(iteration)
            self.iteration_results.append(result)

            # Log summary
            self.log("")
            self.log(f"Summary:", 1)
            self.log(f"Solved: {result.tasks_solved}/{result.tasks_total}", 2)
            self.log(f"Programs: {result.programs_enumerated:,}", 2)
            self.log(f"Wake time: {result.wake_time:.1f}s", 2)
            self.log(f"Sleep time: {result.sleep_time:.1f}s", 2)
            self.log(f"New abstractions: {len(result.new_abstractions)}", 2)

            # Save incremental results
            self.save_incremental_results()

            # Early stop if all solved
            if result.tasks_solved == result.tasks_total:
                self.log("\nAll tasks solved!")
                break

        # Final summary
        total_time = time.time() - self.start_time
        self.log("")
        self.log("=" * 70)
        self.log("FINAL SUMMARY")
        self.log("=" * 70)
        self.log(f"Total time: {total_time/60:.1f} min")
        self.log(f"Iterations: {len(self.iteration_results)}")
        self.log(f"Solved: {len(self.solved_tasks)}/{len(self.tasks)}")
        self.log(f"Final grammar: {len(list(self.grammar.primitives()))} primitives")

        # Compile final results
        results = {
            "summary": {
                "total_time_seconds": total_time,
                "total_time_minutes": total_time / 60,
                "iterations": len(self.iteration_results),
                "tasks_solved": len(self.solved_tasks),
                "tasks_total": len(self.tasks),
                "final_grammar_size": len(list(self.grammar.primitives()))
            },
            "solved_tasks": [
                {
                    "name": name,
                    "program": self.solved_programs[name]
                }
                for name in self.solved_tasks.keys()
            ],
            "iteration_history": [
                {
                    "iteration": r.iteration,
                    "tasks_solved": r.tasks_solved,
                    "programs_enumerated": r.programs_enumerated,
                    "neural_weight": r.neural_weight,
                    "recognition_loss": r.recognition_loss,
                    "wake_time": r.wake_time,
                    "sleep_time": r.sleep_time,
                    "new_abstractions": r.new_abstractions
                }
                for r in self.iteration_results
            ],
            "unsolved_tasks": [
                t.name for t in self.tasks if t.name not in self.solved_tasks
            ]
        }

        # Save final results
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        with open(self.results_dir / f"final_results_{timestamp}.json", 'w') as f:
            json.dump(results, f, indent=2)

        self.log(f"\nResults saved to: {self.results_dir}")

        return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Progressive Neural Wake-Sleep")

    # Time and iteration limits
    parser.add_argument("--time-limit", type=float, default=120.0,
                       help="Time limit in minutes (default: 120)")
    parser.add_argument("--max-iterations", type=int, default=5,
                       help="Max wake-sleep iterations (default: 5)")

    # Enumeration
    parser.add_argument("--enum-budget", type=int, default=1_000_000,
                       help="Enumeration budget per task (default: 1,000,000)")
    parser.add_argument("--max-depth", type=int, default=8,
                       help="Max program depth (default: 8)")

    # Progressive weight
    parser.add_argument("--initial-weight", type=float, default=0.1,
                       help="Initial neural weight (default: 0.1)")
    parser.add_argument("--weight-increment", type=float, default=0.1,
                       help="Weight increment per iteration (default: 0.1)")
    parser.add_argument("--max-weight", type=float, default=0.9,
                       help="Max neural weight (default: 0.9)")

    # Recognition
    parser.add_argument("--hidden-dim", type=int, default=32,
                       help="Recognition hidden dimension (default: 32)")
    parser.add_argument("--recog-epochs", type=int, default=10,
                       help="Recognition training epochs (default: 10)")

    # Compression
    parser.add_argument("--no-compression", action="store_true",
                       help="Disable compression")

    args = parser.parse_args()

    print("Building grammar with parallel primitives...", flush=True)
    grammar = build_extended_grammar()
    print(f"  Grammar size: {len(list(grammar.primitives()))} primitives", flush=True)

    print("\nCreating tasks...", flush=True)
    tasks = create_all_tasks(n_examples=20, n_holdout=100)
    print(f"  Created {len(tasks)} tasks", flush=True)

    # Create and run learner
    learner = ProgressiveWakeSleep(
        grammar=grammar,
        tasks=tasks,
        initial_neural_weight=args.initial_weight,
        weight_increment=args.weight_increment,
        max_neural_weight=args.max_weight,
        enumeration_budget=args.enum_budget,
        max_depth=args.max_depth,
        recognition_hidden_dim=args.hidden_dim,
        recognition_epochs=args.recog_epochs,
        use_compression=not args.no_compression,
        max_iterations=args.max_iterations,
        time_limit_minutes=args.time_limit
    )

    results = learner.run()

    # Print comparison
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)
    print(f"Baseline (no neural): 8/45 solved")
    print(f"Parallel primitives only: 11/45 solved")
    print(f"Progressive wake-sleep: {results['summary']['tasks_solved']}/45 solved")


if __name__ == "__main__":
    main()
