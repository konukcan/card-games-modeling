#!/usr/bin/env python3
"""
Incremental Wake-Sleep with Dream-After-Each-Task

This script runs a wake-sleep learning loop where:
1. After EACH newly solved task, we immediately train the recognition model
2. Neural guidance weight increases with solved tasks (progressive weighting)
3. Uses ContrastiveRecognitionModel (τ = mean(pos) - mean(neg))
4. Parallel primitives for structural patterns

KEY DIFFERENCE FROM run_progressive_wakesleep.py:
- Dreams incrementally after each solved task
- Not batched at end of iteration

Author: Can Konuk
Date: December 2024
"""

import sys
import os
import time
import json
import copy
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
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
from dreamcoder_core.contrastive_wake_sleep import TaskFrontier
from dreamcoder_core.compression import compress_frontiers
from dreamcoder_core.task import Task

from rules.cards import (
    Card, Hand, Suit, Rank, Color, RANK_VALUES, card_color, sample_hand
)
from rules.catalogue import create_all_rules, Rule


# ============================================================================
# PARALLEL PRIMITIVES
# ============================================================================

COLOR = BaseType('color')

def make_parallel_primitives() -> List[Primitive]:
    """Create parallel structure primitives."""
    primitives = []

    # halves_equal_by: (card → α) → hand → bool
    def halves_equal_by(f):
        def check(hand):
            if len(hand) < 2:
                return True
            mid = len(hand) // 2
            left = hand[:mid]
            right = hand[mid:2*mid]
            return [f(c) for c in left] == [f(c) for c in right]
        return check

    primitives.append(Primitive(
        "halves_equal_by",
        arrow(arrow(CARD, TypeVariable('α')), arrow(HAND, BOOL)),
        halves_equal_by
    ))

    # ends_equal_by: (card → α) → hand → bool
    def ends_equal_by(f):
        def check(hand):
            if len(hand) < 2:
                return True
            return f(hand[0]) == f(hand[-1])
        return check

    primitives.append(Primitive(
        "ends_equal_by",
        arrow(arrow(CARD, TypeVariable('α')), arrow(HAND, BOOL)),
        ends_equal_by
    ))

    # is_palindrome_by: (card → α) → hand → bool
    def is_palindrome_by(f):
        def check(hand):
            values = [f(c) for c in hand]
            return values == values[::-1]
        return check

    primitives.append(Primitive(
        "is_palindrome_by",
        arrow(arrow(CARD, TypeVariable('α')), arrow(HAND, BOOL)),
        is_palindrome_by
    ))

    # all_adjacent_satisfy: (card → card → bool) → hand → bool
    def all_adjacent_satisfy(pred):
        def check(hand):
            if len(hand) < 2:
                return True
            for i in range(len(hand) - 1):
                if not pred(hand[i])(hand[i+1]):
                    return False
            return True
        return check

    primitives.append(Primitive(
        "all_adjacent_satisfy",
        arrow(arrow(CARD, arrow(CARD, BOOL)), arrow(HAND, BOOL)),
        all_adjacent_satisfy
    ))

    return primitives


def build_extended_grammar() -> Grammar:
    """Build grammar with lean primitives + parallel primitives."""
    base_grammar = build_lean_grammar()
    parallel_prims = make_parallel_primitives()

    # Get log_probability from existing productions (use uniform)
    if base_grammar.productions:
        log_p = base_grammar.productions[0].log_probability
    else:
        log_p = 0.0

    all_productions = list(base_grammar.productions)
    for prim in parallel_prims:
        all_productions.append(Production(prim, prim.tp, log_p))

    return Grammar(all_productions, base_grammar.log_variable)


def compute_neural_weight(
    n_solved: int,
    n_total: int,
    initial_weight: float = 0.1,
    max_weight: float = 0.9
) -> float:
    """Compute neural guidance weight based on progress."""
    if n_total == 0:
        return initial_weight
    solved_ratio = n_solved / n_total
    # Linear interpolation from initial to max based on solved ratio
    return initial_weight + (max_weight - initial_weight) * solved_ratio


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

    positives, negatives = [], []
    holdout_pos, holdout_neg = [], []

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
        if (len(positives) >= target and len(negatives) >= target and
            len(holdout_pos) >= holdout_target and len(holdout_neg) >= holdout_target):
            break

    examples = positives + negatives
    holdout = holdout_pos + holdout_neg
    random.shuffle(examples)
    random.shuffle(holdout)

    return examples, holdout


def create_all_tasks(n_examples: int = 20, n_holdout: int = 100) -> List[CardTask]:
    """Create all tasks from the rule catalogue."""
    rules = create_all_rules()
    tasks = []

    for rule in rules:
        examples, holdout = create_balanced_examples(rule, n_examples, n_holdout)
        if len(examples) >= n_examples:
            tasks.append(CardTask(
                name=rule.id,
                rule=rule,
                examples=examples,
                holdout=holdout
            ))
        else:
            print(f"  WARNING: Skipping {rule.id} - insufficient examples", flush=True)

    return tasks


# ============================================================================
# INCREMENTAL WAKE-SLEEP
# ============================================================================

class IncrementalWakeSleep:
    """
    DreamCoder with incremental dreaming after each solved task.

    Key difference from batch wake-sleep:
    - After each task is solved, immediately update recognition model
    - This allows neural guidance to improve within an iteration
    """

    def __init__(
        self,
        grammar: Grammar,
        tasks: List[CardTask],

        # Neural settings
        initial_neural_weight: float = 0.1,
        max_neural_weight: float = 0.9,

        # Enumeration settings
        enumeration_budget: int = 1_000_000,
        max_depth: int = 10,

        # Recognition settings
        recognition_hidden_dim: int = 32,
        recognition_epochs: int = 5,  # Fewer epochs since we train more often
        recognition_lr: float = 1e-3,

        # General settings
        max_iterations: int = 3,
        time_limit_minutes: float = 120.0,
        results_dir: str = "results_incremental_wakesleep",
        device: str = 'cpu',
        verbose: bool = True
    ):
        self.grammar = copy.deepcopy(grammar)
        self.tasks = tasks

        # Neural weight
        self.initial_neural_weight = initial_neural_weight
        self.max_neural_weight = max_neural_weight

        # Enumeration
        self.enumeration_budget = enumeration_budget
        self.max_depth = max_depth

        # Recognition
        self.recognition_hidden_dim = recognition_hidden_dim
        self.recognition_epochs = recognition_epochs
        self.recognition_lr = recognition_lr

        # General
        self.max_iterations = max_iterations
        self.time_limit_seconds = time_limit_minutes * 60
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.device = device
        self.verbose = verbose

        # State
        self.solved_tasks: Dict[str, Program] = {}
        self.frontiers: Dict[str, TaskFrontier] = {}
        self.start_time = 0.0
        self.dreams_count = 0

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
                if fn(hand) != expected:
                    return False
            return True
        except Exception:
            return False

    def verify_on_holdout(self, program: Program, holdout: List[Tuple[Hand, bool]]) -> bool:
        """Verify program on holdout examples."""
        return self.evaluate_program(program, holdout)

    def get_current_weight(self) -> float:
        """Compute current neural weight based on progress."""
        return compute_neural_weight(
            len(self.solved_tasks),
            len(self.tasks),
            self.initial_neural_weight,
            self.max_neural_weight
        )

    def dream_on_task(self, task: CardTask, program: Program):
        """
        Incremental dream: update recognition model after solving one task.

        This is the key innovation - we train immediately after each solution.
        """
        self.dreams_count += 1

        # Create frontier for this task
        recognition_task = task.to_recognition_task()
        frontier = TaskFrontier(task=recognition_task)
        frontier.add(EnumerationResult(
            program=program,
            log_probability=0.0,
            log_likelihood=0.0,  # Perfect solution
            description_length=0.0,
            programs_enumerated=0,
            time_seconds=0.0
        ))

        # Update frontiers dict
        self.frontiers[task.name] = frontier

        # Create training data for all solved tasks
        all_tasks = []
        frontiers_dict = {}

        for t in self.tasks:
            if t.name in self.frontiers:
                recognition_t = t.to_recognition_task()
                all_tasks.append(recognition_t)
                frontiers_dict[t.name] = self.frontiers[t.name]

        # Train recognition model
        if len(all_tasks) >= 1:
            try:
                loss = self.recognition.train_on_frontiers(
                    tasks=all_tasks,
                    frontiers=frontiers_dict,
                    epochs=self.recognition_epochs,
                    lambda_struct=0.3,
                    lambda_count=0.1,
                    lambda_pred=1.0
                )
                self.log(f"Dream #{self.dreams_count}: trained on {len(all_tasks)} tasks, loss={loss:.4f}", 2)
            except Exception as e:
                self.log(f"Dream #{self.dreams_count}: training failed - {e}", 2)

    def enumerate_for_task(self, task: CardTask, weight: float) -> Optional[Tuple[Program, int]]:
        """
        Enumerate programs for a task, return first valid solution.

        Returns: (program, programs_checked) or None
        """
        target_type = arrow(HAND, BOOL)
        programs_checked = 0

        for program, cost in self.enumerator.enumerate_memoized(target_type):
            programs_checked += 1

            # Check timeout every 1000 programs
            if programs_checked % 1000 == 0:
                elapsed = time.time() - self.start_time
                if elapsed > self.time_limit_seconds:
                    return None

            # Check budget
            if programs_checked > self.enumeration_budget:
                return None

            # Evaluate on training examples
            if self.evaluate_program(program, task.examples):
                # Verify on holdout
                if self.verify_on_holdout(program, task.holdout):
                    return (program, programs_checked)

        return None

    def run_iteration(self, iteration: int) -> Dict[str, Any]:
        """Run one iteration over all unsolved tasks."""
        iter_start = time.time()
        weight = self.get_current_weight()

        self.log(f"Neural weight: {weight:.2f} ({len(self.solved_tasks)}/{len(self.tasks)} solved)", 1)

        # Get unsolved tasks
        unsolved = [t for t in self.tasks if t.name not in self.solved_tasks]

        new_solutions = []
        total_programs = 0

        for idx, task in enumerate(unsolved):
            # Check timeout
            elapsed = time.time() - self.start_time
            if elapsed > self.time_limit_seconds:
                self.log("Time limit reached", 1)
                break

            result = self.enumerate_for_task(task, weight)

            if result:
                program, programs_checked = result
                total_programs += programs_checked

                # Store solution
                self.solved_tasks[task.name] = program
                new_solutions.append({
                    'name': task.name,
                    'programs': programs_checked,
                    'solution': str(program)
                })

                self.log(f"[{idx+1}/{len(unsolved)}] {task.name}: SOLVED at {programs_checked:,}", 1)

                # INCREMENTAL DREAM: train immediately after solving
                self.dream_on_task(task, program)
            else:
                total_programs += self.enumeration_budget
                self.log(f"[{idx+1}/{len(unsolved)}] {task.name}: unsolved", 1)

        iter_time = time.time() - iter_start

        return {
            'iteration': iteration,
            'new_solutions': new_solutions,
            'total_solved': len(self.solved_tasks),
            'total_programs': total_programs,
            'neural_weight': weight,
            'time_seconds': iter_time,
            'dreams': self.dreams_count
        }

    def save_results(self, results: Dict[str, Any]):
        """Save results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        filename = self.results_dir / f"incremental_results_{timestamp}.json"

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.log(f"Results saved to {filename}")

    def run(self) -> Dict[str, Any]:
        """Run the full incremental wake-sleep loop."""
        self.start_time = time.time()

        self.log("=" * 70)
        self.log("INCREMENTAL WAKE-SLEEP (Dream After Each Task)")
        self.log("=" * 70)
        self.log(f"Tasks: {len(self.tasks)}")
        self.log(f"Grammar: {len(self.grammar.productions)} primitives")
        self.log(f"Max iterations: {self.max_iterations}")
        self.log(f"Time limit: {self.time_limit_seconds/60:.1f} min")
        self.log(f"Neural weight: {self.initial_neural_weight} → {self.max_neural_weight}")
        self.log("")

        all_iterations = []

        for iteration in range(1, self.max_iterations + 1):
            elapsed = time.time() - self.start_time
            if elapsed > self.time_limit_seconds:
                self.log("Time limit reached, stopping")
                break

            self.log("=" * 70)
            self.log(f"ITERATION {iteration}/{self.max_iterations}")
            self.log("=" * 70)

            iter_result = self.run_iteration(iteration)
            all_iterations.append(iter_result)

            self.log(f"\nIteration {iteration} summary:", 1)
            self.log(f"  New solutions: {len(iter_result['new_solutions'])}", 1)
            self.log(f"  Total solved: {iter_result['total_solved']}/{len(self.tasks)}", 1)
            self.log(f"  Dreams so far: {iter_result['dreams']}", 1)
            self.log(f"  Time: {iter_result['time_seconds']:.1f}s", 1)
            self.log("")

            # Stop if all solved
            if len(self.solved_tasks) >= len(self.tasks):
                self.log("All tasks solved!")
                break

        total_time = time.time() - self.start_time

        # Final results
        results = {
            'summary': {
                'total_time_seconds': total_time,
                'total_time_minutes': total_time / 60,
                'tasks_solved': len(self.solved_tasks),
                'tasks_total': len(self.tasks),
                'total_dreams': self.dreams_count
            },
            'iterations': all_iterations,
            'solved_tasks': [
                {
                    'name': name,
                    'solution': str(prog)
                }
                for name, prog in self.solved_tasks.items()
            ],
            'unsolved_tasks': [
                t.name for t in self.tasks if t.name not in self.solved_tasks
            ]
        }

        self.save_results(results)

        self.log("=" * 70)
        self.log("FINAL SUMMARY")
        self.log("=" * 70)
        self.log(f"Tasks solved: {len(self.solved_tasks)}/{len(self.tasks)}")
        self.log(f"Total dreams: {self.dreams_count}")
        self.log(f"Total time: {total_time/60:.1f} minutes")

        return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("Building grammar with parallel primitives...", flush=True)
    grammar = build_extended_grammar()
    print(f"  Grammar size: {len(grammar.productions)} primitives", flush=True)

    print("\nCreating tasks...", flush=True)
    tasks = create_all_tasks(n_examples=20, n_holdout=100)
    print(f"  Created {len(tasks)} tasks", flush=True)

    # Create and run
    ws = IncrementalWakeSleep(
        grammar=grammar,
        tasks=tasks,
        initial_neural_weight=0.1,
        max_neural_weight=0.9,
        enumeration_budget=1_000_000,
        max_depth=10,
        recognition_epochs=5,
        max_iterations=3,
        time_limit_minutes=120.0,
        results_dir="results_incremental_wakesleep",
        verbose=True
    )

    results = ws.run()
    return results


if __name__ == "__main__":
    main()
