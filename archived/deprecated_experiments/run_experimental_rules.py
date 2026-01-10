#!/usr/bin/env python3
"""
Run DreamCoder on Experimental Rules from catalogue.py

This script runs the Set Transformer-based DreamCoder on the 57 experimental
rules from our human study, rather than the pretraining rules.

Key features:
1. Uses experimental rules from catalogue.py (the actual study rules)
2. Set Transformer recognition model with raw feature correlation encoding
3. Improved task discrimination through raw card feature correlations
4. Multiprocessing with PyPy workers for parallel enumeration
"""

import sys
import os
import time
import json
import random
import copy
import pickle
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import torch

from dreamcoder_core.type_system import arrow, HAND, BOOL
from dreamcoder_core.program import Program, Primitive, Invented, parse_program
from dreamcoder_core.grammar import Grammar
from dreamcoder_core.enumeration import enumerate_simple, TopDownEnumerator

# Use Set Transformer recognition model
from dreamcoder_core.set_transformer_recognition import SetTransformerRecognitionModel

# Compression module
from dreamcoder_core.compression import compress_frontiers

# Primitives
from dreamcoder_core.lean_primitives import build_lean_grammar

# DreamCoder v2 components
from dreamcoder_core.dreamcoder_original import (
    Task, SolutionEntry, TaskFrontier, IterationMetrics, TaskMetrics,
    NeuralDreamer, make_eval_fn
)

# EXPERIMENTAL rules from catalogue
from rules.catalogue import create_all_rules, get_rule, get_rules_by_family
from rules.cards import sample_hand


# ============================================================================
# CONFIGURATION
# ============================================================================

# Check for PyPy availability
PYPY_PATH = shutil.which('pypy3.10') or shutil.which('pypy3')
USE_PYPY = PYPY_PATH is not None

# Number of parallel workers (recommend 4 on M1 Mac)
N_WORKERS = 4

# Use multiprocessing or sequential (fallback)
USE_MULTIPROCESSING = True


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


def print_banner(text: str, char: str = "="):
    """Print a banner."""
    line = char * 80
    print(f"\n{line}")
    print(text)
    print(f"{line}\n", flush=True)


@dataclass
class PhaseConfig:
    """Configuration for a training phase."""
    name: str
    iterations: int
    max_depth: int
    max_programs: int
    timeout_per_task: float
    recognition_epochs: int = 5
    dream_tasks: int = 0


# ============================================================================
# TASK CREATION FROM CATALOGUE RULES
# ============================================================================

def sample_balanced_examples(
    rule,
    n_examples: int,
    hand_size: int,
    max_attempts: int = 100000,
    min_per_class: int = 5
) -> Tuple[List[Tuple[List, bool]], bool]:
    """
    Sample balanced examples (50% True, 50% False) for a rule.

    Uses rejection sampling to get equal positive and negative examples.
    If a rule has extreme base rates, this may hit max_attempts before
    getting enough examples of the rare class.

    Args:
        rule: Rule object from catalogue
        n_examples: Total number of examples to generate
        hand_size: Number of cards per hand
        max_attempts: Maximum sampling attempts before giving up
        min_per_class: Minimum examples needed per class, otherwise rule is skipped

    Returns:
        Tuple of (examples, is_valid):
        - examples: List of (hand, result) tuples, balanced as evenly as possible
        - is_valid: True if we got at least min_per_class of each class
    """
    n_positive = n_examples // 2
    n_negative = n_examples - n_positive

    positives = []
    negatives = []
    attempts = 0

    while (len(positives) < n_positive or len(negatives) < n_negative) and attempts < max_attempts:
        hand = sample_hand(hand_size)
        result = rule.eval(hand)
        attempts += 1

        if result and len(positives) < n_positive:
            positives.append((hand, True))
        elif not result and len(negatives) < n_negative:
            negatives.append((hand, False))

    # Check if we have enough of each class
    is_valid = len(positives) >= min_per_class and len(negatives) >= min_per_class

    # Combine and shuffle
    examples = positives + negatives
    random.shuffle(examples)

    # Log warning if we couldn't get balanced examples
    if len(positives) < n_positive or len(negatives) < n_negative:
        status = "SKIPPING" if not is_valid else "WARNING"
        print(f"{status}: {rule.id} - got {len(positives)} positive, "
              f"{len(negatives)} negative examples after {max_attempts} attempts")

    return examples, is_valid


def create_tasks_from_catalogue(
    rules: List = None,
    n_examples: int = 100,
    n_holdout: int = 20,
    hand_size: int = 6,  # Standardized to 6 for all experiments
    seed: int = 42,
    balanced: bool = True,  # Use balanced sampling
    min_per_class: int = 5  # Minimum examples per class (rules with fewer are skipped)
) -> Tuple[List[Task], List[str]]:
    """
    Create Task objects from catalogue rules.

    Args:
        rules: List of Rule objects from catalogue.py (default: all rules)
        n_examples: Number of training examples
        n_holdout: Number of held-out examples for verification
        hand_size: Number of cards per hand
        seed: Random seed for reproducibility
        balanced: If True, ensure ~50% positive and ~50% negative examples
        min_per_class: Minimum examples needed per class (rules with extreme
                       base rates that can't meet this threshold are skipped)

    Returns:
        Tuple of (tasks, skipped_rules):
        - tasks: List of Task objects for rules that could be balanced
        - skipped_rules: List of rule IDs that were skipped due to extreme base rates
    """
    if rules is None:
        rules = create_all_rules()

    random.seed(seed)
    tasks = []
    skipped_rules = []
    request_type = arrow(HAND, BOOL)

    print(f"Creating tasks from {len(rules)} catalogue rules...")
    print(f"Using balanced sampling: {balanced}")
    print(f"Minimum examples per class: {min_per_class}")
    print()

    for rule in rules:
        if balanced:
            # Use balanced sampling - crucial for rules with extreme base rates
            examples, train_valid = sample_balanced_examples(
                rule, n_examples, hand_size, min_per_class=min_per_class
            )
            holdout_examples, holdout_valid = sample_balanced_examples(
                rule, n_holdout, hand_size, min_per_class=min_per_class // 2 or 1
            )

            # Skip rules where we can't get balanced examples
            if not train_valid or not holdout_valid:
                skipped_rules.append(rule.id)
                continue
        else:
            # Original random sampling (can have extreme class imbalance)
            examples = []
            for _ in range(n_examples):
                hand = sample_hand(hand_size)
                result = rule.eval(hand)
                examples.append((hand, result))

            holdout_examples = []
            for _ in range(n_holdout):
                hand = sample_hand(hand_size)
                result = rule.eval(hand)
                holdout_examples.append((hand, result))

        task = Task(
            name=rule.id,
            request_type=request_type,
            examples=examples,
            family=rule.family,
            difficulty_level=rule.level
        )

        # Store holdout examples for verification
        task.holdout_examples = holdout_examples

        tasks.append(task)

    if skipped_rules:
        print()
        print(f"Skipped {len(skipped_rules)} rules due to extreme base rates:")
        for rule_id in skipped_rules:
            print(f"  - {rule_id}")
        print()

    return tasks, skipped_rules


# ============================================================================
# PARALLEL ENUMERATION (same as set transformer script)
# ============================================================================

def enumerate_task(task_data, grammar_productions, max_depth, max_programs, timeout):
    """
    Enumerate programs for a single task using MEMOIZED enumeration.
    This runs in a separate process (potentially with PyPy).

    Uses TopDownEnumerator.enumerate_memoized() for 1000x+ speedup over
    the legacy enumerate_simple() approach.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from dreamcoder_core.type_system import arrow, HAND, BOOL
    from dreamcoder_core.enumeration import TopDownEnumerator
    from dreamcoder_core.lean_primitives import build_lean_grammar
    from rules.cards import Card, Suit, Rank

    # Build grammar
    grammar = build_lean_grammar()
    request_type = arrow(HAND, BOOL)

    if grammar_productions:
        pass  # Could restore grammar weights here

    # Reconstruct examples
    examples = []
    for hand_data, result in task_data['examples']:
        hand = [Card(Suit[c['suit']], Rank[c['rank']]) for c in hand_data]
        examples.append((hand, result))

    # Enumerate using MEMOIZED approach (1000x+ speedup)
    programs_found = []
    programs_enumerated = 0
    start_time = time.time()

    # Create enumerator with memoization
    enumerator = TopDownEnumerator(
        grammar=grammar,
        max_depth=max_depth,
        max_programs=max_programs
    )

    for program, log_prob in enumerator.enumerate_memoized(
        request_type,
        max_cost=50.0,
        timeout_seconds=timeout,
        depth_limit=max_depth
    ):
        programs_enumerated += 1

        if programs_enumerated > max_programs:
            break

        # Check if program solves all examples
        try:
            all_correct = True
            for hand, expected in examples:
                result = program.evaluate([])(hand)
                if result != expected:
                    all_correct = False
                    break

            if all_correct:
                programs_found.append({
                    'program_str': str(program),
                    'log_probability': log_prob,
                    'programs_enumerated': programs_enumerated
                })
        except Exception:
            pass

    return {
        'task_name': task_data['name'],
        'programs_found': programs_found,
        'programs_enumerated': programs_enumerated,
        'time_elapsed': time.time() - start_time,
        'memo_stats': enumerator.get_memo_stats() if hasattr(enumerator, 'get_memo_stats') else None
    }


def serialize_task(task) -> Dict:
    """Serialize a task for multiprocessing."""
    return {
        'name': task.name,
        'family': task.family,
        'examples': [
            ([{'suit': c.suit.name, 'rank': c.rank.name} for c in hand], result)
            for hand, result in task.examples
        ],
        'grammar_productions': None,
    }


def enumerate_task_sequential(task, grammar, max_depth, max_programs, timeout):
    """
    Sequential enumeration for a task using MEMOIZED enumeration.

    Uses TopDownEnumerator.enumerate_memoized() for 1000x+ speedup over
    the legacy enumerate_simple() approach.
    """
    programs_found = []
    programs_enumerated = 0
    start_time = time.time()

    # Create enumerator with memoization
    enumerator = TopDownEnumerator(
        grammar=grammar,
        max_depth=max_depth,
        max_programs=max_programs
    )

    for program, log_prob in enumerator.enumerate_memoized(
        task.request_type,
        max_cost=50.0,
        timeout_seconds=timeout,
        depth_limit=max_depth
    ):
        programs_enumerated += 1

        if programs_enumerated > max_programs:
            break

        try:
            all_correct = True
            for hand, expected in task.examples:
                result = program.evaluate([])(hand)
                if result != expected:
                    all_correct = False
                    break

            if all_correct:
                programs_found.append({
                    'program': program,
                    'log_probability': log_prob,
                    'programs_enumerated': programs_enumerated
                })
        except Exception:
            pass

    return {
        'task_name': task.name,
        'programs_found': programs_found,
        'programs_enumerated': programs_enumerated,
        'time_elapsed': time.time() - start_time,
        'memo_stats': enumerator.get_memo_stats()
    }


# ============================================================================
# DREAMCODER RUNNER
# ============================================================================

class ExperimentalDreamCoder:
    """DreamCoder runner for experimental rules."""

    def __init__(
        self,
        tasks: List[Task],
        grammar: Grammar,
        phases: List[PhaseConfig],
        eval_fn: Callable,
        results_dir: str,
        keep_top_k: int = 5,
        log_level: int = 1
    ):
        self.tasks = tasks
        self.initial_grammar = grammar
        self.grammar = copy.deepcopy(grammar)
        self.phases = phases
        self.eval_fn = eval_fn
        self.results_dir = Path(results_dir)
        self.keep_top_k = keep_top_k
        self.log_level = log_level

        # Results tracking
        self.frontiers: Dict[str, TaskFrontier] = {}
        for task in tasks:
            self.frontiers[task.name] = TaskFrontier(task, max_size=keep_top_k)

        self.task_metrics: Dict[str, TaskMetrics] = {}
        for task in tasks:
            self.task_metrics[task.name] = TaskMetrics(task.name, task.family)

        self.iteration_history: List[IterationMetrics] = []

        # Recognition model
        self.recognition = SetTransformerRecognitionModel(
            grammar=grammar,
            d_model=64,
            num_heads=4,
            num_hand_layers=2,
            num_task_layers=2,
            max_examples=20,
            max_cards=8,
            device='cpu'
        )

        # Neural dreamer
        self.dreamer = NeuralDreamer(
            grammar=grammar,
            recognition_model=self.recognition,
            eval_fn=eval_fn
        )

        # Create results directory
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Log file
        self.log_file = self.results_dir / "run.log"

    def log(self, message: str, level: int = 1):
        """Log a message."""
        if level <= self.log_level:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            formatted = f"[{timestamp}] {message}"
            print(formatted, flush=True)
            with open(self.log_file, 'a') as f:
                f.write(formatted + "\n")

    def get_solved_count(self) -> int:
        """Get number of solved tasks."""
        return sum(1 for f in self.frontiers.values() if f.solved)

    def run(self):
        """Run all phases."""
        print_banner("EXPERIMENTAL RULES DREAMCODER")
        self.log(f"Total tasks: {len(self.tasks)}")
        self.log(f"Rules from: catalogue.py")
        self.log(f"Initial grammar size: {len(self.grammar.productions)}")
        self.log(f"Results dir: {self.results_dir}")

        total_start = time.time()

        for phase_idx, phase in enumerate(self.phases):
            print_banner(f"Phase {phase_idx + 1}: {phase.name}")
            self.run_phase(phase_idx, phase)

            # Save checkpoint
            self.save_checkpoint(f"phase_{phase_idx + 1}")

        total_elapsed = time.time() - total_start
        print_banner(f"COMPLETED in {format_time(total_elapsed)}")
        self.log(f"Final solved: {self.get_solved_count()}/{len(self.tasks)}")

        # Save final results
        self.save_final_results()

    def run_phase(self, phase_idx: int, phase: PhaseConfig):
        """Run a single phase."""
        for iteration in range(phase.iterations):
            iter_start = time.time()
            self.log(f"\n--- Phase {phase_idx + 1}, Iteration {iteration + 1}/{phase.iterations} ---", 1)

            # Wake: enumerate programs
            self.log("Wake phase: enumerating programs...", 1)
            self.run_enumeration(phase)

            # Report progress
            solved = self.get_solved_count()
            self.log(f"Solved: {solved}/{len(self.tasks)} tasks", 1)

            # Sleep: train recognition model if we have solutions
            if solved > 0 and phase.recognition_epochs > 0:
                self.log("Sleep phase: training recognition model...", 1)
                self.train_recognition(phase.recognition_epochs)

            # Compress grammar
            if solved > 3:
                self.log("Compressing grammar...", 1)
                self.compress_grammar()

            iter_elapsed = time.time() - iter_start
            self.log(f"Iteration completed in {format_time(iter_elapsed)}", 1)

    def run_enumeration(self, phase: PhaseConfig):
        """Run program enumeration for all unsolved tasks."""
        unsolved_tasks = [t for t in self.tasks if not self.frontiers[t.name].solved]

        if not unsolved_tasks:
            self.log("All tasks solved!", 1)
            return

        self.log(f"Enumerating for {len(unsolved_tasks)} unsolved tasks", 2)

        if USE_MULTIPROCESSING and len(unsolved_tasks) > 1:
            self._enumerate_parallel(unsolved_tasks, phase)
        else:
            self._enumerate_sequential(unsolved_tasks, phase)

    def _enumerate_sequential(self, tasks: List[Task], phase: PhaseConfig):
        """Sequential enumeration."""
        for task in tasks:
            result = enumerate_task_sequential(
                task, self.grammar,
                phase.max_depth, phase.max_programs, phase.timeout_per_task
            )
            self._process_enumeration_result(task, result, phase)

    def _enumerate_parallel(self, tasks: List[Task], phase: PhaseConfig):
        """Parallel enumeration using multiprocessing."""
        import concurrent.futures

        task_data_list = [serialize_task(t) for t in tasks]

        with concurrent.futures.ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            futures = {}
            for task, task_data in zip(tasks, task_data_list):
                future = executor.submit(
                    enumerate_task,
                    task_data, None,
                    phase.max_depth, phase.max_programs, phase.timeout_per_task
                )
                futures[future] = task

            for future in concurrent.futures.as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                    self._process_enumeration_result(task, result, phase)
                except Exception as e:
                    self.log(f"Error enumerating {task.name}: {e}", 1)

    def _process_enumeration_result(self, task: Task, result: Dict, phase: PhaseConfig):
        """Process enumeration result for a task."""
        frontier = self.frontiers[task.name]

        for prog_data in result['programs_found']:
            # Parse program if string
            if isinstance(prog_data.get('program_str'), str):
                try:
                    primitives_dict = {str(p): p for p in self.grammar.primitives()}
                    program = parse_program(prog_data['program_str'], primitives_dict)
                except Exception as e:
                    self.log(f"Could not parse program: {e}", 2)
                    continue
            else:
                program = prog_data['program']

            # Verify on holdout examples if available
            if hasattr(task, 'holdout_examples') and task.holdout_examples:
                try:
                    all_correct = True
                    for hand, expected in task.holdout_examples:
                        result_val = program.evaluate([])(hand)
                        if result_val != expected:
                            all_correct = False
                            break

                    if not all_correct:
                        self.log(f"Rejected spurious: {task.name}", 2)
                        continue
                except Exception:
                    continue

            # Add to frontier
            entry = SolutionEntry(
                program=program,
                log_probability=prog_data['log_probability'],
                log_likelihood=0.0,
                programs_enumerated=prog_data['programs_enumerated'],
                time_found=time.time()
            )
            was_solved_before = frontier.solved
            frontier.add(entry)

            if frontier.solved and not was_solved_before:
                self.log(f"SOLVED: {task.name}", 1)
                tm = self.task_metrics[task.name]
                if not tm.solved:
                    tm.solved = True
                    tm.iteration_solved = len(self.iteration_history)

    def train_recognition(self, epochs: int):
        """Train recognition model on solved tasks."""
        solved_tasks = [t for t in self.tasks if self.frontiers[t.name].solved]
        if not solved_tasks:
            return

        self.recognition.train_on_frontiers(
            solved_tasks,
            self.frontiers,
            epochs=epochs
        )

    def compress_grammar(self):
        """Compress grammar by finding common abstractions."""
        solved_frontiers = {
            name: f for name, f in self.frontiers.items() if f.solved
        }

        if len(solved_frontiers) < 3:
            return

        try:
            # Convert frontiers dict to list of lists format expected by compress_frontiers
            # Each frontier is a list of [(program, log_likelihood), ...]
            frontier_list = []
            for name, f in solved_frontiers.items():
                if f.best:
                    frontier_list.append([(f.best.program, f.best.log_probability)])

            if len(frontier_list) < 3:
                return

            result = compress_frontiers(
                self.grammar,
                frontier_list,
                max_inventions=5,
                min_savings=2.0,
                use_anti_unification=True
            )
            if result and len(result.new_grammar.productions) > len(self.grammar.productions):
                n_new = len(result.new_grammar.productions) - len(self.grammar.productions)
                self.log(f"Added {n_new} new abstractions", 1)
                for inv in result.new_inventions:
                    self.log(f"  New: {inv}", 2)
                self.grammar = result.new_grammar
                self.recognition.grammar = self.grammar
                self.dreamer.grammar = self.grammar
        except Exception as e:
            self.log(f"Compression error: {e}", 2)

    def save_checkpoint(self, name: str):
        """Save checkpoint."""
        checkpoint_dir = self.results_dir / "checkpoints"
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint = {
            'frontiers': {
                name: {
                    'solved': f.solved,
                    'best_program': str(f.best.program) if f.best else None,
                    'best_log_prob': f.best.log_probability if f.best else None
                }
                for name, f in self.frontiers.items()
            },
            'grammar_size': len(self.grammar.productions),
            'solved_count': self.get_solved_count()
        }

        with open(checkpoint_dir / f"{name}.json", 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def save_final_results(self):
        """Save final results."""
        results = {
            'summary': {
                'total_tasks': len(self.tasks),
                'solved_tasks': self.get_solved_count(),
                'final_grammar_size': len(self.grammar.productions)
            },
            'solved_tasks': {},
            'unsolved_tasks': []
        }

        for task in self.tasks:
            frontier = self.frontiers[task.name]
            if frontier.solved and frontier.best:
                results['solved_tasks'][task.name] = {
                    'family': task.family,
                    'program': str(frontier.best.program),
                    'log_probability': frontier.best.log_probability
                }
            else:
                results['unsolved_tasks'].append({
                    'name': task.name,
                    'family': task.family
                })

        with open(self.results_dir / "final_results.json", 'w') as f:
            json.dump(results, f, indent=2)

        self.log(f"Results saved to {self.results_dir / 'final_results.json'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    # Results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(__file__).parent.parent / "results" / f"experimental_rules_{timestamp}"

    # Create tasks from catalogue rules
    print("Loading experimental rules from catalogue.py...")
    all_rules = create_all_rules()
    print(f"Loaded {len(all_rules)} rules")

    # Create tasks with balanced sampling
    # Rules with extreme base rates (can't get balanced examples) will be skipped
    tasks, skipped_rules = create_tasks_from_catalogue(
        rules=all_rules,
        n_examples=100,
        n_holdout=20,
        hand_size=5,
        seed=42,
        balanced=True,
        min_per_class=5  # Require at least 5 examples of each class
    )
    print(f"Created {len(tasks)} tasks ({len(skipped_rules)} rules skipped due to extreme base rates)")

    # Build grammar
    grammar = build_lean_grammar()
    print(f"Grammar has {len(grammar.productions)} primitives")

    # Create evaluation function
    eval_fn = make_eval_fn()

    # Define phases - OVERNIGHT RUN (much larger)
    # Estimated total time: 8-10 hours
    phases = [
        # Phase 1: Quick exploration - find easy tasks fast
        PhaseConfig(
            name="Quick Exploration",
            iterations=5,
            max_depth=7,
            max_programs=100000,
            timeout_per_task=60,
            recognition_epochs=5,
            dream_tasks=10  # Generate dream tasks for harder rules
        ),
        # Phase 2: Medium search - get more coverage
        PhaseConfig(
            name="Medium Search",
            iterations=10,
            max_depth=9,
            max_programs=250000,
            timeout_per_task=120,
            recognition_epochs=8,
            dream_tasks=20
        ),
        # Phase 3: Deep search - find complex rules
        PhaseConfig(
            name="Deep Search",
            iterations=15,
            max_depth=11,
            max_programs=500000,
            timeout_per_task=180,
            recognition_epochs=10,
            dream_tasks=30
        ),
        # Phase 4: Extended deep search - maximum effort
        PhaseConfig(
            name="Extended Deep Search",
            iterations=20,
            max_depth=13,
            max_programs=1000000,
            timeout_per_task=300,
            recognition_epochs=10,
            dream_tasks=50
        ),
        # Phase 5: Final exhaustive search
        PhaseConfig(
            name="Exhaustive Search",
            iterations=15,
            max_depth=15,
            max_programs=2000000,
            timeout_per_task=600,
            recognition_epochs=15,
            dream_tasks=50
        )
    ]

    # Run DreamCoder
    dc = ExperimentalDreamCoder(
        tasks=tasks,
        grammar=grammar,
        phases=phases,
        eval_fn=eval_fn,
        results_dir=str(results_dir),
        log_level=2
    )

    dc.run()


if __name__ == "__main__":
    main()
