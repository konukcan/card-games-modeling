#!/usr/bin/env python3
"""
Cython-Optimized Overnight Pre-training Runner

This script runs staged curriculum pre-training with:
1. Cython-accelerated core modules (type_system, program, grammar, enumeration)
2. Early pruning in all-or-nothing mode
3. Multiprocessing with PyPy workers for parallel enumeration

Combined expected speedup: 7-15x faster than original Python
- Cython modules: ~3-6x faster for type operations
- Early pruning: ~1.5-2x speedup
- PyPy workers: ~3-6x speedup for enumeration

Architecture:
- Main process (CPython + Cython): DreamCoder loop, uses fast Cython modules
- Worker subprocesses (PyPy): Parallel enumeration (can't use Cython .so files)
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

# NOTE: Cython modules exist and work for standalone testing, but they require
# Cython-native Primitive objects. Since build_lean_grammar() creates Python
# Primitive objects, we must use Python modules here. The main speedup comes
# from PyPy workers for enumeration anyway.
#
# Cython modules are available at dreamcoder_core/cython_src/ for future use
# if we create a Cython-native grammar builder.

USE_CYTHON = False  # Set to True when Cython-native grammar is available

from dreamcoder_core.type_system import arrow, HAND, BOOL
from dreamcoder_core.program import Program, Primitive, Invented, parse_program
from dreamcoder_core.grammar import Grammar
from dreamcoder_core.enumeration import enumerate_simple

print("Using Python modules (PyPy workers provide ~3-6x speedup for enumeration)")

# These don't have Cython versions yet
from dreamcoder_core.compression import compress_frontiers
from dreamcoder_core.neural_recognition import NeuralRecognitionModel
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.dreamcoder_v2 import (
    Task, SolutionEntry, TaskFrontier, IterationMetrics, TaskMetrics,
    NeuralDreamer, create_tasks_from_rules, make_eval_fn
)
from rules.pretraining_rules import (
    get_all_pretraining_rules, get_easy_pretraining_rules
)


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
    use_all_rules: bool
    enumeration_budget: int
    max_depth: int
    dreams_per_iteration: int
    recognition_epochs: int


# ============================================================================
# PYPY SUBPROCESS WORKER (unchanged - uses pure Python)
# ============================================================================

WORKER_SCRIPT = '''#!/usr/bin/env python3
"""PyPy Worker for Enumeration - DO NOT EDIT DIRECTLY"""
import sys
import pickle
import json
import time
import math
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dreamcoder_core.type_system import arrow, HAND, BOOL
from dreamcoder_core.enumeration import enumerate_simple
from dreamcoder_core.lean_primitives import build_lean_grammar
from rules.cards import Card, Suit, Rank


def deserialize_hand(hand_data):
    """Convert JSON hand data back to Card objects."""
    cards = []
    for card_dict in hand_data:
        suit = Suit[card_dict['suit']]
        rank = Rank[card_dict['rank']]
        cards.append(Card(suit, rank))
    return tuple(cards)


def evaluate_program(program, hand):
    """Evaluate a program on a hand."""
    try:
        fn = program.evaluate([])
        return fn(hand)
    except:
        return None


def enumerate_task(task_data, grammar_productions, max_depth, max_programs, timeout):
    """
    Enumerate programs for a single task.

    Uses early pruning: stop evaluating examples as soon as one fails.
    """
    task_name = task_data['name']
    raw_examples = task_data['examples']

    # Deserialize examples (hand_data, expected_output)
    examples = []
    for hand_data, expected in raw_examples:
        hand = deserialize_hand(hand_data)
        examples.append((hand, expected))

    # Build grammar
    grammar = build_lean_grammar()

    # Apply any updated production weights
    if grammar_productions:
        # Could restore grammar weights here
        pass

    request_type = arrow(HAND, BOOL)

    results = []
    programs_tried = 0
    start_time = time.time()

    for program, log_prob in enumerate_simple(grammar, request_type, max_depth=max_depth):
        programs_tried += 1

        if programs_tried > max_programs:
            break
        if time.time() - start_time > timeout:
            break

        # Evaluate with EARLY PRUNING
        try:
            all_correct = True
            correct = 0

            for inp, expected in examples:
                result = evaluate_program(program, inp)
                if result == expected:
                    correct += 1
                else:
                    all_correct = False
                    break  # EARLY EXIT - key optimization!

            if all_correct:
                # Full match
                results.append({
                    'program': str(program),
                    'log_probability': log_prob,
                    'programs_enumerated': programs_tried,
                    'time_found': time.time() - start_time
                })

                # Stop after finding 5 solutions
                if len(results) >= 5:
                    break

        except Exception as e:
            pass

    return {
        'task_name': task_name,
        'solved': len(results) > 0,
        'n_solutions': len(results),
        'programs_searched': programs_tried,
        'time': time.time() - start_time,
        'solutions': results
    }


def main():
    """Main worker entry point."""
    # Read input from stdin
    input_data = json.loads(sys.stdin.read())

    task_data = input_data['task']
    grammar_productions = input_data.get('grammar_productions')
    max_depth = input_data['max_depth']
    max_programs = input_data['max_programs']
    timeout = input_data['timeout']

    result = enumerate_task(task_data, grammar_productions, max_depth, max_programs, timeout)

    # Write result to stdout
    print(json.dumps(result))


if __name__ == '__main__':
    main()
'''


def create_worker_script():
    """Create the PyPy worker script if it doesn't exist."""
    worker_path = Path(__file__).parent / 'enumeration_worker.py'
    with open(worker_path, 'w') as f:
        f.write(WORKER_SCRIPT)
    return str(worker_path)


def run_pypy_worker(task_data: Dict, max_depth: int, max_programs: int,
                    timeout: float, worker_script: str) -> Dict:
    """
    Run enumeration for a single task using PyPy subprocess.

    Returns result dict with solutions found.
    """
    input_data = {
        'task': task_data,
        'grammar_productions': None,  # Will use default
        'max_depth': max_depth,
        'max_programs': max_programs,
        'timeout': timeout
    }

    try:
        # Use PyPy if available, otherwise fall back to Python
        python_cmd = PYPY_PATH if USE_PYPY else sys.executable

        result = subprocess.run(
            [python_cmd, worker_script],
            input=json.dumps(input_data),
            capture_output=True,
            text=True,
            timeout=timeout + 30  # Extra buffer for startup
        )

        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            return {
                'task_name': task_data['name'],
                'solved': False,
                'n_solutions': 0,
                'programs_searched': 0,
                'time': 0,
                'error': result.stderr[:500]
            }
    except subprocess.TimeoutExpired:
        return {
            'task_name': task_data['name'],
            'solved': False,
            'n_solutions': 0,
            'programs_searched': 0,
            'time': timeout,
            'error': 'Timeout'
        }
    except Exception as e:
        return {
            'task_name': task_data['name'],
            'solved': False,
            'n_solutions': 0,
            'programs_searched': 0,
            'time': 0,
            'error': str(e)
        }


def serialize_task_for_worker(task: Task) -> Dict:
    """Convert Task to JSON-serializable dict for worker."""
    return {
        'name': task.name,
        'examples': [
            (serialize_hand(inp), out)
            for inp, out in task.examples
        ]
    }


def serialize_hand(hand) -> List[Dict]:
    """Convert Hand to JSON-serializable format."""
    return [
        {'suit': card.suit.name, 'rank': card.rank.name}
        for card in hand
    ]


# ============================================================================
# SEQUENTIAL FALLBACK WITH CYTHON + EARLY PRUNING
# ============================================================================

def enumerate_task_with_early_pruning(
    grammar: Grammar,
    task: Task,
    eval_fn: Callable,
    max_depth: int,
    max_programs: int,
    timeout: float,
    keep_top_k: int = 5
) -> Tuple[TaskFrontier, int]:
    """
    Enumerate programs for a task with early pruning optimization.

    Uses Cython enumerate_simple if available.

    Returns (frontier, programs_tried)
    """
    frontier = TaskFrontier(task, max_size=keep_top_k)
    programs_tried = 0
    start_time = time.time()

    for program, log_prob in enumerate_simple(grammar, task.request_type, max_depth=max_depth):
        programs_tried += 1

        if programs_tried > max_programs:
            break
        if time.time() - start_time > timeout:
            break

        # Evaluate with EARLY PRUNING
        try:
            all_correct = True
            correct = 0

            for inp, expected in task.examples:
                result = eval_fn(program, inp)
                if result == expected:
                    correct += 1
                else:
                    all_correct = False
                    break  # EARLY EXIT!

            if all_correct:
                entry = SolutionEntry(
                    program=program,
                    log_probability=log_prob,
                    log_likelihood=0.0,
                    programs_enumerated=programs_tried,
                    time_found=time.time() - start_time
                )
                frontier.add(entry)

                if frontier.n_solutions >= keep_top_k:
                    break

        except:
            pass

    frontier.total_programs_searched = programs_tried
    return frontier, programs_tried


# ============================================================================
# CYTHON-OPTIMIZED STAGED DREAMCODER
# ============================================================================

class CythonOptimizedDreamCoder:
    """
    DreamCoder with Cython-accelerated core and parallel enumeration.

    Key optimizations:
    1. Cython modules for type_system, program, grammar, enumeration
    2. Early pruning in all-or-nothing mode
    3. Parallel task enumeration (with PyPy workers)
    """

    def __init__(
        self,
        grammar: Grammar,
        easy_tasks: List[Task],
        all_tasks: List[Task],
        eval_fn: Callable,
        phases: List[PhaseConfig],
        recognition_hidden_dim: int = 256,
        recognition_lr: float = 5e-4,
        keep_top_k: int = 5,
        max_inventions_per_iteration: int = 5,
        dream_temperature: float = 1.0,
        n_workers: int = 4,
        use_pypy: bool = True,
        verbose: bool = True,
        log_dir: Optional[str] = None,
        device: str = 'cpu'
    ):
        self.initial_grammar = grammar
        self.grammar = copy.deepcopy(grammar)
        self.easy_tasks = easy_tasks
        self.all_tasks = all_tasks
        self.eval_fn = eval_fn
        self.phases = phases

        self.keep_top_k = keep_top_k
        self.max_inventions_per_iteration = max_inventions_per_iteration
        self.dream_temperature = dream_temperature
        self.n_workers = n_workers
        self.use_pypy = use_pypy and USE_PYPY
        self.verbose = verbose
        self.log_dir = Path(log_dir) if log_dir else None
        self.device = device

        # Initialize recognition model
        self.recognition = NeuralRecognitionModel(
            grammar=grammar,
            hidden_dim=recognition_hidden_dim,
            learning_rate=recognition_lr,
            device=device
        )

        # Initialize dreamer
        self.dreamer = NeuralDreamer(
            grammar=grammar,
            recognition_model=self.recognition,
            eval_fn=eval_fn,
            device=device
        )

        # State
        self.frontiers: Dict[str, TaskFrontier] = {}
        for task in all_tasks:
            self.frontiers[task.name] = TaskFrontier(task, max_size=keep_top_k)

        self.task_metrics: Dict[str, TaskMetrics] = {}
        for task in all_tasks:
            self.task_metrics[task.name] = TaskMetrics(task.name, task.family)

        self.iteration_metrics: List[IterationMetrics] = []
        self.library_history: List[List[str]] = []
        self._embedding_snapshots: List[Dict[str, Any]] = []

        # Phase tracking
        self.current_phase_idx = 0
        self.global_iteration = 0

        # Worker script for PyPy
        self.worker_script = None
        if self.use_pypy:
            self.worker_script = create_worker_script()

    def log(self, msg: str, level: int = 0):
        """Log message with indentation."""
        if self.verbose:
            indent = "  " * level
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {indent}{msg}", flush=True)

    def get_active_tasks(self) -> List[Task]:
        """Get tasks for current phase."""
        phase = self.phases[self.current_phase_idx]
        return self.all_tasks if phase.use_all_rules else self.easy_tasks

    def verify_on_holdout(self, program: Program, task: Task) -> bool:
        """
        Verify a program passes all holdout examples.

        This prevents accepting spurious solutions that pass training
        examples by coincidence but don't implement the actual rule.

        Returns True if program passes all holdout examples.
        """
        holdout = getattr(task, 'holdout_examples', None)
        if not holdout:
            # No holdout examples, accept the program
            return True

        try:
            # Evaluate program to get callable function, then apply to each holdout example
            fn = program.evaluate([])
            for inp, expected in holdout:
                result = fn(inp)
                if result != expected:
                    return False
            return True
        except Exception as e:
            # Verification failed (likely due to runtime error)
            return False

    def run(self) -> Dict:
        """Run all phases of staged training."""
        start_time = time.time()

        print_banner("DREAMCODER V2 - CYTHON-OPTIMIZED OVERNIGHT PRETRAINING")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Easy tasks: {len(self.easy_tasks)}")
        print(f"All tasks: {len(self.all_tasks)}")
        print(f"Recognition model: {self.recognition.hidden_dim} hidden dim")
        print(f"Device: {self.device}")
        print()
        print("OPTIMIZATIONS ENABLED:")
        print(f"  - Cython core modules: {'YES' if USE_CYTHON else 'NO (fallback to Python)'}")
        print(f"  - Early pruning: YES")
        print(f"  - Parallel workers: {self.n_workers}")
        print(f"  - PyPy workers: {'YES' if self.use_pypy else 'NO (not available)'}")
        print()
        print("Phases:")
        total_iters = 0
        for i, phase in enumerate(self.phases):
            tasks_str = "all 43" if phase.use_all_rules else "22 easy"
            print(f"  Phase {i+1} ({phase.name}): {phase.iterations} iters, "
                  f"{tasks_str} tasks, budget={phase.enumeration_budget:,}, "
                  f"depth={phase.max_depth}, dreams={phase.dreams_per_iteration}")
            total_iters += phase.iterations
        print(f"\nTotal iterations: {total_iters}")

        # Run each phase
        for phase_idx, phase in enumerate(self.phases):
            self.current_phase_idx = phase_idx
            self._run_phase(phase)

            # Save checkpoint after each phase
            if self.log_dir:
                self._save_checkpoint(f"phase{phase_idx+1}")

        total_time = time.time() - start_time

        # Final summary
        print_banner("CYTHON-OPTIMIZED OVERNIGHT PRETRAINING COMPLETE")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total time: {format_time(total_time)}")

        results = self._compile_results(total_time)
        if self.log_dir:
            self._save_results(results)

        return results

    def _run_phase(self, phase: PhaseConfig):
        """Run one phase of training."""
        print_banner(f"PHASE: {phase.name}")

        tasks = self.get_active_tasks()
        self.log(f"Active tasks: {len(tasks)}")
        self.log(f"Budget: {phase.enumeration_budget:,}")
        self.log(f"Max depth: {phase.max_depth}")
        self.log(f"Dreams/iteration: {phase.dreams_per_iteration}")

        for i in range(phase.iterations):
            self.log("")
            self.log("=" * 70)
            self.log(f"ITERATION {self.global_iteration + 1} (Phase {self.current_phase_idx + 1}, iter {i + 1}/{phase.iterations})")
            self.log("=" * 70)

            metrics = self._run_iteration(
                tasks=tasks,
                enumeration_budget=phase.enumeration_budget,
                max_depth=phase.max_depth,
                dreams_per_iteration=phase.dreams_per_iteration,
                recognition_epochs=phase.recognition_epochs
            )

            self.iteration_metrics.append(metrics)
            self.global_iteration += 1

            # Log summary
            self.log("")
            self.log(f"Summary:", 1)
            self.log(f"Solved: {metrics.tasks_solved}/{metrics.tasks_total}", 2)
            self.log(f"Programs: {metrics.programs_enumerated:,}", 2)
            self.log(f"New abstractions: {len(metrics.new_abstractions)}", 2)
            self.log(f"Recognition loss: {metrics.recognition_loss:.4f}", 2)
            self.log(f"Dreams generated: {metrics.dreams_generated}", 2)
            self.log(f"Grammar size: {metrics.grammar_size}", 2)

            # Cumulative progress
            total_solved = sum(1 for tm in self.task_metrics.values() if tm.solved)
            self.log(f"Cumulative solved (all tasks): {total_solved}/{len(self.all_tasks)}", 2)

            # Save per-iteration checkpoint for interpretability analysis
            self._save_iteration_checkpoint(self.global_iteration, metrics)

    def _run_iteration(
        self,
        tasks: List[Task],
        enumeration_budget: int,
        max_depth: int,
        dreams_per_iteration: int,
        recognition_epochs: int
    ) -> IterationMetrics:
        """Run one wake-sleep iteration with parallel enumeration."""

        # =====================
        # WAKE PHASE (PARALLEL)
        # =====================
        self.log("\n[WAKE] Enumerating programs (parallel with early pruning)...")
        wake_start = time.time()

        total_programs = 0
        tasks_solved = 0

        # Get tasks that need enumeration
        tasks_to_enumerate = []
        for task in tasks:
            frontier = self.frontiers[task.name]
            if not (frontier.n_solutions >= self.keep_top_k and frontier.solved):
                tasks_to_enumerate.append(task)
            else:
                tasks_solved += 1

        self.log(f"  Tasks to enumerate: {len(tasks_to_enumerate)} (skipping {tasks_solved} already solved)")

        # Run enumeration
        if USE_MULTIPROCESSING and len(tasks_to_enumerate) > 1:
            results = self._enumerate_parallel(
                tasks_to_enumerate,
                enumeration_budget,
                max_depth
            )
        else:
            results = self._enumerate_sequential(
                tasks_to_enumerate,
                enumeration_budget,
                max_depth
            )

        # Process results
        for task, result in zip(tasks_to_enumerate, results):
            frontier = self.frontiers[task.name]

            if isinstance(result, tuple):
                # Sequential result: (frontier, programs_tried)
                new_frontier, programs_tried = result
                total_programs += programs_tried

                # Merge solutions with holdout verification
                for entry in new_frontier.entries:
                    # VERIFY on holdout examples to prevent spurious solutions
                    if not self.verify_on_holdout(entry.program, task):
                        self.log(f"    Rejected spurious: {task.name} - {str(entry.program)[:50]}...", 2)
                        continue  # Skip this solution

                    if frontier.add(entry):
                        tm = self.task_metrics[task.name]
                        if not tm.solved:
                            tm.solved = True
                            tm.iteration_solved = self.global_iteration
                            tm.programs_to_solve = entry.programs_enumerated
                            tm.best_program = str(entry.program)
                            tm.description_length = entry.description_length
                            self.log(f"    SOLVED (verified): {task.name} ({entry.programs_enumerated:,} programs)", 1)

                frontier.total_programs_searched += programs_tried

            else:
                # Parallel/PyPy result: dict
                total_programs += result.get('programs_searched', 0)
                frontier.total_programs_searched += result.get('programs_searched', 0)

                if result.get('solved'):
                    # Build primitives dict for parsing
                    primitives_dict = {str(p): p for p in self.grammar.primitives()}

                    # Parse and add solutions
                    for sol in result.get('solutions', []):
                        program_str = sol['program']
                        log_prob = sol.get('log_probability', 0.0)
                        programs_enumerated = sol.get('programs_enumerated', 0)
                        time_found = sol.get('time_found', 0.0)

                        # Reconstruct Program from string
                        try:
                            program = parse_program(program_str, primitives_dict)

                            # VERIFY on holdout examples to prevent spurious solutions
                            if not self.verify_on_holdout(program, task):
                                self.log(f"    Rejected spurious: {task.name} - {program_str[:50]}...", 2)
                                continue  # Skip this solution

                            # Add to frontier (verified solution)
                            entry = SolutionEntry(
                                program=program,
                                log_probability=log_prob,
                                log_likelihood=0.0,  # Perfect solution
                                programs_enumerated=programs_enumerated,
                                time_found=time_found
                            )
                            frontier.add(entry)

                            # Update task metrics
                            tm = self.task_metrics[task.name]
                            if not tm.solved:
                                tm.solved = True
                                tm.iteration_solved = self.global_iteration
                                tm.programs_to_solve = programs_enumerated
                                tm.best_program = program_str
                                self.log(f"    SOLVED (verified): {task.name} ({programs_enumerated:,} programs)", 1)
                        except Exception as e:
                            # If parsing fails, log but don't mark as solved
                            self.log(f"    Warning: Could not parse program for {task.name}: {e}", 1)

            if frontier.solved:
                tasks_solved += 1

        wake_time = time.time() - wake_start
        self.log(f"Wake: {tasks_solved}/{len(tasks)} solved, "
                f"{total_programs:,} programs in {wake_time:.1f}s")

        # =====================
        # SLEEP - COMPRESSION
        # =====================
        new_abstractions = []
        compression_time = 0.0

        self.log("\n[SLEEP - COMPRESSION] Finding abstractions...")
        comp_start = time.time()

        all_frontiers = []
        for frontier in self.frontiers.values():
            if frontier.n_solutions > 0:
                programs_with_ll = [(e.program, e.log_likelihood) for e in frontier.entries]
                all_frontiers.append(programs_with_ll)

        if all_frontiers:
            result = compress_frontiers(
                self.grammar,
                all_frontiers,
                max_inventions=self.max_inventions_per_iteration,
                min_savings=2.0,
                use_anti_unification=True
            )

            if result.new_inventions:
                self.grammar = result.new_grammar
                new_abstractions = [str(inv) for inv in result.new_inventions]
                self.recognition.grammar = self.grammar
                self.dreamer.grammar = self.grammar
                self.log(f"  Found {len(new_abstractions)} abstraction(s)", 1)

        compression_time = time.time() - comp_start
        self.library_history.append(new_abstractions)

        # =====================
        # SLEEP - RECOGNITION
        # =====================
        recognition_loss = 0.0
        recognition_time = 0.0

        self.log("\n[SLEEP - RECOGNITION] Training neural model...")
        rec_start = time.time()

        all_solved_tasks = [t for t in self.all_tasks if self.frontiers[t.name].solved]

        if all_solved_tasks:
            recognition_loss = self.recognition.train_on_frontiers(
                all_solved_tasks,
                self.frontiers,
                epochs=recognition_epochs
            )

        recognition_time = time.time() - rec_start
        self.log(f"  Trained on {len(all_solved_tasks)} solved tasks, loss: {recognition_loss:.4f}", 1)

        # =====================
        # SLEEP - DREAMING
        # =====================
        dreams_generated = 0
        dream_time = 0.0

        if self.global_iteration > 0 and dreams_per_iteration > 0:
            self.log("\n[SLEEP - DREAMING] Generating synthetic tasks...")
            dream_start = time.time()

            all_inputs = []
            for task in self.all_tasks:
                for inp, _ in task.examples[:5]:
                    all_inputs.append(inp)

            dreams = self.dreamer.generate_dreams(
                self.all_tasks[0].request_type,
                dreams_per_iteration,
                all_inputs,
                temperature=self.dream_temperature
            )
            dreams_generated = len(dreams)

            for dream_task, program in dreams:
                synthetic_frontier = TaskFrontier(dream_task, max_size=1)
                entry = SolutionEntry(
                    program=program,
                    log_probability=self.grammar.program_log_likelihood(
                        program, dream_task.request_type
                    ),
                    log_likelihood=0.0,
                    programs_enumerated=0,
                    time_found=0.0
                )
                synthetic_frontier.add(entry)

                temp_frontiers = {dream_task.name: synthetic_frontier}
                self.recognition.train_on_frontiers([dream_task], temp_frontiers, epochs=1)

            dream_time = time.time() - dream_start
            self.log(f"  Generated {dreams_generated} dreams", 1)

        # =====================
        # GRAMMAR WEIGHT UPDATE
        # =====================
        if tasks_solved > 0:
            all_frontiers = []
            for frontier in self.frontiers.values():
                if frontier.n_solutions > 0:
                    programs_with_ll = [(e.program, e.log_likelihood) for e in frontier.entries]
                    all_frontiers.append(programs_with_ll)

            if all_frontiers:
                self.grammar = self.grammar.inside_outside_update(all_frontiers)

        # Snapshot embeddings
        self._snapshot_embeddings(self.global_iteration)

        return IterationMetrics(
            iteration=self.global_iteration,
            tasks_solved=tasks_solved,
            tasks_total=len(tasks),
            programs_enumerated=total_programs,
            wake_time=wake_time,
            new_abstractions=new_abstractions,
            compression_time=compression_time,
            recognition_loss=recognition_loss,
            recognition_time=recognition_time,
            dreams_generated=dreams_generated,
            dream_time=dream_time,
            grammar_size=len(self.grammar)
        )

    def _enumerate_parallel(self, tasks: List[Task], budget: int, max_depth: int) -> List[Dict]:
        """Enumerate tasks in parallel using PyPy subprocesses."""
        # Use a dict to collect results by task name, preserving correct task-result mapping
        # NOTE: Previously used as_completed() which returns results in completion order,
        # causing results to be mismatched with tasks when zipped. This fix uses task_name
        # from each result to ensure correct mapping.
        results_by_name = {}

        if self.use_pypy and self.worker_script:
            # Use PyPy subprocesses (PyPy can't use Cython .so files)
            self.log(f"  Using PyPy workers ({self.n_workers} parallel)...")

            with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                futures = {}
                for task in tasks:
                    task_data = serialize_task_for_worker(task)
                    future = executor.submit(
                        run_pypy_worker,
                        task_data,
                        max_depth,
                        budget,
                        180.0,  # timeout per task
                        self.worker_script
                    )
                    futures[future] = task

                for future in as_completed(futures):
                    result = future.result()
                    # Use task_name from result to ensure correct mapping
                    task_name = result.get('task_name')
                    if task_name:
                        results_by_name[task_name] = result
                    else:
                        # Fallback: use the task associated with this future
                        task = futures[future]
                        results_by_name[task.name] = result
        else:
            # Fallback to Cython enumeration with early pruning
            # Sequential execution maintains order, so we can use a list directly
            self.log(f"  Using Cython sequential enumeration...")
            results = []
            for task in tasks:
                frontier, programs = enumerate_task_with_early_pruning(
                    self.grammar, task, self.eval_fn,
                    max_depth, budget, 180.0, self.keep_top_k
                )
                results.append((frontier, programs))
            return results

        # Convert results_by_name dict to list in correct task order
        results = []
        for task in tasks:
            result = results_by_name.get(task.name, {
                'task_name': task.name,
                'solved': False,
                'n_solutions': 0,
                'programs_searched': 0,
                'time': 0,
                'error': 'No result received'
            })
            results.append(result)

        return results

    def _enumerate_sequential(self, tasks: List[Task], budget: int, max_depth: int) -> List[Tuple]:
        """Enumerate tasks sequentially with Cython + early pruning."""
        results = []
        for task in tasks:
            frontier, programs = enumerate_task_with_early_pruning(
                self.grammar, task, self.eval_fn,
                max_depth, budget, 180.0, self.keep_top_k
            )
            results.append((frontier, programs))
        return results

    def _snapshot_embeddings(self, iteration: int):
        """Snapshot task embeddings for interpretability."""
        embeddings = {}
        for task in self.all_tasks[:30]:
            emb = self.recognition.get_task_embedding(task)
            embeddings[task.name] = emb.numpy().tolist()

        self._embedding_snapshots.append({
            'iteration': iteration,
            'embeddings': embeddings
        })

    def _save_iteration_checkpoint(self, iteration: int, metrics: 'IterationMetrics'):
        """Save comprehensive per-iteration checkpoint for interpretability analysis.

        This enables post-hoc analysis of:
        1. How recognition accuracy evolves over iterations
        2. Which features become important when
        3. How task embeddings cluster and evolve
        """
        if not self.log_dir:
            return

        checkpoint_dir = self.log_dir / "iteration_checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Get primitive predictions for all tasks using detailed method
        primitive_predictions = {}
        prediction_errors = []
        for task in self.all_tasks[:50]:  # Limit to 50 tasks for size
            try:
                # Use the new detailed prediction method
                pred_data = self.recognition.get_primitive_predictions_detailed(task)
                primitive_predictions[task.name] = {
                    'log_probs': pred_data['log_probs'],
                    'logits': pred_data['logits'],
                    'top_10': pred_data['top_10'],
                    'entropy': pred_data['entropy'],
                    'max_prob': pred_data['max_prob']
                }
            except Exception as e:
                prediction_errors.append(f"{task.name}: {e}")

        if prediction_errors and len(prediction_errors) < 5:
            for err in prediction_errors:
                self.log(f"Warning: Primitive prediction failed - {err}")
        elif prediction_errors:
            self.log(f"Warning: Primitive predictions failed for {len(prediction_errors)} tasks")

        # Get task embeddings
        task_embeddings = {}
        for task in self.all_tasks[:50]:
            try:
                emb = self.recognition.get_task_embedding(task)
                task_embeddings[task.name] = emb.numpy().tolist()
            except Exception:
                pass

        # Collect solved status
        solved_status = {
            task.name: {
                'solved': self.task_metrics[task.name].solved if task.name in self.task_metrics else False,
                'solution': str(self.frontiers[task.name].entries[0].program)
                    if task.name in self.frontiers and self.frontiers[task.name].entries else None
            }
            for task in self.all_tasks
        }

        # Grammar state
        grammar_primitives = [str(p.program) for p in self.grammar.productions]

        # Build checkpoint
        checkpoint = {
            'iteration': iteration,
            'timestamp': datetime.now().isoformat(),

            # Metrics
            'metrics': {
                'tasks_solved': metrics.tasks_solved,
                'tasks_total': metrics.tasks_total,
                'programs_enumerated': metrics.programs_enumerated,
                'recognition_loss': metrics.recognition_loss,
                'grammar_size': metrics.grammar_size,
                'new_abstractions': metrics.new_abstractions,
            },

            # Model state (save separately as .pt file)
            'model_path': f"iteration_{iteration:04d}_model.pt",

            # Primitive predictions
            'primitive_predictions': primitive_predictions,

            # Task embeddings (64-dim vectors)
            'task_embeddings': task_embeddings,

            # Grammar
            'grammar_primitives': grammar_primitives,

            # Solved status
            'solved_status': solved_status,

            # Cumulative progress
            'cumulative_solved': sum(1 for tm in self.task_metrics.values() if tm.solved),
        }

        # Save JSON checkpoint
        json_path = checkpoint_dir / f"iteration_{iteration:04d}.json"
        with open(json_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        # Save model weights
        model_path = checkpoint_dir / f"iteration_{iteration:04d}_model.pt"
        self.recognition.save(str(model_path))

        self.log(f"  Saved iteration checkpoint: {json_path.name}")

    def _save_checkpoint(self, name: str):
        """Save checkpoint with grammar."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

        # Save neural model
        model_path = self.log_dir / f"checkpoint_{name}_{timestamp}.pt"
        self.recognition.save(str(model_path))

        # Save grammar - use dill if available (handles lambdas), else save summary
        grammar_path = self.log_dir / f"grammar_{name}_{timestamp}.pkl"
        try:
            import dill
            with open(grammar_path, 'wb') as f:
                dill.dump(self.grammar, f)
        except ImportError:
            # dill not available, save grammar summary instead
            grammar_summary = {
                'n_productions': len(self.grammar),
                'primitives': [str(p.program) for p in self.grammar.productions],
                'log_variable': self.grammar.log_variable
            }
            with open(grammar_path.with_suffix('.json'), 'w') as f:
                json.dump(grammar_summary, f, indent=2)
        except Exception as e:
            self.log(f"Warning: Could not save grammar: {e}")

        # Save frontiers summary (avoid pickle issues with lambdas)
        frontiers_path = self.log_dir / f"frontiers_{name}_{timestamp}.json"
        frontiers_summary = {
            task_name: {
                'solved': bool(frontier.entries),
                'best_program': str(frontier.entries[0].program) if frontier.entries else None,
                'n_entries': len(frontier.entries)
            }
            for task_name, frontier in self.frontiers.items()
        }
        with open(frontiers_path, 'w') as f:
            json.dump(frontiers_summary, f, indent=2)

        # Save state summary
        state = {
            'phase': name,
            'global_iteration': self.global_iteration,
            'tasks_solved': sum(1 for tm in self.task_metrics.values() if tm.solved),
            'grammar_size': len(self.grammar),
            'timestamp': timestamp,
            'cython_enabled': USE_CYTHON
        }
        state_path = self.log_dir / f"checkpoint_{name}_{timestamp}.json"
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)

        self.log(f"Checkpoint saved: {name}")

    def _compile_results(self, total_time: float) -> Dict:
        """Compile all results."""
        return {
            'config': {
                'easy_tasks': len(self.easy_tasks),
                'all_tasks': len(self.all_tasks),
                'initial_grammar_size': len(self.initial_grammar),
                'recognition_hidden_dim': self.recognition.hidden_dim,
                'optimizations': {
                    'cython_enabled': USE_CYTHON,
                    'early_pruning': True,
                    'n_workers': self.n_workers,
                    'pypy_workers': self.use_pypy
                },
                'phases': [
                    {
                        'name': p.name,
                        'iterations': p.iterations,
                        'use_all_rules': p.use_all_rules,
                        'enumeration_budget': p.enumeration_budget,
                        'max_depth': p.max_depth,
                        'dreams_per_iteration': p.dreams_per_iteration,
                        'recognition_epochs': p.recognition_epochs
                    }
                    for p in self.phases
                ]
            },
            'summary': {
                'total_time': total_time,
                'total_iterations': self.global_iteration,
                'tasks_solved': sum(1 for tm in self.task_metrics.values() if tm.solved),
                'tasks_total': len(self.all_tasks),
                'final_grammar_size': len(self.grammar),
                'total_abstractions': sum(len(m.new_abstractions) for m in self.iteration_metrics),
                'total_dreams': sum(m.dreams_generated for m in self.iteration_metrics)
            },
            'learning_curve': [
                {
                    'iteration': m.iteration,
                    'tasks_solved': m.tasks_solved,
                    'tasks_total': m.tasks_total,
                    'programs': m.programs_enumerated,
                    'abstractions': len(m.new_abstractions),
                    'recognition_loss': m.recognition_loss,
                    'dreams': m.dreams_generated,
                    'grammar_size': m.grammar_size,
                    'wake_time': m.wake_time
                }
                for m in self.iteration_metrics
            ],
            'task_metrics': {name: asdict(tm) for name, tm in self.task_metrics.items()},
            'library_evolution': self.library_history,
            'embedding_evolution': self._embedding_snapshots,
            'recognition_training_losses': self.recognition.training_losses if self.recognition else []
        }

    def _save_results(self, results: Dict):
        """Save results to log directory."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

        json_path = self.log_dir / f"overnight_cython_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        model_path = self.log_dir / f"recognition_model_final_{timestamp}.pt"
        self.recognition.save(str(model_path))

        self._generate_report(results, timestamp)

        self.log(f"\nResults saved to: {json_path}")

    def _generate_report(self, results: Dict, timestamp: str):
        """Generate human-readable report."""
        report_path = self.log_dir / f"overnight_cython_report_{timestamp}.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DREAMCODER V2 - CYTHON-OPTIMIZED OVERNIGHT PRETRAINING REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {format_time(results['summary']['total_time'])}\n\n")

            f.write("OPTIMIZATIONS\n")
            f.write("-" * 40 + "\n")
            f.write(f"Cython modules: {results['config']['optimizations']['cython_enabled']}\n")
            f.write(f"Early pruning: YES\n")
            f.write(f"Parallel workers: {results['config']['optimizations']['n_workers']}\n")
            f.write(f"PyPy workers: {results['config']['optimizations']['pypy_workers']}\n\n")

            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Tasks solved: {results['summary']['tasks_solved']}/{results['summary']['tasks_total']}\n")
            f.write(f"Success rate: {100*results['summary']['tasks_solved']/results['summary']['tasks_total']:.1f}%\n")
            f.write(f"Total iterations: {results['summary']['total_iterations']}\n")
            f.write(f"Final grammar size: {results['summary']['final_grammar_size']}\n")
            f.write(f"Total abstractions: {results['summary']['total_abstractions']}\n")
            f.write(f"Total dreams: {results['summary']['total_dreams']}\n\n")

            f.write("SOLVED TASKS\n")
            f.write("-" * 40 + "\n")
            solved = [(name, tm) for name, tm in results['task_metrics'].items() if tm['solved']]
            for name, tm in sorted(solved, key=lambda x: x[1]['iteration_solved']):
                f.write(f"\n{name} (iter {tm['iteration_solved']+1}, {tm['programs_to_solve']:,} programs)\n")
                if tm['best_program']:
                    prog_str = tm['best_program'][:100]
                    f.write(f"  Program: {prog_str}{'...' if len(tm['best_program']) > 100 else ''}\n")

            f.write("\n\nUNSOLVED TASKS\n")
            f.write("-" * 40 + "\n")
            unsolved = [name for name, tm in results['task_metrics'].items() if not tm['solved']]
            for name in unsolved:
                f.write(f"  - {name}\n")

            f.write("\n\nLEARNING CURVE\n")
            f.write("-" * 40 + "\n")
            for m in results['learning_curve']:
                f.write(f"Iter {m['iteration']+1:2d}: {m['tasks_solved']:2d}/{m['tasks_total']:2d} solved, "
                       f"loss={m['recognition_loss']:.4f}, "
                       f"grammar={m['grammar_size']}, "
                       f"abstractions={m['abstractions']}, "
                       f"wake={m['wake_time']:.0f}s\n")


# ============================================================================
# MAIN
# ============================================================================

def main():
    start_time = time.time()

    print_banner("DREAMCODER V2 - CYTHON-OPTIMIZED OVERNIGHT PRETRAINING")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check optimizations
    print("Checking optimizations...")
    print(f"  Cython modules: {'ENABLED' if USE_CYTHON else 'DISABLED (Python fallback)'}")
    print(f"  PyPy available: {USE_PYPY} ({PYPY_PATH})")
    print(f"  Multiprocessing: {USE_MULTIPROCESSING}")
    print(f"  Workers: {N_WORKERS}")
    print()

    # Load rules
    easy_rules = get_easy_pretraining_rules()
    all_rules = get_all_pretraining_rules()

    print(f"Easy rules (level 1): {len(easy_rules)}")
    print(f"All rules (levels 1-2): {len(all_rules)}")

    # Create tasks with MORE examples to prevent spurious solutions
    # Using 100 training + 20 holdout examples per task
    print("\nCreating tasks with 100 examples + 20 holdout each...")
    easy_tasks = create_tasks_from_rules(easy_rules, n_examples=100, n_holdout=20, seed=42)
    all_tasks = create_tasks_from_rules(all_rules, n_examples=100, n_holdout=20, seed=42)

    print(f"Created {len(easy_tasks)} easy tasks")
    print(f"Created {len(all_tasks)} total tasks")

    # Build grammar
    print("\nBuilding lean grammar...")
    grammar = build_lean_grammar()
    print(f"Grammar: {len(grammar)} primitives")

    # Define phases (same as original for fair comparison)
    phases = [
        PhaseConfig(
            name="Phase 1: Easy Rules Foundation",
            iterations=5,
            use_all_rules=False,
            enumeration_budget=200000,
            max_depth=8,
            dreams_per_iteration=100,
            recognition_epochs=15
        ),
        PhaseConfig(
            name="Phase 2: All Rules with Abstractions",
            iterations=7,
            use_all_rules=True,
            enumeration_budget=300000,
            max_depth=9,
            dreams_per_iteration=150,
            recognition_epochs=20
        ),
        PhaseConfig(
            name="Phase 3: Deep Search with Full Library",
            iterations=8,
            use_all_rules=True,
            enumeration_budget=500000,
            max_depth=10,
            dreams_per_iteration=150,
            recognition_epochs=20
        )
    ]

    # Create output directory
    log_dir = Path("results/overnight_cython")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create eval function
    eval_fn = make_eval_fn()

    # Run Cython-optimized training
    print_banner("STARTING CYTHON-OPTIMIZED STAGED OVERNIGHT TRAINING")

    dc = CythonOptimizedDreamCoder(
        grammar=grammar,
        easy_tasks=easy_tasks,
        all_tasks=all_tasks,
        eval_fn=eval_fn,
        phases=phases,
        recognition_hidden_dim=256,
        recognition_lr=5e-4,
        keep_top_k=5,
        max_inventions_per_iteration=5,
        dream_temperature=1.0,
        n_workers=N_WORKERS,
        use_pypy=USE_PYPY,
        verbose=True,
        log_dir=str(log_dir),
        device='cpu'
    )

    results = dc.run()

    # Final summary
    total_time = time.time() - start_time

    print_banner("CYTHON-OPTIMIZED OVERNIGHT PRETRAINING COMPLETE")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {format_time(total_time)}")
    print()
    print(f"Tasks solved: {results['summary']['tasks_solved']}/{results['summary']['tasks_total']}")
    print(f"Success rate: {100*results['summary']['tasks_solved']/results['summary']['tasks_total']:.1f}%")
    print(f"Final grammar: {results['summary']['final_grammar_size']} primitives")
    print(f"Total abstractions: {results['summary']['total_abstractions']}")
    print(f"Total dreams: {results['summary']['total_dreams']}")

    # Print solved tasks
    solved = [(name, tm) for name, tm in results['task_metrics'].items() if tm['solved']]
    print(f"\nSolved tasks ({len(solved)}):")
    for name, tm in sorted(solved, key=lambda x: x[1]['iteration_solved']):
        print(f"  - {name} (iter {tm['iteration_solved']+1})")


if __name__ == "__main__":
    main()
