#!/usr/bin/env python3
"""
Overnight Pre-training Runner (Option C+ Enhanced)

This script runs a 12-hour staged curriculum pre-training on all 43 pretraining rules.

Strategy:
- Phase 1 (iterations 1-5):   22 easy rules, budget=200K, depth=8
- Phase 2 (iterations 6-12):  43 all rules, budget=300K, depth=9
- Phase 3 (iterations 13-20): 43 all rules, budget=500K, depth=10

Key features:
- Larger recognition model (256 hidden dim)
- Heavy dreaming (150 dreams/iteration)
- More recognition training (20 epochs)
- Staged difficulty progression
- Comprehensive logging for interpretability

Estimated runtime: 10-12 hours
"""

import sys
import os
import time
import json
import random
import copy
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import torch

from dreamcoder_core.type_system import arrow, HAND, BOOL
from dreamcoder_core.program import Program, Primitive, Invented
from dreamcoder_core.grammar import Grammar
from dreamcoder_core.enumeration import enumerate_simple
from dreamcoder_core.compression import compress_frontiers
from dreamcoder_core.neural_recognition import NeuralRecognitionModel
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.dreamcoder_original import (
    Task, SolutionEntry, TaskFrontier, IterationMetrics, TaskMetrics,
    NeuralDreamer, create_tasks_from_rules, make_eval_fn
)
from rules.pretraining_rules import (
    get_all_pretraining_rules, get_easy_pretraining_rules
)


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
    use_all_rules: bool  # False = easy only, True = all rules
    enumeration_budget: int
    max_depth: int
    dreams_per_iteration: int
    recognition_epochs: int


class StagedDreamCoder:
    """
    DreamCoder with staged curriculum learning.

    Supports changing task sets, budgets, and depths across phases.
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
        self.verbose = verbose
        self.log_dir = Path(log_dir) if log_dir else None
        self.device = device

        # Initialize recognition model with larger capacity
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

        # State - frontiers for ALL tasks (even if not yet active)
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

    def run(self) -> Dict:
        """Run all phases of staged training."""
        start_time = time.time()

        print_banner("DREAMCODER V2 - STAGED OVERNIGHT PRETRAINING")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Easy tasks: {len(self.easy_tasks)}")
        print(f"All tasks: {len(self.all_tasks)}")
        print(f"Recognition model: {self.recognition.hidden_dim} hidden dim")
        print(f"Device: {self.device}")
        print(f"\nPhases:")
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

            # Save intermediate checkpoint after each phase
            if self.log_dir:
                self._save_checkpoint(f"phase{phase_idx+1}")

        total_time = time.time() - start_time

        # Final summary
        print_banner("OVERNIGHT PRETRAINING COMPLETE")
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

            # Log cumulative progress
            total_solved = sum(1 for tm in self.task_metrics.values() if tm.solved)
            self.log(f"Cumulative solved (all tasks): {total_solved}/{len(self.all_tasks)}", 2)

    def _run_iteration(
        self,
        tasks: List[Task],
        enumeration_budget: int,
        max_depth: int,
        dreams_per_iteration: int,
        recognition_epochs: int
    ) -> IterationMetrics:
        """Run one wake-sleep iteration."""

        # =====================
        # WAKE PHASE
        # =====================
        self.log("\n[WAKE] Enumerating programs...")
        wake_start = time.time()

        total_programs = 0
        tasks_solved = 0

        for task in tasks:
            frontier = self.frontiers[task.name]

            # Skip if already well-solved
            if frontier.n_solutions >= self.keep_top_k and frontier.solved:
                tasks_solved += 1
                continue

            # Get task-specific grammar from neural recognition
            if self.global_iteration > 0:
                task_grammar = self.recognition.predict_grammar_weights(task)
            else:
                task_grammar = self.grammar

            # Enumerate
            programs_tried = 0
            enum_start = time.time()
            enum_timeout = 180.0  # 3 minutes per task

            for program, log_prob in enumerate_simple(
                task_grammar,
                task.request_type,
                max_depth=max_depth
            ):
                programs_tried += 1

                if programs_tried > enumeration_budget:
                    break
                if time.time() - enum_start > enum_timeout:
                    break

                # Evaluate
                try:
                    correct = 0
                    for inp, expected in task.examples:
                        result = self.eval_fn(program, inp)
                        if result == expected:
                            correct += 1

                    if correct == len(task.examples):
                        entry = SolutionEntry(
                            program=program,
                            log_probability=log_prob,
                            log_likelihood=0.0,
                            programs_enumerated=programs_tried,
                            time_found=time.time() - enum_start
                        )
                        if frontier.add(entry):
                            tm = self.task_metrics[task.name]
                            if not tm.solved:
                                tm.solved = True
                                tm.iteration_solved = self.global_iteration
                                tm.programs_to_solve = programs_tried
                                tm.best_program = str(program)
                                tm.description_length = entry.description_length
                                self.log(f"  SOLVED: {task.name} ({programs_tried:,} programs)", 1)

                        if frontier.n_solutions >= self.keep_top_k:
                            break
                except (ValueError, TypeError, ZeroDivisionError, IndexError, KeyError, AttributeError, RecursionError):
                    # Expected evaluation errors from malformed programs
                    pass

            frontier.total_programs_searched += programs_tried
            total_programs += programs_tried

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

        # Collect ALL frontiers (not just active tasks) for compression
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

                # Update recognition model grammar
                self.recognition.grammar = self.grammar

                # Update dreamer grammar
                self.dreamer.grammar = self.grammar

                self.log(f"  Found {len(new_abstractions)} abstraction(s)", 1)

        compression_time = time.time() - comp_start
        self.library_history.append(new_abstractions)

        # =====================
        # SLEEP - RECOGNITION TRAINING
        # =====================
        recognition_loss = 0.0
        recognition_time = 0.0

        self.log("\n[SLEEP - RECOGNITION] Training neural model...")
        rec_start = time.time()

        # Train on ALL solved tasks (not just active ones)
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

            # Collect example inputs from ALL tasks
            all_inputs = []
            for task in self.all_tasks:
                for inp, _ in task.examples[:5]:
                    all_inputs.append(inp)

            # Generate dreams
            dreams = self.dreamer.generate_dreams(
                self.all_tasks[0].request_type,
                dreams_per_iteration,
                all_inputs,
                temperature=self.dream_temperature
            )
            dreams_generated = len(dreams)

            # Train recognition on dreams
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

        # =====================
        # SNAPSHOT EMBEDDINGS
        # =====================
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

    def _snapshot_embeddings(self, iteration: int):
        """Snapshot task embeddings for interpretability tracking."""
        embeddings = {}
        for task in self.all_tasks[:30]:  # Sample for efficiency
            emb = self.recognition.get_task_embedding(task)
            embeddings[task.name] = emb.numpy().tolist()

        self._embedding_snapshots.append({
            'iteration': iteration,
            'embeddings': embeddings
        })

    def _save_checkpoint(self, name: str):
        """Save intermediate checkpoint."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

        # Save model
        model_path = self.log_dir / f"checkpoint_{name}_{timestamp}.pt"
        self.recognition.save(str(model_path))

        # Save current state summary
        state = {
            'phase': name,
            'global_iteration': self.global_iteration,
            'tasks_solved': sum(1 for tm in self.task_metrics.values() if tm.solved),
            'grammar_size': len(self.grammar),
            'timestamp': timestamp
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

        # Save JSON results
        json_path = self.log_dir / f"overnight_pretraining_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save model checkpoint
        model_path = self.log_dir / f"recognition_model_final_{timestamp}.pt"
        self.recognition.save(str(model_path))

        # Generate summary report
        self._generate_report(results, timestamp)

        self.log(f"\nResults saved to: {json_path}")

    def _generate_report(self, results: Dict, timestamp: str):
        """Generate a human-readable report."""
        report_path = self.log_dir / f"overnight_report_{timestamp}.txt"

        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DREAMCODER V2 - OVERNIGHT PRETRAINING REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {format_time(results['summary']['total_time'])}\n\n")

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
                f.write(f"  Program: {tm['best_program'][:100]}...\n" if len(tm['best_program']) > 100 else f"  Program: {tm['best_program']}\n")

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
                       f"abstractions={m['abstractions']}\n")


def main():
    start_time = time.time()

    print_banner("DREAMCODER V2 - OVERNIGHT PRETRAINING (Option C+ Enhanced)")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # =========================================================================
    # LOAD RULES AND CREATE TASKS
    # =========================================================================

    easy_rules = get_easy_pretraining_rules()
    all_rules = get_all_pretraining_rules()

    print(f"Easy rules (level 1): {len(easy_rules)}")
    print(f"All rules (levels 1-2): {len(all_rules)}")

    print("\nEasy rules:")
    for r in easy_rules:
        print(f"  - {r.id} ({r.family})")

    print("\nMedium rules (level 2):")
    for r in all_rules:
        if r.level == 2:
            print(f"  - {r.id} ({r.family})")

    # Create tasks
    print("\nCreating tasks with 20 examples each...")
    easy_tasks = create_tasks_from_rules(easy_rules, n_examples=20, seed=42)
    all_tasks = create_tasks_from_rules(all_rules, n_examples=20, seed=42)

    print(f"Created {len(easy_tasks)} easy tasks")
    print(f"Created {len(all_tasks)} total tasks")

    # Build grammar
    print("\nBuilding lean grammar...")
    grammar = build_lean_grammar()
    print(f"Grammar: {len(grammar)} primitives")

    # =========================================================================
    # DEFINE PHASES
    # =========================================================================

    phases = [
        PhaseConfig(
            name="Phase 1: Easy Rules Foundation",
            iterations=5,
            use_all_rules=False,  # 22 easy only
            enumeration_budget=200000,
            max_depth=8,
            dreams_per_iteration=100,
            recognition_epochs=15
        ),
        PhaseConfig(
            name="Phase 2: All Rules with Abstractions",
            iterations=7,
            use_all_rules=True,  # all 43
            enumeration_budget=300000,
            max_depth=9,
            dreams_per_iteration=150,
            recognition_epochs=20
        ),
        PhaseConfig(
            name="Phase 3: Deep Search with Full Library",
            iterations=8,
            use_all_rules=True,  # all 43
            enumeration_budget=500000,
            max_depth=10,
            dreams_per_iteration=150,
            recognition_epochs=20
        )
    ]

    # =========================================================================
    # CREATE OUTPUT DIRECTORY
    # =========================================================================

    log_dir = Path("results/overnight_pretraining")
    log_dir.mkdir(parents=True, exist_ok=True)

    # =========================================================================
    # CREATE EVAL FUNCTION
    # =========================================================================

    eval_fn = make_eval_fn()

    # =========================================================================
    # RUN STAGED TRAINING
    # =========================================================================

    print_banner("STARTING STAGED OVERNIGHT TRAINING")

    dc = StagedDreamCoder(
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
        verbose=True,
        log_dir=str(log_dir),
        device='cpu'
    )

    results = dc.run()

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    total_time = time.time() - start_time

    print_banner("OVERNIGHT PRETRAINING COMPLETE")
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
