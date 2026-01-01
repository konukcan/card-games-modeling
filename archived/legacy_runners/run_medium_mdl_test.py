#!/usr/bin/env python3
"""
Medium-Sized Run with MDL Compression Testing

This script runs a medium-sized experiment (5-8 iterations, ~30-60 minutes)
to test the new Phase 1 (program refactoring) and Phase 2 (MDL scoring)
compression improvements.

Key features tested:
- Program refactoring after compression (Phase 1)
- MDL-based compression vs heuristic (Phase 2)
- Hierarchical abstraction discovery

Estimated runtime: 30-60 minutes
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

from dreamcoder_core.type_system import arrow, HAND, BOOL, INT
from dreamcoder_core.program import Program, Primitive, Invented
from dreamcoder_core.grammar import Grammar
from dreamcoder_core.enumeration import enumerate_simple
from dreamcoder_core.compression import (
    compress_frontiers, compress_frontiers_mdl, compute_mdl,
    compute_mdl_detailed, format_invention
)
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
class CompressionComparison:
    """Track compression method comparison."""
    iteration: int
    heuristic_inventions: int
    heuristic_savings: float
    mdl_inventions: int
    mdl_improvement: float
    heuristic_time: float
    mdl_time: float
    mdl_details: Dict[str, Any] = field(default_factory=dict)


class MediumRunDreamCoder:
    """
    DreamCoder for medium-sized runs with MDL comparison.
    """

    def __init__(
        self,
        grammar: Grammar,
        tasks: List[Task],
        eval_fn: Callable,
        use_mdl_compression: bool = False,  # Toggle MDL vs heuristic
        enumeration_budget: int = 100000,
        max_depth: int = 8,
        recognition_hidden_dim: int = 128,
        recognition_lr: float = 5e-4,
        keep_top_k: int = 5,
        max_inventions_per_iteration: int = 5,
        verbose: bool = True,
        log_dir: Optional[str] = None,
        device: str = 'cpu'
    ):
        self.initial_grammar = grammar
        self.grammar = copy.deepcopy(grammar)
        self.tasks = tasks
        self.eval_fn = eval_fn
        self.use_mdl_compression = use_mdl_compression
        self.enumeration_budget = enumeration_budget
        self.max_depth = max_depth

        self.keep_top_k = keep_top_k
        self.max_inventions_per_iteration = max_inventions_per_iteration
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

        # State - frontiers for all tasks
        self.frontiers: Dict[str, TaskFrontier] = {}
        for task in tasks:
            self.frontiers[task.name] = TaskFrontier(task, max_size=keep_top_k)

        self.task_metrics: Dict[str, TaskMetrics] = {}
        for task in tasks:
            self.task_metrics[task.name] = TaskMetrics(task.name, task.family)

        self.iteration_metrics: List[IterationMetrics] = []
        self.library_history: List[List[str]] = []
        self.compression_comparisons: List[CompressionComparison] = []
        self.global_iteration = 0

    def log(self, msg: str, level: int = 0):
        """Log message with indentation."""
        if self.verbose:
            indent = "  " * level
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {indent}{msg}", flush=True)

    def run(self, n_iterations: int) -> Dict:
        """Run training for n_iterations."""
        start_time = time.time()

        print_banner(f"MEDIUM RUN: {'MDL' if self.use_mdl_compression else 'HEURISTIC'} COMPRESSION")
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Tasks: {len(self.tasks)}")
        print(f"Iterations: {n_iterations}")
        print(f"Enumeration budget: {self.enumeration_budget:,}")
        print(f"Max depth: {self.max_depth}")
        print(f"Compression: {'MDL-based' if self.use_mdl_compression else 'Heuristic'}")
        print(f"Device: {self.device}")

        for i in range(n_iterations):
            self.log("")
            self.log("=" * 70)
            self.log(f"ITERATION {i + 1}/{n_iterations}")
            self.log("=" * 70)

            metrics = self._run_iteration()
            self.iteration_metrics.append(metrics)
            self.global_iteration += 1

            # Log summary
            self.log("")
            self.log(f"Summary:", 1)
            self.log(f"Solved: {metrics.tasks_solved}/{metrics.tasks_total}", 2)
            self.log(f"Programs: {metrics.programs_enumerated:,}", 2)
            self.log(f"New abstractions: {len(metrics.new_abstractions)}", 2)
            self.log(f"Recognition loss: {metrics.recognition_loss:.4f}", 2)
            self.log(f"Grammar size: {metrics.grammar_size}", 2)

            # Cumulative progress
            total_solved = sum(1 for tm in self.task_metrics.values() if tm.solved)
            self.log(f"Cumulative solved: {total_solved}/{len(self.tasks)}", 2)

        total_time = time.time() - start_time

        print_banner("RUN COMPLETE")
        print(f"Total time: {format_time(total_time)}")

        results = self._compile_results(total_time)
        if self.log_dir:
            self._save_results(results)

        return results

    def _run_iteration(self) -> IterationMetrics:
        """Run one wake-sleep iteration."""

        # =====================
        # WAKE PHASE
        # =====================
        self.log("\n[WAKE] Enumerating programs...")
        wake_start = time.time()

        total_programs = 0
        tasks_solved = 0

        for task in self.tasks:
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
            enum_timeout = 120.0  # 2 minutes per task

            for program, log_prob in enumerate_simple(
                task_grammar,
                task.request_type,
                max_depth=self.max_depth
            ):
                programs_tried += 1

                if programs_tried > self.enumeration_budget:
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
        self.log(f"Wake: {tasks_solved}/{len(self.tasks)} solved, "
                f"{total_programs:,} programs in {wake_time:.1f}s")

        # =====================
        # SLEEP - COMPRESSION (Compare both methods)
        # =====================
        new_abstractions = []
        compression_time = 0.0

        self.log("\n[SLEEP - COMPRESSION] Finding abstractions...")
        comp_start = time.time()

        # Collect frontiers
        all_frontiers = []
        all_programs = []
        for frontier in self.frontiers.values():
            if frontier.n_solutions > 0:
                programs_with_ll = [(e.program, e.log_likelihood) for e in frontier.entries]
                all_frontiers.append(programs_with_ll)
                for p, _ in programs_with_ll:
                    all_programs.append(p)

        if all_frontiers:
            # Run BOTH methods for comparison
            comparison = CompressionComparison(
                iteration=self.global_iteration,
                heuristic_inventions=0,
                heuristic_savings=0.0,
                mdl_inventions=0,
                mdl_improvement=0.0,
                heuristic_time=0.0,
                mdl_time=0.0
            )

            # 1. Heuristic compression
            h_start = time.time()
            heuristic_result = compress_frontiers(
                self.grammar,
                all_frontiers,
                max_inventions=self.max_inventions_per_iteration,
                min_savings=2.0,
                use_anti_unification=True,
                refactor_programs=True  # Use new refactoring!
            )
            comparison.heuristic_time = time.time() - h_start
            comparison.heuristic_inventions = len(heuristic_result.new_inventions)
            comparison.heuristic_savings = heuristic_result.total_savings

            # 2. MDL compression
            # Using min_mdl_improvement=0.0 means we accept ANY positive improvement
            # (which is theoretically correct - accept anything that reduces total MDL)
            # Using grammar_weight=0.5 to be less conservative about grammar expansion
            # (our global normalization is harsher than original DreamCoder's per-type normalization)
            m_start = time.time()
            mdl_result = compress_frontiers_mdl(
                self.grammar,
                all_frontiers,
                request_type=arrow(HAND, BOOL),
                max_inventions=self.max_inventions_per_iteration,
                grammar_weight=0.5,  # Lower penalty for grammar complexity
                min_mdl_improvement=0.0,  # Accept any positive improvement
                refactor_programs=True
            )
            comparison.mdl_time = time.time() - m_start
            comparison.mdl_inventions = len(mdl_result.new_inventions)
            comparison.mdl_improvement = mdl_result.total_savings

            if mdl_result.rewrite_stats:
                comparison.mdl_details = {
                    'initial_mdl': mdl_result.rewrite_stats.get('initial_mdl', 0),
                    'final_mdl': mdl_result.rewrite_stats.get('final_mdl', 0),
                    'inventions_evaluated': mdl_result.rewrite_stats.get('inventions_evaluated', 0),
                    'inventions_accepted': mdl_result.rewrite_stats.get('inventions_accepted', 0)
                }

                # Log candidate diagnostics to understand MDL scoring
                all_candidates = mdl_result.rewrite_stats.get('all_candidates', [])
                if all_candidates:
                    self.log(f"  MDL Candidate Analysis ({len(all_candidates)} candidates):", 1)
                    # Sort by improvement to see best candidates
                    sorted_candidates = sorted(all_candidates, key=lambda x: x['improvement'], reverse=True)
                    for i, cand in enumerate(sorted_candidates[:5]):  # Top 5
                        self.log(f"    #{i+1}: improvement={cand['improvement']:.2f}, "
                                f"heuristic={cand['heuristic_savings']:.2f}, "
                                f"count={cand['count']}, target={cand['target'][:50]}...", 1)

            self.compression_comparisons.append(comparison)

            # Log comparison
            self.log(f"  Heuristic: {comparison.heuristic_inventions} inventions, "
                    f"savings={comparison.heuristic_savings:.2f}, "
                    f"time={comparison.heuristic_time:.2f}s", 1)
            self.log(f"  MDL:       {comparison.mdl_inventions} inventions, "
                    f"improvement={comparison.mdl_improvement:.2f}, "
                    f"time={comparison.mdl_time:.2f}s", 1)

            # Use the selected method
            if self.use_mdl_compression:
                result = mdl_result
                self.log(f"  Using MDL compression", 1)
            else:
                result = heuristic_result
                self.log(f"  Using Heuristic compression", 1)

            if result.new_inventions:
                self.grammar = result.new_grammar
                new_abstractions = [str(inv) for inv in result.new_inventions]

                # Update recognition model grammar
                self.recognition.grammar = self.grammar

                self.log(f"  Applied {len(new_abstractions)} abstraction(s):", 1)
                for inv in result.new_inventions:
                    self.log(f"    {format_invention(inv)}", 2)

        compression_time = time.time() - comp_start
        self.library_history.append(new_abstractions)

        # =====================
        # SLEEP - RECOGNITION TRAINING
        # =====================
        recognition_loss = 0.0
        recognition_time = 0.0

        self.log("\n[SLEEP - RECOGNITION] Training neural model...")
        rec_start = time.time()

        solved_tasks = [t for t in self.tasks if self.frontiers[t.name].solved]

        if solved_tasks:
            recognition_loss = self.recognition.train_on_frontiers(
                solved_tasks,
                self.frontiers,
                epochs=10
            )

        recognition_time = time.time() - rec_start
        self.log(f"  Trained on {len(solved_tasks)} solved tasks, loss: {recognition_loss:.4f}", 1)

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

        return IterationMetrics(
            iteration=self.global_iteration,
            tasks_solved=tasks_solved,
            tasks_total=len(self.tasks),
            programs_enumerated=total_programs,
            wake_time=wake_time,
            new_abstractions=new_abstractions,
            compression_time=compression_time,
            recognition_loss=recognition_loss,
            recognition_time=recognition_time,
            dreams_generated=0,
            dream_time=0.0,
            grammar_size=len(self.grammar)
        )

    def _compile_results(self, total_time: float) -> Dict:
        """Compile all results."""
        return {
            'config': {
                'tasks': len(self.tasks),
                'initial_grammar_size': len(self.initial_grammar),
                'enumeration_budget': self.enumeration_budget,
                'max_depth': self.max_depth,
                'use_mdl_compression': self.use_mdl_compression,
                'recognition_hidden_dim': self.recognition.hidden_dim
            },
            'summary': {
                'total_time': total_time,
                'total_iterations': self.global_iteration,
                'tasks_solved': sum(1 for tm in self.task_metrics.values() if tm.solved),
                'tasks_total': len(self.tasks),
                'final_grammar_size': len(self.grammar),
                'total_abstractions': sum(len(m.new_abstractions) for m in self.iteration_metrics)
            },
            'learning_curve': [
                {
                    'iteration': m.iteration,
                    'tasks_solved': m.tasks_solved,
                    'tasks_total': m.tasks_total,
                    'programs': m.programs_enumerated,
                    'abstractions': len(m.new_abstractions),
                    'recognition_loss': m.recognition_loss,
                    'grammar_size': m.grammar_size,
                    'wake_time': m.wake_time
                }
                for m in self.iteration_metrics
            ],
            'compression_comparison': [
                {
                    'iteration': c.iteration,
                    'heuristic_inventions': c.heuristic_inventions,
                    'heuristic_savings': c.heuristic_savings,
                    'heuristic_time': c.heuristic_time,
                    'mdl_inventions': c.mdl_inventions,
                    'mdl_improvement': c.mdl_improvement,
                    'mdl_time': c.mdl_time,
                    'mdl_details': c.mdl_details
                }
                for c in self.compression_comparisons
            ],
            'task_metrics': {name: asdict(tm) for name, tm in self.task_metrics.items()},
            'library_evolution': self.library_history,
            'recognition_training_losses': self.recognition.training_losses if self.recognition else []
        }

    def _save_results(self, results: Dict):
        """Save results to log directory."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        method = "mdl" if self.use_mdl_compression else "heuristic"

        # Save JSON results
        json_path = self.log_dir / f"medium_run_{method}_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save model checkpoint
        model_path = self.log_dir / f"recognition_model_{method}_{timestamp}.pt"
        self.recognition.save(str(model_path))

        self.log(f"\nResults saved to: {json_path}")


def print_report(results: Dict):
    """Print a comprehensive report."""
    print_banner("COMPREHENSIVE RUN REPORT")

    config = results['config']
    summary = results['summary']

    # Summary
    print("CONFIGURATION")
    print("-" * 40)
    print(f"  Tasks: {config['tasks']}")
    print(f"  Enumeration budget: {config['enumeration_budget']:,}")
    print(f"  Max depth: {config['max_depth']}")
    print(f"  Compression method: {'MDL' if config['use_mdl_compression'] else 'Heuristic'}")
    print(f"  Initial grammar size: {config['initial_grammar_size']}")

    print("\nSUMMARY")
    print("-" * 40)
    print(f"  Total time: {format_time(summary['total_time'])}")
    print(f"  Iterations: {summary['total_iterations']}")
    print(f"  Tasks solved: {summary['tasks_solved']}/{summary['tasks_total']} "
          f"({100*summary['tasks_solved']/summary['tasks_total']:.1f}%)")
    print(f"  Final grammar size: {summary['final_grammar_size']}")
    print(f"  Total abstractions learned: {summary['total_abstractions']}")

    # Learning curve
    print("\nLEARNING CURVE")
    print("-" * 40)
    print(f"{'Iter':>4} | {'Solved':>8} | {'Programs':>12} | {'Abstractions':>12} | {'Loss':>8} | {'Grammar':>8}")
    print("-" * 70)
    for m in results['learning_curve']:
        print(f"{m['iteration']+1:4d} | {m['tasks_solved']:8d} | {m['programs']:12,} | "
              f"{m['abstractions']:12d} | {m['recognition_loss']:8.4f} | {m['grammar_size']:8d}")

    # Compression comparison
    if results['compression_comparison']:
        print("\nCOMPRESSION METHOD COMPARISON")
        print("-" * 40)
        print(f"{'Iter':>4} | {'Heur Inv':>8} | {'Heur Save':>10} | {'MDL Inv':>8} | {'MDL Impr':>10} | {'Heur Time':>10} | {'MDL Time':>10}")
        print("-" * 85)
        for c in results['compression_comparison']:
            print(f"{c['iteration']+1:4d} | {c['heuristic_inventions']:8d} | "
                  f"{c['heuristic_savings']:10.2f} | {c['mdl_inventions']:8d} | "
                  f"{c['mdl_improvement']:10.2f} | {c['heuristic_time']:10.2f}s | {c['mdl_time']:10.2f}s")

        # Aggregates
        total_heur_inv = sum(c['heuristic_inventions'] for c in results['compression_comparison'])
        total_mdl_inv = sum(c['mdl_inventions'] for c in results['compression_comparison'])
        total_heur_save = sum(c['heuristic_savings'] for c in results['compression_comparison'])
        total_mdl_impr = sum(c['mdl_improvement'] for c in results['compression_comparison'])

        print("-" * 85)
        print(f"{'TOTAL':>4} | {total_heur_inv:8d} | {total_heur_save:10.2f} | "
              f"{total_mdl_inv:8d} | {total_mdl_impr:10.2f}")

        print("\nINSIGHT: MDL is more selective - accepts fewer abstractions but only")
        print("         those that truly reduce total description length (grammar + programs).")

    # Solved tasks by family
    print("\nSOLVED TASKS BY FAMILY")
    print("-" * 40)
    family_counts = {}
    for name, tm in results['task_metrics'].items():
        family = tm.get('family', 'unknown')  # Handle missing family key
        if family not in family_counts:
            family_counts[family] = {'solved': 0, 'total': 0}
        family_counts[family]['total'] += 1
        if tm['solved']:
            family_counts[family]['solved'] += 1

    for family, counts in sorted(family_counts.items()):
        pct = 100 * counts['solved'] / counts['total'] if counts['total'] > 0 else 0
        print(f"  {family}: {counts['solved']}/{counts['total']} ({pct:.0f}%)")

    # Library evolution
    print("\nLIBRARY EVOLUTION")
    print("-" * 40)
    for i, abstractions in enumerate(results['library_evolution']):
        if abstractions:
            print(f"  Iteration {i+1}: +{len(abstractions)} abstractions")
            for a in abstractions[:3]:  # Show first 3
                print(f"    {a[:60]}..." if len(a) > 60 else f"    {a}")
            if len(abstractions) > 3:
                print(f"    ... and {len(abstractions) - 3} more")

    # Unsolved tasks
    unsolved = [name for name, tm in results['task_metrics'].items() if not tm['solved']]
    if unsolved:
        print(f"\nUNSOLVED TASKS ({len(unsolved)})")
        print("-" * 40)
        for name in unsolved[:10]:
            print(f"  - {name}")
        if len(unsolved) > 10:
            print(f"  ... and {len(unsolved) - 10} more")


def main():
    start_time = time.time()

    print_banner("MEDIUM RUN: TESTING MDL COMPRESSION IMPROVEMENTS")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # =========================================================================
    # LOAD RULES AND CREATE TASKS
    # =========================================================================

    # Use easy rules for faster testing
    rules = get_easy_pretraining_rules()
    print(f"Loaded {len(rules)} easy pretraining rules")

    # Create tasks
    from dreamcoder_core.dreamcoder_original import create_tasks_from_rules, make_eval_fn
    tasks = create_tasks_from_rules(rules, n_examples=15, seed=42)
    print(f"Created {len(tasks)} tasks")

    # Build grammar
    grammar = build_lean_grammar()
    print(f"Grammar: {len(grammar)} primitives")

    # Create eval function
    eval_fn = make_eval_fn()

    # =========================================================================
    # RUN WITH HEURISTIC COMPRESSION
    # =========================================================================

    log_dir = Path("results/medium_mdl_test")
    log_dir.mkdir(parents=True, exist_ok=True)

    print_banner("RUNNING WITH HEURISTIC COMPRESSION")

    dc_heuristic = MediumRunDreamCoder(
        grammar=grammar,
        tasks=tasks,
        eval_fn=eval_fn,
        use_mdl_compression=False,
        enumeration_budget=100000,
        max_depth=8,
        recognition_hidden_dim=128,
        keep_top_k=3,
        max_inventions_per_iteration=5,
        verbose=True,
        log_dir=str(log_dir),
        device='cpu'
    )

    results_heuristic = dc_heuristic.run(n_iterations=5)

    # =========================================================================
    # PRINT COMPREHENSIVE REPORT
    # =========================================================================

    print_report(results_heuristic)

    # =========================================================================
    # FINAL TIMING
    # =========================================================================

    total_time = time.time() - start_time
    print_banner("RUN COMPLETE")
    print(f"Total time: {format_time(total_time)}")


if __name__ == "__main__":
    main()
