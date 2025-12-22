#!/usr/bin/env python3
"""
=============================================================================
TOP-DOWN ENUMERATION TEST RUNNER
=============================================================================

This script tests the new TopDownEnumerator with the full DreamCoder
wake-sleep architecture on pre-training rules.

Configuration:
- 22 easy pre-training rules (level 1)
- Full neural recognition model
- Compression/abstraction learning
- Dreaming with synthetic examples
- 3-5 iterations

Outputs:
- Solve rate comparison
- Programs per solution
- Time per solution
- Memory efficiency (partial vs complete programs explored)
- Learning curve

Usage:
    python run_topdown_test.py
"""

import sys
import os
import time
import json
import random
import copy
import math
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dreamcoder_core.type_system import (
    Type, BaseType, Arrow, ListType, TypeVariable,
    BOOL, INT, CARD, SUIT, RANK, HAND, arrow
)
from dreamcoder_core.program import (
    Program, Primitive, Application, Abstraction, Index, Invented
)
from dreamcoder_core.grammar import Grammar, Production, uniform_grammar
from dreamcoder_core.enumeration import TopDownEnumerator, enumerate_simple, EnumerationResult
from dreamcoder_core.compression import compress_frontiers
from dreamcoder_core.neural_recognition import NeuralRecognitionModel
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.dreamcoder_v2 import (
    Task, SolutionEntry, TaskFrontier, IterationMetrics, TaskMetrics,
    NeuralDreamer, create_tasks_from_rules, make_eval_fn
)
from rules.pretraining_rules import get_easy_pretraining_rules


# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG = {
    # Task settings
    "n_examples": 50,           # Examples per task
    "n_holdout": 20,            # Holdout examples for verification
    "hand_size": 6,

    # Enumeration settings
    "enumeration_budget": 50000,   # Programs per task
    "enumeration_timeout": 60.0,   # Seconds per task
    "max_depth": 7,                # AST depth
    "max_cost": 30.0,              # Max negative log probability

    # Frontier settings
    "keep_top_k": 5,

    # Component flags
    "use_compression": True,
    "use_recognition": True,
    "use_dreaming": True,

    # Compression settings
    "max_inventions_per_iteration": 3,
    "min_compression_savings": 2.0,

    # Recognition settings
    "recognition_hidden_dim": 128,
    "recognition_epochs": 10,
    "recognition_lr": 1e-3,

    # Dreaming settings
    "dreams_per_iteration": 30,
    "dream_temperature": 1.0,

    # General
    "max_iterations": 4,
    "seed": 42,
    "device": "cpu",
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


def print_banner(text: str, char: str = "="):
    """Print a banner."""
    line = char * 70
    print(f"\n{line}")
    print(text)
    print(f"{line}\n", flush=True)


def log(msg: str, level: int = 0):
    """Log message with timestamp."""
    indent = "  " * level
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {indent}{msg}", flush=True)


# =============================================================================
# TOP-DOWN DREAMCODER
# =============================================================================

class TopDownDreamCoder:
    """
    DreamCoder variant using TopDownEnumerator instead of iterative deepening.

    This is a modified version of DreamCoderV2 that uses true top-down
    hole-filling enumeration for the wake phase.
    """

    def __init__(
        self,
        grammar: Grammar,
        tasks: List[Task],
        eval_fn: Callable[[Program, Any], Any],
        config: Dict
    ):
        self.initial_grammar = grammar
        self.grammar = copy.deepcopy(grammar)
        self.tasks = tasks
        self.eval_fn = eval_fn
        self.config = config

        # State
        self.frontiers: Dict[str, TaskFrontier] = {
            t.name: TaskFrontier(t, max_size=config["keep_top_k"])
            for t in tasks
        }
        self.task_metrics: Dict[str, TaskMetrics] = {
            t.name: TaskMetrics(t.name, t.family)
            for t in tasks
        }
        self.iteration_metrics: List[Dict] = []
        self.library_history: List[List[str]] = []

        # Stats for comparison
        self.enumeration_stats: Dict[str, Dict] = {}

        # Neural components
        if config["use_recognition"]:
            self.recognition = NeuralRecognitionModel(
                grammar=grammar,
                hidden_dim=config["recognition_hidden_dim"],
                learning_rate=config["recognition_lr"],
                device=config["device"]
            )
        else:
            self.recognition = None

        if config["use_dreaming"] and self.recognition:
            self.dreamer = NeuralDreamer(
                grammar=grammar,
                recognition_model=self.recognition,
                eval_fn=eval_fn,
                device=config["device"]
            )
        else:
            self.dreamer = None

    def run(self) -> Dict:
        """Run the full wake-sleep learning loop."""
        start_time = time.time()

        print_banner("TOP-DOWN DREAMCODER TEST RUN")
        log(f"Tasks: {len(self.tasks)}")
        log(f"Initial grammar: {len(self.grammar)} primitives")
        log(f"Components: compression={self.config['use_compression']}, "
            f"recognition={self.config['use_recognition']}, "
            f"dreaming={self.config['use_dreaming']}")
        log("")

        for iteration in range(self.config["max_iterations"]):
            print_banner(f"ITERATION {iteration + 1}/{self.config['max_iterations']}")

            metrics = self._run_iteration(iteration)
            self.iteration_metrics.append(metrics)

            # Summary
            log("")
            log(f"Iteration {iteration + 1} Summary:", 1)
            log(f"Solved: {metrics['tasks_solved']}/{metrics['tasks_total']}", 2)
            log(f"Complete programs: {metrics['programs_enumerated']:,}", 2)
            log(f"Partial programs: {metrics['partial_programs_explored']:,}", 2)
            log(f"New abstractions: {len(metrics['new_abstractions'])}", 2)
            log(f"Recognition loss: {metrics['recognition_loss']:.4f}", 2)

            if metrics['tasks_solved'] == metrics['tasks_total']:
                log("\nAll tasks solved!")
                break

        total_time = time.time() - start_time

        # Final report
        print_banner("FINAL RESULTS")
        self._print_final_report(total_time)

        return self._compile_results(total_time)

    def _run_iteration(self, iteration: int) -> Dict:
        """Run one wake-sleep iteration."""

        # =====================
        # WAKE: Top-Down Enumeration
        # =====================
        log("[WAKE] Top-down enumeration...")
        wake_start = time.time()

        total_programs = 0
        total_partial = 0
        tasks_solved = 0

        for task in self.tasks:
            frontier = self.frontiers[task.name]

            if frontier.solved:
                tasks_solved += 1
                continue

            # Get task-specific grammar from recognition
            if self.recognition and iteration > 0:
                task_grammar = self.recognition.predict_grammar_weights(task)
            else:
                task_grammar = self.grammar

            # Create top-down enumerator
            enumerator = TopDownEnumerator(
                task_grammar,
                max_depth=self.config["max_depth"],
                max_programs=self.config["enumeration_budget"]
            )

            # Enumerate
            enum_start = time.time()
            programs_tried = 0
            solution_found = False

            for program, log_prob in enumerator.enumerate(
                task.request_type,
                max_cost=self.config["max_cost"],
                timeout_seconds=self.config["enumeration_timeout"]
            ):
                programs_tried += 1

                if programs_tried > self.config["enumeration_budget"]:
                    break

                # Evaluate on examples
                try:
                    correct = 0
                    for inp, expected in task.examples:
                        result = self.eval_fn(program, inp)
                        if result == expected:
                            correct += 1

                    if correct == len(task.examples):
                        # Verify on holdout
                        holdout_correct = 0
                        holdout = getattr(task, 'holdout_examples', [])
                        for inp, expected in holdout:
                            try:
                                result = self.eval_fn(program, inp)
                                if result == expected:
                                    holdout_correct += 1
                            except:
                                pass

                        if len(holdout) == 0 or holdout_correct >= len(holdout) * 0.9:
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
                                    tm.iteration_solved = iteration
                                    tm.programs_to_solve = programs_tried
                                    tm.best_program = str(program)
                                    tm.description_length = entry.description_length
                                    solution_found = True

                                    # Store enumeration stats
                                    self.enumeration_stats[task.name] = {
                                        "programs_complete": programs_tried,
                                        "programs_partial": enumerator.partial_programs_explored,
                                        "time_seconds": time.time() - enum_start,
                                        "iteration": iteration,
                                        "description_length": entry.description_length
                                    }
                                    break

                except Exception as e:
                    pass

            frontier.total_programs_searched += programs_tried
            total_programs += programs_tried
            total_partial += enumerator.partial_programs_explored

            if frontier.solved:
                tasks_solved += 1

            # Log progress
            status = "SOLVED" if solution_found else "unsolved"
            log(f"{task.name}: {programs_tried:,} progs, "
                f"{enumerator.partial_programs_explored:,} partial [{status}]", 1)

        wake_time = time.time() - wake_start
        log(f"Wake complete: {tasks_solved}/{len(self.tasks)} solved in {wake_time:.1f}s")

        # =====================
        # SLEEP: Compression
        # =====================
        new_abstractions = []
        compression_time = 0.0

        if self.config["use_compression"]:
            log("\n[SLEEP - COMPRESSION] Finding abstractions...")
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
                    max_inventions=self.config["max_inventions_per_iteration"],
                    min_savings=self.config["min_compression_savings"],
                    use_anti_unification=True
                )

                if result.new_inventions:
                    self.grammar = result.new_grammar
                    new_abstractions = [str(inv) for inv in result.new_inventions]
                    if self.recognition:
                        self.recognition.grammar = self.grammar
                    if self.dreamer:
                        self.dreamer.grammar = self.grammar
                    log(f"Found {len(new_abstractions)} abstraction(s)", 1)

            compression_time = time.time() - comp_start
            self.library_history.append(new_abstractions)

        # =====================
        # SLEEP: Recognition
        # =====================
        recognition_loss = 0.0
        recognition_time = 0.0

        if self.config["use_recognition"] and self.recognition:
            log("\n[SLEEP - RECOGNITION] Training neural model...")
            rec_start = time.time()

            recognition_loss = self.recognition.train_on_frontiers(
                self.tasks,
                self.frontiers,
                epochs=self.config["recognition_epochs"]
            )

            recognition_time = time.time() - rec_start
            log(f"Loss: {recognition_loss:.4f}", 1)

        # =====================
        # SLEEP: Dreaming
        # =====================
        dreams_generated = 0
        dream_time = 0.0

        if self.config["use_dreaming"] and self.dreamer and iteration > 0:
            log("\n[SLEEP - DREAMING] Generating synthetic tasks...")
            dream_start = time.time()

            all_inputs = []
            for task in self.tasks:
                for inp, _ in task.examples[:5]:
                    all_inputs.append(inp)

            dreams = self.dreamer.generate_dreams(
                self.tasks[0].request_type,
                self.config["dreams_per_iteration"],
                all_inputs,
                temperature=self.config["dream_temperature"]
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
            log(f"Generated {dreams_generated} dreams", 1)

        # Grammar weight update
        if tasks_solved > 0:
            all_frontiers = []
            for frontier in self.frontiers.values():
                if frontier.n_solutions > 0:
                    programs_with_ll = [(e.program, e.log_likelihood) for e in frontier.entries]
                    all_frontiers.append(programs_with_ll)
            if all_frontiers:
                self.grammar = self.grammar.inside_outside_update(all_frontiers)

        return {
            "iteration": iteration,
            "tasks_solved": tasks_solved,
            "tasks_total": len(self.tasks),
            "programs_enumerated": total_programs,
            "partial_programs_explored": total_partial,
            "wake_time": wake_time,
            "new_abstractions": new_abstractions,
            "compression_time": compression_time,
            "recognition_loss": recognition_loss,
            "recognition_time": recognition_time,
            "dreams_generated": dreams_generated,
            "dream_time": dream_time,
            "grammar_size": len(self.grammar)
        }

    def _print_final_report(self, total_time: float):
        """Print detailed final report."""
        solved = sum(1 for tm in self.task_metrics.values() if tm.solved)
        total = len(self.tasks)

        log(f"Total time: {format_time(total_time)}")
        log(f"Tasks solved: {solved}/{total} ({100*solved/total:.1f}%)")
        log(f"Final grammar size: {len(self.grammar)}")
        log("")

        # Solved tasks by family
        log("SOLVED BY FAMILY:")
        family_stats = defaultdict(lambda: {"solved": 0, "total": 0})
        for task in self.tasks:
            family_stats[task.family]["total"] += 1
            if self.task_metrics[task.name].solved:
                family_stats[task.family]["solved"] += 1

        for family, stats in sorted(family_stats.items()):
            log(f"  {family}: {stats['solved']}/{stats['total']}", 1)

        log("")

        # Enumeration efficiency
        if self.enumeration_stats:
            log("ENUMERATION EFFICIENCY:")
            total_complete = sum(s["programs_complete"] for s in self.enumeration_stats.values())
            total_partial = sum(s["programs_partial"] for s in self.enumeration_stats.values())
            log(f"  Total complete programs: {total_complete:,}", 1)
            log(f"  Total partial programs explored: {total_partial:,}", 1)
            log(f"  Ratio (partial/complete): {total_partial/max(1,total_complete):.2f}", 1)
            log("")

            avg_to_solve = sum(s["programs_complete"] for s in self.enumeration_stats.values()) / len(self.enumeration_stats)
            log(f"  Avg programs to solve: {avg_to_solve:.0f}", 1)

        # Learning curve
        log("")
        log("LEARNING CURVE:")
        for m in self.iteration_metrics:
            log(f"  Iter {m['iteration']+1}: {m['tasks_solved']}/{m['tasks_total']} solved, "
                f"{m['programs_enumerated']:,} complete, {m['partial_programs_explored']:,} partial", 1)

        # Unsolved tasks
        unsolved = [name for name, tm in self.task_metrics.items() if not tm.solved]
        if unsolved:
            log("")
            log(f"UNSOLVED TASKS ({len(unsolved)}):")
            for name in unsolved:
                log(f"  - {name}", 1)

        # Sample solutions
        solved_tasks = [(name, tm) for name, tm in self.task_metrics.items() if tm.solved]
        if solved_tasks:
            log("")
            log(f"SAMPLE SOLUTIONS:")
            for name, tm in sorted(solved_tasks, key=lambda x: x[1].programs_to_solve)[:5]:
                prog = tm.best_program[:60] + "..." if len(tm.best_program) > 60 else tm.best_program
                log(f"  {name}: {prog}", 1)
                log(f"    ({tm.programs_to_solve:,} programs, iter {tm.iteration_solved+1})", 2)

    def _compile_results(self, total_time: float) -> Dict:
        """Compile results for saving."""
        return {
            "config": self.config,
            "summary": {
                "total_time": total_time,
                "iterations_run": len(self.iteration_metrics),
                "tasks_solved": sum(1 for tm in self.task_metrics.values() if tm.solved),
                "tasks_total": len(self.tasks),
                "final_grammar_size": len(self.grammar),
                "total_abstractions": sum(len(m["new_abstractions"]) for m in self.iteration_metrics),
                "total_dreams": sum(m["dreams_generated"] for m in self.iteration_metrics)
            },
            "learning_curve": self.iteration_metrics,
            "task_metrics": {name: asdict(tm) for name, tm in self.task_metrics.items()},
            "enumeration_stats": self.enumeration_stats,
            "library_evolution": self.library_history
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    start_time = time.time()

    print_banner("TOP-DOWN ENUMERATION TEST")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Set seed
    random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])

    # Get rules
    rules = get_easy_pretraining_rules()
    print(f"\nPre-training rules: {len(rules)}")
    print("Rules:")
    for i, rule in enumerate(rules, 1):
        print(f"  {i:2d}. {rule.id} ({rule.family})")

    # Create tasks
    print(f"\nCreating tasks with {CONFIG['n_examples']} examples each...")
    tasks = create_tasks_from_rules(
        rules,
        n_examples=CONFIG["n_examples"],
        n_holdout=CONFIG["n_holdout"],
        hand_size=CONFIG["hand_size"],
        seed=CONFIG["seed"]
    )
    print(f"Created {len(tasks)} tasks")

    # Example balance check
    print("\nExample balance (first 3 tasks):")
    for task in tasks[:3]:
        pos = sum(1 for _, label in task.examples if label)
        neg = sum(1 for _, label in task.examples if not label)
        print(f"  {task.name}: {pos} pos / {neg} neg")

    # Build grammar
    print("\nBuilding lean grammar...")
    grammar = build_lean_grammar()
    print(f"Grammar: {len(grammar)} primitives")

    # Print configuration
    print_banner("CONFIGURATION")
    for k, v in CONFIG.items():
        print(f"  {k}: {v}")

    # Create eval function
    eval_fn = make_eval_fn()

    # Run
    print_banner("STARTING TOP-DOWN WAKE-SLEEP LEARNING")

    dc = TopDownDreamCoder(
        grammar=grammar,
        tasks=tasks,
        eval_fn=eval_fn,
        config=CONFIG
    )

    results = dc.run()

    # Save results
    log_dir = Path("results/topdown_test")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    result_path = log_dir / f"topdown_test_{timestamp}.json"

    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {result_path}")

    total_time = time.time() - start_time
    print_banner("TEST COMPLETE")
    print(f"Total wall time: {format_time(total_time)}")


if __name__ == "__main__":
    main()
