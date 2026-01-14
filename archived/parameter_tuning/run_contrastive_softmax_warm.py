#!/usr/bin/env python3
"""
Contrastive Softmax WARM Experiment
===================================

Runs the WARM condition with ContrastiveRecognitionModel using softmax output.
This tests the hypothesis that the sigmoid output (not the contrastive encoding)
was responsible for the contrastive model's poor search guidance.

Comparison baseline (from previous runs):
- Neural WARM: 6/45 solved (13.3%), +2 transfer
- Contrastive Sigmoid WARM: 4/45 solved (8.9%), 0 transfer

Usage:
    python run_contrastive_softmax_warm.py

    # Quick test
    python run_contrastive_softmax_warm.py --quick-test
"""

import sys
import os
import time
import json
import random
import copy
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Set

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from dreamcoder_core.type_system import arrow, HAND, BOOL
from dreamcoder_core.program import Program, Primitive, Invented
from dreamcoder_core.grammar import Grammar
from dreamcoder_core.enumeration import TopDownEnumerator
from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.dreamcoder_original import (
    Task, SolutionEntry, TaskFrontier, make_eval_fn, create_tasks_from_rules
)

# Rules
from rules.pretraining_rules import get_all_pretraining_rules
from rules.catalogue import create_all_rules as get_catalogue_rules


# ============================================================================
# CONFIGURATION (matching previous experiments)
# ============================================================================

@dataclass
class Config:
    """Configuration matching previous warm-start experiments."""
    seed: int = 42
    run_id: str = ""

    # Pretraining (same as previous)
    pretrain_iterations: int = 5
    pretrain_budget: int = 50000
    pretrain_depth: int = 7
    pretrain_epochs: int = 15
    pretrain_timeout: float = 30.0

    # Main training (same as previous)
    main_iterations: int = 6
    main_budget: int = 200000
    main_depth: int = 9
    main_epochs: int = 10
    main_timeout: float = 60.0

    # Recognition model
    learning_rate: float = 1e-3
    blend_factor: float = 0.5

    # Data
    n_examples: int = 100
    n_holdout: int = 20
    hand_size: int = 6

    # Logging
    log_dir: str = "results/warmstart_experiment"
    verbose: bool = True
    quick_test: bool = False

    def __post_init__(self):
        if not self.run_id:
            self.run_id = f"contrastive_softmax_WARM_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def format_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def print_banner(text: str, char: str = "="):
    line = char * 70
    print(f"\n{line}")
    print(text)
    print(f"{line}\n", flush=True)


def collect_primitives(program: Program) -> Set[str]:
    """Extract all primitive names used in a program."""
    primitives = set()

    def recurse(p):
        if isinstance(p, Primitive):
            primitives.add(str(p))
        elif isinstance(p, Invented):
            primitives.add(str(p))
        elif hasattr(p, 'f'):
            recurse(p.f)
            if hasattr(p, 'x'):
                recurse(p.x)
        elif hasattr(p, 'body'):
            recurse(p.body)

    recurse(program)
    return primitives


# ============================================================================
# EXPERIMENT
# ============================================================================

class ContrastiveSoftmaxExperiment:
    """Run WARM condition with contrastive softmax model."""

    def __init__(self, config: Config):
        self.config = config
        self.device = 'cpu'

        # Set seeds
        random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Create log directory
        self.log_dir = Path(config.log_dir) / config.run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize grammar
        self.grammar = build_lean_grammar()

        # Create tasks
        self._setup_tasks()

        # Evaluation function
        self.eval_fn = make_eval_fn()

        print(f"Experiment initialized: {config.run_id}")
        print(f"Recognition model: CONTRASTIVE (softmax output)")
        print(f"Grammar: {len(self.grammar.productions)} primitives")
        print(f"Pretraining rules: {len(self.pretrain_tasks)}")
        print(f"Catalogue rules: {len(self.catalogue_tasks)}")

    def _setup_tasks(self):
        """Set up pretraining and catalogue tasks."""
        # Pretraining tasks
        pretraining_rules = get_all_pretraining_rules()
        self.pretrain_tasks = create_tasks_from_rules(
            pretraining_rules,
            n_examples=self.config.n_examples,
            n_holdout=self.config.n_holdout,
            hand_size=self.config.hand_size
        )
        self.pretrain_task_map = {t.name: t for t in self.pretrain_tasks}

        # Catalogue tasks
        catalogue_rules = get_catalogue_rules()
        self.catalogue_tasks = create_tasks_from_rules(
            catalogue_rules,
            n_examples=self.config.n_examples,
            n_holdout=self.config.n_holdout,
            hand_size=self.config.hand_size
        )
        self.catalogue_task_map = {t.name: t for t in self.catalogue_tasks}

        # Quick test mode
        if self.config.quick_test:
            self.pretrain_tasks = self.pretrain_tasks[:10]
            self.catalogue_tasks = self.catalogue_tasks[:10]
            self.pretrain_task_map = {t.name: t for t in self.pretrain_tasks}
            self.catalogue_task_map = {t.name: t for t in self.catalogue_tasks}

    def log(self, message: str):
        if self.config.verbose:
            print(message, flush=True)

    def enumerate_task(
        self,
        task: Task,
        grammar: Grammar,
        max_programs: int,
        max_depth: int,
        timeout: float,
        show_progress: bool = False
    ) -> TaskFrontier:
        """Enumerate programs for a task."""
        frontier = TaskFrontier(task)

        enumerator = TopDownEnumerator(
            grammar,
            max_depth=max_depth,
            max_programs=max_programs
        )

        start_time = time.time()
        programs_tried = 0

        for program, log_prob in enumerator.enumerate(
            task.request_type,
            timeout_seconds=timeout
        ):
            programs_tried += 1

            if programs_tried > max_programs:
                break
            if time.time() - start_time > timeout:
                break

            try:
                correct = 0
                for inp, expected in task.examples:
                    result = self.eval_fn(program, inp)
                    if result == expected:
                        correct += 1

                if correct == len(task.examples):
                    # Verify on holdout
                    holdout_correct = 0
                    if hasattr(task, 'holdout_examples') and task.holdout_examples:
                        for inp, expected in task.holdout_examples:
                            try:
                                if self.eval_fn(program, inp) == expected:
                                    holdout_correct += 1
                            except Exception:
                                pass

                        holdout_rate = holdout_correct / len(task.holdout_examples)
                        if holdout_rate < 0.8:
                            continue

                    entry = SolutionEntry(
                        program=program,
                        log_probability=log_prob,
                        log_likelihood=0.0,
                        programs_enumerated=programs_tried,
                        time_found=time.time() - start_time
                    )
                    frontier.add(entry)
                    frontier.total_programs_searched = programs_tried
                    frontier.total_time = time.time() - start_time
                    break

            except Exception:
                pass

        frontier.total_programs_searched = programs_tried
        frontier.total_time = time.time() - start_time
        return frontier

    def run(self) -> Dict:
        """Run the full WARM condition."""
        print_banner("WARM-START EXPERIMENT (Contrastive Softmax)")
        print(f"Run ID: {self.config.run_id}")
        print(f"Log directory: {self.log_dir}")
        print(f"Seed: {self.config.seed}")

        experiment_start = time.time()

        # Create model with softmax output
        model = ContrastiveRecognitionModel(
            grammar=self.grammar,
            card_hidden=64,
            card_out=32,
            pred_hidden=64,
            learning_rate=self.config.learning_rate,
            device=self.device,
            output_mode='softmax'  # KEY: Using softmax instead of sigmoid
        )

        # ===== PHASE 1: PRETRAINING =====
        print_banner("PHASE 1: WARM-START PRETRAINING")
        self.log(f"Training on {len(self.pretrain_tasks)} pretraining rules")
        self.log(f"Budget per task: {self.config.pretrain_budget} programs")
        self.log(f"Max depth: {self.config.pretrain_depth}")

        phase_start = time.time()
        pretrain_frontiers = {}
        all_solved = []

        for iteration in range(self.config.pretrain_iterations):
            iter_start = time.time()
            iter_solved = 0
            iter_programs = 0

            self.log(f"\n--- Pretraining Iteration {iteration + 1}/{self.config.pretrain_iterations} ---")

            for task in self.pretrain_tasks:
                if task.name in pretrain_frontiers and pretrain_frontiers[task.name].solved:
                    iter_solved += 1
                    continue

                # Get grammar weights
                if iteration > 0:
                    biased_grammar = model.predict_grammar_weights(task)
                else:
                    biased_grammar = self.grammar

                frontier = self.enumerate_task(
                    task,
                    biased_grammar,
                    max_programs=self.config.pretrain_budget,
                    max_depth=self.config.pretrain_depth,
                    timeout=self.config.pretrain_timeout
                )

                iter_programs += frontier.total_programs_searched

                if frontier.solved:
                    pretrain_frontiers[task.name] = frontier
                    iter_solved += 1
                    all_solved.append(task.name)
                    if self.config.verbose:
                        print(f"  ✓ {task.name}: {frontier.best.programs_enumerated} programs")

            self.log(f"  Solved: {iter_solved}/{len(self.pretrain_tasks)}")
            self.log(f"  Programs: {iter_programs:,}")
            self.log(f"  Time: {format_time(time.time() - iter_start)}")

            # Train recognition
            if all_solved:
                solved_tasks = [self.pretrain_task_map[name] for name in all_solved]
                loss = model.train_on_frontiers(
                    solved_tasks, pretrain_frontiers, epochs=self.config.pretrain_epochs
                )
                self.log(f"  Recognition loss: {loss:.4f}")

        pretrain_results = {
            'total_time': time.time() - phase_start,
            'tasks_solved': len(pretrain_frontiers),
            'tasks_total': len(self.pretrain_tasks),
            'solve_rate': len(pretrain_frontiers) / len(self.pretrain_tasks),
            'solved_tasks': list(pretrain_frontiers.keys())
        }

        self.log(f"\nPretraining complete: {len(pretrain_frontiers)}/{len(self.pretrain_tasks)} rules solved")
        self.log(f"Total time: {format_time(pretrain_results['total_time'])}")

        # Save pretrained model
        model_path = self.log_dir / "pretrained_recognition.pt"
        model.save(str(model_path))
        self.log(f"Saved pretrained model to: {model_path}")

        # ===== PHASE 2: MAIN TRAINING =====
        print_banner("PHASE 2: MAIN TRAINING (WARM)")
        self.log(f"Training on {len(self.catalogue_tasks)} catalogue rules")
        self.log(f"Budget per task: {self.config.main_budget} programs")
        self.log(f"Max depth: {self.config.main_depth}")

        phase_start = time.time()
        main_frontiers = {}
        task_metrics = {}

        for iteration in range(self.config.main_iterations):
            iter_start = time.time()
            iter_solved = 0
            iter_programs = 0
            newly_solved = []

            self.log(f"\n--- Main Iteration {iteration + 1}/{self.config.main_iterations} ---")

            for task in self.catalogue_tasks:
                if task.name in main_frontiers and main_frontiers[task.name].solved:
                    iter_solved += 1
                    continue

                # Get recognition-guided grammar
                biased_grammar = model.predict_grammar_weights(task)

                frontier = self.enumerate_task(
                    task,
                    biased_grammar,
                    max_programs=self.config.main_budget,
                    max_depth=self.config.main_depth,
                    timeout=self.config.main_timeout
                )

                iter_programs += frontier.total_programs_searched

                # Record metrics
                if task.name not in task_metrics:
                    task_metrics[task.name] = {
                        'solved': False,
                        'programs_enumerated': frontier.total_programs_searched
                    }

                if frontier.solved:
                    main_frontiers[task.name] = frontier
                    iter_solved += 1
                    newly_solved.append(task.name)

                    task_metrics[task.name].update({
                        'solved': True,
                        'iteration_solved': iteration + 1,
                        'programs_enumerated': frontier.best.programs_enumerated,
                        'time_to_solve': frontier.best.time_found,
                        'solution': str(frontier.best.program),
                        'primitives_used': list(collect_primitives(frontier.best.program))
                    })

                    if self.config.verbose:
                        print(f"  ✓ {task.name}: {frontier.best.programs_enumerated} programs")

            self.log(f"  Solved: {iter_solved}/{len(self.catalogue_tasks)}")
            self.log(f"  New this iteration: {len(newly_solved)}")
            self.log(f"  Programs: {iter_programs:,}")
            self.log(f"  Time: {format_time(time.time() - iter_start)}")

            # Train recognition
            all_main_solved = list(main_frontiers.keys())
            if all_main_solved:
                solved_tasks = [self.catalogue_task_map[name] for name in all_main_solved]
                loss = model.train_on_frontiers(
                    solved_tasks, main_frontiers, epochs=self.config.main_epochs
                )
                self.log(f"  Recognition loss: {loss:.4f}")

        main_results = {
            'condition': 'WARM',
            'total_time': time.time() - phase_start,
            'tasks_solved': len(main_frontiers),
            'tasks_total': len(self.catalogue_tasks),
            'solve_rate': len(main_frontiers) / len(self.catalogue_tasks),
            'solved_tasks': list(main_frontiers.keys()),
            'unsolved_tasks': [t.name for t in self.catalogue_tasks if t.name not in main_frontiers],
            'task_metrics': task_metrics
        }

        self.log(f"\nMain training complete: {len(main_frontiers)}/{len(self.catalogue_tasks)} rules solved")
        self.log(f"Solve rate: {main_results['solve_rate']*100:.1f}%")
        self.log(f"Total time: {format_time(main_results['total_time'])}")

        # ===== SAVE RESULTS =====
        results = {
            'condition': 'WARM',
            'model_type': 'contrastive_softmax',
            'start_time': datetime.now().isoformat(),
            'config': asdict(self.config),
            'pretraining': pretrain_results,
            'main_training': main_results,
            'task_metrics': task_metrics,
            'total_time': time.time() - experiment_start,
            'final_solve_rate': main_results['solve_rate']
        }

        results_path = self.log_dir / "results_WARM.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        self.log(f"Results saved to: {results_path}")

        # ===== COMPARISON =====
        print_banner("COMPARISON WITH PREVIOUS RESULTS")
        print("Model                    | Pretrain | Main   | Transfer")
        print("-" * 60)
        print(f"Neural (softmax/CE)      | 19/44    | 6/45   | +2 rules")
        print(f"Contrastive (sigmoid/BCE)| 16/44    | 4/45   | 0 rules")
        print(f"Contrastive (softmax/CE) | {pretrain_results['tasks_solved']}/44    | {main_results['tasks_solved']}/45   | TBD")
        print()

        if main_results['tasks_solved'] > 4:
            print("✓ HYPOTHESIS SUPPORTED: Softmax output improves contrastive model!")
            transfer_tasks = set(main_results['solved_tasks']) - {'Has_pair_ranks', 'Uniform_color', 'Exactly_two_suits', 'At_most_three_suits'}
            if transfer_tasks:
                print(f"  Transfer rules: {transfer_tasks}")
        elif main_results['tasks_solved'] == 4:
            print("✗ No improvement - problem may be in contrastive encoding itself")
        else:
            print("✗ Performance decreased - unexpected result")

        print_banner("EXPERIMENT COMPLETE")
        print(f"Results saved to: {self.log_dir}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Contrastive Softmax WARM Experiment")
    parser.add_argument('--quick-test', action='store_true', help='Run quick test')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    config = Config(
        seed=args.seed,
        quick_test=args.quick_test
    )

    if args.quick_test:
        config.pretrain_iterations = 2
        config.pretrain_budget = 5000
        config.main_iterations = 2
        config.main_budget = 10000

    experiment = ContrastiveSoftmaxExperiment(config)
    experiment.run()


if __name__ == '__main__':
    main()
