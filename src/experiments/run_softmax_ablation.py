#!/usr/bin/env python3
"""
Softmax Ablation Experiment
===========================

This script tests whether the contrastive model's poor search guidance is due to
its sigmoid output (classification) rather than its contrastive encoding (τ = pos - neg).

Hypothesis:
- H1: The problem is the sigmoid output, not the contrastive encoding.
        Adding softmax to contrastive should restore search guidance quality.
- H0: The τ = pos - neg encoding lacks discriminative power.
        Softmax alone won't fix it.

Three Conditions (WARM only - we know pretraining matters):
1. ContrastiveSigmoid: Original contrastive model (sigmoid + BCE)
2. ContrastiveSoftmax: Contrastive encoder + softmax output + CE loss
3. Neural: GRU baseline (softmax + CE)

Key Metrics:
- Solve rate on catalogue rules
- Programs per solution (search efficiency)
- Programs enumerated per task (overall search behavior)
- Prediction entropy (distribution focus)
- Recall@5 (are correct primitives in top predictions?)

Usage:
    # Run full ablation (all 3 conditions)
    python run_softmax_ablation.py

    # Quick test
    python run_softmax_ablation.py --quick-test

    # Single condition
    python run_softmax_ablation.py --model contrastive_softmax

Author: Can Konuk
Date: December 2025
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
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Set, Union
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np

from dreamcoder_core.type_system import arrow, HAND, BOOL
from dreamcoder_core.program import Program, Primitive, Invented
from dreamcoder_core.grammar import Grammar
from dreamcoder_core.enumeration import TopDownEnumerator
from dreamcoder_core.neural_recognition import NeuralRecognitionModel
from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.dreamcoder_original import (
    Task, SolutionEntry, TaskFrontier, make_eval_fn, create_tasks_from_rules
)

# Rules
from rules.pretraining_rules import get_all_pretraining_rules
from rules.catalogue import create_all_rules as get_catalogue_rules
from rules.cards import sample_hand


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AblationConfig:
    """Configuration for the softmax ablation experiment."""

    # Identification
    seed: int = 42
    run_id: str = ""

    # Pretraining configuration
    pretrain_iterations: int = 5
    pretrain_budget: int = 50000
    pretrain_depth: int = 7
    pretrain_epochs: int = 15
    pretrain_timeout: float = 30.0

    # Main training configuration
    main_iterations: int = 6
    main_budget: int = 200000
    main_depth: int = 9
    main_epochs: int = 10
    main_timeout: float = 60.0

    # Recognition model
    hidden_dim: int = 128
    learning_rate: float = 1e-3
    blend_factor: float = 0.5

    # Architecture
    n_examples: int = 100
    n_holdout: int = 20
    hand_size: int = 6

    # Logging
    log_dir: str = "results/softmax_ablation"
    verbose: bool = True

    # Quick test mode
    quick_test: bool = False

    def __post_init__(self):
        if not self.run_id:
            self.run_id = f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


def print_banner(text: str, char: str = "="):
    """Print a banner."""
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


def compute_entropy(probs: torch.Tensor) -> float:
    """Compute entropy of a probability distribution."""
    # Clip to avoid log(0)
    probs = probs.clamp(min=1e-10)
    return -float((probs * probs.log()).sum())


def compute_recall_at_k(predicted_probs: torch.Tensor,
                        actual_primitives: Set[str],
                        primitive_names: List[str],
                        k: int = 5) -> float:
    """Compute Recall@k: what fraction of actual primitives are in top-k predictions."""
    if not actual_primitives:
        return 1.0  # No primitives to recall

    # Get top-k indices
    _, top_k_indices = torch.topk(predicted_probs, min(k, len(primitive_names)))
    top_k_names = {primitive_names[i] for i in top_k_indices.cpu().numpy()}

    # Compute recall
    hits = len(actual_primitives & top_k_names)
    return hits / len(actual_primitives)


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(model_type: str, grammar: Grammar, config: AblationConfig, device: str) -> Any:
    """
    Create a recognition model of the specified type.

    Args:
        model_type: One of 'neural', 'contrastive_sigmoid', 'contrastive_softmax'
        grammar: The grammar to use
        config: Experiment configuration
        device: Device to use ('cpu' or 'cuda')

    Returns:
        Recognition model instance
    """
    if model_type == 'neural':
        # NeuralRecognitionModel uses hidden_dim parameter
        return NeuralRecognitionModel(
            grammar=grammar,
            hidden_dim=config.hidden_dim,
            learning_rate=config.learning_rate,
            device=device
        )
    elif model_type == 'contrastive_sigmoid':
        # ContrastiveRecognitionModel uses card_hidden/card_out/pred_hidden
        return ContrastiveRecognitionModel(
            grammar=grammar,
            card_hidden=64,
            card_out=32,
            pred_hidden=64,
            learning_rate=config.learning_rate,
            device=device,
            output_mode='sigmoid'  # Original BCE-based model
        )
    elif model_type == 'contrastive_softmax':
        # ContrastiveRecognitionModel with softmax output
        return ContrastiveRecognitionModel(
            grammar=grammar,
            card_hidden=64,
            card_out=32,
            pred_hidden=64,
            learning_rate=config.learning_rate,
            device=device,
            output_mode='softmax'  # New CE-based model
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# ABLATION EXPERIMENT RUNNER
# ============================================================================

class AblationExperiment:
    """
    Run the softmax ablation experiment.

    Tests three model types on WARM condition to isolate the effect of
    output activation function vs encoder architecture.
    """

    def __init__(self, config: AblationConfig):
        self.config = config
        self.device = 'cpu'  # Card game tasks are CPU-bound

        # Set random seeds
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Create log directory
        self.log_dir = Path(config.log_dir) / config.run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Initialize grammar
        self.grammar = build_lean_grammar()

        # Create tasks
        self._setup_tasks()

        print(f"Experiment initialized: {config.run_id}")
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

        # Quick test mode: reduce task counts
        if self.config.quick_test:
            self.pretrain_tasks = self.pretrain_tasks[:10]
            self.catalogue_tasks = self.catalogue_tasks[:10]
            self.pretrain_task_map = {t.name: t for t in self.pretrain_tasks}
            self.catalogue_task_map = {t.name: t for t in self.catalogue_tasks}

        # Evaluation function
        self.eval_fn = make_eval_fn()

    def log(self, message: str):
        """Log a message."""
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
        """
        Enumerate programs for a task until solution found or budget exhausted.
        """
        frontier = TaskFrontier(task)

        enumerator = TopDownEnumerator(
            grammar,
            max_depth=max_depth,
            max_programs=max_programs
        )

        start_time = time.time()
        programs_tried = 0
        last_progress = 0

        for program, log_prob in enumerator.enumerate(
            task.request_type,
            timeout_seconds=timeout
        ):
            programs_tried += 1

            # Progress output
            if show_progress and programs_tried - last_progress >= 1000:
                elapsed = time.time() - start_time
                rate = programs_tried / elapsed if elapsed > 0 else 0
                print(f"    [{task.name}] {programs_tried:,} programs ({rate:.0f}/s)...", end='\r', flush=True)
                last_progress = programs_tried

            if programs_tried > max_programs:
                break
            if time.time() - start_time > timeout:
                break

            # Evaluate on examples
            try:
                correct = 0
                for inp, expected in task.examples:
                    result = self.eval_fn(program, inp)
                    if result == expected:
                        correct += 1

                if correct == len(task.examples):
                    # Found a solution - verify on holdout
                    holdout_correct = 0
                    if hasattr(task, 'holdout_examples') and task.holdout_examples:
                        for inp, expected in task.holdout_examples:
                            try:
                                if self.eval_fn(program, inp) == expected:
                                    holdout_correct += 1
                            except Exception:
                                pass

                        # Require at least 80% holdout accuracy
                        holdout_rate = holdout_correct / len(task.holdout_examples)
                        if holdout_rate < 0.8:
                            continue  # Spurious solution

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
                    break  # Found valid solution

            except Exception:
                pass  # Program crashed - skip

        frontier.total_programs_searched = programs_tried
        frontier.total_time = time.time() - start_time
        return frontier

    def run_warm_condition(self, model_type: str) -> Dict:
        """
        Run WARM condition for a specific model type.

        Pretrains on easy rules, then tests on catalogue rules.
        """
        print_banner(f"RUNNING {model_type.upper()}")

        # Create fresh model
        model = create_model(model_type, self.grammar, self.config, self.device)

        # ===== PHASE 1: PRETRAINING =====
        print_banner(f"PHASE 1: PRETRAINING ({model_type})", char="-")
        self.log(f"Training on {len(self.pretrain_tasks)} pretraining rules")

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
                    if model_type == 'neural':
                        biased_grammar = model.predict_grammar_weights(
                            task, blend_factor=self.config.blend_factor
                        )
                    else:
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

        self.log(f"\nPretraining complete: {len(pretrain_frontiers)}/{len(self.pretrain_tasks)} solved")

        # ===== PHASE 2: MAIN TRAINING =====
        print_banner(f"PHASE 2: MAIN TRAINING ({model_type})", char="-")
        self.log(f"Training on {len(self.catalogue_tasks)} catalogue rules")

        phase_start = time.time()
        main_frontiers = {}
        task_metrics = {}

        # Collect prediction metrics before enumeration
        prediction_metrics = self._collect_prediction_metrics(model, self.catalogue_tasks)

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
                if model_type == 'neural':
                    biased_grammar = model.predict_grammar_weights(
                        task, blend_factor=self.config.blend_factor
                    )
                else:
                    biased_grammar = model.predict_grammar_weights(task)

                frontier = self.enumerate_task(
                    task,
                    biased_grammar,
                    max_programs=self.config.main_budget,
                    max_depth=self.config.main_depth,
                    timeout=self.config.main_timeout
                )

                iter_programs += frontier.total_programs_searched

                # Record per-task metrics
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
            all_solved = list(main_frontiers.keys())
            if all_solved:
                solved_tasks = [self.catalogue_task_map[name] for name in all_solved]
                loss = model.train_on_frontiers(
                    solved_tasks, main_frontiers, epochs=self.config.main_epochs
                )
                self.log(f"  Recognition loss: {loss:.4f}")

        main_results = {
            'total_time': time.time() - phase_start,
            'tasks_solved': len(main_frontiers),
            'tasks_total': len(self.catalogue_tasks),
            'solve_rate': len(main_frontiers) / len(self.catalogue_tasks),
            'solved_tasks': list(main_frontiers.keys()),
            'task_metrics': task_metrics,
            'prediction_metrics': prediction_metrics
        }

        self.log(f"\nMain training complete: {len(main_frontiers)}/{len(self.catalogue_tasks)} solved")
        self.log(f"Solve rate: {main_results['solve_rate']*100:.1f}%")

        return {
            'model_type': model_type,
            'pretraining': pretrain_results,
            'main_training': main_results
        }

    def _collect_prediction_metrics(self, model, tasks: List[Task]) -> Dict:
        """
        Collect prediction quality metrics before enumeration.

        For each task, records:
        - Prediction entropy (how focused is the distribution?)
        - Top-5 predicted primitives
        """
        metrics = {}

        for task in tasks:
            probs = model.predict_primitives(task)

            # Get primitive names from model
            if hasattr(model, 'primitive_names'):
                prim_names = model.primitive_names
            else:
                prim_names = [str(p.program) for p in model.grammar.productions]

            entropy = compute_entropy(probs)

            # Top-5 predictions
            values, indices = torch.topk(probs, min(5, len(prim_names)))
            top5 = [(prim_names[i], float(v)) for v, i in zip(values.cpu(), indices.cpu())]

            metrics[task.name] = {
                'entropy': entropy,
                'top5_predictions': top5,
                'max_prob': float(probs.max()),
                'min_prob': float(probs.min()),
                'prob_std': float(probs.std())
            }

        # Aggregate metrics
        avg_entropy = np.mean([m['entropy'] for m in metrics.values()])
        avg_max_prob = np.mean([m['max_prob'] for m in metrics.values()])
        avg_std = np.mean([m['prob_std'] for m in metrics.values()])

        return {
            'per_task': metrics,
            'aggregate': {
                'avg_entropy': float(avg_entropy),
                'avg_max_prob': float(avg_max_prob),
                'avg_prob_std': float(avg_std)
            }
        }

    def run(self, model_types: List[str] = None) -> Dict:
        """
        Run the full ablation experiment.

        Args:
            model_types: List of model types to test.
                        Default: ['neural', 'contrastive_sigmoid', 'contrastive_softmax']

        Returns:
            Results dictionary with all conditions
        """
        if model_types is None:
            model_types = ['neural', 'contrastive_sigmoid', 'contrastive_softmax']

        print_banner("SOFTMAX ABLATION EXPERIMENT")
        print(f"Run ID: {self.config.run_id}")
        print(f"Log directory: {self.log_dir}")
        print(f"Model types: {model_types}")

        results = {
            'config': asdict(self.config),
            'conditions': {}
        }

        start_time = time.time()

        for model_type in model_types:
            condition_results = self.run_warm_condition(model_type)
            results['conditions'][model_type] = condition_results

            # Save intermediate results
            results_path = self.log_dir / f"results_{model_type}.json"
            with open(results_path, 'w') as f:
                json.dump(condition_results, f, indent=2)
            self.log(f"\nSaved results to: {results_path}")

        results['total_time'] = time.time() - start_time

        # Save combined results
        combined_path = self.log_dir / "combined_results.json"
        with open(combined_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Print comparison
        self._print_comparison(results)

        return results

    def _print_comparison(self, results: Dict):
        """Print a comparison table of all conditions."""
        print_banner("COMPARISON: MODEL TYPES")

        conditions = results['conditions']

        print(f"{'Model Type':<25} {'Pretrain':<12} {'Main':<12} {'Avg Entropy':<12}")
        print("-" * 65)

        for model_type, cond in conditions.items():
            pretrain_rate = cond['pretraining']['solve_rate'] * 100
            main_rate = cond['main_training']['solve_rate'] * 100
            pred_metrics = cond['main_training'].get('prediction_metrics', {})
            avg_entropy = pred_metrics.get('aggregate', {}).get('avg_entropy', float('nan'))

            print(f"{model_type:<25} {pretrain_rate:>10.1f}% {main_rate:>10.1f}% {avg_entropy:>10.2f}")

        print()

        # Programs per solution analysis
        print("\nPrograms per Solution (solved tasks only):")
        print(f"{'Model Type':<25} {'Mean':<12} {'Median':<12}")
        print("-" * 50)

        for model_type, cond in conditions.items():
            task_metrics = cond['main_training']['task_metrics']
            programs = [m['programs_enumerated'] for m in task_metrics.values() if m['solved']]
            if programs:
                print(f"{model_type:<25} {np.mean(programs):>10.0f} {np.median(programs):>10.0f}")
            else:
                print(f"{model_type:<25} {'N/A':>10} {'N/A':>10}")

        print()

        # Hypothesis test
        if 'contrastive_sigmoid' in conditions and 'contrastive_softmax' in conditions:
            sigmoid_rate = conditions['contrastive_sigmoid']['main_training']['solve_rate']
            softmax_rate = conditions['contrastive_softmax']['main_training']['solve_rate']

            print("HYPOTHESIS TEST:")
            print(f"  ContrastiveSigmoid solve rate: {sigmoid_rate*100:.1f}%")
            print(f"  ContrastiveSoftmax solve rate: {softmax_rate*100:.1f}%")

            if softmax_rate > sigmoid_rate:
                print("  → H1 SUPPORTED: Softmax output improves contrastive model")
            elif softmax_rate == sigmoid_rate:
                print("  → INCONCLUSIVE: No difference observed")
            else:
                print("  → H0 SUPPORTED: Problem may be the contrastive encoding itself")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Softmax Ablation Experiment")
    parser.add_argument('--model', type=str, default=None,
                        help='Single model to test (neural, contrastive_sigmoid, contrastive_softmax)')
    parser.add_argument('--quick-test', action='store_true',
                        help='Run quick test with reduced budget')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Create config
    config = AblationConfig(
        seed=args.seed,
        quick_test=args.quick_test
    )

    if args.quick_test:
        # Reduce budgets for quick test
        config.pretrain_iterations = 2
        config.pretrain_budget = 5000
        config.main_iterations = 2
        config.main_budget = 10000

    # Determine which models to run
    if args.model:
        model_types = [args.model]
    else:
        model_types = ['neural', 'contrastive_sigmoid', 'contrastive_softmax']

    # Run experiment
    experiment = AblationExperiment(config)
    results = experiment.run(model_types)

    print_banner("EXPERIMENT COMPLETE")
    print(f"Results saved to: {experiment.log_dir}")


if __name__ == '__main__':
    main()
