#!/usr/bin/env python3
"""
Comparison Experiment: Sigmoid vs Softmax Output Activation

This experiment tests whether softmax output activation improves search guidance
compared to sigmoid, now that the encoder produces discriminative task embeddings.

BACKGROUND:
- Earlier experiments (Dec 2025) showed GRU+Softmax > Contrastive+Sigmoid
- But ablation revealed the encoder (not activation) was the problem
- The encoder has since been improved
- This experiment re-tests the hypothesis with the improved encoder

HYPOTHESIS:
- H1: Softmax produces better search guidance (ranking) than sigmoid
- H0: Sigmoid and softmax perform similarly with the improved encoder

Conditions:
1. ContrastiveSigmoid: τ → MLP → sigmoid (multi-label, independent)
2. ContrastiveSoftmax: τ → MLP → softmax (distribution, competitive)

Metrics:
1. Recall@5, Recall@10, MRR (primitive prediction quality)
2. Prediction entropy (how focused is the distribution?)
3. Solve rate on pretraining rules (search guidance effectiveness)
4. Programs per solution (search efficiency)

Usage:
    python3 experiments/compare_sigmoid_softmax.py           # Full comparison
    python3 experiments/compare_sigmoid_softmax.py --quick   # Quick validation

Author: Can Konuk
Date: January 2026
"""

import sys
import os
import time
import json
import random
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import defaultdict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import sample_hand, Card
from rules.pretraining_rules import get_all_pretraining_rules
from rules.catalogue import create_all_rules as get_catalogue_rules
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.type_system import arrow, HAND, BOOL
from dreamcoder_core.grammar import Grammar, Production
from dreamcoder_core.program import Program, Primitive, Application, Abstraction
from dreamcoder_core.enumeration import TopDownEnumerator
from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.dreamcoder_original import (
    Task, SolutionEntry, TaskFrontier, make_eval_fn, create_tasks_from_rules
)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for sigmoid vs softmax comparison experiment."""
    # Task configuration
    n_pretrain_rules: int = 44           # All pretraining rules
    n_catalogue_rules: int = 45          # All catalogue rules
    n_examples_per_task: int = 50        # Examples per task
    n_holdout: int = 20                  # Holdout examples for validation
    hand_size: int = 6

    # Pretraining configuration
    pretrain_iterations: int = 4
    pretrain_budget: int = 50000
    pretrain_depth: int = 7
    pretrain_timeout: float = 30.0

    # Main evaluation
    eval_budget: int = 100000
    eval_depth: int = 8
    eval_timeout: float = 60.0

    # Recognition model configuration
    card_hidden: int = 64
    card_out: int = 32
    pred_hidden: int = 64
    learning_rate: float = 0.001
    epochs_per_iteration: int = 15
    batch_size: int = 8

    # Reproducibility
    seed: int = 42

    # Output
    output_dir: str = "results_sigmoid_softmax"

    # Quick test mode
    quick: bool = False


def print_flush(*args, **kwargs):
    """Print with immediate flush for background runs."""
    print(*args, **kwargs, flush=True)


# ============================================================================
# METRICS COMPUTATION
# ============================================================================

def compute_entropy(probs: torch.Tensor) -> float:
    """Compute entropy of a probability distribution."""
    probs = probs.clamp(min=1e-10)
    return -float((probs * probs.log()).sum())


def compute_recall_at_k(predicted_probs: torch.Tensor,
                        actual_primitives: Set[str],
                        primitive_names: List[str],
                        k: int) -> float:
    """Compute Recall@k."""
    if not actual_primitives:
        return 1.0

    _, top_k_indices = torch.topk(predicted_probs, min(k, len(primitive_names)))
    top_k_names = {primitive_names[i] for i in top_k_indices.cpu().numpy()}

    hits = len(actual_primitives & top_k_names)
    return hits / len(actual_primitives)


def compute_mrr(predicted_probs: torch.Tensor,
                actual_primitives: Set[str],
                primitive_names: List[str]) -> float:
    """Compute Mean Reciprocal Rank."""
    if not actual_primitives:
        return 1.0

    sorted_indices = torch.argsort(predicted_probs, descending=True).cpu().numpy()

    reciprocal_ranks = []
    for prim in actual_primitives:
        if prim in primitive_names:
            idx = primitive_names.index(prim)
            rank = np.where(sorted_indices == idx)[0]
            if len(rank) > 0:
                reciprocal_ranks.append(1.0 / (rank[0] + 1))

    return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0


def extract_primitives(program: Program) -> Set[str]:
    """Extract all primitive names used in a program."""
    primitives = set()

    def recurse(p):
        if isinstance(p, Primitive):
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
# EXPERIMENT RUNNER
# ============================================================================

class SigmoidSoftmaxExperiment:
    """Run sigmoid vs softmax comparison experiment."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = 'cpu'

        # Set seeds
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Initialize
        self.grammar = build_lean_grammar()
        self.eval_fn = make_eval_fn()

        # Create output directory
        self.run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = Path(config.output_dir) / f"comparison_{self.run_id}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        print_flush(f"Experiment initialized: {self.run_id}")
        print_flush(f"Output: {self.run_dir}")

    def create_model(self, output_mode: str) -> ContrastiveRecognitionModel:
        """Create a contrastive recognition model with specified output mode."""
        return ContrastiveRecognitionModel(
            grammar=self.grammar,
            card_hidden=self.config.card_hidden,
            card_out=self.config.card_out,
            pred_hidden=self.config.pred_hidden,
            learning_rate=self.config.learning_rate,
            device=self.device,
            output_mode=output_mode  # 'sigmoid' or 'softmax'
        )

    def create_tasks(self, rules, name: str) -> List[Task]:
        """Create tasks from rules."""
        tasks = create_tasks_from_rules(
            rules,
            n_examples=self.config.n_examples_per_task,
            n_holdout=self.config.n_holdout,
            hand_size=self.config.hand_size
        )
        print_flush(f"  Created {len(tasks)} {name} tasks")
        return tasks

    def enumerate_task(
        self,
        task: Task,
        grammar: Grammar,
        max_programs: int,
        max_depth: int,
        timeout: float
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
                correct = sum(
                    1 for inp, expected in task.examples
                    if self.eval_fn(program, inp) == expected
                )

                if correct == len(task.examples):
                    # Verify on holdout
                    if hasattr(task, 'holdout_examples') and task.holdout_examples:
                        holdout_correct = sum(
                            1 for inp, expected in task.holdout_examples
                            if self.eval_fn(program, inp) == expected
                        )
                        if holdout_correct / len(task.holdout_examples) < 0.8:
                            continue  # Spurious

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

    def pretrain_model(
        self,
        model: ContrastiveRecognitionModel,
        tasks: List[Task],
        output_mode: str
    ) -> Dict:
        """Pretrain model on pretraining tasks."""
        print_flush(f"\n  Pretraining {output_mode} model...")

        frontiers = {}
        solved_by_iteration = []

        for iteration in range(self.config.pretrain_iterations):
            iter_start = time.time()
            new_solved = 0

            for task in tasks:
                if task.name in frontiers and frontiers[task.name].solved:
                    continue

                # Get biased grammar
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

                if frontier.solved:
                    frontiers[task.name] = frontier
                    new_solved += 1

            # Train recognition model
            solved_tasks = [t for t in tasks if t.name in frontiers]
            if solved_tasks:
                loss = model.train_on_frontiers(
                    solved_tasks,
                    frontiers,
                    epochs=self.config.epochs_per_iteration
                )

            solved_by_iteration.append(len(frontiers))
            print_flush(f"    Iter {iteration+1}: {len(frontiers)} solved (+{new_solved}), "
                       f"time: {time.time()-iter_start:.1f}s")

        return {
            'tasks_solved': len(frontiers),
            'tasks_total': len(tasks),
            'solve_rate': len(frontiers) / len(tasks),
            'solved_by_iteration': solved_by_iteration,
            'solved_tasks': list(frontiers.keys())
        }

    def evaluate_predictions(
        self,
        model: ContrastiveRecognitionModel,
        tasks: List[Task],
        frontiers: Dict[str, TaskFrontier]
    ) -> Dict:
        """Evaluate prediction quality on solved tasks."""
        metrics = {
            'recall_at_5': [],
            'recall_at_10': [],
            'mrr': [],
            'entropy': [],
            'max_prob': [],
            'prob_std': []
        }

        prim_names = model.primitive_names

        for task in tasks:
            if task.name not in frontiers:
                continue

            frontier = frontiers[task.name]
            if not frontier.solved:
                continue

            # Get actual primitives
            actual_prims = extract_primitives(frontier.best.program)

            # Get predictions
            with torch.no_grad():
                probs = model.predict_primitives(task)

            # Compute metrics
            metrics['recall_at_5'].append(
                compute_recall_at_k(probs, actual_prims, prim_names, k=5)
            )
            metrics['recall_at_10'].append(
                compute_recall_at_k(probs, actual_prims, prim_names, k=10)
            )
            metrics['mrr'].append(
                compute_mrr(probs, actual_prims, prim_names)
            )
            metrics['entropy'].append(compute_entropy(probs))
            metrics['max_prob'].append(float(probs.max()))
            metrics['prob_std'].append(float(probs.std()))

        # Aggregate
        return {
            k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
            for k, v in metrics.items()
        }

    def run_condition(self, output_mode: str, pretrain_tasks: List[Task]) -> Dict:
        """Run one condition of the experiment."""
        print_flush(f"\n{'='*60}")
        print_flush(f"CONDITION: {output_mode.upper()}")
        print_flush(f"{'='*60}")

        start_time = time.time()

        # Create model
        model = self.create_model(output_mode)

        # Pretrain
        pretrain_results = self.pretrain_model(model, pretrain_tasks, output_mode)

        # Collect frontiers for evaluation
        frontiers = {}
        for task in pretrain_tasks:
            if task.name in pretrain_results['solved_tasks']:
                # Re-enumerate to get frontier
                biased_grammar = model.predict_grammar_weights(task)
                frontier = self.enumerate_task(
                    task,
                    biased_grammar,
                    max_programs=self.config.pretrain_budget,
                    max_depth=self.config.pretrain_depth,
                    timeout=self.config.pretrain_timeout
                )
                if frontier.solved:
                    frontiers[task.name] = frontier

        # Evaluate predictions
        prediction_metrics = self.evaluate_predictions(model, pretrain_tasks, frontiers)

        total_time = time.time() - start_time

        return {
            'output_mode': output_mode,
            'pretrain': pretrain_results,
            'prediction_metrics': prediction_metrics,
            'total_time': total_time
        }

    def run(self) -> Dict:
        """Run full comparison experiment."""
        print_flush("\n" + "=" * 70)
        print_flush("SIGMOID vs SOFTMAX COMPARISON EXPERIMENT")
        print_flush("=" * 70)
        print_flush(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print_flush()

        # Save config
        config_path = self.run_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(asdict(self.config), f, indent=2)

        # Create tasks
        print_flush("Creating tasks...")
        pretraining_rules = get_all_pretraining_rules()
        if self.config.quick:
            pretraining_rules = pretraining_rules[:15]
        pretrain_tasks = self.create_tasks(pretraining_rules, "pretraining")

        # Run conditions
        results = {}

        for output_mode in ['sigmoid', 'softmax']:
            try:
                results[output_mode] = self.run_condition(output_mode, pretrain_tasks)
            except Exception as e:
                print_flush(f"ERROR in {output_mode}: {e}")
                traceback.print_exc()
                results[output_mode] = {'error': str(e)}

        # Print comparison
        self._print_comparison(results)

        # Save results
        results_path = self.run_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        print_flush(f"\nResults saved to: {self.run_dir}")

        return results

    def _print_comparison(self, results: Dict):
        """Print comparison table."""
        print_flush("\n" + "=" * 70)
        print_flush("COMPARISON SUMMARY")
        print_flush("=" * 70)

        print_flush(f"\n{'Metric':<25} {'Sigmoid':<20} {'Softmax':<20}")
        print_flush("-" * 65)

        for mode in ['sigmoid', 'softmax']:
            if 'error' in results.get(mode, {}):
                continue

        sig = results.get('sigmoid', {})
        soft = results.get('softmax', {})

        # Solve rate
        sig_solve = sig.get('pretrain', {}).get('solve_rate', 0) * 100
        soft_solve = soft.get('pretrain', {}).get('solve_rate', 0) * 100
        print_flush(f"{'Solve Rate':<25} {sig_solve:>6.1f}%{'':<13} {soft_solve:>6.1f}%")

        # Prediction metrics
        for metric in ['recall_at_5', 'recall_at_10', 'mrr', 'entropy']:
            sig_m = sig.get('prediction_metrics', {}).get(metric, {})
            soft_m = soft.get('prediction_metrics', {}).get(metric, {})

            sig_val = sig_m.get('mean', 0)
            soft_val = soft_m.get('mean', 0)

            label = metric.replace('_', ' ').title()
            print_flush(f"{label:<25} {sig_val:>18.4f} {soft_val:>18.4f}")

        print_flush("\n" + "-" * 65)

        # Winner determination
        sig_r5 = sig.get('prediction_metrics', {}).get('recall_at_5', {}).get('mean', 0)
        soft_r5 = soft.get('prediction_metrics', {}).get('recall_at_5', {}).get('mean', 0)

        if soft_r5 > sig_r5 * 1.05:
            print_flush("RESULT: Softmax wins (better R@5)")
            print_flush("→ H1 SUPPORTED: Softmax improves search guidance")
        elif sig_r5 > soft_r5 * 1.05:
            print_flush("RESULT: Sigmoid wins (better R@5)")
            print_flush("→ H1 REJECTED: Sigmoid remains better")
        else:
            print_flush("RESULT: No significant difference")
            print_flush("→ Activation function doesn't matter with improved encoder")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Sigmoid vs Softmax Comparison")
    parser.add_argument('--quick', action='store_true', help='Quick test mode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    config = ExperimentConfig(
        seed=args.seed,
        quick=args.quick
    )

    if args.quick:
        config.pretrain_iterations = 2
        config.pretrain_budget = 10000
        config.epochs_per_iteration = 5

    experiment = SigmoidSoftmaxExperiment(config)
    results = experiment.run()

    print_flush("\n" + "=" * 70)
    print_flush("EXPERIMENT COMPLETE")
    print_flush("=" * 70)


if __name__ == '__main__':
    main()
