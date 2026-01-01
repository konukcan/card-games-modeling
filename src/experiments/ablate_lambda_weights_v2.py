#!/usr/bin/env python3
"""
Lambda Weight Grid Search Ablation Experiment (v2 - FIXED)

This script properly trains recognition models with different lambda combinations,
using REAL parsed programs from cached overnight runs.

Key fixes from v1:
1. Uses real Program objects (not fake MinimalProgram stubs)
2. Actually trains models (doesn't return immediately)
3. Grid searches over gradients of lambda values
4. Extracts real primitives from ASTs

Grid search parameters:
- λ_struct: [0.0, 0.1, 0.2, 0.3, 0.5, 1.0]
- λ_count:  [0.0, 0.05, 0.1, 0.2]
- λ_bigram: [0.0, 0.05, 0.1, 0.2]
- λ_pred:   1.0 (fixed baseline)

Pipeline A methodology: Train on rules with cached programs, 3-fold CV.

Expected runtime: ~2-3 hours (96 conditions × 3 folds × 30 epochs)

Usage:
    python3 experiments/ablate_lambda_weights_v2.py
    python3 experiments/ablate_lambda_weights_v2.py --quick  # Fast validation
"""

import sys
import os
import json
import glob
import time
import random
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import KFold

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import sample_hand, Card, Suit, Rank
from rules.catalogue import create_all_rules, ALL_RULES
from rules.pretraining_rules import get_all_pretraining_rules, PretrainingRule
from dreamcoder_core.lean_primitives import build_lean_grammar, build_lean_primitives
from dreamcoder_core.grammar import uniform_grammar
from dreamcoder_core.program import parse_program, Program, Primitive, Application, Abstraction, Invented
from dreamcoder_core.contrastive_recognition import (
    ContrastiveRecognitionModel,
    extract_bigrams,
    build_bigram_vocabulary
)
from dreamcoder_core.type_system import arrow, HAND, BOOL


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AblationConfig:
    """Configuration for the ablation experiment."""
    # Lambda grid values
    lambda_struct_values: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5, 1.0])
    lambda_count_values: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2])
    lambda_bigram_values: List[float] = field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2])
    lambda_pred: float = 1.0  # Fixed

    # Training params
    n_epochs: int = 30
    batch_size: int = 8
    lr: float = 0.001
    hidden_dim: int = 64

    # Data params
    n_examples: int = 50  # Examples per task
    hand_size: int = 6
    n_folds: int = 3

    # Other
    seed: int = 42
    results_dir: str = "results_lambda_grid"


@dataclass
class LambdaConfig:
    """Single lambda combination."""
    lambda_pred: float
    lambda_struct: float
    lambda_count: float
    lambda_bigram: float

    @property
    def name(self) -> str:
        return f"s{self.lambda_struct:.2f}_c{self.lambda_count:.2f}_b{self.lambda_bigram:.2f}"

    def __repr__(self) -> str:
        return f"λ(pred={self.lambda_pred}, struct={self.lambda_struct}, count={self.lambda_count}, bigram={self.lambda_bigram})"


# ============================================================================
# DATA LOADING (Real Programs)
# ============================================================================

def load_cached_programs(results_dirs: List[str] = None) -> Dict[str, Tuple[str, Program]]:
    """
    Load all parseable programs from cached frontier files.

    Returns:
        Dict mapping task_name -> (program_string, Program object)
    """
    if results_dirs is None:
        results_dirs = ["results_archive", "results_overnight", "results_systematic"]

    grammar = build_lean_grammar()
    primitives = {}
    for prod in grammar.productions:
        if hasattr(prod.program, 'name'):
            primitives[prod.program.name] = prod.program

    # Find all frontier files
    frontier_files = []
    for results_dir in results_dirs:
        if os.path.exists(results_dir):
            frontier_files.extend(glob.glob(f'{results_dir}/**/frontiers_*.json', recursive=True))

    logger.info(f"Found {len(frontier_files)} frontier files")

    # Collect all unique solved programs
    all_programs = {}  # task_name -> program_string

    for fpath in frontier_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
            for task_name, info in data.items():
                if isinstance(info, dict) and info.get('solved'):
                    prog_str = info.get('best_program', '')
                    if prog_str and task_name not in all_programs:
                        all_programs[task_name] = prog_str
        except Exception:
            pass

    # Parse programs
    parsed = {}
    for task_name, prog_str in all_programs.items():
        try:
            prog = parse_program(prog_str, primitives)
            parsed[task_name] = (prog_str, prog)
        except Exception:
            pass

    logger.info(f"Successfully parsed {len(parsed)}/{len(all_programs)} programs")
    return parsed, grammar


def extract_primitives_from_program(program: Program) -> Set[str]:
    """Extract primitive names from program AST."""
    primitives = set()

    def visit(node):
        if isinstance(node, Primitive):
            primitives.add(node.name)
        elif isinstance(node, Invented):
            primitives.add(str(node))
        elif isinstance(node, Application):
            visit(node.f)
            visit(node.x)
        elif isinstance(node, Abstraction):
            visit(node.body)

    visit(program)
    return primitives


# ============================================================================
# TASK & FRONTIER WRAPPERS
# ============================================================================

class FrontierEntry:
    """Mimics TaskFrontier entry structure."""
    def __init__(self, program: Program, log_likelihood: float = 0.0):
        self.program = program
        self.log_likelihood = log_likelihood


class RealFrontier:
    """Wrapper with real Program for train_on_frontiers compatibility."""
    def __init__(self, program: Program):
        self.solved = True
        self.entries = [FrontierEntry(program, 0.0)]


class TaskWithProgram:
    """Task that includes actual parsed program."""

    def __init__(self, rule, program: Program, program_str: str,
                 n_examples: int = 50, hand_size: int = 6):
        self.rule = rule
        self.id = rule.id if hasattr(rule, 'id') else rule.name
        self.name = self.id
        self.program = program
        self.program_str = program_str
        self.request = arrow(HAND, BOOL)
        self.examples = []

        # Sample positive and negative examples
        pos_count = neg_count = 0
        max_attempts = n_examples * 100
        attempts = 0

        while (pos_count < n_examples // 2 or neg_count < n_examples // 2) and attempts < max_attempts:
            hand = sample_hand(hand_size)
            try:
                # Handle different rule types
                if hasattr(rule, 'predicate'):
                    label = rule.predicate(hand)
                elif hasattr(rule, 'eval'):
                    label = rule.eval(hand)
                elif hasattr(rule, 'check'):
                    label = rule.check(hand)
                else:
                    label = rule(hand)

                if label and pos_count < n_examples // 2:
                    self.examples.append((hand, True))
                    pos_count += 1
                elif not label and neg_count < n_examples // 2:
                    self.examples.append((hand, False))
                    neg_count += 1
            except:
                pass
            attempts += 1

        # Extract primitives from program
        self.primitives_used = extract_primitives_from_program(program)

        # Extract bigrams from program
        self.bigrams = extract_bigrams(program)


def get_rule_by_id(rule_id: str, pretraining_rules: List, catalogue_rules: List):
    """Find a rule by ID from either pretraining or catalogue rules."""
    for rule in pretraining_rules:
        if rule.id == rule_id or rule.name == rule_id:
            return rule
    for rule in catalogue_rules:
        if rule.id == rule_id or rule.name == rule_id:
            return rule
    return None


def create_tasks_from_cached(
    cached_programs: Dict[str, Tuple[str, Program]],
    pretraining_rules: List,
    catalogue_rules: List,
    config: AblationConfig
) -> List[TaskWithProgram]:
    """Create tasks from cached programs with matching rules."""
    tasks = []

    for task_name, (prog_str, program) in cached_programs.items():
        rule = get_rule_by_id(task_name, pretraining_rules, catalogue_rules)
        if rule:
            try:
                task = TaskWithProgram(
                    rule=rule,
                    program=program,
                    program_str=prog_str,
                    n_examples=config.n_examples,
                    hand_size=config.hand_size
                )
                if len(task.examples) >= 10:  # Need minimum examples
                    tasks.append(task)
            except Exception as e:
                logger.debug(f"Failed to create task {task_name}: {e}")

    return tasks


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, tasks: List[TaskWithProgram]) -> Dict:
    """Evaluate recognition model on tasks."""
    model.eval()

    metrics = {
        'R@5': [], 'R@10': [], 'MRR': [],
        'n_tasks': len(tasks)
    }

    if not tasks:
        return {k: 0.0 if k != 'n_tasks' else 0 for k in metrics}

    with torch.no_grad():
        for task in tasks:
            if not task.primitives_used:
                continue

            # Get predictions as dict
            predictions = model.predict_primitives_dict(task)

            # Sort by probability (descending)
            sorted_preds = sorted(predictions.items(), key=lambda x: -x[1])
            predicted_names = [name for name, _ in sorted_preds]

            used = task.primitives_used

            # R@5: fraction of used primitives in top 5
            top5 = set(predicted_names[:5])
            r5 = len(top5 & used) / len(used) if used else 0
            metrics['R@5'].append(r5)

            # R@10
            top10 = set(predicted_names[:10])
            r10 = len(top10 & used) / len(used) if used else 0
            metrics['R@10'].append(r10)

            # MRR: reciprocal rank of first correct prediction
            mrr = 0.0
            for rank, name in enumerate(predicted_names, 1):
                if name in used:
                    mrr = 1.0 / rank
                    break
            metrics['MRR'].append(mrr)

    # Average
    return {
        'R@5': np.mean(metrics['R@5']) if metrics['R@5'] else 0.0,
        'R@10': np.mean(metrics['R@10']) if metrics['R@10'] else 0.0,
        'MRR': np.mean(metrics['MRR']) if metrics['MRR'] else 0.0,
        'n_tasks': len([t for t in tasks if t.primitives_used])
    }


# ============================================================================
# TRAINING
# ============================================================================

def train_single_condition(
    lambda_config: LambdaConfig,
    train_tasks: List[TaskWithProgram],
    test_tasks: List[TaskWithProgram],
    grammar,
    config: AblationConfig,
    fold_idx: int = 0
) -> Dict:
    """Train a single lambda condition and return metrics."""

    # Create model
    model = ContrastiveRecognitionModel(
        grammar=grammar,
        pred_hidden=config.hidden_dim,
        output_mode='sigmoid'
    )

    # Build bigram vocabulary from training programs
    all_programs = [t.program for t in train_tasks]
    bigram_vocab = build_bigram_vocabulary(all_programs, min_count=1)

    # Set bigram vocabulary if we're using bigram loss
    if lambda_config.lambda_bigram > 0 and bigram_vocab:
        model.bigram_vocab = bigram_vocab
        model.bigram_to_idx = {bg: i for i, bg in enumerate(bigram_vocab)}
        # Rebuild bigram head with correct size
        # The task embedding dimension is card_out (32), not hidden_dim (64)
        # Get the correct input dim from primitive_head.mlp
        task_embed_dim = model.primitive_head.mlp[0].in_features
        model.bigram_head = nn.Sequential(
            nn.Linear(task_embed_dim, task_embed_dim * 2),
            nn.ReLU(),
            nn.Linear(task_embed_dim * 2, len(bigram_vocab)),
            nn.Sigmoid()
        ).to(model.device)

    # Create frontiers with REAL programs
    frontiers = {}
    for task in train_tasks:
        frontiers[task.name] = RealFrontier(task.program)

    # Train
    start_time = time.time()
    final_loss = model.train_on_frontiers(
        tasks=train_tasks,
        frontiers=frontiers,
        epochs=config.n_epochs,
        batch_size=config.batch_size,
        lambda_struct=lambda_config.lambda_struct,
        lambda_count=lambda_config.lambda_count,
        lambda_pred=lambda_config.lambda_pred,
        lambda_bigram=lambda_config.lambda_bigram,
        use_bigram_loss=(lambda_config.lambda_bigram > 0 and bool(bigram_vocab))
    )
    train_time = time.time() - start_time

    # Evaluate
    metrics = evaluate_model(model, test_tasks)
    metrics['train_loss'] = final_loss
    metrics['train_time'] = train_time
    metrics['n_train'] = len(train_tasks)
    metrics['n_test'] = len(test_tasks)
    metrics['fold'] = fold_idx

    return metrics


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_grid_search(config: AblationConfig, quick: bool = False):
    """Run the full grid search ablation experiment."""

    print("=" * 70)
    print("LAMBDA WEIGHT GRID SEARCH ABLATION (v2 - FIXED)")
    print("=" * 70)
    start_time = datetime.now()
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Quick mode
    if quick:
        config.lambda_struct_values = [0.0, 0.3, 1.0]
        config.lambda_count_values = [0.0, 0.1]
        config.lambda_bigram_values = [0.0, 0.1]
        config.n_epochs = 10
        config.n_folds = 2
        print("QUICK MODE: Reduced grid and epochs")

    # Generate all lambda combinations
    lambda_configs = []
    for ls, lc, lb in product(
        config.lambda_struct_values,
        config.lambda_count_values,
        config.lambda_bigram_values
    ):
        lambda_configs.append(LambdaConfig(
            lambda_pred=config.lambda_pred,
            lambda_struct=ls,
            lambda_count=lc,
            lambda_bigram=lb
        ))

    print(f"Grid search: {len(lambda_configs)} lambda combinations")
    print(f"  λ_struct: {config.lambda_struct_values}")
    print(f"  λ_count:  {config.lambda_count_values}")
    print(f"  λ_bigram: {config.lambda_bigram_values}")
    print(f"Folds: {config.n_folds}")
    print(f"Epochs: {config.n_epochs}")
    print()

    # Load cached programs
    print("Loading cached programs...")
    cached_programs, grammar = load_cached_programs()

    if len(cached_programs) < 10:
        print("WARNING: Not enough cached programs found!")
        print("The experiment needs solved programs from overnight runs.")
        print("Checking for alternative data sources...")

        # Fall back to trying to parse expected_program strings from pretraining rules
        # (This is less reliable but better than nothing)
        primitives = {}
        for prod in grammar.productions:
            if hasattr(prod.program, 'name'):
                primitives[prod.program.name] = prod.program

        pretraining_rules = get_all_pretraining_rules()
        for rule in pretraining_rules:
            if hasattr(rule, 'expected_program') and rule.expected_program:
                try:
                    prog = parse_program(rule.expected_program, primitives)
                    cached_programs[rule.id] = (rule.expected_program, prog)
                except:
                    pass

        print(f"After fallback: {len(cached_programs)} programs available")

    if len(cached_programs) < 5:
        print("ERROR: Not enough programs to run experiment.")
        print("Please run an overnight experiment first to generate cached programs.")
        return None

    # Load rules
    pretraining_rules = get_all_pretraining_rules()
    catalogue_rules = ALL_RULES

    # Create tasks from cached programs
    print("Creating tasks from cached programs...")
    all_tasks = create_tasks_from_cached(
        cached_programs, pretraining_rules, catalogue_rules, config
    )
    print(f"Created {len(all_tasks)} tasks with real programs")

    if len(all_tasks) < 10:
        print("ERROR: Not enough tasks created. Need at least 10.")
        return None

    # Set seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Results storage
    all_results = defaultdict(list)

    # Cross-validation
    kfold = KFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
    task_names = [t.name for t in all_tasks]

    total_conditions = len(lambda_configs) * config.n_folds
    condition_idx = 0

    for fold_idx, (train_idx, test_idx) in enumerate(kfold.split(task_names)):
        print()
        print("=" * 70)
        print(f"FOLD {fold_idx + 1}/{config.n_folds}")
        print("=" * 70)

        train_tasks = [all_tasks[i] for i in train_idx]
        test_tasks = [all_tasks[i] for i in test_idx]

        print(f"Train: {len(train_tasks)} tasks, Test: {len(test_tasks)} tasks")

        for lc in lambda_configs:
            condition_idx += 1
            print(f"\n  [{condition_idx}/{total_conditions}] {lc}")

            try:
                metrics = train_single_condition(
                    lambda_config=lc,
                    train_tasks=train_tasks,
                    test_tasks=test_tasks,
                    grammar=grammar,
                    config=config,
                    fold_idx=fold_idx
                )

                # Store results
                all_results[lc.name].append(metrics)

                print(f"    R@5={metrics['R@5']:.3f}, R@10={metrics['R@10']:.3f}, "
                      f"MRR={metrics['MRR']:.3f}, loss={metrics['train_loss']:.4f}, "
                      f"time={metrics['train_time']:.1f}s")

            except Exception as e:
                print(f"    ERROR: {e}")
                import traceback
                traceback.print_exc()

    # Aggregate results
    print()
    print("=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)

    summary = []
    for lc in lambda_configs:
        if lc.name in all_results and all_results[lc.name]:
            fold_metrics = all_results[lc.name]
            r5_vals = [m['R@5'] for m in fold_metrics]
            r10_vals = [m['R@10'] for m in fold_metrics]
            mrr_vals = [m['MRR'] for m in fold_metrics]

            summary.append({
                'config': lc.name,
                'lambda_struct': lc.lambda_struct,
                'lambda_count': lc.lambda_count,
                'lambda_bigram': lc.lambda_bigram,
                'R@5_mean': np.mean(r5_vals),
                'R@5_std': np.std(r5_vals),
                'R@10_mean': np.mean(r10_vals),
                'R@10_std': np.std(r10_vals),
                'MRR_mean': np.mean(mrr_vals),
                'MRR_std': np.std(mrr_vals),
            })

    # Sort by R@5
    summary.sort(key=lambda x: -x['R@5_mean'])

    print("\nTop 10 configurations (by R@5):")
    print("-" * 80)
    print(f"{'Config':<25} {'λ_s':>5} {'λ_c':>5} {'λ_b':>5}  {'R@5':>12}  {'R@10':>12}  {'MRR':>12}")
    print("-" * 80)

    for i, s in enumerate(summary[:10]):
        print(f"{s['config']:<25} {s['lambda_struct']:>5.2f} {s['lambda_count']:>5.2f} {s['lambda_bigram']:>5.2f}  "
              f"{s['R@5_mean']:.3f}±{s['R@5_std']:.3f}  "
              f"{s['R@10_mean']:.3f}±{s['R@10_std']:.3f}  "
              f"{s['MRR_mean']:.3f}±{s['MRR_std']:.3f}")

    # Save results
    os.makedirs(config.results_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    results_path = os.path.join(config.results_dir, f"grid_search_{timestamp}.json")
    with open(results_path, 'w') as f:
        json.dump({
            'config': {
                'lambda_struct_values': config.lambda_struct_values,
                'lambda_count_values': config.lambda_count_values,
                'lambda_bigram_values': config.lambda_bigram_values,
                'n_epochs': config.n_epochs,
                'n_folds': config.n_folds,
                'n_tasks': len(all_tasks),
            },
            'summary': summary,
            'all_results': {k: v for k, v in all_results.items()},
        }, f, indent=2)

    print(f"\nResults saved to: {results_path}")

    # Print key findings
    print()
    print("=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    if summary:
        best = summary[0]
        print(f"Best configuration: λ_struct={best['lambda_struct']}, "
              f"λ_count={best['lambda_count']}, λ_bigram={best['lambda_bigram']}")
        print(f"  R@5={best['R@5_mean']:.3f}±{best['R@5_std']:.3f}")

        # Find baseline (0.3, 0.1, 0.1) if it exists
        baseline = next((s for s in summary
                        if abs(s['lambda_struct'] - 0.3) < 0.01
                        and abs(s['lambda_count'] - 0.1) < 0.01
                        and abs(s['lambda_bigram'] - 0.1) < 0.01), None)

        if baseline:
            print(f"\nBaseline (0.3, 0.1, 0.1): R@5={baseline['R@5_mean']:.3f}±{baseline['R@5_std']:.3f}")
            improvement = (best['R@5_mean'] - baseline['R@5_mean']) / baseline['R@5_mean'] * 100
            print(f"Improvement: {improvement:+.1f}%")

    end_time = datetime.now()
    print(f"\nTotal runtime: {end_time - start_time}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lambda weight grid search ablation")
    parser.add_argument("--quick", action="store_true", help="Quick validation run")
    args = parser.parse_args()

    config = AblationConfig()
    run_grid_search(config, quick=args.quick)
