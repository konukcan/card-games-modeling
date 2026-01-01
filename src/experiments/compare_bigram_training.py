#!/usr/bin/env python3
"""
Compare BigramHead Training: WITH vs WITHOUT Bigram Loss

This script follows Pipeline A methodology but uses REAL parsed programs
from cached overnight runs to extract actual bigrams from ASTs.

Key differences from previous tests:
- Uses real Program objects (not fake primitive sets)
- Extracts actual bigrams from AST structure
- Fair comparison: same training data, only difference is bigram loss

Expected runtime: ~2-5 minutes
"""

import sys
import os
import json
import glob
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import sample_hand, Card
from rules.catalogue import create_all_rules
from rules.pretraining_rules import get_all_pretraining_rules, PretrainingRule
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.program import parse_program, Program
from dreamcoder_core.contrastive_recognition import (
    ContrastiveRecognitionModel,
    extract_bigrams,
    build_bigram_vocabulary
)


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
class BigramTestConfig:
    """Configuration for bigram comparison test."""
    n_examples: int = 50          # Examples per task (25 pos, 25 neg)
    n_epochs: int = 30            # Training epochs
    batch_size: int = 8
    lr: float = 0.001
    hand_size: int = 6
    hidden_dim: int = 64
    n_folds: int = 3              # Cross-validation folds
    lambda_bigram: float = 0.1    # Bigram loss weight
    seed: int = 42
    quick_mode: bool = False      # For testing


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_cached_programs(results_dir: str = "results_archive") -> Dict[str, Tuple[str, Program]]:
    """
    Load all parseable programs from cached frontier files.

    Returns:
        Dict mapping task_name -> (program_string, Program object)
    """
    grammar = build_lean_grammar()
    primitives = {}
    for prod in grammar.productions:
        if hasattr(prod.program, 'name'):
            primitives[prod.program.name] = prod.program

    # Find all frontier files
    frontier_files = glob.glob(f'{results_dir}/**/frontiers_*.json', recursive=True)
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


def get_rule_by_id(rule_id: str, pretraining_rules: List, catalogue_rules: List):
    """Find a rule by ID from either pretraining or catalogue rules."""
    for rule in pretraining_rules:
        if rule.id == rule_id:
            return rule
    for rule in catalogue_rules:
        if rule.id == rule_id:
            return rule
    return None


# ============================================================================
# FRONTIER WRAPPER
# ============================================================================

class FrontierEntry:
    """Mimics TaskFrontier entry structure."""
    def __init__(self, program: Program, log_likelihood: float = 0.0):
        self.program = program
        self.log_likelihood = log_likelihood


class FakeFrontier:
    """Wrapper to make parsed programs compatible with train_on_frontiers."""
    def __init__(self, program: Program):
        self.solved = True
        self.entries = [FrontierEntry(program, 0.0)]


# ============================================================================
# TASK CREATION
# ============================================================================

class TaskWithProgram:
    """Task that includes actual parsed program for bigram extraction."""

    def __init__(self, rule, program: Program, program_str: str, n_examples: int = 50, hand_size: int = 6):
        self.rule = rule
        self.id = rule.id if hasattr(rule, 'id') else str(rule)
        self.name = self.id
        self.program = program
        self.program_str = program_str
        self.examples = []

        # Sample positive and negative examples
        pos_count = neg_count = 0
        max_attempts = n_examples * 50
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
        self.primitives_used = self._extract_primitives(program)

        # Extract bigrams from program
        self.bigrams = extract_bigrams(program)

    def _extract_primitives(self, program: Program) -> List[str]:
        """Extract primitive names from program AST."""
        from dreamcoder_core.program import Primitive, Application, Abstraction, Invented

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
        return list(primitives)


def create_tasks_from_cached(
    cached_programs: Dict[str, Tuple[str, Program]],
    pretraining_rules: List,
    catalogue_rules: List,
    config: BigramTestConfig
) -> List[TaskWithProgram]:
    """Create tasks from cached programs with matching rules."""
    tasks = []

    for task_name, (prog_str, program) in cached_programs.items():
        rule = get_rule_by_id(task_name, pretraining_rules, catalogue_rules)
        if rule:
            task = TaskWithProgram(
                rule=rule,
                program=program,
                program_str=prog_str,
                n_examples=config.n_examples,
                hand_size=config.hand_size
            )
            tasks.append(task)

    return tasks


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def evaluate_model(model, tasks: List[TaskWithProgram], grammar) -> Dict:
    """Evaluate recognition model on tasks."""
    model.eval()

    metrics = {
        'R@5': 0, 'R@10': 0, 'MRR': 0,
        'n_tasks': len(tasks)
    }

    if not tasks:
        return metrics

    # Get primitive list from grammar
    primitive_names = [str(p.program) for p in grammar.productions
                       if hasattr(p.program, 'name')]

    with torch.no_grad():
        for task in tasks:
            # Get predictions as dict
            predictions = model.predict_primitives_dict(task)

            # Sort by probability (descending)
            sorted_preds = sorted(predictions.items(), key=lambda x: -x[1])
            predicted_names = [name for name, _ in sorted_preds]

            # Check if any used primitive is in top-k
            used = set(task.primitives_used)

            # R@5
            top5 = set(predicted_names[:5])
            if used & top5:
                metrics['R@5'] += 1

            # R@10
            top10 = set(predicted_names[:10])
            if used & top10:
                metrics['R@10'] += 1

            # MRR
            for rank, name in enumerate(predicted_names, 1):
                if name in used:
                    metrics['MRR'] += 1.0 / rank
                    break

    # Normalize
    for key in ['R@5', 'R@10', 'MRR']:
        metrics[key] /= len(tasks)

    return metrics


def evaluate_bigram_predictions(model, tasks: List[TaskWithProgram]) -> Dict:
    """Evaluate bigram prediction quality."""
    if not hasattr(model, 'bigram_vocab') or not model.bigram_vocab:
        return {}

    model.eval()

    total_precision = 0
    total_recall = 0
    n_tasks_with_bigrams = 0

    with torch.no_grad():
        for task in tasks:
            if not task.bigrams:
                continue

            # Get predicted bigrams
            pred_bigrams = model.predict_bigrams(task)
            if not pred_bigrams:
                continue

            # Top-5 predicted
            sorted_preds = sorted(pred_bigrams.items(), key=lambda x: -x[1])
            top5_pred = set([bg for bg, _ in sorted_preds[:5]])

            # Actual bigrams (only those in vocabulary)
            actual = set(task.bigrams) & set(model.bigram_vocab)

            if actual:
                # Precision@5
                precision = len(top5_pred & actual) / min(5, len(pred_bigrams))
                # Recall@5
                recall = len(top5_pred & actual) / len(actual)

                total_precision += precision
                total_recall += recall
                n_tasks_with_bigrams += 1

    if n_tasks_with_bigrams == 0:
        return {}

    avg_precision = total_precision / n_tasks_with_bigrams
    avg_recall = total_recall / n_tasks_with_bigrams
    f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall + 1e-8)

    return {
        'bigram_P@5': avg_precision,
        'bigram_R@5': avg_recall,
        'bigram_F1': f1
    }


# ============================================================================
# MAIN COMPARISON
# ============================================================================

def run_bigram_comparison(config: BigramTestConfig):
    """Run the main comparison: WITH vs WITHOUT bigram training."""

    logger.info("=" * 70)
    logger.info("BIGRAM TRAINING COMPARISON (Pipeline A with Real Programs)")
    logger.info("=" * 70)

    # Set seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load cached programs
    cached_programs, grammar = load_all_cached_programs()

    if len(cached_programs) < 10:
        logger.error(f"Not enough parseable programs ({len(cached_programs)}). Need at least 10.")
        return None

    # Load rules
    pretraining_rules = get_all_pretraining_rules()
    catalogue_rules = create_all_rules()

    # Create tasks
    tasks = create_tasks_from_cached(
        cached_programs, pretraining_rules, catalogue_rules, config
    )

    logger.info(f"Created {len(tasks)} tasks with parsed programs")

    # Collect all bigrams
    all_bigrams = set()
    for task in tasks:
        all_bigrams.update(task.bigrams)
    logger.info(f"Total unique bigrams: {len(all_bigrams)}")

    if config.quick_mode:
        config.n_epochs = 10
        config.n_folds = 2
        tasks = tasks[:12]
        logger.info(f"Quick mode: {len(tasks)} tasks, {config.n_epochs} epochs")

    # K-fold cross-validation
    results_no_bigram = []
    results_with_bigram = []

    fold_size = len(tasks) // config.n_folds

    for fold in range(config.n_folds):
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold+1}/{config.n_folds}")
        logger.info(f"{'='*60}")

        # Split
        start_idx = fold * fold_size
        end_idx = start_idx + fold_size if fold < config.n_folds - 1 else len(tasks)

        test_tasks = tasks[start_idx:end_idx]
        train_tasks = tasks[:start_idx] + tasks[end_idx:]

        logger.info(f"Train: {len(train_tasks)}, Test: {len(test_tasks)}")

        # Create frontiers dict for training (using FakeFrontier wrapper)
        frontiers = {}
        for task in train_tasks:
            frontiers[task.name] = FakeFrontier(task.program)

        # ================================================
        # Model WITHOUT bigram training
        # ================================================
        logger.info("\nTraining WITHOUT bigram loss...")
        torch.manual_seed(config.seed + fold)

        model_no_bigram = ContrastiveRecognitionModel(
            grammar=grammar,
            card_hidden=128,
            card_out=config.hidden_dim,
            pred_hidden=128,
            normalize_embeddings=True,
            embedding_scale=20.0,
            output_mode='sigmoid'
        )

        model_no_bigram.train_on_frontiers(
            tasks=train_tasks,
            frontiers=frontiers,
            epochs=config.n_epochs,
            batch_size=config.batch_size,
            use_bigram_loss=False
        )

        eval_no = evaluate_model(model_no_bigram, test_tasks, grammar)
        results_no_bigram.append(eval_no)
        logger.info(f"  R@5: {eval_no['R@5']:.3f}, MRR: {eval_no['MRR']:.3f}")

        # ================================================
        # Model WITH bigram training
        # ================================================
        logger.info("\nTraining WITH bigram loss...")
        torch.manual_seed(config.seed + fold)

        model_with_bigram = ContrastiveRecognitionModel(
            grammar=grammar,
            card_hidden=128,
            card_out=config.hidden_dim,
            pred_hidden=128,
            normalize_embeddings=True,
            embedding_scale=20.0,
            output_mode='sigmoid'
        )

        # Build bigram vocabulary from training programs
        train_programs = [t.program for t in train_tasks]
        model_with_bigram.build_bigram_vocabulary(train_programs, min_count=1)
        logger.info(f"  Bigram vocabulary size: {len(model_with_bigram.bigram_vocab)}")

        model_with_bigram.train_on_frontiers(
            tasks=train_tasks,
            frontiers=frontiers,
            epochs=config.n_epochs,
            batch_size=config.batch_size,
            use_bigram_loss=True,
            lambda_bigram=config.lambda_bigram
        )

        eval_with = evaluate_model(model_with_bigram, test_tasks, grammar)
        bigram_eval = evaluate_bigram_predictions(model_with_bigram, test_tasks)
        eval_with.update(bigram_eval)
        results_with_bigram.append(eval_with)

        logger.info(f"  R@5: {eval_with['R@5']:.3f}, MRR: {eval_with['MRR']:.3f}")
        if 'bigram_P@5' in eval_with:
            logger.info(f"  Bigram P@5: {eval_with['bigram_P@5']:.3f}, R@5: {eval_with['bigram_R@5']:.3f}")

    # ================================================
    # Aggregate Results
    # ================================================
    def aggregate(results_list):
        agg = {}
        for key in results_list[0].keys():
            if key != 'n_tasks':
                values = [r[key] for r in results_list if key in r]
                if values:
                    agg[key] = {'mean': np.mean(values), 'std': np.std(values)}
        return agg

    agg_no = aggregate(results_no_bigram)
    agg_with = aggregate(results_with_bigram)

    # Print summary
    print("\n" + "=" * 80)
    print("BIGRAM TRAINING COMPARISON RESULTS")
    print("=" * 80)
    print(f"\nDataset: {len(tasks)} tasks with real parsed programs, {len(all_bigrams)} unique bigrams")
    print(f"Training: {config.n_epochs} epochs, {config.n_folds}-fold CV")
    print()

    print("-" * 80)
    print(f"{'Metric':<20} {'WITHOUT Bigram':<25} {'WITH Bigram':<25} {'Difference':<15}")
    print("-" * 80)

    for metric in ['R@5', 'R@10', 'MRR']:
        no_val = agg_no[metric]
        with_val = agg_with[metric]
        diff = with_val['mean'] - no_val['mean']
        pct = 100 * diff / (no_val['mean'] + 1e-8)

        print(f"{metric:<20} {no_val['mean']:.3f}±{no_val['std']:.3f}           "
              f"{with_val['mean']:.3f}±{with_val['std']:.3f}           "
              f"{diff:+.3f} ({pct:+.1f}%)")

    print()
    if 'bigram_P@5' in agg_with:
        print("-" * 80)
        print("Bigram Prediction Quality (WITH bigram training):")
        print(f"  Precision@5: {agg_with['bigram_P@5']['mean']:.3f}±{agg_with['bigram_P@5']['std']:.3f}")
        print(f"  Recall@5:    {agg_with['bigram_R@5']['mean']:.3f}±{agg_with['bigram_R@5']['std']:.3f}")
        print(f"  F1:          {agg_with['bigram_F1']['mean']:.3f}±{agg_with['bigram_F1']['std']:.3f}")

    # Conclusion
    print("\n" + "=" * 80)
    r5_diff = agg_with['R@5']['mean'] - agg_no['R@5']['mean']
    if r5_diff > 0.01:
        print(f"CONCLUSION: Bigram training HELPS R@5 by {r5_diff:.3f}")
    elif r5_diff < -0.01:
        print(f"CONCLUSION: Bigram training HURTS R@5 by {abs(r5_diff):.3f}")
    else:
        print(f"CONCLUSION: Bigram training has NO SIGNIFICANT EFFECT (diff={r5_diff:.3f})")
    print("=" * 80)

    # Save results
    results = {
        'config': {
            'n_tasks': len(tasks),
            'n_bigrams': len(all_bigrams),
            'n_epochs': config.n_epochs,
            'n_folds': config.n_folds,
            'lambda_bigram': config.lambda_bigram
        },
        'without_bigram': agg_no,
        'with_bigram': agg_with,
        'timestamp': datetime.now().isoformat()
    }

    results_dir = Path('results_bigram_comparison')
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"bigram_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with open(results_file, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            return obj
        json.dump(convert(results), f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare bigram training effect")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--folds", type=int, default=3, help="Cross-validation folds")
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    parser.add_argument("--lambda-bigram", type=float, default=0.1, help="Bigram loss weight")
    args = parser.parse_args()

    config = BigramTestConfig(
        n_epochs=args.epochs,
        n_folds=args.folds,
        quick_mode=args.quick,
        lambda_bigram=args.lambda_bigram
    )

    run_bigram_comparison(config)


if __name__ == "__main__":
    main()
