#!/usr/bin/env python3
"""
Comparison of Recognition Model Variants WITH New Head Components

This script re-runs the variant comparisons with the newly integrated components:
1. BigramHead training (enabled)
2. CountHead evaluation (metrics logged)
3. ContextualGrammarNetwork (when available)

Purpose:
- Determine if BigramHead training changes variant rankings
- Evaluate if CountHead predictions are meaningful
- Prepare for ContextualGrammarNetwork integration

Usage:
    python experiments/compare_with_new_heads.py [--quick]

Author: Can Konuk
Date: December 2024
"""

import sys
import json
import random
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass
import traceback

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.contrastive_recognition import (
    ContrastiveRecognitionModel,
    extract_bigrams,
    build_bigram_vocabulary
)
from dreamcoder_core.lean_primitives import build_lean_grammar
from rules.catalogue import create_all_rules, Rule
from rules.pretraining_rules import get_all_pretraining_rules, PretrainingRule
from rules.cards import sample_hand

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ============================================================================
# Task Creation
# ============================================================================

@dataclass
class RevelationTask:
    """A task for rule revelation training."""
    name: str
    examples: List[Tuple[Any, bool]]
    primitives_used: Set[str]
    source: str


def create_task_from_catalogue_rule(rule: Rule, n_examples: int = 50, seed: int = 42) -> RevelationTask:
    """Create a revelation task from a catalogue rule."""
    rng = random.Random(seed + hash(rule.id) % 10000)

    examples = []
    pos_count = 0
    neg_count = 0
    attempts = 0

    while (pos_count < n_examples // 2 or neg_count < n_examples // 2) and attempts < n_examples * 20:
        hand = sample_hand(size=6)
        try:
            label = rule.predicate(hand)
            if label and pos_count < n_examples // 2:
                examples.append((hand, True))
                pos_count += 1
            elif not label and neg_count < n_examples // 2:
                examples.append((hand, False))
                neg_count += 1
        except Exception:
            pass
        attempts += 1

    return RevelationTask(
        name=rule.id,
        examples=examples,
        primitives_used=set(rule.primitives) if hasattr(rule, 'primitives') else set(),
        source='catalogue'
    )


def extract_primitives_from_expected_program(expected_program: str) -> Set[str]:
    """Extract primitive names from expected_program string."""
    # Common primitives in the grammar
    known_primitives = {
        'lt', 'le', 'gt', 'ge', 'eq', 'ne',
        'add', 'sub', 'mul', 'div', 'mod',
        'and_', 'or_', 'not_', 'if_',
        'length', 'unique', 'map', 'filter', 'any', 'all', 'count',
        'get_rank', 'get_suit', 'get_color', 'rank_val',
        'is_face', 'is_ace', 'is_red', 'is_black',
        'sorted_by_rank', 'consecutive',
        'fold', 'head', 'tail', 'cons', 'nil', 'empty', 'append',
        'max', 'min', 'sum', 'abs', 'even', 'odd',
        'hearts', 'diamonds', 'clubs', 'spades',
        'hand', 'ranks', 'suits', 'colors'
    }

    primitives = set()
    # Simple extraction: find known primitives in the expected_program string
    for prim in known_primitives:
        if prim in expected_program:
            primitives.add(prim)

    return primitives


def create_task_from_pretraining_rule(rule: PretrainingRule, n_examples: int = 50, seed: int = 42) -> RevelationTask:
    """Create a revelation task from a pretraining rule."""
    rng = random.Random(seed + hash(rule.name) % 10000)

    examples = []
    pos_count = 0
    neg_count = 0
    attempts = 0

    while (pos_count < n_examples // 2 or neg_count < n_examples // 2) and attempts < n_examples * 20:
        hand = sample_hand(size=6)
        try:
            label = rule.eval(hand)  # Use 'eval' not 'predicate'
            if label and pos_count < n_examples // 2:
                examples.append((hand, True))
                pos_count += 1
            elif not label and neg_count < n_examples // 2:
                examples.append((hand, False))
                neg_count += 1
        except Exception:
            pass
        attempts += 1

    # Extract primitives from expected_program
    primitives = extract_primitives_from_expected_program(rule.expected_program)

    return RevelationTask(
        name=rule.name,
        examples=examples,
        primitives_used=primitives,
        source='pretraining'
    )


# ============================================================================
# Fake Frontier for Training
# ============================================================================

@dataclass
class FakeFrontierEntry:
    program: Any
    log_likelihood: float


@dataclass
class FakeFrontier:
    entries: List[FakeFrontierEntry]
    solved: bool = True


def create_fake_program_with_primitives(primitives: Set[str]):
    """Create a fake program object that stores primitive names."""
    class FakeProgram:
        def __init__(self, prims):
            self.primitives = prims
    return FakeProgram(primitives)


# ============================================================================
# Evaluation Functions
# ============================================================================

def evaluate_model_on_tasks(
    model: ContrastiveRecognitionModel,
    tasks: List[RevelationTask],
    k_values: List[int] = [5, 10]
) -> Dict[str, float]:
    """
    Evaluate model with recall@k, MRR, and probability metrics.
    """
    model.eval()

    all_recalls = {k: [] for k in k_values}
    all_mrrs = []
    prob_ratios = []

    with torch.no_grad():
        for task in tasks:
            if not task.primitives_used:
                continue

            # Get predictions
            probs = model.predict_primitives(task)

            # Get indices of solution primitives
            solution_indices = []
            for prim_name in task.primitives_used:
                if prim_name in model.primitive_to_idx:
                    solution_indices.append(model.primitive_to_idx[prim_name])

            if not solution_indices:
                continue

            # Sort by probability
            sorted_indices = torch.argsort(probs, descending=True).cpu().numpy()
            sorted_probs = probs[sorted_indices].cpu().numpy()

            # Recall@k
            for k in k_values:
                top_k = set(sorted_indices[:k])
                recall = len(top_k.intersection(solution_indices)) / len(solution_indices)
                all_recalls[k].append(recall)

            # MRR
            ranks = []
            for idx in solution_indices:
                rank = np.where(sorted_indices == idx)[0]
                if len(rank) > 0:
                    ranks.append(1.0 / (rank[0] + 1))
            if ranks:
                all_mrrs.append(np.mean(ranks))

            # Probability ratio
            sol_probs = probs[solution_indices].cpu().numpy()
            non_sol_mask = np.ones(len(probs), dtype=bool)
            non_sol_mask[solution_indices] = False
            non_sol_probs = probs[non_sol_mask].cpu().numpy()

            if len(non_sol_probs) > 0 and np.mean(non_sol_probs) > 0:
                prob_ratios.append(np.mean(sol_probs) / np.mean(non_sol_probs))

    results = {}
    for k in k_values:
        results[f'recall@{k}'] = np.mean(all_recalls[k]) if all_recalls[k] else 0.0
    results['mrr'] = np.mean(all_mrrs) if all_mrrs else 0.0
    results['prob_ratio'] = np.mean(prob_ratios) if prob_ratios else 0.0

    return results


# ============================================================================
# Variant Configurations
# ============================================================================

VARIANT_CONFIGS = {
    'baseline_sigmoid': {
        'output_mode': 'sigmoid',
        'use_bigram_loss': False,
        'description': 'Baseline with sigmoid output, no bigram'
    },
    'baseline_sigmoid_bigram': {
        'output_mode': 'sigmoid',
        'use_bigram_loss': True,
        'description': 'Baseline with sigmoid output, WITH bigram training'
    },
    'baseline_softmax': {
        'output_mode': 'softmax',
        'use_bigram_loss': False,
        'description': 'Baseline with softmax output, no bigram'
    },
    'baseline_softmax_bigram': {
        'output_mode': 'softmax',
        'use_bigram_loss': True,
        'description': 'Baseline with softmax output, WITH bigram training'
    },
    'layernorm_sigmoid': {
        'output_mode': 'sigmoid',
        'normalize_embeddings': True,
        'use_bigram_loss': False,
        'description': 'LayerNorm+Scale with sigmoid, no bigram'
    },
    'layernorm_sigmoid_bigram': {
        'output_mode': 'sigmoid',
        'normalize_embeddings': True,
        'use_bigram_loss': True,
        'description': 'LayerNorm+Scale with sigmoid, WITH bigram'
    },
    'layernorm_softmax': {
        'output_mode': 'softmax',
        'normalize_embeddings': True,
        'use_bigram_loss': False,
        'description': 'LayerNorm+Scale with softmax, no bigram'
    },
    'layernorm_softmax_bigram': {
        'output_mode': 'softmax',
        'normalize_embeddings': True,
        'use_bigram_loss': True,
        'description': 'LayerNorm+Scale with softmax, WITH bigram'
    }
}


# ============================================================================
# Main Experiment
# ============================================================================

def run_single_variant(
    config_name: str,
    config: Dict,
    train_tasks: List[RevelationTask],
    test_tasks: List[RevelationTask],
    grammar,
    epochs: int = 20,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Run a single variant configuration and return results.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    try:
        # Create model
        model = ContrastiveRecognitionModel(
            grammar=grammar,
            card_hidden=128,
            card_out=64,
            pred_hidden=128,
            normalize_embeddings=config.get('normalize_embeddings', False),
            embedding_scale=20.0,
            output_mode=config.get('output_mode', 'sigmoid')
        )

        # Build bigram vocabulary if needed
        use_bigram = config.get('use_bigram_loss', False)
        if use_bigram:
            # Build vocabulary from training task primitives
            fake_programs = []
            for task in train_tasks:
                if task.primitives_used:
                    fake_prog = create_fake_program_with_primitives(task.primitives_used)
                    fake_programs.append(fake_prog)

            if fake_programs:
                model.build_bigram_vocabulary(fake_programs, min_count=1)

        # Create fake frontiers for training
        frontiers = {}
        for task in train_tasks:
            if task.primitives_used:
                fake_prog = create_fake_program_with_primitives(task.primitives_used)
                frontiers[task.name] = FakeFrontier(
                    entries=[FakeFrontierEntry(fake_prog, 0.0)],
                    solved=True
                )

        # Train
        start_time = time.time()
        final_loss = model.train_on_frontiers(
            tasks=train_tasks,
            frontiers=frontiers,
            epochs=epochs,
            batch_size=8,
            use_bigram_loss=use_bigram,
            lambda_bigram=0.1 if use_bigram else 0.0
        )
        train_time = time.time() - start_time

        # Evaluate on test set
        test_metrics = evaluate_model_on_tasks(model, test_tasks)

        # Evaluate CountHead
        count_metrics = model.evaluate_count_head(train_tasks, frontiers)

        # Evaluate BigramHead if enabled
        bigram_metrics = {}
        if use_bigram and model.bigram_vocab:
            bigram_metrics = model.evaluate_bigram_head(train_tasks, frontiers, k=5)

        return {
            'config_name': config_name,
            'config': config,
            'final_loss': final_loss,
            'train_time': train_time,
            **{f'test_{k}': v for k, v in test_metrics.items()},
            **{f'count_{k}': v for k, v in count_metrics.items()},
            **{f'bigram_{k}': v for k, v in bigram_metrics.items()},
            'success': True
        }

    except Exception as e:
        logger.error(f"Error in {config_name}: {e}")
        traceback.print_exc()
        return {
            'config_name': config_name,
            'config': config,
            'error': str(e),
            'success': False
        }


def run_comparison(
    tasks: List[RevelationTask],
    n_folds: int = 3,
    epochs: int = 20,
    output_dir: str = 'results_new_heads_comparison'
) -> Dict[str, Any]:
    """
    Run full comparison across all variants with cross-validation.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    grammar = build_lean_grammar()
    all_results = []

    # Cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    task_indices = list(range(len(tasks)))

    for config_name, config in VARIANT_CONFIGS.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {config_name}")
        logger.info(f"Description: {config.get('description', 'N/A')}")
        logger.info(f"{'='*60}")

        fold_results = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(task_indices)):
            train_tasks = [tasks[i] for i in train_idx]
            test_tasks = [tasks[i] for i in test_idx]

            logger.info(f"  Fold {fold + 1}/{n_folds}: {len(train_tasks)} train, {len(test_tasks)} test")

            result = run_single_variant(
                config_name=config_name,
                config=config,
                train_tasks=train_tasks,
                test_tasks=test_tasks,
                grammar=grammar,
                epochs=epochs,
                seed=42 + fold
            )

            fold_results.append(result)

            if result['success']:
                logger.info(f"    R@5: {result.get('test_recall@5', 0):.3f}, "
                           f"MRR: {result.get('test_mrr', 0):.3f}, "
                           f"Count MAE: {result.get('count_mae', 'N/A')}")

        # Aggregate fold results
        successful_folds = [r for r in fold_results if r['success']]
        if successful_folds:
            aggregated = {
                'config_name': config_name,
                'config': config,
                'n_folds': len(successful_folds)
            }

            # Aggregate numeric metrics
            numeric_keys = ['test_recall@5', 'test_recall@10', 'test_mrr',
                           'test_prob_ratio', 'count_mae', 'count_correlation',
                           'bigram_precision@k', 'bigram_recall@k', 'final_loss']

            for key in numeric_keys:
                values = [r.get(key) for r in successful_folds if key in r and r.get(key) is not None]
                if values:
                    aggregated[f'{key}_mean'] = np.mean(values)
                    aggregated[f'{key}_std'] = np.std(values)

            all_results.append(aggregated)

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_path / f'comparison_{timestamp}.json'

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {results_file}")

    # Print summary
    print_summary(all_results)

    return {'results': all_results, 'output_file': str(results_file)}


def print_summary(results: List[Dict]):
    """Print comparison summary table."""
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY (with new heads)")
    print("=" * 100)

    # Sort by R@5
    sorted_results = sorted(results, key=lambda x: -x.get('test_recall@5_mean', 0))

    header = f"{'Variant':<30} {'R@5':<12} {'MRR':<12} {'Count MAE':<12} {'Bigram P@5':<12}"
    print(header)
    print("-" * 100)

    for r in sorted_results:
        name = r['config_name'][:28]
        r5 = f"{r.get('test_recall@5_mean', 0):.3f}±{r.get('test_recall@5_std', 0):.2f}"
        mrr = f"{r.get('test_mrr_mean', 0):.3f}±{r.get('test_mrr_std', 0):.2f}"
        count_mae = f"{r.get('count_mae_mean', float('inf')):.2f}" if 'count_mae_mean' in r else 'N/A'
        bigram_p = f"{r.get('bigram_precision@k_mean', 0):.3f}" if 'bigram_precision@k_mean' in r else 'N/A'

        print(f"{name:<30} {r5:<12} {mrr:<12} {count_mae:<12} {bigram_p:<12}")

    # Check if rankings changed with bigram
    print("\n" + "-" * 100)
    print("BIGRAM IMPACT ANALYSIS")
    print("-" * 100)

    baseline_pairs = [
        ('baseline_sigmoid', 'baseline_sigmoid_bigram'),
        ('baseline_softmax', 'baseline_softmax_bigram'),
        ('layernorm_sigmoid', 'layernorm_sigmoid_bigram'),
        ('layernorm_softmax', 'layernorm_softmax_bigram')
    ]

    for base, with_bigram in baseline_pairs:
        base_result = next((r for r in results if r['config_name'] == base), None)
        bigram_result = next((r for r in results if r['config_name'] == with_bigram), None)

        if base_result and bigram_result:
            base_r5 = base_result.get('test_recall@5_mean', 0)
            bigram_r5 = bigram_result.get('test_recall@5_mean', 0)
            diff = bigram_r5 - base_r5
            pct = (diff / base_r5 * 100) if base_r5 > 0 else 0

            impact = "IMPROVED" if diff > 0.01 else ("DEGRADED" if diff < -0.01 else "NO CHANGE")
            print(f"{base} → {with_bigram}: {impact} ({diff:+.3f}, {pct:+.1f}%)")


# ============================================================================
# Entry Point
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Compare recognition model variants with new heads')
    parser.add_argument('--quick', action='store_true', help='Quick test with fewer tasks')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--folds', type=int, default=3, help='Cross-validation folds')
    args = parser.parse_args()

    # Load tasks
    logger.info("Loading tasks...")

    # Use pretraining rules for consistency
    pretraining_rules = get_all_pretraining_rules()  # Returns a list
    tasks = []

    n_rules = 15 if args.quick else len(pretraining_rules)
    for rule in pretraining_rules[:n_rules]:
        task = create_task_from_pretraining_rule(rule)
        if task.examples and task.primitives_used:
            tasks.append(task)

    logger.info(f"Created {len(tasks)} tasks")

    # Run comparison
    results = run_comparison(
        tasks=tasks,
        n_folds=args.folds,
        epochs=args.epochs if not args.quick else 5,
        output_dir='results_new_heads_comparison'
    )

    return results


if __name__ == '__main__':
    main()
