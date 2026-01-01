#!/usr/bin/env python3
"""
Lambda Weight Ablation Experiment

Tests the effect of different loss weight combinations on recognition model performance.
Runs independently - safe to run in parallel with other experiments.

Conditions tested:
1. baseline:    λ_pred=1.0, λ_struct=0.3, λ_count=0.1, λ_bigram=0.1 (current defaults)
2. no_struct:   λ_pred=1.0, λ_struct=0.0, λ_count=0.1, λ_bigram=0.1
3. high_struct: λ_pred=1.0, λ_struct=1.0, λ_count=0.1, λ_bigram=0.1
4. no_count:    λ_pred=1.0, λ_struct=0.3, λ_count=0.0, λ_bigram=0.1
5. no_bigram:   λ_pred=1.0, λ_struct=0.3, λ_count=0.1, λ_bigram=0.0
6. pred_only:   λ_pred=1.0, λ_struct=0.0, λ_count=0.0, λ_bigram=0.0

Pipeline A methodology: Train on pretraining rules, test on catalogue rules.
3-fold cross-validation, 30 epochs per fold.

Expected runtime: ~1 hour

Usage:
    python3 experiments/ablate_lambda_weights.py
    python3 experiments/ablate_lambda_weights.py --quick  # Fast validation run
"""

import sys
from pathlib import Path
import json
import random
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional
import argparse

# Add src to path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold

from dreamcoder_core.type_system import arrow, HAND, BOOL
from dreamcoder_core.grammar import uniform_grammar
from dreamcoder_core.lean_primitives import build_lean_primitives
from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from rules.pretraining_rules import get_all_pretraining_rules
from rules.catalogue import ALL_RULES as CATALOGUE_RULES


@dataclass
class LambdaConfig:
    """Configuration for a single ablation condition."""
    name: str
    lambda_pred: float
    lambda_struct: float
    lambda_count: float
    lambda_bigram: float
    use_bigram_loss: bool = True


# Define ablation conditions
ABLATION_CONDITIONS = [
    LambdaConfig("baseline", 1.0, 0.3, 0.1, 0.1, use_bigram_loss=True),
    LambdaConfig("no_struct", 1.0, 0.0, 0.1, 0.1, use_bigram_loss=True),
    LambdaConfig("high_struct", 1.0, 1.0, 0.1, 0.1, use_bigram_loss=True),
    LambdaConfig("no_count", 1.0, 0.3, 0.0, 0.1, use_bigram_loss=True),
    LambdaConfig("no_bigram", 1.0, 0.3, 0.1, 0.0, use_bigram_loss=False),
    LambdaConfig("pred_only", 1.0, 0.0, 0.0, 0.0, use_bigram_loss=False),
]


class FrontierEntry:
    """Mimics TaskFrontier entry structure."""
    def __init__(self, program, log_likelihood: float = 0.0):
        self.program = program
        self.log_likelihood = log_likelihood


class FakeFrontier:
    """Wrapper to make parsed programs compatible with train_on_frontiers."""
    def __init__(self, program):
        self.solved = True
        self.entries = [FrontierEntry(program, 0.0)]


def create_task_from_rule(rule, grammar, hand_size: int = 6):
    """Create a Task object from a rule definition (Rule or PretrainingRule class)."""
    from rules.cards import Card, Suit, Rank

    class Task:
        def __init__(self, name, examples, request):
            self.name = name
            self.examples = examples
            self.request = request

    # Generate a full deck
    deck = [Card(suit, rank) for suit in Suit for rank in Rank]
    examples = []

    # Get rule name and eval function (handles both Rule and PretrainingRule)
    rule_name = rule.name
    if hasattr(rule, 'predicate'):
        rule_fn = rule.predicate  # Rule class
    elif hasattr(rule, 'eval') and callable(rule.eval):
        rule_fn = rule.eval  # PretrainingRule class
    else:
        rule_fn = lambda h: True  # Fallback

    # Generate examples
    n_positive = 0
    n_negative = 0
    max_per_class = 20

    random.seed(hash(rule_name) % (2**32))

    for _ in range(200):
        if n_positive >= max_per_class and n_negative >= max_per_class:
            break
        hand = random.sample(deck, hand_size)
        try:
            result = rule_fn(hand)
            if result and n_positive < max_per_class:
                examples.append((hand, True))
                n_positive += 1
            elif not result and n_negative < max_per_class:
                examples.append((hand, False))
                n_negative += 1
        except:
            continue

    return Task(rule_name, examples, arrow(HAND, BOOL))


def evaluate_model(model, test_tasks: List, k_values: List[int] = [5, 10]) -> Dict:
    """Evaluate model on test tasks."""
    model.eval()

    results = {f'R@{k}': [] for k in k_values}
    results['MRR'] = []

    with torch.no_grad():
        for task in test_tasks:
            if not hasattr(task, '_target_primitives'):
                continue

            target_prims = task._target_primitives
            if not target_prims:
                continue

            # Get predictions
            pred_dict = model.predict_primitives_dict(task)

            # Sort by probability
            sorted_prims = sorted(pred_dict.items(), key=lambda x: -x[1])
            ranked_prims = [p[0] for p in sorted_prims]

            # Compute R@k
            for k in k_values:
                top_k = set(ranked_prims[:k])
                recall = len(top_k & target_prims) / len(target_prims)
                results[f'R@{k}'].append(recall)

            # Compute MRR
            for i, prim in enumerate(ranked_prims):
                if prim in target_prims:
                    results['MRR'].append(1.0 / (i + 1))
                    break
            else:
                results['MRR'].append(0.0)

    # Average
    return {k: np.mean(v) if v else 0.0 for k, v in results.items()}


def run_single_condition(
    config: LambdaConfig,
    train_rules: List[Dict],
    test_rules: List[Dict],
    grammar,
    primitives,
    epochs: int = 30,
    hidden_dim: int = 64
) -> Dict:
    """Run a single ablation condition."""

    # Create tasks
    train_tasks = [create_task_from_rule(r, grammar) for r in train_rules]
    test_tasks = [create_task_from_rule(r, grammar) for r in test_rules]

    # Build bigram vocabulary from train programs (if using bigram loss)
    bigram_vocab = None
    if config.use_bigram_loss:
        bigram_vocab = []  # Would need parsed programs; skip for now

    # Create model
    model = ContrastiveRecognitionModel(
        grammar=grammar,
        pred_hidden=hidden_dim,
        output_mode='sigmoid'
    )

    # Create fake frontiers for training
    frontiers = {}
    for task, rule in zip(train_tasks, train_rules):
        # Extract primitives from rule (simplified - use rule name patterns)
        target_prims = set()
        rule_name = rule.name.lower()

        # Heuristic primitive assignment based on rule patterns
        if 'uniform' in rule_name or 'same' in rule_name:
            target_prims.update(['all', 'eq'])
        if 'suit' in rule_name or 'color' in rule_name:
            target_prims.add('get_suit')
        if 'rank' in rule_name:
            target_prims.add('get_rank')
        if 'pair' in rule_name or 'unique' in rule_name:
            target_prims.update(['n_unique_ranks', 'eq', 'le'])
        if 'sorted' in rule_name or 'ascending' in rule_name:
            target_prims.add('is_sorted')
        if 'hearts' in rule_name:
            target_prims.add('HEARTS')
        if 'spades' in rule_name:
            target_prims.add('SPADES')
        if 'filter' in rule_name or 'contains' in rule_name:
            target_prims.add('filter')
        if 'any' in rule_name or 'least' in rule_name:
            target_prims.add('any')
        if 'all' in rule_name or 'every' in rule_name:
            target_prims.add('all')
        if 'count' in rule_name or 'exactly' in rule_name:
            target_prims.update(['length', 'eq'])
        if 'greater' in rule_name or 'more' in rule_name:
            target_prims.add('gt')
        if 'less' in rule_name or 'fewer' in rule_name:
            target_prims.add('lt')

        # Ensure we have some primitives
        if not target_prims:
            target_prims = {'filter', 'all', 'any'}

        # Filter to only primitives in vocabulary
        target_prims = {p for p in target_prims if p in model.primitive_to_idx}

        task._target_primitives = target_prims

        # Create minimal frontier
        class MinimalProgram:
            def __init__(self):
                self.body = None

        frontiers[task.name] = FakeFrontier(MinimalProgram())

    # Also set target primitives on test tasks
    for task, rule in zip(test_tasks, test_rules):
        target_prims = set()
        rule_name = rule.name.lower()

        if 'uniform' in rule_name or 'same' in rule_name:
            target_prims.update(['all', 'eq'])
        if 'suit' in rule_name or 'color' in rule_name:
            target_prims.add('get_suit')
        if 'rank' in rule_name:
            target_prims.add('get_rank')
        if 'pair' in rule_name or 'unique' in rule_name:
            target_prims.update(['n_unique_ranks', 'eq', 'le'])
        if 'sorted' in rule_name or 'ascending' in rule_name:
            target_prims.add('is_sorted')
        if 'hearts' in rule_name:
            target_prims.add('HEARTS')
        if 'spades' in rule_name:
            target_prims.add('SPADES')
        if 'filter' in rule_name or 'contains' in rule_name:
            target_prims.add('filter')
        if 'any' in rule_name or 'least' in rule_name:
            target_prims.add('any')
        if 'all' in rule_name or 'every' in rule_name:
            target_prims.add('all')
        if 'count' in rule_name or 'exactly' in rule_name:
            target_prims.update(['length', 'eq'])
        if 'greater' in rule_name or 'more' in rule_name:
            target_prims.add('gt')
        if 'less' in rule_name or 'fewer' in rule_name:
            target_prims.add('lt')

        if not target_prims:
            target_prims = {'filter', 'all', 'any'}

        target_prims = {p for p in target_prims if p in model.primitive_to_idx}
        task._target_primitives = target_prims

    # Train
    model.train_on_frontiers(
        tasks=train_tasks,
        frontiers=frontiers,
        epochs=epochs,
        batch_size=8,
        lambda_struct=config.lambda_struct,
        lambda_count=config.lambda_count,
        lambda_pred=config.lambda_pred,
        lambda_bigram=config.lambda_bigram,
        use_bigram_loss=False  # Simplified - no parsed programs available
    )

    # Evaluate
    metrics = evaluate_model(model, test_tasks)

    return metrics


def run_ablation_experiment(
    n_folds: int = 3,
    epochs: int = 30,
    quick: bool = False
):
    """Run the full ablation experiment."""

    print("=" * 70)
    print("LAMBDA WEIGHT ABLATION EXPERIMENT")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Quick mode for validation
    if quick:
        n_folds = 2
        epochs = 5
        print("QUICK MODE: 2 folds, 5 epochs")

    # Setup
    primitives = build_lean_primitives()
    grammar = uniform_grammar(primitives)

    # Get rules
    pretraining_rules = get_all_pretraining_rules()
    catalogue_rules = CATALOGUE_RULES

    # Use pretraining for train, catalogue for test (Pipeline A)
    all_train_rules = pretraining_rules
    all_test_rules = catalogue_rules[:15]  # Subset for speed

    print(f"Training rules: {len(all_train_rules)}")
    print(f"Test rules: {len(all_test_rules)}")
    print(f"Folds: {n_folds}")
    print(f"Epochs per fold: {epochs}")
    print(f"Conditions: {len(ABLATION_CONDITIONS)}")
    print()

    # Results storage
    results = {config.name: {'R@5': [], 'R@10': [], 'MRR': []} for config in ABLATION_CONDITIONS}

    # K-fold on training rules
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(all_train_rules)):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'='*70}")

        train_rules = [all_train_rules[i] for i in train_idx]
        # Use catalogue rules as test (not validation fold)
        test_rules = all_test_rules

        for config in ABLATION_CONDITIONS:
            print(f"\n  Condition: {config.name}")
            print(f"    λ_pred={config.lambda_pred}, λ_struct={config.lambda_struct}, "
                  f"λ_count={config.lambda_count}, λ_bigram={config.lambda_bigram}")

            metrics = run_single_condition(
                config=config,
                train_rules=train_rules,
                test_rules=test_rules,
                grammar=grammar,
                primitives=primitives,
                epochs=epochs
            )

            print(f"    R@5={metrics['R@5']:.3f}, R@10={metrics['R@10']:.3f}, MRR={metrics['MRR']:.3f}")

            results[config.name]['R@5'].append(metrics['R@5'])
            results[config.name]['R@10'].append(metrics['R@10'])
            results[config.name]['MRR'].append(metrics['MRR'])

    # Aggregate results
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    summary = {}
    for config in ABLATION_CONDITIONS:
        name = config.name
        summary[name] = {
            'R@5': {'mean': np.mean(results[name]['R@5']), 'std': np.std(results[name]['R@5'])},
            'R@10': {'mean': np.mean(results[name]['R@10']), 'std': np.std(results[name]['R@10'])},
            'MRR': {'mean': np.mean(results[name]['MRR']), 'std': np.std(results[name]['MRR'])},
            'config': {
                'lambda_pred': config.lambda_pred,
                'lambda_struct': config.lambda_struct,
                'lambda_count': config.lambda_count,
                'lambda_bigram': config.lambda_bigram
            }
        }

        print(f"\n{name}:")
        print(f"  R@5:  {summary[name]['R@5']['mean']:.3f} ± {summary[name]['R@5']['std']:.3f}")
        print(f"  R@10: {summary[name]['R@10']['mean']:.3f} ± {summary[name]['R@10']['std']:.3f}")
        print(f"  MRR:  {summary[name]['MRR']['mean']:.3f} ± {summary[name]['MRR']['std']:.3f}")

    # Comparison table
    print("\n" + "-" * 70)
    print("COMPARISON TABLE (R@5)")
    print("-" * 70)
    print(f"{'Condition':<15} {'R@5':<12} {'vs baseline':<12} {'Interpretation'}")
    print("-" * 70)

    baseline_r5 = summary['baseline']['R@5']['mean']
    for config in ABLATION_CONDITIONS:
        name = config.name
        r5 = summary[name]['R@5']['mean']
        diff = r5 - baseline_r5
        diff_pct = (diff / baseline_r5 * 100) if baseline_r5 > 0 else 0

        if name == 'baseline':
            interp = "(reference)"
        elif diff > 0.02:
            interp = "HELPS"
        elif diff < -0.02:
            interp = "HURTS"
        else:
            interp = "neutral"

        print(f"{name:<15} {r5:.3f}        {diff_pct:+.1f}%         {interp}")

    # Save results
    output_dir = Path("results_lambda_ablation")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"ablation_results_{timestamp}.json"

    with open(output_file, 'w') as f:
        json.dump({
            'summary': summary,
            'raw_results': results,
            'config': {
                'n_folds': n_folds,
                'epochs': epochs,
                'n_train_rules': len(all_train_rules),
                'n_test_rules': len(all_test_rules),
                'quick_mode': quick
            },
            'timestamp': timestamp
        }, f, indent=2)

    print(f"\n\nResults saved to: {output_file}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Key findings
    print("\n" + "=" * 70)
    print("KEY FINDINGS")
    print("=" * 70)

    # Find best and worst
    r5_scores = [(name, summary[name]['R@5']['mean']) for name in summary]
    r5_scores.sort(key=lambda x: -x[1])

    print(f"Best:  {r5_scores[0][0]} (R@5={r5_scores[0][1]:.3f})")
    print(f"Worst: {r5_scores[-1][0]} (R@5={r5_scores[-1][1]:.3f})")

    # Specific findings
    struct_diff = summary['no_struct']['R@5']['mean'] - baseline_r5
    if abs(struct_diff) > 0.02:
        print(f"\nStructural loss effect: {struct_diff:+.3f} R@5 when removed")
        if struct_diff < 0:
            print("  → Structural loss HELPS (removing it hurts)")
        else:
            print("  → Structural loss may be HURTING (removing it helps)")
    else:
        print(f"\nStructural loss effect: negligible ({struct_diff:+.3f})")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lambda weight ablation experiment")
    parser.add_argument("--quick", action="store_true", help="Quick validation run")
    args = parser.parse_args()

    run_ablation_experiment(quick=args.quick)
