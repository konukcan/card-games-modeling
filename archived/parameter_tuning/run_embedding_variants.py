#!/usr/bin/env python3
"""
Re-run the embedding head variants that failed due to dimension mismatch.
"""

import sys
from pathlib import Path
import json
import logging
from datetime import datetime
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.recognition_variants import RecognitionModelVariant
from dreamcoder_core.lean_primitives import build_lean_grammar
from rules.catalogue import create_all_rules
from rules.pretraining_rules import get_all_pretraining_rules as create_pretraining_rules
from rules.cards import sample_hand

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

EMBEDDING_VARIANTS = [
    {'card_encoder': 'standard', 'hand_encoder': 'mean', 'task_encoder': 'standard',
     'prediction_head': 'embedding', 'loss': 'bce', 'name': 'embedding_head'},
    {'card_encoder': 'enhanced', 'hand_encoder': 'attention', 'task_encoder': 'standard',
     'prediction_head': 'embedding', 'loss': 'bce', 'name': 'enhanced+attention+embed'},
    {'card_encoder': 'enhanced', 'hand_encoder': 'attention', 'task_encoder': 'standard',
     'prediction_head': 'embedding', 'loss': 'focal', 'name': 'enhanced+attention+embed+focal'},
    {'card_encoder': 'enhanced', 'hand_encoder': 'deepsets', 'task_encoder': 'standard',
     'prediction_head': 'embedding', 'loss': 'bce', 'name': 'enhanced+deepsets+embed'},
    {'card_encoder': 'enhanced', 'hand_encoder': 'multiscale', 'task_encoder': 'multihead',
     'prediction_head': 'embedding', 'loss': 'focal', 'name': 'full_enhanced'},
    {'card_encoder': 'standard', 'hand_encoder': 'attention', 'task_encoder': 'multihead',
     'prediction_head': 'embedding', 'loss': 'focal', 'name': 'attention+multi+embed+focal'},
]


class Task:
    """Task wrapper for rules."""
    # TODO: Consider consolidating Task classes across experiment files - Dec 2024
    def __init__(self, rule, n_examples=50, hand_size=6):
        self.rule = rule
        self.id = rule.id
        self.examples = []

        pos_count = 0
        neg_count = 0
        max_attempts = 5000
        attempts = 0

        while (pos_count < n_examples or neg_count < n_examples) and attempts < max_attempts:
            hand = sample_hand(hand_size)
            try:
                label = rule.predicate(hand)
                if label and pos_count < n_examples:
                    self.examples.append((hand, True))
                    pos_count += 1
                elif not label and neg_count < n_examples:
                    self.examples.append((hand, False))
                    neg_count += 1
            except:
                pass
            attempts += 1


def evaluate_model(model, test_tasks, k_values=[5, 10, 20]):
    """Evaluate model on test tasks."""
    model.eval()
    metrics = defaultdict(list)

    # Build primitive name to index mapping
    primitives = model.grammar.primitives()
    prim_to_idx = {p.name: i for i, p in enumerate(primitives)}

    with torch.no_grad():
        for task in test_tasks:
            preds = model.predict_primitives(task)

            # Get ground truth primitives using primitives_used attribute
            gt_indices = set()
            if hasattr(task.rule, 'primitives_used') and task.rule.primitives_used:
                for pname in task.rule.primitives_used:
                    if pname in prim_to_idx:
                        gt_indices.add(prim_to_idx[pname])

            if not gt_indices:
                continue

            # Calculate metrics
            pred_indices = torch.argsort(preds, descending=True)

            for k in k_values:
                top_k = set(pred_indices[:k].tolist())
                hits = len(top_k & gt_indices)
                recall = hits / len(gt_indices) if gt_indices else 0
                metrics[f'recall@{k}'].append(recall)

            # MRR
            ranks = []
            for p in gt_indices:
                rank = (pred_indices == p).nonzero(as_tuple=True)[0]
                if len(rank) > 0:
                    ranks.append(1.0 / (rank[0].item() + 1))
            if ranks:
                metrics['mrr'].append(np.mean(ranks))

    return {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}


def extract_primitives_from_program(program):
    """Extract primitive indices from program string."""
    from dreamcoder_core.lean_primitives import build_lean_grammar
    grammar = build_lean_grammar()
    prim_names = [p.name for p in grammar.primitives()]

    found = set()
    for i, name in enumerate(prim_names):
        if name in str(program):
            found.add(i)
    return found


def train_fold(model, train_tasks, val_tasks, epochs=100, lr=1e-3, patience=10):
    """Train model with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n_batches = 0

        for task in train_tasks:
            optimizer.zero_grad()
            loss = model.compute_loss(task)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for task in val_tasks:
                val_loss += model.compute_loss(task).item()
        val_loss /= len(val_tasks) if val_tasks else 1

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    return best_epoch


def run_experiment(config, grammar, train_rules, test_rules, n_folds=5):
    """Run cross-validation experiment for a variant."""
    results = {
        'config': config,
        'fold_results': [],
        'mean_metrics': {},
        'std_metrics': {}
    }

    # Split training rules into folds
    indices = np.arange(len(train_rules))
    np.random.shuffle(indices)
    fold_size = len(indices) // n_folds

    for fold in range(n_folds):
        logger.info(f"    Fold {fold+1}/{n_folds}")

        try:
            # Split into train/val
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else len(indices)
            val_indices = indices[val_start:val_end]
            train_indices = np.concatenate([indices[:val_start], indices[val_end:]])

            train_tasks = [Task(train_rules[i]) for i in train_indices]
            val_tasks = [Task(train_rules[i]) for i in val_indices]
            test_tasks = [Task(r) for r in test_rules]

            # Create model
            model = RecognitionModelVariant(
                grammar=grammar,
                card_encoder_type=config['card_encoder'],
                hand_encoder_type=config['hand_encoder'],
                task_encoder_type=config['task_encoder'],
                prediction_head_type=config['prediction_head'],
                loss_type=config['loss']
            )

            # Train
            best_epoch = train_fold(model, train_tasks, val_tasks)

            # Evaluate
            metrics = evaluate_model(model, test_tasks)
            metrics['convergence_epoch'] = best_epoch

            results['fold_results'].append(metrics)

        except Exception as e:
            logger.error(f"    Fold {fold+1} failed: {e}")
            results['fold_results'].append({'error': str(e)})

    # Compute mean/std
    valid_folds = [f for f in results['fold_results'] if 'error' not in f]
    if valid_folds:
        for key in valid_folds[0].keys():
            values = [f[key] for f in valid_folds if key in f]
            results['mean_metrics'][key] = np.mean(values)
            results['std_metrics'][key] = np.std(values)

    return results


def main():
    logger.info("="*70)
    logger.info("RE-RUNNING EMBEDDING HEAD VARIANTS")
    logger.info("="*70)

    # Setup
    grammar = build_lean_grammar()
    train_rules = create_pretraining_rules()
    test_rules = create_all_rules()

    logger.info(f"Grammar: {len(grammar.primitives())} primitives")
    logger.info(f"Pre-training rules: {len(train_rules)}")
    logger.info(f"Test rules: {len(test_rules)}")

    # Results
    all_results = []

    for i, config in enumerate(EMBEDDING_VARIANTS):
        logger.info(f"\n[{i+1}/{len(EMBEDDING_VARIANTS)}] Testing: {config['name']}")

        results = run_experiment(config, grammar, train_rules, test_rules)
        all_results.append(results)

        if results['mean_metrics']:
            r5 = results['mean_metrics'].get('recall@5', 0)
            mrr = results['mean_metrics'].get('mrr', 0)
            logger.info(f"  => R@5: {r5:.3f}, MRR: {mrr:.3f}")
        else:
            logger.info(f"  => All folds failed")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'results_embedding_variants_{timestamp}')
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print summary
    logger.info("\n" + "="*70)
    logger.info("EMBEDDING VARIANTS SUMMARY")
    logger.info("="*70)

    for r in sorted(all_results, key=lambda x: x['mean_metrics'].get('recall@5', 0), reverse=True):
        name = r['config']['name']
        if r['mean_metrics']:
            r5 = r['mean_metrics'].get('recall@5', 0)
            r10 = r['mean_metrics'].get('recall@10', 0)
            mrr = r['mean_metrics'].get('mrr', 0)
            logger.info(f"{name:35} R@5: {r5:.3f}  R@10: {r10:.3f}  MRR: {mrr:.3f}")
        else:
            logger.info(f"{name:35} FAILED")

    logger.info(f"\nResults saved to: {output_dir}")


if __name__ == '__main__':
    main()
