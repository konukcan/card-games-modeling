#!/usr/bin/env python3
"""
Run experiments testing random contrast task encoding variants.

Tests three new contrastive encoding strategies:
1. RandomAugmented: τ = mean(pos) - mean(neg) + λ*(mean(pos+neg) - mean(random))
2. PositiveVsRandom: τ = mean(pos) - mean(random)
3. Triple: concat([pos-neg, pos-random, neg-random])

Also tests combinations with the best-performing components from the
architectural variants comparison (enhanced_card encoder).
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


# Define the random contrast variants to test
RANDOM_CONTRAST_VARIANTS = [
    # Baseline for comparison
    {
        'card_encoder': 'standard',
        'hand_encoder': 'mean',
        'task_encoder': 'standard',
        'prediction_head': 'sigmoid',
        'loss': 'bce',
        'n_random_hands': 0,
        'random_lambda': 0.0,
        'name': 'baseline_standard'
    },
    {
        'card_encoder': 'enhanced',
        'hand_encoder': 'mean',
        'task_encoder': 'standard',
        'prediction_head': 'sigmoid',
        'loss': 'bce',
        'n_random_hands': 0,
        'random_lambda': 0.0,
        'name': 'baseline_enhanced'
    },

    # Random Augmented variants (τ = pos-neg + λ*(combined-random))
    {
        'card_encoder': 'standard',
        'hand_encoder': 'mean',
        'task_encoder': 'random_augmented',
        'prediction_head': 'sigmoid',
        'loss': 'bce',
        'n_random_hands': 25,
        'random_lambda': 0.5,
        'name': 'random_augmented_lambda0.5'
    },
    {
        'card_encoder': 'standard',
        'hand_encoder': 'mean',
        'task_encoder': 'random_augmented',
        'prediction_head': 'sigmoid',
        'loss': 'bce',
        'n_random_hands': 25,
        'random_lambda': 1.0,
        'name': 'random_augmented_lambda1.0'
    },
    {
        'card_encoder': 'enhanced',
        'hand_encoder': 'mean',
        'task_encoder': 'random_augmented',
        'prediction_head': 'sigmoid',
        'loss': 'bce',
        'n_random_hands': 25,
        'random_lambda': 0.5,
        'name': 'enhanced+random_augmented'
    },

    # Positive vs Random variants (τ = pos - random)
    {
        'card_encoder': 'standard',
        'hand_encoder': 'mean',
        'task_encoder': 'pos_vs_random',
        'prediction_head': 'sigmoid',
        'loss': 'bce',
        'n_random_hands': 25,
        'random_lambda': 0.0,
        'name': 'pos_vs_random_standard'
    },
    {
        'card_encoder': 'enhanced',
        'hand_encoder': 'mean',
        'task_encoder': 'pos_vs_random',
        'prediction_head': 'sigmoid',
        'loss': 'bce',
        'n_random_hands': 25,
        'random_lambda': 0.0,
        'name': 'pos_vs_random_enhanced'
    },

    # Triple Contrast variants (concat [pos-neg, pos-random, neg-random])
    {
        'card_encoder': 'standard',
        'hand_encoder': 'mean',
        'task_encoder': 'triple',
        'prediction_head': 'sigmoid',
        'loss': 'bce',
        'n_random_hands': 25,
        'random_lambda': 0.0,
        'name': 'triple_contrast_standard'
    },
    {
        'card_encoder': 'enhanced',
        'hand_encoder': 'mean',
        'task_encoder': 'triple',
        'prediction_head': 'sigmoid',
        'loss': 'bce',
        'n_random_hands': 25,
        'random_lambda': 0.0,
        'name': 'triple_contrast_enhanced'
    },

    # Combinations with other best components
    {
        'card_encoder': 'enhanced',
        'hand_encoder': 'attention',
        'task_encoder': 'triple',
        'prediction_head': 'sigmoid',
        'loss': 'bce',
        'n_random_hands': 25,
        'random_lambda': 0.0,
        'name': 'enhanced+attention+triple'
    },

    # Varying number of random hands
    {
        'card_encoder': 'standard',
        'hand_encoder': 'mean',
        'task_encoder': 'triple',
        'prediction_head': 'sigmoid',
        'loss': 'bce',
        'n_random_hands': 50,
        'random_lambda': 0.0,
        'name': 'triple_contrast_50random'
    },
    {
        'card_encoder': 'standard',
        'hand_encoder': 'mean',
        'task_encoder': 'random_augmented',
        'prediction_head': 'sigmoid',
        'loss': 'bce',
        'n_random_hands': 50,
        'random_lambda': 0.5,
        'name': 'random_augmented_50random'
    },
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

            # Probability ratio (higher prob for correct vs incorrect primitives)
            gt_probs = preds[list(gt_indices)].mean().item()
            non_gt_indices = [i for i in range(len(preds)) if i not in gt_indices]
            non_gt_probs = preds[non_gt_indices].mean().item() if non_gt_indices else 0.001
            if non_gt_probs > 0:
                metrics['prob_ratio'].append(gt_probs / non_gt_probs)

            # Prediction diversity (std of predictions)
            metrics['pred_std'].append(preds.std().item())
            metrics['pred_range'].append((preds.max() - preds.min()).item())

    return {k: np.mean(v) if v else 0.0 for k, v in metrics.items()}


def train_fold(model, train_tasks, val_tasks, epochs=100, lr=1e-3, patience=15):
    """Train model with early stopping."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_losses = []

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

        avg_train_loss = train_loss / n_batches if n_batches > 0 else 0
        train_losses.append(avg_train_loss)

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

    return best_epoch, train_losses[-1] if train_losses else 0


def run_experiment(config, grammar, train_rules, test_rules, n_folds=5):
    """Run cross-validation experiment for a variant."""
    results = {
        'config': config,
        'config_name': config['name'],
        'fold_results': [],
    }

    # Split training rules into folds
    indices = np.arange(len(train_rules))
    np.random.seed(42)  # For reproducibility
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
                loss_type=config['loss'],
                n_random_hands=config.get('n_random_hands', 25),
                random_lambda=config.get('random_lambda', 0.5)
            )

            # Train
            best_epoch, final_train_loss = train_fold(model, train_tasks, val_tasks)

            # Evaluate
            metrics = evaluate_model(model, test_tasks)
            metrics['convergence_epoch'] = best_epoch
            metrics['train_loss'] = final_train_loss

            results['fold_results'].append(metrics)

        except Exception as e:
            logger.error(f"    Fold {fold+1} failed: {e}")
            import traceback
            traceback.print_exc()
            results['fold_results'].append({'error': str(e)})

    # Compute mean/std across folds
    valid_folds = [f for f in results['fold_results'] if 'error' not in f]
    if valid_folds:
        for key in valid_folds[0].keys():
            values = [f[key] for f in valid_folds if key in f]
            results[f'{key}_mean'] = np.mean(values)
            results[f'{key}_std'] = np.std(values)
        results['successful_folds'] = len(valid_folds)
    else:
        results['error'] = 'All folds failed'

    return results


def generate_report(all_results, output_dir):
    """Generate a markdown comparison report."""
    report = []
    report.append("# Random Contrast Task Encoding Variants: Comparison Report\n")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    report.append(f"**Total variants tested:** {len(all_results)}\n")

    # Separate successful and failed
    successful = [r for r in all_results if 'error' not in r]
    failed = [r for r in all_results if 'error' in r]

    report.append(f"\n**Successful:** {len(successful)}, **Failed:** {len(failed)}\n")

    # Sort by recall@5
    successful.sort(key=lambda x: x.get('recall@5_mean', 0), reverse=True)

    report.append("\n## Main Results Table\n")
    report.append("| Variant | R@5 | R@10 | MRR | Prob Ratio | Conv. Epoch |")
    report.append("|---------|-----|------|-----|------------|-------------|")

    for r in successful:
        name = r['config_name']
        r5 = r.get('recall@5_mean', 0)
        r5_std = r.get('recall@5_std', 0)
        r10 = r.get('recall@10_mean', 0)
        mrr = r.get('mrr_mean', 0)
        mrr_std = r.get('mrr_std', 0)
        prob = r.get('prob_ratio_mean', 0)
        epoch = r.get('convergence_epoch_mean', 0)
        report.append(f"| {name} | {r5:.3f}±{r5_std:.2f} | {r10:.3f} | {mrr:.3f}±{mrr_std:.2f} | {prob:.1f} | {epoch:.0f} |")

    # Component-wise analysis
    report.append("\n## Task Encoder Analysis\n")
    report.append("| Encoder Type | Mean R@5 | Mean MRR | Count |")
    report.append("|--------------|----------|----------|-------|")

    encoder_stats = defaultdict(lambda: {'r5': [], 'mrr': []})
    for r in successful:
        enc = r['config']['task_encoder']
        encoder_stats[enc]['r5'].append(r.get('recall@5_mean', 0))
        encoder_stats[enc]['mrr'].append(r.get('mrr_mean', 0))

    for enc, stats in sorted(encoder_stats.items()):
        r5 = np.mean(stats['r5'])
        mrr = np.mean(stats['mrr'])
        n = len(stats['r5'])
        report.append(f"| {enc} | {r5:.3f} | {mrr:.3f} | {n} |")

    # Failed variants
    if failed:
        report.append("\n## Failed Variants\n")
        for r in failed:
            report.append(f"- {r['config_name']}: {r.get('error', 'Unknown error')}")

    # Key insights
    report.append("\n## Key Insights\n")
    if successful:
        best = successful[0]
        report.append(f"1. **Best performing variant:** {best['config_name']} (R@5={best.get('recall@5_mean', 0):.3f})")

        # Compare to baseline
        baseline = next((r for r in successful if 'baseline' in r['config_name']), None)
        if baseline:
            baseline_r5 = baseline.get('recall@5_mean', 0)
            best_r5 = best.get('recall@5_mean', 0)
            diff = (best_r5 - baseline_r5) / baseline_r5 * 100 if baseline_r5 > 0 else 0
            report.append(f"2. **Improvement over baseline:** {diff:+.1f}%")

    report.append("\n")
    return "\n".join(report)


def main():
    logger.info("=" * 70)
    logger.info("RANDOM CONTRAST TASK ENCODING VARIANTS COMPARISON")
    logger.info("=" * 70)

    # Setup
    grammar = build_lean_grammar()
    train_rules = create_pretraining_rules()
    test_rules = create_all_rules()

    logger.info(f"Grammar: {len(grammar.primitives())} primitives")
    logger.info(f"Pre-training rules: {len(train_rules)}")
    logger.info(f"Test rules: {len(test_rules)}")
    logger.info(f"Variants to test: {len(RANDOM_CONTRAST_VARIANTS)}")

    # Results
    all_results = []

    for i, config in enumerate(RANDOM_CONTRAST_VARIANTS):
        logger.info(f"\n[{i+1}/{len(RANDOM_CONTRAST_VARIANTS)}] Testing: {config['name']}")

        results = run_experiment(config, grammar, train_rules, test_rules)
        all_results.append(results)

        if 'error' not in results:
            r5 = results.get('recall@5_mean', 0)
            mrr = results.get('mrr_mean', 0)
            logger.info(f"  => R@5: {r5:.3f}, MRR: {mrr:.3f}")
        else:
            logger.info(f"  => All folds failed: {results.get('error', 'Unknown')}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'results_random_contrast_{timestamp}')
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'full_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Generate and save report
    report = generate_report(all_results, output_dir)
    with open(output_dir / 'comparison_report.md', 'w') as f:
        f.write(report)

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("RANDOM CONTRAST VARIANTS SUMMARY")
    logger.info("=" * 70)

    successful = [r for r in all_results if 'error' not in r]
    successful.sort(key=lambda x: x.get('recall@5_mean', 0), reverse=True)

    for r in successful:
        name = r['config_name']
        r5 = r.get('recall@5_mean', 0)
        r10 = r.get('recall@10_mean', 0)
        mrr = r.get('mrr_mean', 0)
        logger.info(f"{name:35} R@5: {r5:.3f}  R@10: {r10:.3f}  MRR: {mrr:.3f}")

    logger.info(f"\nResults saved to: {output_dir}")
    logger.info(f"Report saved to: {output_dir / 'comparison_report.md'}")

    return output_dir


if __name__ == '__main__':
    main()
