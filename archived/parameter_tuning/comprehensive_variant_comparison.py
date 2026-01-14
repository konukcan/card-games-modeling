#!/usr/bin/env python3
"""
Comprehensive Comparison of Recognition Model Variants

This script tests all architectural variants systematically:
1. Card encoder: standard vs enhanced (color + rank value)
2. Hand encoder: mean vs attention vs deepsets vs multiscale
3. Task encoder: standard vs multihead
4. Prediction head: sigmoid vs embedding
5. Loss: BCE vs focal

All combinations are tested with cross-validation and compared using:
- Recall@k metrics
- MRR (Mean Reciprocal Rank)
- ProbRatio (solution vs non-solution probability ratio)
- Prediction diversity
- Training convergence
- Interpretability analysis

Author: Can Konuk
Date: December 2024
"""

import sys
import json
import random
import logging
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field
from itertools import product
import traceback

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.recognition_variants import (
    RecognitionModelVariant,
    EnhancedCardEncoder,
    SelfAttentionHandEncoder,
    DeepSetsEncoder,
    MultiScalePoolingEncoder,
    MultiHeadContrastEncoder,
    PrimitiveEmbeddingHead,
    FocalLoss
)
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.grammar import Grammar
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
        hand = sample_hand(size=6)  # Standardized to 6
        try:
            label = rule.predicate(hand)
            if label and pos_count < n_examples // 2:
                examples.append((hand, True))
                pos_count += 1
            elif not label and neg_count < n_examples // 2:
                examples.append((hand, False))
                neg_count += 1
        except:
            pass
        attempts += 1

    return RevelationTask(
        name=rule.id,
        examples=examples,
        primitives_used=set(rule.primitives_used),
        source='catalogue'
    )


def create_task_from_pretraining_rule(rule: PretrainingRule, grammar: Grammar, n_examples: int = 50, seed: int = 42) -> RevelationTask:
    """Create a revelation task from a pre-training rule."""
    rng = random.Random(seed + hash(rule.id) % 10000)

    examples = []
    pos_count = 0
    neg_count = 0
    attempts = 0

    while (pos_count < n_examples // 2 or neg_count < n_examples // 2) and attempts < n_examples * 20:
        hand = sample_hand(size=6)
        try:
            label = rule.eval(hand)
            if label and pos_count < n_examples // 2:
                examples.append((hand, True))
                pos_count += 1
            elif not label and neg_count < n_examples // 2:
                examples.append((hand, False))
                neg_count += 1
        except:
            pass
        attempts += 1

    prim_names = {p.name for p in grammar.primitives()}
    words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', rule.expected_program)
    primitives = {w for w in words if w in prim_names}

    return RevelationTask(
        name=rule.id,
        examples=examples,
        primitives_used=primitives,
        source='pretraining'
    )


# ============================================================================
# Training
# ============================================================================

def train_model(
    model: RecognitionModelVariant,
    tasks: List[RevelationTask],
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 0.001
) -> Dict[str, Any]:
    """Train model and return training stats."""
    if not tasks:
        return {'final_loss': 0.0, 'loss_history': [], 'convergence_epoch': 0}

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    training_data = []
    for task in tasks:
        if len(task.examples) < 10:
            continue

        target = torch.zeros(model.num_primitives, device=model.device)
        for prim_name in task.primitives_used:
            if prim_name in model.primitive_to_idx:
                target[model.primitive_to_idx[prim_name]] = 1.0

        if target.sum() > 0:
            training_data.append((task, target))

    if not training_data:
        return {'final_loss': 0.0, 'loss_history': [], 'convergence_epoch': 0}

    loss_history = []
    best_loss = float('inf')
    patience = 20
    patience_counter = 0
    convergence_epoch = epochs

    for epoch in range(epochs):
        random.shuffle(training_data)
        epoch_losses = []

        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]

            batch_preds = []
            batch_targets = []

            for task, target in batch:
                try:
                    τ = model.encode_task_batched(task)
                    if isinstance(model.primitive_head, torch.nn.Sequential):
                        pred = model.primitive_head(τ.unsqueeze(0))
                    else:
                        pred = model.primitive_head(τ.unsqueeze(0))
                    batch_preds.append(pred.squeeze(0))
                    batch_targets.append(target)
                except Exception as e:
                    continue

            if not batch_preds:
                continue

            preds = torch.stack(batch_preds)
            targets = torch.stack(batch_targets)

            # Compute loss based on model's loss type
            if isinstance(model.loss_fn, FocalLoss):
                probs = torch.sigmoid(preds) if isinstance(model.primitive_head, torch.nn.Sequential) else preds
                loss = model.loss_fn(probs, targets)
            elif isinstance(model.loss_fn, torch.nn.BCEWithLogitsLoss):
                loss = model.loss_fn(preds, targets)
            else:
                probs = torch.sigmoid(preds) if isinstance(model.primitive_head, torch.nn.Sequential) else preds
                loss = F.binary_cross_entropy(probs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            loss_history.append(avg_loss)

            # Early stopping check
            if avg_loss < best_loss - 0.001:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience and convergence_epoch == epochs:
                convergence_epoch = epoch - patience

    return {
        'final_loss': loss_history[-1] if loss_history else 0.0,
        'loss_history': loss_history,
        'convergence_epoch': convergence_epoch
    }


# ============================================================================
# Evaluation Metrics
# ============================================================================

def compute_metrics(predictions: np.ndarray, solution_prims: Set[str], prim_names: List[str]) -> Dict[str, float]:
    """Compute comprehensive prediction quality metrics."""
    sorted_idx = np.argsort(predictions)[::-1]
    sorted_names = [prim_names[i] for i in sorted_idx]

    n_sol = len(solution_prims)
    n_prims = len(prim_names)

    # Find ranks of solution primitives
    ranks = []
    for i, name in enumerate(sorted_names):
        if name in solution_prims:
            ranks.append(i + 1)

    # Recall@k
    def recall_k(k):
        if n_sol == 0:
            return 1.0
        return len(solution_prims & set(sorted_names[:k])) / n_sol

    # Precision@k
    def precision_k(k):
        return len(solution_prims & set(sorted_names[:k])) / k

    # MRR
    mrr = np.mean([1.0/r for r in ranks]) if ranks else 0.0
    mean_rank = np.mean(ranks) if ranks else n_prims

    # ProbRatio
    sol_idx = [prim_names.index(p) for p in solution_prims if p in prim_names]
    other_idx = [i for i in range(n_prims) if prim_names[i] not in solution_prims]

    if sol_idx and other_idx:
        mean_sol_prob = float(np.mean(predictions[sol_idx]))
        mean_other_prob = float(np.mean(predictions[other_idx]))
        prob_ratio = mean_sol_prob / mean_other_prob if mean_other_prob > 0 else 0.0
    else:
        prob_ratio = 1.0

    # Prediction spread (diversity indicator)
    pred_std = float(np.std(predictions))
    pred_range = float(np.max(predictions) - np.min(predictions))

    return {
        'recall@5': float(recall_k(5)),
        'recall@10': float(recall_k(10)),
        'recall@15': float(recall_k(15)),
        'precision@5': float(precision_k(5)),
        'mrr': float(mrr),
        'mean_rank': float(mean_rank),
        'prob_ratio': float(prob_ratio),
        'pred_std': pred_std,
        'pred_range': pred_range,
        'top5': sorted_names[:5]
    }


def evaluate_model(model: RecognitionModelVariant, tasks: List[RevelationTask]) -> Dict[str, Any]:
    """Comprehensive evaluation of a model."""
    model.eval()
    all_metrics = []
    all_predictions = []
    all_embeddings = []

    with torch.no_grad():
        for task in tasks:
            if len(task.examples) < 10:
                continue

            try:
                # Get embedding
                τ = model.encode_task_batched(task)
                all_embeddings.append(τ.cpu().numpy())

                # Get predictions
                pred = model.predict_primitives(task).cpu().numpy()
                all_predictions.append(pred)

                # Compute metrics
                metrics = compute_metrics(pred, task.primitives_used, model.primitive_names)
                metrics['task'] = task.name
                all_metrics.append(metrics)
            except Exception as e:
                continue

    if not all_metrics:
        return {'error': 'No tasks evaluated'}

    # Aggregate metrics
    result = {}
    for key in ['recall@5', 'recall@10', 'recall@15', 'precision@5', 'mrr', 'mean_rank', 'prob_ratio', 'pred_std', 'pred_range']:
        values = [m[key] for m in all_metrics]
        result[f'mean_{key}'] = float(np.mean(values))
        result[f'std_{key}'] = float(np.std(values))

    # Embedding analysis
    if all_embeddings:
        embeddings = np.array(all_embeddings)
        result['embedding_norm_mean'] = float(np.mean(np.linalg.norm(embeddings, axis=1)))
        result['embedding_norm_std'] = float(np.std(np.linalg.norm(embeddings, axis=1)))

        # Pairwise similarity
        if len(embeddings) > 1:
            from sklearn.metrics.pairwise import cosine_similarity
            sims = cosine_similarity(embeddings)
            np.fill_diagonal(sims, 0)
            result['embedding_similarity_mean'] = float(np.mean(sims))

    # Prediction diversity
    if all_predictions:
        predictions = np.array(all_predictions)
        # Count unique top-5 sets
        top5_sets = [tuple(np.argsort(p)[-5:]) for p in predictions]
        result['unique_top5_count'] = len(set(top5_sets))
        result['total_tasks'] = len(predictions)
        result['diversity_ratio'] = result['unique_top5_count'] / max(1, result['total_tasks'])

    result['n_tasks_evaluated'] = len(all_metrics)

    return result


# ============================================================================
# Variant Configuration
# ============================================================================

def get_all_variant_configs() -> List[Dict[str, str]]:
    """Generate all variant configurations to test."""
    # Define options for each component
    card_encoders = ['standard', 'enhanced']
    hand_encoders = ['mean', 'attention', 'deepsets', 'multiscale']
    task_encoders = ['standard', 'multihead']
    prediction_heads = ['sigmoid', 'embedding']
    losses = ['bce', 'focal']

    # Generate all combinations
    all_configs = []

    for card, hand, task, pred, loss in product(
        card_encoders, hand_encoders, task_encoders, prediction_heads, losses
    ):
        all_configs.append({
            'card_encoder': card,
            'hand_encoder': hand,
            'task_encoder': task,
            'prediction_head': pred,
            'loss': loss
        })

    return all_configs


def get_priority_variant_configs() -> List[Dict[str, str]]:
    """Get priority variants based on theoretical analysis."""
    return [
        # Baseline (reference)
        {'card_encoder': 'standard', 'hand_encoder': 'mean', 'task_encoder': 'standard', 'prediction_head': 'sigmoid', 'loss': 'bce', 'name': 'baseline'},

        # Single improvements (to isolate effects)
        {'card_encoder': 'enhanced', 'hand_encoder': 'mean', 'task_encoder': 'standard', 'prediction_head': 'sigmoid', 'loss': 'bce', 'name': 'enhanced_card'},
        {'card_encoder': 'standard', 'hand_encoder': 'attention', 'task_encoder': 'standard', 'prediction_head': 'sigmoid', 'loss': 'bce', 'name': 'attention_hand'},
        {'card_encoder': 'standard', 'hand_encoder': 'deepsets', 'task_encoder': 'standard', 'prediction_head': 'sigmoid', 'loss': 'bce', 'name': 'deepsets_hand'},
        {'card_encoder': 'standard', 'hand_encoder': 'multiscale', 'task_encoder': 'standard', 'prediction_head': 'sigmoid', 'loss': 'bce', 'name': 'multiscale_hand'},
        {'card_encoder': 'standard', 'hand_encoder': 'mean', 'task_encoder': 'multihead', 'prediction_head': 'sigmoid', 'loss': 'bce', 'name': 'multihead_task'},
        {'card_encoder': 'standard', 'hand_encoder': 'mean', 'task_encoder': 'standard', 'prediction_head': 'embedding', 'loss': 'bce', 'name': 'embedding_head'},
        {'card_encoder': 'standard', 'hand_encoder': 'mean', 'task_encoder': 'standard', 'prediction_head': 'sigmoid', 'loss': 'focal', 'name': 'focal_loss'},

        # Top combinations (based on theory)
        {'card_encoder': 'enhanced', 'hand_encoder': 'attention', 'task_encoder': 'standard', 'prediction_head': 'sigmoid', 'loss': 'bce', 'name': 'enhanced+attention'},
        {'card_encoder': 'enhanced', 'hand_encoder': 'attention', 'task_encoder': 'standard', 'prediction_head': 'embedding', 'loss': 'bce', 'name': 'enhanced+attention+embed'},
        {'card_encoder': 'enhanced', 'hand_encoder': 'attention', 'task_encoder': 'standard', 'prediction_head': 'embedding', 'loss': 'focal', 'name': 'enhanced+attention+embed+focal'},
        {'card_encoder': 'enhanced', 'hand_encoder': 'deepsets', 'task_encoder': 'standard', 'prediction_head': 'embedding', 'loss': 'bce', 'name': 'enhanced+deepsets+embed'},
        {'card_encoder': 'enhanced', 'hand_encoder': 'multiscale', 'task_encoder': 'multihead', 'prediction_head': 'embedding', 'loss': 'focal', 'name': 'full_enhanced'},
        {'card_encoder': 'standard', 'hand_encoder': 'attention', 'task_encoder': 'multihead', 'prediction_head': 'embedding', 'loss': 'focal', 'name': 'attention+multi+embed+focal'},
    ]


# ============================================================================
# Cross-Validation Runner
# ============================================================================

def run_cross_validation(
    config: Dict[str, str],
    train_tasks: List[RevelationTask],
    test_tasks: List[RevelationTask],
    grammar,
    n_folds: int = 5,
    epochs: int = 100
) -> Dict[str, Any]:
    """Run cross-validation for a single configuration."""
    config_name = config.get('name', f"{config['card_encoder']}_{config['hand_encoder']}_{config['task_encoder']}_{config['prediction_head']}_{config['loss']}")

    logger.info(f"  Testing: {config_name}")

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_tasks)):
        fold_train = [train_tasks[i] for i in train_idx]
        fold_val = [train_tasks[i] for i in val_idx]

        try:
            # Create model
            model = RecognitionModelVariant(
                grammar=grammar,
                card_encoder_type=config['card_encoder'],
                hand_encoder_type=config['hand_encoder'],
                task_encoder_type=config['task_encoder'],
                prediction_head_type=config['prediction_head'],
                loss_type=config['loss'],
                card_hidden=128,
                card_out=64,
                pred_hidden=128,
                normalize_embeddings=True,
                embedding_scale=20.0
            )

            # Train
            train_stats = train_model(model, fold_train, epochs=epochs)

            # Evaluate on validation fold
            val_metrics = evaluate_model(model, fold_val)

            # Evaluate on test set
            test_metrics = evaluate_model(model, test_tasks)

            fold_results.append({
                'fold': fold_idx + 1,
                'train_loss': train_stats['final_loss'],
                'convergence_epoch': train_stats['convergence_epoch'],
                'val_metrics': val_metrics,
                'test_metrics': test_metrics
            })

        except Exception as e:
            logger.error(f"    Fold {fold_idx + 1} failed: {e}")
            fold_results.append({'fold': fold_idx + 1, 'error': str(e)})

    # Aggregate results
    successful_folds = [f for f in fold_results if 'error' not in f]

    if not successful_folds:
        return {'config': config, 'error': 'All folds failed', 'fold_results': fold_results}

    # Aggregate test metrics
    aggregate = {'config': config, 'config_name': config_name}

    test_metric_keys = [k for k in successful_folds[0]['test_metrics'].keys() if k.startswith('mean_')]

    for key in test_metric_keys:
        values = [f['test_metrics'][key] for f in successful_folds]
        short_key = key.replace('mean_', '')
        aggregate[f'test_{short_key}_mean'] = float(np.mean(values))
        aggregate[f'test_{short_key}_std'] = float(np.std(values))

    # Training stats
    aggregate['train_loss_mean'] = float(np.mean([f['train_loss'] for f in successful_folds]))
    aggregate['convergence_epoch_mean'] = float(np.mean([f['convergence_epoch'] for f in successful_folds]))

    # Diversity
    if 'diversity_ratio' in successful_folds[0]['test_metrics']:
        aggregate['diversity_ratio_mean'] = float(np.mean([f['test_metrics']['diversity_ratio'] for f in successful_folds]))

    aggregate['successful_folds'] = len(successful_folds)
    aggregate['fold_results'] = fold_results

    return aggregate


# ============================================================================
# Interpretability Analysis
# ============================================================================

def analyze_primitive_embeddings(model: RecognitionModelVariant) -> Dict[str, Any]:
    """Analyze learned primitive embeddings (for embedding head variant)."""
    if not isinstance(model.primitive_head, PrimitiveEmbeddingHead):
        return {}

    with torch.no_grad():
        sim_matrix = model.primitive_head.get_primitive_similarity()
        sim_np = sim_matrix.cpu().numpy()

    # Find most similar primitive pairs
    n_prims = len(model.primitive_names)
    pairs = []
    for i in range(n_prims):
        for j in range(i + 1, n_prims):
            pairs.append({
                'prim1': model.primitive_names[i],
                'prim2': model.primitive_names[j],
                'similarity': float(sim_np[i, j])
            })

    pairs.sort(key=lambda x: -x['similarity'])

    return {
        'top_similar_pairs': pairs[:10],
        'bottom_similar_pairs': pairs[-10:],
        'mean_similarity': float(np.mean(sim_np[np.triu_indices(n_prims, k=1)])),
        'similarity_std': float(np.std(sim_np[np.triu_indices(n_prims, k=1)]))
    }


def analyze_attention_patterns(model: RecognitionModelVariant, tasks: List[RevelationTask]) -> Dict[str, Any]:
    """Analyze attention patterns (for attention hand encoder variant)."""
    if model.config['hand_encoder'] != 'attention':
        return {}

    # This would require modifying the attention layer to return attention weights
    # For now, we'll return placeholder analysis
    return {
        'attention_analysis': 'Attention patterns require model modification to extract weights'
    }


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(all_results: List[Dict], output_dir: Path) -> str:
    """Generate comprehensive comparison report."""
    report = []
    report.append("=" * 100)
    report.append("COMPREHENSIVE RECOGNITION MODEL VARIANT COMPARISON")
    report.append("=" * 100)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total variants tested: {len(all_results)}")
    report.append("")

    # Filter successful results
    successful = [r for r in all_results if 'error' not in r]
    failed = [r for r in all_results if 'error' in r]

    report.append(f"Successful: {len(successful)}, Failed: {len(failed)}")

    if failed:
        report.append("\nFailed configurations:")
        for r in failed:
            report.append(f"  - {r.get('config_name', r.get('config', 'unknown'))}: {r.get('error', 'unknown error')}")

    if not successful:
        return "\n".join(report)

    # Sort by test recall@5
    successful.sort(key=lambda x: -x.get('test_recall@5_mean', 0))

    # Main comparison table
    report.append("\n" + "=" * 100)
    report.append("MAIN RESULTS TABLE (sorted by Test R@5)")
    report.append("=" * 100)
    report.append("")

    header = f"{'Variant':<40} {'R@5':<12} {'R@10':<12} {'MRR':<12} {'Diversity':<12} {'Conv.Epoch':<12}"
    report.append(header)
    report.append("-" * 100)

    for r in successful:
        name = r.get('config_name', 'unknown')[:38]
        r5 = f"{r.get('test_recall@5_mean', 0):.3f}±{r.get('test_recall@5_std', 0):.2f}"
        r10 = f"{r.get('test_recall@10_mean', 0):.3f}±{r.get('test_recall@10_std', 0):.2f}"
        mrr = f"{r.get('test_mrr_mean', 0):.3f}±{r.get('test_mrr_std', 0):.2f}"
        div = f"{r.get('diversity_ratio_mean', 0):.2f}"
        conv = f"{r.get('convergence_epoch_mean', 0):.0f}"
        report.append(f"{name:<40} {r5:<12} {r10:<12} {mrr:<12} {div:<12} {conv:<12}")

    # Best variants analysis
    report.append("\n" + "=" * 100)
    report.append("BEST VARIANTS BY METRIC")
    report.append("=" * 100)

    metrics_to_maximize = ['test_recall@5_mean', 'test_recall@10_mean', 'test_mrr_mean', 'test_prob_ratio_mean', 'diversity_ratio_mean']
    metrics_to_minimize = ['test_mean_rank_mean', 'train_loss_mean', 'convergence_epoch_mean']

    for metric in metrics_to_maximize:
        best = max(successful, key=lambda x: x.get(metric, 0))
        clean_metric = metric.replace('test_', '').replace('_mean', '')
        report.append(f"\nBest {clean_metric}: {best.get('config_name', 'unknown')}")
        report.append(f"  Value: {best.get(metric, 0):.4f}")

    for metric in metrics_to_minimize:
        best = min(successful, key=lambda x: x.get(metric, float('inf')))
        clean_metric = metric.replace('test_', '').replace('_mean', '')
        report.append(f"\nBest {clean_metric} (lowest): {best.get('config_name', 'unknown')}")
        report.append(f"  Value: {best.get(metric, 0):.4f}")

    # Component analysis
    report.append("\n" + "=" * 100)
    report.append("COMPONENT-WISE ANALYSIS")
    report.append("=" * 100)

    # Analyze each component's effect
    components = {
        'card_encoder': ['standard', 'enhanced'],
        'hand_encoder': ['mean', 'attention', 'deepsets', 'multiscale'],
        'task_encoder': ['standard', 'multihead'],
        'prediction_head': ['sigmoid', 'embedding'],
        'loss': ['bce', 'focal']
    }

    for component, options in components.items():
        report.append(f"\n{component.upper()} comparison:")

        for option in options:
            matching = [r for r in successful if r.get('config', {}).get(component) == option]
            if matching:
                avg_r5 = np.mean([r.get('test_recall@5_mean', 0) for r in matching])
                avg_mrr = np.mean([r.get('test_mrr_mean', 0) for r in matching])
                report.append(f"  {option:15s}: R@5={avg_r5:.3f}, MRR={avg_mrr:.3f} (n={len(matching)})")

    # Statistical significance note
    report.append("\n" + "=" * 100)
    report.append("NOTES")
    report.append("=" * 100)
    report.append("""
1. Results are from 5-fold cross-validation on pre-training rules
2. Test set is the 45 catalogue rules
3. ± values indicate standard deviation across folds
4. Diversity ratio = unique top-5 prediction sets / total tasks
5. Convergence epoch = epoch where loss stopped improving
""")

    report_text = "\n".join(report)

    # Save report
    report_file = output_dir / "comparison_report.txt"
    with open(report_file, 'w') as f:
        f.write(report_text)

    return report_text


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    print("=" * 100)
    print("COMPREHENSIVE RECOGNITION MODEL VARIANT COMPARISON")
    print("=" * 100)
    print()

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results_variant_comparison_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")

    # Load grammar and tasks
    grammar = build_lean_grammar()
    print(f"Grammar: {len(grammar.productions)} primitives")

    pretraining_rules = get_all_pretraining_rules()
    catalogue_rules = create_all_rules()

    pretraining_tasks = [create_task_from_pretraining_rule(r, grammar) for r in pretraining_rules]
    pretraining_tasks = [t for t in pretraining_tasks if len(t.examples) >= 10 and len(t.primitives_used) > 0]

    catalogue_tasks = [create_task_from_catalogue_rule(r) for r in catalogue_rules]
    catalogue_tasks = [t for t in catalogue_tasks if len(t.examples) >= 10 and len(t.primitives_used) > 0]

    print(f"Pre-training tasks: {len(pretraining_tasks)}")
    print(f"Catalogue tasks: {len(catalogue_tasks)}")

    # Get variant configurations
    configs = get_priority_variant_configs()
    print(f"\nTesting {len(configs)} priority variants...")

    # Run experiments
    all_results = []

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Testing variant: {config.get('name', 'unknown')}")

        try:
            result = run_cross_validation(
                config=config,
                train_tasks=pretraining_tasks,
                test_tasks=catalogue_tasks,
                grammar=grammar,
                n_folds=5,
                epochs=100
            )
            all_results.append(result)

            # Print summary
            if 'error' not in result:
                r5 = result.get('test_recall@5_mean', 0)
                mrr = result.get('test_mrr_mean', 0)
                print(f"  => R@5: {r5:.3f}, MRR: {mrr:.3f}")
            else:
                print(f"  => Error: {result['error']}")

        except Exception as e:
            print(f"  => Failed: {e}")
            traceback.print_exc()
            all_results.append({'config': config, 'error': str(e)})

    # Generate report
    print("\n" + "=" * 100)
    print("GENERATING REPORT")
    print("=" * 100)

    report = generate_report(all_results, output_dir)
    print(report)

    # Save full results
    results_file = output_dir / "full_results.json"

    # Clean results for JSON serialization
    clean_results = []
    for r in all_results:
        clean_r = {}
        for k, v in r.items():
            if k == 'fold_results':
                continue  # Skip detailed fold results for main file
            if isinstance(v, (np.floating, np.integer)):
                clean_r[k] = float(v)
            elif isinstance(v, np.ndarray):
                clean_r[k] = v.tolist()
            else:
                clean_r[k] = v
        clean_results.append(clean_r)

    with open(results_file, 'w') as f:
        json.dump(clean_results, f, indent=2)

    print(f"\nFull results saved to: {results_file}")
    print(f"Report saved to: {output_dir / 'comparison_report.txt'}")


if __name__ == "__main__":
    main()
