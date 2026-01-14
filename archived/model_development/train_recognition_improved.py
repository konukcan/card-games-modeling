#!/usr/bin/env python3
"""
Improved Recognition Model Training with:
1. Deeper programs for synthetic rule generation (max_depth=12)
2. Many more epochs (200)
3. K-fold cross-validation
4. Better diversity in synthetic sampling

Two training pipelines:
1. Train on pre-training rules (44) → Test on catalogue rules (45)
2. Train on synthetic rules (random programs) → Test on both rule sets
"""

import sys
import json
import re
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass
from sklearn.model_selection import KFold
import copy

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.grammar import Grammar
from dreamcoder_core.type_system import arrow, HAND, BOOL
from dreamcoder_core.program import Program
from dreamcoder_core.enumeration import TopDownEnumerator
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
    source: str  # 'catalogue', 'pretraining', 'synthetic'
    depth: int = 0  # Program depth for synthetic tasks


def create_task_from_catalogue_rule(rule: Rule, n_examples: int = 50, seed: int = 42) -> RevelationTask:
    """Create a revelation task from a catalogue rule."""
    rng = random.Random(seed + hash(rule.id) % 10000)

    examples = []
    pos_count = 0
    neg_count = 0
    attempts = 0
    max_attempts = n_examples * 20

    while (pos_count < n_examples // 2 or neg_count < n_examples // 2) and attempts < max_attempts:
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
    max_attempts = n_examples * 20

    while (pos_count < n_examples // 2 or neg_count < n_examples // 2) and attempts < max_attempts:
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

    # Extract primitives from expected_program string
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
# Improved Synthetic Rule Generation
# ============================================================================

def extract_primitives(program: Program) -> Set[str]:
    """Extract all primitive names from a program."""
    prims = set()
    def traverse(p):
        ptype = type(p).__name__
        if ptype == 'Primitive' and hasattr(p, 'name'):
            prims.add(p.name)
        if hasattr(p, 'f'):
            traverse(p.f)
        if hasattr(p, 'x'):
            traverse(p.x)
        if hasattr(p, 'body'):
            traverse(p.body)
    traverse(program)
    return prims


def compute_program_depth(program: Program) -> int:
    """Compute the AST depth of a program."""
    def depth(p) -> int:
        ptype = type(p).__name__
        if ptype == 'Primitive' or ptype == 'Index':
            return 1
        elif ptype == 'Application':
            return 1 + max(depth(p.f), depth(p.x))
        elif ptype == 'Abstraction':
            return 1 + depth(p.body)
        elif ptype == 'Invented':
            return 1
        return 1
    return depth(program)


def generate_synthetic_rules_improved(
    grammar: Grammar,
    n_rules: int = 100,
    max_depth: int = 12,
    min_depth: int = 3,
    seed: int = 42,
    timeout_seconds: float = 120.0
) -> List[RevelationTask]:
    """
    Generate synthetic rules with DEEPER programs.

    Improvements:
    1. max_depth=12 (vs 8 before)
    2. Filter for min_depth to ensure non-trivial programs
    3. Diversity sampling by primitive set
    4. Longer timeout for more programs
    """
    rng = random.Random(seed)
    tasks = []

    logger.info(f"Generating {n_rules} synthetic rules (depth {min_depth}-{max_depth})...")

    # Enumerate programs of type HAND -> BOOL
    request_type = arrow(HAND, BOOL)
    enumerator = TopDownEnumerator(grammar, max_depth=max_depth, max_programs=n_rules * 10)

    programs_by_depth = {}  # Organize by depth for diversity
    programs_by_prims = {}  # Organize by primitive set for diversity

    for prog, lp in enumerator.enumerate(request_type, timeout_seconds=timeout_seconds):
        depth = compute_program_depth(prog)
        prims = extract_primitives(prog)

        # Skip trivial programs
        if not prims or prims == {'true'} or prims == {'false'}:
            continue
        if depth < min_depth:
            continue

        # Key by primitive set for diversity
        prim_key = frozenset(prims)

        if prim_key not in programs_by_prims:
            programs_by_prims[prim_key] = []
        programs_by_prims[prim_key].append((prog, lp, depth))

        # Also track by depth
        if depth not in programs_by_depth:
            programs_by_depth[depth] = []
        programs_by_depth[depth].append((prog, lp, prims))

        if len(programs_by_prims) >= n_rules * 2:
            break

    logger.info(f"  Enumerated programs across {len(programs_by_depth)} depth levels")
    logger.info(f"  Unique primitive sets: {len(programs_by_prims)}")

    # Report depth distribution
    for d in sorted(programs_by_depth.keys()):
        logger.info(f"    Depth {d}: {len(programs_by_depth[d])} programs")

    # Sample diversely: one program per unique primitive set
    selected = []
    for prim_key, prog_list in programs_by_prims.items():
        # Pick the deepest program for this primitive set
        prog_list.sort(key=lambda x: -x[2])  # Sort by depth descending
        selected.append((prog_list[0][0], prog_list[0][1], prim_key))
        if len(selected) >= n_rules:
            break

    logger.info(f"  Selected {len(selected)} diverse programs")

    # Create tasks from selected programs
    for i, (prog, lp, prim_key) in enumerate(selected):
        primitives = set(prim_key)
        depth = compute_program_depth(prog)

        # Generate examples
        examples = []
        try:
            fn = prog.evaluate([])
            pos_count = 0
            neg_count = 0

            for _ in range(500):  # Try up to 500 hands
                hand = sample_hand(size=6)  # Standardized to 6
                try:
                    result = fn(hand)
                    if result and pos_count < 25:
                        examples.append((hand, True))
                        pos_count += 1
                    elif not result and neg_count < 25:
                        examples.append((hand, False))
                        neg_count += 1
                except:
                    continue

                if pos_count >= 25 and neg_count >= 25:
                    break

            # Only keep if we have balanced examples
            if pos_count >= 10 and neg_count >= 10:
                tasks.append(RevelationTask(
                    name=f"synthetic_{i:03d}",
                    examples=examples,
                    primitives_used=primitives,
                    source='synthetic',
                    depth=depth
                ))
        except:
            continue

    logger.info(f"  Created {len(tasks)} valid synthetic tasks")

    # Report depth distribution of final tasks
    if tasks:
        depths = [t.depth for t in tasks]
        logger.info(f"  Task depths: min={min(depths)}, max={max(depths)}, mean={np.mean(depths):.1f}")

    return tasks


# ============================================================================
# Training with Cross-Validation
# ============================================================================

def train_model_with_epochs(
    model: ContrastiveRecognitionModel,
    tasks: List[RevelationTask],
    epochs: int = 200,
    batch_size: int = 8,
    lr: float = 0.001,
    log_every: int = 20
) -> Tuple[float, List[float]]:
    """Train model with many epochs and return loss history."""
    if not tasks:
        return 0.0, []

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Create target vectors
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
        return 0.0, []

    losses = []
    for epoch in range(epochs):
        random.shuffle(training_data)
        epoch_losses = []

        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]

            batch_preds = []
            batch_targets = []

            for task, target in batch:
                # Create task-like object for encoding
                class TaskProxy:
                    def __init__(self, t):
                        self.name = t.name
                        self.examples = t.examples

                try:
                    τ = model.encode_task_batched(TaskProxy(task))
                    pred = model.primitive_head(τ.unsqueeze(0))
                    batch_preds.append(pred.squeeze(0))
                    batch_targets.append(target)
                except Exception as e:
                    continue

            if not batch_preds:
                continue

            preds = torch.stack(batch_preds)
            targets = torch.stack(batch_targets)

            # BCE loss
            loss = torch.nn.functional.binary_cross_entropy(
                torch.sigmoid(preds), targets
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)
            if epoch % log_every == 0 or epoch == epochs - 1:
                logger.info(f"    Epoch {epoch+1}/{epochs}: loss = {avg_loss:.4f}")

    return losses[-1] if losses else 0.0, losses


def cross_validate(
    model_factory,
    tasks: List[RevelationTask],
    test_tasks: List[RevelationTask],
    n_folds: int = 5,
    epochs: int = 200
) -> Dict[str, Any]:
    """
    K-fold cross-validation for model evaluation.

    Args:
        model_factory: Function that creates a new model instance
        tasks: Training tasks to split into folds
        test_tasks: Separate test set (always evaluated)
        n_folds: Number of folds
        epochs: Training epochs per fold

    Returns:
        Dictionary with per-fold and aggregate metrics
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(tasks)):
        logger.info(f"\n  === Fold {fold_idx + 1}/{n_folds} ===")

        train_tasks = [tasks[i] for i in train_idx]
        val_tasks = [tasks[i] for i in val_idx]

        # Create fresh model for this fold
        model = model_factory()

        # Train
        final_loss, loss_history = train_model_with_epochs(
            model, train_tasks, epochs=epochs
        )

        # Evaluate on validation fold
        val_metrics = evaluate_model(model, val_tasks, prefix='val')

        # Evaluate on test set
        test_metrics = evaluate_model(model, test_tasks, prefix='test')

        fold_results.append({
            'fold': fold_idx + 1,
            'train_size': len(train_tasks),
            'val_size': len(val_tasks),
            'final_loss': final_loss,
            'val_metrics': val_metrics,
            'test_metrics': test_metrics
        })

        logger.info(f"    Val R@5: {val_metrics['val_recall@5']:.3f}, Test R@5: {test_metrics['test_recall@5']:.3f}")

    # Aggregate metrics across folds
    aggregate = {}
    metric_keys = list(fold_results[0]['test_metrics'].keys())

    for key in metric_keys:
        values = [fr['test_metrics'][key] for fr in fold_results]
        aggregate[f'mean_{key}'] = np.mean(values)
        aggregate[f'std_{key}'] = np.std(values)

    return {
        'fold_results': fold_results,
        'aggregate': aggregate
    }


# ============================================================================
# Evaluation
# ============================================================================

def compute_metrics(predictions: np.ndarray, solution_prims: Set[str], prim_names: List[str]) -> Dict[str, float]:
    """Compute prediction quality metrics."""
    sorted_idx = np.argsort(predictions)[::-1]
    sorted_names = [prim_names[i] for i in sorted_idx]

    n_sol = len(solution_prims)
    n_prims = len(prim_names)

    # Find ranks
    ranks = []
    for i, name in enumerate(sorted_names):
        if name in solution_prims:
            ranks.append(i + 1)

    # Recall@k
    def recall_k(k):
        if n_sol == 0:
            return 1.0
        return len(solution_prims & set(sorted_names[:k])) / n_sol

    # MRR
    mrr = np.mean([1.0/r for r in ranks]) if ranks else 0.0

    # ProbRatio
    sol_idx = [prim_names.index(p) for p in solution_prims if p in prim_names]
    other_idx = [i for i in range(n_prims) if prim_names[i] not in solution_prims]

    if sol_idx and other_idx:
        mean_sol_prob = float(np.mean(predictions[sol_idx]))
        mean_other_prob = float(np.mean(predictions[other_idx]))
        prob_ratio = mean_sol_prob / mean_other_prob if mean_other_prob > 0 else 0.0
    else:
        mean_sol_prob = 0.0
        mean_other_prob = 0.0
        prob_ratio = 1.0

    # LogBoost
    uniform_p = 1.0 / n_prims
    if sol_idx:
        sol_probs = np.clip(predictions[sol_idx], 1e-10, 1.0)
        log_boost = float(np.mean(np.log(sol_probs) - np.log(uniform_p)))
    else:
        log_boost = 0.0

    return {
        'recall@5': float(recall_k(5)),
        'recall@10': float(recall_k(10)),
        'recall@15': float(recall_k(15)),
        'mrr': float(mrr),
        'prob_ratio': float(prob_ratio),
        'log_boost': float(log_boost)
    }


def evaluate_model(model: ContrastiveRecognitionModel, tasks: List[RevelationTask], prefix: str = '') -> Dict[str, float]:
    """Evaluate model on a set of tasks."""
    model.eval()
    all_metrics = []

    class TaskProxy:
        def __init__(self, t):
            self.name = t.name
            self.examples = t.examples

    with torch.no_grad():
        for task in tasks:
            if len(task.examples) < 10:
                continue

            try:
                pred = model.predict_primitives(TaskProxy(task)).cpu().numpy()
                metrics = compute_metrics(pred, task.primitives_used, model.primitive_names)
                all_metrics.append(metrics)
            except:
                continue

    if not all_metrics:
        return {f'{prefix}_recall@5': 0.0, f'{prefix}_recall@10': 0.0,
                f'{prefix}_mrr': 0.0, f'{prefix}_prob_ratio': 1.0, f'{prefix}_log_boost': 0.0}

    # Aggregate
    result = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        result[f'{prefix}_{key}' if prefix else key] = float(np.mean(values))

    return result


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    print("=" * 80)
    print("IMPROVED RECOGNITION MODEL TRAINING")
    print("=" * 80)
    print("\nImprovements:")
    print("  - Deeper synthetic programs (max_depth=12 vs 8)")
    print("  - More training epochs (200 vs 50)")
    print("  - 5-fold cross-validation")
    print("  - Diversity sampling by primitive set")
    print()

    # Load grammar
    grammar = build_lean_grammar()
    print(f"Grammar: {len(grammar.productions)} primitives")

    # Model configurations
    model_configs = [
        ("baseline", {"normalize_embeddings": False, "encoding_mode": "standard"}),
        ("layernorm_s20", {"normalize_embeddings": True, "embedding_scale": 20.0, "encoding_mode": "standard"}),
        ("triple_s20", {"normalize_embeddings": True, "embedding_scale": 20.0, "encoding_mode": "triple_contrast", "n_random_hands": 10}),
    ]

    # ========================================================================
    # PIPELINE 1: Pre-training → Catalogue
    # ========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE 1: Pre-training Rules → Catalogue Rules")
    print("=" * 80)

    # Load tasks
    pretraining_rules = get_all_pretraining_rules()
    catalogue_rules = create_all_rules()

    pretraining_tasks = [create_task_from_pretraining_rule(r, grammar) for r in pretraining_rules]
    pretraining_tasks = [t for t in pretraining_tasks if len(t.examples) >= 10 and len(t.primitives_used) > 0]

    catalogue_tasks = [create_task_from_catalogue_rule(r) for r in catalogue_rules]
    catalogue_tasks = [t for t in catalogue_tasks if len(t.examples) >= 10 and len(t.primitives_used) > 0]

    print(f"\nPre-training tasks: {len(pretraining_tasks)}")
    print(f"Catalogue tasks (test): {len(catalogue_tasks)}")

    pipeline1_results = {}

    for model_name, config in model_configs:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        def model_factory():
            return ContrastiveRecognitionModel(
                grammar=grammar,
                card_hidden=128,
                card_out=64,
                pred_hidden=128,
                **config
            )

        # Cross-validation
        cv_results = cross_validate(
            model_factory=model_factory,
            tasks=pretraining_tasks,
            test_tasks=catalogue_tasks,
            n_folds=5,
            epochs=200
        )

        pipeline1_results[model_name] = cv_results['aggregate']

        print(f"\nAggregate Results:")
        print(f"  Test R@5:  {cv_results['aggregate']['mean_test_recall@5']:.3f} ± {cv_results['aggregate']['std_test_recall@5']:.3f}")
        print(f"  Test R@10: {cv_results['aggregate']['mean_test_recall@10']:.3f} ± {cv_results['aggregate']['std_test_recall@10']:.3f}")
        print(f"  Test MRR:  {cv_results['aggregate']['mean_test_mrr']:.3f} ± {cv_results['aggregate']['std_test_mrr']:.3f}")

    # ========================================================================
    # PIPELINE 2: Synthetic → Both
    # ========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE 2: Synthetic Rules → Catalogue + Pre-training")
    print("=" * 80)

    # Generate improved synthetic rules
    synthetic_tasks = generate_synthetic_rules_improved(
        grammar,
        n_rules=150,
        max_depth=12,
        min_depth=3,
        timeout_seconds=180.0
    )

    # Combined test set
    combined_test = catalogue_tasks + pretraining_tasks

    print(f"\nSynthetic tasks: {len(synthetic_tasks)}")
    print(f"Combined test: {len(combined_test)}")

    pipeline2_results = {}

    for model_name, config in model_configs:
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        def model_factory():
            return ContrastiveRecognitionModel(
                grammar=grammar,
                card_hidden=128,
                card_out=64,
                pred_hidden=128,
                **config
            )

        if len(synthetic_tasks) >= 10:
            cv_results = cross_validate(
                model_factory=model_factory,
                tasks=synthetic_tasks,
                test_tasks=combined_test,
                n_folds=min(5, len(synthetic_tasks) // 2),
                epochs=200
            )

            pipeline2_results[model_name] = cv_results['aggregate']

            print(f"\nAggregate Results:")
            print(f"  Test R@5:  {cv_results['aggregate']['mean_test_recall@5']:.3f} ± {cv_results['aggregate']['std_test_recall@5']:.3f}")
            print(f"  Test R@10: {cv_results['aggregate']['mean_test_recall@10']:.3f} ± {cv_results['aggregate']['std_test_recall@10']:.3f}")
            print(f"  Test MRR:  {cv_results['aggregate']['mean_test_mrr']:.3f} ± {cv_results['aggregate']['std_test_mrr']:.3f}")
        else:
            print("  Not enough synthetic tasks for cross-validation")

    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print("\nPIPELINE 1: Pre-training → Catalogue")
    print(f"{'Model':<20} {'R@5':<12} {'R@10':<12} {'MRR':<12}")
    print("-" * 60)
    for name, results in pipeline1_results.items():
        r5 = f"{results['mean_test_recall@5']:.3f}±{results['std_test_recall@5']:.2f}"
        r10 = f"{results['mean_test_recall@10']:.3f}±{results['std_test_recall@10']:.2f}"
        mrr = f"{results['mean_test_mrr']:.3f}±{results['std_test_mrr']:.2f}"
        print(f"{name:<20} {r5:<12} {r10:<12} {mrr:<12}")

    if pipeline2_results:
        print("\nPIPELINE 2: Synthetic → Combined")
        print(f"{'Model':<20} {'R@5':<12} {'R@10':<12} {'MRR':<12}")
        print("-" * 60)
        for name, results in pipeline2_results.items():
            r5 = f"{results['mean_test_recall@5']:.3f}±{results['std_test_recall@5']:.2f}"
            r10 = f"{results['mean_test_recall@10']:.3f}±{results['std_test_recall@10']:.2f}"
            mrr = f"{results['mean_test_mrr']:.3f}±{results['std_test_mrr']:.2f}"
            print(f"{name:<20} {r5:<12} {r10:<12} {mrr:<12}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(f"results/recognition_improved_{timestamp}.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, 'w') as f:
        json.dump({
            'pipeline1': pipeline1_results,
            'pipeline2': pipeline2_results,
            'config': {
                'epochs': 200,
                'n_folds': 5,
                'synthetic_max_depth': 12,
                'synthetic_min_depth': 3
            }
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
