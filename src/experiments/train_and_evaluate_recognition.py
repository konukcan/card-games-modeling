#!/usr/bin/env python3
"""
Train and Evaluate Recognition Model with Rule Revelation

Two training pipelines:
1. Train on pre-training rules (44) → Test on catalogue rules (45)
2. Train on synthetic rules (random programs) → Test on both rule sets

Metrics: Recall@k, MRR, ProbRatio, LogBoost
"""

import sys
import json
import re
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Tuple, Any, Optional
from dataclasses import dataclass, field

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


def create_task_from_catalogue_rule(rule: Rule, n_examples: int = 50, seed: int = 42) -> RevelationTask:
    """Create a revelation task from a catalogue rule."""
    rng = random.Random(seed + hash(rule.id) % 10000)

    examples = []
    pos_count = 0
    neg_count = 0
    attempts = 0
    max_attempts = n_examples * 20

    while (pos_count < n_examples // 2 or neg_count < n_examples // 2) and attempts < max_attempts:
        hand = sample_hand(size=5)
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
# Synthetic Rule Generation
# ============================================================================

def generate_synthetic_rules(grammar: Grammar, n_rules: int = 100, seed: int = 42) -> List[RevelationTask]:
    """Generate synthetic rules by enumerating random programs."""
    rng = random.Random(seed)
    tasks = []

    logger.info(f"Generating {n_rules} synthetic rules from grammar...")

    # Enumerate programs of type HAND -> BOOL
    request_type = arrow(HAND, BOOL)
    enumerator = TopDownEnumerator(grammar, max_depth=8, max_programs=n_rules * 5)

    programs = []
    for prog, lp in enumerator.enumerate(request_type, timeout_seconds=30.0):
        programs.append((prog, lp))
        if len(programs) >= n_rules * 3:
            break

    logger.info(f"  Enumerated {len(programs)} candidate programs")

    # Sample n_rules programs with diversity
    if len(programs) > n_rules:
        selected = rng.sample(programs, n_rules)
    else:
        selected = programs

    for i, (prog, lp) in enumerate(selected):
        # Extract primitives from program
        primitives = extract_primitives(prog)

        if not primitives or primitives == {'true'} or primitives == {'false'}:
            continue

        # Generate examples
        examples = []
        try:
            fn = prog.evaluate([])
            pos_count = 0
            neg_count = 0

            for _ in range(500):  # Try up to 500 hands
                hand = sample_hand(size=5)
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
                    source='synthetic'
                ))
        except:
            continue

    logger.info(f"  Created {len(tasks)} valid synthetic tasks")
    return tasks


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


# ============================================================================
# Training
# ============================================================================

def train_model(
    model: ContrastiveRecognitionModel,
    tasks: List[RevelationTask],
    epochs: int = 50,
    batch_size: int = 8
) -> float:
    """Train model on revelation tasks."""
    if not tasks:
        return 0.0

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

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
        return 0.0

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
            if epoch % 10 == 0 or epoch == epochs - 1:
                logger.info(f"    Epoch {epoch+1}/{epochs}: loss = {avg_loss:.4f}")

    return losses[-1] if losses else 0.0


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
    uniform_p = 1.0 / n_prims
    sol_idx = [prim_names.index(p) for p in solution_prims if p in prim_names]
    other_idx = [i for i in range(n_prims) if prim_names[i] not in solution_prims]

    if sol_idx and other_idx:
        mean_sol_prob = np.mean(predictions[sol_idx])
        mean_other_prob = np.mean(predictions[other_idx])
        prob_ratio = mean_sol_prob / mean_other_prob if mean_other_prob > 0 else 0.0
    else:
        mean_sol_prob = 0.0
        mean_other_prob = 0.0
        prob_ratio = 1.0

    # LogBoost
    if sol_idx:
        sol_probs = np.clip(predictions[sol_idx], 1e-10, 1.0)
        log_boost = np.mean(np.log(sol_probs) - np.log(uniform_p))
    else:
        log_boost = 0.0

    return {
        'recall@5': recall_k(5),
        'recall@10': recall_k(10),
        'mrr': mrr,
        'prob_ratio': prob_ratio,
        'log_boost': log_boost,
        'top5': sorted_names[:5]
    }


def evaluate_model(
    model: ContrastiveRecognitionModel,
    tasks: List[RevelationTask]
) -> Dict[str, float]:
    """Evaluate model on a set of tasks."""
    model.eval()

    all_metrics = []

    with torch.no_grad():
        for task in tasks:
            if len(task.examples) < 10 or not task.primitives_used:
                continue

            class TaskProxy:
                def __init__(self, t):
                    self.name = t.name
                    self.examples = t.examples

            try:
                pred = model.predict_primitives(TaskProxy(task)).cpu().numpy()
                metrics = compute_metrics(pred, task.primitives_used, model.primitive_names)
                all_metrics.append(metrics)
            except:
                continue

    if not all_metrics:
        return {'recall@5': 0, 'recall@10': 0, 'mrr': 0, 'prob_ratio': 1, 'log_boost': 0}

    return {
        'recall@5': np.mean([m['recall@5'] for m in all_metrics]),
        'recall@10': np.mean([m['recall@10'] for m in all_metrics]),
        'mrr': np.mean([m['mrr'] for m in all_metrics]),
        'prob_ratio': np.mean([m['prob_ratio'] for m in all_metrics]),
        'log_boost': np.mean([m['log_boost'] for m in all_metrics]),
        'n_tasks': len(all_metrics)
    }


# ============================================================================
# Main Experiment
# ============================================================================

def main():
    print("=" * 80)
    print("RECOGNITION MODEL TRAINING PIPELINE EVALUATION")
    print("=" * 80)

    # Setup
    grammar = build_lean_grammar()
    logger.info(f"\nLoaded grammar with {len(grammar.productions)} primitives")

    # Create tasks from rules
    logger.info("\nCreating tasks from rules...")

    catalogue_rules = create_all_rules()
    pretraining_rules = get_all_pretraining_rules()

    catalogue_tasks = [create_task_from_catalogue_rule(r) for r in catalogue_rules]
    pretraining_tasks = [create_task_from_pretraining_rule(r, grammar) for r in pretraining_rules]

    catalogue_tasks = [t for t in catalogue_tasks if len(t.examples) >= 20 and t.primitives_used]
    pretraining_tasks = [t for t in pretraining_tasks if len(t.examples) >= 20 and t.primitives_used]

    logger.info(f"  Catalogue tasks: {len(catalogue_tasks)}")
    logger.info(f"  Pre-training tasks: {len(pretraining_tasks)}")

    # Generate synthetic rules
    logger.info("\nGenerating synthetic rules...")
    synthetic_tasks = generate_synthetic_rules(grammar, n_rules=100, seed=42)
    logger.info(f"  Synthetic tasks: {len(synthetic_tasks)}")

    # Results storage
    all_results = {}

    # ========================================================================
    # PIPELINE 1: Pre-training rules → Catalogue rules
    # ========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE 1: Train on Pre-training Rules → Test on Catalogue Rules")
    print("=" * 80)

    for model_name, config in [
        ("baseline", {"normalize_embeddings": False}),
        ("layernorm_s20", {"normalize_embeddings": True, "embedding_scale": 20.0}),
    ]:
        logger.info(f"\n--- {model_name} ---")

        model = ContrastiveRecognitionModel(
            grammar=grammar,
            card_hidden=128,
            card_out=64,
            pred_hidden=128,
            encoding_mode='standard',
            **config
        )

        # Evaluate untrained
        untrained_metrics = evaluate_model(model, catalogue_tasks)
        logger.info(f"  Untrained: R@5={untrained_metrics['recall@5']:.3f}, MRR={untrained_metrics['mrr']:.3f}")

        # Train on pre-training rules
        logger.info(f"  Training on {len(pretraining_tasks)} pre-training tasks...")
        train_model(model, pretraining_tasks, epochs=50, batch_size=8)

        # Evaluate on catalogue rules
        trained_metrics = evaluate_model(model, catalogue_tasks)
        logger.info(f"  Trained: R@5={trained_metrics['recall@5']:.3f}, MRR={trained_metrics['mrr']:.3f}, ProbRatio={trained_metrics['prob_ratio']:.2f}")

        all_results[f"pipeline1_{model_name}"] = {
            'pipeline': 'pretraining→catalogue',
            'model': model_name,
            'untrained': untrained_metrics,
            'trained': trained_metrics
        }

    # ========================================================================
    # PIPELINE 2: Synthetic rules → Both rule sets
    # ========================================================================
    print("\n" + "=" * 80)
    print("PIPELINE 2: Train on Synthetic Rules → Test on Both")
    print("=" * 80)

    for model_name, config in [
        ("baseline", {"normalize_embeddings": False}),
        ("layernorm_s20", {"normalize_embeddings": True, "embedding_scale": 20.0}),
    ]:
        logger.info(f"\n--- {model_name} ---")

        model = ContrastiveRecognitionModel(
            grammar=grammar,
            card_hidden=128,
            card_out=64,
            pred_hidden=128,
            encoding_mode='standard',
            **config
        )

        # Train on synthetic rules
        logger.info(f"  Training on {len(synthetic_tasks)} synthetic tasks...")
        train_model(model, synthetic_tasks, epochs=50, batch_size=8)

        # Evaluate on both
        catalogue_metrics = evaluate_model(model, catalogue_tasks)
        pretraining_metrics = evaluate_model(model, pretraining_tasks)

        logger.info(f"  On catalogue: R@5={catalogue_metrics['recall@5']:.3f}, MRR={catalogue_metrics['mrr']:.3f}")
        logger.info(f"  On pretraining: R@5={pretraining_metrics['recall@5']:.3f}, MRR={pretraining_metrics['mrr']:.3f}")

        all_results[f"pipeline2_{model_name}"] = {
            'pipeline': 'synthetic→both',
            'model': model_name,
            'on_catalogue': catalogue_metrics,
            'on_pretraining': pretraining_metrics
        }

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n--- Pipeline 1: Pre-training → Catalogue ---")
    print(f"{'Model':<20} {'R@5 (before)':<15} {'R@5 (after)':<15} {'MRR (after)':<15} {'ProbRatio'}")
    print("-" * 80)
    for key, r in all_results.items():
        if 'pipeline1' in key:
            print(f"{r['model']:<20} {r['untrained']['recall@5']:<15.3f} {r['trained']['recall@5']:<15.3f} "
                  f"{r['trained']['mrr']:<15.3f} {r['trained']['prob_ratio']:.2f}")

    print("\n--- Pipeline 2: Synthetic → Both ---")
    print(f"{'Model':<20} {'R@5 (catalogue)':<18} {'R@5 (pretrain)':<18} {'MRR (catalogue)':<18}")
    print("-" * 80)
    for key, r in all_results.items():
        if 'pipeline2' in key:
            print(f"{r['model']:<20} {r['on_catalogue']['recall@5']:<18.3f} {r['on_pretraining']['recall@5']:<18.3f} "
                  f"{r['on_catalogue']['mrr']:<18.3f}")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(f"results/training_pipeline_eval_{timestamp}.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy values for JSON
    def to_json(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: to_json(v) for k, v in obj.items()}
        return obj

    with open(results_file, 'w') as f:
        json.dump(to_json(all_results), f, indent=2)

    logger.info(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
