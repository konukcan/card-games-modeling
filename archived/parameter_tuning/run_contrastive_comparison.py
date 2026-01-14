#!/usr/bin/env python3
"""
Run comparison experiments with fixed ContrastiveRecognitionModel.

This script compares:
1. Baseline (no normalization) - CONTROL
2. Standard + LayerNorm - THE FIX
3. TripleContrast + LayerNorm - ENRICHED REPRESENTATION

Each configuration is trained and then evaluated on task-solving with
recognition-guided enumeration.
"""

import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.dreamcoder_original import create_tasks_from_rules
from dreamcoder_core.grammar import Grammar
from dreamcoder_core.enumeration import TopDownEnumerator
from rules.catalogue import create_all_rules

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_model_config(name: str) -> Dict[str, Any]:
    """Create model configuration by name."""
    configs = {
        'baseline': {
            'normalize_embeddings': False,
            'encoding_mode': 'standard',
            'description': 'Baseline (no normalization)'
        },
        'standard_layernorm': {
            'normalize_embeddings': True,
            'embedding_scale': 20.0,
            'encoding_mode': 'standard',
            'description': 'Standard + LayerNorm (THE FIX)'
        },
        'triple_contrast': {
            'normalize_embeddings': True,
            'embedding_scale': 20.0,
            'encoding_mode': 'triple_contrast',
            'n_random_hands': 10,
            'description': 'TripleContrast + LayerNorm'
        },
        'random_contrast': {
            'normalize_embeddings': True,
            'embedding_scale': 20.0,
            'encoding_mode': 'random_contrast',
            'n_random_hands': 10,
            'lambda_random': 0.5,
            'description': 'RandomContrast + LayerNorm'
        }
    }
    return configs.get(name, configs['standard_layernorm'])


def solve_task(task, grammar: Grammar, timeout: float = 2.0, max_programs: int = 10000):
    """Attempt to solve a single task using enumeration."""
    enumerator = TopDownEnumerator(grammar, max_depth=15, max_programs=max_programs)
    start_time = time.time()

    for program, log_prob in enumerator.enumerate(task.request_type, timeout_seconds=timeout):
        if time.time() - start_time > timeout:
            break

        try:
            fn = program.evaluate([])
            correct = 0
            for inp, expected in task.examples:
                result = fn(inp)
                if result == expected:
                    correct += 1

            if correct == len(task.examples):
                return program, log_prob
        except Exception:
            continue

    return None, None


def pretrain_recognition_model(
    model: ContrastiveRecognitionModel,
    tasks: List,
    grammar: Grammar,
    n_pretraining_tasks: int = 20,
    pretraining_epochs: int = 20,
    enum_timeout: float = 2.0
) -> Dict[str, Any]:
    """
    Pre-train recognition model by solving easy tasks and learning from them.

    Returns training statistics.
    """
    logger.info(f"Pre-training on {n_pretraining_tasks} tasks...")

    # Sort tasks by expected difficulty (use grammar entropy as proxy)
    # For now, just take first n tasks
    training_tasks = tasks[:n_pretraining_tasks]

    # Solve tasks with uniform grammar
    frontiers = {}
    solved_count = 0

    for task in training_tasks:
        program, log_prob = solve_task(task, grammar, timeout=enum_timeout)

        if program is not None:
            frontiers[task.name] = type('Frontier', (), {
                'solved': True,
                'entries': [type('Entry', (), {
                    'program': program,
                    'log_likelihood': 0.0
                })()]
            })()
            solved_count += 1

    logger.info(f"  Solved {solved_count}/{len(training_tasks)} pre-training tasks")

    # Train recognition model on solved tasks
    if solved_count > 0:
        loss = model.train_on_frontiers(
            tasks=training_tasks,
            frontiers=frontiers,
            epochs=pretraining_epochs,
            batch_size=8
        )
        logger.info(f"  Final training loss: {loss:.4f}")
    else:
        loss = 0.0

    return {
        'n_tasks': len(training_tasks),
        'n_solved': solved_count,
        'final_loss': loss
    }


def evaluate_predictions(
    model: ContrastiveRecognitionModel,
    tasks: List,
    n_eval_tasks: int = 10
) -> Dict[str, Any]:
    """Evaluate prediction diversity and quality."""
    eval_tasks = tasks[:n_eval_tasks]

    predictions = []
    embeddings = []

    model.eval()
    with torch.no_grad():
        for task in eval_tasks:
            pred = model.predict_primitives(task).cpu().numpy()
            emb = model.encode_task_batched(task).cpu().numpy()
            predictions.append(pred)
            embeddings.append(emb)

    predictions_np = np.array(predictions)
    embeddings_np = np.array(embeddings)

    # Metrics
    top5_sets = [frozenset(np.argsort(pred)[::-1][:5]) for pred in predictions_np]
    unique_top5 = len(set(top5_sets))

    pred_spreads = np.max(predictions_np, axis=1) - np.min(predictions_np, axis=1)
    emb_norms = np.linalg.norm(embeddings_np, axis=1)

    return {
        'unique_top5': unique_top5,
        'total_tasks': n_eval_tasks,
        'mean_pred_spread': float(np.mean(pred_spreads)),
        'mean_emb_norm': float(np.mean(emb_norms))
    }


def evaluate_task_solving(
    model: ContrastiveRecognitionModel,
    tasks: List,
    grammar: Grammar,
    timeout: float = 5.0,
    max_tasks: int = 20
) -> Dict[str, Any]:
    """Evaluate task-solving with recognition-guided enumeration."""
    eval_tasks = tasks[:max_tasks]

    solved = 0
    total_time = 0.0
    results = []

    for task in eval_tasks:
        # Get task-specific grammar weights from recognition model
        guided_grammar = model.predict_grammar_weights(task)

        start = time.time()
        program, _ = solve_task(task, guided_grammar, timeout=timeout)
        elapsed = time.time() - start
        total_time += elapsed

        task_solved = program is not None
        if task_solved:
            solved += 1

        results.append({
            'task': task.name,
            'solved': task_solved,
            'time': elapsed
        })

    return {
        'n_tasks': len(eval_tasks),
        'n_solved': solved,
        'solve_rate': solved / len(eval_tasks) if eval_tasks else 0.0,
        'total_time': total_time,
        'results': results
    }


def run_single_config(
    config_name: str,
    grammar: Grammar,
    tasks: List,
    pretraining_tasks: int = 20,
    pretraining_epochs: int = 20,
    eval_timeout: float = 5.0
) -> Dict[str, Any]:
    """Run a single configuration."""
    config = create_model_config(config_name)
    logger.info(f"\n{'='*60}")
    logger.info(f"Running: {config['description']}")
    logger.info(f"{'='*60}")

    # Create model
    model_kwargs = {
        'grammar': grammar,
        'card_hidden': 128,
        'card_out': 64,
        'pred_hidden': 128,
    }
    model_kwargs.update({k: v for k, v in config.items() if k != 'description'})

    model = ContrastiveRecognitionModel(**model_kwargs)

    # Pre-train
    pretrain_stats = pretrain_recognition_model(
        model=model,
        tasks=tasks,
        grammar=grammar,
        n_pretraining_tasks=pretraining_tasks,
        pretraining_epochs=pretraining_epochs
    )

    # Evaluate predictions
    pred_stats = evaluate_predictions(model, tasks)
    logger.info(f"  Prediction diversity: {pred_stats['unique_top5']}/10 unique top-5")
    logger.info(f"  Mean spread: {pred_stats['mean_pred_spread']:.4f}")

    # Evaluate task solving
    solve_stats = evaluate_task_solving(
        model=model,
        tasks=tasks,
        grammar=grammar,
        timeout=eval_timeout
    )
    logger.info(f"  Task solving: {solve_stats['n_solved']}/{solve_stats['n_tasks']} ({solve_stats['solve_rate']*100:.1f}%)")

    return {
        'config_name': config_name,
        'description': config['description'],
        'config': config,
        'pretrain': pretrain_stats,
        'predictions': pred_stats,
        'solving': solve_stats
    }


def main():
    print("=" * 70)
    print("CONTRASTIVE RECOGNITION MODEL - COMPARISON EXPERIMENT")
    print("=" * 70)

    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"results/contrastive_comparison_{timestamp}")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load grammar and tasks
    logger.info("\nLoading grammar and tasks...")
    grammar = build_lean_grammar()
    rules = create_all_rules()
    tasks = create_tasks_from_rules(rules, n_examples=20, seed=42)
    logger.info(f"  Grammar: {len(grammar.productions)} primitives")
    logger.info(f"  Tasks: {len(tasks)}")

    # Configurations to test
    configs = ['baseline', 'standard_layernorm', 'triple_contrast']

    # Run experiments
    all_results = []
    for config_name in configs:
        try:
            result = run_single_config(
                config_name=config_name,
                grammar=grammar,
                tasks=tasks,
                pretraining_tasks=15,
                pretraining_epochs=15,
                eval_timeout=3.0
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"Error in {config_name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Config':<25} {'Unique Top-5':<15} {'Spread':<12} {'Solve Rate':<12}")
    print("-" * 70)

    for r in all_results:
        print(f"{r['config_name']:<25} {r['predictions']['unique_top5']}/10{'':<7} "
              f"{r['predictions']['mean_pred_spread']:<12.4f} "
              f"{r['solving']['solve_rate']*100:<10.1f}%")

    # Save results
    results_file = results_dir / "comparison_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy values for JSON serialization
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        json.dump(all_results, f, indent=2, default=convert)

    logger.info(f"\nResults saved to: {results_file}")

    # Final verdict
    print("\n" + "=" * 70)
    baseline = next((r for r in all_results if r['config_name'] == 'baseline'), None)
    fixed = next((r for r in all_results if r['config_name'] == 'standard_layernorm'), None)

    if baseline and fixed:
        if fixed['predictions']['unique_top5'] > baseline['predictions']['unique_top5']:
            print("✅ LAYERNORM FIX IMPROVES PREDICTION DIVERSITY!")
        if fixed['solving']['solve_rate'] >= baseline['solving']['solve_rate']:
            print("✅ TASK SOLVING MAINTAINED OR IMPROVED!")
    print("=" * 70)


if __name__ == "__main__":
    main()
