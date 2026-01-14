#!/usr/bin/env python3
"""
Detailed evaluation of primitive prediction quality.

This script:
1. Solves tasks with a generous timeout to get more ground truth
2. Shows detailed per-task prediction analysis
3. Compares predictions to solution primitives
4. Measures actual enumeration speedup
"""

import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set, Optional, Tuple

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.dreamcoder_original import create_tasks_from_rules
from dreamcoder_core.grammar import Grammar
from dreamcoder_core.enumeration import TopDownEnumerator
from dreamcoder_core.program import Program
from rules.catalogue import create_all_rules

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_primitives(program: Program) -> Set[str]:
    """Extract all primitive names from a program."""
    prims = set()
    def traverse(p):
        ptype = type(p).__name__
        if ptype == 'Primitive' and hasattr(p, 'name'):
            prims.add(p.name)
        # Application nodes use 'f' and 'x' attributes
        if hasattr(p, 'f'):
            traverse(p.f)
        if hasattr(p, 'x'):
            traverse(p.x)
        # Abstraction uses 'body'
        if hasattr(p, 'body'):
            traverse(p.body)
    traverse(program)
    return prims


def solve_task(task, grammar: Grammar, timeout: float = 10.0) -> Tuple[Optional[Program], float, int]:
    """Solve a task and return (solution, time, programs_tried)."""
    enumerator = TopDownEnumerator(grammar, max_depth=15, max_programs=100000)
    start = time.time()
    tried = 0

    for program, _ in enumerator.enumerate(task.request_type, timeout_seconds=timeout):
        tried += 1
        if time.time() - start > timeout:
            break
        try:
            fn = program.evaluate([])
            if all(fn(inp) == exp for inp, exp in task.examples):
                return program, time.time() - start, tried
        except:
            continue

    return None, time.time() - start, tried


def compute_metrics(predictions: np.ndarray, solution_prims: Set[str], prim_names: List[str]) -> Dict[str, float]:
    """Compute prediction quality metrics."""
    sorted_idx = np.argsort(predictions)[::-1]
    sorted_names = [prim_names[i] for i in sorted_idx]

    # Find ranks of solution primitives
    ranks = []
    for i, name in enumerate(sorted_names):
        if name in solution_prims:
            ranks.append(i + 1)  # 1-indexed

    n_sol = len(solution_prims)
    n_prims = len(prim_names)

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

    # Log-likelihood boost
    uniform_p = 1.0 / n_prims
    sol_idx = [prim_names.index(p) for p in solution_prims if p in prim_names]
    if sol_idx:
        sol_probs = np.clip(predictions[sol_idx], 1e-10, 1.0)
        log_boost = np.mean(np.log(sol_probs) - np.log(uniform_p))
    else:
        log_boost = 0.0

    return {
        'recall@5': recall_k(5),
        'recall@10': recall_k(10),
        'precision@5': precision_k(5),
        'mrr': mrr,
        'mean_rank': mean_rank,
        'log_boost': log_boost,
        'top5': sorted_names[:5],
        'ranks': ranks
    }


def main():
    print("=" * 80)
    print("DETAILED PRIMITIVE PREDICTION QUALITY ANALYSIS")
    print("=" * 80)

    # Load
    grammar = build_lean_grammar()
    rules = create_all_rules()
    tasks = create_tasks_from_rules(rules, n_examples=20, seed=42)
    print(f"\nLoaded {len(tasks)} tasks, {len(grammar.productions)} primitives")

    # First, solve tasks with uniform grammar (generous timeout)
    print("\n" + "=" * 80)
    print("PHASE 1: SOLVING TASKS WITH UNIFORM GRAMMAR")
    print("=" * 80)

    solutions = {}
    for task in tasks[:30]:  # Limit for time
        prog, t, n = solve_task(task, grammar, timeout=10.0)
        if prog:
            prims = extract_primitives(prog)
            solutions[task.name] = {
                'program': prog,
                'str': str(prog),
                'primitives': prims,
                'time': t,
                'n_programs': n
            }
            print(f"  ✓ {task.name}: {len(prims)} prims, {t:.2f}s, {n} programs")
        else:
            print(f"  ✗ {task.name}: timeout")

    print(f"\nSolved {len(solutions)}/{30} tasks")

    if not solutions:
        print("No tasks solved - cannot evaluate predictions")
        return

    # Create models to evaluate
    model_configs = [
        ("baseline", {"normalize_embeddings": False, "encoding_mode": "standard"}),
        ("layernorm_s10", {"normalize_embeddings": True, "embedding_scale": 10.0, "encoding_mode": "standard"}),
        ("layernorm_s20", {"normalize_embeddings": True, "embedding_scale": 20.0, "encoding_mode": "standard"}),
        ("triple_s20", {"normalize_embeddings": True, "embedding_scale": 20.0, "encoding_mode": "triple_contrast", "n_random_hands": 10}),
    ]

    all_results = {}

    for name, config in model_configs:
        print("\n" + "=" * 80)
        print(f"EVALUATING: {name}")
        print("=" * 80)

        # Create and train model
        model = ContrastiveRecognitionModel(
            grammar=grammar,
            card_hidden=128,
            card_out=64,
            pred_hidden=128,
            **config
        )

        # Pre-train on solved tasks
        frontiers = {}
        for task_name, sol in solutions.items():
            frontiers[task_name] = type('F', (), {
                'solved': True,
                'entries': [type('E', (), {'program': sol['program'], 'log_likelihood': 0.0})()]
            })()

        training_tasks = [t for t in tasks if t.name in solutions]
        if training_tasks:
            loss = model.train_on_frontiers(
                tasks=training_tasks,
                frontiers=frontiers,
                epochs=50,  # More training
                batch_size=8
            )
            print(f"  Training loss: {loss:.4f}")

        # Evaluate predictions
        model.eval()
        task_results = []

        print(f"\n  Per-task analysis:")
        print(f"  {'Task':<35} {'Sol Prims':<12} {'R@5':<8} {'MRR':<8} {'Top-5 Predictions'}")
        print("  " + "-" * 100)

        with torch.no_grad():
            for task_name, sol in solutions.items():
                task = next(t for t in tasks if t.name == task_name)
                pred = model.predict_primitives(task).cpu().numpy()
                metrics = compute_metrics(pred, sol['primitives'], model.primitive_names)
                metrics['task'] = task_name
                metrics['solution_prims'] = list(sol['primitives'])
                task_results.append(metrics)

                sol_prims_str = ','.join(sorted(sol['primitives']))[:20]
                top5_str = ','.join(metrics['top5'])[:40]
                print(f"  {task_name:<35} {sol_prims_str:<12} {metrics['recall@5']:<8.2f} {metrics['mrr']:<8.3f} {top5_str}")

        # Aggregate metrics
        mean_recall5 = np.mean([r['recall@5'] for r in task_results])
        mean_recall10 = np.mean([r['recall@10'] for r in task_results])
        mean_mrr = np.mean([r['mrr'] for r in task_results])
        mean_log_boost = np.mean([r['log_boost'] for r in task_results])

        # Test guided enumeration speedup
        print(f"\n  Testing enumeration speedup...")
        speedups = []
        guided_solved = 0

        for task_name, sol in solutions.items():
            task = next(t for t in tasks if t.name == task_name)
            guided_grammar = model.predict_grammar_weights(task)

            prog, guided_time, guided_n = solve_task(task, guided_grammar, timeout=10.0)
            if prog:
                guided_solved += 1
                uniform_time = sol['time']
                if guided_time > 0:
                    speedups.append(uniform_time / guided_time)

        mean_speedup = np.mean(speedups) if speedups else 1.0

        all_results[name] = {
            'mean_recall@5': mean_recall5,
            'mean_recall@10': mean_recall10,
            'mean_mrr': mean_mrr,
            'mean_log_boost': mean_log_boost,
            'guided_solve_rate': guided_solved / len(solutions),
            'mean_speedup': mean_speedup,
            'task_results': task_results
        }

        print(f"\n  Summary for {name}:")
        print(f"    Mean Recall@5:  {mean_recall5:.3f}")
        print(f"    Mean Recall@10: {mean_recall10:.3f}")
        print(f"    Mean MRR:       {mean_mrr:.3f}")
        print(f"    Mean LogBoost:  {mean_log_boost:.3f}")
        print(f"    Mean Speedup:   {mean_speedup:.2f}x")

    # Final comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    print(f"\n{'Model':<20} {'R@5':<8} {'R@10':<8} {'MRR':<8} {'LogBoost':<10} {'Speedup'}")
    print("-" * 80)
    for name, results in all_results.items():
        print(f"{name:<20} {results['mean_recall@5']:<8.3f} {results['mean_recall@10']:<8.3f} "
              f"{results['mean_mrr']:<8.3f} {results['mean_log_boost']:<10.3f} {results['mean_speedup']:.2f}x")

    # Best model
    best_by_mrr = max(all_results.items(), key=lambda x: x[1]['mean_mrr'])
    best_by_speedup = max(all_results.items(), key=lambda x: x[1]['mean_speedup'])

    print(f"\n✓ Best by MRR: {best_by_mrr[0]} ({best_by_mrr[1]['mean_mrr']:.3f})")
    print(f"✓ Best by Speedup: {best_by_speedup[0]} ({best_by_speedup[1]['mean_speedup']:.2f}x)")

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(f"results/prediction_quality_detailed_{timestamp}.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)

    # Convert for JSON
    save_results = {}
    for name, r in all_results.items():
        save_results[name] = {k: v for k, v in r.items() if k != 'task_results'}
        save_results[name]['n_tasks'] = len(r['task_results'])

    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
