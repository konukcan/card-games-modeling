#!/usr/bin/env python3
"""
Evaluate primitive prediction quality using existing solved tasks as ground truth.

This script:
1. Loads solved tasks from previous experiment results
2. Parses solution strings to extract primitives
3. Tests contrastive model predictions against ground truth
4. Measures how well predictions correlate with solution primitives
"""

import sys
import json
import re
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Set, Tuple

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.dreamcoder_original import create_tasks_from_rules
from dreamcoder_core.grammar import Grammar
from rules.catalogue import create_all_rules

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_primitives_from_string(solution_str: str, grammar: Grammar) -> Set[str]:
    """Extract primitive names from a solution string like '(λ gt 5 (n_unique_ranks $0))'."""
    # Get all primitive names from grammar
    prim_names = {p.name for p in grammar.primitives()}

    # Find all words in the solution string
    words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', solution_str)

    # Keep only those that are primitives
    return {w for w in words if w in prim_names}


def load_ground_truth(results_file: Path, grammar: Grammar) -> Dict[str, Dict[str, Any]]:
    """Load solved tasks from a results file."""
    with open(results_file) as f:
        data = json.load(f)

    ground_truth = {}

    # Navigate to task details
    if 'iterations' in data:
        for iteration in data['iterations']:
            if 'task_details' in iteration:
                for task_name, details in iteration['task_details'].items():
                    if details.get('solved', False) and 'solution' in details:
                        solution_str = details['solution']
                        primitives = extract_primitives_from_string(solution_str, grammar)

                        # Skip trivial solutions like (λ false) or (λ true)
                        if primitives and primitives not in [{'false'}, {'true'}]:
                            ground_truth[task_name] = {
                                'solution': solution_str,
                                'primitives': primitives,
                                'solution_size': details.get('solution_size', 0)
                            }

    return ground_truth


def compute_metrics(predictions: np.ndarray, solution_prims: Set[str], prim_names: List[str]) -> Dict[str, float]:
    """Compute prediction quality metrics."""
    sorted_idx = np.argsort(predictions)[::-1]
    sorted_names = [prim_names[i] for i in sorted_idx]

    n_sol = len(solution_prims)
    n_prims = len(prim_names)

    # Find ranks of solution primitives (1-indexed)
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

    # MRR (Mean Reciprocal Rank)
    mrr = np.mean([1.0/r for r in ranks]) if ranks else 0.0
    mean_rank = np.mean(ranks) if ranks else n_prims

    # Log-likelihood boost over uniform
    uniform_p = 1.0 / n_prims
    sol_idx = [prim_names.index(p) for p in solution_prims if p in prim_names]
    if sol_idx:
        sol_probs = np.clip(predictions[sol_idx], 1e-10, 1.0)
        log_boost = np.mean(np.log(sol_probs) - np.log(uniform_p))
        mean_sol_prob = np.mean(sol_probs)
    else:
        log_boost = 0.0
        mean_sol_prob = 0.0

    # Probability ratio: solution prims vs others
    other_idx = [i for i in range(n_prims) if prim_names[i] not in solution_prims]
    mean_other_prob = np.mean(predictions[other_idx]) if other_idx else 0.0
    prob_ratio = mean_sol_prob / mean_other_prob if mean_other_prob > 0 else 0.0

    return {
        'recall@5': recall_k(5),
        'recall@10': recall_k(10),
        'recall@15': recall_k(15),
        'precision@5': precision_k(5),
        'mrr': mrr,
        'mean_rank': mean_rank,
        'log_boost': log_boost,
        'mean_sol_prob': mean_sol_prob,
        'mean_other_prob': mean_other_prob,
        'prob_ratio': prob_ratio,
        'top5': sorted_names[:5],
        'top10': sorted_names[:10],
        'ranks': ranks
    }


def create_task_lookup(tasks: List) -> Dict[str, Any]:
    """Create lookup from task name to task object."""
    # Normalize names for matching
    lookup = {}
    for task in tasks:
        lookup[task.name] = task
        # Also add normalized versions
        normalized = task.name.lower().replace('_', '')
        lookup[normalized] = task
    return lookup


def main():
    print("=" * 80)
    print("PRIMITIVE PREDICTION QUALITY - USING GROUND TRUTH")
    print("=" * 80)

    # Load grammar and tasks
    grammar = build_lean_grammar()
    rules = create_all_rules()
    tasks = create_tasks_from_rules(rules, n_examples=20, seed=42)
    task_lookup = create_task_lookup(tasks)

    print(f"\nLoaded {len(tasks)} tasks, {len(grammar.productions)} primitives")

    # Load ground truth from results files
    results_dir = Path("results_archive/test_runs")
    results_files = list(results_dir.rglob("final_results.json"))

    print(f"\nFound {len(results_files)} result files")

    # Aggregate ground truth from all files
    all_ground_truth = {}
    for rf in results_files:
        gt = load_ground_truth(rf, grammar)
        all_ground_truth.update(gt)

    print(f"Found {len(all_ground_truth)} non-trivial solved tasks")

    if not all_ground_truth:
        print("No ground truth found!")
        return

    # Show ground truth
    print("\nGround truth tasks:")
    for name, info in sorted(all_ground_truth.items()):
        prims = ', '.join(sorted(info['primitives']))
        print(f"  {name}: {prims}")

    # Model configurations to test
    model_configs = [
        ("baseline", {"normalize_embeddings": False, "encoding_mode": "standard"}),
        ("layernorm_s10", {"normalize_embeddings": True, "embedding_scale": 10.0, "encoding_mode": "standard"}),
        ("layernorm_s20", {"normalize_embeddings": True, "embedding_scale": 20.0, "encoding_mode": "standard"}),
        ("layernorm_s50", {"normalize_embeddings": True, "embedding_scale": 50.0, "encoding_mode": "standard"}),
        ("triple_s20", {"normalize_embeddings": True, "embedding_scale": 20.0, "encoding_mode": "triple_contrast", "n_random_hands": 10}),
        ("random_s20", {"normalize_embeddings": True, "embedding_scale": 20.0, "encoding_mode": "random_contrast", "n_random_hands": 10, "lambda_random": 0.5}),
    ]

    all_results = {}

    for model_name, config in model_configs:
        print("\n" + "=" * 80)
        print(f"EVALUATING: {model_name}")
        print("=" * 80)

        # Create model
        model = ContrastiveRecognitionModel(
            grammar=grammar,
            card_hidden=128,
            card_out=64,
            pred_hidden=128,
            **config
        )

        # We'll test on UNTRAINED model first to see baseline diversity
        # Then train and compare

        model.eval()
        task_results = []

        print("\nPer-task predictions (UNTRAINED):")
        print(f"{'Task':<25} {'Solution Prims':<20} {'R@5':<7} {'R@10':<7} {'MRR':<7} {'Top-5 Predictions'}")
        print("-" * 100)

        with torch.no_grad():
            for task_name, gt_info in sorted(all_ground_truth.items()):
                # Find matching task
                task = None
                for t in tasks:
                    if task_name.lower() in t.name.lower() or t.name.lower() in task_name.lower():
                        task = t
                        break

                if task is None:
                    continue

                pred = model.predict_primitives(task).cpu().numpy()
                metrics = compute_metrics(pred, gt_info['primitives'], model.primitive_names)
                metrics['task'] = task_name
                metrics['solution_prims'] = list(gt_info['primitives'])
                task_results.append(metrics)

                sol_prims_str = ','.join(sorted(gt_info['primitives']))[:18]
                top5_str = ','.join(metrics['top5'])[:35]
                print(f"{task_name:<25} {sol_prims_str:<20} {metrics['recall@5']:<7.2f} {metrics['recall@10']:<7.2f} {metrics['mrr']:<7.3f} {top5_str}")

        if task_results:
            # Aggregate metrics
            mean_recall5 = np.mean([r['recall@5'] for r in task_results])
            mean_recall10 = np.mean([r['recall@10'] for r in task_results])
            mean_mrr = np.mean([r['mrr'] for r in task_results])
            mean_log_boost = np.mean([r['log_boost'] for r in task_results])
            mean_prob_ratio = np.mean([r['prob_ratio'] for r in task_results])

            all_results[model_name] = {
                'mean_recall@5': mean_recall5,
                'mean_recall@10': mean_recall10,
                'mean_mrr': mean_mrr,
                'mean_log_boost': mean_log_boost,
                'mean_prob_ratio': mean_prob_ratio,
                'n_tasks': len(task_results),
                'task_results': task_results
            }

            print(f"\nSummary for {model_name}:")
            print(f"  Mean Recall@5:   {mean_recall5:.3f}")
            print(f"  Mean Recall@10:  {mean_recall10:.3f}")
            print(f"  Mean MRR:        {mean_mrr:.3f}")
            print(f"  Mean LogBoost:   {mean_log_boost:.3f}")
            print(f"  Mean ProbRatio:  {mean_prob_ratio:.2f}")

    # Final comparison
    print("\n" + "=" * 80)
    print("FINAL COMPARISON (UNTRAINED MODELS)")
    print("=" * 80)

    print(f"\n{'Model':<20} {'R@5':<8} {'R@10':<8} {'MRR':<8} {'LogBoost':<10} {'ProbRatio'}")
    print("-" * 80)

    for name, results in all_results.items():
        print(f"{name:<20} {results['mean_recall@5']:<8.3f} {results['mean_recall@10']:<8.3f} "
              f"{results['mean_mrr']:<8.3f} {results['mean_log_boost']:<10.3f} {results['mean_prob_ratio']:<.2f}")

    # Best model
    if all_results:
        best_by_recall = max(all_results.items(), key=lambda x: x[1]['mean_recall@5'])
        best_by_mrr = max(all_results.items(), key=lambda x: x[1]['mean_mrr'])

        print(f"\n✓ Best by Recall@5: {best_by_recall[0]} ({best_by_recall[1]['mean_recall@5']:.3f})")
        print(f"✓ Best by MRR: {best_by_mrr[0]} ({best_by_mrr[1]['mean_mrr']:.3f})")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(f"results/ground_truth_eval_{timestamp}.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)

    save_results = {}
    for name, r in all_results.items():
        save_results[name] = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                              for k, v in r.items() if k != 'task_results'}

    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
