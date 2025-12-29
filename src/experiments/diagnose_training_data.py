#!/usr/bin/env python3
"""
Diagnostic: What primitives were in the training data?

The neural model predicts all_same_suit, all_same_color, n_unique_ranks for
ALL tasks. This might be because:
1. Only a few tasks were solved during training
2. Those solved tasks predominantly used these primitives
3. The model learned to predict "what primitives typically appear in solutions"
   rather than "what primitives are needed for THIS task"

Let's examine the training data to verify this hypothesis.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
from collections import Counter

RESULTS_DIR = Path("results/warmstart_experiment")


def extract_primitives_from_solution(solution_str):
    """Extract primitive names from a solution string."""
    # Simple heuristic: split on parentheses and spaces, filter for primitive names
    # This won't be perfect but gives us an idea
    primitives = []
    # Common primitives we're looking for
    known_primitives = [
        'all_same_suit', 'all_same_color', 'n_unique_ranks', 'n_unique_suits',
        'has_suit', 'le', 'gt', 'eq', 'not', 'and', 'or', 'if',
        'map_suit', 'map_rank', 'map_color', 'filter', 'length', 'head', 'last',
        'first_half', 'second_half', 'reverse', 'is_sorted', 'has_ap',
        'HEARTS', 'DIAMONDS', 'CLUBS', 'SPADES', 'RED', 'BLACK',
        'is_face', 'is_ace', 'odd', 'even', '+', '-', 'mod'
    ]
    for prim in known_primitives:
        if prim in solution_str:
            primitives.append(prim)
    return primitives


def main():
    print("=" * 70)
    print("TRAINING DATA ANALYSIS")
    print("=" * 70)

    # Load all results files
    result_files = [
        ("Neural BOTH", RESULTS_DIR / "neural_BOTH_20251225_145356" / "results_WARM.json"),
        ("Contrastive Sigmoid", RESULTS_DIR / "contrastive_BOTH_20251225_145818" / "results_WARM.json"),
        ("Contrastive Softmax", RESULTS_DIR / "contrastive_softmax_WARM_20251226_174158" / "results_WARM.json"),
    ]

    all_primitives = Counter()

    for name, path in result_files:
        if not path.exists():
            print(f"\n{name}: File not found")
            continue

        with open(path) as f:
            results = json.load(f)

        print(f"\n{'='*70}")
        print(f"{name}")
        print(f"{'='*70}")

        # Check pretraining
        pretraining = results.get('pretraining', {})
        pretrain_solved = pretraining.get('solved_tasks', [])
        print(f"\nPretraining solved: {len(pretrain_solved)} tasks")
        if pretrain_solved:
            print(f"  Tasks: {pretrain_solved}")

        # Check main training
        main_training = results.get('main_training', {})
        task_metrics = main_training.get('task_metrics', {})

        solved_count = 0
        solved_tasks = []
        model_primitives = Counter()

        for task_name, metrics in task_metrics.items():
            if metrics.get('solved'):
                solved_count += 1
                solved_tasks.append(task_name)

                # Get primitives used
                prims = metrics.get('primitives_used', [])
                for p in prims:
                    model_primitives[p] += 1
                    all_primitives[p] += 1

                # Also try to get from solution string if available
                solution = metrics.get('solution', '')
                if solution:
                    extra_prims = extract_primitives_from_solution(solution)
                    for p in extra_prims:
                        if p not in prims:
                            model_primitives[p] += 1
                            all_primitives[p] += 1

        print(f"\nMain training solved: {solved_count} tasks")
        print(f"  Tasks: {solved_tasks}")

        print(f"\nPrimitives used in solutions:")
        for prim, count in model_primitives.most_common():
            print(f"  {prim}: {count}")

        # Check iterations
        iterations = main_training.get('iterations', [])
        if iterations:
            print(f"\nIteration details:")
            for i, it in enumerate(iterations):
                solved = it.get('tasks_solved', 0)
                new_solved = it.get('newly_solved', [])
                print(f"  Iteration {i+1}: {solved} solved, new: {new_solved}")

    print("\n" + "=" * 70)
    print("COMBINED PRIMITIVE FREQUENCY (All Models)")
    print("=" * 70)
    for prim, count in all_primitives.most_common(20):
        print(f"  {prim}: {count}")

    # Check if the top predicted primitives match the most frequent in training
    print("\n" + "=" * 70)
    print("HYPOTHESIS CHECK")
    print("=" * 70)

    predicted_top = ['all_same_suit', 'all_same_color', 'n_unique_ranks', 'le', 'second_half']
    print("\nTop predicted primitives by all models:")
    for p in predicted_top:
        freq = all_primitives.get(p, 0)
        print(f"  {p}: {freq} occurrences in training data")

    # Check how many unique primitives were seen
    print(f"\nTotal unique primitives in training: {len(all_primitives)}")
    print(f"Grammar has ~59 primitives")

    if len(all_primitives) < 10:
        print("\n⚠️ VERY FEW PRIMITIVES SEEN IN TRAINING!")
        print("   The model can only learn to predict primitives it has seen.")
        print("   With such limited training data, it will overfit to the few")
        print("   primitives that appeared in the solved tasks.")


if __name__ == '__main__':
    main()
