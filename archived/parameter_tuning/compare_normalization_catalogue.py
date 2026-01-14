#!/usr/bin/env python3
"""
Run normalization comparison on CATALOGUE tasks (not pretraining).
This tests on a different set of rules to see if results generalize.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import random
import numpy as np
import torch
from datetime import datetime

from experiments.compare_normalization_wakesleep import (
    ExperimentConfig, run_wake_sleep,
    LayerNormRecognitionModel, L2NormRecognitionModel,
    build_lean_grammar, print_flush, TaskWrapper
)
from dreamcoder_core.task_generation import load_prerecorded_tasks


def create_catalogue_tasks(config: ExperimentConfig):
    """Load catalogue tasks instead of pretraining tasks."""
    tasks_path = Path(__file__).parent.parent / "data" / "prerecorded_tasks" / "catalogue_tasks.json"
    print_flush(f"Loading CATALOGUE tasks from {tasks_path.name}...")

    all_tasks = load_prerecorded_tasks(tasks_path)
    n_rules = min(config.n_rules, len(all_tasks))
    selected_tasks = all_tasks[:n_rules]

    # Verify balance
    for task in selected_tasks[:3]:
        pos = sum(1 for _, l in task.examples if l)
        neg = sum(1 for _, l in task.examples if not l)
        print_flush(f"  {task.name}: {pos}+/{neg}- examples")

    wrapped_tasks = [TaskWrapper(t) for t in selected_tasks]
    print_flush(f"Loaded {len(wrapped_tasks)} catalogue tasks with guaranteed balance")

    return wrapped_tasks


def main():
    print_flush("=" * 70)
    print_flush("NORMALIZATION COMPARISON - CATALOGUE RULES")
    print_flush("=" * 70)
    print_flush(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    config = ExperimentConfig(
        n_rules=44,  # All catalogue rules
        n_iterations=5,
        enumeration_budget=100_000,
    )

    # Load grammar
    print_flush("\nLoading grammar...")
    grammar = build_lean_grammar()
    num_primitives = len(grammar.productions)
    print_flush(f"  Primitives: {num_primitives}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results_normalization_catalogue/comparison_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump({
            'n_rules': config.n_rules,
            'n_iterations': config.n_iterations,
            'enumeration_budget': config.enumeration_budget,
            'task_source': 'catalogue_tasks.json'
        }, f, indent=2)

    # Define approaches
    approaches = {
        'LayerNorm+Scale': lambda: LayerNormRecognitionModel(
            num_primitives, config.hidden_dim, config.layernorm_scale_init
        ),
        'L2Norm+Temperature': lambda: L2NormRecognitionModel(
            num_primitives, config.hidden_dim, temperature_init=20.0
        )
    }

    all_results = {}

    for name, model_fn in approaches.items():
        # Reset seeds
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Create tasks
        tasks = create_catalogue_tasks(config)

        # Create model
        model = model_fn()

        # Run wake-sleep
        results = run_wake_sleep(name, model, tasks, grammar, config)
        all_results[name] = results

        # Save results
        results_file = output_dir / f"{name.replace('+', '_').replace(' ', '_')}_results.json"
        with open(results_file, 'w') as f:
            json.dump([{
                'iteration': r.iteration,
                'tasks_solved_total': r.tasks_solved_total,
                'tasks_solved_new': r.tasks_solved_new,
                'programs_enumerated': r.programs_enumerated,
                'wall_time': r.wall_time,
                'training_loss': r.training_loss,
                'solved_task_names': r.solved_task_names
            } for r in results], f, indent=2)

    # Create summary
    summary = {
        'task_source': 'catalogue_tasks.json',
        'learning_curves': {
            name: [r.tasks_solved_total for r in results]
            for name, results in all_results.items()
        },
        'final_solved': {
            name: results[-1].tasks_solved_total
            for name, results in all_results.items()
        },
        'total_time': {
            name: sum(r.wall_time for r in results)
            for name, results in all_results.items()
        },
        'final_training_loss': {
            name: results[-1].training_loss
            for name, results in all_results.items()
        }
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Print comparison
    print_flush("\n" + "=" * 70)
    print_flush("CATALOGUE RULES - COMPARISON SUMMARY")
    print_flush("=" * 70)

    for name, results in all_results.items():
        final = results[-1]
        print_flush(f"\n{name}:")
        print_flush(f"  Tasks solved: {final.tasks_solved_total}/{config.n_rules}")
        print_flush(f"  Learning curve: {[r.tasks_solved_total for r in results]}")
        print_flush(f"  Final loss: {final.training_loss:.4f}")
        print_flush(f"  Total time: {sum(r.wall_time for r in results):.1f}s")

    print_flush(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
