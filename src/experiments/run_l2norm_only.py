#!/usr/bin/env python3
"""Run just L2Norm+Temperature to complete the comparison."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import random
import numpy as np
import torch
from datetime import datetime

from experiments.compare_normalization_wakesleep import (
    ExperimentConfig, create_tasks, run_wake_sleep,
    L2NormRecognitionModel, build_lean_grammar, print_flush
)

def main():
    print_flush("=" * 70)
    print_flush("L2Norm+Temperature Only Run")
    print_flush("=" * 70)
    print_flush(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    config = ExperimentConfig()

    # Load grammar
    print_flush("\nLoading grammar...")
    grammar = build_lean_grammar()
    num_primitives = len(grammar.productions)
    print_flush(f"  Primitives: {num_primitives}")

    # Create tasks
    print_flush("\nCreating tasks...")
    tasks = create_tasks(config)
    print_flush(f"  Tasks created: {len(tasks)}")

    # Output directory - use same as the incomplete run
    output_dir = Path("results_normalization_wakesleep/comparison_20260101_211404")

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Recreate tasks with same seed for fair comparison
    tasks = create_tasks(config)

    # Create L2Norm model
    model = L2NormRecognitionModel(num_primitives, config.hidden_dim, temperature_init=20.0)

    # Run wake-sleep
    results = run_wake_sleep('L2Norm+Temperature', model, tasks, grammar, config)

    # Save results
    results_file = output_dir / "L2Norm_Temperature_results.json"
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

    print_flush(f"\nResults saved to: {results_file}")

    # Print summary
    final = results[-1]
    print_flush("\n" + "=" * 70)
    print_flush("L2Norm+Temperature FINAL RESULTS")
    print_flush("=" * 70)
    print_flush(f"Tasks solved: {final.tasks_solved_total}/35")
    print_flush(f"Total programs: {sum(r.programs_enumerated for r in results):,}")
    print_flush(f"Total time: {sum(r.wall_time for r in results):.1f}s")

    # Now create combined summary
    layernorm_file = output_dir / "LayerNorm_Scale_results.json"
    if layernorm_file.exists():
        with open(layernorm_file) as f:
            ln_results = json.load(f)

        ln_final = ln_results[-1]
        l2_final = results[-1]

        summary = {
            'config': {
                'n_rules': config.n_rules,
                'n_examples_per_task': config.n_examples_per_task,
                'hand_size': config.hand_size,
                'n_iterations': config.n_iterations,
                'enumeration_budget': config.enumeration_budget,
            },
            'learning_curves': {
                'LayerNorm+Scale': [r['tasks_solved_total'] for r in ln_results],
                'L2Norm+Temperature': [r.tasks_solved_total for r in results]
            },
            'final_solved': {
                'LayerNorm+Scale': ln_final['tasks_solved_total'],
                'L2Norm+Temperature': l2_final.tasks_solved_total
            },
            'total_time': {
                'LayerNorm+Scale': sum(r['wall_time'] for r in ln_results),
                'L2Norm+Temperature': sum(r.wall_time for r in results)
            },
            'final_training_loss': {
                'LayerNorm+Scale': ln_final['training_loss'],
                'L2Norm+Temperature': l2_final.training_loss
            }
        }

        with open(output_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)

        print_flush("\n" + "=" * 70)
        print_flush("COMPARISON SUMMARY")
        print_flush("=" * 70)
        print_flush(f"LayerNorm+Scale:     {ln_final['tasks_solved_total']}/35 solved")
        print_flush(f"L2Norm+Temperature:  {l2_final.tasks_solved_total}/35 solved")
        print_flush(f"\nLearning curves:")
        print_flush(f"  LayerNorm: {summary['learning_curves']['LayerNorm+Scale']}")
        print_flush(f"  L2Norm:    {summary['learning_curves']['L2Norm+Temperature']}")

if __name__ == "__main__":
    main()
