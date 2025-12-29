#!/usr/bin/env python3
"""
5-Iteration Memoized Enumeration Test with Contrastive Wake-Sleep

Extended version of test_memoized_enumeration.py with 5 iterations.
Expected runtime: ~20-25 hours based on previous 3-iteration run.

Usage:
    nohup caffeinate -d -i -s python3 run_5iter_memoized.py > 5iter.out 2>&1 &
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.contrastive_wake_sleep import ContrastiveWakeSleep, create_tasks_from_rules
from dreamcoder_core.program import Program
from rules.cards import sample_hand
from rules.catalogue import create_all_rules


def main():
    print("=" * 70)
    print("MEMOIZED ENUMERATION - 5 ITERATIONS")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Build grammar
    print("Building grammar...")
    grammar = build_lean_grammar()
    print(f"  Grammar size: {len(list(grammar.primitives()))} primitives")

    # Create all experimental rules
    print("\nCreating tasks from experimental rules (catalogue.py)...")
    all_rules = create_all_rules()
    print(f"  Total rules: {len(all_rules)}")

    # Create tasks using the same method as 3-iteration test
    tasks = create_tasks_from_rules(all_rules, n_examples=20, seed=42)
    print(f"  Created {len(tasks)} tasks with balanced sampling (6-card hands)")

    # Evaluation function
    def eval_fn(program: Program, hand):
        fn = program.evaluate([])
        return fn(hand)

    def sample_hand_fn():
        return sample_hand(6)

    def sample_card_fn():
        return sample_hand(1)[0]

    # Results directory
    results_dir = Path(__file__).parent / "results_5iter"
    results_dir.mkdir(exist_ok=True)

    # Initialize contrastive wake-sleep with memoized enumeration
    print("\nInitializing contrastive wake-sleep with memoized enumeration...")
    learner = ContrastiveWakeSleep(
        grammar=grammar,
        tasks=tasks,
        eval_fn=eval_fn,
        sample_hand_fn=sample_hand_fn,
        sample_card_fn=sample_card_fn,

        # Enumeration settings - use memoized!
        enumeration_budget=50000,
        enumeration_timeout=45.0,
        max_depth=7,

        # Frontier settings
        keep_top_k=3,

        # Enable all phases
        use_compression=True,
        use_recognition=True,
        use_dreaming=True,

        # Recognition settings
        recognition_hidden_dim=64,
        recognition_epochs=10,
        structural_similarity_weight=0.1,

        # Dreaming settings
        dreams_per_iteration=20,
        contrastive_dream_ratio=0.5,

        # Run settings - 5 ITERATIONS
        max_iterations=5,
        verbose=True,
        log_dir=str(results_dir)
    )

    print(f"  Model parameters: {sum(p.numel() for p in learner.recognition.parameters()):,}")
    print()

    # Run!
    print("Starting wake-sleep loop with memoized enumeration...")
    print("-" * 70)
    start_time = time.time()

    results = learner.run()

    total_time = time.time() - start_time
    print("-" * 70)
    print()

    # Report results
    print("=" * 70)
    print("5-ITERATION TEST COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes, {total_time/3600:.1f} hours)")
    print()
    print("Summary:")
    print(f"  Tasks solved: {results['summary']['tasks_solved']}/{results['summary']['tasks_total']}")
    print(f"  Success rate: {100*results['summary']['tasks_solved']/results['summary']['tasks_total']:.1f}%")
    print(f"  Final grammar size: {results['summary']['final_grammar_size']} primitives")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"5iter_results_{timestamp}.json"

    # Save summary
    summary = {
        'timestamp': timestamp,
        'total_time_seconds': total_time,
        'total_time_hours': total_time / 3600,
        'n_iterations': 5,
        'tasks_solved': results['summary']['tasks_solved'],
        'tasks_total': results['summary']['tasks_total'],
        'final_grammar_size': results['summary']['final_grammar_size'],
        'config': {
            'enumeration_budget': 50000,
            'enumeration_timeout': 45.0,
            'max_depth': 7,
            'keep_top_k': 3,
            'recognition_hidden_dim': 64,
            'dreams_per_iteration': 20,
        }
    }

    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
