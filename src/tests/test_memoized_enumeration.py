#!/usr/bin/env python3
"""
Test Memoized Enumeration with Contrastive Wake-Sleep

This script tests the new memoized enumeration (Option C) integrated
into the contrastive wake-sleep loop. It runs 3 iterations with:
- 45 experimental rules from catalogue.py
- Contrastive recognition model
- Contrastive dreaming
- Memoized enumeration (1000x+ speedup)

Usage:
    python test_memoized_enumeration.py
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dreamcoder_core.primitives import build_lean_grammar
from dreamcoder_core.contrastive_wake_sleep import ContrastiveWakeSleep, create_tasks_from_rules
from dreamcoder_core.program import Program
from rules.cards import sample_hand
from rules.catalogue import create_all_rules


def main():
    print("=" * 70)
    print("MEMOIZED ENUMERATION TEST - 3 ITERATIONS")
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

    # Create tasks with balanced sampling
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
    results_dir = Path("results_memoized_test")
    results_dir.mkdir(exist_ok=True)

    # Run contrastive wake-sleep with memoized enumeration
    print("\nInitializing contrastive wake-sleep with memoized enumeration...")
    learner = ContrastiveWakeSleep(
        grammar=grammar,
        tasks=tasks,
        eval_fn=eval_fn,
        sample_hand_fn=sample_hand_fn,
        sample_card_fn=sample_card_fn,

        # Enumeration settings - should be MUCH faster now with memoization
        enumeration_budget=50000,
        enumeration_timeout=45.0,  # Shorter timeout since memoization is fast
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

        # Run settings
        max_iterations=3,
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
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print()
    print("Summary:")
    print(f"  Tasks solved: {results['summary']['tasks_solved']}/{results['summary']['tasks_total']}")
    print(f"  Success rate: {100*results['summary']['tasks_solved']/results['summary']['tasks_total']:.1f}%")
    print(f"  Final grammar size: {results['summary']['final_grammar_size']} primitives")
    print(f"  Total dreams: {results['summary']['total_dreams']}")
    print(f"  Contrastive dreams: {results['summary']['total_contrastive_dreams']}")

    # Save results
    results_file = results_dir / f"memoized_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump({
            'summary': results['summary'],
            'total_time_seconds': total_time,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    # Per-iteration breakdown
    print("\nPer-iteration breakdown:")
    for i, metrics in enumerate(results['iterations']):
        print(f"  Iteration {i+1}:")
        print(f"    Solved: {metrics['tasks_solved']}/{metrics['tasks_total']}")
        print(f"    Programs enumerated: {metrics['total_programs_enumerated']:,}")
        print(f"    Wake time: {metrics['wake_time']:.1f}s")
        if metrics['new_abstractions']:
            print(f"    New abstractions: {len(metrics['new_abstractions'])}")

    return results


if __name__ == "__main__":
    main()
