#!/usr/bin/env python3
"""
Two-Phase Overnight Training Run

Phase 5: Intensive pretraining consolidation
- 43 pretraining rules
- 10 iterations with aggressive search
- Build strong grammar and model

Phase 6: Transfer test to full catalogue
- 50 catalogue rules (with overlap)
- 8 iterations
- Test if learned abstractions transfer
"""

import sys
import os
import time
import argparse
import random
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from run_overnight_cython import (
    CythonOptimizedDreamCoder,
    PhaseConfig,
    print_banner,
    format_time,
    N_WORKERS,
    USE_PYPY,
    USE_CYTHON
)
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.dreamcoder_v2 import create_tasks_from_rules, Task
from dreamcoder_core.type_system import arrow, HAND, BOOL
from rules.pretraining_rules import get_easy_pretraining_rules, get_all_pretraining_rules
from rules.catalogue import create_all_rules
from rules.cards import sample_hand


def make_eval_fn():
    """Create evaluation function for tasks."""
    def eval_fn(program, task):
        try:
            for io in task.examples[:10]:
                hand = io[0][0]
                expected = io[1]
                result = program.run(hand)
                if result != expected:
                    return False
            return True
        except Exception:
            return False
    return eval_fn


def create_catalogue_tasks(n_examples=100, n_holdout=20, hand_size=6, seed=42):
    """Create tasks from the full catalogue of rules."""
    catalogue_rules = create_all_rules()

    random.seed(seed)

    tasks = []
    for rule in catalogue_rules:
        # Generate examples using the same approach as create_tasks_from_rules
        positives = []
        negatives = []
        holdout_positives = []
        holdout_negatives = []

        target = n_examples // 2
        holdout_target = n_holdout // 2

        # Use rule-specific seed for reproducibility
        rule_seed = seed + hash(rule.id) % 10000
        random.seed(rule_seed)

        # Sample hands to get balanced examples
        for _ in range(50000):
            hand = sample_hand(hand_size)
            try:
                label = rule.predicate(hand)
                if label:
                    if len(positives) < target:
                        positives.append((hand, True))
                    elif len(holdout_positives) < holdout_target:
                        holdout_positives.append((hand, True))
                else:
                    if len(negatives) < target:
                        negatives.append((hand, False))
                    elif len(holdout_negatives) < holdout_target:
                        holdout_negatives.append((hand, False))
            except:
                continue

            # Check if we have enough
            if (len(positives) >= target and len(negatives) >= target and
                len(holdout_positives) >= holdout_target and
                len(holdout_negatives) >= holdout_target):
                break

        examples = positives[:target] + negatives[:target]
        random.shuffle(examples)

        holdout = holdout_positives[:holdout_target] + holdout_negatives[:holdout_target]
        random.shuffle(holdout)

        # Create task
        task = Task(
            name=rule.id,
            request_type=arrow(HAND, BOOL),
            examples=examples,
            family=rule.family,
            difficulty_level=rule.level
        )
        task.holdout_examples = holdout
        tasks.append(task)

    return tasks


def main():
    parser = argparse.ArgumentParser(description="Two-Phase Overnight Training")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test run with 2 iterations per phase")
    parser.add_argument("--phase5-only", action="store_true",
                        help="Only run Phase 5 (pretraining consolidation)")
    parser.add_argument("--phase6-only", action="store_true",
                        help="Only run Phase 6 (requires checkpoint)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    start_time = time.time()

    print_banner("TWO-PHASE OVERNIGHT TRAINING")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check optimizations
    print("Checking optimizations...")
    print(f"  Cython modules: {'ENABLED' if USE_CYTHON else 'DISABLED (Python fallback)'}")
    print(f"  PyPy available: {USE_PYPY}")
    print(f"  Workers: {N_WORKERS}")
    print()

    # Load pretraining rules
    easy_rules = get_easy_pretraining_rules()
    all_pretraining_rules = get_all_pretraining_rules()

    print(f"Pretraining rules (easy): {len(easy_rules)}")
    print(f"Pretraining rules (all): {len(all_pretraining_rules)}")

    # Create pretraining tasks
    print("\nCreating pretraining tasks (100 examples + 20 holdout)...")
    easy_tasks = create_tasks_from_rules(easy_rules, n_examples=100, n_holdout=20, seed=42)
    pretraining_tasks = create_tasks_from_rules(all_pretraining_rules, n_examples=100, n_holdout=20, seed=42)
    print(f"Created {len(pretraining_tasks)} pretraining tasks")

    # Create catalogue tasks (for Phase 6)
    print("\nCreating catalogue tasks (100 examples + 20 holdout)...")
    catalogue_tasks = create_catalogue_tasks(n_examples=100, n_holdout=20, seed=42)
    print(f"Created {len(catalogue_tasks)} catalogue tasks")

    # Build grammar
    print("\nBuilding lean grammar...")
    grammar = build_lean_grammar()
    print(f"Grammar: {len(grammar)} primitives")

    # Define phases based on arguments
    if args.dry_run:
        print("\n*** DRY RUN MODE: 2 iterations per phase ***")
        phases = [
            PhaseConfig(
                name="Phase 5 (DRY RUN): Pretraining Consolidation",
                iterations=2,
                use_all_rules=True,
                enumeration_budget=100000,
                max_depth=8,
                dreams_per_iteration=50,
                recognition_epochs=5
            ),
        ]
        if not args.phase5_only:
            phases.append(PhaseConfig(
                name="Phase 6 (DRY RUN): Transfer Test",
                iterations=2,
                use_all_rules=True,
                enumeration_budget=100000,
                max_depth=8,
                dreams_per_iteration=50,
                recognition_epochs=5
            ))
    else:
        # Full overnight configuration
        phases = []

        if not args.phase6_only:
            # Phase 5: Intensive pretraining consolidation (~6 hours)
            phases.append(PhaseConfig(
                name="Phase 5: Pretraining Consolidation",
                iterations=10,
                use_all_rules=True,
                enumeration_budget=1000000,
                max_depth=12,
                dreams_per_iteration=300,
                recognition_epochs=25
            ))

        if not args.phase5_only:
            # Phase 6: Transfer test on full catalogue (~6 hours)
            phases.append(PhaseConfig(
                name="Phase 6: Transfer to Catalogue",
                iterations=8,
                use_all_rules=True,
                enumeration_budget=800000,
                max_depth=12,
                dreams_per_iteration=250,
                recognition_epochs=20
            ))

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "dry_run" if args.dry_run else "twophase"
    log_dir = Path(f"results/overnight_twophase/{run_name}_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create eval function
    eval_fn = make_eval_fn()

    # Determine which task set to use based on phase
    # Phase 5 uses pretraining_tasks, Phase 6 uses catalogue_tasks
    # For simplicity, we'll run them as separate executions if needed

    if args.phase6_only:
        print("\n*** PHASE 6 ONLY: Using catalogue tasks ***")
        tasks_to_use = catalogue_tasks
        easy_to_use = catalogue_tasks[:20]  # First 20 as "easy"
    else:
        print("\n*** PHASE 5: Using pretraining tasks ***")
        tasks_to_use = pretraining_tasks
        easy_to_use = easy_tasks

    print_banner("STARTING TWO-PHASE TRAINING")
    print(f"Output directory: {log_dir}")
    print(f"Tasks: {len(tasks_to_use)}")
    print(f"Phases: {len(phases)}")
    for i, phase in enumerate(phases):
        total_programs = phase.enumeration_budget * phase.iterations
        print(f"  {phase.name}: {phase.iterations} iters × {phase.enumeration_budget:,} = {total_programs:,} programs")
    print()

    # Run training
    dc = CythonOptimizedDreamCoder(
        grammar=grammar,
        easy_tasks=easy_to_use,
        all_tasks=tasks_to_use,
        eval_fn=eval_fn,
        phases=phases,
        recognition_hidden_dim=256,
        recognition_lr=5e-4,
        keep_top_k=5,
        max_inventions_per_iteration=5,
        dream_temperature=1.0,
        n_workers=N_WORKERS,
        use_pypy=USE_PYPY,
        verbose=True,
        log_dir=str(log_dir),
        device='cpu'
    )

    results = dc.run()

    # Final summary
    total_time = time.time() - start_time

    print_banner("TWO-PHASE TRAINING COMPLETE")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {format_time(total_time)}")
    print()
    print(f"Tasks solved: {results['summary']['tasks_solved']}/{results['summary']['tasks_total']}")
    print(f"Success rate: {100*results['summary']['tasks_solved']/results['summary']['tasks_total']:.1f}%")
    print(f"Final grammar: {results['summary']['final_grammar_size']} primitives")
    print(f"Total abstractions: {results['summary']['total_abstractions']}")
    print(f"Total dreams: {results['summary']['total_dreams']}")

    # Print solved tasks
    solved = [(name, tm) for name, tm in results['task_metrics'].items() if tm['solved']]
    print(f"\nSolved tasks ({len(solved)}):")
    for name, tm in sorted(solved, key=lambda x: x[1]['iteration_solved']):
        print(f"  - {name} (iter {tm['iteration_solved']+1})")

    # Save config for reproducibility
    config = {
        "run_type": "two_phase",
        "dry_run": args.dry_run,
        "phase5_only": args.phase5_only,
        "phase6_only": args.phase6_only,
        "total_time": total_time,
        "n_pretraining_tasks": len(pretraining_tasks),
        "n_catalogue_tasks": len(catalogue_tasks),
        "phases": [
            {
                "name": p.name,
                "iterations": p.iterations,
                "enumeration_budget": p.enumeration_budget,
                "max_depth": p.max_depth
            }
            for p in phases
        ],
        "results_summary": results['summary']
    }
    import json
    with open(log_dir / "run_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\nResults saved to: {log_dir}")


if __name__ == "__main__":
    main()
