#!/usr/bin/env python3
"""
Pre-training Runner for DreamCoder V2

This script runs pre-training on the 22 easy pretraining rules to:
1. Build up useful abstractions
2. Train the neural recognition model
3. Create a warm-started system for harder rules

Progress is logged in detail for hourly monitoring.
"""

import sys
import os
import time
import json
import random
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dreamcoder_core.dreamcoder_v2 import (
    DreamCoderV2, create_tasks_from_rules, make_eval_fn
)
from dreamcoder_core.lean_primitives import build_lean_grammar
from rules.pretraining_rules import get_easy_pretraining_rules, get_all_pretraining_rules


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


def print_banner(text: str, char: str = "="):
    """Print a banner."""
    line = char * 70
    print(f"\n{line}")
    print(text)
    print(f"{line}\n", flush=True)


def main():
    start_time = time.time()

    print_banner("DREAMCODER V2 - PRE-TRAINING RUN")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # =========================================================================
    # CONFIGURATION
    # =========================================================================

    # Get easy pretraining rules
    rules = get_easy_pretraining_rules()
    print(f"Pre-training rules: {len(rules)}")

    # Print rule details
    print("\nRules to learn:")
    for i, rule in enumerate(rules, 1):
        print(f"  {i:2d}. {rule.id} ({rule.family}): {rule.description[:50]}...")
    print()

    # Create tasks
    print("Creating tasks with 20 examples each...")
    tasks = create_tasks_from_rules(rules, n_examples=20, seed=42)
    print(f"Created {len(tasks)} tasks")

    # Check example balance
    print("\nExample balance check:")
    for task in tasks[:3]:
        pos = sum(1 for _, label in task.examples if label)
        neg = sum(1 for _, label in task.examples if not label)
        print(f"  {task.name}: {pos} pos / {neg} neg")
    print("  ...")

    # Build grammar
    print("\nBuilding lean grammar...")
    grammar = build_lean_grammar()
    print(f"Grammar: {len(grammar)} primitives")

    # Configuration
    config = {
        "enumeration_budget": 100000,
        "enumeration_timeout": 120.0,
        "max_depth": 7,
        "keep_top_k": 5,
        "use_compression": True,
        "use_recognition": True,
        "use_dreaming": True,
        "recognition_hidden_dim": 128,
        "recognition_epochs": 10,
        "recognition_lr": 1e-3,
        "dreams_per_iteration": 30,
        "dream_temperature": 1.0,
        "max_iterations": 5,
    }

    print_banner("CONFIGURATION")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Estimated time
    est_time_per_task = 100000 / 3000  # ~33 seconds per task at 3000 prog/sec
    est_total = est_time_per_task * len(tasks) * config["max_iterations"]
    print(f"\nEstimated time: {format_time(est_total)} (very rough)")
    print("(Actual time depends on task difficulty and compression)")

    # Create output directory
    log_dir = Path("results/pretraining")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # Create eval function
    eval_fn = make_eval_fn()

    print_banner("STARTING WAKE-SLEEP LEARNING")

    # =========================================================================
    # RUN DREAMCODER
    # =========================================================================

    dc = DreamCoderV2(
        grammar=grammar,
        tasks=tasks,
        eval_fn=eval_fn,
        enumeration_budget=config["enumeration_budget"],
        enumeration_timeout=config["enumeration_timeout"],
        max_depth=config["max_depth"],
        keep_top_k=config["keep_top_k"],
        use_compression=config["use_compression"],
        use_recognition=config["use_recognition"],
        use_dreaming=config["use_dreaming"],
        recognition_hidden_dim=config["recognition_hidden_dim"],
        recognition_epochs=config["recognition_epochs"],
        dreams_per_iteration=config["dreams_per_iteration"],
        dream_temperature=config["dream_temperature"],
        max_iterations=config["max_iterations"],
        verbose=True,
        log_dir=str(log_dir),
        device='cpu'
    )

    results = dc.run()

    # =========================================================================
    # FINAL REPORT
    # =========================================================================

    total_time = time.time() - start_time

    print_banner("PRE-TRAINING COMPLETE")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {format_time(total_time)}")
    print()

    # Summary
    print("SUMMARY:")
    print(f"  Tasks solved: {results['summary']['tasks_solved']}/{results['summary']['tasks_total']}")
    print(f"  Solve rate: {100*results['summary']['tasks_solved']/results['summary']['tasks_total']:.1f}%")
    print(f"  Final grammar size: {results['summary']['final_grammar_size']}")
    print(f"  Total abstractions learned: {results['summary']['total_abstractions']}")
    print(f"  Total dreams generated: {results['summary']['total_dreams']}")
    print()

    # Learning curve
    print("LEARNING CURVE:")
    for m in results['learning_curve']:
        print(f"  Iter {m['iteration']+1}: {m['tasks_solved']}/{results['summary']['tasks_total']} solved, "
              f"{m['programs']:,} programs, {m['abstractions']} abstractions, "
              f"loss={m['recognition_loss']:.4f}")
    print()

    # Solved tasks by family
    print("SOLVED TASKS BY FAMILY:")
    family_stats = {}
    for name, tm in results['task_metrics'].items():
        # Find the task's family
        task_family = None
        for task in tasks:
            if task.name == name:
                task_family = task.family
                break
        if task_family:
            if task_family not in family_stats:
                family_stats[task_family] = {'solved': 0, 'total': 0}
            family_stats[task_family]['total'] += 1
            if tm['solved']:
                family_stats[task_family]['solved'] += 1

    for family, stats in sorted(family_stats.items()):
        print(f"  {family}: {stats['solved']}/{stats['total']}")
    print()

    # Unsolved tasks
    unsolved = [name for name, tm in results['task_metrics'].items() if not tm['solved']]
    if unsolved:
        print(f"UNSOLVED TASKS ({len(unsolved)}):")
        for name in unsolved:
            print(f"  - {name}")
    print()

    # Solved tasks with program
    solved_tasks = [(name, tm) for name, tm in results['task_metrics'].items() if tm['solved']]
    if solved_tasks:
        print(f"SOLVED TASKS ({len(solved_tasks)}):")
        for name, tm in sorted(solved_tasks, key=lambda x: x[1]['iteration_solved']):
            print(f"  - {name} (iter {tm['iteration_solved']+1}, {tm['programs_to_solve']:,} programs)")
            print(f"    Program: {tm['best_program'][:80]}...")
    print()

    # Library evolution
    if results['library_evolution']:
        print("NEW ABSTRACTIONS LEARNED:")
        for i, abstractions in enumerate(results['library_evolution']):
            if abstractions:
                print(f"  Iteration {i+1}:")
                for abstr in abstractions:
                    print(f"    - {abstr[:60]}...")
    print()

    # Save detailed report
    report_path = log_dir / f"pretraining_report_{timestamp}.txt"
    with open(report_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DREAMCODER V2 PRE-TRAINING REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Duration: {format_time(total_time)}\n\n")

        f.write("CONFIGURATION:\n")
        for k, v in config.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("SUMMARY:\n")
        for k, v in results['summary'].items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

        f.write("ALL TASK RESULTS:\n")
        for name, tm in sorted(results['task_metrics'].items()):
            status = "SOLVED" if tm['solved'] else "UNSOLVED"
            f.write(f"\n  {name} [{status}]\n")
            if tm['solved']:
                f.write(f"    Iteration: {tm['iteration_solved']+1}\n")
                f.write(f"    Programs: {tm['programs_to_solve']:,}\n")
                f.write(f"    Description length: {tm['description_length']:.2f} bits\n")
                f.write(f"    Program: {tm['best_program']}\n")

    print(f"Detailed report saved to: {report_path}")

    # Final message
    print_banner("PRE-TRAINING RUN FINISHED", "=")
    if results['summary']['tasks_solved'] == 0:
        print("⚠️  WARNING: No tasks were solved!")
        print("   Consider:")
        print("   - Increasing enumeration budget")
        print("   - Adding more primitives to grammar")
        print("   - Starting with even simpler rules")
    elif results['summary']['tasks_solved'] < len(tasks) // 2:
        print("📊 Partial success - some tasks solved")
        print("   The recognition model has some training signal")
        print("   Abstractions (if any) can help harder tasks")
    else:
        print("✅ Good progress! Most tasks solved")
        print("   Recognition model has good training data")
        print("   Ready to attempt harder rules")


if __name__ == "__main__":
    main()
