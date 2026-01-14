#!/usr/bin/env python3
"""
Generate Pre-recorded Tasks for All Rules
==========================================

This script generates a complete set of training and holdout tasks for:
1. All 45 catalogue rules (from rules/catalogue.py)
2. All 44 pretraining rules (from rules/pretraining_rules.py)

The generated datasets are saved to JSON files for reproducible experiments.
This eliminates the task generation bug where rare rules like sym_ranks_palindrome
would silently generate trivially-solvable tasks with 0 positive examples.

Usage:
    python generate_prerecorded_tasks.py

Output:
    data/prerecorded_tasks/
    ├── catalogue_tasks.json       # 45 catalogue rules
    ├── pretraining_tasks.json     # 44 pretraining rules
    └── combined_tasks.json        # All 89 rules combined
"""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dreamcoder_core.task_generation import (
    TaskGenerationConfig,
    generate_and_save_dataset,
    load_prerecorded_tasks,
    serialize_task
)
from rules.catalogue import ALL_RULES as CATALOGUE_RULES
from rules.pretraining_rules import get_all_pretraining_rules

PRETRAINING_RULES = get_all_pretraining_rules()


def main():
    print("=" * 70)
    print("GENERATING PRE-RECORDED TASKS")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    # Output directory
    output_dir = Path(__file__).parent / "data" / "prerecorded_tasks"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration for robust task generation
    config = TaskGenerationConfig(
        n_training_positives=20,
        n_seed_positives=20,    # Hidden seeds for near-miss generation
        n_training_negatives=20,
        n_holdout_positives=20,
        n_holdout_negatives=20,
        hand_size=6,
        max_sampling_attempts=200_000,
        max_near_miss_attempts_per_seed=200,
        use_near_miss_negatives=True,
        near_miss_positions_to_try=4,
        allow_random_negative_fallback=True,
        require_exact_balance=True,
    )

    print(f"Configuration:")
    print(f"  Training examples per task: {config.n_training_positives + config.n_training_negatives}")
    print(f"  Holdout examples per task: {config.n_holdout_positives + config.n_holdout_negatives}")
    print(f"  Near-miss negatives: {config.use_near_miss_negatives}")
    print(f"  Max sampling attempts: {config.max_sampling_attempts:,}")
    print()

    # ================================================================
    # 1. Generate Catalogue Tasks
    # ================================================================
    print("-" * 70)
    print("1. CATALOGUE RULES (45 rules)")
    print("-" * 70)

    catalogue_output = output_dir / "catalogue_tasks.json"
    catalogue_tasks, catalogue_dataset = generate_and_save_dataset(
        rules=CATALOGUE_RULES,
        output_path=catalogue_output,
        config=config,
        seed=42,
        rule_source="rules/catalogue.py - ALL_RULES"
    )

    # ================================================================
    # 2. Generate Pretraining Tasks
    # ================================================================
    print()
    print("-" * 70)
    print("2. PRETRAINING RULES (44 rules)")
    print("-" * 70)

    pretraining_output = output_dir / "pretraining_tasks.json"
    pretraining_tasks, pretraining_dataset = generate_and_save_dataset(
        rules=PRETRAINING_RULES,
        output_path=pretraining_output,
        config=config,
        seed=1000,  # Different seed for variety
        rule_source="rules/pretraining_rules.py - PRETRAINING_RULES"
    )

    # ================================================================
    # 3. Combined Dataset (for experiments using both)
    # ================================================================
    print()
    print("-" * 70)
    print("3. COMBINED DATASET")
    print("-" * 70)

    # Combine unique rules (some may overlap)
    all_rules = list(CATALOGUE_RULES) + list(PRETRAINING_RULES)
    seen_ids = set()
    unique_rules = []
    for rule in all_rules:
        if rule.id not in seen_ids:
            seen_ids.add(rule.id)
            unique_rules.append(rule)

    print(f"Total rules: {len(all_rules)}")
    print(f"Unique rules: {len(unique_rules)}")

    combined_output = output_dir / "combined_tasks.json"
    combined_tasks, combined_dataset = generate_and_save_dataset(
        rules=unique_rules,
        output_path=combined_output,
        config=config,
        seed=2000,
        rule_source="Combined: catalogue.py + pretraining_rules.py (deduplicated)"
    )

    # ================================================================
    # Summary
    # ================================================================
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Catalogue tasks: {len(catalogue_tasks)}/45")
    print(f"Pretraining tasks: {len(pretraining_tasks)}/44")
    print(f"Combined unique tasks: {len(combined_tasks)}/{len(unique_rules)}")
    print()
    print("Output files:")
    print(f"  {catalogue_output}")
    print(f"  {pretraining_output}")
    print(f"  {combined_output}")
    print()

    # ================================================================
    # Verification: Print sample tasks for review
    # ================================================================
    print("=" * 70)
    print("SAMPLE TASKS FOR REVIEW")
    print("=" * 70)

    # Show first 5 catalogue tasks
    print("\nCatalogue Tasks (first 5):")
    for i, task in enumerate(catalogue_tasks[:5]):
        pos_count = sum(1 for _, l in task.examples if l)
        neg_count = sum(1 for _, l in task.examples if not l)
        holdout_pos = sum(1 for _, l in task.holdout if l)
        holdout_neg = sum(1 for _, l in task.holdout if not l)
        print(f"  {i+1}. {task.name}")
        print(f"     Training: {pos_count}+ / {neg_count}- = {len(task.examples)} total")
        print(f"     Holdout:  {holdout_pos}+ / {holdout_neg}- = {len(task.holdout)} total")

    # Show any failures
    catalogue_failures = catalogue_dataset.metadata.get('failures', [])
    if catalogue_failures:
        print(f"\nCatalogue Failures ({len(catalogue_failures)}):")
        for failure in catalogue_failures:
            print(f"  - {failure['rule_id']}: {failure['reason'][:80]}...")

    pretraining_failures = pretraining_dataset.metadata.get('failures', [])
    if pretraining_failures:
        print(f"\nPretraining Failures ({len(pretraining_failures)}):")
        for failure in pretraining_failures:
            print(f"  - {failure['rule_id']}: {failure['reason'][:80]}...")

    print()
    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)

    return catalogue_tasks, pretraining_tasks, combined_tasks


if __name__ == "__main__":
    main()
