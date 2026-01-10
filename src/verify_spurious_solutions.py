#!/usr/bin/env python3
"""
Verification script to check if disputed tasks have spurious solutions.

This script:
1. Loads disputed tasks with their training and holdout examples
2. Enumerates programs to find solutions
3. Checks if solutions pass training but fail holdout (spurious)
"""

import json
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dreamcoder_core.lean_primitives import build_lean_primitives, build_lean_grammar
from dreamcoder_core.enumeration import TopDownEnumerator
from dreamcoder_core.type_system import HAND, BOOL, arrow
from rules.cards import Card

# Disputed tasks identified from the two studies
DISPUTED_PRETRAINING = [
    'sol_alternating',
    'simple_all_even',
    'simple_all_odd',
    'comp_skip1_same_color',
    'comp_shift2_color',
    'comp_color_pairs'
]

DISPUTED_CATALOGUE = [
    'Odd_opens_next_closes',
    'Halves_copy_colors',
    'Halves_copy_ranks',
    'Only_one_odd_rank',
    'Uniform_rank_parity',
    'Even_opens_next_closes'
]


def load_prerecorded_tasks(task_file: str) -> Dict:
    """Load tasks from prerecorded JSON file."""
    with open(task_file) as f:
        data = json.load(f)
    return {task['name']: task for task in data['tasks']}


def parse_hand(hand_json: list) -> tuple:
    """Convert JSON hand to tuple of Card objects."""
    cards = []
    for card_json in hand_json:
        card = Card(card_json['suit'], card_json['rank'])
        cards.append(card)
    return tuple(cards)


def parse_examples(examples_json: list) -> list:
    """Parse examples from JSON format to (hand, label) tuples."""
    result = []
    for ex in examples_json:
        hand = parse_hand(ex['hand'])
        label = ex['label']
        result.append((hand, label))
    return result


def check_program_on_examples(program_fn, examples: list) -> Tuple[int, int]:
    """
    Check how many examples the program gets correct.
    Returns (correct, total).
    """
    correct = 0
    total = len(examples)

    for hand, expected in examples:
        try:
            result = program_fn(hand)
            if result == expected:
                correct += 1
        except Exception:
            pass  # Program failed on this example

    return correct, total


def verify_task(task_name: str, task_data: Dict, budget: int = 500_000, verbose: bool = True) -> Dict:
    """
    Verify a single task by enumerating programs and checking against holdout.

    Returns dict with:
        - task_name
        - training_examples: count
        - holdout_examples: count
        - first_training_solution: (program_str, index)
        - first_holdout_solution: (program_str, index) or None
        - is_spurious: True if training solution fails holdout
    """
    # Parse examples
    training = parse_examples(task_data['training_examples'])
    holdout = parse_examples(task_data['holdout_examples'])

    if verbose:
        print(f"\n{'='*60}")
        print(f"Task: {task_name}")
        print(f"Training examples: {len(training)}")
        print(f"Holdout examples: {len(holdout)}")

    # Build grammar
    grammar = build_lean_grammar()

    # Request type: HAND -> BOOL
    request_type = arrow(HAND, BOOL)

    # Create enumerator
    enumerator = TopDownEnumerator(grammar, max_depth=8, max_programs=budget)

    # Enumerate programs
    first_training_solution = None
    first_holdout_solution = None
    programs_tried = 0

    for program_obj, log_prob in enumerator.enumerate(request_type, timeout_seconds=600.0):
        programs_tried += 1

        if programs_tried > 0 and programs_tried % 100_000 == 0 and verbose:
            print(f"  ... enumerated {programs_tried:,} programs")

        # Build lambda function
        try:
            program_fn = program_obj.to_lambda()
        except Exception:
            continue

        # Check on training
        train_correct, train_total = check_program_on_examples(program_fn, training)

        if train_correct == train_total:
            # Found a program that passes ALL training examples
            if first_training_solution is None:
                first_training_solution = (str(program_obj), programs_tried)
                if verbose:
                    print(f"\n  FIRST TRAINING SOLUTION at #{programs_tried:,}:")
                    print(f"    Program: {program_obj}")

                # Check on holdout
                hold_correct, hold_total = check_program_on_examples(program_fn, holdout)

                if verbose:
                    print(f"    Training: {train_correct}/{train_total} (PASS)")
                    print(f"    Holdout:  {hold_correct}/{hold_total} {'(PASS)' if hold_correct == hold_total else '(FAIL - SPURIOUS!)'}")

                if hold_correct == hold_total:
                    first_holdout_solution = (str(program_obj), programs_tried)
                    break  # Found a genuine solution, done

    # If we found a training solution but no holdout solution, it's spurious
    is_spurious = first_training_solution is not None and first_holdout_solution is None

    result = {
        'task_name': task_name,
        'n_training': len(training),
        'n_holdout': len(holdout),
        'first_training_solution': first_training_solution,
        'first_holdout_solution': first_holdout_solution,
        'is_spurious': is_spurious,
        'programs_enumerated': programs_tried
    }

    if verbose:
        print(f"\n  VERDICT: {'SPURIOUS' if is_spurious else 'GENUINE' if first_holdout_solution else 'NO SOLUTION FOUND'}")

    return result


def main():
    print("="*60)
    print("SPURIOUS SOLUTION VERIFICATION")
    print("="*60)

    # Load task files
    data_dir = Path(__file__).parent / 'data' / 'prerecorded_tasks'
    pretraining_tasks = load_prerecorded_tasks(data_dir / 'pretraining_tasks.json')
    catalogue_tasks = load_prerecorded_tasks(data_dir / 'catalogue_tasks.json')

    # Results storage
    all_results = {
        'pretraining': [],
        'catalogue': []
    }

    # Verify disputed pretraining tasks
    print("\n\n" + "="*60)
    print("PART 1: DISPUTED PRETRAINING TASKS")
    print("="*60)

    for task_name in DISPUTED_PRETRAINING:
        if task_name in pretraining_tasks:
            result = verify_task(task_name, pretraining_tasks[task_name], budget=500_000)
            all_results['pretraining'].append(result)
        else:
            print(f"WARNING: Task {task_name} not found in pretraining_tasks.json")

    # Verify disputed catalogue tasks
    print("\n\n" + "="*60)
    print("PART 2: DISPUTED CATALOGUE TASKS")
    print("="*60)

    for task_name in DISPUTED_CATALOGUE:
        if task_name in catalogue_tasks:
            result = verify_task(task_name, catalogue_tasks[task_name], budget=500_000)
            all_results['catalogue'].append(result)
        else:
            print(f"WARNING: Task {task_name} not found in catalogue_tasks.json")

    # Summary
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\nPRETRAINING TASKS:")
    spurious_pre = sum(1 for r in all_results['pretraining'] if r['is_spurious'])
    genuine_pre = sum(1 for r in all_results['pretraining'] if r['first_holdout_solution'])
    no_solution_pre = sum(1 for r in all_results['pretraining'] if not r['first_training_solution'])
    print(f"  Spurious: {spurious_pre}/{len(DISPUTED_PRETRAINING)}")
    print(f"  Genuine:  {genuine_pre}/{len(DISPUTED_PRETRAINING)}")
    print(f"  No solution: {no_solution_pre}/{len(DISPUTED_PRETRAINING)}")

    for r in all_results['pretraining']:
        status = "SPURIOUS" if r['is_spurious'] else "GENUINE" if r['first_holdout_solution'] else "NO SOLUTION"
        print(f"    {r['task_name']}: {status}")
        if r['first_training_solution']:
            print(f"      Program: {r['first_training_solution'][0]}")

    print("\nCATALOGUE TASKS:")
    spurious_cat = sum(1 for r in all_results['catalogue'] if r['is_spurious'])
    genuine_cat = sum(1 for r in all_results['catalogue'] if r['first_holdout_solution'])
    no_solution_cat = sum(1 for r in all_results['catalogue'] if not r['first_training_solution'])
    print(f"  Spurious: {spurious_cat}/{len(DISPUTED_CATALOGUE)}")
    print(f"  Genuine:  {genuine_cat}/{len(DISPUTED_CATALOGUE)}")
    print(f"  No solution: {no_solution_cat}/{len(DISPUTED_CATALOGUE)}")

    for r in all_results['catalogue']:
        status = "SPURIOUS" if r['is_spurious'] else "GENUINE" if r['first_holdout_solution'] else "NO SOLUTION"
        print(f"    {r['task_name']}: {status}")
        if r['first_training_solution']:
            print(f"      Program: {r['first_training_solution'][0]}")

    # Save results
    output_file = Path(__file__).parent / 'spurious_verification_results.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
