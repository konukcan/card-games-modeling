#!/usr/bin/env python3
"""
Quick validation script to find and print actual program solutions.
This validates that solutions are not spurious (like λ false).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.task_generation import load_prerecorded_tasks
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.enumeration import TopDownEnumerator
from dreamcoder_core.program import Program
from dreamcoder_core.type_system import arrow, HAND, BOOL
from typing import Optional


def eval_program(program: Program, hand) -> Optional[bool]:
    """Evaluate a program on a hand."""
    try:
        fn = program.evaluate([])
        result = fn(hand)
        return bool(result) if result is not None else None
    except Exception:
        return None


def find_solution_for_task(task, grammar, max_programs: int = 200_000) -> Optional[Program]:
    """Find a solution for a task."""
    target_type = arrow(HAND, BOOL)
    enumerator = TopDownEnumerator(grammar, max_depth=8, max_programs=max_programs)

    programs_tried = 0
    for program, _ in enumerator.enumerate(target_type):
        if programs_tried >= max_programs:
            break
        programs_tried += 1

        # Check all examples
        all_correct = True
        for hand, label in task.examples:
            result = eval_program(program, hand)
            if result is None or result != label:
                all_correct = False
                break

        if all_correct:
            return program

    return None


def main():
    print("=" * 70)
    print("SOLUTION VALIDATION: Checking actual program solutions")
    print("=" * 70)

    # Load tasks
    tasks_path = Path(__file__).parent.parent / "data" / "prerecorded_tasks" / "pretraining_tasks.json"
    tasks = load_prerecorded_tasks(tasks_path)
    print(f"\nLoaded {len(tasks)} tasks")

    # Create grammar
    grammar = build_lean_grammar()
    print(f"Grammar has {len(grammar.productions)} primitives")

    # Select a few tasks to validate - mix of easy and harder ones
    tasks_to_check = [
        "poker_flush",           # Should be easy (check all same suit)
        "poker_has_pair",        # Common pattern (any pair)
        "simple_first_red",      # Simple color check
        "bj_sum_even",           # Requires sum computation
        "rummy_all_different",   # Requires uniqueness check
    ]

    print("\n" + "-" * 70)
    print("FINDING SOLUTIONS (this may take a minute per task)")
    print("-" * 70)

    found_solutions = {}

    for task_name in tasks_to_check:
        # Find the task
        task = next((t for t in tasks if t.name == task_name), None)
        if not task:
            print(f"\n⚠ Task '{task_name}' not found")
            continue

        # Show task details
        pos_count = sum(1 for _, l in task.examples if l)
        neg_count = sum(1 for _, l in task.examples if not l)
        print(f"\n📋 {task_name}: {pos_count}+/{neg_count}- examples")

        # Find solution
        print(f"   Searching (up to 200k programs)...")
        solution = find_solution_for_task(task, grammar, max_programs=200_000)

        if solution:
            found_solutions[task_name] = solution
            program_str = str(solution)
            print(f"   ✅ SOLUTION: {program_str}")

            # Check if it's the trivial (λ false) or (λ true)
            if program_str.strip() in ["(λ false)", "(λ true)", "false", "true"]:
                print(f"   ⚠️  WARNING: Trivial solution detected!")
        else:
            print(f"   ❌ No solution found in 200k programs")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nChecked {len(tasks_to_check)} tasks:")
    print(f"  - Found solutions: {len(found_solutions)}")

    print("\n📦 SOLUTIONS FOUND:")
    for name, prog in found_solutions.items():
        prog_str = str(prog)
        trivial = "⚠️ TRIVIAL" if prog_str.strip() in ["(λ false)", "(λ true)", "false", "true"] else "✓"
        print(f"  {name}: {prog_str} {trivial}")

    # Validate solutions are not trivial
    trivial_count = sum(
        1 for p in found_solutions.values()
        if str(p).strip() in ["(λ false)", "(λ true)", "false", "true"]
    )

    if trivial_count > 0:
        print(f"\n❌ VALIDATION FAILED: {trivial_count} trivial solutions found!")
        return False
    elif len(found_solutions) == 0:
        print(f"\n⚠️  No solutions found to validate")
        return False
    else:
        print(f"\n✅ VALIDATION PASSED: All solutions are non-trivial")
        return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
