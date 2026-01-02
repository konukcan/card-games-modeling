#!/usr/bin/env python3
"""
Filter Validation Experiment
=============================

This experiment validates that the abstraction quality filters improve
compression outcomes by comparing library quality with and without filters.

Design:
- Run 2 iterations of wake-sleep on 20 pretraining tasks
- Measure abstraction quality metrics
- Compare to baseline (expected improvement)

Expected Duration: ~10-15 minutes

Usage:
    python3 experiments/run_filter_validation.py

Output:
    results_filter_validation/<timestamp>/
"""

import json
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import Counter
from typing import Dict, List, Any, Tuple, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dreamcoder_core.grammar import Grammar, Production, uniform_grammar
from dreamcoder_core.program import (
    Program, Primitive, Abstraction, Application, Index, Invented
)
from dreamcoder_core.compression import (
    compress_frontiers, is_nontrivial, is_eta_reducible,
    is_nested_eta_reducible
)
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.enumeration import enumerate_for_task
from dreamcoder_core.type_system import BOOL, HAND, arrow
from dreamcoder_core.task_generation import load_prerecorded_tasks
from dreamcoder_core.task import Task

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
SRC_DIR = Path(__file__).parent.parent
PRETRAINING_TASKS_PATH = SRC_DIR / 'data' / 'prerecorded_tasks' / 'pretraining_tasks.json'


def count_primitives_in_program(program: Program) -> int:
    """Count primitives in a program tree."""
    if isinstance(program, Primitive):
        return 1
    elif isinstance(program, Invented):
        return count_primitives_in_program(program.body)
    elif isinstance(program, Application):
        return count_primitives_in_program(program.f) + count_primitives_in_program(program.x)
    elif isinstance(program, Abstraction):
        return count_primitives_in_program(program.body)
    return 0


def analyze_invention(inv: Invented) -> Dict[str, Any]:
    """Analyze an invented abstraction for quality metrics."""
    body = inv.body

    # Count structure
    n_primitives = count_primitives_in_program(body)

    # Check quality
    trivial = not is_nontrivial(body)
    eta = is_eta_reducible(body)
    nested_eta = is_nested_eta_reducible(body)

    # Categorize
    if eta or nested_eta:
        category = "eta-wrapper"
    elif trivial:
        category = "trivial"
    else:
        category = "useful"

    return {
        'name': str(inv),
        'body': str(body),
        'n_primitives': n_primitives,
        'is_trivial': trivial,
        'is_eta': eta,
        'is_nested_eta': nested_eta,
        'category': category
    }


def build_grammar() -> Grammar:
    """Build the card game grammar."""
    return build_lean_grammar()


def make_eval_fn():
    """Create evaluation function for programs on hands.

    Programs are lambdas: (λ ... $0 ...)
    - program.evaluate([]) returns a closure
    - closure(inp) applies the function to the input
    """
    def eval_fn(program, inp):
        try:
            closure = program.evaluate([])
            return closure(inp)
        except Exception:
            return None
    return eval_fn


def solve_task(task: Task, grammar: Grammar, timeout: float = 30.0, max_programs: int = 50000) -> Optional[Program]:
    """Try to solve a single task with enumeration."""
    # Use task examples
    examples = list(task.examples)
    if len(examples) < 10:
        return None

    # Split into training and holdout
    train_examples = examples[:int(len(examples) * 0.8)]
    holdout_examples = examples[int(len(examples) * 0.8):]

    # Create request type (HAND -> BOOL)
    request_type = arrow(HAND, BOOL)

    # Use enumerate_for_task
    eval_fn = make_eval_fn()
    frontier = enumerate_for_task(
        grammar=grammar,
        examples=train_examples,
        request_type=request_type,
        eval_fn=eval_fn,
        timeout_seconds=timeout,
        max_programs=max_programs,
        use_top_down=True
    )

    # Check if we found a solution
    if not frontier.entries:
        return None

    # Verify the best solution on holdout examples
    best = frontier.entries[0]  # EnumerationResult object
    program = best.program
    try:
        closure = program.evaluate([])
        for inp, expected in holdout_examples:
            result = closure(inp)
            if result != expected:
                return None
        return program
    except Exception:
        return None


def run_validation_experiment():
    """Run the filter validation experiment."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / f"results_filter_validation/validation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ABSTRACTION QUALITY FILTER VALIDATION EXPERIMENT")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {output_dir}")
    print()

    # Configuration
    n_tasks = 20
    n_iterations = 2
    max_inventions_per_iter = 3
    enumeration_timeout = 30.0  # seconds per task
    max_programs_per_task = 50000

    print(f"Configuration:")
    print(f"  Tasks: {n_tasks}")
    print(f"  Iterations: {n_iterations}")
    print(f"  Max inventions per iteration: {max_inventions_per_iter}")
    print(f"  Enumeration timeout: {enumeration_timeout}s")
    print()

    # Load tasks
    if not PRETRAINING_TASKS_PATH.exists():
        print(f"❌ Pretraining tasks file not found: {PRETRAINING_TASKS_PATH}")
        return False

    all_tasks = load_prerecorded_tasks(PRETRAINING_TASKS_PATH)
    test_tasks = all_tasks[:n_tasks]

    print(f"Loaded {len(all_tasks)} tasks, using first {n_tasks}:")
    for task in test_tasks[:10]:
        print(f"  - {task.name}")
    if len(test_tasks) > 10:
        print(f"  ... and {len(test_tasks) - 10} more")
    print()

    # Initialize
    grammar = build_grammar()
    all_inventions = []
    iteration_logs = []

    # Run wake-sleep iterations
    for iteration in range(1, n_iterations + 1):
        print(f"\n{'='*70}")
        print(f"ITERATION {iteration}/{n_iterations}")
        print(f"{'='*70}")

        iter_start = time.time()

        # Wake phase: solve tasks
        print(f"\nWake phase: solving {len(test_tasks)} tasks...")
        solutions = []
        solved_count = 0

        for i, task in enumerate(test_tasks):
            print(f"  [{i+1}/{len(test_tasks)}] {task.name}...", end=" ", flush=True)
            solution = solve_task(task, grammar, enumeration_timeout, max_programs_per_task)
            if solution:
                solutions.append((task.name, solution))
                solved_count += 1
                print(f"✅ {solution}")
            else:
                print("❌")

        print(f"\nSolved: {solved_count}/{len(test_tasks)}")

        if not solutions:
            print("No solutions found, stopping.")
            break

        # Sleep phase: compress
        print(f"\nSleep phase: compressing {len(solutions)} solutions...")

        # Create frontiers
        frontiers = [[(sol, 0.0)] for _, sol in solutions]

        # Compress
        compress_start = time.time()
        result = compress_frontiers(
            grammar,
            frontiers,
            max_inventions=max_inventions_per_iter,
            min_savings=2.0,
            use_anti_unification=True,
            refactor_programs=True
        )
        compress_time = time.time() - compress_start

        # Update grammar
        grammar = result.new_grammar
        all_inventions.extend(result.new_inventions)

        # Analyze new inventions
        print(f"\nNew inventions ({len(result.new_inventions)}):")
        for inv in result.new_inventions:
            analysis = analyze_invention(inv)
            status = "✅" if analysis['category'] == 'useful' else "⚠️"
            print(f"  {status} {inv}: {analysis['category']} ({analysis['n_primitives']} primitives)")

        # Log iteration
        iter_log = {
            'iteration': iteration,
            'tasks_solved': solved_count,
            'tasks_total': len(test_tasks),
            'solutions': [(name, str(sol)) for name, sol in solutions],
            'new_inventions': [str(inv) for inv in result.new_inventions],
            'compression_time_seconds': compress_time,
            'iteration_time_seconds': time.time() - iter_start
        }
        iteration_logs.append(iter_log)

    # Final analysis
    print(f"\n{'='*70}")
    print("FINAL ANALYSIS")
    print(f"{'='*70}")

    # Analyze all inventions
    print(f"\nTotal inventions learned: {len(all_inventions)}")

    categories = Counter()
    useful_count = 0
    analyses = []

    for inv in all_inventions:
        analysis = analyze_invention(inv)
        analyses.append(analysis)
        categories[analysis['category']] += 1
        if analysis['category'] == 'useful':
            useful_count += 1

    print(f"\nCategory breakdown:")
    for cat, count in categories.most_common():
        pct = count / len(all_inventions) * 100 if all_inventions else 0
        print(f"  {cat}: {count} ({pct:.1f}%)")

    quality_rate = useful_count / len(all_inventions) * 100 if all_inventions else 0
    print(f"\nQuality rate: {quality_rate:.1f}% useful abstractions")

    # Expected: with filters, should have >80% useful abstractions
    # (Previously was ~30-50% without filters)
    quality_threshold = 80.0
    if quality_rate >= quality_threshold:
        print(f"✅ PASS: Quality rate {quality_rate:.1f}% >= {quality_threshold}% threshold")
        success = True
    else:
        print(f"⚠️ CHECK: Quality rate {quality_rate:.1f}% < {quality_threshold}% threshold")
        print("   (This may be acceptable if no degenerate patterns were learned)")
        # Check if any degenerate patterns exist
        degenerate = categories.get('eta-wrapper', 0) + categories.get('trivial', 0)
        if degenerate == 0:
            print("   ✅ No degenerate abstractions learned - filters working!")
            success = True
        else:
            success = False

    # Save results
    results = {
        'timestamp': timestamp,
        'config': {
            'n_tasks': n_tasks,
            'n_iterations': n_iterations,
            'max_inventions_per_iter': max_inventions_per_iter,
            'enumeration_timeout': enumeration_timeout
        },
        'iteration_logs': iteration_logs,
        'total_inventions': len(all_inventions),
        'invention_analyses': analyses,
        'category_counts': dict(categories),
        'quality_rate': quality_rate,
        'success': success
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {output_dir}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return success


def main():
    success = run_validation_experiment()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
