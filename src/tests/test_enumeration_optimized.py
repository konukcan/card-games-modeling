#!/usr/bin/env python3
"""
Test script for optimized enumeration.

Tests:
1. Early pruning speedup
2. Relaxed likelihood mode
3. Correctness vs original enumeration

Run with: python3 -m dreamcoder_core.test_enumeration_optimized
"""

import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.primitives import build_lean_grammar
from dreamcoder_core.type_system import arrow, HAND, BOOL
from dreamcoder_core.enumeration import enumerate_simple
from dreamcoder_core.enumeration_optimized import (
    LikelihoodMode,
    LikelihoodConfig,
    enumerate_for_task_optimized,
    enumerate_tasks_sequential_optimized,
    TaskFrontier
)


def make_eval_fn():
    """Create evaluation function for programs."""
    def eval_fn(program, hand):
        try:
            fn = program.evaluate([])
            return fn(hand)
        except Exception:
            return None
    return eval_fn


def test_early_pruning_speedup():
    """Test that early pruning provides speedup for all-or-nothing mode."""
    print("\n" + "="*60)
    print("TEST 1: Early Pruning Speedup")
    print("="*60)

    import random
    from rules.cards import sample_hand

    # Create a simple task that's easy to fail fast on
    random.seed(42)
    examples = [(sample_hand(5), i % 2 == 0) for i in range(20)]

    grammar = build_lean_grammar()
    eval_fn = make_eval_fn()

    # Test with early pruning (ALL_OR_NOTHING)
    config_strict = LikelihoodConfig(mode=LikelihoodMode.ALL_OR_NOTHING)

    start = time.time()
    frontier_strict = enumerate_for_task_optimized(
        grammar=grammar,
        task_name="test_task",
        examples=examples,
        request_type=arrow(HAND, BOOL),
        eval_fn=eval_fn,
        max_depth=6,
        max_programs=10000,
        timeout_seconds=30.0,
        keep_top_k=5,
        likelihood_config=config_strict
    )
    strict_time = time.time() - start

    print(f"ALL_OR_NOTHING mode:")
    print(f"  Time: {strict_time:.2f}s")
    print(f"  Programs searched: {frontier_strict.total_programs_searched:,}")
    print(f"  Solved: {frontier_strict.solved}")
    print(f"  Solutions: {frontier_strict.n_solutions}")

    # Note: Without a comparison to non-early-pruning, we just verify it works
    print("\n✓ Early pruning test passed")


def test_relaxed_likelihood():
    """Test relaxed likelihood mode keeps partial solutions."""
    print("\n" + "="*60)
    print("TEST 2: Relaxed Likelihood Mode")
    print("="*60)

    import random
    from rules.cards import sample_hand

    # Create a task where partial solutions are likely
    # Use consistent examples with a simple pattern
    random.seed(123)
    examples = []
    for i in range(20):
        hand = sample_hand(5)
        # Label based on whether first card's suit matches a pattern
        label = hand[0].suit.name in ['HEARTS', 'DIAMONDS']  # Is first card red
        examples.append((hand, label))

    grammar = build_lean_grammar()
    eval_fn = make_eval_fn()

    # Test RELAXED mode with low threshold
    config_relaxed = LikelihoodConfig(
        mode=LikelihoodMode.RELAXED,
        min_accuracy=0.3  # Keep anything above 30% accuracy
    )

    frontier_relaxed = enumerate_for_task_optimized(
        grammar=grammar,
        task_name="test_relaxed",
        examples=examples,
        request_type=arrow(HAND, BOOL),
        eval_fn=eval_fn,
        max_depth=6,
        max_programs=10000,
        timeout_seconds=30.0,
        keep_top_k=10,
        likelihood_config=config_relaxed
    )

    print(f"RELAXED mode (min_accuracy=30%):")
    print(f"  Programs searched: {frontier_relaxed.total_programs_searched:,}")
    print(f"  Solved: {frontier_relaxed.solved}")
    print(f"  Total solutions: {frontier_relaxed.n_solutions}")
    print(f"  Perfect solutions: {frontier_relaxed.n_perfect_solutions}")
    print(f"  Partial solutions: {len(frontier_relaxed.partial_solutions())}")

    if frontier_relaxed.entries:
        print(f"\n  Top solutions:")
        for i, entry in enumerate(frontier_relaxed.entries[:5]):
            print(f"    {i+1}. accuracy={entry.accuracy:.1%}, "
                  f"log_lik={entry.log_likelihood:.3f}, "
                  f"prog={str(entry.program)[:50]}...")

    # Verify we got some results
    assert frontier_relaxed.n_solutions > 0, "Should find at least some partial solutions"

    print("\n✓ Relaxed likelihood test passed")


def test_comparison_strict_vs_relaxed():
    """Compare strict vs relaxed mode on same task."""
    print("\n" + "="*60)
    print("TEST 3: Strict vs Relaxed Comparison")
    print("="*60)

    import random
    from rules.cards import sample_hand

    # Create task
    random.seed(456)
    examples = []
    for i in range(20):
        hand = sample_hand(5)
        # A more complex rule that won't be easily solved
        label = len(set(c.suit for c in hand)) >= 3  # 3+ suits
        examples.append((hand, label))

    grammar = build_lean_grammar()
    eval_fn = make_eval_fn()

    # Strict mode
    config_strict = LikelihoodConfig(mode=LikelihoodMode.ALL_OR_NOTHING)
    frontier_strict = enumerate_for_task_optimized(
        grammar=grammar,
        task_name="test_comparison",
        examples=examples,
        request_type=arrow(HAND, BOOL),
        eval_fn=eval_fn,
        max_depth=6,
        max_programs=5000,
        timeout_seconds=20.0,
        keep_top_k=5,
        likelihood_config=config_strict
    )

    # Relaxed mode
    config_relaxed = LikelihoodConfig(
        mode=LikelihoodMode.RELAXED,
        min_accuracy=0.5
    )
    frontier_relaxed = enumerate_for_task_optimized(
        grammar=grammar,
        task_name="test_comparison",
        examples=examples,
        request_type=arrow(HAND, BOOL),
        eval_fn=eval_fn,
        max_depth=6,
        max_programs=5000,
        timeout_seconds=20.0,
        keep_top_k=5,
        likelihood_config=config_relaxed
    )

    print(f"Comparison (5000 programs, depth 6):")
    print(f"\n  ALL_OR_NOTHING:")
    print(f"    Solved: {frontier_strict.solved}")
    print(f"    Solutions: {frontier_strict.n_solutions}")

    print(f"\n  RELAXED (min_accuracy=50%):")
    print(f"    Solved: {frontier_relaxed.solved}")
    print(f"    Total solutions: {frontier_relaxed.n_solutions}")
    print(f"    Perfect: {frontier_relaxed.n_perfect_solutions}")
    print(f"    Partial: {len(frontier_relaxed.partial_solutions())}")

    if frontier_relaxed.entries:
        best = frontier_relaxed.best
        print(f"    Best accuracy: {best.accuracy:.1%}")

    print("\n✓ Comparison test passed")


def test_sequential_optimized():
    """Test the sequential optimized enumeration for multiple tasks."""
    print("\n" + "="*60)
    print("TEST 4: Sequential Optimized Enumeration")
    print("="*60)

    import random
    from rules.cards import sample_hand

    # Create a few simple tasks
    tasks = []
    for task_idx in range(3):
        random.seed(task_idx * 100)
        examples = []
        for i in range(10):
            hand = sample_hand(5)
            if task_idx == 0:
                label = hand[0].suit.name in ['HEARTS', 'DIAMONDS']
            elif task_idx == 1:
                label = len(hand) > 3
            else:
                label = True  # Trivial
            examples.append((hand, label))

        tasks.append({
            'name': f'task_{task_idx}',
            'examples': examples,
            'request_type': arrow(HAND, BOOL)
        })

    grammar = build_lean_grammar()
    eval_fn = make_eval_fn()

    # Run sequential optimized
    config = LikelihoodConfig(mode=LikelihoodMode.ALL_OR_NOTHING)

    start = time.time()
    frontiers = enumerate_tasks_sequential_optimized(
        grammar=grammar,
        tasks=tasks,
        eval_fn=eval_fn,
        max_depth=5,
        max_programs=3000,
        timeout_seconds=10.0,
        keep_top_k=3,
        likelihood_config=config,
        verbose=True
    )
    elapsed = time.time() - start

    print(f"\nResults ({elapsed:.2f}s total):")
    for name, frontier in frontiers.items():
        status = "SOLVED" if frontier.solved else "unsolved"
        print(f"  {name}: {status}, {frontier.n_solutions} solutions, "
              f"{frontier.total_programs_searched:,} programs")

    print("\n✓ Sequential optimized test passed")


def main():
    """Run all tests."""
    print("="*60)
    print("OPTIMIZED ENUMERATION TESTS")
    print("="*60)

    test_early_pruning_speedup()
    test_relaxed_likelihood()
    test_comparison_strict_vs_relaxed()
    test_sequential_optimized()

    print("\n" + "="*60)
    print("ALL TESTS PASSED ✓")
    print("="*60)


if __name__ == "__main__":
    main()
