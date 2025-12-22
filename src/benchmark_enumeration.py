#!/usr/bin/env python3
"""
Benchmark script for comparing enumeration strategies.

Tests the TopDownEnumerator with predictive depth pruning
at different max_depth values to measure the depth scaling fix.

This measures whether the predictive pruning fix (Option B)
resolves the partial program explosion problem.
"""

import sys
import time
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))

from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.enumeration import TopDownEnumerator
from dreamcoder_core.type_system import arrow, HAND, BOOL


def benchmark_enumeration(
    grammar,
    request_type,
    max_depth: int,
    timeout: float = 30.0,
    max_programs: int = 50000,
    use_cost_bands: bool = False,
    use_memoized: bool = False
) -> dict:
    """
    Benchmark enumeration with given parameters.

    Returns dict with timing and counts.
    """
    enumerator = TopDownEnumerator(
        grammar=grammar,
        max_depth=max_depth,
        max_programs=max_programs
    )

    start_time = time.time()

    if use_memoized:
        programs = list(enumerator.enumerate_memoized(
            request_type,
            max_cost=50.0,
            timeout_seconds=timeout,
            depth_limit=max_depth
        ))
        memo_stats = enumerator.get_memo_stats()
    elif use_cost_bands:
        programs = list(enumerator.enumerate_with_cost_bands(
            request_type,
            max_cost=50.0,
            timeout_seconds=timeout,
            initial_budget=12.0,
            budget_increment=1.5
        ))
        memo_stats = None
    else:
        programs = list(enumerator.enumerate(
            request_type,
            max_cost=50.0,
            timeout_seconds=timeout
        ))
        memo_stats = None

    elapsed = time.time() - start_time

    result = {
        'max_depth': max_depth,
        'use_cost_bands': use_cost_bands,
        'use_memoized': use_memoized,
        'programs_found': len(programs),
        'partial_explored': enumerator.partial_programs_explored,
        'elapsed_seconds': elapsed,
        'programs_per_second': len(programs) / elapsed if elapsed > 0 else 0,
        'efficiency': len(programs) / enumerator.partial_programs_explored if enumerator.partial_programs_explored > 0 else 0
    }

    if memo_stats:
        result['memo_cache_size'] = memo_stats['cache_size']
        result['memo_hit_rate'] = memo_stats['hit_rate']

    return result


def print_result(result: dict):
    """Pretty print a benchmark result."""
    if result.get('use_memoized'):
        method = "Memoized"
    elif result['use_cost_bands']:
        method = "Cost Bands"
    else:
        method = "Single Pass"

    print(f"  {method} @ depth {result['max_depth']}:")
    print(f"    Programs found:   {result['programs_found']:,}")
    print(f"    Partials explored: {result['partial_explored']:,}")
    print(f"    Time:             {result['elapsed_seconds']:.2f}s")
    print(f"    Rate:             {result['programs_per_second']:.1f} prog/s")
    print(f"    Efficiency:       {result['efficiency']*100:.2f}%")

    if result.get('memo_cache_size') is not None:
        print(f"    Cache size:       {result['memo_cache_size']:,}")
        print(f"    Cache hit rate:   {result['memo_hit_rate']*100:.1f}%")

    print()


def main():
    print("=" * 70)
    print("ENUMERATION BENCHMARK: Original vs Memoized (Option C)")
    print("=" * 70)
    print()

    # Build grammar
    print("Building grammar...")
    grammar = build_lean_grammar()
    print(f"  Grammar size: {len(list(grammar.primitives()))} primitives")
    print()

    # Request type: HAND -> BOOL (card game predicates)
    request_type = arrow(HAND, BOOL)

    # Test parameters
    timeout = 30.0  # seconds per test
    max_programs = 50000

    results = []

    for max_depth in [6, 7, 8]:
        print(f"\n--- Testing max_depth = {max_depth} ---")

        # Test original (queue-based)
        print("Testing original (priority queue)...")
        result_original = benchmark_enumeration(
            grammar, request_type, max_depth,
            timeout=timeout, max_programs=max_programs,
            use_cost_bands=False, use_memoized=False
        )
        results.append(result_original)
        print_result(result_original)

        # Test memoized (Option C - DreamCoder style)
        print("Testing MEMOIZED (Option C - DreamCoder style)...")
        result_memo = benchmark_enumeration(
            grammar, request_type, max_depth,
            timeout=timeout, max_programs=max_programs,
            use_cost_bands=False, use_memoized=True
        )
        results.append(result_memo)
        print_result(result_memo)

        # Compare
        if result_original['programs_per_second'] > 0:
            speedup = result_memo['programs_per_second'] / result_original['programs_per_second']
            print(f"  Speedup (memoized / original): {speedup:.2f}x")

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Depth':<6} {'Method':<12} {'Programs':<10} {'Partials':<12} {'Time':<8} {'Rate':<10} {'Efficiency'}")
    print("-" * 70)

    for r in results:
        if r.get('use_memoized'):
            method = "Memoized"
        elif r['use_cost_bands']:
            method = "Bands"
        else:
            method = "Original"
        print(f"{r['max_depth']:<6} {method:<12} {r['programs_found']:<10,} {r['partial_explored']:<12,} {r['elapsed_seconds']:<8.2f} {r['programs_per_second']:<10.1f} {r['efficiency']*100:.2f}%")

    print()
    print("Key insight: If memoized shows better DEPTH SCALING (less slowdown at depth 7/8),")
    print("            then the DreamCoder-style caching is working!")


if __name__ == '__main__':
    main()
