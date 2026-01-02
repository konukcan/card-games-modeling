#!/usr/bin/env python3
"""
Test Abstraction Quality Filters
================================

This script validates that the abstraction quality filters (nontrivial, eta-reducible,
single-task) are working correctly and improving library quality.

Test Design:
- Run compression on a medium-sized task set (20 rules)
- Compare before/after: number of degenerate abstractions learned
- Measure: what percentage of learned abstractions are "useful" (contain 2+ primitives)

Expected Results:
- Before filters: ~30-50% of abstractions are degenerate wrappers
- After filters: <5% should be degenerate (ideally 0%)

Usage:
    python3 experiments/test_abstraction_quality.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from collections import Counter

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dreamcoder_core.program import Primitive, Abstraction, Application, Index, Invented
from dreamcoder_core.compression import (
    is_nontrivial, is_eta_reducible,
    is_nested_eta_reducible, passes_abstraction_quality_checks
)


def count_primitives(program) -> int:
    """Count primitives in a program."""
    if isinstance(program, Primitive):
        return 1
    elif isinstance(program, Invented):
        return count_primitives(program.body)
    elif isinstance(program, Application):
        return count_primitives(program.f) + count_primitives(program.x)
    elif isinstance(program, Abstraction):
        return count_primitives(program.body)
    else:
        return 0


def analyze_abstraction(invention: Invented) -> dict:
    """Analyze an abstraction for quality metrics."""
    body = invention.body

    # Count structure
    n_primitives = count_primitives(body)

    # Check our filters
    is_trivial = not is_nontrivial(body)
    is_eta = is_eta_reducible(body)
    is_nested_eta = is_nested_eta_reducible(body)

    # Determine quality category
    if is_eta or is_nested_eta:
        category = "eta-wrapper"
    elif is_trivial:
        category = "trivial"
    else:
        category = "useful"

    return {
        'body': str(body),
        'n_primitives': n_primitives,
        'is_trivial': is_trivial,
        'is_eta_reducible': is_eta,
        'is_nested_eta': is_nested_eta,
        'category': category,
        'would_pass_filters': passes_abstraction_quality_checks(invention)
    }


def run_quality_test():
    """Run the abstraction quality validation test."""
    print("=" * 70)
    print("ABSTRACTION QUALITY FILTER VALIDATION TEST")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Test the filtering functions directly on known-good and known-bad abstractions
    print("Testing abstraction quality filters on synthetic examples...")

    # For now, we'll test the filtering functions directly
    # Create some test programs that would trigger filters

    test_cases = []

    # Test case 1: Eta-expanded wrapper (should be rejected)
    last = Primitive("last", None, lambda x: x[-1] if x else None)
    eta_last = Invented(Abstraction(Application(last, Index(0))))
    test_cases.append(("(λ last $0) - eta wrapper", eta_last))

    # Test case 2: Identity function (should be rejected)
    identity = Invented(Abstraction(Index(0)))
    test_cases.append(("(λ $0) - identity", identity))

    # Test case 3: Nested eta (should be rejected)
    nested = Invented(Abstraction(Application(
        Abstraction(Application(last, Index(0))),
        Index(0)
    )))
    test_cases.append(("(λ (λ last $0) $0) - nested eta", nested))

    # Test case 4: Useful abstraction with 2 primitives (should be accepted)
    first = Primitive("first", None, lambda x: x[0] if x else None)
    useful = Invented(Abstraction(Abstraction(Application(
        Application(first, Index(0)),
        Application(last, Index(1))
    ))))
    test_cases.append(("(λ λ (first $0) (last $1)) - useful", useful))

    # Test case 5: Useful abstraction with duplicated variable (should be accepted)
    eq = Primitive("eq", None, lambda x, y: x == y)
    dup_var = Invented(Abstraction(Application(Application(eq, Index(0)), Index(0))))
    test_cases.append(("(λ eq $0 $0) - duplicate var", dup_var))

    # Test case 6: Multi-primitive composition (should be accepted)
    get_suit = Primitive("get_suit", None, lambda c: c.suit if hasattr(c, 'suit') else None)
    compose = Invented(Abstraction(Application(get_suit, Application(first, Index(0)))))
    test_cases.append(("(λ get_suit (first $0)) - composition", compose))

    print("\n" + "=" * 70)
    print("INDIVIDUAL ABSTRACTION ANALYSIS")
    print("=" * 70)

    results = []
    for name, invention in test_cases:
        analysis = analyze_abstraction(invention)
        results.append({**analysis, 'name': name})

        status = "✅ PASS" if analysis['would_pass_filters'] else "❌ REJECT"
        print(f"\n{status} {name}")
        print(f"    Body: {analysis['body']}")
        print(f"    Primitives: {analysis['n_primitives']}")
        print(f"    Category: {analysis['category']}")
        print(f"    is_trivial: {analysis['is_trivial']}, is_eta: {analysis['is_eta_reducible']}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    categories = Counter(r['category'] for r in results)
    passed = sum(1 for r in results if r['would_pass_filters'])
    rejected = len(results) - passed

    print(f"\nCategory breakdown:")
    for cat, count in categories.items():
        print(f"  {cat}: {count}")

    print(f"\nFilter results:")
    print(f"  Passed: {passed}")
    print(f"  Rejected: {rejected}")
    print(f"  Rejection rate: {rejected / len(results) * 100:.1f}%")

    # Verify expected behavior
    print("\n" + "=" * 70)
    print("EXPECTED vs ACTUAL")
    print("=" * 70)

    expected = {
        "(λ last $0) - eta wrapper": False,
        "(λ $0) - identity": False,
        "(λ (λ last $0) $0) - nested eta": False,
        "(λ λ (first $0) (last $1)) - useful": True,
        "(λ eq $0 $0) - duplicate var": True,
        "(λ get_suit (first $0)) - composition": True,
    }

    all_correct = True
    for r in results:
        exp = expected.get(r['name'])
        if exp is not None:
            actual = r['would_pass_filters']
            status = "✅" if actual == exp else "❌"
            if actual != exp:
                all_correct = False
            print(f"  {status} {r['name']}: expected={exp}, actual={actual}")

    print("\n" + "=" * 70)
    if all_correct:
        print("✅ ALL TESTS PASSED - Filters working correctly!")
    else:
        print("❌ SOME TESTS FAILED - Check filter implementation")
    print("=" * 70)

    return all_correct


def main():
    """Main entry point."""
    success = run_quality_test()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
