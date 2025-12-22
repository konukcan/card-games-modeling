#!/usr/bin/env python3
"""Quick test to verify memoized enumeration produces correct programs."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.enumeration import TopDownEnumerator
from dreamcoder_core.type_system import arrow, HAND, BOOL

def main():
    print("=" * 70)
    print("MEMOIZED ENUMERATION FIX VERIFICATION")
    print("=" * 70)
    print()

    # Build grammar
    print("Building grammar...")
    grammar = build_lean_grammar()
    print(f"  Primitives: {len(list(grammar.primitives()))}")

    # Request type: HAND -> BOOL
    request_type = arrow(HAND, BOOL)
    print(f"\nEnumerating programs of type: {request_type}")
    print()

    # Test original enumerate()
    print("=== ORIGINAL enumerate() ===")
    enumerator1 = TopDownEnumerator(grammar, max_depth=6, max_programs=20)
    original_progs = []
    for i, (prog, log_prob) in enumerate(enumerator1.enumerate(request_type, timeout_seconds=10.0)):
        original_progs.append(str(prog))
        if i < 10:
            print(f"  {i+1}: {str(prog)} (log_prob={log_prob:.2f})")
    print(f"  ... {len(original_progs)} total programs")
    print()

    # Test memoized enumerate_memoized()
    print("=== MEMOIZED enumerate_memoized() ===")
    enumerator2 = TopDownEnumerator(grammar, max_depth=6, max_programs=20)
    memoized_progs = []
    for i, (prog, log_prob) in enumerate(enumerator2.enumerate_memoized(request_type, timeout_seconds=10.0)):
        memoized_progs.append(str(prog))
        if i < 10:
            print(f"  {i+1}: {str(prog)} (log_prob={log_prob:.2f})")

    stats = enumerator2.get_memo_stats()
    print(f"  ... {len(memoized_progs)} total programs")
    print(f"  Cache stats: {stats['hits']} hits, {stats['misses']} misses, {stats['hit_rate']*100:.1f}% hit rate")
    print()

    # Compare
    print("=== COMPARISON ===")

    # Check for expected programs
    expected = [
        "(λ true)",
        "(λ false)",
        "(λ not true)",
        "(λ not false)",
        "(λ all_same_suit $0)",
        "(λ all_same_rank $0)",
    ]

    original_set = set(original_progs)
    memoized_set = set(memoized_progs)

    print("Checking for expected programs:")
    all_found = True
    for prog in expected:
        in_orig = prog in original_set
        in_memo = prog in memoized_set
        status = "✓" if (in_orig and in_memo) else "✗"
        print(f"  {status} {prog}: original={in_orig}, memoized={in_memo}")
        if not in_memo:
            all_found = False

    print()

    # Check overlap
    overlap = original_set & memoized_set
    only_original = original_set - memoized_set
    only_memoized = memoized_set - original_set

    print(f"Programs in both: {len(overlap)}")
    print(f"Only in original: {len(only_original)}")
    print(f"Only in memoized: {len(only_memoized)}")

    if only_original:
        print("\n  Missing from memoized (first 5):")
        for prog in list(only_original)[:5]:
            print(f"    - {prog}")

    if only_memoized:
        print("\n  Extra in memoized (first 5):")
        for prog in list(only_memoized)[:5]:
            print(f"    - {prog}")

    print()
    if all_found and len(overlap) > 0:
        print("✓ FIX VERIFIED: Memoized enumeration produces expected programs!")
    else:
        print("✗ FIX INCOMPLETE: Some expected programs are missing from memoized")

    return all_found

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
