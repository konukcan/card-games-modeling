#!/usr/bin/env python3
"""
A/B Comparison Test: Backup vs New Modular Compression

This is the definitive backwards compatibility test.

It imports the BACKUP compression.py and the NEW modular compression/
package, runs both on identical inputs, and verifies outputs match.

This proves the refactoring didn't change any behavior.
"""

import sys
import importlib.util
from pathlib import Path

# =============================================================================
# SETUP: Import both versions
# =============================================================================

# Path to backup
BACKUP_PATH = Path(__file__).parent.parent.parent / "archived" / "legacy_compression" / "compression_backup_20260103_003112.py"

# Check backup exists
if not BACKUP_PATH.exists():
    print(f"❌ Backup not found: {BACKUP_PATH}")
    sys.exit(1)

print("=" * 70)
print("A/B COMPARISON: Backup vs New Modular Compression")
print("=" * 70)
print(f"\nBackup: {BACKUP_PATH.name}")
print("New: dreamcoder_core.compression (modular package)")

# Import the NEW modular version
print("\n1. Importing NEW modular compression...")
from dreamcoder_core.compression import (
    find_common_subtrees as new_find_common_subtrees,
    abstract_subtree as new_abstract_subtree,
    anti_unify as new_anti_unify,
    is_nontrivial as new_is_nontrivial,
    is_eta_reducible as new_is_eta_reducible,
    passes_abstraction_quality_checks as new_passes_checks,
    compress_frontiers as new_compress_frontiers,
    compute_mdl as new_compute_mdl,
    rewrite_with_invention as new_rewrite,
)
print("  ✓ New modular imports successful")

# Import the BACKUP version dynamically
print("\n2. Importing BACKUP compression.py...")
spec = importlib.util.spec_from_file_location("backup_compression", BACKUP_PATH)
backup = importlib.util.module_from_spec(spec)

# Need to add parent to path for backup's imports
sys.path.insert(0, str(BACKUP_PATH.parent.parent.parent / "src"))
try:
    spec.loader.exec_module(backup)
    print("  ✓ Backup imports successful")
except Exception as e:
    print(f"  ⚠ Backup import warning: {e}")
    # Try to continue - some imports might fail but functions might work

# Extract backup functions
old_find_common_subtrees = backup.find_common_subtrees
old_abstract_subtree = backup.abstract_subtree
old_anti_unify = backup.anti_unify
old_is_nontrivial = backup.is_nontrivial
old_is_eta_reducible = backup.is_eta_reducible
old_passes_checks = backup.passes_abstraction_quality_checks
old_compress_frontiers = backup.compress_frontiers
old_compute_mdl = backup.compute_mdl
old_rewrite = backup.rewrite_with_invention

# Import shared types (same in both)
from dreamcoder_core.program import Primitive, Application, Abstraction, Index, Invented
from dreamcoder_core.type_system import INT, arrow
from dreamcoder_core.grammar import uniform_grammar

# =============================================================================
# TEST FIXTURES
# =============================================================================

print("\n3. Creating test fixtures...")

add = Primitive('+', arrow(INT, INT, INT), lambda a: lambda b: a + b)
mul = Primitive('*', arrow(INT, INT, INT), lambda a: lambda b: a * b)
one = Primitive('1', INT, 1)
two = Primitive('2', INT, 2)
three = Primitive('3', INT, 3)

grammar = uniform_grammar([add, mul, one, two, three])

test_programs = [
    Application(Application(add, one), one),      # (+ 1 1) = 2
    Application(Application(add, one), two),      # (+ 1 2) = 3
    Application(Application(add, one), three),    # (+ 1 3) = 4
    Application(Application(add, one),            # (+ 1 (+ 1 1)) = 3
               Application(Application(add, one), one)),
    Application(Application(mul, two), three),    # (* 2 3) = 6
    Application(Application(add, Application(Application(mul, two), two)),  # (+ (* 2 2) 1) = 5
               one),
]

frontiers = [[(p, 0.0)] for p in test_programs]

print(f"  Created {len(test_programs)} test programs")

# =============================================================================
# A/B COMPARISON TESTS
# =============================================================================

passed = 0
failed = 0

def compare_test(name, old_fn, new_fn, *args, compare_fn=None):
    """Run same function on both versions and compare results."""
    global passed, failed
    try:
        old_result = old_fn(*args)
        new_result = new_fn(*args)

        if compare_fn:
            match = compare_fn(old_result, new_result)
        else:
            match = old_result == new_result

        if match:
            print(f"  ✓ {name}: MATCH")
            passed += 1
        else:
            print(f"  ✗ {name}: MISMATCH")
            print(f"      Old: {old_result}")
            print(f"      New: {new_result}")
            failed += 1
    except Exception as e:
        print(f"  ✗ {name}: ERROR - {e}")
        failed += 1

print("\n4. Running A/B comparison tests...")
print("-" * 50)

# Test 1: is_nontrivial
print("\n[Quality Filters]")
for prog in [Index(0), one, Application(add, one)]:
    compare_test(
        f"is_nontrivial({prog})",
        old_is_nontrivial, new_is_nontrivial,
        prog
    )

# Test 2: is_eta_reducible
eta_prog = Abstraction(Application(add, Index(0)))
non_eta = Abstraction(Application(Application(add, Index(0)), one))
compare_test("is_eta_reducible (eta)", old_is_eta_reducible, new_is_eta_reducible, eta_prog)
compare_test("is_eta_reducible (non-eta)", old_is_eta_reducible, new_is_eta_reducible, non_eta)

# Test 3: find_common_subtrees
print("\n[Subtree Finding]")
def compare_subtrees(old, new):
    """Compare SubtreeOccurrence lists."""
    if len(old) != len(new):
        return False
    # Compare by subtree string and savings
    old_set = {(str(o.subtree), o.count, o.savings) for o in old}
    new_set = {(str(n.subtree), n.count, n.savings) for n in new}
    return old_set == new_set

compare_test(
    "find_common_subtrees",
    old_find_common_subtrees, new_find_common_subtrees,
    test_programs, 2, 2,
    compare_fn=compare_subtrees
)

# Test 4: abstract_subtree
print("\n[Abstraction]")
target = Application(Application(add, Index(0)), one)
def compare_abstraction(old, new):
    old_inv, old_args = old
    new_inv, new_args = new
    return str(old_inv) == str(new_inv) and old_args == new_args

compare_test(
    "abstract_subtree",
    old_abstract_subtree, new_abstract_subtree,
    target,
    compare_fn=compare_abstraction
)

# Test 5: anti_unify
print("\n[Anti-unification]")
p1 = Application(Application(add, one), two)
p2 = Application(Application(add, one), three)

def compare_anti_unify(old, new):
    old_pat, old_subs, old_map = old
    new_pat, new_subs, new_map = new
    return (str(old_pat) == str(new_pat) and
            len(old_subs) == len(new_subs))

compare_test(
    "anti_unify",
    old_anti_unify, new_anti_unify,
    p1, p2,
    compare_fn=compare_anti_unify
)

# Test 6: compute_mdl
print("\n[MDL Scoring]")
def compare_float(old, new, tolerance=1e-10):
    return abs(old - new) < tolerance

compare_test(
    "compute_mdl",
    lambda g, p, t: old_compute_mdl(g, p, t),
    lambda g, p, t: new_compute_mdl(g, p, t),
    grammar, test_programs, INT,
    compare_fn=compare_float
)

# Test 7: compress_frontiers
print("\n[Compression]")
def compare_compression(old, new):
    # Compare number of inventions
    if len(old.new_inventions) != len(new.new_inventions):
        print(f"    Invention count: old={len(old.new_inventions)}, new={len(new.new_inventions)}")
        return False
    # Compare total savings
    if abs(old.total_savings - new.total_savings) > 1e-10:
        print(f"    Savings: old={old.total_savings}, new={new.total_savings}")
        return False
    # Compare invention bodies
    old_invs = sorted([str(i) for i in old.new_inventions])
    new_invs = sorted([str(i) for i in new.new_inventions])
    if old_invs != new_invs:
        print(f"    Inventions differ:")
        print(f"      Old: {old_invs}")
        print(f"      New: {new_invs}")
        return False
    return True

compare_test(
    "compress_frontiers",
    lambda g, f: old_compress_frontiers(g, f, max_inventions=3, min_savings=1.0, refactor_programs=False),
    lambda g, f: new_compress_frontiers(g, f, max_inventions=3, min_savings=1.0, refactor_programs=False),
    grammar, frontiers,
    compare_fn=compare_compression
)

# Test 8: compress_frontiers with refactoring
compare_test(
    "compress_frontiers (refactor)",
    lambda g, f: old_compress_frontiers(g, f, max_inventions=3, min_savings=1.0, refactor_programs=True),
    lambda g, f: new_compress_frontiers(g, f, max_inventions=3, min_savings=1.0, refactor_programs=True),
    grammar, frontiers,
    compare_fn=compare_compression
)

# Test 9: rewrite_with_invention
print("\n[Rewriting]")
target = Application(Application(add, Index(0)), one)
inv_body = Abstraction(Application(Application(add, Index(0)), one))
inv = Invented(inv_body)
orig = Abstraction(Application(Application(add, Index(0)), one))

def compare_rewrite(old, new):
    return str(old) == str(new)

compare_test(
    "rewrite_with_invention",
    lambda p, t, i, n: old_rewrite(p, t, i, n),
    lambda p, t, i, n: new_rewrite(p, t, i, n),
    orig, target, inv, 1,
    compare_fn=compare_rewrite
)

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print(f"A/B COMPARISON RESULTS: {passed} passed, {failed} failed")
print("=" * 70)

if failed == 0:
    print("✅ BACKWARDS COMPATIBILITY VERIFIED!")
    print("   The new modular code produces identical results to the backup.")
else:
    print(f"⚠️  {failed} tests show differences between backup and new code!")
    print("   Investigate before assuming backwards compatibility.")

sys.exit(0 if failed == 0 else 1)
