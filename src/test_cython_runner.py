#!/usr/bin/env python3
"""
Comprehensive Test Suite for Full Cython DreamCoder Runner

This script validates all components before running overnight training.
Each test must pass before proceeding to ensure correctness.
"""

import sys
import time
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional

sys.path.insert(0, str(Path(__file__).parent))

# Test counters
TESTS_RUN = 0
TESTS_PASSED = 0
TESTS_FAILED = 0

@dataclass
class TestResult:
    name: str
    passed: bool
    message: str
    duration: float

def test(name: str):
    """Decorator to mark test functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            global TESTS_RUN, TESTS_PASSED, TESTS_FAILED
            TESTS_RUN += 1
            print(f"\n{'='*60}")
            print(f"TEST {TESTS_RUN}: {name}")
            print(f"{'='*60}")
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start
                TESTS_PASSED += 1
                print(f"✓ PASSED ({duration:.2f}s)")
                return TestResult(name, True, "OK", duration)
            except Exception as e:
                duration = time.time() - start
                TESTS_FAILED += 1
                print(f"✗ FAILED: {e}")
                traceback.print_exc()
                return TestResult(name, False, str(e), duration)
        return wrapper
    return decorator


# ============================================================================
# TEST 1: Import Cython Modules
# ============================================================================

@test("Import Cython Modules")
def test_cython_imports():
    """Verify all Cython modules can be imported."""
    print("  Importing type_system_cy...")
    from dreamcoder_core.cython_src.type_system_cy import (
        arrow, HAND, BOOL, INT, Type, Arrow, TypeContext, BaseType, ListType
    )
    print("    ✓ type_system_cy OK")

    print("  Importing program_cy...")
    from dreamcoder_core.cython_src.program_cy import (
        Program, Primitive, Invented, Application, Abstraction, Index
    )
    print("    ✓ program_cy OK")

    print("  Importing grammar_cy...")
    from dreamcoder_core.cython_src.grammar_cy import (
        Grammar, Production
    )
    print("    ✓ grammar_cy OK")

    print("  Importing enumeration_cy...")
    from dreamcoder_core.cython_src.enumeration_cy import enumerate_simple
    print("    ✓ enumeration_cy OK")

    print("  Importing lean_primitives_cy...")
    from dreamcoder_core.cython_src.lean_primitives_cy import (
        build_lean_primitives_cy, build_lean_grammar_cy
    )
    print("    ✓ lean_primitives_cy OK")

    return True


# ============================================================================
# TEST 2: Cython Type System
# ============================================================================

@test("Cython Type System Operations")
def test_type_system():
    """Verify type system operations work correctly."""
    from dreamcoder_core.cython_src.type_system_cy import (
        arrow, HAND, BOOL, INT, Type, Arrow, BaseType, ListType, TypeVariable
    )

    # Test base types exist
    print("  Checking base types...")
    assert HAND is not None, "HAND should exist"
    assert BOOL is not None, "BOOL should exist"
    assert INT is not None, "INT should exist"
    print("    ✓ Base types exist")

    # Test arrow type construction
    print("  Testing arrow type construction...")
    func_type = arrow(HAND, BOOL)
    assert isinstance(func_type, Arrow), f"Should be Arrow, got {type(func_type)}"
    print(f"    ✓ arrow(HAND, BOOL) = {func_type}")

    # Test complex arrow
    func_type2 = arrow(INT, INT, BOOL)
    print(f"    ✓ arrow(INT, INT, BOOL) = {func_type2}")

    # Test type variables
    print("  Testing type variables...")
    a = TypeVariable(0)
    b = TypeVariable(1)
    polymorphic = arrow(a, a)
    print(f"    ✓ Polymorphic type: {polymorphic}")

    return True


# ============================================================================
# TEST 3: Cython Primitives
# ============================================================================

@test("Cython Primitives Construction and Execution")
def test_primitives():
    """Verify primitives are correctly constructed and can execute."""
    from dreamcoder_core.cython_src.lean_primitives_cy import build_lean_primitives_cy
    from dreamcoder_core.cython_src.program_cy import Primitive

    print("  Building Cython primitives...")
    primitives = build_lean_primitives_cy()
    print(f"    ✓ Built {len(primitives)} primitives")

    # Check primitive types
    print("  Verifying primitive types...")
    for p in primitives[:5]:
        assert isinstance(p, Primitive), f"Expected Primitive, got {type(p)}"
    print("    ✓ All are Cython Primitive objects")

    # Test execution of some key primitives
    print("  Testing primitive execution...")

    # Find and test 'not'
    not_prim = None
    for p in primitives:
        if p.name == 'not':
            not_prim = p
            break
    assert not_prim is not None, "'not' primitive not found"
    result = not_prim.value(True)
    assert result == False, f"not(True) should be False, got {result}"
    print(f"    ✓ not(True) = {result}")

    # Find and test 'and'
    and_prim = None
    for p in primitives:
        if p.name == 'and':
            and_prim = p
            break
    assert and_prim is not None, "'and' primitive not found"
    result = and_prim.value(True)(False)
    assert result == False, f"and(True)(False) should be False, got {result}"
    print(f"    ✓ and(True)(False) = {result}")

    # Find and test '+'
    add_prim = None
    for p in primitives:
        if p.name == '+':
            add_prim = p
            break
    assert add_prim is not None, "'+' primitive not found"
    result = add_prim.value(3)(4)
    assert result == 7, f"3 + 4 should be 7, got {result}"
    print(f"    ✓ 3 + 4 = {result}")

    return True


# ============================================================================
# TEST 4: Cython Grammar Construction
# ============================================================================

@test("Cython Grammar Construction")
def test_grammar():
    """Verify grammar is correctly constructed."""
    from dreamcoder_core.cython_src.lean_primitives_cy import build_lean_grammar_cy
    from dreamcoder_core.cython_src.grammar_cy import Grammar

    print("  Building Cython grammar...")
    grammar = build_lean_grammar_cy()
    print(f"    ✓ Built grammar with {len(grammar)} productions")

    assert isinstance(grammar, Grammar), f"Expected Grammar, got {type(grammar)}"
    assert len(grammar) > 50, f"Grammar should have >50 productions, got {len(grammar)}"

    # Verify grammar has expected primitives
    print("  Checking grammar contents...")
    prod_names = [str(p.primitive) if hasattr(p, 'primitive') else str(p) for p in grammar.productions]

    # Key primitives that should exist
    expected = ['get_suit', 'get_rank', 'eq', 'and', 'or', 'not', 'all', 'any', 'map', 'filter']
    for exp in expected:
        found = any(exp in str(p) for p in grammar.productions)
        assert found, f"Expected primitive '{exp}' not in grammar"
        print(f"    ✓ Found '{exp}' in grammar")

    return True


# ============================================================================
# TEST 5: Cython Enumeration Produces Valid Programs
# ============================================================================

@test("Cython Enumeration Produces Valid Programs")
def test_enumeration():
    """Verify enumeration produces valid programs."""
    from dreamcoder_core.cython_src.type_system_cy import arrow, HAND, BOOL, INT
    from dreamcoder_core.cython_src.enumeration_cy import enumerate_simple
    from dreamcoder_core.cython_src.lean_primitives_cy import build_lean_grammar_cy
    from dreamcoder_core.cython_src.program_cy import Program

    grammar = build_lean_grammar_cy()

    print("  Enumerating HAND -> BOOL programs...")
    programs = []
    for prog, log_prob in enumerate_simple(grammar, arrow(HAND, BOOL), max_depth=5):
        programs.append((prog, log_prob))
        if len(programs) >= 100:
            break

    print(f"    ✓ Enumerated {len(programs)} programs")

    # Verify they are Program objects
    print("  Verifying program types...")
    for prog, log_prob in programs[:10]:
        assert isinstance(prog, Program), f"Expected Program, got {type(prog)}"
        assert isinstance(log_prob, float), f"Expected float log_prob, got {type(log_prob)}"
        assert log_prob <= 0, f"Log probability should be <= 0, got {log_prob}"
    print("    ✓ All are valid Program objects with valid log_probs")

    # Show some examples
    print("  Sample programs:")
    for prog, log_prob in programs[:5]:
        print(f"    {prog} (log_prob={log_prob:.2f})")

    return True


# ============================================================================
# TEST 6: Programs Execute Correctly on Card Data
# ============================================================================

@test("Programs Execute Correctly on Card Data")
def test_program_execution():
    """Verify programs can execute on actual card hands."""
    from dreamcoder_core.cython_src.type_system_cy import arrow, HAND, BOOL
    from dreamcoder_core.cython_src.enumeration_cy import enumerate_simple
    from dreamcoder_core.cython_src.lean_primitives_cy import build_lean_grammar_cy
    from rules.cards import Card, Suit, Rank

    grammar = build_lean_grammar_cy()

    # Create test hand
    test_hand = (
        Card(Suit.HEARTS, Rank.ACE),
        Card(Suit.SPADES, Rank.KING),
        Card(Suit.DIAMONDS, Rank.QUEEN)
    )
    print(f"  Test hand: {[str(c) for c in test_hand]}")

    # Enumerate some programs and execute them
    print("  Executing programs on test hand...")
    executed = 0
    errors = 0

    for prog, log_prob in enumerate_simple(grammar, arrow(HAND, BOOL), max_depth=5):
        try:
            fn = prog.evaluate([])
            result = fn(test_hand)
            assert isinstance(result, bool), f"Expected bool, got {type(result)}"
            executed += 1
        except Exception as e:
            errors += 1

        if executed + errors >= 50:
            break

    print(f"    ✓ Executed {executed} programs successfully")
    if errors > 0:
        print(f"    ! {errors} programs failed (expected for some invalid combinations)")

    # Verify at least some programs execute
    assert executed >= 20, f"Too few programs executed: {executed}"

    return True


# ============================================================================
# TEST 7: Frontier Population Works Correctly
# ============================================================================

@test("Frontier Population Works Correctly")
def test_frontier_population():
    """Verify frontiers are correctly populated when programs match."""
    from dreamcoder_core.cython_src.type_system_cy import arrow, HAND, BOOL
    from dreamcoder_core.cython_src.enumeration_cy import enumerate_simple
    from dreamcoder_core.cython_src.lean_primitives_cy import build_lean_grammar_cy
    from dreamcoder_core.dreamcoder_v2 import Task, TaskFrontier, SolutionEntry
    from rules.cards import Card, Suit, Rank

    grammar = build_lean_grammar_cy()

    # Create a simple task: "is the hand non-empty?"
    # This matches programs like "not (empty? $0)" or similar
    examples = [
        ((Card(Suit.HEARTS, Rank.ACE),), True),   # 1 card - non-empty
        ((), False),  # empty hand
        ((Card(Suit.SPADES, Rank.KING), Card(Suit.DIAMONDS, Rank.QUEEN)), True),  # 2 cards
    ]

    task = Task(
        name="test_nonempty",
        request_type=arrow(HAND, BOOL),
        examples=examples,
        family="test"
    )

    frontier = TaskFrontier(task, max_size=5)

    print(f"  Task: {task.name}")
    print(f"  Examples: {len(examples)}")

    # Find programs that solve the task
    solutions_found = 0
    programs_tried = 0

    for prog, log_prob in enumerate_simple(grammar, arrow(HAND, BOOL), max_depth=6):
        programs_tried += 1
        if programs_tried > 10000:
            break

        try:
            fn = prog.evaluate([])

            # Check all examples
            all_correct = True
            for inp, expected in examples:
                result = fn(inp)
                if result != expected:
                    all_correct = False
                    break

            if all_correct:
                entry = SolutionEntry(
                    program=prog,
                    log_probability=log_prob,
                    log_likelihood=0.0,
                    programs_enumerated=programs_tried,
                    time_found=0.0
                )
                frontier.add(entry)
                solutions_found += 1
                print(f"    SOLUTION {solutions_found}: {prog}")

                if solutions_found >= 3:
                    break
        except:
            pass

    print(f"  ✓ Searched {programs_tried} programs")
    print(f"  ✓ Found {solutions_found} solutions")
    print(f"  ✓ Frontier has {len(frontier.entries)} entries")
    print(f"  ✓ Frontier.solved = {frontier.solved}")

    # CRITICAL: Verify frontier is populated
    assert frontier.solved, "Frontier should be marked as solved"
    assert len(frontier.entries) > 0, "Frontier should have entries"
    assert frontier.entries[0].program is not None, "Entry should have program"

    # Verify we can access the program
    best_prog = frontier.entries[0].program
    print(f"  ✓ Best program: {best_prog}")

    return True


# ============================================================================
# TEST 8: Recognition Model Trains with Non-Zero Loss
# ============================================================================

@test("Recognition Model Trains with Non-Zero Loss")
def test_recognition_training():
    """Verify recognition model trains and produces non-zero loss."""
    from dreamcoder_core.cython_src.type_system_cy import arrow, HAND, BOOL
    from dreamcoder_core.cython_src.lean_primitives_cy import build_lean_grammar_cy
    from dreamcoder_core.lean_primitives import build_lean_grammar  # Python version for neural
    from dreamcoder_core.neural_recognition import NeuralRecognitionModel
    from dreamcoder_core.dreamcoder_v2 import Task, TaskFrontier, SolutionEntry
    from dreamcoder_core.cython_src.enumeration_cy import enumerate_simple
    from rules.cards import Card, Suit, Rank

    # Build Python grammar for recognition model
    python_grammar = build_lean_grammar()

    print("  Creating recognition model...")
    recognition = NeuralRecognitionModel(
        grammar=python_grammar,
        hidden_dim=64,  # Small for testing
        learning_rate=1e-3,
        device='cpu'
    )
    print(f"    ✓ Model created with {recognition.hidden_dim} hidden dim")

    # Create a few simple tasks with solutions
    grammar = build_lean_grammar_cy()

    tasks = []
    frontiers = {}

    # Task 1: always true
    examples1 = [
        ((Card(Suit.HEARTS, Rank.ACE),), True),
        ((Card(Suit.SPADES, Rank.KING),), True),
    ]
    task1 = Task(name="always_true", request_type=arrow(HAND, BOOL), examples=examples1, family="test")
    tasks.append(task1)

    # Find solution for task1
    frontier1 = TaskFrontier(task1, max_size=3)
    for prog, log_prob in enumerate_simple(grammar, arrow(HAND, BOOL), max_depth=4):
        try:
            fn = prog.evaluate([])
            if all(fn(inp) == exp for inp, exp in examples1):
                # Convert Cython program to string for storage
                entry = SolutionEntry(
                    program=prog,
                    log_probability=log_prob,
                    log_likelihood=0.0,
                    programs_enumerated=1,
                    time_found=0.0
                )
                frontier1.add(entry)
                if len(frontier1.entries) >= 1:
                    break
        except:
            pass
    frontiers[task1.name] = frontier1
    print(f"    ✓ Task 1 solved: {frontier1.solved}")

    # Task 2: always false
    examples2 = [
        ((Card(Suit.HEARTS, Rank.ACE),), False),
        ((Card(Suit.SPADES, Rank.KING),), False),
    ]
    task2 = Task(name="always_false", request_type=arrow(HAND, BOOL), examples=examples2, family="test")
    tasks.append(task2)

    frontier2 = TaskFrontier(task2, max_size=3)
    for prog, log_prob in enumerate_simple(grammar, arrow(HAND, BOOL), max_depth=4):
        try:
            fn = prog.evaluate([])
            if all(fn(inp) == exp for inp, exp in examples2):
                entry = SolutionEntry(
                    program=prog,
                    log_probability=log_prob,
                    log_likelihood=0.0,
                    programs_enumerated=1,
                    time_found=0.0
                )
                frontier2.add(entry)
                if len(frontier2.entries) >= 1:
                    break
        except:
            pass
    frontiers[task2.name] = frontier2
    print(f"    ✓ Task 2 solved: {frontier2.solved}")

    # Train recognition model
    print("  Training recognition model...")
    solved_tasks = [t for t in tasks if frontiers[t.name].solved]
    print(f"    Solved tasks: {len(solved_tasks)}")

    if len(solved_tasks) == 0:
        raise ValueError("No tasks were solved - cannot test recognition training")

    loss = recognition.train_on_frontiers(solved_tasks, frontiers, epochs=5)

    print(f"    ✓ Training loss: {loss:.6f}")

    # CRITICAL: Loss should be non-zero
    assert loss > 0, f"Recognition loss should be > 0, got {loss}"

    return True


# ============================================================================
# TEST 9: Mini Wake-Sleep Loop End-to-End
# ============================================================================

@test("Mini Wake-Sleep Loop End-to-End")
def test_mini_wake_sleep():
    """Run a mini wake-sleep iteration to verify the full loop works."""
    import time
    from dreamcoder_core.cython_src.type_system_cy import arrow, HAND, BOOL
    from dreamcoder_core.cython_src.lean_primitives_cy import build_lean_grammar_cy
    from dreamcoder_core.cython_src.enumeration_cy import enumerate_simple
    from dreamcoder_core.lean_primitives import build_lean_grammar
    from dreamcoder_core.neural_recognition import NeuralRecognitionModel
    from dreamcoder_core.dreamcoder_v2 import Task, TaskFrontier, SolutionEntry, make_eval_fn
    from rules.cards import Card, Suit, Rank

    print("  Setting up mini wake-sleep test...")

    # Build grammars
    cython_grammar = build_lean_grammar_cy()
    python_grammar = build_lean_grammar()

    # Create recognition model
    recognition = NeuralRecognitionModel(
        grammar=python_grammar,
        hidden_dim=64,
        learning_rate=1e-3,
        device='cpu'
    )

    # Create simple tasks
    eval_fn = make_eval_fn()

    tasks = []
    frontiers = {}

    # Task: "always true"
    examples1 = [
        ((Card(Suit.HEARTS, Rank.ACE),), True),
        ((Card(Suit.SPADES, Rank.KING), Card(Suit.DIAMONDS, Rank.QUEEN)), True),
        ((Card(Suit.CLUBS, Rank.TWO),), True),
    ]
    task1 = Task(name="const_true", request_type=arrow(HAND, BOOL), examples=examples1, family="test")
    tasks.append(task1)
    frontiers[task1.name] = TaskFrontier(task1, max_size=3)

    # Task: "always false"
    examples2 = [
        ((Card(Suit.HEARTS, Rank.ACE),), False),
        ((Card(Suit.SPADES, Rank.KING), Card(Suit.DIAMONDS, Rank.QUEEN)), False),
    ]
    task2 = Task(name="const_false", request_type=arrow(HAND, BOOL), examples=examples2, family="test")
    tasks.append(task2)
    frontiers[task2.name] = TaskFrontier(task2, max_size=3)

    print(f"    ✓ Created {len(tasks)} tasks")

    # WAKE: Enumerate and solve
    print("\n  WAKE PHASE:")
    solved_count = 0
    total_programs = 0

    for task in tasks:
        frontier = frontiers[task.name]
        start = time.time()

        for prog, log_prob in enumerate_simple(cython_grammar, arrow(HAND, BOOL), max_depth=5):
            total_programs += 1

            if total_programs > 5000:
                break

            try:
                all_correct = True
                for inp, expected in task.examples:
                    result = eval_fn(prog, inp)
                    if result != expected:
                        all_correct = False
                        break

                if all_correct:
                    entry = SolutionEntry(
                        program=prog,
                        log_probability=log_prob,
                        log_likelihood=0.0,
                        programs_enumerated=total_programs,
                        time_found=time.time() - start
                    )
                    frontier.add(entry)
                    print(f"    SOLVED {task.name}: {prog}")
                    solved_count += 1
                    break
            except:
                pass

    print(f"    ✓ Wake: Solved {solved_count}/{len(tasks)} tasks")
    print(f"    ✓ Enumerated {total_programs} programs")

    # Verify we solved at least one task
    assert solved_count > 0, "Should solve at least one task in wake phase"

    # SLEEP: Train recognition
    print("\n  SLEEP PHASE:")
    solved_tasks = [t for t in tasks if frontiers[t.name].solved]

    if solved_tasks:
        loss = recognition.train_on_frontiers(solved_tasks, frontiers, epochs=3)
        print(f"    ✓ Recognition trained on {len(solved_tasks)} tasks")
        print(f"    ✓ Loss: {loss:.6f}")

        # CRITICAL: Verify non-zero loss
        assert loss > 0, f"Recognition loss should be > 0, got {loss}"

    print("\n  ✓ Mini wake-sleep loop completed successfully!")

    return True


# ============================================================================
# TEST 10: Full Integration with Real Rules
# ============================================================================

@test("Full Integration - Enumeration Stress Test")
def test_real_rules_integration():
    """Stress test enumeration to verify it doesn't crash."""
    import time
    from dreamcoder_core.cython_src.type_system_cy import arrow, HAND, BOOL
    from dreamcoder_core.cython_src.lean_primitives_cy import build_lean_grammar_cy
    from dreamcoder_core.cython_src.enumeration_cy import enumerate_simple

    print("  Building grammar...")
    grammar = build_lean_grammar_cy()
    print(f"    ✓ Grammar has {len(grammar)} productions")

    print("  Running enumeration stress test (50,000 programs)...")
    start = time.time()
    count = 0

    for prog, log_prob in enumerate_simple(grammar, arrow(HAND, BOOL), max_depth=8):
        count += 1

        # Just enumerate, don't execute
        if count % 10000 == 0:
            print(f"    {count} programs ({time.time() - start:.1f}s)")

        if count >= 50000:
            break

    duration = time.time() - start
    print(f"    ✓ Enumerated {count} programs in {duration:.1f}s")
    print(f"    ✓ Rate: {count / duration:.0f} programs/sec")

    assert count >= 50000, f"Should enumerate 50,000 programs, got {count}"
    assert duration < 60, f"Should complete in <60s, took {duration:.1f}s"

    return True


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("COMPREHENSIVE CYTHON RUNNER TEST SUITE")
    print("=" * 70)
    print()
    print("Running all tests to verify the full Cython DreamCoder pipeline...")
    print("Each test must pass before launching overnight training.")
    print()

    results = []

    # Run all tests
    results.append(test_cython_imports())
    results.append(test_type_system())
    results.append(test_primitives())
    results.append(test_grammar())
    results.append(test_enumeration())
    results.append(test_program_execution())
    results.append(test_frontier_population())
    results.append(test_recognition_training())
    results.append(test_mini_wake_sleep())
    results.append(test_real_rules_integration())

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print()

    for r in results:
        status = "✓ PASS" if r.passed else "✗ FAIL"
        print(f"  {status}: {r.name} ({r.duration:.2f}s)")

    print()
    print(f"Total: {TESTS_RUN} tests")
    print(f"Passed: {TESTS_PASSED}")
    print(f"Failed: {TESTS_FAILED}")
    print()

    if TESTS_FAILED == 0:
        print("=" * 70)
        print("ALL TESTS PASSED - SAFE TO LAUNCH OVERNIGHT RUN")
        print("=" * 70)
        return 0
    else:
        print("=" * 70)
        print("SOME TESTS FAILED - DO NOT LAUNCH OVERNIGHT RUN")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
