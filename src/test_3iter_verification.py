#!/usr/bin/env python3
"""
3-Iteration Verification Test - Simplified

Tests:
1. Recognition guidance works (no slowdown after normalize_probabilities fix)
2. Dream quality (sample_requiring_variable produces non-trivial programs)
3. Full wake-sleep loop runs correctly with TopDownEnumerator
"""

import sys
import time
import random
import json
from pathlib import Path
from datetime import datetime

# Set random seed for reproducibility
random.seed(42)

sys.path.insert(0, str(Path(__file__).parent))

from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.type_system import arrow, HAND, BOOL
from dreamcoder_core.task import Task
from dreamcoder_core.enumeration import TopDownEnumerator
from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.program import Abstraction, uses_variable
from rules.cards import Card, Suit, Rank
from rules.catalogue import ALL_RULES

print("=" * 70)
print("3-ITERATION VERIFICATION TEST")
print("=" * 70)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Build grammar
grammar = build_lean_grammar()
print(f"Grammar: {len(grammar)} primitives")

# Create recognition model
recognition = ContrastiveRecognitionModel(grammar)
print(f"Recognition model: {sum(p.numel() for p in recognition.parameters())} parameters")

# Create simple tasks for testing
def sample_hand(n=6):
    """Sample a random hand of n cards."""
    all_cards = [Card(s, r) for s in Suit for r in Rank]
    return tuple(random.sample(all_cards, n))

def make_eval_fn():
    """Create evaluation function."""
    def eval_fn(program, hand):
        try:
            fn = program.evaluate([])
            return fn(hand)
        except (ValueError, TypeError, ZeroDivisionError, IndexError, KeyError, AttributeError, RecursionError):
            return None
    return eval_fn

eval_fn = make_eval_fn()

# Select a subset of simple rules for quick testing (use actual IDs from catalogue)
test_rules = [r for r in ALL_RULES if r.id in [
    'Sorted_by_rank', 'Has_pair_ranks', 'Uniform_color', 
    'Exactly_two_suits', 'Exactly_one_club'
]][:5]

if not test_rules:
    # Fallback to first few rules
    test_rules = ALL_RULES[:5]

print(f"\nTest rules: {[r.id for r in test_rules]}")

# Create tasks
def make_task(name: str, rule, n_examples: int = 20) -> Task:
    """Create a Task from a rule by generating examples."""
    examples = []
    for _ in range(n_examples * 3):  # Generate extra to ensure we get enough
        hand = sample_hand(6)
        try:
            result = rule.checker(hand)
            examples.append((hand, result))
        except (ValueError, TypeError, ZeroDivisionError, IndexError, KeyError, AttributeError):
            pass
    # Keep n_examples
    examples = examples[:n_examples]
    return Task(name=name, request_type=arrow(HAND, BOOL), examples=examples)

tasks = [make_task(r.id, r) for r in test_rules]
print(f"Created {len(tasks)} tasks with ~20 examples each")

# Track metrics
metrics = {
    'iterations': [],
    'recognition_guidance_test': None,
    'dream_quality_test': None
}

# ============================================================================
# TEST 1: Recognition Guidance Performance
# ============================================================================
print("\n" + "=" * 70)
print("TEST 1: Recognition Guidance Performance")
print("=" * 70)

request_type = arrow(HAND, BOOL)
test_task = tasks[0] if tasks else None

if test_task:
    # Base grammar enumeration
    print("\n1a. Enumerating with BASE grammar...")
    enumerator = TopDownEnumerator(grammar, max_depth=5, max_programs=2000)
    start = time.time()
    base_count = 0
    for prog, lp in enumerator.enumerate(request_type, timeout_seconds=10):
        base_count += 1
        if base_count >= 2000:
            break
    base_time = time.time() - start
    base_rate = base_count / base_time if base_time > 0 else 0
    print(f"   Enumerated {base_count} programs in {base_time:.2f}s")
    print(f"   Rate: {base_rate:.1f} programs/second")

    # Guided grammar enumeration
    print("\n1b. Enumerating with GUIDED grammar...")
    guided_grammar = recognition.predict_grammar_weights(test_task)
    enumerator = TopDownEnumerator(guided_grammar, max_depth=5, max_programs=2000)
    start = time.time()
    guided_count = 0
    for prog, lp in enumerator.enumerate(request_type, timeout_seconds=10):
        guided_count += 1
        if guided_count >= 2000:
            break
    guided_time = time.time() - start
    guided_rate = guided_count / guided_time if guided_time > 0 else 0
    print(f"   Enumerated {guided_count} programs in {guided_time:.2f}s")
    print(f"   Rate: {guided_rate:.1f} programs/second")

    ratio = guided_rate / base_rate if base_rate > 0 else 0
    print(f"\n   Ratio (guided/base): {ratio:.2f}x")
    if ratio > 0.5:
        print("   ✅ PASS: Recognition guidance does NOT slow enumeration")
        metrics['recognition_guidance_test'] = 'PASS'
    else:
        print("   ❌ FAIL: Recognition guidance too slow")
        metrics['recognition_guidance_test'] = 'FAIL'
else:
    print("   ⚠️ No test task available")
    metrics['recognition_guidance_test'] = 'SKIP'

# ============================================================================
# TEST 2: Dream Quality
# ============================================================================
print("\n" + "=" * 70)
print("TEST 2: Dream Quality (sample_requiring_variable)")
print("=" * 70)

# Test regular sample
print("\n2a. Regular Grammar.sample()...")
uses_var_count = 0
total = 0
for _ in range(50):
    result = grammar.sample(request_type, max_depth=6)
    if result:
        prog, _ = result
        total += 1
        if isinstance(prog, Abstraction) and uses_variable(prog.body, 0):
            uses_var_count += 1
pct_regular = 100 * uses_var_count / total if total > 0 else 0
print(f"   Sampled {total} programs, {uses_var_count} use $0 ({pct_regular:.1f}%)")

# Test sample_requiring_variable
print("\n2b. Grammar.sample_requiring_variable()...")
uses_var_count = 0
total = 0
for _ in range(50):
    result = grammar.sample_requiring_variable(request_type, max_depth=6, max_retries=15)
    if result:
        prog, _ = result
        total += 1
        if isinstance(prog, Abstraction) and uses_variable(prog.body, 0):
            uses_var_count += 1
pct_required = 100 * uses_var_count / total if total > 0 else 0
print(f"   Sampled {total} programs, {uses_var_count} use $0 ({pct_required:.1f}%)")

if pct_required >= 95:
    print("\n   ✅ PASS: sample_requiring_variable produces quality dreams")
    metrics['dream_quality_test'] = 'PASS'
else:
    print("\n   ❌ FAIL: Too many trivial dreams")
    metrics['dream_quality_test'] = 'FAIL'

# ============================================================================
# TEST 3: 3-Iteration Wake-Sleep Loop
# ============================================================================
print("\n" + "=" * 70)
print("TEST 3: 3-Iteration Wake-Sleep Loop")
print("=" * 70)

for iteration in range(1, 4):
    print(f"\n--- Iteration {iteration}/3 ---")
    iter_start = time.time()
    iter_metrics = {
        'iteration': iteration,
        'tasks_solved': 0,
        'programs_enumerated': 0,
        'solutions': []
    }
    
    # WAKE: Enumerate for each task
    print(f"  WAKE: Enumerating (budget=10000, depth=6)...")
    wake_start = time.time()
    
    for task in tasks:
        # Get guided grammar (this is where recognition guidance is applied)
        if iteration > 1:
            task_grammar = recognition.predict_grammar_weights(task)
        else:
            task_grammar = grammar
        
        enumerator = TopDownEnumerator(task_grammar, max_depth=6, max_programs=10000)
        programs_tried = 0
        
        for prog, lp in enumerator.enumerate(task.request_type, timeout_seconds=30):
            programs_tried += 1
            if programs_tried > 10000:
                break
            
            # Evaluate
            try:
                correct = sum(1 for inp, exp in task.examples if eval_fn(prog, inp) == exp)
                if correct == len(task.examples):
                    iter_metrics['tasks_solved'] += 1
                    iter_metrics['solutions'].append({
                        'task': task.name,
                        'program': str(prog)[:50],
                        'programs_tried': programs_tried
                    })
                    break
            except (ValueError, TypeError, ZeroDivisionError, IndexError, KeyError, AttributeError, RecursionError):
                pass
        
        iter_metrics['programs_enumerated'] += programs_tried
    
    wake_time = time.time() - wake_start
    print(f"       Solved: {iter_metrics['tasks_solved']}/{len(tasks)} tasks")
    print(f"       Programs: {iter_metrics['programs_enumerated']:,} in {wake_time:.1f}s")
    
    # SLEEP-DREAM: Generate dreams
    print(f"  SLEEP-DREAM: Generating 10 dreams...")
    dream_start = time.time()
    dreams = []
    for _ in range(20):  # Try more to get 10 valid dreams
        result = grammar.sample_requiring_variable(request_type, max_depth=6, max_retries=20)
        if result:
            prog, _ = result
            # Generate examples for this dream
            dream_examples = []
            try:
                fn = prog.evaluate([])
                for _ in range(10):
                    hand = sample_hand(6)
                    try:
                        out = fn(hand)
                        dream_examples.append((hand, out))
                    except (ValueError, TypeError, ZeroDivisionError, IndexError, KeyError, AttributeError, RecursionError):
                        pass
                if len(dream_examples) >= 5:
                    dreams.append({'program': prog, 'examples': dream_examples})
                    if len(dreams) >= 10:
                        break
            except (ValueError, TypeError, ZeroDivisionError, IndexError, KeyError, AttributeError, RecursionError):
                pass
    dream_time = time.time() - dream_start
    print(f"       Generated {len(dreams)} dreams in {dream_time:.1f}s")
    
    # Show example dreams
    if dreams and iteration == 1:
        print("       Example dreams:")
        for d in dreams[:3]:
            print(f"         {str(d['program'])[:50]}")
    
    iter_time = time.time() - iter_start
    print(f"  Iteration time: {iter_time:.1f}s")
    
    iter_metrics['wake_time'] = wake_time
    iter_metrics['dream_time'] = dream_time
    iter_metrics['total_time'] = iter_time
    iter_metrics['n_dreams'] = len(dreams)
    metrics['iterations'].append(iter_metrics)

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nRecognition Guidance Test: {metrics['recognition_guidance_test']}")
print(f"Dream Quality Test: {metrics['dream_quality_test']}")

print("\nPer-Iteration Results:")
for m in metrics['iterations']:
    print(f"  Iteration {m['iteration']}: {m['tasks_solved']}/{len(tasks)} solved, "
          f"{m['programs_enumerated']:,} programs, {m.get('n_dreams', 0)} dreams, {m['total_time']:.1f}s")

# Solutions found
all_solutions = []
for m in metrics['iterations']:
    all_solutions.extend(m['solutions'])

if all_solutions:
    print(f"\nSolutions found ({len(all_solutions)} total):")
    seen = set()
    for sol in all_solutions:
        if sol['task'] not in seen:
            print(f"  {sol['task']}: {sol['program']}")
            seen.add(sol['task'])

# Overall pass/fail
all_passed = (
    metrics['recognition_guidance_test'] == 'PASS' and 
    metrics['dream_quality_test'] == 'PASS'
)

print("\n" + "=" * 70)
if all_passed:
    print("✅ ALL TESTS PASSED")
    if not all_solutions:
        print("   (No solutions found, but core fixes verified)")
else:
    print("❌ SOME TESTS FAILED")
print("=" * 70)

# Save results
results_path = Path(__file__).parent / 'test_3iter_results.json'
with open(results_path, 'w') as f:
    # Convert to serializable format
    serializable = {
        'recognition_guidance_test': metrics['recognition_guidance_test'],
        'dream_quality_test': metrics['dream_quality_test'],
        'iterations': [
            {k: v for k, v in m.items() if k != 'solutions'} | 
            {'solutions': [{'task': s['task'], 'program': s['program']} for s in m.get('solutions', [])]}
            for m in metrics['iterations']
        ],
        'all_passed': all_passed
    }
    json.dump(serializable, f, indent=2)
print(f"\nResults saved to: {results_path}")
