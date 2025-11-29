"""
Transfer Effects Experiment for Card Game Rules

This module measures how learning some rules affects learning others.
The key insight from DreamCoder is that library learning enables transfer:
- Learning rule A may discover abstractions useful for rule B
- Order of presentation matters for difficulty

Transfer effects to measure:
1. Does learning "Ends_same_color" help with "Ends_same_suit"?
2. Does learning "Uniform_color" help with "Uniform_rank_parity"?
3. Does learning simple rules help with complex ones in the same family?

Experiment design:
- Run DreamCoder on pairs/sequences of rules
- Measure programs enumerated for second rule
- Compare with baseline (second rule learned alone)
"""

import sys
from pathlib import Path
import json
import time
import random
import csv
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import copy

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.type_system import arrow, HAND, BOOL
from dreamcoder_core.program import Program
from dreamcoder_core.grammar import Grammar
from dreamcoder_core.enumeration import enumerate_simple
from dreamcoder_core.compression import compress_frontiers
from dreamcoder_core.wake_sleep import DreamCoder, Task
from dreamcoder_core.card_primitives import build_card_grammar
from dreamcoder_core.difficulty_experiment import (
    generate_balanced_examples, RuleDifficultyResult
)

# Import card domain
from rules.cards import sample_hand
from rules.catalogue import ALL_RULES, Rule, RULE_DICT


# ============================================================================
# TRANSFER EXPERIMENT
# ============================================================================

@dataclass
class TransferResult:
    """Result of a transfer experiment."""
    source_rule: str  # First rule learned
    target_rule: str  # Second rule measured

    # Baseline: target learned alone
    baseline_programs: int
    baseline_solved: bool
    baseline_time: float

    # With transfer: target learned after source
    transfer_programs: int
    transfer_solved: bool
    transfer_time: float

    # Library state
    primitives_after_source: int
    primitives_after_target: int
    new_abstractions: List[str]

    @property
    def transfer_benefit(self) -> float:
        """
        Ratio of baseline to transfer programs.

        > 1 means transfer helped
        < 1 means transfer hurt (negative transfer)
        = 1 means no effect
        """
        if not self.baseline_solved or not self.transfer_solved:
            return float('nan')
        if self.transfer_programs == 0:
            return float('inf')
        return self.baseline_programs / self.transfer_programs

    @property
    def programs_saved(self) -> int:
        """How many fewer programs needed with transfer."""
        if not self.baseline_solved or not self.transfer_solved:
            return 0
        return self.baseline_programs - self.transfer_programs


def measure_transfer(
    source_rule: Rule,
    target_rule: Rule,
    enumeration_budget: int = 500000,
    enumeration_timeout: float = 300.0,
    max_depth: int = 8,
    n_examples: int = 20,
    seed: int = 42
) -> TransferResult:
    """
    Measure transfer from source_rule to target_rule.

    Steps:
    1. Learn target_rule alone (baseline)
    2. Learn source_rule first, then target_rule (transfer condition)
    3. Compare programs enumerated
    """
    random.seed(seed)

    print(f"\n{'='*60}")
    print(f"TRANSFER: {source_rule.id} -> {target_rule.id}")
    print(f"{'='*60}")

    # Generate examples for both rules
    source_examples, _ = generate_balanced_examples(source_rule, n_examples, seed=seed)
    target_examples, _ = generate_balanced_examples(target_rule, n_examples, seed=seed+1)

    if len(source_examples) < 4 or len(target_examples) < 4:
        print("  SKIP: Insufficient examples")
        return TransferResult(
            source_rule=source_rule.id,
            target_rule=target_rule.id,
            baseline_programs=0,
            baseline_solved=False,
            baseline_time=0.0,
            transfer_programs=0,
            transfer_solved=False,
            transfer_time=0.0,
            primitives_after_source=0,
            primitives_after_target=0,
            new_abstractions=[]
        )

    request_type = arrow(HAND, BOOL)

    # =========================================================================
    # BASELINE: Learn target alone
    # =========================================================================
    print(f"\n  BASELINE: Learning {target_rule.id} alone")

    grammar = build_card_grammar()
    baseline_start = time.time()
    baseline_programs = 0
    baseline_solved = False

    for program, log_prob in enumerate_simple(grammar, request_type, max_depth=max_depth):
        baseline_programs += 1

        if baseline_programs > enumeration_budget:
            break
        if time.time() - baseline_start > enumeration_timeout:
            break

        # Evaluate
        try:
            fn = program.evaluate([])
            correct = sum(1 for h, expected in target_examples if fn(h) == expected)
            if correct == len(target_examples):
                baseline_solved = True
                break
        except:
            pass

    baseline_time = time.time() - baseline_start
    print(f"    {'SOLVED' if baseline_solved else 'UNSOLVED'} in {baseline_programs:,} programs ({baseline_time:.1f}s)")

    # =========================================================================
    # TRANSFER: Learn source first, then target
    # =========================================================================
    print(f"\n  TRANSFER: Learning {source_rule.id} first...")

    grammar = build_card_grammar()
    initial_primitives = len(grammar)
    source_solution = None

    # Learn source rule
    source_start = time.time()
    source_programs = 0

    for program, log_prob in enumerate_simple(grammar, request_type, max_depth=max_depth):
        source_programs += 1

        if source_programs > enumeration_budget:
            break
        if time.time() - source_start > enumeration_timeout:
            break

        try:
            fn = program.evaluate([])
            correct = sum(1 for h, expected in source_examples if fn(h) == expected)
            if correct == len(source_examples):
                source_solution = (program, log_prob)
                break
        except:
            pass

    source_time = time.time() - source_start
    print(f"    Source {'SOLVED' if source_solution else 'UNSOLVED'} in {source_programs:,} programs ({source_time:.1f}s)")

    # Run compression if source was solved
    new_abstractions = []
    if source_solution:
        print(f"    Running compression...")
        frontiers = [[(source_solution[0], 0.0)]]  # Perfect solution
        result = compress_frontiers(
            grammar,
            frontiers,
            max_inventions=3,
            min_savings=1.0
        )
        if result.new_inventions:
            grammar = result.new_grammar
            new_abstractions = [str(inv) for inv in result.new_inventions]
            print(f"    Discovered {len(new_abstractions)} abstractions:")
            for inv in result.new_inventions:
                print(f"      {inv}")

    primitives_after_source = len(grammar)

    # Now learn target with (potentially) updated grammar
    print(f"\n  Now learning {target_rule.id} with updated grammar...")

    transfer_start = time.time()
    transfer_programs = 0
    transfer_solved = False

    for program, log_prob in enumerate_simple(grammar, request_type, max_depth=max_depth):
        transfer_programs += 1

        if transfer_programs > enumeration_budget:
            break
        if time.time() - transfer_start > enumeration_timeout:
            break

        try:
            fn = program.evaluate([])
            correct = sum(1 for h, expected in target_examples if fn(h) == expected)
            if correct == len(target_examples):
                transfer_solved = True
                print(f"    Solution: {program}")
                break
        except:
            pass

    transfer_time = time.time() - transfer_start
    print(f"    {'SOLVED' if transfer_solved else 'UNSOLVED'} in {transfer_programs:,} programs ({transfer_time:.1f}s)")

    primitives_after_target = len(grammar)

    result = TransferResult(
        source_rule=source_rule.id,
        target_rule=target_rule.id,
        baseline_programs=baseline_programs,
        baseline_solved=baseline_solved,
        baseline_time=baseline_time,
        transfer_programs=transfer_programs,
        transfer_solved=transfer_solved,
        transfer_time=transfer_time,
        primitives_after_source=primitives_after_source,
        primitives_after_target=primitives_after_target,
        new_abstractions=new_abstractions
    )

    # Summary
    print(f"\n  SUMMARY:")
    print(f"    Baseline: {baseline_programs:,} programs")
    print(f"    Transfer: {transfer_programs:,} programs")
    if result.transfer_benefit == result.transfer_benefit:  # not nan
        print(f"    Transfer benefit: {result.transfer_benefit:.2f}x")
        print(f"    Programs saved: {result.programs_saved:,}")

    return result


def run_transfer_experiment(
    pairs: List[Tuple[str, str]],
    **kwargs
) -> List[TransferResult]:
    """Run transfer experiments on multiple rule pairs."""
    results = []

    for source_id, target_id in pairs:
        source = RULE_DICT.get(source_id)
        target = RULE_DICT.get(target_id)

        if source is None or target is None:
            print(f"WARNING: Unknown rule ID: {source_id} or {target_id}")
            continue

        result = measure_transfer(source, target, **kwargs)
        results.append(result)

    return results


def save_transfer_results(results: List[TransferResult], name: str = "transfer"):
    """Save transfer results to CSV."""
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3] + "Z"
    csv_path = output_dir / f"transfer_{name}_{timestamp}.csv"

    with open(csv_path, 'w', newline='') as f:
        fieldnames = [
            'source_rule', 'target_rule',
            'baseline_programs', 'baseline_solved', 'baseline_time',
            'transfer_programs', 'transfer_solved', 'transfer_time',
            'transfer_benefit', 'programs_saved',
            'primitives_after_source', 'new_abstractions'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            writer.writerow({
                'source_rule': r.source_rule,
                'target_rule': r.target_rule,
                'baseline_programs': r.baseline_programs,
                'baseline_solved': r.baseline_solved,
                'baseline_time': r.baseline_time,
                'transfer_programs': r.transfer_programs,
                'transfer_solved': r.transfer_solved,
                'transfer_time': r.transfer_time,
                'transfer_benefit': r.transfer_benefit,
                'programs_saved': r.programs_saved,
                'primitives_after_source': r.primitives_after_source,
                'new_abstractions': ';'.join(r.new_abstractions)
            })

    print(f"\nTransfer results saved to: {csv_path}")


# ============================================================================
# PRESET EXPERIMENTS
# ============================================================================

def run_same_structure_transfer():
    """
    Test transfer between rules with same structure.

    Hypothesis: Learning Ends_same_color should help Ends_same_suit
    because they share the structure: eq (property (first h)) (property (last h))
    """
    pairs = [
        ('Ends_same_color', 'Ends_same_suit'),
        ('Ends_same_suit', 'Ends_same_color'),
        ('Uniform_color', 'Suits_palindrome'),
    ]

    results = run_transfer_experiment(
        pairs,
        enumeration_budget=500000,
        enumeration_timeout=300.0
    )

    save_transfer_results(results, "same_structure")
    return results


def run_quick_transfer_test():
    """Quick transfer test between similar rules."""
    pairs = [
        ('Ends_same_color', 'Ends_same_suit'),
    ]

    results = run_transfer_experiment(
        pairs,
        enumeration_budget=500000,
        enumeration_timeout=300.0
    )

    save_transfer_results(results, "quick_test")
    return results


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run transfer experiments")
    parser.add_argument("--source", help="Source rule ID")
    parser.add_argument("--target", help="Target rule ID")
    parser.add_argument("--preset", choices=["quick", "same_structure"],
                       help="Preset experiment")

    args = parser.parse_args()

    if args.source and args.target:
        results = run_transfer_experiment(
            [(args.source, args.target)],
            enumeration_budget=500000,
            enumeration_timeout=300.0
        )
        save_transfer_results(results, "custom")
    elif args.preset == "quick":
        run_quick_transfer_test()
    elif args.preset == "same_structure":
        run_same_structure_transfer()
    else:
        run_quick_transfer_test()
