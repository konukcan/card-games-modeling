"""
Difficulty Measurement Experiment for Card Game Rules

This module runs DreamCoder on a representative sample of card game rules
to measure their relative learning difficulty.

Key metrics captured:
1. Programs enumerated before finding solution
2. Time to solution
3. Solution complexity (AST size, description length)
4. Whether rule is solvable within budget

The goal is cognitive realism - ranking rules by how hard they are for
a computational learner, which can then be compared to human data.
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

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.type_system import (
    Type, BaseType, Arrow, ListType, TypeVariable,
    BOOL, INT, CARD, HAND, arrow
)
from dreamcoder_core.program import Program, Primitive, Application, Abstraction, Index
from dreamcoder_core.grammar import Grammar, uniform_grammar
from dreamcoder_core.enumeration import enumerate_simple
from dreamcoder_core.wake_sleep import DreamCoder, Task, DreamCoderResult, LearningMetrics
from dreamcoder_core.card_primitives import build_card_grammar

# Import card domain
from rules.cards import Card, Hand, Suit, Rank, sample_hand
from rules.catalogue import ALL_RULES, Rule, get_rules_by_family, get_all_families


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class DifficultyExperimentConfig:
    """Configuration for difficulty measurement."""
    name: str = "difficulty_experiment"

    # Rule selection
    rules: Optional[List[str]] = None  # None = test all
    families: Optional[List[str]] = None  # Filter by family
    levels: Optional[List[int]] = None  # Filter by compositional level

    # Enumeration settings
    enumeration_budget: int = 500000  # Programs to try per rule
    enumeration_timeout: float = 300.0  # Seconds per rule
    max_depth: int = 8

    # Example generation
    n_examples: int = 20
    balance_examples: bool = True

    # Random seed for reproducibility
    seed: int = 42

    # Output
    output_dir: str = "results"
    save_solutions: bool = True


@dataclass
class RuleDifficultyResult:
    """Result for a single rule."""
    rule_id: str
    rule_name: str
    family: str
    level: int

    # Difficulty metrics
    solved: bool
    programs_enumerated: int
    time_seconds: float

    # Solution details (if solved)
    solution: Optional[str] = None
    solution_size: int = 0
    description_length: float = 0.0

    # Example balance
    n_positive_examples: int = 0
    n_negative_examples: int = 0

    def difficulty_score(self) -> float:
        """
        Compute a difficulty score.

        Lower = easier
        Scale: log(programs_enumerated) for solved
               infinity for unsolved
        """
        if not self.solved:
            return float('inf')
        import math
        return math.log(self.programs_enumerated + 1)


# ============================================================================
# EXAMPLE GENERATION
# ============================================================================

def generate_balanced_examples(
    rule: Rule,
    n_examples: int = 20,
    max_attempts: int = 20000,
    seed: Optional[int] = None
) -> Tuple[List[Tuple[Hand, bool]], Dict]:
    """
    Generate balanced training examples for a rule.

    Returns:
        examples: List of (hand, label) tuples
        stats: Dictionary with generation statistics
    """
    if seed is not None:
        random.seed(seed)

    positives = []
    negatives = []
    target_each = n_examples // 2
    attempts = 0

    for _ in range(max_attempts):
        attempts += 1
        hand = sample_hand(6)
        try:
            label = rule.eval(hand)
            if label and len(positives) < target_each:
                positives.append((hand, True))
            elif not label and len(negatives) < target_each:
                negatives.append((hand, False))
        except Exception:
            continue

        if len(positives) >= target_each and len(negatives) >= target_each:
            break

    # Build examples
    n_pos = min(len(positives), target_each)
    n_neg = min(len(negatives), target_each)

    examples = positives[:n_pos] + negatives[:n_neg]
    random.shuffle(examples)

    stats = {
        'attempts': attempts,
        'positives_found': len(positives),
        'negatives_found': len(negatives),
        'positives_used': n_pos,
        'negatives_used': n_neg,
        'balance_achieved': n_pos == n_neg == target_each
    }

    return examples, stats


# ============================================================================
# SINGLE RULE EVALUATION
# ============================================================================

def evaluate_rule_difficulty(
    rule: Rule,
    grammar: Grammar,
    config: DifficultyExperimentConfig
) -> RuleDifficultyResult:
    """
    Evaluate the difficulty of learning a single rule.

    This is the core measurement: how many programs do we need to enumerate
    before finding one that solves all examples?
    """
    print(f"\n  [{rule.id}] ({rule.family}, level {rule.level})")
    print(f"    {rule.name}")

    # Generate examples
    examples, example_stats = generate_balanced_examples(
        rule,
        n_examples=config.n_examples,
        seed=config.seed
    )

    if len(examples) < 4:
        print(f"    SKIP: Insufficient examples ({len(examples)})")
        return RuleDifficultyResult(
            rule_id=rule.id,
            rule_name=rule.name,
            family=rule.family,
            level=rule.level,
            solved=False,
            programs_enumerated=0,
            time_seconds=0.0,
            n_positive_examples=example_stats['positives_used'],
            n_negative_examples=example_stats['negatives_used']
        )

    print(f"    Examples: {example_stats['positives_used']}+ / {example_stats['negatives_used']}-")

    # Create task
    request_type = arrow(HAND, BOOL)

    # Enumeration
    start_time = time.time()
    programs_tried = 0
    solution = None

    for program, log_prob in enumerate_simple(
        grammar,
        request_type,
        max_depth=config.max_depth
    ):
        programs_tried += 1

        if programs_tried > config.enumeration_budget:
            break
        if time.time() - start_time > config.enumeration_timeout:
            break

        # Progress update
        if programs_tried % 100000 == 0:
            print(f"    ... {programs_tried} programs, {time.time() - start_time:.1f}s")

        # Evaluate on examples
        try:
            fn = program.evaluate([])
            correct = 0
            for hand, expected in examples:
                result = fn(hand)
                if result == expected:
                    correct += 1

            if correct == len(examples):
                solution = program
                break
        except Exception:
            pass

    elapsed = time.time() - start_time

    if solution:
        print(f"    SOLVED in {programs_tried:,} programs ({elapsed:.1f}s)")
        print(f"    Solution: {solution}")

        import math
        return RuleDifficultyResult(
            rule_id=rule.id,
            rule_name=rule.name,
            family=rule.family,
            level=rule.level,
            solved=True,
            programs_enumerated=programs_tried,
            time_seconds=elapsed,
            solution=str(solution),
            solution_size=solution.size(),
            description_length=-log_prob / math.log(2),
            n_positive_examples=example_stats['positives_used'],
            n_negative_examples=example_stats['negatives_used']
        )
    else:
        print(f"    UNSOLVED after {programs_tried:,} programs ({elapsed:.1f}s)")
        return RuleDifficultyResult(
            rule_id=rule.id,
            rule_name=rule.name,
            family=rule.family,
            level=rule.level,
            solved=False,
            programs_enumerated=programs_tried,
            time_seconds=elapsed,
            n_positive_examples=example_stats['positives_used'],
            n_negative_examples=example_stats['negatives_used']
        )


# ============================================================================
# FULL EXPERIMENT
# ============================================================================

def run_difficulty_experiment(
    config: DifficultyExperimentConfig
) -> List[RuleDifficultyResult]:
    """
    Run a complete difficulty measurement experiment.

    Returns list of results for each rule tested.
    """
    random.seed(config.seed)

    # Select rules to test
    rules = ALL_RULES

    if config.rules:
        rules = [r for r in rules if r.id in config.rules]
    if config.families:
        rules = [r for r in rules if r.family in config.families]
    if config.levels:
        rules = [r for r in rules if r.level in config.levels]

    print("=" * 70)
    print(f"DIFFICULTY MEASUREMENT EXPERIMENT: {config.name}")
    print("=" * 70)
    print(f"Rules to test: {len(rules)}")
    print(f"Enumeration budget: {config.enumeration_budget:,} programs")
    print(f"Timeout: {config.enumeration_timeout}s per rule")
    print(f"Max depth: {config.max_depth}")
    print(f"Examples per rule: {config.n_examples}")

    # Build grammar
    grammar = build_card_grammar()
    print(f"Grammar: {len(grammar)} primitives")

    # Run experiments
    results = []
    start_time = time.time()

    for i, rule in enumerate(rules):
        print(f"\n[{i+1}/{len(rules)}]", end="")
        result = evaluate_rule_difficulty(rule, grammar, config)
        results.append(result)

    total_time = time.time() - start_time

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    solved = [r for r in results if r.solved]
    unsolved = [r for r in results if not r.solved]

    print(f"\nTotal time: {total_time:.1f}s")
    print(f"Rules tested: {len(results)}")
    print(f"Solved: {len(solved)} ({100*len(solved)/len(results):.1f}%)")
    print(f"Unsolved: {len(unsolved)}")

    if solved:
        # Sort by difficulty (programs enumerated)
        solved_sorted = sorted(solved, key=lambda r: r.programs_enumerated)

        print("\nEasiest rules (fewest programs):")
        for r in solved_sorted[:5]:
            print(f"  {r.rule_id}: {r.programs_enumerated:,} programs ({r.time_seconds:.1f}s)")

        print("\nHardest solved rules:")
        for r in solved_sorted[-5:]:
            print(f"  {r.rule_id}: {r.programs_enumerated:,} programs ({r.time_seconds:.1f}s)")

        # Difficulty by family
        print("\nDifficulty by family (avg programs enumerated):")
        family_stats = defaultdict(list)
        for r in solved:
            family_stats[r.family].append(r.programs_enumerated)

        for family in sorted(family_stats.keys()):
            progs = family_stats[family]
            avg = sum(progs) / len(progs)
            print(f"  {family}: {avg:,.0f} (n={len(progs)})")

    if unsolved:
        print("\nUnsolved rules:")
        for r in unsolved:
            print(f"  {r.rule_id} ({r.family}, level {r.level})")

    # Save results
    save_results(results, config)

    return results


def save_results(results: List[RuleDifficultyResult], config: DifficultyExperimentConfig):
    """Save results to CSV and JSON."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S-%f")[:-3] + "Z"
    base_name = f"results_{config.name}_{timestamp}"

    # CSV for easy analysis
    csv_path = output_dir / f"{base_name}.csv"
    with open(csv_path, 'w', newline='') as f:
        fieldnames = [
            'rule_id', 'rule_name', 'family', 'level',
            'solved', 'programs_enumerated', 'time_seconds',
            'solution_size', 'description_length', 'difficulty_score',
            'n_positive_examples', 'n_negative_examples'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for r in results:
            row = {
                'rule_id': r.rule_id,
                'rule_name': r.rule_name,
                'family': r.family,
                'level': r.level,
                'solved': r.solved,
                'programs_enumerated': r.programs_enumerated,
                'time_seconds': r.time_seconds,
                'solution_size': r.solution_size,
                'description_length': r.description_length,
                'difficulty_score': r.difficulty_score() if r.solved else float('inf'),
                'n_positive_examples': r.n_positive_examples,
                'n_negative_examples': r.n_negative_examples
            }
            writer.writerow(row)

    print(f"\nResults saved to: {csv_path}")

    # JSON with full details
    if config.save_solutions:
        json_path = output_dir / f"{base_name}.json"
        data = {
            'config': asdict(config) if hasattr(config, '__dataclass_fields__') else vars(config),
            'results': [asdict(r) for r in results]
        }
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"Full results saved to: {json_path}")


# ============================================================================
# PRESET EXPERIMENTS
# ============================================================================

def run_simple_rules_experiment():
    """Test simple rules (level 0-1) that should be solvable."""
    config = DifficultyExperimentConfig(
        name="simple_rules",
        levels=[0, 1, 4],  # Position rules and basic counting rules
        enumeration_budget=500000,
        enumeration_timeout=300.0,
        max_depth=8,
        n_examples=20
    )
    return run_difficulty_experiment(config)


def run_selected_rules_experiment():
    """Test a curated selection of rules across families."""
    selected_rules = [
        # Simple - should be solvable
        'Ends_same_color',      # eq (get_color (first h)) (get_color (last h))
        'Ends_same_suit',       # eq (get_suit (first h)) (get_suit (last h))
        'Uniform_color',        # all same color

        # Medium
        'Has_pair_ranks',       # Duplicate ranks
        'Sorted_by_rank',       # Sorted

        # Complex
        'Suits_palindrome',     # Suit sequence is palindrome
        'Uniform_rank_parity',  # All odd or all even
    ]

    config = DifficultyExperimentConfig(
        name="selected_rules",
        rules=selected_rules,
        enumeration_budget=500000,
        enumeration_timeout=300.0,
        max_depth=8,
        n_examples=20
    )
    return run_difficulty_experiment(config)


def run_quick_test():
    """Quick test on 2 simple rules."""
    config = DifficultyExperimentConfig(
        name="quick_test",
        rules=['Ends_same_color', 'Ends_same_suit'],
        enumeration_budget=500000,
        enumeration_timeout=300.0,
        max_depth=8,
        n_examples=20
    )
    return run_difficulty_experiment(config)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run difficulty measurement experiment")
    parser.add_argument("--preset", choices=["quick", "simple", "selected", "all"],
                       default="quick", help="Preset experiment to run")
    parser.add_argument("--rules", nargs="+", help="Specific rule IDs to test")
    parser.add_argument("--budget", type=int, default=500000, help="Enumeration budget")
    parser.add_argument("--timeout", type=float, default=300.0, help="Timeout per rule")

    args = parser.parse_args()

    if args.rules:
        config = DifficultyExperimentConfig(
            name="custom",
            rules=args.rules,
            enumeration_budget=args.budget,
            enumeration_timeout=args.timeout
        )
        run_difficulty_experiment(config)
    elif args.preset == "quick":
        run_quick_test()
    elif args.preset == "simple":
        run_simple_rules_experiment()
    elif args.preset == "selected":
        run_selected_rules_experiment()
    else:
        # All rules
        config = DifficultyExperimentConfig(
            name="all_rules",
            enumeration_budget=args.budget,
            enumeration_timeout=args.timeout
        )
        run_difficulty_experiment(config)
