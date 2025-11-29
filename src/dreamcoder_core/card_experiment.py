"""
Card Game Rule Learning Experiment

This module runs DreamCoder on actual card game rules to measure:
1. Learning difficulty for each rule
2. Transfer effects between rules
3. Library growth and abstraction discovery
4. Comparison to human learning patterns

The experiment puts DreamCoder in the same situation as human participants:
- Start with basic primitives
- See examples from each rule (hands labeled True/False)
- Try to infer the rule
- Can build on learned abstractions
"""

import sys
from pathlib import Path
import json
import time
import random
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
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
from dreamcoder_core.compression import compress_frontiers
from dreamcoder_core.wake_sleep import (
    DreamCoder, Task, DreamCoderResult, LearningMetrics
)
from dreamcoder_core.card_primitives import build_minimal_primitives, build_card_grammar

# Import card domain
from rules.cards import Card, Hand, Suit, Rank, sample_hand
from rules.catalogue import ALL_RULES, Rule


# ============================================================================
# TASK GENERATION FROM RULES
# ============================================================================

def generate_examples_for_rule(
    rule: Rule,
    n_examples: int = 20,
    balance: bool = True,
    max_attempts: int = 10000
) -> List[Tuple[Hand, bool]]:
    """
    Generate training examples for a rule.

    Args:
        rule: The rule to generate examples for
        n_examples: Number of examples to generate
        balance: Try to balance True/False examples
        max_attempts: Maximum random hands to try

    Returns:
        List of (hand, label) tuples
    """
    positives = []
    negatives = []
    target_each = n_examples // 2 if balance else n_examples

    # Generate many candidates
    for _ in range(max_attempts):
        hand = sample_hand(6)
        try:
            label = rule.eval(hand)
            if label and len(positives) < target_each:
                positives.append((hand, True))
            elif not label and len(negatives) < target_each:
                negatives.append((hand, False))
        except Exception:
            continue

        # Early exit if we have enough
        if balance:
            if len(positives) >= target_each and len(negatives) >= target_each:
                break
        else:
            if len(positives) + len(negatives) >= n_examples:
                break

    # If we couldn't find enough positives, try to construct some
    if balance and len(positives) < target_each:
        # For rules where positives are rare, try harder
        # This is a fallback - ideally we'd have rule-specific generators
        pass

    # Build final examples
    if balance:
        n_each = min(len(positives), len(negatives), target_each)
        if n_each == 0:
            # Couldn't balance - use what we have
            examples = positives + negatives
        else:
            examples = positives[:n_each] + negatives[:n_each]
    else:
        examples = (positives + negatives)[:n_examples]

    random.shuffle(examples)
    return examples


def rule_to_task(rule: Rule, n_examples: int = 20) -> Task:
    """Convert a Rule to a DreamCoder Task."""
    examples = generate_examples_for_rule(rule, n_examples)

    # The request type is: Hand -> Bool
    # (which is list(card) -> bool)
    request_type = arrow(HAND, BOOL)

    return Task(
        name=rule.id,
        request_type=request_type,
        examples=[(hand, label) for hand, label in examples]
    )


# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def eval_program_on_hand(program: Program, hand: Hand) -> Any:
    """
    Evaluate a program on a hand.

    The program should be of type Hand -> Bool.
    """
    try:
        fn = program.evaluate([])
        result = fn(hand)
        return result
    except Exception as e:
        return None


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a card game learning experiment."""
    name: str = "card_experiment"
    rules: List[str] = field(default_factory=list)  # Rule IDs to test
    n_examples_per_rule: int = 20
    max_iterations: int = 5
    enumeration_timeout: float = 30.0
    enumeration_budget: int = 5000
    max_depth: int = 5
    n_runs: int = 1
    shuffle_rules: bool = False
    seed: int = 42


@dataclass
class ExperimentResult:
    """Result of a complete experiment."""
    config: ExperimentConfig
    runs: List[DreamCoderResult]
    rule_difficulty: Dict[str, Dict]  # Per-rule metrics
    transfer_matrix: Dict[str, Dict[str, float]]  # Transfer effects
    total_time: float

    def save(self, path: Path):
        """Save results to JSON."""
        data = {
            'config': {
                'name': self.config.name,
                'rules': self.config.rules,
                'n_examples': self.config.n_examples_per_rule,
                'max_iterations': self.config.max_iterations,
                'n_runs': self.config.n_runs,
            },
            'rule_difficulty': self.rule_difficulty,
            'transfer_matrix': self.transfer_matrix,
            'total_time': self.total_time,
            'runs': [run.to_json() for run in self.runs]
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


def run_experiment(config: ExperimentConfig) -> ExperimentResult:
    """
    Run a complete card game learning experiment.

    This puts DreamCoder in the participant's shoes.
    """
    random.seed(config.seed)
    start_time = time.time()

    # Get the rules to test
    if config.rules:
        rules = [r for r in ALL_RULES if r.id in config.rules]
    else:
        rules = ALL_RULES

    print(f"\n{'='*60}")
    print(f"CARD GAME LEARNING EXPERIMENT: {config.name}")
    print(f"{'='*60}")
    print(f"Rules: {len(rules)}")
    print(f"Examples per rule: {config.n_examples_per_rule}")
    print(f"Max iterations: {config.max_iterations}")
    print(f"Runs: {config.n_runs}")

    # Build initial grammar
    grammar = build_card_grammar()
    print(f"Initial primitives: {len(grammar)}")

    # Run experiments
    all_runs = []
    rule_metrics: Dict[str, List[LearningMetrics]] = defaultdict(list)

    for run_idx in range(config.n_runs):
        print(f"\n--- RUN {run_idx + 1}/{config.n_runs} ---")

        # Generate tasks
        run_rules = list(rules)
        if config.shuffle_rules:
            random.shuffle(run_rules)

        tasks = [rule_to_task(r, config.n_examples_per_rule) for r in run_rules]

        # Run DreamCoder
        dc = DreamCoder(
            grammar=grammar,
            tasks=tasks,
            eval_fn=eval_program_on_hand,
            max_iterations=config.max_iterations,
            enumeration_timeout=config.enumeration_timeout,
            enumeration_budget=config.enumeration_budget,
            max_depth=config.max_depth,
            verbose=True
        )

        result = dc.run()
        all_runs.append(result)

        # Collect metrics
        for name, metrics in result.task_metrics.items():
            rule_metrics[name].append(metrics)

    # Aggregate difficulty metrics
    rule_difficulty = {}
    for name, metrics_list in rule_metrics.items():
        solved_count = sum(1 for m in metrics_list if m.solved)
        avg_iteration = sum(m.iteration_solved or float('inf') for m in metrics_list) / len(metrics_list)
        avg_programs = sum(m.programs_enumerated for m in metrics_list) / len(metrics_list)

        rule_difficulty[name] = {
            'solve_rate': solved_count / len(metrics_list),
            'avg_iteration_solved': avg_iteration,
            'avg_programs_enumerated': avg_programs,
            'difficulty_score': avg_programs * (1 + avg_iteration) / (solved_count + 0.1)
        }

    # Compute transfer matrix (simplified - would need more runs)
    transfer_matrix = {}  # TODO: Implement transfer analysis

    return ExperimentResult(
        config=config,
        runs=all_runs,
        rule_difficulty=rule_difficulty,
        transfer_matrix=transfer_matrix,
        total_time=time.time() - start_time
    )


# ============================================================================
# QUICK TEST
# ============================================================================

def quick_test():
    """Run a quick test on a few simple rules."""
    print("\n" + "=" * 60)
    print("QUICK TEST: Learning a few card game rules")
    print("=" * 60)

    # Select rules with good True/False balance and varying complexity
    test_rules = [
        'Ends_same_color',     # First and last same color (simple)
        'Ends_same_suit',      # First and last same suit (medium)
    ]

    config = ExperimentConfig(
        name="quick_test",
        rules=test_rules,
        n_examples_per_rule=20,  # More examples for better learning
        max_iterations=2,
        enumeration_timeout=300.0,  # 5 minutes per rule
        enumeration_budget=500000,  # Half million programs
        max_depth=8,  # Allow deeper programs for complex compositions
        n_runs=1
    )

    result = run_experiment(config)

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    print("\nRule Difficulty (higher = harder):")
    for name, diff in sorted(result.rule_difficulty.items(),
                             key=lambda x: x[1]['difficulty_score']):
        print(f"  {name}:")
        print(f"    Solve rate: {diff['solve_rate']*100:.0f}%")
        print(f"    Avg programs: {diff['avg_programs_enumerated']:.0f}")
        print(f"    Difficulty: {diff['difficulty_score']:.2f}")

    if result.runs:
        print(result.runs[0].summary())

    return result


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    quick_test()
