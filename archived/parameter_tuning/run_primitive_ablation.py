#!/usr/bin/env python3
"""
Primitive Ablation Experiment
=============================

This experiment tests which "chunky" convenience primitives can be replaced by
compositions of more basic primitives without losing the ability to solve rules.

GOAL: Identify primitives that could be removed from the library while maintaining
expressiveness, at the cost of increased search depth.

DESIGN:
- Baseline: Current v3 library (59 primitives)
- Ablation variants: Remove selected "convenience" primitives
- Compare: Rules solved, search cost, solution depths

ABLATION CANDIDATES (primitives expressible as compositions):
- all_same_suit, all_same_color  → eq 1 (n_unique_* hand)
- n_unique_suits/ranks/colors   → length (unique (map get_* hand))
- has_suit, has_color           → any (λ eq X (get_* $0)) hand
- count_suit, count_color       → length (filter (λ eq X (get_* $0)) hand)
- first_half, second_half       → take/drop (half_len hand) hand

NOT ABLATABLE (no fold in library):
- sum_ranks, max_rank, min_rank → would need fold

Usage:
    # Run full experiment (all variants)
    python3 run_primitive_ablation.py

    # Run specific variant
    python3 run_primitive_ablation.py --variant baseline
    python3 run_primitive_ablation.py --variant no_gestalt

    # Quick test (1 iteration, subset of rules)
    python3 run_primitive_ablation.py --quick-test

    # Dry run (show what would be run)
    python3 run_primitive_ablation.py --dry-run

Launch with caffeinate for overnight:
    nohup caffeinate -d -i -s python3 run_primitive_ablation.py > ablation.log 2>&1 &
"""

import sys
import os
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field, asdict
from copy import deepcopy

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.type_system import (
    Type, BaseType, Arrow, ListType, TypeVariable,
    BOOL, INT, CARD, SUIT, RANK, HAND, arrow, LIST_BOOL
)
from dreamcoder_core.program import Primitive
from dreamcoder_core.grammar import Grammar, uniform_grammar
from dreamcoder_core.enumeration import TopDownEnumerator, Frontier
from dreamcoder_core.task import Task
from dreamcoder_core.lean_primitives import (
    build_lean_primitives,
    make_constants, make_card_accessors, make_position_ops,
    make_list_slicing, make_direct_queries, make_aggregates,
    make_comparisons, make_boolean_ops, make_higher_order, make_arithmetic,
    COLOR
)

from rules.cards import (
    Card, Hand, Suit, Rank, Color, RANK_VALUES,
    card_color, sample_hand
)
from rules.catalogue import create_all_rules
from rules.pretraining_rules import get_all_pretraining_rules
from dreamcoder_core.task_generation import load_prerecorded_tasks

# Pre-recorded task paths
PRERECORDED_TASKS_DIR = Path(__file__).parent.parent / 'data' / 'prerecorded_tasks'
PRETRAINING_TASKS_PATH = PRERECORDED_TASKS_DIR / 'pretraining_tasks.json'
CATALOGUE_TASKS_PATH = PRERECORDED_TASKS_DIR / 'catalogue_tasks.json'


# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AblationConfig:
    """Configuration for the ablation experiment."""

    # Experiment parameters
    variant: str = 'baseline'  # Which library variant to test
    n_iterations: int = 3

    # Enumeration parameters
    enumeration_budget: int = 100_000
    enumeration_timeout: float = 60.0
    max_depth: int = 8

    # Task parameters
    n_examples_per_task: int = 50
    n_holdout_per_task: int = 20
    hand_size: int = 6

    # Output
    results_dir: str = 'results_ablation'

    def run_dir(self) -> Path:
        """Return directory for this run."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return Path(self.results_dir) / f"{self.variant}_{timestamp}"


# ============================================================================
# ABLATION VARIANTS
# ============================================================================

# Primitives to remove in each variant
ABLATION_VARIANTS = {
    'baseline': [],  # Full library

    # Individual ablations
    'no_gestalt': ['all_same_suit', 'all_same_color'],
    'no_n_unique': ['n_unique_suits', 'n_unique_ranks', 'n_unique_colors'],
    'no_has': ['has_suit', 'has_color'],
    'no_count': ['count_suit', 'count_color'],
    'no_halves': ['first_half', 'second_half'],

    # Combined ablations
    'no_gestalt_n_unique': [
        'all_same_suit', 'all_same_color',
        'n_unique_suits', 'n_unique_ranks', 'n_unique_colors'
    ],
    'no_direct_queries': [
        'all_same_suit', 'all_same_color',
        'n_unique_suits', 'n_unique_ranks', 'n_unique_colors',
        'has_suit', 'has_color',
        'count_suit', 'count_color'
    ],
    'minimal': [
        'all_same_suit', 'all_same_color',
        'n_unique_suits', 'n_unique_ranks', 'n_unique_colors',
        'has_suit', 'has_color',
        'count_suit', 'count_color',
        'first_half', 'second_half'
    ],
}

# Document what each removed primitive requires for composition
COMPOSITION_REQUIREMENTS = {
    'all_same_suit': {
        'composition': 'eq 1 (n_unique_suits hand)',
        'requires': ['eq', 'n_unique_suits'],
        'depth_increase': 2,
        'note': 'If n_unique also removed: eq 1 (length (unique (map get_suit hand)))'
    },
    'all_same_color': {
        'composition': 'eq 1 (n_unique_colors hand)',
        'requires': ['eq', 'n_unique_colors'],
        'depth_increase': 2,
    },
    'n_unique_suits': {
        'composition': 'length (unique (map get_suit hand))',
        'requires': ['length', 'unique', 'map', 'get_suit'],
        'depth_increase': 3,
    },
    'n_unique_ranks': {
        'composition': 'length (unique (map get_rank hand))',
        'requires': ['length', 'unique', 'map', 'get_rank'],
        'depth_increase': 3,
    },
    'n_unique_colors': {
        'composition': 'length (unique (map get_color hand))',
        'requires': ['length', 'unique', 'map', 'get_color'],
        'depth_increase': 3,
    },
    'has_suit': {
        'composition': 'any (λ eq SUIT (get_suit $0)) hand',
        'requires': ['any', 'eq', 'get_suit', 'SUIT_CONSTANT'],
        'depth_increase': 4,
    },
    'has_color': {
        'composition': 'any (λ eq COLOR (get_color $0)) hand',
        'requires': ['any', 'eq', 'get_color', 'COLOR_CONSTANT'],
        'depth_increase': 4,
    },
    'count_suit': {
        'composition': 'length (filter (λ eq SUIT (get_suit $0)) hand)',
        'requires': ['length', 'filter', 'eq', 'get_suit', 'SUIT_CONSTANT'],
        'depth_increase': 4,
    },
    'count_color': {
        'composition': 'length (filter (λ eq COLOR (get_color $0)) hand)',
        'requires': ['length', 'filter', 'eq', 'get_color', 'COLOR_CONSTANT'],
        'depth_increase': 4,
    },
    'first_half': {
        'composition': 'take (half_len hand) hand',
        'requires': ['take', 'half_len'],
        'depth_increase': 1,
    },
    'second_half': {
        'composition': 'drop (half_len hand) hand',
        'requires': ['drop', 'half_len'],
        'depth_increase': 1,
    },
}


# ============================================================================
# LIBRARY CONSTRUCTION
# ============================================================================

def build_ablated_grammar(variant: str) -> Tuple[Grammar, List[str]]:
    """
    Build a grammar with specified primitives removed.

    Returns:
        (grammar, list of removed primitive names)
    """
    if variant not in ABLATION_VARIANTS:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(ABLATION_VARIANTS.keys())}")

    primitives_to_remove = set(ABLATION_VARIANTS[variant])

    # Build full primitive list
    all_primitives = build_lean_primitives()

    # Filter out removed primitives
    kept_primitives = [p for p in all_primitives if p.name not in primitives_to_remove]
    removed_names = [p.name for p in all_primitives if p.name in primitives_to_remove]

    # Build grammar
    grammar = uniform_grammar(kept_primitives)

    logger.info(f"Variant '{variant}': {len(kept_primitives)} primitives "
                f"(removed {len(removed_names)}: {removed_names})")

    return grammar, removed_names


# ============================================================================
# TASK CREATION
# ============================================================================

def create_tasks(
    rules: List,
    n_examples: int = 50,
    n_holdout: int = 20,
    hand_size: int = 6
) -> List[Task]:
    """Create Task objects from rule definitions."""
    tasks = []

    for rule in rules:
        # Handle both Rule objects and PretrainingRule objects
        if hasattr(rule, 'eval'):
            rule_fn = rule.eval
            rule_name = rule.id
        elif hasattr(rule, 'predicate'):
            rule_fn = rule.predicate
            rule_name = rule.id
        else:
            rule_fn = rule
            rule_name = getattr(rule_fn, '__name__', str(rule_fn))

        # Generate training examples
        examples = []
        attempts = 0
        max_attempts = n_examples * 20

        # Ensure balanced examples (try to get both True and False)
        true_examples = []
        false_examples = []
        target_each = n_examples // 2

        while (len(true_examples) < target_each or len(false_examples) < target_each) and attempts < max_attempts:
            hand = sample_hand(hand_size)
            try:
                result = rule_fn(hand)
                if result and len(true_examples) < target_each:
                    true_examples.append((hand, True))
                elif not result and len(false_examples) < target_each:
                    false_examples.append((hand, False))
            except Exception:
                pass
            attempts += 1

        examples = true_examples + false_examples

        # Generate holdout examples
        holdout = []
        attempts = 0
        while len(holdout) < n_holdout and attempts < n_holdout * 20:
            hand = sample_hand(hand_size)
            try:
                result = rule_fn(hand)
                holdout.append((hand, result))
            except Exception:
                pass
            attempts += 1

        if len(examples) < n_examples // 2:
            logger.warning(f"Rule {rule_name}: only generated {len(examples)} examples")

        task = Task(
            name=rule_name,
            request_type=arrow(HAND, BOOL),
            examples=examples,
            holdout=holdout
        )
        tasks.append(task)

    return tasks


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_program(program, hand: Hand) -> Optional[bool]:
    """Safely evaluate a program on a hand.

    Programs of type HAND -> BOOL are lambda abstractions.
    We first evaluate with empty env to get the function,
    then call that function with the hand.
    """
    try:
        # Get the function (programs are lambdas: λhand. body)
        fn = program.evaluate([])
        # Call the function with the hand
        result = fn(hand)
        if isinstance(result, bool):
            return result
        return None
    except Exception:
        return None


def check_solution(program, examples: List[Tuple[Hand, bool]]) -> Tuple[bool, int]:
    """
    Check if a program solves all examples.

    Returns:
        (is_correct, n_correct)
    """
    n_correct = 0
    for hand, expected in examples:
        result = evaluate_program(program, hand)
        if result == expected:
            n_correct += 1
        else:
            return False, n_correct
    return True, n_correct


# ============================================================================
# MAIN ENUMERATION LOOP
# ============================================================================

@dataclass
class TaskResult:
    """Result of enumeration for a single task."""
    task_name: str
    solved: bool
    programs_enumerated: int
    solution_str: Optional[str] = None
    solution_depth: Optional[int] = None
    solution_size: Optional[int] = None
    time_seconds: float = 0.0
    holdout_verified: bool = False


@dataclass
class IterationResult:
    """Result of a single iteration."""
    iteration: int
    tasks_solved: int
    total_tasks: int
    total_programs: int
    total_time: float
    task_results: List[TaskResult] = field(default_factory=list)


def run_iteration(
    grammar: Grammar,
    tasks: List[Task],
    config: AblationConfig,
    iteration: int
) -> IterationResult:
    """Run one iteration of enumeration on all tasks."""

    logger.info(f"\n{'='*60}")
    logger.info(f"ITERATION {iteration + 1}")
    logger.info(f"{'='*60}")

    task_results = []
    total_programs = 0
    total_time = 0.0
    tasks_solved = 0

    for task_idx, task in enumerate(tasks):
        start_time = time.time()

        logger.info(f"\n[{task_idx+1}/{len(tasks)}] Task: {task.name}")

        # Create enumerator
        enumerator = TopDownEnumerator(
            grammar,
            max_depth=config.max_depth,
            max_programs=config.enumeration_budget
        )

        # Enumerate
        solution = None
        programs_tried = 0

        try:
            for program, log_prob in enumerator.enumerate_memoized(
                task.request_type,
                max_cost=50.0,
                timeout_seconds=config.enumeration_timeout
            ):
                programs_tried += 1

                # Check solution
                is_correct, _ = check_solution(program, task.examples)

                if is_correct:
                    # Verify on holdout
                    holdout_ok, _ = check_solution(program, task.holdout)

                    if holdout_ok:
                        solution = program
                        logger.info(f"  ✓ SOLVED after {programs_tried} programs: {program}")
                        break

                if programs_tried >= config.enumeration_budget:
                    break

        except Exception as e:
            logger.warning(f"  Enumeration error: {e}")

        elapsed = time.time() - start_time
        total_time += elapsed
        total_programs += programs_tried

        if solution:
            tasks_solved += 1
            task_results.append(TaskResult(
                task_name=task.name,
                solved=True,
                programs_enumerated=programs_tried,
                solution_str=str(solution),
                solution_depth=solution.depth(),
                solution_size=solution.size(),
                time_seconds=elapsed,
                holdout_verified=True
            ))
        else:
            logger.info(f"  ✗ Not solved after {programs_tried} programs ({elapsed:.1f}s)")
            task_results.append(TaskResult(
                task_name=task.name,
                solved=False,
                programs_enumerated=programs_tried,
                time_seconds=elapsed
            ))

    return IterationResult(
        iteration=iteration,
        tasks_solved=tasks_solved,
        total_tasks=len(tasks),
        total_programs=total_programs,
        total_time=total_time,
        task_results=task_results
    )


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

@dataclass
class ExperimentResult:
    """Complete result of an ablation experiment."""
    variant: str
    removed_primitives: List[str]
    n_primitives: int
    config: Dict
    iterations: List[IterationResult] = field(default_factory=list)

    def summary(self) -> Dict:
        """Generate summary statistics."""
        if not self.iterations:
            return {}

        final = self.iterations[-1]

        # Track which tasks were ever solved
        ever_solved = set()
        for it in self.iterations:
            for tr in it.task_results:
                if tr.solved:
                    ever_solved.add(tr.task_name)

        # Get solution depths for solved tasks
        depths = []
        for tr in final.task_results:
            if tr.solution_depth:
                depths.append(tr.solution_depth)

        return {
            'variant': self.variant,
            'removed': self.removed_primitives,
            'n_primitives': self.n_primitives,
            'tasks_solved_final': final.tasks_solved,
            'tasks_ever_solved': len(ever_solved),
            'total_tasks': final.total_tasks,
            'solve_rate': final.tasks_solved / final.total_tasks,
            'avg_depth': sum(depths) / len(depths) if depths else 0,
            'max_depth': max(depths) if depths else 0,
            'total_programs': sum(it.total_programs for it in self.iterations),
            'total_time': sum(it.total_time for it in self.iterations),
        }


def run_experiment(
    config: AblationConfig,
    rules: Optional[List] = None,
    prerecorded_tasks: Optional[List[Task]] = None
) -> ExperimentResult:
    """Run the complete experiment for one variant.

    Args:
        config: Ablation configuration
        rules: List of rules to create tasks from (if not using prerecorded)
        prerecorded_tasks: Pre-recorded tasks (preferred, avoids class imbalance issues)
    """

    logger.info(f"\n{'#'*70}")
    logger.info(f"# ABLATION EXPERIMENT: {config.variant}")
    logger.info(f"{'#'*70}")

    # Build grammar for this variant
    grammar, removed = build_ablated_grammar(config.variant)

    # Get tasks (either pre-recorded or generate from rules)
    if prerecorded_tasks is not None:
        tasks = prerecorded_tasks
        logger.info(f"Using {len(tasks)} pre-recorded tasks (balanced, reliable)")
    elif rules is not None:
        logger.info(f"\nCreating tasks from {len(rules)} rules...")
        tasks = create_tasks(
            rules,
            n_examples=config.n_examples_per_task,
            n_holdout=config.n_holdout_per_task,
            hand_size=config.hand_size
        )
        logger.info(f"Created {len(tasks)} tasks")
    else:
        raise ValueError("Must provide either rules or prerecorded_tasks")

    # Create result container
    result = ExperimentResult(
        variant=config.variant,
        removed_primitives=removed,
        n_primitives=len(grammar.primitives()),
        config=asdict(config)
    )

    # Run iterations
    for iteration in range(config.n_iterations):
        iter_result = run_iteration(grammar, tasks, config, iteration)
        result.iterations.append(iter_result)

        # Log progress
        logger.info(f"\nIteration {iteration + 1} Summary:")
        logger.info(f"  Solved: {iter_result.tasks_solved}/{iter_result.total_tasks}")
        logger.info(f"  Programs: {iter_result.total_programs:,}")
        logger.info(f"  Time: {iter_result.total_time:.1f}s")

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Primitive Ablation Experiment')

    parser.add_argument('--variant', type=str, default=None,
                        help='Specific variant to run (default: all)')
    parser.add_argument('--variants', type=str, nargs='+', default=None,
                        help='List of variants to run')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Number of iterations per variant')
    parser.add_argument('--budget', type=int, default=100_000,
                        help='Enumeration budget per task')
    parser.add_argument('--timeout', type=float, default=60.0,
                        help='Timeout per task in seconds')
    parser.add_argument('--max-depth', type=int, default=8,
                        help='Maximum program depth')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test mode (1 iteration, 10 rules)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be run without running')
    parser.add_argument('--use-pretraining', action='store_true',
                        help='Use pretraining rules instead of catalogue')
    parser.add_argument('--use-prerecorded', action='store_true', default=True,
                        help='Use pre-recorded tasks (default: True, more reliable)')
    parser.add_argument('--no-prerecorded', action='store_true',
                        help='Do NOT use pre-recorded tasks (generate on the fly)')
    parser.add_argument('--results-dir', type=str, default='results_ablation',
                        help='Directory for results')

    args = parser.parse_args()

    # Determine which variants to run
    if args.variant:
        variants = [args.variant]
    elif args.variants:
        variants = args.variants
    else:
        # Default: run baseline + key ablations
        variants = ['baseline', 'no_gestalt', 'no_n_unique', 'no_direct_queries', 'minimal']

    # Validate variants
    for v in variants:
        if v not in ABLATION_VARIANTS:
            print(f"Error: Unknown variant '{v}'")
            print(f"Available: {list(ABLATION_VARIANTS.keys())}")
            return 1

    # Load tasks (prefer pre-recorded for balanced examples)
    use_prerecorded = args.use_prerecorded and not args.no_prerecorded
    prerecorded_tasks = None
    rules = None

    if use_prerecorded:
        # Load pre-recorded tasks (preferred - avoids class imbalance issues)
        task_file = PRETRAINING_TASKS_PATH if args.use_pretraining else CATALOGUE_TASKS_PATH
        if task_file.exists():
            logger.info(f"Loading pre-recorded tasks from: {task_file}")
            prerecorded_tasks = load_prerecorded_tasks(task_file)
            logger.info(f"Loaded {len(prerecorded_tasks)} pre-recorded tasks")
        else:
            logger.warning(f"Pre-recorded tasks not found: {task_file}")
            logger.warning("Falling back to on-the-fly task generation")
            use_prerecorded = False

    if not use_prerecorded:
        # Fall back to generating tasks from rules
        if args.use_pretraining:
            rules = get_all_pretraining_rules()
            logger.info(f"Using {len(rules)} pretraining rules (on-the-fly generation)")
        else:
            rules = create_all_rules()
            logger.info(f"Using {len(rules)} catalogue rules (on-the-fly generation)")

    # Quick test mode
    if args.quick_test:
        args.iterations = 1
        args.budget = 10_000
        args.timeout = 30.0
        if prerecorded_tasks:
            prerecorded_tasks = prerecorded_tasks[:10]
        elif rules:
            rules = rules[:10]
        logger.info("QUICK TEST MODE: 1 iteration, 10 rules, 10k budget")

    # Dry run
    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Variants to run: {variants}")
        n_tasks = len(prerecorded_tasks) if prerecorded_tasks else len(rules) if rules else 0
        print(f"Tasks: {n_tasks} ({'pre-recorded' if prerecorded_tasks else 'from rules'})")
        print(f"Iterations: {args.iterations}")
        print(f"Budget: {args.budget:,}")
        print(f"Timeout: {args.timeout}s")
        print(f"Max depth: {args.max_depth}")
        print()

        for v in variants:
            removed = ABLATION_VARIANTS[v]
            print(f"\n{v}:")
            print(f"  Removed: {removed if removed else '(none)'}")
            if removed:
                for prim in removed:
                    if prim in COMPOSITION_REQUIREMENTS:
                        info = COMPOSITION_REQUIREMENTS[prim]
                        print(f"    {prim}: +{info['depth_increase']} depth")
                        print(f"      → {info['composition']}")
        return 0

    # Create results directory
    results_dir = Path(args.results_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = results_dir / f"ablation_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Results will be saved to: {run_dir}")

    # Run experiments
    all_results = []

    for variant in variants:
        config = AblationConfig(
            variant=variant,
            n_iterations=args.iterations,
            enumeration_budget=args.budget,
            enumeration_timeout=args.timeout,
            max_depth=args.max_depth,
            results_dir=str(run_dir)
        )

        result = run_experiment(config, rules=rules, prerecorded_tasks=prerecorded_tasks)
        all_results.append(result)

        # Save individual result
        variant_file = run_dir / f"{variant}_result.json"
        with open(variant_file, 'w') as f:
            # Convert to serializable format
            data = {
                'variant': result.variant,
                'removed_primitives': result.removed_primitives,
                'n_primitives': result.n_primitives,
                'config': result.config,
                'summary': result.summary(),
                'iterations': [
                    {
                        'iteration': it.iteration,
                        'tasks_solved': it.tasks_solved,
                        'total_tasks': it.total_tasks,
                        'total_programs': it.total_programs,
                        'total_time': it.total_time,
                        'task_results': [asdict(tr) for tr in it.task_results]
                    }
                    for it in result.iterations
                ]
            }
            json.dump(data, f, indent=2)
        logger.info(f"Saved: {variant_file}")

    # Generate comparison summary
    print("\n" + "="*80)
    print("ABLATION EXPERIMENT SUMMARY")
    print("="*80)

    print(f"\n{'Variant':<25} {'Primitives':<12} {'Solved':<10} {'Rate':<8} {'Avg Depth':<10}")
    print("-"*70)

    for result in all_results:
        s = result.summary()
        print(f"{s['variant']:<25} {s['n_primitives']:<12} "
              f"{s['tasks_solved_final']}/{s['total_tasks']:<7} "
              f"{s['solve_rate']:.1%}    {s['avg_depth']:.1f}")

    # Save comparison
    comparison_file = run_dir / "comparison.json"
    with open(comparison_file, 'w') as f:
        json.dump([r.summary() for r in all_results], f, indent=2)

    print(f"\n\nResults saved to: {run_dir}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
