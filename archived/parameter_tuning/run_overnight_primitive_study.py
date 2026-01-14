#!/usr/bin/env python3
"""
Comprehensive Overnight Primitive Library Study
================================================

This script implements the full experimental plan for determining the optimal
primitive library configuration. It runs three tracks:

Track A: Fine-Grained Ablation (individual + grouped primitive removal)
Track B: Addition Experiments (testing removed technical primitives)
Track C: Wake-Sleep Dynamics (multi-iteration library learning)

Runtime: ~10-12 hours total

Usage:
    python3 run_overnight_primitive_study.py --tracks A B C
    python3 run_overnight_primitive_study.py --tracks A --dry-run
    python3 run_overnight_primitive_study.py --tracks C --quick-test
"""

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from functools import reduce
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.type_system import (
    arrow, BOOL, INT, CARD, SUIT, RANK, HAND,
    TypeVariable, ListType, BaseType
)
from dreamcoder_core.program import Primitive
from dreamcoder_core.grammar import Grammar, Production, uniform_grammar
from dreamcoder_core.enumeration import TopDownEnumerator, enumerate_simple
from dreamcoder_core.lean_primitives import (
    build_lean_primitives, build_lean_grammar,
    make_constants, make_card_accessors, make_position_ops, make_list_slicing,
    make_direct_queries, make_aggregates, make_comparisons, make_boolean_ops,
    make_higher_order, make_arithmetic, COLOR
)
from dreamcoder_core.task import Task
from dreamcoder_core.task_generation import load_prerecorded_tasks

from rules.cards import Card, Hand, Suit, Rank, Color, RANK_VALUES, card_color, sample_hand

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ============================================================================
# PATHS
# ============================================================================

SCRIPT_DIR = Path(__file__).parent
SRC_DIR = SCRIPT_DIR.parent
DATA_DIR = SRC_DIR / 'data' / 'prerecorded_tasks'
PRETRAINING_TASKS_PATH = DATA_DIR / 'pretraining_tasks.json'
CATALOGUE_TASKS_PATH = DATA_DIR / 'catalogue_tasks.json'


# ============================================================================
# PRIMITIVES THAT CAN BE ADDED BACK
# ============================================================================

def make_addition_primitives() -> Dict[str, Primitive]:
    """
    Primitives that were removed from v3 but could be added back.

    These are 'technical' primitives, not gestalt ones.
    """
    a = TypeVariable(0)
    b = TypeVariable(1)
    c = TypeVariable(2)

    return {
        # Abstract combinators
        'compose': Primitive(
            'compose',
            arrow(arrow(b, c), arrow(a, b), a, c),
            lambda f: lambda g: lambda x: f(g(x))
        ),
        'flip': Primitive(
            'flip',
            arrow(arrow(a, b, c), b, a, c),
            lambda f: lambda x: lambda y: f(y)(x)
        ),
        'const': Primitive(
            'const',
            arrow(a, b, a),
            lambda x: lambda _: x
        ),
        'id': Primitive(
            'id',
            arrow(a, a),
            lambda x: x
        ),

        # Operators removed in v3
        'neq': Primitive(
            'neq',
            arrow(a, a, BOOL),
            lambda x: lambda y: x != y
        ),
        'fold': Primitive(
            'fold',
            arrow(arrow(a, b, b), b, ListType(a), b),
            lambda f: lambda z: lambda xs: reduce(lambda acc, x: f(x)(acc), xs, z)
        ),
        'tail': Primitive(
            'tail',
            arrow(ListType(a), ListType(a)),
            lambda xs: xs[1:] if xs else []
        ),
        'cons': Primitive(
            'cons',
            arrow(a, ListType(a), ListType(a)),
            lambda x: lambda xs: [x] + list(xs)
        ),

        # Rank constants (removed because too specific)
        'RANK_10': Primitive('RANK_10', INT, 10),
        'RANK_11': Primitive('RANK_11', INT, 11),
        'RANK_12': Primitive('RANK_12', INT, 12),
        'RANK_13': Primitive('RANK_13', INT, 13),
        'RANK_14': Primitive('RANK_14', INT, 14),
        'RANK_17': Primitive('RANK_17', INT, 17),
        'RANK_21': Primitive('RANK_21', INT, 21),
    }


# ============================================================================
# ABLATION CONFIGURATIONS
# ============================================================================

# Track A1: Individual primitive ablation
INDIVIDUAL_ABLATIONS = {
    'baseline': [],
    'no_all_same_suit': ['all_same_suit'],
    'no_all_same_color': ['all_same_color'],
    'no_n_unique_suits': ['n_unique_suits'],
    'no_n_unique_ranks': ['n_unique_ranks'],
    'no_n_unique_colors': ['n_unique_colors'],
    'no_has_suit': ['has_suit'],
    'no_has_color': ['has_color'],
    'no_count_suit': ['count_suit'],
    'no_count_color': ['count_color'],
    'no_first_half': ['first_half'],
    'no_second_half': ['second_half'],
}

# Track A2: Grouped category ablation
GROUPED_ABLATIONS = {
    'no_gestalt_perception': ['all_same_suit', 'all_same_color'],
    'no_uniqueness_queries': ['n_unique_suits', 'n_unique_ranks', 'n_unique_colors'],
    'no_membership_queries': ['has_suit', 'has_color'],
    'no_counting_queries': ['count_suit', 'count_color'],
    'no_halves': ['first_half', 'second_half'],
    'no_aggregates': ['sum_ranks', 'max_rank', 'min_rank'],
    'no_list_slicing': ['take', 'drop', 'half_len', 'adjacent_pairs'],
    'no_position_access': ['head', 'last', 'at', 'reverse'],
    'minimal_gestalt': [
        'all_same_suit', 'all_same_color',
        'n_unique_suits', 'n_unique_ranks', 'n_unique_colors',
        'has_suit', 'has_color',
        'count_suit', 'count_color',
        'first_half', 'second_half'
    ],
}

# Track A3: Interaction effects
INTERACTION_ABLATIONS = {
    'no_gestalt_no_halves': [
        'all_same_suit', 'all_same_color', 'first_half', 'second_half'
    ],
    'no_count_no_has': [
        'count_suit', 'count_color', 'has_suit', 'has_color'
    ],
    'no_unique_no_gestalt': [
        'n_unique_suits', 'n_unique_ranks', 'n_unique_colors',
        'all_same_suit', 'all_same_color'
    ],
    'no_slicing_no_position': [
        'take', 'drop', 'head', 'last', 'at'
    ],
}

# Track B: Addition experiments
ADDITION_EXPERIMENTS = {
    'add_compose': ['compose'],
    'add_flip': ['flip'],
    'add_const': ['const'],
    'add_id': ['id'],
    'add_all_combinators': ['compose', 'flip', 'const', 'id'],
    'add_neq': ['neq'],
    'add_fold': ['fold'],
    'add_tail': ['tail'],
    'add_neq_fold': ['neq', 'fold'],
    'add_face_cards': ['RANK_10', 'RANK_11', 'RANK_12', 'RANK_13', 'RANK_14'],
    'add_blackjack': ['RANK_17', 'RANK_21'],
}


# ============================================================================
# GRAMMAR BUILDING
# ============================================================================

def build_ablated_grammar(remove_primitives: List[str]) -> Tuple[Grammar, int]:
    """Build grammar with specified primitives removed."""
    all_primitives = build_lean_primitives()
    remove_set = set(remove_primitives)

    filtered = [p for p in all_primitives if str(p.program) not in remove_set]
    grammar = uniform_grammar(filtered)

    return grammar, len(filtered)


def build_extended_grammar(add_primitive_names: List[str]) -> Tuple[Grammar, int]:
    """Build grammar with specified primitives added."""
    all_primitives = build_lean_primitives()
    addition_pool = make_addition_primitives()

    for name in add_primitive_names:
        if name in addition_pool:
            all_primitives.append(addition_pool[name])
        else:
            logger.warning(f"Unknown addition primitive: {name}")

    grammar = uniform_grammar(all_primitives)
    return grammar, len(all_primitives)


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_program(program, hand: Hand) -> Optional[bool]:
    """Safely evaluate a program on a hand."""
    try:
        fn = program.evaluate([])
        result = fn(hand)
        if isinstance(result, bool):
            return result
        return None
    except Exception:
        return None


def check_solution(program, examples: List[Tuple[Hand, bool]]) -> Tuple[bool, int]:
    """Check if a program solves all examples."""
    n_correct = 0
    for hand, expected in examples:
        result = evaluate_program(program, hand)
        if result == expected:
            n_correct += 1
        else:
            return False, n_correct
    return True, n_correct


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    variant_type: str  # 'ablation' or 'addition'
    primitives_changed: List[str]
    budget: int = 100_000
    timeout: float = 60.0
    max_depth: int = 8


@dataclass
class ExperimentResult:
    """Result of a single experiment."""
    name: str
    n_primitives: int
    tasks_solved: int
    total_tasks: int
    solve_rate: float
    avg_depth: float
    total_programs: int
    total_time: float
    solutions: Dict[str, str]


def run_single_experiment(
    config: ExperimentConfig,
    tasks: List[Task],
    grammar: Grammar,
    n_primitives: int
) -> ExperimentResult:
    """Run enumeration experiment with given grammar."""

    logger.info(f"\n--- {config.name} ({n_primitives} primitives) ---")

    solutions = {}
    depths = []
    total_programs = 0
    start_time = time.time()

    for i, task in enumerate(tasks, 1):
        task_start = time.time()
        solved = False

        enumerator = TopDownEnumerator(grammar, max_depth=config.max_depth, max_programs=config.budget)

        for program, log_prob in enumerator.enumerate_memoized(
            arrow(HAND, BOOL),
            max_cost=100.0
        ):
            total_programs += 1

            if total_programs > config.budget:
                break
            if time.time() - task_start > config.timeout:
                break

            is_correct, _ = check_solution(program, task.examples)

            if is_correct:
                # Verify on holdout
                holdout_correct, _ = check_solution(program, task.holdout)
                if holdout_correct:
                    solutions[task.name] = str(program)
                    depths.append(program.depth())
                    solved = True
                    break

        status = "✓" if solved else "✗"
        logger.info(f"  [{i}/{len(tasks)}] {task.name}: {status}")

    elapsed = time.time() - start_time

    return ExperimentResult(
        name=config.name,
        n_primitives=n_primitives,
        tasks_solved=len(solutions),
        total_tasks=len(tasks),
        solve_rate=len(solutions) / len(tasks) if tasks else 0,
        avg_depth=sum(depths) / len(depths) if depths else 0,
        total_programs=total_programs,
        total_time=elapsed,
        solutions=solutions
    )


# ============================================================================
# TRACK RUNNERS
# ============================================================================

def run_track_A(
    tasks: List[Task],
    output_dir: Path,
    budget: int = 100_000,
    timeout: float = 60.0,
    quick_test: bool = False
) -> Dict[str, ExperimentResult]:
    """Track A: Fine-Grained Ablation Experiments."""

    logger.info("\n" + "=" * 70)
    logger.info("TRACK A: FINE-GRAINED ABLATION EXPERIMENTS")
    logger.info("=" * 70)

    results = {}

    # A1: Individual ablations
    logger.info("\n[A1] Individual Primitive Ablation")
    a1_dir = output_dir / 'A1_individual'
    a1_dir.mkdir(parents=True, exist_ok=True)

    variants = INDIVIDUAL_ABLATIONS
    if quick_test:
        variants = {k: v for i, (k, v) in enumerate(variants.items()) if i < 3}

    for name, remove in variants.items():
        grammar, n_prims = build_ablated_grammar(remove)
        config = ExperimentConfig(
            name=name,
            variant_type='ablation',
            primitives_changed=remove,
            budget=budget,
            timeout=timeout
        )
        result = run_single_experiment(config, tasks, grammar, n_prims)
        results[f"A1_{name}"] = result

        # Save individual result
        with open(a1_dir / f"{name}.json", 'w') as f:
            json.dump(asdict(result), f, indent=2)

    # A2: Grouped ablations
    logger.info("\n[A2] Grouped Category Ablation")
    a2_dir = output_dir / 'A2_grouped'
    a2_dir.mkdir(parents=True, exist_ok=True)

    variants = GROUPED_ABLATIONS
    if quick_test:
        variants = {k: v for i, (k, v) in enumerate(variants.items()) if i < 3}

    for name, remove in variants.items():
        grammar, n_prims = build_ablated_grammar(remove)
        config = ExperimentConfig(
            name=name,
            variant_type='ablation',
            primitives_changed=remove,
            budget=budget,
            timeout=timeout
        )
        result = run_single_experiment(config, tasks, grammar, n_prims)
        results[f"A2_{name}"] = result

        with open(a2_dir / f"{name}.json", 'w') as f:
            json.dump(asdict(result), f, indent=2)

    # A3: Interaction effects
    if not quick_test:
        logger.info("\n[A3] Interaction Effects")
        a3_dir = output_dir / 'A3_interactions'
        a3_dir.mkdir(parents=True, exist_ok=True)

        for name, remove in INTERACTION_ABLATIONS.items():
            grammar, n_prims = build_ablated_grammar(remove)
            config = ExperimentConfig(
                name=name,
                variant_type='ablation',
                primitives_changed=remove,
                budget=budget,
                timeout=timeout
            )
            result = run_single_experiment(config, tasks, grammar, n_prims)
            results[f"A3_{name}"] = result

            with open(a3_dir / f"{name}.json", 'w') as f:
                json.dump(asdict(result), f, indent=2)

    return results


def run_track_B(
    tasks: List[Task],
    output_dir: Path,
    budget: int = 100_000,
    timeout: float = 60.0,
    quick_test: bool = False
) -> Dict[str, ExperimentResult]:
    """Track B: Addition Experiments."""

    logger.info("\n" + "=" * 70)
    logger.info("TRACK B: ADDITION EXPERIMENTS")
    logger.info("=" * 70)

    results = {}
    b_dir = output_dir / 'track_B_addition'
    b_dir.mkdir(parents=True, exist_ok=True)

    # Baseline (no additions)
    grammar, n_prims = build_lean_grammar(), 59
    baseline_config = ExperimentConfig(
        name='baseline_no_additions',
        variant_type='addition',
        primitives_changed=[],
        budget=budget,
        timeout=timeout
    )
    result = run_single_experiment(baseline_config, tasks, grammar, n_prims)
    results['B_baseline'] = result

    # Addition variants
    variants = ADDITION_EXPERIMENTS
    if quick_test:
        variants = {k: v for i, (k, v) in enumerate(variants.items()) if i < 4}

    for name, add_prims in variants.items():
        grammar, n_prims = build_extended_grammar(add_prims)
        config = ExperimentConfig(
            name=name,
            variant_type='addition',
            primitives_changed=add_prims,
            budget=budget,
            timeout=timeout
        )
        result = run_single_experiment(config, tasks, grammar, n_prims)
        results[f"B_{name}"] = result

        with open(b_dir / f"{name}.json", 'w') as f:
            json.dump(asdict(result), f, indent=2)

    return results


def run_track_C(
    tasks: List[Task],
    output_dir: Path,
    budget: int = 150_000,
    timeout: float = 120.0,
    iterations: int = 6,
    quick_test: bool = False
) -> Dict[str, Any]:
    """
    Track C: Wake-Sleep Dynamics.

    This tests multi-iteration learning with recognition model training.
    """
    logger.info("\n" + "=" * 70)
    logger.info("TRACK C: WAKE-SLEEP DYNAMICS")
    logger.info("=" * 70)

    if quick_test:
        iterations = 2
        budget = 50_000

    results = {}
    c_dir = output_dir / 'track_C_wakesleep'
    c_dir.mkdir(parents=True, exist_ok=True)

    # C1: Baseline wake-sleep (full library)
    logger.info("\n[C1] Wake-Sleep with Full Library")
    c1_results = run_wakesleep_experiment(
        tasks=tasks,
        variant='full_library',
        remove_primitives=[],
        iterations=iterations,
        budget=budget,
        timeout=timeout,
        output_dir=c_dir / 'C1_full_library'
    )
    results['C1_full_library'] = c1_results

    # C2: Wake-sleep with minimal library
    logger.info("\n[C2] Wake-Sleep with Minimal Library")
    c2_results = run_wakesleep_experiment(
        tasks=tasks,
        variant='minimal_library',
        remove_primitives=GROUPED_ABLATIONS['minimal_gestalt'],
        iterations=iterations,
        budget=budget,
        timeout=timeout,
        output_dir=c_dir / 'C2_minimal_library'
    )
    results['C2_minimal_library'] = c2_results

    # C3: Wake-sleep with no gestalt
    logger.info("\n[C3] Wake-Sleep with No Gestalt")
    c3_results = run_wakesleep_experiment(
        tasks=tasks,
        variant='no_gestalt',
        remove_primitives=GROUPED_ABLATIONS['no_gestalt_perception'],
        iterations=iterations,
        budget=budget,
        timeout=timeout,
        output_dir=c_dir / 'C3_no_gestalt'
    )
    results['C3_no_gestalt'] = c3_results

    return results


def run_wakesleep_experiment(
    tasks: List[Task],
    variant: str,
    remove_primitives: List[str],
    iterations: int,
    budget: int,
    timeout: float,
    output_dir: Path
) -> Dict[str, Any]:
    """Run a multi-iteration wake-sleep experiment."""

    output_dir.mkdir(parents=True, exist_ok=True)

    grammar, n_prims = build_ablated_grammar(remove_primitives)

    iteration_results = []
    cumulative_solved = set()

    for iteration in range(1, iterations + 1):
        logger.info(f"\n=== {variant} Iteration {iteration}/{iterations} ===")

        config = ExperimentConfig(
            name=f"{variant}_iter{iteration}",
            variant_type='wakesleep',
            primitives_changed=remove_primitives,
            budget=budget,
            timeout=timeout
        )

        result = run_single_experiment(config, tasks, grammar, n_prims)

        # Track cumulative progress
        for task_name in result.solutions:
            cumulative_solved.add(task_name)

        iter_data = {
            'iteration': iteration,
            'solved_this_iter': result.tasks_solved,
            'cumulative_solved': len(cumulative_solved),
            'total_tasks': result.total_tasks,
            'solve_rate': result.solve_rate,
            'cumulative_rate': len(cumulative_solved) / result.total_tasks,
            'time': result.total_time,
            'solutions': result.solutions
        }
        iteration_results.append(iter_data)

        # Save iteration result
        with open(output_dir / f"iter_{iteration}.json", 'w') as f:
            json.dump(iter_data, f, indent=2)

    # Summary
    summary = {
        'variant': variant,
        'n_primitives': n_prims,
        'removed': remove_primitives,
        'iterations': iterations,
        'final_cumulative_solved': len(cumulative_solved),
        'final_cumulative_rate': len(cumulative_solved) / len(tasks),
        'iteration_results': iteration_results
    }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_summary_report(
    all_results: Dict[str, Any],
    output_dir: Path
):
    """Generate a comprehensive summary report."""

    report_lines = [
        "# Overnight Primitive Library Study - Summary Report",
        f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "\n---\n"
    ]

    # Track A Summary
    if any(k.startswith('A') for k in all_results):
        report_lines.append("## Track A: Ablation Experiments\n")
        report_lines.append("| Variant | Primitives | Solved | Rate | Avg Depth |")
        report_lines.append("|---------|------------|--------|------|-----------|")

        for key, result in sorted(all_results.items()):
            if key.startswith('A') and isinstance(result, ExperimentResult):
                report_lines.append(
                    f"| {result.name} | {result.n_primitives} | "
                    f"{result.tasks_solved}/{result.total_tasks} | "
                    f"{result.solve_rate:.1%} | {result.avg_depth:.1f} |"
                )
        report_lines.append("")

    # Track B Summary
    if any(k.startswith('B') for k in all_results):
        report_lines.append("## Track B: Addition Experiments\n")
        report_lines.append("| Variant | Primitives | Solved | Rate | Delta |")
        report_lines.append("|---------|------------|--------|------|-------|")

        baseline_rate = 0
        for key, result in sorted(all_results.items()):
            if key == 'B_baseline':
                baseline_rate = result.solve_rate

        for key, result in sorted(all_results.items()):
            if key.startswith('B') and isinstance(result, ExperimentResult):
                delta = result.solve_rate - baseline_rate
                delta_str = f"+{delta:.1%}" if delta > 0 else f"{delta:.1%}"
                report_lines.append(
                    f"| {result.name} | {result.n_primitives} | "
                    f"{result.tasks_solved}/{result.total_tasks} | "
                    f"{result.solve_rate:.1%} | {delta_str} |"
                )
        report_lines.append("")

    # Track C Summary
    if any(k.startswith('C') for k in all_results):
        report_lines.append("## Track C: Wake-Sleep Dynamics\n")

        for key, result in sorted(all_results.items()):
            if key.startswith('C') and isinstance(result, dict):
                report_lines.append(f"### {result['variant']}")
                report_lines.append(f"- Primitives: {result['n_primitives']}")
                report_lines.append(f"- Final solve rate: {result['final_cumulative_rate']:.1%}")
                report_lines.append(f"- Iterations: {result['iterations']}")
                report_lines.append("")

    # Write report
    report_path = output_dir / 'summary_report.md'
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"Report saved to: {report_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Overnight Primitive Library Study'
    )

    parser.add_argument('--tracks', type=str, nargs='+', default=['A', 'B', 'C'],
                        choices=['A', 'B', 'C'],
                        help='Which tracks to run')
    parser.add_argument('--output-dir', type=str,
                        default='results_overnight_primitive_study',
                        help='Output directory')
    parser.add_argument('--budget', type=int, default=100_000,
                        help='Enumeration budget per task')
    parser.add_argument('--timeout', type=float, default=60.0,
                        help='Timeout per task in seconds')
    parser.add_argument('--iterations', type=int, default=6,
                        help='Iterations for Track C')
    parser.add_argument('--use-pretraining', action='store_true',
                        help='Use pretraining tasks (default: both)')
    parser.add_argument('--use-catalogue', action='store_true',
                        help='Use catalogue tasks (default: both)')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test mode (fewer variants)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be run without running')

    args = parser.parse_args()

    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"study_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load tasks
    tasks = []
    if args.use_pretraining or (not args.use_pretraining and not args.use_catalogue):
        if PRETRAINING_TASKS_PATH.exists():
            tasks.extend(load_prerecorded_tasks(PRETRAINING_TASKS_PATH))
            logger.info(f"Loaded {len(tasks)} pretraining tasks")

    if args.use_catalogue or (not args.use_pretraining and not args.use_catalogue):
        if CATALOGUE_TASKS_PATH.exists():
            catalogue_tasks = load_prerecorded_tasks(CATALOGUE_TASKS_PATH)
            tasks.extend(catalogue_tasks)
            logger.info(f"Loaded {len(catalogue_tasks)} catalogue tasks")

    if not tasks:
        logger.error("No tasks loaded!")
        return 1

    logger.info(f"Total tasks: {len(tasks)}")

    # Dry run
    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Tracks: {args.tracks}")
        print(f"Tasks: {len(tasks)}")
        print(f"Budget: {args.budget:,}")
        print(f"Output: {output_dir}")

        if 'A' in args.tracks:
            print(f"\nTrack A: {len(INDIVIDUAL_ABLATIONS) + len(GROUPED_ABLATIONS) + len(INTERACTION_ABLATIONS)} variants")
        if 'B' in args.tracks:
            print(f"Track B: {len(ADDITION_EXPERIMENTS) + 1} variants")
        if 'C' in args.tracks:
            print(f"Track C: 3 wake-sleep experiments × {args.iterations} iterations")

        return 0

    # Run experiments
    logger.info(f"\nOutput directory: {output_dir}")

    all_results = {}
    start_time = time.time()

    if 'A' in args.tracks:
        track_a_results = run_track_A(
            tasks=tasks,
            output_dir=output_dir / 'track_A_ablation',
            budget=args.budget,
            timeout=args.timeout,
            quick_test=args.quick_test
        )
        all_results.update(track_a_results)

    if 'B' in args.tracks:
        track_b_results = run_track_B(
            tasks=tasks,
            output_dir=output_dir,
            budget=args.budget,
            timeout=args.timeout,
            quick_test=args.quick_test
        )
        all_results.update(track_b_results)

    if 'C' in args.tracks:
        track_c_results = run_track_C(
            tasks=tasks,
            output_dir=output_dir,
            budget=args.budget * 1.5 if not args.quick_test else args.budget,
            timeout=args.timeout * 2 if not args.quick_test else args.timeout,
            iterations=args.iterations,
            quick_test=args.quick_test
        )
        all_results.update(track_c_results)

    # Generate report
    generate_summary_report(all_results, output_dir)

    # Final summary
    elapsed = time.time() - start_time
    logger.info(f"\n{'=' * 70}")
    logger.info(f"STUDY COMPLETE")
    logger.info(f"Total time: {timedelta(seconds=int(elapsed))}")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'=' * 70}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
