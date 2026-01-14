#!/usr/bin/env python3
"""
Comprehensive Library Ablation Study

PURPOSE:
Find the minimal sufficient primitive library that can solve all 45 rules.
This script systematically tests which primitives are truly essential vs.
which are "shortcuts" that can be composed from more basic operations.

RESEARCH QUESTIONS:
1. Which primitive categories are essential vs. optional?
2. Can high-level primitives (n_unique, has_suit, etc.) be replaced by compositions?
3. What is the minimal library that achieves full coverage?
4. How does library size affect search efficiency?

ABLATION STRATEGY:
The 59 primitives are organized into categories. We test:

1. CATEGORY ABLATIONS (8 variants):
   - Remove each category individually to measure impact

2. "SHORTCUT" PRIMITIVE ABLATIONS (6 variants):
   - Remove high-level primitives that could be composed
   - direct_queries: has_suit, has_color, count_suit, count_color, n_unique_*
   - aggregates: sum_ranks, max_rank, min_rank
   - halves: first_half, second_half, half_len

3. COMBINATION ABLATIONS (12+ variants):
   - Remove multiple categories together
   - Test which combinations still achieve good coverage

4. MINIMAL CORE EXPERIMENTS (5 variants):
   - Progressively add categories to find minimal sufficient set

CONFIGURATION:
- Budget: 1M programs per task (very thorough search)
- Timeout: 600s per task (allows deep search)
- Iterations: 5 per phase (enough for library learning)
- Two-phase transfer: pretraining → catalogue

USAGE:
    # Full study (estimated 48-72 hours)
    PYTHONUNBUFFERED=1 nohup caffeinate -d -i -s python3 experiments/run_comprehensive_library_ablation.py > library_ablation.out 2>&1 &

    # Quick test (1 variant, reduced params)
    python3 experiments/run_comprehensive_library_ablation.py --quick-test

    # Specific variants only
    python3 experiments/run_comprehensive_library_ablation.py --variants baseline no_direct_queries no_aggregates

    # List all variants
    python3 experiments/run_comprehensive_library_ablation.py --list-variants

    # Dry run (show what would be tested)
    python3 experiments/run_comprehensive_library_ablation.py --dry-run
"""

import sys
import os
import json
import argparse
import logging
import torch
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple, Any, Set
from itertools import combinations

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.grammar import Grammar, uniform_grammar
from dreamcoder_core.primitives import (
    build_primitives,
    make_constants,
    make_card_accessors,
    make_position_ops,
    make_list_slicing,
    make_direct_queries,
    make_aggregates,
    make_comparisons,
    make_boolean_ops,
    make_higher_order,
    make_arithmetic,
)
from dreamcoder_core.task import Task
from dreamcoder_core.task_generation import load_prerecorded_tasks
from dreamcoder_core.contrastive_wake_sleep import ContrastiveWakeSleep
from dreamcoder_core.program import Program
from rules.cards import sample_hand

# ============================================================================
# PATHS
# ============================================================================

SRC_DIR = Path(__file__).parent.parent
PRETRAINING_TASKS_PATH = SRC_DIR / 'data' / 'prerecorded_tasks' / 'pretraining_tasks.json'
CATALOGUE_TASKS_PATH = SRC_DIR / 'data' / 'prerecorded_tasks' / 'catalogue_tasks.json'

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# PRIMITIVE CATEGORY DEFINITIONS
# ============================================================================

# Map category names to the primitives they contain
PRIMITIVE_CATEGORIES = {
    'constants': [p.name for p in make_constants()],
    'card_accessors': [p.name for p in make_card_accessors()],
    'position_ops': [p.name for p in make_position_ops()],
    'list_slicing': [p.name for p in make_list_slicing()],
    'direct_queries': [p.name for p in make_direct_queries()],
    'aggregates': [p.name for p in make_aggregates()],
    'comparisons': [p.name for p in make_comparisons()],
    'boolean_ops': [p.name for p in make_boolean_ops()],
    'higher_order': [p.name for p in make_higher_order()],
    'arithmetic': [p.name for p in make_arithmetic()],
}

# Subgroups within categories (for finer-grained ablation)
PRIMITIVE_SUBGROUPS = {
    # Within direct_queries
    'has_primitives': ['has_suit', 'has_color'],
    'count_primitives': ['count_suit', 'count_color'],
    'n_unique_primitives': ['n_unique_suits', 'n_unique_ranks', 'n_unique_colors'],

    # Within list_slicing
    'halves_primitives': ['first_half', 'second_half', 'half_len'],
    'zip_primitives': ['zip_with', 'adjacent_pairs'],
    'take_drop': ['take', 'drop'],

    # Within aggregates
    'rank_aggregates': ['sum_ranks', 'max_rank', 'min_rank'],

    # Within position_ops
    'endpoints': ['head', 'last'],
    'indexing': ['at', 'length'],

    # Within higher_order
    'quantifiers': ['all', 'any'],
    'transformers': ['map', 'filter'],
}

# Essential primitives that should never be removed (core type system support)
ESSENTIAL_PRIMITIVES = {
    'true', 'false',  # Boolean constants
    'eq',  # Equality is fundamental
    'and', 'or', 'not',  # Boolean operations
}

# ============================================================================
# ABLATION VARIANT DEFINITIONS
# ============================================================================

@dataclass
class AblationVariant:
    """Definition of one ablation variant to test."""
    name: str
    description: str
    remove_primitives: List[str]
    priority: int = 1  # Lower = run first (for important variants)

    def __post_init__(self):
        # Filter out essential primitives
        self.remove_primitives = [
            p for p in self.remove_primitives
            if p not in ESSENTIAL_PRIMITIVES
        ]

    @property
    def n_removed(self) -> int:
        return len(self.remove_primitives)


def build_all_variants() -> List[AblationVariant]:
    """
    Build the comprehensive list of ablation variants to test.

    Organized by priority:
    1. Baseline (control)
    2. Category ablations (remove one category at a time)
    3. Shortcut ablations (remove convenience primitives)
    4. Combination ablations (remove multiple categories)
    5. Minimal core experiments (start minimal, add back)
    """
    variants = []

    # =========================================================================
    # 1. BASELINE (Control)
    # =========================================================================
    variants.append(AblationVariant(
        name='baseline',
        description='Full 59-primitive library (control condition)',
        remove_primitives=[],
        priority=1
    ))

    # =========================================================================
    # 2. CATEGORY ABLATIONS (Remove one category at a time)
    # =========================================================================

    # Test removing each non-essential category
    removable_categories = [
        ('direct_queries', 'Remove all direct query primitives'),
        ('aggregates', 'Remove aggregate primitives (sum/max/min_ranks)'),
        ('list_slicing', 'Remove list slicing (take/drop/zip_with/halves)'),
        ('position_ops', 'Remove position operations (head/last/at/length)'),
        ('higher_order', 'Remove higher-order (map/filter/all/any)'),
        ('arithmetic', 'Remove arithmetic (+/-/mod)'),
    ]

    for category, desc in removable_categories:
        variants.append(AblationVariant(
            name=f'no_{category}',
            description=desc,
            remove_primitives=PRIMITIVE_CATEGORIES[category],
            priority=2
        ))

    # =========================================================================
    # 3. SHORTCUT ABLATIONS (Remove convenience/composed primitives)
    # =========================================================================

    # These are high-level primitives that could theoretically be composed
    shortcut_ablations = [
        ('no_n_unique', 'Remove n_unique_* (test if composable via unique+length)',
         PRIMITIVE_SUBGROUPS['n_unique_primitives']),

        ('no_has', 'Remove has_suit/has_color (test if composable via any)',
         PRIMITIVE_SUBGROUPS['has_primitives']),

        ('no_count', 'Remove count_suit/count_color (test if composable via filter+length)',
         PRIMITIVE_SUBGROUPS['count_primitives']),

        ('no_halves', 'Remove halves primitives (test if composable via take/drop)',
         PRIMITIVE_SUBGROUPS['halves_primitives']),

        ('no_zip', 'Remove zip_with/adjacent_pairs',
         PRIMITIVE_SUBGROUPS['zip_primitives']),
    ]

    for name, desc, prims in shortcut_ablations:
        variants.append(AblationVariant(
            name=name,
            description=desc,
            remove_primitives=prims,
            priority=3
        ))

    # =========================================================================
    # 4. COMBINATION ABLATIONS (Remove multiple categories/groups)
    # =========================================================================

    # Key combinations to test
    combination_ablations = [
        # Remove all "convenience" primitives
        ('no_shortcuts', 'Remove all shortcut primitives (n_unique + has + count + halves + aggregates)',
         PRIMITIVE_SUBGROUPS['n_unique_primitives'] +
         PRIMITIVE_SUBGROUPS['has_primitives'] +
         PRIMITIVE_SUBGROUPS['count_primitives'] +
         PRIMITIVE_SUBGROUPS['halves_primitives'] +
         PRIMITIVE_SUBGROUPS['rank_aggregates']),

        # Remove direct queries but keep aggregates
        ('no_queries_keep_agg', 'Remove direct queries, keep aggregates',
         PRIMITIVE_CATEGORIES['direct_queries']),

        # Remove aggregates but keep queries
        ('no_agg_keep_queries', 'Remove aggregates, keep direct queries',
         PRIMITIVE_CATEGORIES['aggregates']),

        # Remove both direct_queries and aggregates
        ('no_high_level', 'Remove both direct queries AND aggregates',
         PRIMITIVE_CATEGORIES['direct_queries'] + PRIMITIVE_CATEGORIES['aggregates']),

        # Remove list_slicing and position_ops (test minimal list ops)
        ('minimal_lists', 'Remove most list operations (keep only higher-order)',
         PRIMITIVE_CATEGORIES['list_slicing'] + PRIMITIVE_SUBGROUPS['endpoints'] + PRIMITIVE_SUBGROUPS['indexing']),

        # Test without arithmetic
        ('no_arithmetic_full', 'Remove all arithmetic operations',
         PRIMITIVE_CATEGORIES['arithmetic']),

        # Combined: no shortcuts + no arithmetic
        ('lean_library', 'Remove shortcuts + arithmetic (lean library)',
         PRIMITIVE_SUBGROUPS['n_unique_primitives'] +
         PRIMITIVE_SUBGROUPS['has_primitives'] +
         PRIMITIVE_SUBGROUPS['count_primitives'] +
         PRIMITIVE_SUBGROUPS['halves_primitives'] +
         PRIMITIVE_SUBGROUPS['rank_aggregates'] +
         PRIMITIVE_CATEGORIES['arithmetic']),
    ]

    for name, desc, prims in combination_ablations:
        variants.append(AblationVariant(
            name=name,
            description=desc,
            remove_primitives=prims,
            priority=4
        ))

    # =========================================================================
    # 5. MINIMAL CORE EXPERIMENTS
    # =========================================================================

    # Start with minimal set and progressively add
    all_primitives = build_primitives()
    all_prim_names = {p.name for p in all_primitives}

    # Minimal core: constants + card_accessors + comparisons + boolean_ops
    minimal_core = (
        set(PRIMITIVE_CATEGORIES['constants']) |
        set(PRIMITIVE_CATEGORIES['card_accessors']) |
        set(PRIMITIVE_CATEGORIES['comparisons']) |
        set(PRIMITIVE_CATEGORIES['boolean_ops'])
    )

    # What to remove to get minimal core
    to_remove_for_minimal = list(all_prim_names - minimal_core)

    variants.append(AblationVariant(
        name='minimal_core',
        description='Minimal core: constants + card_accessors + comparisons + boolean_ops',
        remove_primitives=to_remove_for_minimal,
        priority=5
    ))

    # Minimal + higher order
    minimal_plus_ho = minimal_core | set(PRIMITIVE_CATEGORIES['higher_order'])
    to_remove_plus_ho = list(all_prim_names - minimal_plus_ho)

    variants.append(AblationVariant(
        name='minimal_plus_higher_order',
        description='Minimal core + higher-order functions (map/filter/all/any)',
        remove_primitives=to_remove_plus_ho,
        priority=5
    ))

    # Minimal + higher order + position
    minimal_plus_ho_pos = minimal_plus_ho | set(PRIMITIVE_CATEGORIES['position_ops'])
    to_remove_plus_ho_pos = list(all_prim_names - minimal_plus_ho_pos)

    variants.append(AblationVariant(
        name='minimal_plus_ho_position',
        description='Minimal + higher-order + position operations',
        remove_primitives=to_remove_plus_ho_pos,
        priority=5
    ))

    # Minimal + higher order + position + list_slicing
    minimal_plus_lists = minimal_plus_ho_pos | set(PRIMITIVE_CATEGORIES['list_slicing'])
    to_remove_plus_lists = list(all_prim_names - minimal_plus_lists)

    variants.append(AblationVariant(
        name='minimal_plus_lists',
        description='Minimal + higher-order + position + list slicing',
        remove_primitives=to_remove_plus_lists,
        priority=5
    ))

    # Sort by priority, then name
    variants.sort(key=lambda v: (v.priority, v.name))

    return variants


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

@dataclass
class StudyConfig:
    """Configuration for the entire ablation study."""
    # Experiment parameters
    iterations_phase1: int = 5  # Pretraining iterations
    iterations_phase2: int = 5  # Catalogue iterations
    enumeration_budget: int = 1_000_000  # 1M programs per task
    timeout_seconds: int = 600  # 10 minutes per task
    recognition_epochs: int = 20
    max_depth: int = 12  # Allow deep programs

    # Output
    results_dir: Path = field(default_factory=lambda: SRC_DIR / 'results_library_ablation')

    # Variants to run (empty = all)
    variants_to_run: List[str] = field(default_factory=list)

    # Quick test mode
    quick_test: bool = False

    # Overnight mode (12-hour run with high-priority variants)
    overnight: bool = False

    def __post_init__(self):
        if self.quick_test:
            self.iterations_phase1 = 2
            self.iterations_phase2 = 2
            self.enumeration_budget = 50_000
            self.timeout_seconds = 60
            self.recognition_epochs = 5
        elif self.overnight:
            # Intermediate parameters for ~12 hour run
            self.iterations_phase1 = 4
            self.iterations_phase2 = 4
            self.enumeration_budget = 300_000  # 300K programs per task
            self.timeout_seconds = 180  # 3 minutes per task
            self.recognition_epochs = 15


# High-priority variants for overnight run (~6 variants, ~12 hours)
OVERNIGHT_VARIANTS = [
    'baseline',           # Control condition (essential)
    'no_direct_queries',  # Remove has_*, count_*, n_unique_* (high impact)
    'no_higher_order',    # Remove map/filter/all/any (already tested)
    'no_shortcuts',       # Remove all convenience primitives
    'minimal_core',       # Absolute minimum (constants + accessors + comparisons + boolean)
    'minimal_plus_higher_order',  # Minimum + higher-order (test if HO helps minimal)
]


# ============================================================================
# GRAMMAR BUILDING
# ============================================================================

def build_ablated_grammar(variant: AblationVariant) -> Tuple[Grammar, List[str], int]:
    """
    Build a grammar with specified primitives removed.

    Returns:
        (grammar, list of kept primitive names, count of removed)
    """
    to_remove = set(variant.remove_primitives)

    # Build full primitive list
    all_primitives = build_primitives()

    # Filter
    kept = [p for p in all_primitives if p.name not in to_remove]
    removed = [p.name for p in all_primitives if p.name in to_remove]

    grammar = uniform_grammar(kept)

    logger.info(f"Variant '{variant.name}': {len(kept)} primitives "
                f"(removed {len(removed)})")

    return grammar, [p.name for p in kept], len(removed)


# ============================================================================
# TASK LOADING
# ============================================================================

def load_pretraining_tasks() -> List[Task]:
    """Load prerecorded pretraining tasks."""
    if not PRETRAINING_TASKS_PATH.exists():
        raise FileNotFoundError(f"Pretraining tasks not found: {PRETRAINING_TASKS_PATH}")
    tasks = load_prerecorded_tasks(PRETRAINING_TASKS_PATH)
    logger.info(f"Loaded {len(tasks)} pretraining tasks")
    return tasks


def load_catalogue_tasks() -> List[Task]:
    """Load prerecorded catalogue tasks."""
    if not CATALOGUE_TASKS_PATH.exists():
        raise FileNotFoundError(f"Catalogue tasks not found: {CATALOGUE_TASKS_PATH}")
    tasks = load_prerecorded_tasks(CATALOGUE_TASKS_PATH)
    logger.info(f"Loaded {len(tasks)} catalogue tasks")
    return tasks


# ============================================================================
# RESULT DATA STRUCTURES
# ============================================================================

@dataclass
class PhaseResult:
    """Result from one phase of experiment."""
    phase: int
    tasks_total: int
    tasks_solved: int
    solve_rate: float
    abstractions_learned: int
    final_grammar_size: int
    total_programs_enumerated: int
    total_time_seconds: float
    solved_task_names: List[str]


@dataclass
class VariantResult:
    """Complete result for one variant."""
    name: str
    description: str
    removed_primitives: List[str]
    kept_primitives: List[str]
    initial_grammar_size: int

    phase1: Optional[PhaseResult] = None
    phase2: Optional[PhaseResult] = None

    start_time: str = ""
    end_time: str = ""

    def total_solved(self) -> int:
        p1 = self.phase1.tasks_solved if self.phase1 else 0
        p2 = self.phase2.tasks_solved if self.phase2 else 0
        return p1 + p2

    def total_tasks(self) -> int:
        p1 = self.phase1.tasks_total if self.phase1 else 0
        p2 = self.phase2.tasks_total if self.phase2 else 0
        return p1 + p2

    def solve_rate(self) -> float:
        total = self.total_tasks()
        return self.total_solved() / total if total > 0 else 0.0


# ============================================================================
# EVALUATION HELPER
# ============================================================================

def eval_program(program: Program, hand) -> Optional[bool]:
    """Safely evaluate a program on a hand of cards."""
    try:
        fn = program.evaluate([])  # Get function from program
        result = fn(hand)          # Apply to hand
        return result if isinstance(result, bool) else None
    except Exception:
        return None


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

def run_phase(
    tasks: List[Task],
    grammar: Grammar,
    config: StudyConfig,
    phase_num: int,
    variant_name: str,
    results_dir: Path,
    transfer_state: Optional[Dict] = None,
) -> Tuple[PhaseResult, Dict]:
    """
    Run one phase of the experiment.

    Returns:
        (PhaseResult, transfer_state_for_next_phase)
    """
    import time
    import copy

    phase_dir = results_dir / f'phase{phase_num}'
    phase_dir.mkdir(parents=True, exist_ok=True)

    iterations = config.iterations_phase1 if phase_num == 1 else config.iterations_phase2

    # Create wake-sleep learner
    learner = ContrastiveWakeSleep(
        grammar=grammar,
        tasks=tasks,
        eval_fn=eval_program,
        sample_hand_fn=lambda: sample_hand(6),
        sample_card_fn=lambda: sample_hand(1)[0],
        enumeration_budget=config.enumeration_budget,
        enumeration_timeout=config.timeout_seconds,
        max_depth=config.max_depth,
        max_iterations=iterations,
        recognition_epochs=config.recognition_epochs,
        recognition_hidden_dim=32,
        recognition_lr=1e-3,
        use_compression=True,
        use_recognition=True,
        use_dreaming=True,
        verbose=True,
        log_dir=str(phase_dir),
        device='cpu',
    )

    # Load transfer state if provided
    if transfer_state and 'model_state_dict' in transfer_state:
        try:
            learner.recognition.load_state_dict(transfer_state['model_state_dict'])
            logger.info("Loaded recognition model from Phase 1")
        except Exception as e:
            logger.warning(f"Could not load transfer state: {e}")

    # Run
    start_time = time.time()
    results_dict = learner.run()
    elapsed = time.time() - start_time

    # Extract results from the results dictionary
    summary = results_dict.get('summary', {})
    total_solved = summary.get('tasks_solved', 0)
    total_programs = summary.get('total_programs', 0)

    # Get solved task names from frontiers
    solved_names = []
    frontiers = results_dict.get('frontiers', {})
    for task_name, frontier_data in frontiers.items():
        if frontier_data.get('solved', False):
            solved_names.append(task_name)

    result = PhaseResult(
        phase=phase_num,
        tasks_total=len(tasks),
        tasks_solved=total_solved,
        solve_rate=total_solved / len(tasks) if tasks else 0.0,
        abstractions_learned=len(learner.grammar.productions) - len(grammar.productions),
        final_grammar_size=len(learner.grammar.productions),
        total_programs_enumerated=total_programs,
        total_time_seconds=elapsed,
        solved_task_names=list(solved_names),
    )

    # Prepare transfer state
    new_transfer_state = {
        'model_state_dict': learner.recognition.state_dict(),
        'grammar': learner.grammar,
    }

    return result, new_transfer_state


def run_variant(
    variant: AblationVariant,
    config: StudyConfig,
    pretraining_tasks: List[Task],
    catalogue_tasks: List[Task],
) -> VariantResult:
    """Run a single ablation variant (both phases)."""
    logger.info(f"\n{'='*70}")
    logger.info(f"VARIANT: {variant.name}")
    logger.info(f"Description: {variant.description}")
    logger.info(f"Removing {variant.n_removed} primitives")
    logger.info(f"{'='*70}")

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    variant_dir = config.results_dir / f'study_{timestamp}' / variant.name
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Build grammar
    grammar, kept_prims, n_removed = build_ablated_grammar(variant)

    # Initialize result
    result = VariantResult(
        name=variant.name,
        description=variant.description,
        removed_primitives=variant.remove_primitives,
        kept_primitives=kept_prims,
        initial_grammar_size=len(grammar.productions),
        start_time=datetime.now().isoformat(),
    )

    # Save config
    config_dict = asdict(config)
    config_dict['results_dir'] = str(config_dict['results_dir'])  # Convert Path to string
    with open(variant_dir / 'config.json', 'w') as f:
        json.dump({
            'variant': variant.name,
            'description': variant.description,
            'removed_primitives': variant.remove_primitives,
            'n_primitives': len(kept_prims),
            'study_config': config_dict,
        }, f, indent=2)

    # Phase 1: Pretraining
    logger.info(f"\n--- Phase 1: Pretraining ({len(pretraining_tasks)} tasks) ---")
    phase1_result, transfer_state = run_phase(
        tasks=pretraining_tasks,
        grammar=grammar,
        config=config,
        phase_num=1,
        variant_name=variant.name,
        results_dir=variant_dir,
    )
    result.phase1 = phase1_result

    logger.info(f"Phase 1 complete: {phase1_result.tasks_solved}/{phase1_result.tasks_total} solved")

    # Phase 2: Catalogue (with transfer)
    logger.info(f"\n--- Phase 2: Catalogue ({len(catalogue_tasks)} tasks) ---")
    phase2_result, _ = run_phase(
        tasks=catalogue_tasks,
        grammar=grammar,  # Use same grammar (abstractions learned in phase 1 are in transfer)
        config=config,
        phase_num=2,
        variant_name=variant.name,
        results_dir=variant_dir,
        transfer_state=transfer_state,
    )
    result.phase2 = phase2_result

    logger.info(f"Phase 2 complete: {phase2_result.tasks_solved}/{phase2_result.tasks_total} solved")

    result.end_time = datetime.now().isoformat()

    # Save result
    with open(variant_dir / 'variant_result.json', 'w') as f:
        json.dump(asdict(result), f, indent=2, default=str)

    return result


# ============================================================================
# MAIN STUDY RUNNER
# ============================================================================

def run_study(config: StudyConfig) -> Dict[str, VariantResult]:
    """Run the complete ablation study."""
    logger.info("="*70)
    logger.info("COMPREHENSIVE LIBRARY ABLATION STUDY")
    logger.info("="*70)

    # Create results directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    study_dir = config.results_dir / f'study_{timestamp}'
    study_dir.mkdir(parents=True, exist_ok=True)

    # Load tasks
    pretraining_tasks = load_pretraining_tasks()
    catalogue_tasks = load_catalogue_tasks()

    # Get variants to run
    all_variants = build_all_variants()

    if config.variants_to_run:
        variants = [v for v in all_variants if v.name in config.variants_to_run]
        if not variants:
            logger.error(f"No matching variants found. Available: {[v.name for v in all_variants]}")
            return {}
    else:
        variants = all_variants

    logger.info(f"Running {len(variants)} variants")
    for v in variants:
        logger.info(f"  - {v.name}: {v.description}")

    # Save study config
    study_config_dict = asdict(config)
    study_config_dict['results_dir'] = str(study_config_dict['results_dir'])
    with open(study_dir / 'study_config.json', 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'n_variants': len(variants),
            'variants': [v.name for v in variants],
            'config': study_config_dict,
        }, f, indent=2)

    # Run variants
    results = {}
    for i, variant in enumerate(variants, 1):
        logger.info(f"\n{'#'*70}")
        logger.info(f"# VARIANT {i}/{len(variants)}: {variant.name}")
        logger.info(f"{'#'*70}")

        try:
            result = run_variant(
                variant=variant,
                config=config,
                pretraining_tasks=pretraining_tasks,
                catalogue_tasks=catalogue_tasks,
            )
            results[variant.name] = result
        except Exception as e:
            logger.error(f"Variant {variant.name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Save summary
    summary = {
        'timestamp': timestamp,
        'n_variants_completed': len(results),
        'results': {
            name: {
                'total_solved': r.total_solved(),
                'solve_rate': r.solve_rate(),
                'n_removed': len(r.removed_primitives),
                'phase1_solved': r.phase1.tasks_solved if r.phase1 else 0,
                'phase2_solved': r.phase2.tasks_solved if r.phase2 else 0,
            }
            for name, r in results.items()
        }
    }

    with open(study_dir / 'study_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Print summary
    logger.info("\n" + "="*70)
    logger.info("STUDY COMPLETE - SUMMARY")
    logger.info("="*70)

    # Sort by solve rate
    sorted_results = sorted(results.items(), key=lambda x: x[1].solve_rate(), reverse=True)

    logger.info(f"\n{'Variant':<30} {'Removed':<8} {'Solved':<10} {'Rate':<10}")
    logger.info("-"*60)
    for name, r in sorted_results:
        logger.info(f"{name:<30} {len(r.removed_primitives):<8} "
                   f"{r.total_solved()}/{r.total_tasks():<10} "
                   f"{r.solve_rate():.1%}")

    logger.info(f"\nResults saved to: {study_dir}")

    return results


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Comprehensive Library Ablation Study',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--quick-test', action='store_true',
                           help='Quick test (~30 min, 2 variants, minimal params)')
    mode_group.add_argument('--overnight', action='store_true',
                           help='Overnight run (~12 hours, 6 high-priority variants)')
    mode_group.add_argument('--full', action='store_true',
                           help='Full study (~48-72 hours, all 23 variants)')

    parser.add_argument('--variants', nargs='+', help='Override: specific variants to run')
    parser.add_argument('--list-variants', action='store_true', help='List all variants and exit')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be run')

    # Manual parameter overrides
    parser.add_argument('--budget', type=int, help='Override enumeration budget')
    parser.add_argument('--timeout', type=int, help='Override timeout per task (seconds)')
    parser.add_argument('--iterations', type=int, help='Override iterations per phase')

    args = parser.parse_args()

    # List variants
    if args.list_variants:
        variants = build_all_variants()
        print(f"\n{'Name':<35} {'Priority':<10} {'Removed':<10} Description")
        print("-"*100)
        for v in variants:
            print(f"{v.name:<35} {v.priority:<10} {v.n_removed:<10} {v.description}")
        print(f"\nTotal: {len(variants)} variants")
        print(f"\nOvernight variants ({len(OVERNIGHT_VARIANTS)}):")
        for name in OVERNIGHT_VARIANTS:
            print(f"  - {name}")
        return

    # Build config based on mode
    config = StudyConfig(
        quick_test=args.quick_test,
        overnight=args.overnight,
    )

    # Determine variants to run
    if args.variants:
        # Manual override
        config.variants_to_run = args.variants
    elif args.overnight:
        # Use overnight preset
        config.variants_to_run = OVERNIGHT_VARIANTS
    # else: run all variants (empty list)

    # Apply manual parameter overrides
    if args.budget:
        config.enumeration_budget = args.budget
    if args.timeout:
        config.timeout_seconds = args.timeout
    if args.iterations:
        config.iterations_phase1 = args.iterations
        config.iterations_phase2 = args.iterations

    # Dry run
    if args.dry_run:
        variants = build_all_variants()
        if config.variants_to_run:
            variants = [v for v in variants if v.name in config.variants_to_run]

        mode = "OVERNIGHT" if args.overnight else "QUICK TEST" if args.quick_test else "FULL"
        print(f"\nDRY RUN ({mode}) - Would run {len(variants)} variants:")
        print(f"  Budget: {config.enumeration_budget:,} programs")
        print(f"  Timeout: {config.timeout_seconds}s per task")
        print(f"  Iterations: {config.iterations_phase1} (phase1), {config.iterations_phase2} (phase2)")
        print(f"  Recognition epochs: {config.recognition_epochs}")
        print(f"\nVariants:")
        for v in variants:
            print(f"  - {v.name}: remove {v.n_removed} primitives")

        # Estimate runtime
        est_minutes_per_variant = (config.timeout_seconds / 60) * 88 * config.iterations_phase1 / 10
        est_hours = (est_minutes_per_variant * len(variants)) / 60
        print(f"\nEstimated runtime: ~{est_hours:.1f} hours")
        return

    # Run study
    run_study(config)


if __name__ == '__main__':
    main()
