#!/usr/bin/env python3
"""
Targeted Primitive Ablation Study with Two-Phase Transfer Learning

Research Questions:
1. Gestalt redundancy: Confirm no_gestalt ≈ baseline (only removing 2 primitives)
2. Critical primitives: Which primitives are truly essential vs convenience?
3. Transfer benefit: How much does Phase 1 pretraining help Phase 2?

Variants:
- baseline: Full 59-primitive library
- no_gestalt: Remove all_same_suit, all_same_color (2 primitives)
- no_halves: Remove first_half, second_half (2 primitives)
- no_n_unique: Remove n_unique_suits/ranks/colors (3 primitives)
- no_gestalt_n_unique: Remove gestalt + n_unique (5 primitives)

Design:
- Phase 1: 3 iterations on pretraining_rules (44 tasks)
- Phase 2: 3 iterations on catalogue (44 tasks) with transfer
- Budget: 500k per iteration (enumerates ~10-20M programs)

Usage:
    # Run all variants
    python3 run_targeted_ablation_study.py

    # Run specific variants
    python3 run_targeted_ablation_study.py --variants baseline no_gestalt

    # Quick test
    python3 run_targeted_ablation_study.py --quick-test

    # Dry run
    python3 run_targeted_ablation_study.py --dry-run

Launch overnight:
    PYTHONUNBUFFERED=1 nohup caffeinate -d -i -s python3 run_targeted_ablation_study.py > targeted_ablation.out 2>&1 &
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
from typing import List, Dict, Optional, Tuple, Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.grammar import Grammar, uniform_grammar
from dreamcoder_core.lean_primitives import build_lean_primitives
from dreamcoder_core.task import Task
from dreamcoder_core.task_generation import load_prerecorded_tasks

# Import LoggingWakeSleep from the overnight study script
from experiments.run_overnight_wakesleep_study import LoggingWakeSleep

# Paths to prerecorded tasks
SRC_DIR = Path(__file__).parent.parent
PRETRAINING_TASKS_PATH = SRC_DIR / 'data' / 'prerecorded_tasks' / 'pretraining_tasks.json'
CATALOGUE_TASKS_PATH = SRC_DIR / 'data' / 'prerecorded_tasks' / 'catalogue_tasks.json'


# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# ABLATION VARIANT DEFINITIONS
# ============================================================================

# Individual primitive groups to ablate
ABLATION_GROUPS = {
    'gestalt': ['all_same_suit', 'all_same_color'],
    'n_unique': ['n_unique_suits', 'n_unique_ranks', 'n_unique_colors'],
    'halves': ['first_half', 'second_half'],
    'has': ['has_suit', 'has_color'],
    'count': ['count_suit', 'count_color'],
}

# Variant definitions: name -> list of primitives to remove
VARIANTS = {
    'baseline': [],
    'no_gestalt': ABLATION_GROUPS['gestalt'],
    'no_halves': ABLATION_GROUPS['halves'],
    'no_n_unique': ABLATION_GROUPS['n_unique'],
    'no_gestalt_n_unique': ABLATION_GROUPS['gestalt'] + ABLATION_GROUPS['n_unique'],
}


@dataclass
class AblationConfig:
    """Configuration for an ablation variant."""
    name: str
    remove_primitives: List[str]
    description: str

    # Experiment parameters
    iterations_phase1: int = 3
    iterations_phase2: int = 3
    enumeration_budget: int = 500_000
    recognition_epochs: int = 15

    def n_primitives(self) -> int:
        """Calculate resulting primitive count."""
        return 59 - len(self.remove_primitives)


def get_ablation_configs() -> List[AblationConfig]:
    """Get all ablation variant configurations."""
    return [
        AblationConfig(
            name='baseline',
            remove_primitives=[],
            description='Full 59-primitive library (control)',
        ),
        AblationConfig(
            name='no_gestalt',
            remove_primitives=ABLATION_GROUPS['gestalt'],
            description='Remove gestalt primitives (all_same_suit, all_same_color)',
        ),
        AblationConfig(
            name='no_halves',
            remove_primitives=ABLATION_GROUPS['halves'],
            description='Remove halves primitives (first_half, second_half)',
        ),
        AblationConfig(
            name='no_n_unique',
            remove_primitives=ABLATION_GROUPS['n_unique'],
            description='Remove n_unique primitives (n_unique_suits/ranks/colors)',
        ),
        AblationConfig(
            name='no_gestalt_n_unique',
            remove_primitives=ABLATION_GROUPS['gestalt'] + ABLATION_GROUPS['n_unique'],
            description='Remove gestalt + n_unique (hardest - no uniformity checking)',
        ),
    ]


# ============================================================================
# GRAMMAR BUILDING
# ============================================================================

def build_ablated_grammar(config: AblationConfig) -> Tuple[Grammar, List[str]]:
    """
    Build a grammar with specified primitives removed.

    Returns:
        (grammar, list of removed primitive names)
    """
    primitives_to_remove = set(config.remove_primitives)

    # Build full primitive list
    all_primitives = build_lean_primitives()

    # Filter out removed primitives
    kept_primitives = [p for p in all_primitives if p.name not in primitives_to_remove]
    removed_names = [p.name for p in all_primitives if p.name in primitives_to_remove]

    # Build grammar
    grammar = uniform_grammar(kept_primitives)

    logger.info(f"Variant '{config.name}': {len(kept_primitives)} primitives "
                f"(removed {len(removed_names)}: {removed_names})")

    return grammar, removed_names


# ============================================================================
# TASK LOADING
# ============================================================================

def load_pretraining_tasks() -> List[Task]:
    """Load pretraining tasks."""
    rules = get_pretraining_rules()
    tasks = []
    for rule in rules:
        task = generate_balanced_task(rule, n_positive=8, n_negative=8)
        if task:
            tasks.append(task)
    logger.info(f"Loaded {len(tasks)} pretraining tasks")
    return tasks


def load_catalogue_tasks() -> List[Task]:
    """Load catalogue tasks."""
    rules = get_catalogue_rules()
    tasks = []
    for rule in rules:
        task = generate_balanced_task(rule, n_positive=8, n_negative=8)
        if task:
            tasks.append(task)
    logger.info(f"Loaded {len(tasks)} catalogue tasks")
    return tasks


# ============================================================================
# TRANSFER STATE MANAGEMENT
# ============================================================================

def save_transfer_state(learner: LoggingWakeSleep, path: Path) -> None:
    """Save model weights and grammar for transfer to Phase 2."""
    # Serialize grammar productions (handle both Primitive and Invented types)
    grammar_prods = []
    for p in learner.grammar.productions:
        prog_str = str(p.program)
        log_prob = p.log_probability
        # Get type - Primitive has .tp, Invented needs .infer()
        if hasattr(p.program, 'tp'):
            type_str = str(p.program.tp)
        elif hasattr(p.program, 'infer'):
            type_str = str(p.program.infer())
        else:
            type_str = "?"
        grammar_prods.append((prog_str, log_prob, type_str))

    state = {
        'model_state_dict': learner.recognition.state_dict(),
        'grammar_productions': grammar_prods,
        'all_abstractions': learner.all_abstractions,
        'cumulative_solved': list(learner.cumulative_solved),
    }
    torch.save(state, path)
    print(f"Saved transfer state to {path}")


def load_transfer_state(path: Path) -> Dict:
    """Load transfer state from file."""
    state = torch.load(path, weights_only=False)
    print(f"Loaded transfer state from {path}")
    print(f"  Abstractions learned: {len(state.get('all_abstractions', []))}")
    print(f"  Tasks previously solved: {len(state.get('cumulative_solved', []))}")
    return state


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

@dataclass
class PhaseResult:
    """Result from one phase of the experiment."""
    phase: int
    iterations: int
    tasks_total: int
    tasks_solved: int
    solve_rate: float
    abstractions_learned: int
    final_grammar_size: int
    total_programs_enumerated: int
    total_time_seconds: float


@dataclass
class VariantResult:
    """Complete result for one variant."""
    name: str
    removed_primitives: List[str]
    initial_primitives: int
    phase1: Optional[PhaseResult] = None
    phase2: Optional[PhaseResult] = None

    def combined_solved(self) -> int:
        p1 = self.phase1.tasks_solved if self.phase1 else 0
        p2 = self.phase2.tasks_solved if self.phase2 else 0
        return p1 + p2

    def combined_rate(self) -> float:
        p1_total = self.phase1.tasks_total if self.phase1 else 0
        p2_total = self.phase2.tasks_total if self.phase2 else 0
        total = p1_total + p2_total
        if total == 0:
            return 0.0
        return self.combined_solved() / total


def run_phase(
    learner: LoggingWakeSleep,
    tasks: List[Task],
    n_iterations: int,
    phase_name: str,
    output_dir: Path,
) -> PhaseResult:
    """Run one phase of the experiment."""
    import time

    start_time = time.time()
    total_programs = 0

    for iteration in range(n_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{n_iterations}")
        print(f"{'='*60}")

        # Run wake-sleep iteration
        result = learner.iteration(tasks)

        # Track programs enumerated
        if hasattr(result, 'programs_enumerated'):
            total_programs += result.programs_enumerated

        # Save iteration log
        iter_log = {
            'iteration': iteration + 1,
            'solved': len(learner.cumulative_solved),
            'total_tasks': len(tasks),
            'grammar_size': len(learner.grammar.primitives()),
            'abstractions': len(learner.all_abstractions),
        }

        log_path = output_dir / f"iter_{iteration + 1:02d}_log.json"
        with open(log_path, 'w') as f:
            json.dump(iter_log, f, indent=2)

    elapsed = time.time() - start_time

    return PhaseResult(
        phase=1 if 'phase1' in phase_name.lower() else 2,
        iterations=n_iterations,
        tasks_total=len(tasks),
        tasks_solved=len(learner.cumulative_solved),
        solve_rate=len(learner.cumulative_solved) / len(tasks),
        abstractions_learned=len(learner.all_abstractions),
        final_grammar_size=len(learner.grammar.primitives()),
        total_programs_enumerated=total_programs,
        total_time_seconds=elapsed,
    )


def run_variant(
    config: AblationConfig,
    pretraining_tasks: List[Task],
    catalogue_tasks: List[Task],
    output_dir: Path,
    device: str = 'cpu',
) -> VariantResult:
    """Run complete two-phase experiment for one variant."""
    import time

    print(f"\n{'#'*70}")
    print(f"# VARIANT: {config.name}")
    print(f"{'#'*70}")

    # Build grammar for this variant
    grammar, removed = build_ablated_grammar(config)
    print(f"Grammar: {len(grammar.primitives())} primitives")
    if removed:
        print(f"Removed: {removed}")

    result = VariantResult(
        name=config.name,
        removed_primitives=removed,
        initial_primitives=len(grammar.primitives()),
    )

    variant_dir = output_dir / config.name
    variant_dir.mkdir(parents=True, exist_ok=True)

    # ========== PHASE 1: PRETRAINING ==========
    print(f"\n{'='*60}")
    print(f"PHASE 1: PRETRAINING ({config.iterations_phase1} iterations)")
    print(f"{'='*60}")

    phase1_dir = variant_dir / "phase1_pretraining"
    phase1_dir.mkdir(parents=True, exist_ok=True)

    # Create learner for Phase 1
    learner = LoggingWakeSleep(
        grammar=grammar,
        budget=config.enumeration_budget,
        recognition_epochs=config.recognition_epochs,
        output_dir=phase1_dir,
        device=device,
    )

    # Run Phase 1
    result.phase1 = run_phase(
        learner=learner,
        tasks=pretraining_tasks,
        n_iterations=config.iterations_phase1,
        phase_name="phase1_pretraining",
        output_dir=phase1_dir,
    )

    print(f"\nPhase 1 complete: {result.phase1.tasks_solved}/{result.phase1.tasks_total} solved")
    print(f"  Abstractions: {result.phase1.abstractions_learned}")
    print(f"  Time: {result.phase1.total_time_seconds:.1f}s")

    # Save transfer state
    transfer_path = variant_dir / "transfer_state.pt"
    save_transfer_state(learner, transfer_path)

    # ========== PHASE 2: CATALOGUE WITH TRANSFER ==========
    print(f"\n{'='*60}")
    print(f"PHASE 2: CATALOGUE TASKS ({config.iterations_phase2} iterations with transfer)")
    print(f"{'='*60}")
    print(f"Transferring: {result.phase1.abstractions_learned} abstractions")
    print(f"             Model weights from {result.phase1.tasks_solved} solved tasks")

    phase2_dir = variant_dir / "phase2_catalogue"
    phase2_dir.mkdir(parents=True, exist_ok=True)

    # Load transfer state
    transfer_state = load_transfer_state(transfer_path)

    # Create fresh grammar for Phase 2 (same primitives, no inventions)
    grammar2, _ = build_ablated_grammar(config)

    # Create new learner for Phase 2
    learner2 = LoggingWakeSleep(
        grammar=grammar2,
        budget=config.enumeration_budget,
        recognition_epochs=config.recognition_epochs,
        output_dir=phase2_dir,
        device=device,
    )

    # Transfer model weights (with dimension mismatch handling)
    try:
        learner2.recognition.load_state_dict(transfer_state['model_state_dict'])
        print("  Loaded model weights successfully")
    except RuntimeError as e:
        print(f"  Warning: Model weight mismatch (likely due to new primitives): {e}")
        print("  Will use partially loaded weights where dimensions match")
        # Load compatible weights only
        model_dict = learner2.recognition.state_dict()
        pretrained_dict = transfer_state['model_state_dict']
        compatible_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        model_dict.update(compatible_dict)
        learner2.recognition.load_state_dict(model_dict)
        print(f"  Loaded {len(compatible_dict)}/{len(pretrained_dict)} weight tensors")

    # Run Phase 2
    result.phase2 = run_phase(
        learner=learner2,
        tasks=catalogue_tasks,
        n_iterations=config.iterations_phase2,
        phase_name="phase2_catalogue",
        output_dir=phase2_dir,
    )

    print(f"\nPhase 2 complete: {result.phase2.tasks_solved}/{result.phase2.tasks_total} solved")
    print(f"  Abstractions: {result.phase2.abstractions_learned}")
    print(f"  Time: {result.phase2.total_time_seconds:.1f}s")

    # Save variant summary
    summary = {
        'variant': config.name,
        'removed_primitives': removed,
        'config': asdict(config),
        'phase1': asdict(result.phase1) if result.phase1 else None,
        'phase2': asdict(result.phase2) if result.phase2 else None,
        'combined': {
            'total_tasks': (result.phase1.tasks_total if result.phase1 else 0) +
                          (result.phase2.tasks_total if result.phase2 else 0),
            'total_solved': result.combined_solved(),
            'combined_rate': result.combined_rate(),
            'total_abstractions': (result.phase1.abstractions_learned if result.phase1 else 0) +
                                 (result.phase2.abstractions_learned if result.phase2 else 0),
        }
    }

    with open(variant_dir / "variant_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Targeted Primitive Ablation Study with Two-Phase Transfer'
    )
    parser.add_argument('--variants', type=str, nargs='+',
                        default=list(VARIANTS.keys()),
                        help='Which variants to run')
    parser.add_argument('--output-dir', type=str,
                        default='results_targeted_ablation',
                        help='Output directory')
    parser.add_argument('--budget', type=int, default=500_000,
                        help='Enumeration budget per iteration')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test mode (1 iteration, 100k budget)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show config without running')

    args = parser.parse_args()

    # Get all variant configs
    all_configs = {c.name: c for c in get_ablation_configs()}

    # Filter to requested variants
    configs = []
    for name in args.variants:
        if name in all_configs:
            configs.append(all_configs[name])
        else:
            print(f"Warning: Unknown variant '{name}'")
            print(f"Available: {list(all_configs.keys())}")

    if not configs:
        print("No valid variants specified!")
        return 1

    # Apply budget override
    for config in configs:
        config.enumeration_budget = args.budget

    # Quick test mode
    if args.quick_test:
        print("Quick test mode: 1 iteration per phase, 100k budget")
        for config in configs:
            config.iterations_phase1 = 1
            config.iterations_phase2 = 1
            config.enumeration_budget = 100_000

    # Dry run - show what would be run
    if args.dry_run:
        print("\n" + "="*70)
        print("DRY RUN - Configuration Summary")
        print("="*70)

        for config in configs:
            print(f"\n{config.name}:")
            print(f"  Description: {config.description}")
            print(f"  Primitives: {config.n_primitives()} (removes {len(config.remove_primitives)})")
            if config.remove_primitives:
                print(f"  Removes: {config.remove_primitives}")
            print(f"  Budget: {config.enumeration_budget:,}")
            print(f"  Iterations: {config.iterations_phase1} (P1) + {config.iterations_phase2} (P2)")

        return 0

    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f"study_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Results will be saved to: {output_dir}")

    # Load tasks
    pretraining_tasks = load_pretraining_tasks()
    catalogue_tasks = load_catalogue_tasks()

    # Run experiments
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n{'#'*70}")
        print(f"# VARIANT {i}/{len(configs)}: {config.name}")
        print(f"{'#'*70}")

        result = run_variant(
            config=config,
            pretraining_tasks=pretraining_tasks,
            catalogue_tasks=catalogue_tasks,
            output_dir=output_dir,
        )
        results.append(result)

    # Print final summary
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Results: {output_dir}")

    print("\n--- Summary ---")
    print(f"{'Variant':<25} {'Removed':<10} {'P1 Solved':<12} {'P2 Solved':<12} {'Combined':<12}")
    print("-" * 75)

    for result in results:
        p1_str = f"{result.phase1.tasks_solved}/{result.phase1.tasks_total}" if result.phase1 else "N/A"
        p2_str = f"{result.phase2.tasks_solved}/{result.phase2.tasks_total}" if result.phase2 else "N/A"
        combined_str = f"{result.combined_solved()} ({result.combined_rate()*100:.1f}%)"
        print(f"{result.name:<25} {len(result.removed_primitives):<10} {p1_str:<12} {p2_str:<12} {combined_str:<12}")

    # Save overall summary
    summary = {
        'timestamp': timestamp,
        'output_dir': str(output_dir),
        'variants': [
            {
                'name': r.name,
                'removed': r.removed_primitives,
                'phase1_solved': r.phase1.tasks_solved if r.phase1 else 0,
                'phase1_total': r.phase1.tasks_total if r.phase1 else 0,
                'phase2_solved': r.phase2.tasks_solved if r.phase2 else 0,
                'phase2_total': r.phase2.tasks_total if r.phase2 else 0,
                'combined_solved': r.combined_solved(),
                'combined_rate': r.combined_rate(),
            }
            for r in results
        ]
    }

    with open(output_dir / "study_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    return 0


if __name__ == '__main__':
    import time
    start = time.time()
    exit_code = main()
    elapsed = time.time() - start
    print(f"\nTotal time: {elapsed/3600:.1f} hours ({elapsed:.0f}s)")
    sys.exit(exit_code)
