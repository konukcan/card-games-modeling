#!/usr/bin/env python3
"""
Recognition × Dream Strategy Experiment Runner
================================================

This script implements the 2×3 factorial experiment design:
- 2 Recognition Models: GRU (legacy) vs Contrastive (new)
- 3 Dream Strategies: Standard vs Balanced vs Contrastive

See recognition_dream_experiment_plan.md for full experiment details.

Usage:
    # Run all conditions (full experiment)
    python3 run_recognition_dream_experiment.py

    # Run specific condition
    python3 run_recognition_dream_experiment.py --recognition gru --dreams balanced

    # Quick test (2 iterations, 3 rules)
    python3 run_recognition_dream_experiment.py --quick-test

    # Dry run (just show what would be run)
    python3 run_recognition_dream_experiment.py --dry-run

Launch with caffeinate for overnight runs:
    nohup caffeinate -d -i -s python3 run_recognition_dream_experiment.py > experiment.log 2>&1 &
"""

import sys
import os
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.neural_recognition import NeuralRecognitionModel
from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.contrastive_dreaming import ConfigurableDreamer
from dreamcoder_core.type_system import arrow, HAND, BOOL
from dreamcoder_core.enumeration import TopDownEnumerator, EnumerationResult, Frontier
from dreamcoder_core.compression import compress_frontiers
from dreamcoder_core.grammar import Grammar
from dreamcoder_core.task import Task
from dreamcoder_core.task_generation import (
    create_tasks_from_rules as unified_create_tasks,
    TaskGenerationConfig,
    load_prerecorded_tasks
)

from rules.cards import sample_hand
from rules.pretraining_rules import get_all_pretraining_rules, get_easy_pretraining_rules


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for the experiment."""

    # Experiment conditions
    recognition_type: str = 'gru'  # 'gru' or 'contrastive'
    dream_strategy: str = 'balanced'  # 'standard', 'balanced', or 'contrastive'
    run_id: int = 1

    # Wake phase
    enumeration_budget: int = 200_000
    enumeration_timeout: float = 120.0
    max_depth: int = 8

    # Compression
    use_compression: bool = True
    max_inventions_per_iteration: int = 5
    min_compression_savings: float = 2.0

    # Dreaming
    dreams_per_iteration: int = 100
    dream_temperature: float = 1.0
    n_examples_per_dream: int = 10

    # Recognition training
    recognition_hidden_dim: int = 128
    recognition_lr: float = 1e-3
    recognition_epochs: int = 10
    structural_similarity_weight: float = 0.1  # For contrastive model only

    # General
    keep_top_k: int = 5
    max_iterations: int = 10
    n_examples_per_task: int = 100
    n_holdout_per_task: int = 20

    # Output
    results_dir: str = 'results'

    def condition_name(self) -> str:
        """Return the condition nickname."""
        return f"{self.recognition_type}_{self.dream_strategy}"

    def run_dir(self) -> Path:
        """Return the directory for this run."""
        return Path(self.results_dir) / self.condition_name() / f"run_{self.run_id}"


CONDITIONS = [
    {'recognition': 'gru', 'dreams': 'standard'},
    {'recognition': 'gru', 'dreams': 'balanced'},
    {'recognition': 'gru', 'dreams': 'contrastive'},
    {'recognition': 'contrastive', 'dreams': 'standard'},
    {'recognition': 'contrastive', 'dreams': 'balanced'},
    {'recognition': 'contrastive', 'dreams': 'contrastive'},
]

N_RUNS_PER_CONDITION = 3


# ============================================================================
# TASK CREATION
# ============================================================================

def create_tasks_from_rules(
    rules: List,
    n_examples: int = 100,
    n_holdout: int = 20,
    hand_size: int = 6  # Standardized to 6
) -> List[Task]:
    """
    DEPRECATED: Use task_generation.create_tasks_from_rules() instead.

    This wrapper delegates to the unified task generation system which provides:
    - Guaranteed balanced examples (equal positives/negatives)
    - Near-miss negative generation (flip one card from positive)
    - Disjoint seed/training/holdout pools to prevent data leakage
    - Explicit failure if balance cannot be achieved

    Args:
        rules: List of PretrainingRule objects or callables
        n_examples: Number of training examples
        n_holdout: Number of held-out examples
        hand_size: Number of cards per hand (default 6)

    Returns:
        List of Task objects with balanced examples
    """
    import warnings
    warnings.warn(
        "run_recognition_dream_experiment.create_tasks_from_rules() is deprecated. "
        "Use task_generation.create_tasks_from_rules() directly.",
        DeprecationWarning,
        stacklevel=2
    )

    # Map old parameters to new config
    config = TaskGenerationConfig(
        n_training_positives=n_examples // 2,
        n_seed_positives=n_examples // 4,  # For near-miss generation
        n_training_negatives=n_examples // 2,
        n_holdout_positives=n_holdout // 2,
        n_holdout_negatives=n_holdout // 2,
        hand_size=hand_size,
        use_near_miss_negatives=True,
        allow_random_negative_fallback=True,
        require_exact_balance=False,  # Be lenient for backwards compat
    )

    # Delegate to unified implementation
    return unified_create_tasks(rules, config=config)


# ============================================================================
# EVALUATION
# ============================================================================

def eval_program_on_hand(program, hand):
    """Evaluate a program on a hand of cards."""
    try:
        fn = program.evaluate([])
        return fn(hand)
    except Exception:
        return None


def verify_on_holdout(program, holdout: List[Tuple[Any, bool]]) -> bool:
    """Verify a program on held-out examples."""
    for hand, expected in holdout:
        result = eval_program_on_hand(program, hand)
        if result != expected:
            return False
    return True


# ============================================================================
# ITERATION RESULT
# ============================================================================

@dataclass
class IterationResult:
    """Results from a single wake-sleep iteration."""
    iteration: int

    # Wake results
    tasks_solved: int
    total_tasks: int
    programs_enumerated: int
    enumeration_time: float

    # Compression results
    new_inventions: List[str] = field(default_factory=list)
    grammar_size: int = 0

    # Recognition results
    recognition_loss: float = 0.0

    # Dreaming results
    dreams_generated: int = 0

    # Timing
    iteration_time: float = 0.0

    # Solutions found
    solutions: Dict[str, str] = field(default_factory=dict)


# ============================================================================
# WAKE-SLEEP LOOP
# ============================================================================

def run_wake_sleep_iteration(
    config: ExperimentConfig,
    grammar: Grammar,
    tasks: List[Task],
    recognition_model,
    dreamer: ConfigurableDreamer,
    iteration: int,
    logger: logging.Logger
) -> Tuple[IterationResult, Grammar, Dict]:
    """Run a single wake-sleep iteration."""

    start_time = time.time()
    result = IterationResult(iteration=iteration, tasks_solved=0, total_tasks=len(tasks),
                             programs_enumerated=0, enumeration_time=0.0)

    frontiers = {}

    # =========================================================================
    # WAKE PHASE: Enumerate programs to solve tasks
    # =========================================================================
    logger.info(f"  WAKE: Enumerating programs for {len(tasks)} tasks...")

    enum_start = time.time()
    total_programs = 0

    for task in tasks:
        # Get task-specific grammar weights from recognition model
        if recognition_model is not None and iteration > 0:
            try:
                task_grammar = recognition_model.predict_grammar_weights(task)
            except Exception as e:
                logger.warning(f"    Recognition prediction failed for {task.name}: {e}")
                task_grammar = grammar
        else:
            task_grammar = grammar

        # Enumerate
        enumerator = TopDownEnumerator(
            task_grammar,
            max_depth=config.max_depth,
            max_programs=config.enumeration_budget
        )

        frontier = Frontier(
            task_name=task.name,
            request_type=task.request_type,
            max_size=config.keep_top_k
        )

        for program, log_prob in enumerator.enumerate(
            task.request_type,
            max_cost=30.0,
            timeout_seconds=config.enumeration_timeout
        ):
            total_programs += 1

            # Check if program solves the task
            correct = 0
            for hand, expected in task.examples:
                result_val = eval_program_on_hand(program, hand)
                if result_val == expected:
                    correct += 1

            if correct == len(task.examples):
                # Verify on holdout
                if verify_on_holdout(program, task.holdout):
                    enum_result = EnumerationResult(
                        program=program,
                        log_probability=log_prob,
                        log_likelihood=0.0,
                        description_length=-log_prob / 0.693,  # Convert to bits
                        programs_enumerated=enumerator.programs_enumerated,
                        partial_programs_explored=enumerator.partial_programs_explored,
                        time_seconds=time.time() - enum_start
                    )
                    frontier.add(enum_result)
                    break  # Found a valid solution

        frontiers[task.name] = frontier

        if not frontier.empty:
            result.tasks_solved += 1
            result.solutions[task.name] = str(frontier.best.program)

    result.programs_enumerated = total_programs
    result.enumeration_time = time.time() - enum_start

    logger.info(f"    Solved {result.tasks_solved}/{result.total_tasks} tasks")
    logger.info(f"    Enumerated {total_programs:,} programs in {result.enumeration_time:.1f}s")

    # =========================================================================
    # SLEEP-COMPRESS: Find reusable abstractions
    # =========================================================================
    if config.use_compression and result.tasks_solved > 0:
        logger.info("  SLEEP-COMPRESS: Finding abstractions...")

        # Convert frontiers to format expected by compression
        frontier_list = []
        for task_name, frontier in frontiers.items():
            if not frontier.empty:
                frontier_list.append([
                    (entry.program, entry.log_likelihood)
                    for entry in frontier.entries
                ])

        if frontier_list:
            try:
                compression_result = compress_frontiers(
                    grammar,
                    frontier_list,
                    max_inventions=config.max_inventions_per_iteration,
                    min_savings=config.min_compression_savings
                )

                grammar = compression_result.new_grammar
                result.new_inventions = [str(inv) for inv in compression_result.new_inventions]
                result.grammar_size = len(grammar)

                # Update recognition model vocabulary
                if recognition_model is not None:
                    for inv in compression_result.new_inventions:
                        recognition_model.add_invention(inv)

                logger.info(f"    Found {len(result.new_inventions)} new abstractions")

            except Exception as e:
                logger.warning(f"    Compression failed: {e}")

    result.grammar_size = len(grammar)

    # =========================================================================
    # SLEEP-RECOGNIZE: Train recognition model
    # =========================================================================
    if recognition_model is not None and result.tasks_solved > 0:
        logger.info("  SLEEP-RECOGNIZE: Training recognition model...")

        # Get solved tasks
        solved_tasks = [t for t in tasks if frontiers.get(t.name) and not frontiers[t.name].empty]

        try:
            if hasattr(recognition_model, 'train_on_frontiers'):
                loss = recognition_model.train_on_frontiers(
                    tasks=solved_tasks,
                    frontiers=frontiers,
                    epochs=config.recognition_epochs
                )
                result.recognition_loss = float(loss) if loss is not None else 0.0

            logger.info(f"    Training loss: {result.recognition_loss:.4f}")

        except Exception as e:
            logger.warning(f"    Recognition training failed: {e}")

    # =========================================================================
    # SLEEP-DREAM: Generate dreamed examples (for future iterations)
    # =========================================================================
    if dreamer is not None and config.dreams_per_iteration > 0:
        logger.info("  SLEEP-DREAM: Generating dreams...")

        try:
            # Get programs to dream from
            programs = []
            for frontier in frontiers.values():
                if not frontier.empty:
                    programs.append(frontier.best.program)

            if programs:
                dreams = dreamer.generate_dreams(
                    programs=programs[:20],  # Limit to avoid too many dreams
                    n_dreams=config.dreams_per_iteration,
                    n_examples_per_dream=config.n_examples_per_dream
                )
                result.dreams_generated = len(dreams)

                logger.info(f"    Generated {result.dreams_generated} dreams")

        except Exception as e:
            logger.warning(f"    Dreaming failed: {e}")

    result.iteration_time = time.time() - start_time

    return result, grammar, frontiers


def run_condition(config: ExperimentConfig, logger: logging.Logger) -> Dict:
    """Run a complete experiment for one condition."""

    logger.info(f"\n{'='*70}")
    logger.info(f"RUNNING: {config.condition_name()} (run {config.run_id})")
    logger.info(f"{'='*70}")

    # Create output directory
    run_dir = config.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)

    # Build grammar
    logger.info("Building grammar...")
    grammar = build_lean_grammar()
    logger.info(f"  Grammar size: {len(grammar)} primitives")

    # Create tasks
    logger.info("Creating tasks...")
    rules = get_all_pretraining_rules()
    tasks = create_tasks_from_rules(
        rules,
        n_examples=config.n_examples_per_task,
        n_holdout=config.n_holdout_per_task
    )
    logger.info(f"  Created {len(tasks)} tasks")

    # Initialize recognition model
    logger.info(f"Initializing {config.recognition_type} recognition model...")
    if config.recognition_type == 'gru':
        recognition_model = NeuralRecognitionModel(
            grammar,
            hidden_dim=config.recognition_hidden_dim,
            learning_rate=config.recognition_lr
        )
    else:
        recognition_model = ContrastiveRecognitionModel(
            grammar,
            card_out=config.recognition_hidden_dim // 4,
            pred_hidden=config.recognition_hidden_dim // 2,
            learning_rate=config.recognition_lr
        )
    logger.info(f"  Model parameters: {sum(p.numel() for p in recognition_model.parameters()):,}")

    # Initialize dreamer
    logger.info(f"Initializing {config.dream_strategy} dreamer...")

    def sample_card_fn():
        return sample_hand(1)[0]

    dreamer = ConfigurableDreamer(
        grammar=grammar,
        eval_fn=eval_program_on_hand,
        sample_hand_fn=sample_hand,
        sample_card_fn=sample_card_fn,
        strategy=config.dream_strategy
    )

    # Run iterations
    results = {
        'config': asdict(config),
        'start_time': datetime.now().isoformat(),
        'iterations': []
    }

    for iteration in range(1, config.max_iterations + 1):
        logger.info(f"\n--- Iteration {iteration}/{config.max_iterations} ---")

        iter_result, grammar, frontiers = run_wake_sleep_iteration(
            config=config,
            grammar=grammar,
            tasks=tasks,
            recognition_model=recognition_model,
            dreamer=dreamer,
            iteration=iteration,
            logger=logger
        )

        # Update dreamer's grammar
        dreamer.grammar = grammar

        results['iterations'].append(asdict(iter_result))

        # Save checkpoint
        checkpoint_path = run_dir / f"checkpoint_iter_{iteration}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save model
        model_path = run_dir / f"model_iter_{iteration}.pt"
        recognition_model.save(str(model_path))

        logger.info(f"  Saved checkpoint to {checkpoint_path}")

    # Final summary
    results['end_time'] = datetime.now().isoformat()

    final_path = run_dir / "final_results.json"
    with open(final_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    logger.info(f"\nCompleted {config.condition_name()} run {config.run_id}")
    logger.info(f"Results saved to {final_path}")

    return results


# ============================================================================
# MAIN
# ============================================================================

def setup_logger(log_file: Optional[Path] = None) -> logging.Logger:
    """Set up logging."""
    logger = logging.getLogger('experiment')
    logger.setLevel(logging.INFO)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(console)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(file_handler)

    return logger


def main():
    parser = argparse.ArgumentParser(description='Run Recognition × Dream Strategy Experiment')

    parser.add_argument('--recognition', choices=['gru', 'contrastive'],
                       help='Recognition model type (run single condition)')
    parser.add_argument('--dreams', choices=['standard', 'balanced', 'contrastive'],
                       help='Dream strategy (run single condition)')
    parser.add_argument('--run-id', type=int, default=1,
                       help='Run ID (default: 1)')
    parser.add_argument('--max-iterations', type=int, default=10,
                       help='Maximum iterations (default: 10)')
    parser.add_argument('--results-dir', type=str, default='results',
                       help='Results directory (default: results)')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test mode (2 iterations, 3 rules)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be run without running')

    args = parser.parse_args()

    # Determine which conditions to run
    if args.recognition and args.dreams:
        # Run single condition
        conditions = [{'recognition': args.recognition, 'dreams': args.dreams}]
        runs = [args.run_id]
    elif args.dry_run:
        # Show all conditions
        print("\nExperiment Conditions:")
        print("-" * 50)
        for cond in CONDITIONS:
            for run_id in range(1, N_RUNS_PER_CONDITION + 1):
                print(f"  {cond['recognition']}_{cond['dreams']} run {run_id}")
        print(f"\nTotal: {len(CONDITIONS) * N_RUNS_PER_CONDITION} runs")
        return
    else:
        # Run all conditions
        conditions = CONDITIONS
        runs = list(range(1, N_RUNS_PER_CONDITION + 1))

    # Quick test mode
    if args.quick_test:
        print("\n*** QUICK TEST MODE ***")
        args.max_iterations = 2
        # Will be handled in task creation

    # Set up results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger
    log_file = results_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger(log_file)

    logger.info("=" * 70)
    logger.info("RECOGNITION × DREAM STRATEGY EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Conditions: {len(conditions)}")
    logger.info(f"Runs per condition: {len(runs)}")
    logger.info(f"Max iterations: {args.max_iterations}")
    logger.info(f"Results directory: {results_dir}")

    # Run experiments
    all_results = []

    for cond in conditions:
        for run_id in runs:
            config = ExperimentConfig(
                recognition_type=cond['recognition'],
                dream_strategy=cond['dreams'],
                run_id=run_id,
                max_iterations=args.max_iterations,
                results_dir=str(results_dir)
            )

            if args.quick_test:
                config.enumeration_budget = 10_000
                config.enumeration_timeout = 30.0
                config.dreams_per_iteration = 10
                config.recognition_epochs = 2

            try:
                result = run_condition(config, logger)
                all_results.append(result)
            except Exception as e:
                logger.error(f"Condition {config.condition_name()} run {run_id} failed: {e}")
                import traceback
                logger.error(traceback.format_exc())

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Completed {len(all_results)} runs")
    logger.info(f"Results saved to {results_dir}")


if __name__ == '__main__':
    main()
