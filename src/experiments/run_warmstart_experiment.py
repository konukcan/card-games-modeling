#!/usr/bin/env python3
"""
Warm-Start Pretraining Experiment
=================================

This script tests whether warm-starting the recognition model on simpler
pretraining rules improves performance on the main catalogue rules.

Two Experimental Conditions:
1. COLD: Train directly on catalogue rules (no pretraining)
2. WARM: Pretrain on 44 pretraining rules, then fine-tune on catalogue rules

The script can run BOTH conditions in a single invocation for comparison,
or a single condition for separate runs.

Usage:
    # Run both conditions (recommended for comparison)
    python run_warmstart_experiment.py --both

    # Run just warm-start condition
    python run_warmstart_experiment.py --condition WARM

    # Run just cold-start condition
    python run_warmstart_experiment.py --condition COLD

    # Quick test with reduced budget
    python run_warmstart_experiment.py --quick-test

Key Metrics:
- Solve rate on catalogue rules
- Programs enumerated per solution
- Time to first solution per rule
- Recognition model prediction accuracy
"""

import sys
import os
import time
import json
import random
import copy
import shutil
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple, Set, Union
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from dreamcoder_core.type_system import Type, arrow, HAND, BOOL
from dreamcoder_core.program import Program, Primitive, Invented
from dreamcoder_core.grammar import Grammar
from dreamcoder_core.enumeration import TopDownEnumerator
from dreamcoder_core.neural_recognition import NeuralRecognitionModel
from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.compression import compress_frontiers
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.dreamcoder_original import (
    Task, SolutionEntry, TaskFrontier, make_eval_fn, create_tasks_from_rules
)
from dreamcoder_core.contrastive_dreaming import (
    ConfigurableDreamer, ContrastiveDream, ContrastiveDreamer,
    BalancedDreamer, StandardDreamer
)

# Type alias for recognition models
RecognitionModel = Union[NeuralRecognitionModel, ContrastiveRecognitionModel]

# Rules
from rules.pretraining_rules import get_all_pretraining_rules, get_easy_pretraining_rules
from rules.catalogue import create_all_rules as get_catalogue_rules
from rules.cards import sample_hand, Card, Suit, Rank


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for the warm-start experiment."""

    # Identification
    condition: str  # 'COLD', 'WARM', 'BOTH'
    seed: int = 42
    run_id: str = ""
    model_type: str = "neural"  # 'neural' or 'contrastive'

    # Pretraining configuration (for WARM condition)
    pretrain_iterations: int = 5
    pretrain_budget: int = 50000  # Small budget - rules are easy
    pretrain_depth: int = 7
    pretrain_epochs: int = 15
    pretrain_timeout: float = 30.0  # seconds per task

    # Main training configuration
    main_iterations: int = 6
    main_budget: int = 200000
    main_depth: int = 9
    main_epochs: int = 10
    main_timeout: float = 60.0  # seconds per task

    # Recognition model
    hidden_dim: int = 128
    learning_rate: float = 1e-3
    blend_factor: float = 0.5  # How much to trust recognition predictions

    # Architecture
    n_examples: int = 100
    n_holdout: int = 20
    hand_size: int = 6

    # Logging
    log_dir: str = "results/warmstart_experiment"
    checkpoint_every: int = 1
    verbose: bool = True

    # Quick test mode
    quick_test: bool = False
    max_pretrain_rules: int = 0  # 0 = use all rules
    max_catalogue_rules: int = 0  # 0 = use all rules

    # Dreaming configuration (CRITICAL for learning P(primitive|task))
    use_dreaming: bool = True  # Enable fantasy generation
    dreams_per_iteration: int = 50  # Dreams to generate per iteration
    dream_strategy: str = "balanced"  # 'standard', 'balanced', or 'contrastive'
    dream_examples_per_task: int = 10  # Examples per dream task
    dream_temperature: float = 1.0  # Sampling temperature for dream programs
    dream_epochs_per_dream: int = 1  # Training epochs per dream

    # Fantasy pre-training (BEFORE seeing any real tasks)
    fantasy_pretrain: int = 500  # Number of fantasies before iteration 1 (0 = disabled)

    def __post_init__(self):
        if not self.run_id:
            self.run_id = f"{self.model_type}_{self.condition}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


def print_banner(text: str, char: str = "="):
    """Print a banner."""
    line = char * 70
    print(f"\n{line}")
    print(text)
    print(f"{line}\n", flush=True)


def collect_primitives(program: Program) -> Set[str]:
    """Extract all primitive names used in a program."""
    primitives = set()

    def recurse(p):
        if isinstance(p, (Primitive, Invented)):
            primitives.add(str(p))
        elif hasattr(p, 'f') and hasattr(p, 'x'):  # Application
            recurse(p.f)
            recurse(p.x)
        elif hasattr(p, 'body'):  # Abstraction
            recurse(p.body)

    recurse(program)
    return primitives


# ============================================================================
# WARM-START EXPERIMENT RUNNER
# ============================================================================

class WarmStartExperiment:
    """
    Runs the warm-start pretraining experiment.

    Compares:
    - COLD: Direct training on catalogue rules
    - WARM: Pretrain on simpler rules, then train on catalogue rules
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.log_dir = Path(config.log_dir) / config.run_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Set random seed
        random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Build grammar
        self.grammar = build_lean_grammar()
        self.eval_fn = make_eval_fn()

        # Initialize tasks
        self._init_tasks()

        # Results storage
        self.results = {
            'config': asdict(config),
            'pretraining': {},
            'main_training': {},
            'task_metrics': {},
            'recognition_metrics': {}
        }

        # Initialize dreamer if enabled
        self.dreamer = None
        if config.use_dreaming:
            self._init_dreamer()

        if config.verbose:
            print(f"Experiment initialized: {config.run_id}")
            print(f"Recognition model: {config.model_type.upper()}")
            print(f"Grammar: {len(self.grammar)} primitives")
            print(f"Pretraining rules: {len(self.pretrain_tasks)}")
            print(f"Catalogue rules: {len(self.catalogue_tasks)}")
            if config.use_dreaming:
                print(f"Dreaming: ENABLED ({config.dreams_per_iteration} dreams/iter, strategy={config.dream_strategy})")
            else:
                print(f"Dreaming: DISABLED")

    def _init_dreamer(self):
        """Initialize the dream generator."""
        def sample_hand_fn():
            return sample_hand(self.config.hand_size)

        def sample_card_fn():
            """Sample a single random card."""
            return sample_hand(1)[0]

        # Select dreamer based on strategy
        if self.config.dream_strategy == 'contrastive':
            self.dreamer = ContrastiveDreamer(
                grammar=self.grammar,
                eval_fn=self.eval_fn,
                sample_hand_fn=sample_hand_fn,
                sample_card_fn=sample_card_fn,
                device='cpu'
            )
        elif self.config.dream_strategy == 'balanced':
            self.dreamer = BalancedDreamer(
                grammar=self.grammar,
                eval_fn=self.eval_fn,
                sample_hand_fn=sample_hand_fn,
                device='cpu'
            )
        elif self.config.dream_strategy == 'standard':
            self.dreamer = StandardDreamer(
                grammar=self.grammar,
                eval_fn=self.eval_fn,
                sample_hand_fn=sample_hand_fn,
                device='cpu'
            )
        else:
            raise ValueError(f"Unknown dream strategy: {self.config.dream_strategy}")

        self.log(f"Initialized {self.config.dream_strategy} dreamer")

    def _init_tasks(self):
        """Initialize tasks from rules."""
        # Pretraining tasks (44 simpler rules)
        pretrain_rules = get_all_pretraining_rules()
        if self.config.max_pretrain_rules > 0:
            pretrain_rules = pretrain_rules[:self.config.max_pretrain_rules]
        self.pretrain_tasks = create_tasks_from_rules(
            pretrain_rules,
            n_examples=self.config.n_examples,
            n_holdout=self.config.n_holdout,
            hand_size=self.config.hand_size,
            seed=self.config.seed
        )

        # Catalogue tasks (45 experimental rules)
        catalogue_rules = get_catalogue_rules()
        if self.config.max_catalogue_rules > 0:
            catalogue_rules = catalogue_rules[:self.config.max_catalogue_rules]
        self.catalogue_tasks = create_tasks_from_rules(
            catalogue_rules,
            n_examples=self.config.n_examples,
            n_holdout=self.config.n_holdout,
            hand_size=self.config.hand_size,
            seed=self.config.seed + 1  # Different seed for diversity
        )

        # Task name to task mapping
        self.pretrain_task_map = {t.name: t for t in self.pretrain_tasks}
        self.catalogue_task_map = {t.name: t for t in self.catalogue_tasks}

    def log(self, msg: str):
        """Log a message if verbose."""
        if self.config.verbose:
            print(msg, flush=True)

    def log_primitive_predictions(
        self,
        recognition: RecognitionModel,
        tasks: List[Task],
        label: str = ""
    ):
        """
        Log primitive predictions for a sample of tasks to monitor diversity.

        This helps diagnose the "repetitive predictions" problem where the model
        predicts the same primitives for all tasks.
        """
        import torch

        self.log(f"\n  --- Primitive Prediction Analysis {label} ---")

        # Sample up to 5 tasks for analysis
        sample_tasks = tasks[:min(5, len(tasks))]

        all_top_prims = []

        with torch.no_grad():
            for task in sample_tasks:
                # Get predictions based on model type
                if self.config.model_type == 'contrastive':
                    probs_dict = recognition.predict_primitives_dict(task)
                    top_prims = sorted(probs_dict.items(), key=lambda x: -x[1])[:5]
                else:
                    probs_tensor = recognition.predict_primitive_probs(task)
                    top_indices = probs_tensor.topk(5).indices.tolist()
                    top_prims = [(recognition.primitive_names[i], float(probs_tensor[i])) for i in top_indices]

                all_top_prims.append([p[0] for p in top_prims])

                self.log(f"    {task.name}:")
                for prim, prob in top_prims:
                    self.log(f"      {prim}: {prob:.4f}")

        # Compute diversity: how many unique primitives across all top-5s?
        all_unique = set()
        for prims in all_top_prims:
            all_unique.update(prims)

        # Compute overlap: what fraction of primitives appear in ALL tasks' top-5?
        if all_top_prims:
            common = set(all_top_prims[0])
            for prims in all_top_prims[1:]:
                common &= set(prims)
            overlap_pct = 100 * len(common) / 5 if all_top_prims else 0
        else:
            overlap_pct = 0
            common = set()

        self.log(f"  Diversity: {len(all_unique)} unique primitives in top-5s")
        self.log(f"  Overlap: {len(common)}/5 primitives appear in ALL tasks ({overlap_pct:.0f}%)")

        if overlap_pct > 60:
            self.log(f"  WARNING: High overlap suggests repetitive predictions!")
        else:
            self.log(f"  GOOD: Predictions show task-specific variation")

        return {
            'unique_count': len(all_unique),
            'overlap_count': len(common),
            'overlap_pct': overlap_pct,
            'common_prims': list(common)
        }

    def run_fantasy_pretrain(
        self,
        recognition: RecognitionModel,
        request_type: Type
    ) -> Dict[str, Any]:
        """
        Run fantasy pre-training BEFORE seeing any real tasks.

        This teaches the recognition model the grammar's structure and
        primitive relationships before it encounters real tasks.
        """
        if self.config.fantasy_pretrain <= 0:
            return {'skipped': True, 'reason': 'fantasy_pretrain=0'}

        if not self.config.use_dreaming or self.dreamer is None:
            return {'skipped': True, 'reason': 'dreaming disabled'}

        print_banner("FANTASY PRE-TRAINING")
        self.log(f"Training on {self.config.fantasy_pretrain} grammar-sampled fantasies")
        self.log(f"Strategy: {self.config.dream_strategy}")
        self.log("This teaches P(primitive | grammar structure) before real tasks")

        pretrain_start = time.time()

        # Generate and train on fantasies in batches
        batch_size = 50
        total_generated = 0
        total_trained = 0
        all_primitives = set()

        n_batches = (self.config.fantasy_pretrain + batch_size - 1) // batch_size

        for batch_idx in range(n_batches):
            batch_n = min(batch_size, self.config.fantasy_pretrain - total_generated)

            self.log(f"\n  Batch {batch_idx + 1}/{n_batches}: Generating {batch_n} fantasies...")

            stats = self.generate_and_train_on_dreams(
                recognition=recognition,
                frontiers={},  # No real frontiers yet
                request_type=request_type,
                n_dreams=batch_n,
                iteration=-1,  # Special marker for pre-training
                is_pretrain=True
            )

            total_generated += stats.get('dreams_generated', 0)
            total_trained += stats.get('dreams_trained', 0)

            if 'primitives_covered' in stats:
                prims = stats['primitives_covered']
                if isinstance(prims, list):
                    all_primitives.update(prims)
                else:
                    all_primitives.update(prims)

            # Log intermediate primitive predictions every few batches
            if (batch_idx + 1) % 3 == 0:
                sample_tasks = self.catalogue_tasks[:3]
                self.log_primitive_predictions(recognition, sample_tasks, f"(after batch {batch_idx + 1})")

        pretrain_time = time.time() - pretrain_start

        # Final prediction analysis
        self.log_primitive_predictions(recognition, self.catalogue_tasks[:5], "(after pre-training)")

        self.log(f"\n  Fantasy pre-training complete!")
        self.log(f"  Total fantasies: {total_generated}")
        self.log(f"  Trained on: {total_trained}")
        self.log(f"  Primitives covered: {len(all_primitives)}/{len(self.grammar)} ({100*len(all_primitives)/len(self.grammar):.1f}%)")
        self.log(f"  Time: {format_time(pretrain_time)}")

        return {
            'skipped': False,
            'fantasies_generated': total_generated,
            'fantasies_trained': total_trained,
            'primitives_covered': list(all_primitives),
            'n_primitives_covered': len(all_primitives),
            'coverage_pct': 100 * len(all_primitives) / len(self.grammar),
            'time': pretrain_time
        }

    def generate_and_train_on_dreams(
        self,
        recognition: RecognitionModel,
        frontiers: Dict[str, TaskFrontier],
        request_type: Type,
        n_dreams: int,
        iteration: int,
        is_pretrain: bool = False
    ) -> Dict[str, Any]:
        """
        Generate dreams and train the recognition model on them.

        This is CRITICAL for learning P(primitive|task) rather than P(primitive|solved).
        Dreams provide diverse training data by sampling programs from the grammar
        and creating synthetic tasks from them.

        How it works:
        1. Sample programs from the current grammar using stochastic sampling
        2. For each program, generate input-output examples by running on random hands
        3. Create a Task object with these examples
        4. Train the recognition model to predict the program's primitives from examples

        Args:
            recognition: The recognition model to train
            frontiers: Current solved task frontiers (for context, not used directly)
            request_type: Type of programs to sample (HAND → BOOL)
            n_dreams: Number of dreams to generate
            iteration: Current iteration number
            is_pretrain: If True, skip the iteration==0 check (for fantasy pre-training)

        Returns:
            Dictionary with dream generation statistics
        """
        if not self.config.use_dreaming or self.dreamer is None:
            return {'dreams_generated': 0, 'primitives_covered': set()}

        # Skip iteration 0 only if not doing pre-training
        if iteration == 0 and not is_pretrain:
            return {'dreams_generated': 0, 'primitives_covered': set(), 'skipped': 'first_iteration'}

        self.log(f"  Generating {n_dreams} dreams (strategy: {self.config.dream_strategy})...")
        dream_start = time.time()

        # Generate dreams
        dreams = self.dreamer.generate_dreams(
            request_type=request_type,
            n_dreams=n_dreams,
            n_examples_per_dream=self.config.dream_examples_per_task,
            temperature=self.config.dream_temperature,
            verbose=False
        )

        if not dreams:
            self.log(f"  Warning: No dreams generated")
            return {'dreams_generated': 0, 'primitives_covered': set(), 'generation_failed': True}

        # Collect statistics about primitives covered
        all_primitives_in_dreams = set()
        for dream in dreams:
            all_primitives_in_dreams.update(dream.primitives_used)

        # Train recognition model on each dream
        # Each dream is treated as a "solved task" with known solution
        dreams_trained = 0
        for dream in dreams:
            # Create a temporary frontier with the dream's program as the solution
            temp_frontier = TaskFrontier(dream.task)

            # Compute log probability of the dream program under current grammar
            log_prob = self.grammar.program_log_likelihood(
                dream.program, dream.task.request_type
            )

            entry = SolutionEntry(
                program=dream.program,
                log_probability=log_prob,
                log_likelihood=0.0,
                programs_enumerated=0,
                time_found=0.0
            )
            temp_frontier.add(entry)

            # Train recognition on this dream
            try:
                recognition.train_on_frontiers(
                    tasks=[dream.task],
                    frontiers={dream.task.name: temp_frontier},
                    epochs=self.config.dream_epochs_per_dream
                )
                dreams_trained += 1
            except Exception as e:
                self.log(f"  Warning: Failed to train on dream: {e}")

        dream_time = time.time() - dream_start
        self.log(f"  Generated {len(dreams)} dreams, trained on {dreams_trained}")
        self.log(f"  Primitives covered by dreams: {len(all_primitives_in_dreams)}")
        self.log(f"  Dream time: {dream_time:.1f}s")

        return {
            'dreams_generated': len(dreams),
            'dreams_trained': dreams_trained,
            'primitives_covered': list(all_primitives_in_dreams),
            'n_primitives_covered': len(all_primitives_in_dreams),
            'dream_time': dream_time
        }

    def enumerate_task(
        self,
        task: Task,
        grammar: Grammar,
        max_programs: int,
        max_depth: int,
        timeout: float,
        show_progress: bool = False
    ) -> TaskFrontier:
        """
        Enumerate programs for a single task.

        Returns a TaskFrontier with solutions found.
        """
        frontier = TaskFrontier(task)

        enumerator = TopDownEnumerator(
            grammar,
            max_depth=max_depth,
            max_programs=max_programs
        )

        start_time = time.time()
        programs_tried = 0
        last_progress = 0

        for program, log_prob in enumerator.enumerate(
            task.request_type,
            timeout_seconds=timeout
        ):
            programs_tried += 1

            # Progress output every 1000 programs
            if show_progress and programs_tried - last_progress >= 1000:
                elapsed = time.time() - start_time
                rate = programs_tried / elapsed if elapsed > 0 else 0
                print(f"    [{task.name}] {programs_tried:,} programs ({rate:.0f}/s)...", end='\r', flush=True)
                last_progress = programs_tried

            if programs_tried > max_programs:
                break
            if time.time() - start_time > timeout:
                break

            # Evaluate on examples
            try:
                correct = 0
                for inp, expected in task.examples:
                    result = self.eval_fn(program, inp)
                    if result == expected:
                        correct += 1

                if correct == len(task.examples):
                    # Found a solution - verify on holdout
                    holdout_correct = 0
                    if hasattr(task, 'holdout_examples') and task.holdout_examples:
                        for inp, expected in task.holdout_examples:
                            try:
                                if self.eval_fn(program, inp) == expected:
                                    holdout_correct += 1
                            except Exception:
                                pass

                        # Require at least 80% holdout accuracy
                        holdout_rate = holdout_correct / len(task.holdout_examples)
                        if holdout_rate < 0.8:
                            continue  # Spurious solution

                    entry = SolutionEntry(
                        program=program,
                        log_probability=log_prob,
                        log_likelihood=0.0,
                        programs_enumerated=programs_tried,
                        time_found=time.time() - start_time
                    )
                    frontier.add(entry)
                    frontier.total_programs_searched = programs_tried
                    frontier.total_time = time.time() - start_time
                    break  # Found valid solution

            except Exception:
                pass  # Program crashed - skip

        frontier.total_programs_searched = programs_tried
        frontier.total_time = time.time() - start_time

        return frontier

    def run_pretraining(self, recognition: NeuralRecognitionModel) -> Dict:
        """
        Run warm-start pretraining on simpler rules.

        Returns metrics about pretraining phase.
        """
        print_banner("PHASE 1: WARM-START PRETRAINING")
        self.log(f"Training on {len(self.pretrain_tasks)} pretraining rules")
        self.log(f"Budget per task: {self.config.pretrain_budget} programs")
        self.log(f"Max depth: {self.config.pretrain_depth}")

        phase_start = time.time()
        frontiers = {}
        all_solved = []

        for iteration in range(self.config.pretrain_iterations):
            iter_start = time.time()
            iter_solved = 0
            iter_programs = 0

            self.log(f"\n--- Pretraining Iteration {iteration + 1}/{self.config.pretrain_iterations} ---")

            # Get recognition-biased grammar for this iteration
            if iteration > 0:
                # After first iteration, use recognition to guide search
                biased_grammar = copy.deepcopy(self.grammar)
            else:
                biased_grammar = self.grammar

            # Enumerate for each unsolved pretraining task
            for task in self.pretrain_tasks:
                if task.name in frontiers and frontiers[task.name].solved:
                    iter_solved += 1
                    continue  # Already solved

                # Get task-specific grammar weights
                if iteration > 0:
                    # Handle different model interfaces
                    if self.config.model_type == 'contrastive':
                        biased_grammar = recognition.predict_grammar_weights(task)
                    else:
                        biased_grammar = recognition.predict_grammar_weights(
                            task, blend_factor=self.config.blend_factor
                        )

                frontier = self.enumerate_task(
                    task,
                    biased_grammar,
                    max_programs=self.config.pretrain_budget,
                    max_depth=self.config.pretrain_depth,
                    timeout=self.config.pretrain_timeout,
                    show_progress=self.config.quick_test
                )

                iter_programs += frontier.total_programs_searched
                if self.config.quick_test:
                    print()  # Clear progress line

                if frontier.solved:
                    frontiers[task.name] = frontier
                    iter_solved += 1
                    all_solved.append(task.name)
                    if self.config.verbose:
                        print(f"  ✓ {task.name}: {frontier.best.programs_enumerated} programs")

            self.log(f"  Solved: {iter_solved}/{len(self.pretrain_tasks)}")
            self.log(f"  Programs: {iter_programs:,}")
            self.log(f"  Time: {format_time(time.time() - iter_start)}")

            # Train recognition on solved tasks
            loss = None
            if all_solved:
                solved_tasks = [self.pretrain_task_map[name] for name in all_solved]
                loss = recognition.train_on_frontiers(
                    solved_tasks, frontiers, epochs=self.config.pretrain_epochs
                )
                self.log(f"  Recognition loss: {loss:.4f}")

            # DREAMING: Train on synthetic tasks to learn P(primitive|task)
            dream_stats = self.generate_and_train_on_dreams(
                recognition=recognition,
                frontiers=frontiers,
                request_type=self.pretrain_tasks[0].request_type,
                n_dreams=self.config.dreams_per_iteration,
                iteration=iteration
            )

            # Checkpoint
            if (iteration + 1) % self.config.checkpoint_every == 0:
                checkpoint_path = self.log_dir / f"pretrain_checkpoint_iter{iteration+1}.pt"
                recognition.save(str(checkpoint_path))

        # Compile pretraining results
        pretrain_results = {
            'total_time': time.time() - phase_start,
            'tasks_solved': len(frontiers),
            'tasks_total': len(self.pretrain_tasks),
            'solve_rate': len(frontiers) / len(self.pretrain_tasks),
            'solved_tasks': list(frontiers.keys()),
            'final_recognition_loss': loss if all_solved else None
        }

        self.log(f"\nPretraining complete: {len(frontiers)}/{len(self.pretrain_tasks)} rules solved")
        self.log(f"Total time: {format_time(pretrain_results['total_time'])}")

        # Save pretrained model
        pretrain_model_path = self.log_dir / "pretrained_recognition.pt"
        recognition.save(str(pretrain_model_path))
        self.log(f"Saved pretrained model to: {pretrain_model_path}")

        return pretrain_results

    def run_main_training(
        self,
        recognition: NeuralRecognitionModel,
        condition: str
    ) -> Dict:
        """
        Run main training on catalogue rules.

        Args:
            recognition: Recognition model (pretrained for WARM, fresh for COLD)
            condition: 'WARM' or 'COLD'

        Returns metrics about main training phase.
        """
        print_banner(f"PHASE 2: MAIN TRAINING ({condition})")
        self.log(f"Training on {len(self.catalogue_tasks)} catalogue rules")
        self.log(f"Budget per task: {self.config.main_budget} programs")
        self.log(f"Max depth: {self.config.main_depth}")

        phase_start = time.time()
        frontiers = {}
        task_metrics = {}

        for iteration in range(self.config.main_iterations):
            iter_start = time.time()
            iter_solved = 0
            iter_programs = 0
            newly_solved = []

            self.log(f"\n--- Main Iteration {iteration + 1}/{self.config.main_iterations} ---")

            for task in self.catalogue_tasks:
                task_start = time.time()

                if task.name in frontiers and frontiers[task.name].solved:
                    iter_solved += 1
                    continue

                # Get recognition-guided grammar
                if iteration > 0 or condition == 'WARM':
                    # Handle different model interfaces
                    if self.config.model_type == 'contrastive':
                        biased_grammar = recognition.predict_grammar_weights(task)
                    else:
                        biased_grammar = recognition.predict_grammar_weights(
                            task, blend_factor=self.config.blend_factor
                        )
                else:
                    biased_grammar = self.grammar

                frontier = self.enumerate_task(
                    task,
                    biased_grammar,
                    max_programs=self.config.main_budget,
                    max_depth=self.config.main_depth,
                    timeout=self.config.main_timeout,
                    show_progress=self.config.quick_test
                )

                iter_programs += frontier.total_programs_searched
                if self.config.quick_test:
                    print()  # Clear progress line

                if frontier.solved:
                    frontiers[task.name] = frontier
                    iter_solved += 1
                    newly_solved.append(task.name)

                    # Record metrics
                    task_metrics[task.name] = {
                        'solved': True,
                        'iteration_solved': iteration + 1,
                        'programs_enumerated': frontier.total_programs_searched,
                        'time_to_solve': time.time() - task_start,
                        'solution': str(frontier.best.program),
                        'primitives_used': list(collect_primitives(frontier.best.program))
                    }

                    if self.config.verbose:
                        print(f"  ✓ {task.name}: {frontier.best.programs_enumerated} programs")
                else:
                    # Track unsolved task progress
                    if task.name not in task_metrics:
                        task_metrics[task.name] = {
                            'solved': False,
                            'programs_enumerated': frontier.total_programs_searched
                        }
                    else:
                        task_metrics[task.name]['programs_enumerated'] += frontier.total_programs_searched

            self.log(f"  Solved: {iter_solved}/{len(self.catalogue_tasks)}")
            self.log(f"  New this iteration: {len(newly_solved)}")
            self.log(f"  Programs: {iter_programs:,}")
            self.log(f"  Time: {format_time(time.time() - iter_start)}")

            # Train recognition on all solved tasks
            all_solved = list(frontiers.keys())
            if newly_solved:
                solved_tasks = [self.catalogue_task_map[name] for name in all_solved]
                loss = recognition.train_on_frontiers(
                    solved_tasks, frontiers, epochs=self.config.main_epochs
                )
                self.log(f"  Recognition loss: {loss:.4f}")

            # DREAMING: Train on synthetic tasks to learn P(primitive|task)
            dream_stats = self.generate_and_train_on_dreams(
                recognition=recognition,
                frontiers=frontiers,
                request_type=self.catalogue_tasks[0].request_type,
                n_dreams=self.config.dreams_per_iteration,
                iteration=iteration
            )

            # Log primitive predictions to monitor for repetitive prediction issue
            # Sample unsolved tasks for analysis
            unsolved_tasks = [t for t in self.catalogue_tasks if t.name not in frontiers]
            if unsolved_tasks:
                sample_unsolved = unsolved_tasks[:min(3, len(unsolved_tasks))]
                pred_stats = self.log_primitive_predictions(
                    recognition, sample_unsolved,
                    f"(iter {iteration + 1}, on unsolved tasks)"
                )

                # Store iteration stats for later analysis
                if 'iteration_stats' not in task_metrics:
                    task_metrics['iteration_stats'] = []
                task_metrics['iteration_stats'].append({
                    'iteration': iteration + 1,
                    'solved': iter_solved,
                    'new_solved': len(newly_solved),
                    'prediction_diversity': pred_stats['unique_count'],
                    'prediction_overlap_pct': pred_stats['overlap_pct'],
                    'dreams_generated': dream_stats.get('dreams_generated', 0)
                })

            # Early stopping if all solved
            if iter_solved == len(self.catalogue_tasks):
                self.log("All tasks solved! Stopping early.")
                break

        # Compile main training results
        main_results = {
            'condition': condition,
            'total_time': time.time() - phase_start,
            'tasks_solved': len(frontiers),
            'tasks_total': len(self.catalogue_tasks),
            'solve_rate': len(frontiers) / len(self.catalogue_tasks),
            'solved_tasks': list(frontiers.keys()),
            'unsolved_tasks': [t.name for t in self.catalogue_tasks if t.name not in frontiers],
            'task_metrics': task_metrics
        }

        self.log(f"\nMain training complete: {len(frontiers)}/{len(self.catalogue_tasks)} rules solved")
        self.log(f"Solve rate: {100 * main_results['solve_rate']:.1f}%")
        self.log(f"Total time: {format_time(main_results['total_time'])}")

        return main_results

    def run_condition(self, condition: str) -> Dict:
        """
        Run a single experimental condition.

        Args:
            condition: 'WARM' or 'COLD'

        Returns complete results for this condition.
        """
        start_time = time.time()

        # Initialize fresh recognition model based on model_type
        if self.config.model_type == 'contrastive':
            recognition = ContrastiveRecognitionModel(
                grammar=self.grammar,
                card_hidden=self.config.hidden_dim,
                card_out=self.config.hidden_dim // 2,
                pred_hidden=self.config.hidden_dim,
                learning_rate=self.config.learning_rate,
                device='cpu'
            )
        else:  # default to neural
            recognition = NeuralRecognitionModel(
                grammar=self.grammar,
                hidden_dim=self.config.hidden_dim,
                learning_rate=self.config.learning_rate,
                device='cpu'
            )

        results = {
            'condition': condition,
            'start_time': datetime.now().isoformat(),
            'config': asdict(self.config),
            'model_type': self.config.model_type
        }

        # Log initial primitive predictions (before any training)
        self.log_primitive_predictions(
            recognition,
            self.catalogue_tasks[:5],
            "(initial - before any training)"
        )

        # Run fantasy pre-training BEFORE any real tasks
        # This teaches P(primitive | grammar structure) to prevent collapse
        fantasy_pretrain_results = self.run_fantasy_pretrain(
            recognition,
            request_type=self.catalogue_tasks[0].request_type
        )
        results['fantasy_pretrain'] = fantasy_pretrain_results

        if condition == 'WARM':
            # Phase 1: Pretrain on simpler rules
            pretrain_results = self.run_pretraining(recognition)
            results['pretraining'] = pretrain_results

        # Phase 2: Main training on catalogue rules
        main_results = self.run_main_training(recognition, condition)
        results['main_training'] = main_results
        results['task_metrics'] = main_results['task_metrics']

        # Final metrics
        results['total_time'] = time.time() - start_time
        results['final_solve_rate'] = main_results['solve_rate']

        # Save results
        results_path = self.log_dir / f"results_{condition}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        self.log(f"Results saved to: {results_path}")

        # Save final model
        model_path = self.log_dir / f"recognition_model_{condition}.pt"
        recognition.save(str(model_path))

        return results

    def run(self) -> Dict:
        """
        Run the full experiment based on configuration.

        Returns:
            Dictionary with results for all conditions run.
        """
        print_banner("WARM-START PRETRAINING EXPERIMENT")
        print(f"Run ID: {self.config.run_id}")
        print(f"Condition: {self.config.condition}")
        print(f"Log directory: {self.log_dir}")
        print(f"Seed: {self.config.seed}")

        all_results = {}

        if self.config.condition in ['COLD', 'BOTH']:
            self.log("\n" + "="*70)
            self.log("Running COLD condition (no pretraining)")
            self.log("="*70)
            all_results['COLD'] = self.run_condition('COLD')

        if self.config.condition in ['WARM', 'BOTH']:
            self.log("\n" + "="*70)
            self.log("Running WARM condition (with pretraining)")
            self.log("="*70)
            all_results['WARM'] = self.run_condition('WARM')

        # Compare results if both conditions run
        if 'COLD' in all_results and 'WARM' in all_results:
            self._compare_conditions(all_results['COLD'], all_results['WARM'])

        # Save combined results
        combined_path = self.log_dir / "combined_results.json"
        with open(combined_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        print_banner("EXPERIMENT COMPLETE")
        print(f"Results saved to: {self.log_dir}")

        return all_results

    def _compare_conditions(self, cold: Dict, warm: Dict):
        """Compare COLD vs WARM results."""
        print_banner("COMPARISON: COLD vs WARM")

        cold_rate = cold['final_solve_rate']
        warm_rate = warm['final_solve_rate']
        diff = warm_rate - cold_rate

        print(f"COLD solve rate: {100*cold_rate:.1f}%")
        print(f"WARM solve rate: {100*warm_rate:.1f}%")
        print(f"Difference: {100*diff:+.1f} percentage points")
        print()

        # Tasks solved by WARM but not COLD (positive transfer)
        cold_solved = set(cold['main_training']['solved_tasks'])
        warm_solved = set(warm['main_training']['solved_tasks'])

        positive_transfer = warm_solved - cold_solved
        negative_transfer = cold_solved - warm_solved

        if positive_transfer:
            print(f"Positive transfer ({len(positive_transfer)} rules):")
            for task in sorted(positive_transfer):
                print(f"  + {task}")

        if negative_transfer:
            print(f"\nNegative transfer ({len(negative_transfer)} rules):")
            for task in sorted(negative_transfer):
                print(f"  - {task}")

        # Time comparison
        cold_time = cold['total_time']
        warm_time = warm['total_time']
        print(f"\nCOLD total time: {format_time(cold_time)}")
        print(f"WARM total time: {format_time(warm_time)}")

        # Efficiency: improvement per extra time
        if warm_time > cold_time and diff > 0:
            efficiency = diff / ((warm_time - cold_time) / cold_time)
            print(f"Transfer efficiency: {efficiency:.2f} (improvement per unit extra time)")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Warm-Start Pretraining Experiment"
    )
    parser.add_argument(
        '--condition',
        choices=['COLD', 'WARM', 'BOTH'],
        default='BOTH',
        help="Which condition(s) to run"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help="Run with reduced budget for quick testing"
    )
    parser.add_argument(
        '--pretrain-iters',
        type=int,
        default=5,
        help="Number of pretraining iterations"
    )
    parser.add_argument(
        '--main-iters',
        type=int,
        default=6,
        help="Number of main training iterations"
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default="results/warmstart_experiment",
        help="Directory for results"
    )
    parser.add_argument(
        '--model',
        choices=['neural', 'contrastive'],
        default='neural',
        help="Recognition model type: 'neural' (GRU-based) or 'contrastive' (factored embeddings)"
    )
    parser.add_argument(
        '--no-dreaming',
        action='store_true',
        help="Disable dream generation (NOT recommended - causes P(primitive|task) collapse)"
    )
    parser.add_argument(
        '--dreams-per-iter',
        type=int,
        default=50,
        help="Number of dreams to generate per iteration (default: 50)"
    )
    parser.add_argument(
        '--dream-strategy',
        choices=['standard', 'balanced', 'contrastive'],
        default='balanced',
        help="Dream generation strategy: 'standard', 'balanced', or 'contrastive' (default: balanced)"
    )
    parser.add_argument(
        '--fantasy-pretrain',
        type=int,
        default=500,
        help="Number of fantasies to train on BEFORE seeing any real tasks (default: 500, 0 to disable)"
    )

    args = parser.parse_args()

    # Build configuration
    config = ExperimentConfig(
        condition=args.condition,
        seed=args.seed,
        pretrain_iterations=args.pretrain_iters,
        main_iterations=args.main_iters,
        log_dir=args.log_dir,
        model_type=args.model,
        use_dreaming=not args.no_dreaming,
        dreams_per_iteration=args.dreams_per_iter,
        dream_strategy=args.dream_strategy,
        fantasy_pretrain=args.fantasy_pretrain
    )

    # Quick test mode: reduce budgets and limit rules
    if args.quick_test:
        config.quick_test = True
        config.pretrain_iterations = 1
        config.pretrain_budget = 2000
        config.main_iterations = 1
        config.main_budget = 3000
        config.pretrain_epochs = 3
        config.main_epochs = 2
        config.n_examples = 30
        config.max_pretrain_rules = 5  # Only 5 pretraining rules
        config.max_catalogue_rules = 5  # Only 5 catalogue rules
        config.pretrain_timeout = 10.0  # Shorter timeout
        config.main_timeout = 15.0
        config.dreams_per_iteration = 10  # Reduced dreams for quick test
        config.fantasy_pretrain = 50  # Reduced pre-training for quick test
        print("QUICK TEST MODE: Using 5 rules, reduced budgets, 50 fantasy pretrain, 10 dreams/iter")

    # Run experiment
    experiment = WarmStartExperiment(config)
    results = experiment.run()

    return results


if __name__ == "__main__":
    main()
