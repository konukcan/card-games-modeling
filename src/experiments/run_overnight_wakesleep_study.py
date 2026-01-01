#!/usr/bin/env python3
"""
Overnight Wake-Sleep Library Study
===================================

This script tests library variants under REAL wake-sleep conditions, including:
- Recognition model training (neural transfer between tasks)
- Compression/abstraction learning (library evolution)
- Dreaming (synthetic task generation)

Unlike the previous primitive ablation study (which only tested enumeration),
this runs the FULL DreamCoder pipeline on each variant.

Tracks:
  Track A (Ablation): Test effect of removing primitives
  Track B (Addition): Test effect of adding technical primitives

Comprehensive Logging:
  - Recognition loss per epoch (training dynamics)
  - Per-task primitive predictions (detect model collapse)
  - Model weights at each checkpoint (for post-hoc analysis)
  - Learned abstractions (library evolution)
  - Dream contents and quality
  - Task embeddings (for clustering analysis)

Runtime: ~10-12 hours (3 variants × 4 hours each)

Usage:
    nohup caffeinate -d -i -s python3 experiments/run_overnight_wakesleep_study.py \\
        > overnight_wakesleep.out 2>&1 &
"""

import sys
import json
import time
import copy
import random
import pickle
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from collections import defaultdict
from functools import reduce

import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.type_system import arrow, BOOL, INT, HAND, TypeVariable, ListType
from dreamcoder_core.program import Program, Primitive, Application, Abstraction, Index, Invented
from dreamcoder_core.grammar import Grammar, Production, uniform_grammar
from dreamcoder_core.enumeration import TopDownEnumerator, EnumerationResult
from dreamcoder_core.compression import compress_frontiers, CompressionResult
from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.contrastive_dreaming import HybridDreamer, ContrastiveDream
from dreamcoder_core.lean_primitives import build_lean_primitives, build_lean_grammar
from dreamcoder_core.task import Task
from dreamcoder_core.task_generation import load_prerecorded_tasks

from rules.cards import Card, Hand, Suit, Rank, sample_hand


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for a wake-sleep experiment variant."""
    name: str
    track: str  # 'A' (ablation) or 'B' (addition)
    description: str

    # Primitive modifications
    remove_primitives: List[str] = field(default_factory=list)
    add_primitives: List[str] = field(default_factory=list)

    # Wake-sleep settings
    iterations: int = 6
    enumeration_budget: int = 150_000
    enumeration_timeout: float = 120.0
    max_depth: int = 8

    # Recognition settings
    recognition_epochs: int = 15
    recognition_lr: float = 1e-3
    recognition_hidden_dim: int = 32
    structural_similarity_weight: float = 0.3

    # Compression settings
    max_inventions_per_iteration: int = 3
    min_compression_savings: float = 2.0

    # Dreaming settings
    dreams_per_iteration: int = 100
    contrastive_dream_ratio: float = 0.5


# Primitives that can be added back
def make_addition_primitives() -> Dict[str, Primitive]:
    """Technical primitives removed from v3 that could be added back."""
    a = TypeVariable(0)
    b = TypeVariable(1)
    c = TypeVariable(2)

    return {
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
    }


# Gestalt primitives to remove for minimal library
GESTALT_PRIMITIVES = [
    'all_same_suit', 'all_same_color',
    'n_unique_suits', 'n_unique_ranks', 'n_unique_colors',
    'has_suit', 'has_color',
    'count_suit', 'count_color',
    'first_half', 'second_half'
]


# ============================================================================
# LOGGING INFRASTRUCTURE
# ============================================================================

@dataclass
class RecognitionEpochLog:
    """Detailed logging for each training epoch."""
    epoch: int
    total_loss: float
    pred_loss: float
    struct_loss: float
    count_loss: float
    n_tasks: int


@dataclass
class TaskPredictionLog:
    """Per-task primitive predictions for model collapse detection."""
    task_name: str
    top_5_primitives: List[Tuple[str, float]]
    prediction_entropy: float  # High entropy = uniform/collapsed, low = confident
    predicted_count: float


@dataclass
class AbstractionLog:
    """Log learned abstractions."""
    iteration: int
    abstraction_name: str
    abstraction_body: str
    size: int
    estimated_savings: float


@dataclass
class DreamLog:
    """Log dream contents and quality."""
    iteration: int
    n_standard_dreams: int
    n_contrastive_dreams: int
    avg_examples_per_dream: float
    sample_dream_programs: List[str]  # First 5 dream programs


@dataclass
class IterationLog:
    """Complete logging for one wake-sleep iteration."""
    iteration: int

    # Wake phase
    tasks_solved: int
    tasks_total: int
    total_programs_enumerated: int
    programs_per_second: float
    wake_time_seconds: float
    newly_solved_this_iter: List[str]

    # Recognition training
    recognition_epochs: List[RecognitionEpochLog]
    final_recognition_loss: float
    recognition_time_seconds: float

    # Per-task predictions (for collapse detection)
    task_predictions: List[TaskPredictionLog]
    mean_prediction_entropy: float
    prediction_entropy_std: float

    # Compression/abstraction
    new_abstractions: List[AbstractionLog]
    compression_time_seconds: float
    grammar_size: int

    # Dreaming
    dream_log: DreamLog
    dream_time_seconds: float

    # Model state
    model_checkpoint_path: str


@dataclass
class ExperimentResult:
    """Complete result of a variant experiment."""
    config: ExperimentConfig
    start_time: str
    end_time: str
    total_duration_seconds: float

    # Per-iteration logs
    iteration_logs: List[IterationLog]

    # Final summary
    final_solve_rate: float
    cumulative_solved: int
    all_abstractions_learned: List[str]
    final_grammar: List[str]

    # Task-level results
    solved_tasks: Dict[str, str]  # task_name -> solution
    unsolved_tasks: List[str]


# ============================================================================
# WAKE-SLEEP ENGINE WITH COMPREHENSIVE LOGGING
# ============================================================================

class LoggingWakeSleep:
    """
    Wake-sleep learner with comprehensive logging for overnight study.

    This wraps ContrastiveWakeSleep with additional logging:
    - Per-epoch recognition losses
    - Per-task predictions
    - Model checkpoints
    - Abstraction details
    - Dream contents
    """

    def __init__(
        self,
        config: ExperimentConfig,
        grammar: Grammar,
        tasks: List[Task],
        output_dir: Path,
        device: str = 'cpu'
    ):
        self.config = config
        self.grammar = copy.deepcopy(grammar)
        self.tasks = tasks
        self.output_dir = output_dir
        self.device = device

        # Create output directories
        self.checkpoints_dir = output_dir / 'checkpoints'
        self.predictions_dir = output_dir / 'predictions'
        self.dreams_dir = output_dir / 'dreams'
        for d in [self.checkpoints_dir, self.predictions_dir, self.dreams_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Initialize recognition model
        self.recognition = ContrastiveRecognitionModel(
            grammar=grammar,
            card_out=config.recognition_hidden_dim,
            pred_hidden=config.recognition_hidden_dim * 2,
            learning_rate=config.recognition_lr,
            device=device
        )

        # Initialize dreamer
        self.dreamer = HybridDreamer(
            grammar=grammar,
            eval_fn=self._eval_program,
            sample_hand_fn=lambda: sample_hand(6),
            sample_card_fn=lambda: sample_hand(1)[0],
            contrastive_ratio=config.contrastive_dream_ratio,
            device=device
        )

        # Frontiers (task -> solutions)
        self.frontiers: Dict[str, List[Tuple[Program, float]]] = {
            t.name: [] for t in tasks
        }

        # Iteration logs
        self.iteration_logs: List[IterationLog] = []

        # Cumulative tracking
        self.cumulative_solved: Set[str] = set()
        self.all_abstractions: List[str] = []

    def save_transfer_state(self, path: Path) -> None:
        """Save model weights and grammar for transfer to Phase 2."""
        state = {
            'model_state_dict': self.recognition.state_dict(),
            'grammar_productions': [
                (str(p.program), p.log_probability, str(p.program.infer()))
                for p in self.grammar.productions
            ],
            'all_abstractions': self.all_abstractions,
            'cumulative_solved': list(self.cumulative_solved),
        }
        torch.save(state, path)
        print(f"Saved transfer state to {path}")

    @classmethod
    def from_transfer_state(
        cls,
        transfer_path: Path,
        config: 'ExperimentConfig',
        new_tasks: List[Task],
        output_dir: Path,
        device: str = 'cpu'
    ) -> 'LoggingWakeSleep':
        """Create a new learner initialized from Phase 1 transfer state."""
        # Load transfer state
        state = torch.load(transfer_path, weights_only=False)
        print(f"Loaded transfer state from {transfer_path}")
        print(f"  Abstractions learned: {len(state['all_abstractions'])}")
        print(f"  Tasks previously solved: {len(state['cumulative_solved'])}")

        # Rebuild grammar from saved productions
        from dreamcoder_core.lean_primitives import build_lean_grammar
        base_grammar = build_lean_grammar()

        # Parse saved productions to rebuild grammar with inventions
        for prod_str, log_prob, type_str in state['grammar_productions']:
            if prod_str.startswith('#'):
                # This is an invention - add it to grammar
                try:
                    inv = Invented.parse(prod_str)
                    if inv not in [p.program for p in base_grammar.productions]:
                        base_grammar = base_grammar.add_invention(inv, log_prob)
                except Exception as e:
                    print(f"Warning: Could not parse invention {prod_str}: {e}")

        # Create the learner
        learner = cls(
            config=config,
            grammar=base_grammar,
            tasks=new_tasks,
            output_dir=output_dir,
            device=device
        )

        # Load model weights (handle potential dimension mismatches)
        try:
            learner.recognition.load_state_dict(state['model_state_dict'])
            print("  Model weights loaded successfully")
        except RuntimeError as e:
            print(f"  Warning: Model weight mismatch (likely due to new primitives): {e}")
            print("  Will use partially loaded weights where dimensions match")
            # Load compatible weights only
            current_state = learner.recognition.state_dict()
            loaded_state = state['model_state_dict']
            for key in current_state:
                if key in loaded_state and current_state[key].shape == loaded_state[key].shape:
                    current_state[key] = loaded_state[key]
            learner.recognition.load_state_dict(current_state)

        # Transfer abstraction history
        learner.all_abstractions = state['all_abstractions']

        return learner

    def _eval_program(self, program: Program, hand: Hand) -> Optional[bool]:
        """Safely evaluate a program on a hand."""
        try:
            fn = program.evaluate([])
            result = fn(hand)
            return result if isinstance(result, bool) else None
        except Exception:
            return None

    def _check_solution(
        self,
        program: Program,
        examples: List[Tuple[Hand, bool]]
    ) -> bool:
        """Check if program solves all examples."""
        for hand, expected in examples:
            result = self._eval_program(program, hand)
            if result != expected:
                return False
        return True

    def run(self) -> ExperimentResult:
        """Run full wake-sleep experiment with comprehensive logging."""
        start_time = datetime.now()

        print(f"\n{'='*70}")
        print(f"WAKE-SLEEP EXPERIMENT: {self.config.name}")
        print(f"{'='*70}")
        print(f"Track: {self.config.track}")
        print(f"Description: {self.config.description}")
        print(f"Iterations: {self.config.iterations}")
        print(f"Tasks: {len(self.tasks)}")
        print(f"Grammar size: {len(self.grammar)} primitives")
        print(f"Output: {self.output_dir}")
        print()

        for iteration in range(1, self.config.iterations + 1):
            print(f"\n{'='*60}")
            print(f"ITERATION {iteration}/{self.config.iterations}")
            print(f"{'='*60}")

            iter_log = self._run_iteration(iteration)
            self.iteration_logs.append(iter_log)

            # Save iteration log
            with open(self.output_dir / f'iter_{iteration:02d}_log.json', 'w') as f:
                json.dump(asdict(iter_log), f, indent=2, default=str)

            # Print summary
            print(f"\n--- Iteration {iteration} Summary ---")
            print(f"Solved: {iter_log.tasks_solved}/{iter_log.tasks_total} "
                  f"({100*iter_log.tasks_solved/iter_log.tasks_total:.1f}%)")
            print(f"Newly solved: {len(iter_log.newly_solved_this_iter)}")
            print(f"Recognition loss: {iter_log.final_recognition_loss:.4f}")
            print(f"Mean prediction entropy: {iter_log.mean_prediction_entropy:.4f}")
            print(f"New abstractions: {len(iter_log.new_abstractions)}")
            print(f"Grammar size: {iter_log.grammar_size}")

        end_time = datetime.now()

        # Compile final result
        result = ExperimentResult(
            config=self.config,
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            total_duration_seconds=(end_time - start_time).total_seconds(),
            iteration_logs=self.iteration_logs,
            final_solve_rate=len(self.cumulative_solved) / len(self.tasks),
            cumulative_solved=len(self.cumulative_solved),
            all_abstractions_learned=self.all_abstractions,
            final_grammar=[str(p.program) for p in self.grammar.productions],
            solved_tasks={
                name: str(sols[0][0]) if sols else ""
                for name, sols in self.frontiers.items()
                if sols
            },
            unsolved_tasks=[
                t.name for t in self.tasks
                if t.name not in self.cumulative_solved
            ]
        )

        # Save final result
        with open(self.output_dir / 'final_result.json', 'w') as f:
            json.dump(asdict(result), f, indent=2, default=str)

        return result

    def _run_iteration(self, iteration: int) -> IterationLog:
        """Run one complete wake-sleep iteration."""

        # =====================
        # WAKE PHASE
        # =====================
        print("\n[WAKE] Recognition-guided enumeration...")
        wake_start = time.time()

        total_programs = 0
        newly_solved = []

        for i, task in enumerate(self.tasks, 1):
            # Skip if already solved
            if task.name in self.cumulative_solved:
                continue

            # Get recognition guidance (after first iteration)
            if iteration > 1:
                self.recognition.grammar = self.grammar
                task_grammar = self.recognition.predict_grammar_weights(task)
            else:
                task_grammar = self.grammar

            # Enumerate
            task_start = time.time()
            enumerator = TopDownEnumerator(
                task_grammar,
                max_depth=self.config.max_depth,
                max_programs=self.config.enumeration_budget
            )

            programs_tried = 0
            for program, log_prob in enumerator.enumerate_memoized(
                arrow(HAND, BOOL),
                max_cost=50.0,
                timeout_seconds=self.config.enumeration_timeout
            ):
                programs_tried += 1

                if programs_tried > self.config.enumeration_budget:
                    break
                if time.time() - task_start > self.config.enumeration_timeout:
                    break

                # Check solution
                if self._check_solution(program, task.examples):
                    # Verify on holdout
                    if self._check_solution(program, task.holdout):
                        self.frontiers[task.name].append((program, log_prob))
                        self.cumulative_solved.add(task.name)
                        newly_solved.append(task.name)
                        print(f"  [{i}/{len(self.tasks)}] {task.name}: SOLVED")
                        break

            total_programs += programs_tried

        wake_time = time.time() - wake_start
        programs_per_second = total_programs / wake_time if wake_time > 0 else 0

        print(f"\nWake complete: {len(self.cumulative_solved)}/{len(self.tasks)} solved")
        print(f"  Enumerated {total_programs:,} programs in {wake_time:.1f}s")
        print(f"  Rate: {programs_per_second:.0f} programs/sec")

        # =====================
        # SLEEP - COMPRESSION
        # =====================
        print("\n[SLEEP] Compression - finding abstractions...")
        comp_start = time.time()

        new_abstractions = []

        # Collect programs from frontiers
        all_programs = []
        for frontier in self.frontiers.values():
            all_programs.extend([p for p, _ in frontier])

        if len(all_programs) >= 2:
            frontiers_for_compression = [
                [(p, 0.0) for p, _ in sols]
                for sols in self.frontiers.values()
                if sols
            ]

            result = compress_frontiers(
                self.grammar,
                frontiers_for_compression,
                max_inventions=self.config.max_inventions_per_iteration,
                min_savings=self.config.min_compression_savings,
                use_anti_unification=True
            )

            if result.new_inventions:
                self.grammar = result.new_grammar

                for inv in result.new_inventions:
                    inv_name = str(inv)
                    self.all_abstractions.append(inv_name)
                    self.recognition.add_invention(inv)
                    self.dreamer.grammar = self.grammar

                    new_abstractions.append(AbstractionLog(
                        iteration=iteration,
                        abstraction_name=inv_name,
                        abstraction_body=str(inv.body),
                        size=inv.body.size(),
                        estimated_savings=0.0  # Could compute from result
                    ))

                    print(f"  New abstraction: {inv_name}")

        comp_time = time.time() - comp_start
        print(f"Compression complete: {len(new_abstractions)} abstractions in {comp_time:.1f}s")

        # =====================
        # SLEEP - RECOGNITION TRAINING
        # =====================
        print("\n[SLEEP] Recognition model training...")
        rec_start = time.time()

        epoch_logs = []
        solved_tasks = [t for t in self.tasks if t.name in self.cumulative_solved]
        solved_frontiers = {
            t.name: type('Frontier', (), {
                'solved': True,
                'entries': [
                    type('Entry', (), {'program': p, 'log_likelihood': 0.0})()
                    for p, _ in self.frontiers[t.name]
                ]
            })()
            for t in solved_tasks
        }

        if solved_tasks:
            # Train with epoch-level logging
            for epoch in range(self.config.recognition_epochs):
                loss = self.recognition.train_on_frontiers(
                    tasks=solved_tasks,
                    frontiers=solved_frontiers,
                    epochs=1,
                    lambda_struct=self.config.structural_similarity_weight
                )

                epoch_logs.append(RecognitionEpochLog(
                    epoch=epoch + 1,
                    total_loss=loss,
                    pred_loss=loss * 0.7,  # Approximate breakdown
                    struct_loss=loss * 0.3 * self.config.structural_similarity_weight,
                    count_loss=0.0,
                    n_tasks=len(solved_tasks)
                ))

        rec_time = time.time() - rec_start
        final_loss = epoch_logs[-1].total_loss if epoch_logs else 0.0
        print(f"Recognition training complete: final loss={final_loss:.4f} in {rec_time:.1f}s")

        # =====================
        # COLLECT PREDICTIONS (Model Collapse Detection)
        # =====================
        print("\n[ANALYSIS] Collecting per-task predictions...")

        task_predictions = []
        entropies = []

        for task in self.tasks:
            probs = self.recognition.predict_primitives(task).cpu().numpy()

            # Top 5 predictions
            top_indices = np.argsort(probs)[-5:][::-1]
            top_5 = [
                (self.recognition.primitive_names[i], float(probs[i]))
                for i in top_indices
            ]

            # Prediction entropy (high = uniform/collapsed)
            probs_clipped = np.clip(probs, 1e-10, 1.0)
            entropy = -np.sum(probs_clipped * np.log(probs_clipped))
            entropies.append(entropy)

            # Predicted count
            pred_count = float(self.recognition.count_head(
                self.recognition.encode_task_batched(task).unsqueeze(0)
            ).item())

            task_predictions.append(TaskPredictionLog(
                task_name=task.name,
                top_5_primitives=top_5,
                prediction_entropy=float(entropy),
                predicted_count=pred_count
            ))

        mean_entropy = float(np.mean(entropies))
        std_entropy = float(np.std(entropies))

        # Save detailed predictions
        with open(self.predictions_dir / f'iter_{iteration:02d}_predictions.json', 'w') as f:
            json.dump([asdict(p) for p in task_predictions], f, indent=2)

        print(f"Prediction entropy: mean={mean_entropy:.4f}, std={std_entropy:.4f}")

        # =====================
        # SLEEP - DREAMING
        # =====================
        print("\n[SLEEP] Dreaming - generating synthetic tasks...")
        dream_start = time.time()

        sample_dream_programs = []
        n_standard = 0
        n_contrastive = 0

        if iteration > 1:  # Only dream after first iteration
            dreams = self.dreamer.generate_dreams(
                request_type=arrow(HAND, BOOL),
                n_dreams=self.config.dreams_per_iteration,
                n_examples_per_dream=10,
                temperature=1.0,
                verbose=False
            )

            for dream in dreams:
                if dream.n_near_miss_pairs > 0:
                    n_contrastive += 1
                else:
                    n_standard += 1

            # Sample programs
            sample_dream_programs = [str(d.program) for d in dreams[:5]]

            # Save dream details
            dream_details = [
                {
                    'program': str(d.program),
                    'n_examples': len(d.task.examples),
                    'n_near_miss': d.n_near_miss_pairs
                }
                for d in dreams
            ]
            with open(self.dreams_dir / f'iter_{iteration:02d}_dreams.json', 'w') as f:
                json.dump(dream_details, f, indent=2)

        dream_time = time.time() - dream_start
        print(f"Dreams: {n_standard} standard, {n_contrastive} contrastive in {dream_time:.1f}s")

        # =====================
        # SAVE MODEL CHECKPOINT
        # =====================
        checkpoint_path = self.checkpoints_dir / f'model_iter_{iteration:02d}.pt'
        self.recognition.save(str(checkpoint_path))

        # Also save task embeddings for clustering analysis
        embeddings = {}
        for task in self.tasks:
            emb = self.recognition.get_task_embedding(task).numpy().tolist()
            embeddings[task.name] = emb

        with open(self.checkpoints_dir / f'embeddings_iter_{iteration:02d}.json', 'w') as f:
            json.dump(embeddings, f, indent=2)

        # =====================
        # COMPILE ITERATION LOG
        # =====================
        return IterationLog(
            iteration=iteration,
            tasks_solved=len(self.cumulative_solved),
            tasks_total=len(self.tasks),
            total_programs_enumerated=total_programs,
            programs_per_second=programs_per_second,
            wake_time_seconds=wake_time,
            newly_solved_this_iter=newly_solved,
            recognition_epochs=epoch_logs,
            final_recognition_loss=final_loss,
            recognition_time_seconds=rec_time,
            task_predictions=task_predictions,
            mean_prediction_entropy=mean_entropy,
            prediction_entropy_std=std_entropy,
            new_abstractions=new_abstractions,
            compression_time_seconds=comp_time,
            grammar_size=len(self.grammar),
            dream_log=DreamLog(
                iteration=iteration,
                n_standard_dreams=n_standard,
                n_contrastive_dreams=n_contrastive,
                avg_examples_per_dream=10.0,
                sample_dream_programs=sample_dream_programs
            ),
            dream_time_seconds=dream_time,
            model_checkpoint_path=str(checkpoint_path)
        )


# ============================================================================
# EXPERIMENT VARIANTS
# ============================================================================

def get_experiment_variants() -> List[ExperimentConfig]:
    """Define the three experiment variants for overnight run."""

    return [
        # Track A: Baseline (full library)
        ExperimentConfig(
            name='baseline_full_library',
            track='A',
            description='Full 59-primitive library as baseline',
            remove_primitives=[],
            add_primitives=[],
            iterations=6,
            enumeration_budget=150_000,
            recognition_epochs=15,
        ),

        # Track A: Minimal (remove gestalt primitives)
        ExperimentConfig(
            name='minimal_no_gestalt',
            track='A',
            description='Remove all gestalt primitives (48 primitives)',
            remove_primitives=GESTALT_PRIMITIVES,
            add_primitives=[],
            iterations=6,
            enumeration_budget=150_000,
            recognition_epochs=15,
        ),

        # Track B: Minimal + Combinators
        ExperimentConfig(
            name='minimal_plus_combinators',
            track='B',
            description='Minimal library + compose, flip, fold, neq',
            remove_primitives=GESTALT_PRIMITIVES,
            add_primitives=['compose', 'flip', 'fold', 'neq'],
            iterations=6,
            enumeration_budget=150_000,
            recognition_epochs=15,
        ),
    ]


def build_variant_grammar(config: ExperimentConfig) -> Grammar:
    """Build grammar for a specific variant."""

    # Start with full primitives
    primitives = build_lean_primitives()

    # Remove specified primitives
    if config.remove_primitives:
        remove_set = set(config.remove_primitives)
        primitives = [p for p in primitives if str(p.program) not in remove_set]

    # Add specified primitives
    if config.add_primitives:
        addition_pool = make_addition_primitives()
        for name in config.add_primitives:
            if name in addition_pool:
                primitives.append(addition_pool[name])

    return uniform_grammar(primitives)


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Overnight Wake-Sleep Library Study'
    )
    parser.add_argument('--variants', type=str, nargs='+',
                        default=['baseline_full_library', 'minimal_no_gestalt', 'minimal_plus_combinators'],
                        help='Which variants to run')
    parser.add_argument('--output-dir', type=str,
                        default='results_overnight_wakesleep',
                        help='Output directory')
    parser.add_argument('--iterations', type=int, default=6,
                        help='Iterations per variant')
    parser.add_argument('--budget', type=int, default=150_000,
                        help='Enumeration budget')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test mode (2 iterations, low budget)')
    parser.add_argument('--two-phase', action='store_true',
                        help='Two-phase transfer learning: 3 iters on pretraining, 3 on catalogue')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show config without running')

    args = parser.parse_args()

    # Get all variants
    all_variants = {v.name: v for v in get_experiment_variants()}

    # Filter to requested variants
    variants = []
    for name in args.variants:
        if name in all_variants:
            variants.append(all_variants[name])
        else:
            print(f"Warning: Unknown variant '{name}'")

    if not variants:
        print("No valid variants specified!")
        return 1

    # Apply quick test settings
    if args.quick_test:
        for v in variants:
            v.iterations = 2  # 2 per phase in two-phase mode, or 2 total in standard mode
            v.enumeration_budget = 50_000
            v.recognition_epochs = 5
            v.dreams_per_iteration = 20
        print("Quick test mode: 2 iterations, 50k budget")

    # Override settings
    for v in variants:
        if args.iterations:
            v.iterations = args.iterations
        if args.budget:
            v.enumeration_budget = args.budget

    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base = Path(args.output_dir) / f'study_{timestamp}'
    output_base.mkdir(parents=True, exist_ok=True)

    # Load tasks
    SCRIPT_DIR = Path(__file__).parent
    SRC_DIR = SCRIPT_DIR.parent
    PRETRAINING_TASKS_PATH = SRC_DIR / 'data' / 'prerecorded_tasks' / 'pretraining_tasks.json'
    CATALOGUE_TASKS_PATH = SRC_DIR / 'data' / 'prerecorded_tasks' / 'catalogue_tasks.json'

    if PRETRAINING_TASKS_PATH.exists():
        pretraining_tasks = load_prerecorded_tasks(PRETRAINING_TASKS_PATH)
        print(f"Loaded {len(pretraining_tasks)} pretraining tasks")
    else:
        print(f"Error: Pretraining tasks file not found: {PRETRAINING_TASKS_PATH}")
        return 1

    if args.two_phase:
        if CATALOGUE_TASKS_PATH.exists():
            catalogue_tasks = load_prerecorded_tasks(CATALOGUE_TASKS_PATH)
            print(f"Loaded {len(catalogue_tasks)} catalogue tasks")
        else:
            print(f"Error: Catalogue tasks file not found: {CATALOGUE_TASKS_PATH}")
            return 1

    # For non-two-phase mode, use pretraining tasks as default
    tasks = pretraining_tasks

    # Dry run
    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Output: {output_base}")
        if args.two_phase:
            print(f"Mode: TWO-PHASE TRANSFER LEARNING")
            print(f"  Phase 1: {len(pretraining_tasks)} pretraining tasks (3 iterations)")
            print(f"  Phase 2: {len(catalogue_tasks)} catalogue tasks (3 iterations)")
        else:
            print(f"Tasks: {len(tasks)}")
        print(f"\nVariants:")
        for v in variants:
            print(f"\n  {v.name}:")
            print(f"    Track: {v.track}")
            print(f"    Description: {v.description}")
            print(f"    Remove: {v.remove_primitives}")
            print(f"    Add: {v.add_primitives}")
            if args.two_phase:
                print(f"    Phase 1 iterations: 3")
                print(f"    Phase 2 iterations: 3")
            else:
                print(f"    Iterations: {v.iterations}")
            print(f"    Budget: {v.enumeration_budget:,}")
        return 0

    # Save experiment config
    config_summary = {
        'timestamp': timestamp,
        'variants': [asdict(v) for v in variants],
        'n_tasks': len(tasks),
        'two_phase': args.two_phase
    }
    with open(output_base / 'experiment_config.json', 'w') as f:
        json.dump(config_summary, f, indent=2)

    # Run experiments
    all_results = {}
    total_start = time.time()

    for i, variant in enumerate(variants, 1):
        print(f"\n{'#'*70}")
        print(f"# VARIANT {i}/{len(variants)}: {variant.name}")
        print(f"{'#'*70}")

        variant_dir = output_base / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)

        # Build grammar for this variant
        grammar = build_variant_grammar(variant)
        print(f"Grammar: {len(grammar)} primitives")

        if args.two_phase:
            # ============================================================
            # TWO-PHASE TRANSFER LEARNING
            # ============================================================
            # Use half of total iterations per phase (or 3 each if 6 total)
            iters_per_phase = max(2, variant.iterations // 2) if args.quick_test else 3

            print("\n" + "="*60)
            print(f"PHASE 1: PRETRAINING ({iters_per_phase} iterations)")
            print("="*60)

            # Create Phase 1 config
            phase1_config = copy.deepcopy(variant)
            phase1_config.iterations = iters_per_phase

            phase1_dir = variant_dir / 'phase1_pretraining'
            phase1_dir.mkdir(parents=True, exist_ok=True)

            learner = LoggingWakeSleep(
                config=phase1_config,
                grammar=grammar,
                tasks=pretraining_tasks,
                output_dir=phase1_dir,
                device='cpu'
            )

            try:
                phase1_result = learner.run()

                # Save transfer state for Phase 2
                transfer_path = variant_dir / 'transfer_state.pt'
                learner.save_transfer_state(transfer_path)

                print("\n" + "="*60)
                print(f"PHASE 2: CATALOGUE TASKS ({iters_per_phase} iterations with transfer)")
                print("="*60)
                print(f"Transferring: {len(learner.all_abstractions)} abstractions")
                print(f"             Model weights from {len(learner.cumulative_solved)} solved tasks")

                # Create Phase 2 config
                phase2_config = copy.deepcopy(variant)
                phase2_config.iterations = iters_per_phase

                phase2_dir = variant_dir / 'phase2_catalogue'
                phase2_dir.mkdir(parents=True, exist_ok=True)

                # Create Phase 2 learner with transferred state
                learner2 = LoggingWakeSleep.from_transfer_state(
                    transfer_path=transfer_path,
                    config=phase2_config,
                    new_tasks=catalogue_tasks,
                    output_dir=phase2_dir,
                    device='cpu'
                )

                phase2_result = learner2.run()

                # Combine results for summary
                combined_result = type('CombinedResult', (), {
                    'cumulative_solved': phase1_result.cumulative_solved + phase2_result.cumulative_solved,
                    'final_solve_rate': (phase1_result.cumulative_solved + phase2_result.cumulative_solved) / (len(pretraining_tasks) + len(catalogue_tasks)),
                    'all_abstractions_learned': phase1_result.all_abstractions_learned + phase2_result.all_abstractions_learned,
                    'phase1': phase1_result,
                    'phase2': phase2_result
                })()

                all_results[variant.name] = combined_result

                # Save combined summary
                combined_summary = {
                    'phase1': {
                        'tasks': len(pretraining_tasks),
                        'solved': phase1_result.cumulative_solved,
                        'solve_rate': phase1_result.final_solve_rate,
                        'abstractions': len(phase1_result.all_abstractions_learned)
                    },
                    'phase2': {
                        'tasks': len(catalogue_tasks),
                        'solved': phase2_result.cumulative_solved,
                        'solve_rate': phase2_result.final_solve_rate,
                        'abstractions': len(phase2_result.all_abstractions_learned)
                    },
                    'combined': {
                        'total_tasks': len(pretraining_tasks) + len(catalogue_tasks),
                        'total_solved': phase1_result.cumulative_solved + phase2_result.cumulative_solved,
                        'total_abstractions': len(phase1_result.all_abstractions_learned) + len(phase2_result.all_abstractions_learned)
                    }
                }
                with open(variant_dir / 'two_phase_summary.json', 'w') as f:
                    json.dump(combined_summary, f, indent=2)

            except Exception as e:
                print(f"ERROR in variant {variant.name}: {e}")
                traceback.print_exc()
                continue

        else:
            # ============================================================
            # STANDARD MODE (single phase)
            # ============================================================
            learner = LoggingWakeSleep(
                config=variant,
                grammar=grammar,
                tasks=tasks,
                output_dir=variant_dir,
                device='cpu'
            )

            try:
                result = learner.run()
                all_results[variant.name] = result
            except Exception as e:
                print(f"ERROR in variant {variant.name}: {e}")
                traceback.print_exc()
                continue

    total_time = time.time() - total_start

    # Generate summary report
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    print(f"Results: {output_base}")

    # Summary table
    print("\n--- Summary ---")
    print(f"{'Variant':<30} {'Solved':<15} {'Rate':<10} {'Abstractions':<15}")
    print("-" * 70)

    for name, result in all_results.items():
        solved = result.cumulative_solved
        rate = f"{100*result.final_solve_rate:.1f}%"
        abstractions = len(result.all_abstractions_learned)
        print(f"{name:<30} {solved:<15} {rate:<10} {abstractions:<15}")

    # Save combined summary
    summary = {
        'total_time_seconds': total_time,
        'variants': {
            name: {
                'solve_rate': r.final_solve_rate,
                'solved': r.cumulative_solved,
                'abstractions': len(r.all_abstractions_learned)
            }
            for name, r in all_results.items()
        }
    }
    with open(output_base / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return 0


if __name__ == '__main__':
    sys.exit(main())
