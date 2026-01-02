#!/usr/bin/env python3
"""
Recognition-Guided Compression Ablation Study
==============================================

This script compares standard MDL-only compression with DreamDecompiler-style
recognition-guided compression (Palmarini et al., ICML 2024).

Ablation Variables:
  1. use_recognition_guided_compression: True/False
  2. corpus_guidance_alpha: 0.5, 0.7, 0.9 (if recognition-guided)

Two-Phase Transfer Learning:
  Phase 1: Pre-training rules (44 rules) - builds initial library
  Phase 2: Main catalogue rules (45 rules) - tests transfer

Variants:
  - baseline_mdl_only: Standard compression (alpha=1.0 equivalent)
  - recognition_guided_070: Recognition-guided with alpha=0.7 (recommended)
  - recognition_guided_050: Recognition-guided with alpha=0.5 (more forward)

Key Metrics:
  - Tasks solved per phase
  - Abstractions learned (quantity and quality)
  - Transfer effectiveness (Phase 2 solve rate improvement)
  - Compression scoring breakdown (backward vs forward contribution)

Usage:
    # Quick test (2 iterations per phase)
    python3 experiments/run_recognition_compression_ablation.py --quick-test

    # Full overnight run (3 iterations per phase)
    nohup caffeinate -d -i -s python3 experiments/run_recognition_compression_ablation.py \\
        > recognition_compression_ablation.out 2>&1 &

    # Dry run (show config without running)
    python3 experiments/run_recognition_compression_ablation.py --dry-run

Runtime Estimates:
    Quick test: ~30-45 minutes (3 variants × 2 phases × 2 iterations)
    Full run: ~8-12 hours (3 variants × 2 phases × 3 iterations)
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

import torch
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.type_system import arrow, BOOL, INT, HAND
from dreamcoder_core.program import Program, Primitive, Application, Abstraction, Index, Invented
from dreamcoder_core.grammar import Grammar, Production, uniform_grammar
from dreamcoder_core.enumeration import TopDownEnumerator, EnumerationResult
from dreamcoder_core.compression import (
    compress_frontiers,
    compress_frontiers_recognition,
    CompressionResult
)
from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.contrastive_dreaming import HybridDreamer
from dreamcoder_core.lean_primitives import build_lean_primitives, build_lean_grammar
from dreamcoder_core.task import Task
from dreamcoder_core.task_generation import load_prerecorded_tasks

from rules.cards import Card, Hand, Suit, Rank, sample_hand


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class AblationConfig:
    """Configuration for a compression ablation variant."""
    name: str
    description: str

    # Key ablation variable
    use_recognition_guided_compression: bool
    corpus_guidance_alpha: float  # 0.7 = mostly backward, 0.5 = balanced

    # Wake-sleep settings
    iterations_per_phase: int = 3
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


# ============================================================================
# LOGGING INFRASTRUCTURE
# ============================================================================

@dataclass
class CompressionAnalysis:
    """Detailed analysis of compression decisions."""
    iteration: int
    n_candidates_evaluated: int
    n_abstractions_added: int

    # If recognition-guided
    recognition_guided: bool
    alpha_used: float

    # Score breakdowns (for recognition-guided runs)
    backward_scores: List[float] = field(default_factory=list)
    forward_scores: List[float] = field(default_factory=list)
    combined_scores: List[float] = field(default_factory=list)
    abstraction_names: List[str] = field(default_factory=list)

    # Abstraction details
    abstractions_added: List[Dict] = field(default_factory=list)


@dataclass
class IterationLog:
    """Complete logging for one wake-sleep iteration."""
    iteration: int
    phase: int  # 1 or 2

    # Wake phase
    tasks_solved: int
    tasks_total: int
    total_programs_enumerated: int
    wake_time_seconds: float
    newly_solved_this_iter: List[str]

    # Compression analysis (key for this ablation)
    compression_analysis: CompressionAnalysis
    compression_time_seconds: float
    grammar_size: int

    # Recognition training
    final_recognition_loss: float
    recognition_time_seconds: float

    # Dreaming
    n_dreams: int
    dream_time_seconds: float


@dataclass
class PhaseResult:
    """Result of one phase (pretraining or catalogue)."""
    phase: int
    task_set: str  # 'pretraining' or 'catalogue'
    n_tasks: int
    tasks_solved: int
    solve_rate: float
    abstractions_learned: List[str]
    iteration_logs: List[IterationLog]
    total_time_seconds: float


@dataclass
class AblationResult:
    """Complete result of one ablation variant."""
    config: AblationConfig
    start_time: str
    end_time: str
    total_duration_seconds: float

    # Phase results
    phase1_result: PhaseResult  # Pretraining
    phase2_result: PhaseResult  # Catalogue

    # Summary metrics
    total_tasks_solved: int
    total_abstractions: int
    transfer_effectiveness: float  # Phase 2 solve rate / Phase 1 solve rate


# ============================================================================
# ABLATION WAKE-SLEEP ENGINE
# ============================================================================

class AblationWakeSleep:
    """
    Wake-sleep learner for compression ablation study.

    Key feature: Switchable compression method (standard vs recognition-guided)
    with detailed logging of compression decisions.
    """

    def __init__(
        self,
        config: AblationConfig,
        grammar: Grammar,
        tasks: List[Task],
        output_dir: Path,
        phase: int = 1,
        device: str = 'cpu'
    ):
        self.config = config
        self.grammar = copy.deepcopy(grammar)
        self.tasks = tasks
        self.output_dir = output_dir
        self.phase = phase
        self.device = device

        # Create output directories
        self.checkpoints_dir = output_dir / 'checkpoints'
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

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

        # Frontiers
        self.frontiers: Dict[str, List[Tuple[Program, float]]] = {
            t.name: [] for t in tasks
        }

        # Tracking
        self.cumulative_solved: Set[str] = set()
        self.all_abstractions: List[str] = []
        self.iteration_logs: List[IterationLog] = []

    def save_transfer_state(self, path: Path) -> None:
        """Save model weights and grammar for transfer to Phase 2."""
        grammar_prods = []
        for p in self.grammar.productions:
            prog_str = str(p.program)
            log_prob = p.log_probability
            if hasattr(p.program, 'tp'):
                type_str = str(p.program.tp)
            elif hasattr(p.program, 'infer'):
                type_str = str(p.program.infer())
            else:
                type_str = "?"
            grammar_prods.append((prog_str, log_prob, type_str))

        state = {
            'model_state_dict': self.recognition.state_dict(),
            'grammar_productions': grammar_prods,
            'all_abstractions': self.all_abstractions,
            'cumulative_solved': list(self.cumulative_solved),
        }
        torch.save(state, path)
        print(f"  Saved transfer state to {path}")

    @classmethod
    def from_transfer_state(
        cls,
        transfer_path: Path,
        config: AblationConfig,
        new_tasks: List[Task],
        output_dir: Path,
        device: str = 'cpu'
    ) -> 'AblationWakeSleep':
        """Create learner initialized from Phase 1 state."""
        state = torch.load(transfer_path, weights_only=False)
        print(f"  Loaded transfer state: {len(state['all_abstractions'])} abstractions, "
              f"{len(state['cumulative_solved'])} tasks solved")

        # Rebuild grammar
        base_grammar = build_lean_grammar()

        # Add learned inventions
        for prod_str, log_prob, type_str in state['grammar_productions']:
            if prod_str.startswith('#'):
                try:
                    inv = Invented.parse(prod_str)
                    if inv not in [p.program for p in base_grammar.productions]:
                        base_grammar = base_grammar.add_invention(inv, log_prob)
                except Exception as e:
                    print(f"    Warning: Could not parse invention {prod_str}: {e}")

        # Create learner
        learner = cls(
            config=config,
            grammar=base_grammar,
            tasks=new_tasks,
            output_dir=output_dir,
            phase=2,
            device=device
        )

        # Load model weights
        try:
            learner.recognition.load_state_dict(state['model_state_dict'])
            print("  Model weights loaded successfully")
        except RuntimeError as e:
            print(f"  Warning: Model weight mismatch: {e}")
            # Partial load
            current_state = learner.recognition.state_dict()
            loaded_state = state['model_state_dict']
            for key in current_state:
                if key in loaded_state and current_state[key].shape == loaded_state[key].shape:
                    current_state[key] = loaded_state[key]
            learner.recognition.load_state_dict(current_state)

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

    def _check_solution(self, program: Program, examples: List[Tuple[Hand, bool]]) -> bool:
        """Check if program solves all examples."""
        for hand, expected in examples:
            if self._eval_program(program, hand) != expected:
                return False
        return True

    def run(self) -> PhaseResult:
        """Run wake-sleep experiment for this phase."""
        start_time = time.time()
        task_set = 'pretraining' if self.phase == 1 else 'catalogue'

        print(f"\n{'='*60}")
        print(f"PHASE {self.phase}: {task_set.upper()} ({len(self.tasks)} tasks)")
        print(f"{'='*60}")
        print(f"Compression: {'Recognition-Guided (alpha={:.1f})'.format(self.config.corpus_guidance_alpha) if self.config.use_recognition_guided_compression else 'Standard MDL-only'}")
        print(f"Iterations: {self.config.iterations_per_phase}")
        print(f"Grammar size: {len(self.grammar)}")

        for iteration in range(1, self.config.iterations_per_phase + 1):
            print(f"\n--- Iteration {iteration}/{self.config.iterations_per_phase} ---")
            iter_log = self._run_iteration(iteration)
            self.iteration_logs.append(iter_log)

            # Save iteration log
            with open(self.output_dir / f'phase{self.phase}_iter{iteration:02d}_log.json', 'w') as f:
                json.dump(asdict(iter_log), f, indent=2, default=str)

        total_time = time.time() - start_time

        return PhaseResult(
            phase=self.phase,
            task_set=task_set,
            n_tasks=len(self.tasks),
            tasks_solved=len(self.cumulative_solved),
            solve_rate=len(self.cumulative_solved) / len(self.tasks),
            abstractions_learned=self.all_abstractions.copy(),
            iteration_logs=self.iteration_logs,
            total_time_seconds=total_time
        )

    def _run_iteration(self, iteration: int) -> IterationLog:
        """Run one complete wake-sleep iteration."""

        # =====================
        # WAKE PHASE
        # =====================
        print("  [WAKE] Enumeration...")
        wake_start = time.time()

        total_programs = 0
        newly_solved = []

        for task in self.tasks:
            if task.name in self.cumulative_solved:
                continue

            # Get recognition guidance (after first iteration)
            if iteration > 1 and self.recognition:
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

                if self._check_solution(program, task.examples):
                    if self._check_solution(program, task.holdout):
                        self.frontiers[task.name].append((program, log_prob))
                        self.cumulative_solved.add(task.name)
                        newly_solved.append(task.name)
                        break

            total_programs += programs_tried

        wake_time = time.time() - wake_start
        print(f"    Solved: {len(self.cumulative_solved)}/{len(self.tasks)} "
              f"(+{len(newly_solved)} new)")

        # =====================
        # SLEEP - COMPRESSION (Key ablation point!)
        # =====================
        print("  [SLEEP] Compression...")
        comp_start = time.time()

        compression_analysis = self._run_compression(iteration)

        comp_time = time.time() - comp_start
        print(f"    Abstractions: +{compression_analysis.n_abstractions_added} "
              f"(grammar: {len(self.grammar)})")

        # =====================
        # SLEEP - RECOGNITION TRAINING
        # =====================
        print("  [SLEEP] Recognition training...")
        rec_start = time.time()

        final_loss = 0.0
        solved_tasks = [t for t in self.tasks if t.name in self.cumulative_solved]

        if solved_tasks:
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

            for epoch in range(self.config.recognition_epochs):
                final_loss = self.recognition.train_on_frontiers(
                    tasks=solved_tasks,
                    frontiers=solved_frontiers,
                    epochs=1,
                    lambda_struct=self.config.structural_similarity_weight
                )

        rec_time = time.time() - rec_start
        print(f"    Loss: {final_loss:.4f}")

        # =====================
        # SLEEP - DREAMING
        # =====================
        print("  [SLEEP] Dreaming...")
        dream_start = time.time()

        n_dreams = 0
        if iteration > 1:
            dreams = self.dreamer.generate_dreams(
                request_type=arrow(HAND, BOOL),
                n_dreams=self.config.dreams_per_iteration,
                n_examples_per_dream=10,
                temperature=1.0,
                verbose=False
            )
            n_dreams = len(dreams)

        dream_time = time.time() - dream_start

        # Save checkpoint
        checkpoint_path = self.checkpoints_dir / f'phase{self.phase}_iter{iteration:02d}.pt'
        self.recognition.save(str(checkpoint_path))

        return IterationLog(
            iteration=iteration,
            phase=self.phase,
            tasks_solved=len(self.cumulative_solved),
            tasks_total=len(self.tasks),
            total_programs_enumerated=total_programs,
            wake_time_seconds=wake_time,
            newly_solved_this_iter=newly_solved,
            compression_analysis=compression_analysis,
            compression_time_seconds=comp_time,
            grammar_size=len(self.grammar),
            final_recognition_loss=final_loss,
            recognition_time_seconds=rec_time,
            n_dreams=n_dreams,
            dream_time_seconds=dream_time
        )

    def _run_compression(self, iteration: int) -> CompressionAnalysis:
        """
        Run compression with detailed logging.

        This is the KEY ABLATION POINT - switches between:
        - Standard MDL-only compression (compress_frontiers)
        - Recognition-guided compression (compress_frontiers_recognition)
        """

        # Collect programs from frontiers
        frontiers_for_compression = [
            [(p, 0.0) for p, _ in sols]
            for sols in self.frontiers.values()
            if sols
        ]

        if len(frontiers_for_compression) < 2:
            return CompressionAnalysis(
                iteration=iteration,
                n_candidates_evaluated=0,
                n_abstractions_added=0,
                recognition_guided=self.config.use_recognition_guided_compression,
                alpha_used=self.config.corpus_guidance_alpha
            )

        # Collect unsolved tasks for forward scoring
        unsolved_tasks = [t for t in self.tasks if t.name not in self.cumulative_solved]

        # Choose compression method
        use_recognition_guidance = (
            self.config.use_recognition_guided_compression
            and self.recognition is not None
            and iteration > 1  # Need at least one training iteration
            and len(unsolved_tasks) > 0  # Need unsolved tasks
        )

        if use_recognition_guidance:
            print(f"    Using recognition-guided compression (alpha={self.config.corpus_guidance_alpha})")
            print(f"    Forward scoring on {len(unsolved_tasks)} unsolved tasks")

            result = compress_frontiers_recognition(
                self.grammar,
                frontiers_for_compression,
                unsolved_tasks=unsolved_tasks,
                recognition_model=self.recognition,
                max_inventions=self.config.max_inventions_per_iteration,
                min_savings=self.config.min_compression_savings,
                use_anti_unification=True,
                alpha=self.config.corpus_guidance_alpha
            )
        else:
            reason = "first iteration" if iteration == 1 else "no unsolved tasks" if len(unsolved_tasks) == 0 else "disabled"
            print(f"    Using standard MDL compression ({reason})")

            result = compress_frontiers(
                self.grammar,
                frontiers_for_compression,
                max_inventions=self.config.max_inventions_per_iteration,
                min_savings=self.config.min_compression_savings,
                use_anti_unification=True
            )

        # Process results
        abstractions_added = []
        if result.new_inventions:
            self.grammar = result.new_grammar

            for inv in result.new_inventions:
                inv_name = str(inv)
                self.all_abstractions.append(inv_name)
                self.recognition.add_invention(inv)
                self.dreamer.grammar = self.grammar

                abstractions_added.append({
                    'name': inv_name,
                    'body': str(inv.body),
                    'size': inv.body.size()
                })

        return CompressionAnalysis(
            iteration=iteration,
            n_candidates_evaluated=len(frontiers_for_compression),
            n_abstractions_added=len(result.new_inventions) if result.new_inventions else 0,
            recognition_guided=use_recognition_guidance,
            alpha_used=self.config.corpus_guidance_alpha if use_recognition_guidance else 1.0,
            abstractions_added=abstractions_added
        )


# ============================================================================
# EXPERIMENT VARIANTS
# ============================================================================

def get_ablation_variants() -> List[AblationConfig]:
    """Define the ablation study variants."""

    return [
        # Baseline: Standard MDL-only compression
        AblationConfig(
            name='baseline_mdl_only',
            description='Standard MDL-only compression (no recognition guidance)',
            use_recognition_guided_compression=False,
            corpus_guidance_alpha=1.0,  # Not used, but for clarity
        ),

        # Recognition-guided with alpha=0.7 (recommended balance)
        AblationConfig(
            name='recognition_guided_070',
            description='Recognition-guided compression with alpha=0.7 (70% backward, 30% forward)',
            use_recognition_guided_compression=True,
            corpus_guidance_alpha=0.7,
        ),

        # Recognition-guided with alpha=0.5 (more forward-looking)
        AblationConfig(
            name='recognition_guided_050',
            description='Recognition-guided compression with alpha=0.5 (50% backward, 50% forward)',
            use_recognition_guided_compression=True,
            corpus_guidance_alpha=0.5,
        ),
    ]


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Recognition-Guided Compression Ablation Study'
    )
    parser.add_argument('--variants', type=str, nargs='+',
                        default=['baseline_mdl_only', 'recognition_guided_070', 'recognition_guided_050'],
                        help='Which variants to run')
    parser.add_argument('--output-dir', type=str,
                        default='results_recognition_compression_ablation',
                        help='Output directory')
    parser.add_argument('--iterations', type=int, default=3,
                        help='Iterations per phase')
    parser.add_argument('--budget', type=int, default=150_000,
                        help='Enumeration budget')
    parser.add_argument('--quick-test', action='store_true',
                        help='Quick test mode (2 iterations, 50k budget)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Show config without running')

    args = parser.parse_args()

    # Get all variants
    all_variants = {v.name: v for v in get_ablation_variants()}

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
            v.iterations_per_phase = 2
            v.enumeration_budget = 50_000
            v.recognition_epochs = 5
            v.dreams_per_iteration = 20
        print("Quick test mode: 2 iterations per phase, 50k budget")

    # Override settings
    for v in variants:
        if args.iterations:
            v.iterations_per_phase = args.iterations
        if args.budget:
            v.enumeration_budget = args.budget

    # Setup output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_base = Path(args.output_dir) / f'ablation_{timestamp}'
    output_base.mkdir(parents=True, exist_ok=True)

    # Load pre-recorded tasks
    SCRIPT_DIR = Path(__file__).parent
    SRC_DIR = SCRIPT_DIR.parent
    PRETRAINING_TASKS_PATH = SRC_DIR / 'data' / 'prerecorded_tasks' / 'pretraining_tasks.json'
    CATALOGUE_TASKS_PATH = SRC_DIR / 'data' / 'prerecorded_tasks' / 'catalogue_tasks.json'

    if not PRETRAINING_TASKS_PATH.exists():
        print(f"Error: Pretraining tasks file not found: {PRETRAINING_TASKS_PATH}")
        return 1

    if not CATALOGUE_TASKS_PATH.exists():
        print(f"Error: Catalogue tasks file not found: {CATALOGUE_TASKS_PATH}")
        return 1

    pretraining_tasks = load_prerecorded_tasks(PRETRAINING_TASKS_PATH)
    catalogue_tasks = load_prerecorded_tasks(CATALOGUE_TASKS_PATH)
    print(f"Loaded tasks: {len(pretraining_tasks)} pretraining, {len(catalogue_tasks)} catalogue")

    # Dry run
    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Output: {output_base}")
        print(f"\nTwo-Phase Transfer Learning:")
        print(f"  Phase 1: {len(pretraining_tasks)} pretraining tasks")
        print(f"  Phase 2: {len(catalogue_tasks)} catalogue tasks")
        print(f"\nVariants ({len(variants)}):")
        for v in variants:
            print(f"\n  {v.name}:")
            print(f"    Recognition-guided: {v.use_recognition_guided_compression}")
            print(f"    Alpha: {v.corpus_guidance_alpha}")
            print(f"    Iterations per phase: {v.iterations_per_phase}")
            print(f"    Budget: {v.enumeration_budget:,}")

        # Estimate runtime
        est_time_per_iter = 30  # minutes per iteration (rough)
        total_iters = len(variants) * 2 * variants[0].iterations_per_phase
        est_total_hours = total_iters * est_time_per_iter / 60
        print(f"\nEstimated runtime: ~{est_total_hours:.1f} hours")
        return 0

    # Save experiment config
    config_summary = {
        'timestamp': timestamp,
        'variants': [asdict(v) for v in variants],
        'n_pretraining_tasks': len(pretraining_tasks),
        'n_catalogue_tasks': len(catalogue_tasks),
        'quick_test': args.quick_test
    }
    with open(output_base / 'experiment_config.json', 'w') as f:
        json.dump(config_summary, f, indent=2)

    print(f"\n{'='*70}")
    print("RECOGNITION-GUIDED COMPRESSION ABLATION STUDY")
    print(f"{'='*70}")
    print(f"Output: {output_base}")
    print(f"Variants: {[v.name for v in variants]}")
    print()

    # Run experiments
    all_results: Dict[str, AblationResult] = {}
    total_start = time.time()

    for i, variant in enumerate(variants, 1):
        print(f"\n{'#'*70}")
        print(f"# VARIANT {i}/{len(variants)}: {variant.name}")
        print(f"# {variant.description}")
        print(f"{'#'*70}")

        variant_dir = output_base / variant.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        variant_start = time.time()

        try:
            # ===== PHASE 1: PRETRAINING =====
            grammar = build_lean_grammar()
            print(f"\nGrammar: {len(grammar)} primitives")

            phase1_dir = variant_dir / 'phase1_pretraining'
            phase1_dir.mkdir(parents=True, exist_ok=True)

            learner1 = AblationWakeSleep(
                config=variant,
                grammar=grammar,
                tasks=pretraining_tasks,
                output_dir=phase1_dir,
                phase=1,
                device='cpu'
            )

            phase1_result = learner1.run()

            # Save transfer state
            transfer_path = variant_dir / 'transfer_state.pt'
            learner1.save_transfer_state(transfer_path)

            # ===== PHASE 2: CATALOGUE =====
            phase2_dir = variant_dir / 'phase2_catalogue'
            phase2_dir.mkdir(parents=True, exist_ok=True)

            learner2 = AblationWakeSleep.from_transfer_state(
                transfer_path=transfer_path,
                config=variant,
                new_tasks=catalogue_tasks,
                output_dir=phase2_dir,
                device='cpu'
            )

            phase2_result = learner2.run()

            # Calculate transfer effectiveness
            transfer_effectiveness = (
                phase2_result.solve_rate / phase1_result.solve_rate
                if phase1_result.solve_rate > 0 else 0.0
            )

            # Compile variant result
            variant_end = time.time()
            result = AblationResult(
                config=variant,
                start_time=datetime.fromtimestamp(variant_start).isoformat(),
                end_time=datetime.fromtimestamp(variant_end).isoformat(),
                total_duration_seconds=variant_end - variant_start,
                phase1_result=phase1_result,
                phase2_result=phase2_result,
                total_tasks_solved=phase1_result.tasks_solved + phase2_result.tasks_solved,
                total_abstractions=len(phase1_result.abstractions_learned) + len(phase2_result.abstractions_learned),
                transfer_effectiveness=transfer_effectiveness
            )

            all_results[variant.name] = result

            # Save variant result
            with open(variant_dir / 'result.json', 'w') as f:
                json.dump(asdict(result), f, indent=2, default=str)

            # Print variant summary
            print(f"\n--- {variant.name} Summary ---")
            print(f"Phase 1: {phase1_result.tasks_solved}/{phase1_result.n_tasks} "
                  f"({100*phase1_result.solve_rate:.1f}%)")
            print(f"Phase 2: {phase2_result.tasks_solved}/{phase2_result.n_tasks} "
                  f"({100*phase2_result.solve_rate:.1f}%)")
            print(f"Total abstractions: {result.total_abstractions}")
            print(f"Transfer effectiveness: {transfer_effectiveness:.2f}")
            print(f"Time: {timedelta(seconds=int(variant_end - variant_start))}")

        except Exception as e:
            print(f"ERROR in variant {variant.name}: {e}")
            traceback.print_exc()
            continue

    total_time = time.time() - total_start

    # ===== FINAL SUMMARY =====
    print(f"\n{'='*70}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*70}")
    print(f"Total time: {timedelta(seconds=int(total_time))}")
    print(f"Results: {output_base}")

    # Summary table
    print(f"\n{'Variant':<25} {'P1 Solved':<12} {'P2 Solved':<12} {'Transfer':<10} {'Abstractions'}")
    print("-" * 75)

    for name, result in all_results.items():
        p1 = f"{result.phase1_result.tasks_solved}/{result.phase1_result.n_tasks}"
        p2 = f"{result.phase2_result.tasks_solved}/{result.phase2_result.n_tasks}"
        transfer = f"{result.transfer_effectiveness:.2f}x"
        abstractions = result.total_abstractions
        print(f"{name:<25} {p1:<12} {p2:<12} {transfer:<10} {abstractions}")

    # Save combined summary
    summary = {
        'total_time_seconds': total_time,
        'variants': {
            name: {
                'phase1_solve_rate': r.phase1_result.solve_rate,
                'phase2_solve_rate': r.phase2_result.solve_rate,
                'transfer_effectiveness': r.transfer_effectiveness,
                'total_abstractions': r.total_abstractions,
                'total_tasks_solved': r.total_tasks_solved
            }
            for name, r in all_results.items()
        }
    }
    with open(output_base / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary saved to: {output_base / 'summary.json'}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
