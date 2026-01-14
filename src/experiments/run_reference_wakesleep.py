#!/usr/bin/env python3
"""
DreamCoder Wake-Sleep Reference Implementation
==============================================

This is the CANONICAL reference script for running the DreamCoder wake-sleep
learning loop. 


USAGE:
------
    # Quick test (~10-15 minutes)
    python3 run_reference_wakesleep.py --quick --verbose 3

    # Overnight run (~12 hours)
    nohup caffeinate -d -i -s python3 run_reference_wakesleep.py --overnight > ref.out 2>&1 &

    # Full run (~24-48 hours)
    python3 run_reference_wakesleep.py --full

    # Resume from checkpoint
    python3 run_reference_wakesleep.py --resume results/run_20260113_142530/

    # Dry run (show configuration without executing)
    python3 run_reference_wakesleep.py --overnight --dry-run

VERBOSE LEVELS:
---------------
    --verbose 1: Iteration summaries only (default)
    --verbose 2: + Phase progress (wake/compression/recognition/dreaming)
    --verbose 3: + Per-task details and diagnostic info

See ARCHITECTURE.md for detailed explanation of design decisions.
"""

import sys
import os
import json
import math
import time
import argparse
import logging
import copy
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Any, Tuple, Set
from collections import defaultdict

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.grammar import Grammar, Production, uniform_grammar
from dreamcoder_core.primitives import build_primitives
from dreamcoder_core.type_system import arrow, HAND, BOOL
from dreamcoder_core.program import Program, Primitive, Invented
from dreamcoder_core.enumeration import TopDownEnumerator, EnumerationResult, Frontier
from dreamcoder_core.task import Task
from dreamcoder_core.task_generation import load_prerecorded_tasks
from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.compression import compress_frontiers, compress_frontiers_recognition
from dreamcoder_core.contrastive_dreaming import HybridDreamer
from rules.cards import sample_hand

# ============================================================================
# PATHS
# ============================================================================

SRC_DIR = Path(__file__).parent.parent
PRETRAINING_TASKS_PATH = SRC_DIR / 'data' / 'prerecorded_tasks' / 'pretraining_tasks.json'
CATALOGUE_TASKS_PATH = SRC_DIR / 'data' / 'prerecorded_tasks' / 'catalogue_tasks.json'
DEFAULT_RESULTS_DIR = SRC_DIR / 'results_reference'

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class WakeSleepConfig:
    """
    Complete configuration for DreamCoder wake-sleep loop.

    Mode Presets:
    - QUICK (~10-15 min): For sanity checking and debugging
    - OVERNIGHT (~12 hours): For meaningful results on reduced budget
    - FULL (~24-48 hours): For publication-quality comprehensive runs

    Architecture: Sequential CPython with full neural guidance.
    See ARCHITECTURE.md for parallelization options and trade-offs.
    """

    # === Mode ===
    mode: str = "overnight"  # "quick", "overnight", "full"

    # === Wake Phase (Enumeration) ===
    enumeration_budget: int = 500_000      # Max programs per task
    enumeration_timeout: float = 180.0     # Seconds per task
    max_depth: int = 12                    # Program tree depth limit

    # === Sleep - Compression ===
    use_compression: bool = True
    max_inventions_per_iteration: int = 5
    min_compression_savings: float = 2.0   # MDL threshold
    use_recognition_guided_compression: bool = True
    corpus_guidance_alpha: float = 0.7     # Weight for unsolved task fit

    # === Sleep - Recognition ===
    use_recognition: bool = True
    recognition_hidden_dim: int = 32
    recognition_epochs: int = 15
    recognition_lr: float = 1e-3

    # === Sleep - Dreaming ===
    use_dreaming: bool = True
    dreams_per_iteration: int = 50
    contrastive_dream_ratio: float = 0.5

    # === General ===
    max_iterations: int = 4
    keep_top_k: int = 5                    # Solutions per task
    checkpoint_every: int = 1              # Save state every N iterations

    # === Output ===
    results_dir: Path = field(default_factory=lambda: DEFAULT_RESULTS_DIR)
    verbose: int = 1                       # 1=summary, 2=phases, 3=tasks

    # === Resume ===
    resume_from: Optional[str] = None

    def __post_init__(self):
        """Apply mode presets."""
        if self.mode == "quick":
            self.max_iterations = 2
            self.enumeration_budget = 100_000
            self.enumeration_timeout = 60.0
            self.recognition_epochs = 5
            self.dreams_per_iteration = 20
        elif self.mode == "overnight":
            self.max_iterations = 4
            self.enumeration_budget = 500_000
            self.enumeration_timeout = 180.0
            self.recognition_epochs = 15
            self.dreams_per_iteration = 50
        elif self.mode == "full":
            self.max_iterations = 6
            self.enumeration_budget = 1_000_000
            self.enumeration_timeout = 600.0
            self.recognition_epochs = 20
            self.dreams_per_iteration = 100


# ============================================================================
# VERBOSE LOGGER
# ============================================================================

class VerboseLogger:
    """
    Multi-level verbose logging for understanding model dynamics.

    Level 1: Iteration summaries only
        - Tasks solved, success rate, abstractions learned, final loss

    Level 2: + Phase progress and statistics
        - Wake: Tasks solved, programs enumerated, solutions found per task
        - Compression: New abstractions with bodies and savings
        - Recognition: Loss trajectory, epochs trained
        - Dreaming: Standard vs contrastive dreams generated
        - Library: Current grammar size and new primitives added

    Level 3: + Per-task details and model internals
        - Per-task enumeration results and solutions
        - Top primitive predictions per task (neural guidance)
        - Recognition model confidence scores
        - Full abstraction details (all tasks that use them)
        - Grammar evolution (primitives added/removed)
        - Timing breakdown per phase
    """

    def __init__(self, level: int = 1):
        self.level = level
        self._iteration_start = None
        self._phase_start = None
        self._wake_stats = {}  # Accumulate stats during wake phase
        self._library_history = []  # Track grammar evolution

    def iteration_start(self, iteration: int, max_iterations: int):
        """Log iteration start."""
        self._iteration_start = time.time()
        self._wake_stats = {'newly_solved': [], 'solutions_found': 0}
        if self.level >= 1:
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration}/{max_iterations}")
            print(f"{'='*70}")

    def iteration_end(self, metrics: dict):
        """Log iteration summary with enhanced statistics."""
        elapsed = time.time() - self._iteration_start if self._iteration_start else 0

        if self.level >= 1:
            print(f"\n{'─'*70}")
            print(f"ITERATION {metrics['iteration']}/{metrics['max_iterations']} SUMMARY")
            print(f"  Tasks: {metrics['solved']}/{metrics['total']} solved ({metrics['rate']:.1%})")
            print(f"  Library: {metrics['grammar_size']} primitives (+{metrics['abstractions']} new)")
            print(f"  Recognition loss: {metrics['loss']:.4f}")
            print(f"  Time: {elapsed:.1f}s")

        if self.level >= 2:
            # Show newly solved tasks this iteration
            if self._wake_stats.get('newly_solved'):
                print(f"  Newly solved this iteration: {len(self._wake_stats['newly_solved'])}")
                for task_name in self._wake_stats['newly_solved'][:5]:
                    print(f"    + {task_name}")
                if len(self._wake_stats['newly_solved']) > 5:
                    print(f"    ... and {len(self._wake_stats['newly_solved']) - 5} more")

    def phase_start(self, phase: str):
        """Log phase start."""
        self._phase_start = time.time()
        if self.level >= 2:
            print(f"\n[{phase}]")

    def phase_end(self, phase: str, summary: str, extra_stats: dict = None):
        """Log phase completion with optional detailed statistics."""
        elapsed = time.time() - self._phase_start if self._phase_start else 0
        if self.level >= 2:
            print(f"  → {summary} ({elapsed:.1f}s)")

            # Level 2: Show key statistics
            if extra_stats:
                if phase == "WAKE" and self.level >= 2:
                    if 'programs_per_sec' in extra_stats:
                        print(f"    Throughput: {extra_stats['programs_per_sec']:,.0f} programs/sec")
                    if 'avg_programs_per_task' in extra_stats:
                        print(f"    Avg programs/task: {extra_stats['avg_programs_per_task']:,.0f}")

                if phase == "COMPRESSION" and self.level >= 2:
                    if 'total_savings' in extra_stats:
                        print(f"    Total MDL savings: {extra_stats['total_savings']:.2f} nats")

                if phase == "RECOGNITION" and self.level >= 2:
                    if 'loss_trajectory' in extra_stats:
                        traj = extra_stats['loss_trajectory']
                        if len(traj) > 3:
                            print(f"    Loss trajectory: {traj[0]:.3f} → {traj[len(traj)//2]:.3f} → {traj[-1]:.3f}")

        # Level 3: Show timing breakdown
        if self.level >= 3:
            print(f"    Phase duration: {elapsed:.2f}s")

    def task_result(self, task_name: str, solved: bool, programs: int,
                    solution: str = None, was_already_solved: bool = False,
                    n_solutions: int = 1):
        """Log individual task result with solution details."""
        if solved and not was_already_solved:
            self._wake_stats.setdefault('newly_solved', []).append(task_name)
        if solved:
            self._wake_stats['solutions_found'] = self._wake_stats.get('solutions_found', 0) + n_solutions

        if self.level >= 3:
            if solved:
                sol_str = f": {solution[:60]}..." if solution and len(solution) > 60 else f": {solution}" if solution else ""
                status = "[OK]" if not was_already_solved else "[==]"  # [==] means was already solved
                print(f"    {status} {task_name} [{programs:,} prog, {n_solutions} sol]{sol_str}")
            else:
                print(f"    [ ] {task_name} - unsolved after {programs:,} programs")

    def predictions(self, task_name: str, top_preds: List[Tuple[str, float]]):
        """Log top primitive predictions with confidence scores."""
        if self.level >= 3 and top_preds:
            preds_str = ", ".join([f"{p[0][:15]}({p[1]:.2f})" for p in top_preds[:5]])
            print(f"      Top predictions: {preds_str}")

    def abstraction(self, name: str, body: str, savings: float, tasks: int):
        """Log a learned abstraction with full details at level 3."""
        if self.level >= 2:
            body_short = body[:50] + "..." if len(body) > 50 else body
            print(f"    {name}: {body_short}")
            print(f"         Saves {savings:.2f} nats across {tasks} tasks")

        if self.level >= 3:
            # Full body for level 3
            if len(body) > 50:
                print(f"         Full: {body}")

    def library_update(self, old_size: int, new_size: int, new_primitives: List[str]):
        """Log library/grammar evolution."""
        self._library_history.append({
            'old_size': old_size,
            'new_size': new_size,
            'added': new_primitives
        })

        if self.level >= 2:
            print(f"  Library: {old_size} → {new_size} primitives")

        if self.level >= 3 and new_primitives:
            print(f"    New primitives added:")
            for prim in new_primitives[:5]:
                prim_short = prim[:60] + "..." if len(prim) > 60 else prim
                print(f"      + {prim_short}")
            if len(new_primitives) > 5:
                print(f"      ... and {len(new_primitives) - 5} more")

    def recognition_training(self, epoch: int, total_epochs: int, loss: float,
                            accuracy: float = None, top_k_accuracy: float = None):
        """Log recognition model training progress."""
        if self.level >= 3:
            acc_str = f", acc={accuracy:.1%}" if accuracy is not None else ""
            top_k_str = f", top-5={top_k_accuracy:.1%}" if top_k_accuracy is not None else ""
            print(f"    Epoch {epoch}/{total_epochs}: loss={loss:.4f}{acc_str}{top_k_str}")

    def recognition_summary(self, initial_loss: float, final_loss: float,
                           n_tasks: int, avg_confidence: float = None):
        """Log recognition training summary."""
        if self.level >= 2:
            improvement = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0
            print(f"    Training: {initial_loss:.4f} → {final_loss:.4f} ({improvement:.1f}% improvement)")
            print(f"    Trained on {n_tasks} solved tasks")

        if self.level >= 3 and avg_confidence is not None:
            print(f"    Average prediction confidence: {avg_confidence:.3f}")

    def dreaming_details(self, standard_count: int, contrastive_count: int,
                        sample_dreams: List[str] = None):
        """Log dreaming phase details."""
        if self.level >= 2:
            print(f"    Generated: {standard_count} standard + {contrastive_count} contrastive dreams")

        if self.level >= 3 and sample_dreams:
            print(f"    Sample dream programs:")
            for dream in sample_dreams[:3]:
                dream_short = dream[:60] + "..." if len(dream) > 60 else dream
                print(f"      {dream_short}")

    def enumeration_stats(self, total_programs: int, total_time: float,
                         programs_per_task: Dict[str, int] = None):
        """Log enumeration statistics."""
        if self.level >= 2:
            rate = total_programs / total_time if total_time > 0 else 0
            print(f"    Total: {total_programs:,} programs in {total_time:.1f}s ({rate:,.0f} prog/s)")

        if self.level >= 3 and programs_per_task:
            # Show distribution
            counts = list(programs_per_task.values())
            if counts:
                avg = sum(counts) / len(counts)
                min_c, max_c = min(counts), max(counts)
                print(f"    Per-task: avg={avg:,.0f}, min={min_c:,}, max={max_c:,}")

    def info(self, message: str, indent: int = 0, level_required: int = 2):
        """Log informational message at specified level."""
        if self.level >= level_required:
            prefix = "  " * indent
            print(f"{prefix}{message}")


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

@dataclass
class CheckpointState:
    """Complete state for resume capability."""
    iteration: int
    config: dict
    grammar_productions: List[dict]
    model_state_dict: dict
    frontiers: Dict[str, dict]
    metrics_history: List[dict]
    timestamp: str


def save_checkpoint(
    checkpoint_dir: Path,
    iteration: int,
    config: WakeSleepConfig,
    grammar: Grammar,
    recognition: ContrastiveRecognitionModel,
    frontiers: Dict[str, 'TaskFrontier'],
    metrics_history: List[dict]
):
    """Save complete state for resume."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    iter_dir = checkpoint_dir / f"iter_{iteration:02d}"
    iter_dir.mkdir(exist_ok=True)

    # Save model weights
    torch.save(recognition.state_dict(), iter_dir / "model.pt")

    # Save grammar as JSON
    grammar_data = [
        {"program": str(p.program), "type": str(p.tp), "log_prob": p.log_probability}
        for p in grammar.productions
    ]
    with open(iter_dir / "grammar.json", "w") as f:
        json.dump(grammar_data, f, indent=2)

    # Save frontiers
    frontiers_data = {}
    for name, frontier in frontiers.items():
        frontiers_data[name] = {
            "solved": frontier.solved,
            "n_solutions": frontier.n_solutions,
            "programs_searched": frontier.total_programs_searched,
            "solutions": [
                {"program": str(e.program), "log_prob": e.log_probability}
                for e in frontier.entries
            ]
        }
    with open(iter_dir / "frontiers.json", "w") as f:
        json.dump(frontiers_data, f, indent=2)

    # Save metrics history
    with open(iter_dir / "metrics.json", "w") as f:
        json.dump(metrics_history, f, indent=2)

    # Save config
    config_dict = asdict(config)
    config_dict['results_dir'] = str(config_dict['results_dir'])
    with open(iter_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)


def load_checkpoint(checkpoint_dir: Path) -> Optional[dict]:
    """Load checkpoint for resume. Returns None if not found."""
    if not checkpoint_dir.exists():
        return None

    # Find latest iteration checkpoint
    iter_dirs = sorted([d for d in checkpoint_dir.iterdir() if d.name.startswith("iter_")])
    if not iter_dirs:
        return None

    latest_dir = iter_dirs[-1]
    iteration = int(latest_dir.name.split("_")[1])

    try:
        # Load model
        model_path = latest_dir / "model.pt"
        model_state = torch.load(model_path, map_location='cpu') if model_path.exists() else None

        # Load grammar
        with open(latest_dir / "grammar.json") as f:
            grammar_data = json.load(f)

        # Load frontiers
        with open(latest_dir / "frontiers.json") as f:
            frontiers_data = json.load(f)

        # Load metrics
        with open(latest_dir / "metrics.json") as f:
            metrics_history = json.load(f)

        # Load config
        with open(latest_dir / "config.json") as f:
            config_data = json.load(f)

        return {
            "iteration": iteration,
            "model_state": model_state,
            "grammar_data": grammar_data,
            "frontiers_data": frontiers_data,
            "metrics_history": metrics_history,
            "config": config_data
        }
    except Exception as e:
        print(f"Warning: Could not load checkpoint from {latest_dir}: {e}")
        return None


# ============================================================================
# TASK FRONTIER
# ============================================================================

@dataclass
class TaskFrontier:
    """Top-k solutions for a single task."""
    task: Task
    entries: List[EnumerationResult] = field(default_factory=list)
    max_size: int = 5
    total_programs_searched: int = 0
    _seen_hashes: Set[int] = field(default_factory=set)

    def add(self, entry: EnumerationResult) -> bool:
        """Add solution if it improves the frontier."""
        prog_hash = hash(entry.program)
        if prog_hash in self._seen_hashes:
            return False
        self._seen_hashes.add(prog_hash)
        self.entries.append(entry)
        self.entries.sort(key=lambda e: -e.log_probability)
        if len(self.entries) > self.max_size:
            self.entries.pop()
        return True

    @property
    def best(self) -> Optional[EnumerationResult]:
        return self.entries[0] if self.entries else None

    @property
    def solved(self) -> bool:
        return len(self.entries) > 0

    @property
    def n_solutions(self) -> int:
        return len(self.entries)


# ============================================================================
# EVALUATION HELPER
# ============================================================================

def eval_program(program: Program, hand) -> Optional[bool]:
    """Safely evaluate a program on a hand of cards."""
    try:
        fn = program.evaluate([])
        result = fn(hand)
        return result if isinstance(result, bool) else None
    except Exception:
        return None


# ============================================================================
# WAKE-SLEEP LEARNER
# ============================================================================

class ReferenceWakeSleep:
    """
    DreamCoder Wake-Sleep Reference Implementation.

    This class implements the full wake-sleep loop with:
    - Recognition-guided enumeration (wake phase)
    - Compression-based library learning (sleep)
    - Contrastive recognition model training (sleep)
    - Near-miss dream generation (sleep)

    Architecture: Sequential CPython with full neural guidance.
    """

    def __init__(
        self,
        grammar: Grammar,
        tasks: List[Task],
        config: WakeSleepConfig,
        logger: VerboseLogger
    ):
        self.grammar = grammar
        self.tasks = tasks
        self.config = config
        self.logger = logger

        # Initialize frontiers
        self.frontiers: Dict[str, TaskFrontier] = {
            task.name: TaskFrontier(task=task, max_size=config.keep_top_k)
            for task in tasks
        }

        # Initialize recognition model
        self.recognition = ContrastiveRecognitionModel(
            grammar=grammar,
            card_hidden=32,
            card_out=config.recognition_hidden_dim,
            pred_hidden=64,
            learning_rate=config.recognition_lr,
            device='cpu'
        )

        # Initialize dreamer (deferred - needs eval_fn which isn't available at init time)
        # Will be created on first use in _run_dreaming_phase
        self.dreamer = None
        self._dreamer_initialized = not config.use_dreaming  # Skip init if dreaming disabled

        # Metrics history
        self.metrics_history: List[dict] = []

    def run(self) -> dict:
        """
        Execute the full wake-sleep loop.

        Returns:
            Dictionary with final results and metrics history.
        """
        start_time = time.time()

        for iteration in range(1, self.config.max_iterations + 1):
            self.logger.iteration_start(iteration, self.config.max_iterations)

            # ==========================================
            # WAKE PHASE: Recognition-Guided Enumeration
            # ==========================================
            self.logger.phase_start("WAKE")
            wake_metrics = self._run_wake_phase(iteration)
            self.logger.phase_end("WAKE",
                f"{wake_metrics['solved']}/{wake_metrics['total']} solved, "
                f"{wake_metrics['programs']:,} programs",
                extra_stats={
                    'programs_per_sec': wake_metrics.get('programs_per_sec', 0),
                    'avg_programs_per_task': wake_metrics.get('avg_programs_per_task', 0)
                })

            # ==========================================
            # SLEEP - COMPRESSION: Library Learning
            # ==========================================
            if self.config.use_compression:
                self.logger.phase_start("COMPRESSION")
                comp_metrics = self._run_compression_phase(iteration)
                self.logger.phase_end("COMPRESSION",
                    f"{comp_metrics['new_abstractions']} new abstractions, "
                    f"grammar: {comp_metrics['grammar_size']} primitives",
                    extra_stats={'total_savings': comp_metrics.get('total_savings', 0)})
            else:
                comp_metrics = {"new_abstractions": 0, "grammar_size": len(self.grammar.productions), "total_savings": 0}

            # ==========================================
            # SLEEP - RECOGNITION: Neural Training
            # ==========================================
            if self.config.use_recognition:
                self.logger.phase_start("RECOGNITION")
                recog_metrics = self._run_recognition_phase(iteration)
                self.logger.phase_end("RECOGNITION",
                    f"loss: {recog_metrics['initial_loss']:.3f} -> {recog_metrics['final_loss']:.3f}",
                    extra_stats={'loss_trajectory': recog_metrics.get('loss_trajectory', [])})
            else:
                recog_metrics = {"initial_loss": 0, "final_loss": 0, "n_tasks": 0}

            # ==========================================
            # SLEEP - DREAMING: Synthetic Tasks
            # ==========================================
            if self.config.use_dreaming and iteration >= 1:
                self.logger.phase_start("DREAMING")
                dream_metrics = self._run_dreaming_phase(iteration)
                self.logger.phase_end("DREAMING",
                    f"{dream_metrics['standard']} standard + {dream_metrics['contrastive']} contrastive")
            else:
                dream_metrics = {"standard": 0, "contrastive": 0, "total": 0}

            # ==========================================
            # Collect metrics
            # ==========================================
            iteration_metrics = {
                "iteration": iteration,
                "max_iterations": self.config.max_iterations,
                "solved": wake_metrics['solved'],
                "total": wake_metrics['total'],
                "rate": wake_metrics['solved'] / wake_metrics['total'] if wake_metrics['total'] > 0 else 0,
                "programs": wake_metrics['programs'],
                "abstractions": comp_metrics['new_abstractions'],
                "grammar_size": comp_metrics['grammar_size'],
                "loss": recog_metrics['final_loss'],
                "dreams": dream_metrics['standard'] + dream_metrics['contrastive'],
            }
            self.metrics_history.append(iteration_metrics)
            self.logger.iteration_end(iteration_metrics)

            # ==========================================
            # Checkpoint
            # ==========================================
            if iteration % self.config.checkpoint_every == 0:
                checkpoint_dir = self.config.results_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    iteration=iteration,
                    config=self.config,
                    grammar=self.grammar,
                    recognition=self.recognition,
                    frontiers=self.frontiers,
                    metrics_history=self.metrics_history
                )
                self.logger.info(f"Checkpoint saved to {checkpoint_dir}")

            # Early termination if all solved
            if wake_metrics['solved'] == wake_metrics['total']:
                self.logger.info(f"All tasks solved! Terminating early.")
                break

        # Final summary
        total_time = time.time() - start_time
        solved_count = sum(1 for f in self.frontiers.values() if f.solved)

        return {
            "summary": {
                "total_time": total_time,
                "iterations_run": len(self.metrics_history),
                "tasks_solved": solved_count,
                "tasks_total": len(self.tasks),
                "final_grammar_size": len(self.grammar.productions),
            },
            "metrics_history": self.metrics_history,
            "frontiers": {
                name: {
                    "solved": f.solved,
                    "n_solutions": f.n_solutions,
                    "best_program": str(f.best.program) if f.best else None,
                }
                for name, f in self.frontiers.items()
            }
        }

    def _run_wake_phase(self, iteration: int) -> dict:
        """
        WAKE PHASE: Recognition-guided enumeration.

        For each unsolved task:
        1. Get task-specific grammar weights from recognition model
        2. Enumerate programs using memoized best-first search
        3. Evaluate on examples, verify correctness
        4. Add solutions to frontier
        """
        total_programs = 0
        solved_count = 0
        wake_start = time.time()
        programs_per_task = {}

        for task in self.tasks:
            frontier = self.frontiers[task.name]
            was_already_solved = frontier.solved

            # Skip if already solved with max solutions
            if frontier.n_solutions >= self.config.keep_top_k:
                solved_count += 1
                self.logger.task_result(
                    task.name, True, 0,
                    str(frontier.best.program) if frontier.best else None,
                    was_already_solved=True,
                    n_solutions=frontier.n_solutions
                )
                continue

            # Get task-specific grammar weights (if using recognition)
            if self.config.use_recognition and iteration > 1:
                self.recognition.grammar = self.grammar
                task_grammar = self.recognition.predict_grammar_weights(task)
                top_preds = self.recognition.get_top_predictions(task, n=5)
                self.logger.predictions(task.name, top_preds)
            else:
                task_grammar = self.grammar

            # Enumerate using TopDownEnumerator with MEMOIZATION
            programs_tried = 0
            enum_start = time.time()
            enumerator = TopDownEnumerator(
                task_grammar,
                max_depth=self.config.max_depth,
                max_programs=self.config.enumeration_budget
            )

            # Use enumerate_memoized() for 1000x+ speedup
            for program, log_prob in enumerator.enumerate_memoized(
                task.request_type,
                max_cost=50.0,
                timeout_seconds=self.config.enumeration_timeout,
                depth_limit=self.config.max_depth
            ):
                programs_tried += 1

                if programs_tried > self.config.enumeration_budget:
                    break
                if time.time() - enum_start > self.config.enumeration_timeout:
                    break

                # Evaluate on training examples
                try:
                    training_correct = sum(
                        1 for inp, expected in task.examples
                        if eval_program(program, inp) == expected
                    )

                    if training_correct == len(task.examples):
                        # CRITICAL: Also verify on holdout examples
                        # This prevents spurious solutions that memorize training data
                        holdout_correct = 0
                        if task.holdout:
                            holdout_correct = sum(
                                1 for inp, expected in task.holdout
                                if eval_program(program, inp) == expected
                            )
                            passes_holdout = (holdout_correct == len(task.holdout))
                        else:
                            passes_holdout = True  # No holdout = skip verification

                        if passes_holdout:
                            entry = EnumerationResult(
                                program=program,
                                log_probability=log_prob,
                                log_likelihood=0.0,
                                description_length=-log_prob / math.log(2),
                                programs_enumerated=programs_tried,
                                time_seconds=time.time() - enum_start
                            )
                            frontier.add(entry)

                            if frontier.n_solutions >= self.config.keep_top_k:
                                break
                        # else: program passed training but failed holdout (spurious)
                except Exception:
                    pass

            frontier.total_programs_searched += programs_tried
            total_programs += programs_tried
            programs_per_task[task.name] = programs_tried

            if frontier.solved:
                solved_count += 1
                self.logger.task_result(
                    task.name, True, programs_tried,
                    str(frontier.best.program) if frontier.best else None,
                    was_already_solved=was_already_solved,
                    n_solutions=frontier.n_solutions
                )
            else:
                self.logger.task_result(task.name, False, programs_tried)

        # Log enumeration statistics
        wake_time = time.time() - wake_start
        self.logger.enumeration_stats(total_programs, wake_time, programs_per_task)

        return {
            "solved": solved_count,
            "total": len(self.tasks),
            "programs": total_programs,
            "time": wake_time,
            "programs_per_sec": total_programs / wake_time if wake_time > 0 else 0,
            "avg_programs_per_task": total_programs / len(self.tasks) if self.tasks else 0
        }

    def _run_compression_phase(self, iteration: int) -> dict:
        """
        SLEEP - COMPRESSION: Library learning via abstraction.

        1. Collect all programs from solved frontiers
        2. Find common patterns via anti-unification
        3. Apply quality filters (nontrivial, eta-reduction, single-task)
        4. Score by MDL compression + forward-looking (unsolved task fit)
        5. Add top-k inventions to grammar
        """
        old_grammar_size = len(self.grammar.productions)

        # Collect all programs from frontiers
        all_frontiers = []
        total_solutions = 0
        for frontier in self.frontiers.values():
            if frontier.n_solutions > 0:
                programs_with_ll = [(e.program, e.log_likelihood) for e in frontier.entries]
                all_frontiers.append(programs_with_ll)
                total_solutions += frontier.n_solutions

        if not all_frontiers:
            return {"new_abstractions": 0, "grammar_size": len(self.grammar.productions), "total_savings": 0}

        self.logger.info(f"Compressing {total_solutions} solutions from {len(all_frontiers)} tasks", 1)

        # Collect unsolved tasks for forward-looking scoring
        unsolved_tasks = [f.task for f in self.frontiers.values() if not f.solved]
        self.logger.info(f"Forward-looking scoring using {len(unsolved_tasks)} unsolved tasks", 1)

        # Use recognition-guided compression if enabled
        use_recognition_guidance = (
            self.config.use_recognition_guided_compression
            and self.recognition is not None
            and iteration > 1
            and len(unsolved_tasks) > 0
        )

        if use_recognition_guidance:
            self.logger.info(f"Using recognition-guided compression (alpha={self.config.corpus_guidance_alpha})", 1)
            result = compress_frontiers_recognition(
                self.grammar,
                all_frontiers,
                unsolved_tasks=unsolved_tasks,
                recognition_model=self.recognition,
                max_inventions=self.config.max_inventions_per_iteration,
                min_savings=self.config.min_compression_savings,
                use_anti_unification=True,
                alpha=self.config.corpus_guidance_alpha
            )
        else:
            self.logger.info("Using standard compression (no recognition guidance)", 1)
            result = compress_frontiers(
                self.grammar,
                all_frontiers,
                max_inventions=self.config.max_inventions_per_iteration,
                min_savings=self.config.min_compression_savings,
                use_anti_unification=True
            )

        # Update grammar and recognition model if new inventions found
        new_primitives = []
        if result.new_inventions:
            self.grammar = result.new_grammar

            # Update recognition model with new grammar (expands vocabulary)
            self.recognition.update_grammar(self.grammar)

            # Log each new invention
            for idx, inv in enumerate(result.new_inventions, 1):
                inv_str = str(inv)
                new_primitives.append(inv_str)
                self.logger.abstraction(
                    f"#{idx}",
                    inv_str,
                    result.total_savings / len(result.new_inventions),
                    len(all_frontiers)
                )

        # Log library evolution
        self.logger.library_update(old_grammar_size, len(self.grammar.productions), new_primitives)

        return {
            "new_abstractions": len(result.new_inventions),
            "grammar_size": len(self.grammar.productions),
            "total_savings": result.total_savings
        }

    def _run_recognition_phase(self, iteration: int) -> dict:
        """
        SLEEP - RECOGNITION: Train contrastive recognition model.

        1. Encode tasks contrastively: tau = mean(positives) - mean(negatives)
        2. Train to predict which primitives are useful for each task
        3. Add structural similarity loss (cluster similar tasks)
        """
        # Collect solved tasks for training
        solved_frontiers = {
            name: frontier for name, frontier in self.frontiers.items()
            if frontier.solved
        }

        if not solved_frontiers:
            return {"initial_loss": 0, "final_loss": 0, "epochs": 0, "n_tasks": 0}

        # Update recognition grammar
        self.recognition.update_grammar(self.grammar)

        # Prepare data in the format expected by train_on_frontiers:
        # tasks: List[Task], frontiers: Dict[task_name -> Frontier object with .solved, .entries]
        tasks = [f.task for f in solved_frontiers.values()]

        # The recognition model expects Frontier objects with .solved and .entries attributes
        # Our TaskFrontier class has these, so we can pass them directly
        frontiers_dict = {name: f for name, f in solved_frontiers.items()}

        # Train for configured epochs (train_on_frontiers handles multiple epochs internally)
        initial_loss = None
        final_loss = 0
        loss_trajectory = []

        for epoch in range(self.config.recognition_epochs):
            epoch_loss = self.recognition.train_on_frontiers(
                tasks=tasks,
                frontiers=frontiers_dict,
                epochs=1,  # One epoch at a time for tracking
                batch_size=min(8, len(tasks))
            )
            loss_trajectory.append(epoch_loss)

            if initial_loss is None:
                initial_loss = epoch_loss
            final_loss = epoch_loss

            # Log per-epoch progress at level 3
            self.logger.recognition_training(
                epoch + 1,
                self.config.recognition_epochs,
                epoch_loss
            )

        # Log summary at level 2
        self.logger.recognition_summary(
            initial_loss or 0,
            final_loss,
            len(solved_frontiers)
        )

        return {
            "initial_loss": initial_loss or 0,
            "final_loss": final_loss,
            "epochs": self.config.recognition_epochs,
            "n_tasks": len(solved_frontiers),
            "loss_trajectory": loss_trajectory
        }

    def _run_dreaming_phase(self, iteration: int) -> dict:
        """
        SLEEP - DREAMING: Generate synthetic tasks.

        1. Sample programs from current grammar
        2. Generate positive/negative examples
        3. Include near-miss contrastive dreams (differ by one card)
        4. Train recognition on dreams for better generalization
        """
        if not self.config.use_dreaming:
            return {"standard": 0, "contrastive": 0, "total": 0}

        # Initialize dreamer on first use (needs current grammar)
        if self.dreamer is None:
            # Create a generic eval function for dreams
            def dream_eval_fn(program, hand):
                try:
                    fn = program.evaluate([])
                    return fn(hand)
                except Exception:
                    return None

            self.dreamer = HybridDreamer(
                grammar=self.grammar,
                eval_fn=dream_eval_fn,
                sample_hand_fn=lambda: sample_hand(6),
                sample_card_fn=lambda: sample_hand(1)[0],
                contrastive_ratio=self.config.contrastive_dream_ratio,
                device='cpu'
            )
        else:
            # Update dreamer grammar for new iteration
            self.dreamer.grammar = self.grammar

        # Generate dreams
        n_contrastive = int(self.config.dreams_per_iteration * self.config.contrastive_dream_ratio)
        n_standard = self.config.dreams_per_iteration - n_contrastive

        try:
            dreams = self.dreamer.generate_dreams(
                request_type=arrow(HAND, BOOL),
                n_dreams=self.config.dreams_per_iteration,
                verbose=(self.config.verbose >= 3)
            )
        except Exception as e:
            self.logger.info(f"Dream generation failed: {e}", 1)
            return {"standard": 0, "contrastive": 0, "total": 0, "error": str(e)}

        # Collect sample dream programs for logging
        sample_dreams = []
        if dreams:
            for dream in dreams[:5]:  # Collect first 5 for logging
                if hasattr(dream, 'program'):
                    sample_dreams.append(str(dream.program))

        # Train recognition on dreams by converting to frontier format
        dreams_trained = 0
        if dreams:
            # Convert dreams to frontier format for train_on_frontiers()
            dream_tasks = []
            dream_frontiers = {}

            for i, dream in enumerate(dreams):
                if hasattr(dream, 'task') and hasattr(dream, 'program'):
                    task = dream.task
                    # Create a minimal frontier-like object
                    entry = EnumerationResult(
                        program=dream.program,
                        log_probability=0.0,
                        log_likelihood=0.0,
                        description_length=1.0,
                        programs_enumerated=1,
                        time_seconds=0.0
                    )
                    frontier = TaskFrontier(task=task, max_size=1)
                    frontier.add(entry)

                    dream_tasks.append(task)
                    dream_frontiers[task.name] = frontier
                    dreams_trained += 1

            # Train if we have valid dreams
            if dream_tasks:
                try:
                    self.recognition.train_on_frontiers(
                        tasks=dream_tasks,
                        frontiers=dream_frontiers,
                        epochs=1,
                        batch_size=min(8, len(dream_tasks))
                    )
                except Exception as e:
                    self.logger.info(f"Dream training failed: {e}", 2)

        # Log dreaming details
        self.logger.dreaming_details(n_standard, n_contrastive, sample_dreams)
        self.logger.info(f"Trained recognition on {dreams_trained} dreams", 1)

        return {
            "standard": n_standard,
            "contrastive": n_contrastive,
            "total": len(dreams) if dreams else 0,
            "trained": dreams_trained
        }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="DreamCoder Wake-Sleep Reference Implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument('--quick', action='store_true',
                           help='Quick test (~10-15 min, 2 iterations)')
    mode_group.add_argument('--overnight', action='store_true',
                           help='Overnight run (~12 hours, 4 iterations)')
    mode_group.add_argument('--full', action='store_true',
                           help='Full run (~24-48 hours, 6 iterations)')

    # Verbosity
    parser.add_argument('--verbose', '-v', type=int, default=1, choices=[1, 2, 3],
                       help='Verbosity level: 1=summary, 2=phases, 3=tasks')

    # Resume
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint directory')

    # Dry run
    parser.add_argument('--dry-run', action='store_true',
                       help='Show configuration without executing')

    # Task selection
    parser.add_argument('--tasks', type=str, default='pretraining',
                       choices=['pretraining', 'catalogue', 'both'],
                       help='Which task set to use')

    # Output
    parser.add_argument('--results-dir', type=str, default=None,
                       help='Override results directory')

    args = parser.parse_args()

    # Determine mode
    mode = "quick" if args.quick else "full" if args.full else "overnight"

    # Build config
    config = WakeSleepConfig(
        mode=mode,
        verbose=args.verbose,
        resume_from=args.resume
    )
    if args.results_dir:
        config.results_dir = Path(args.results_dir)

    # Dry run
    if args.dry_run:
        print(f"\nDRY RUN - Configuration ({mode} mode):")
        print(f"  Iterations: {config.max_iterations}")
        print(f"  Budget: {config.enumeration_budget:,} programs/task")
        print(f"  Timeout: {config.enumeration_timeout}s/task")
        print(f"  Recognition epochs: {config.recognition_epochs}")
        print(f"  Dreams: {config.dreams_per_iteration}/iteration")
        print(f"  Verbose level: {config.verbose}")
        print(f"  Tasks: {args.tasks}")
        if args.resume:
            print(f"  Resume from: {args.resume}")
        return

    # Initialize logger
    logger = VerboseLogger(level=config.verbose)

    # Load tasks
    print(f"\nLoading tasks ({args.tasks})...")
    if args.tasks == 'pretraining':
        tasks = load_prerecorded_tasks(PRETRAINING_TASKS_PATH)
    elif args.tasks == 'catalogue':
        tasks = load_prerecorded_tasks(CATALOGUE_TASKS_PATH)
    else:
        tasks = (load_prerecorded_tasks(PRETRAINING_TASKS_PATH) +
                 load_prerecorded_tasks(CATALOGUE_TASKS_PATH))
    print(f"Loaded {len(tasks)} tasks")

    # Build grammar
    primitives = build_primitives()
    grammar = uniform_grammar(primitives)
    print(f"Grammar: {len(grammar.productions)} primitives")

    # Create learner
    learner = ReferenceWakeSleep(
        grammar=grammar,
        tasks=tasks,
        config=config,
        logger=logger
    )

    # Handle resume
    if args.resume:
        checkpoint = load_checkpoint(Path(args.resume))
        if checkpoint:
            print(f"Resuming from iteration {checkpoint['iteration']}")
            learner.recognition.load_state_dict(checkpoint['model_state'])
            learner.metrics_history = checkpoint['metrics_history']
            # Note: Would need to restore grammar and frontiers too for full resume

    # Run
    print(f"\nStarting wake-sleep loop ({mode} mode)...")
    print(f"{'='*70}")
    results = learner.run()

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"Total time: {results['summary']['total_time']:.1f}s")
    print(f"Iterations: {results['summary']['iterations_run']}")
    print(f"Solved: {results['summary']['tasks_solved']}/{results['summary']['tasks_total']}")
    print(f"Final grammar: {results['summary']['final_grammar_size']} primitives")

    # Save final results
    results_dir = config.results_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "final_results.json", "w") as f:
        # Convert non-serializable items
        results_json = copy.deepcopy(results)
        json.dump(results_json, f, indent=2, default=str)
    print(f"\nResults saved to: {results_dir}")


if __name__ == '__main__':
    main()
