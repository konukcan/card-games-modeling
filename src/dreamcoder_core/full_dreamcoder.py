#!/usr/bin/env python3
"""
Full DreamCoder Implementation for Card Game Rule Learning

This module implements the complete DreamCoder system as described in Ellis et al. (2021):
1. WAKE phase: Enumerate programs to solve tasks, keeping top-k solutions
2. SLEEP phase - Compression: Extract common patterns into library via anti-unification
3. SLEEP phase - Recognition: Train neural network to predict grammar weights
4. SLEEP phase - Dreaming: Generate synthetic tasks from recognition model

Key features for interpretability:
- Detailed logging of each component's contribution
- Learning curves and transfer effects
- Library evolution tracking
- Recognition model interpretability

Each component can be individually enabled/disabled for ablation studies.
"""

import sys
import os
import math
import time
import random
import json
import copy
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict
import pickle

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.type_system import (
    Type, BaseType, Arrow, ListType, TypeVariable,
    TypeContext, UnificationError,
    BOOL, INT, CARD, SUIT, RANK, HAND, arrow, type_arity
)
from dreamcoder_core.program import (
    Program, Primitive, Application, Abstraction, Index, Invented,
    apply_args
)
from dreamcoder_core.grammar import Grammar, Production, uniform_grammar
from dreamcoder_core.enumeration import (
    enumerate_simple, Frontier, EnumerationResult
)
from dreamcoder_core.compression import (
    compress_frontiers, CompressionResult, find_anti_unified_patterns
)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class Task:
    """A learning task defined by examples."""
    name: str
    request_type: Type
    examples: List[Tuple[Any, Any]]  # [(input, output), ...]

    # Optional metadata
    family: str = ""
    difficulty_level: int = 0

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


@dataclass
class SolutionEntry:
    """A single program solution with metadata."""
    program: Program
    log_probability: float  # Under current grammar
    log_likelihood: float   # How well it fits (0 = perfect)
    programs_enumerated: int  # Search effort to find this
    time_found: float  # When found (seconds from start)

    @property
    def posterior(self) -> float:
        """Joint probability: prior * likelihood."""
        return self.log_probability + self.log_likelihood

    @property
    def description_length(self) -> float:
        """Description length in bits."""
        return -self.log_probability / math.log(2)

    def to_dict(self) -> Dict:
        return {
            'program': str(self.program),
            'log_probability': self.log_probability,
            'log_likelihood': self.log_likelihood,
            'description_length': self.description_length,
            'programs_enumerated': self.programs_enumerated,
            'time_found': self.time_found
        }


@dataclass
class TaskFrontier:
    """
    Top-k solutions for a single task.

    This is crucial for compression - having multiple solutions increases
    the chance of finding common patterns across programs.
    """
    task: Task
    entries: List[SolutionEntry] = field(default_factory=list)
    max_size: int = 10

    # Tracking
    total_programs_searched: int = 0
    total_time: float = 0.0

    def add(self, entry: SolutionEntry) -> bool:
        """
        Add a solution if it improves the frontier.
        Returns True if added.
        """
        # Check for duplicates
        for e in self.entries:
            if str(e.program) == str(entry.program):
                return False

        self.entries.append(entry)

        # Sort by posterior (higher is better = less negative)
        self.entries.sort(key=lambda e: -e.posterior)

        # Keep only top-k
        if len(self.entries) > self.max_size:
            removed = self.entries.pop()
            return entry != removed
        return True

    @property
    def best(self) -> Optional[SolutionEntry]:
        return self.entries[0] if self.entries else None

    @property
    def solved(self) -> bool:
        """Task is solved if we have at least one perfect solution."""
        return any(e.log_likelihood == 0.0 for e in self.entries)

    @property
    def n_solutions(self) -> int:
        return len(self.entries)

    def all_programs(self) -> List[Program]:
        """Get all programs in this frontier."""
        return [e.program for e in self.entries]


@dataclass
class IterationMetrics:
    """Metrics from one wake-sleep iteration."""
    iteration: int

    # Wake phase
    tasks_solved: int
    tasks_total: int
    total_programs_enumerated: int
    wake_time: float

    # Sleep - Compression
    new_abstractions: List[str]  # String representations
    compression_savings: float
    compression_time: float

    # Sleep - Recognition (if enabled)
    recognition_loss: Optional[float] = None
    recognition_time: float = 0.0

    # Sleep - Dreaming (if enabled)
    dreams_generated: int = 0
    dream_time: float = 0.0

    # Grammar state
    grammar_size: int = 0

    @property
    def success_rate(self) -> float:
        return self.tasks_solved / self.tasks_total if self.tasks_total > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            'iteration': self.iteration,
            'tasks_solved': self.tasks_solved,
            'tasks_total': self.tasks_total,
            'success_rate': self.success_rate,
            'programs_enumerated': self.total_programs_enumerated,
            'wake_time': self.wake_time,
            'new_abstractions': self.new_abstractions,
            'compression_savings': self.compression_savings,
            'compression_time': self.compression_time,
            'recognition_loss': self.recognition_loss,
            'recognition_time': self.recognition_time,
            'dreams_generated': self.dreams_generated,
            'dream_time': self.dream_time,
            'grammar_size': self.grammar_size
        }


@dataclass
class TaskMetrics:
    """Learning metrics for a single task across iterations."""
    task_name: str
    task_family: str

    # When/if solved
    solved: bool = False
    iteration_solved: Optional[int] = None
    programs_to_solve: int = 0  # Total programs enumerated before solution
    time_to_solve: float = 0.0

    # Solution details
    best_program: Optional[str] = None
    description_length: float = float('inf')

    # Per-iteration tracking
    programs_per_iteration: List[int] = field(default_factory=list)
    solutions_per_iteration: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# RECOGNITION MODEL
# ============================================================================

class RecognitionModel:
    """
    Neural recognition model that predicts grammar weights given a task.

    In the original DreamCoder, this is a neural network (GRU/MLP).
    Here we implement a simpler version based on feature matching
    that captures the key idea: biasing search toward useful primitives.

    The model learns which primitives tend to be useful for tasks with
    certain characteristics (e.g., tasks involving color comparisons
    tend to use get_color primitive).
    """

    def __init__(self, grammar: Grammar, hidden_dim: int = 64):
        self.grammar = grammar
        self.hidden_dim = hidden_dim

        # Feature-based weights: maps (feature, primitive) -> weight
        # Features are derived from task examples
        self._weights: Dict[Tuple[str, str], float] = defaultdict(float)

        # Counts for normalization
        self._feature_counts: Dict[str, int] = defaultdict(int)
        self._primitive_uses: Dict[str, int] = defaultdict(int)

        # Training history
        self.training_losses: List[float] = []

    def extract_features(self, task: Task) -> Set[str]:
        """
        Extract features from a task's examples.

        Features capture patterns in the inputs/outputs that might
        indicate which primitives are useful.
        """
        features = set()

        features.add(f"family:{task.family}")
        features.add(f"level:{task.difficulty_level}")

        # Analyze examples
        n_pos = sum(1 for _, out in task.examples if out == True)
        n_neg = len(task.examples) - n_pos
        features.add(f"balance:{n_pos}/{n_neg}")

        # Analyze inputs (hands of cards)
        for inp, out in task.examples[:5]:  # Sample a few
            if hasattr(inp, '__iter__') and len(inp) > 0:
                try:
                    # Check for card-specific features
                    first_card = inp[0] if inp else None
                    last_card = inp[-1] if inp else None

                    if first_card and last_card:
                        if hasattr(first_card, 'suit') and hasattr(last_card, 'suit'):
                            if first_card.suit == last_card.suit:
                                features.add("terminals_same_suit")
                            if first_card.color == last_card.color:
                                features.add("terminals_same_color")
                        if hasattr(first_card, 'rank') and hasattr(last_card, 'rank'):
                            if first_card.rank == last_card.rank:
                                features.add("terminals_same_rank")

                    # Check for uniformity
                    if hasattr(inp[0], 'color'):
                        colors = [c.color for c in inp]
                        if len(set(colors)) == 1:
                            features.add("uniform_color")
                        suits = [c.suit for c in inp]
                        if len(set(suits)) == 1:
                            features.add("uniform_suit")
                except:
                    pass

        return features

    def predict_grammar_weights(self, task: Task) -> Grammar:
        """
        Given a task, predict which primitives are likely useful.
        Returns a grammar with adjusted weights.
        """
        features = self.extract_features(task)

        # Start with base log-probabilities
        new_productions = []

        for prod in self.grammar.productions:
            prim_name = str(prod.program)

            # Sum feature contributions
            bonus = 0.0
            for feat in features:
                key = (feat, prim_name)
                if key in self._weights:
                    bonus += self._weights[key]

            # Apply bonus (capped to avoid extreme values)
            bonus = max(-2.0, min(2.0, bonus))
            new_lp = prod.log_probability + bonus

            new_productions.append(Production(prod.program, prod.tp, new_lp))

        return Grammar(new_productions, self.grammar.log_variable).normalize_probabilities()

    def train_on_frontier(self, task: Task, frontier: TaskFrontier):
        """
        Update weights based on which primitives appear in solutions.

        This is a simplified version of recognition training:
        - Extract features from task
        - Count which primitives appear in solutions
        - Increase weights for (feature, primitive) pairs that co-occur
        """
        if not frontier.solved:
            return

        features = self.extract_features(task)

        # Count primitives in solutions
        primitive_counts: Dict[str, int] = defaultdict(int)
        for entry in frontier.entries:
            if entry.log_likelihood == 0.0:  # Perfect solution
                self._count_primitives(entry.program, primitive_counts)

        if not primitive_counts:
            return

        # Update weights
        total_uses = sum(primitive_counts.values())
        for feat in features:
            self._feature_counts[feat] += 1
            for prim, count in primitive_counts.items():
                key = (feat, prim)
                # Simple update rule: increase weight proportionally
                self._weights[key] += 0.1 * (count / total_uses)
                self._primitive_uses[prim] += count

    def _count_primitives(self, program: Program, counts: Dict[str, int]):
        """Count primitive occurrences in a program."""
        if isinstance(program, (Primitive, Invented)):
            counts[str(program)] += 1
        elif isinstance(program, Application):
            self._count_primitives(program.f, counts)
            self._count_primitives(program.x, counts)
        elif isinstance(program, Abstraction):
            self._count_primitives(program.body, counts)

    def get_top_predictions(self, task: Task, n: int = 10) -> List[Tuple[str, float]]:
        """Get top-n predicted primitives for interpretability."""
        features = self.extract_features(task)

        scores = defaultdict(float)
        for prod in self.grammar.productions:
            prim_name = str(prod.program)
            for feat in features:
                key = (feat, prim_name)
                if key in self._weights:
                    scores[prim_name] += self._weights[key]

        sorted_scores = sorted(scores.items(), key=lambda x: -x[1])
        return sorted_scores[:n]


# ============================================================================
# DREAMING
# ============================================================================

class Dreamer:
    """
    Generates synthetic tasks ("dreams") for self-supervised learning.

    Dreams help the recognition model generalize by providing more
    training signal than just the solved tasks.

    Process:
    1. Sample a program from the grammar
    2. Generate input-output examples by running the program
    3. Use these synthetic tasks to train recognition
    """

    def __init__(self, grammar: Grammar, eval_fn: Callable):
        self.grammar = grammar
        self.eval_fn = eval_fn

    def generate_dreams(
        self,
        request_type: Type,
        n_dreams: int,
        example_inputs: List[Any],
        max_programs: int = 1000
    ) -> List[Tuple[Program, List[Tuple[Any, Any]]]]:
        """
        Generate synthetic tasks by sampling programs.

        Args:
            request_type: Type of programs to sample
            n_dreams: Number of dreams to generate
            example_inputs: Pool of inputs to use for examples
            max_programs: Max programs to sample from

        Returns:
            List of (program, examples) pairs
        """
        dreams = []
        programs_seen = set()

        for prog, log_prob in enumerate_simple(self.grammar, request_type, max_depth=6):
            if len(dreams) >= n_dreams:
                break
            if len(programs_seen) >= max_programs:
                break

            prog_str = str(prog)
            if prog_str in programs_seen:
                continue
            programs_seen.add(prog_str)

            # Generate examples
            examples = []
            try:
                fn = prog.evaluate([])
                for inp in example_inputs[:10]:
                    try:
                        out = fn(inp)
                        examples.append((inp, out))
                    except:
                        continue
            except:
                continue

            # Only keep if we got enough valid examples
            if len(examples) >= 5:
                dreams.append((prog, examples))

        return dreams


# ============================================================================
# FULL DREAMCODER
# ============================================================================

class FullDreamCoder:
    """
    Complete DreamCoder implementation with all components.

    Supports ablation via flags:
    - use_compression: Enable library learning
    - use_recognition: Enable neural guidance
    - use_dreaming: Enable dream-based self-supervision
    - keep_top_k: How many solutions to keep per task
    """

    def __init__(
        self,
        grammar: Grammar,
        tasks: List[Task],
        eval_fn: Callable[[Program, Any], Any],

        # Wake settings
        enumeration_budget: int = 100000,
        enumeration_timeout: float = 60.0,
        max_depth: int = 6,

        # Frontier settings
        keep_top_k: int = 5,

        # Component flags
        use_compression: bool = True,
        use_recognition: bool = True,
        use_dreaming: bool = True,

        # Compression settings
        max_inventions_per_iteration: int = 3,
        min_compression_savings: float = 2.0,

        # Recognition settings
        recognition_hidden_dim: int = 64,

        # Dreaming settings
        dreams_per_iteration: int = 50,

        # General
        max_iterations: int = 10,
        verbose: bool = True,
        log_dir: Optional[str] = None
    ):
        self.initial_grammar = grammar
        self.grammar = copy.deepcopy(grammar)
        self.tasks = tasks
        self.eval_fn = eval_fn

        # Wake settings
        self.enumeration_budget = enumeration_budget
        self.enumeration_timeout = enumeration_timeout
        self.max_depth = max_depth

        # Frontier settings
        self.keep_top_k = keep_top_k

        # Component flags
        self.use_compression = use_compression
        self.use_recognition = use_recognition
        self.use_dreaming = use_dreaming

        # Compression settings
        self.max_inventions_per_iteration = max_inventions_per_iteration
        self.min_compression_savings = min_compression_savings

        # Recognition
        if use_recognition:
            self.recognition = RecognitionModel(grammar, recognition_hidden_dim)
        else:
            self.recognition = None

        # Dreaming
        self.dreams_per_iteration = dreams_per_iteration

        # General
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.log_dir = Path(log_dir) if log_dir else None

        # State
        self.frontiers: Dict[str, TaskFrontier] = {
            t.name: TaskFrontier(t, max_size=keep_top_k)
            for t in tasks
        }
        self.iteration_metrics: List[IterationMetrics] = []
        self.task_metrics: Dict[str, TaskMetrics] = {
            t.name: TaskMetrics(t.name, t.family)
            for t in tasks
        }
        self.library_history: List[List[str]] = []  # Abstractions added per iteration

    def log(self, msg: str, level: int = 0):
        """Log message with indentation."""
        if self.verbose:
            indent = "  " * level
            print(f"{indent}{msg}", flush=True)

    def run(self) -> Dict:
        """
        Run the full wake-sleep learning loop.

        Returns:
            Dictionary with all results and metrics
        """
        start_time = time.time()

        self.log("=" * 70)
        self.log("FULL DREAMCODER - WAKE-SLEEP LEARNING")
        self.log("=" * 70)
        self.log(f"Tasks: {len(self.tasks)}")
        self.log(f"Initial grammar: {len(self.grammar)} primitives")
        self.log(f"Components: compression={self.use_compression}, "
                f"recognition={self.use_recognition}, dreaming={self.use_dreaming}")
        self.log(f"Keep top-k: {self.keep_top_k}")
        self.log("")

        for iteration in range(self.max_iterations):
            self.log("=" * 70)
            self.log(f"ITERATION {iteration + 1}/{self.max_iterations}")
            self.log("=" * 70)

            metrics = self._run_iteration(iteration)
            self.iteration_metrics.append(metrics)

            # Log iteration summary
            self.log("")
            self.log(f"Summary:", 1)
            self.log(f"Solved: {metrics.tasks_solved}/{metrics.tasks_total} "
                    f"({100*metrics.success_rate:.1f}%)", 2)
            self.log(f"Programs enumerated: {metrics.total_programs_enumerated:,}", 2)
            self.log(f"New abstractions: {len(metrics.new_abstractions)}", 2)
            for abstr in metrics.new_abstractions:
                self.log(f"  {abstr}", 3)
            self.log(f"Grammar size: {metrics.grammar_size}", 2)

            # Check for early stopping
            if metrics.success_rate == 1.0:
                self.log("\nAll tasks solved! Stopping early.")
                break

        total_time = time.time() - start_time

        # Final summary
        self.log("")
        self.log("=" * 70)
        self.log("FINAL SUMMARY")
        self.log("=" * 70)

        solved_tasks = [t for t in self.tasks if self.task_metrics[t.name].solved]
        unsolved_tasks = [t for t in self.tasks if not self.task_metrics[t.name].solved]

        self.log(f"Total time: {total_time:.1f}s")
        self.log(f"Iterations: {len(self.iteration_metrics)}")
        self.log(f"Solved: {len(solved_tasks)}/{len(self.tasks)}")
        self.log(f"Final grammar: {len(self.grammar)} primitives "
                f"(started with {len(self.initial_grammar)})")

        # Library growth
        total_inventions = sum(len(m.new_abstractions) for m in self.iteration_metrics)
        self.log(f"Total new abstractions: {total_inventions}")

        # Save results
        results = self._compile_results(total_time)
        if self.log_dir:
            self._save_results(results)

        return results

    def _run_iteration(self, iteration: int) -> IterationMetrics:
        """Run one complete wake-sleep iteration."""

        # =====================
        # WAKE PHASE
        # =====================
        self.log("\n[WAKE] Enumerating programs...")
        wake_start = time.time()

        total_programs = 0
        tasks_solved = 0

        for task in self.tasks:
            self.log(f"Task: {task.name}", 1)

            frontier = self.frontiers[task.name]

            # Skip if already have max solutions and all are perfect
            if frontier.n_solutions >= self.keep_top_k and frontier.solved:
                self.log(f"(already solved with {frontier.n_solutions} solutions)", 2)
                tasks_solved += 1
                continue

            # Get task-specific grammar (if using recognition)
            task_grammar = self.grammar
            if self.recognition and iteration > 0:
                task_grammar = self.recognition.predict_grammar_weights(task)
                top_preds = self.recognition.get_top_predictions(task, 5)
                if top_preds:
                    self.log(f"Recognition predictions: {[p[0] for p in top_preds]}", 2)

            # Enumerate
            programs_tried = 0
            solutions_found = 0
            enum_start = time.time()

            for program, log_prob in enumerate_simple(
                task_grammar,
                task.request_type,
                max_depth=self.max_depth
            ):
                programs_tried += 1

                if programs_tried > self.enumeration_budget:
                    break
                if time.time() - enum_start > self.enumeration_timeout:
                    break

                # Evaluate
                try:
                    correct = 0
                    for inp, expected in task.examples:
                        result = self.eval_fn(program, inp)
                        if result == expected:
                            correct += 1

                    if correct == len(task.examples):
                        # Perfect solution!
                        entry = SolutionEntry(
                            program=program,
                            log_probability=log_prob,
                            log_likelihood=0.0,
                            programs_enumerated=programs_tried,
                            time_found=time.time() - enum_start
                        )
                        if frontier.add(entry):
                            solutions_found += 1
                            self.log(f"Found solution #{frontier.n_solutions}: {program}", 2)

                            # Update task metrics
                            tm = self.task_metrics[task.name]
                            if not tm.solved:
                                tm.solved = True
                                tm.iteration_solved = iteration
                                tm.programs_to_solve = programs_tried
                                tm.time_to_solve = entry.time_found
                                tm.best_program = str(program)
                                tm.description_length = entry.description_length

                        # Stop if we have enough solutions
                        if frontier.n_solutions >= self.keep_top_k:
                            break
                except:
                    pass

            frontier.total_programs_searched += programs_tried
            frontier.total_time += time.time() - enum_start
            total_programs += programs_tried

            # Update per-iteration metrics
            tm = self.task_metrics[task.name]
            tm.programs_per_iteration.append(programs_tried)
            tm.solutions_per_iteration.append(frontier.n_solutions)

            if frontier.solved:
                tasks_solved += 1
                self.log(f"=> Solved with {frontier.n_solutions} solution(s), "
                        f"{programs_tried:,} programs", 2)
            else:
                self.log(f"=> Unsolved after {programs_tried:,} programs", 2)

        wake_time = time.time() - wake_start
        self.log(f"\nWake phase: {tasks_solved}/{len(self.tasks)} solved, "
                f"{total_programs:,} programs in {wake_time:.1f}s")

        # =====================
        # SLEEP - COMPRESSION
        # =====================
        new_abstractions = []
        compression_savings = 0.0
        compression_time = 0.0

        if self.use_compression:
            self.log("\n[SLEEP - COMPRESSION] Finding abstractions...")
            comp_start = time.time()

            # Collect all programs from frontiers
            all_frontiers = []
            for frontier in self.frontiers.values():
                if frontier.n_solutions > 0:
                    programs_with_ll = [(e.program, e.log_likelihood) for e in frontier.entries]
                    all_frontiers.append(programs_with_ll)

            if all_frontiers:
                # Run compression
                result = compress_frontiers(
                    self.grammar,
                    all_frontiers,
                    max_inventions=self.max_inventions_per_iteration,
                    min_savings=self.min_compression_savings,
                    use_anti_unification=True
                )

                if result.new_inventions:
                    self.grammar = result.new_grammar
                    new_abstractions = [str(inv) for inv in result.new_inventions]
                    compression_savings = result.total_savings

                    # Update recognition model with new grammar
                    if self.recognition:
                        self.recognition.grammar = self.grammar

                    self.log(f"Found {len(new_abstractions)} new abstraction(s):", 1)
                    for abstr in new_abstractions:
                        self.log(abstr, 2)
                else:
                    self.log("No new abstractions found", 1)

            compression_time = time.time() - comp_start
            self.library_history.append(new_abstractions)

        # =====================
        # SLEEP - RECOGNITION TRAINING
        # =====================
        recognition_loss = None
        recognition_time = 0.0

        if self.use_recognition and self.recognition:
            self.log("\n[SLEEP - RECOGNITION] Training recognition model...")
            rec_start = time.time()

            # Train on all solved tasks
            for task in self.tasks:
                frontier = self.frontiers[task.name]
                if frontier.solved:
                    self.recognition.train_on_frontier(task, frontier)

            recognition_time = time.time() - rec_start
            self.log(f"Trained on {sum(1 for f in self.frontiers.values() if f.solved)} tasks", 1)

        # =====================
        # SLEEP - DREAMING
        # =====================
        dreams_generated = 0
        dream_time = 0.0

        if self.use_dreaming and self.recognition and iteration > 0:
            self.log("\n[SLEEP - DREAMING] Generating synthetic tasks...")
            dream_start = time.time()

            # Collect example inputs from tasks
            all_inputs = []
            for task in self.tasks:
                for inp, _ in task.examples[:3]:
                    all_inputs.append(inp)

            # Generate dreams
            dreamer = Dreamer(self.grammar, self.eval_fn)
            dreams = dreamer.generate_dreams(
                self.tasks[0].request_type,
                self.dreams_per_iteration,
                all_inputs
            )
            dreams_generated = len(dreams)

            # Train recognition on dreams
            for prog, examples in dreams:
                # Create synthetic task
                synthetic_task = Task(
                    name=f"dream_{dreams_generated}",
                    request_type=self.tasks[0].request_type,
                    examples=examples
                )

                # Create frontier with the known solution
                synthetic_frontier = TaskFrontier(synthetic_task, max_size=1)
                entry = SolutionEntry(
                    program=prog,
                    log_probability=self.grammar.program_log_likelihood(
                        prog, synthetic_task.request_type
                    ),
                    log_likelihood=0.0,
                    programs_enumerated=0,
                    time_found=0.0
                )
                synthetic_frontier.add(entry)

                # Train
                self.recognition.train_on_frontier(synthetic_task, synthetic_frontier)

            dream_time = time.time() - dream_start
            self.log(f"Generated {dreams_generated} dreams", 1)

        # =====================
        # GRAMMAR WEIGHT UPDATE
        # =====================
        if tasks_solved > 0:
            # Update grammar weights based on what worked
            all_frontiers = []
            for frontier in self.frontiers.values():
                if frontier.n_solutions > 0:
                    programs_with_ll = [(e.program, e.log_likelihood) for e in frontier.entries]
                    all_frontiers.append(programs_with_ll)

            if all_frontiers:
                self.grammar = self.grammar.inside_outside_update(all_frontiers)

        return IterationMetrics(
            iteration=iteration,
            tasks_solved=tasks_solved,
            tasks_total=len(self.tasks),
            total_programs_enumerated=total_programs,
            wake_time=wake_time,
            new_abstractions=new_abstractions,
            compression_savings=compression_savings,
            compression_time=compression_time,
            recognition_loss=recognition_loss,
            recognition_time=recognition_time,
            dreams_generated=dreams_generated,
            dream_time=dream_time,
            grammar_size=len(self.grammar)
        )

    def _compile_results(self, total_time: float) -> Dict:
        """Compile all results into a dictionary."""
        return {
            'config': {
                'tasks': len(self.tasks),
                'initial_grammar_size': len(self.initial_grammar),
                'enumeration_budget': self.enumeration_budget,
                'enumeration_timeout': self.enumeration_timeout,
                'max_depth': self.max_depth,
                'keep_top_k': self.keep_top_k,
                'use_compression': self.use_compression,
                'use_recognition': self.use_recognition,
                'use_dreaming': self.use_dreaming,
                'max_iterations': self.max_iterations
            },
            'summary': {
                'total_time': total_time,
                'iterations_run': len(self.iteration_metrics),
                'tasks_solved': sum(1 for tm in self.task_metrics.values() if tm.solved),
                'tasks_total': len(self.tasks),
                'final_grammar_size': len(self.grammar),
                'total_abstractions': sum(len(m.new_abstractions) for m in self.iteration_metrics)
            },
            'learning_curve': [m.to_dict() for m in self.iteration_metrics],
            'task_metrics': {name: tm.to_dict() for name, tm in self.task_metrics.items()},
            'library_evolution': self.library_history,
            'final_grammar': [str(p.program) for p in self.grammar.productions]
        }

    def _save_results(self, results: Dict):
        """Save results to log directory."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

        # Save JSON results
        json_path = self.log_dir / f"dreamcoder_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.log(f"\nResults saved to: {json_path}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_tasks_from_rules(rules: List, n_examples: int = 20, seed: int = 42) -> List[Task]:
    """
    Create Task objects from rule definitions.

    Args:
        rules: List of Rule objects from rules.catalogue
        n_examples: Number of examples per task
        seed: Random seed

    Returns:
        List of Task objects
    """
    from rules.cards import sample_hand

    random.seed(seed)
    tasks = []

    for rule in rules:
        # Generate balanced examples
        positives = []
        negatives = []
        target = n_examples // 2

        for _ in range(10000):
            hand = sample_hand(6)
            try:
                label = rule.eval(hand)
                if label and len(positives) < target:
                    positives.append((hand, True))
                elif not label and len(negatives) < target:
                    negatives.append((hand, False))
            except:
                continue

            if len(positives) >= target and len(negatives) >= target:
                break

        examples = positives[:target] + negatives[:target]
        random.shuffle(examples)

        task = Task(
            name=rule.id,
            request_type=arrow(HAND, BOOL),
            examples=examples,
            family=rule.family,
            difficulty_level=rule.level
        )
        tasks.append(task)

    return tasks


def make_eval_fn():
    """Create the evaluation function for card game tasks."""
    def eval_fn(program: Program, hand):
        fn = program.evaluate([])
        return fn(hand)
    return eval_fn


# ============================================================================
# DEMO / TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("FULL DREAMCODER DEMO")
    print("=" * 70)

    # Create simple test
    from dreamcoder_core.card_primitives import build_card_grammar
    from rules.catalogue import RULE_DICT

    # Select a few rules for testing
    test_rule_ids = [
        'Uniform_color',
        'Suits_palindrome',
        'Ends_same_color',
        'Ends_same_suit',
    ]

    rules = [RULE_DICT[rid] for rid in test_rule_ids if rid in RULE_DICT]
    print(f"\nTest rules: {[r.id for r in rules]}")

    # Create tasks
    tasks = create_tasks_from_rules(rules, n_examples=20, seed=42)
    print(f"Created {len(tasks)} tasks")

    # Build grammar
    grammar = build_card_grammar()
    print(f"Grammar: {len(grammar)} primitives")

    # Create evaluation function
    eval_fn = make_eval_fn()

    # Run DreamCoder with all components
    dc = FullDreamCoder(
        grammar=grammar,
        tasks=tasks,
        eval_fn=eval_fn,

        # Faster settings for demo
        enumeration_budget=100000,
        enumeration_timeout=60.0,
        max_depth=8,

        keep_top_k=3,

        use_compression=True,
        use_recognition=True,
        use_dreaming=False,  # Skip dreaming for speed

        max_inventions_per_iteration=2,

        max_iterations=3,
        verbose=True,
        log_dir="results/demo"
    )

    results = dc.run()

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)
    print(f"Final solve rate: {results['summary']['tasks_solved']}/{results['summary']['tasks_total']}")
    print(f"Abstractions learned: {results['summary']['total_abstractions']}")
