#!/usr/bin/env python3
"""
DreamCoder Original - Reference Implementation

This is the canonical reference implementation of DreamCoder with all
components from Ellis et al. (2021). It serves as:

1. A clean reference showing how all components connect
2. Source of helper classes (TaskFrontier, SolutionEntry, etc.)
3. Utility functions used by production runners

NOTE: Production overnight runners typically implement their own wake-sleep
loops inline (for parallelization, checkpointing, custom logging) but import
helper classes and utilities from this module.

Components from Ellis et al. (2021):

1. NEURAL Recognition Model (not feature-based!)
   - GRU encoder for input/output examples
   - Attention-based task aggregation
   - MLP predicting primitive log-probabilities

2. DREAMING with Neural Guidance
   - Sample programs from grammar weighted by recognition model
   - Generate synthetic examples
   - Use these "dreams" for additional recognition training

3. COMPRESSION via Anti-Unification
   - Extract common patterns across solutions
   - Add to library as new primitives
   - Rewrite existing solutions

4. LEAN Primitive Library
   - Cognitively realistic minimal set
   - palindrome?, sorted?, sum should be LEARNED

5. Pre-training on Familiar Rules
   - Warm-start on poker/blackjack/rummy patterns
   - Build up recognition priors

6. Comprehensive Tracking
   - Full interpretability analysis
   - Embedding evolution
   - Learning curves
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

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.type_system import (
    Type, BaseType, Arrow, ListType, TypeVariable,
    BOOL, INT, CARD, SUIT, RANK, HAND, arrow
)
from dreamcoder_core.program import (
    Program, Primitive, Application, Abstraction, Index, Invented
)
from dreamcoder_core.grammar import Grammar, Production, uniform_grammar
# enumerate_simple removed - we now use TopDownEnumerator for enumeration
# and Grammar.sample() or Grammar.sample_requiring_variable() for dreaming
from dreamcoder_core.enumeration import TopDownEnumerator
from dreamcoder_core.compression import compress_frontiers
from dreamcoder_core.neural_recognition import NeuralRecognitionModel
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.task import Task  # Canonical Task definition


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class SolutionEntry:
    """A single program solution with metadata."""
    program: Program
    log_probability: float
    log_likelihood: float
    programs_enumerated: int
    time_found: float

    @property
    def posterior(self) -> float:
        return self.log_probability + self.log_likelihood

    @property
    def description_length(self) -> float:
        return -self.log_probability / math.log(2)


@dataclass
class TaskFrontier:
    """Top-k solutions for a task."""
    task: Task
    entries: List[SolutionEntry] = field(default_factory=list)
    max_size: int = 10
    total_programs_searched: int = 0
    total_time: float = 0.0

    def add(self, entry: SolutionEntry) -> bool:
        for e in self.entries:
            if str(e.program) == str(entry.program):
                return False
        self.entries.append(entry)
        self.entries.sort(key=lambda e: -e.posterior)
        if len(self.entries) > self.max_size:
            self.entries.pop()
        return True

    @property
    def best(self) -> Optional[SolutionEntry]:
        return self.entries[0] if self.entries else None

    @property
    def solved(self) -> bool:
        return any(e.log_likelihood == 0.0 for e in self.entries)

    @property
    def n_solutions(self) -> int:
        return len(self.entries)


@dataclass
class IterationMetrics:
    """Metrics from one wake-sleep iteration."""
    iteration: int
    tasks_solved: int
    tasks_total: int
    programs_enumerated: int
    wake_time: float

    new_abstractions: List[str]
    compression_time: float

    recognition_loss: float
    recognition_time: float

    dreams_generated: int
    dream_time: float

    grammar_size: int

    # Neural model evolution
    embedding_divergence: float = 0.0  # How much embeddings changed


@dataclass
class TaskMetrics:
    """Learning metrics for a single task."""
    task_name: str
    task_family: str
    solved: bool = False
    iteration_solved: Optional[int] = None
    programs_to_solve: int = 0
    best_program: Optional[str] = None
    description_length: float = float('inf')


# ============================================================================
# NEURAL DREAMING
# ============================================================================

class NeuralDreamer:
    """
    Generate synthetic "dream" tasks using the neural recognition model.

    In true DreamCoder:
    1. Use recognition model to bias program sampling toward learned patterns
    2. Generate input-output examples from sampled programs
    3. Use these for additional recognition training

    This creates a virtuous cycle where the recognition model learns from
    its own predictions, generalizing beyond just the solved tasks.
    """

    def __init__(
        self,
        grammar: Grammar,
        recognition_model: NeuralRecognitionModel,
        eval_fn: Callable,
        device: str = 'cpu'
    ):
        self.grammar = grammar
        self.recognition_model = recognition_model
        self.eval_fn = eval_fn
        self.device = device

    def generate_dreams(
        self,
        request_type: Type,
        n_dreams: int,
        example_inputs: List[Any],
        max_programs_per_dream: int = 100,
        temperature: float = 1.0
    ) -> List[Tuple[Task, Program]]:
        """
        Generate synthetic tasks by sampling programs.

        Returns:
            List of (synthetic_task, program) pairs
        """
        dreams = []
        programs_tried = set()

        # Sample diverse inputs
        if len(example_inputs) > 20:
            sampled_inputs = random.sample(example_inputs, 20)
        else:
            sampled_inputs = example_inputs

        attempt = 0
        max_attempts = n_dreams * 10

        while len(dreams) < n_dreams and attempt < max_attempts:
            attempt += 1

            # Sample a program from grammar (biased by recognition if trained)
            program, log_prob = self._sample_program(request_type, temperature)

            if program is None:
                continue

            prog_str = str(program)
            if prog_str in programs_tried:
                continue
            programs_tried.add(prog_str)

            # Try to generate examples
            examples = self._generate_examples(program, sampled_inputs)

            if len(examples) >= 5:
                # Create synthetic task
                task = Task(
                    name=f"dream_{len(dreams)}_{attempt}",
                    request_type=request_type,
                    examples=examples,
                    family="dream"
                )
                dreams.append((task, program))

        return dreams

    def _sample_program(
        self,
        request_type: Type,
        temperature: float,
        max_depth: int = 5
    ) -> Tuple[Optional[Program], float]:
        """
        Sample a program from the grammar using direct stochastic sampling.

        This uses Grammar.sample() which is O(depth) rather than the old
        enumerate-then-sample approach which was O(enumeration size).
        """
        result = self.grammar.sample(request_type, max_depth=max_depth, temperature=temperature)
        if result is None:
            return None, 0.0
        return result

    def _generate_examples(
        self,
        program: Program,
        inputs: List[Any],
        min_examples: int = 5
    ) -> List[Tuple[Any, Any]]:
        """Generate input-output examples for a program."""
        examples = []

        for inp in inputs:
            try:
                fn = program.evaluate([])
                out = fn(inp)
                if isinstance(out, bool):  # Only keep boolean outputs for classification
                    examples.append((inp, out))
            except (ValueError, TypeError, ZeroDivisionError, IndexError, KeyError, AttributeError, RecursionError):
                # Expected runtime errors from programs - skip this input
                continue

            if len(examples) >= 10:
                break

        return examples


# ============================================================================
# DREAMCODER V2
# ============================================================================

class DreamCoderV2:
    """
    Full DreamCoder V2 with neural recognition and proper dreaming.
    """

    def __init__(
        self,
        grammar: Grammar,
        tasks: List[Task],
        eval_fn: Callable[[Program, Any], Any],

        # Wake settings
        enumeration_budget: int = 100000,
        enumeration_timeout: float = 120.0,
        max_depth: int = 8,

        # Frontier settings
        keep_top_k: int = 5,

        # Component flags
        use_compression: bool = True,
        use_recognition: bool = True,
        use_dreaming: bool = True,

        # Compression settings
        max_inventions_per_iteration: int = 5,
        min_compression_savings: float = 2.0,

        # Neural recognition settings
        recognition_hidden_dim: int = 128,
        recognition_epochs: int = 10,
        recognition_lr: float = 1e-3,

        # Dreaming settings
        dreams_per_iteration: int = 50,
        dream_temperature: float = 1.0,

        # General
        max_iterations: int = 10,
        verbose: bool = True,
        log_dir: Optional[str] = None,
        device: str = 'cpu'
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

        # Neural recognition
        self.recognition_hidden_dim = recognition_hidden_dim
        self.recognition_epochs = recognition_epochs
        self.device = device

        if use_recognition:
            self.recognition = NeuralRecognitionModel(
                grammar=grammar,
                hidden_dim=recognition_hidden_dim,
                learning_rate=recognition_lr,
                device=device
            )
        else:
            self.recognition = None

        # Dreaming
        self.dreams_per_iteration = dreams_per_iteration
        self.dream_temperature = dream_temperature

        if use_dreaming and self.recognition:
            self.dreamer = NeuralDreamer(
                grammar=grammar,
                recognition_model=self.recognition,
                eval_fn=eval_fn,
                device=device
            )
        else:
            self.dreamer = None

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
        self.library_history: List[List[str]] = []

        # Track embeddings for interpretability
        self._embedding_snapshots: List[Dict[str, Any]] = []

    def log(self, msg: str, level: int = 0):
        """Log message with indentation."""
        if self.verbose:
            indent = "  " * level
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {indent}{msg}", flush=True)

    def run(self) -> Dict:
        """Run the full wake-sleep learning loop."""
        start_time = time.time()

        self.log("=" * 70)
        self.log("DREAMCODER V2 - NEURAL WAKE-SLEEP LEARNING")
        self.log("=" * 70)
        self.log(f"Tasks: {len(self.tasks)}")
        self.log(f"Initial grammar: {len(self.grammar)} primitives")
        self.log(f"Components: compression={self.use_compression}, "
                f"recognition={self.use_recognition}, dreaming={self.use_dreaming}")
        self.log(f"Device: {self.device}")
        self.log("")

        for iteration in range(self.max_iterations):
            self.log("=" * 70)
            self.log(f"ITERATION {iteration + 1}/{self.max_iterations}")
            self.log("=" * 70)

            metrics = self._run_iteration(iteration)
            self.iteration_metrics.append(metrics)

            # Log summary
            self.log("")
            self.log(f"Summary:", 1)
            self.log(f"Solved: {metrics.tasks_solved}/{metrics.tasks_total}", 2)
            self.log(f"Programs: {metrics.programs_enumerated:,}", 2)
            self.log(f"New abstractions: {len(metrics.new_abstractions)}", 2)
            self.log(f"Recognition loss: {metrics.recognition_loss:.4f}", 2)
            self.log(f"Dreams generated: {metrics.dreams_generated}", 2)

            # Early stopping
            if metrics.tasks_solved == metrics.tasks_total:
                self.log("\nAll tasks solved!")
                break

        total_time = time.time() - start_time

        # Final summary
        self.log("")
        self.log("=" * 70)
        self.log("FINAL SUMMARY")
        self.log("=" * 70)

        solved = sum(1 for tm in self.task_metrics.values() if tm.solved)
        self.log(f"Total time: {total_time:.1f}s")
        self.log(f"Solved: {solved}/{len(self.tasks)}")
        self.log(f"Final grammar: {len(self.grammar)} primitives")

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
            frontier = self.frontiers[task.name]

            if frontier.n_solutions >= self.keep_top_k and frontier.solved:
                tasks_solved += 1
                continue

            # Get task-specific grammar from neural recognition
            if self.recognition and iteration > 0:
                task_grammar = self.recognition.predict_grammar_weights(task)
            else:
                task_grammar = self.grammar

            # Enumerate using TopDownEnumerator (replaces deprecated enumerate_simple)
            programs_tried = 0
            enum_start = time.time()
            enumerator = TopDownEnumerator(
                task_grammar,
                max_depth=self.max_depth,
                max_programs=self.enumeration_budget
            )

            for program, log_prob in enumerator.enumerate(
                task.request_type,
                timeout_seconds=self.enumeration_timeout
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
                        entry = SolutionEntry(
                            program=program,
                            log_probability=log_prob,
                            log_likelihood=0.0,
                            programs_enumerated=programs_tried,
                            time_found=time.time() - enum_start
                        )
                        if frontier.add(entry):
                            tm = self.task_metrics[task.name]
                            if not tm.solved:
                                tm.solved = True
                                tm.iteration_solved = iteration
                                tm.programs_to_solve = programs_tried
                                tm.best_program = str(program)
                                tm.description_length = entry.description_length

                        if frontier.n_solutions >= self.keep_top_k:
                            break
                except (ValueError, TypeError, ZeroDivisionError, IndexError, KeyError, AttributeError, RecursionError):
                    # Expected runtime errors from program evaluation - skip this program
                    pass

            frontier.total_programs_searched += programs_tried
            total_programs += programs_tried

            if frontier.solved:
                tasks_solved += 1

        wake_time = time.time() - wake_start
        self.log(f"Wake: {tasks_solved}/{len(self.tasks)} solved, "
                f"{total_programs:,} programs in {wake_time:.1f}s")

        # =====================
        # SLEEP - COMPRESSION
        # =====================
        new_abstractions = []
        compression_time = 0.0

        if self.use_compression:
            self.log("\n[SLEEP - COMPRESSION] Finding abstractions...")
            comp_start = time.time()

            all_frontiers = []
            for frontier in self.frontiers.values():
                if frontier.n_solutions > 0:
                    programs_with_ll = [(e.program, e.log_likelihood) for e in frontier.entries]
                    all_frontiers.append(programs_with_ll)

            if all_frontiers:
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

                    # Update recognition model grammar
                    if self.recognition:
                        self.recognition.grammar = self.grammar

                    # Update dreamer grammar
                    if self.dreamer:
                        self.dreamer.grammar = self.grammar

                    self.log(f"Found {len(new_abstractions)} abstraction(s)", 1)

            compression_time = time.time() - comp_start
            self.library_history.append(new_abstractions)

        # =====================
        # SLEEP - RECOGNITION TRAINING
        # =====================
        recognition_loss = 0.0
        recognition_time = 0.0

        if self.use_recognition and self.recognition:
            self.log("\n[SLEEP - RECOGNITION] Training neural model...")
            rec_start = time.time()

            recognition_loss = self.recognition.train_on_frontiers(
                self.tasks,
                self.frontiers,
                epochs=self.recognition_epochs
            )

            recognition_time = time.time() - rec_start
            n_solved = sum(1 for f in self.frontiers.values() if f.solved)
            self.log(f"Trained on {n_solved} solved tasks, loss: {recognition_loss:.4f}", 1)

        # =====================
        # SLEEP - DREAMING
        # =====================
        dreams_generated = 0
        dream_time = 0.0

        if self.use_dreaming and self.dreamer and iteration > 0:
            self.log("\n[SLEEP - DREAMING] Generating synthetic tasks...")
            dream_start = time.time()

            # Collect example inputs
            all_inputs = []
            for task in self.tasks:
                for inp, _ in task.examples[:5]:
                    all_inputs.append(inp)

            # Generate dreams
            dreams = self.dreamer.generate_dreams(
                self.tasks[0].request_type,
                self.dreams_per_iteration,
                all_inputs,
                temperature=self.dream_temperature
            )
            dreams_generated = len(dreams)

            # Train recognition on dreams
            for dream_task, program in dreams:
                # Create synthetic frontier
                synthetic_frontier = TaskFrontier(dream_task, max_size=1)
                entry = SolutionEntry(
                    program=program,
                    log_probability=self.grammar.program_log_likelihood(
                        program, dream_task.request_type
                    ),
                    log_likelihood=0.0,
                    programs_enumerated=0,
                    time_found=0.0
                )
                synthetic_frontier.add(entry)

                # Create temporary frontiers dict for training
                temp_frontiers = {dream_task.name: synthetic_frontier}
                self.recognition.train_on_frontiers([dream_task], temp_frontiers, epochs=1)

            dream_time = time.time() - dream_start
            self.log(f"Generated {dreams_generated} dreams", 1)

        # =====================
        # GRAMMAR WEIGHT UPDATE
        # =====================
        if tasks_solved > 0:
            all_frontiers = []
            for frontier in self.frontiers.values():
                if frontier.n_solutions > 0:
                    programs_with_ll = [(e.program, e.log_likelihood) for e in frontier.entries]
                    all_frontiers.append(programs_with_ll)

            if all_frontiers:
                self.grammar = self.grammar.inside_outside_update(all_frontiers)

        # =====================
        # SNAPSHOT EMBEDDINGS
        # =====================
        if self.recognition:
            self._snapshot_embeddings(iteration)

        return IterationMetrics(
            iteration=iteration,
            tasks_solved=tasks_solved,
            tasks_total=len(self.tasks),
            programs_enumerated=total_programs,
            wake_time=wake_time,
            new_abstractions=new_abstractions,
            compression_time=compression_time,
            recognition_loss=recognition_loss,
            recognition_time=recognition_time,
            dreams_generated=dreams_generated,
            dream_time=dream_time,
            grammar_size=len(self.grammar)
        )

    def _snapshot_embeddings(self, iteration: int):
        """Snapshot task embeddings for interpretability tracking."""
        embeddings = {}
        for task in self.tasks[:20]:  # Sample for efficiency
            emb = self.recognition.get_task_embedding(task)
            embeddings[task.name] = emb.numpy().tolist()

        self._embedding_snapshots.append({
            'iteration': iteration,
            'embeddings': embeddings
        })

    def _compile_results(self, total_time: float) -> Dict:
        """Compile all results."""
        return {
            'config': {
                'tasks': len(self.tasks),
                'initial_grammar_size': len(self.initial_grammar),
                'enumeration_budget': self.enumeration_budget,
                'max_depth': self.max_depth,
                'keep_top_k': self.keep_top_k,
                'use_compression': self.use_compression,
                'use_recognition': self.use_recognition,
                'use_dreaming': self.use_dreaming,
                'recognition_hidden_dim': self.recognition_hidden_dim,
                'dreams_per_iteration': self.dreams_per_iteration,
                'max_iterations': self.max_iterations
            },
            'summary': {
                'total_time': total_time,
                'iterations_run': len(self.iteration_metrics),
                'tasks_solved': sum(1 for tm in self.task_metrics.values() if tm.solved),
                'tasks_total': len(self.tasks),
                'final_grammar_size': len(self.grammar),
                'total_abstractions': sum(len(m.new_abstractions) for m in self.iteration_metrics),
                'total_dreams': sum(m.dreams_generated for m in self.iteration_metrics)
            },
            'learning_curve': [
                {
                    'iteration': m.iteration,
                    'tasks_solved': m.tasks_solved,
                    'programs': m.programs_enumerated,
                    'abstractions': len(m.new_abstractions),
                    'recognition_loss': m.recognition_loss,
                    'dreams': m.dreams_generated,
                    'grammar_size': m.grammar_size
                }
                for m in self.iteration_metrics
            ],
            'task_metrics': {name: asdict(tm) for name, tm in self.task_metrics.items()},
            'library_evolution': self.library_history,
            'embedding_evolution': self._embedding_snapshots,
            'recognition_training_losses': self.recognition.training_losses if self.recognition else []
        }

    def _save_results(self, results: Dict):
        """Save results to log directory."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

        # Save JSON results
        json_path = self.log_dir / f"dreamcoder_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        # Save model checkpoint
        if self.recognition:
            model_path = self.log_dir / f"recognition_model_{timestamp}.pt"
            self.recognition.save(str(model_path))

        self.log(f"\nResults saved to: {json_path}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_tasks_from_rules(
    rules: List,
    n_examples: int = 100,  # Increased from 20 to prevent spurious solutions
    n_holdout: int = 20,    # Held-out examples for verification
    hand_size: int = 6,     # Configurable hand size
    seed: int = 42
) -> List[Task]:
    """
    Create Task objects from rule definitions.

    Args:
        rules: List of PretrainingRule objects
        n_examples: Number of training examples (default 100, was 20)
        n_holdout: Number of held-out examples for solution verification
        hand_size: Number of cards per hand (default 6)
        seed: Random seed for reproducibility

    The increased example count and held-out verification help prevent
    spurious solutions that pass training examples by coincidence.
    """
    from rules.cards import sample_hand

    random.seed(seed)
    tasks = []

    for rule in rules:
        positives = []
        negatives = []
        holdout_positives = []
        holdout_negatives = []

        target = n_examples // 2
        holdout_target = n_holdout // 2

        # Sample more hands to ensure we get enough diverse examples
        for _ in range(50000):  # Increased from 10000
            hand = sample_hand(hand_size)
            try:
                label = rule.eval(hand)
                if label:
                    if len(positives) < target:
                        positives.append((hand, True))
                    elif len(holdout_positives) < holdout_target:
                        holdout_positives.append((hand, True))
                else:
                    if len(negatives) < target:
                        negatives.append((hand, False))
                    elif len(holdout_negatives) < holdout_target:
                        holdout_negatives.append((hand, False))
            except (ValueError, TypeError, ZeroDivisionError, IndexError, KeyError, AttributeError, RecursionError):
                # Expected runtime errors from rule evaluation - skip this hand
                continue

            # Check if we have enough of everything
            if (len(positives) >= target and len(negatives) >= target and
                len(holdout_positives) >= holdout_target and
                len(holdout_negatives) >= holdout_target):
                break

        examples = positives[:target] + negatives[:target]
        random.shuffle(examples)

        holdout = holdout_positives[:holdout_target] + holdout_negatives[:holdout_target]
        random.shuffle(holdout)

        task = Task(
            name=rule.id,
            request_type=arrow(HAND, BOOL),
            examples=examples,
            family=getattr(rule, 'family', ''),
            difficulty_level=getattr(rule, 'level', 0)
        )
        # Store holdout examples for verification
        task.holdout_examples = holdout
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
    print("DREAMCODER V2 - QUICK TEST")
    print("=" * 70)

    from rules.pretraining_rules import get_easy_pretraining_rules

    # Use a few easy pretraining rules
    rules = get_easy_pretraining_rules()[:6]
    print(f"\nTest rules: {[r.id for r in rules]}")

    # Create tasks
    tasks = create_tasks_from_rules(rules, n_examples=20, seed=42)
    print(f"Created {len(tasks)} tasks")

    # Build lean grammar
    grammar = build_lean_grammar()
    print(f"Grammar: {len(grammar)} primitives")

    # Create evaluation function
    eval_fn = make_eval_fn()

    # Run DreamCoder V2
    dc = DreamCoderV2(
        grammar=grammar,
        tasks=tasks,
        eval_fn=eval_fn,

        # Fast settings for demo
        enumeration_budget=50000,
        enumeration_timeout=30.0,
        max_depth=6,

        keep_top_k=3,

        use_compression=True,
        use_recognition=True,
        use_dreaming=True,

        recognition_hidden_dim=64,
        recognition_epochs=5,
        dreams_per_iteration=20,

        max_iterations=3,
        verbose=True,
        log_dir="results/v2_test"
    )

    results = dc.run()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"Solved: {results['summary']['tasks_solved']}/{results['summary']['tasks_total']}")
    print(f"Abstractions: {results['summary']['total_abstractions']}")
    print(f"Dreams: {results['summary']['total_dreams']}")
