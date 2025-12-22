#!/usr/bin/env python3
"""
Contrastive Wake-Sleep Learning Loop (Phase 5 Integration)

This module integrates the contrastive recognition model with the full
wake-sleep learning loop:

1. WAKE: Recognition-guided enumeration using contrastive embeddings
2. SLEEP - Compression: Extract abstractions and expand recognition model
3. SLEEP - Recognition: Train on solved tasks with structural similarity loss
4. SLEEP - Dreaming: Generate contrastive dreams with near-miss pairs

Key innovations over standard DreamCoder:
- Contrastive task encoding: τ = mean(pos) - mean(neg)
- Structural similarity loss: cluster tasks by primitive usage
- Near-miss dream generation: negatives differ by one card
- Dynamic vocabulary expansion for inventions

Author: Can Konuk (with Claude)
"""

import sys
import math
import time
import random
import copy
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime

import torch

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.type_system import Type, arrow, HAND, BOOL
from dreamcoder_core.program import Program, Primitive, Application, Abstraction, Index, Invented
from dreamcoder_core.grammar import Grammar, Production
from dreamcoder_core.enumeration import TopDownEnumerator, Frontier, EnumerationResult
# NOTE: enumerate_simple is deprecated - use TopDownEnumerator instead
from dreamcoder_core.compression import compress_frontiers, CompressionResult
from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.contrastive_dreaming import (
    ContrastiveDreamer, StandardDreamer, HybridDreamer, ContrastiveDream
)
from dreamcoder_core.task import Task


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class TaskFrontier:
    """Top-k solutions for a single task."""
    task: Task
    entries: List[EnumerationResult] = field(default_factory=list)
    max_size: int = 5

    total_programs_searched: int = 0
    total_time: float = 0.0

    def add(self, entry: EnumerationResult) -> bool:
        """Add solution if it improves the frontier."""
        # Check for duplicates
        for e in self.entries:
            if str(e.program) == str(entry.program):
                return False

        self.entries.append(entry)
        self.entries.sort(key=lambda e: -e.log_probability)

        if len(self.entries) > self.max_size:
            removed = self.entries.pop()
            return entry != removed
        return True

    @property
    def best(self) -> Optional[EnumerationResult]:
        return self.entries[0] if self.entries else None

    @property
    def solved(self) -> bool:
        return any(e.log_likelihood == 0.0 for e in self.entries)

    @property
    def n_solutions(self) -> int:
        return len(self.entries)

    def all_programs(self) -> List[Program]:
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
    new_abstractions: List[str]
    compression_savings: float
    compression_time: float

    # Sleep - Recognition
    recognition_loss: float
    structural_similarity_loss: float
    recognition_time: float

    # Sleep - Dreaming
    dreams_generated: int
    contrastive_dreams: int
    dream_time: float

    # Grammar state
    grammar_size: int

    @property
    def success_rate(self) -> float:
        return self.tasks_solved / self.tasks_total if self.tasks_total > 0 else 0.0


# ============================================================================
# CONTRASTIVE WAKE-SLEEP LEARNER
# ============================================================================

class ContrastiveWakeSleep:
    """
    DreamCoder with contrastive recognition model and near-miss dreaming.

    Key differences from standard DreamCoder:
    1. Uses ContrastiveRecognitionModel instead of GRU-based model
    2. Trains with structural similarity loss
    3. Generates contrastive dreams with near-miss negatives
    4. Dynamically expands vocabulary for inventions
    """

    def __init__(
        self,
        grammar: Grammar,
        tasks: List[Task],
        eval_fn: Callable[[Program, Any], Any],
        sample_hand_fn: Callable,
        sample_card_fn: Callable,

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

        # Recognition settings
        recognition_hidden_dim: int = 32,
        recognition_epochs: int = 10,
        recognition_lr: float = 1e-3,
        structural_similarity_weight: float = 0.1,

        # Dreaming settings
        dreams_per_iteration: int = 50,
        contrastive_dream_ratio: float = 0.5,
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
        self.sample_hand_fn = sample_hand_fn
        self.sample_card_fn = sample_card_fn

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

        # Recognition settings
        self.recognition_hidden_dim = recognition_hidden_dim
        self.recognition_epochs = recognition_epochs
        self.recognition_lr = recognition_lr
        self.structural_similarity_weight = structural_similarity_weight

        # Dreaming settings
        self.dreams_per_iteration = dreams_per_iteration
        self.contrastive_dream_ratio = contrastive_dream_ratio
        self.dream_temperature = dream_temperature

        # General
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.log_dir = Path(log_dir) if log_dir else None
        self.device = device

        # Initialize recognition model
        if use_recognition:
            self.recognition = ContrastiveRecognitionModel(
                grammar=grammar,
                card_out=recognition_hidden_dim,
                pred_hidden=recognition_hidden_dim * 2,
                learning_rate=recognition_lr,
                device=device
            )
        else:
            self.recognition = None

        # Initialize dreamer
        if use_dreaming:
            self.dreamer = HybridDreamer(
                grammar=grammar,
                eval_fn=eval_fn,
                sample_hand_fn=sample_hand_fn,
                sample_card_fn=sample_card_fn,
                contrastive_ratio=contrastive_dream_ratio,
                device=device
            )
        else:
            self.dreamer = None

        # State
        self.frontiers: Dict[str, TaskFrontier] = {
            t.name: TaskFrontier(t, max_size=keep_top_k)
            for t in tasks
        }
        self.iteration_metrics: List[IterationMetrics] = []

    def log(self, msg: str, level: int = 0):
        """Log message with indentation."""
        if self.verbose:
            indent = "  " * level
            print(f"{indent}{msg}", flush=True)

    def run(self) -> Dict:
        """Run the full wake-sleep learning loop."""
        start_time = time.time()

        self.log("=" * 70)
        self.log("CONTRASTIVE WAKE-SLEEP LEARNING")
        self.log("=" * 70)
        self.log(f"Tasks: {len(self.tasks)}")
        self.log(f"Initial grammar: {len(self.grammar)} primitives")
        self.log(f"Recognition: {self.use_recognition} "
                f"(contrastive, hidden_dim={self.recognition_hidden_dim})")
        self.log(f"Dreaming: {self.use_dreaming} "
                f"(contrastive_ratio={self.contrastive_dream_ratio})")
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
            self.log(f"Recognition loss: {metrics.recognition_loss:.4f}", 2)
            self.log(f"Structural similarity loss: {metrics.structural_similarity_loss:.4f}", 2)
            self.log(f"Dreams: {metrics.dreams_generated} "
                    f"({metrics.contrastive_dreams} contrastive)", 2)
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

        solved = sum(1 for f in self.frontiers.values() if f.solved)
        self.log(f"Total time: {total_time:.1f}s")
        self.log(f"Iterations: {len(self.iteration_metrics)}")
        self.log(f"Solved: {solved}/{len(self.tasks)}")
        self.log(f"Final grammar: {len(self.grammar)} primitives")

        # Compile and return results
        results = self._compile_results(total_time)
        if self.log_dir:
            self._save_results(results)

        return results

    def _run_iteration(self, iteration: int) -> IterationMetrics:
        """Run one complete wake-sleep iteration."""

        # =====================
        # WAKE PHASE
        # =====================
        self.log("\n[WAKE] Recognition-guided enumeration...")
        wake_start = time.time()

        total_programs = 0
        tasks_solved = 0

        for task in self.tasks:
            self.log(f"Task: {task.name}", 1)

            frontier = self.frontiers[task.name]

            # Skip if already solved with max solutions
            if frontier.n_solutions >= self.keep_top_k and frontier.solved:
                self.log(f"(already solved with {frontier.n_solutions} solutions)", 2)
                tasks_solved += 1
                continue

            # Get task-specific grammar weights (if using recognition)
            if self.recognition and iteration > 0:
                # Update recognition model's grammar if it changed
                self.recognition.grammar = self.grammar
                task_grammar = self.recognition.predict_grammar_weights(task)
                top_preds = self.recognition.get_top_predictions(task, n=5)
                if top_preds:
                    self.log(f"Top predictions: {[p[0][:15] for p in top_preds]}", 2)
            else:
                task_grammar = self.grammar

            # Enumerate using TopDownEnumerator with MEMOIZATION (1000x+ speedup)
            programs_tried = 0
            enum_start = time.time()
            enumerator = TopDownEnumerator(
                task_grammar,
                max_depth=self.max_depth,
                max_programs=self.enumeration_budget
            )

            # Use enumerate_memoized() for DreamCoder-style dynamic programming
            # This caches subproblem solutions for 1000x+ speedup
            for program, log_prob in enumerator.enumerate_memoized(
                task.request_type,
                max_cost=50.0,
                timeout_seconds=self.enumeration_timeout,
                depth_limit=self.max_depth
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
                        entry = EnumerationResult(
                            program=program,
                            log_probability=log_prob,
                            log_likelihood=0.0,
                            description_length=-log_prob / math.log(2),
                            programs_enumerated=programs_tried,
                            time_seconds=time.time() - enum_start
                        )
                        if frontier.add(entry):
                            self.log(f"Found solution #{frontier.n_solutions}: "
                                    f"{str(program)[:40]}...", 2)

                        if frontier.n_solutions >= self.keep_top_k:
                            break
                except (ValueError, TypeError, ZeroDivisionError, IndexError, KeyError, AttributeError, RecursionError):
                    # Expected evaluation errors from malformed programs
                    pass

            frontier.total_programs_searched += programs_tried
            total_programs += programs_tried

            if frontier.solved:
                tasks_solved += 1
                self.log(f"=> Solved with {frontier.n_solutions} solution(s)", 2)
            else:
                self.log(f"=> Unsolved after {programs_tried:,} programs", 2)

        wake_time = time.time() - wake_start
        self.log(f"\nWake phase: {tasks_solved}/{len(self.tasks)} solved in {wake_time:.1f}s")

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

                    # Update recognition model with new primitives
                    if self.recognition:
                        for inv in result.new_inventions:
                            self.recognition.add_invention(inv)
                            self.log(f"Added invention to recognition: {inv}", 2)

                    # Update dreamer grammar
                    if self.dreamer:
                        self.dreamer.grammar = self.grammar

                    self.log(f"Found {len(new_abstractions)} abstraction(s)", 1)
                else:
                    self.log("No new abstractions found", 1)

            compression_time = time.time() - comp_start

        # =====================
        # SLEEP - RECOGNITION TRAINING
        # =====================
        recognition_loss = 0.0
        structural_similarity_loss = 0.0
        recognition_time = 0.0

        if self.use_recognition and self.recognition:
            self.log("\n[SLEEP - RECOGNITION] Training contrastive model...")
            rec_start = time.time()

            # Train on solved tasks
            solved_frontiers = {f.task.name: f for f in self.frontiers.values() if f.solved}
            solved_tasks = [f.task for f in self.frontiers.values() if f.solved]
            recognition_loss = self.recognition.train_on_frontiers(
                tasks=solved_tasks,
                frontiers=solved_frontiers,
                epochs=self.recognition_epochs,
                lambda_struct=self.structural_similarity_weight
            )

            # Compute structural similarity loss separately for logging
            structural_similarity_loss = self._compute_structural_similarity_loss()

            recognition_time = time.time() - rec_start
            n_solved = sum(1 for f in self.frontiers.values() if f.solved)
            self.log(f"Trained on {n_solved} solved tasks, "
                    f"loss={recognition_loss:.4f}", 1)

        # =====================
        # SLEEP - DREAMING
        # =====================
        dreams_generated = 0
        contrastive_dreams = 0
        dream_time = 0.0

        if self.use_dreaming and self.dreamer and iteration > 0:
            self.log("\n[SLEEP - DREAMING] Generating contrastive dreams...")
            dream_start = time.time()

            dreams = self.dreamer.generate_dreams(
                request_type=self.tasks[0].request_type,
                n_dreams=self.dreams_per_iteration,
                n_examples_per_dream=10,
                temperature=self.dream_temperature,
                verbose=False
            )

            dreams_generated = len(dreams)
            contrastive_dreams = sum(1 for d in dreams if d.n_near_miss_pairs > 0)

            # Train recognition on dreams
            if self.recognition:
                for dream in dreams:
                    # Create temporary frontier with known solution
                    temp_frontier = TaskFrontier(dream.task, max_size=1)
                    entry = EnumerationResult(
                        program=dream.program,
                        log_probability=self.grammar.program_log_likelihood(
                            dream.program, dream.task.request_type
                        ),
                        log_likelihood=0.0,
                        description_length=0.0,
                        programs_enumerated=0,
                        time_seconds=0.0
                    )
                    temp_frontier.add(entry)

                    # Train on this dream
                    self.recognition.train_on_frontiers(
                        tasks=[dream.task],
                        frontiers={dream.task.name: temp_frontier},
                        epochs=1,
                        lambda_struct=0.0  # Skip structural loss for dreams
                    )

            dream_time = time.time() - dream_start
            self.log(f"Generated {dreams_generated} dreams "
                    f"({contrastive_dreams} contrastive)", 1)

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
            structural_similarity_loss=structural_similarity_loss,
            recognition_time=recognition_time,
            dreams_generated=dreams_generated,
            contrastive_dreams=contrastive_dreams,
            dream_time=dream_time,
            grammar_size=len(self.grammar)
        )

    def _compute_structural_similarity_loss(self) -> float:
        """Compute structural similarity loss for logging."""
        if not self.recognition:
            return 0.0

        solved_tasks = [f.task for f in self.frontiers.values() if f.solved]
        if len(solved_tasks) < 2:
            return 0.0

        # Get primitives used for each task
        task_primitives = {}
        for f in self.frontiers.values():
            if f.solved:
                prims = set()
                for entry in f.entries:
                    self._collect_primitives(entry.program, prims)
                task_primitives[f.task.name] = prims

        return self.recognition.compute_structural_similarity_loss(
            solved_tasks[:10],  # Limit for efficiency
            task_primitives
        ).item()

    def _collect_primitives(self, program: Program, primitives: Set[str]):
        """Collect primitive names from a program."""
        if isinstance(program, (Primitive, Invented)):
            primitives.add(str(program))
        elif isinstance(program, Application):
            self._collect_primitives(program.f, primitives)
            self._collect_primitives(program.x, primitives)
        elif isinstance(program, Abstraction):
            self._collect_primitives(program.body, primitives)

    def _compile_results(self, total_time: float) -> Dict:
        """Compile all results into a dictionary."""
        solved = sum(1 for f in self.frontiers.values() if f.solved)
        total_dreams = sum(m.dreams_generated for m in self.iteration_metrics)
        total_contrastive = sum(m.contrastive_dreams for m in self.iteration_metrics)

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
                'contrastive_dream_ratio': self.contrastive_dream_ratio,
                'recognition_hidden_dim': self.recognition_hidden_dim,
                'structural_similarity_weight': self.structural_similarity_weight,
                'max_iterations': self.max_iterations
            },
            'summary': {
                'total_time': total_time,
                'iterations_run': len(self.iteration_metrics),
                'tasks_solved': solved,
                'tasks_total': len(self.tasks),
                'final_grammar_size': len(self.grammar),
                'total_abstractions': sum(len(m.new_abstractions)
                                         for m in self.iteration_metrics),
                'total_dreams': total_dreams,
                'total_contrastive_dreams': total_contrastive
            },
            'learning_curve': [
                {
                    'iteration': m.iteration,
                    'tasks_solved': m.tasks_solved,
                    'success_rate': m.success_rate,
                    'recognition_loss': m.recognition_loss,
                    'structural_similarity_loss': m.structural_similarity_loss,
                    'dreams': m.dreams_generated,
                    'contrastive_dreams': m.contrastive_dreams,
                    'grammar_size': m.grammar_size
                }
                for m in self.iteration_metrics
            ],
            'final_grammar': [str(p.program) for p in self.grammar.productions]
        }

    def _save_results(self, results: Dict):
        """Save results to log directory."""
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        json_path = self.log_dir / f"contrastive_results_{timestamp}.json"

        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.log(f"\nResults saved to: {json_path}")

    @property
    def n_solutions(self):
        """Alias for compatibility with TaskFrontier interface."""
        return len(self.entries) if hasattr(self, 'entries') else 0


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_tasks_from_rules(
    rules: List,
    n_examples: int = 20,
    seed: int = 42
) -> List[Task]:
    """Create Task objects from rule definitions."""
    from rules.cards import sample_hand

    random.seed(seed)
    tasks = []

    for rule in rules:
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
            except (ValueError, TypeError, ZeroDivisionError, IndexError, KeyError, AttributeError):
                # Rule evaluation failed for this hand - skip it
                continue

            if len(positives) >= target and len(negatives) >= target:
                break

        examples = positives[:target] + negatives[:target]
        random.shuffle(examples)

        task = Task(
            name=rule.id,
            request_type=arrow(HAND, BOOL),
            examples=examples,
            family=getattr(rule, 'family', ''),
            difficulty_level=getattr(rule, 'level', 0)
        )
        tasks.append(task)

    return tasks


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CONTRASTIVE WAKE-SLEEP TEST")
    print("=" * 70)

    # Import dependencies
    from dreamcoder_core.lean_primitives import build_lean_grammar
    from rules.cards import sample_hand
    from rules.catalogue import RULE_DICT

    # Build grammar
    grammar = build_lean_grammar()
    print(f"\nGrammar: {len(grammar)} primitives")

    # Select easy test rules
    test_rule_ids = ['Uniform_color', 'Ends_same_color']
    rules = [RULE_DICT[rid] for rid in test_rule_ids if rid in RULE_DICT]
    print(f"Test rules: {[r.id for r in rules]}")

    # Create tasks
    tasks = create_tasks_from_rules(rules, n_examples=20, seed=42)
    print(f"Created {len(tasks)} tasks")

    # Create eval function
    def eval_fn(program: Program, hand):
        fn = program.evaluate([])
        return fn(hand)

    def sample_hand_fn():
        return sample_hand(6)

    def sample_card_fn():
        return sample_hand(1)[0]

    # Run contrastive wake-sleep
    learner = ContrastiveWakeSleep(
        grammar=grammar,
        tasks=tasks,
        eval_fn=eval_fn,
        sample_hand_fn=sample_hand_fn,
        sample_card_fn=sample_card_fn,

        enumeration_budget=50000,
        enumeration_timeout=30.0,
        max_depth=7,

        keep_top_k=3,

        use_compression=True,
        use_recognition=True,
        use_dreaming=True,

        recognition_hidden_dim=32,
        recognition_epochs=5,
        structural_similarity_weight=0.1,

        dreams_per_iteration=10,
        contrastive_dream_ratio=0.5,

        max_iterations=2,
        verbose=True,
        log_dir="results/contrastive_test"
    )

    results = learner.run()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    print(f"Solved: {results['summary']['tasks_solved']}/{results['summary']['tasks_total']}")
    print(f"Grammar: {results['summary']['final_grammar_size']} primitives")
    print(f"Dreams: {results['summary']['total_dreams']} "
          f"({results['summary']['total_contrastive_dreams']} contrastive)")
