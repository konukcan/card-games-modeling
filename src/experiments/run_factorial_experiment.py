#!/usr/bin/env python3
"""
2×3×3 Factorial Experiment Runner
==================================

This script implements a full factorial experiment design:
- 2 Recognition Models: GRU vs Contrastive
- 3 Dream Strategies: Standard vs Balanced vs Contrastive
- 3 Primitive Libraries: Lean vs Lean+Fold vs Minimal

Total: 18 conditions

Features:
- Increasing enumeration budget per iteration (progressive difficulty)
- Comprehensive logging for post-hoc analysis
- Checkpoint/resume capability
- Timing calibration (~1 hour per condition)

Usage:
    # Dry run (show conditions)
    python3 run_factorial_experiment.py --dry-run

    # Quick smoke test
    python3 run_factorial_experiment.py --smoke-test

    # Full experiment (with caffeinate)
    nohup caffeinate -d -i -s python3 run_factorial_experiment.py > experiment.log 2>&1 &

    # Single condition
    python3 run_factorial_experiment.py --recognition gru --dreams balanced --primitives lean
"""

import sys
import os
import json
import time
import argparse
import logging
import traceback
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
import pickle

# Number of parallel workers for enumeration
# Reduced from cpu_count()-1 (11) to 6 to avoid memory pressure and system lag
N_WORKERS = 6

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.neural_recognition import NeuralRecognitionModel
from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.contrastive_dreaming import ConfigurableDreamer
from dreamcoder_core.type_system import arrow, HAND, BOOL
from dreamcoder_core.enumeration import TopDownEnumerator, EnumerationResult, Frontier
from dreamcoder_core.compression import compress_frontiers
from dreamcoder_core.grammar import Grammar
from dreamcoder_core.program import parse_program

from experiments.primitive_variants import (
    build_grammar_for_variant, get_primitive_variant, PRIMITIVE_VARIANTS
)
from rules.cards import sample_hand
from rules.catalogue import ALL_RULES, create_all_rules, get_rules_by_family
from rules.pretraining_rules import get_easy_pretraining_rules, get_all_pretraining_rules


# ============================================================================
# PARALLEL ENUMERATION HELPER
# ============================================================================

def _enumerate_task_worker(args: Dict) -> Dict:
    """
    Worker function for parallel task enumeration with recognition-guided search.

    This function runs in a separate process and handles enumeration
    for a single task. It rebuilds the grammar in-process to avoid
    pickling issues with lambda functions.

    RECOGNITION INTEGRATION:
    If 'predicted_log_probs' is provided, the worker applies these weights
    to the grammar using adaptive blending. This implements the key
    DreamCoder insight: recognition model predictions guide enumeration
    toward useful primitives.

    COST-BANDING:
    Instead of using a single max_cost=50.0, we use iterative cost-banding:
    first enumerate programs up to cost 15, then 20, then 25, etc.
    This ensures low-cost (high-probability) programs are explored first,
    which is how recognition guidance takes effect.

    Args:
        args: Dictionary with task_name, examples, holdout, request_type,
              primitive_variant, budget, max_depth, timeout, keep_top_k,
              and optionally predicted_log_probs (dict of primitive -> log_prob)
              and blend_factor (float, default 0.5)

    Returns:
        Dictionary with task_name, frontier entries, programs_enumerated,
        time_seconds, and solution info.
    """
    task_name = args['task_name']
    examples = args['examples']
    holdout = args['holdout']
    request_type = args['request_type']
    primitive_variant = args['primitive_variant']
    budget = args['budget']
    max_depth = args['max_depth']
    timeout = args['timeout']
    keep_top_k = args['keep_top_k']

    # Optional recognition-guided weights
    predicted_log_probs = args.get('predicted_log_probs', None)
    blend_factor = args.get('blend_factor', 0.5)

    # Rebuild grammar in worker process (avoids pickle issues with lambdas)
    grammar = build_grammar_for_variant(primitive_variant)

    # Apply recognition model predictions to grammar weights
    if predicted_log_probs is not None:
        from dreamcoder_core.grammar import Grammar, Production

        new_productions = []
        for prod in grammar.productions:
            prim_name = str(prod.program)
            if prim_name in predicted_log_probs:
                # Blend original and predicted with configurable factor
                new_lp = (1 - blend_factor) * prod.log_probability + \
                         blend_factor * predicted_log_probs[prim_name]
            else:
                new_lp = prod.log_probability
            new_productions.append(Production(prod.program, prod.tp, new_lp))

        grammar = Grammar(new_productions, grammar.log_variable).normalize_probabilities()

    task_start = time.time()

    # Create frontier
    frontier = Frontier(
        task_name=task_name,
        request_type=request_type,
        max_size=keep_top_k
    )

    programs_enumerated = 0
    solution_found = False

    # COST-BANDING: Iterate through progressively higher cost bounds
    # This ensures we explore low-cost (recognition-favored) programs first
    initial_cost = 15.0
    cost_increment = 5.0
    max_cost = 50.0

    try:
        cost_bound = initial_cost
        while cost_bound <= max_cost and not solution_found:
            # Check timeout
            elapsed = time.time() - task_start
            if elapsed > timeout:
                break

            remaining_timeout = timeout - elapsed
            remaining_budget = budget - programs_enumerated

            if remaining_budget <= 0:
                break

            # Create enumerator for this cost band
            enumerator = TopDownEnumerator(
                grammar,
                max_depth=max_depth,
                max_programs=remaining_budget
            )

            for program, log_prob in enumerator.enumerate(
                request_type,
                max_cost=cost_bound,
                timeout_seconds=remaining_timeout
            ):
                programs_enumerated += 1

                # Check if program solves the task
                correct = 0
                for hand, expected in examples:
                    result_val = eval_program_on_hand(program, hand)
                    if result_val == expected:
                        correct += 1

                if correct == len(examples):
                    # Verify on holdout
                    passed, accuracy = verify_on_holdout(program, holdout)
                    if passed:
                        enum_result = EnumerationResult(
                            program=program,
                            log_probability=log_prob,
                            log_likelihood=0.0,
                            description_length=-log_prob / 0.693,
                            programs_enumerated=programs_enumerated,
                            partial_programs_explored=enumerator.partial_programs_explored,
                            time_seconds=time.time() - task_start
                        )
                        frontier.add(enum_result)
                        solution_found = True
                        break

            cost_bound += cost_increment

    except Exception as e:
        pass  # Silent failure, will return empty frontier

    task_time = time.time() - task_start

    # Convert frontier to serializable format
    # NOTE: We cannot return Program objects through multiprocessing because they
    # contain lambda functions (in Primitive.value) which are not picklable.
    # We return program_str instead and reconstruct if needed on the receiving side.
    entries_data = []
    for entry in frontier.entries:
        entries_data.append({
            'program_str': str(entry.program),
            # 'program': entry.program,  # REMOVED: unpicklable lambdas cause MaybeEncodingError
            'log_probability': entry.log_probability,
            'log_likelihood': entry.log_likelihood,
            'description_length': entry.description_length,
            'programs_enumerated': entry.programs_enumerated,
            'partial_programs_explored': entry.partial_programs_explored,
            'time_seconds': entry.time_seconds,
        })

    return {
        'task_name': task_name,
        'solved': solution_found,
        'frontier_entries': entries_data,
        'programs_enumerated': programs_enumerated,
        'time_seconds': task_time,
        'solution_str': str(frontier.best.program) if not frontier.empty else None,
        'solution_size': frontier.best.program.size() if not frontier.empty else None,
    }


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

# 2×3×3 = 18 conditions
CONDITIONS = []
for recognition in ['gru', 'contrastive']:
    for dreams in ['standard', 'balanced', 'contrastive']:
        for primitives in ['lean', 'lean_plus_fold', 'minimal']:
            CONDITIONS.append({
                'recognition': recognition,
                'dreams': dreams,
                'primitives': primitives,
            })


@dataclass
class ExperimentConfig:
    """Configuration for a single experimental condition."""

    # Condition identifiers
    recognition_type: str = 'gru'
    dream_strategy: str = 'balanced'
    primitive_variant: str = 'lean'
    run_id: int = 1

    # Progressive enumeration budget (increases per iteration)
    # Increased to match successful overnight runs (200k → 500k)
    # Iteration 1: 200k, Iteration 2: 275k, Iteration 3: 350k, Iteration 4: 425k, Iteration 5: 500k
    base_enumeration_budget: int = 200_000
    budget_increment_per_iteration: int = 75_000
    max_enumeration_budget: int = 500_000

    # Progressive depth limit (same as before, proven adequate)
    base_max_depth: int = 7
    depth_increment_per_iteration: int = 1
    max_depth_limit: int = 11

    # Timeouts (per task) - dramatically increased for proper enumeration
    # At ~42 programs/sec, 300s = ~12,600 programs, 600s = ~25,200 programs
    base_timeout: float = 300.0  # 5 minutes per task
    timeout_increment: float = 60.0
    max_timeout: float = 600.0  # 10 minutes per task

    # Compression
    use_compression: bool = True
    max_inventions_per_iteration: int = 3
    min_compression_savings: float = 2.0

    # Dreaming
    dreams_per_iteration: int = 50
    n_examples_per_dream: int = 10

    # Recognition training
    recognition_hidden_dim: int = 64
    recognition_lr: float = 1e-3
    recognition_epochs: int = 5

    # General
    keep_top_k: int = 3
    max_iterations: int = 4  # 4 iterations provides good learning curve data
    n_examples_per_task: int = 50
    n_holdout_per_task: int = 20

    # Task selection
    # Use easy rules for ~1 hour per condition (22 rules instead of 43)
    use_easy_rules_only: bool = True
    use_pretraining_rules: bool = False  # Use easier pretraining rules instead of experimental
    max_tasks: Optional[int] = None  # None = all tasks

    # Output
    results_dir: str = 'results_factorial'

    def condition_name(self) -> str:
        """Return the condition nickname."""
        return f"{self.recognition_type}_{self.dream_strategy}_{self.primitive_variant}"

    def run_dir(self) -> Path:
        """Return the directory for this run."""
        return Path(self.results_dir) / self.condition_name() / f"run_{self.run_id}"

    def get_budget_for_iteration(self, iteration: int) -> int:
        """Get enumeration budget for a specific iteration."""
        budget = self.base_enumeration_budget + (iteration - 1) * self.budget_increment_per_iteration
        return min(budget, self.max_enumeration_budget)

    def get_depth_for_iteration(self, iteration: int) -> int:
        """Get max depth for a specific iteration."""
        depth = self.base_max_depth + (iteration - 1) * self.depth_increment_per_iteration
        return min(depth, self.max_depth_limit)

    def get_timeout_for_iteration(self, iteration: int) -> float:
        """Get timeout for a specific iteration."""
        timeout = self.base_timeout + (iteration - 1) * self.timeout_increment
        return min(timeout, self.max_timeout)


# ============================================================================
# TASK CREATION
# ============================================================================

@dataclass
class Task:
    """A synthesis task defined by examples."""
    name: str
    request_type: Any
    examples: List[Tuple[Any, bool]]
    holdout: List[Tuple[Any, bool]]
    rule_fn: Any


def sample_balanced_examples(
    rule,
    n_examples: int,
    hand_size: int = 6,
    max_attempts: int = 100000,
    min_per_class: int = 5
) -> Tuple[List[Tuple[Any, bool]], bool]:
    """
    Sample balanced examples (50% True, 50% False) for a rule.

    Uses rejection sampling to get equal positive and negative examples.
    If a rule has extreme base rates, this may hit max_attempts before
    getting enough examples of the rare class.

    Args:
        rule: Rule object with .eval method or callable
        n_examples: Total number of examples to generate
        hand_size: Number of cards per hand (default 6)
        max_attempts: Maximum sampling attempts before giving up
        min_per_class: Minimum examples needed per class

    Returns:
        Tuple of (examples, is_valid):
        - examples: List of (hand, result) tuples, balanced as evenly as possible
        - is_valid: True if we got at least min_per_class of each class
    """
    import random

    n_positive = n_examples // 2
    n_negative = n_examples - n_positive

    # Get rule function
    if hasattr(rule, 'eval'):
        rule_fn = rule.eval
    else:
        rule_fn = rule

    positives = []
    negatives = []
    attempts = 0

    while (len(positives) < n_positive or len(negatives) < n_negative) and attempts < max_attempts:
        hand = sample_hand(hand_size)
        try:
            result = rule_fn(hand)
        except Exception:
            attempts += 1
            continue

        attempts += 1

        if result and len(positives) < n_positive:
            positives.append((hand, True))
        elif not result and len(negatives) < n_negative:
            negatives.append((hand, False))

    # Check if we have enough of each class
    is_valid = len(positives) >= min_per_class and len(negatives) >= min_per_class

    # Combine and shuffle
    examples = positives + negatives
    random.shuffle(examples)

    return examples, is_valid


def create_tasks_from_rules(
    rules: List,
    n_examples: int = 50,
    n_holdout: int = 20,
    hand_size: int = 6,  # Changed from 5 to 6
    balanced: bool = True,  # Use balanced sampling
    min_per_class: int = 5
) -> List[Task]:
    """
    Create tasks from catalogue rules with balanced sampling.

    Args:
        rules: List of Rule objects from catalogue.py
        n_examples: Number of training examples per task
        n_holdout: Number of holdout examples per task
        hand_size: Number of cards per hand (default 6)
        balanced: If True, use balanced sampling (50% True, 50% False)
        min_per_class: Minimum examples per class (rules that can't meet this are skipped)

    Returns:
        List of Task objects
    """
    import random
    tasks = []
    skipped = []

    for rule in rules:
        if hasattr(rule, 'eval'):
            rule_fn = rule.eval
            rule_name = rule.id
        else:
            rule_fn = rule
            rule_name = getattr(rule_fn, '__name__', str(rule_fn))

        if balanced:
            # Use balanced sampling for training examples
            examples, train_valid = sample_balanced_examples(
                rule, n_examples, hand_size, min_per_class=min_per_class
            )
            # Use balanced sampling for holdout examples
            holdout, holdout_valid = sample_balanced_examples(
                rule, n_holdout, hand_size, min_per_class=min_per_class // 2 or 1
            )

            if not train_valid:
                skipped.append(rule_name)
                continue
        else:
            # Unbalanced random sampling (original behavior)
            examples = []
            for _ in range(n_examples):
                hand = sample_hand(hand_size)
                try:
                    result = rule_fn(hand)
                    examples.append((hand, result))
                except Exception:
                    continue

            holdout = []
            for _ in range(n_holdout):
                hand = sample_hand(hand_size)
                try:
                    result = rule_fn(hand)
                    holdout.append((hand, result))
                except Exception:
                    continue

        if len(examples) >= n_examples // 2:  # Need at least half
            task = Task(
                name=rule_name,
                request_type=arrow(HAND, BOOL),
                examples=examples,
                holdout=holdout,
                rule_fn=rule_fn
            )
            tasks.append(task)

    if skipped:
        print(f"Skipped {len(skipped)} rules due to extreme base rates: {skipped[:5]}...")

    return tasks


# ============================================================================
# EVALUATION
# ============================================================================

def eval_program_on_hand(program, hand):
    """Evaluate a program on a hand of cards."""
    try:
        fn = program.evaluate([])
        result = fn(hand)
        return result
    except Exception:
        return None


def verify_on_holdout(program, holdout: List[Tuple[Any, bool]]) -> Tuple[bool, float]:
    """Verify a program on held-out examples. Returns (passed, accuracy)."""
    if not holdout:
        return True, 1.0

    correct = 0
    for hand, expected in holdout:
        result = eval_program_on_hand(program, hand)
        if result == expected:
            correct += 1

    accuracy = correct / len(holdout)
    passed = accuracy == 1.0
    return passed, accuracy


# ============================================================================
# ITERATION RESULT
# ============================================================================

@dataclass
class IterationResult:
    """Results from a single wake-sleep iteration."""
    iteration: int

    # Configuration for this iteration
    enumeration_budget: int = 0
    max_depth: int = 0
    timeout_per_task: float = 0.0

    # Wake results
    tasks_solved: int = 0
    total_tasks: int = 0
    programs_enumerated: int = 0
    enumeration_time: float = 0.0

    # Per-task details
    task_details: Dict[str, Dict] = field(default_factory=dict)

    # Compression results
    new_inventions: List[str] = field(default_factory=list)
    grammar_size: int = 0

    # Recognition results
    recognition_loss: float = 0.0
    recognition_train_time: float = 0.0

    # Dreaming results
    dreams_generated: int = 0

    # Timing
    iteration_time: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================================
# EXPERIMENT RUNNER
# ============================================================================

class FactorialExperimentRunner:
    """Runner for a single experimental condition."""

    def __init__(self, config: ExperimentConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger

        # Will be initialized in run()
        self.grammar = None
        self.tasks = None
        self.recognition_model = None
        self.dreamer = None
        self.results = None

    def _init_grammar(self):
        """Initialize grammar from primitive variant."""
        self.logger.info(f"Building grammar: {self.config.primitive_variant}")
        self.grammar = build_grammar_for_variant(self.config.primitive_variant)
        self.logger.info(f"  Grammar size: {len(self.grammar)} primitives")

    def _init_tasks(self):
        """Initialize tasks from rules."""
        if self.config.use_pretraining_rules:
            self.logger.info("Creating tasks from PRETRAINING rules (easier)...")
            rules = get_easy_pretraining_rules()  # 22 level-1 rules
            self.logger.info(f"  Using {len(rules)} easy pretraining rules")
        else:
            self.logger.info("Creating tasks from experimental rules (catalogue.py)...")
            rules = list(ALL_RULES)  # 57 experimental rules
            self.logger.info(f"  Using {len(rules)} experimental rules from catalogue.py")

        if self.config.max_tasks:
            rules = rules[:self.config.max_tasks]

        self.tasks = create_tasks_from_rules(
            rules,
            n_examples=self.config.n_examples_per_task,
            n_holdout=self.config.n_holdout_per_task,
            hand_size=6,  # 6 cards per hand
            balanced=True,  # Use balanced sampling (50% True, 50% False)
            min_per_class=5
        )
        self.logger.info(f"  Created {len(self.tasks)} tasks with balanced sampling (6-card hands)")

    def _init_recognition(self):
        """Initialize recognition model."""
        self.logger.info(f"Initializing {self.config.recognition_type} recognition model...")

        if self.config.recognition_type == 'gru':
            self.recognition_model = NeuralRecognitionModel(
                self.grammar,
                hidden_dim=self.config.recognition_hidden_dim,
                learning_rate=self.config.recognition_lr
            )
        else:
            self.recognition_model = ContrastiveRecognitionModel(
                self.grammar,
                card_out=self.config.recognition_hidden_dim // 4,
                pred_hidden=self.config.recognition_hidden_dim // 2,
                learning_rate=self.config.recognition_lr
            )

        n_params = sum(p.numel() for p in self.recognition_model.parameters())
        self.logger.info(f"  Model parameters: {n_params:,}")

    def _init_dreamer(self):
        """Initialize dreamer with 6-card hands."""
        self.logger.info(f"Initializing {self.config.dream_strategy} dreamer...")

        # Use 6-card hands for dream generation
        def sample_hand_6():
            return sample_hand(6)

        def sample_card_fn():
            return sample_hand(1)[0]

        self.dreamer = ConfigurableDreamer(
            grammar=self.grammar,
            eval_fn=eval_program_on_hand,
            sample_hand_fn=sample_hand_6,  # 6-card hands for dreams
            sample_card_fn=sample_card_fn,
            strategy=self.config.dream_strategy
        )

    def _run_wake_phase(self, iteration: int) -> Tuple[Dict[str, Frontier], IterationResult]:
        """Run the WAKE phase: enumerate programs to solve tasks (PARALLEL)."""
        result = IterationResult(iteration=iteration)

        budget = self.config.get_budget_for_iteration(iteration)
        max_depth = self.config.get_depth_for_iteration(iteration)
        timeout = self.config.get_timeout_for_iteration(iteration)

        result.enumeration_budget = budget
        result.max_depth = max_depth
        result.timeout_per_task = timeout
        result.total_tasks = len(self.tasks)

        self.logger.info(f"  WAKE: budget={budget:,}, depth={max_depth}, timeout={timeout:.0f}s, workers={N_WORKERS}")

        # Step 1: Pre-compute recognition model predictions for all tasks (neural network in main thread)
        # We use get_primitive_log_probs_dict() to get serializable dictionaries (not Grammar objects)
        # which can be passed to worker processes without pickling issues.
        task_log_probs = {}

        # Calculate blend factor based on iteration progress (0.3 → 0.8)
        # Early iterations trust prior more, later iterations trust neural predictions more
        blend_factor = 0.3 + (0.8 - 0.3) * ((iteration - 1) / max(1, self.config.max_iterations - 1))

        if self.recognition_model is not None and iteration > 1:
            self.logger.info(f"    Using recognition guidance with blend_factor={blend_factor:.2f}")
            for task in self.tasks:
                try:
                    task_log_probs[task.name] = self.recognition_model.get_primitive_log_probs_dict(task)
                except Exception as e:
                    self.logger.debug(f"    Recognition failed for {task.name}: {e}")
                    task_log_probs[task.name] = None
        else:
            self.logger.info("    Using base grammar (no recognition guidance)")
            for task in self.tasks:
                task_log_probs[task.name] = None

        # Step 2: Build worker arguments for all tasks
        # Note: We pass primitive_variant to rebuild grammar, plus recognition predictions
        worker_args = []
        for task in self.tasks:
            worker_args.append({
                'task_name': task.name,
                'examples': task.examples,
                'holdout': task.holdout,
                'request_type': task.request_type,
                'primitive_variant': self.config.primitive_variant,
                'budget': budget,
                'max_depth': max_depth,
                'timeout': timeout,
                'keep_top_k': self.config.keep_top_k,
                # Recognition integration: pass predicted log-probs and blend factor
                'predicted_log_probs': task_log_probs[task.name],
                'blend_factor': blend_factor,
            })

        # Step 3: Run enumeration in parallel
        frontiers = {}
        enum_start = time.time()
        total_programs = 0
        completed = 0

        with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
            # Submit all tasks - use dictionary keying to avoid result scrambling
            futures = {executor.submit(_enumerate_task_worker, args): args['task_name']
                       for args in worker_args}

            # Collect results as they complete
            for future in as_completed(futures):
                task_name = futures[future]
                completed += 1

                try:
                    task_result = future.result()

                    # Reconstruct frontier from worker result
                    frontier = Frontier(
                        task_name=task_result['task_name'],
                        request_type=next(t.request_type for t in self.tasks if t.name == task_name),
                        max_size=self.config.keep_top_k
                    )

                    # Build primitives dict for parsing program strings
                    primitives_dict = {str(p): p for p in self.grammar.primitives()}

                    for entry_data in task_result['frontier_entries']:
                        # Parse program from string (we can't pass Program objects through
                        # multiprocessing because they contain unpicklable lambda functions)
                        program = parse_program(entry_data['program_str'], primitives_dict)
                        enum_result = EnumerationResult(
                            program=program,
                            log_probability=entry_data['log_probability'],
                            log_likelihood=entry_data['log_likelihood'],
                            description_length=entry_data['description_length'],
                            programs_enumerated=entry_data['programs_enumerated'],
                            partial_programs_explored=entry_data['partial_programs_explored'],
                            time_seconds=entry_data['time_seconds']
                        )
                        frontier.add(enum_result)

                    frontiers[task_name] = frontier
                    total_programs += task_result['programs_enumerated']

                    # Record task details
                    result.task_details[task_name] = {
                        'solved': task_result['solved'],
                        'programs_enumerated': task_result['programs_enumerated'],
                        'time_seconds': task_result['time_seconds'],
                        'solution': task_result['solution_str'],
                        'solution_size': task_result['solution_size'],
                    }

                    if task_result['solved']:
                        result.tasks_solved += 1

                except Exception as e:
                    self.logger.debug(f"    Worker error for {task_name}: {e}")
                    # Create empty frontier for failed task
                    frontier = Frontier(
                        task_name=task_name,
                        request_type=next(t.request_type for t in self.tasks if t.name == task_name),
                        max_size=self.config.keep_top_k
                    )
                    frontiers[task_name] = frontier
                    result.task_details[task_name] = {
                        'solved': False,
                        'programs_enumerated': 0,
                        'time_seconds': 0,
                        'solution': None,
                        'solution_size': None,
                    }

                # Progress logging (every 10 tasks)
                if completed % 10 == 0:
                    self.logger.info(f"    Progress: {completed}/{len(self.tasks)} tasks, "
                                   f"{result.tasks_solved} solved")

        result.programs_enumerated = total_programs
        result.enumeration_time = time.time() - enum_start

        self.logger.info(f"    Solved: {result.tasks_solved}/{result.total_tasks} tasks")
        self.logger.info(f"    Programs: {total_programs:,} in {result.enumeration_time:.1f}s")

        return frontiers, result

    def _run_compression_phase(self, frontiers: Dict[str, Frontier], result: IterationResult):
        """Run SLEEP-COMPRESS: find reusable abstractions."""
        if not self.config.use_compression or result.tasks_solved == 0:
            return

        self.logger.info("  SLEEP-COMPRESS: Finding abstractions...")

        # Convert frontiers to compression format
        frontier_list = []
        for task_name, frontier in frontiers.items():
            if not frontier.empty:
                frontier_list.append([
                    (entry.program, entry.log_likelihood)
                    for entry in frontier.entries
                ])

        if not frontier_list:
            return

        try:
            compression_result = compress_frontiers(
                self.grammar,
                frontier_list,
                max_inventions=self.config.max_inventions_per_iteration,
                min_savings=self.config.min_compression_savings
            )

            self.grammar = compression_result.new_grammar
            result.new_inventions = [str(inv) for inv in compression_result.new_inventions]
            result.grammar_size = len(self.grammar)

            # Update recognition model vocabulary
            if self.recognition_model is not None:
                for inv in compression_result.new_inventions:
                    self.recognition_model.add_invention(inv)

            self.logger.info(f"    Found {len(result.new_inventions)} new abstractions")
            for inv in result.new_inventions[:3]:  # Show first 3
                self.logger.info(f"      {inv[:60]}...")

        except Exception as e:
            self.logger.warning(f"    Compression failed: {e}")

        result.grammar_size = len(self.grammar)

    def _run_recognition_phase(self, frontiers: Dict[str, Frontier], result: IterationResult):
        """Run SLEEP-RECOGNIZE: train recognition model."""
        if self.recognition_model is None or result.tasks_solved == 0:
            return

        self.logger.info("  SLEEP-RECOGNIZE: Training recognition model...")

        train_start = time.time()

        # Get solved tasks
        solved_tasks = [t for t in self.tasks
                       if frontiers.get(t.name) and not frontiers[t.name].empty]

        try:
            if hasattr(self.recognition_model, 'train_on_frontiers'):
                loss = self.recognition_model.train_on_frontiers(
                    tasks=solved_tasks,
                    frontiers=frontiers,
                    epochs=self.config.recognition_epochs
                )
                result.recognition_loss = float(loss) if loss is not None else 0.0

            result.recognition_train_time = time.time() - train_start
            self.logger.info(f"    Training loss: {result.recognition_loss:.4f}")

        except Exception as e:
            self.logger.warning(f"    Recognition training failed: {e}")

    def _run_dream_phase(self, frontiers: Dict[str, Frontier], result: IterationResult):
        """Run SLEEP-DREAM: generate dreamed examples."""
        if self.dreamer is None or self.config.dreams_per_iteration == 0:
            return

        self.logger.info("  SLEEP-DREAM: Generating dreams...")

        try:
            # Generate dreams using the dreamer's grammar
            dreams = self.dreamer.generate_dreams(
                request_type=arrow(HAND, BOOL),
                n_dreams=self.config.dreams_per_iteration,
                n_examples_per_dream=self.config.n_examples_per_dream
            )
            result.dreams_generated = len(dreams)
            self.logger.info(f"    Generated {result.dreams_generated} dreams")

        except Exception as e:
            self.logger.warning(f"    Dreaming failed: {e}")

    def _save_checkpoint(self, iteration: int, frontiers: Dict[str, Frontier]):
        """Save checkpoint after each iteration."""
        run_dir = self.config.run_dir()
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save results JSON
        results_path = run_dir / "results.json"
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        # Save model checkpoint
        model_path = run_dir / f"model_iter_{iteration}.pt"
        if self.recognition_model is not None:
            try:
                self.recognition_model.save(str(model_path))
            except Exception as e:
                self.logger.debug(f"Could not save model: {e}")

        # Save grammar info (not the full grammar - lambdas can't be pickled)
        grammar_info_path = run_dir / f"grammar_info_iter_{iteration}.json"
        grammar_info = {
            'size': len(self.grammar),
            'primitives': [str(p.program) for p in self.grammar.productions],
            'iteration': iteration,
        }
        with open(grammar_info_path, 'w') as f:
            json.dump(grammar_info, f, indent=2)

        self.logger.info(f"  Checkpoint saved to {run_dir}")

    def run(self) -> Dict:
        """Run the complete experiment for this condition."""
        self.logger.info(f"\n{'='*70}")
        self.logger.info(f"RUNNING: {self.config.condition_name()} (run {self.config.run_id})")
        self.logger.info(f"{'='*70}")

        # Initialize components
        self._init_grammar()
        self._init_tasks()
        self._init_recognition()
        self._init_dreamer()

        # Initialize results
        self.results = {
            'config': asdict(self.config),
            'start_time': datetime.now().isoformat(),
            'condition': self.config.condition_name(),
            'iterations': [],
        }

        # Run iterations
        for iteration in range(1, self.config.max_iterations + 1):
            iter_start = time.time()
            self.logger.info(f"\n--- Iteration {iteration}/{self.config.max_iterations} ---")

            # WAKE
            frontiers, iter_result = self._run_wake_phase(iteration)

            # SLEEP-COMPRESS
            self._run_compression_phase(frontiers, iter_result)

            # SLEEP-RECOGNIZE
            self._run_recognition_phase(frontiers, iter_result)

            # SLEEP-DREAM
            self._run_dream_phase(frontiers, iter_result)

            # Update dreamer's grammar
            if self.dreamer:
                self.dreamer.grammar = self.grammar

            # Record iteration time
            iter_result.iteration_time = time.time() - iter_start

            # Store results
            self.results['iterations'].append(asdict(iter_result))

            # Save checkpoint
            self._save_checkpoint(iteration, frontiers)

            self.logger.info(f"  Iteration time: {iter_result.iteration_time:.1f}s")

        # Finalize
        self.results['end_time'] = datetime.now().isoformat()
        self.results['total_time'] = (
            datetime.fromisoformat(self.results['end_time']) -
            datetime.fromisoformat(self.results['start_time'])
        ).total_seconds()

        # Save final results
        run_dir = self.config.run_dir()
        final_path = run_dir / "final_results.json"
        with open(final_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        self.logger.info(f"\nCompleted {self.config.condition_name()} run {self.config.run_id}")
        self.logger.info(f"Total time: {self.results['total_time']:.1f}s")
        self.logger.info(f"Results saved to {final_path}")

        return self.results


# ============================================================================
# MAIN
# ============================================================================

def setup_logger(log_file: Optional[Path] = None) -> logging.Logger:
    """Set up logging."""
    logger = logging.getLogger('factorial_experiment')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(console)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    return logger


def run_smoke_test(logger: logging.Logger) -> bool:
    """Run a quick smoke test to validate everything works."""
    logger.info("=" * 70)
    logger.info("SMOKE TEST: Validating experiment components")
    logger.info("=" * 70)

    # Test 1: Primitive variants
    logger.info("\n1. Testing primitive variants...")
    from experiments.primitive_variants import validate_all_variants
    if not validate_all_variants():
        logger.error("Primitive variants validation failed!")
        return False
    logger.info("   ✅ Primitive variants OK")

    # Test 2: Recognition models
    logger.info("\n2. Testing recognition models...")
    try:
        grammar = build_grammar_for_variant('lean')

        gru = NeuralRecognitionModel(grammar, hidden_dim=32)
        contrastive = ContrastiveRecognitionModel(grammar, card_out=16, pred_hidden=32)

        # Test add_invention
        from dreamcoder_core.program import Invented, Abstraction, Index
        fake_inv = Invented(Abstraction(Index(0)))
        gru.add_invention(fake_inv)
        contrastive.add_invention(fake_inv)

        logger.info("   ✅ Recognition models OK")
    except Exception as e:
        logger.error(f"   ❌ Recognition models failed: {e}")
        return False

    # Test 3: Dreamers
    logger.info("\n3. Testing dreamers...")
    try:
        def sample_card_fn():
            return sample_hand(1)[0]

        def sample_hand_6():
            return sample_hand(6)

        for strategy in ['standard', 'balanced', 'contrastive']:
            dreamer = ConfigurableDreamer(
                grammar=grammar,
                eval_fn=eval_program_on_hand,
                sample_hand_fn=sample_hand_6,  # 6-card hands
                sample_card_fn=sample_card_fn,
                strategy=strategy
            )
        logger.info("   ✅ Dreamers OK (6-card hands)")
    except Exception as e:
        logger.error(f"   ❌ Dreamers failed: {e}")
        return False

    # Test 4: Task creation with catalogue rules
    logger.info("\n4. Testing task creation with catalogue rules...")
    try:
        rules = list(ALL_RULES)[:3]  # First 3 experimental rules
        tasks = create_tasks_from_rules(rules, n_examples=10, n_holdout=5, hand_size=6, balanced=True)
        if len(tasks) < 1:
            raise ValueError("No tasks created")
        logger.info(f"   ✅ Task creation OK ({len(tasks)} tasks)")
    except Exception as e:
        logger.error(f"   ❌ Task creation failed: {e}")
        return False

    # Test 5: Mini enumeration
    logger.info("\n5. Testing enumeration...")
    try:
        from dreamcoder_core.enumeration import TopDownEnumerator

        enumerator = TopDownEnumerator(grammar, max_depth=4, max_programs=100)
        count = 0
        for program, log_prob in enumerator.enumerate(arrow(HAND, BOOL), max_cost=20.0, timeout_seconds=5.0):
            count += 1
            if count >= 10:
                break
        logger.info(f"   ✅ Enumeration OK ({count} programs)")
    except Exception as e:
        logger.error(f"   ❌ Enumeration failed: {e}")
        return False

    # Test 6: Mini run (1 iteration, 1 task)
    logger.info("\n6. Testing mini experiment run...")
    try:
        config = ExperimentConfig(
            recognition_type='gru',
            dream_strategy='balanced',
            primitive_variant='lean',
            run_id=0,
            max_iterations=1,
            base_enumeration_budget=1000,
            base_timeout=10.0,
            max_tasks=2,
            use_easy_rules_only=True,
            dreams_per_iteration=5,
            recognition_epochs=1,
            results_dir='results_smoke_test',
        )

        runner = FactorialExperimentRunner(config, logger)
        results = runner.run()

        if results and 'iterations' in results and len(results['iterations']) == 1:
            logger.info("   ✅ Mini run OK")
        else:
            raise ValueError("Mini run produced invalid results")

    except Exception as e:
        logger.error(f"   ❌ Mini run failed: {e}")
        traceback.print_exc()
        return False

    logger.info("\n" + "=" * 70)
    logger.info("✅ ALL SMOKE TESTS PASSED")
    logger.info("=" * 70)
    return True


def main():
    parser = argparse.ArgumentParser(description='2×3×3 Factorial Experiment Runner')

    parser.add_argument('--recognition', choices=['gru', 'contrastive'],
                       help='Recognition model (single condition)')
    parser.add_argument('--dreams', choices=['standard', 'balanced', 'contrastive'],
                       help='Dream strategy (single condition)')
    parser.add_argument('--primitives', choices=['lean', 'lean_plus_fold', 'minimal'],
                       help='Primitive variant (single condition)')
    parser.add_argument('--run-id', type=int, default=1,
                       help='Run ID (default: 1)')
    parser.add_argument('--max-iterations', type=int, default=5,
                       help='Max iterations per condition (default: 5)')
    parser.add_argument('--results-dir', type=str, default='results_factorial',
                       help='Results directory (default: results_factorial)')
    parser.add_argument('--smoke-test', action='store_true',
                       help='Run smoke test to validate components')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show conditions without running')
    parser.add_argument('--easy-rules-only', action='store_true', default=True,
                       help='Use only easy rules (default: True for ~1hr timing)')
    parser.add_argument('--full-rules', action='store_true',
                       help='Use all rules instead of easy rules only')
    parser.add_argument('--pretraining-rules', action='store_true',
                       help='Use easier pretraining rules (22 rules) instead of experimental rules')
    parser.add_argument('--max-tasks', type=int, default=None,
                       help='Maximum number of tasks (default: all)')

    args = parser.parse_args()

    # Set up results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Set up logger
    log_file = results_dir / f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger(log_file)

    # Dry run: show conditions
    if args.dry_run:
        print("\n2×3×3 Factorial Experiment Conditions:")
        print("-" * 60)
        for i, cond in enumerate(CONDITIONS, 1):
            name = f"{cond['recognition']}_{cond['dreams']}_{cond['primitives']}"
            print(f"  {i:2}. {name}")
        print(f"\nTotal: {len(CONDITIONS)} conditions")
        return

    # Smoke test
    if args.smoke_test:
        success = run_smoke_test(logger)
        sys.exit(0 if success else 1)

    # Determine conditions to run
    if args.recognition and args.dreams and args.primitives:
        conditions = [{
            'recognition': args.recognition,
            'dreams': args.dreams,
            'primitives': args.primitives,
        }]
    else:
        conditions = CONDITIONS

    # Run experiments
    logger.info("=" * 70)
    logger.info("2×3×3 FACTORIAL EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Conditions: {len(conditions)}")
    logger.info(f"Iterations per condition: {args.max_iterations}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Log file: {log_file}")

    all_results = []
    failed_conditions = []

    for cond_idx, cond in enumerate(conditions, 1):
        logger.info(f"\n[{cond_idx}/{len(conditions)}] Starting condition...")

        config = ExperimentConfig(
            recognition_type=cond['recognition'],
            dream_strategy=cond['dreams'],
            primitive_variant=cond['primitives'],
            run_id=args.run_id,
            max_iterations=args.max_iterations,
            results_dir=str(results_dir),
            use_easy_rules_only=not args.full_rules,
            use_pretraining_rules=args.pretraining_rules,
            max_tasks=args.max_tasks,
        )

        try:
            runner = FactorialExperimentRunner(config, logger)
            result = runner.run()
            all_results.append(result)

        except Exception as e:
            logger.error(f"Condition {config.condition_name()} failed: {e}")
            traceback.print_exc()
            failed_conditions.append(config.condition_name())

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Completed: {len(all_results)}/{len(conditions)} conditions")
    if failed_conditions:
        logger.info(f"Failed: {failed_conditions}")
    logger.info(f"Results saved to: {results_dir}")


if __name__ == '__main__':
    main()
