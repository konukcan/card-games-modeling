#!/usr/bin/env python3
"""
Curriculum Learning Runner for DreamCoder

Processes rules in batches, where knowledge (library + recognition model)
transfers from one batch to the next. This demonstrates how learning
accumulates across tasks.

Key features:
- Batched processing with knowledge transfer
- Comprehensive tracking of all metrics
- HTML report generation with interactive charts
- Progressive library growth visualization
"""

import sys
import time
import json
import copy
import random
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from dreamcoder_core.full_dreamcoder import (
    FullDreamCoder, Task, TaskFrontier, SolutionEntry,
    create_tasks_from_rules, make_eval_fn, RecognitionModel
)
from dreamcoder_core.card_primitives import build_card_grammar
from dreamcoder_core.grammar import Grammar
from dreamcoder_core.enumeration import enumerate_simple
from dreamcoder_core.compression import compress_frontiers
from dreamcoder_core.html_report import generate_html_report
from rules.catalogue import ALL_RULES, RULE_DICT


# ============================================================================
# CURRICULUM TRACKER
# ============================================================================

@dataclass
class BatchResult:
    """Results from processing one batch."""
    batch_num: int
    tasks: List[str]
    tasks_solved: int
    tasks_total: int
    solve_rate: float
    programs_enumerated: int
    time: float
    new_abstractions: List[str]
    grammar_size: int
    task_details: List[Dict]
    recognition_weights: Dict[str, float]
    grammar_probs: Dict[str, float]


@dataclass
class CurriculumTracker:
    """Tracks all data across the curriculum."""
    experiment_name: str = "curriculum"
    start_time: float = 0.0

    # Initial state
    initial_grammar_size: int = 0
    initial_grammar_probs: Dict[str, float] = field(default_factory=dict)

    # Per-batch results
    batches: List[BatchResult] = field(default_factory=list)

    # Cumulative tracking
    all_abstractions: List[Tuple[int, str]] = field(default_factory=list)
    cumulative_solved: int = 0
    cumulative_total: int = 0
    task_results: Dict[str, Dict] = field(default_factory=dict)

    def get_cumulative_data(self) -> Dict:
        """Get cumulative data for reporting."""

        # Grammar changes
        grammar_changes = []
        if self.batches:
            final_probs = self.batches[-1].grammar_probs
            for prim, init_prob in self.initial_grammar_probs.items():
                if prim in final_probs:
                    change = final_probs[prim] - init_prob
                    grammar_changes.append((prim, init_prob, final_probs[prim], change))
            grammar_changes.sort(key=lambda x: -abs(x[3]))

        # Top primitives for chart
        top_prims = [c[0] for c in grammar_changes[:15]]
        initial_probs = [self.initial_grammar_probs.get(p, -4) for p in top_prims]
        final_probs = [self.batches[-1].grammar_probs.get(p, -4) for p in top_prims] if self.batches else initial_probs

        return {
            'total_batches': len(self.batches),
            'total_tasks_solved': self.cumulative_solved,
            'total_tasks': self.cumulative_total,
            'total_abstractions': len(self.all_abstractions),
            'final_grammar_size': self.batches[-1].grammar_size if self.batches else self.initial_grammar_size,
            'total_time': time.time() - self.start_time,
            'total_programs': sum(b.programs_enumerated for b in self.batches),
            'all_abstractions': self.all_abstractions,
            'task_results': self.task_results,
            'grammar_changes': grammar_changes,
            'top_primitives': top_prims,
            'initial_probs': initial_probs,
            'final_probs': final_probs
        }

    def get_batches_data(self) -> List[Dict]:
        """Get batch data for charts."""
        data = []
        cumulative_solved = 0
        cumulative_total = 0

        for batch in self.batches:
            cumulative_solved += batch.tasks_solved
            cumulative_total += batch.tasks_total

            # Calculate recognition divergence (sum of weights)
            rec_divergence = sum(abs(w) for w in batch.recognition_weights.values())

            data.append({
                'batch_num': batch.batch_num,
                'tasks': batch.task_details,
                'tasks_solved': batch.tasks_solved,
                'tasks_total': batch.tasks_total,
                'solve_rate': batch.solve_rate,
                'cumulative_solve_rate': cumulative_solved / cumulative_total if cumulative_total > 0 else 0,
                'programs_enumerated': batch.programs_enumerated,
                'avg_programs_per_task': batch.programs_enumerated / batch.tasks_total if batch.tasks_total > 0 else 0,
                'time': batch.time,
                'new_abstractions': batch.new_abstractions,
                'grammar_size': batch.grammar_size,
                'recognition_divergence': rec_divergence
            })

        return data


# ============================================================================
# CURRICULUM RUNNER
# ============================================================================

class CurriculumRunner:
    """
    Runs DreamCoder in curriculum mode: process batches sequentially,
    transferring learned knowledge between batches.
    """

    def __init__(
        self,
        rules: List,
        batch_size: int = 6,
        iterations_per_batch: int = 2,
        enumeration_budget: int = 100000,
        enumeration_timeout: float = 300.0,
        max_depth: int = 8,
        keep_top_k: int = 5,
        output_dir: str = "results/curriculum",
        verbose: bool = True
    ):
        self.rules = rules
        self.batch_size = batch_size
        self.iterations_per_batch = iterations_per_batch
        self.enumeration_budget = enumeration_budget
        self.enumeration_timeout = enumeration_timeout
        self.max_depth = max_depth
        self.keep_top_k = keep_top_k
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        # State that persists across batches
        self.grammar = build_card_grammar()
        self.recognition = RecognitionModel(self.grammar)
        self.eval_fn = make_eval_fn()

        # Tracking
        self.tracker = CurriculumTracker(experiment_name="curriculum")

    def log(self, msg: str, level: int = 0):
        if self.verbose:
            indent = "  " * level
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] {indent}{msg}", flush=True)

    def create_batches(self) -> List[List]:
        """Split rules into batches."""
        batches = []
        for i in range(0, len(self.rules), self.batch_size):
            batches.append(self.rules[i:i + self.batch_size])
        return batches

    def run(self) -> Dict:
        """Run the full curriculum."""
        self.log("=" * 70)
        self.log("CURRICULUM LEARNING - DREAMCODER")
        self.log("=" * 70)

        self.tracker.start_time = time.time()
        self.tracker.initial_grammar_size = len(self.grammar)
        self.tracker.initial_grammar_probs = {
            str(p.program): p.log_probability
            for p in self.grammar.productions
        }

        batches = self.create_batches()

        self.log(f"Total rules: {len(self.rules)}")
        self.log(f"Batch size: {self.batch_size}")
        self.log(f"Number of batches: {len(batches)}")
        self.log(f"Iterations per batch: {self.iterations_per_batch}")
        self.log(f"Enumeration budget: {self.enumeration_budget:,}")
        self.log("")

        for batch_num, batch_rules in enumerate(batches):
            self.log("=" * 70)
            self.log(f"BATCH {batch_num + 1}/{len(batches)}")
            self.log("=" * 70)
            self.log(f"Rules: {[r.id for r in batch_rules]}")

            batch_result = self._process_batch(batch_num, batch_rules)
            self.tracker.batches.append(batch_result)

            # Update cumulative
            self.tracker.cumulative_solved += batch_result.tasks_solved
            self.tracker.cumulative_total += batch_result.tasks_total

            self.log(f"\nBatch {batch_num + 1} complete:")
            self.log(f"  Solved: {batch_result.tasks_solved}/{batch_result.tasks_total}", 1)
            self.log(f"  New abstractions: {len(batch_result.new_abstractions)}", 1)
            self.log(f"  Grammar size: {batch_result.grammar_size}", 1)
            self.log(f"  Time: {batch_result.time:.1f}s", 1)

        # Final summary
        total_time = time.time() - self.tracker.start_time

        self.log("")
        self.log("=" * 70)
        self.log("CURRICULUM COMPLETE")
        self.log("=" * 70)
        self.log(f"Total time: {total_time:.1f}s ({total_time/60:.1f} min)")
        self.log(f"Total solved: {self.tracker.cumulative_solved}/{self.tracker.cumulative_total}")
        self.log(f"Total abstractions: {len(self.tracker.all_abstractions)}")
        self.log(f"Final grammar: {len(self.grammar)} primitives")

        # Generate reports
        self._generate_reports()

        return self.tracker.get_cumulative_data()

    def _process_batch(self, batch_num: int, batch_rules: List) -> BatchResult:
        """Process a single batch of rules."""
        batch_start = time.time()

        # Create tasks for this batch
        tasks = create_tasks_from_rules(batch_rules, n_examples=20, seed=42 + batch_num)

        # Create frontiers
        frontiers = {t.name: TaskFrontier(t, max_size=self.keep_top_k) for t in tasks}

        total_programs = 0
        tasks_solved = 0
        task_details = []

        # Run iterations on this batch
        for iteration in range(self.iterations_per_batch):
            self.log(f"\n  Iteration {iteration + 1}/{self.iterations_per_batch}")

            iter_programs = 0

            for task in tasks:
                frontier = frontiers[task.name]

                # Skip if already have max solutions
                if frontier.n_solutions >= self.keep_top_k and frontier.solved:
                    continue

                # Get task-specific grammar from recognition
                task_grammar = self.grammar
                if batch_num > 0 or iteration > 0:
                    task_grammar = self.recognition.predict_grammar_weights(task)

                # Enumerate
                programs_tried = 0
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
                        correct = sum(
                            1 for inp, exp in task.examples
                            if self.eval_fn(program, inp) == exp
                        )

                        if correct == len(task.examples):
                            entry = SolutionEntry(
                                program=program,
                                log_probability=log_prob,
                                log_likelihood=0.0,
                                programs_enumerated=programs_tried,
                                time_found=time.time() - enum_start
                            )
                            frontier.add(entry)

                            if frontier.n_solutions >= self.keep_top_k:
                                break
                    except:
                        pass

                frontier.total_programs_searched += programs_tried
                iter_programs += programs_tried

            total_programs += iter_programs

            # Compression
            all_frontiers = []
            for frontier in frontiers.values():
                if frontier.n_solutions > 0:
                    programs_with_ll = [(e.program, e.log_likelihood) for e in frontier.entries]
                    all_frontiers.append(programs_with_ll)

            new_abstractions = []
            if all_frontiers:
                result = compress_frontiers(
                    self.grammar,
                    all_frontiers,
                    max_inventions=3,
                    min_savings=2.0,
                    use_anti_unification=True
                )

                if result.new_inventions:
                    self.grammar = result.new_grammar
                    new_abstractions = [str(inv) for inv in result.new_inventions]
                    self.recognition.grammar = self.grammar

                    for a in new_abstractions:
                        self.tracker.all_abstractions.append((batch_num, a))

                    self.log(f"    New abstractions: {len(new_abstractions)}", 1)

            # Recognition training
            for task in tasks:
                frontier = frontiers[task.name]
                if frontier.solved:
                    self.recognition.train_on_frontier(task, frontier)

            # Grammar weight update
            if all_frontiers:
                self.grammar = self.grammar.inside_outside_update(all_frontiers)

        # Collect results
        for task in tasks:
            frontier = frontiers[task.name]
            solved = frontier.solved
            best = frontier.best

            if solved:
                tasks_solved += 1

            detail = {
                'name': task.name,
                'solved': solved,
                'programs': frontier.total_programs_searched,
                'solutions': frontier.n_solutions
            }
            task_details.append(detail)

            # Update cumulative task tracking
            if task.name not in self.tracker.task_results:
                self.tracker.task_results[task.name] = {
                    'solved': solved,
                    'batch_solved': batch_num + 1 if solved else None,
                    'programs_to_solve': best.programs_enumerated if best else frontier.total_programs_searched,
                    'best_program': str(best.program) if best else None,
                    'description_length': f"{best.description_length:.1f}" if best else None
                }
            elif solved and not self.tracker.task_results[task.name]['solved']:
                self.tracker.task_results[task.name].update({
                    'solved': True,
                    'batch_solved': batch_num + 1,
                    'programs_to_solve': best.programs_enumerated if best else 0,
                    'best_program': str(best.program) if best else None,
                    'description_length': f"{best.description_length:.1f}" if best else None
                })

        # Get current recognition weights
        rec_weights = {}
        all_weights = defaultdict(float)
        for (feat, prim), w in self.recognition._weights.items():
            all_weights[prim] += abs(w)
        rec_weights = dict(sorted(all_weights.items(), key=lambda x: -x[1])[:20])

        # Get current grammar probs
        grammar_probs = {
            str(p.program): p.log_probability
            for p in self.grammar.productions
        }

        return BatchResult(
            batch_num=batch_num,
            tasks=[t.name for t in tasks],
            tasks_solved=tasks_solved,
            tasks_total=len(tasks),
            solve_rate=tasks_solved / len(tasks) if tasks else 0,
            programs_enumerated=total_programs,
            time=time.time() - batch_start,
            new_abstractions=[a for b, a in self.tracker.all_abstractions if b == batch_num],
            grammar_size=len(self.grammar),
            task_details=task_details,
            recognition_weights=rec_weights,
            grammar_probs=grammar_probs
        )

    def _generate_reports(self):
        """Generate all reports."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

        # HTML report
        html_path = self.output_dir / f"report_{timestamp}.html"
        generate_html_report(
            experiment_name=self.tracker.experiment_name,
            batches_data=self.tracker.get_batches_data(),
            cumulative_data=self.tracker.get_cumulative_data(),
            output_path=str(html_path)
        )
        self.log(f"\nHTML report: {html_path}")

        # JSON data
        json_path = self.output_dir / f"data_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump({
                'cumulative': self.tracker.get_cumulative_data(),
                'batches': self.tracker.get_batches_data()
            }, f, indent=2, default=str)
        self.log(f"JSON data: {json_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run DreamCoder Curriculum Learning")

    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--budget", type=int, default=100000)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--output", default="results/curriculum")
    parser.add_argument("--mode", choices=["test", "easy", "all"], default="test")

    args = parser.parse_args()

    # Select rules based on mode
    if args.mode == "test":
        # 2 batches of 3 for quick testing
        rule_ids = [
            'Uniform_color', 'Suits_palindrome', 'Colors_palindrome',
            'Ranks_palindrome', 'Sorted_by_rank', 'Shift2_plus3'
        ]
    elif args.mode == "easy":
        # Rules that were solved in overnight run
        rule_ids = [
            'Uniform_color', 'Suits_palindrome', 'Colors_palindrome',
            'Ranks_palindrome', 'Sorted_by_rank', 'Shift2_plus3',
            'Ends_same_color', 'Ends_same_suit',
            'Halves_radial_nonincreasing', 'Global_radial_no_dominance'
        ]
    else:
        # All rules
        rule_ids = [r.id for r in ALL_RULES]

    rules = [RULE_DICT[rid] for rid in rule_ids if rid in RULE_DICT]

    runner = CurriculumRunner(
        rules=rules,
        batch_size=args.batch_size,
        iterations_per_batch=args.iterations,
        enumeration_budget=args.budget,
        enumeration_timeout=args.timeout,
        output_dir=args.output,
        verbose=True
    )

    runner.run()


if __name__ == "__main__":
    main()
