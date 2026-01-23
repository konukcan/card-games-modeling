#!/usr/bin/env python3
"""
DreamCoder Visualization and Reporting Module

Generates comprehensive, interpretable reports showing:
1. Learning curves and performance evolution
2. Library growth and abstraction discovery
3. Recognition model evolution
4. Per-task solution tracking
5. Grammar distribution changes

All visualizations are text-based for maximum portability.
"""

import json
import math
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict


# ============================================================================
# DATA COLLECTORS
# ============================================================================

@dataclass
class IterationSnapshot:
    """Complete snapshot of system state at one iteration."""
    iteration: int
    timestamp: float  # seconds from start

    # Performance
    tasks_solved: int
    tasks_total: int
    programs_enumerated: int

    # Library state
    grammar_size: int
    new_abstractions: List[str]
    abstraction_sources: List[Dict]  # What programs they came from

    # Per-task details
    task_solutions: Dict[str, List[Dict]]  # task_name -> list of solutions
    task_programs_tried: Dict[str, int]

    # Recognition model state
    recognition_weights: Dict[str, float]  # top weighted primitives
    recognition_predictions: Dict[str, List[str]]  # task -> predicted primitives

    # Grammar distribution
    grammar_distribution: Dict[str, float]  # primitive -> log_prob


@dataclass
class ExperimentTracker:
    """
    Tracks all metrics throughout a DreamCoder run for visualization.

    Use this class to collect data during the run, then generate reports.
    """

    # Configuration
    experiment_name: str = "experiment"
    start_time: float = 0.0

    # Snapshots per iteration
    snapshots: List[IterationSnapshot] = field(default_factory=list)

    # Initial state
    initial_grammar_size: int = 0
    initial_grammar_dist: Dict[str, float] = field(default_factory=dict)

    # Cumulative tracking
    all_abstractions: List[Tuple[int, str, str]] = field(default_factory=list)  # (iter, abstraction, source)
    library_sizes: List[int] = field(default_factory=list)
    solve_rates: List[float] = field(default_factory=list)

    def record_initial_state(self, grammar, tasks):
        """Record initial grammar state."""
        import time
        self.start_time = time.time()
        self.initial_grammar_size = len(grammar)
        self.initial_grammar_dist = {
            str(p.program): p.log_probability
            for p in grammar.productions
        }
        self.library_sizes.append(len(grammar))

    def record_iteration(
        self,
        iteration: int,
        grammar,
        frontiers: Dict,
        new_abstractions: List,
        recognition_model=None,
        tasks=None
    ):
        """Record state after one iteration."""
        import time

        # Task solutions
        task_solutions = {}
        task_programs = {}
        for task_name, frontier in frontiers.items():
            task_solutions[task_name] = [
                {
                    'program': str(e.program),
                    'description_length': e.description_length,
                    'programs_to_find': e.programs_enumerated,
                    'time_found': e.time_found
                }
                for e in frontier.entries
            ]
            task_programs[task_name] = frontier.total_programs_searched

        # Recognition state
        rec_weights = {}
        rec_predictions = {}
        if recognition_model and tasks:
            # Get top weighted primitives
            all_weights = defaultdict(float)
            for (feat, prim), weight in recognition_model._weights.items():
                all_weights[prim] += abs(weight)
            rec_weights = dict(sorted(all_weights.items(), key=lambda x: -x[1])[:20])

            # Get predictions per task
            for task in tasks:
                preds = recognition_model.get_top_predictions(task, 5)
                rec_predictions[task.name] = [p[0] for p in preds]

        # Grammar distribution
        grammar_dist = {
            str(p.program): p.log_probability
            for p in grammar.productions
        }

        # Abstraction sources (simplified)
        abstraction_sources = [
            {'abstraction': str(a), 'iteration': iteration}
            for a in new_abstractions
        ]

        snapshot = IterationSnapshot(
            iteration=iteration,
            timestamp=time.time() - self.start_time,
            tasks_solved=sum(1 for f in frontiers.values() if f.solved),
            tasks_total=len(frontiers),
            programs_enumerated=sum(task_programs.values()),
            grammar_size=len(grammar),
            new_abstractions=[str(a) for a in new_abstractions],
            abstraction_sources=abstraction_sources,
            task_solutions=task_solutions,
            task_programs_tried=task_programs,
            recognition_weights=rec_weights,
            recognition_predictions=rec_predictions,
            grammar_distribution=grammar_dist
        )

        self.snapshots.append(snapshot)
        self.library_sizes.append(len(grammar))
        self.solve_rates.append(snapshot.tasks_solved / snapshot.tasks_total)

        # Track abstractions
        for a in new_abstractions:
            self.all_abstractions.append((iteration, str(a), "compression"))


# ============================================================================
# REPORT GENERATION
# ============================================================================

class ReportGenerator:
    """Generates comprehensive text-based reports."""

    def __init__(self, tracker: ExperimentTracker):
        self.tracker = tracker
        self.width = 80

    def generate_full_report(self) -> str:
        """Generate complete report with all sections."""
        sections = [
            self._header(),
            self._executive_summary(),
            self._learning_curve(),
            self._library_evolution(),
            self._per_task_analysis(),
            self._recognition_analysis(),
            self._grammar_distribution_analysis(),
            self._detailed_solutions(),
            self._footer()
        ]
        return "\n".join(sections)

    def _header(self) -> str:
        lines = []
        lines.append("=" * self.width)
        lines.append(self._center("DREAMCODER EXPERIMENT REPORT"))
        lines.append(self._center(f"Experiment: {self.tracker.experiment_name}"))
        lines.append(self._center(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"))
        lines.append("=" * self.width)
        return "\n".join(lines)

    def _executive_summary(self) -> str:
        lines = []
        lines.append("\n" + "─" * self.width)
        lines.append(self._center("EXECUTIVE SUMMARY"))
        lines.append("─" * self.width)

        if not self.tracker.snapshots:
            lines.append("No data recorded.")
            return "\n".join(lines)

        final = self.tracker.snapshots[-1]

        lines.append(f"""
┌{'─'*30}┬{'─'*30}┐
│ {'Metric':<28} │ {'Value':<28} │
├{'─'*30}┼{'─'*30}┤
│ {'Total Iterations':<28} │ {len(self.tracker.snapshots):<28} │
│ {'Total Time':<28} │ {final.timestamp:.1f}s{'':<22} │
│ {'Tasks Solved':<28} │ {final.tasks_solved}/{final.tasks_total} ({100*final.tasks_solved/final.tasks_total:.1f}%){'':<10} │
│ {'Programs Enumerated':<28} │ {sum(s.programs_enumerated for s in self.tracker.snapshots):,}{'':<15} │
│ {'Initial Grammar Size':<28} │ {self.tracker.initial_grammar_size:<28} │
│ {'Final Grammar Size':<28} │ {final.grammar_size:<28} │
│ {'Abstractions Learned':<28} │ {len(self.tracker.all_abstractions):<28} │
└{'─'*30}┴{'─'*30}┘
""")
        return "\n".join(lines)

    def _learning_curve(self) -> str:
        lines = []
        lines.append("\n" + "─" * self.width)
        lines.append(self._center("LEARNING CURVE"))
        lines.append("─" * self.width)

        if not self.tracker.snapshots:
            return "\n".join(lines)

        # ASCII chart
        max_rate = 1.0
        chart_height = 10
        chart_width = min(50, len(self.tracker.snapshots) * 8)

        lines.append("\nSolve Rate Over Iterations:")
        lines.append("")

        # Y-axis labels and bars
        for row in range(chart_height, -1, -1):
            threshold = row / chart_height * max_rate
            label = f"{threshold*100:5.0f}% │"
            bar = ""
            for i, snap in enumerate(self.tracker.snapshots):
                rate = snap.tasks_solved / snap.tasks_total
                if rate >= threshold:
                    bar += "████████"
                else:
                    bar += "        "
            lines.append(label + bar[:chart_width])

        # X-axis
        lines.append("      └" + "─" * chart_width)
        x_labels = "        " + "".join(f"Iter {i+1:<3}" for i in range(len(self.tracker.snapshots)))
        lines.append(x_labels[:self.width])

        # Numeric table
        lines.append("\n┌────────┬──────────┬─────────────┬───────────────┬──────────────┐")
        lines.append("│  Iter  │  Solved  │   Programs  │  Abstractions │    Time (s)  │")
        lines.append("├────────┼──────────┼─────────────┼───────────────┼──────────────┤")

        for snap in self.tracker.snapshots:
            lines.append(
                f"│ {snap.iteration+1:^6} │ "
                f"{snap.tasks_solved:>3}/{snap.tasks_total:<3} │ "
                f"{snap.programs_enumerated:>11,} │ "
                f"{len(snap.new_abstractions):^13} │ "
                f"{snap.timestamp:>11.1f}s │"
            )

        lines.append("└────────┴──────────┴─────────────┴───────────────┴──────────────┘")

        return "\n".join(lines)

    def _library_evolution(self) -> str:
        lines = []
        lines.append("\n" + "─" * self.width)
        lines.append(self._center("LIBRARY EVOLUTION"))
        lines.append("─" * self.width)

        # Library size over time
        lines.append("\nGrammar Size: " + " → ".join(str(s) for s in self.tracker.library_sizes))

        # All learned abstractions
        lines.append("\nLearned Abstractions Timeline:")
        lines.append("")

        for iteration, abstraction, source in self.tracker.all_abstractions:
            # Truncate long abstractions
            display = abstraction[:60] + "..." if len(abstraction) > 60 else abstraction
            lines.append(f"  Iter {iteration+1}: {display}")

        if not self.tracker.all_abstractions:
            lines.append("  (No abstractions learned)")

        # Abstraction frequency by iteration
        lines.append("\nAbstractions per Iteration:")
        abstraction_counts = defaultdict(int)
        for iter_num, _, _ in self.tracker.all_abstractions:
            abstraction_counts[iter_num] += 1

        max_count = max(abstraction_counts.values()) if abstraction_counts else 1
        for snap in self.tracker.snapshots:
            count = abstraction_counts.get(snap.iteration, 0)
            bar = "█" * int(count / max_count * 30)
            lines.append(f"  Iter {snap.iteration+1}: {bar} ({count})")

        return "\n".join(lines)

    def _per_task_analysis(self) -> str:
        lines = []
        lines.append("\n" + "─" * self.width)
        lines.append(self._center("PER-TASK ANALYSIS"))
        lines.append("─" * self.width)

        if not self.tracker.snapshots:
            return "\n".join(lines)

        final = self.tracker.snapshots[-1]

        # Sort tasks by solve status, then by programs needed
        task_data = []
        for task_name, solutions in final.task_solutions.items():
            solved = len(solutions) > 0
            programs = final.task_programs_tried.get(task_name, 0)
            best_dl = solutions[0]['description_length'] if solutions else float('inf')
            task_data.append((task_name, solved, programs, best_dl, len(solutions)))

        task_data.sort(key=lambda x: (not x[1], x[2]))  # Solved first, then by programs

        lines.append("\n┌" + "─"*25 + "┬" + "─"*10 + "┬" + "─"*15 + "┬" + "─"*10 + "┬" + "─"*12 + "┐")
        lines.append(f"│ {'Task Name':<23} │ {'Status':<8} │ {'Programs':<13} │ {'Best DL':<8} │ {'Solutions':<10} │")
        lines.append("├" + "─"*25 + "┼" + "─"*10 + "┼" + "─"*15 + "┼" + "─"*10 + "┼" + "─"*12 + "┤")

        for name, solved, programs, dl, n_solutions in task_data:
            status = "✓ SOLVED" if solved else "✗ UNSOLVED"
            dl_str = f"{dl:.1f}" if dl != float('inf') else "N/A"
            name_display = name[:23] if len(name) <= 23 else name[:20] + "..."
            lines.append(
                f"│ {name_display:<23} │ {status:<8} │ {programs:>13,} │ {dl_str:>8} │ {n_solutions:>10} │"
            )

        lines.append("└" + "─"*25 + "┴" + "─"*10 + "┴" + "─"*15 + "┴" + "─"*10 + "┴" + "─"*12 + "┘")

        # Search effort evolution per task
        lines.append("\nSearch Effort Evolution (programs enumerated per iteration):")
        lines.append("")

        for task_name, _, _, _, _ in task_data[:10]:  # Top 10 tasks
            efforts = []
            for snap in self.tracker.snapshots:
                if task_name in snap.task_programs_tried:
                    efforts.append(snap.task_programs_tried[task_name])

            if efforts:
                trend = " → ".join(f"{e:,}" for e in efforts)
                # Calculate speedup
                if len(efforts) > 1 and efforts[0] > 0 and efforts[-1] > 0:
                    if efforts[-1] < efforts[0]:
                        speedup = f" ({efforts[0]/efforts[-1]:.1f}x faster)"
                    else:
                        speedup = ""
                else:
                    speedup = ""
                lines.append(f"  {task_name[:25]:<25}: {trend}{speedup}")

        return "\n".join(lines)

    def _recognition_analysis(self) -> str:
        lines = []
        lines.append("\n" + "─" * self.width)
        lines.append(self._center("RECOGNITION MODEL ANALYSIS"))
        lines.append("─" * self.width)

        if not self.tracker.snapshots:
            return "\n".join(lines)

        # Check if recognition was used
        has_recognition = any(snap.recognition_weights for snap in self.tracker.snapshots)

        if not has_recognition:
            lines.append("\nRecognition model was not enabled for this run.")
            return "\n".join(lines)

        # Evolution of top-weighted primitives
        lines.append("\nTop Primitive Weights Over Iterations:")
        lines.append("(Higher weight = more frequently used in solutions)")
        lines.append("")

        # Collect all primitives that ever had weight
        all_prims = set()
        for snap in self.tracker.snapshots:
            all_prims.update(snap.recognition_weights.keys())

        # Show top 10 by final weight
        final_weights = self.tracker.snapshots[-1].recognition_weights if self.tracker.snapshots else {}
        top_prims = sorted(final_weights.items(), key=lambda x: -x[1])[:10]

        if top_prims:
            lines.append("Top 10 primitives by final weight:")
            for prim, weight in top_prims:
                # Show evolution
                evolution = []
                for snap in self.tracker.snapshots:
                    w = snap.recognition_weights.get(prim, 0)
                    evolution.append(f"{w:.2f}")
                trend = " → ".join(evolution)
                prim_display = prim[:30] if len(prim) <= 30 else prim[:27] + "..."
                lines.append(f"  {prim_display:<30}: {trend}")

        # Recognition predictions accuracy
        lines.append("\nRecognition Predictions per Task (final iteration):")
        final = self.tracker.snapshots[-1]

        for task_name, predictions in list(final.recognition_predictions.items())[:10]:
            pred_str = ", ".join(predictions[:5])
            lines.append(f"  {task_name[:20]:<20}: {pred_str}")

        return "\n".join(lines)

    def _grammar_distribution_analysis(self) -> str:
        lines = []
        lines.append("\n" + "─" * self.width)
        lines.append(self._center("GRAMMAR DISTRIBUTION ANALYSIS"))
        lines.append("─" * self.width)

        if not self.tracker.snapshots:
            return "\n".join(lines)

        lines.append("\nPrimitive Probability Changes (initial → final):")
        lines.append("(Positive change = primitive became more likely)")
        lines.append("")

        initial = self.tracker.initial_grammar_dist
        final = self.tracker.snapshots[-1].grammar_distribution

        # Calculate changes
        changes = []
        for prim in initial:
            if prim in final:
                change = final[prim] - initial[prim]
                changes.append((prim, initial[prim], final[prim], change))

        # Sort by absolute change
        changes.sort(key=lambda x: -abs(x[3]))

        # Show top movers
        lines.append("Top 15 probability changes:")
        lines.append("")
        lines.append(f"  {'Primitive':<35} {'Initial':>10} {'Final':>10} {'Change':>10}")
        lines.append("  " + "─" * 67)

        for prim, init, fin, change in changes[:15]:
            prim_display = prim[:35] if len(prim) <= 35 else prim[:32] + "..."
            change_str = f"+{change:.3f}" if change > 0 else f"{change:.3f}"
            lines.append(f"  {prim_display:<35} {init:>10.3f} {fin:>10.3f} {change_str:>10}")

        # New primitives (learned abstractions)
        new_prims = [p for p in final if p not in initial]
        if new_prims:
            lines.append(f"\nNew primitives added: {len(new_prims)}")
            for prim in new_prims[:10]:
                prim_display = prim[:60] if len(prim) <= 60 else prim[:57] + "..."
                lines.append(f"  {prim_display}")

        # Distribution divergence (KL-like metric)
        lines.append("\nDistribution Divergence from Initial:")

        for snap in self.tracker.snapshots:
            # Simple divergence: sum of absolute changes
            total_change = 0
            for prim in initial:
                if prim in snap.grammar_distribution:
                    total_change += abs(snap.grammar_distribution[prim] - initial[prim])
            lines.append(f"  Iter {snap.iteration+1}: {total_change:.3f}")

        return "\n".join(lines)

    def _detailed_solutions(self) -> str:
        lines = []
        lines.append("\n" + "─" * self.width)
        lines.append(self._center("DETAILED SOLUTIONS"))
        lines.append("─" * self.width)

        if not self.tracker.snapshots:
            return "\n".join(lines)

        final = self.tracker.snapshots[-1]

        for task_name in sorted(final.task_solutions.keys()):
            solutions = final.task_solutions[task_name]

            lines.append(f"\n▶ {task_name}")
            lines.append("  " + "─" * (self.width - 4))

            if not solutions:
                lines.append("  No solutions found")
                continue

            for i, sol in enumerate(solutions[:5]):  # Top 5 solutions
                prog = sol['program']
                # Wrap long programs
                if len(prog) > 60:
                    prog = prog[:57] + "..."

                lines.append(f"  Solution {i+1}:")
                lines.append(f"    Program: {prog}")
                lines.append(f"    Description Length: {sol['description_length']:.2f} bits")
                lines.append(f"    Found after: {sol['programs_to_find']:,} programs")

        return "\n".join(lines)

    def _footer(self) -> str:
        lines = []
        lines.append("\n" + "=" * self.width)
        lines.append(self._center("END OF REPORT"))
        lines.append("=" * self.width)
        return "\n".join(lines)

    def _center(self, text: str) -> str:
        return text.center(self.width)


# ============================================================================
# INTEGRATION WITH FULL DREAMCODER
# ============================================================================

def create_tracked_dreamcoder(
    grammar,
    tasks,
    eval_fn,
    experiment_name: str = "experiment",
    **kwargs
):
    """
    Create a DreamCoder instance with comprehensive tracking.

    Returns (dreamcoder, tracker) tuple.
    """
    from dreamcoder_core.full_dreamcoder import FullDreamCoder

    tracker = ExperimentTracker(experiment_name=experiment_name)
    tracker.record_initial_state(grammar, tasks)

    dc = FullDreamCoder(grammar, tasks, eval_fn, **kwargs)

    return dc, tracker


def run_tracked_experiment(
    grammar,
    tasks,
    eval_fn,
    experiment_name: str = "experiment",
    output_dir: str = "results",
    **kwargs
) -> Tuple[Dict, str]:
    """
    Run a full DreamCoder experiment with tracking and generate report.

    Returns (results_dict, report_string).
    """
    import time
    import copy
    from dreamcoder_core.full_dreamcoder import FullDreamCoder

    # Initialize tracker
    tracker = ExperimentTracker(experiment_name=experiment_name)
    tracker.record_initial_state(grammar, tasks)

    # We need to hook into the DreamCoder to record each iteration
    # For now, we'll run it and reconstruct tracking from results

    dc = FullDreamCoder(
        grammar=grammar,
        tasks=tasks,
        eval_fn=eval_fn,
        log_dir=output_dir,
        **kwargs
    )

    # Store reference for tracking
    dc._tracker = tracker

    # Override _run_iteration to add tracking
    original_run_iteration = dc._run_iteration

    def tracked_run_iteration(iteration):
        result = original_run_iteration(iteration)

        # Record to tracker
        tracker.record_iteration(
            iteration=iteration,
            grammar=dc.grammar,
            frontiers=dc.frontiers,
            new_abstractions=[inv for inv in dc.grammar.primitives()
                            if str(inv).startswith('#(')],  # Invented
            recognition_model=dc.recognition,
            tasks=dc.tasks
        )

        return result

    dc._run_iteration = tracked_run_iteration

    # Run
    results = dc.run()

    # Generate report
    reporter = ReportGenerator(tracker)
    report = reporter.generate_full_report()

    # Save report
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    report_path = output_path / f"report_{experiment_name}_{timestamp}.txt"

    with open(report_path, 'w') as f:
        f.write(report)

    print(f"\nReport saved to: {report_path}")

    return results, report


# ============================================================================
# STANDALONE TEST
# ============================================================================

if __name__ == "__main__":
    # Demo with synthetic data
    print("Generating demo report with synthetic data...")

    tracker = ExperimentTracker(experiment_name="demo")
    tracker.initial_grammar_size = 61
    tracker.initial_grammar_dist = {f"prim_{i}": -3.0 for i in range(61)}
    tracker.library_sizes = [61]

    # Simulate 3 iterations
    for i in range(3):
        snap = IterationSnapshot(
            iteration=i,
            timestamp=(i+1) * 100.0,
            tasks_solved=3 + i,
            tasks_total=6,
            programs_enumerated=100000 * (3 - i),
            grammar_size=61 + (i+1) * 3,
            new_abstractions=[f"#(abstraction_{i}_{j})" for j in range(3)],
            abstraction_sources=[],
            task_solutions={
                "Task_A": [{"program": "(λ solution_a)", "description_length": 10.5, "programs_to_find": 1000, "time_found": 1.0}],
                "Task_B": [{"program": "(λ solution_b)", "description_length": 12.3, "programs_to_find": 5000, "time_found": 5.0}] if i > 0 else [],
                "Task_C": [],
            },
            task_programs_tried={"Task_A": 10000, "Task_B": 50000, "Task_C": 100000},
            recognition_weights={"map": 0.5 + i*0.1, "filter": 0.3, "fold": 0.2},
            recognition_predictions={"Task_A": ["map", "filter"], "Task_B": ["fold", "map"]},
            grammar_distribution={f"prim_{j}": -3.0 + (j % 5) * 0.1 * (i+1) for j in range(61)}
        )
        tracker.snapshots.append(snap)
        tracker.library_sizes.append(snap.grammar_size)
        tracker.solve_rates.append(snap.tasks_solved / snap.tasks_total)
        for a in snap.new_abstractions:
            tracker.all_abstractions.append((i, a, "compression"))

    # Generate report
    reporter = ReportGenerator(tracker)
    report = reporter.generate_full_report()
    print(report)
