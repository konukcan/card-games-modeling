#!/usr/bin/env python3
"""
Factorial Experiment Analysis Script

Analyzes results from the 2×3×3 factorial experiment:
- Recognition: GRU vs Contrastive
- Dreams: Standard vs Balanced vs Contrastive
- Primitives: Lean vs Lean+Fold vs Minimal

Usage:
    python3 experiments/analyze_factorial_results.py --results-dir results_factorial
    python3 experiments/analyze_factorial_results.py --results-dir results_factorial --output-dir analysis_output
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict
import statistics

# Try to import optional dependencies
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    print("Note: Install pandas for enhanced analysis (pip install pandas)")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Note: Install matplotlib for visualizations (pip install matplotlib)")


@dataclass
class ConditionResults:
    """Results for a single experimental condition."""
    recognition: str
    dreams: str
    primitives: str

    # Per-iteration metrics
    tasks_solved_per_iter: List[int]
    programs_enumerated_per_iter: List[int]
    recognition_loss_per_iter: List[float]
    iteration_time_per_iter: List[float]

    # Final metrics
    final_tasks_solved: int
    total_tasks: int
    total_time: float
    inventions_discovered: int

    # Task-level details
    solved_tasks: List[str]
    unsolved_tasks: List[str]

    @property
    def solve_rate(self) -> float:
        return self.final_tasks_solved / self.total_tasks if self.total_tasks > 0 else 0

    @property
    def condition_name(self) -> str:
        return f"{self.recognition}_{self.dreams}_{self.primitives}"


def load_condition_results(results_dir: Path) -> List[ConditionResults]:
    """Load all condition results from the results directory."""
    results = []

    for condition_dir in sorted(results_dir.iterdir()):
        if not condition_dir.is_dir() or condition_dir.name.startswith('.'):
            continue
        if condition_dir.name.startswith('experiment_'):
            continue  # Skip log files

        # Parse condition name
        parts = condition_dir.name.split('_')
        if len(parts) < 3:
            continue

        recognition = parts[0]
        dreams = parts[1]
        primitives = '_'.join(parts[2:])  # Handle lean_plus_fold

        # Find the run directory
        run_dirs = list(condition_dir.glob('run_*'))
        if not run_dirs:
            continue

        for run_dir in run_dirs:
            results_file = run_dir / 'final_results.json'
            if not results_file.exists():
                continue

            with open(results_file) as f:
                data = json.load(f)

            # Extract per-iteration metrics
            iterations = data.get('iterations', [])
            tasks_solved = [it.get('tasks_solved', 0) for it in iterations]
            programs_enum = [it.get('programs_enumerated', 0) for it in iterations]
            recog_loss = [it.get('recognition_loss', 0) for it in iterations]
            iter_time = [it.get('iteration_time', 0) for it in iterations]

            # Count inventions
            inventions = sum(len(it.get('new_inventions', [])) for it in iterations)

            # Get final task details
            final_iter = iterations[-1] if iterations else {}
            task_details = final_iter.get('task_details', {})
            solved_tasks = [name for name, info in task_details.items() if info.get('solved')]
            unsolved_tasks = [name for name, info in task_details.items() if not info.get('solved')]

            result = ConditionResults(
                recognition=recognition,
                dreams=dreams,
                primitives=primitives,
                tasks_solved_per_iter=tasks_solved,
                programs_enumerated_per_iter=programs_enum,
                recognition_loss_per_iter=recog_loss,
                iteration_time_per_iter=iter_time,
                final_tasks_solved=tasks_solved[-1] if tasks_solved else 0,
                total_tasks=final_iter.get('total_tasks', 22),
                total_time=data.get('total_time', sum(iter_time)),
                inventions_discovered=inventions,
                solved_tasks=solved_tasks,
                unsolved_tasks=unsolved_tasks
            )
            results.append(result)

    return results


def print_summary_table(results: List[ConditionResults]):
    """Print a summary table of all conditions."""
    print("\n" + "="*80)
    print("FACTORIAL EXPERIMENT RESULTS SUMMARY")
    print("="*80)

    # Group by factors
    print("\n### Overall Summary ###\n")
    print(f"{'Condition':<35} {'Solved':>8} {'Rate':>8} {'Time':>10} {'Inventions':>10}")
    print("-" * 75)

    for r in sorted(results, key=lambda x: x.condition_name):
        print(f"{r.condition_name:<35} {r.final_tasks_solved:>5}/{r.total_tasks:<2} "
              f"{r.solve_rate:>7.1%} {r.total_time:>9.1f}s {r.inventions_discovered:>10}")


def analyze_main_effects(results: List[ConditionResults]):
    """Analyze main effects of each factor."""
    print("\n" + "="*80)
    print("MAIN EFFECTS ANALYSIS")
    print("="*80)

    # Group by each factor
    by_recognition = defaultdict(list)
    by_dreams = defaultdict(list)
    by_primitives = defaultdict(list)

    for r in results:
        by_recognition[r.recognition].append(r.solve_rate)
        by_dreams[r.dreams].append(r.solve_rate)
        by_primitives[r.primitives].append(r.solve_rate)

    print("\n### Recognition Model Effect ###")
    print(f"{'Model':<15} {'Mean Solve Rate':>15} {'Std Dev':>10} {'N':>5}")
    print("-" * 50)
    for model, rates in sorted(by_recognition.items()):
        mean_rate = statistics.mean(rates)
        std_rate = statistics.stdev(rates) if len(rates) > 1 else 0
        print(f"{model:<15} {mean_rate:>14.1%} {std_rate:>10.3f} {len(rates):>5}")

    print("\n### Dream Strategy Effect ###")
    print(f"{'Strategy':<15} {'Mean Solve Rate':>15} {'Std Dev':>10} {'N':>5}")
    print("-" * 50)
    for strategy, rates in sorted(by_dreams.items()):
        mean_rate = statistics.mean(rates)
        std_rate = statistics.stdev(rates) if len(rates) > 1 else 0
        print(f"{strategy:<15} {mean_rate:>14.1%} {std_rate:>10.3f} {len(rates):>5}")

    print("\n### Primitive Library Effect ###")
    print(f"{'Library':<20} {'Mean Solve Rate':>15} {'Std Dev':>10} {'N':>5}")
    print("-" * 55)
    for library, rates in sorted(by_primitives.items()):
        mean_rate = statistics.mean(rates)
        std_rate = statistics.stdev(rates) if len(rates) > 1 else 0
        print(f"{library:<20} {mean_rate:>14.1%} {std_rate:>10.3f} {len(rates):>5}")


def analyze_interactions(results: List[ConditionResults]):
    """Analyze two-way interactions between factors."""
    print("\n" + "="*80)
    print("INTERACTION ANALYSIS")
    print("="*80)

    # Recognition × Dreams
    print("\n### Recognition × Dreams Interaction ###")
    interaction = defaultdict(list)
    for r in results:
        key = (r.recognition, r.dreams)
        interaction[key].append(r.solve_rate)

    print(f"{'Recognition':<12} {'Dreams':<12} {'Mean Rate':>12} {'N':>5}")
    print("-" * 45)
    for (recog, dreams), rates in sorted(interaction.items()):
        mean_rate = statistics.mean(rates)
        print(f"{recog:<12} {dreams:<12} {mean_rate:>11.1%} {len(rates):>5}")

    # Recognition × Primitives
    print("\n### Recognition × Primitives Interaction ###")
    interaction = defaultdict(list)
    for r in results:
        key = (r.recognition, r.primitives)
        interaction[key].append(r.solve_rate)

    print(f"{'Recognition':<12} {'Primitives':<18} {'Mean Rate':>12} {'N':>5}")
    print("-" * 50)
    for (recog, prims), rates in sorted(interaction.items()):
        mean_rate = statistics.mean(rates)
        print(f"{recog:<12} {prims:<18} {mean_rate:>11.1%} {len(rates):>5}")

    # Dreams × Primitives
    print("\n### Dreams × Primitives Interaction ###")
    interaction = defaultdict(list)
    for r in results:
        key = (r.dreams, r.primitives)
        interaction[key].append(r.solve_rate)

    print(f"{'Dreams':<12} {'Primitives':<18} {'Mean Rate':>12} {'N':>5}")
    print("-" * 50)
    for (dreams, prims), rates in sorted(interaction.items()):
        mean_rate = statistics.mean(rates)
        print(f"{dreams:<12} {prims:<18} {mean_rate:>11.1%} {len(rates):>5}")


def analyze_learning_curves(results: List[ConditionResults]):
    """Analyze learning curves across iterations."""
    print("\n" + "="*80)
    print("LEARNING CURVE ANALYSIS")
    print("="*80)

    # Group by recognition model
    print("\n### Tasks Solved per Iteration (by Recognition) ###")
    by_recognition = defaultdict(list)
    for r in results:
        by_recognition[r.recognition].append(r.tasks_solved_per_iter)

    for model, curves in sorted(by_recognition.items()):
        print(f"\n{model.upper()}:")
        n_iters = max(len(c) for c in curves)
        for i in range(n_iters):
            values = [c[i] for c in curves if i < len(c)]
            mean_val = statistics.mean(values)
            print(f"  Iteration {i+1}: {mean_val:.1f} tasks solved (avg across {len(values)} conditions)")

    # Recognition loss trajectory
    print("\n### Recognition Loss Trajectory ###")
    for model, curves in sorted(by_recognition.items()):
        loss_curves = [r.recognition_loss_per_iter for r in results if r.recognition == model]
        print(f"\n{model.upper()}:")
        n_iters = max(len(c) for c in loss_curves)
        for i in range(n_iters):
            values = [c[i] for c in loss_curves if i < len(c) and c[i] > 0]
            if values:
                mean_val = statistics.mean(values)
                print(f"  Iteration {i+1}: {mean_val:.4f} loss")


def analyze_task_difficulty(results: List[ConditionResults]):
    """Analyze which tasks are easy vs hard across conditions."""
    print("\n" + "="*80)
    print("TASK DIFFICULTY ANALYSIS")
    print("="*80)

    # Count how often each task was solved
    task_solve_counts = defaultdict(int)
    task_total_counts = defaultdict(int)

    for r in results:
        for task in r.solved_tasks:
            task_solve_counts[task] += 1
            task_total_counts[task] += 1
        for task in r.unsolved_tasks:
            task_total_counts[task] += 1

    # Sort by solve rate
    task_rates = [(task, task_solve_counts[task] / task_total_counts[task], task_total_counts[task])
                  for task in task_total_counts]
    task_rates.sort(key=lambda x: -x[1])

    print("\n### Tasks by Solve Rate ###")
    print(f"{'Task':<30} {'Solve Rate':>12} {'Solved/Total':>15}")
    print("-" * 60)

    for task, rate, total in task_rates:
        solved = task_solve_counts[task]
        print(f"{task:<30} {rate:>11.1%} {solved:>6}/{total:<6}")

    # Identify consistently hard tasks
    print("\n### Consistently Hard Tasks (never solved) ###")
    never_solved = [task for task, rate, _ in task_rates if rate == 0]
    for task in never_solved:
        print(f"  - {task}")

    print("\n### Consistently Easy Tasks (always solved) ###")
    always_solved = [task for task, rate, _ in task_rates if rate == 1.0]
    for task in always_solved:
        print(f"  - {task}")


def generate_plots(results: List[ConditionResults], output_dir: Path):
    """Generate visualization plots."""
    if not HAS_MATPLOTLIB:
        print("\nSkipping plots (matplotlib not installed)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Bar chart: Solve rates by condition
    fig, ax = plt.subplots(figsize=(14, 6))
    conditions = [r.condition_name for r in sorted(results, key=lambda x: x.condition_name)]
    rates = [r.solve_rate * 100 for r in sorted(results, key=lambda x: x.condition_name)]

    colors = []
    for r in sorted(results, key=lambda x: x.condition_name):
        if r.recognition == 'gru':
            colors.append('#1f77b4')  # Blue
        else:
            colors.append('#ff7f0e')  # Orange

    bars = ax.bar(range(len(conditions)), rates, color=colors)
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Solve Rate (%)')
    ax.set_title('Task Solve Rate by Condition')
    ax.set_ylim(0, 100)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1f77b4', label='GRU'),
                       Patch(facecolor='#ff7f0e', label='Contrastive')]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.tight_layout()
    plt.savefig(output_dir / 'solve_rates_by_condition.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'solve_rates_by_condition.png'}")

    # 2. Learning curves by recognition model
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, model in zip(axes, ['gru', 'contrastive']):
        model_results = [r for r in results if r.recognition == model]
        for r in model_results:
            label = f"{r.dreams}_{r.primitives}"
            ax.plot(range(1, len(r.tasks_solved_per_iter) + 1),
                   r.tasks_solved_per_iter,
                   marker='o', label=label, alpha=0.7)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Tasks Solved')
        ax.set_title(f'{model.upper()} Recognition')
        ax.legend(fontsize=7, loc='upper left')
        ax.set_ylim(0, max(r.total_tasks for r in results) + 1)

    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'learning_curves.png'}")

    # 3. Main effects plot
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Recognition effect
    by_recognition = defaultdict(list)
    for r in results:
        by_recognition[r.recognition].append(r.solve_rate * 100)

    models = list(by_recognition.keys())
    means = [statistics.mean(by_recognition[m]) for m in models]
    stds = [statistics.stdev(by_recognition[m]) if len(by_recognition[m]) > 1 else 0 for m in models]
    axes[0].bar(models, means, yerr=stds, capsize=5, color=['#1f77b4', '#ff7f0e'])
    axes[0].set_ylabel('Solve Rate (%)')
    axes[0].set_title('Recognition Model')
    axes[0].set_ylim(0, 100)

    # Dreams effect
    by_dreams = defaultdict(list)
    for r in results:
        by_dreams[r.dreams].append(r.solve_rate * 100)

    strategies = list(by_dreams.keys())
    means = [statistics.mean(by_dreams[s]) for s in strategies]
    stds = [statistics.stdev(by_dreams[s]) if len(by_dreams[s]) > 1 else 0 for s in strategies]
    axes[1].bar(strategies, means, yerr=stds, capsize=5, color='#2ca02c')
    axes[1].set_ylabel('Solve Rate (%)')
    axes[1].set_title('Dream Strategy')
    axes[1].set_ylim(0, 100)

    # Primitives effect
    by_primitives = defaultdict(list)
    for r in results:
        by_primitives[r.primitives].append(r.solve_rate * 100)

    libraries = list(by_primitives.keys())
    means = [statistics.mean(by_primitives[l]) for l in libraries]
    stds = [statistics.stdev(by_primitives[l]) if len(by_primitives[l]) > 1 else 0 for l in libraries]
    axes[2].bar(libraries, means, yerr=stds, capsize=5, color='#9467bd')
    axes[2].set_ylabel('Solve Rate (%)')
    axes[2].set_title('Primitive Library')
    axes[2].set_ylim(0, 100)
    axes[2].tick_params(axis='x', rotation=15)

    plt.tight_layout()
    plt.savefig(output_dir / 'main_effects.png', dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'main_effects.png'}")


def export_to_csv(results: List[ConditionResults], output_dir: Path):
    """Export results to CSV for further analysis in R or other tools."""
    if not HAS_PANDAS:
        print("\nSkipping CSV export (pandas not installed)")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Condition-level summary
    rows = []
    for r in results:
        rows.append({
            'condition': r.condition_name,
            'recognition': r.recognition,
            'dreams': r.dreams,
            'primitives': r.primitives,
            'tasks_solved': r.final_tasks_solved,
            'total_tasks': r.total_tasks,
            'solve_rate': r.solve_rate,
            'total_time': r.total_time,
            'inventions': r.inventions_discovered,
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'condition_summary.csv', index=False)
    print(f"Saved: {output_dir / 'condition_summary.csv'}")

    # Iteration-level data (for learning curve analysis)
    rows = []
    for r in results:
        for i, (solved, programs, loss, time) in enumerate(zip(
            r.tasks_solved_per_iter,
            r.programs_enumerated_per_iter,
            r.recognition_loss_per_iter,
            r.iteration_time_per_iter
        )):
            rows.append({
                'condition': r.condition_name,
                'recognition': r.recognition,
                'dreams': r.dreams,
                'primitives': r.primitives,
                'iteration': i + 1,
                'tasks_solved': solved,
                'programs_enumerated': programs,
                'recognition_loss': loss,
                'iteration_time': time,
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / 'iteration_data.csv', index=False)
    print(f"Saved: {output_dir / 'iteration_data.csv'}")


def main():
    parser = argparse.ArgumentParser(description='Analyze factorial experiment results')
    parser.add_argument('--results-dir', type=str, default='results_factorial',
                       help='Directory containing experiment results')
    parser.add_argument('--output-dir', type=str, default='analysis_output',
                       help='Directory for analysis outputs')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    # Load results
    print(f"Loading results from: {results_dir}")
    results = load_condition_results(results_dir)

    if not results:
        print("No results found!")
        return

    print(f"Loaded {len(results)} condition results")

    # Run analyses
    print_summary_table(results)
    analyze_main_effects(results)
    analyze_interactions(results)
    analyze_learning_curves(results)
    analyze_task_difficulty(results)

    # Generate outputs
    generate_plots(results, output_dir)
    export_to_csv(results, output_dir)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutputs saved to: {output_dir}/")
    print("\nNext steps:")
    print("1. Review the summary tables above")
    print("2. Check the plots in the output directory")
    print("3. Load CSV files into R for statistical tests:")
    print("   - 3-way ANOVA: aov(solve_rate ~ recognition * dreams * primitives)")
    print("   - Post-hoc tests: TukeyHSD()")


if __name__ == '__main__':
    main()
