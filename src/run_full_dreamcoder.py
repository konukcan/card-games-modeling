#!/usr/bin/env python3
"""
Full DreamCoder Experiment Runner

This script runs comprehensive experiments with the full DreamCoder system
including all components:
- Wake phase: Enumerate programs with top-k frontiers
- Sleep - Compression: Anti-unification based library learning
- Sleep - Recognition: Feature-based recognition model
- Sleep - Dreaming: Synthetic task generation (optional)

Experiment modes:
1. quick: 2 rules, fast settings for testing
2. selected: 10 representative rules
3. all: All 57 rules (overnight run)
4. ablation: Compare different component combinations

Output:
- JSON results with full metrics
- Learning curves
- Library evolution
- Per-task difficulty measurements
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dreamcoder_core.full_dreamcoder import (
    FullDreamCoder, create_tasks_from_rules, make_eval_fn, Task
)
from dreamcoder_core.card_primitives import build_card_grammar
from rules.catalogue import ALL_RULES, RULE_DICT


def log(msg: str):
    """Timestamped logging."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)


def get_rule_selection(mode: str):
    """Get rules based on experiment mode."""

    if mode == "quick":
        # 2 easy rules for quick testing
        return ['Uniform_color', 'Suits_palindrome']

    elif mode == "selected":
        # 10 representative rules across families and difficulties
        return [
            # Easy - palindrome family (~42k programs)
            'Uniform_color',
            'Suits_palindrome',
            'Colors_palindrome',
            'Ranks_palindrome',

            # Medium - terminal comparisons (~427k programs)
            'Ends_same_color',
            'Ends_same_suit',

            # Sorting
            'Sorted_by_rank',

            # Center-based
            'Halves_radial_nonincreasing',
            'Global_radial_no_dominance',

            # Special case (very easy)
            'Shift2_plus3',
        ]

    elif mode == "all":
        # All rules
        return [r.id for r in ALL_RULES]

    elif mode == "easy":
        # Only rules that were solved in overnight run
        return [
            'Sorted_by_rank',
            'Ends_same_suit',
            'Ends_same_color',
            'Uniform_color',
            'Suits_palindrome',
            'Colors_palindrome',
            'Ranks_palindrome',
            'Halves_radial_nonincreasing',
            'Global_radial_no_dominance',
            'Shift2_plus3',
        ]

    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_experiment(
    mode: str,
    use_compression: bool = True,
    use_recognition: bool = True,
    use_dreaming: bool = False,
    enumeration_budget: int = 500000,
    max_iterations: int = 5,
    output_dir: str = "results"
):
    """Run a full DreamCoder experiment."""

    log("=" * 70)
    log("FULL DREAMCODER EXPERIMENT")
    log("=" * 70)

    # Get rules
    rule_ids = get_rule_selection(mode)
    rules = [RULE_DICT[rid] for rid in rule_ids if rid in RULE_DICT]
    log(f"Mode: {mode}")
    log(f"Rules: {len(rules)}")
    log(f"Components: compression={use_compression}, recognition={use_recognition}, dreaming={use_dreaming}")
    log(f"Budget: {enumeration_budget:,} programs per task")
    log(f"Max iterations: {max_iterations}")

    # Create tasks
    log("\nCreating tasks...")
    tasks = create_tasks_from_rules(rules, n_examples=20, seed=42)
    log(f"Created {len(tasks)} tasks")

    # Build grammar
    log("Building grammar...")
    grammar = build_card_grammar()
    log(f"Grammar: {len(grammar)} primitives")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Create evaluation function
    eval_fn = make_eval_fn()

    # Configure DreamCoder
    log("\nStarting DreamCoder...")
    dc = FullDreamCoder(
        grammar=grammar,
        tasks=tasks,
        eval_fn=eval_fn,

        # Wake settings
        enumeration_budget=enumeration_budget,
        enumeration_timeout=600.0,  # 10 minutes per task
        max_depth=8,

        # Frontier settings
        keep_top_k=5,

        # Component flags
        use_compression=use_compression,
        use_recognition=use_recognition,
        use_dreaming=use_dreaming,

        # Compression settings
        max_inventions_per_iteration=5,
        min_compression_savings=2.0,

        # Dreaming settings
        dreams_per_iteration=100,

        # General
        max_iterations=max_iterations,
        verbose=True,
        log_dir=str(output_path)
    )

    # Run
    results = dc.run()

    return results


def run_ablation_study(output_dir: str = "results"):
    """
    Run ablation study comparing different component combinations.

    This helps understand the contribution of each component.
    """
    log("=" * 70)
    log("ABLATION STUDY")
    log("=" * 70)

    # Use selected rules for ablation (not too many, not too few)
    rule_ids = get_rule_selection("selected")

    configurations = [
        # Baseline: no learning
        {"name": "baseline", "compression": False, "recognition": False, "dreaming": False},

        # Single components
        {"name": "compression_only", "compression": True, "recognition": False, "dreaming": False},
        {"name": "recognition_only", "compression": False, "recognition": True, "dreaming": False},

        # Combinations
        {"name": "compress_recog", "compression": True, "recognition": True, "dreaming": False},

        # Full system
        {"name": "full_system", "compression": True, "recognition": True, "dreaming": True},
    ]

    all_results = {}

    for config in configurations:
        log(f"\n{'='*70}")
        log(f"Configuration: {config['name']}")
        log(f"{'='*70}")

        results = run_experiment(
            mode="selected",
            use_compression=config["compression"],
            use_recognition=config["recognition"],
            use_dreaming=config["dreaming"],
            enumeration_budget=100000,  # Smaller budget for ablation
            max_iterations=3,
            output_dir=f"{output_dir}/ablation_{config['name']}"
        )

        all_results[config['name']] = {
            'config': config,
            'solved': results['summary']['tasks_solved'],
            'total': results['summary']['tasks_total'],
            'abstractions': results['summary']['total_abstractions'],
            'time': results['summary']['total_time']
        }

    # Summary
    log("\n" + "=" * 70)
    log("ABLATION SUMMARY")
    log("=" * 70)

    for name, r in all_results.items():
        log(f"{name}: {r['solved']}/{r['total']} solved, {r['abstractions']} abstractions, {r['time']:.1f}s")

    # Save ablation results
    ablation_path = Path(output_dir) / "ablation_summary.json"
    with open(ablation_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    log(f"\nAblation results saved to: {ablation_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run Full DreamCoder Experiments")

    parser.add_argument(
        "--mode",
        choices=["quick", "selected", "all", "easy", "ablation"],
        default="quick",
        help="Experiment mode"
    )

    parser.add_argument(
        "--no-compression",
        action="store_true",
        help="Disable compression (library learning)"
    )

    parser.add_argument(
        "--no-recognition",
        action="store_true",
        help="Disable recognition model"
    )

    parser.add_argument(
        "--dreaming",
        action="store_true",
        help="Enable dreaming (self-supervised)"
    )

    parser.add_argument(
        "--budget",
        type=int,
        default=500000,
        help="Enumeration budget per task"
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Maximum wake-sleep iterations"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="results",
        help="Output directory"
    )

    args = parser.parse_args()

    if args.mode == "ablation":
        run_ablation_study(args.output)
    else:
        run_experiment(
            mode=args.mode,
            use_compression=not args.no_compression,
            use_recognition=not args.no_recognition,
            use_dreaming=args.dreaming,
            enumeration_budget=args.budget,
            max_iterations=args.iterations,
            output_dir=args.output
        )


if __name__ == "__main__":
    main()
