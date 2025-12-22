#!/usr/bin/env python3
"""
Overnight Experiment Runner for Card Game Rule Learning

This script runs comprehensive experiments overnight:
1. Difficulty measurement on all 45 rules
2. Transfer experiments on key rule pairs

Designed to run unattended with:
- Robust error handling (continues on individual failures)
- Progress logging to files
- Periodic checkpointing
- Summary at the end
"""

import sys
import os
import time
import traceback
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dreamcoder_core.difficulty_experiment import (
    DifficultyExperimentConfig,
    run_difficulty_experiment,
    RuleDifficultyResult
)
from dreamcoder_core.transfer_experiment import (
    measure_transfer,
    save_transfer_results,
    TransferResult
)
from rules.catalogue import ALL_RULES, RULE_DICT


def log(msg: str, logfile=None):
    """Print and optionally log to file."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    if logfile:
        with open(logfile, 'a') as f:
            f.write(line + "\n")


def run_all_difficulty_experiments(logfile: str) -> list:
    """Run difficulty measurement on all rules."""
    log("=" * 70, logfile)
    log("STARTING DIFFICULTY EXPERIMENT ON ALL RULES", logfile)
    log("=" * 70, logfile)

    config = DifficultyExperimentConfig(
        name="overnight_all_rules",
        rules=None,  # All rules
        enumeration_budget=500000,
        enumeration_timeout=600.0,  # 10 minutes per rule
        max_depth=8,
        n_examples=20,
        seed=42
    )

    log(f"Total rules: {len(ALL_RULES)}", logfile)
    log(f"Budget: {config.enumeration_budget:,} programs per rule", logfile)
    log(f"Timeout: {config.enumeration_timeout}s per rule", logfile)

    try:
        results = run_difficulty_experiment(config)
        log(f"Difficulty experiment completed: {len(results)} rules processed", logfile)
        return results
    except Exception as e:
        log(f"ERROR in difficulty experiment: {e}", logfile)
        log(traceback.format_exc(), logfile)
        return []


def run_all_transfer_experiments(logfile: str) -> list:
    """Run transfer experiments on key rule pairs."""
    log("=" * 70, logfile)
    log("STARTING TRANSFER EXPERIMENTS", logfile)
    log("=" * 70, logfile)

    # Key pairs that should show transfer effects
    transfer_pairs = [
        # Same structure, different property accessor
        ('Ends_same_color', 'Ends_same_suit'),
        ('Ends_same_suit', 'Ends_same_color'),

        # Palindrome family
        ('Suits_palindrome', 'Uniform_color'),
        ('Uniform_color', 'Suits_palindrome'),

        # Counting rules
        ('Uniform_color', 'Uniform_rank_parity'),

        # Position rules
        ('Sorted_by_rank', 'Ends_same_color'),

        # Cross-family transfer
        ('Suits_palindrome', 'Ends_same_suit'),
    ]

    results = []

    for i, (source_id, target_id) in enumerate(transfer_pairs):
        log(f"\nTransfer pair {i+1}/{len(transfer_pairs)}: {source_id} -> {target_id}", logfile)

        source = RULE_DICT.get(source_id)
        target = RULE_DICT.get(target_id)

        if source is None or target is None:
            log(f"  SKIP: Unknown rule ID", logfile)
            continue

        try:
            result = measure_transfer(
                source,
                target,
                enumeration_budget=500000,
                enumeration_timeout=600.0,
                max_depth=8,
                n_examples=20
            )
            results.append(result)

            log(f"  Baseline: {result.baseline_programs:,} programs (solved={result.baseline_solved})", logfile)
            log(f"  Transfer: {result.transfer_programs:,} programs (solved={result.transfer_solved})", logfile)
            if result.transfer_benefit == result.transfer_benefit:  # not NaN
                log(f"  Benefit: {result.transfer_benefit:.2f}x", logfile)

        except Exception as e:
            log(f"  ERROR: {e}", logfile)
            log(traceback.format_exc(), logfile)

    # Save results
    if results:
        save_transfer_results(results, "overnight_transfer")
        log(f"\nTransfer experiments completed: {len(results)} pairs processed", logfile)

    return results


def main():
    """Main entry point for overnight experiments."""
    start_time = time.time()

    # Create results directory
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # Create log file
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    logfile = results_dir / f"overnight_log_{timestamp}.txt"

    log("=" * 70, logfile)
    log("OVERNIGHT EXPERIMENT RUNNER", logfile)
    log(f"Started at: {datetime.now().isoformat()}", logfile)
    log("=" * 70, logfile)

    # Run experiments
    difficulty_results = run_all_difficulty_experiments(logfile)
    transfer_results = run_all_transfer_experiments(logfile)

    # Final summary
    elapsed = time.time() - start_time
    hours = elapsed / 3600

    log("\n" + "=" * 70, logfile)
    log("OVERNIGHT EXPERIMENTS COMPLETE", logfile)
    log("=" * 70, logfile)
    log(f"Total time: {hours:.1f} hours", logfile)
    log(f"Difficulty: {len(difficulty_results)} rules processed", logfile)
    if difficulty_results:
        solved = sum(1 for r in difficulty_results if r.solved)
        log(f"  Solved: {solved}/{len(difficulty_results)} ({100*solved/len(difficulty_results):.1f}%)", logfile)
    log(f"Transfer: {len(transfer_results)} pairs processed", logfile)
    log(f"Results saved to: {results_dir}", logfile)
    log("=" * 70, logfile)

    return 0


if __name__ == "__main__":
    sys.exit(main())
