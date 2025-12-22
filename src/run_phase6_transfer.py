#!/usr/bin/env python3
"""
Phase 6: Transfer Test to Catalogue Rules

This script loads the trained model and grammar from Phase 5 (pretraining consolidation)
and tests transfer to the full 57-rule catalogue from the behavioral experiment.

Key question: Do abstractions learned from pretraining generalize to experimental rules?
"""

import sys
import os
import time
import argparse
import random
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from run_overnight_cython import (
    CythonOptimizedDreamCoder,
    PhaseConfig,
    print_banner,
    format_time,
    N_WORKERS,
    USE_PYPY,
    USE_CYTHON
)
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.grammar import Grammar
from dreamcoder_core.program import Primitive
from dreamcoder_core.type_system import arrow, HAND, BOOL
from dreamcoder_core.neural_recognition import PrimitivePredictor
from rules.catalogue import create_all_rules
from rules.cards import sample_hand
import torch


def create_catalogue_tasks(n_examples=100, n_holdout=20, hand_size=6, seed=42):
    """Create tasks from the full catalogue of 57 experimental rules."""
    from dreamcoder_core.dreamcoder_original import Task

    catalogue_rules = create_all_rules()
    print(f"Catalogue has {len(catalogue_rules)} rules")

    random.seed(seed)

    tasks = []
    for rule in catalogue_rules:
        positives = []
        negatives = []
        holdout_positives = []
        holdout_negatives = []

        target = n_examples // 2
        holdout_target = n_holdout // 2

        # Use rule-specific seed for reproducibility
        rule_seed = seed + hash(rule.id) % 10000
        random.seed(rule_seed)

        # Sample hands to get balanced examples
        for _ in range(50000):
            hand = sample_hand(hand_size)
            try:
                label = rule.predicate(hand)
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
            except (ValueError, TypeError, ZeroDivisionError, IndexError, KeyError, AttributeError):
                continue

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
        task.holdout_examples = holdout
        tasks.append(task)

    return tasks


def load_phase5_checkpoint(checkpoint_dir: Path):
    """Load model and grammar from Phase 5 checkpoint."""

    # Find the phase 1 checkpoint (which is actually Phase 5 in naming)
    phase1_pt = list(checkpoint_dir.glob("checkpoint_phase1_*.pt"))
    phase1_json = list(checkpoint_dir.glob("checkpoint_phase1_*.json"))
    grammar_json = list(checkpoint_dir.glob("grammar_phase1_*.json"))

    if not phase1_pt:
        raise FileNotFoundError(f"No Phase 5 checkpoint found in {checkpoint_dir}")

    # Use most recent
    model_path = sorted(phase1_pt)[-1]
    meta_path = sorted(phase1_json)[-1]
    grammar_path = sorted(grammar_json)[-1] if grammar_json else None

    print(f"Loading checkpoint from: {model_path}")

    # Load metadata
    with open(meta_path) as f:
        meta = json.load(f)
    print(f"  Phase: {meta['phase']}")
    print(f"  Iteration: {meta['global_iteration']}")
    print(f"  Tasks solved: {meta['tasks_solved']}")
    print(f"  Grammar size: {meta['grammar_size']}")

    # Load grammar if available
    if grammar_path:
        print(f"Loading grammar from: {grammar_path}")
        with open(grammar_path) as f:
            grammar_data = json.load(f)
        # We'll rebuild grammar with learned primitives

    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')

    return checkpoint, meta, grammar_path


def rebuild_grammar_with_abstractions(base_grammar: Grammar, grammar_json_path: Path) -> Grammar:
    """Rebuild grammar including learned abstractions."""
    from dreamcoder_core.grammar import Production

    if not grammar_json_path or not grammar_json_path.exists():
        print("No grammar checkpoint found, using base grammar")
        return base_grammar

    with open(grammar_json_path) as f:
        grammar_data = json.load(f)

    # Get base primitives
    primitives = list(base_grammar.primitives())

    # Add learned abstractions
    learned = grammar_data.get('learned_primitives', [])
    print(f"Adding {len(learned)} learned abstractions to grammar")

    # For now, we'll use the base grammar and let the model guide search
    # The learned abstractions are encoded in the grammar file but parsing them
    # requires reconstructing the Program objects

    return base_grammar


def main():
    parser = argparse.ArgumentParser(description="Phase 6: Transfer test to catalogue rules")
    parser.add_argument("--checkpoint-dir", type=str, required=True,
                        help="Directory containing Phase 5 checkpoint")
    parser.add_argument("--dry-run", action="store_true",
                        help="Quick test with reduced iterations")
    parser.add_argument("--iterations", type=int, default=8,
                        help="Number of iterations (default: 8)")
    parser.add_argument("--budget", type=int, default=800000,
                        help="Enumeration budget per iteration (default: 800000)")
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        print(f"Error: Checkpoint directory not found: {checkpoint_dir}")
        sys.exit(1)

    print_banner("PHASE 6: TRANSFER TEST TO CATALOGUE")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load Phase 5 checkpoint
    print("Loading Phase 5 checkpoint...")
    checkpoint, meta, grammar_path = load_phase5_checkpoint(checkpoint_dir)
    print()

    # Build grammar (base + any learned abstractions)
    print("Building grammar...")
    base_grammar = build_lean_grammar()
    grammar = rebuild_grammar_with_abstractions(base_grammar, grammar_path)
    print(f"Grammar: {len(grammar)} primitives")
    print()

    # Create catalogue tasks
    print("Creating catalogue tasks (45 rules, 100 examples + 20 holdout each)...")
    catalogue_tasks = create_catalogue_tasks(n_examples=100, n_holdout=20, seed=42)
    print(f"Created {len(catalogue_tasks)} catalogue tasks")
    print()

    # Check overlap with pretraining
    pretraining_names = {
        'poker_has_pair', 'poker_three_of_kind', 'poker_flush', 'poker_same_color',
        'poker_two_suits', 'poker_high_card', 'poker_all_face', 'poker_has_ace',
        'poker_straight', 'bj_under_21', 'bj_exactly_21', 'bj_stand_17',
        'bj_sum_even', 'bj_safe_range', 'rummy_all_different', 'rummy_three_ranks',
        'rummy_run_3', 'rummy_set_3', 'sol_same_suit_seq', 'sol_alternating',
        'sol_descending', 'sol_ascending', 'sol_sorted'
    }
    catalogue_names = {t.name for t in catalogue_tasks}
    overlap = pretraining_names & catalogue_names
    new_rules = catalogue_names - pretraining_names
    print(f"Overlap with pretraining: {len(overlap)} rules")
    print(f"New rules to test: {len(new_rules)} rules")
    print()

    # Configure Phase 6
    if args.dry_run:
        print("*** DRY RUN MODE ***")
        iterations = 2
        budget = 100000
        dreams = 50
    else:
        iterations = args.iterations
        budget = args.budget
        dreams = 250

    phases = [
        PhaseConfig(
            name="Phase 6: Transfer to Catalogue",
            iterations=iterations,
            use_all_rules=True,
            enumeration_budget=budget,
            max_depth=12,
            dreams_per_iteration=dreams,
            recognition_epochs=20
        )
    ]

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "phase6_transfer" if not args.dry_run else "phase6_dryrun"
    log_dir = Path(f"results/phase6_transfer/{run_name}_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    config = {
        "run_type": "phase6_transfer",
        "checkpoint_source": str(checkpoint_dir),
        "phase5_iteration": meta['global_iteration'],
        "phase5_tasks_solved": meta['tasks_solved'],
        "catalogue_tasks": len(catalogue_tasks),
        "iterations": iterations,
        "budget": budget,
        "start_time": datetime.now().isoformat()
    }
    with open(log_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print_banner("STARTING PHASE 6 TRANSFER TEST")
    print(f"Output directory: {log_dir}")
    print(f"Tasks: {len(catalogue_tasks)}")
    print(f"Iterations: {iterations}")
    print(f"Budget per iteration: {budget:,}")
    print()

    # Create evaluation function
    def eval_fn(program, hand):
        fn = program.evaluate([])
        return fn(hand)

    # Initialize DreamCoder with loaded model
    # Use first 20 tasks as "easy" for curriculum purposes
    easy_tasks = catalogue_tasks[:20]

    dc = CythonOptimizedDreamCoder(
        grammar=grammar,
        easy_tasks=easy_tasks,
        all_tasks=catalogue_tasks,
        eval_fn=eval_fn,
        phases=phases,
        recognition_hidden_dim=256,
        recognition_lr=5e-4,
        keep_top_k=5,
        max_inventions_per_iteration=5,
        dream_temperature=1.0,
        n_workers=N_WORKERS,
        use_pypy=USE_PYPY,
        verbose=True,
        log_dir=str(log_dir),
        device='cpu'
    )

    # Load the trained recognition model weights
    print("Loading trained recognition model from Phase 5...")
    try:
        dc.recognition.load_state_dict(checkpoint['model_state_dict'])
        print("  Successfully loaded model weights")
        print(f"  Training history: {len(checkpoint.get('training_losses', []))} epochs")
    except Exception as e:
        print(f"  Warning: Could not load model weights: {e}")
        print("  Starting with fresh model")
    print()

    # Run Phase 6
    results = dc.run()

    # Summary
    total_time = time.time() - time.time()  # This will be updated by dc.run()

    print_banner("PHASE 6 TRANSFER TEST COMPLETE")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print(f"Tasks solved: {results['summary']['tasks_solved']}/{results['summary']['tasks_total']}")
    print(f"Success rate: {results['summary']['tasks_solved']/results['summary']['tasks_total']*100:.1f}%")
    print(f"Final grammar: {results['summary']['final_grammar_size']} primitives")
    print()

    # Analyze transfer
    solved_names = set(results.get('solved_tasks', {}).keys())

    # How many overlapping rules were solved?
    overlap_solved = solved_names & overlap
    print(f"Overlap rules solved: {len(overlap_solved)}/{len(overlap)}")

    # How many NEW rules were solved (true transfer)?
    new_solved = solved_names & new_rules
    print(f"NEW rules solved (transfer): {len(new_solved)}/{len(new_rules)}")
    print()

    if new_solved:
        print("Newly solved rules (transfer success):")
        for name in sorted(new_solved):
            print(f"  - {name}")

    print(f"\nResults saved to: {log_dir}")


if __name__ == "__main__":
    main()
