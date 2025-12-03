#!/usr/bin/env python3
"""
Overnight Run: Extended Primitives - Full Lambda Calculus Test

This script tests whether the lambda calculus paradigm can solve our 56 rules
when equipped with the FULL set of primitives from:
- DreamCoder: fold, cons, tail, empty
- Yang & Piantadosi: pair, fst, snd
- Type-gap bridging: all_true, any_true
- Convenience: first_half, second_half, is_sorted_by, is_palindrome_by

Design Philosophy:
- We add ALL the primitives that could theoretically help
- We use a 6-phase curriculum with increasing search depth and budget
- We track which rules are solved at each phase
- We log extensively for post-hoc analysis

Expected Runtime: 10-12 hours
Expected Outcome: This will tell us the practical limits of lambda calculus
for our rule set. If rules remain unsolved, it's a paradigm issue, not a
primitive issue.

IMPORTANT: Run with caffeinate to prevent sleep!
    nohup caffeinate -d -i -s python3 run_overnight_extended.py > extended.out 2>&1 &
"""

import sys
import os
import time
import json
import random
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Set, Tuple, Any

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
from dreamcoder_core.extended_primitives import build_extended_grammar, build_extended_primitives
from dreamcoder_core.type_system import arrow, HAND, BOOL
from rules.catalogue import create_all_rules, Rule
from rules.cards import sample_hand


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    """Set up comprehensive logging."""
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('overnight_extended')
    logger.setLevel(logging.DEBUG)

    # File handler - detailed logs
    fh = logging.FileHandler(log_dir / 'detailed.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))
    logger.addHandler(fh)

    # Console handler - summary
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)

    return logger


# ============================================================================
# RULE CLASSIFICATION BY EXPECTED PROGRAM DEPTH
# ============================================================================

def classify_rules_by_depth() -> Dict[str, List[Rule]]:
    """
    Classify rules by expected program depth with extended primitives.

    This is a more nuanced classification than before, based on our analysis
    of what it takes to express each rule with the new primitives.
    """
    all_rules = create_all_rules()

    # Phase 1: Trivial (depth 2-3) - Direct primitives available
    trivial_ids = {
        # Direct aggregate queries
        'Uniform_color',           # all_same_color
        'Exactly_two_suits',       # eq 2 (n_unique_suits)
        'At_most_three_suits',     # le (n_unique_suits) 3
        'Exactly_one_club',        # eq 1 (count_suit CLUBS)
        'Has_pair_ranks',          # lt (n_unique_ranks) (length)
        'Uniform_rank_parity',     # all same parity (with get_parity)
    }

    # Phase 2: Easy (depth 3-4) - Simple compositions
    easy_ids = {
        # Endpoints
        'Ends_same_suit',          # eq (get_suit (head)) (get_suit (last))
        'Ends_same_color',         # eq (get_color (head)) (get_color (last))
        'Ends_same_altcolor1',     # eq (get_altcolor1 (head)) (get_altcolor1 (last))

        # Position checks
        'Pos3_is_JQK',             # member (get_rank (at 2)) {J,Q,K}
        'Pos4_is_2_5_7',           # member (get_rank (at 3)) {2,5,7}

        # Token presence
        'Has_Ace_of_Spades',       # any (and (eq suit S) (eq rank A))
        'Has_6_of_Diamonds',       # any (and (eq suit D) (eq rank 6))

        # Sorted (with is_sorted_by)
        'Sorted_by_rank',          # is_sorted_by rank_val

        # Parity
        'Only_one_odd_rank',       # eq 1 (count_true (map is_odd_rank))
    }

    # Phase 3: Medium (depth 4-5) - Palindromes and halves
    medium_ids = {
        # Palindromes (with is_palindrome_by)
        'Suits_palindrome',        # is_palindrome_by get_suit
        'Colors_palindrome',       # is_palindrome_by get_color
        'Ranks_palindrome',        # is_palindrome_by get_rank
        'AltColor1_palindrome',    # is_palindrome_by get_altcolor1
        'AltColor2_palindrome',    # is_palindrome_by get_altcolor2

        # Half comparisons
        'Half_or_more_same_suit',  # ge (max count_suit) (half_len)
        'Halves_uniform_color_equal',  # eq (all_same_color first_half) (all_same_color second_half)
        'Halves_hearts_presence_equal', # eq (has_heart first_half) (has_heart second_half)
        'Halves_same_suit_set',    # eq (unique suits first_half) (unique suits second_half)
    }

    # Phase 4: Hard (depth 5-7) - Complex halves and sequences
    hard_ids = {
        # Halves copy (with list_eq and first_half/second_half)
        'Halves_copy_suits',       # list_eq (map get_suit first_half) (map get_suit second_half)
        'Halves_copy_colors',      # list_eq (map get_color first_half) (map get_color second_half)
        'Halves_copy_ranks',       # list_eq (map get_rank first_half) (map get_rank second_half)
        'Halves_copy_altcolor1',   # list_eq (map get_altcolor1 first_half) (map get_altcolor1 second_half)
        'Halves_copy_altcolor2',   # list_eq (map get_altcolor2 first_half) (map get_altcolor2 second_half)

        # Halves boolean properties
        'Halves_uniform_parity_equal',   # eq (uniform_parity first_half) (uniform_parity second_half)
        'Halves_AP_step1_equal',   # eq (is_run first_half) (is_run second_half)
        'Halves_AP_len2_step1_equal',  # eq (has_adjacent first_half) (has_adjacent second_half)

        # Ordering patterns
        'S_before_H',              # Needs scan/fold

        # Suit cycles
        'Half_map_samepos_M1',     # all (eq (cycle_m1 (get_suit i)) (get_suit i+k))
        'Half_map_samepos_M2',     # all (eq (cycle_m2 (get_suit i)) (get_suit i+k))
    }

    # Phase 5: Very Hard (depth 7-10) - Complex patterns
    very_hard_ids = {
        # Shift patterns
        'Shift_half_plus_two',     # all shifted pairs differ by 2
        'Shift2_plus3',            # all pairs at offset 2 differ by 3
        'Shift_half_ge',           # all right >= left

        # Adjacent patterns
        'Adj_same_rank_or_suit',   # all adjacent pairs share rank or suit
        'Skip2_same_rank_or_suit', # all pairs at offset 2 share rank or suit
        'Adj_rank_gap_le3',        # all adjacent pairs differ by <= 3

        # Suit cycle patterns
        'Step2_back_map_M1',       # suit[j] = cycle_m1(suit[j-2])
        'Step2_back_map_M2',       # suit[j] = cycle_m2(suit[j-2])
        'Adj_same_or_map_M1',      # adjacent same or next in M1
        'Adj_same_or_map_M2',      # adjacent same or next in M2

        # Score rules
        'Half_sum_diff_geN',       # left_sum - right_sum >= N
        'Half_sum_one_side_ge_2x_other',  # one half >= 2x other
    }

    # Phase 6: Extreme (depth 10+) - Require fold/recursion
    extreme_ids = {
        # LANG family - bracket matching (requires fold + state)
        'Well_formed_brackets_by_suit',
        'Even_opens_next_closes',
        'Odd_opens_next_closes',

        # CENTER family - radial patterns
        'Halves_radial_nonincreasing',
        'Global_radial_no_dominance',

        # Complex scores
        'Score_threshold_Rstar',   # Complex multi-component score

        # AP patterns
        'AP_len3_anywhere_anyk',   # Find AP of length 3
        'AP_len3_step2_anywhere',  # Find AP with step 2
        'AP_len4_step2_anywhere',  # Find AP of length 4
        'Halves_AP_len3_any_equal',  # Both halves have AP or both don't
    }

    # Build classification
    classified = {
        'trivial': [],
        'easy': [],
        'medium': [],
        'hard': [],
        'very_hard': [],
        'extreme': []
    }

    rule_dict = {r.id: r for r in all_rules}

    for phase, ids in [
        ('trivial', trivial_ids),
        ('easy', easy_ids),
        ('medium', medium_ids),
        ('hard', hard_ids),
        ('very_hard', very_hard_ids),
        ('extreme', extreme_ids)
    ]:
        for rule_id in ids:
            if rule_id in rule_dict:
                classified[phase].append(rule_dict[rule_id])

    # Any remaining rules go to extreme
    all_classified = trivial_ids | easy_ids | medium_ids | hard_ids | very_hard_ids | extreme_ids
    for rule in all_rules:
        if rule.id not in all_classified:
            classified['extreme'].append(rule)

    return classified


# ============================================================================
# TASK CREATION
# ============================================================================

def create_tasks_from_rules(
    rules: List[Rule],
    n_examples: int = 100,
    n_holdout: int = 20,
    hand_size: int = 6,
    seed: int = 42
) -> List:
    """Create tasks from rules with balanced examples."""
    from dreamcoder_core.dreamcoder_v2 import Task

    tasks = []
    for rule in rules:
        positives = []
        negatives = []
        holdout_positives = []
        holdout_negatives = []

        target = n_examples // 2
        holdout_target = n_holdout // 2

        # Use rule-specific seed
        rule_seed = seed + hash(rule.id) % 10000
        random.seed(rule_seed)

        attempts = 0
        max_attempts = 100000

        while attempts < max_attempts:
            attempts += 1
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
            except Exception:
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


# ============================================================================
# PHASE CONFIGURATION
# ============================================================================

def create_phase_configs(dry_run: bool = False) -> List[PhaseConfig]:
    """Create 6-phase curriculum with increasing depth and budget."""
    if dry_run:
        return [
            PhaseConfig(
                name="Dry Run",
                iterations=2,
                use_all_rules=False,
                enumeration_budget=50000,
                max_depth=8,
                dreams_per_iteration=20,
                recognition_epochs=3
            ),
        ]

    return [
        # Phase 1: Trivial rules (depth 2-3)
        PhaseConfig(
            name="Phase 1: Trivial Rules (depth 2-3)",
            iterations=3,
            use_all_rules=False,
            enumeration_budget=200000,
            max_depth=8,
            dreams_per_iteration=80,
            recognition_epochs=10
        ),

        # Phase 2: Easy rules (depth 3-4)
        PhaseConfig(
            name="Phase 2: Easy Rules (depth 3-4)",
            iterations=4,
            use_all_rules=False,
            enumeration_budget=400000,
            max_depth=10,
            dreams_per_iteration=100,
            recognition_epochs=15
        ),

        # Phase 3: Medium rules - palindromes and halves (depth 4-5)
        PhaseConfig(
            name="Phase 3: Medium Rules (depth 4-5)",
            iterations=5,
            use_all_rules=False,
            enumeration_budget=600000,
            max_depth=12,
            dreams_per_iteration=150,
            recognition_epochs=15
        ),

        # Phase 4: Hard rules - complex halves (depth 5-7)
        PhaseConfig(
            name="Phase 4: Hard Rules (depth 5-7)",
            iterations=6,
            use_all_rules=False,
            enumeration_budget=800000,
            max_depth=14,
            dreams_per_iteration=200,
            recognition_epochs=20
        ),

        # Phase 5: Very hard rules (depth 7-10)
        PhaseConfig(
            name="Phase 5: Very Hard Rules (depth 7-10)",
            iterations=6,
            use_all_rules=False,
            enumeration_budget=1000000,
            max_depth=16,
            dreams_per_iteration=250,
            recognition_epochs=20
        ),

        # Phase 6: Extreme rules - push limits (depth 10+)
        PhaseConfig(
            name="Phase 6: Extreme Rules (depth 10+)",
            iterations=8,
            use_all_rules=True,
            enumeration_budget=1500000,
            max_depth=18,
            dreams_per_iteration=300,
            recognition_epochs=25
        ),
    ]


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extended primitives overnight run - full lambda calculus test"
    )
    parser.add_argument("--dry-run", action="store_true", help="Quick test run")
    parser.add_argument("--phase", type=int, default=0,
                        help="Start from specific phase (1-6), 0=all")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint directory")
    args = parser.parse_args()

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "extended_prims" if not args.dry_run else "extended_dryrun"
    log_dir = Path(f"results/overnight_extended/{run_name}_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging
    logger = setup_logging(log_dir)

    print_banner("OVERNIGHT RUN: EXTENDED PRIMITIVES")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Log directory: {log_dir}")
    logger.info(f"PyPy: {'Available' if USE_PYPY else 'Not available'}")
    logger.info(f"Workers: {N_WORKERS}")
    print()

    # Build grammar with extended primitives
    logger.info("Building grammar with extended primitives...")
    grammar = build_extended_grammar()
    prims = build_extended_primitives()
    logger.info(f"Grammar: {len(grammar)} primitives")

    # Log new primitives
    new_primitive_categories = {
        'fold_prims': ['fold', 'foldr', 'cons', 'empty', 'tail', 'is_empty'],
        'pair_prims': ['pair', 'fst', 'snd', 'thd', 'triple'],
        'bool_agg': ['all_true', 'any_true', 'none_true', 'count_true'],
        'halves': ['first_half', 'second_half', 'list_eq'],
        'seq_checks': ['is_sorted_by', 'is_strictly_sorted_by', 'is_palindrome_by'],
        'ext_arith': ['abs', 'max2', 'min2', '*', '//'],
        'suit_cycles': ['cycle_m1', 'cycle_m2'],
        'altcolor': ['get_altcolor1', 'get_altcolor2', 'get_parity', 'is_odd_rank', 'is_even_rank']
    }

    prim_names = {p.name for p in prims}
    logger.info("\nNew primitives by category:")
    for category, expected in new_primitive_categories.items():
        present = [p for p in expected if p in prim_names]
        missing = [p for p in expected if p not in prim_names]
        logger.info(f"  {category}: {len(present)}/{len(expected)}")
        for p in present:
            logger.info(f"    ✓ {p}")
        for p in missing:
            logger.info(f"    ✗ {p} MISSING!")
    print()

    # Classify rules by expected depth
    logger.info("Classifying rules by expected program depth...")
    classified = classify_rules_by_depth()
    for phase_name, rules in classified.items():
        logger.info(f"  {phase_name}: {len(rules)} rules")
        for r in rules[:3]:
            logger.info(f"    - {r.id}")
        if len(rules) > 3:
            logger.info(f"    ... and {len(rules) - 3} more")

    total_rules = sum(len(v) for v in classified.values())
    logger.info(f"  Total: {total_rules} rules")
    print()

    # Create tasks
    logger.info("Creating tasks...")
    trivial_tasks = create_tasks_from_rules(classified['trivial'])
    easy_tasks = create_tasks_from_rules(classified['easy'])
    medium_tasks = create_tasks_from_rules(classified['medium'])
    hard_tasks = create_tasks_from_rules(classified['hard'])
    very_hard_tasks = create_tasks_from_rules(classified['very_hard'])
    extreme_tasks = create_tasks_from_rules(classified['extreme'])

    all_tasks = trivial_tasks + easy_tasks + medium_tasks + hard_tasks + very_hard_tasks + extreme_tasks
    logger.info(f"Created {len(all_tasks)} tasks")

    # Log task balance
    for t in all_tasks[:5]:
        pos_count = sum(1 for ex in t.examples if ex[1])
        neg_count = len(t.examples) - pos_count
        logger.debug(f"  {t.name}: {pos_count}+ / {neg_count}-")
    print()

    # Configure phases
    phases = create_phase_configs(args.dry_run)
    if args.phase > 0:
        phases = phases[args.phase - 1:]
        logger.info(f"Starting from Phase {args.phase}")

    # Task sets for progressive curriculum
    task_sets = {
        0: trivial_tasks,
        1: trivial_tasks + easy_tasks,
        2: trivial_tasks + easy_tasks + medium_tasks,
        3: trivial_tasks + easy_tasks + medium_tasks + hard_tasks,
        4: trivial_tasks + easy_tasks + medium_tasks + hard_tasks + very_hard_tasks,
        5: all_tasks
    }

    # Save run configuration
    config = {
        "run_type": "extended_primitives_full_test",
        "grammar_size": len(grammar),
        "primitives_added": sum(len(v) for v in new_primitive_categories.values()),
        "classification": {k: len(v) for k, v in classified.items()},
        "phases": [
            {
                "name": p.name,
                "iterations": p.iterations,
                "budget": p.enumeration_budget,
                "max_depth": p.max_depth
            }
            for p in phases
        ],
        "start_time": datetime.now().isoformat(),
        "dry_run": args.dry_run
    }
    with open(log_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print_banner("STARTING CURRICULUM TRAINING")
    logger.info(f"Output directory: {log_dir}")
    logger.info(f"Phases: {len(phases)}")
    estimated_hours = (3 + 4 + 5 + 6 + 6 + 8) * 0.3 if not args.dry_run else 0.1
    logger.info(f"Estimated runtime: {estimated_hours:.1f} hours")
    print()

    # Evaluation function
    def eval_fn(program, hand):
        fn = program.evaluate([])
        return fn(hand)

    # Initialize DreamCoder
    dc = CythonOptimizedDreamCoder(
        grammar=grammar,
        easy_tasks=trivial_tasks + easy_tasks,
        all_tasks=all_tasks,
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

    # Override get_active_tasks for phase-based selection
    def get_active_tasks_progressive():
        """Return tasks based on current phase."""
        phase_idx = dc.current_phase_idx
        if args.dry_run:
            return all_tasks[:10]  # Just a few for dry run
        return task_sets.get(phase_idx, all_tasks)

    dc.get_active_tasks = get_active_tasks_progressive

    # Initialize frontiers for all tasks
    for task in all_tasks:
        if task.name not in dc.frontiers:
            from dreamcoder_core.dreamcoder_v2 import TaskFrontier
            dc.frontiers[task.name] = TaskFrontier(task, max_size=dc.keep_top_k)
        if task.name not in dc.task_metrics:
            from dreamcoder_core.dreamcoder_v2 import TaskMetrics
            dc.task_metrics[task.name] = TaskMetrics(task_name=task.name, family=task.family)

    # Run training
    start_time = time.time()
    results = dc.run()
    total_time = time.time() - start_time

    # Final summary
    print_banner("OVERNIGHT RUN COMPLETE")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Total time: {format_time(total_time)}")
    print()

    solved_count = results['summary']['tasks_solved']
    total_count = results['summary']['tasks_total']
    logger.info(f"Tasks solved: {solved_count}/{total_count} ({100*solved_count/total_count:.1f}%)")
    logger.info(f"Final grammar: {results['summary']['final_grammar_size']} primitives")
    print()

    # Breakdown by difficulty
    solved_names = set(results.get('solved_tasks', {}).keys())

    breakdown = {}
    for phase_name, phase_tasks in [
        ('trivial', trivial_tasks),
        ('easy', easy_tasks),
        ('medium', medium_tasks),
        ('hard', hard_tasks),
        ('very_hard', very_hard_tasks),
        ('extreme', extreme_tasks)
    ]:
        solved = sum(1 for t in phase_tasks if t.name in solved_names)
        breakdown[phase_name] = f"{solved}/{len(phase_tasks)}"
        logger.info(f"  {phase_name}: {solved}/{len(phase_tasks)}")

    print()

    # Log unsolved rules by family
    unsolved = [t.name for t in all_tasks if t.name not in solved_names]
    unsolved_by_family = defaultdict(list)
    for task in all_tasks:
        if task.name not in solved_names:
            unsolved_by_family[task.family].append(task.name)

    logger.info(f"Unsolved rules by family ({len(unsolved)} total):")
    for family, rules in sorted(unsolved_by_family.items()):
        logger.info(f"  {family}: {len(rules)} - {rules[:3]}{'...' if len(rules) > 3 else ''}")

    # Update config with results
    config['end_time'] = datetime.now().isoformat()
    config['total_time_seconds'] = total_time
    config['tasks_solved'] = solved_count
    config['breakdown'] = breakdown
    config['unsolved_by_family'] = dict(unsolved_by_family)

    with open(log_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    logger.info(f"\nResults saved to: {log_dir}")

    # Print key insight
    print_banner("KEY INSIGHTS")
    if solved_count >= 45:
        logger.info("SUCCESS: Extended primitives significantly improved coverage!")
    elif solved_count >= 30:
        logger.info("PARTIAL SUCCESS: Some improvement, but paradigm limits reached.")
    else:
        logger.info("LIMITED IMPROVEMENT: Lambda calculus may not be ideal for these rules.")

    logger.info(f"\nCompare with previous run (8/57 solved) to see improvement.")


if __name__ == "__main__":
    main()
