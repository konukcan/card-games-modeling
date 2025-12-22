#!/usr/bin/env python3
"""
Overnight Run: List Primitives + Multi-Phase Curriculum

This script runs a multi-phase curriculum starting from easy rules
and progressively adding harder rules. Uses the newly added list
primitives (take, drop, zip_with, adjacent_pairs, half_len).

Curriculum Structure:
- Phase 1: Easy aggregate rules (COUNT, LOCAL without halves)
- Phase 2: Medium rules (POSITION, TOKEN, simpler HIER rules)
- Phase 3: Hard rules (PAL, COPY, SHIFT, ADJ, MAP)
- Phase 4: All remaining rules (LANG, CENTER, SCORE, complex AP)

Expected duration: 8-12 hours
Expected outcome: ~30-40 of 45 rules solved (vs 8 without list primitives)
"""

import sys
import os
import time
import json
import random
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
from dreamcoder_core.type_system import arrow, HAND, BOOL
from rules.catalogue import create_all_rules, Rule
from rules.cards import sample_hand


# ============================================================================
# RULE CLASSIFICATION BY DIFFICULTY
# ============================================================================

def classify_rules() -> dict:
    """
    Classify all catalogue rules by expected difficulty.

    Difficulty based on:
    - Required primitives (list vs aggregate)
    - Compositional depth
    - Rule family complexity
    """
    all_rules = create_all_rules()

    # Phase 1: Easy aggregate rules - should solve immediately
    easy_ids = {
        # COUNT family - basic cardinality
        'Uniform_color',           # all_same_color
        'Exactly_two_suits',       # n_unique_suits == 2
        'At_most_three_suits',     # n_unique_suits <= 3
        'Exactly_one_club',        # count_suit CLUBS == 1
        'Has_pair_ranks',          # n_unique_ranks < length
        'Half_or_more_same_suit',  # max count_suit >= half

        # LOCAL family - simple comparisons
        'Ends_same_suit',          # get_suit head == get_suit last
        'Ends_same_color',         # get_color head == get_color last
    }

    # Phase 2: Medium rules - need list ops or positions
    medium_ids = {
        # POSITION family
        'Pos3_is_JQK',             # at 2 in {J,Q,K}
        'Pos4_is_2_5_7',           # at 3 in {2,5,7}

        # TOKEN family
        'Has_Ace_of_Spades',       # any (suit==S and rank==A)
        'Has_6_of_Diamonds',       # any (suit==D and rank==6)

        # PARITY family
        'Only_one_odd_rank',       # count odd == 1
        'Uniform_rank_parity',     # all same parity

        # Simpler HIER (need halves but simple properties)
        'Halves_uniform_color_equal',    # uniform_color left == uniform_color right
        'Halves_hearts_presence_equal',  # has_heart left == has_heart right
        'Halves_same_suit_set',          # unique suits left == unique suits right

        # LOCAL - ordering (needs adjacent_pairs)
        'Sorted_by_rank',          # all adjacent pairs non-decreasing
        'S_before_H',              # exists spade before heart

        # ALT COLOR endpoints
        'Ends_same_altcolor1',     # first/last same pointy/round
    }

    # Phase 3: Hard rules - complex list operations
    hard_ids = {
        # PAL family (need reverse + zip_with)
        'Suits_palindrome',
        'Colors_palindrome',
        'Ranks_palindrome',
        'AltColor1_palindrome',
        'AltColor2_palindrome',

        # COPY family (need take/drop + zip_with)
        'Halves_copy_suits',
        'Halves_copy_colors',
        'Halves_copy_ranks',
        'Halves_copy_altcolor1',
        'Halves_copy_altcolor2',

        # SHIFT family (need shifted pairs)
        'Shift_half_plus_two',
        'Shift2_plus3',
        'Shift_half_ge',

        # ADJ family
        'Adj_same_rank_or_suit',
        'Skip2_same_rank_or_suit',
        'Adj_rank_gap_le3',

        # More HIER
        'Halves_uniform_parity_equal',
        'Halves_AP_step1_equal',
        'Halves_AP_len2_step1_equal',
    }

    # Phase 4: Very hard rules - complex compositions or unusual primitives
    very_hard_ids = {
        # LANG family (bracket matching - may need PDA)
        'Well_formed_brackets_by_suit',
        'Even_opens_next_closes',
        'Odd_opens_next_closes',

        # CENTER family
        'Halves_radial_nonincreasing',
        'Global_radial_no_dominance',

        # SCORE family
        'Score_threshold_Rstar',
        'Half_sum_diff_geN',
        'Half_sum_one_side_ge_2x_other',

        # AP family
        'AP_len3_anywhere_anyk',
        'AP_len3_step2_anywhere',
        'AP_len4_step2_anywhere',
        'Halves_AP_len3_any_equal',

        # MAP family (suit cycles)
        'Half_map_samepos_M1',
        'Half_map_samepos_M2',
        'Step2_back_map_M1',
        'Step2_back_map_M2',
        'Adj_same_or_map_M1',
        'Adj_same_or_map_M2',
    }

    # Build rule lists
    classified = {
        'easy': [],
        'medium': [],
        'hard': [],
        'very_hard': []
    }

    rule_dict = {r.id: r for r in all_rules}

    for rule_id in easy_ids:
        if rule_id in rule_dict:
            classified['easy'].append(rule_dict[rule_id])

    for rule_id in medium_ids:
        if rule_id in rule_dict:
            classified['medium'].append(rule_dict[rule_id])

    for rule_id in hard_ids:
        if rule_id in rule_dict:
            classified['hard'].append(rule_dict[rule_id])

    for rule_id in very_hard_ids:
        if rule_id in rule_dict:
            classified['very_hard'].append(rule_dict[rule_id])

    # Any remaining rules go to very_hard
    classified_ids = easy_ids | medium_ids | hard_ids | very_hard_ids
    for rule in all_rules:
        if rule.id not in classified_ids:
            classified['very_hard'].append(rule)

    return classified


def create_tasks_from_rules_list(rules: list, n_examples=100, n_holdout=20, hand_size=6, seed=42):
    """Create tasks from a list of Rule objects."""
    from dreamcoder_core.dreamcoder_original import Task

    tasks = []
    for rule in rules:
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


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Overnight run with list primitives")
    parser.add_argument("--dry-run", action="store_true", help="Quick test run")
    parser.add_argument("--phase", type=int, default=0,
                        help="Start from specific phase (1-4), 0=all")
    args = parser.parse_args()

    print_banner("OVERNIGHT RUN: LIST PRIMITIVES + CURRICULUM")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"PyPy: {'Available' if USE_PYPY else 'Not available'}")
    print(f"Workers: {N_WORKERS}")
    print()

    # Build grammar with new list primitives
    print("Building grammar...")
    grammar = build_lean_grammar()
    print(f"Grammar: {len(grammar)} primitives")

    # Check for new primitives
    prim_names = [p.name for p in grammar.primitives()]
    new_prims = ['take', 'drop', 'zip_with', 'adjacent_pairs', 'half_len']
    for p in new_prims:
        if p in prim_names:
            print(f"  ✓ {p}")
        else:
            print(f"  ✗ {p} MISSING!")
    print()

    # Classify rules by difficulty
    print("Classifying rules by difficulty...")
    classified = classify_rules()
    print(f"  Easy: {len(classified['easy'])} rules")
    print(f"  Medium: {len(classified['medium'])} rules")
    print(f"  Hard: {len(classified['hard'])} rules")
    print(f"  Very hard: {len(classified['very_hard'])} rules")
    total_rules = sum(len(v) for v in classified.values())
    print(f"  Total: {total_rules} rules")
    print()

    # Create tasks for each phase
    print("Creating tasks...")
    easy_tasks = create_tasks_from_rules_list(classified['easy'])
    medium_tasks = create_tasks_from_rules_list(classified['medium'])
    hard_tasks = create_tasks_from_rules_list(classified['hard'])
    very_hard_tasks = create_tasks_from_rules_list(classified['very_hard'])

    all_tasks = easy_tasks + medium_tasks + hard_tasks + very_hard_tasks
    print(f"Created {len(all_tasks)} tasks")
    print()

    # Configure phases
    if args.dry_run:
        print("*** DRY RUN MODE ***")
        phases = [
            PhaseConfig(
                name="Phase 1: Easy (dry run)",
                iterations=2,
                use_all_rules=False,
                enumeration_budget=50000,
                max_depth=10,
                dreams_per_iteration=20,
                recognition_epochs=5
            ),
        ]
        # Just use easy + a few medium for dry run
        all_tasks = easy_tasks + medium_tasks[:3]
    else:
        phases = [
            # Phase 1: Easy aggregate rules - quick wins
            PhaseConfig(
                name="Phase 1: Easy Rules",
                iterations=4,
                use_all_rules=False,  # Just easy tasks
                enumeration_budget=300000,
                max_depth=10,
                dreams_per_iteration=100,
                recognition_epochs=15
            ),
            # Phase 2: Add medium rules - more iterations
            PhaseConfig(
                name="Phase 2: Medium Rules",
                iterations=6,
                use_all_rules=False,  # Easy + medium
                enumeration_budget=500000,
                max_depth=12,
                dreams_per_iteration=150,
                recognition_epochs=20
            ),
            # Phase 3: Add hard rules - longer enumeration
            PhaseConfig(
                name="Phase 3: Hard Rules",
                iterations=6,
                use_all_rules=False,  # Easy + medium + hard
                enumeration_budget=800000,
                max_depth=14,
                dreams_per_iteration=200,
                recognition_epochs=20
            ),
            # Phase 4: All rules - intensive search
            PhaseConfig(
                name="Phase 4: All Rules",
                iterations=8,
                use_all_rules=True,
                enumeration_budget=1000000,
                max_depth=15,
                dreams_per_iteration=250,
                recognition_epochs=25
            ),
        ]

    # If starting from specific phase, adjust
    if args.phase > 0:
        phases = phases[args.phase-1:]
        print(f"Starting from Phase {args.phase}")

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "listprims_curriculum" if not args.dry_run else "listprims_dryrun"
    log_dir = Path(f"results/overnight_listprims/{run_name}_{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save run config
    config = {
        "run_type": "listprims_curriculum",
        "grammar_size": len(grammar),
        "new_primitives": new_prims,
        "n_easy": len(classified['easy']),
        "n_medium": len(classified['medium']),
        "n_hard": len(classified['hard']),
        "n_very_hard": len(classified['very_hard']),
        "total_rules": total_rules,
        "phases": [
            {
                "name": p.name,
                "iterations": p.iterations,
                "budget": p.enumeration_budget
            }
            for p in phases
        ],
        "start_time": datetime.now().isoformat(),
        "dry_run": args.dry_run
    }
    with open(log_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print_banner("STARTING CURRICULUM TRAINING")
    print(f"Output directory: {log_dir}")
    print(f"Phases: {len(phases)}")
    print()

    # Create evaluation function
    def eval_fn(program, hand):
        fn = program.evaluate([])
        return fn(hand)

    # Store task sets for phase-based selection (cumulative)
    task_sets = {
        0: easy_tasks,
        1: easy_tasks + medium_tasks,
        2: easy_tasks + medium_tasks + hard_tasks,
        3: all_tasks
    }

    # Initialize DreamCoder
    dc = CythonOptimizedDreamCoder(
        grammar=grammar,
        easy_tasks=easy_tasks,
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

    # Override get_active_tasks to use phase-based task selection
    def get_active_tasks_progressive():
        """Return tasks based on current phase."""
        phase_idx = dc.current_phase_idx
        if args.dry_run:
            return all_tasks
        return task_sets.get(phase_idx, all_tasks)

    dc.get_active_tasks = get_active_tasks_progressive

    # Ensure frontiers exist for all tasks upfront
    for task in all_tasks:
        if task.name not in dc.frontiers:
            from dreamcoder_core.dreamcoder_original import TaskFrontier
            dc.frontiers[task.name] = TaskFrontier(task, max_size=dc.keep_top_k)
        if task.name not in dc.task_metrics:
            from dreamcoder_core.dreamcoder_original import TaskMetrics
            dc.task_metrics[task.name] = TaskMetrics(task_name=task.name, family=task.family)

    # Run training
    start_time = time.time()
    results = dc.run()
    total_time = time.time() - start_time

    # Final summary
    print_banner("OVERNIGHT RUN COMPLETE")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {format_time(total_time)}")
    print()

    solved_count = results['summary']['tasks_solved']
    total_count = results['summary']['tasks_total']
    print(f"Tasks solved: {solved_count}/{total_count} ({100*solved_count/total_count:.1f}%)")
    print(f"Final grammar: {results['summary']['final_grammar_size']} primitives")
    print()

    # Breakdown by difficulty
    solved_names = set(results.get('solved_tasks', {}).keys())

    easy_solved = sum(1 for t in easy_tasks if t.name in solved_names)
    medium_solved = sum(1 for t in medium_tasks if t.name in solved_names)
    hard_solved = sum(1 for t in hard_tasks if t.name in solved_names)
    very_hard_solved = sum(1 for t in very_hard_tasks if t.name in solved_names)

    print("Breakdown by difficulty:")
    print(f"  Easy: {easy_solved}/{len(easy_tasks)}")
    print(f"  Medium: {medium_solved}/{len(medium_tasks)}")
    print(f"  Hard: {hard_solved}/{len(hard_tasks)}")
    print(f"  Very hard: {very_hard_solved}/{len(very_hard_tasks)}")
    print()

    print(f"Results saved to: {log_dir}")

    # Update config with final results
    config['end_time'] = datetime.now().isoformat()
    config['total_time_seconds'] = total_time
    config['tasks_solved'] = solved_count
    config['breakdown'] = {
        'easy': f"{easy_solved}/{len(easy_tasks)}",
        'medium': f"{medium_solved}/{len(medium_tasks)}",
        'hard': f"{hard_solved}/{len(hard_tasks)}",
        'very_hard': f"{very_hard_solved}/{len(very_hard_tasks)}"
    }
    with open(log_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
