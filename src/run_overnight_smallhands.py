#!/usr/bin/env python3
"""
Smaller Hands Overnight Pre-training Runner

This script uses a curriculum that starts with smaller hands (3 cards),
progressively increasing to 4, 5, and finally 6 cards. The hypothesis is
that smaller hands have a simpler search space, allowing the system to
discover useful primitives and abstractions before tackling full-size hands.

Architecture:
- Phase 1: 3-card hands (easy rules only)
- Phase 2: 4-card hands (easy rules)
- Phase 3: 5-card hands (all rules)
- Phase 4: 6-card hands with full library (all rules)
"""

import sys
import os
import time
import json
import random
import copy
import pickle
import subprocess
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import torch

USE_CYTHON = False

from dreamcoder_core.type_system import arrow, HAND, BOOL
from dreamcoder_core.program import Program, Primitive, Invented, parse_program
from dreamcoder_core.grammar import Grammar
from dreamcoder_core.enumeration import enumerate_simple

print("Using Python modules (PyPy workers provide ~3-6x speedup for enumeration)")

from dreamcoder_core.compression import compress_frontiers
from dreamcoder_core.neural_recognition import NeuralRecognitionModel
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.dreamcoder_v2 import (
    Task, SolutionEntry, TaskFrontier, IterationMetrics, TaskMetrics,
    NeuralDreamer, create_tasks_from_rules, make_eval_fn
)
from rules.pretraining_rules import (
    get_all_pretraining_rules, get_easy_pretraining_rules
)

# Import the optimized runner components
from run_overnight_cython import (
    PhaseConfig, CythonOptimizedDreamCoder, print_banner, format_time,
    USE_PYPY, PYPY_PATH, N_WORKERS, USE_MULTIPROCESSING
)


# ============================================================================
# MAIN
# ============================================================================

def main():
    start_time = time.time()

    print_banner("DREAMCODER V2 - SMALLER HANDS CURRICULUM OVERNIGHT PRETRAINING")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check optimizations
    print("Checking optimizations...")
    print(f"  Cython modules: {'ENABLED' if USE_CYTHON else 'DISABLED (Python fallback)'}")
    print(f"  PyPy available: {USE_PYPY} ({PYPY_PATH})")
    print(f"  Multiprocessing: {USE_MULTIPROCESSING}")
    print(f"  Workers: {N_WORKERS}")
    print()

    # Load rules
    easy_rules = get_easy_pretraining_rules()
    all_rules = get_all_pretraining_rules()

    print(f"Easy rules (level 1): {len(easy_rules)}")
    print(f"All rules (levels 1-2): {len(all_rules)}")

    # Create tasks at different hand sizes
    print("\n" + "=" * 60)
    print("CREATING TASKS AT DIFFERENT HAND SIZES")
    print("=" * 60)

    # 3-card hands - simpler search space
    print("\nCreating 3-card hand tasks...")
    easy_tasks_3 = create_tasks_from_rules(easy_rules, n_examples=100, n_holdout=20, hand_size=3, seed=42)
    all_tasks_3 = create_tasks_from_rules(all_rules, n_examples=100, n_holdout=20, hand_size=3, seed=42)
    print(f"  Created {len(easy_tasks_3)} easy tasks, {len(all_tasks_3)} total tasks (3 cards)")

    # 4-card hands
    print("\nCreating 4-card hand tasks...")
    easy_tasks_4 = create_tasks_from_rules(easy_rules, n_examples=100, n_holdout=20, hand_size=4, seed=43)
    all_tasks_4 = create_tasks_from_rules(all_rules, n_examples=100, n_holdout=20, hand_size=4, seed=43)
    print(f"  Created {len(easy_tasks_4)} easy tasks, {len(all_tasks_4)} total tasks (4 cards)")

    # 5-card hands
    print("\nCreating 5-card hand tasks...")
    easy_tasks_5 = create_tasks_from_rules(easy_rules, n_examples=100, n_holdout=20, hand_size=5, seed=44)
    all_tasks_5 = create_tasks_from_rules(all_rules, n_examples=100, n_holdout=20, hand_size=5, seed=44)
    print(f"  Created {len(easy_tasks_5)} easy tasks, {len(all_tasks_5)} total tasks (5 cards)")

    # 6-card hands (standard)
    print("\nCreating 6-card hand tasks...")
    easy_tasks_6 = create_tasks_from_rules(easy_rules, n_examples=100, n_holdout=20, hand_size=6, seed=45)
    all_tasks_6 = create_tasks_from_rules(all_rules, n_examples=100, n_holdout=20, hand_size=6, seed=45)
    print(f"  Created {len(easy_tasks_6)} easy tasks, {len(all_tasks_6)} total tasks (6 cards)")

    # Build grammar
    print("\nBuilding lean grammar...")
    grammar = build_lean_grammar()
    print(f"Grammar: {len(grammar)} primitives")

    # Define phases with progressive hand sizes
    # Key insight: smaller hands = smaller search space = faster convergence
    phases = [
        # Phase 1: Start with 3-card hands (very simple)
        PhaseConfig(
            name="Phase 1: 3-Card Easy Rules",
            iterations=3,
            use_all_rules=False,
            enumeration_budget=100000,  # Smaller budget needed
            max_depth=7,
            dreams_per_iteration=50,
            recognition_epochs=10
        ),
        # Phase 2: Move to 4-card hands
        PhaseConfig(
            name="Phase 2: 4-Card Easy Rules",
            iterations=3,
            use_all_rules=False,
            enumeration_budget=150000,
            max_depth=8,
            dreams_per_iteration=75,
            recognition_epochs=12
        ),
        # Phase 3: 5-card hands with all rules
        PhaseConfig(
            name="Phase 3: 5-Card All Rules",
            iterations=4,
            use_all_rules=True,
            enumeration_budget=200000,
            max_depth=8,
            dreams_per_iteration=100,
            recognition_epochs=15
        ),
        # Phase 4: Full 6-card hands
        PhaseConfig(
            name="Phase 4: 6-Card Easy Foundation",
            iterations=4,
            use_all_rules=False,
            enumeration_budget=200000,
            max_depth=8,
            dreams_per_iteration=100,
            recognition_epochs=15
        ),
        # Phase 5: Deep 6-card search
        PhaseConfig(
            name="Phase 5: 6-Card All Rules Deep",
            iterations=6,
            use_all_rules=True,
            enumeration_budget=400000,
            max_depth=10,
            dreams_per_iteration=150,
            recognition_epochs=20
        )
    ]

    # Create output directory (DIFFERENT from main run!)
    log_dir = Path("results/overnight_smallhands")
    log_dir.mkdir(parents=True, exist_ok=True)

    # Create eval function
    eval_fn = make_eval_fn()

    # We need a custom runner that switches tasks between phases
    print_banner("STARTING SMALLER HANDS CURRICULUM TRAINING")

    # For this curriculum, we'll run multiple DreamCoder instances
    # Each phase uses different hand-size tasks but shares grammar/recognition

    total_solved = 0
    final_grammar = grammar
    all_results = []

    # Task sets for each phase
    phase_tasks = [
        (easy_tasks_3, all_tasks_3),  # Phase 1
        (easy_tasks_4, all_tasks_4),  # Phase 2
        (easy_tasks_5, all_tasks_5),  # Phase 3
        (easy_tasks_6, all_tasks_6),  # Phase 4
        (easy_tasks_6, all_tasks_6),  # Phase 5
    ]

    recognition_model = None

    for phase_idx, phase in enumerate(phases):
        easy_tasks, all_tasks = phase_tasks[phase_idx]

        print_banner(f"PHASE {phase_idx + 1}: {phase.name}")
        print(f"Tasks: {len(easy_tasks)} easy, {len(all_tasks)} total")

        # Create DreamCoder for this phase
        dc = CythonOptimizedDreamCoder(
            grammar=final_grammar,
            easy_tasks=easy_tasks,
            all_tasks=all_tasks,
            eval_fn=eval_fn,
            phases=[phase],  # Single phase
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

        # Transfer recognition model from previous phase
        if recognition_model is not None:
            dc.recognition = recognition_model
            dc.dreamer.recognition_model = recognition_model

        # Run this phase
        phase_results = dc.run()
        all_results.append(phase_results)

        # Carry forward grammar and recognition
        final_grammar = dc.grammar
        recognition_model = dc.recognition

        # Update solve count
        phase_solved = phase_results['summary']['tasks_solved']
        print(f"\nPhase {phase_idx + 1} solved: {phase_solved}/{phase_results['summary']['tasks_total']}")

        # Save intermediate checkpoint
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        checkpoint_path = log_dir / f"phase{phase_idx+1}_checkpoint_{timestamp}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(phase_results, f, indent=2, default=str)

    # Final summary
    total_time = time.time() - start_time

    print_banner("SMALLER HANDS CURRICULUM OVERNIGHT PRETRAINING COMPLETE")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {format_time(total_time)}")
    print()

    # Aggregate results
    print("RESULTS BY PHASE:")
    for i, result in enumerate(all_results):
        print(f"  Phase {i+1}: {result['summary']['tasks_solved']}/{result['summary']['tasks_total']} solved")

    print(f"\nFinal grammar: {len(final_grammar)} primitives")

    # Save final combined results
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    final_results = {
        'method': 'smaller_hands_curriculum',
        'total_time': total_time,
        'phases': [
            {
                'name': phases[i].name,
                'solved': all_results[i]['summary']['tasks_solved'],
                'total': all_results[i]['summary']['tasks_total'],
                'grammar_size': all_results[i]['summary']['final_grammar_size']
            }
            for i in range(len(all_results))
        ],
        'final_grammar_size': len(final_grammar),
        'detailed_results': all_results
    }

    final_path = log_dir / f"smallhands_final_{timestamp}.json"
    with open(final_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\nResults saved to: {final_path}")


if __name__ == "__main__":
    main()
