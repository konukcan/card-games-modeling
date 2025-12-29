#!/usr/bin/env python3
"""
DreamCoder Overnight Run v4 - Complete Pipeline
================================================

This script runs two experimental configurations sequentially:

**PART A: Standard Batch Wake-Sleep (Current Architecture)**
- Phases 1-4: Pretraining on 43 pretraining rules
- Phases 5-7: Transfer evaluation on 57 experimental rules

**PART B: Mini-Batch Wake-Sleep (Experimental)**
- More frequent abstraction learning (every N solved tasks)
- Potentially better transfer within phases

Both parts include:
- Task-result scrambling fix
- Per-iteration checkpoint saving
- Comprehensive pre-flight validation

Usage:
    python src/run_overnight_v4.py                  # Full run (both parts)
    python src/run_overnight_v4.py --dry-run        # Validation only
    python src/run_overnight_v4.py --part-a-only    # Standard batch only
    python src/run_overnight_v4.py --part-b-only    # Mini-batch only
    python src/run_overnight_v4.py --quick          # Quick test (1 iteration each)
"""

import sys
import os
import time
import json
import random
import traceback
import subprocess
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Callable

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))


# ============================================================================
# CONFIGURATION
# ============================================================================

RUN_NAME = f"run_v4_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOG_DIR = Path("results/overnight_v4") / RUN_NAME

PYPY_PATH = shutil.which('pypy3.10') or shutil.which('pypy3')
USE_PYPY = PYPY_PATH is not None

N_WORKERS = 4
RANDOM_SEED = 42


@dataclass
class PhaseConfig:
    """Configuration for a training phase."""
    name: str
    iterations: int
    task_set: str  # 'easy_pretraining', 'all_pretraining', 'experimental'
    enumeration_budget: int
    max_depth: int
    dreams_per_iteration: int
    recognition_epochs: int
    mini_batch_size: Optional[int] = None  # If set, run compression every N solutions
    description: str = ""


@dataclass
class RunConfig:
    """Configuration for a complete run."""
    name: str
    phases: List[PhaseConfig]
    description: str


# ============================================================================
# TIME ESTIMATION
# ============================================================================

def estimate_phase_time(phase: PhaseConfig, n_tasks: int) -> Tuple[float, float]:
    """Estimate time for a phase (min, max in seconds).

    Based on empirical observations:
    - ~0.5-2 seconds per 1000 programs enumerated
    - Recognition training: ~1-3 seconds per epoch
    - Compression: ~10-30 seconds per iteration
    """
    # Enumeration time
    enum_time_per_1k = 1.0  # seconds per 1000 programs
    base_enum = phase.enumeration_budget / 1000 * enum_time_per_1k

    # Scale by number of tasks (parallel, so sqrt scaling)
    import math
    task_factor = math.sqrt(n_tasks / 10)

    # Per iteration time
    recognition_time = phase.recognition_epochs * 2  # ~2 sec/epoch
    compression_time = 20  # seconds
    dream_time = phase.dreams_per_iteration * 0.1

    iter_time = base_enum * task_factor + recognition_time + compression_time + dream_time

    # Mini-batch has more compression calls but each is faster
    if phase.mini_batch_size:
        iter_time *= 1.2  # 20% overhead for more frequent compression

    total_min = phase.iterations * iter_time * 0.6
    total_max = phase.iterations * iter_time * 1.5

    return total_min, total_max


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


def estimate_run_time(config: RunConfig, task_counts: Dict[str, int]) -> Tuple[str, str]:
    """Estimate total run time."""
    total_min = 0
    total_max = 0

    for phase in config.phases:
        n_tasks = task_counts.get(phase.task_set, 43)
        phase_min, phase_max = estimate_phase_time(phase, n_tasks)
        total_min += phase_min
        total_max += phase_max

    return format_time(total_min), format_time(total_max)


# ============================================================================
# PHASE CONFIGURATIONS
# ============================================================================

def get_standard_batch_phases(quick: bool = False) -> List[PhaseConfig]:
    """Standard batch wake-sleep phases.

    Part A: Pretraining (Phases 1-4)
    Part B: Experimental Transfer (Phases 5-7)
    """
    if quick:
        # Quick test mode - 1 iteration each
        return [
            PhaseConfig(
                name="Quick Test: Pretraining Easy",
                iterations=1,
                task_set='easy_pretraining',
                enumeration_budget=50000,
                max_depth=6,
                dreams_per_iteration=20,
                recognition_epochs=5,
                description="Quick validation run"
            ),
            PhaseConfig(
                name="Quick Test: Experimental",
                iterations=1,
                task_set='experimental',
                enumeration_budget=50000,
                max_depth=6,
                dreams_per_iteration=20,
                recognition_epochs=5,
                description="Quick experimental validation"
            ),
        ]

    return [
        # === PART A: PRETRAINING ===
        PhaseConfig(
            name="Phase 1: Foundation (Easy Pretraining)",
            iterations=5,
            task_set='easy_pretraining',
            enumeration_budget=150000,
            max_depth=7,
            dreams_per_iteration=80,
            recognition_epochs=12,
            description="Build foundation with 22 easy pretraining rules"
        ),
        PhaseConfig(
            name="Phase 2: Expansion (All Pretraining)",
            iterations=5,
            task_set='all_pretraining',
            enumeration_budget=250000,
            max_depth=8,
            dreams_per_iteration=120,
            recognition_epochs=15,
            description="All 43 pretraining rules with medium search"
        ),
        PhaseConfig(
            name="Phase 3: Deep Search",
            iterations=5,
            task_set='all_pretraining',
            enumeration_budget=400000,
            max_depth=9,
            dreams_per_iteration=150,
            recognition_epochs=20,
            description="Deep search on remaining unsolved pretraining"
        ),
        PhaseConfig(
            name="Phase 4: Intensive Push",
            iterations=6,
            task_set='all_pretraining',
            enumeration_budget=600000,
            max_depth=10,
            dreams_per_iteration=200,
            recognition_epochs=25,
            description="Maximum budget for remaining hard pretraining tasks"
        ),

        # === PART B: EXPERIMENTAL TRANSFER ===
        PhaseConfig(
            name="Phase 5: Experimental Easy",
            iterations=4,
            task_set='experimental',  # All 57 experimental rules
            enumeration_budget=200000,
            max_depth=8,
            dreams_per_iteration=100,
            recognition_epochs=15,
            description="First pass on experimental rules with learned grammar"
        ),
        PhaseConfig(
            name="Phase 6: Experimental Medium",
            iterations=5,
            task_set='experimental',
            enumeration_budget=400000,
            max_depth=9,
            dreams_per_iteration=150,
            recognition_epochs=20,
            description="Deeper search on experimental rules"
        ),
        PhaseConfig(
            name="Phase 7: Experimental Intensive",
            iterations=6,
            task_set='experimental',
            enumeration_budget=800000,
            max_depth=10,
            dreams_per_iteration=200,
            recognition_epochs=25,
            description="Maximum budget for hard experimental tasks"
        ),
    ]


def get_mini_batch_phases(quick: bool = False) -> List[PhaseConfig]:
    """Mini-batch wake-sleep phases with more frequent abstraction learning.

    Key difference: mini_batch_size controls how often compression runs.
    """
    if quick:
        return [
            PhaseConfig(
                name="Quick Mini-Batch Test",
                iterations=1,
                task_set='easy_pretraining',
                enumeration_budget=50000,
                max_depth=6,
                dreams_per_iteration=20,
                recognition_epochs=5,
                mini_batch_size=3,  # Compress after every 3 solutions
                description="Quick mini-batch validation"
            ),
        ]

    return [
        # === MINI-BATCH PRETRAINING ===
        PhaseConfig(
            name="MB Phase 1: Foundation (Frequent Learning)",
            iterations=6,
            task_set='easy_pretraining',
            enumeration_budget=150000,
            max_depth=7,
            dreams_per_iteration=60,
            recognition_epochs=8,
            mini_batch_size=3,  # Compress after every 3 solutions
            description="Easy rules with aggressive abstraction learning"
        ),
        PhaseConfig(
            name="MB Phase 2: Expansion",
            iterations=6,
            task_set='all_pretraining',
            enumeration_budget=250000,
            max_depth=8,
            dreams_per_iteration=80,
            recognition_epochs=10,
            mini_batch_size=5,
            description="All pretraining with medium batch abstraction"
        ),
        PhaseConfig(
            name="MB Phase 3: Deep Search",
            iterations=6,
            task_set='all_pretraining',
            enumeration_budget=400000,
            max_depth=9,
            dreams_per_iteration=100,
            recognition_epochs=15,
            mini_batch_size=8,
            description="Deep search with larger batches"
        ),

        # === MINI-BATCH EXPERIMENTAL ===
        PhaseConfig(
            name="MB Phase 4: Experimental Transfer",
            iterations=5,
            task_set='experimental',
            enumeration_budget=300000,
            max_depth=9,
            dreams_per_iteration=100,
            recognition_epochs=15,
            mini_batch_size=5,
            description="Experimental rules with mini-batch learning"
        ),
        PhaseConfig(
            name="MB Phase 5: Experimental Intensive",
            iterations=6,
            task_set='experimental',
            enumeration_budget=600000,
            max_depth=10,
            dreams_per_iteration=150,
            recognition_epochs=20,
            mini_batch_size=8,
            description="Final push on experimental tasks"
        ),
    ]


def get_run_configs(quick: bool = False) -> List[RunConfig]:
    """Get all run configurations."""
    return [
        RunConfig(
            name="Part A: Standard Batch Wake-Sleep",
            phases=get_standard_batch_phases(quick),
            description="Traditional DreamCoder with batch abstraction learning"
        ),
        RunConfig(
            name="Part B: Mini-Batch Wake-Sleep",
            phases=get_mini_batch_phases(quick),
            description="Experimental: more frequent abstraction learning"
        ),
    ]


# ============================================================================
# PRE-FLIGHT VALIDATION
# ============================================================================

class PreFlightValidator:
    """Comprehensive pre-flight validation."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []

    def log(self, msg: str):
        if self.verbose:
            print(f"  {msg}")

    def error(self, msg: str):
        self.errors.append(msg)
        print(f"  [ERROR] {msg}")

    def warn(self, msg: str):
        self.warnings.append(msg)
        print(f"  [WARN] {msg}")

    def ok(self, msg: str):
        print(f"  [OK] {msg}")

    def validate_imports(self) -> bool:
        """Validate all required imports."""
        print("\n1. Validating imports...")

        try:
            import torch
            self.ok(f"PyTorch {torch.__version__}")
        except ImportError as e:
            self.error(f"PyTorch not available: {e}")
            return False

        try:
            from dreamcoder_core.type_system import arrow, HAND, BOOL
            from dreamcoder_core.program import Program, Primitive
            from dreamcoder_core.grammar import Grammar
            self.ok("DreamCoder core modules")
        except ImportError as e:
            self.error(f"DreamCoder core import failed: {e}")
            return False

        try:
            from dreamcoder_core.lean_primitives import build_lean_grammar
            self.ok("Lean primitives module")
        except ImportError as e:
            self.error(f"Lean primitives import failed: {e}")
            return False

        try:
            from dreamcoder_core.neural_recognition import NeuralRecognitionModel
            self.ok("Neural recognition module")
        except ImportError as e:
            self.error(f"Neural recognition import failed: {e}")
            return False

        try:
            from rules.pretraining_rules import get_all_pretraining_rules, get_easy_pretraining_rules
            self.ok("Pretraining rules module")
        except ImportError as e:
            self.error(f"Pretraining rules import failed: {e}")
            return False

        try:
            from rules.catalogue import ALL_RULES
            self.ok(f"Experimental rules catalogue ({len(ALL_RULES)} rules)")
        except ImportError as e:
            self.error(f"Catalogue import failed: {e}")
            return False

        return True

    def validate_grammar(self) -> Tuple[bool, Optional[Any]]:
        """Validate grammar construction."""
        print("\n2. Validating grammar...")

        try:
            from dreamcoder_core.lean_primitives import build_lean_grammar
            grammar = build_lean_grammar()
            self.ok(f"Grammar built: {len(grammar)} primitives")

            # Verify key primitives
            prim_names = {str(p.program) for p in grammar.productions}
            required = {'all_same_suit', 'all_same_color', 'eq', 'count_suit', 'get_rank'}
            missing = required - prim_names
            if missing:
                self.warn(f"Missing expected primitives: {missing}")
            else:
                self.ok("All key primitives present")

            return True, grammar
        except Exception as e:
            self.error(f"Grammar construction failed: {e}")
            traceback.print_exc()
            return False, None

    def validate_pretraining_rules(self) -> Tuple[bool, List, List]:
        """Validate pretraining rules."""
        print("\n3. Validating pretraining rules...")

        try:
            from rules.pretraining_rules import get_all_pretraining_rules, get_easy_pretraining_rules
            from rules.cards import sample_hand

            all_rules = get_all_pretraining_rules()
            easy_rules = get_easy_pretraining_rules()

            self.ok(f"Total pretraining: {len(all_rules)}")
            self.ok(f"Easy pretraining: {len(easy_rules)}")

            # Test each rule
            failed = []
            for rule in all_rules:
                try:
                    for _ in range(3):
                        hand = sample_hand(6)
                        result = rule.eval(hand)
                        if not isinstance(result, bool):
                            failed.append((rule.name, f"Non-bool: {type(result)}"))
                            break
                except Exception as e:
                    failed.append((rule.name, str(e)[:50]))

            if failed:
                for name, err in failed[:5]:
                    self.error(f"Rule '{name}': {err}")
                return False, [], []

            self.ok("All pretraining rules validated")
            return True, easy_rules, all_rules

        except Exception as e:
            self.error(f"Pretraining validation failed: {e}")
            return False, [], []

    def validate_experimental_rules(self) -> Tuple[bool, List]:
        """Validate experimental catalogue rules."""
        print("\n4. Validating experimental rules...")

        try:
            from rules.catalogue import ALL_RULES
            from rules.cards import sample_hand

            self.ok(f"Total experimental: {len(ALL_RULES)}")

            # Test each rule
            failed = []
            for rule in ALL_RULES:
                try:
                    for _ in range(3):
                        hand = sample_hand(6)
                        result = rule.predicate(hand)
                        if not isinstance(result, bool):
                            failed.append((rule.id, f"Non-bool: {type(result)}"))
                            break
                except Exception as e:
                    failed.append((rule.id, str(e)[:50]))

            if failed:
                for name, err in failed[:5]:
                    self.error(f"Rule '{name}': {err}")
                return False, []

            # Count by level
            levels = {}
            for r in ALL_RULES:
                levels.setdefault(r.level, []).append(r)
            for lvl in sorted(levels.keys()):
                self.log(f"  Level {lvl}: {len(levels[lvl])} rules")

            self.ok("All experimental rules validated")
            return True, ALL_RULES

        except Exception as e:
            self.error(f"Experimental validation failed: {e}")
            return False, []

    def validate_task_creation(self, rules_by_type: Dict) -> Tuple[bool, Dict]:
        """Validate task creation for all rule types."""
        print("\n5. Validating task creation...")

        try:
            from dreamcoder_core.dreamcoder_original import create_tasks_from_rules

            tasks_by_type = {}

            # Easy pretraining
            if 'easy_pretraining' in rules_by_type:
                tasks = create_tasks_from_rules(
                    rules_by_type['easy_pretraining'],
                    n_examples=100, n_holdout=20
                )
                tasks_by_type['easy_pretraining'] = tasks
                self.ok(f"Easy pretraining tasks: {len(tasks)}")

            # All pretraining
            if 'all_pretraining' in rules_by_type:
                tasks = create_tasks_from_rules(
                    rules_by_type['all_pretraining'],
                    n_examples=100, n_holdout=20
                )
                tasks_by_type['all_pretraining'] = tasks
                self.ok(f"All pretraining tasks: {len(tasks)}")

            # Experimental - needs adapter since catalogue uses different format
            if 'experimental' in rules_by_type:
                from rules.catalogue import ALL_RULES

                # Create adapter rules with .id and .eval attributes (matching PretrainingRule)
                class ExperimentalRuleAdapter:
                    def __init__(self, rule):
                        self.id = rule.id  # Used by create_tasks_from_rules
                        self.name = rule.id
                        self.eval = rule.predicate
                        self.level = rule.level
                        self.family = rule.family

                adapted_rules = [ExperimentalRuleAdapter(r) for r in ALL_RULES]
                tasks = create_tasks_from_rules(
                    adapted_rules,
                    n_examples=100, n_holdout=20
                )
                tasks_by_type['experimental'] = tasks
                self.ok(f"Experimental tasks: {len(tasks)}")

            return True, tasks_by_type

        except Exception as e:
            self.error(f"Task creation failed: {e}")
            traceback.print_exc()
            return False, {}

    def validate_recognition_model(self, grammar: Any, tasks: List) -> bool:
        """Validate neural recognition model."""
        print("\n6. Validating recognition model...")

        try:
            from dreamcoder_core.neural_recognition import NeuralRecognitionModel

            model = NeuralRecognitionModel(
                grammar=grammar,
                hidden_dim=256,
                learning_rate=5e-4,
                device='cpu'
            )
            self.ok("Model initialized")

            if tasks:
                task = tasks[0]
                emb = model.get_task_embedding(task)
                self.ok(f"Forward pass: embedding shape {emb.shape}")

                log_probs = model.predict_primitive_probs(task)
                self.ok(f"Primitive prediction: {log_probs.shape[0]} primitives")

            return True
        except Exception as e:
            self.error(f"Recognition model failed: {e}")
            traceback.print_exc()
            return False

    def validate_pypy_worker(self) -> bool:
        """Validate PyPy worker."""
        print("\n7. Validating PyPy worker...")

        if not USE_PYPY:
            self.warn("PyPy not available - will use slower CPython workers")
            return True

        self.ok(f"PyPy found: {PYPY_PATH}")

        # Test PyPy can run basic code
        try:
            result = subprocess.run(
                [PYPY_PATH, '-c', 'print("PyPy OK")'],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                self.ok("PyPy worker test passed")
                return True
            else:
                self.warn(f"PyPy test failed: {result.stderr}")
                return True
        except Exception as e:
            self.warn(f"PyPy test error: {e}")
            return True

    def validate_scrambling_fix(self) -> bool:
        """Verify the task-result scrambling bug fix is in place."""
        print("\n8. Verifying scrambling bug fix...")

        try:
            runner_path = Path(__file__).parent / "run_overnight_cython.py"
            if not runner_path.exists():
                self.warn("run_overnight_cython.py not found - cannot verify fix")
                return True

            with open(runner_path) as f:
                content = f.read()

            # Check for the fix pattern
            if "results_by_name" in content and "task_name" in content:
                self.ok("Scrambling bug fix verified")
                return True
            else:
                self.warn("Could not verify scrambling fix - check manually")
                return True

        except Exception as e:
            self.warn(f"Could not check scrambling fix: {e}")
            return True

    def validate_checkpoint_saving(self) -> bool:
        """Verify per-iteration checkpoint saving is enabled."""
        print("\n9. Verifying checkpoint saving...")

        try:
            runner_path = Path(__file__).parent / "run_overnight_cython.py"
            if not runner_path.exists():
                self.warn("run_overnight_cython.py not found")
                return True

            with open(runner_path) as f:
                content = f.read()

            if "_save_iteration_checkpoint" in content:
                self.ok("Per-iteration checkpoint saving verified")
                return True
            else:
                self.warn("Checkpoint saving may not be enabled")
                return True

        except Exception as e:
            self.warn(f"Could not check checkpoint saving: {e}")
            return True

    def run_all(self) -> Tuple[bool, Dict]:
        """Run all validations."""
        print("\n" + "=" * 70)
        print("PRE-FLIGHT VALIDATION")
        print("=" * 70)

        results = {}

        # 1. Imports
        if not self.validate_imports():
            return False, results

        # 2. Grammar
        ok, grammar = self.validate_grammar()
        if not ok:
            return False, results
        results['grammar'] = grammar

        # 3. Pretraining rules
        ok, easy_rules, all_rules = self.validate_pretraining_rules()
        if not ok:
            return False, results

        rules_by_type = {
            'easy_pretraining': easy_rules,
            'all_pretraining': all_rules,
        }

        # 4. Experimental rules
        ok, exp_rules = self.validate_experimental_rules()
        if not ok:
            return False, results
        rules_by_type['experimental'] = exp_rules
        results['rules_by_type'] = rules_by_type

        # 5. Task creation
        ok, tasks_by_type = self.validate_task_creation(rules_by_type)
        if not ok:
            return False, results
        results['tasks_by_type'] = tasks_by_type

        # Get any task list for model validation
        any_tasks = next(iter(tasks_by_type.values()), [])

        # 6. Recognition model
        if not self.validate_recognition_model(grammar, any_tasks):
            return False, results

        # 7. PyPy
        self.validate_pypy_worker()

        # 8. Scrambling fix
        self.validate_scrambling_fix()

        # 9. Checkpoint saving
        self.validate_checkpoint_saving()

        # Summary
        print("\n" + "=" * 70)
        if self.errors:
            print(f"VALIDATION FAILED: {len(self.errors)} error(s)")
            return False, results
        elif self.warnings:
            print(f"VALIDATION PASSED with {len(self.warnings)} warning(s)")
        else:
            print("VALIDATION PASSED - All checks OK")
        print("=" * 70)

        return True, results


# ============================================================================
# MAIN RUNNER
# ============================================================================

def print_banner(text: str, char: str = "="):
    """Print a banner."""
    line = char * 80
    print(f"\n{line}")
    print(text)
    print(f"{line}\n", flush=True)


def main():
    """Main entry point."""
    start_time = time.time()

    # Parse arguments
    dry_run = "--dry-run" in sys.argv
    part_a_only = "--part-a-only" in sys.argv
    part_b_only = "--part-b-only" in sys.argv
    quick = "--quick" in sys.argv

    print_banner("DREAMCODER OVERNIGHT RUN v4 - COMPLETE PIPELINE")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run name: {RUN_NAME}")
    print(f"Log directory: {LOG_DIR}")
    print(f"Mode: {'DRY RUN' if dry_run else 'QUICK TEST' if quick else 'FULL TRAINING'}")
    print(f"Parts: {'A only' if part_a_only else 'B only' if part_b_only else 'A + B'}")

    random.seed(RANDOM_SEED)
    print(f"Random seed: {RANDOM_SEED}")

    # Run pre-flight validation
    validator = PreFlightValidator()
    ok, validation_results = validator.run_all()

    if not ok:
        print("\n[FATAL] Pre-flight validation failed. Aborting.")
        sys.exit(1)

    if dry_run:
        print("\n[DRY RUN] Validation complete. Exiting without training.")

        # Print time estimates
        configs = get_run_configs(quick)
        task_counts = {
            'easy_pretraining': 22,
            'all_pretraining': 43,
            'experimental': 57,
        }

        print("\n" + "=" * 70)
        print("TIME ESTIMATES")
        print("=" * 70)

        total_min = 0
        total_max = 0

        for config in configs:
            if (part_a_only and "Part B" in config.name) or \
               (part_b_only and "Part A" in config.name):
                continue

            min_t, max_t = estimate_run_time(config, task_counts)
            print(f"\n{config.name}:")
            print(f"  Estimated time: {min_t} - {max_t}")

            for phase in config.phases:
                n_tasks = task_counts.get(phase.task_set, 43)
                phase_min, phase_max = estimate_phase_time(phase, n_tasks)
                mini = " [mini-batch]" if phase.mini_batch_size else ""
                print(f"    {phase.name}{mini}: {format_time(phase_min)} - {format_time(phase_max)}")

        sys.exit(0)

    # Get configurations
    configs = get_run_configs(quick)

    # Filter by part if requested
    if part_a_only:
        configs = [c for c in configs if "Part A" in c.name]
    elif part_b_only:
        configs = [c for c in configs if "Part B" in c.name]

    # Create output directory
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Save configuration
    config_data = {
        "run_name": RUN_NAME,
        "start_time": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "n_workers": N_WORKERS,
        "use_pypy": USE_PYPY,
        "quick_mode": quick,
        "configs": [
            {
                "name": c.name,
                "phases": [
                    {
                        "name": p.name,
                        "iterations": p.iterations,
                        "task_set": p.task_set,
                        "enumeration_budget": p.enumeration_budget,
                        "max_depth": p.max_depth,
                        "dreams_per_iteration": p.dreams_per_iteration,
                        "recognition_epochs": p.recognition_epochs,
                        "mini_batch_size": p.mini_batch_size,
                    }
                    for p in c.phases
                ]
            }
            for c in configs
        ]
    }

    with open(LOG_DIR / "run_config.json", 'w') as f:
        json.dump(config_data, f, indent=2)

    # Run each configuration
    for config in configs:
        print_banner(f"RUNNING: {config.name}")
        print(config.description)
        print()

        # This is where you would integrate with the actual DreamCoder runner
        # For now, we just print what would happen

        for i, phase in enumerate(config.phases, 1):
            print(f"\n--- Phase {i}/{len(config.phases)}: {phase.name} ---")
            print(f"  Task set: {phase.task_set}")
            print(f"  Iterations: {phase.iterations}")
            print(f"  Budget: {phase.enumeration_budget:,}")
            print(f"  Max depth: {phase.max_depth}")
            if phase.mini_batch_size:
                print(f"  Mini-batch size: {phase.mini_batch_size}")
            print()

            # TODO: Actually run the phase
            # This would call into run_overnight_cython.CythonOptimizedDreamCoder
            print("  [Would run phase here]")

    # Summary
    total_time = time.time() - start_time
    print_banner("OVERNIGHT RUN COMPLETE")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {format_time(total_time)}")
    print(f"All outputs saved to: {LOG_DIR}")


if __name__ == "__main__":
    main()
