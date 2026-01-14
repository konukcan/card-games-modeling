#!/usr/bin/env python3
"""
DreamCoder Overnight Run v3 - Airtight Edition
==============================================

This script runs a comprehensive overnight training session with all recent fixes:

1. Task-Result Scrambling Bug: FIXED
   - Uses results_by_name dict keyed by task_name from result
   - Eliminates as_completed() ordering issues

2. Per-Iteration Checkpoints: ENABLED
   - Saves model weights, task embeddings, primitive predictions after each iteration
   - Enables post-hoc analysis of recognition model evolution

3. Curriculum Learning: IMPLEMENTED
   - Phase 1: Easy rules only (level 1) for foundation building
   - Phase 2: All rules with increased budget
   - Phase 3: Deep search with full library
   - Phase 4: Intensive final push

4. Pre-flight Validation: COMPREHENSIVE
   - Verifies all rules can be evaluated
   - Checks PyPy availability
   - Validates grammar construction
   - Tests worker subprocess communication

Usage:
    python src/run_overnight_v3.py
    python src/run_overnight_v3.py --dry-run  # Validation only, no training
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
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

# ============================================================================
# CONFIGURATION
# ============================================================================

# Output directory
RUN_NAME = f"run_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
LOG_DIR = Path("results/overnight_v3") / RUN_NAME

# Check for PyPy availability
PYPY_PATH = shutil.which('pypy3.10') or shutil.which('pypy3')
USE_PYPY = PYPY_PATH is not None

# Number of parallel workers
N_WORKERS = 4

# Random seed for reproducibility
RANDOM_SEED = 42


@dataclass
class PhaseConfig:
    """Configuration for a training phase."""
    name: str
    iterations: int
    use_all_rules: bool
    enumeration_budget: int
    max_depth: int
    dreams_per_iteration: int
    recognition_epochs: int
    description: str = ""


# ============================================================================
# PRE-FLIGHT VALIDATION
# ============================================================================

class PreFlightValidator:
    """Comprehensive pre-flight validation to catch issues before training."""

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
        """Validate all required imports work."""
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
            from dreamcoder_core.enumeration import enumerate_simple
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
            from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
            self.ok("Contrastive recognition module")
        except ImportError as e:
            self.error(f"Contrastive recognition import failed: {e}")
            return False

        try:
            from rules.pretraining_rules import get_all_pretraining_rules, get_easy_pretraining_rules
            self.ok("Pretraining rules module")
        except ImportError as e:
            self.error(f"Pretraining rules import failed: {e}")
            return False

        return True

    def validate_grammar(self) -> Tuple[bool, Optional[Any]]:
        """Validate grammar construction."""
        print("\n2. Validating grammar...")

        try:
            from dreamcoder_core.lean_primitives import build_lean_grammar
            grammar = build_lean_grammar()
            self.ok(f"Grammar built: {len(grammar)} primitives")

            # Verify key primitives exist
            prim_names = {str(p.program) for p in grammar.productions}
            required = {'n_unique_suits', 'n_unique_colors', 'eq', 'count_suit', 'get_rank', 'lt'}
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

    def validate_rules(self) -> Tuple[bool, List, List]:
        """Validate all rules can be evaluated."""
        print("\n3. Validating rules...")

        try:
            from rules.pretraining_rules import get_all_pretraining_rules, get_easy_pretraining_rules
            from rules.cards import sample_hand

            all_rules = get_all_pretraining_rules()
            easy_rules = get_easy_pretraining_rules()

            self.ok(f"Total rules: {len(all_rules)}")
            self.ok(f"Easy rules (level 1): {len(easy_rules)}")

            # Test each rule
            failed_rules = []
            for rule in all_rules:
                try:
                    # Generate test hands
                    for _ in range(5):
                        hand = sample_hand(6)
                        result = rule.eval(hand)
                        if not isinstance(result, bool):
                            failed_rules.append((rule.name, f"Non-bool result: {type(result)}"))
                            break
                except Exception as e:
                    failed_rules.append((rule.name, str(e)))

            if failed_rules:
                for name, err in failed_rules:
                    self.error(f"Rule '{name}' failed: {err}")
                return False, [], []

            self.ok(f"All {len(all_rules)} rules validated successfully")

            # Group by level
            levels = {}
            for r in all_rules:
                levels.setdefault(r.level, []).append(r)
            for level in sorted(levels.keys()):
                self.log(f"  Level {level}: {len(levels[level])} rules")

            return True, easy_rules, all_rules
        except Exception as e:
            self.error(f"Rule validation failed: {e}")
            traceback.print_exc()
            return False, [], []

    def validate_task_creation(self, easy_rules: List, all_rules: List) -> Tuple[bool, List, List]:
        """Validate task creation from rules."""
        print("\n4. Validating task creation...")

        try:
            from dreamcoder_core.dreamcoder_original import create_tasks_from_rules

            easy_tasks = create_tasks_from_rules(
                easy_rules, n_examples=100, n_holdout=20
            )
            self.ok(f"Created {len(easy_tasks)} easy tasks")

            all_tasks = create_tasks_from_rules(
                all_rules, n_examples=100, n_holdout=20
            )
            self.ok(f"Created {len(all_tasks)} total tasks")

            # Verify task structure
            for task in all_tasks[:3]:
                if len(task.examples) < 10:
                    self.warn(f"Task '{task.name}' has few examples: {len(task.examples)}")
                if not hasattr(task, 'holdout_examples') or len(task.holdout_examples) < 5:
                    self.warn(f"Task '{task.name}' may have insufficient holdout")

            return True, easy_tasks, all_tasks
        except Exception as e:
            self.error(f"Task creation failed: {e}")
            traceback.print_exc()
            return False, [], []

    def validate_recognition_model(self, grammar: Any, tasks: List) -> bool:
        """Validate contrastive recognition model initialization."""
        print("\n5. Validating recognition model...")

        try:
            from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel

            model = ContrastiveRecognitionModel(
                grammar=grammar,
                card_hidden=128,
                learning_rate=5e-4,
                device='cpu',
                output_mode='softmax'
            )
            self.ok(f"Model initialized: ContrastiveRecognitionModel")

            # Test forward pass on a task
            if tasks:
                task = tasks[0]
                emb = model.get_task_embedding(task)
                self.ok(f"Forward pass works: embedding shape {emb.shape}")

                # Test primitive prediction
                probs = model.predict_primitives(task)
                self.ok(f"Primitive prediction works: {probs.shape[0]} primitives")

            return True
        except Exception as e:
            self.error(f"Recognition model validation failed: {e}")
            traceback.print_exc()
            return False

    def validate_pypy_worker(self) -> bool:
        """Validate PyPy worker subprocess communication."""
        print("\n6. Validating PyPy worker...")

        if not USE_PYPY:
            self.warn("PyPy not available - will use sequential enumeration")
            return True

        self.ok(f"PyPy found: {PYPY_PATH}")

        # Test that PyPy can import our modules
        try:
            result = subprocess.run(
                [PYPY_PATH, '-c', '''
import sys
sys.path.insert(0, "src")
from dreamcoder_core.lean_primitives import build_lean_grammar
g = build_lean_grammar()
print(f"Grammar: {len(g)} primitives")
'''],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(Path(__file__).parent.parent)
            )

            if result.returncode == 0:
                self.ok(f"PyPy worker test passed: {result.stdout.strip()}")
                return True
            else:
                self.warn(f"PyPy worker test failed: {result.stderr}")
                self.warn("Will fall back to sequential enumeration")
                return True  # Not fatal
        except subprocess.TimeoutExpired:
            self.warn("PyPy worker test timed out")
            return True
        except Exception as e:
            self.warn(f"PyPy worker test error: {e}")
            return True

    def validate_output_directory(self) -> bool:
        """Validate output directory can be created."""
        print("\n7. Validating output directory...")

        try:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            test_file = LOG_DIR / "test.txt"
            test_file.write_text("test")
            test_file.unlink()
            self.ok(f"Output directory: {LOG_DIR}")
            return True
        except Exception as e:
            self.error(f"Cannot create output directory: {e}")
            return False

    def validate_scrambling_fix(self) -> bool:
        """Verify the task-result scrambling bug fix is in place."""
        print("\n8. Verifying scrambling bug fix...")

        try:
            # Check the source code for the fix
            runner_path = Path(__file__).parent / "run_overnight_cython.py"
            content = runner_path.read_text()

            # Look for the fix pattern
            if "results_by_name" in content and "task_name = result.get('task_name')" in content:
                self.ok("Scrambling bug fix verified in run_overnight_cython.py")
                return True
            else:
                self.error("Scrambling bug fix not found - check run_overnight_cython.py")
                return False
        except Exception as e:
            self.warn(f"Could not verify scrambling fix: {e}")
            return True

    def validate_checkpoint_saving(self) -> bool:
        """Verify per-iteration checkpoint saving is in place."""
        print("\n9. Verifying checkpoint saving...")

        try:
            runner_path = Path(__file__).parent / "run_overnight_cython.py"
            content = runner_path.read_text()

            if "_save_iteration_checkpoint" in content:
                self.ok("Per-iteration checkpoint saving verified")
                return True
            else:
                self.warn("Per-iteration checkpoint saving not found")
                return True  # Not fatal
        except Exception as e:
            self.warn(f"Could not verify checkpoint saving: {e}")
            return True

    def run_all(self) -> Tuple[bool, Dict]:
        """Run all validations and return results."""
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

        # 3. Rules
        ok, easy_rules, all_rules = self.validate_rules()
        if not ok:
            return False, results
        results['easy_rules'] = easy_rules
        results['all_rules'] = all_rules

        # 4. Tasks
        ok, easy_tasks, all_tasks = self.validate_task_creation(easy_rules, all_rules)
        if not ok:
            return False, results
        results['easy_tasks'] = easy_tasks
        results['all_tasks'] = all_tasks

        # 5. Recognition model
        if not self.validate_recognition_model(grammar, all_tasks):
            return False, results

        # 6. PyPy worker
        self.validate_pypy_worker()

        # 7. Output directory
        if not self.validate_output_directory():
            return False, results

        # 8. Scrambling fix
        self.validate_scrambling_fix()

        # 9. Checkpoint saving
        self.validate_checkpoint_saving()

        # Summary
        print("\n" + "=" * 70)
        if self.errors:
            print(f"VALIDATION FAILED: {len(self.errors)} error(s)")
            for err in self.errors:
                print(f"  - {err}")
            return False, results
        elif self.warnings:
            print(f"VALIDATION PASSED with {len(self.warnings)} warning(s)")
            for warn in self.warnings:
                print(f"  - {warn}")
        else:
            print("VALIDATION PASSED - All checks OK")
        print("=" * 70)

        return True, results


# ============================================================================
# PHASE CONFIGURATION
# ============================================================================

def get_phases() -> List[PhaseConfig]:
    """Get the training phase configuration.

    This is an aggressive overnight configuration designed for:
    - Building strong foundations with easy rules first
    - Gradually increasing difficulty
    - Deep search in later phases
    - Total estimated runtime: 6-10 hours
    """
    return [
        PhaseConfig(
            name="Phase 1: Foundation (Easy Rules Only)",
            iterations=6,
            use_all_rules=False,  # Only level 1 rules
            enumeration_budget=150000,
            max_depth=7,
            dreams_per_iteration=80,
            recognition_epochs=12,
            description="Build foundation with 22 easy rules, modest search"
        ),
        PhaseConfig(
            name="Phase 2: Expansion (All Rules, Medium Search)",
            iterations=6,
            use_all_rules=True,  # All 43 rules
            enumeration_budget=250000,
            max_depth=8,
            dreams_per_iteration=120,
            recognition_epochs=15,
            description="Introduce harder rules, increase search budget"
        ),
        PhaseConfig(
            name="Phase 3: Deep Search",
            iterations=6,
            use_all_rules=True,
            enumeration_budget=400000,
            max_depth=9,
            dreams_per_iteration=150,
            recognition_epochs=20,
            description="Deep search with more dreams for abstraction learning"
        ),
        PhaseConfig(
            name="Phase 4: Intensive Push",
            iterations=8,
            use_all_rules=True,
            enumeration_budget=600000,
            max_depth=10,
            dreams_per_iteration=200,
            recognition_epochs=25,
            description="Maximum budget for remaining hard tasks"
        )
    ]


# ============================================================================
# MAIN RUNNER
# ============================================================================

def print_banner(text: str, char: str = "="):
    """Print a banner."""
    line = char * 80
    print(f"\n{line}")
    print(text)
    print(f"{line}\n", flush=True)


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    return str(timedelta(seconds=int(seconds)))


def estimate_runtime(phases: List[PhaseConfig]) -> str:
    """Estimate total runtime based on phase configuration."""
    total_iters = sum(p.iterations for p in phases)

    # Rough estimate: 10-20 minutes per iteration depending on budget
    min_time = total_iters * 10 * 60  # 10 min/iter
    max_time = total_iters * 25 * 60  # 25 min/iter

    return f"{format_time(min_time)} - {format_time(max_time)}"


def main():
    """Main entry point."""
    start_time = time.time()

    # Check for dry-run mode
    dry_run = "--dry-run" in sys.argv

    print_banner("DREAMCODER OVERNIGHT RUN v3 - AIRTIGHT EDITION")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run name: {RUN_NAME}")
    print(f"Log directory: {LOG_DIR}")
    print(f"Mode: {'DRY RUN (validation only)' if dry_run else 'FULL TRAINING'}")

    # Set random seed
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
        sys.exit(0)

    # Extract validated components
    grammar = validation_results['grammar']
    easy_tasks = validation_results['easy_tasks']
    all_tasks = validation_results['all_tasks']

    # Get phase configuration
    phases = get_phases()

    print_banner("PHASE CONFIGURATION")
    total_iters = 0
    for i, phase in enumerate(phases, 1):
        print(f"{i}. {phase.name}")
        print(f"   Iterations: {phase.iterations}")
        print(f"   Rules: {'All 43' if phase.use_all_rules else 'Easy 22'}")
        print(f"   Budget: {phase.enumeration_budget:,} programs")
        print(f"   Max depth: {phase.max_depth}")
        print(f"   Dreams/iter: {phase.dreams_per_iteration}")
        print(f"   Recognition epochs: {phase.recognition_epochs}")
        print(f"   Description: {phase.description}")
        print()
        total_iters += phase.iterations

    print(f"Total iterations: {total_iters}")
    print(f"Estimated runtime: {estimate_runtime(phases)}")

    # Import the DreamCoder runner
    from run_overnight_cython import (
        CythonOptimizedDreamCoder,
        PhaseConfig as RunnerPhaseConfig,
        make_eval_fn
    )

    # Convert our phases to runner phases
    runner_phases = [
        RunnerPhaseConfig(
            name=p.name,
            iterations=p.iterations,
            use_all_rules=p.use_all_rules,
            enumeration_budget=p.enumeration_budget,
            max_depth=p.max_depth,
            dreams_per_iteration=p.dreams_per_iteration,
            recognition_epochs=p.recognition_epochs
        )
        for p in phases
    ]

    # Create eval function
    eval_fn = make_eval_fn()

    # Initialize DreamCoder
    print_banner("INITIALIZING DREAMCODER")

    dc = CythonOptimizedDreamCoder(
        grammar=grammar,
        easy_tasks=easy_tasks,
        all_tasks=all_tasks,
        eval_fn=eval_fn,
        phases=runner_phases,
        recognition_hidden_dim=256,
        recognition_lr=5e-4,
        keep_top_k=5,
        max_inventions_per_iteration=5,
        dream_temperature=1.0,
        n_workers=N_WORKERS,
        use_pypy=USE_PYPY,
        verbose=True,
        log_dir=str(LOG_DIR),
        device='cpu'
    )

    # Save configuration
    config_path = LOG_DIR / "run_config.json"
    config = {
        "run_name": RUN_NAME,
        "start_time": datetime.now().isoformat(),
        "random_seed": RANDOM_SEED,
        "n_workers": N_WORKERS,
        "use_pypy": USE_PYPY,
        "pypy_path": PYPY_PATH,
        "total_tasks": len(all_tasks),
        "easy_tasks": len(easy_tasks),
        "grammar_size": len(grammar),
        "phases": [
            {
                "name": p.name,
                "iterations": p.iterations,
                "use_all_rules": p.use_all_rules,
                "enumeration_budget": p.enumeration_budget,
                "max_depth": p.max_depth,
                "dreams_per_iteration": p.dreams_per_iteration,
                "recognition_epochs": p.recognition_epochs,
                "description": p.description
            }
            for p in phases
        ],
        "total_iterations": total_iters,
        "estimated_runtime": estimate_runtime(phases),
        "fixes_applied": [
            "task_result_scrambling_fix",
            "per_iteration_checkpoints",
            "curriculum_learning"
        ]
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Configuration saved to: {config_path}")

    # Run training
    print_banner("STARTING OVERNIGHT TRAINING")
    print(f"This will take approximately {estimate_runtime(phases)}")
    print("Progress will be logged continuously.\n")

    try:
        results = dc.run()
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Training interrupted by user")
        results = dc._compile_results(time.time() - start_time)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        traceback.print_exc()
        results = {"error": str(e)}

    # Final summary
    total_time = time.time() - start_time

    print_banner("OVERNIGHT TRAINING COMPLETE")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {format_time(total_time)}")
    print()

    if 'summary' in results:
        summary = results['summary']
        print(f"Tasks solved: {summary['tasks_solved']}/{summary['tasks_total']}")
        print(f"Success rate: {100*summary['tasks_solved']/summary['tasks_total']:.1f}%")
        print(f"Final grammar: {summary['final_grammar_size']} primitives")
        print(f"Total abstractions: {summary['total_abstractions']}")
        print(f"Total dreams: {summary['total_dreams']}")

        # Print solved tasks
        if 'task_metrics' in results:
            solved = [(name, tm) for name, tm in results['task_metrics'].items() if tm.get('solved')]
            print(f"\nSolved tasks ({len(solved)}):")
            for name, tm in sorted(solved, key=lambda x: x[1].get('iteration_solved', 0)):
                print(f"  - {name} (iter {tm.get('iteration_solved', 0)+1})")

    # Save final results
    results_path = LOG_DIR / "final_results.json"
    with open(results_path, 'w') as f:
        # Convert to JSON-serializable
        json_results = json.loads(json.dumps(results, default=str))
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Generate report
    print("\nGenerating HTML report...")
    try:
        from generate_overnight_report import parse_log_file, generate_html_report

        log_files = list(LOG_DIR.glob("*.log"))
        if log_files:
            log_path = log_files[0]
            summary, iterations = parse_log_file(log_path)
            report_path = LOG_DIR / "report.html"
            generate_html_report(summary, report_path, log_path.name)
            print(f"Report generated: {report_path}")
    except Exception as e:
        print(f"Could not generate report: {e}")

    print("\n" + "=" * 80)
    print("OVERNIGHT RUN COMPLETE")
    print(f"All outputs saved to: {LOG_DIR}")
    print("=" * 80)


if __name__ == "__main__":
    main()
