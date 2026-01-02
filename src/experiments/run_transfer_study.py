#!/usr/bin/env python3
"""
Transfer Learning Study: L2 Normalization vs LayerNorm+Scale

This experiment tests whether warm-up on pretraining rules helps learn catalogue rules.

DESIGN:
  Phase 1 (WARM-UP): Train recognition model on 44 pretraining rules (5 iterations)
  Phase 2 (CATALOGUE): Continue learning on 44 catalogue rules (5 iterations)
                       Model keeps learning - NOT frozen

KEY QUESTION: Does pre-training accelerate catalogue rule learning?

RUNTIME ESTIMATE: ~6-10 hours with 500K enumeration budget

To run:
    nohup caffeinate -d -i -s python3 experiments/run_transfer_study.py > transfer_study.out 2>&1 &

Author: Can Konuk
Date: January 2, 2025
"""

import sys
import os
import time
import json
import random
import copy
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import sample_hand, Card
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.type_system import arrow, HAND, BOOL
from dreamcoder_core.task_generation import load_prerecorded_tasks
from dreamcoder_core.grammar import Grammar, Production
from dreamcoder_core.program import Program, Primitive, Application, Abstraction, collect_primitive_names
from dreamcoder_core.enumeration import TopDownEnumerator


def print_flush(*args, **kwargs):
    """Print with immediate flush for real-time output in background runs."""
    print(*args, **kwargs, flush=True)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class TransferStudyConfig:
    """Configuration for the transfer learning study."""

    # ===== TASK CONFIGURATION =====
    n_pretraining_rules: int = 44    # ALL pretraining rules
    n_catalogue_rules: int = 44      # ALL catalogue rules

    # ===== PHASE CONFIGURATION =====
    warmup_iterations: int = 5       # Phase 1: Warm-up on pretraining
    catalogue_iterations: int = 5    # Phase 2: Continue learning on catalogue

    # ===== ENUMERATION BUDGET =====
    enumeration_budget: int = 500_000   # Programs per task per iteration
    max_depth: int = 8                  # Max program depth

    # ===== RECOGNITION MODEL CONFIGURATION =====
    hidden_dim: int = 64
    recognition_epochs: int = 20
    recognition_lr: float = 0.001
    batch_size: int = 8

    # ===== NORMALIZATION-SPECIFIC =====
    layernorm_scale_init: float = 20.0
    l2norm_temperature_init: float = 20.0

    # ===== REPRODUCIBILITY =====
    seed: int = 42

    # ===== OUTPUT =====
    output_dir: str = "results_transfer_study"


# ============================================================================
# TASK WRAPPER
# ============================================================================

class TaskWrapper:
    """Wrapper to give pre-recorded tasks the expected interface."""

    def __init__(self, task):
        self.name = task.name
        self.family = getattr(task, 'family', 'unknown')
        self.level = getattr(task, 'difficulty_level', 1)
        self.examples = task.examples
        self.holdout = task.holdout if hasattr(task, 'holdout') else []
        self.primitives_used: Set[str] = set()
        self.solution: Optional[Program] = None


# ============================================================================
# RECOGNITION MODELS
# ============================================================================

def hand_to_tensors(hand, max_cards: int = 8):
    """Convert hand to tensor representation."""
    suit_map = {'clubs': 0, 'diamonds': 1, 'hearts': 2, 'spades': 3,
                'CLUBS': 0, 'DIAMONDS': 1, 'HEARTS': 2, 'SPADES': 3}
    rank_map = {'A': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6,
                '8': 7, '9': 8, '10': 9, 'J': 10, 'Q': 11, 'K': 12}

    suits = torch.zeros(max_cards, dtype=torch.long)
    ranks = torch.zeros(max_cards, dtype=torch.long)

    for i, card in enumerate(hand[:max_cards]):
        suit_val = card.suit.value if hasattr(card.suit, 'value') else str(card.suit)
        rank_val = card.rank.value if hasattr(card.rank, 'value') else str(card.rank)
        suits[i] = suit_map.get(suit_val, suit_map.get(suit_val.upper(), 0))
        ranks[i] = rank_map.get(rank_val, 0)

    return suits, ranks


class CardEncoder(nn.Module):
    """Card encoder with suit, rank, color, and value embeddings."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.suit_embed = nn.Embedding(4, 8)
        self.rank_embed = nn.Embedding(13, 16)
        self.color_embed = nn.Embedding(2, 4)
        self.rank_value_proj = nn.Linear(1, 4)
        self.output_proj = nn.Linear(32, hidden_dim)

    def forward(self, suits, ranks):
        suit_emb = self.suit_embed(suits)
        rank_emb = self.rank_embed(ranks)
        colors = ((suits == 1) | (suits == 2)).long()
        color_emb = self.color_embed(colors)
        rank_vals = (ranks.float() + 1) / 13.0
        rank_val_emb = self.rank_value_proj(rank_vals.unsqueeze(-1))
        combined = torch.cat([suit_emb, rank_emb, color_emb, rank_val_emb], dim=-1)
        return self.output_proj(combined)


class HandEncoder(nn.Module):
    """Hand encoder with mean pooling."""

    def __init__(self, card_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(card_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, card_embeddings):
        features = self.mlp(card_embeddings)
        return features.mean(dim=1)


class LayerNormRecognitionModel(nn.Module):
    """Recognition model with LayerNorm + learned scale."""

    def __init__(self, num_primitives: int, hidden_dim: int = 64, scale_init: float = 20.0):
        super().__init__()
        self.card_encoder = CardEncoder(hidden_dim)
        self.hand_encoder = HandEncoder(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.scale = nn.Parameter(torch.tensor(scale_init))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_primitives)
        )
        self.num_primitives = num_primitives
        self.hidden_dim = hidden_dim

    def encode_hand(self, hand) -> torch.Tensor:
        suits, ranks = hand_to_tensors(hand)
        suits = suits.unsqueeze(0)
        ranks = ranks.unsqueeze(0)
        card_emb = self.card_encoder(suits, ranks)
        return self.hand_encoder(card_emb).squeeze(0)

    def encode_task(self, task) -> torch.Tensor:
        pos_hands = [h for h, label in task.examples if label]
        neg_hands = [h for h, label in task.examples if not label]

        if not pos_hands or not neg_hands:
            return torch.zeros(self.hidden_dim)

        pos_embs = torch.stack([self.encode_hand(h) for h in pos_hands])
        neg_embs = torch.stack([self.encode_hand(h) for h in neg_hands])

        pos_mean = pos_embs.mean(dim=0)
        neg_mean = neg_embs.mean(dim=0)
        tau = pos_mean - neg_mean
        tau = self.layer_norm(tau) * self.scale
        return tau

    def forward(self, task) -> torch.Tensor:
        tau = self.encode_task(task)
        logits = self.head(tau)
        return torch.sigmoid(logits)

    def predict_log_probs(self, task) -> torch.Tensor:
        probs = self.forward(task)
        return torch.log(probs.clamp(min=1e-10))


class L2NormRecognitionModel(nn.Module):
    """Recognition model with L2 normalization + learned temperature."""

    def __init__(self, num_primitives: int, hidden_dim: int = 64, temperature_init: float = 20.0):
        super().__init__()
        self.card_encoder = CardEncoder(hidden_dim)
        self.hand_encoder = HandEncoder(hidden_dim, hidden_dim)
        self.temperature = nn.Parameter(torch.tensor(temperature_init))
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_primitives)
        )
        self.num_primitives = num_primitives
        self.hidden_dim = hidden_dim

    def encode_hand(self, hand) -> torch.Tensor:
        suits, ranks = hand_to_tensors(hand)
        suits = suits.unsqueeze(0)
        ranks = ranks.unsqueeze(0)
        card_emb = self.card_encoder(suits, ranks)
        return self.hand_encoder(card_emb).squeeze(0)

    def encode_task(self, task) -> torch.Tensor:
        pos_hands = [h for h, label in task.examples if label]
        neg_hands = [h for h, label in task.examples if not label]

        if not pos_hands or not neg_hands:
            return torch.zeros(self.hidden_dim)

        pos_embs = torch.stack([self.encode_hand(h) for h in pos_hands])
        neg_embs = torch.stack([self.encode_hand(h) for h in neg_hands])

        pos_mean = pos_embs.mean(dim=0)
        neg_mean = neg_embs.mean(dim=0)
        tau = pos_mean - neg_mean
        tau = F.normalize(tau, p=2, dim=-1) * self.temperature
        return tau

    def forward(self, task) -> torch.Tensor:
        tau = self.encode_task(task)
        logits = self.head(tau)
        return torch.sigmoid(logits)

    def predict_log_probs(self, task) -> torch.Tensor:
        probs = self.forward(task)
        return torch.log(probs.clamp(min=1e-10))


# ============================================================================
# ENUMERATION AND TRAINING
# ============================================================================

def create_eval_fn(grammar: Grammar):
    """Create an evaluation function for programs on hands."""
    def eval_fn(program: Program, hand) -> Optional[bool]:
        try:
            fn = program.evaluate([])
            result = fn(hand)
            return result
        except:
            return None
    return eval_fn


def enumerate_for_task(
    task: TaskWrapper,
    grammar: Grammar,
    eval_fn,
    budget: int,
    max_depth: int,
    recognition_model: Optional[nn.Module] = None
) -> Tuple[Optional[Program], int]:
    """Enumerate programs for a task with optional recognition guidance."""
    target_type = arrow(HAND, BOOL)

    if recognition_model is not None:
        log_probs = recognition_model.predict_log_probs(task).detach().cpu().numpy()

        new_productions = []
        prim_names = [str(p.program) for p in grammar.productions]
        prim_to_idx = {name: i for i, name in enumerate(prim_names)}

        for prod in grammar.productions:
            prim_name = str(prod.program)
            if prim_name in prim_to_idx:
                idx = prim_to_idx[prim_name]
                new_lp = 0.5 * prod.log_probability + 0.5 * log_probs[idx]
            else:
                new_lp = prod.log_probability
            new_productions.append(Production(prod.program, prod.tp, new_lp))

        biased_grammar = Grammar(new_productions, grammar.log_variable)
    else:
        biased_grammar = grammar

    enumerator = TopDownEnumerator(biased_grammar, max_depth=max_depth, max_programs=budget)

    programs_tried = 0
    solution = None

    for program, log_prob in enumerator.enumerate(target_type):
        if programs_tried >= budget:
            break

        programs_tried += 1

        correct = 0
        for hand, expected in task.examples:
            try:
                result = eval_fn(program, hand)
                if result == expected:
                    correct += 1
            except:
                pass

        if correct == len(task.examples):
            solution = program
            break

    return solution, programs_tried


def train_recognition_model(
    model: nn.Module,
    solved_tasks: List[TaskWrapper],
    grammar: Grammar,
    config: TransferStudyConfig
) -> float:
    """Train recognition model on solved tasks."""
    if not solved_tasks:
        return 0.0

    optimizer = Adam(model.parameters(), lr=config.recognition_lr)
    loss_fn = nn.BCELoss()

    prim_names = [str(p.program) for p in grammar.productions]
    prim_to_idx = {name: i for i, name in enumerate(prim_names)}
    num_prims = len(prim_names)

    total_loss = 0.0
    n_batches = 0

    for epoch in range(config.recognition_epochs):
        random.shuffle(solved_tasks)

        for i in range(0, len(solved_tasks), config.batch_size):
            batch = solved_tasks[i:i+config.batch_size]

            optimizer.zero_grad()
            batch_loss = 0.0

            for task in batch:
                preds = model(task)

                target = torch.zeros(num_prims)
                for pname in task.primitives_used:
                    if pname in prim_to_idx:
                        target[prim_to_idx[pname]] = 1.0

                if target.sum() > 0:
                    loss = loss_fn(preds, target)
                    batch_loss += loss

            if batch_loss > 0:
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()
                n_batches += 1

    return total_loss / max(n_batches, 1)


# ============================================================================
# PHASE RUNNERS
# ============================================================================

@dataclass
class PhaseResult:
    """Results from one phase of the experiment."""
    phase_name: str
    iterations: List[Dict]
    total_solved: int
    total_time: float
    final_loss: float
    solved_task_names: List[str]


def run_phase(
    phase_name: str,
    model_name: str,
    model: nn.Module,
    tasks: List[TaskWrapper],
    grammar: Grammar,
    config: TransferStudyConfig,
    n_iterations: int,
    existing_solved: List[TaskWrapper] = None
) -> Tuple[PhaseResult, List[TaskWrapper]]:
    """
    Run a wake-sleep phase.

    Args:
        existing_solved: Tasks solved in previous phases (for training continuity)
    """
    print_flush(f"\n{'='*70}")
    print_flush(f"{phase_name.upper()} ({model_name})")
    print_flush(f"Learning on {len(tasks)} tasks, {n_iterations} iterations")
    print_flush(f"{'='*70}")

    eval_fn = create_eval_fn(grammar)

    # Track which tasks from THIS phase are solved
    phase_solved: List[TaskWrapper] = []
    phase_solved_names: Set[str] = set()

    # For training, include tasks solved in previous phases
    all_solved_for_training = list(existing_solved) if existing_solved else []

    iteration_results = []
    phase_start = time.time()

    for iteration in range(n_iterations):
        iter_start = time.time()

        print_flush(f"\n--- {phase_name} Iteration {iteration + 1}/{n_iterations} ---")

        # WAKE: Enumerate for unsolved tasks in THIS phase
        new_solved = []
        total_programs = 0

        unsolved = [t for t in tasks if t.name not in phase_solved_names]
        print_flush(f"  Unsolved: {len(unsolved)}, Solved (this phase): {len(phase_solved)}")

        for task in unsolved:
            # Use recognition model if we have ANY solved tasks
            recog = model if all_solved_for_training else None

            solution, programs_tried = enumerate_for_task(
                task, grammar, eval_fn,
                config.enumeration_budget,
                config.max_depth,
                recog
            )

            total_programs += programs_tried

            if solution is not None:
                task.primitives_used = collect_primitive_names(solution)
                task.solution = solution
                new_solved.append(task)
                phase_solved_names.add(task.name)
                print_flush(f"    SOLVED: {task.name} ({programs_tried:,} programs, {len(task.primitives_used)} primitives)")

        phase_solved.extend(new_solved)
        all_solved_for_training.extend(new_solved)

        # SLEEP: Train on ALL solved tasks (from all phases)
        if all_solved_for_training:
            train_loss = train_recognition_model(model, all_solved_for_training, grammar, config)
        else:
            train_loss = 0.0

        iter_time = time.time() - iter_start

        iteration_results.append({
            'iteration': iteration,
            'tasks_solved_total': len(phase_solved),
            'tasks_solved_new': len(new_solved),
            'programs_enumerated': total_programs,
            'wall_time': iter_time,
            'training_loss': train_loss,
            'solved_task_names': [t.name for t in new_solved],
            'total_training_tasks': len(all_solved_for_training)
        })

        print_flush(f"  New: {len(new_solved)}, Phase total: {len(phase_solved)}/{len(tasks)}")
        print_flush(f"  Training on {len(all_solved_for_training)} total tasks, Loss: {train_loss:.6f}")
        print_flush(f"  Time: {iter_time:.1f}s")

        # Early stopping if all solved
        if len(phase_solved) == len(tasks):
            print_flush(f"  All tasks solved! Stopping early.")
            break

    phase_time = time.time() - phase_start

    result = PhaseResult(
        phase_name=phase_name,
        iterations=iteration_results,
        total_solved=len(phase_solved),
        total_time=phase_time,
        final_loss=iteration_results[-1]['training_loss'] if iteration_results else 0.0,
        solved_task_names=[t.name for t in phase_solved]
    )

    return result, all_solved_for_training


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def load_tasks(task_file: str, n_tasks: int) -> List[TaskWrapper]:
    """Load pre-recorded tasks from file."""
    tasks_path = Path(__file__).parent.parent / "data" / "prerecorded_tasks" / task_file

    if not tasks_path.exists():
        raise FileNotFoundError(f"Tasks not found: {tasks_path}")

    all_tasks = load_prerecorded_tasks(tasks_path)
    n = min(n_tasks, len(all_tasks))

    wrapped = [TaskWrapper(t) for t in all_tasks[:n]]

    # Verify balance
    for t in wrapped[:3]:
        pos = sum(1 for _, l in t.examples if l)
        neg = sum(1 for _, l in t.examples if not l)
        print_flush(f"  {t.name}: {pos}+/{neg}- examples")

    return wrapped


def run_full_study():
    """Run the complete transfer learning study."""

    print_flush("=" * 70)
    print_flush("TRANSFER LEARNING STUDY")
    print_flush("Does warm-up on pretraining rules help learn catalogue rules?")
    print_flush("=" * 70)
    print_flush(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_flush()

    config = TransferStudyConfig()

    # Print configuration
    print_flush("CONFIGURATION:")
    print_flush(f"  Pretraining rules: {config.n_pretraining_rules}")
    print_flush(f"  Catalogue rules: {config.n_catalogue_rules}")
    print_flush(f"  Warmup iterations: {config.warmup_iterations}")
    print_flush(f"  Catalogue iterations: {config.catalogue_iterations}")
    print_flush(f"  Enumeration budget: {config.enumeration_budget:,}")
    print_flush(f"  Seed: {config.seed}")
    print_flush()

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Load grammar
    print_flush("Loading grammar...")
    grammar = build_lean_grammar()
    num_primitives = len(grammar.productions)
    print_flush(f"  Primitives: {num_primitives}")

    # Create output directory
    output_dir = Path(__file__).parent.parent / config.output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"study_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print_flush(f"\nOutput directory: {run_dir}")

    # Save config
    config_dict = {k: v for k, v in config.__dict__.items()}
    with open(run_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Define approaches
    approaches = {
        'LayerNorm_Scale': lambda: LayerNormRecognitionModel(
            num_primitives, config.hidden_dim, config.layernorm_scale_init
        ),
        'L2Norm_Temperature': lambda: L2NormRecognitionModel(
            num_primitives, config.hidden_dim, config.l2norm_temperature_init
        )
    }

    all_results = {}

    for model_name, model_fn in approaches.items():
        print_flush(f"\n{'#'*70}")
        print_flush(f"# MODEL: {model_name}")
        print_flush(f"{'#'*70}")

        # Reset seeds for fair comparison
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Create fresh model
        model = model_fn()

        # Load tasks (fresh copies for each model)
        print_flush(f"\nLoading pretraining tasks...")
        pretraining_tasks = load_tasks("pretraining_tasks.json", config.n_pretraining_rules)
        print_flush(f"  Loaded {len(pretraining_tasks)} pretraining tasks")

        print_flush(f"\nLoading catalogue tasks...")
        catalogue_tasks = load_tasks("catalogue_tasks.json", config.n_catalogue_rules)
        print_flush(f"  Loaded {len(catalogue_tasks)} catalogue tasks")

        # PHASE 1: Warm-up on pretraining rules
        warmup_result, warmup_solved = run_phase(
            "Warmup", model_name, model, pretraining_tasks, grammar, config,
            n_iterations=config.warmup_iterations,
            existing_solved=None
        )

        # PHASE 2: Continue learning on catalogue rules
        # Model keeps learning, starts with warmup knowledge
        catalogue_result, final_solved = run_phase(
            "Catalogue", model_name, model, catalogue_tasks, grammar, config,
            n_iterations=config.catalogue_iterations,
            existing_solved=warmup_solved  # Continue training on ALL solved tasks
        )

        # Store results
        all_results[model_name] = {
            'warmup': warmup_result,
            'catalogue': catalogue_result
        }

        # Save model results
        model_results = {
            'warmup': {
                'iterations': warmup_result.iterations,
                'total_solved': warmup_result.total_solved,
                'total_time': warmup_result.total_time,
                'solved_task_names': warmup_result.solved_task_names
            },
            'catalogue': {
                'iterations': catalogue_result.iterations,
                'total_solved': catalogue_result.total_solved,
                'total_time': catalogue_result.total_time,
                'solved_task_names': catalogue_result.solved_task_names
            }
        }

        with open(run_dir / f"{model_name}_results.json", 'w') as f:
            json.dump(model_results, f, indent=2)

        # Save model checkpoint
        torch.save(model.state_dict(), run_dir / f"{model_name}_final.pt")

    # =========================================================================
    # GENERATE COMPARISON SUMMARY
    # =========================================================================

    print_flush("\n" + "=" * 70)
    print_flush("FINAL COMPARISON SUMMARY")
    print_flush("=" * 70)

    summary = {
        'config': config_dict,
        'models': list(approaches.keys()),
        'results': {}
    }

    for model_name in approaches.keys():
        results = all_results[model_name]

        warmup = results['warmup']
        catalogue = results['catalogue']

        print_flush(f"\n{model_name}:")
        print_flush(f"  WARMUP: {warmup.total_solved}/{config.n_pretraining_rules} pretraining rules")
        print_flush(f"          Time: {warmup.total_time:.1f}s")
        print_flush(f"  CATALOGUE: {catalogue.total_solved}/{config.n_catalogue_rules} catalogue rules")
        print_flush(f"             Time: {catalogue.total_time:.1f}s")
        print_flush(f"  TOTAL TIME: {warmup.total_time + catalogue.total_time:.1f}s")

        warmup_curve = [r['tasks_solved_total'] for r in warmup.iterations]
        catalogue_curve = [r['tasks_solved_total'] for r in catalogue.iterations]

        print_flush(f"  Warmup curve: {warmup_curve}")
        print_flush(f"  Catalogue curve: {catalogue_curve}")

        summary['results'][model_name] = {
            'warmup_solved': warmup.total_solved,
            'warmup_time': warmup.total_time,
            'warmup_curve': warmup_curve,
            'catalogue_solved': catalogue.total_solved,
            'catalogue_time': catalogue.total_time,
            'catalogue_curve': catalogue_curve,
            'total_time': warmup.total_time + catalogue.total_time
        }

    # Winner determination
    print_flush("\n" + "-" * 70)
    print_flush("COMPARISON:")
    print_flush("-" * 70)

    models = list(approaches.keys())
    warmup_scores = {m: all_results[m]['warmup'].total_solved for m in models}
    catalogue_scores = {m: all_results[m]['catalogue'].total_solved for m in models}

    warmup_winner = max(warmup_scores, key=warmup_scores.get)
    catalogue_winner = max(catalogue_scores, key=catalogue_scores.get)

    print_flush(f"Warmup winner: {warmup_winner} ({warmup_scores[warmup_winner]}/{config.n_pretraining_rules})")
    print_flush(f"Catalogue winner: {catalogue_winner} ({catalogue_scores[catalogue_winner]}/{config.n_catalogue_rules})")

    # Save summary
    with open(run_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print_flush(f"\n\nResults saved to: {run_dir}")
    print_flush(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_flush("=" * 70)

    return all_results


# ============================================================================
# SANITY CHECKS
# ============================================================================

def run_sanity_checks():
    """Run sanity checks to verify everything is set up correctly."""

    print_flush("=" * 70)
    print_flush("SANITY CHECKS")
    print_flush("=" * 70)

    checks_passed = 0
    checks_failed = 0

    # Check 1: Grammar loads
    print_flush("\n[1] Loading grammar...")
    try:
        grammar = build_lean_grammar()
        num_prims = len(grammar.productions)
        print_flush(f"    ✓ Grammar loaded: {num_prims} primitives")
        checks_passed += 1
    except Exception as e:
        print_flush(f"    ✗ Failed: {e}")
        checks_failed += 1
        return False

    # Check 2: Pretraining tasks exist
    print_flush("\n[2] Loading pretraining tasks...")
    try:
        tasks_path = Path(__file__).parent.parent / "data" / "prerecorded_tasks" / "pretraining_tasks.json"
        tasks = load_prerecorded_tasks(tasks_path)
        print_flush(f"    ✓ Loaded {len(tasks)} pretraining tasks")
        for t in tasks[:2]:
            pos = sum(1 for _, l in t.examples if l)
            neg = sum(1 for _, l in t.examples if not l)
            print_flush(f"      {t.name}: {pos}+/{neg}- (balanced: {pos == neg})")
        checks_passed += 1
    except Exception as e:
        print_flush(f"    ✗ Failed: {e}")
        checks_failed += 1

    # Check 3: Catalogue tasks exist
    print_flush("\n[3] Loading catalogue tasks...")
    try:
        tasks_path = Path(__file__).parent.parent / "data" / "prerecorded_tasks" / "catalogue_tasks.json"
        tasks = load_prerecorded_tasks(tasks_path)
        print_flush(f"    ✓ Loaded {len(tasks)} catalogue tasks")
        checks_passed += 1
    except Exception as e:
        print_flush(f"    ✗ Failed: {e}")
        checks_failed += 1

    # Check 4: Models work
    print_flush("\n[4] Creating recognition models...")
    try:
        layernorm = LayerNormRecognitionModel(num_prims, 64, 20.0)
        l2norm = L2NormRecognitionModel(num_prims, 64, 20.0)
        print_flush(f"    ✓ LayerNorm: {sum(p.numel() for p in layernorm.parameters())} params")
        print_flush(f"    ✓ L2Norm: {sum(p.numel() for p in l2norm.parameters())} params")
        checks_passed += 1
    except Exception as e:
        print_flush(f"    ✗ Failed: {e}")
        checks_failed += 1

    # Check 5: Enumeration works
    print_flush("\n[5] Quick enumeration test...")
    try:
        target_type = arrow(HAND, BOOL)
        enumerator = TopDownEnumerator(grammar, max_depth=8, max_programs=100)
        count = sum(1 for _ in enumerator.enumerate(target_type))
        print_flush(f"    ✓ Enumerated {count} programs")
        checks_passed += 1
    except Exception as e:
        print_flush(f"    ✗ Failed: {e}")
        checks_failed += 1

    # Check 6: Output directory
    print_flush("\n[6] Checking output directory...")
    try:
        output_dir = Path(__file__).parent.parent / "results_transfer_study"
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / "test.tmp"
        test_file.write_text("test")
        test_file.unlink()
        print_flush(f"    ✓ Output directory writable")
        checks_passed += 1
    except Exception as e:
        print_flush(f"    ✗ Failed: {e}")
        checks_failed += 1

    # Summary
    print_flush("\n" + "=" * 70)
    print_flush(f"SANITY CHECK RESULTS: {checks_passed} passed, {checks_failed} failed")
    print_flush("=" * 70)

    if checks_failed == 0:
        print_flush("\n✓ All checks passed! Safe to run.")

        config = TransferStudyConfig()

        # Runtime estimate (simplified)
        # Based on previous experiments: ~10K programs/sec
        # Warmup: 5 iterations × 44 tasks × ~100K avg = 22M programs
        # Catalogue: 5 iterations × 44 tasks × ~300K avg = 66M programs (harder rules)
        # Total per model: ~88M, both models: ~176M
        # At 10K/sec: ~5 hours

        est_hours = 176_000_000 / (10_000 * 3600)

        print_flush(f"\nRuntime estimate: ~{est_hours:.1f} hours")
        print_flush(f"  Phases: Warmup({config.warmup_iterations}) + Catalogue({config.catalogue_iterations})")
        print_flush(f"  Budget per task: {config.enumeration_budget:,}")

        return True
    else:
        print_flush("\n✗ Some checks failed.")
        return False


def run_quick_test():
    """Quick test with minimal settings."""

    print_flush("=" * 70)
    print_flush("QUICK TEST MODE")
    print_flush("=" * 70)

    config = TransferStudyConfig()
    config.n_pretraining_rules = 5
    config.n_catalogue_rules = 5
    config.warmup_iterations = 1
    config.catalogue_iterations = 1
    config.enumeration_budget = 1000
    config.recognition_epochs = 2

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    grammar = build_lean_grammar()

    model = LayerNormRecognitionModel(len(grammar.productions), config.hidden_dim)

    pretraining_tasks = load_tasks("pretraining_tasks.json", config.n_pretraining_rules)
    catalogue_tasks = load_tasks("catalogue_tasks.json", config.n_catalogue_rules)

    warmup_result, warmup_solved = run_phase(
        "Warmup", "QuickTest", model, pretraining_tasks, grammar, config,
        n_iterations=config.warmup_iterations
    )

    catalogue_result, _ = run_phase(
        "Catalogue", "QuickTest", model, catalogue_tasks, grammar, config,
        n_iterations=config.catalogue_iterations,
        existing_solved=warmup_solved
    )

    print_flush("\n✓ Quick test completed!")
    print_flush(f"  Warmup: {warmup_result.total_solved}/5")
    print_flush(f"  Catalogue: {catalogue_result.total_solved}/5")

    return True


if __name__ == "__main__":
    import sys

    if "--check" in sys.argv:
        run_sanity_checks()
    elif "--quick" in sys.argv:
        run_quick_test()
    else:
        run_full_study()
