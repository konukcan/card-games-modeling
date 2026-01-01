#!/usr/bin/env python3
"""
Comparison Experiment: L2 Normalization vs LayerNorm+Scale in Wake-Sleep Learning

This experiment tests whether L2 normalization's faster learning with limited data
translates to better performance in early wake-sleep iterations.

HYPOTHESIS:
- L2 normalization learns faster with limited data (verified in quick comparison test)
- This should translate to better recognition guidance in early wake-sleep iterations
- LayerNorm+Scale may catch up in later iterations with more solved tasks

Metrics:
1. Rules solved per iteration (learning curve)
2. Enumeration speed (programs/second)
3. Transfer across tasks (does early solving accelerate later solving?)

RUNTIME ESTIMATES:
- Quick test (--quick): ~1 minute
- Full experiment: ~2-3 hours with default config

NOTE: This script uses pure Python enumeration which is slow (~30-50 programs/sec).
For production experiments, use run_overnight_v3.py which uses PyPy workers
(~10-50x faster enumeration).

To run:
    python3 experiments/compare_normalization_wakesleep.py           # Full experiment
    python3 experiments/compare_normalization_wakesleep.py --quick   # Quick validation

Author: Can Konuk
Date: December 31, 2024
"""

import sys
import os
import time
import json
import random
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any, Set
from collections import defaultdict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import sample_hand, Card
from rules.pretraining_rules import get_all_pretraining_rules, get_easy_pretraining_rules
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.type_system import arrow, HAND, BOOL
from dreamcoder_core.task_generation import load_prerecorded_tasks


def print_flush(*args, **kwargs):
    """Print with immediate flush for real-time output in background runs."""
    print(*args, **kwargs, flush=True)
from dreamcoder_core.grammar import Grammar, Production
from dreamcoder_core.program import Program, Primitive, Application, Abstraction
from dreamcoder_core.enumeration import TopDownEnumerator


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for normalization comparison experiment."""
    # Task configuration
    n_rules: int = 35                  # Number of pretraining rules to use
    n_examples_per_task: int = 40      # Examples per task
    hand_size: int = 6

    # Wake-sleep configuration
    n_iterations: int = 5              # Wake-sleep iterations
    enumeration_budget: int = 100000   # Programs per task per iteration
    max_depth: int = 8                 # Max program depth

    # Recognition model configuration
    hidden_dim: int = 64               # Recognition model hidden dimension
    recognition_epochs: int = 15       # Epochs per iteration
    recognition_lr: float = 0.001
    batch_size: int = 8

    # Normalization-specific
    layernorm_scale_init: float = 20.0  # Initial scale for LayerNorm approach

    # Parallelization
    n_workers: int = 4

    # Reproducibility
    seed: int = 42

    # Output
    output_dir: str = "results_normalization_wakesleep"


# ============================================================================
# TASK CREATION (Using pre-recorded balanced tasks)
# ============================================================================

class TaskWrapper:
    """
    Wrapper to give pre-recorded tasks the expected interface.

    Pre-recorded tasks have:
    - Guaranteed 50/50 balance (equal positives and negatives)
    - Near-miss negatives (differ from positives by one card)
    - Disjoint training/holdout pools
    """

    def __init__(self, task):
        self.name = task.name
        self.family = getattr(task, 'family', 'unknown')
        self.level = getattr(task, 'difficulty_level', 1)
        self.examples = task.examples
        self.holdout = task.holdout if hasattr(task, 'holdout') else []
        self.primitives_used: Set[str] = set()


def create_tasks(config: ExperimentConfig) -> List[TaskWrapper]:
    """
    Load pre-recorded tasks instead of generating on-the-fly.

    Uses unified task generation system which guarantees:
    - Balanced examples (50/50 positive/negative)
    - Near-miss negatives (differ by one card from positives)
    - Disjoint training/holdout pools (no data leakage)
    - Explicit failure for rare rules (no spurious 0-positive tasks)

    This fixes the sym_ranks_palindrome bug where 0 positives
    allowed trivial (λ false) solutions.
    """
    tasks_path = Path(__file__).parent.parent / "data" / "prerecorded_tasks" / "pretraining_tasks.json"

    if not tasks_path.exists():
        raise FileNotFoundError(
            f"Pre-recorded tasks not found at {tasks_path}. "
            "Run generate_prerecorded_tasks.py first."
        )

    print_flush(f"Loading pre-recorded tasks from {tasks_path.name}...")
    all_tasks = load_prerecorded_tasks(tasks_path)

    # Take requested number of tasks
    n_rules = min(config.n_rules, len(all_tasks))
    selected_tasks = all_tasks[:n_rules]

    # Wrap tasks to provide expected interface
    wrapped_tasks = [TaskWrapper(t) for t in selected_tasks]

    # Verify and report balance
    for t in wrapped_tasks[:3]:
        pos = sum(1 for _, l in t.examples if l)
        neg = sum(1 for _, l in t.examples if not l)
        print_flush(f"  {t.name}: {pos}+/{neg}- examples")

    print_flush(f"Loaded {len(wrapped_tasks)} tasks with guaranteed balance")

    return wrapped_tasks


# ============================================================================
# RECOGNITION MODEL VARIANTS
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
    """Enhanced card encoder (same for both approaches)."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.suit_embed = nn.Embedding(4, 8)
        self.rank_embed = nn.Embedding(13, 16)
        self.color_embed = nn.Embedding(2, 4)
        self.rank_value_proj = nn.Linear(1, 4)

        # 8 + 16 + 4 + 4 = 32
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
        # card_embeddings: (batch, max_cards, card_dim)
        features = self.mlp(card_embeddings)
        return features.mean(dim=1)  # (batch, hidden_dim)


class LayerNormRecognitionModel(nn.Module):
    """Recognition model with LayerNorm + learned scale (current approach)."""

    def __init__(self, num_primitives: int, hidden_dim: int = 64, scale_init: float = 20.0):
        super().__init__()
        self.card_encoder = CardEncoder(hidden_dim)
        self.hand_encoder = HandEncoder(hidden_dim, hidden_dim)

        # LayerNorm + learned scale
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.scale = nn.Parameter(torch.tensor(scale_init))

        # Prediction head
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

        # Contrastive encoding
        tau = pos_mean - neg_mean

        # LayerNorm + scale
        tau = self.layer_norm(tau) * self.scale

        return tau

    def forward(self, task) -> torch.Tensor:
        tau = self.encode_task(task)
        logits = self.head(tau)
        return torch.sigmoid(logits)

    def predict_log_probs(self, task) -> torch.Tensor:
        """Get log-probabilities for grammar biasing."""
        probs = self.forward(task)
        return torch.log(probs.clamp(min=1e-10))


class L2NormRecognitionModel(nn.Module):
    """Recognition model with L2 normalization + learned temperature."""

    def __init__(self, num_primitives: int, hidden_dim: int = 64, temperature_init: float = 20.0):
        super().__init__()
        self.card_encoder = CardEncoder(hidden_dim)
        self.hand_encoder = HandEncoder(hidden_dim, hidden_dim)

        # Learned temperature/scale parameter (like CLIP)
        # This amplifies the normalized embeddings before prediction
        self.temperature = nn.Parameter(torch.tensor(temperature_init))

        # Prediction head (same as LayerNorm model)
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

        # Contrastive encoding
        tau = pos_mean - neg_mean

        # L2 normalize to unit sphere (direction only)
        tau = F.normalize(tau, p=2, dim=-1)

        # Scale by learned temperature (critical for discrimination!)
        tau = tau * self.temperature

        return tau

    def forward(self, task) -> torch.Tensor:
        tau = self.encode_task(task)
        logits = self.head(tau)
        return torch.sigmoid(logits)

    def predict_log_probs(self, task) -> torch.Tensor:
        """Get log-probabilities for grammar biasing."""
        probs = self.forward(task)
        return torch.log(probs.clamp(min=1e-10))


# ============================================================================
# SIMPLE WAKE-SLEEP ENGINE
# ============================================================================

@dataclass
class IterationResult:
    """Results from one wake-sleep iteration."""
    iteration: int
    tasks_solved_total: int
    tasks_solved_new: int
    programs_enumerated: int
    wall_time: float
    training_loss: float
    solved_task_names: List[str] = field(default_factory=list)


def create_eval_fn(grammar: Grammar):
    """Create an evaluation function for programs on hands."""
    def eval_fn(program: Program, hand) -> Optional[bool]:
        try:
            fn = program.evaluate([])  # Compile program to function
            result = fn(hand)          # Execute on hand
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
    """
    Enumerate programs for a task.

    Returns:
        (solution, programs_tried)
    """
    target_type = arrow(HAND, BOOL)

    # Get grammar weights (biased by recognition model if available)
    if recognition_model is not None:
        log_probs = recognition_model.predict_log_probs(task).detach().cpu().numpy()

        new_productions = []
        prim_names = [str(p.program) for p in grammar.productions]
        prim_to_idx = {name: i for i, name in enumerate(prim_names)}

        for prod in grammar.productions:
            prim_name = str(prod.program)
            if prim_name in prim_to_idx:
                idx = prim_to_idx[prim_name]
                # Blend original and predicted
                new_lp = 0.5 * prod.log_probability + 0.5 * log_probs[idx]
            else:
                new_lp = prod.log_probability
            new_productions.append(Production(prod.program, prod.tp, new_lp))

        biased_grammar = Grammar(new_productions, grammar.log_variable)
    else:
        biased_grammar = grammar

    # Enumerate
    enumerator = TopDownEnumerator(biased_grammar, max_depth=max_depth, max_programs=budget)

    programs_tried = 0
    solution = None

    for program, log_prob in enumerator.enumerate(target_type):
        if programs_tried >= budget:
            break

        programs_tried += 1

        # Evaluate on examples
        correct = 0
        for hand, expected in task.examples:
            try:
                result = eval_fn(program, hand)
                if result == expected:
                    correct += 1
            except:
                pass

        # Check if solution
        if correct == len(task.examples):
            solution = program
            break

    return solution, programs_tried


def train_recognition_model(
    model: nn.Module,
    solved_tasks: List[TaskWrapper],
    grammar: Grammar,
    config: ExperimentConfig
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

                # Build target from primitives used
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


def run_wake_sleep(
    model_name: str,
    model: nn.Module,
    tasks: List[TaskWrapper],
    grammar: Grammar,
    config: ExperimentConfig
) -> List[IterationResult]:
    """Run wake-sleep iterations for one normalization strategy."""

    print_flush(f"\n{'='*60}")
    print_flush(f"Running: {model_name}")
    print_flush(f"{'='*60}")

    eval_fn = create_eval_fn(grammar)
    solved_tasks: List[TaskWrapper] = []
    solved_names: Set[str] = set()
    results: List[IterationResult] = []

    for iteration in range(config.n_iterations):
        iter_start = time.time()

        print_flush(f"\n--- Iteration {iteration + 1}/{config.n_iterations} ---")

        # WAKE PHASE: Enumerate for unsolved tasks
        new_solved = []
        total_programs = 0

        unsolved_tasks = [t for t in tasks if t.name not in solved_names]
        print_flush(f"  Unsolved tasks: {len(unsolved_tasks)}")

        for task in unsolved_tasks:
            # Use recognition model only if we have solved tasks
            recog = model if solved_tasks else None

            solution, programs_tried = enumerate_for_task(
                task, grammar, eval_fn,
                config.enumeration_budget,
                config.max_depth,
                recog
            )

            total_programs += programs_tried

            if solution is not None:
                new_solved.append(task)
                solved_names.add(task.name)
                print_flush(f"    SOLVED: {task.name} after {programs_tried} programs")

        # Update solved tasks
        solved_tasks.extend(new_solved)

        # SLEEP PHASE: Train recognition model
        if solved_tasks:
            train_loss = train_recognition_model(model, solved_tasks, grammar, config)
        else:
            train_loss = 0.0

        iter_time = time.time() - iter_start

        result = IterationResult(
            iteration=iteration,
            tasks_solved_total=len(solved_tasks),
            tasks_solved_new=len(new_solved),
            programs_enumerated=total_programs,
            wall_time=iter_time,
            training_loss=train_loss,
            solved_task_names=[t.name for t in new_solved]
        )
        results.append(result)

        print_flush(f"  New solved: {len(new_solved)}, Total solved: {len(solved_tasks)}")
        print_flush(f"  Programs enumerated: {total_programs:,}")
        print_flush(f"  Training loss: {train_loss:.4f}")
        print_flush(f"  Time: {iter_time:.1f}s")

    return results


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    """Run the full comparison experiment."""

    print_flush("=" * 70)
    print_flush("NORMALIZATION COMPARISON: L2 Norm vs LayerNorm+Scale in Wake-Sleep")
    print_flush("=" * 70)
    print_flush(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_flush()

    config = ExperimentConfig()

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Load grammar
    print_flush("Loading grammar...")
    grammar = build_lean_grammar()
    num_primitives = len(grammar.productions)
    print_flush(f"  Primitives: {num_primitives}")

    # Create tasks
    print_flush("\nCreating tasks...")
    tasks = create_tasks(config)
    print_flush(f"  Tasks created: {len(tasks)}")

    # Output directory
    output_dir = Path(__file__).parent.parent / config.output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"comparison_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print_flush(f"\nOutput directory: {run_dir}")

    # Save config
    config_dict = {k: v for k, v in config.__dict__.items()}
    with open(run_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    # Define approaches
    approaches = {
        'LayerNorm+Scale': lambda: LayerNormRecognitionModel(
            num_primitives, config.hidden_dim, config.layernorm_scale_init
        ),
        'L2Norm+Temperature': lambda: L2NormRecognitionModel(
            num_primitives, config.hidden_dim, temperature_init=20.0
        )
    }

    all_results = {}

    for name, model_fn in approaches.items():
        # Reset seeds for fair comparison
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Recreate tasks with same seed
        tasks = create_tasks(config)

        # Create fresh model
        model = model_fn()

        # Run wake-sleep
        results = run_wake_sleep(name, model, tasks, grammar, config)
        all_results[name] = results

        # Save intermediate results
        results_dict = [
            {
                'iteration': r.iteration,
                'tasks_solved_total': r.tasks_solved_total,
                'tasks_solved_new': r.tasks_solved_new,
                'programs_enumerated': r.programs_enumerated,
                'wall_time': r.wall_time,
                'training_loss': r.training_loss,
                'solved_task_names': r.solved_task_names
            }
            for r in results
        ]
        with open(run_dir / f"{name.replace('+', '_')}_results.json", 'w') as f:
            json.dump(results_dict, f, indent=2)

    # Generate comparison summary
    print_flush("\n" + "=" * 70)
    print_flush("COMPARISON SUMMARY")
    print_flush("=" * 70)

    print_flush(f"\n{'Iteration':<12}", end='')
    for name in approaches.keys():
        print_flush(f"{name:<25}", end='')
    print_flush()

    print_flush("-" * 62)

    for i in range(config.n_iterations):
        print_flush(f"Iter {i+1:<6}", end='')
        for name in approaches.keys():
            solved = all_results[name][i].tasks_solved_total
            new = all_results[name][i].tasks_solved_new
            print_flush(f"{solved:>3} solved (+{new:<2})         ", end='')
        print_flush()

    # Final comparison
    print_flush("\n" + "-" * 62)
    print_flush("FINAL RESULTS:")
    print_flush("-" * 62)

    for name in approaches.keys():
        final = all_results[name][-1]
        total_time = sum(r.wall_time for r in all_results[name])
        total_programs = sum(r.programs_enumerated for r in all_results[name])
        print_flush(f"\n{name}:")
        print_flush(f"  Tasks solved: {final.tasks_solved_total}/{len(tasks)}")
        print_flush(f"  Total time: {total_time:.1f}s")
        print_flush(f"  Total programs: {total_programs:,}")
        print_flush(f"  Avg programs/task: {total_programs/final.tasks_solved_total:,.0f}" if final.tasks_solved_total > 0 else "  Avg programs/task: N/A")

    # Learning curve comparison
    print_flush("\n" + "-" * 62)
    print_flush("LEARNING CURVE (tasks solved by iteration):")
    print_flush("-" * 62)

    for name in approaches.keys():
        curve = [r.tasks_solved_total for r in all_results[name]]
        print_flush(f"{name}: {curve}")

    # Determine winner at each iteration
    print_flush("\n" + "-" * 62)
    print_flush("ITERATION-BY-ITERATION WINNER:")
    print_flush("-" * 62)

    names = list(approaches.keys())
    for i in range(config.n_iterations):
        scores = {name: all_results[name][i].tasks_solved_total for name in names}
        winner = max(scores, key=scores.get)
        diff = scores[names[0]] - scores[names[1]]
        if diff == 0:
            print_flush(f"Iter {i+1}: TIE ({scores[names[0]]} tasks each)")
        else:
            print_flush(f"Iter {i+1}: {winner} wins ({scores[winner]} vs {min(scores.values())})")

    # Save final summary
    summary = {
        'config': config_dict,
        'approaches': list(approaches.keys()),
        'learning_curves': {
            name: [r.tasks_solved_total for r in all_results[name]]
            for name in approaches.keys()
        },
        'final_solved': {
            name: all_results[name][-1].tasks_solved_total
            for name in approaches.keys()
        },
        'total_time': {
            name: sum(r.wall_time for r in all_results[name])
            for name in approaches.keys()
        },
        'total_programs': {
            name: sum(r.programs_enumerated for r in all_results[name])
            for name in approaches.keys()
        }
    }

    with open(run_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print_flush(f"\n\nResults saved to: {run_dir}")
    print_flush("=" * 70)

    return all_results


def run_quick_test():
    """Quick test with minimal configuration to verify everything works."""
    print_flush("=" * 60)
    print_flush("QUICK TEST MODE")
    print_flush("=" * 60)

    config = ExperimentConfig(
        n_rules=3,              # Just 3 rules
        n_iterations=1,         # Just 1 iteration
        enumeration_budget=500, # Very small budget (flow test only)
        recognition_epochs=2    # Few epochs
    )

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    grammar = build_lean_grammar()
    tasks = create_tasks(config)
    print_flush(f"Quick test: {len(tasks)} tasks, {config.n_iterations} iterations")

    # Just test one approach
    model = LayerNormRecognitionModel(len(grammar.productions), config.hidden_dim)
    results = run_wake_sleep("QuickTest", model, tasks, grammar, config)

    print_flush("\nQuick test completed successfully!")
    return results


if __name__ == "__main__":
    import sys

    if "--quick" in sys.argv:
        results = run_quick_test()
    else:
        results = run_experiment()
