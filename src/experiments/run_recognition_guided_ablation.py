#!/usr/bin/env python3
"""
Recognition-Guided Compression Ablation Study
==============================================

This script compares three conditions for library learning:
1. Recognition-guided compression with L2Norm + Temperature model
2. Recognition-guided compression with LayerNorm + Scale model
3. Standard backward-only compression with L2Norm + Temperature model

Two-Phase Transfer Learning:
  Phase 1: 5 iterations on pre-training rules (44 rules)
  Phase 2: 5 iterations on catalogue rules (45 rules) with transfer

Program Budget per Iteration:
  - Iteration 1: 250,000 programs
  - Iterations 2,3,4: 500,000 programs
  - Iteration 5: 1,000,000 programs + max_depth=8

Logging:
  - All solutions found with program index when hit
  - All abstractions found with their formulae
  - Recognition model losses and predictions

Usage:
    # Full overnight run
    nohup caffeinate -d -i -s python3 experiments/run_recognition_guided_ablation.py \\
        > recognition_guided_ablation.out 2>&1 &

    # Quick sanity test (reduced budgets)
    python3 experiments/run_recognition_guided_ablation.py --quick-test

    # Dry run (print config without running)
    python3 experiments/run_recognition_guided_ablation.py --dry-run

Author: Can Konuk
Date: January 3, 2026
"""

import sys
import json
import time
import copy
import pickle
import random
import argparse
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.type_system import arrow, BOOL, HAND
from dreamcoder_core.program import Program, Primitive, Application, Abstraction, Index, Invented, collect_primitive_names
from dreamcoder_core.grammar import Grammar, Production
from dreamcoder_core.enumeration import TopDownEnumerator
from dreamcoder_core.compression import (
    compress_frontiers,
    compress_frontiers_recognition,
    CompressionResult
)
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.task_generation import load_prerecorded_tasks
from rules.cards import sample_hand, Card, Hand


def print_flush(*args, **kwargs):
    """Print with immediate flush for real-time output."""
    print(*args, **kwargs, flush=True)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class WorkflowConfig:
    """
    Reusable workflow configuration template.

    This template defines the standard two-phase transfer learning workflow
    with escalating program budgets. Copy and modify for future ablations.
    """
    # Phase structure
    n_phases: int = 2
    iterations_per_phase: int = 5

    # Program budgets per iteration (1-indexed within phase)
    def get_budget(self, iteration: int) -> int:
        """Return program budget for this iteration (1-indexed)."""
        if iteration == 1:
            return 250_000
        elif iteration == 5:
            return 1_000_000
        else:
            return 500_000

    def get_max_depth(self, iteration: int) -> int:
        """Return max program depth for this iteration."""
        return 8 if iteration == 5 else 7

    # Task sources
    phase1_tasks: str = "pretraining_tasks.json"
    phase2_tasks: str = "catalogue_tasks.json"

    # Recognition settings
    recognition_epochs: int = 15
    recognition_lr: float = 1e-3
    recognition_hidden_dim: int = 64

    # Compression settings
    max_inventions_per_iteration: int = 3
    min_compression_savings: float = 2.0
    recognition_alpha: float = 0.7  # For recognition-guided compression

    # Enumeration settings
    enumeration_timeout: float = 180.0

    # Seeds
    seed: int = 42


@dataclass
class VariantConfig:
    """Configuration for a specific ablation variant."""
    name: str
    description: str

    # Key ablation variables
    model_type: str  # 'l2norm' or 'layernorm'
    use_recognition_guided_compression: bool

    # Model parameters
    temperature_init: float = 20.0  # For L2Norm
    scale_init: float = 20.0  # For LayerNorm

    def __post_init__(self):
        self.short_name = self.name.replace(' ', '_').replace('+', '_')


# ============================================================================
# LOGGING INFRASTRUCTURE
# ============================================================================

@dataclass
class SolutionLog:
    """Record of a solution found."""
    task_name: str
    task_family: str
    program: str
    program_size: int
    primitives_used: List[str]
    program_index: int  # Which program number found this solution
    iteration: int
    phase: int
    log_probability: float


@dataclass
class AbstractionLog:
    """Record of an abstraction learned."""
    abstraction_name: str
    abstraction_body: str
    size: int
    iteration: int
    phase: int
    backward_savings: float
    forward_score: Optional[float]  # Only for recognition-guided


@dataclass
class IterationLog:
    """Complete log for one iteration."""
    phase: int
    iteration: int

    # Wake phase
    tasks_solved_cumulative: int
    tasks_solved_new: int
    tasks_total: int
    total_programs_enumerated: int
    programs_budget: int
    max_depth: int
    wake_time_seconds: float

    # Solutions found this iteration
    solutions_found: List[SolutionLog]

    # Compression
    abstractions_found: List[AbstractionLog]
    compression_time_seconds: float
    grammar_size: int

    # Recognition
    recognition_loss_final: float
    recognition_time_seconds: float


@dataclass
class PhaseResult:
    """Result of one phase."""
    phase: int
    iterations: List[IterationLog]
    final_solve_rate: float
    final_solved: int
    total_tasks: int
    abstractions_learned: List[str]


@dataclass
class VariantResult:
    """Complete result for one variant."""
    config: VariantConfig
    workflow: WorkflowConfig
    start_time: str
    end_time: str
    total_duration_seconds: float

    phase_results: List[PhaseResult]

    # Final summary
    phase1_final_solved: int
    phase2_final_solved: int
    total_abstractions: int

    # All solutions and abstractions
    all_solutions: List[SolutionLog]
    all_abstractions: List[AbstractionLog]


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
    """Card embedding encoder."""

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
        tau = pos_embs.mean(dim=0) - neg_embs.mean(dim=0)
        tau = F.normalize(tau, p=2, dim=-1)
        tau = tau * self.temperature
        return tau

    def forward(self, task) -> torch.Tensor:
        tau = self.encode_task(task)
        logits = self.head(tau)
        return torch.sigmoid(logits)

    def predict_log_probs(self, task) -> torch.Tensor:
        probs = self.forward(task)
        return torch.log(probs.clamp(min=1e-10))

    def predict_primitives_dict(self, task) -> Dict[str, float]:
        """Return primitive predictions as a dict (for recognition-guided compression)."""
        probs = self.forward(task).detach().cpu().numpy()
        # This requires knowing primitive names - will be set externally
        return {f"prim_{i}": float(p) for i, p in enumerate(probs)}


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
        tau = pos_embs.mean(dim=0) - neg_embs.mean(dim=0)
        tau = self.layer_norm(tau) * self.scale
        return tau

    def forward(self, task) -> torch.Tensor:
        tau = self.encode_task(task)
        logits = self.head(tau)
        return torch.sigmoid(logits)

    def predict_log_probs(self, task) -> torch.Tensor:
        probs = self.forward(task)
        return torch.log(probs.clamp(min=1e-10))

    def predict_primitives_dict(self, task) -> Dict[str, float]:
        probs = self.forward(task).detach().cpu().numpy()
        return {f"prim_{i}": float(p) for i, p in enumerate(probs)}


# ============================================================================
# TASK WRAPPER
# ============================================================================

class TaskWrapper:
    """Wrapper to give pre-recorded tasks the expected interface."""

    def __init__(self, task):
        self.name = task.name
        self.family = getattr(task, 'family', 'unknown')
        self.examples = task.examples
        self.holdout = task.holdout if hasattr(task, 'holdout') else []
        self.primitives_used: Set[str] = set()

        # Populate primitives_used if available
        if hasattr(task, 'primitives_used'):
            self.primitives_used = set(task.primitives_used)


# ============================================================================
# WAKE-SLEEP ENGINE
# ============================================================================

class RecognitionGuidedLearner:
    """
    Wake-sleep learner with optional recognition-guided compression.
    """

    def __init__(
        self,
        variant_config: VariantConfig,
        workflow_config: WorkflowConfig,
        grammar: Grammar,
        tasks: List[TaskWrapper],
        output_dir: Path,
        primitive_names: List[str]
    ):
        self.variant = variant_config
        self.workflow = workflow_config
        self.grammar = copy.deepcopy(grammar)
        self.tasks = tasks
        self.output_dir = output_dir
        self.primitive_names = primitive_names

        num_primitives = len(grammar.productions)

        # Create recognition model based on variant
        if variant_config.model_type == 'l2norm':
            self.recognition = L2NormRecognitionModel(
                num_primitives=num_primitives,
                hidden_dim=workflow_config.recognition_hidden_dim,
                temperature_init=variant_config.temperature_init
            )
        else:  # layernorm
            self.recognition = LayerNormRecognitionModel(
                num_primitives=num_primitives,
                hidden_dim=workflow_config.recognition_hidden_dim,
                scale_init=variant_config.scale_init
            )

        self.optimizer = Adam(self.recognition.parameters(), lr=workflow_config.recognition_lr)

        # Frontiers (task -> solutions)
        self.frontiers: Dict[str, List[Tuple[Program, float, int]]] = {
            t.name: [] for t in tasks
        }

        # Tracking
        self.cumulative_solved: Set[str] = set()
        self.all_abstractions: List[str] = []
        self.all_solutions: List[SolutionLog] = []
        self.all_abstraction_logs: List[AbstractionLog] = []

    def _eval_program(self, program: Program, hand) -> Optional[bool]:
        """Safely evaluate a program on a hand."""
        try:
            fn = program.evaluate([])
            result = fn(hand)
            return result if isinstance(result, bool) else None
        except Exception:
            return None

    def _check_solution(self, program: Program, examples: List[Tuple[Any, bool]]) -> bool:
        """Check if program solves all examples."""
        for hand, expected in examples:
            result = self._eval_program(program, hand)
            if result != expected:
                return False
        return True

    def run_iteration(self, phase: int, iteration: int) -> IterationLog:
        """Run one wake-sleep iteration."""
        budget = self.workflow.get_budget(iteration)
        max_depth = self.workflow.get_max_depth(iteration)

        print_flush(f"\n  Iteration {iteration}: budget={budget:,}, max_depth={max_depth}")

        # =====================
        # WAKE PHASE
        # =====================
        wake_start = time.time()
        total_programs = 0
        solutions_this_iter: List[SolutionLog] = []

        for task in self.tasks:
            if task.name in self.cumulative_solved:
                continue

            # Get recognition-biased grammar (after first iteration)
            if iteration > 1:
                task_grammar = self._get_biased_grammar(task)
            else:
                task_grammar = self.grammar

            # Enumerate
            task_start = time.time()
            enumerator = TopDownEnumerator(task_grammar, max_depth=max_depth, max_programs=budget)

            programs_tried = 0
            for program, log_prob in enumerator.enumerate_memoized(
                arrow(HAND, BOOL),
                max_cost=50.0,
                timeout_seconds=self.workflow.enumeration_timeout
            ):
                programs_tried += 1
                total_programs += 1

                if programs_tried > budget:
                    break
                if time.time() - task_start > self.workflow.enumeration_timeout:
                    break

                # Check solution
                if self._check_solution(program, task.examples):
                    # Verify on holdout
                    if self._check_solution(program, task.holdout):
                        # Record solution
                        prims = list(collect_primitive_names(program))
                        sol_log = SolutionLog(
                            task_name=task.name,
                            task_family=task.family,
                            program=str(program),
                            program_size=program.size(),
                            primitives_used=prims,
                            program_index=total_programs,
                            iteration=iteration,
                            phase=phase,
                            log_probability=log_prob
                        )
                        solutions_this_iter.append(sol_log)
                        self.all_solutions.append(sol_log)

                        self.frontiers[task.name].append((program, log_prob, total_programs))
                        self.cumulative_solved.add(task.name)
                        print_flush(f"    SOLVED {task.name} at program #{total_programs}: {program}")
                        break

        wake_time = time.time() - wake_start

        print_flush(f"    Wake: {len(self.cumulative_solved)}/{len(self.tasks)} solved, "
                    f"{total_programs:,} programs in {wake_time:.1f}s")

        # =====================
        # COMPRESSION
        # =====================
        comp_start = time.time()
        abstractions_this_iter: List[AbstractionLog] = []

        # Collect frontiers for compression
        frontiers_for_compression = [
            [(p, lp) for p, lp, _ in sols]
            for sols in self.frontiers.values()
            if sols
        ]

        if len(frontiers_for_compression) >= 2:
            # Get unsolved tasks for forward scoring
            unsolved_tasks = [t for t in self.tasks if t.name not in self.cumulative_solved]

            if self.variant.use_recognition_guided_compression and iteration > 1:
                # Use recognition-guided compression
                result = compress_frontiers_recognition(
                    self.grammar,
                    frontiers_for_compression,
                    unsolved_tasks=unsolved_tasks,
                    recognition_model=self._make_recognition_wrapper(),
                    max_inventions=self.workflow.max_inventions_per_iteration,
                    min_savings=self.workflow.min_compression_savings,
                    use_anti_unification=True,
                    alpha=self.workflow.recognition_alpha
                )
            else:
                # Use standard backward-only compression
                result = compress_frontiers(
                    self.grammar,
                    frontiers_for_compression,
                    max_inventions=self.workflow.max_inventions_per_iteration,
                    min_savings=self.workflow.min_compression_savings,
                    use_anti_unification=True
                )

            if result.new_inventions:
                self.grammar = result.new_grammar

                for inv in result.new_inventions:
                    inv_name = str(inv)
                    self.all_abstractions.append(inv_name)

                    abs_log = AbstractionLog(
                        abstraction_name=inv_name,
                        abstraction_body=str(inv.body),
                        size=inv.body.size(),
                        iteration=iteration,
                        phase=phase,
                        backward_savings=0.0,  # Could extract from result
                        forward_score=None
                    )
                    abstractions_this_iter.append(abs_log)
                    self.all_abstraction_logs.append(abs_log)

                    print_flush(f"    Abstraction: {inv_name}")

        comp_time = time.time() - comp_start

        # =====================
        # RECOGNITION TRAINING
        # =====================
        rec_start = time.time()
        final_loss = 0.0

        # Get solved tasks for training
        solved_tasks = [t for t in self.tasks if t.name in self.cumulative_solved]

        if solved_tasks:
            for epoch in range(self.workflow.recognition_epochs):
                epoch_loss = 0.0
                random.shuffle(solved_tasks)

                for task in solved_tasks:
                    self.optimizer.zero_grad()

                    # Get predictions
                    preds = self.recognition(task)

                    # Create target from primitives used in solution
                    target = torch.zeros(self.recognition.num_primitives)
                    prims_in_solution = set()
                    if self.frontiers[task.name]:
                        prog = self.frontiers[task.name][0][0]
                        prims_in_solution = set(collect_primitive_names(prog))

                    for i, pname in enumerate(self.primitive_names):
                        if pname in prims_in_solution:
                            target[i] = 1.0

                    # BCE loss
                    loss = F.binary_cross_entropy(preds, target)
                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item()

                final_loss = epoch_loss / len(solved_tasks)

        rec_time = time.time() - rec_start

        return IterationLog(
            phase=phase,
            iteration=iteration,
            tasks_solved_cumulative=len(self.cumulative_solved),
            tasks_solved_new=len(solutions_this_iter),
            tasks_total=len(self.tasks),
            total_programs_enumerated=total_programs,
            programs_budget=budget,
            max_depth=max_depth,
            wake_time_seconds=wake_time,
            solutions_found=solutions_this_iter,
            abstractions_found=abstractions_this_iter,
            compression_time_seconds=comp_time,
            grammar_size=len(self.grammar.productions),
            recognition_loss_final=final_loss,
            recognition_time_seconds=rec_time
        )

    def _get_biased_grammar(self, task) -> Grammar:
        """Get grammar with recognition-biased weights."""
        log_probs = self.recognition.predict_log_probs(task).detach().cpu().numpy()

        new_productions = []
        prim_to_idx = {name: i for i, name in enumerate(self.primitive_names)}

        for prod in self.grammar.productions:
            prim_name = str(prod.program)
            if prim_name in prim_to_idx:
                idx = prim_to_idx[prim_name]
                new_lp = 0.5 * prod.log_probability + 0.5 * log_probs[idx]
            else:
                new_lp = prod.log_probability
            new_productions.append(Production(prod.program, prod.tp, new_lp))

        return Grammar(new_productions, self.grammar.log_variable)

    def _make_recognition_wrapper(self):
        """Create a wrapper that makes our model compatible with compress_frontiers_recognition."""
        class RecognitionWrapper:
            def __init__(wrapper_self, model, prim_names):
                wrapper_self.model = model
                wrapper_self.prim_names = prim_names

            def predict_primitives_dict(wrapper_self, task):
                probs = wrapper_self.model(task).detach().cpu().numpy()
                return {name: float(probs[i]) for i, name in enumerate(wrapper_self.prim_names)}

        return RecognitionWrapper(self.recognition, self.primitive_names)

    def save_state(self, path: Path) -> None:
        """Save model and grammar state for transfer."""
        state = {
            'model_state_dict': self.recognition.state_dict(),
            'grammar_productions': [(str(p.program), p.log_probability) for p in self.grammar.productions],
            'all_abstractions': self.all_abstractions,
            'cumulative_solved': list(self.cumulative_solved),
        }
        with open(path, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, path: Path) -> None:
        """Load model and grammar state from transfer."""
        with open(path, 'rb') as f:
            state = pickle.load(f)

        # Load model weights (with dimension check)
        try:
            self.recognition.load_state_dict(state['model_state_dict'])
        except RuntimeError:
            print_flush("  Warning: Model dimensions changed, loading compatible weights only")
            current_state = self.recognition.state_dict()
            loaded_state = state['model_state_dict']
            for key in current_state:
                if key in loaded_state and current_state[key].shape == loaded_state[key].shape:
                    current_state[key] = loaded_state[key]
            self.recognition.load_state_dict(current_state)

        self.all_abstractions = state['all_abstractions']


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_variant(
    variant: VariantConfig,
    workflow: WorkflowConfig,
    output_dir: Path,
    quick_test: bool = False
) -> VariantResult:
    """Run a complete two-phase experiment for one variant."""

    start_time = datetime.now()
    print_flush(f"\n{'='*70}")
    print_flush(f"VARIANT: {variant.name}")
    print_flush(f"{'='*70}")
    print_flush(f"Description: {variant.description}")
    print_flush(f"Model: {variant.model_type}")
    print_flush(f"Recognition-guided compression: {variant.use_recognition_guided_compression}")
    print_flush(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Create output directory for this variant
    variant_dir = output_dir / variant.short_name
    variant_dir.mkdir(parents=True, exist_ok=True)

    # Load grammar
    grammar = build_lean_grammar()
    primitive_names = [str(p.program) for p in grammar.productions]

    # Override workflow for quick test
    if quick_test:
        workflow = copy.deepcopy(workflow)
        workflow.iterations_per_phase = 2
        workflow.recognition_epochs = 5
        # Override budgets for quick test
        original_get_budget = workflow.get_budget
        def quick_budget(iteration):
            return min(original_get_budget(iteration), 50_000)
        workflow.get_budget = quick_budget

    phase_results = []
    transfer_state_path = variant_dir / 'phase1_transfer_state.pkl'

    # =====================
    # PHASE 1: Pre-training
    # =====================
    print_flush(f"\n{'='*60}")
    print_flush("PHASE 1: Pre-training Rules")
    print_flush(f"{'='*60}")

    # Load pre-training tasks
    tasks_path = Path(__file__).parent.parent / "data" / "prerecorded_tasks" / workflow.phase1_tasks
    phase1_tasks_raw = load_prerecorded_tasks(tasks_path)
    phase1_tasks = [TaskWrapper(t) for t in phase1_tasks_raw]
    print_flush(f"Loaded {len(phase1_tasks)} pre-training tasks")

    # Create learner for phase 1
    learner = RecognitionGuidedLearner(
        variant_config=variant,
        workflow_config=workflow,
        grammar=grammar,
        tasks=phase1_tasks,
        output_dir=variant_dir / 'phase1',
        primitive_names=primitive_names
    )

    # Run phase 1 iterations
    phase1_iters = []
    for iteration in range(1, workflow.iterations_per_phase + 1):
        iter_log = learner.run_iteration(phase=1, iteration=iteration)
        phase1_iters.append(iter_log)

        # Save iteration log
        with open(variant_dir / f'phase1_iter{iteration:02d}.json', 'w') as f:
            json.dump(asdict(iter_log), f, indent=2, default=str)

    # Save transfer state
    learner.save_state(transfer_state_path)

    phase1_result = PhaseResult(
        phase=1,
        iterations=phase1_iters,
        final_solve_rate=len(learner.cumulative_solved) / len(phase1_tasks),
        final_solved=len(learner.cumulative_solved),
        total_tasks=len(phase1_tasks),
        abstractions_learned=learner.all_abstractions.copy()
    )
    phase_results.append(phase1_result)

    print_flush(f"\nPhase 1 complete: {len(learner.cumulative_solved)}/{len(phase1_tasks)} solved")
    print_flush(f"Abstractions learned: {len(learner.all_abstractions)}")

    # =====================
    # PHASE 2: Catalogue with Transfer
    # =====================
    print_flush(f"\n{'='*60}")
    print_flush("PHASE 2: Catalogue Rules (with transfer)")
    print_flush(f"{'='*60}")

    # Load catalogue tasks
    tasks_path = Path(__file__).parent.parent / "data" / "prerecorded_tasks" / workflow.phase2_tasks
    phase2_tasks_raw = load_prerecorded_tasks(tasks_path)
    phase2_tasks = [TaskWrapper(t) for t in phase2_tasks_raw]
    print_flush(f"Loaded {len(phase2_tasks)} catalogue tasks")

    # Create learner for phase 2 with transferred state
    learner2 = RecognitionGuidedLearner(
        variant_config=variant,
        workflow_config=workflow,
        grammar=learner.grammar,  # Use learned grammar
        tasks=phase2_tasks,
        output_dir=variant_dir / 'phase2',
        primitive_names=primitive_names + learner.all_abstractions  # Include learned abstractions
    )
    learner2.load_state(transfer_state_path)

    # Update primitive names to include new productions
    learner2.primitive_names = [str(p.program) for p in learner2.grammar.productions]

    # Run phase 2 iterations
    phase2_iters = []
    for iteration in range(1, workflow.iterations_per_phase + 1):
        iter_log = learner2.run_iteration(phase=2, iteration=iteration)
        phase2_iters.append(iter_log)

        # Save iteration log
        with open(variant_dir / f'phase2_iter{iteration:02d}.json', 'w') as f:
            json.dump(asdict(iter_log), f, indent=2, default=str)

    phase2_result = PhaseResult(
        phase=2,
        iterations=phase2_iters,
        final_solve_rate=len(learner2.cumulative_solved) / len(phase2_tasks),
        final_solved=len(learner2.cumulative_solved),
        total_tasks=len(phase2_tasks),
        abstractions_learned=learner2.all_abstractions.copy()
    )
    phase_results.append(phase2_result)

    print_flush(f"\nPhase 2 complete: {len(learner2.cumulative_solved)}/{len(phase2_tasks)} solved")
    print_flush(f"Total abstractions: {len(learner2.all_abstractions)}")

    end_time = datetime.now()

    # Compile full result
    result = VariantResult(
        config=variant,
        workflow=workflow,
        start_time=start_time.isoformat(),
        end_time=end_time.isoformat(),
        total_duration_seconds=(end_time - start_time).total_seconds(),
        phase_results=phase_results,
        phase1_final_solved=phase1_result.final_solved,
        phase2_final_solved=phase2_result.final_solved,
        total_abstractions=len(learner2.all_abstractions),
        all_solutions=learner.all_solutions + learner2.all_solutions,
        all_abstractions=learner.all_abstraction_logs + learner2.all_abstraction_logs
    )

    # Save full result
    with open(variant_dir / 'full_result.json', 'w') as f:
        json.dump(asdict(result), f, indent=2, default=str)

    return result


def main():
    parser = argparse.ArgumentParser(description="Recognition-guided compression ablation study")
    parser.add_argument('--quick-test', action='store_true', help='Run quick test with reduced budgets')
    parser.add_argument('--dry-run', action='store_true', help='Print config without running')
    args = parser.parse_args()

    # Set seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Create workflow config
    workflow = WorkflowConfig()

    # Define variants
    variants = [
        VariantConfig(
            name="L2Norm+Temp RecogGuided",
            description="L2 normalization + temperature, recognition-guided compression",
            model_type='l2norm',
            use_recognition_guided_compression=True,
            temperature_init=20.0
        ),
        VariantConfig(
            name="LayerNorm+Scale RecogGuided",
            description="LayerNorm + scale, recognition-guided compression",
            model_type='layernorm',
            use_recognition_guided_compression=True,
            scale_init=20.0
        ),
        VariantConfig(
            name="L2Norm+Temp Backward Only",
            description="L2 normalization + temperature, standard backward-only compression",
            model_type='l2norm',
            use_recognition_guided_compression=False,
            temperature_init=20.0
        ),
    ]

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"results_recognition_guided_ablation/study_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print_flush("=" * 70)
    print_flush("RECOGNITION-GUIDED COMPRESSION ABLATION STUDY")
    print_flush("=" * 70)
    print_flush(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print_flush(f"Output: {output_dir}")
    print_flush(f"Quick test: {args.quick_test}")
    print_flush()

    print_flush("Workflow Configuration:")
    print_flush(f"  Phases: {workflow.n_phases}")
    print_flush(f"  Iterations per phase: {workflow.iterations_per_phase}")
    print_flush(f"  Program budgets: 250k → 500k → 500k → 500k → 1M")
    print_flush(f"  Recognition alpha: {workflow.recognition_alpha}")
    print_flush()

    print_flush("Variants:")
    for i, v in enumerate(variants, 1):
        print_flush(f"  {i}. {v.name}")
        print_flush(f"     {v.description}")
    print_flush()

    if args.dry_run:
        print_flush("DRY RUN - exiting without running experiments")
        return

    # Save experiment config
    config_summary = {
        'timestamp': timestamp,
        'workflow': asdict(workflow) if hasattr(workflow, '__dataclass_fields__') else {
            'n_phases': workflow.n_phases,
            'iterations_per_phase': workflow.iterations_per_phase,
            'recognition_alpha': workflow.recognition_alpha,
            'phase1_tasks': workflow.phase1_tasks,
            'phase2_tasks': workflow.phase2_tasks,
        },
        'variants': [asdict(v) for v in variants],
        'quick_test': args.quick_test
    }
    with open(output_dir / 'experiment_config.json', 'w') as f:
        json.dump(config_summary, f, indent=2)

    # Run all variants
    all_results = {}

    for variant in variants:
        try:
            result = run_variant(variant, workflow, output_dir, quick_test=args.quick_test)
            all_results[variant.name] = result
        except Exception as e:
            print_flush(f"\nERROR in variant {variant.name}: {e}")
            traceback.print_exc()
            continue

    # Create summary
    print_flush("\n" + "=" * 70)
    print_flush("FINAL SUMMARY")
    print_flush("=" * 70)

    summary = {
        'total_duration': str(timedelta(seconds=sum(
            r.total_duration_seconds for r in all_results.values()
        ))),
        'results': {}
    }

    for name, result in all_results.items():
        print_flush(f"\n{name}:")
        print_flush(f"  Phase 1 solved: {result.phase1_final_solved}")
        print_flush(f"  Phase 2 solved: {result.phase2_final_solved}")
        print_flush(f"  Abstractions: {result.total_abstractions}")
        print_flush(f"  Duration: {timedelta(seconds=result.total_duration_seconds)}")

        summary['results'][name] = {
            'phase1_solved': result.phase1_final_solved,
            'phase2_solved': result.phase2_final_solved,
            'total_abstractions': result.total_abstractions,
            'duration_seconds': result.total_duration_seconds
        }

    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print_flush(f"\n\nAll results saved to: {output_dir}")
    print_flush("Done!")


if __name__ == "__main__":
    main()
