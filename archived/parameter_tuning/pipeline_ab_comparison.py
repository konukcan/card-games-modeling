#!/usr/bin/env python3
"""
Pipeline A/B Comparison: Train on Pretraining Rules → Evaluate on Catalogue

This experiment tests whether a recognition model trained on the 44 pretraining
rules can generalize to predict primitives for the 45 catalogue rules.

Two pipelines:
- Pipeline A: Train on pretraining rules → Test on catalogue rules
- Pipeline B: Train on synthetic rules → Test on both rule sets

Comparison conditions:
- LayerNorm+Scale (current approach)
- L2 Normalization (simpler approach)

Metrics: Recall@5, Recall@10, MRR, ProbRatio

RUNTIME ESTIMATE: ~10-15 minutes

Author: Can Konuk
Date: January 2026
"""

import sys
import os
import json
import random
import re
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set, Tuple, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import sample_hand, Card
from rules.catalogue import create_all_rules, Rule
from rules.pretraining_rules import get_all_pretraining_rules, PretrainingRule
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.grammar import Grammar
from dreamcoder_core.program import Program
from dreamcoder_core.enumeration import TopDownEnumerator
from dreamcoder_core.type_system import arrow, HAND, BOOL


def print_flush(*args, **kwargs):
    """Print with immediate flush for real-time output in background runs."""
    print(*args, **kwargs, flush=True)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class ExperimentConfig:
    """Configuration for pipeline comparison experiment."""
    # Task configuration
    n_examples_per_task: int = 50       # Examples per task
    hand_size: int = 6

    # Synthetic rule generation
    n_synthetic_rules: int = 100        # Number of synthetic rules to generate
    synthetic_timeout: float = 60.0     # Timeout for program enumeration

    # Training configuration
    n_epochs: int = 50                  # Training epochs
    batch_size: int = 8
    learning_rate: float = 0.001

    # Model configuration
    hidden_dim: int = 64
    card_hidden: int = 128
    card_out: int = 64

    # Reproducibility
    seed: int = 42

    # Output
    output_dir: str = "results_pipeline_ab"


# ============================================================================
# TASK CREATION
# ============================================================================

@dataclass
class RevelationTask:
    """A task for primitive prediction evaluation."""
    name: str
    examples: List[Tuple[Any, bool]]
    primitives_used: Set[str]
    source: str  # 'catalogue', 'pretraining', 'synthetic'


def create_task_from_catalogue_rule(
    rule: Rule,
    n_examples: int = 50,
    hand_size: int = 6,
    seed: int = 42
) -> Optional[RevelationTask]:
    """Create a task from a catalogue rule."""
    rng = random.Random(seed + hash(rule.id) % 10000)

    examples = []
    pos_count = neg_count = 0
    max_attempts = n_examples * 30

    for _ in range(max_attempts):
        hand = sample_hand(size=hand_size)
        try:
            label = rule.predicate(hand)
            if label and pos_count < n_examples // 2:
                examples.append((hand, True))
                pos_count += 1
            elif not label and neg_count < n_examples // 2:
                examples.append((hand, False))
                neg_count += 1
        except:
            continue

        if pos_count >= n_examples // 2 and neg_count >= n_examples // 2:
            break

    if len(examples) < n_examples // 2:
        return None

    return RevelationTask(
        name=rule.id,
        examples=examples,
        primitives_used=set(rule.primitives_used),
        source='catalogue'
    )


def create_task_from_pretraining_rule(
    rule: PretrainingRule,
    grammar: Grammar,
    n_examples: int = 50,
    hand_size: int = 6,
    seed: int = 42
) -> Optional[RevelationTask]:
    """Create a task from a pretraining rule."""
    rng = random.Random(seed + hash(rule.id) % 10000)

    examples = []
    pos_count = neg_count = 0
    max_attempts = n_examples * 30

    for _ in range(max_attempts):
        hand = sample_hand(size=hand_size)
        try:
            label = rule.eval(hand)
            if label and pos_count < n_examples // 2:
                examples.append((hand, True))
                pos_count += 1
            elif not label and neg_count < n_examples // 2:
                examples.append((hand, False))
                neg_count += 1
        except:
            continue

        if pos_count >= n_examples // 2 and neg_count >= n_examples // 2:
            break

    if len(examples) < n_examples // 2:
        return None

    # Extract primitives from expected_program string
    prim_names = {p.name for p in grammar.primitives()}
    words = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', rule.expected_program)
    primitives = {w for w in words if w in prim_names}

    if not primitives:
        return None

    return RevelationTask(
        name=rule.id,
        examples=examples,
        primitives_used=primitives,
        source='pretraining'
    )


def generate_synthetic_tasks(
    grammar: Grammar,
    n_rules: int = 100,
    n_examples: int = 50,
    hand_size: int = 6,
    timeout: float = 60.0,
    seed: int = 42
) -> List[RevelationTask]:
    """Generate synthetic tasks by enumerating random programs."""
    print_flush(f"  Generating synthetic rules (max {n_rules})...")

    rng = random.Random(seed)
    tasks = []

    # Enumerate programs of type HAND -> BOOL
    request_type = arrow(HAND, BOOL)
    enumerator = TopDownEnumerator(grammar, max_depth=7)

    programs = []
    start_time = time.time()

    # Uses memoized enumeration (now default)
    for prog, cost in enumerator.enumerate(request_type, timeout_seconds=timeout):
        programs.append((prog, cost))
        if len(programs) >= n_rules * 3:
            break

    elapsed = time.time() - start_time
    print_flush(f"    Enumerated {len(programs)} programs in {elapsed:.1f}s")

    # Sample diverse programs
    if len(programs) > n_rules:
        selected = rng.sample(programs, n_rules)
    else:
        selected = programs

    for i, (prog, cost) in enumerate(selected):
        # Extract primitives
        primitives = extract_primitives(prog)

        # Skip trivial programs
        if not primitives or primitives == {'true'} or primitives == {'false'}:
            continue

        # Generate examples
        try:
            fn = prog.evaluate([])
            examples = []
            pos_count = neg_count = 0

            for _ in range(500):
                hand = sample_hand(size=hand_size)
                try:
                    result = fn(hand)
                    if result and pos_count < n_examples // 2:
                        examples.append((hand, True))
                        pos_count += 1
                    elif not result and neg_count < n_examples // 2:
                        examples.append((hand, False))
                        neg_count += 1
                except:
                    continue

                if pos_count >= n_examples // 2 and neg_count >= n_examples // 2:
                    break

            if pos_count >= 10 and neg_count >= 10:
                tasks.append(RevelationTask(
                    name=f"synthetic_{i:03d}",
                    examples=examples,
                    primitives_used=primitives,
                    source='synthetic'
                ))
        except:
            continue

    print_flush(f"    Created {len(tasks)} valid synthetic tasks")
    return tasks


def extract_primitives(program: Program) -> Set[str]:
    """Extract all primitive names from a program."""
    prims = set()

    def traverse(p):
        ptype = type(p).__name__
        if ptype == 'Primitive' and hasattr(p, 'name'):
            prims.add(p.name)
        if hasattr(p, 'f'):
            traverse(p.f)
        if hasattr(p, 'x'):
            traverse(p.x)
        if hasattr(p, 'body'):
            traverse(p.body)

    traverse(program)
    return prims


# ============================================================================
# RECOGNITION MODELS
# ============================================================================

def hand_to_tensors(hand, max_cards: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert hand to tensor representation."""
    ranks = torch.zeros(max_cards, dtype=torch.long)
    suits = torch.zeros(max_cards, dtype=torch.long)

    rank_map = {
        '2': 0, '3': 1, '4': 2, '5': 3, '6': 4, '7': 5, '8': 6,
        '9': 7, '10': 8, 'J': 9, 'Q': 10, 'K': 11, 'A': 12
    }
    suit_map = {'hearts': 0, 'diamonds': 1, 'clubs': 2, 'spades': 3}

    for i, card in enumerate(hand[:max_cards]):
        rank_str = str(card.rank.value) if hasattr(card.rank, 'value') else str(card.rank)
        suit_str = card.suit.value if hasattr(card.suit, 'value') else str(card.suit)

        ranks[i] = rank_map.get(rank_str, 0)
        suits[i] = suit_map.get(suit_str.lower(), 0)

    return ranks, suits


class LayerNormRecognitionModel(nn.Module):
    """Recognition model with LayerNorm + learned scale."""

    def __init__(self, num_primitives: int, hidden_dim: int = 64, scale_init: float = 20.0):
        super().__init__()
        self.num_primitives = num_primitives
        self.hidden_dim = hidden_dim

        # Card embeddings
        self.rank_embed = nn.Embedding(13, 16)
        self.suit_embed = nn.Embedding(4, 8)

        # Card encoder
        self.card_encoder = nn.Sequential(
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )

        # Task encoder (bidirectional GRU)
        self.task_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)

        # Reduce bidirectional output
        self.reduce = nn.Linear(hidden_dim * 2, hidden_dim)

        # LayerNorm + learned scale
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.scale = nn.Parameter(torch.tensor(scale_init))

        # Prediction head
        self.pred_head = nn.Linear(hidden_dim, num_primitives)

    def encode_hand(self, hand) -> torch.Tensor:
        """Encode a single hand."""
        ranks, suits = hand_to_tensors(hand)

        rank_emb = self.rank_embed(ranks)  # [max_cards, 16]
        suit_emb = self.suit_embed(suits)  # [max_cards, 8]
        card_emb = torch.cat([rank_emb, suit_emb], dim=-1)  # [max_cards, 24]

        card_enc = self.card_encoder(card_emb)  # [max_cards, hidden]
        hand_enc = card_enc.mean(dim=0)  # [hidden]

        return hand_enc

    def encode_task(self, examples: List[Tuple[Any, bool]]) -> torch.Tensor:
        """Encode a task from its examples."""
        hand_encodings = []

        for hand, label in examples[:20]:  # Limit for efficiency
            hand_enc = self.encode_hand(hand)
            # Add label information
            label_weight = 1.0 if label else -0.5
            hand_encodings.append(hand_enc * label_weight)

        if not hand_encodings:
            return torch.zeros(self.hidden_dim)

        sequence = torch.stack(hand_encodings).unsqueeze(0)  # [1, n_examples, hidden]
        output, _ = self.task_gru(sequence)  # [1, n_examples, hidden*2]
        pooled = output.mean(dim=1)  # [1, hidden*2]
        reduced = self.reduce(pooled)  # [1, hidden]

        # LayerNorm + scale
        normalized = self.layer_norm(reduced)
        scaled = normalized * self.scale

        return scaled.squeeze(0)

    def forward(self, examples: List[Tuple[Any, bool]]) -> torch.Tensor:
        """Forward pass: examples -> primitive logits."""
        task_emb = self.encode_task(examples)
        logits = self.pred_head(task_emb)
        return logits


class L2NormRecognitionModel(nn.Module):
    """Recognition model with L2 normalization."""

    def __init__(self, num_primitives: int, hidden_dim: int = 64):
        super().__init__()
        self.num_primitives = num_primitives
        self.hidden_dim = hidden_dim

        # Same architecture, different normalization
        self.rank_embed = nn.Embedding(13, 16)
        self.suit_embed = nn.Embedding(4, 8)

        self.card_encoder = nn.Sequential(
            nn.Linear(24, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )

        self.task_gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.reduce = nn.Linear(hidden_dim * 2, hidden_dim)

        # Simple prediction head
        self.pred_head = nn.Linear(hidden_dim, num_primitives)

    def encode_hand(self, hand) -> torch.Tensor:
        ranks, suits = hand_to_tensors(hand)
        rank_emb = self.rank_embed(ranks)
        suit_emb = self.suit_embed(suits)
        card_emb = torch.cat([rank_emb, suit_emb], dim=-1)
        card_enc = self.card_encoder(card_emb)
        return card_enc.mean(dim=0)

    def encode_task(self, examples: List[Tuple[Any, bool]]) -> torch.Tensor:
        hand_encodings = []

        for hand, label in examples[:20]:
            hand_enc = self.encode_hand(hand)
            label_weight = 1.0 if label else -0.5
            hand_encodings.append(hand_enc * label_weight)

        if not hand_encodings:
            return torch.zeros(self.hidden_dim)

        sequence = torch.stack(hand_encodings).unsqueeze(0)
        output, _ = self.task_gru(sequence)
        pooled = output.mean(dim=1)
        reduced = self.reduce(pooled)

        # L2 normalize to unit sphere
        normalized = F.normalize(reduced, p=2, dim=-1)

        return normalized.squeeze(0)

    def forward(self, examples: List[Tuple[Any, bool]]) -> torch.Tensor:
        task_emb = self.encode_task(examples)
        logits = self.pred_head(task_emb)
        return logits


# ============================================================================
# TRAINING
# ============================================================================

def train_model(
    model: nn.Module,
    tasks: List[RevelationTask],
    primitive_names: List[str],
    config: ExperimentConfig
) -> List[float]:
    """Train model on tasks."""
    if not tasks:
        return []

    model.train()
    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    # Create primitive name to index mapping
    prim_to_idx = {name: i for i, name in enumerate(primitive_names)}

    # Create training data with targets
    training_data = []
    for task in tasks:
        if len(task.examples) < 10:
            continue

        target = torch.zeros(len(primitive_names))
        for prim_name in task.primitives_used:
            if prim_name in prim_to_idx:
                target[prim_to_idx[prim_name]] = 1.0

        if target.sum() > 0:
            training_data.append((task, target))

    if not training_data:
        return []

    losses = []
    for epoch in range(config.n_epochs):
        random.shuffle(training_data)
        epoch_losses = []

        for i in range(0, len(training_data), config.batch_size):
            batch = training_data[i:i + config.batch_size]

            batch_preds = []
            batch_targets = []

            for task, target in batch:
                try:
                    logits = model(task.examples)
                    batch_preds.append(logits)
                    batch_targets.append(target)
                except Exception as e:
                    continue

            if not batch_preds:
                continue

            preds = torch.stack(batch_preds)
            targets = torch.stack(batch_targets)

            # BCE with logits loss
            loss = F.binary_cross_entropy_with_logits(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        if epoch_losses:
            avg_loss = np.mean(epoch_losses)
            losses.append(avg_loss)

            if epoch % 10 == 0 or epoch == config.n_epochs - 1:
                print_flush(f"      Epoch {epoch+1}/{config.n_epochs}: loss = {avg_loss:.4f}")

    return losses


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_metrics(
    predictions: np.ndarray,
    solution_prims: Set[str],
    prim_names: List[str]
) -> Dict[str, Any]:
    """Compute prediction quality metrics."""
    sorted_idx = np.argsort(predictions)[::-1]
    sorted_names = [prim_names[i] for i in sorted_idx]

    n_sol = len(solution_prims)
    n_prims = len(prim_names)

    # Find ranks of solution primitives
    ranks = []
    for i, name in enumerate(sorted_names):
        if name in solution_prims:
            ranks.append(i + 1)

    # Recall@k
    def recall_k(k):
        if n_sol == 0:
            return 1.0
        return len(solution_prims & set(sorted_names[:k])) / n_sol

    # MRR (Mean Reciprocal Rank)
    mrr = np.mean([1.0 / r for r in ranks]) if ranks else 0.0

    # ProbRatio
    sol_idx = [prim_names.index(p) for p in solution_prims if p in prim_names]
    other_idx = [i for i in range(n_prims) if prim_names[i] not in solution_prims]

    if sol_idx and other_idx:
        mean_sol_prob = np.mean(predictions[sol_idx])
        mean_other_prob = np.mean(predictions[other_idx])
        prob_ratio = mean_sol_prob / mean_other_prob if mean_other_prob > 0 else 0.0
    else:
        prob_ratio = 1.0

    return {
        'recall@5': recall_k(5),
        'recall@10': recall_k(10),
        'mrr': mrr,
        'prob_ratio': prob_ratio,
        'top5': sorted_names[:5]
    }


def evaluate_model(
    model: nn.Module,
    tasks: List[RevelationTask],
    primitive_names: List[str]
) -> Dict[str, float]:
    """Evaluate model on a set of tasks."""
    model.eval()
    all_metrics = []

    with torch.no_grad():
        for task in tasks:
            if len(task.examples) < 10 or not task.primitives_used:
                continue

            try:
                logits = model(task.examples)
                probs = torch.sigmoid(logits).cpu().numpy()
                metrics = compute_metrics(probs, task.primitives_used, primitive_names)
                all_metrics.append(metrics)
            except:
                continue

    if not all_metrics:
        return {'recall@5': 0, 'recall@10': 0, 'mrr': 0, 'prob_ratio': 1, 'n_tasks': 0}

    return {
        'recall@5': np.mean([m['recall@5'] for m in all_metrics]),
        'recall@10': np.mean([m['recall@10'] for m in all_metrics]),
        'mrr': np.mean([m['mrr'] for m in all_metrics]),
        'prob_ratio': np.mean([m['prob_ratio'] for m in all_metrics]),
        'n_tasks': len(all_metrics)
    }


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print_flush("=" * 80)
    print_flush("PIPELINE A/B COMPARISON: Train → Evaluate Recognition Models")
    print_flush("=" * 80)
    print_flush(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    config = ExperimentConfig()

    # Set seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Load grammar
    print_flush("\nLoading grammar...")
    grammar = build_lean_grammar()
    primitive_names = [p.name for p in grammar.primitives()]
    num_primitives = len(primitive_names)
    print_flush(f"  Primitives: {num_primitives}")

    # Create tasks
    print_flush("\nCreating tasks...")

    catalogue_rules = create_all_rules()
    pretraining_rules = get_all_pretraining_rules()

    catalogue_tasks = []
    for rule in catalogue_rules:
        task = create_task_from_catalogue_rule(rule, config.n_examples_per_task, config.hand_size)
        if task:
            catalogue_tasks.append(task)

    pretraining_tasks = []
    for rule in pretraining_rules:
        task = create_task_from_pretraining_rule(rule, grammar, config.n_examples_per_task, config.hand_size)
        if task:
            pretraining_tasks.append(task)

    print_flush(f"  Catalogue tasks: {len(catalogue_tasks)}")
    print_flush(f"  Pretraining tasks: {len(pretraining_tasks)}")

    # Generate synthetic tasks
    print_flush("\nGenerating synthetic tasks...")
    synthetic_tasks = generate_synthetic_tasks(
        grammar,
        config.n_synthetic_rules,
        config.n_examples_per_task,
        config.hand_size,
        config.synthetic_timeout,
        config.seed
    )

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / config.output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print_flush(f"\nOutput directory: {output_dir}")

    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config.__dict__, f, indent=2)

    # Results storage
    all_results = {}

    # Define model variants
    models = {
        'LayerNorm+Scale': lambda: LayerNormRecognitionModel(num_primitives, config.hidden_dim),
        'L2Norm': lambda: L2NormRecognitionModel(num_primitives, config.hidden_dim)
    }

    # ========================================================================
    # PIPELINE A: Pretraining → Catalogue
    # ========================================================================
    print_flush("\n" + "=" * 80)
    print_flush("PIPELINE A: Train on Pretraining Rules → Test on Catalogue Rules")
    print_flush("=" * 80)

    for model_name, model_fn in models.items():
        print_flush(f"\n--- {model_name} ---")

        # Reset seed for fair comparison
        random.seed(config.seed)
        torch.manual_seed(config.seed)

        model = model_fn()

        # Evaluate untrained
        print_flush("  Evaluating untrained model...")
        untrained_metrics = evaluate_model(model, catalogue_tasks, primitive_names)
        print_flush(f"    Untrained: R@5={untrained_metrics['recall@5']:.3f}, MRR={untrained_metrics['mrr']:.3f}")

        # Train on pretraining rules
        print_flush(f"  Training on {len(pretraining_tasks)} pretraining tasks...")
        losses = train_model(model, pretraining_tasks, primitive_names, config)

        # Evaluate on catalogue rules
        print_flush("  Evaluating trained model on catalogue...")
        trained_metrics = evaluate_model(model, catalogue_tasks, primitive_names)
        print_flush(f"    Trained: R@5={trained_metrics['recall@5']:.3f}, MRR={trained_metrics['mrr']:.3f}, "
                   f"ProbRatio={trained_metrics['prob_ratio']:.2f}")

        all_results[f"pipelineA_{model_name}"] = {
            'pipeline': 'pretraining→catalogue',
            'model': model_name,
            'untrained': untrained_metrics,
            'trained': trained_metrics,
            'final_loss': losses[-1] if losses else None
        }

    # ========================================================================
    # PIPELINE B: Synthetic → Both
    # ========================================================================
    print_flush("\n" + "=" * 80)
    print_flush("PIPELINE B: Train on Synthetic Rules → Test on Both")
    print_flush("=" * 80)

    for model_name, model_fn in models.items():
        print_flush(f"\n--- {model_name} ---")

        # Reset seed
        random.seed(config.seed)
        torch.manual_seed(config.seed)

        model = model_fn()

        # Train on synthetic rules
        print_flush(f"  Training on {len(synthetic_tasks)} synthetic tasks...")
        losses = train_model(model, synthetic_tasks, primitive_names, config)

        # Evaluate on both
        print_flush("  Evaluating on catalogue...")
        catalogue_metrics = evaluate_model(model, catalogue_tasks, primitive_names)

        print_flush("  Evaluating on pretraining...")
        pretraining_metrics = evaluate_model(model, pretraining_tasks, primitive_names)

        print_flush(f"    On catalogue: R@5={catalogue_metrics['recall@5']:.3f}, MRR={catalogue_metrics['mrr']:.3f}")
        print_flush(f"    On pretraining: R@5={pretraining_metrics['recall@5']:.3f}, MRR={pretraining_metrics['mrr']:.3f}")

        all_results[f"pipelineB_{model_name}"] = {
            'pipeline': 'synthetic→both',
            'model': model_name,
            'on_catalogue': catalogue_metrics,
            'on_pretraining': pretraining_metrics,
            'final_loss': losses[-1] if losses else None
        }

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print_flush("\n" + "=" * 80)
    print_flush("SUMMARY")
    print_flush("=" * 80)

    print_flush("\n--- Pipeline A: Pretraining → Catalogue ---")
    print_flush(f"{'Model':<20} {'R@5 (before)':<15} {'R@5 (after)':<15} {'MRR (after)':<15} {'ProbRatio'}")
    print_flush("-" * 80)
    for key, r in all_results.items():
        if 'pipelineA' in key:
            print_flush(f"{r['model']:<20} {r['untrained']['recall@5']:<15.3f} {r['trained']['recall@5']:<15.3f} "
                       f"{r['trained']['mrr']:<15.3f} {r['trained']['prob_ratio']:.2f}")

    print_flush("\n--- Pipeline B: Synthetic → Both ---")
    print_flush(f"{'Model':<20} {'R@5 (catalogue)':<18} {'R@5 (pretrain)':<18} {'MRR (catalogue)':<18}")
    print_flush("-" * 80)
    for key, r in all_results.items():
        if 'pipelineB' in key:
            print_flush(f"{r['model']:<20} {r['on_catalogue']['recall@5']:<18.3f} "
                       f"{r['on_pretraining']['recall@5']:<18.3f} {r['on_catalogue']['mrr']:<18.3f}")

    # Save results
    def to_json(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, set):
            return list(obj)
        return obj

    results_json = json.loads(json.dumps(all_results, default=to_json))
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results_json, f, indent=2)

    print_flush(f"\n✅ Results saved to: {output_dir}")
    print_flush(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
