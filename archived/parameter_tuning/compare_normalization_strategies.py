#!/usr/bin/env python3
"""
Quick Comparison: LayerNorm+Scale vs L2 Normalization for Task Embeddings

This is a parenthetical test during the recognition model code review.
Goal: Determine if L2 normalization is a better approach than LayerNorm+Scale.

Expected runtime: ~5-10 minutes (reduced dataset, fewer epochs)

Author: Can Konuk
Date: December 30, 2024
"""

import sys
import os
import time
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.cards import sample_hand, Card
from rules.catalogue import create_all_rules
from rules.pretraining_rules import get_all_pretraining_rules
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.task_generation import load_prerecorded_tasks


# ============================================================================
# CONFIGURATION - Keep it fast!
# ============================================================================

@dataclass
class QuickTestConfig:
    """Configuration for quick comparison test."""
    n_train_rules: int = 35       # More training rules for stability
    n_test_rules: int = 30        # More test rules for stability
    n_examples: int = 40          # Examples per task
    n_epochs: int = 50            # Training epochs
    batch_size: int = 8
    lr: float = 0.001
    hand_size: int = 6
    hidden_dim: int = 64
    seed: int = 42


# ============================================================================
# TASK CREATION (Using pre-recorded balanced tasks)
# ============================================================================

class TaskWrapper:
    """Wrapper to give pre-recorded tasks the expected interface."""
    def __init__(self, task):
        self.id = task.name
        self.name = task.name
        self.examples = task.examples
        self.holdout = task.holdout if hasattr(task, 'holdout') else []
        self.primitives_used = []
        self.family = getattr(task, 'family', 'unknown')


def create_quick_tasks(rules: Optional[List], config: QuickTestConfig) -> List[TaskWrapper]:
    """
    Load pre-recorded tasks instead of generating on-the-fly.

    Uses unified task generation system with:
    - Guaranteed balanced examples (50/50 positive/negative)
    - Near-miss negatives (differ by one card)
    - Disjoint training/holdout pools
    """
    # Load pre-recorded pretraining tasks
    tasks_path = Path(__file__).parent.parent / "data" / "prerecorded_tasks" / "pretraining_tasks.json"

    if not tasks_path.exists():
        raise FileNotFoundError(
            f"Pre-recorded tasks not found at {tasks_path}. "
            "Run generate_prerecorded_tasks.py first."
        )

    all_tasks = load_prerecorded_tasks(tasks_path)

    # Take requested number of tasks
    n_train = min(config.n_train_rules, len(all_tasks))
    selected_tasks = all_tasks[:n_train]

    # Wrap tasks to provide expected interface
    return [TaskWrapper(t) for t in selected_tasks]


def create_test_tasks(config: QuickTestConfig) -> List[TaskWrapper]:
    """
    Load pre-recorded tasks for testing (using catalogue tasks).
    """
    tasks_path = Path(__file__).parent.parent / "data" / "prerecorded_tasks" / "catalogue_tasks.json"

    if not tasks_path.exists():
        # Fall back to pretraining tasks if catalogue not available
        tasks_path = Path(__file__).parent.parent / "data" / "prerecorded_tasks" / "pretraining_tasks.json"

    all_tasks = load_prerecorded_tasks(tasks_path)

    # Take last n_test_rules for testing (to avoid overlap with training)
    n_test = min(config.n_test_rules, len(all_tasks))
    selected_tasks = all_tasks[-n_test:]

    return [TaskWrapper(t) for t in selected_tasks]


# ============================================================================
# SHARED COMPONENTS
# ============================================================================

def hand_to_tensors(hand, max_cards=8):
    """Convert hand to tensors."""
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


class SharedCardEncoder(nn.Module):
    """Card encoder shared by both approaches."""
    def __init__(self, hidden_dim=64):
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


class SharedHandEncoder(nn.Module):
    """Hand encoder with mean pooling."""
    def __init__(self, card_dim=64, hidden_dim=64):
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


# ============================================================================
# APPROACH 1: LayerNorm + Learned Scale (Current)
# ============================================================================

class LayerNormModel(nn.Module):
    """Current approach: LayerNorm + learned scale."""

    def __init__(self, num_primitives: int, hidden_dim: int = 64, scale_init: float = 20.0):
        super().__init__()
        self.card_encoder = SharedCardEncoder(hidden_dim)
        self.hand_encoder = SharedHandEncoder(hidden_dim, hidden_dim)

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

    def encode_hand(self, hand):
        suits, ranks = hand_to_tensors(hand)
        suits = suits.unsqueeze(0)
        ranks = ranks.unsqueeze(0)
        card_emb = self.card_encoder(suits, ranks)
        return self.hand_encoder(card_emb).squeeze(0)

    def encode_task(self, task):
        pos_hands = [h for h, label in task.examples if label]
        neg_hands = [h for h, label in task.examples if not label]

        # Handle edge cases
        if not pos_hands or not neg_hands:
            return torch.zeros(self.head[0].in_features)

        pos_embs = torch.stack([self.encode_hand(h) for h in pos_hands])
        neg_embs = torch.stack([self.encode_hand(h) for h in neg_hands])

        pos_mean = pos_embs.mean(dim=0)
        neg_mean = neg_embs.mean(dim=0)

        # Contrastive encoding
        tau = pos_mean - neg_mean

        # LayerNorm + scale
        tau = self.layer_norm(tau) * self.scale

        return tau

    def forward(self, task):
        tau = self.encode_task(task)
        logits = self.head(tau)
        return torch.sigmoid(logits)


# ============================================================================
# APPROACH 2: L2 Normalization + Temperature (Proposed)
# ============================================================================

class L2NormModel(nn.Module):
    """Proposed approach: L2 normalization + temperature scaling."""

    def __init__(self, num_primitives: int, hidden_dim: int = 64, temperature: float = 0.1):
        super().__init__()
        self.card_encoder = SharedCardEncoder(hidden_dim)
        self.hand_encoder = SharedHandEncoder(hidden_dim, hidden_dim)

        # L2 normalization (no learned parameters needed for the norm itself)
        self.temperature = temperature

        # Learned primitive embeddings (like CLIP/SimCLR style)
        self.primitive_embeddings = nn.Parameter(torch.randn(num_primitives, hidden_dim))
        nn.init.xavier_uniform_(self.primitive_embeddings)

        self.num_primitives = num_primitives
        self.hidden_dim = hidden_dim

    def encode_hand(self, hand):
        suits, ranks = hand_to_tensors(hand)
        suits = suits.unsqueeze(0)
        ranks = ranks.unsqueeze(0)
        card_emb = self.card_encoder(suits, ranks)
        return self.hand_encoder(card_emb).squeeze(0)

    def encode_task(self, task):
        pos_hands = [h for h, label in task.examples if label]
        neg_hands = [h for h, label in task.examples if not label]

        # Handle edge cases
        if not pos_hands or not neg_hands:
            return torch.zeros(self.hidden_dim)

        pos_embs = torch.stack([self.encode_hand(h) for h in pos_hands])
        neg_embs = torch.stack([self.encode_hand(h) for h in neg_hands])

        pos_mean = pos_embs.mean(dim=0)
        neg_mean = neg_embs.mean(dim=0)

        # Contrastive encoding
        tau = pos_mean - neg_mean

        # L2 normalize to unit sphere
        tau = F.normalize(tau, p=2, dim=-1)

        return tau

    def forward(self, task):
        tau = self.encode_task(task)

        # Normalize primitive embeddings
        prim_normed = F.normalize(self.primitive_embeddings, p=2, dim=-1)

        # Cosine similarity with temperature
        scores = torch.matmul(tau, prim_normed.t()) / self.temperature

        return torch.sigmoid(scores)


# ============================================================================
# APPROACH 3: Hybrid - L2 Norm on hands, then standard head
# ============================================================================

class L2NormSimpleModel(nn.Module):
    """Hybrid: L2 normalize task embedding, but use standard MLP head."""

    def __init__(self, num_primitives: int, hidden_dim: int = 64):
        super().__init__()
        self.card_encoder = SharedCardEncoder(hidden_dim)
        self.hand_encoder = SharedHandEncoder(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim

        # Prediction head (same as LayerNorm model)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_primitives)
        )

        self.num_primitives = num_primitives

    def encode_hand(self, hand):
        suits, ranks = hand_to_tensors(hand)
        suits = suits.unsqueeze(0)
        ranks = ranks.unsqueeze(0)
        card_emb = self.card_encoder(suits, ranks)
        return self.hand_encoder(card_emb).squeeze(0)

    def encode_task(self, task):
        pos_hands = [h for h, label in task.examples if label]
        neg_hands = [h for h, label in task.examples if not label]

        # Handle edge cases
        if not pos_hands or not neg_hands:
            return torch.zeros(self.hidden_dim)

        pos_embs = torch.stack([self.encode_hand(h) for h in pos_hands])
        neg_embs = torch.stack([self.encode_hand(h) for h in neg_hands])

        pos_mean = pos_embs.mean(dim=0)
        neg_mean = neg_embs.mean(dim=0)

        # Contrastive encoding
        tau = pos_mean - neg_mean

        # L2 normalize (direction only)
        tau = F.normalize(tau, p=2, dim=-1)

        return tau

    def forward(self, task):
        tau = self.encode_task(task)
        logits = self.head(tau)
        return torch.sigmoid(logits)


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def train_model(model, train_tasks, grammar, config: QuickTestConfig):
    """Train a model quickly."""
    optimizer = Adam(model.parameters(), lr=config.lr)
    loss_fn = nn.BCELoss()

    primitives = grammar.primitives()
    prim_to_idx = {p.name: i for i, p in enumerate(primitives)}
    num_prims = len(primitives)

    losses = []

    for epoch in range(config.n_epochs):
        model.train()
        epoch_loss = 0.0

        for task in train_tasks:
            optimizer.zero_grad()

            # Get predictions
            preds = model(task)

            # Build target
            target = torch.zeros(num_prims)
            for pname in task.primitives_used:
                if pname in prim_to_idx:
                    target[prim_to_idx[pname]] = 1.0

            if target.sum() == 0:
                continue

            loss = loss_fn(preds, target)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        losses.append(epoch_loss / max(len(train_tasks), 1))

    return losses


def evaluate_model(model, test_tasks, grammar) -> Dict:
    """Evaluate model on test tasks."""
    model.eval()

    primitives = grammar.primitives()
    prim_names = [p.name for p in primitives]
    prim_to_idx = {p.name: i for i, p in enumerate(primitives)}

    recalls_at_5 = []
    recalls_at_10 = []
    mrrs = []
    pred_stds = []

    with torch.no_grad():
        for task in test_tasks:
            preds = model(task)
            pred_stds.append(preds.std().item())

            # Get ground truth
            gt_indices = set()
            for pname in task.primitives_used:
                if pname in prim_to_idx:
                    gt_indices.add(prim_to_idx[pname])

            if not gt_indices:
                continue

            # Sort predictions
            sorted_indices = torch.argsort(preds, descending=True).tolist()

            # Recall@5
            top5 = set(sorted_indices[:5])
            r5 = len(top5 & gt_indices) / len(gt_indices)
            recalls_at_5.append(r5)

            # Recall@10
            top10 = set(sorted_indices[:10])
            r10 = len(top10 & gt_indices) / len(gt_indices)
            recalls_at_10.append(r10)

            # MRR
            for rank, idx in enumerate(sorted_indices, 1):
                if idx in gt_indices:
                    mrrs.append(1.0 / rank)
                    break
            else:
                mrrs.append(0.0)

    return {
        'R@5': np.mean(recalls_at_5) if recalls_at_5 else 0.0,
        'R@10': np.mean(recalls_at_10) if recalls_at_10 else 0.0,
        'MRR': np.mean(mrrs) if mrrs else 0.0,
        'pred_std': np.mean(pred_stds) if pred_stds else 0.0,
        'n_evaluated': len(recalls_at_5)
    }


# ============================================================================
# MAIN COMPARISON
# ============================================================================

def run_comparison():
    """Run quick comparison of normalization strategies."""
    print("=" * 70)
    print("QUICK COMPARISON: LayerNorm+Scale vs L2 Normalization")
    print("=" * 70)
    print()

    config = QuickTestConfig()
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Create grammar
    print("Loading grammar...")
    grammar = build_lean_grammar()
    num_primitives = len(grammar.primitives())
    print(f"  Primitives: {num_primitives}")

    # Create tasks from pre-recorded balanced datasets
    print("\nLoading pre-recorded tasks (balanced, near-miss negatives)...")
    train_tasks = create_quick_tasks(None, config)  # rules param ignored
    test_tasks = create_test_tasks(config)

    print(f"  Train tasks: {len(train_tasks)} (from pretraining_tasks.json)")
    print(f"  Test tasks: {len(test_tasks)} (from catalogue_tasks.json)")

    # Verify balance
    for t in train_tasks[:2]:
        pos = sum(1 for _, l in t.examples if l)
        neg = sum(1 for _, l in t.examples if not l)
        print(f"    Sample: {t.name} = {pos}+/{neg}-")

    # Define approaches to test
    approaches = {
        'LayerNorm+Scale': lambda: LayerNormModel(num_primitives, config.hidden_dim),
        'L2Norm+Temperature': lambda: L2NormModel(num_primitives, config.hidden_dim, temperature=0.1),
        'L2Norm+SimpleHead': lambda: L2NormSimpleModel(num_primitives, config.hidden_dim),
    }

    results = {}

    for name, model_fn in approaches.items():
        print(f"\n{'='*50}")
        print(f"Testing: {name}")
        print(f"{'='*50}")

        # Reset seeds for fair comparison
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        model = model_fn()

        # Train
        start_time = time.time()
        losses = train_model(model, train_tasks, grammar, config)
        train_time = time.time() - start_time

        print(f"  Training time: {train_time:.1f}s")
        print(f"  Final loss: {losses[-1]:.4f}")

        # Evaluate
        metrics = evaluate_model(model, test_tasks, grammar)
        metrics['train_time'] = train_time
        metrics['final_loss'] = losses[-1]
        metrics['losses'] = losses

        results[name] = metrics

        print(f"  R@5: {metrics['R@5']:.3f}")
        print(f"  R@10: {metrics['R@10']:.3f}")
        print(f"  MRR: {metrics['MRR']:.3f}")
        print(f"  Pred std: {metrics['pred_std']:.4f}")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print()
    print(f"{'Approach':<25} {'R@5':>8} {'R@10':>8} {'MRR':>8} {'Pred Std':>10}")
    print("-" * 60)

    for name, metrics in results.items():
        print(f"{name:<25} {metrics['R@5']:>8.3f} {metrics['R@10']:>8.3f} "
              f"{metrics['MRR']:>8.3f} {metrics['pred_std']:>10.4f}")

    # Determine winner
    print("\n" + "-" * 60)
    best_approach = max(results.keys(), key=lambda k: results[k]['R@5'])
    print(f"Best R@5: {best_approach} ({results[best_approach]['R@5']:.3f})")

    baseline_r5 = results['LayerNorm+Scale']['R@5']
    for name, metrics in results.items():
        if name != 'LayerNorm+Scale':
            if baseline_r5 > 0:
                diff = (metrics['R@5'] - baseline_r5) / baseline_r5 * 100
                print(f"{name} vs LayerNorm+Scale: {diff:+.1f}%")
            else:
                diff = metrics['R@5'] - baseline_r5
                print(f"{name} vs LayerNorm+Scale: {diff:+.3f} (absolute)")

    # Save results
    output_dir = Path(__file__).parent.parent / "results_normalization_comparison"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"comparison_{timestamp}.json"

    # Convert losses to list for JSON serialization
    save_results = {}
    for name, metrics in results.items():
        save_results[name] = {k: v if k != 'losses' else v for k, v in metrics.items()}

    with open(output_file, 'w') as f:
        json.dump(save_results, f, indent=2)

    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    results = run_comparison()
