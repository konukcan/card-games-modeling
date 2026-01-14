#!/usr/bin/env python3
"""
Train recognition models specifically for interpretability analysis.

Trains both LayerNorm+Scale and L2Norm+Temperature models on pretraining rules,
saves the trained models, and runs interpretability analysis on them.
"""

import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.dreamcoder_original import create_tasks_from_rules
from rules.pretraining_rules import get_all_pretraining_rules
from rules.cards import Suit, Rank, RANK_VALUES


def print_flush(msg: str):
    print(msg, flush=True)


# =============================================================================
# MODEL DEFINITIONS (matching the wake-sleep experiment)
# =============================================================================

class CardEncoder(nn.Module):
    """Factored card encoder: separate embeddings for suit, rank, position."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.d_suit = 8
        self.d_rank = 16
        self.d_pos = 8

        self.suit_embed = nn.Embedding(4, self.d_suit)
        self.rank_embed = nn.Embedding(13, self.d_rank)
        self.pos_embed = nn.Embedding(8, self.d_pos)

    def forward(self, suits, ranks, positions):
        suit_emb = self.suit_embed(suits)
        rank_emb = self.rank_embed(ranks)
        pos_emb = self.pos_embed(positions)
        return torch.cat([suit_emb, rank_emb, pos_emb], dim=-1)


class LayerNormRecognitionModel(nn.Module):
    """Recognition model with LayerNorm + learned scale."""

    def __init__(self, num_primitives: int, hidden_dim: int = 64, scale_init: float = 20.0):
        super().__init__()
        self.card_encoder = CardEncoder(hidden_dim)
        d_card = self.card_encoder.d_suit + self.card_encoder.d_rank + self.card_encoder.d_pos

        self.card_mlp = nn.Sequential(
            nn.Linear(d_card, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.embedding_norm = nn.LayerNorm(hidden_dim)
        self.embedding_scale = nn.Parameter(torch.tensor(scale_init))

        self.primitive_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_primitives)
        )

        self.num_primitives = num_primitives
        self.hidden_dim = hidden_dim

    def encode_hand(self, hand) -> torch.Tensor:
        """Encode a single hand."""
        if not hand:
            return torch.zeros(self.hidden_dim)

        suits, ranks, positions = [], [], []
        for i, card in enumerate(hand):
            suit_map = {Suit.SPADES: 0, Suit.HEARTS: 1, Suit.DIAMONDS: 2, Suit.CLUBS: 3}
            suits.append(suit_map.get(card.suit, 0))
            ranks.append(RANK_VALUES[card.rank] - 2)
            positions.append(min(i, 7))

        suits = torch.tensor(suits)
        ranks = torch.tensor(ranks)
        positions = torch.tensor(positions)

        card_emb = self.card_encoder(suits, ranks, positions)
        card_features = self.card_mlp(card_emb)
        return card_features.mean(dim=0)

    def encode_task(self, task) -> torch.Tensor:
        """Encode task using contrastive encoding."""
        pos_reprs, neg_reprs = [], []

        for hand, label in task.examples:
            repr = self.encode_hand(hand)
            if label:
                pos_reprs.append(repr)
            else:
                neg_reprs.append(repr)

        pos_mean = torch.stack(pos_reprs).mean(dim=0) if pos_reprs else torch.zeros(self.hidden_dim)
        neg_mean = torch.stack(neg_reprs).mean(dim=0) if neg_reprs else torch.zeros(self.hidden_dim)

        tau = pos_mean - neg_mean
        tau = self.embedding_norm(tau) * self.embedding_scale
        return tau

    def forward(self, task) -> torch.Tensor:
        """Get primitive predictions for a task."""
        tau = self.encode_task(task)
        logits = self.primitive_head(tau)
        return torch.sigmoid(logits)


class L2NormRecognitionModel(nn.Module):
    """Recognition model with L2 normalization + learned temperature."""

    def __init__(self, num_primitives: int, hidden_dim: int = 64, temperature_init: float = 20.0):
        super().__init__()
        self.card_encoder = CardEncoder(hidden_dim)
        d_card = self.card_encoder.d_suit + self.card_encoder.d_rank + self.card_encoder.d_pos

        self.card_mlp = nn.Sequential(
            nn.Linear(d_card, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.temperature = nn.Parameter(torch.tensor(temperature_init))

        self.primitive_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_primitives)
        )

        self.num_primitives = num_primitives
        self.hidden_dim = hidden_dim

    def encode_hand(self, hand) -> torch.Tensor:
        """Encode a single hand."""
        if not hand:
            return torch.zeros(self.hidden_dim)

        suits, ranks, positions = [], [], []
        for i, card in enumerate(hand):
            suit_map = {Suit.SPADES: 0, Suit.HEARTS: 1, Suit.DIAMONDS: 2, Suit.CLUBS: 3}
            suits.append(suit_map.get(card.suit, 0))
            ranks.append(RANK_VALUES[card.rank] - 2)
            positions.append(min(i, 7))

        suits = torch.tensor(suits)
        ranks = torch.tensor(ranks)
        positions = torch.tensor(positions)

        card_emb = self.card_encoder(suits, ranks, positions)
        card_features = self.card_mlp(card_emb)
        return card_features.mean(dim=0)

    def encode_task(self, task) -> torch.Tensor:
        """Encode task using contrastive encoding with L2 norm + temperature."""
        pos_reprs, neg_reprs = [], []

        for hand, label in task.examples:
            repr = self.encode_hand(hand)
            if label:
                pos_reprs.append(repr)
            else:
                neg_reprs.append(repr)

        pos_mean = torch.stack(pos_reprs).mean(dim=0) if pos_reprs else torch.zeros(self.hidden_dim)
        neg_mean = torch.stack(neg_reprs).mean(dim=0) if neg_reprs else torch.zeros(self.hidden_dim)

        tau = pos_mean - neg_mean
        tau = F.normalize(tau, p=2, dim=-1)  # L2 normalize to unit sphere
        tau = tau * self.temperature  # Scale by learned temperature
        return tau

    def forward(self, task) -> torch.Tensor:
        """Get primitive predictions for a task."""
        tau = self.encode_task(task)
        logits = self.primitive_head(tau)
        return torch.sigmoid(logits)


# =============================================================================
# TRAINING
# =============================================================================

def train_model(model: nn.Module, tasks: List, primitive_names: List[str],
                epochs: int = 100, lr: float = 0.001) -> Dict[str, Any]:
    """Train recognition model on tasks with known solutions."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    # Create targets: for each task, mark which primitives are in the solution
    # Since we don't have actual solutions, we'll use a proxy based on task name
    # This is supervised training on the primitive prediction task

    training_history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for task in tasks:
            optimizer.zero_grad()

            # Get predictions
            probs = model(task)

            # Create target based on task type (heuristic)
            target = create_target_for_task(task.name, len(primitive_names), primitive_names)

            loss = criterion(probs, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(tasks)
        training_history.append(avg_loss)

        if (epoch + 1) % 20 == 0:
            print_flush(f"    Epoch {epoch+1}/{epochs}: loss = {avg_loss:.4f}")

    return {'training_history': training_history, 'final_loss': avg_loss}


def create_target_for_task(task_name: str, num_primitives: int, primitive_names: List[str]) -> torch.Tensor:
    """Create a target vector based on task semantics."""
    target = torch.zeros(num_primitives)

    # Map task types to relevant primitives
    task_primitive_map = {
        # Suit-based tasks
        'poker_flush': ['n_unique_suits', 'lt', '2'],  # (lt (n_unique_suits hand) 2)
        'poker_same_color': ['n_unique_colors', 'lt', '2'],  # (lt (n_unique_colors hand) 2)
        'simple_has_spade': ['has_suit', 'get_suit', 'eq'],
        'simple_has_heart': ['has_suit', 'get_suit', 'eq'],
        'simple_first_red': ['first', 'get_color', 'is_red'],
        'simple_last_black': ['last', 'get_color', 'is_black'],
        'sol_same_suit_seq': ['n_unique_suits', 'lt', '2', 'get_suit'],  # Sequence of same suit
        'count_more_red': ['count_color', 'gt', 'is_red'],

        # Rank-based tasks
        'poker_has_pair': ['has_pair', 'get_rank', 'eq', 'n_unique_ranks'],
        'rummy_all_different': ['n_unique_ranks', 'length', 'eq'],
        'rummy_three_ranks': ['n_unique_ranks', '3', 'eq'],
        'bj_sum_even': ['sum_ranks', 'mod', '2', 'eq', '0'],
        'bj_sum_odd': ['sum_ranks', 'mod', '2', 'eq', '1'],
        'sym_ranks_palindrome': ['get_rank', 'reverse', 'eq'],
        'simple_diverse_ranks': ['n_unique_ranks', 'length', 'eq'],

        # Position-based tasks
        'sol_ascending': ['is_sorted', 'get_rank', 'lt'],
        'poker_straight': ['is_consecutive', 'get_rank'],
    }

    # Find matching primitives
    relevant_prims = []
    for prefix, prims in task_primitive_map.items():
        if task_name.startswith(prefix) or task_name == prefix:
            relevant_prims = prims
            break

    # Set target
    for prim in relevant_prims:
        if prim in primitive_names:
            idx = primitive_names.index(prim)
            target[idx] = 1.0

    # If no specific mapping, use generic primitives
    if target.sum() == 0:
        for prim in ['get_rank', 'get_suit', 'eq', 'map']:
            if prim in primitive_names:
                idx = primitive_names.index(prim)
                target[idx] = 0.5

    return target


# =============================================================================
# INTERPRETABILITY ANALYSIS
# =============================================================================

@dataclass
class FeatureImportance:
    """Feature importance for a single task."""
    task_name: str
    suit_importance: float
    rank_importance: float
    position_importance: float
    dominant_feature: str
    top_primitives: List[Dict[str, Any]]


def analyze_feature_importance(model: nn.Module, task, primitive_names: List[str]) -> FeatureImportance:
    """Analyze which features the model uses for this task."""
    model.eval()

    with torch.no_grad():
        # Baseline prediction
        probs_baseline = model(task)
        entropy_baseline = compute_entropy(probs_baseline)

        # We need to ablate at the embedding level
        # Store original encode_hand method
        original_encode = model.encode_hand

        entropy_changes = {}

        for feature in ['suit', 'rank', 'position']:
            # Create ablated encoder
            def make_ablated_encoder(ablate_feature):
                def ablated_encode_hand(hand):
                    if not hand:
                        return torch.zeros(model.hidden_dim)

                    suits, ranks, positions = [], [], []
                    for i, card in enumerate(hand):
                        suit_map = {Suit.SPADES: 0, Suit.HEARTS: 1, Suit.DIAMONDS: 2, Suit.CLUBS: 3}
                        suits.append(suit_map.get(card.suit, 0))
                        ranks.append(RANK_VALUES[card.rank] - 2)
                        positions.append(min(i, 7))

                    suits_t = torch.tensor(suits)
                    ranks_t = torch.tensor(ranks)
                    positions_t = torch.tensor(positions)

                    # Get embeddings
                    suit_emb = model.card_encoder.suit_embed(suits_t)
                    rank_emb = model.card_encoder.rank_embed(ranks_t)
                    pos_emb = model.card_encoder.pos_embed(positions_t)

                    # Ablate the specified feature
                    if ablate_feature == 'suit':
                        suit_emb = torch.zeros_like(suit_emb)
                    elif ablate_feature == 'rank':
                        rank_emb = torch.zeros_like(rank_emb)
                    elif ablate_feature == 'position':
                        pos_emb = torch.zeros_like(pos_emb)

                    card_emb = torch.cat([suit_emb, rank_emb, pos_emb], dim=-1)
                    card_features = model.card_mlp(card_emb)
                    return card_features.mean(dim=0)

                return ablated_encode_hand

            # Temporarily replace encoder
            model.encode_hand = make_ablated_encoder(feature)
            probs_ablated = model(task)
            entropy_ablated = compute_entropy(probs_ablated)

            # Restore original
            model.encode_hand = original_encode

            # Higher entropy after ablation = more important feature
            entropy_changes[feature] = max(0, entropy_ablated - entropy_baseline)

        # Normalize to get importance
        total = sum(entropy_changes.values()) + 1e-10
        suit_imp = entropy_changes['suit'] / total
        rank_imp = entropy_changes['rank'] / total
        pos_imp = entropy_changes['position'] / total

        # Determine dominant feature
        max_imp = max(suit_imp, rank_imp, pos_imp)
        if max_imp == suit_imp:
            dominant = 'suit'
        elif max_imp == rank_imp:
            dominant = 'rank'
        else:
            dominant = 'position'

        # Get top primitives
        top_k = min(5, len(primitive_names))
        top_values, top_indices = torch.topk(probs_baseline, top_k)
        top_primitives = [
            {'name': primitive_names[idx.item()], 'prob': float(top_values[i])}
            for i, idx in enumerate(top_indices)
        ]

        return FeatureImportance(
            task_name=task.name,
            suit_importance=float(suit_imp),
            rank_importance=float(rank_imp),
            position_importance=float(pos_imp),
            dominant_feature=dominant,
            top_primitives=top_primitives
        )


def compute_entropy(probs: torch.Tensor) -> float:
    """Compute binary entropy."""
    eps = 1e-10
    entropy = -(probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps))
    return entropy.sum().item()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_flush("=" * 70)
    print_flush("TRAINING MODELS FOR INTERPRETABILITY ANALYSIS")
    print_flush("=" * 70)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Setup
    print_flush("\nLoading grammar and tasks...")
    grammar = build_lean_grammar()
    primitive_names = [p.name for p in grammar.primitives()]
    num_primitives = len(primitive_names)
    print_flush(f"  Primitives: {num_primitives}")

    rules = get_all_pretraining_rules()
    tasks = create_tasks_from_rules(rules, n_examples=20, seed=42)
    print_flush(f"  Tasks: {len(tasks)}")

    # Output directory
    output_dir = Path(f"results_interpretability/train_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    print_flush(f"  Output: {output_dir}")

    # Define models to train
    models_to_train = {
        'LayerNorm+Scale': lambda: LayerNormRecognitionModel(num_primitives, hidden_dim=64, scale_init=20.0),
        'L2Norm+Temperature': lambda: L2NormRecognitionModel(num_primitives, hidden_dim=64, temperature_init=20.0)
    }

    all_results = {}

    for model_name, model_fn in models_to_train.items():
        print_flush(f"\n{'='*60}")
        print_flush(f"Training: {model_name}")
        print_flush("=" * 60)

        model = model_fn()

        # Train
        print_flush("\n  Training...")
        train_result = train_model(model, tasks, primitive_names, epochs=100, lr=0.001)

        # Save model
        model_path = output_dir / f"{model_name.replace('+', '_')}_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'primitive_names': primitive_names,
            'training_history': train_result['training_history'],
            'model_type': model_name
        }, model_path)
        print_flush(f"  Model saved: {model_path.name}")

        # Run interpretability analysis
        print_flush("\n  Running interpretability analysis...")

        # Expected features by task
        suit_tasks = ['poker_flush', 'poker_same_color', 'sol_same_suit_seq',
                      'simple_has_spade', 'simple_has_heart', 'simple_first_red',
                      'simple_last_black', 'count_more_red']
        rank_tasks = ['poker_has_pair', 'rummy_all_different', 'rummy_three_ranks',
                      'bj_sum_even', 'bj_sum_odd', 'sym_ranks_palindrome', 'simple_diverse_ranks']

        interpretability_results = []
        correct_suit = 0
        correct_rank = 0
        total_suit = 0
        total_rank = 0

        for task in tasks:
            result = analyze_feature_importance(model, task, primitive_names)
            interpretability_results.append(result)

            # Check correctness
            if task.name in suit_tasks:
                total_suit += 1
                if result.dominant_feature == 'suit':
                    correct_suit += 1
            elif task.name in rank_tasks:
                total_rank += 1
                if result.dominant_feature == 'rank':
                    correct_rank += 1

        # Summary stats
        mean_suit = np.mean([r.suit_importance for r in interpretability_results])
        mean_rank = np.mean([r.rank_importance for r in interpretability_results])
        mean_pos = np.mean([r.position_importance for r in interpretability_results])

        print_flush(f"\n  Feature Usage:")
        print_flush(f"    Suit:     {mean_suit:.3f}")
        print_flush(f"    Rank:     {mean_rank:.3f}")
        print_flush(f"    Position: {mean_pos:.3f}")

        print_flush(f"\n  Correctness:")
        if total_suit > 0:
            print_flush(f"    Suit tasks: {correct_suit}/{total_suit} ({100*correct_suit/total_suit:.0f}%)")
        if total_rank > 0:
            print_flush(f"    Rank tasks: {correct_rank}/{total_rank} ({100*correct_rank/total_rank:.0f}%)")

        # Check prediction diversity
        all_top_preds = [r.top_primitives[0]['name'] for r in interpretability_results]
        unique_top = len(set(all_top_preds))
        print_flush(f"\n  Prediction Diversity:")
        print_flush(f"    Unique top-1 predictions: {unique_top}/{len(tasks)}")

        # Store results
        all_results[model_name] = {
            'training': train_result,
            'interpretability': [asdict(r) for r in interpretability_results],
            'summary': {
                'mean_suit_importance': float(mean_suit),
                'mean_rank_importance': float(mean_rank),
                'mean_position_importance': float(mean_pos),
                'suit_task_accuracy': correct_suit / total_suit if total_suit > 0 else 0,
                'rank_task_accuracy': correct_rank / total_rank if total_rank > 0 else 0,
                'prediction_diversity': unique_top / len(tasks)
            }
        }

        # Print per-task results
        print_flush(f"\n  Per-Task Analysis:")
        for result in interpretability_results[:10]:  # First 10
            expected = None
            if result.task_name in suit_tasks:
                expected = 'suit'
            elif result.task_name in rank_tasks:
                expected = 'rank'

            match = ""
            if expected:
                match = "✓" if result.dominant_feature == expected else "✗"

            print_flush(f"    {result.task_name}: S={result.suit_importance:.2f} R={result.rank_importance:.2f} P={result.position_importance:.2f} → {result.dominant_feature} {match}")

    # Save all results
    results_path = output_dir / "results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Final comparison
    print_flush("\n" + "=" * 70)
    print_flush("COMPARISON SUMMARY")
    print_flush("=" * 70)

    for model_name, results in all_results.items():
        summary = results['summary']
        print_flush(f"\n{model_name}:")
        print_flush(f"  Suit task accuracy:  {100*summary['suit_task_accuracy']:.0f}%")
        print_flush(f"  Rank task accuracy:  {100*summary['rank_task_accuracy']:.0f}%")
        print_flush(f"  Prediction diversity: {100*summary['prediction_diversity']:.0f}%")

    print_flush(f"\nResults saved to: {output_dir}")
    print_flush("=" * 70)


if __name__ == "__main__":
    main()
