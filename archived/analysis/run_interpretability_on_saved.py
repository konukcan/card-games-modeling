#!/usr/bin/env python3
"""
Run interpretability analysis on saved recognition models.

Uses the factored ablation method to determine which features
(suit, rank, position) the model relies on for each task.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.dreamcoder_original import create_tasks_from_rules
from rules.pretraining_rules import get_all_pretraining_rules
from rules.catalogue import create_all_rules
from rules.cards import Suit, Rank, RANK_VALUES


def print_flush(msg: str):
    print(msg, flush=True)


@dataclass
class FeatureImportance:
    """Feature importance for a single task."""
    task_name: str
    suit_importance: float
    rank_importance: float
    position_importance: float
    dominant_feature: str
    top_primitives: List[Dict[str, Any]]
    entropy_baseline: float
    entropy_changes: Dict[str, float]


class SavedModelAnalyzer:
    """Analyze saved ContrastiveRecognitionModel checkpoints."""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(checkpoint_path)
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.device = torch.device('cpu')

        # Extract model info
        self.primitive_names = self.checkpoint.get('primitive_names', [])
        self.config = self.checkpoint.get('config', {})
        self.state_dict = self.checkpoint['model_state_dict']

        # Infer dimensions from state dict
        self._infer_dimensions()

    def _infer_dimensions(self):
        """Infer model dimensions from state dict."""
        # Get embedding dimensions
        if 'card_encoder.suit_embed.weight' in self.state_dict:
            suit_weight = self.state_dict['card_encoder.suit_embed.weight']
            self.d_suit = suit_weight.shape[1]
            self.n_suits = suit_weight.shape[0]

        if 'card_encoder.rank_embed.weight' in self.state_dict:
            rank_weight = self.state_dict['card_encoder.rank_embed.weight']
            self.d_rank = rank_weight.shape[1]
            self.n_ranks = rank_weight.shape[0]

        if 'card_encoder.pos_embed.weight' in self.state_dict:
            pos_weight = self.state_dict['card_encoder.pos_embed.weight']
            self.d_pos = pos_weight.shape[1]
            self.max_cards = pos_weight.shape[0]

        print_flush(f"  Dimensions: suit={self.d_suit}, rank={self.d_rank}, pos={self.d_pos}")
        print_flush(f"  Vocab: {self.n_suits} suits, {self.n_ranks} ranks, {self.max_cards} positions")
        print_flush(f"  Primitives: {len(self.primitive_names)}")

    def _encode_hand(self, hand, ablate_feature: str = None) -> torch.Tensor:
        """Encode a hand using the saved weights, optionally ablating a feature."""
        # Get embeddings
        suit_embed = self.state_dict['card_encoder.suit_embed.weight']
        rank_embed = self.state_dict['card_encoder.rank_embed.weight']
        pos_embed = self.state_dict['card_encoder.pos_embed.weight']

        # Card MLP weights
        mlp_w1 = self.state_dict['card_mlp.mlp.0.weight']
        mlp_b1 = self.state_dict['card_mlp.mlp.0.bias']
        mlp_w2 = self.state_dict['card_mlp.mlp.3.weight']
        mlp_b2 = self.state_dict['card_mlp.mlp.3.bias']

        # Encode each card
        card_embeddings = []
        for i, card in enumerate(hand):
            # Map suit to index (0-3)
            suit_map = {Suit.SPADES: 0, Suit.HEARTS: 1, Suit.DIAMONDS: 2, Suit.CLUBS: 3}
            suit_idx = suit_map.get(card.suit, 0)
            rank_idx = RANK_VALUES[card.rank] - 2  # 2-14 -> 0-12
            pos_idx = min(i, self.max_cards - 1)

            # Get embeddings
            suit_emb = suit_embed[suit_idx] if ablate_feature != 'suit' else torch.zeros(self.d_suit)
            rank_emb = rank_embed[rank_idx] if ablate_feature != 'rank' else torch.zeros(self.d_rank)
            pos_emb = pos_embed[pos_idx] if ablate_feature != 'position' else torch.zeros(self.d_pos)

            # Concatenate
            card_emb = torch.cat([suit_emb, rank_emb, pos_emb])
            card_embeddings.append(card_emb)

        if not card_embeddings:
            hidden_dim = mlp_w2.shape[0]
            return torch.zeros(hidden_dim)

        # Stack and apply MLP
        cards = torch.stack(card_embeddings)  # (n_cards, d_card)

        # MLP forward
        x = F.relu(cards @ mlp_w1.t() + mlp_b1)
        x = x @ mlp_w2.t() + mlp_b2  # (n_cards, hidden_dim)

        # Mean pool
        hand_repr = x.mean(dim=0)
        return hand_repr

    def _encode_task(self, task, ablate_feature: str = None) -> torch.Tensor:
        """Encode a task using contrastive encoding."""
        pos_reprs = []
        neg_reprs = []

        for hand, label in task.examples:
            repr = self._encode_hand(hand, ablate_feature)
            if label:
                pos_reprs.append(repr)
            else:
                neg_reprs.append(repr)

        # Contrastive: positive mean - negative mean
        if pos_reprs:
            pos_mean = torch.stack(pos_reprs).mean(dim=0)
        else:
            pos_mean = torch.zeros_like(neg_reprs[0]) if neg_reprs else torch.zeros(64)

        if neg_reprs:
            neg_mean = torch.stack(neg_reprs).mean(dim=0)
        else:
            neg_mean = torch.zeros_like(pos_reprs[0]) if pos_reprs else torch.zeros(64)

        tau = pos_mean - neg_mean

        # Apply normalization if present
        if 'embedding_norm.weight' in self.state_dict:
            norm_weight = self.state_dict['embedding_norm.weight']
            norm_bias = self.state_dict['embedding_norm.bias']
            scale = self.state_dict.get('embedding_scale', torch.tensor(20.0))

            # LayerNorm
            mean = tau.mean()
            var = tau.var(unbiased=False)
            tau = (tau - mean) / torch.sqrt(var + 1e-5)
            tau = tau * norm_weight + norm_bias
            tau = tau * scale

        return tau

    def _predict(self, tau: torch.Tensor) -> torch.Tensor:
        """Get primitive predictions from task embedding."""
        # Get prediction head weights (structure: primitive_head.mlp.X.weight)
        head_w1 = self.state_dict['primitive_head.mlp.0.weight']
        head_b1 = self.state_dict['primitive_head.mlp.0.bias']
        head_w2 = self.state_dict['primitive_head.mlp.3.weight']
        head_b2 = self.state_dict['primitive_head.mlp.3.bias']

        # Forward through prediction head
        x = F.relu(tau @ head_w1.t() + head_b1)
        logits = x @ head_w2.t() + head_b2

        # Sigmoid for probabilities
        probs = torch.sigmoid(logits)
        return probs

    def analyze_task(self, task) -> FeatureImportance:
        """Analyze which features matter for this task."""
        # Baseline prediction
        tau_baseline = self._encode_task(task)
        probs_baseline = self._predict(tau_baseline)
        entropy_baseline = -(probs_baseline * torch.log(probs_baseline + 1e-10) +
                            (1 - probs_baseline) * torch.log(1 - probs_baseline + 1e-10)).sum().item()

        # Ablate each feature
        entropy_changes = {}
        for feature in ['suit', 'rank', 'position']:
            tau_ablated = self._encode_task(task, ablate_feature=feature)
            probs_ablated = self._predict(tau_ablated)
            entropy_ablated = -(probs_ablated * torch.log(probs_ablated + 1e-10) +
                               (1 - probs_ablated) * torch.log(1 - probs_ablated + 1e-10)).sum().item()
            # Increase in entropy = feature was important
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
        top_k = min(5, len(self.primitive_names))
        top_values, top_indices = torch.topk(probs_baseline, top_k)
        top_primitives = [
            {'name': self.primitive_names[idx.item()], 'prob': float(top_values[i])}
            for i, idx in enumerate(top_indices)
        ]

        return FeatureImportance(
            task_name=task.name,
            suit_importance=float(suit_imp),
            rank_importance=float(rank_imp),
            position_importance=float(pos_imp),
            dominant_feature=dominant,
            top_primitives=top_primitives,
            entropy_baseline=entropy_baseline,
            entropy_changes=entropy_changes
        )


def main():
    print_flush("=" * 70)
    print_flush("INTERPRETABILITY ANALYSIS ON SAVED MODELS")
    print_flush("=" * 70)

    # Find saved models
    model_dir = Path("results/warmstart_experiment/contrastive_BOTH_20251225_145818")
    warm_model = model_dir / "recognition_model_WARM.pt"

    if not warm_model.exists():
        print_flush(f"ERROR: Model not found at {warm_model}")
        return

    print_flush(f"\nLoading model: {warm_model.name}")
    analyzer = SavedModelAnalyzer(str(warm_model))

    # Create tasks
    print_flush("\nCreating tasks...")
    rules = get_all_pretraining_rules()
    tasks = create_tasks_from_rules(rules, n_examples=20, seed=42)
    print_flush(f"  Tasks: {len(tasks)}")

    # Analyze each task
    print_flush("\n" + "=" * 70)
    print_flush("FEATURE IMPORTANCE BY TASK")
    print_flush("=" * 70)

    results = []

    # Group by expected dominant feature
    suit_tasks = ['poker_flush', 'poker_same_color', 'sol_same_suit_seq',
                  'simple_has_spade', 'simple_has_heart', 'simple_first_red',
                  'simple_last_black', 'count_more_red', 'simple_ends_color']
    rank_tasks = ['poker_has_pair', 'rummy_all_different', 'rummy_three_ranks',
                  'bj_sum_even', 'bj_sum_odd', 'sym_ranks_palindrome', 'simple_diverse_ranks']
    position_tasks = ['sol_ascending', 'sym_ranks_palindrome']

    for task in tasks:
        result = analyzer.analyze_task(task)
        results.append(result)

        # Determine if model matches expectation
        expected = None
        if task.name in suit_tasks:
            expected = 'suit'
        elif task.name in rank_tasks:
            expected = 'rank'
        elif task.name in position_tasks:
            expected = 'position'

        match_symbol = ""
        if expected:
            match_symbol = "✓" if result.dominant_feature == expected else "✗"

        print_flush(f"\n{task.name}:")
        print_flush(f"  Suit:     {result.suit_importance:.3f}")
        print_flush(f"  Rank:     {result.rank_importance:.3f}")
        print_flush(f"  Position: {result.position_importance:.3f}")
        print_flush(f"  Dominant: {result.dominant_feature} {match_symbol}")
        print_flush(f"  Top preds: {[p['name'] for p in result.top_primitives[:3]]}")

    # Summary
    print_flush("\n" + "=" * 70)
    print_flush("SUMMARY")
    print_flush("=" * 70)

    # Aggregate by task type
    suit_results = [r for r in results if r.task_name in suit_tasks]
    rank_results = [r for r in results if r.task_name in rank_tasks]

    if suit_results:
        mean_suit = np.mean([r.suit_importance for r in suit_results])
        print_flush(f"\nSuit-based tasks (n={len(suit_results)}):")
        print_flush(f"  Mean suit importance: {mean_suit:.3f}")
        correct = sum(1 for r in suit_results if r.dominant_feature == 'suit')
        print_flush(f"  Correctly identified: {correct}/{len(suit_results)}")

    if rank_results:
        mean_rank = np.mean([r.rank_importance for r in rank_results])
        print_flush(f"\nRank-based tasks (n={len(rank_results)}):")
        print_flush(f"  Mean rank importance: {mean_rank:.3f}")
        correct = sum(1 for r in rank_results if r.dominant_feature == 'rank')
        print_flush(f"  Correctly identified: {correct}/{len(rank_results)}")

    # Overall feature usage
    print_flush("\nOverall feature usage:")
    mean_suit = np.mean([r.suit_importance for r in results])
    mean_rank = np.mean([r.rank_importance for r in results])
    mean_pos = np.mean([r.position_importance for r in results])
    print_flush(f"  Suit: {mean_suit:.3f}")
    print_flush(f"  Rank: {mean_rank:.3f}")
    print_flush(f"  Position: {mean_pos:.3f}")

    # Save results
    output_path = Path("results_interpretability")
    output_path.mkdir(exist_ok=True)

    results_json = {
        'model': str(warm_model),
        'n_tasks': len(results),
        'summary': {
            'mean_suit_importance': float(mean_suit),
            'mean_rank_importance': float(mean_rank),
            'mean_position_importance': float(mean_pos),
        },
        'tasks': [asdict(r) for r in results]
    }

    out_file = output_path / "interpretability_results.json"
    with open(out_file, 'w') as f:
        json.dump(results_json, f, indent=2)

    print_flush(f"\nResults saved to: {out_file}")
    print_flush("\n" + "=" * 70)


if __name__ == "__main__":
    main()
