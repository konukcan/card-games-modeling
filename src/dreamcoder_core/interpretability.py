#!/usr/bin/env python3
"""
Interpretability Module for ContrastiveRecognitionModel

This module provides tools to understand WHY the recognition model makes
certain primitive predictions for each task. Key questions:
- Which card features (suit, rank, position) drive predictions?
- Which hands (positive vs negative) contribute most?
- How do embeddings decompose by feature type?

Methods implemented:
1. Integrated Gradients: Attribution of output to input features
2. Factored Attribution: Decompose by suit/rank/position embeddings
3. Hand Importance: Which examples matter most for task encoding
4. Saliency Maps: First-order gradient-based importance

Author: Can Konuk
Date: December 2024
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


@dataclass
class InterpretabilityResult:
    """Container for interpretability analysis results."""
    task_name: str
    # Feature attributions (mean absolute attribution per feature type)
    suit_attribution: float  # How much suit information matters
    rank_attribution: float  # How much rank information matters
    position_attribution: float  # How much position information matters
    # Per-example importance
    example_importances: List[Tuple[bool, float]]  # [(is_positive, importance), ...]
    # Top predicted primitives with their attributions
    top_primitives: List[Dict[str, Any]]
    # Raw attributions if needed
    raw_card_attributions: Optional[np.ndarray] = None


class InterpretabilityAnalyzer:
    """
    Analyze ContrastiveRecognitionModel predictions for interpretability.

    Uses Integrated Gradients to attribute predictions to input features.
    """

    def __init__(
        self,
        model: nn.Module,
        n_integration_steps: int = 50,
        baseline_type: str = 'zero'  # 'zero' or 'random'
    ):
        """
        Args:
            model: ContrastiveRecognitionModel instance
            n_integration_steps: Number of steps for IG approximation
            baseline_type: Type of baseline ('zero' or 'random')
        """
        self.model = model
        self.n_steps = n_integration_steps
        self.baseline_type = baseline_type

        # Cache dimensions from model
        self.d_suit = model.card_encoder.d_suit
        self.d_rank = model.card_encoder.d_rank
        self.d_pos = model.card_encoder.d_pos

    def _get_baseline(self, shape: Tuple[int, ...], device: torch.device) -> torch.Tensor:
        """Get baseline embedding for IG."""
        if self.baseline_type == 'zero':
            return torch.zeros(shape, device=device)
        else:  # random
            return torch.randn(shape, device=device) * 0.01

    def integrated_gradients(
        self,
        task,
        target_primitive_idx: Optional[int] = None,
        return_raw: bool = False
    ) -> Dict[str, Any]:
        """
        Compute Integrated Gradients for a task's primitive predictions.

        Args:
            task: Task object with .examples
            target_primitive_idx: If specified, compute IG for this primitive only.
                                  Otherwise, compute for top predicted primitive.
            return_raw: If True, include raw attributions in result

        Returns:
            Dict with attribution analysis results
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        # Encode hands and track gradients
        pos_hands_data = []
        neg_hands_data = []

        for hand, label in task.examples:
            encoded = self.model._encode_hand_raw(hand)
            if label:
                pos_hands_data.append(encoded)
            else:
                neg_hands_data.append(encoded)

        # Get card embeddings with gradients enabled
        def get_card_embeddings(hands_data) -> torch.Tensor:
            """Get stacked card embeddings for a set of hands."""
            if not hands_data:
                return None

            suits = torch.stack([h[0] for h in hands_data]).to(device)
            ranks = torch.stack([h[1] for h in hands_data]).to(device)
            positions = torch.stack([h[2] for h in hands_data]).to(device)
            masks = torch.stack([h[3] for h in hands_data]).to(device)

            # Get factored embeddings
            card_emb = self.model.card_encoder(suits, ranks, positions)
            return card_emb, masks

        # Get embeddings for positive and negative hands
        pos_emb = None
        neg_emb = None
        pos_masks = None
        neg_masks = None

        if pos_hands_data:
            pos_emb, pos_masks = get_card_embeddings(pos_hands_data)
        if neg_hands_data:
            neg_emb, neg_masks = get_card_embeddings(neg_hands_data)

        # Forward pass to get target primitive
        with torch.no_grad():
            probs = self.model.predict_primitives(task)
            if target_primitive_idx is None:
                target_primitive_idx = probs.argmax().item()

        # Compute Integrated Gradients
        # We'll use the sum of pos and neg embeddings as our "input"
        # and trace attributions through the network

        attributions_pos = None
        attributions_neg = None

        if pos_emb is not None:
            attributions_pos = self._compute_ig_for_embedding(
                pos_emb, pos_masks, neg_emb, neg_masks,
                target_primitive_idx, is_positive=True
            )

        if neg_emb is not None:
            attributions_neg = self._compute_ig_for_embedding(
                neg_emb, neg_masks, pos_emb, pos_masks,
                target_primitive_idx, is_positive=False
            )

        # Aggregate attributions by feature type
        suit_attr = 0.0
        rank_attr = 0.0
        pos_attr = 0.0
        example_importances = []

        if attributions_pos is not None:
            for i, attr in enumerate(attributions_pos):
                # attr is (n_cards, embedding_dim)
                suit_attr += np.abs(attr[:, :self.d_suit]).sum()
                rank_attr += np.abs(attr[:, self.d_suit:self.d_suit+self.d_rank]).sum()
                pos_attr += np.abs(attr[:, self.d_suit+self.d_rank:]).sum()
                example_importances.append((True, float(np.abs(attr).sum())))

        if attributions_neg is not None:
            for i, attr in enumerate(attributions_neg):
                suit_attr += np.abs(attr[:, :self.d_suit]).sum()
                rank_attr += np.abs(attr[:, self.d_suit:self.d_suit+self.d_rank]).sum()
                pos_attr += np.abs(attr[:, self.d_suit+self.d_rank:]).sum()
                example_importances.append((False, float(np.abs(attr).sum())))

        # Normalize
        total = suit_attr + rank_attr + pos_attr + 1e-10
        suit_attr /= total
        rank_attr /= total
        pos_attr /= total

        # Get top primitives with attributions
        top_k = min(5, len(self.model.primitive_names))
        top_values, top_indices = torch.topk(probs, top_k)

        top_primitives = [
            {
                'name': self.model.primitive_names[idx.item()],
                'probability': float(top_values[i]),
                'is_target': idx.item() == target_primitive_idx
            }
            for i, idx in enumerate(top_indices)
        ]

        result = InterpretabilityResult(
            task_name=task.name,
            suit_attribution=float(suit_attr),
            rank_attribution=float(rank_attr),
            position_attribution=float(pos_attr),
            example_importances=example_importances,
            top_primitives=top_primitives
        )

        if return_raw:
            raw = []
            if attributions_pos is not None:
                raw.extend(attributions_pos)
            if attributions_neg is not None:
                raw.extend(attributions_neg)
            result.raw_card_attributions = raw

        return result

    def _compute_ig_for_embedding(
        self,
        target_emb: torch.Tensor,
        target_masks: torch.Tensor,
        other_emb: Optional[torch.Tensor],
        other_masks: Optional[torch.Tensor],
        target_primitive_idx: int,
        is_positive: bool
    ) -> List[np.ndarray]:
        """
        Compute Integrated Gradients for a set of card embeddings.

        Returns list of attribution arrays, one per hand in target_emb.
        """
        device = target_emb.device
        batch_size, max_cards, emb_dim = target_emb.shape

        # Baseline
        baseline = self._get_baseline(target_emb.shape, device)

        # Compute gradients at each step
        attributions = []

        for batch_idx in range(batch_size):
            hand_attributions = torch.zeros(max_cards, emb_dim, device=device)
            hand_emb = target_emb[batch_idx:batch_idx+1]  # (1, max_cards, emb_dim)
            hand_base = baseline[batch_idx:batch_idx+1]
            hand_mask = target_masks[batch_idx:batch_idx+1]

            for step in range(self.n_steps):
                alpha = step / self.n_steps

                # Interpolate between baseline and actual embedding
                interp_emb = hand_base + alpha * (hand_emb - hand_base)
                interp_emb.requires_grad_(True)

                # Forward pass through rest of network
                # Apply card MLP
                card_features = self.model.card_mlp(interp_emb)

                # Mean pool
                if hand_mask is not None:
                    mask_expanded = hand_mask.unsqueeze(-1).float()
                    card_features = card_features * mask_expanded
                    hand_repr = card_features.sum(dim=1) / hand_mask.sum(dim=1, keepdim=True).clamp(min=1)
                else:
                    hand_repr = card_features.mean(dim=1)

                # Contrastive encoding
                if other_emb is not None:
                    other_features = self.model.card_mlp(other_emb)
                    if other_masks is not None:
                        mask_exp = other_masks.unsqueeze(-1).float()
                        other_features = other_features * mask_exp
                        other_repr = other_features.sum(dim=1) / other_masks.sum(dim=1, keepdim=True).clamp(min=1)
                    else:
                        other_repr = other_features.mean(dim=1)
                    other_mean = other_repr.mean(dim=0, keepdim=True)
                else:
                    other_mean = torch.zeros_like(hand_repr)

                if is_positive:
                    τ = hand_repr - other_mean
                else:
                    # For negative hands, they are subtracted from positives
                    τ = -hand_repr + other_mean

                # Apply normalization if present
                if self.model.normalize_embeddings and self.model.embedding_norm is not None:
                    τ = self.model.embedding_norm(τ) * self.model.embedding_scale

                # Get prediction
                pred = self.model.primitive_head(τ)
                if self.model.output_mode == 'softmax':
                    pred = torch.exp(pred)

                # Compute gradient
                target_pred = pred[0, target_primitive_idx]
                target_pred.backward(retain_graph=True)

                if interp_emb.grad is not None:
                    hand_attributions += interp_emb.grad[0]
                    interp_emb.grad.zero_()

            # Scale by (input - baseline)
            hand_attributions = hand_attributions * (hand_emb[0] - hand_base[0]) / self.n_steps
            attributions.append(hand_attributions.detach().cpu().numpy())

        return attributions

    def factored_importance(self, task) -> Dict[str, float]:
        """
        Compute simple feature importance by ablating each factor.

        Faster than IG but less precise. Good for quick analysis.
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        with torch.no_grad():
            # Get baseline prediction
            base_probs = self.model.predict_primitives(task)
            base_entropy = -(base_probs * torch.log(base_probs + 1e-10)).sum().item()

            # Encode hands
            pos_hands = []
            neg_hands = []
            for hand, label in task.examples:
                encoded = self.model._encode_hand_raw(hand)
                if label:
                    pos_hands.append(encoded)
                else:
                    neg_hands.append(encoded)

            # Ablate each factor and measure change
            importances = {'suit': 0.0, 'rank': 0.0, 'position': 0.0}

            for factor, (start_idx, end_idx) in [
                ('suit', (0, self.d_suit)),
                ('rank', (self.d_suit, self.d_suit + self.d_rank)),
                ('position', (self.d_suit + self.d_rank, self.d_suit + self.d_rank + self.d_pos))
            ]:
                # Zero out this factor in the embeddings
                ablated_probs = self._predict_with_ablation(
                    task, pos_hands, neg_hands, start_idx, end_idx
                )
                ablated_entropy = -(ablated_probs * torch.log(ablated_probs + 1e-10)).sum().item()

                # Higher entropy after ablation = more important factor
                importances[factor] = max(0, ablated_entropy - base_entropy)

            # Normalize
            total = sum(importances.values()) + 1e-10
            return {k: v/total for k, v in importances.items()}

    def _predict_with_ablation(
        self,
        task,
        pos_hands: List,
        neg_hands: List,
        ablate_start: int,
        ablate_end: int
    ) -> torch.Tensor:
        """Predict with part of embedding ablated (zeroed out)."""
        device = next(self.model.parameters()).device
        hidden_dim = self.model.card_mlp.mlp[-2].out_features

        def batch_encode_ablated(hands_data):
            if not hands_data:
                return torch.zeros(hidden_dim, device=device)

            suits = torch.stack([h[0] for h in hands_data]).to(device)
            ranks = torch.stack([h[1] for h in hands_data]).to(device)
            positions = torch.stack([h[2] for h in hands_data]).to(device)
            masks = torch.stack([h[3] for h in hands_data]).to(device)

            # Get factored embeddings
            card_emb = self.model.card_encoder(suits, ranks, positions)

            # Ablate specified range
            card_emb[:, :, ablate_start:ablate_end] = 0

            # Rest of encoding
            card_features = self.model.card_mlp(card_emb)
            mask_exp = masks.unsqueeze(-1).float()
            card_features = card_features * mask_exp
            hand_repr = card_features.sum(dim=1) / masks.sum(dim=1, keepdim=True).clamp(min=1)
            return hand_repr.mean(dim=0)

        pos_mean = batch_encode_ablated(pos_hands)
        neg_mean = batch_encode_ablated(neg_hands)

        τ = pos_mean - neg_mean

        if self.model.normalize_embeddings and self.model.embedding_norm is not None:
            τ = self.model.embedding_norm(τ) * self.model.embedding_scale

        probs = self.model.primitive_head(τ.unsqueeze(0))
        if self.model.output_mode == 'softmax':
            probs = torch.exp(probs)

        return probs.squeeze(0)

    def analyze_task(
        self,
        task,
        method: str = 'factored',  # 'ig' or 'factored'
        **kwargs
    ) -> InterpretabilityResult:
        """
        Main entry point for interpretability analysis.

        Args:
            task: Task to analyze
            method: 'ig' for Integrated Gradients, 'factored' for ablation-based

        Returns:
            InterpretabilityResult with attributions
        """
        if method == 'ig':
            return self.integrated_gradients(task, **kwargs)
        else:
            # Use factored importance
            importance = self.factored_importance(task)

            # Get predictions
            probs = self.model.predict_primitives(task)
            top_k = min(5, len(self.model.primitive_names))
            top_values, top_indices = torch.topk(probs, top_k)

            top_primitives = [
                {
                    'name': self.model.primitive_names[idx.item()],
                    'probability': float(top_values[i])
                }
                for i, idx in enumerate(top_indices)
            ]

            return InterpretabilityResult(
                task_name=task.name,
                suit_attribution=importance['suit'],
                rank_attribution=importance['rank'],
                position_attribution=importance['position'],
                example_importances=[],  # Not computed in factored method
                top_primitives=top_primitives
            )


def analyze_multiple_tasks(
    model: nn.Module,
    tasks: List,
    method: str = 'factored'
) -> Dict[str, InterpretabilityResult]:
    """
    Analyze multiple tasks and return aggregated results.

    Args:
        model: ContrastiveRecognitionModel
        tasks: List of tasks to analyze
        method: 'ig' or 'factored'

    Returns:
        Dict mapping task names to InterpretabilityResult
    """
    analyzer = InterpretabilityAnalyzer(model)
    results = {}

    for task in tasks:
        try:
            result = analyzer.analyze_task(task, method=method)
            results[task.name] = result
        except Exception as e:
            print(f"Error analyzing {task.name}: {e}")
            continue

    return results


def summarize_results(results: Dict[str, InterpretabilityResult]) -> Dict[str, Any]:
    """
    Summarize interpretability results across multiple tasks.

    Returns:
        Summary statistics
    """
    if not results:
        return {}

    suit_attrs = [r.suit_attribution for r in results.values()]
    rank_attrs = [r.rank_attribution for r in results.values()]
    pos_attrs = [r.position_attribution for r in results.values()]

    return {
        'n_tasks': len(results),
        'mean_suit_attribution': float(np.mean(suit_attrs)),
        'mean_rank_attribution': float(np.mean(rank_attrs)),
        'mean_position_attribution': float(np.mean(pos_attrs)),
        'std_suit_attribution': float(np.std(suit_attrs)),
        'std_rank_attribution': float(np.std(rank_attrs)),
        'std_position_attribution': float(np.std(pos_attrs)),
        'dominant_feature': max(
            [('suit', np.mean(suit_attrs)),
             ('rank', np.mean(rank_attrs)),
             ('position', np.mean(pos_attrs))],
            key=lambda x: x[1]
        )[0]
    }


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("INTERPRETABILITY MODULE TEST")
    print("=" * 70)

    from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
    from dreamcoder_core.primitives import build_lean_grammar
    from dreamcoder_core.dreamcoder_original import create_tasks_from_rules
    from rules.catalogue import create_all_rules

    # Load grammar and tasks
    print("\nLoading...")
    grammar = build_lean_grammar()
    rules = create_all_rules()
    tasks = create_tasks_from_rules(rules, n_examples=20, seed=42)

    # Create model with fix
    print("Creating model...")
    model = ContrastiveRecognitionModel(
        grammar=grammar,
        card_hidden=128,
        card_out=64,
        pred_hidden=128,
        normalize_embeddings=True,
        embedding_scale=20.0,
        encoding_mode='standard'
    )

    # Test interpretability
    print("\nTesting factored importance...")
    analyzer = InterpretabilityAnalyzer(model)

    for task in tasks[:5]:
        result = analyzer.analyze_task(task, method='factored')
        print(f"\n{task.name}:")
        print(f"  Suit: {result.suit_attribution:.3f}")
        print(f"  Rank: {result.rank_attribution:.3f}")
        print(f"  Position: {result.position_attribution:.3f}")
        print(f"  Top predictions: {[p['name'] for p in result.top_primitives[:3]]}")

    # Summary
    print("\n" + "-" * 50)
    print("SUMMARY (first 10 tasks)")
    results = analyze_multiple_tasks(model, tasks[:10], method='factored')
    summary = summarize_results(results)
    print(f"  Mean suit attribution: {summary['mean_suit_attribution']:.3f}")
    print(f"  Mean rank attribution: {summary['mean_rank_attribution']:.3f}")
    print(f"  Mean position attribution: {summary['mean_position_attribution']:.3f}")
    print(f"  Dominant feature: {summary['dominant_feature']}")

    print("\n" + "=" * 70)
    print("INTERPRETABILITY TEST COMPLETE")
    print("=" * 70)
