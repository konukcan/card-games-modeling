#!/usr/bin/env python3
"""
Systematic test of embedding magnitude fixes for ContrastiveRecognitionModel.

Tests:
1. LayerNorm + learned scaling
2. L2 normalization + fixed/learned scale
3. Larger initialization for prediction head
4. Random hands contrast variants

Metrics:
- Embedding diversity (cosine similarity)
- Embedding magnitude (L2 norm)
- Prediction diversity (unique top-5 sets)
- Prediction spread (std of probabilities)
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional
import random
from dataclasses import dataclass

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.contrastive_recognition import (
    ContrastiveRecognitionModel,
    FactoredCardEncoder,
    CardInteractionMLP,
    HandEncoder,
    ContrastiveTaskEncoder,
    PrimitiveHead
)
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.dreamcoder_original import create_tasks_from_rules
from rules.catalogue import create_all_rules
from rules.cards import sample_hand


@dataclass
class EvaluationResult:
    """Results from evaluating an embedding approach."""
    name: str
    embedding_mean_sim: float
    embedding_std_sim: float
    embedding_mean_norm: float
    embedding_std_norm: float
    prediction_unique_top5: int
    prediction_mean_std: float
    prediction_spread: float  # max - min probability
    sample_predictions: Dict[str, List[str]]  # task -> top-3 primitives


# ============================================================================
# FIX 1: LAYER NORMALIZATION + LEARNED SCALE
# ============================================================================

class NormalizedPrimitiveHead(nn.Module):
    """Primitive head with LayerNorm on input embeddings."""

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
        num_primitives: int = 67,
        init_scale: float = 10.0
    ):
        super().__init__()
        self.num_primitives = num_primitives

        # Layer normalization on input
        self.layer_norm = nn.LayerNorm(input_dim)

        # Learned scale factor
        self.scale = nn.Parameter(torch.tensor(init_scale))

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_primitives)
        )

        self.thresholds = nn.Parameter(torch.zeros(num_primitives))

    def forward(self, τ: torch.Tensor, return_binary: bool = False) -> torch.Tensor:
        # Normalize and scale
        τ_normalized = self.layer_norm(τ) * self.scale

        logits = self.mlp(τ_normalized)
        probs = torch.sigmoid(logits)

        if return_binary:
            thresholds = torch.sigmoid(self.thresholds)
            binary = (probs > thresholds).float()
            return probs, binary
        return probs


# ============================================================================
# FIX 2: L2 NORMALIZATION + FIXED/LEARNED SCALE
# ============================================================================

class L2NormalizedPrimitiveHead(nn.Module):
    """Primitive head with L2 normalization on input embeddings."""

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
        num_primitives: int = 67,
        scale: float = 10.0,
        learnable_scale: bool = True
    ):
        super().__init__()
        self.num_primitives = num_primitives

        if learnable_scale:
            self.scale = nn.Parameter(torch.tensor(scale))
        else:
            self.register_buffer('scale', torch.tensor(scale))

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_primitives)
        )

        self.thresholds = nn.Parameter(torch.zeros(num_primitives))

    def forward(self, τ: torch.Tensor, return_binary: bool = False) -> torch.Tensor:
        # L2 normalize and scale
        τ_normalized = F.normalize(τ, p=2, dim=-1) * self.scale

        logits = self.mlp(τ_normalized)
        probs = torch.sigmoid(logits)

        if return_binary:
            thresholds = torch.sigmoid(self.thresholds)
            binary = (probs > thresholds).float()
            return probs, binary
        return probs


# ============================================================================
# FIX 3: LARGER INITIALIZATION
# ============================================================================

class LargeInitPrimitiveHead(nn.Module):
    """Primitive head with larger weight initialization."""

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
        num_primitives: int = 67,
        init_gain: float = 5.0
    ):
        super().__init__()
        self.num_primitives = num_primitives

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_primitives)
        )

        # Initialize with larger weights
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=init_gain)
                nn.init.zeros_(layer.bias)

        self.thresholds = nn.Parameter(torch.zeros(num_primitives))

    def forward(self, τ: torch.Tensor, return_binary: bool = False) -> torch.Tensor:
        logits = self.mlp(τ)
        probs = torch.sigmoid(logits)

        if return_binary:
            thresholds = torch.sigmoid(self.thresholds)
            binary = (probs > thresholds).float()
            return probs, binary
        return probs


# ============================================================================
# VARIANT 4: RANDOM HANDS CONTRAST - NEW TASK ENCODER
# ============================================================================

class RandomContrastTaskEncoder(nn.Module):
    """
    Task encoder with random hands contrast.

    τ = mean(pos) - mean(neg) + λ * (mean(all) - mean(random))

    The random contrast helps distinguish "what makes these hands special"
    from any arbitrary hand.
    """

    def __init__(
        self,
        hand_encoder: HandEncoder,
        lambda_random: float = 0.5,
        n_random_hands: int = 10,
        hand_size: int = 6
    ):
        super().__init__()
        self.hand_encoder = hand_encoder
        self.lambda_random = lambda_random
        self.n_random_hands = n_random_hands
        self.hand_size = hand_size

    def _encode_hand_raw(self, hand, max_cards: int = 8):
        """Convert a hand to tensor indices."""
        from rules.cards import Suit, Rank

        suit_map = {Suit.CLUBS: 0, Suit.DIAMONDS: 1, Suit.HEARTS: 2, Suit.SPADES: 3}
        rank_list = list(Rank)
        rank_map = {r: i for i, r in enumerate(rank_list)}

        n_cards = min(len(hand), max_cards)
        suits = torch.zeros(max_cards, dtype=torch.long)
        ranks = torch.zeros(max_cards, dtype=torch.long)
        positions = torch.arange(max_cards, dtype=torch.long)
        mask = torch.zeros(max_cards, dtype=torch.bool)

        for i, card in enumerate(hand[:max_cards]):
            suits[i] = suit_map[card.suit]
            ranks[i] = rank_map[card.rank]
            mask[i] = True

        return suits, ranks, positions, mask

    def forward(
        self,
        pos_hands: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
        neg_hands: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]
    ) -> torch.Tensor:
        hidden_dim = self.hand_encoder.card_mlp.mlp[-2].out_features
        device = next(self.hand_encoder.parameters()).device

        # Encode positive examples
        if pos_hands:
            pos_embeddings = []
            for suits, ranks, positions, mask in pos_hands:
                h = self.hand_encoder(
                    suits.unsqueeze(0).to(device),
                    ranks.unsqueeze(0).to(device),
                    positions.unsqueeze(0).to(device),
                    mask.unsqueeze(0).to(device) if mask is not None else None
                )
                pos_embeddings.append(h.squeeze(0))
            pos_mean = torch.stack(pos_embeddings).mean(dim=0)
        else:
            pos_mean = torch.zeros(hidden_dim, device=device)

        # Encode negative examples
        if neg_hands:
            neg_embeddings = []
            for suits, ranks, positions, mask in neg_hands:
                h = self.hand_encoder(
                    suits.unsqueeze(0).to(device),
                    ranks.unsqueeze(0).to(device),
                    positions.unsqueeze(0).to(device),
                    mask.unsqueeze(0).to(device) if mask is not None else None
                )
                neg_embeddings.append(h.squeeze(0))
            neg_mean = torch.stack(neg_embeddings).mean(dim=0)
        else:
            neg_mean = torch.zeros(hidden_dim, device=device)

        # Sample and encode random hands
        random_hands = [sample_hand(self.hand_size) for _ in range(self.n_random_hands)]
        random_embeddings = []
        for hand in random_hands:
            encoded = self._encode_hand_raw(hand)
            suits, ranks, positions, mask = encoded
            h = self.hand_encoder(
                suits.unsqueeze(0).to(device),
                ranks.unsqueeze(0).to(device),
                positions.unsqueeze(0).to(device),
                mask.unsqueeze(0).to(device)
            )
            random_embeddings.append(h.squeeze(0))
        random_mean = torch.stack(random_embeddings).mean(dim=0)

        # Combined mean of task examples
        all_embeddings = (pos_embeddings if pos_hands else []) + (neg_embeddings if neg_hands else [])
        if all_embeddings:
            all_mean = torch.stack(all_embeddings).mean(dim=0)
        else:
            all_mean = torch.zeros(hidden_dim, device=device)

        # Contrastive encoding with random contrast
        pos_neg_diff = pos_mean - neg_mean
        random_contrast = all_mean - random_mean

        return pos_neg_diff + self.lambda_random * random_contrast


# ============================================================================
# VARIANT 5: POSITIVE VS RANDOM ONLY
# ============================================================================

class PositiveVsRandomTaskEncoder(nn.Module):
    """
    Task encoder contrasting positives against random hands.

    τ = mean(pos) - mean(random)

    This focuses on "what makes positive examples special" without
    relying on the negative examples.
    """

    def __init__(
        self,
        hand_encoder: HandEncoder,
        n_random_hands: int = 20,
        hand_size: int = 6
    ):
        super().__init__()
        self.hand_encoder = hand_encoder
        self.n_random_hands = n_random_hands
        self.hand_size = hand_size

    def _encode_hand_raw(self, hand, max_cards: int = 8):
        from rules.cards import Suit, Rank

        suit_map = {Suit.CLUBS: 0, Suit.DIAMONDS: 1, Suit.HEARTS: 2, Suit.SPADES: 3}
        rank_list = list(Rank)
        rank_map = {r: i for i, r in enumerate(rank_list)}

        suits = torch.zeros(max_cards, dtype=torch.long)
        ranks = torch.zeros(max_cards, dtype=torch.long)
        positions = torch.arange(max_cards, dtype=torch.long)
        mask = torch.zeros(max_cards, dtype=torch.bool)

        for i, card in enumerate(hand[:max_cards]):
            suits[i] = suit_map[card.suit]
            ranks[i] = rank_map[card.rank]
            mask[i] = True

        return suits, ranks, positions, mask

    def forward(
        self,
        pos_hands: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
        neg_hands: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]
    ) -> torch.Tensor:
        hidden_dim = self.hand_encoder.card_mlp.mlp[-2].out_features
        device = next(self.hand_encoder.parameters()).device

        # Encode positive examples
        if pos_hands:
            pos_embeddings = []
            for suits, ranks, positions, mask in pos_hands:
                h = self.hand_encoder(
                    suits.unsqueeze(0).to(device),
                    ranks.unsqueeze(0).to(device),
                    positions.unsqueeze(0).to(device),
                    mask.unsqueeze(0).to(device) if mask is not None else None
                )
                pos_embeddings.append(h.squeeze(0))
            pos_mean = torch.stack(pos_embeddings).mean(dim=0)
        else:
            pos_mean = torch.zeros(hidden_dim, device=device)

        # Sample and encode random hands
        random_hands = [sample_hand(self.hand_size) for _ in range(self.n_random_hands)]
        random_embeddings = []
        for hand in random_hands:
            encoded = self._encode_hand_raw(hand)
            suits, ranks, positions, mask = encoded
            h = self.hand_encoder(
                suits.unsqueeze(0).to(device),
                ranks.unsqueeze(0).to(device),
                positions.unsqueeze(0).to(device),
                mask.unsqueeze(0).to(device)
            )
            random_embeddings.append(h.squeeze(0))
        random_mean = torch.stack(random_embeddings).mean(dim=0)

        return pos_mean - random_mean


# ============================================================================
# VARIANT 6: CONCATENATED TRIPLE CONTRAST
# ============================================================================

class TripleContrastTaskEncoder(nn.Module):
    """
    Task encoder concatenating three contrast vectors.

    τ = concat[mean(pos)-mean(neg), mean(pos)-mean(random), mean(neg)-mean(random)]

    This provides the richest representation by capturing:
    1. What distinguishes positives from negatives
    2. What makes positives special vs random
    3. What makes negatives special vs random
    """

    def __init__(
        self,
        hand_encoder: HandEncoder,
        n_random_hands: int = 10,
        hand_size: int = 6
    ):
        super().__init__()
        self.hand_encoder = hand_encoder
        self.n_random_hands = n_random_hands
        self.hand_size = hand_size
        self.output_dim_multiplier = 3  # Triple the output dimension

    def _encode_hand_raw(self, hand, max_cards: int = 8):
        from rules.cards import Suit, Rank

        suit_map = {Suit.CLUBS: 0, Suit.DIAMONDS: 1, Suit.HEARTS: 2, Suit.SPADES: 3}
        rank_list = list(Rank)
        rank_map = {r: i for i, r in enumerate(rank_list)}

        suits = torch.zeros(max_cards, dtype=torch.long)
        ranks = torch.zeros(max_cards, dtype=torch.long)
        positions = torch.arange(max_cards, dtype=torch.long)
        mask = torch.zeros(max_cards, dtype=torch.bool)

        for i, card in enumerate(hand[:max_cards]):
            suits[i] = suit_map[card.suit]
            ranks[i] = rank_map[card.rank]
            mask[i] = True

        return suits, ranks, positions, mask

    def forward(
        self,
        pos_hands: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
        neg_hands: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]
    ) -> torch.Tensor:
        hidden_dim = self.hand_encoder.card_mlp.mlp[-2].out_features
        device = next(self.hand_encoder.parameters()).device

        # Encode positive examples
        if pos_hands:
            pos_embeddings = []
            for suits, ranks, positions, mask in pos_hands:
                h = self.hand_encoder(
                    suits.unsqueeze(0).to(device),
                    ranks.unsqueeze(0).to(device),
                    positions.unsqueeze(0).to(device),
                    mask.unsqueeze(0).to(device) if mask is not None else None
                )
                pos_embeddings.append(h.squeeze(0))
            pos_mean = torch.stack(pos_embeddings).mean(dim=0)
        else:
            pos_mean = torch.zeros(hidden_dim, device=device)

        # Encode negative examples
        if neg_hands:
            neg_embeddings = []
            for suits, ranks, positions, mask in neg_hands:
                h = self.hand_encoder(
                    suits.unsqueeze(0).to(device),
                    ranks.unsqueeze(0).to(device),
                    positions.unsqueeze(0).to(device),
                    mask.unsqueeze(0).to(device) if mask is not None else None
                )
                neg_embeddings.append(h.squeeze(0))
            neg_mean = torch.stack(neg_embeddings).mean(dim=0)
        else:
            neg_mean = torch.zeros(hidden_dim, device=device)

        # Sample and encode random hands
        random_hands = [sample_hand(self.hand_size) for _ in range(self.n_random_hands)]
        random_embeddings = []
        for hand in random_hands:
            encoded = self._encode_hand_raw(hand)
            suits, ranks, positions, mask = encoded
            h = self.hand_encoder(
                suits.unsqueeze(0).to(device),
                ranks.unsqueeze(0).to(device),
                positions.unsqueeze(0).to(device),
                mask.unsqueeze(0).to(device)
            )
            random_embeddings.append(h.squeeze(0))
        random_mean = torch.stack(random_embeddings).mean(dim=0)

        # Three contrast vectors
        pos_neg = pos_mean - neg_mean
        pos_random = pos_mean - random_mean
        neg_random = neg_mean - random_mean

        return torch.cat([pos_neg, pos_random, neg_random], dim=-1)


# ============================================================================
# EVALUATION FRAMEWORK
# ============================================================================

def evaluate_model(
    model: ContrastiveRecognitionModel,
    tasks: list,
    primitive_head: nn.Module,
    name: str,
    custom_encoder: Optional[nn.Module] = None,
    n_tasks: int = 10
) -> EvaluationResult:
    """Evaluate an embedding approach."""
    model.eval()

    embeddings = []
    predictions = []
    task_names = []

    with torch.no_grad():
        for task in tasks[:n_tasks]:
            # Get embedding
            if custom_encoder is not None:
                # For custom task encoders, we need to encode differently
                pos_hands = []
                neg_hands = []
                for hand, label in task.examples:
                    encoded = model._encode_hand_raw(hand)
                    if label:
                        pos_hands.append(encoded)
                    else:
                        neg_hands.append(encoded)
                emb = custom_encoder(pos_hands, neg_hands)
            else:
                emb = model.encode_task_batched(task)

            embeddings.append(emb.cpu().numpy())

            # Get prediction
            pred = primitive_head(emb.unsqueeze(0)).squeeze(0)
            predictions.append(pred.cpu().numpy())
            task_names.append(task.name)

    embeddings = np.array(embeddings)
    predictions = np.array(predictions)

    # Compute metrics
    sim_matrix = cosine_similarity(embeddings)
    upper_tri = sim_matrix[np.triu_indices(n_tasks, k=1)]

    norms = np.linalg.norm(embeddings, axis=1)

    # Top-5 diversity
    top5_sets = []
    for pred in predictions:
        top5_idx = set(np.argsort(pred)[::-1][:5])
        top5_sets.append(frozenset(top5_idx))
    unique_top5 = len(set(top5_sets))

    # Prediction spread
    pred_stds = np.std(predictions, axis=1)
    pred_spreads = np.max(predictions, axis=1) - np.min(predictions, axis=1)

    # Sample predictions
    primitives = model.primitive_names
    sample_preds = {}
    for i in range(min(3, n_tasks)):
        top3_idx = np.argsort(predictions[i])[::-1][:3]
        sample_preds[task_names[i]] = [primitives[idx] for idx in top3_idx]

    return EvaluationResult(
        name=name,
        embedding_mean_sim=float(np.mean(upper_tri)),
        embedding_std_sim=float(np.std(upper_tri)),
        embedding_mean_norm=float(np.mean(norms)),
        embedding_std_norm=float(np.std(norms)),
        prediction_unique_top5=unique_top5,
        prediction_mean_std=float(np.mean(pred_stds)),
        prediction_spread=float(np.mean(pred_spreads)),
        sample_predictions=sample_preds
    )


def run_all_tests():
    """Run all embedding fix tests."""
    print("="*80)
    print("EMBEDDING FIX SYSTEMATIC TEST")
    print("="*80)

    # Setup
    print("\nLoading grammar and tasks...")
    grammar = build_lean_grammar()
    rules = create_all_rules()
    tasks = create_tasks_from_rules(rules, n_examples=20, seed=42)
    print(f"Loaded {len(tasks)} tasks")

    # Create base model
    model = ContrastiveRecognitionModel(
        grammar=grammar,
        card_hidden=128,
        card_out=64,
        pred_hidden=128
    )

    num_primitives = model.num_primitives
    card_out = 64  # Must match model config
    pred_hidden = 128

    results = []

    # ========================================================================
    # TEST 0: BASELINE (Original)
    # ========================================================================
    print("\n" + "-"*60)
    print("TEST 0: BASELINE (Original Model)")
    print("-"*60)

    result = evaluate_model(
        model, tasks, model.primitive_head,
        "Baseline (Original)"
    )
    results.append(result)
    print(f"  Embedding similarity: {result.embedding_mean_sim:.4f} ± {result.embedding_std_sim:.4f}")
    print(f"  Embedding norm: {result.embedding_mean_norm:.4f} ± {result.embedding_std_norm:.4f}")
    print(f"  Unique top-5: {result.prediction_unique_top5}/10")
    print(f"  Prediction std: {result.prediction_mean_std:.4f}")
    print(f"  Prediction spread: {result.prediction_spread:.4f}")

    # ========================================================================
    # TEST 1: LAYER NORMALIZATION + LEARNED SCALE
    # ========================================================================
    print("\n" + "-"*60)
    print("TEST 1: LayerNorm + Learned Scale")
    print("-"*60)

    for init_scale in [5.0, 10.0, 20.0, 50.0]:
        head = NormalizedPrimitiveHead(
            input_dim=card_out,
            hidden_dim=pred_hidden,
            num_primitives=num_primitives,
            init_scale=init_scale
        )
        result = evaluate_model(
            model, tasks, head,
            f"LayerNorm (scale={init_scale})"
        )
        results.append(result)
        print(f"  Scale={init_scale}: unique top-5={result.prediction_unique_top5}/10, "
              f"std={result.prediction_mean_std:.4f}, spread={result.prediction_spread:.4f}")

    # ========================================================================
    # TEST 2: L2 NORMALIZATION + SCALE
    # ========================================================================
    print("\n" + "-"*60)
    print("TEST 2: L2 Normalization + Scale")
    print("-"*60)

    for scale in [5.0, 10.0, 20.0, 50.0]:
        head = L2NormalizedPrimitiveHead(
            input_dim=card_out,
            hidden_dim=pred_hidden,
            num_primitives=num_primitives,
            scale=scale,
            learnable_scale=False
        )
        result = evaluate_model(
            model, tasks, head,
            f"L2Norm (scale={scale})"
        )
        results.append(result)
        print(f"  Scale={scale}: unique top-5={result.prediction_unique_top5}/10, "
              f"std={result.prediction_mean_std:.4f}, spread={result.prediction_spread:.4f}")

    # ========================================================================
    # TEST 3: LARGER INITIALIZATION
    # ========================================================================
    print("\n" + "-"*60)
    print("TEST 3: Larger Weight Initialization")
    print("-"*60)

    for gain in [2.0, 5.0, 10.0, 20.0]:
        head = LargeInitPrimitiveHead(
            input_dim=card_out,
            hidden_dim=pred_hidden,
            num_primitives=num_primitives,
            init_gain=gain
        )
        result = evaluate_model(
            model, tasks, head,
            f"LargeInit (gain={gain})"
        )
        results.append(result)
        print(f"  Gain={gain}: unique top-5={result.prediction_unique_top5}/10, "
              f"std={result.prediction_mean_std:.4f}, spread={result.prediction_spread:.4f}")

    # ========================================================================
    # TEST 4: RANDOM HANDS CONTRAST
    # ========================================================================
    print("\n" + "-"*60)
    print("TEST 4: Random Hands Contrast (pos-neg + λ*(all-random))")
    print("-"*60)

    for lambda_random in [0.25, 0.5, 1.0, 2.0]:
        encoder = RandomContrastTaskEncoder(
            hand_encoder=model.hand_encoder,
            lambda_random=lambda_random,
            n_random_hands=10
        )
        # Use L2Norm head since it performed well
        head = L2NormalizedPrimitiveHead(
            input_dim=card_out,
            hidden_dim=pred_hidden,
            num_primitives=num_primitives,
            scale=10.0
        )
        result = evaluate_model(
            model, tasks, head,
            f"RandomContrast (λ={lambda_random})",
            custom_encoder=encoder
        )
        results.append(result)
        print(f"  λ={lambda_random}: unique top-5={result.prediction_unique_top5}/10, "
              f"norm={result.embedding_mean_norm:.4f}, spread={result.prediction_spread:.4f}")

    # ========================================================================
    # TEST 5: POSITIVE VS RANDOM
    # ========================================================================
    print("\n" + "-"*60)
    print("TEST 5: Positive vs Random (pos - random)")
    print("-"*60)

    for n_random in [5, 10, 20, 40]:
        encoder = PositiveVsRandomTaskEncoder(
            hand_encoder=model.hand_encoder,
            n_random_hands=n_random
        )
        head = L2NormalizedPrimitiveHead(
            input_dim=card_out,
            hidden_dim=pred_hidden,
            num_primitives=num_primitives,
            scale=10.0
        )
        result = evaluate_model(
            model, tasks, head,
            f"PosVsRandom (n={n_random})",
            custom_encoder=encoder
        )
        results.append(result)
        print(f"  n_random={n_random}: unique top-5={result.prediction_unique_top5}/10, "
              f"norm={result.embedding_mean_norm:.4f}, spread={result.prediction_spread:.4f}")

    # ========================================================================
    # TEST 6: TRIPLE CONTRAST (CONCATENATED)
    # ========================================================================
    print("\n" + "-"*60)
    print("TEST 6: Triple Contrast (concat[pos-neg, pos-rand, neg-rand])")
    print("-"*60)

    encoder = TripleContrastTaskEncoder(
        hand_encoder=model.hand_encoder,
        n_random_hands=10
    )
    # Triple the input dimension
    head = L2NormalizedPrimitiveHead(
        input_dim=card_out * 3,  # Triple dimension
        hidden_dim=pred_hidden,
        num_primitives=num_primitives,
        scale=10.0
    )
    result = evaluate_model(
        model, tasks, head,
        "TripleContrast",
        custom_encoder=encoder
    )
    results.append(result)
    print(f"  Triple contrast: unique top-5={result.prediction_unique_top5}/10, "
          f"norm={result.embedding_mean_norm:.4f}, spread={result.prediction_spread:.4f}")

    # ========================================================================
    # TEST 7: COMBINATIONS
    # ========================================================================
    print("\n" + "-"*60)
    print("TEST 7: Best Combinations")
    print("-"*60)

    # RandomContrast + LayerNorm
    encoder = RandomContrastTaskEncoder(
        hand_encoder=model.hand_encoder,
        lambda_random=0.5,
        n_random_hands=10
    )
    head = NormalizedPrimitiveHead(
        input_dim=card_out,
        hidden_dim=pred_hidden,
        num_primitives=num_primitives,
        init_scale=20.0
    )
    result = evaluate_model(
        model, tasks, head,
        "RandomContrast + LayerNorm",
        custom_encoder=encoder
    )
    results.append(result)
    print(f"  RandomContrast + LayerNorm: unique top-5={result.prediction_unique_top5}/10, "
          f"spread={result.prediction_spread:.4f}")

    # TripleContrast + LayerNorm
    encoder = TripleContrastTaskEncoder(
        hand_encoder=model.hand_encoder,
        n_random_hands=10
    )
    head = NormalizedPrimitiveHead(
        input_dim=card_out * 3,
        hidden_dim=pred_hidden,
        num_primitives=num_primitives,
        init_scale=20.0
    )
    result = evaluate_model(
        model, tasks, head,
        "TripleContrast + LayerNorm",
        custom_encoder=encoder
    )
    results.append(result)
    print(f"  TripleContrast + LayerNorm: unique top-5={result.prediction_unique_top5}/10, "
          f"spread={result.prediction_spread:.4f}")

    # ========================================================================
    # SUMMARY REPORT
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY REPORT")
    print("="*80)

    # Sort by prediction diversity
    sorted_results = sorted(results, key=lambda r: (r.prediction_unique_top5, r.prediction_spread), reverse=True)

    print("\n" + "-"*80)
    print(f"{'Approach':<40} {'Top5':<8} {'Spread':<10} {'Norm':<10}")
    print("-"*80)

    for r in sorted_results:
        print(f"{r.name:<40} {r.prediction_unique_top5:<8} {r.prediction_spread:<10.4f} {r.embedding_mean_norm:<10.4f}")

    # Best approaches
    print("\n" + "="*80)
    print("TOP 5 APPROACHES")
    print("="*80)

    for i, r in enumerate(sorted_results[:5], 1):
        print(f"\n{i}. {r.name}")
        print(f"   Unique top-5: {r.prediction_unique_top5}/10")
        print(f"   Prediction spread: {r.prediction_spread:.4f}")
        print(f"   Embedding norm: {r.embedding_mean_norm:.4f}")
        print(f"   Sample predictions:")
        for task_name, prims in r.sample_predictions.items():
            print(f"      {task_name[:25]}: {prims}")

    return results


if __name__ == "__main__":
    results = run_all_tests()
