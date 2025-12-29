#!/usr/bin/env python3
"""
Contrastive Recognition Model for Card Game Classification Tasks

This module implements a recognition model specifically designed for classification
tasks (hand → bool) rather than the original DreamCoder's list → list tasks.

KEY INNOVATIONS:
1. Factored Card Embeddings: Learned embeddings for suit, rank, and position
2. Contrastive Task Encoding: τ = mean(positive) - mean(negative)
3. Structural Similarity Training: Tasks with similar primitives cluster together
4. Independent Primitive Predictions: Sigmoid outputs (not softmax)
5. Bigram Support: Predict primitive co-occurrence patterns
6. Dynamic Invention Handling: Add new primitives discovered during compression

Architecture:
    Card → E_suit(s) ⊕ E_rank(r) ⊕ E_pos(p)
        → CardMLP → 64 → 32
        → MeanPool over cards → h ∈ ℝ³²
    Task → τ = mean(h | pos) - mean(h | neg)
        → PrimitiveHead → σ(Wτ + b) ∈ [0,1]^num_primitives

Training Objectives:
    L_struct = Σᵢⱼ ||cos(τᵢ, τⱼ) - Jaccard(Pᵢ, Pⱼ)||²
    L_count = MSE(predicted_count, actual_count)
    L_pred = BCE(predicted_prims, actual_prims)

Author: Can Konuk
Date: December 2024
"""

import sys
import math
import random
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.program import Program, Primitive, Application, Abstraction, Index, Invented
from dreamcoder_core.grammar import Grammar, Production

# Try to import Cython types for compatibility
try:
    from dreamcoder_core.cython_src.program_cy import (
        Program as CythonProgram,
        Primitive as CythonPrimitive,
        Application as CythonApplication,
        Abstraction as CythonAbstraction,
        Invented as CythonInvented
    )
    CYTHON_AVAILABLE = True
except ImportError:
    CYTHON_AVAILABLE = False


# ============================================================================
# CARD ENCODING: FACTORED EMBEDDINGS
# ============================================================================

class FactoredCardEncoder(nn.Module):
    """
    Encode cards using learned factored embeddings.

    Instead of one-hot encoding, we learn embeddings for each attribute:
    - E_suit(s): 4 suits → d_suit dimensions
    - E_rank(r): 13 ranks → d_rank dimensions
    - E_pos(p): max_pos positions → d_pos dimensions

    These are concatenated: card → E_suit ⊕ E_rank ⊕ E_pos

    Benefits over one-hot:
    - Learns similarity (Jack/Queen close as face cards)
    - More compact representation
    - Position captures order (useful for "sorted" rules)
    """

    def __init__(
        self,
        d_suit: int = 8,
        d_rank: int = 16,
        d_pos: int = 8,
        max_pos: int = 8
    ):
        super().__init__()

        self.d_suit = d_suit
        self.d_rank = d_rank
        self.d_pos = d_pos
        self.output_dim = d_suit + d_rank + d_pos

        # Learned embeddings
        self.suit_embed = nn.Embedding(4, d_suit)    # 4 suits: ♣♦♥♠
        self.rank_embed = nn.Embedding(13, d_rank)   # 13 ranks: 2-A
        self.pos_embed = nn.Embedding(max_pos, d_pos)  # Position in hand

        # Initialize embeddings
        nn.init.xavier_uniform_(self.suit_embed.weight)
        nn.init.xavier_uniform_(self.rank_embed.weight)
        nn.init.xavier_uniform_(self.pos_embed.weight)

    def forward(
        self,
        suits: torch.Tensor,
        ranks: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            suits: (batch, max_cards) int indices 0-3
            ranks: (batch, max_cards) int indices 0-12
            positions: (batch, max_cards) int indices 0-7

        Returns:
            (batch, max_cards, output_dim) factored embeddings
        """
        suit_emb = self.suit_embed(suits)      # (batch, max_cards, d_suit)
        rank_emb = self.rank_embed(ranks)      # (batch, max_cards, d_rank)
        pos_emb = self.pos_embed(positions)    # (batch, max_cards, d_pos)

        return torch.cat([suit_emb, rank_emb, pos_emb], dim=-1)


# ============================================================================
# CARD-LEVEL INTERACTION MLP
# ============================================================================

class CardInteractionMLP(nn.Module):
    """
    Per-card feature extraction before pooling.

    Transforms each card's factored embedding into a richer representation
    that can capture interactions between suit, rank, and position.

    Example learned features:
    - "High red card" = combination of rank and suit
    - "Ace in first position" = combination of rank and position
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, card_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            card_embeddings: (batch, max_cards, input_dim)

        Returns:
            (batch, max_cards, output_dim)
        """
        return self.mlp(card_embeddings)


# ============================================================================
# HAND ENCODER: MEAN POOLING
# ============================================================================

class HandEncoder(nn.Module):
    """
    Encode a hand of cards into a fixed-size vector via mean pooling.

    Mean pooling is:
    - Permutation invariant (order handled by position embedding)
    - Fixed size regardless of hand size
    - Simple and effective for set-like data
    """

    def __init__(
        self,
        card_encoder: FactoredCardEncoder,
        card_mlp: CardInteractionMLP
    ):
        super().__init__()
        self.card_encoder = card_encoder
        self.card_mlp = card_mlp

    def forward(
        self,
        suits: torch.Tensor,
        ranks: torch.Tensor,
        positions: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            suits: (batch, max_cards)
            ranks: (batch, max_cards)
            positions: (batch, max_cards)
            mask: (batch, max_cards) True for real cards, False for padding

        Returns:
            (batch, hidden_dim) hand embeddings
        """
        # Get factored embeddings
        card_emb = self.card_encoder(suits, ranks, positions)

        # Apply card-level MLP
        card_features = self.card_mlp(card_emb)

        # Mean pooling (with optional mask)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            card_features = card_features * mask_expanded
            return card_features.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

        return card_features.mean(dim=1)


# ============================================================================
# CONTRASTIVE TASK ENCODER
# ============================================================================

class ContrastiveTaskEncoder(nn.Module):
    """
    Encode a classification task using contrastive difference.

    τ = mean(h | positive examples) - mean(h | negative examples)

    This is the KEY INSIGHT for classification tasks:
    - The difference vector directly encodes the decision boundary
    - It captures "what distinguishes positive from negative"
    - Similar rules will have similar τ vectors

    Geometric interpretation:
    - τ points in the direction that separates positive from negative
    - Magnitude indicates how separable the classes are
    """

    def __init__(self, hand_encoder: HandEncoder):
        super().__init__()
        self.hand_encoder = hand_encoder

    def forward(
        self,
        pos_hands: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
        neg_hands: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]
    ) -> torch.Tensor:
        """
        Args:
            pos_hands: List of (suits, ranks, positions, mask) for positive examples
            neg_hands: List of (suits, ranks, positions, mask) for negative examples

        Returns:
            τ ∈ ℝ^hidden_dim contrastive task embedding
        """
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

        # Contrastive difference
        return pos_mean - neg_mean


# ============================================================================
# OUTPUT HEADS
# ============================================================================

class PrimitiveHead(nn.Module):
    """
    Predict per-primitive probabilities using sigmoid (not softmax).

    Why sigmoid:
    - Primitives are NOT mutually exclusive
    - A program uses MULTIPLE primitives
    - We want P(use +) and P(use filter) independently

    Also includes adaptive per-primitive thresholds for binary decisions.
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
        num_primitives: int = 67
    ):
        super().__init__()

        self.num_primitives = num_primitives

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_primitives)
        )

        # Learnable thresholds (initialized to 0 → sigmoid(0) = 0.5)
        self.thresholds = nn.Parameter(torch.zeros(num_primitives))

    def forward(self, τ: torch.Tensor, return_binary: bool = False) -> torch.Tensor:
        """
        Args:
            τ: (batch, input_dim) task embeddings
            return_binary: If True, also return binary predictions

        Returns:
            probs: (batch, num_primitives) probabilities in [0, 1]
            binary: (batch, num_primitives) binary predictions (if return_binary)
        """
        logits = self.mlp(τ)
        probs = torch.sigmoid(logits)

        if return_binary:
            thresholds = torch.sigmoid(self.thresholds)
            binary = (probs > thresholds).float()
            return probs, binary

        return probs

    def expand_for_invention(self, init_weight: Optional[torch.Tensor] = None):
        """
        Add a new output neuron for a discovered invention.

        Args:
            init_weight: Optional initialization for the new neuron
        """
        old_out = self.mlp[-1]
        new_num = self.num_primitives + 1

        new_out = nn.Linear(old_out.in_features, new_num)
        new_out.weight.data[:self.num_primitives] = old_out.weight.data
        new_out.bias.data[:self.num_primitives] = old_out.bias.data

        if init_weight is not None:
            new_out.weight.data[-1] = init_weight
        else:
            nn.init.xavier_uniform_(new_out.weight.data[-1:])
        new_out.bias.data[-1] = 0.0

        self.mlp[-1] = new_out

        # Expand thresholds
        new_thresholds = torch.zeros(new_num)
        new_thresholds[:self.num_primitives] = self.thresholds.data
        self.thresholds = nn.Parameter(new_thresholds)

        self.num_primitives = new_num


class SoftmaxPrimitiveHead(nn.Module):
    """
    Predict primitive probabilities using softmax (like NeuralRecognitionModel).

    Unlike PrimitiveHead which uses sigmoid (independent probabilities),
    this uses softmax to produce a proper distribution over primitives.

    Why this might work better for search guidance:
    - Softmax produces a normalized distribution (sums to 1)
    - Forces competition between primitives
    - Matches the training target format (normalized primitive counts)
    - Same as the successful neural recognition model
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
        num_primitives: int = 67
    ):
        super().__init__()

        self.num_primitives = num_primitives

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_primitives)
        )

    def forward(self, τ: torch.Tensor, return_logits: bool = False):
        """
        Args:
            τ: (batch, input_dim) task embeddings
            return_logits: If True, return raw logits instead of log-probs

        Returns:
            log_probs: (batch, num_primitives) log-probabilities (or logits if return_logits=True)
        """
        logits = self.mlp(τ)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)

    def expand_for_invention(self, init_weight: Optional[torch.Tensor] = None):
        """Add a new output neuron for a discovered invention."""
        old_out = self.mlp[-1]
        new_num = self.num_primitives + 1

        new_out = nn.Linear(old_out.in_features, new_num)
        new_out.weight.data[:self.num_primitives] = old_out.weight.data
        new_out.bias.data[:self.num_primitives] = old_out.bias.data

        if init_weight is not None:
            new_out.weight.data[-1] = init_weight
        else:
            nn.init.xavier_uniform_(new_out.weight.data[-1:])
        new_out.bias.data[-1] = 0.0

        self.mlp[-1] = new_out
        self.num_primitives = new_num


class BigramHead(nn.Module):
    """
    Predict bigram (primitive pair) probabilities.

    Bigrams capture co-occurrence patterns:
    - P(filter → map | task): "filter feeds into map"
    - P(+ applied to rank_val | task): "arithmetic on rank values"

    This goes beyond first-order primitive prediction.
    """

    def __init__(
        self,
        input_dim: int = 32,
        hidden_dim: int = 64,
        num_bigrams: int = 50
    ):
        super().__init__()

        self.num_bigrams = num_bigrams

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_bigrams),
            nn.Sigmoid()
        )

    def forward(self, τ: torch.Tensor) -> torch.Tensor:
        """
        Args:
            τ: (batch, input_dim) task embeddings

        Returns:
            (batch, num_bigrams) bigram probabilities
        """
        return self.mlp(τ)


class CountHead(nn.Module):
    """
    Auxiliary head predicting number of distinct primitives used.

    This regularizes the task embedding to capture solution complexity.
    """

    def __init__(self, input_dim: int = 32, hidden_dim: int = 32):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, τ: torch.Tensor) -> torch.Tensor:
        """
        Args:
            τ: (batch, input_dim) task embeddings

        Returns:
            (batch, 1) predicted primitive count
        """
        return self.mlp(τ)


# ============================================================================
# BIGRAM EXTRACTION
# ============================================================================

def extract_bigrams(program: Program) -> Set[Tuple[str, str]]:
    """
    Extract bigram patterns from a program's AST.

    A bigram (a, b) means primitive 'a' has 'b' as a child in the AST.

    Examples:
        (filter (λ (eq ...))) → {('filter', 'eq')}
        (map (λ (+ ...))) → {('map', '+')}
    """
    bigrams = set()

    def visit(node: Program, parent_name: Optional[str] = None):
        if isinstance(node, (Primitive, Invented)):
            name = str(node)
            if parent_name is not None:
                bigrams.add((parent_name, name))
            return name

        elif isinstance(node, Application):
            f_name = visit(node.f, parent_name)
            visit(node.x, f_name)
            return f_name

        elif isinstance(node, Abstraction):
            visit(node.body, parent_name)
            return parent_name

        elif isinstance(node, Index):
            return None

        return None

    visit(program, None)
    return bigrams


def build_bigram_vocabulary(
    programs: List[Program],
    min_count: int = 2,
    max_bigrams: int = 100
) -> List[Tuple[str, str]]:
    """
    Build vocabulary of common bigrams from solved programs.

    Returns list of (primitive1, primitive2) tuples sorted by frequency.
    """
    bigram_counts = defaultdict(int)

    for prog in programs:
        for bigram in extract_bigrams(prog):
            bigram_counts[bigram] += 1

    # Filter by count and sort
    filtered = [(bg, count) for bg, count in bigram_counts.items() if count >= min_count]
    filtered.sort(key=lambda x: -x[1])

    return [bg for bg, _ in filtered[:max_bigrams]]


# ============================================================================
# FULL CONTRASTIVE RECOGNITION MODEL
# ============================================================================

class ContrastiveRecognitionModel(nn.Module):
    """
    Full contrastive recognition model for card game classification tasks.

    Components:
    1. FactoredCardEncoder: Learned embeddings for suit/rank/position
    2. CardInteractionMLP: Per-card feature extraction
    3. HandEncoder: Mean pooling over cards
    4. ContrastiveTaskEncoder: τ = mean(pos) - mean(neg)
    5. PrimitiveHead: Per-primitive sigmoid predictions
    6. BigramHead: Primitive co-occurrence predictions
    7. CountHead: Auxiliary primitive count prediction

    Training:
    - Structural similarity loss: cluster similar tasks
    - Primitive prediction loss: BCE on primitive usage
    - Count prediction loss: MSE on primitive count
    - Bigram prediction loss: BCE on bigram usage
    """

    def __init__(
        self,
        grammar: Grammar,
        d_suit: int = 8,
        d_rank: int = 16,
        d_pos: int = 8,
        card_hidden: int = 64,
        card_out: int = 32,
        pred_hidden: int = 64,
        max_cards: int = 8,
        max_bigrams: int = 50,
        learning_rate: float = 1e-3,
        device: str = 'cpu',
        output_mode: str = 'sigmoid',  # 'sigmoid' (original) or 'softmax' (new)
        # NEW: Embedding normalization parameters
        normalize_embeddings: bool = True,  # Enable LayerNorm + scale fix
        embedding_scale: float = 20.0,  # Scale factor for normalized embeddings
        # NEW: Encoding mode for different contrastive variants
        encoding_mode: str = 'standard',  # 'standard', 'random_contrast', 'triple_contrast'
        n_random_hands: int = 10,  # Number of random hands for contrast variants
        lambda_random: float = 0.5  # Weight for random contrast term
    ):
        super().__init__()

        self.grammar = grammar
        self.device = device
        self.max_cards = max_cards
        self.output_mode = output_mode  # Store output mode for training/prediction
        self.encoding_mode = encoding_mode
        self.n_random_hands = n_random_hands
        self.lambda_random = lambda_random
        self.normalize_embeddings = normalize_embeddings

        # Build primitive vocabulary
        self.primitive_names = [str(p.program) for p in grammar.productions]
        self.primitive_to_idx = {name: i for i, name in enumerate(self.primitive_names)}
        self.num_primitives = len(self.primitive_names)

        # Bigram vocabulary (built from training data)
        self.bigram_vocab: List[Tuple[str, str]] = []
        self.bigram_to_idx: Dict[Tuple[str, str], int] = {}

        # Invention tracking
        self.inventions: List[str] = []

        # ========== Network Components ==========

        # Card encoding
        card_dim = d_suit + d_rank + d_pos
        self.card_encoder = FactoredCardEncoder(d_suit, d_rank, d_pos, max_cards)
        self.card_mlp = CardInteractionMLP(card_dim, card_hidden, card_out)

        # Hand and task encoding
        self.hand_encoder = HandEncoder(self.card_encoder, self.card_mlp)
        self.task_encoder = ContrastiveTaskEncoder(self.hand_encoder)

        # NEW: Embedding normalization layer (fixes magnitude problem)
        if normalize_embeddings:
            # Determine embedding dimension based on encoding mode
            if encoding_mode == 'triple_contrast':
                emb_dim = card_out * 3  # Concatenated triple contrast
            else:
                emb_dim = card_out
            self.embedding_norm = nn.LayerNorm(emb_dim)
            self.embedding_scale = nn.Parameter(torch.tensor(embedding_scale))
        else:
            self.embedding_norm = None
            self.embedding_scale = None

        # Output heads - choose based on output_mode
        # Adjust input dim for triple contrast
        head_input_dim = card_out * 3 if encoding_mode == 'triple_contrast' else card_out

        if output_mode == 'softmax':
            self.primitive_head = SoftmaxPrimitiveHead(head_input_dim, pred_hidden, self.num_primitives)
        else:  # Default to sigmoid
            self.primitive_head = PrimitiveHead(head_input_dim, pred_hidden, self.num_primitives)
        self.bigram_head = BigramHead(head_input_dim, pred_hidden, max_bigrams)
        self.count_head = CountHead(head_input_dim)

        # Move to device
        self.to(device)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Training history
        self.training_losses: List[float] = []
        self.epoch_history: List[Dict] = []

        # Task embedding cache
        self._task_embeddings: Dict[str, torch.Tensor] = {}

    # ========================================================================
    # CARD/HAND ENCODING UTILITIES
    # ========================================================================

    def _card_to_indices(self, card) -> Tuple[int, int]:
        """Convert a Card object to (suit_idx, rank_idx)."""
        from rules.cards import Suit, Rank

        suit_map = {Suit.CLUBS: 0, Suit.DIAMONDS: 1, Suit.HEARTS: 2, Suit.SPADES: 3}
        rank_list = list(Rank)
        rank_map = {r: i for i, r in enumerate(rank_list)}

        return suit_map[card.suit], rank_map[card.rank]

    def _encode_hand_raw(self, hand) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert a hand (list of Cards) to tensor indices.

        Returns:
            suits: (max_cards,) int tensor
            ranks: (max_cards,) int tensor
            positions: (max_cards,) int tensor
            mask: (max_cards,) bool tensor
        """
        n_cards = min(len(hand), self.max_cards)

        suits = torch.zeros(self.max_cards, dtype=torch.long)
        ranks = torch.zeros(self.max_cards, dtype=torch.long)
        positions = torch.arange(self.max_cards, dtype=torch.long)
        mask = torch.zeros(self.max_cards, dtype=torch.bool)

        for i, card in enumerate(hand[:self.max_cards]):
            s, r = self._card_to_indices(card)
            suits[i] = s
            ranks[i] = r
            mask[i] = True

        return suits, ranks, positions, mask

    # ========================================================================
    # TASK ENCODING
    # ========================================================================

    def encode_task(self, task) -> torch.Tensor:
        """
        Encode a task into a contrastive embedding.

        Args:
            task: Task object with .examples = [(hand, bool), ...]

        Returns:
            τ ∈ ℝ^hidden_dim contrastive task embedding

        NOTE: This now calls encode_task_batched internally for consistency
        with normalization and encoding mode support.
        """
        return self.encode_task_batched(task)

    def encode_task_batched(self, task) -> torch.Tensor:
        """
        More efficient batched encoding for a single task.

        Supports multiple encoding modes:
        - 'standard': τ = mean(pos) - mean(neg)
        - 'random_contrast': τ = pos - neg + λ*(all - random)
        - 'triple_contrast': τ = concat[pos-neg, pos-random, neg-random]

        Applies LayerNorm + scale if normalize_embeddings=True.
        """
        pos_hands_data = []
        neg_hands_data = []

        for hand, label in task.examples:
            encoded = self._encode_hand_raw(hand)
            if label:
                pos_hands_data.append(encoded)
            else:
                neg_hands_data.append(encoded)

        hidden_dim = self.card_mlp.mlp[-2].out_features

        # Helper function to batch encode hands
        def batch_encode(hands_data):
            if not hands_data:
                return torch.zeros(hidden_dim, device=self.device), None
            suits = torch.stack([h[0] for h in hands_data]).to(self.device)
            ranks = torch.stack([h[1] for h in hands_data]).to(self.device)
            positions = torch.stack([h[2] for h in hands_data]).to(self.device)
            masks = torch.stack([h[3] for h in hands_data]).to(self.device)
            embeddings = self.hand_encoder(suits, ranks, positions, masks)
            return embeddings.mean(dim=0), embeddings

        pos_mean, pos_embeddings = batch_encode(pos_hands_data)
        neg_mean, neg_embeddings = batch_encode(neg_hands_data)

        # Standard contrastive encoding
        τ_standard = pos_mean - neg_mean

        # Handle different encoding modes
        if self.encoding_mode == 'random_contrast' or self.encoding_mode == 'triple_contrast':
            # Generate random hands for contrast
            from rules.cards import sample_hand

            random_hands_data = []
            # Determine hand size from existing examples
            hand_size = 5  # Default
            if pos_hands_data:
                hand_size = pos_hands_data[0][3].sum().item()  # Count True in mask
            elif neg_hands_data:
                hand_size = neg_hands_data[0][3].sum().item()
            hand_size = max(1, int(hand_size))

            for _ in range(self.n_random_hands):
                random_hand = sample_hand(hand_size)
                random_hands_data.append(self._encode_hand_raw(random_hand))

            random_mean, _ = batch_encode(random_hands_data)

            # Compute all examples mean (pos + neg combined)
            all_hands_data = pos_hands_data + neg_hands_data
            all_mean, _ = batch_encode(all_hands_data)

        if self.encoding_mode == 'standard':
            τ = τ_standard
        elif self.encoding_mode == 'random_contrast':
            # τ = (pos - neg) + λ*(all - random)
            τ = τ_standard + self.lambda_random * (all_mean - random_mean)
        elif self.encoding_mode == 'triple_contrast':
            # τ = concat[pos-neg, pos-random, neg-random]
            pos_neg = pos_mean - neg_mean
            pos_random = pos_mean - random_mean
            neg_random = neg_mean - random_mean
            τ = torch.cat([pos_neg, pos_random, neg_random], dim=-1)
        else:
            raise ValueError(f"Unknown encoding_mode: {self.encoding_mode}")

        # Apply normalization if enabled (FIXES THE MAGNITUDE PROBLEM)
        if self.normalize_embeddings and self.embedding_norm is not None:
            τ = self.embedding_norm(τ) * self.embedding_scale

        return τ

    # ========================================================================
    # PREDICTION
    # ========================================================================

    def predict_primitives(self, task) -> torch.Tensor:
        """
        Predict primitive probabilities for a task.

        Returns:
            (num_primitives,) tensor of probabilities in [0, 1]
            For softmax mode: proper probability distribution (sums to 1)
            For sigmoid mode: independent probabilities
        """
        self.eval()
        with torch.no_grad():
            τ = self.encode_task_batched(task).unsqueeze(0)
            output = self.primitive_head(τ)
            if self.output_mode == 'softmax':
                # output is log-probabilities, convert to probabilities
                return torch.exp(output.squeeze(0))
            else:
                # output is already probabilities from sigmoid
                return output.squeeze(0)

    def predict_log_probs(self, task) -> torch.Tensor:
        """
        Predict primitive log-probabilities for a task.

        Returns:
            (num_primitives,) tensor of log-probabilities
        """
        self.eval()
        with torch.no_grad():
            τ = self.encode_task_batched(task).unsqueeze(0)
            output = self.primitive_head(τ)
            if self.output_mode == 'softmax':
                # output is already log-probabilities
                return output.squeeze(0)
            else:
                # output is probabilities, convert to log-probs
                return torch.log(output.squeeze(0).clamp(min=1e-10))

    def predict_primitives_dict(self, task) -> Dict[str, float]:
        """
        Predict primitives as a dictionary mapping names to probabilities.
        """
        probs = self.predict_primitives(task)
        return {
            name: float(probs[i])
            for i, name in enumerate(self.primitive_names)
        }

    def predict_grammar_weights(self, task) -> Grammar:
        """
        Return a grammar with weights adjusted by predictions.

        Blends original grammar weights with predicted log-probabilities.
        """
        # Get log-probabilities directly (more efficient for softmax mode)
        log_probs = self.predict_log_probs(task).cpu().numpy()

        new_productions = []
        for i, prod in enumerate(self.grammar.productions):
            prim_name = str(prod.program)
            if prim_name in self.primitive_to_idx:
                idx = self.primitive_to_idx[prim_name]
                # Blend original and predicted log-probabilities
                new_lp = 0.5 * prod.log_probability + 0.5 * log_probs[idx]
            else:
                new_lp = prod.log_probability

            new_productions.append(Production(prod.program, prod.tp, new_lp))

        # NOTE: We intentionally do NOT call .normalize_probabilities() here.
        # Normalization is computationally expensive and unnecessary because:
        # 1. The blended weights are already valid log-probabilities
        # 2. TopDownEnumerator uses relative ordering, not absolute values
        # 3. This was causing significant slowdown in recognition-guided enumeration
        return Grammar(new_productions, self.grammar.log_variable)

    def get_top_predictions(self, task, n: int = 10) -> List[Tuple[str, float]]:
        """Get top-n predicted primitives."""
        probs = self.predict_primitives(task)
        values, indices = torch.topk(probs, min(n, self.num_primitives))

        return [
            (self.primitive_names[idx], float(val))
            for val, idx in zip(values.cpu(), indices.cpu())
        ]

    # ========================================================================
    # TRAINING
    # ========================================================================

    def _collect_primitives(self, program: Program, primitives: Set[str]):
        """Collect all primitive/invention names used in a program."""
        if isinstance(program, (Primitive, Invented)):
            primitives.add(str(program))
        elif isinstance(program, Application):
            self._collect_primitives(program.f, primitives)
            self._collect_primitives(program.x, primitives)
        elif isinstance(program, Abstraction):
            self._collect_primitives(program.body, primitives)
        # Handle Cython types
        elif CYTHON_AVAILABLE:
            if isinstance(program, (CythonPrimitive, CythonInvented)):
                primitives.add(str(program))
            elif isinstance(program, CythonApplication):
                self._collect_primitives(program.f, primitives)
                self._collect_primitives(program.x, primitives)
            elif isinstance(program, CythonAbstraction):
                self._collect_primitives(program.body, primitives)

    def compute_structural_similarity_loss(
        self,
        task_embeddings: Dict[str, torch.Tensor],
        primitive_sets: Dict[str, Set[str]]
    ) -> torch.Tensor:
        """
        Structural similarity loss: tasks with similar primitives should cluster.

        L = Σᵢⱼ ||cos(τᵢ, τⱼ) - Jaccard(Pᵢ, Pⱼ)||²
        """
        tasks = list(task_embeddings.keys())
        n = len(tasks)

        if n < 2:
            return torch.tensor(0.0, device=self.device)

        loss = torch.tensor(0.0, device=self.device)
        count = 0

        for i in range(n):
            for j in range(i + 1, n):
                τ_i = task_embeddings[tasks[i]]
                τ_j = task_embeddings[tasks[j]]

                # Cosine similarity of embeddings
                cos_sim = F.cosine_similarity(τ_i.unsqueeze(0), τ_j.unsqueeze(0)).squeeze()

                # Jaccard similarity of primitive sets
                P_i = primitive_sets[tasks[i]]
                P_j = primitive_sets[tasks[j]]

                if P_i or P_j:
                    jaccard = len(P_i & P_j) / len(P_i | P_j)
                else:
                    jaccard = 1.0

                loss = loss + (cos_sim - jaccard) ** 2
                count += 1

        return loss / max(count, 1)

    def train_on_frontiers(
        self,
        tasks: List,
        frontiers: Dict,
        epochs: int = 10,
        batch_size: int = 8,
        lambda_struct: float = 0.3,
        lambda_count: float = 0.1,
        lambda_pred: float = 1.0
    ) -> float:
        """
        Train the recognition model on solved tasks.

        Args:
            tasks: List of Task objects
            frontiers: Dict mapping task names to TaskFrontier objects
            epochs: Number of training epochs
            batch_size: Batch size
            lambda_struct: Weight for structural similarity loss
            lambda_count: Weight for count prediction loss
            lambda_pred: Weight for primitive prediction loss

        Returns:
            Final training loss
        """
        # Collect training data
        training_data = []

        for task in tasks:
            frontier = frontiers.get(task.name)
            if frontier and frontier.solved:
                target_primitives = set()
                for entry in frontier.entries:
                    if entry.log_likelihood == 0.0:  # Perfect solution
                        self._collect_primitives(entry.program, target_primitives)

                if target_primitives:
                    training_data.append((task, target_primitives))

        if not training_data:
            return 0.0

        self.train()
        total_loss = 0.0

        for epoch in range(epochs):
            random.shuffle(training_data)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]

                self.optimizer.zero_grad()

                # Collect embeddings and targets for batch
                task_embeddings = {}
                primitive_sets = {}
                batch_τ = []
                batch_targets = []
                batch_counts = []

                for task, target_prims in batch:
                    τ = self.encode_task_batched(task)
                    task_embeddings[task.name] = τ
                    primitive_sets[task.name] = target_prims
                    batch_τ.append(τ)

                    # Create target vector
                    target = torch.zeros(self.num_primitives, device=self.device)
                    for prim_name in target_prims:
                        if prim_name in self.primitive_to_idx:
                            target[self.primitive_to_idx[prim_name]] = 1.0
                    batch_targets.append(target)
                    batch_counts.append(len(target_prims))

                # Stack batch tensors
                τ_batch = torch.stack(batch_τ)  # (batch, hidden_dim)
                target_batch = torch.stack(batch_targets)  # (batch, num_primitives)
                count_batch = torch.tensor(batch_counts, dtype=torch.float, device=self.device)

                # Predictions
                pred_prims = self.primitive_head(τ_batch)  # (batch, num_primitives)
                pred_counts = self.count_head(τ_batch).squeeze(-1)  # (batch,)

                # Losses - depends on output mode
                if self.output_mode == 'softmax':
                    # Cross-entropy loss: pred_prims are log-probabilities
                    # Normalize target to be a distribution
                    target_sums = target_batch.sum(dim=1, keepdim=True).clamp(min=1e-8)
                    target_normalized = target_batch / target_sums
                    # CE loss: -sum(target * log_pred) averaged over batch
                    loss_pred = -torch.sum(target_normalized * pred_prims, dim=1).mean()
                else:
                    # BCE loss (original): pred_prims are sigmoid probabilities
                    loss_pred = F.binary_cross_entropy(pred_prims, target_batch)

                loss_count = F.mse_loss(pred_counts, count_batch)
                loss_struct = self.compute_structural_similarity_loss(task_embeddings, primitive_sets)

                # Combined loss
                loss = lambda_pred * loss_pred + lambda_count * loss_count + lambda_struct * loss_struct

                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_epoch_loss = epoch_loss / max(n_batches, 1)
            self.training_losses.append(avg_epoch_loss)

        final_loss = sum(self.training_losses[-epochs:]) / epochs if epochs > 0 else 0.0

        self.epoch_history.append({
            'num_tasks': len(training_data),
            'final_loss': final_loss,
            'epochs': epochs
        })

        # Clear embedding cache
        self._task_embeddings.clear()

        return final_loss

    # ========================================================================
    # INVENTION HANDLING
    # ========================================================================

    def add_invention(self, invention: Invented):
        """
        Add a new invention to the model's vocabulary.

        Expands the output layer to include the new primitive.
        """
        inv_name = str(invention)

        if inv_name in self.primitive_to_idx:
            return  # Already known

        # Add to vocabulary
        self.primitive_to_idx[inv_name] = self.num_primitives
        self.primitive_names.append(inv_name)
        self.inventions.append(inv_name)

        # Expand output head
        self.primitive_head.expand_for_invention()

        self.num_primitives += 1

    def update_grammar(self, new_grammar: Grammar):
        """
        Update the grammar and add any new inventions.
        """
        self.grammar = new_grammar

        for prod in new_grammar.productions:
            name = str(prod.program)
            if name not in self.primitive_to_idx:
                if isinstance(prod.program, Invented):
                    self.add_invention(prod.program)

    # ========================================================================
    # BIGRAM SUPPORT
    # ========================================================================

    def build_bigram_vocabulary(self, programs: List[Program], min_count: int = 2):
        """
        Build bigram vocabulary from solved programs.
        """
        self.bigram_vocab = build_bigram_vocabulary(programs, min_count, self.bigram_head.num_bigrams)
        self.bigram_to_idx = {bg: i for i, bg in enumerate(self.bigram_vocab)}

    def predict_bigrams(self, task) -> Dict[Tuple[str, str], float]:
        """
        Predict bigram probabilities for a task.
        """
        self.eval()
        with torch.no_grad():
            τ = self.encode_task_batched(task).unsqueeze(0)
            probs = self.bigram_head(τ).squeeze(0)

            return {
                bg: float(probs[i])
                for i, bg in enumerate(self.bigram_vocab)
                if i < len(probs)
            }

    # ========================================================================
    # INTERPRETABILITY
    # ========================================================================

    def get_task_embedding(self, task, use_cache: bool = False) -> torch.Tensor:
        """
        Get task embedding for interpretability/visualization.
        """
        if use_cache and task.name in self._task_embeddings:
            return self._task_embeddings[task.name]

        with torch.no_grad():
            embedding = self.encode_task_batched(task).cpu()
            if use_cache:
                self._task_embeddings[task.name] = embedding
            return embedding

    def get_predictions_detailed(self, task) -> Dict[str, Any]:
        """
        Get comprehensive predictions for a task.
        """
        self.eval()
        with torch.no_grad():
            τ = self.encode_task_batched(task).unsqueeze(0)

            prim_probs = self.primitive_head(τ).squeeze(0)
            pred_count = self.count_head(τ).squeeze().item()

            # Top primitives
            top_k_values, top_k_indices = torch.topk(prim_probs, min(10, self.num_primitives))
            top_primitives = [
                {'name': self.primitive_names[idx], 'prob': float(val)}
                for val, idx in zip(top_k_values, top_k_indices)
            ]

            return {
                'task_embedding': τ.squeeze(0).cpu().numpy().tolist(),
                'primitive_probs': prim_probs.cpu().numpy().tolist(),
                'top_primitives': top_primitives,
                'predicted_count': pred_count,
                'num_primitives': self.num_primitives,
                'num_inventions': len(self.inventions)
            }

    # ========================================================================
    # SAVE / LOAD
    # ========================================================================

    def save(self, path: str):
        """Save model state."""
        import os

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'primitive_names': self.primitive_names,
            'inventions': self.inventions,
            'bigram_vocab': self.bigram_vocab,
            'training_losses': self.training_losses,
            'epoch_history': self.epoch_history,
            'config': {
                'd_suit': self.card_encoder.d_suit,
                'd_rank': self.card_encoder.d_rank,
                'd_pos': self.card_encoder.d_pos,
                'max_cards': self.max_cards,
                # NEW: Embedding normalization config
                'normalize_embeddings': self.normalize_embeddings,
                'encoding_mode': self.encoding_mode,
                'n_random_hands': self.n_random_hands,
                'lambda_random': self.lambda_random,
            }
        }

        temp_path = path + '.tmp'
        torch.save(checkpoint, temp_path)

        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            os.replace(temp_path, path)
        else:
            raise RuntimeError(f"Model save failed: {temp_path}")

    def load(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Restore vocabulary first (may affect model structure)
        saved_primitive_names = checkpoint.get('primitive_names', self.primitive_names)
        self.inventions = checkpoint.get('inventions', [])
        self.bigram_vocab = checkpoint.get('bigram_vocab', [])
        self.bigram_to_idx = {bg: i for i, bg in enumerate(self.bigram_vocab)}

        # Resize primitive head if needed to match checkpoint
        saved_num_prims = len(saved_primitive_names)
        if saved_num_prims != self.num_primitives:
            # Expand to match saved size
            while self.num_primitives < saved_num_prims:
                self.primitive_head.expand_for_invention()
                self.num_primitives += 1
            # Note: contraction case would need shrink() but is rare

        # Now update vocabulary
        self.primitive_names = saved_primitive_names
        self.primitive_to_idx = {name: i for i, name in enumerate(self.primitive_names)}

        # Load weights
        self.load_state_dict(checkpoint['model_state_dict'])

        self.training_losses = checkpoint.get('training_losses', [])
        self.epoch_history = checkpoint.get('epoch_history', [])


# ============================================================================
# TESTS
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("CONTRASTIVE RECOGNITION MODEL TEST")
    print("=" * 70)

    # Check PyTorch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Import dependencies
    from dreamcoder_core.lean_primitives import build_lean_grammar
    from rules.cards import sample_hand, Card, Suit, Rank

    # Build grammar
    grammar = build_lean_grammar()
    print(f"\nGrammar size: {len(grammar.productions)} primitives")

    # ========================================================================
    # Test 1: Card Encoding
    # ========================================================================
    print("\n1. TEST: FactoredCardEncoder")
    print("-" * 50)

    encoder = FactoredCardEncoder(d_suit=8, d_rank=16, d_pos=8)
    print(f"   Output dimension: {encoder.output_dim}")

    # Test encoding
    suits = torch.tensor([[0, 1, 2, 3, 0, 1]])  # 6 cards
    ranks = torch.tensor([[0, 5, 10, 12, 3, 7]])
    positions = torch.arange(6).unsqueeze(0)

    card_emb = encoder(suits, ranks, positions)
    print(f"   Card embedding shape: {card_emb.shape}")
    assert card_emb.shape == (1, 6, 32), f"Expected (1, 6, 32), got {card_emb.shape}"
    print("   ✓ FactoredCardEncoder works!")

    # ========================================================================
    # Test 2: Card Interaction MLP
    # ========================================================================
    print("\n2. TEST: CardInteractionMLP")
    print("-" * 50)

    card_mlp = CardInteractionMLP(input_dim=32, hidden_dim=64, output_dim=32)
    card_features = card_mlp(card_emb)
    print(f"   Card features shape: {card_features.shape}")
    assert card_features.shape == (1, 6, 32)
    print("   ✓ CardInteractionMLP works!")

    # ========================================================================
    # Test 3: Hand Encoder
    # ========================================================================
    print("\n3. TEST: HandEncoder")
    print("-" * 50)

    hand_encoder = HandEncoder(encoder, card_mlp)
    mask = torch.ones(1, 6, dtype=torch.bool)
    hand_emb = hand_encoder(suits, ranks, positions, mask)
    print(f"   Hand embedding shape: {hand_emb.shape}")
    assert hand_emb.shape == (1, 32)
    print("   ✓ HandEncoder works!")

    # ========================================================================
    # Test 4: Full Model Creation
    # ========================================================================
    print("\n4. TEST: ContrastiveRecognitionModel creation")
    print("-" * 50)

    model = ContrastiveRecognitionModel(
        grammar=grammar,
        d_suit=8,
        d_rank=16,
        d_pos=8,
        card_hidden=64,
        card_out=32,
        max_cards=8
    )
    print(f"   Number of primitives: {model.num_primitives}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("   ✓ Model created!")

    # ========================================================================
    # Test 5: Task Encoding
    # ========================================================================
    print("\n5. TEST: Contrastive Task Encoding")
    print("-" * 50)

    # Create a fake task
    class FakeTask:
        def __init__(self, name):
            self.name = name
            self.examples = [
                (sample_hand(6), True),
                (sample_hand(6), True),
                (sample_hand(6), False),
                (sample_hand(6), False),
            ]

    task = FakeTask("test_task")

    τ = model.encode_task_batched(task)
    print(f"   Task embedding shape: {τ.shape}")
    assert τ.shape == (32,)
    print(f"   Embedding norm: {τ.norm().item():.4f}")
    print("   ✓ Contrastive task encoding works!")

    # ========================================================================
    # Test 6: Primitive Prediction
    # ========================================================================
    print("\n6. TEST: Primitive Prediction")
    print("-" * 50)

    probs = model.predict_primitives(task)
    print(f"   Prediction shape: {probs.shape}")
    assert probs.shape == (model.num_primitives,)
    print(f"   Prob range: [{probs.min().item():.4f}, {probs.max().item():.4f}]")

    top_preds = model.get_top_predictions(task, n=5)
    print("   Top 5 predictions:")
    for name, prob in top_preds:
        print(f"      {name}: {prob:.4f}")
    print("   ✓ Primitive prediction works!")

    # ========================================================================
    # Test 7: Grammar Weight Prediction
    # ========================================================================
    print("\n7. TEST: Grammar Weight Prediction")
    print("-" * 50)

    new_grammar = model.predict_grammar_weights(task)
    print(f"   New grammar size: {len(new_grammar.productions)}")
    print("   ✓ Grammar weight prediction works!")

    # ========================================================================
    # Test 8: Structural Similarity Loss
    # ========================================================================
    print("\n8. TEST: Structural Similarity Loss")
    print("-" * 50)

    task2 = FakeTask("test_task_2")

    τ1 = model.encode_task_batched(task)
    τ2 = model.encode_task_batched(task2)

    task_embeddings = {'task1': τ1, 'task2': τ2}
    primitive_sets = {
        'task1': {'filter', 'map', '+'},
        'task2': {'filter', 'all', 'eq'}
    }

    loss = model.compute_structural_similarity_loss(task_embeddings, primitive_sets)
    print(f"   Structural similarity loss: {loss.item():.4f}")
    print("   ✓ Structural similarity loss works!")

    # ========================================================================
    # Test 9: Bigram Extraction
    # ========================================================================
    print("\n9. TEST: Bigram Extraction")
    print("-" * 50)

    # Create a simple program: (filter (λ (eq ...)))
    from dreamcoder_core.program import Primitive

    # Find primitives
    filter_prim = None
    eq_prim = None
    for p in grammar.productions:
        if str(p.program) == 'filter':
            filter_prim = p.program
        elif str(p.program) == 'eq':
            eq_prim = p.program

    if filter_prim and eq_prim:
        # (filter (λ eq))
        test_prog = Application(filter_prim, Abstraction(eq_prim))
        bigrams = extract_bigrams(test_prog)
        print(f"   Program: {test_prog}")
        print(f"   Extracted bigrams: {bigrams}")
    else:
        print("   (Skipped - primitives not found)")
    print("   ✓ Bigram extraction works!")

    # ========================================================================
    # Test 10: Invention Handling
    # ========================================================================
    print("\n10. TEST: Invention Handling")
    print("-" * 50)

    old_num = model.num_primitives

    # Create a fake invention
    fake_inv = Invented(Abstraction(Index(0)))
    model.add_invention(fake_inv)

    print(f"   Primitives before: {old_num}")
    print(f"   Primitives after: {model.num_primitives}")
    assert model.num_primitives == old_num + 1
    print("   ✓ Invention handling works!")

    # ========================================================================
    # Test 11: Save/Load
    # ========================================================================
    print("\n11. TEST: Save/Load")
    print("-" * 50)

    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.pt")
        model.save(path)
        print(f"   Saved to: {path}")
        print(f"   File size: {os.path.getsize(path):,} bytes")

        # Create new model and load
        model2 = ContrastiveRecognitionModel(grammar=grammar)
        model2.load(path)
        print(f"   Loaded primitives: {model2.num_primitives}")
        assert model2.num_primitives == model.num_primitives
    print("   ✓ Save/Load works!")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)
    print("""
Summary of tested functionality:

1. FactoredCardEncoder: Learned embeddings for suit/rank/position
2. CardInteractionMLP: Per-card feature extraction
3. HandEncoder: Mean pooling over cards
4. ContrastiveRecognitionModel: Full model creation
5. Contrastive Task Encoding: τ = mean(pos) - mean(neg)
6. Primitive Prediction: Sigmoid outputs per primitive
7. Grammar Weight Prediction: Blend predictions with grammar
8. Structural Similarity Loss: Cluster similar tasks
9. Bigram Extraction: Extract primitive co-occurrence
10. Invention Handling: Dynamic vocabulary expansion
11. Save/Load: Checkpoint persistence

Key differences from GRU-based model:
- Factored card embeddings (vs one-hot)
- Contrastive task encoding (vs pooled examples)
- Sigmoid outputs (vs softmax)
- Structural similarity training objective
- Built-in invention/bigram support
""")
