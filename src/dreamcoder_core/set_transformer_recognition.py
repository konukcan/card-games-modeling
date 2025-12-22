#!/usr/bin/env python3
"""
Set Transformer Recognition Model for DreamCoder

This module implements a recognition model based on the Set Transformer architecture
(Lee et al., 2019), specifically designed for card-based classification tasks.

Key Differences from Original GRU-based Model:
1. Uses self-attention to model card-card interactions explicitly
2. Employs learnable positional encodings for position-sensitive rules
3. Hierarchical structure: cards → hand embedding → task embedding
4. Better suited for classification tasks where output is True/False

Architecture Overview:
- CardEmbedding: Per-card feature encoding with positional information
- SetAttentionBlock (SAB): Self-attention over cards/examples
- PoolingByMultiheadAttention (PMA): Learnable pooling with seed vectors
- HandEncoder: SAB layers + PMA to encode a hand of cards
- TaskEncoder: SAB layers + PMA to encode a set of examples
- PrimitivePredictor: MLP from task embedding to primitive log-probabilities

References:
- Lee et al. (2019). "Set Transformer: A Framework for Attention-based
  Permutation-Invariant Neural Networks"
- Zaheer et al. (2017). "Deep Sets"
"""

import sys
import math
import random
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict
from dataclasses import dataclass, field
import json

import numpy as np
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
# CARD FEATURE EXTRACTION
# ============================================================================

@dataclass
class CardFeatures:
    """Feature representation for a single card."""
    suit_idx: int      # 0-3 for CLUBS, DIAMONDS, HEARTS, SPADES
    rank_idx: int      # 0-12 for 2-A
    color_idx: int     # 0 (RED) or 1 (BLACK)
    rank_value: int    # 2-14
    is_face: bool      # J, Q, K
    is_ace: bool       # Ace

    def to_vector(self) -> List[float]:
        """Convert to fixed-size feature vector (24 dimensions)."""
        vec = [0.0] * 24  # 4 suit + 13 rank + 2 color + 5 properties

        # One-hot suit (4)
        vec[self.suit_idx] = 1.0

        # One-hot rank (13)
        vec[4 + self.rank_idx] = 1.0

        # One-hot color (2)
        vec[17 + self.color_idx] = 1.0

        # Normalized rank value
        vec[19] = (self.rank_value - 2) / 12.0

        # Boolean features
        vec[20] = float(self.is_face)
        vec[21] = float(self.is_ace)
        vec[22] = float(self.rank_value % 2 == 0)  # Even
        vec[23] = float(self.rank_value % 2 == 1)  # Odd

        return vec


def extract_card_features(card) -> CardFeatures:
    """Extract features from a Card object."""
    from rules.cards import Suit, Rank, RANK_VALUES, card_color, Color

    suit_map = {Suit.CLUBS: 0, Suit.DIAMONDS: 1, Suit.HEARTS: 2, Suit.SPADES: 3}
    rank_map = {r: i for i, r in enumerate(Rank)}
    color_map = {Color.RED: 0, Color.BLACK: 1}

    return CardFeatures(
        suit_idx=suit_map[card.suit],
        rank_idx=rank_map[card.rank],
        color_idx=color_map[card_color(card)],
        rank_value=RANK_VALUES[card.rank],
        is_face=card.rank in (Rank.JACK, Rank.QUEEN, Rank.KING),
        is_ace=card.rank == Rank.ACE
    )


def encode_hand(hand, max_cards: int = 8) -> torch.Tensor:
    """
    Encode a hand of cards as a tensor.

    Args:
        hand: List of Card objects
        max_cards: Maximum number of cards (for padding)

    Returns:
        Tensor of shape (max_cards, card_feature_dim)
    """
    card_dim = 24  # From CardFeatures.to_vector()

    features = torch.zeros(max_cards, card_dim)

    for i, card in enumerate(hand[:max_cards]):
        cf = extract_card_features(card)
        features[i] = torch.tensor(cf.to_vector())

    return features


def encode_output(output: Any) -> torch.Tensor:
    """Encode an output value (True/False for classification tasks)."""
    if isinstance(output, bool):
        return torch.tensor([1.0 if output else 0.0, 0.0 if output else 1.0])
    else:
        # For other outputs, try simple numeric encoding
        return torch.tensor([float(output), 0.0])


# ============================================================================
# SET TRANSFORMER BUILDING BLOCKS
# ============================================================================

class MultiheadAttention(nn.Module):
    """
    Multihead attention module following Vaswani et al. (2017).

    Computes attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (batch, seq_q, d_model)
            key: (batch, seq_k, d_model)
            value: (batch, seq_k, d_model)
            mask: Optional (batch, seq_q, seq_k) mask

        Returns:
            output: (batch, seq_q, d_model)
        """
        batch_size = query.size(0)

        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # Shape: (batch, num_heads, seq, d_k)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, heads, seq_q, seq_k)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # (batch, heads, seq_q, d_k)

        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)

        return output


class MultiheadAttentionBlock(nn.Module):
    """
    Multihead Attention Block (MAB) from Set Transformer paper.

    Uses Pre-LayerNorm (LN before attention/FF) which is more stable and
    preserves more discriminative information than Post-LayerNorm.

    Pre-LN: output = X + Attention(LayerNorm(X), LayerNorm(Y))
    This avoids normalizing the residual path, preserving variance.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, use_layernorm: bool = True):
        super().__init__()

        self.attention = MultiheadAttention(d_model, num_heads, dropout)
        self.use_layernorm = use_layernorm

        if use_layernorm:
            self.norm1 = nn.LayerNorm(d_model)
            self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            X: Query sequence (batch, seq_x, d_model)
            Y: Key/Value sequence (batch, seq_y, d_model)
            mask: Optional attention mask

        Returns:
            output: (batch, seq_x, d_model)
        """
        if self.use_layernorm:
            # Pre-LayerNorm: normalize inputs, not outputs
            # This preserves variance in the residual stream
            X_norm = self.norm1(X)
            Y_norm = self.norm1(Y) if Y is X else Y  # Same norm for self-attention
            H = X + self.attention(X_norm, Y_norm, Y_norm, mask)

            H_norm = self.norm2(H)
            output = H + self.ff(H_norm)
        else:
            # No normalization - maximum variance preservation
            H = X + self.attention(X, Y, Y, mask)
            output = H + self.ff(H)

        return output


class SetAttentionBlock(nn.Module):
    """
    Set Attention Block (SAB) = MAB(X, X)

    Self-attention over a set, allowing each element to attend to all others.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, use_layernorm: bool = True):
        super().__init__()
        self.mab = MultiheadAttentionBlock(d_model, num_heads, dropout, use_layernorm)

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            X: Set elements (batch, num_elements, d_model)
            mask: Optional attention mask

        Returns:
            output: (batch, num_elements, d_model) with context from all elements
        """
        return self.mab(X, X, mask)


class InducedSetAttentionBlock(nn.Module):
    """
    Induced Set Attention Block (ISAB) for efficiency on large sets.

    Uses m inducing points to reduce complexity from O(n²) to O(nm).
    Not needed for our small card hands but included for completeness.

    ISAB_m(X) = MAB(X, H) where H = MAB(I, X)
    """

    def __init__(self, d_model: int, num_heads: int, num_inducing: int, dropout: float = 0.1):
        super().__init__()

        # Learnable inducing points
        self.inducing_points = nn.Parameter(torch.randn(1, num_inducing, d_model))

        self.mab1 = MultiheadAttentionBlock(d_model, num_heads, dropout)
        self.mab2 = MultiheadAttentionBlock(d_model, num_heads, dropout)

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = X.size(0)

        # Expand inducing points for batch
        I = self.inducing_points.expand(batch_size, -1, -1)

        # Inducing points attend to X
        H = self.mab1(I, X, mask)

        # X attends to compressed representation
        return self.mab2(X, H)


class PoolingByMultiheadAttention(nn.Module):
    """
    Pooling by Multihead Attention (PMA).

    Uses k learnable "seed" vectors to query the set and produce k output vectors.
    This is more expressive than simple mean/sum pooling.

    PMA_k(X) = MAB(S, X) where S are k learnable seeds
    """

    def __init__(self, d_model: int, num_heads: int, num_seeds: int, dropout: float = 0.1, use_layernorm: bool = True):
        super().__init__()

        # Learnable seed vectors
        self.seeds = nn.Parameter(torch.randn(1, num_seeds, d_model))
        nn.init.xavier_uniform_(self.seeds)

        self.mab = MultiheadAttentionBlock(d_model, num_heads, dropout, use_layernorm)

    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            X: Set elements (batch, num_elements, d_model)
            mask: Optional attention mask

        Returns:
            output: (batch, num_seeds, d_model)
        """
        batch_size = X.size(0)

        # Expand seeds for batch
        S = self.seeds.expand(batch_size, -1, -1)

        # Seeds query the set
        return self.mab(S, X, mask)


# ============================================================================
# POSITIONAL ENCODING
# ============================================================================

class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding for sequences.

    Unlike sinusoidal encoding, this learns position-specific embeddings
    that can capture arbitrary positional patterns important for our rules
    (e.g., "ends same suit" requires knowing first and last positions).
    """

    def __init__(self, max_len: int, d_model: int):
        super().__init__()
        self.position_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Args:
            seq_len: Length of sequence
            device: Device to create tensor on

        Returns:
            Positional encoding (1, seq_len, d_model)
        """
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        return self.position_embeddings(positions)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from Vaswani et al. (2017).

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, max_len: int, d_model: int):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        return self.pe[:, :seq_len, :].to(device)


# ============================================================================
# HAND ENCODER
# ============================================================================

class SetTransformerHandEncoder(nn.Module):
    """
    Encodes a hand of cards using Set Transformer architecture.

    Architecture:
    1. Card embedding: per-card features → d_model
    2. Positional encoding: adds position information
    3. SAB layers: self-attention for card-card interactions (NO LayerNorm to preserve variance)
    4. PMA: pools cards into single hand representation

    NOTE: We disable LayerNorm in SAB and PMA to preserve discriminative information.
    LayerNorm normalizes each sample to zero mean and unit variance, which destroys
    the differences between hands when they have similar structure.
    """

    def __init__(
        self,
        card_feature_dim: int = 24,
        d_model: int = 64,
        num_heads: int = 4,
        num_sab_layers: int = 2,
        num_seeds: int = 1,
        max_cards: int = 8,
        use_positional: bool = True,
        dropout: float = 0.1,
        use_layernorm: bool = False  # Disabled by default to preserve variance
    ):
        super().__init__()

        self.d_model = d_model
        self.use_positional = use_positional

        # Card feature embedding with LayerNorm at the end for stable input scale
        self.card_embed = nn.Sequential(
            nn.Linear(card_feature_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model)  # Only normalize here, not in attention blocks
        )

        # Positional encoding (learnable for position-sensitive rules)
        if use_positional:
            self.pos_encoding = LearnablePositionalEncoding(max_cards, d_model)

        # Self-attention layers - NO LayerNorm to preserve variance
        self.sab_layers = nn.ModuleList([
            SetAttentionBlock(d_model, num_heads, dropout, use_layernorm=use_layernorm)
            for _ in range(num_sab_layers)
        ])

        # Pooling to single vector - NO LayerNorm
        self.pma = PoolingByMultiheadAttention(d_model, num_heads, num_seeds, dropout, use_layernorm=use_layernorm)

        # Final projection if using multiple seeds
        if num_seeds > 1:
            self.output_proj = nn.Linear(d_model * num_seeds, d_model)
        else:
            self.output_proj = nn.Identity()

    def forward(
        self,
        hand_features: torch.Tensor,
        lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hand_features: (batch, max_cards, card_feature_dim)
            lengths: Optional (batch,) actual hand lengths for masking

        Returns:
            hand_embedding: (batch, d_model)
        """
        batch_size, max_cards, _ = hand_features.shape

        # Embed card features
        x = self.card_embed(hand_features)  # (batch, max_cards, d_model)

        # Add positional encoding
        if self.use_positional:
            pos_enc = self.pos_encoding(max_cards, x.device)
            x = x + pos_enc

        # Create attention mask if lengths provided
        mask = None
        if lengths is not None:
            # Mask out padding positions
            seq_range = torch.arange(max_cards, device=x.device).unsqueeze(0)
            mask = seq_range < lengths.unsqueeze(1)  # (batch, max_cards)
            # Expand for attention: (batch, max_cards, max_cards)
            mask = mask.unsqueeze(1).expand(-1, max_cards, -1)

        # Self-attention layers
        for sab in self.sab_layers:
            x = sab(x, mask)

        # Pool to single vector per hand
        pooled = self.pma(x, mask)  # (batch, num_seeds, d_model)

        # Flatten if multiple seeds
        pooled = pooled.flatten(start_dim=1)  # (batch, num_seeds * d_model)

        return self.output_proj(pooled)  # (batch, d_model)


# ============================================================================
# EXAMPLE ENCODER
# ============================================================================

class SetTransformerExampleEncoder(nn.Module):
    """
    Encodes a single (hand, label) example.

    Uses FiLM (Feature-wise Linear Modulation) to combine hand and label,
    ensuring the label actually modulates the hand representation rather
    than being washed out by concatenation.

    Architecture:
    - Hand encoding from SetTransformerHandEncoder
    - Label → (gamma, beta) parameters for FiLM
    - FiLM: output = gamma * hand + beta
    - This ensures the label fundamentally changes the representation
    """

    def __init__(
        self,
        d_model: int = 64,
        label_dim: int = 2,
        card_feature_dim: int = 24,
        num_heads: int = 4,
        num_sab_layers: int = 2,
        max_cards: int = 8,
        use_positional: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        self.d_model = d_model

        # Hand encoder
        self.hand_encoder = SetTransformerHandEncoder(
            card_feature_dim=card_feature_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_sab_layers=num_sab_layers,
            max_cards=max_cards,
            use_positional=use_positional,
            dropout=dropout
        )

        # FiLM: Label → (gamma, beta) for modulating hand embedding
        # gamma scales, beta shifts - this ensures label fundamentally changes the output
        self.label_to_gamma = nn.Sequential(
            nn.Linear(label_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.label_to_beta = nn.Sequential(
            nn.Linear(label_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        # Final projection after FiLM
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model)
        )

    def forward(
        self,
        hand_features: torch.Tensor,
        label_features: torch.Tensor,
        hand_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            hand_features: (batch, max_cards, card_feature_dim)
            label_features: (batch, label_dim)
            hand_lengths: Optional (batch,) actual hand lengths

        Returns:
            example_embedding: (batch, d_model)
        """
        hand_emb = self.hand_encoder(hand_features, hand_lengths)  # (batch, d_model)

        # FiLM modulation: gamma * hand + beta
        # This ensures True/False labels produce DIFFERENT representations
        gamma = self.label_to_gamma(label_features)  # (batch, d_model)
        beta = self.label_to_beta(label_features)    # (batch, d_model)

        # Apply FiLM: element-wise scaling and shifting
        modulated = gamma * hand_emb + beta  # (batch, d_model)

        return self.output_proj(modulated)  # (batch, d_model)


# ============================================================================
# TASK ENCODER
# ============================================================================

class SetTransformerTaskEncoder(nn.Module):
    """
    Encodes a task using RAW CARD FEATURE correlations.

    Key insight: The learned embeddings (d_model=64) collapse discriminative information,
    but raw card features (24-dim) preserve it. Different rules create different
    correlation patterns between card features and True/False labels.

    Architecture:
    - Compute correlation between RAW card features and the label
    - Expand to d_model by tiling/padding (no learned projection initially)
    - Use identity initialization for projection to preserve the signal
    """

    def __init__(
        self,
        d_model: int = 64,
        num_heads: int = 4,  # Kept for API compatibility
        num_sab_layers: int = 2,  # Kept for API compatibility
        num_seeds: int = 1,  # Kept for API compatibility
        max_examples: int = 20,
        dropout: float = 0.1,
        use_layernorm: bool = False,  # Kept for API compatibility
        raw_feature_dim: int = 24  # Raw card feature dimension
    ):
        super().__init__()

        self.d_model = d_model
        self.raw_feature_dim = raw_feature_dim

        # Total raw signal dimension: correlation (24) + diff (24) = 48
        # We'll pad/tile to get to d_model
        self.raw_signal_dim = raw_feature_dim * 2

        # Simple expansion: linear layer initialized near identity
        # This maps 48 -> d_model while preserving the signal
        self.expand = nn.Linear(self.raw_signal_dim, d_model, bias=True)

        # Initialize to preserve raw signal:
        # First 48 dims get identity-like projection, rest get small random
        with torch.no_grad():
            self.expand.weight.zero_()
            self.expand.bias.zero_()
            # Copy input directly to first 48 output dims
            min_dim = min(self.raw_signal_dim, d_model)
            for i in range(min_dim):
                self.expand.weight[i, i] = 1.0
            # Add small noise to remaining output dims
            if d_model > self.raw_signal_dim:
                self.expand.weight[self.raw_signal_dim:, :].normal_(0, 0.1)

    def forward(
        self,
        example_embeddings: torch.Tensor,
        example_mask: Optional[torch.Tensor] = None,
        label_mask: Optional[torch.Tensor] = None,
        raw_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            example_embeddings: (batch, max_examples, d_model) - learned embeddings (unused)
            example_mask: Optional (batch, max_examples) True for valid examples
            label_mask: Optional (batch, max_examples) True for positive examples
            raw_features: Optional (batch, max_examples, raw_feature_dim) - RAW card features

        Returns:
            task_embedding: (batch, d_model)
        """
        batch_size = example_embeddings.size(0)
        device = example_embeddings.device

        # Initialize outputs for RAW features
        raw_correlation = torch.zeros(batch_size, self.raw_feature_dim, device=device)
        raw_diff = torch.zeros(batch_size, self.raw_feature_dim, device=device)

        if label_mask is not None and example_mask is not None and raw_features is not None:
            for b in range(batch_size):
                # Get valid examples
                valid_indices = example_mask[b].nonzero(as_tuple=True)[0]
                if len(valid_indices) < 2:
                    continue

                labels = label_mask[b, valid_indices].float()  # (n_valid,)
                label_std = labels.std() + 1e-8

                # Skip if all same label (no discrimination possible)
                if label_std < 1e-6:
                    continue

                label_centered = labels - labels.mean()

                # === RAW FEATURE CORRELATION (the key discriminative signal) ===
                raw_feat = raw_features[b, valid_indices]  # (n_valid, raw_feature_dim)
                raw_centered = raw_feat - raw_feat.mean(dim=0, keepdim=True)
                raw_cov = (raw_centered * label_centered.unsqueeze(1)).mean(dim=0)
                raw_std = raw_feat.std(dim=0) + 1e-8
                raw_correlation[b] = raw_cov / (raw_std * label_std)

                # Prototype difference for raw features
                pos_indices = (example_mask[b] & label_mask[b]).nonzero(as_tuple=True)[0]
                neg_indices = (example_mask[b] & ~label_mask[b]).nonzero(as_tuple=True)[0]

                if len(pos_indices) > 0 and len(neg_indices) > 0:
                    raw_pos_mean = raw_features[b, pos_indices].mean(dim=0)
                    raw_neg_mean = raw_features[b, neg_indices].mean(dim=0)
                    raw_diff[b] = raw_pos_mean - raw_neg_mean

        # === EXPAND to d_model (identity-initialized) ===
        # Concatenate correlation and diff - these are the key discriminative signals
        raw_combined = torch.cat([raw_correlation, raw_diff], dim=-1)  # (batch, 48)

        # Expand to d_model while preserving the signal (identity-initialized)
        return self.expand(raw_combined)  # (batch, d_model)


# ============================================================================
# PRIMITIVE PREDICTOR
# ============================================================================

class PrimitivePredictor(nn.Module):
    """
    Predicts log-probabilities for each grammar primitive.

    Takes a task encoding and outputs a vector of log-probabilities,
    one per primitive in the grammar.
    """

    def __init__(
        self,
        d_model: int = 64,
        num_primitives: int = 76,
        hidden_dim: int = 128,
        dropout: float = 0.1
    ):
        super().__init__()

        self.predictor = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_primitives)
        )

    def forward(self, task_encoding: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        Args:
            task_encoding: (batch, d_model)
            return_logits: If True, return raw logits instead of log-probabilities

        Returns:
            Log-probabilities or logits: (batch, num_primitives)
        """
        logits = self.predictor(task_encoding)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)


# ============================================================================
# FULL SET TRANSFORMER RECOGNITION MODEL
# ============================================================================

class SetTransformerRecognitionModel(nn.Module):
    """
    Complete Set Transformer-based recognition model for DreamCoder.

    Hierarchical structure:
    1. Cards → Hand embedding (SetTransformerHandEncoder)
    2. (Hand, Label) → Example embedding (SetTransformerExampleEncoder)
    3. Examples → Task embedding (SetTransformerTaskEncoder)
    4. Task → Primitive probabilities (PrimitivePredictor)
    """

    def __init__(
        self,
        grammar: Grammar,
        d_model: int = 64,
        num_heads: int = 4,
        num_hand_layers: int = 2,
        num_task_layers: int = 2,
        max_examples: int = 20,
        max_cards: int = 8,
        use_positional: bool = True,
        learning_rate: float = 1e-3,
        dropout: float = 0.1,
        device: str = 'cpu'
    ):
        super().__init__()

        self.grammar = grammar
        self.d_model = d_model
        self.max_examples = max_examples
        self.max_cards = max_cards
        self.device = device

        # Create primitive name -> index mapping
        self.primitive_names = [str(p.program) for p in grammar.productions]
        self.primitive_to_idx = {name: i for i, name in enumerate(self.primitive_names)}
        self.num_primitives = len(self.primitive_names)

        # Network components
        self.example_encoder = SetTransformerExampleEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_sab_layers=num_hand_layers,
            max_cards=max_cards,
            use_positional=use_positional,
            dropout=dropout
        )

        self.task_encoder = SetTransformerTaskEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_sab_layers=num_task_layers,
            max_examples=max_examples,
            dropout=dropout
        )

        self.primitive_predictor = PrimitivePredictor(
            d_model=d_model,
            num_primitives=self.num_primitives,
            hidden_dim=d_model * 2,
            dropout=dropout
        )

        # Move to device
        self.to(device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

        # Training history
        self.training_losses: List[float] = []
        self.epoch_history: List[Dict] = []

        # Embedding cache for interpretability
        self._task_embeddings: Dict[str, torch.Tensor] = {}

        # Hidden dim property for compatibility
        self.hidden_dim = d_model

    def encode_task(self, task) -> torch.Tensor:
        """
        Encode a task into a fixed-size vector.

        Args:
            task: Task object with examples

        Returns:
            Task embedding: (d_model,)
        """
        # Encode each example and track labels + raw features
        example_embeddings = []
        raw_feature_list = []  # RAW card features for correlation computation
        labels = []  # Track which examples are positive (True)

        for inp, out in task.examples[:self.max_examples]:
            # Encode hand
            hand_features = encode_hand(inp, self.max_cards).unsqueeze(0).to(self.device)
            label_features = encode_output(out).unsqueeze(0).to(self.device)

            # Get example embedding
            example_emb = self.example_encoder(hand_features, label_features)
            example_embeddings.append(example_emb.squeeze(0))
            labels.append(out == True)  # Track positive examples

            # Store RAW features: mean-pool card features for this hand
            # hand_features is (1, max_cards, 24) - we mean pool over cards
            num_cards = len(inp) if hasattr(inp, '__len__') else self.max_cards
            raw_hand_feat = hand_features[0, :num_cards, :].mean(dim=0)  # (24,)
            raw_feature_list.append(raw_hand_feat)

        if not example_embeddings:
            return torch.zeros(self.d_model, device=self.device)

        # Stack examples
        stacked = torch.stack(example_embeddings, dim=0).unsqueeze(0)  # (1, num_examples, d_model)

        # Stack raw features
        raw_stacked = torch.stack(raw_feature_list, dim=0).unsqueeze(0)  # (1, num_examples, 24)

        # Create mask for valid examples
        num_examples = len(example_embeddings)
        example_mask = torch.ones(1, self.max_examples, dtype=torch.bool, device=self.device)
        example_mask[0, num_examples:] = False

        # Create label mask (True for positive examples)
        label_mask = torch.zeros(1, self.max_examples, dtype=torch.bool, device=self.device)
        for i, is_positive in enumerate(labels):
            label_mask[0, i] = is_positive

        # Pad examples if needed
        if num_examples < self.max_examples:
            padding = torch.zeros(
                1, self.max_examples - num_examples, self.d_model,
                device=self.device
            )
            stacked = torch.cat([stacked, padding], dim=1)

            raw_padding = torch.zeros(
                1, self.max_examples - num_examples, 24,
                device=self.device
            )
            raw_stacked = torch.cat([raw_stacked, raw_padding], dim=1)

        # Encode task with label information AND raw features
        task_emb = self.task_encoder(stacked, example_mask, label_mask, raw_features=raw_stacked)

        return task_emb.squeeze(0)

    def predict_primitive_probs(self, task) -> torch.Tensor:
        """
        Predict log-probabilities for each primitive given a task.

        Returns:
            Log-probabilities: (num_primitives,)
        """
        self.eval()
        with torch.no_grad():
            task_enc = self.encode_task(task).unsqueeze(0)
            log_probs = self.primitive_predictor(task_enc)
            return log_probs.squeeze(0)

    def predict_primitive_logits(self, task) -> torch.Tensor:
        """
        Predict raw logits (before softmax) for each primitive.

        Returns:
            Logits: (num_primitives,)
        """
        self.eval()
        with torch.no_grad():
            task_enc = self.encode_task(task).unsqueeze(0)
            logits = self.primitive_predictor(task_enc, return_logits=True)
            return logits.squeeze(0)

    def get_primitive_predictions_detailed(self, task) -> Dict[str, Any]:
        """
        Get comprehensive primitive predictions for a task.

        Returns dict with:
        - log_probs: List of log-probabilities
        - logits: List of raw logits
        - top_10: List of dicts with primitive name, log_prob, logit, prob
        - entropy: Entropy of the distribution
        - max_prob: Maximum probability
        """
        self.eval()
        with torch.no_grad():
            task_enc = self.encode_task(task).unsqueeze(0)

            logits = self.primitive_predictor(task_enc, return_logits=True).squeeze(0)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)

            # Compute metrics
            entropy = -torch.sum(probs * log_probs).item()
            max_prob = torch.max(probs).item()

            # Top 10
            top_k_values, top_k_indices = torch.topk(log_probs, min(10, len(log_probs)))

            top_k = []
            for lp, idx in zip(top_k_values, top_k_indices):
                idx_int = int(idx)
                prim_name = self.primitive_names[idx_int] if idx_int < len(self.primitive_names) else f"unknown_{idx_int}"
                top_k.append({
                    'primitive': prim_name,
                    'log_prob': float(lp),
                    'logit': float(logits[idx_int]),
                    'prob': float(torch.exp(lp))
                })

            return {
                'log_probs': log_probs.cpu().numpy().tolist(),
                'logits': logits.cpu().numpy().tolist(),
                'top_10': top_k,
                'entropy': entropy,
                'max_prob': max_prob,
                'num_primitives': len(self.primitive_names)
            }

    def predict_grammar_weights(self, task) -> Grammar:
        """
        Predict grammar weights for a task.
        Returns a grammar with adjusted primitive probabilities.
        """
        log_probs = self.predict_primitive_probs(task)
        log_probs_np = log_probs.cpu().numpy()

        new_productions = []
        for i, prod in enumerate(self.grammar.productions):
            prim_name = str(prod.program)
            if prim_name in self.primitive_to_idx:
                idx = self.primitive_to_idx[prim_name]
                # Blend original and predicted
                new_lp = 0.5 * prod.log_probability + 0.5 * log_probs_np[idx]
            else:
                new_lp = prod.log_probability

            new_productions.append(Production(prod.program, prod.tp, new_lp))

        return Grammar(new_productions, self.grammar.log_variable).normalize_probabilities()

    def train_on_frontiers(
        self,
        tasks: List,
        frontiers: Dict,
        epochs: int = 10,
        batch_size: int = 8
    ) -> float:
        """
        Train the recognition model on solved tasks.

        Args:
            tasks: List of Task objects
            frontiers: Dict mapping task names to TaskFrontier objects
            epochs: Number of training epochs
            batch_size: Batch size

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
                    if entry.log_likelihood == 0.0:
                        self._collect_primitives(entry.program, target_primitives)

                if target_primitives:
                    training_data.append((task, target_primitives))

        if not training_data:
            return 0.0

        self.train()
        total_loss = 0.0
        n_batches = 0

        for epoch in range(epochs):
            random.shuffle(training_data)
            epoch_loss = 0.0

            for i in range(0, len(training_data), batch_size):
                batch = training_data[i:i+batch_size]

                self.optimizer.zero_grad()
                batch_loss = 0.0

                for task, target_prims in batch:
                    # Encode task
                    task_enc = self.encode_task(task).unsqueeze(0)

                    # Predict primitives
                    log_probs = self.primitive_predictor(task_enc).squeeze(0)

                    # Build target distribution
                    target = torch.zeros(self.num_primitives, device=self.device)
                    for prim_name in target_prims:
                        if prim_name in self.primitive_to_idx:
                            target[self.primitive_to_idx[prim_name]] = 1.0

                    # Normalize
                    if target.sum() > 0:
                        target = target / target.sum()

                    # Cross-entropy loss
                    loss = -torch.sum(target * log_probs)
                    batch_loss += loss

                batch_loss = batch_loss / len(batch)
                batch_loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                self.optimizer.step()

                epoch_loss += batch_loss.item()
                n_batches += 1

            avg_epoch_loss = epoch_loss / max(1, len(training_data) // batch_size)
            self.training_losses.append(avg_epoch_loss)

        # Update scheduler
        final_loss = sum(self.training_losses[-epochs:]) / epochs if epochs > 0 else 0.0
        self.scheduler.step(final_loss)

        # Record history
        self.epoch_history.append({
            'num_tasks': len(training_data),
            'final_loss': final_loss,
            'epochs': epochs
        })

        # Clear embedding cache
        self.clear_embedding_cache()

        return final_loss

    def _collect_primitives(self, program, primitives: Set[str]):
        """Collect all primitive names used in a program."""
        if isinstance(program, (Primitive, Invented)):
            primitives.add(str(program))
        elif isinstance(program, Application):
            self._collect_primitives(program.f, primitives)
            self._collect_primitives(program.x, primitives)
        elif isinstance(program, Abstraction):
            self._collect_primitives(program.body, primitives)
        elif CYTHON_AVAILABLE:
            if isinstance(program, (CythonPrimitive, CythonInvented)):
                primitives.add(str(program))
            elif isinstance(program, CythonApplication):
                self._collect_primitives(program.f, primitives)
                self._collect_primitives(program.x, primitives)
            elif isinstance(program, CythonAbstraction):
                self._collect_primitives(program.body, primitives)

    def get_task_embedding(self, task, use_cache: bool = False) -> torch.Tensor:
        """
        Get task embedding for interpretability.

        Args:
            task: Task object
            use_cache: If True, use cached embedding if available

        Returns:
            Task embedding tensor
        """
        if use_cache and task.name in self._task_embeddings:
            return self._task_embeddings[task.name]

        with torch.no_grad():
            embedding = self.encode_task(task).cpu()
            if use_cache:
                self._task_embeddings[task.name] = embedding
            return embedding

    def clear_embedding_cache(self):
        """Clear the task embedding cache."""
        self._task_embeddings.clear()

    def compute_task_discrimination(self, tasks: List, n_sample: int = 20) -> Dict[str, Any]:
        """
        Compute task discrimination metrics for interpretability.

        Measures how well the model distinguishes between different tasks
        by computing pairwise cosine similarities of task embeddings.

        Args:
            tasks: List of Task objects
            n_sample: Max number of tasks to sample

        Returns:
            Dict with:
            - mean_similarity: Average pairwise cosine similarity (lower = better)
            - min_similarity: Minimum similarity (diversity)
            - max_similarity: Maximum similarity (potential confusion)
            - similarity_matrix: Full pairwise similarity matrix
            - task_names: Names of tasks in matrix
        """
        self.eval()
        with torch.no_grad():
            # Sample tasks if too many
            sample_tasks = tasks[:n_sample]

            embeddings = []
            task_names = []
            for task in sample_tasks:
                emb = self.encode_task(task)
                embeddings.append(emb)
                task_names.append(task.name)

            if len(embeddings) < 2:
                return {
                    'mean_similarity': 1.0,
                    'min_similarity': 1.0,
                    'max_similarity': 1.0,
                    'similarity_matrix': [[1.0]],
                    'task_names': task_names
                }

            # Stack embeddings: (n_tasks, d_model)
            emb_matrix = torch.stack(embeddings, dim=0)

            # Normalize for cosine similarity
            emb_norm = F.normalize(emb_matrix, p=2, dim=1)

            # Compute pairwise similarities
            sim_matrix = torch.mm(emb_norm, emb_norm.t())

            # Extract upper triangle (excluding diagonal)
            n = len(embeddings)
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            pairwise_sims = sim_matrix[mask]

            return {
                'mean_similarity': float(pairwise_sims.mean()),
                'min_similarity': float(pairwise_sims.min()),
                'max_similarity': float(pairwise_sims.max()),
                'std_similarity': float(pairwise_sims.std()),
                'similarity_matrix': sim_matrix.cpu().numpy().tolist(),
                'task_names': task_names
            }

    def get_prediction_diversity(self, tasks: List, n_sample: int = 20) -> Dict[str, Any]:
        """
        Measure diversity of primitive predictions across tasks.

        For the legacy GRU model, all tasks got nearly identical predictions.
        This metric checks if Set Transformer produces more diverse predictions.

        Args:
            tasks: List of Task objects
            n_sample: Max number of tasks to sample

        Returns:
            Dict with:
            - entropy_mean: Mean entropy across tasks
            - entropy_std: Std of entropy (higher = more variation)
            - prediction_cosine_mean: Mean cosine sim of prediction vectors (lower = more diverse)
            - top_primitive_diversity: How many different primitives appear as top-1 across tasks
        """
        self.eval()
        with torch.no_grad():
            sample_tasks = tasks[:n_sample]

            entropies = []
            predictions = []
            top_primitives = []

            for task in sample_tasks:
                log_probs = self.predict_primitive_probs(task)
                probs = torch.exp(log_probs)

                # Entropy
                entropy = -torch.sum(probs * log_probs).item()
                entropies.append(entropy)

                predictions.append(log_probs)
                top_primitives.append(int(torch.argmax(log_probs)))

            # Stack predictions
            pred_matrix = torch.stack(predictions, dim=0)  # (n_tasks, n_prims)

            # Cosine similarity of prediction vectors
            pred_norm = F.normalize(pred_matrix, p=2, dim=1)
            pred_sim_matrix = torch.mm(pred_norm, pred_norm.t())
            n = len(predictions)
            mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
            pairwise_pred_sims = pred_sim_matrix[mask]

            return {
                'entropy_mean': float(np.mean(entropies)),
                'entropy_std': float(np.std(entropies)),
                'prediction_cosine_mean': float(pairwise_pred_sims.mean()),
                'prediction_cosine_std': float(pairwise_pred_sims.std()),
                'top_primitive_diversity': len(set(top_primitives)),
                'n_tasks': len(sample_tasks)
            }

    def get_top_predictions(self, task, n: int = 10) -> List[Tuple[str, float]]:
        """Get top-n predicted primitives."""
        log_probs = self.predict_primitive_probs(task)
        values, indices = torch.topk(log_probs, min(n, self.num_primitives))

        results = []
        for val, idx in zip(values.cpu().numpy(), indices.cpu().numpy()):
            results.append((self.primitive_names[idx], float(val)))

        return results

    def save(self, path: str):
        """Save model state."""
        import os

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'primitive_names': self.primitive_names,
            'd_model': self.d_model,
            'training_losses': self.training_losses,
            'epoch_history': self.epoch_history,
            'model_type': 'SetTransformerRecognitionModel'
        }

        temp_path = path + '.tmp'
        torch.save(checkpoint, temp_path)

        if os.path.exists(temp_path):
            if os.path.getsize(temp_path) > 0:
                os.replace(temp_path, path)
            else:
                raise RuntimeError(f"Model save failed: empty file")
        else:
            raise RuntimeError(f"Model save failed: file not created")

    def load(self, path: str):
        """Load model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.training_losses = checkpoint.get('training_losses', [])
        self.epoch_history = checkpoint.get('epoch_history', [])


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_recognition_model(
    grammar: Grammar,
    model_type: str = 'set_transformer',
    **kwargs
) -> nn.Module:
    """
    Factory function to create recognition model.

    Args:
        grammar: Grammar object
        model_type: 'set_transformer' or 'legacy_gru'
        **kwargs: Model-specific parameters

    Returns:
        Recognition model instance
    """
    if model_type == 'set_transformer':
        return SetTransformerRecognitionModel(grammar, **kwargs)
    elif model_type == 'legacy_gru':
        from dreamcoder_core.neural_recognition_legacy_gru import NeuralRecognitionModel
        return NeuralRecognitionModel(grammar, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# ============================================================================
# DEMO / TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SET TRANSFORMER RECOGNITION MODEL TEST")
    print("=" * 70)

    # Check PyTorch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Import dependencies
    from dreamcoder_core.lean_primitives import build_lean_grammar
    from rules.cards import sample_hand

    # Build grammar
    grammar = build_lean_grammar()
    print(f"\nGrammar size: {len(grammar)} primitives")

    # Create model
    model = SetTransformerRecognitionModel(
        grammar=grammar,
        d_model=64,
        num_heads=4,
        num_hand_layers=2,
        num_task_layers=2,
        max_examples=10,
        max_cards=6
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test encoding
    print("\nTesting card encoding...")
    hand = sample_hand(6)
    features = encode_hand(hand, max_cards=6)
    print(f"  Hand: {hand}")
    print(f"  Features shape: {features.shape}")

    # Test hand encoder
    print("\nTesting hand encoder...")
    hand_enc = model.example_encoder.hand_encoder(
        features.unsqueeze(0)  # Add batch dimension
    )
    print(f"  Hand encoding shape: {hand_enc.shape}")

    # Test full task encoding
    print("\nTesting full model with fake task...")

    from dreamcoder_core.type_system import arrow, HAND, BOOL

    class FakeTask:
        def __init__(self):
            self.name = "test_task"
            self.request_type = arrow(HAND, BOOL)
            self.examples = [(sample_hand(6), True if i % 2 == 0 else False) for i in range(10)]

    task = FakeTask()

    # Test encoding
    task_enc = model.encode_task(task)
    print(f"  Task encoding shape: {task_enc.shape}")
    print(f"  Task encoding norm: {task_enc.norm().item():.4f}")

    # Test predictions
    log_probs = model.predict_primitive_probs(task)
    print(f"  Predictions shape: {log_probs.shape}")
    print(f"  Sum of probs: {torch.exp(log_probs).sum().item():.4f}")

    # Get detailed predictions
    detailed = model.get_primitive_predictions_detailed(task)
    print(f"  Entropy: {detailed['entropy']:.4f}")
    print(f"  Max prob: {detailed['max_prob']:.4f}")
    print(f"  Top 5 primitives:")
    for pred in detailed['top_10'][:5]:
        print(f"    {pred['primitive']}: {pred['prob']:.4f}")

    # Test that different tasks give different predictions
    print("\nTesting task discrimination...")
    task2 = FakeTask()
    task2.name = "test_task_2"
    task2.examples = [(sample_hand(6), True) for _ in range(10)]  # All True

    enc1 = model.encode_task(task)
    enc2 = model.encode_task(task2)

    cosine_sim = F.cosine_similarity(enc1.unsqueeze(0), enc2.unsqueeze(0)).item()
    print(f"  Task encoding cosine similarity: {cosine_sim:.4f}")
    print(f"  (Lower is better - indicates discrimination)")

    # Test grammar generation
    print("\nTesting grammar weight prediction...")
    new_grammar = model.predict_grammar_weights(task)
    print(f"  New grammar size: {len(new_grammar)} primitives")

    print("\n" + "=" * 70)
    print("SET TRANSFORMER RECOGNITION MODEL TEST COMPLETE")
    print("=" * 70)
