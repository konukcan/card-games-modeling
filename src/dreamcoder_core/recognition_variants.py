#!/usr/bin/env python3
"""
Recognition Model Architectural Variants

This module implements various architectural improvements to the ContrastiveRecognitionModel
based on theoretical analysis. Each variant can be used independently or combined.

Variants:
1. SelfAttentionHandEncoder - Multi-head self-attention instead of mean pooling
2. EnhancedCardEncoder - Adds color embedding and numeric rank value
3. PrimitiveEmbeddingHead - Learns primitive embeddings for dot-product scoring
4. MultiHeadContrastEncoder - Multiple projection heads for task encoding
5. FocalLoss - Addresses primitive class imbalance
6. DeepSetsEncoder - Proven set-function architecture
7. GatedResidualMLP - Residual connections with gating

Author: Can Konuk
Date: December 2024
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Any


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def hand_to_tensors(hand, max_cards: int = 8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a hand (list of Card objects) to tensor indices.

    Args:
        hand: List of Card objects with .suit and .rank attributes
        max_cards: Maximum number of cards to encode

    Returns:
        suits: (max_cards,) tensor of suit indices
        ranks: (max_cards,) tensor of rank indices
        positions: (max_cards,) tensor of position indices
    """
    # Handle both string and Enum suits
    suit_map = {
        'clubs': 0, 'diamonds': 1, 'hearts': 2, 'spades': 3,
        'CLUBS': 0, 'DIAMONDS': 1, 'HEARTS': 2, 'SPADES': 3
    }
    # Handle both string and Enum ranks
    rank_map = {
        'A': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6,
        '8': 7, '9': 8, '10': 9, 'J': 10, 'Q': 11, 'K': 12
    }

    n_cards = min(len(hand), max_cards)

    suits = torch.zeros(max_cards, dtype=torch.long)
    ranks = torch.zeros(max_cards, dtype=torch.long)
    positions = torch.arange(max_cards, dtype=torch.long)

    for i, card in enumerate(hand[:max_cards]):
        # Handle Enum values by getting .value or using str representation
        suit_val = card.suit.value if hasattr(card.suit, 'value') else str(card.suit)
        rank_val = card.rank.value if hasattr(card.rank, 'value') else str(card.rank)

        suits[i] = suit_map.get(suit_val, suit_map.get(suit_val.upper(), 0))
        ranks[i] = rank_map.get(rank_val, 0)

    return suits, ranks, positions


# ============================================================================
# VARIANT 1: ENHANCED CARD ENCODER (Color + Rank Value)
# ============================================================================

class EnhancedCardEncoder(nn.Module):
    """
    Enhanced card encoder with additional features:
    - Color embedding (2 colors: red, black)
    - Numeric rank value (1-13 as continuous feature)

    Benefits:
    - Color is explicit (no need to learn suit→color mapping)
    - Rank value helps with arithmetic rules (sum, compare)
    """

    def __init__(
        self,
        d_suit: int = 8,
        d_rank: int = 16,
        d_pos: int = 8,
        d_color: int = 4,
        max_pos: int = 8,
        include_rank_value: bool = True
    ):
        super().__init__()

        self.d_suit = d_suit
        self.d_rank = d_rank
        self.d_pos = d_pos
        self.d_color = d_color
        self.include_rank_value = include_rank_value

        # Original embeddings
        self.suit_embed = nn.Embedding(4, d_suit)
        self.rank_embed = nn.Embedding(13, d_rank)
        self.pos_embed = nn.Embedding(max_pos, d_pos)

        # New: Color embedding (0=black for clubs/spades, 1=red for diamonds/hearts)
        self.color_embed = nn.Embedding(2, d_color)

        # Rank value projection (continuous 1-13 → d_rank_val)
        self.rank_value_proj = nn.Linear(1, 4) if include_rank_value else None

        self.output_dim = d_suit + d_rank + d_pos + d_color + (4 if include_rank_value else 0)

        # Initialize
        for embed in [self.suit_embed, self.rank_embed, self.pos_embed, self.color_embed]:
            nn.init.xavier_uniform_(embed.weight)

    def forward(
        self,
        suits: torch.Tensor,
        ranks: torch.Tensor,
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            suits: (batch, max_cards) int indices 0-3 (clubs=0, diamonds=1, hearts=2, spades=3)
            ranks: (batch, max_cards) int indices 0-12
            positions: (batch, max_cards) int indices 0-7

        Returns:
            (batch, max_cards, output_dim) enhanced embeddings
        """
        suit_emb = self.suit_embed(suits)
        rank_emb = self.rank_embed(ranks)
        pos_emb = self.pos_embed(positions)

        # Derive color from suit (clubs=0→black, diamonds=1→red, hearts=2→red, spades=3→black)
        colors = (suits == 1) | (suits == 2)  # True for red (diamonds, hearts)
        colors = colors.long()
        color_emb = self.color_embed(colors)

        embeddings = [suit_emb, rank_emb, pos_emb, color_emb]

        if self.include_rank_value:
            # Normalize rank to [0, 1] range
            rank_values = (ranks.float() + 1) / 13.0  # 1-13 normalized
            rank_val_emb = self.rank_value_proj(rank_values.unsqueeze(-1))
            embeddings.append(rank_val_emb)

        return torch.cat(embeddings, dim=-1)


# ============================================================================
# VARIANT 2: SELF-ATTENTION HAND ENCODER
# ============================================================================

class SelfAttentionHandEncoder(nn.Module):
    """
    Encode a hand using multi-head self-attention.

    Benefits over mean pooling:
    - Cards can attend to each other ("this card is highest")
    - Captures relational patterns (sorted, pairs, sequences)
    - Maintains permutation equivariance (with position embedding)

    Architecture:
    - Multi-head self-attention over card embeddings
    - Followed by mean pooling of attended representations
    """

    def __init__(
        self,
        card_dim: int = 32,
        hidden_dim: int = 64,
        output_dim: int = 32,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.card_dim = card_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Project card embeddings to hidden dim
        self.input_proj = nn.Linear(card_dim, hidden_dim)

        # Self-attention layers
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(n_layers)
        ])

        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(n_layers)
        ])

        # Feed-forward layers
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            for _ in range(n_layers)
        ])

        self.ffn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim)
            for _ in range(n_layers)
        ])

        # Output projection
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(
        self,
        card_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            card_embeddings: (batch, max_cards, card_dim)
            mask: (batch, max_cards) True for real cards, False for padding

        Returns:
            (batch, output_dim) hand embeddings
        """
        # Project to hidden dim
        x = self.input_proj(card_embeddings)  # (batch, max_cards, hidden_dim)

        # Create attention mask (True means ignore)
        if mask is not None:
            attn_mask = ~mask  # Invert: True for padding positions
        else:
            attn_mask = None

        # Apply self-attention layers
        for i, (attn, ln, ffn, ffn_ln) in enumerate(zip(
            self.attention_layers, self.layer_norms,
            self.ffn_layers, self.ffn_norms
        )):
            # Self-attention with residual
            attn_out, _ = attn(x, x, x, key_padding_mask=attn_mask)
            x = ln(x + attn_out)

            # FFN with residual
            ffn_out = ffn(x)
            x = ffn_ln(x + ffn_out)

        # Pool attended representations
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            x = x.mean(dim=1)

        # Project to output
        return self.output_proj(x)


# ============================================================================
# VARIANT 3: DEEP SETS ENCODER
# ============================================================================

class DeepSetsEncoder(nn.Module):
    """
    Deep Sets architecture: ρ(Σ φ(x_i))

    Proven architecture for learning set functions.
    More expressive than simple mean pooling.
    """

    def __init__(
        self,
        card_dim: int = 32,
        phi_hidden: int = 64,
        phi_output: int = 64,
        rho_hidden: int = 64,
        output_dim: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()

        # φ network: per-element transformation
        self.phi = nn.Sequential(
            nn.Linear(card_dim, phi_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(phi_hidden, phi_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(phi_hidden, phi_output)
        )

        # ρ network: post-aggregation transformation
        self.rho = nn.Sequential(
            nn.Linear(phi_output, rho_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(rho_hidden, output_dim)
        )

        self.output_dim = output_dim

    def forward(
        self,
        card_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            card_embeddings: (batch, max_cards, card_dim)
            mask: (batch, max_cards) True for real cards

        Returns:
            (batch, output_dim) hand embeddings
        """
        # Apply φ to each card
        phi_out = self.phi(card_embeddings)  # (batch, max_cards, phi_output)

        # Sum pooling (with mask)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            pooled = (phi_out * mask_expanded).sum(dim=1)
        else:
            pooled = phi_out.sum(dim=1)

        # Apply ρ
        return self.rho(pooled)


# ============================================================================
# VARIANT 4: MULTI-SCALE POOLING
# ============================================================================

class MultiScalePoolingEncoder(nn.Module):
    """
    Concatenate multiple pooling operations:
    - Mean pooling
    - Max pooling
    - Min pooling

    Captures different statistics of the hand.
    """

    def __init__(
        self,
        card_dim: int = 32,
        hidden_dim: int = 64,
        output_dim: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()

        # Card MLP
        self.card_mlp = nn.Sequential(
            nn.Linear(card_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Combine pooled features (3 * hidden_dim → output_dim)
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        self.output_dim = output_dim

    def forward(
        self,
        card_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Apply card MLP
        features = self.card_mlp(card_embeddings)

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            # For max/min, set masked positions to extreme values
            masked_features = features * mask_expanded

            # Mean pooling
            mean_pool = masked_features.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)

            # Max pooling (set masked to -inf)
            max_pool, _ = (features - (1 - mask_expanded) * 1e9).max(dim=1)

            # Min pooling (set masked to +inf)
            min_pool, _ = (features + (1 - mask_expanded) * 1e9).min(dim=1)
        else:
            mean_pool = features.mean(dim=1)
            max_pool, _ = features.max(dim=1)
            min_pool, _ = features.min(dim=1)

        # Concatenate and combine
        pooled = torch.cat([mean_pool, max_pool, min_pool], dim=-1)
        return self.combine(pooled)


# ============================================================================
# VARIANT 5: MULTI-HEAD CONTRAST ENCODER
# ============================================================================

class MultiHeadContrastEncoder(nn.Module):
    """
    Project hands through K different heads, compute K contrastive differences,
    and concatenate.

    Each head can capture a different aspect of the task:
    - Head 1: Color-focused contrast
    - Head 2: Rank-focused contrast
    - Head 3: Position-focused contrast
    """

    def __init__(
        self,
        hand_dim: int = 32,
        n_heads: int = 4,
        output_dim: int = 32
    ):
        super().__init__()

        self.n_heads = n_heads
        head_dim = output_dim // n_heads

        # Projection heads
        self.heads = nn.ModuleList([
            nn.Linear(hand_dim, head_dim)
            for _ in range(n_heads)
        ])

        self.output_dim = head_dim * n_heads

    def forward(
        self,
        pos_embeddings: torch.Tensor,  # (n_pos, hand_dim)
        neg_embeddings: torch.Tensor   # (n_neg, hand_dim)
    ) -> torch.Tensor:
        """
        Compute multi-head contrastive encoding.

        Returns:
            (output_dim,) contrastive encoding
        """
        contrasts = []

        for head in self.heads:
            pos_proj = head(pos_embeddings).mean(dim=0)
            neg_proj = head(neg_embeddings).mean(dim=0)
            contrasts.append(pos_proj - neg_proj)

        return torch.cat(contrasts, dim=-1)


# ============================================================================
# VARIANT 6: PRIMITIVE EMBEDDING HEAD
# ============================================================================

class PrimitiveEmbeddingHead(nn.Module):
    """
    Instead of fixed output weights, learn primitive embeddings.

    Scoring: score_i = τ · E_prim[i] / temperature

    Benefits:
    - Primitives with similar functions cluster
    - Better transfer to new primitives
    - More interpretable embedding space
    """

    def __init__(
        self,
        task_dim: int = 32,
        prim_dim: int = 32,
        num_primitives: int = 67,
        temperature: float = 1.0,
        use_sigmoid: bool = True
    ):
        super().__init__()

        self.num_primitives = num_primitives
        self.temperature = temperature
        self.use_sigmoid = use_sigmoid

        # Project task embedding to primitive space
        self.task_proj = nn.Sequential(
            nn.Linear(task_dim, prim_dim * 2),
            nn.ReLU(),
            nn.Linear(prim_dim * 2, prim_dim)
        )

        # Learned primitive embeddings
        self.prim_embeddings = nn.Parameter(torch.randn(num_primitives, prim_dim))
        nn.init.xavier_uniform_(self.prim_embeddings)

        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(prim_dim)

    def forward(self, τ: torch.Tensor) -> torch.Tensor:
        """
        Args:
            τ: (batch, task_dim) task embeddings

        Returns:
            (batch, num_primitives) scores
        """
        # Project task embedding
        task_proj = self.task_proj(τ)  # (batch, prim_dim)
        task_proj = self.layer_norm(task_proj)

        # Normalize primitive embeddings
        prim_normed = F.normalize(self.prim_embeddings, p=2, dim=-1)

        # Dot product scoring
        scores = torch.matmul(task_proj, prim_normed.t()) / self.temperature

        if self.use_sigmoid:
            return torch.sigmoid(scores)
        else:
            return F.log_softmax(scores, dim=-1)

    def get_primitive_similarity(self) -> torch.Tensor:
        """Get similarity matrix between primitives for interpretability."""
        normed = F.normalize(self.prim_embeddings, p=2, dim=-1)
        return torch.matmul(normed, normed.t())

    def expand_for_invention(self, init_embedding: Optional[torch.Tensor] = None):
        """Add a new primitive embedding."""
        new_num = self.num_primitives + 1
        prim_dim = self.prim_embeddings.shape[1]

        new_embeddings = torch.zeros(new_num, prim_dim, device=self.prim_embeddings.device)
        new_embeddings[:self.num_primitives] = self.prim_embeddings.data

        if init_embedding is not None:
            new_embeddings[-1] = init_embedding
        else:
            nn.init.xavier_uniform_(new_embeddings[-1:])

        self.prim_embeddings = nn.Parameter(new_embeddings)
        self.num_primitives = new_num


# ============================================================================
# VARIANT 7: HIERARCHICAL PREDICTION HEAD
# ============================================================================

class HierarchicalPrimitiveHead(nn.Module):
    """
    Hierarchical prediction: First predict category, then primitive within category.

    Categories:
    - Arithmetic: +, -, *, /, mod, gt, lt, eq, ...
    - List: map, filter, fold, length, ...
    - Card: get_suit, get_rank, get_color, ...
    - Boolean: and, or, not, true, false, ...
    - Constant: numbers 0-13, suits, colors, ...
    """

    def __init__(
        self,
        task_dim: int = 32,
        hidden_dim: int = 64,
        primitive_categories: Dict[str, List[int]] = None,
        num_primitives: int = 67
    ):
        super().__init__()

        self.num_primitives = num_primitives

        # Default categories if not provided
        if primitive_categories is None:
            primitive_categories = {
                'arithmetic': list(range(0, 15)),
                'list': list(range(15, 30)),
                'card': list(range(30, 45)),
                'boolean': list(range(45, 55)),
                'constant': list(range(55, 67))
            }

        self.categories = primitive_categories
        self.n_categories = len(primitive_categories)

        # Category prediction head
        self.category_head = nn.Sequential(
            nn.Linear(task_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.n_categories)
        )

        # Per-category primitive heads
        self.primitive_heads = nn.ModuleDict({
            cat: nn.Sequential(
                nn.Linear(task_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, len(indices))
            )
            for cat, indices in primitive_categories.items()
        })

        # Index mapping
        self.cat_to_prim_idx = primitive_categories
        self.cat_names = list(primitive_categories.keys())

    def forward(self, τ: torch.Tensor) -> torch.Tensor:
        """
        Args:
            τ: (batch, task_dim) task embeddings

        Returns:
            (batch, num_primitives) probabilities
        """
        batch_size = τ.shape[0]
        device = τ.device

        # Predict category probabilities
        cat_logits = self.category_head(τ)  # (batch, n_categories)
        cat_probs = F.softmax(cat_logits, dim=-1)

        # Predict primitive probabilities within each category
        full_probs = torch.zeros(batch_size, self.num_primitives, device=device)

        for i, (cat_name, prim_indices) in enumerate(self.cat_to_prim_idx.items()):
            prim_logits = self.primitive_heads[cat_name](τ)  # (batch, n_prims_in_cat)
            prim_probs = torch.sigmoid(prim_logits)

            # Weight by category probability
            weighted_probs = cat_probs[:, i:i+1] * prim_probs

            for j, prim_idx in enumerate(prim_indices):
                if prim_idx < self.num_primitives:
                    full_probs[:, prim_idx] = weighted_probs[:, j]

        return full_probs


# ============================================================================
# VARIANT 8: GATED RESIDUAL MLP
# ============================================================================

class GatedResidualMLP(nn.Module):
    """
    MLP with gated residual connections.

    output = gate * MLP(x) + (1 - gate) * x

    Allows the network to pass through raw features when transformation isn't helpful.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()

        self.use_residual = (input_dim == output_dim)

        layers = []
        for i in range(n_layers):
            in_d = input_dim if i == 0 else hidden_dim
            out_d = output_dim if i == n_layers - 1 else hidden_dim
            layers.extend([
                nn.Linear(in_d, out_d),
                nn.ReLU() if i < n_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < n_layers - 1 else nn.Identity()
            ])

        self.mlp = nn.Sequential(*layers)

        # Gating mechanism
        if self.use_residual:
            self.gate = nn.Sequential(
                nn.Linear(input_dim, 1),
                nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mlp_out = self.mlp(x)

        if self.use_residual:
            gate = self.gate(x)
            return gate * mlp_out + (1 - gate) * x
        else:
            return mlp_out


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling primitive class imbalance.

    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)

    Down-weights well-classified primitives, focuses on hard ones.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: (batch, num_classes) sigmoid probabilities
            targets: (batch, num_classes) binary targets

        Returns:
            Scalar loss
        """
        # Clamp for numerical stability
        predictions = torch.clamp(predictions, 1e-7, 1 - 1e-7)

        # Compute focal weight
        p_t = predictions * targets + (1 - predictions) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Compute cross-entropy
        bce = -targets * torch.log(predictions) - (1 - targets) * torch.log(1 - predictions)

        # Apply focal weight and alpha
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * focal_weight * bce

        return focal_loss.mean()


class ContrastivePrimitiveLoss(nn.Module):
    """
    Contrastive loss on primitive embeddings.

    Pushes primitives that co-occur together, pulls apart primitives that don't.
    """

    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(
        self,
        prim_embeddings: torch.Tensor,  # (num_prims, dim)
        cooccurrence: torch.Tensor       # (num_prims, num_prims) binary matrix
    ) -> torch.Tensor:
        """Compute contrastive loss on primitive embeddings."""
        # Normalize embeddings
        normed = F.normalize(prim_embeddings, p=2, dim=-1)

        # Compute similarities
        sim = torch.matmul(normed, normed.t()) / self.temperature

        # For co-occurring primitives, maximize similarity
        positive_mask = cooccurrence > 0
        positive_loss = -torch.log(torch.sigmoid(sim[positive_mask])).mean() if positive_mask.any() else 0

        # For non-co-occurring, push apart with margin
        negative_mask = cooccurrence == 0
        # Exclude diagonal
        negative_mask.fill_diagonal_(False)
        negative_loss = torch.relu(sim[negative_mask] - self.margin).mean() if negative_mask.any() else 0

        return positive_loss + negative_loss


# ============================================================================
# FULL VARIANT MODEL
# ============================================================================

class RecognitionModelVariant(nn.Module):
    """
    Flexible recognition model that can use any combination of variants.
    """

    def __init__(
        self,
        grammar,
        card_encoder_type: str = 'standard',  # 'standard', 'enhanced'
        hand_encoder_type: str = 'mean',       # 'mean', 'attention', 'deepsets', 'multiscale'
        task_encoder_type: str = 'standard',   # 'standard', 'multihead'
        prediction_head_type: str = 'sigmoid', # 'sigmoid', 'embedding', 'hierarchical'
        loss_type: str = 'bce',                # 'bce', 'focal'
        card_hidden: int = 128,
        card_out: int = 64,
        pred_hidden: int = 128,
        normalize_embeddings: bool = True,
        embedding_scale: float = 20.0,
        n_attention_heads: int = 4,
        n_contrast_heads: int = 4,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()

        self.grammar = grammar
        self.config = {
            'card_encoder': card_encoder_type,
            'hand_encoder': hand_encoder_type,
            'task_encoder': task_encoder_type,
            'prediction_head': prediction_head_type,
            'loss': loss_type
        }

        # Get primitives
        primitives = list(grammar.primitives())
        self.num_primitives = len(primitives)
        self.primitive_names = [p.name for p in primitives]
        self.primitive_to_idx = {name: i for i, name in enumerate(self.primitive_names)}

        # Normalization settings
        self.normalize_embeddings = normalize_embeddings
        self.embedding_scale = embedding_scale

        # Build card encoder
        if card_encoder_type == 'enhanced':
            self.card_encoder = EnhancedCardEncoder(
                d_suit=8, d_rank=16, d_pos=8, d_color=4,
                include_rank_value=True
            )
            card_dim = self.card_encoder.output_dim
        else:
            from dreamcoder_core.contrastive_recognition import FactoredCardEncoder
            self.card_encoder = FactoredCardEncoder(d_suit=8, d_rank=16, d_pos=8)
            card_dim = 32

        # Build card interaction MLP
        if hand_encoder_type == 'attention':
            # For attention, we project first then apply attention
            self.card_mlp = nn.Sequential(
                nn.Linear(card_dim, card_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(card_hidden, card_out)
            )
            self.hand_encoder = SelfAttentionHandEncoder(
                card_dim=card_out,
                hidden_dim=card_hidden,
                output_dim=card_out,
                n_heads=n_attention_heads,
                n_layers=2
            )
        elif hand_encoder_type == 'deepsets':
            self.card_mlp = nn.Identity()
            self.hand_encoder = DeepSetsEncoder(
                card_dim=card_dim,
                phi_hidden=card_hidden,
                phi_output=card_hidden,
                rho_hidden=card_hidden,
                output_dim=card_out
            )
        elif hand_encoder_type == 'multiscale':
            self.card_mlp = nn.Identity()
            self.hand_encoder = MultiScalePoolingEncoder(
                card_dim=card_dim,
                hidden_dim=card_hidden,
                output_dim=card_out
            )
        else:  # mean pooling (standard)
            # NOTE: No final ReLU - this preserves gradient flow and prevents
            # embedding collapse where all outputs become identical
            self.card_mlp = nn.Sequential(
                nn.Linear(card_dim, card_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(card_hidden, card_out)
                # No activation here - important for diverse embeddings!
            )
            self.hand_encoder = None  # Use inline mean pooling

        # Task encoder
        if task_encoder_type == 'multihead':
            self.task_encoder = MultiHeadContrastEncoder(
                hand_dim=card_out,
                n_heads=n_contrast_heads,
                output_dim=card_out
            )
            task_dim = self.task_encoder.output_dim
        else:
            self.task_encoder = None
            task_dim = card_out

        # Store task_dim for use in encode_task_batched
        self.task_dim = task_dim

        # Embedding normalization
        if normalize_embeddings:
            self.embedding_norm = nn.LayerNorm(task_dim)
            self.embedding_scale_param = nn.Parameter(torch.tensor(embedding_scale))

        # Prediction head
        if prediction_head_type == 'embedding':
            self.primitive_head = PrimitiveEmbeddingHead(
                task_dim=task_dim,
                prim_dim=pred_hidden,
                num_primitives=self.num_primitives,
                temperature=1.0,
                use_sigmoid=True
            )
        elif prediction_head_type == 'hierarchical':
            self.primitive_head = HierarchicalPrimitiveHead(
                task_dim=task_dim,
                hidden_dim=pred_hidden,
                num_primitives=self.num_primitives
            )
        else:  # sigmoid (standard)
            self.primitive_head = nn.Sequential(
                nn.Linear(task_dim, pred_hidden),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(pred_hidden, self.num_primitives)
            )

        # Loss function
        if loss_type == 'focal':
            self.loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss() if prediction_head_type == 'sigmoid' else nn.BCELoss()

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

    def encode_hand(self, hand) -> torch.Tensor:
        """Encode a single hand into a vector."""
        suits, ranks, positions = hand_to_tensors(hand)
        suits = suits.unsqueeze(0).to(self.device)
        ranks = ranks.unsqueeze(0).to(self.device)
        positions = positions.unsqueeze(0).to(self.device)

        # Get card embeddings
        card_emb = self.card_encoder(suits, ranks, positions)

        if self.hand_encoder is not None:
            # Process through card MLP first
            card_features = self.card_mlp(card_emb)
            return self.hand_encoder(card_features).squeeze(0)
        else:
            # Standard mean pooling
            card_features = self.card_mlp(card_emb)
            return card_features.mean(dim=1).squeeze(0)

    def encode_task_batched(self, task) -> torch.Tensor:
        """Encode a task into a contrastive embedding."""
        pos_hands = [h for h, label in task.examples if label]
        neg_hands = [h for h, label in task.examples if not label]

        if not pos_hands or not neg_hands:
            return torch.zeros(self.task_dim, device=self.device)

        # Encode hands
        pos_embeddings = torch.stack([self.encode_hand(h) for h in pos_hands])
        neg_embeddings = torch.stack([self.encode_hand(h) for h in neg_hands])

        # Compute contrastive encoding
        if self.task_encoder is not None:
            τ = self.task_encoder(pos_embeddings, neg_embeddings)
        else:
            τ = pos_embeddings.mean(dim=0) - neg_embeddings.mean(dim=0)

        # Apply normalization
        if self.normalize_embeddings:
            τ = self.embedding_norm(τ.unsqueeze(0)).squeeze(0)
            τ = τ * self.embedding_scale_param

        return τ

    def predict_primitives(self, task) -> torch.Tensor:
        """Predict primitive probabilities for a task."""
        τ = self.encode_task_batched(task)

        if isinstance(self.primitive_head, nn.Sequential):
            logits = self.primitive_head(τ.unsqueeze(0))
            return torch.sigmoid(logits).squeeze(0)
        else:
            return self.primitive_head(τ.unsqueeze(0)).squeeze(0)

    def compute_loss(self, task) -> torch.Tensor:
        """Compute training loss for a task."""
        # Get ground truth primitives
        gt_prims = set()
        primitives = self.grammar.primitives()
        if hasattr(task, 'rule') and hasattr(task.rule, 'program') and task.rule.program:
            program_str = str(task.rule.program)
            for i, prim in enumerate(primitives):
                if prim.name in program_str:
                    gt_prims.add(i)

        # Create target vector
        target = torch.zeros(self.num_primitives, device=self.device)
        for idx in gt_prims:
            target[idx] = 1.0

        # Get predictions
        τ = self.encode_task_batched(task)

        if isinstance(self.primitive_head, nn.Sequential):
            logits = self.primitive_head(τ.unsqueeze(0))
            if isinstance(self.loss_fn, FocalLoss):
                probs = torch.sigmoid(logits)
                loss = self.loss_fn(probs, target.unsqueeze(0))
            else:
                loss = self.loss_fn(logits, target.unsqueeze(0))
        else:
            probs = self.primitive_head(τ.unsqueeze(0))
            if isinstance(self.loss_fn, FocalLoss):
                loss = self.loss_fn(probs, target.unsqueeze(0))
            else:
                # BCE loss expects probabilities in [0, 1]
                loss = self.loss_fn(probs, target.unsqueeze(0))

        return loss

    def get_config_string(self) -> str:
        """Get a short config string for logging."""
        return f"{self.config['card_encoder']}_{self.config['hand_encoder']}_{self.config['task_encoder']}_{self.config['prediction_head']}_{self.config['loss']}"
