#!/usr/bin/env python3
"""
Neural Recognition Model for DreamCoder

This module implements a TRUE neural recognition model following Ellis et al. (2021):
1. Example Encoder: Encodes each (input, output) example pair
2. Task Encoder: Aggregates example encodings into a task representation
3. Primitive Predictor: Predicts log-probabilities for each grammar primitive

The key insight is that the recognition model learns to:
- Recognize patterns in examples that indicate which primitives are useful
- Generalize across tasks to predict useful primitives for NEW tasks
- Guide enumeration toward more promising parts of the search space

Architecture (following DreamCoder paper):
- Input encoder: GRU over serialized input representation
- Output encoder: GRU over serialized output representation
- Example encoder: MLP combining input and output encodings
- Task encoder: Mean pooling over example encodings
- Primitive head: MLP predicting log-probabilities per primitive
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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.program import Program, Primitive, Application, Abstraction, Index, Invented
from dreamcoder_core.grammar import Grammar, Production
from collections import defaultdict

# Try to import Cython types for compatibility
# This allows the model to work with both Python and Cython programs
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
        """Convert to fixed-size feature vector."""
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
# NEURAL NETWORK COMPONENTS
# ============================================================================

class CardEncoder(nn.Module):
    """
    Encodes a hand of cards using a GRU over card features.

    Architecture:
    - Per-card MLP: card_features -> hidden_dim
    - Bidirectional GRU: sequence of card encodings -> context vectors
    - Pooling: final hidden states -> single vector
    """

    def __init__(
        self,
        card_feature_dim: int = 24,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()

        self.card_feature_dim = card_feature_dim
        self.hidden_dim = hidden_dim

        # Per-card embedding
        self.card_embed = nn.Sequential(
            nn.Linear(card_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Sequence encoder
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Combine bidirectional outputs
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Card features, shape (batch, max_cards, card_feature_dim)
            lengths: Actual sequence lengths, shape (batch,)

        Returns:
            Hand encoding, shape (batch, hidden_dim)
        """
        batch_size, max_cards, _ = x.shape

        # Embed each card
        card_embeds = self.card_embed(x)  # (batch, max_cards, hidden_dim)

        # Run GRU
        if lengths is not None:
            # Pack for variable length sequences
            packed = nn.utils.rnn.pack_padded_sequence(
                card_embeds, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.gru(packed)
        else:
            _, hidden = self.gru(card_embeds)

        # hidden: (num_layers * 2, batch, hidden_dim)
        # Take final layer, concatenate forward and backward
        forward_hidden = hidden[-2]  # (batch, hidden_dim)
        backward_hidden = hidden[-1]  # (batch, hidden_dim)

        combined = torch.cat([forward_hidden, backward_hidden], dim=-1)
        output = self.combine(combined)  # (batch, hidden_dim)

        return output


class ExampleEncoder(nn.Module):
    """
    Encodes a single (input, output) example pair.

    Combines:
    - Input encoding (from CardEncoder)
    - Output encoding (simple embedding)
    """

    def __init__(self, hidden_dim: int = 64, output_dim: int = 2):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.input_encoder = CardEncoder(hidden_dim=hidden_dim)

        self.output_embed = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU()
        )

        # Combine input and output
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(
        self,
        input_features: torch.Tensor,
        output_features: torch.Tensor,
        input_lengths: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_features: (batch, max_cards, card_dim)
            output_features: (batch, output_dim)
            input_lengths: (batch,) actual input lengths

        Returns:
            Example encoding: (batch, hidden_dim)
        """
        input_enc = self.input_encoder(input_features, input_lengths)
        output_enc = self.output_embed(output_features)

        combined = torch.cat([input_enc, output_enc], dim=-1)
        return self.combine(combined)


class TaskEncoder(nn.Module):
    """
    Encodes a task from multiple example encodings.

    Uses attention-weighted pooling over examples.
    """

    def __init__(self, hidden_dim: int = 64):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Attention for pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        # Final projection
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, example_encodings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            example_encodings: (batch, num_examples, hidden_dim)
            mask: (batch, num_examples) True for valid examples

        Returns:
            Task encoding: (batch, hidden_dim)
        """
        # Compute attention weights
        attn_scores = self.attention(example_encodings).squeeze(-1)  # (batch, num_examples)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # (batch, num_examples, 1)

        # Weighted sum
        pooled = (example_encodings * attn_weights).sum(dim=1)  # (batch, hidden_dim)

        return self.project(pooled)


class PrimitivePredictor(nn.Module):
    """
    Predicts log-probabilities for each grammar primitive.

    Takes a task encoding and outputs a vector of log-probabilities,
    one per primitive in the grammar.

    Supports dynamic expansion for newly discovered inventions.
    """

    def __init__(self, hidden_dim: int = 64, num_primitives: int = 61):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_primitives = num_primitives

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_primitives)
        )

    def forward(self, task_encoding: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        """
        Args:
            task_encoding: (batch, hidden_dim)
            return_logits: If True, return raw logits instead of log-probabilities

        Returns:
            Log-probabilities or raw logits: (batch, num_primitives)
        """
        logits = self.predictor(task_encoding)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)

    def expand_for_invention(self, init_weight: Optional[torch.Tensor] = None):
        """
        Add a new output neuron for a discovered invention.

        This allows the model to predict probabilities for newly learned
        abstractions discovered during compression.

        Args:
            init_weight: Optional initialization for the new neuron's weights.
                        If None, uses Xavier uniform initialization.
        """
        # Get the final linear layer
        old_out = self.predictor[-1]
        new_num = self.num_primitives + 1

        # Create new layer with one more output
        new_out = nn.Linear(old_out.in_features, new_num)

        # Copy existing weights
        new_out.weight.data[:self.num_primitives] = old_out.weight.data
        new_out.bias.data[:self.num_primitives] = old_out.bias.data

        # Initialize new neuron
        if init_weight is not None:
            new_out.weight.data[-1] = init_weight
        else:
            nn.init.xavier_uniform_(new_out.weight.data[-1:])
        new_out.bias.data[-1] = 0.0

        # Replace the layer
        self.predictor[-1] = new_out
        self.num_primitives = new_num


# ============================================================================
# BIGRAM/COUNT HEADS
# ============================================================================

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
        input_dim: int = 64,
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

    def __init__(self, input_dim: int = 64, hidden_dim: int = 32):
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
# FULL NEURAL RECOGNITION MODEL
# ============================================================================

class NeuralRecognitionModel(nn.Module):
    """
    Complete neural recognition model for DreamCoder.

    Given a task (list of input-output examples), predicts which
    grammar primitives are likely to be useful.

    Training:
    - On each solved task, the "target" is the set of primitives used in the solution
    - We train to maximize likelihood of those primitives

    Inference:
    - Given new task, predict primitive probabilities
    - Use these to bias the grammar during enumeration
    """

    def __init__(
        self,
        grammar: Grammar,
        hidden_dim: int = 128,
        max_examples: int = 20,
        max_cards: int = 8,
        max_bigrams: int = 50,
        learning_rate: float = 1e-3,
        device: str = 'cpu'
    ):
        super().__init__()

        self.grammar = grammar
        self.hidden_dim = hidden_dim
        self.max_examples = max_examples
        self.max_cards = max_cards
        self.device = device

        # Create primitive name -> index mapping
        self.primitive_names = [str(p.program) for p in grammar.productions]
        self.primitive_to_idx = {name: i for i, name in enumerate(self.primitive_names)}
        self.num_primitives = len(self.primitive_names)

        # Network components
        self.example_encoder = ExampleEncoder(hidden_dim=hidden_dim)
        self.task_encoder = TaskEncoder(hidden_dim=hidden_dim)
        self.primitive_predictor = PrimitivePredictor(
            hidden_dim=hidden_dim,
            num_primitives=self.num_primitives
        )

        # Additional heads (matching ContrastiveRecognitionModel)
        self.bigram_head = BigramHead(hidden_dim, hidden_dim, max_bigrams)
        self.count_head = CountHead(hidden_dim)

        # Bigram vocabulary (populated via build_bigram_vocabulary)
        self.bigram_vocab: List[Tuple[str, str]] = []
        self.bigram_to_idx: Dict[Tuple[str, str], int] = {}

        # Move to device
        self.to(device)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Training history
        self.training_losses: List[float] = []
        self.epoch_history: List[Dict] = []

        # Embedding cache for interpretability
        self._task_embeddings: Dict[str, torch.Tensor] = {}

    def encode_task(self, task) -> torch.Tensor:
        """
        Encode a task into a fixed-size vector.

        Args:
            task: Task object with examples

        Returns:
            Task embedding: (hidden_dim,)
        """
        # Encode each example
        example_encodings = []

        for inp, out in task.examples[:self.max_examples]:
            input_features = encode_hand(inp, self.max_cards).unsqueeze(0).to(self.device)
            output_features = encode_output(out).unsqueeze(0).to(self.device)

            enc = self.example_encoder(input_features, output_features)
            example_encodings.append(enc)

        # Stack and pool
        if example_encodings:
            stacked = torch.stack([e.squeeze(0) for e in example_encodings], dim=0)
            stacked = stacked.unsqueeze(0)  # (1, num_examples, hidden_dim)
            task_enc = self.task_encoder(stacked)
            return task_enc.squeeze(0)
        else:
            return torch.zeros(self.hidden_dim, device=self.device)

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
        Predict raw logits (before softmax) for each primitive given a task.

        This provides the model's "raw intuition" about primitive relevance
        before normalization. Useful for interpretability analysis.

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
        - top_k: List of (primitive_name, log_prob, logit) tuples for top-10
        - entropy: Entropy of the distribution (uncertainty measure)
        - max_prob: Maximum probability (confidence measure)
        """
        self.eval()
        with torch.no_grad():
            task_enc = self.encode_task(task).unsqueeze(0)

            # Get both logits and log-probs
            logits = self.primitive_predictor(task_enc, return_logits=True).squeeze(0)
            log_probs = F.log_softmax(logits, dim=-1)
            probs = torch.exp(log_probs)

            # Compute metrics
            entropy = -torch.sum(probs * log_probs).item()
            max_prob = torch.max(probs).item()

            # Get top-10 predictions
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

    def predict_grammar_weights(self, task, blend_factor: float = 0.5) -> Grammar:
        """
        Given a task, predict which primitives are likely useful.
        Returns a grammar with adjusted weights.

        Args:
            task: The task to predict weights for
            blend_factor: How much to weight neural predictions (0=prior only, 1=neural only)
                         Default 0.5 for backward compatibility. Recommended: 0.7-0.8 for
                         later iterations when recognition model is well-trained.
        """
        log_probs = self.predict_primitive_probs(task)

        # Convert to numpy for grammar construction
        log_probs_np = log_probs.cpu().numpy()

        # Create new productions with predicted weights
        new_productions = []
        for i, prod in enumerate(self.grammar.productions):
            prim_name = str(prod.program)
            if prim_name in self.primitive_to_idx:
                idx = self.primitive_to_idx[prim_name]
                # Blend original and predicted with configurable factor
                new_lp = (1 - blend_factor) * prod.log_probability + blend_factor * log_probs_np[idx]
            else:
                new_lp = prod.log_probability

            new_productions.append(Production(prod.program, prod.tp, new_lp))

        # NOTE: We intentionally do NOT call .normalize_probabilities() here.
        # Normalization is computationally expensive and unnecessary because:
        # 1. The blended weights are already valid log-probabilities
        # 2. TopDownEnumerator uses relative ordering, not absolute values
        # 3. This was causing significant slowdown in recognition-guided enumeration
        return Grammar(new_productions, self.grammar.log_variable)

    def predict_grammar_weights_adaptive(self, task, iteration: int, max_iterations: int) -> Grammar:
        """
        Predict grammar weights with adaptive blending based on training progress.

        Early iterations: Conservative (30% neural, 70% prior) - recognition model is untrained
        Late iterations: Aggressive (80% neural, 20% prior) - recognition model is trained

        This follows the original DreamCoder approach where trust in the recognition
        model increases as it receives more training signal from solved tasks.

        Args:
            task: The task to predict weights for
            iteration: Current iteration number (1-indexed)
            max_iterations: Maximum number of iterations

        Returns:
            Grammar with adaptively blended weights
        """
        # Linear schedule: 0.3 at iteration 1, 0.8 at max_iterations
        blend_factor = 0.3 + (0.8 - 0.3) * ((iteration - 1) / max(1, max_iterations - 1))
        return self.predict_grammar_weights(task, blend_factor=blend_factor)

    def get_primitive_log_probs_dict(self, task) -> Dict[str, float]:
        """
        Get predicted log-probabilities as a serializable dictionary.

        This is useful for passing predictions to worker processes that
        cannot receive the full Grammar object due to pickle issues.

        Args:
            task: The task to predict weights for

        Returns:
            Dictionary mapping primitive names to predicted log-probabilities
        """
        log_probs = self.predict_primitive_probs(task)
        log_probs_np = log_probs.cpu().numpy()

        return {
            name: float(log_probs_np[idx])
            for name, idx in self.primitive_to_idx.items()
        }

    def train_on_frontiers(
        self,
        tasks: List,
        frontiers: Dict,
        epochs: int = 10,
        batch_size: int = 8,
        lambda_count: float = 0.1,
        lambda_bigram: float = 0.1,
        use_bigram_loss: bool = False
    ) -> float:
        """
        Train the recognition model on solved tasks.

        Args:
            tasks: List of Task objects
            frontiers: Dict mapping task names to TaskFrontier objects
            epochs: Number of training epochs
            batch_size: Batch size
            lambda_count: Weight for count prediction loss
            lambda_bigram: Weight for bigram prediction loss
            use_bigram_loss: Whether to include bigram loss (requires bigram_vocab)

        Returns:
            Final training loss
        """
        # Collect training data: (task, target_primitives, program) tuples
        training_data = []

        for task in tasks:
            frontier = frontiers.get(task.name)
            if frontier and frontier.solved:
                # Get primitives and program from solution
                target_primitives = set()
                best_program = None
                for entry in frontier.entries:
                    if entry.log_likelihood == 0.0:  # Perfect solution
                        self._collect_primitives(entry.program, target_primitives)
                        best_program = entry.program
                        break

                if target_primitives:
                    training_data.append((task, target_primitives, best_program))

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

                for task, target_prims, program in batch:
                    # Encode task
                    task_enc = self.encode_task(task).unsqueeze(0)

                    # ==========================================
                    # 1. Primitive prediction loss (cross-entropy)
                    # ==========================================
                    log_probs = self.primitive_predictor(task_enc).squeeze(0)

                    # Target: maximize probability of used primitives
                    target = torch.zeros(self.num_primitives, device=self.device)
                    for prim_name in target_prims:
                        if prim_name in self.primitive_to_idx:
                            target[self.primitive_to_idx[prim_name]] = 1.0

                    # Normalize target
                    if target.sum() > 0:
                        target = target / target.sum()

                    # Cross-entropy loss
                    loss_pred = -torch.sum(target * log_probs)

                    # ==========================================
                    # 2. Count prediction loss (MSE)
                    # ==========================================
                    actual_count = float(len(target_prims))
                    pred_count = self.count_head(task_enc).squeeze()
                    loss_count = (pred_count - actual_count) ** 2

                    # ==========================================
                    # 3. Bigram prediction loss (BCE)
                    # ==========================================
                    loss_bigram = torch.tensor(0.0, device=self.device)
                    if use_bigram_loss and self.bigram_vocab and program is not None:
                        # Extract bigrams from program
                        prog_bigrams = extract_bigrams(program)

                        # Build target vector
                        bigram_target = torch.zeros(len(self.bigram_vocab), device=self.device)
                        for bg in prog_bigrams:
                            if bg in self.bigram_to_idx:
                                bigram_target[self.bigram_to_idx[bg]] = 1.0

                        # Predict bigrams
                        pred_bigrams = self.bigram_head(task_enc).squeeze(0)

                        # BCE loss
                        loss_bigram = F.binary_cross_entropy(
                            pred_bigrams[:len(self.bigram_vocab)],
                            bigram_target,
                            reduction='mean'
                        )

                    # ==========================================
                    # Total loss
                    # ==========================================
                    total_task_loss = loss_pred + lambda_count * loss_count
                    if use_bigram_loss:
                        total_task_loss = total_task_loss + lambda_bigram * loss_bigram

                    batch_loss += total_task_loss

                batch_loss = batch_loss / len(batch)
                batch_loss.backward()
                self.optimizer.step()

                epoch_loss += batch_loss.item()
                n_batches += 1

            self.training_losses.append(epoch_loss / max(1, len(training_data) // batch_size))

        total_loss = sum(self.training_losses[-epochs:]) / epochs if epochs > 0 else 0.0

        # Store epoch info for tracking
        self.epoch_history.append({
            'num_tasks': len(training_data),
            'final_loss': total_loss,
            'epochs': epochs,
            'bigram_training_enabled': use_bigram_loss and bool(self.bigram_vocab)
        })

        # Clear embedding cache after training so subsequent calls
        # to get_task_embedding() use the updated model weights
        self.clear_embedding_cache()

        return total_loss

    def _collect_primitives(self, program, primitives: Set[str]):
        """
        Collect all primitive names used in a program.

        Works with both Python and Cython program types.
        """
        # Check Python types
        if isinstance(program, (Primitive, Invented)):
            primitives.add(str(program))
        elif isinstance(program, Application):
            self._collect_primitives(program.f, primitives)
            self._collect_primitives(program.x, primitives)
        elif isinstance(program, Abstraction):
            self._collect_primitives(program.body, primitives)
        # Check Cython types if available
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
            task: Task object with examples
            use_cache: If True, use cached embedding if available.
                      Default is False to ensure fresh embeddings after training.

        Returns:
            Task embedding tensor of shape (hidden_dim,)

        Note:
            The cache should be cleared after training (via clear_embedding_cache())
            to ensure embeddings reflect the updated model weights.
        """
        if use_cache and task.name in self._task_embeddings:
            return self._task_embeddings[task.name]

        with torch.no_grad():
            embedding = self.encode_task(task).cpu()
            if use_cache:
                self._task_embeddings[task.name] = embedding
            return embedding

    def clear_embedding_cache(self):
        """
        Clear the task embedding cache.

        This should be called after training to ensure that subsequent calls
        to get_task_embedding() return embeddings computed with the updated
        model weights rather than stale cached values.
        """
        self._task_embeddings.clear()

    def get_top_predictions(self, task, n: int = 10) -> List[Tuple[str, float]]:
        """Get top-n predicted primitives for interpretability."""
        log_probs = self.predict_primitive_probs(task)

        # Get top-n indices
        values, indices = torch.topk(log_probs, min(n, self.num_primitives))

        results = []
        for val, idx in zip(values.cpu().numpy(), indices.cpu().numpy()):
            results.append((self.primitive_names[idx], float(val)))

        return results

    # =========================================================================
    # BIGRAM SUPPORT
    # =========================================================================

    def build_bigram_vocabulary_from_programs(self, programs: List[Program], min_count: int = 2):
        """
        Build bigram vocabulary from solved programs.

        Args:
            programs: List of Program objects from solutions
            min_count: Minimum occurrence count to include a bigram
        """
        self.bigram_vocab = build_bigram_vocabulary(
            programs, min_count, self.bigram_head.num_bigrams
        )
        self.bigram_to_idx = {bg: i for i, bg in enumerate(self.bigram_vocab)}

    # =========================================================================
    # EVALUATION METHODS
    # =========================================================================

    def evaluate_count_head(
        self,
        tasks: List,
        frontiers: Dict
    ) -> Dict[str, float]:
        """
        Evaluate CountHead predictions against ground truth.

        Args:
            tasks: List of Task objects
            frontiers: Dict mapping task names to TaskFrontier objects

        Returns:
            Dictionary with evaluation metrics:
            - 'mse': Mean squared error
            - 'mae': Mean absolute error
            - 'correlation': Pearson correlation
            - 'within_1': Fraction of predictions within ±1 of ground truth
            - 'within_2': Fraction within ±2
        """
        import numpy as np
        from scipy.stats import pearsonr

        self.eval()

        predictions = []
        actuals = []

        with torch.no_grad():
            for task in tasks:
                frontier = frontiers.get(task.name)
                if not frontier or not frontier.solved:
                    continue

                # Get actual count from solution
                actual_prims = set()
                for entry in frontier.entries:
                    if entry.log_likelihood == 0.0:
                        self._collect_primitives(entry.program, actual_prims)
                        break

                if not actual_prims:
                    continue

                actual_count = len(actual_prims)

                # Get predicted count
                task_enc = self.encode_task(task).unsqueeze(0)
                pred_count = self.count_head(task_enc).squeeze().item()

                predictions.append(pred_count)
                actuals.append(actual_count)

        if not predictions:
            return {
                'mse': float('inf'), 'mae': float('inf'),
                'correlation': 0.0, 'p_value': 1.0,
                'within_1': 0.0, 'within_2': 0.0,
                'n_tasks': 0
            }

        preds = np.array(predictions)
        acts = np.array(actuals)

        mse = float(np.mean((preds - acts) ** 2))
        mae = float(np.mean(np.abs(preds - acts)))

        if len(preds) > 2:
            try:
                correlation, p_value = pearsonr(preds, acts)
                correlation = float(correlation)
                p_value = float(p_value)
            except Exception:
                correlation, p_value = 0.0, 1.0
        else:
            correlation, p_value = 0.0, 1.0

        within_1 = float(np.mean(np.abs(preds - acts) <= 1))
        within_2 = float(np.mean(np.abs(preds - acts) <= 2))

        return {
            'mse': mse,
            'mae': mae,
            'correlation': correlation,
            'p_value': p_value,
            'within_1': within_1,
            'within_2': within_2,
            'n_tasks': len(predictions),
            'mean_actual': float(np.mean(acts)),
            'mean_predicted': float(np.mean(preds))
        }

    def evaluate_bigram_head(
        self,
        tasks: List,
        frontiers: Dict,
        k: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate BigramHead predictions against ground truth.

        Args:
            tasks: List of Task objects
            frontiers: Dict mapping task names to TaskFrontier objects
            k: Number of top predictions to consider

        Returns:
            Dictionary with evaluation metrics:
            - 'precision@k': Fraction of top-k predictions that are correct
            - 'recall@k': Fraction of true bigrams in top-k predictions
            - 'f1@k': Harmonic mean of precision and recall
        """
        import numpy as np

        if not self.bigram_vocab:
            return {
                'precision@k': 0.0, 'recall@k': 0.0, 'f1@k': 0.0,
                'n_tasks': 0, 'note': 'No bigram vocabulary'
            }

        self.eval()

        precisions = []
        recalls = []

        with torch.no_grad():
            for task in tasks:
                frontier = frontiers.get(task.name)
                if not frontier or not frontier.solved:
                    continue

                # Get ground truth bigrams from solution
                gt_bigrams = set()
                for entry in frontier.entries:
                    if entry.log_likelihood == 0.0:
                        gt_bigrams = extract_bigrams(entry.program)
                        break

                gt_indices = {self.bigram_to_idx[bg] for bg in gt_bigrams
                              if bg in self.bigram_to_idx}

                if not gt_indices:
                    continue

                # Get predictions
                task_enc = self.encode_task(task).unsqueeze(0)
                pred_probs = self.bigram_head(task_enc).squeeze(0)

                # Top-k predictions
                _, top_k_indices = torch.topk(pred_probs, min(k, len(pred_probs)))
                pred_indices = set(top_k_indices.cpu().numpy().tolist())

                # Compute metrics
                correct = len(pred_indices & gt_indices)
                precision = correct / k if k > 0 else 0
                recall = correct / len(gt_indices) if gt_indices else 0

                precisions.append(precision)
                recalls.append(recall)

        if not precisions:
            return {
                'precision@k': 0.0, 'recall@k': 0.0, 'f1@k': 0.0,
                'n_tasks': 0
            }

        mean_precision = float(np.mean(precisions))
        mean_recall = float(np.mean(recalls))
        f1 = (2 * mean_precision * mean_recall / (mean_precision + mean_recall)
              if (mean_precision + mean_recall) > 0 else 0.0)

        return {
            'precision@k': mean_precision,
            'recall@k': mean_recall,
            'f1@k': f1,
            'n_tasks': len(precisions),
            'k': k
        }

    # =========================================================================
    # INVENTION HANDLING
    # =========================================================================

    def add_invention(self, invention: Invented):
        """
        Add a new invention to the model's vocabulary.

        This is called when compression discovers a new abstraction.
        The model expands its output layer to predict probabilities for the
        new invention.

        Args:
            invention: The newly discovered Invented abstraction
        """
        name = str(invention)

        # Already have it?
        if name in self.primitive_to_idx:
            return

        # Expand vocabulary
        self.primitive_names.append(name)
        self.primitive_to_idx[name] = self.num_primitives

        # Expand primitive predictor
        self.primitive_predictor.expand_for_invention()

        self.num_primitives += 1

    def update_grammar(self, new_grammar: Grammar):
        """
        Update the grammar and add any new inventions.

        Args:
            new_grammar: The updated grammar (possibly with new inventions)
        """
        self.grammar = new_grammar

        # Add any new inventions to vocabulary
        for prod in new_grammar.productions:
            name = str(prod.program)
            if name not in self.primitive_to_idx:
                if isinstance(prod.program, Invented):
                    self.add_invention(prod.program)

    def save(self, path: str):
        """Save model state with verification."""
        import os

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'primitive_names': self.primitive_names,
            'hidden_dim': self.hidden_dim,
            'training_losses': self.training_losses,
            'epoch_history': self.epoch_history
        }

        # Save with explicit file handling to ensure data is written
        temp_path = path + '.tmp'
        torch.save(checkpoint, temp_path)

        # Verify file was written
        if os.path.exists(temp_path):
            file_size = os.path.getsize(temp_path)
            if file_size > 0:
                # Atomic rename
                os.replace(temp_path, path)
            else:
                raise RuntimeError(f"Model save failed: file {temp_path} is empty")
        else:
            raise RuntimeError(f"Model save failed: file {temp_path} was not created")

    def load(self, path: str):
        """Load model state, handling vocabulary expansion if needed."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        # Get saved vocabulary info
        saved_primitive_names = checkpoint.get('primitive_names', self.primitive_names)
        saved_num_prims = len(saved_primitive_names)

        # Resize primitive predictor if needed to match checkpoint
        if saved_num_prims != self.num_primitives:
            # Expand to match saved size
            while self.num_primitives < saved_num_prims:
                self.primitive_predictor.expand_for_invention()
                self.num_primitives += 1

        # Update vocabulary
        self.primitive_names = saved_primitive_names
        self.primitive_to_idx = {name: i for i, name in enumerate(self.primitive_names)}

        # Load weights
        self.load_state_dict(checkpoint['model_state_dict'])
        self.training_losses = checkpoint.get('training_losses', [])
        self.epoch_history = checkpoint.get('epoch_history', [])


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def create_training_batch(
    tasks: List,
    frontiers: Dict,
    grammar: Grammar,
    max_examples: int = 20,
    max_cards: int = 8
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a batched training example from solved tasks.

    Returns:
        input_features: (batch, max_examples, max_cards, card_dim)
        output_features: (batch, max_examples, 2)
        target_primitives: (batch, num_primitives) one-hot
    """
    batch_inputs = []
    batch_outputs = []
    batch_targets = []

    primitive_names = [str(p.program) for p in grammar.productions]
    primitive_to_idx = {name: i for i, name in enumerate(primitive_names)}

    for task in tasks:
        frontier = frontiers.get(task.name)
        if not frontier or not frontier.solved:
            continue

        # Encode examples
        task_inputs = []
        task_outputs = []

        for inp, out in task.examples[:max_examples]:
            task_inputs.append(encode_hand(inp, max_cards))
            task_outputs.append(encode_output(out))

        # Pad to max_examples
        while len(task_inputs) < max_examples:
            task_inputs.append(torch.zeros(max_cards, 24))
            task_outputs.append(torch.zeros(2))

        batch_inputs.append(torch.stack(task_inputs))
        batch_outputs.append(torch.stack(task_outputs))

        # Get target primitives
        target = torch.zeros(len(primitive_names))
        for entry in frontier.entries:
            if entry.log_likelihood == 0.0:
                _collect_prims_into_tensor(entry.program, primitive_to_idx, target)

        batch_targets.append(target)

    if not batch_inputs:
        return None, None, None

    return (
        torch.stack(batch_inputs),
        torch.stack(batch_outputs),
        torch.stack(batch_targets)
    )


def _collect_prims_into_tensor(program, prim_to_idx: Dict[str, int], target: torch.Tensor):
    """
    Helper to collect primitives into target tensor.

    Works with both Python and Cython program types.
    """
    # Check Python types
    if isinstance(program, (Primitive, Invented)):
        name = str(program)
        if name in prim_to_idx:
            target[prim_to_idx[name]] = 1.0
    elif isinstance(program, Application):
        _collect_prims_into_tensor(program.f, prim_to_idx, target)
        _collect_prims_into_tensor(program.x, prim_to_idx, target)
    elif isinstance(program, Abstraction):
        _collect_prims_into_tensor(program.body, prim_to_idx, target)
    # Check Cython types if available
    elif CYTHON_AVAILABLE:
        if isinstance(program, (CythonPrimitive, CythonInvented)):
            name = str(program)
            if name in prim_to_idx:
                target[prim_to_idx[name]] = 1.0
        elif isinstance(program, CythonApplication):
            _collect_prims_into_tensor(program.f, prim_to_idx, target)
            _collect_prims_into_tensor(program.x, prim_to_idx, target)
        elif isinstance(program, CythonAbstraction):
            _collect_prims_into_tensor(program.body, prim_to_idx, target)


# ============================================================================
# DEMO / TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("NEURAL RECOGNITION MODEL TEST")
    print("=" * 70)

    # Check PyTorch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Import dependencies
    from dreamcoder_core.lean_primitives import build_lean_grammar as build_card_grammar
    from rules.cards import sample_hand

    # Build grammar
    grammar = build_card_grammar()
    print(f"\nGrammar size: {len(grammar)} primitives")

    # Create model
    model = NeuralRecognitionModel(
        grammar=grammar,
        hidden_dim=64,
        max_examples=10,
        max_cards=6
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test encoding
    print("\nTesting card encoding...")
    hand = sample_hand(6)
    features = encode_hand(hand, max_cards=6)
    print(f"  Hand: {hand}")
    print(f"  Features shape: {features.shape}")

    # Test example encoding
    print("\nTesting example encoder...")
    input_features = features.unsqueeze(0)  # (1, 6, 24)
    output_features = encode_output(True).unsqueeze(0)  # (1, 2)

    example_enc = model.example_encoder(input_features, output_features)
    print(f"  Example encoding shape: {example_enc.shape}")

    # Test task encoding
    print("\nTesting task encoder...")
    # Simulate multiple examples
    example_encs = torch.randn(1, 10, 64)  # (batch, examples, hidden)
    task_enc = model.task_encoder(example_encs)
    print(f"  Task encoding shape: {task_enc.shape}")

    # Test primitive prediction
    print("\nTesting primitive predictor...")
    log_probs = model.primitive_predictor(task_enc)
    print(f"  Log-probs shape: {log_probs.shape}")
    print(f"  Sum of probs: {torch.exp(log_probs).sum().item():.4f}")

    # Test full forward pass
    print("\nTesting full model...")
    from dreamcoder_core.type_system import arrow, HAND, BOOL

    # Create a fake task
    class FakeTask:
        def __init__(self):
            self.name = "test_task"
            self.request_type = arrow(HAND, BOOL)
            self.examples = [(sample_hand(6), True) for _ in range(10)]

    task = FakeTask()

    # Predict
    log_probs = model.predict_primitive_probs(task)
    print(f"  Predictions shape: {log_probs.shape}")

    # Get top predictions
    top_preds = model.get_top_predictions(task, n=5)
    print(f"  Top 5 primitives:")
    for name, score in top_preds:
        print(f"    {name}: {score:.3f}")

    # Test grammar generation
    print("\nTesting grammar weight prediction...")
    new_grammar = model.predict_grammar_weights(task)
    print(f"  New grammar size: {len(new_grammar)} primitives")

    print("\n" + "=" * 70)
    print("NEURAL RECOGNITION MODEL TEST COMPLETE")
    print("=" * 70)
