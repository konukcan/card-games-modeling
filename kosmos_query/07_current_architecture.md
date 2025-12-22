# Current Recognition Model Architecture

This document describes the current neural recognition model architecture and its components.

## Overview

The recognition model takes a **task** (set of labeled examples) and outputs **log-probabilities over grammar primitives**. These probabilities guide the program enumerator to search likely programs first.

## Input Representation

### Card Encoding (24 dimensions per card)
```python
features = [
    # One-hot suit (4 dims)
    suit == CLUBS, suit == DIAMONDS, suit == HEARTS, suit == SPADES,

    # One-hot rank (13 dims)
    rank == 2, rank == 3, ..., rank == A,

    # One-hot color (2 dims)
    color == RED, color == BLACK,

    # Derived features (5 dims)
    normalized_rank_value,  # (rank - 2) / 12
    is_face_card,           # J, Q, K
    is_ace,
    is_even_rank,
    is_odd_rank
]
```

### Example Encoding
```
Example = (hand: List[Card], label: bool)
```

### Task Encoding
```
Task = {
    name: str,
    examples: List[(hand, label)],  # ~20 examples, balanced True/False
    request_type: hand -> bool
}
```

## Architecture Components

### 1. Hand Encoder (Set Transformer)

```python
class SetTransformerHandEncoder(nn.Module):
    """
    Encodes a hand of 6 cards using self-attention.

    Input: (batch, 6, 24) - card features
    Output: (batch, 64) - hand embedding
    """
    def __init__(self):
        self.card_embed = Linear(24 -> 64)  # Per-card projection
        self.pos_encoding = LearnablePositional(max_len=8, dim=64)
        self.sab1 = SetAttentionBlock(dim=64, heads=4)  # Card-card attention
        self.sab2 = SetAttentionBlock(dim=64, heads=4)
        self.pma = PoolingByMultiheadAttention(dim=64, seeds=1)  # Pool to single vector

    def forward(self, hand_features):
        x = self.card_embed(hand_features)      # (batch, 6, 64)
        x = x + self.pos_encoding(6)            # Add position info
        x = self.sab1(x)                        # Self-attention
        x = self.sab2(x)
        x = self.pma(x)                         # Pool: (batch, 1, 64) -> (batch, 64)
        return x
```

### 2. Example Encoder (FiLM Modulation)

```python
class ExampleEncoder(nn.Module):
    """
    Combines hand embedding with label using FiLM.

    Input: hand_features (batch, 6, 24), label (batch, 2)
    Output: example_embedding (batch, 64)
    """
    def __init__(self):
        self.hand_encoder = SetTransformerHandEncoder()
        self.label_to_gamma = MLP(2 -> 64)  # Scale parameters
        self.label_to_beta = MLP(2 -> 64)   # Shift parameters

    def forward(self, hand_features, label_onehot):
        hand_emb = self.hand_encoder(hand_features)

        # FiLM: gamma * hand + beta
        gamma = self.label_to_gamma(label_onehot)
        beta = self.label_to_beta(label_onehot)
        return gamma * hand_emb + beta
```

### 3. Task Encoder (Raw Feature Correlation)

```python
class TaskEncoder(nn.Module):
    """
    Aggregates examples into task embedding.

    Current approach: compute correlation between raw features and labels.
    This bypasses learned embeddings to preserve discriminative signal.

    Input: example_embeddings (batch, 20, 64), labels (batch, 20)
    Output: task_embedding (batch, 64)
    """
    def __init__(self):
        self.expand = Linear(48 -> 64)  # Expand correlation vector

    def forward(self, examples, labels, raw_features):
        # Compute per-feature correlation with True/False labels
        # raw_features: (batch, 20, 24) - mean-pooled card features per example

        correlation = []
        for feature_dim in range(24):
            feat_values = raw_features[:, :, feature_dim]  # (batch, 20)
            corr = pearson_correlation(feat_values, labels)
            correlation.append(corr)

        # Also compute prototype difference (mean of True - mean of False)
        pos_mask = labels == True
        neg_mask = labels == False
        pos_mean = raw_features[pos_mask].mean(dim=0)  # (24,)
        neg_mean = raw_features[neg_mask].mean(dim=0)  # (24,)
        prototype_diff = pos_mean - neg_mean           # (24,)

        # Concatenate: (24 correlation + 24 prototype diff) = 48
        raw_signal = concat([correlation, prototype_diff])  # (48,)

        # Expand to d_model=64
        return self.expand(raw_signal)  # (64,)
```

### 4. Primitive Predictor

```python
class PrimitivePredictor(nn.Module):
    """
    Maps task embedding to primitive log-probabilities.

    Input: task_embedding (batch, 64)
    Output: log_probs (batch, ~60)
    """
    def __init__(self, num_primitives=60):
        self.mlp = Sequential(
            Linear(64, 128), GELU, Dropout(0.1),
            Linear(128, 128), GELU, Dropout(0.1),
            Linear(128, num_primitives)
        )

    def forward(self, task_embedding):
        logits = self.mlp(task_embedding)
        return log_softmax(logits, dim=-1)
```

## Full Pipeline

```
Task (20 examples) →
  ┌─────────────────────────────────────────────────────────────┐
  │  For each example (hand, label):                            │
  │    hand → Card Features (6×24) → Hand Encoder → hand_emb    │
  │    (hand_emb, label) → FiLM Example Encoder → example_emb   │
  └─────────────────────────────────────────────────────────────┘
  ↓
  Stack examples: (20, 64)
  ↓
  Task Encoder (using raw feature correlation)
  ↓
  task_embedding (64,)
  ↓
  Primitive Predictor
  ↓
  log_probs over ~60 primitives
```

## Training

### Training Signal
- Only solved tasks provide training data
- Target: primitives that appear in the solution program
- Loss: Cross-entropy between predicted distribution and multi-hot target

```python
def training_step(task, solution_primitives):
    task_emb = encode_task(task)
    log_probs = predict_primitives(task_emb)

    # Multi-hot target from solution
    target = zeros(num_primitives)
    for prim in solution_primitives:
        target[prim_to_idx[prim]] = 1
    target = target / target.sum()  # Normalize

    loss = -sum(target * log_probs)  # Cross-entropy
    return loss
```

## Parameter Count

| Component | Parameters |
|-----------|------------|
| Card embedding | 24 × 64 = 1,536 |
| Positional encoding | 8 × 64 = 512 |
| SAB blocks (×2) | ~8,000 each = 16,000 |
| PMA | ~4,000 |
| FiLM (gamma/beta) | 2 × (2 × 64 + 64 × 64) = ~8,500 |
| Task encoder expand | 48 × 64 = 3,072 |
| Primitive predictor | 64 × 128 + 128 × 128 + 128 × 60 = ~32,000 |
| **Total** | **~65,000** |

## What Doesn't Work

Despite this architecture:

1. **Task embeddings collapse**: Different tasks produce similar embeddings
2. **Predictions are uniform**: All tasks get nearly identical primitive distributions
3. **No speedup**: Recognition doesn't accelerate enumeration meaningfully

The raw feature correlation helps somewhat but is insufficient for the task complexity.
