#!/usr/bin/env python3
"""
DreamCoder Recognition Network - Full Integration

This module integrates the recognition network with the full rule catalogue (57 rules)
and complete primitive vocabulary (60+ primitives). It provides:

1. Task generation from the rule catalogue
2. Feature extraction for card hands
3. Neural network training for primitive prediction
4. Per-rule accuracy metrics
5. Visualization of predictions and embeddings

Based on Ellis et al. (2023) DreamCoder architecture, adapted for card game domain.

Usage:
    python src/dreamcoder/recognition.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Dict, Optional, Set
from dataclasses import dataclass
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our domain
from rules.cards import (
    Card, Hand, Suit, Rank, Color, AltColor1, AltColor2, Parity,
    sample_hand, hand_to_string, RANK_VALUES,
    card_color, suit_to_altcolor1, suit_to_altcolor2, rank_parity
)
from rules.catalogue import ALL_RULES, Rule

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# ============================================================================
# PRIMITIVE VOCABULARY
# ============================================================================

def build_primitive_vocabulary() -> Tuple[List[str], Dict[str, int]]:
    """
    Build the complete primitive vocabulary from all rules.
    Returns: (list of primitive names, name->index mapping)
    """
    all_primitives: Set[str] = set()

    for rule in ALL_RULES:
        all_primitives.update(rule.primitives_used)

    # Sort for consistency
    primitive_list = sorted(all_primitives)
    primitive_to_idx = {name: idx for idx, name in enumerate(primitive_list)}

    return primitive_list, primitive_to_idx


PRIMITIVE_LIST, PRIMITIVE_TO_IDX = build_primitive_vocabulary()
NUM_PRIMITIVES = len(PRIMITIVE_LIST)

print(f"Primitive vocabulary: {NUM_PRIMITIVES} primitives")
print(f"Sample primitives: {PRIMITIVE_LIST[:10]}...")


# ============================================================================
# FEATURIZATION: Convert Cards to Vectors
# ============================================================================

def featurize_card(card: Card) -> np.ndarray:
    """
    Convert a card to a feature vector (21 dimensions).

    Features:
    - Rank one-hot (13 dims): A, 2, 3, ..., K
    - Suit one-hot (4 dims): Clubs, Diamonds, Hearts, Spades
    - Color (1 dim): 0=Black, 1=Red
    - Parity (1 dim): 0=Even, 1=Odd
    - AltColor1 (1 dim): 0=Pointy, 1=Round
    - AltColor2 (1 dim): 0=SH, 1=DC
    """
    features = np.zeros(21, dtype=np.float32)

    # Rank one-hot (13 dims)
    rank_idx = list(Rank).index(card.rank)
    features[rank_idx] = 1.0

    # Suit one-hot (4 dims)
    suit_idx = list(Suit).index(card.suit)
    features[13 + suit_idx] = 1.0

    # Color (1 dim)
    features[17] = 1.0 if card_color(card) == Color.RED else 0.0

    # Parity (1 dim)
    features[18] = 1.0 if rank_parity(card.rank) == Parity.ODD else 0.0

    # AltColor1 (1 dim)
    features[19] = 1.0 if suit_to_altcolor1(card.suit) == AltColor1.ROUND else 0.0

    # AltColor2 (1 dim)
    features[20] = 1.0 if suit_to_altcolor2(card.suit) == AltColor2.DC else 0.0

    return features


def featurize_hand(hand: Hand, max_cards: int = 6) -> np.ndarray:
    """
    Convert a hand to a feature vector.

    Features per card: 21
    Total: 21 * max_cards = 126 dims

    Plus global features (30 dims):
    - Rank statistics (mean, std, min, max): 4
    - Suit counts: 4
    - Color balance: 2
    - Sorted indicator: 1
    - Has pair indicator: 1
    - Palindrome indicators (suit, color, rank): 3
    - Terminal equality (suit, color, rank): 3
    - Halves similarity (suit, color, rank): 3
    - Position-specific features: 9
    """
    card_features = []

    # Per-card features
    for i in range(max_cards):
        if i < len(hand):
            card_features.extend(featurize_card(hand[i]))
        else:
            card_features.extend(np.zeros(21))

    card_features = np.array(card_features, dtype=np.float32)

    # Global features
    global_features = np.zeros(30, dtype=np.float32)

    if hand:
        # Rank statistics
        rank_vals = [RANK_VALUES[c.rank] for c in hand]
        global_features[0] = np.mean(rank_vals) / 14.0  # Normalized
        global_features[1] = np.std(rank_vals) / 7.0 if len(rank_vals) > 1 else 0.0
        global_features[2] = min(rank_vals) / 14.0
        global_features[3] = max(rank_vals) / 14.0

        # Suit counts (normalized)
        for i, suit in enumerate(Suit):
            count = sum(1 for c in hand if c.suit == suit)
            global_features[4 + i] = count / len(hand)

        # Color balance
        red_count = sum(1 for c in hand if card_color(c) == Color.RED)
        global_features[8] = red_count / len(hand)
        global_features[9] = 1.0 if red_count == len(hand) or red_count == 0 else 0.0  # Uniform

        # Sorted indicator
        global_features[10] = 1.0 if all(
            RANK_VALUES[hand[i].rank] <= RANK_VALUES[hand[i+1].rank]
            for i in range(len(hand)-1)
        ) else 0.0

        # Has pair indicator
        global_features[11] = 1.0 if len(set(c.rank for c in hand)) < len(hand) else 0.0

        # Palindrome indicators
        suits = [c.suit for c in hand]
        global_features[12] = 1.0 if suits == suits[::-1] else 0.0

        colors = [card_color(c) for c in hand]
        global_features[13] = 1.0 if colors == colors[::-1] else 0.0

        ranks = [c.rank for c in hand]
        global_features[14] = 1.0 if ranks == ranks[::-1] else 0.0

        # Terminal equality
        if len(hand) >= 2:
            global_features[15] = 1.0 if hand[0].suit == hand[-1].suit else 0.0
            global_features[16] = 1.0 if card_color(hand[0]) == card_color(hand[-1]) else 0.0
            global_features[17] = 1.0 if hand[0].rank == hand[-1].rank else 0.0

        # Halves similarity (for 6-card hands)
        if len(hand) == 6:
            left = hand[:3]
            right = hand[3:]

            left_suits = [c.suit for c in left]
            right_suits = [c.suit for c in right]
            global_features[18] = 1.0 if left_suits == right_suits else 0.0

            left_colors = [card_color(c) for c in left]
            right_colors = [card_color(c) for c in right]
            global_features[19] = 1.0 if left_colors == right_colors else 0.0

            left_ranks = [RANK_VALUES[c.rank] for c in left]
            right_ranks = [RANK_VALUES[c.rank] for c in right]
            global_features[20] = 1.0 if left_ranks == right_ranks else 0.0

        # Position-specific features (first 3 positions)
        for i in range(min(3, len(hand))):
            global_features[21 + i*3] = RANK_VALUES[hand[i].rank] / 14.0
            global_features[22 + i*3] = list(Suit).index(hand[i].suit) / 3.0
            global_features[23 + i*3] = 1.0 if card_color(hand[i]) == Color.RED else 0.0

    return np.concatenate([card_features, global_features])


def featurize_example(hand: Hand, label: bool) -> np.ndarray:
    """Featurize a single example (hand + label)."""
    hand_features = featurize_hand(hand)
    label_feature = np.array([1.0, 0.0] if label else [0.0, 1.0], dtype=np.float32)
    return np.concatenate([hand_features, label_feature])


# Feature dimension
EXAMPLE_FEATURE_DIM = 6 * 21 + 30 + 2  # 158 dims


# ============================================================================
# TASK GENERATION
# ============================================================================

@dataclass
class Task:
    """A task consists of examples for one rule."""
    rule_id: str
    rule_token: str
    examples: List[Tuple[Hand, bool]]
    primitive_indices: List[int]

    def to_tensor(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert to PyTorch tensors."""
        # Featurize examples
        example_features = [featurize_example(h, l) for h, l in self.examples]
        examples_tensor = torch.tensor(np.stack(example_features), dtype=torch.float32)

        # Multi-hot target
        target = torch.zeros(NUM_PRIMITIVES, dtype=torch.float32)
        for idx in self.primitive_indices:
            target[idx] = 1.0

        return examples_tensor, target


def generate_task(rule: Rule, num_examples: int = 8) -> Task:
    """
    Generate a task with balanced positive/negative examples for a rule.
    """
    examples = []
    pos_count = 0
    neg_count = 0
    target_per_class = num_examples // 2

    max_attempts = 2000
    attempts = 0

    while (pos_count < target_per_class or neg_count < target_per_class) and attempts < max_attempts:
        hand = sample_hand(6)
        try:
            label = rule.eval(hand)

            if label and pos_count < target_per_class:
                examples.append((hand, True))
                pos_count += 1
            elif not label and neg_count < target_per_class:
                examples.append((hand, False))
                neg_count += 1
        except Exception:
            pass

        attempts += 1

    # Pad if necessary
    while len(examples) < num_examples:
        hand = sample_hand(6)
        try:
            label = rule.eval(hand)
            examples.append((hand, label))
        except:
            examples.append((hand, False))

    # Get primitive indices
    primitive_indices = [
        PRIMITIVE_TO_IDX[p] for p in rule.primitives_used
        if p in PRIMITIVE_TO_IDX
    ]

    return Task(
        rule_id=rule.id,
        rule_token=rule.token,
        examples=examples[:num_examples],
        primitive_indices=primitive_indices
    )


# ============================================================================
# PYTORCH DATASET
# ============================================================================

class CardRuleDataset(Dataset):
    """Dataset of card rule tasks."""

    def __init__(self, tasks: List[Task]):
        self.tasks = tasks

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]
        examples_tensor, target = task.to_tensor()
        return examples_tensor, target, task.rule_id


# ============================================================================
# NEURAL ARCHITECTURE
# ============================================================================

class ExampleEncoder(nn.Module):
    """Encodes a single example into an embedding."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)


class AttentionAggregator(nn.Module):
    """Aggregates example embeddings using self-attention."""

    def __init__(self, embedding_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embedding_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, example_embeddings):
        """
        Args:
            example_embeddings: (batch_size, num_examples, embedding_dim)
        Returns:
            task_embedding: (batch_size, embedding_dim)
        """
        # Self-attention
        attn_output, _ = self.attention(
            example_embeddings, example_embeddings, example_embeddings
        )

        # Residual + norm
        attn_output = self.norm(attn_output + example_embeddings)

        # Max-pool for final representation
        task_embedding = torch.max(attn_output, dim=1)[0]

        return task_embedding


class RecognitionNetwork(nn.Module):
    """
    Full recognition network: task examples → primitive distribution.

    Architecture:
    1. Example Encoder: Encode each (hand, label) example
    2. Attention Aggregator: Combine examples (permutation-invariant)
    3. Primitive Predictor: Predict probability of each primitive
    """

    def __init__(self,
                 example_feature_dim: int,
                 embedding_dim: int = 128,
                 num_primitives: int = NUM_PRIMITIVES):
        super().__init__()

        self.example_encoder = ExampleEncoder(
            input_dim=example_feature_dim,
            hidden_dim=256,
            output_dim=embedding_dim
        )

        self.aggregator = AttentionAggregator(embedding_dim, num_heads=4)

        self.primitive_predictor = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_primitives),
        )

    def forward(self, examples_batch, return_intermediates=False):
        """
        Args:
            examples_batch: (batch_size, num_examples, feature_dim)
        Returns:
            logits: (batch_size, num_primitives)
        """
        batch_size, num_examples, feature_dim = examples_batch.shape

        # Encode each example
        examples_flat = examples_batch.view(-1, feature_dim)
        embeddings_flat = self.example_encoder(examples_flat)
        embedding_dim = embeddings_flat.shape[-1]
        example_embeddings = embeddings_flat.view(batch_size, num_examples, embedding_dim)

        # Aggregate
        task_embedding = self.aggregator(example_embeddings)

        # Predict primitives
        logits = self.primitive_predictor(task_embedding)

        if return_intermediates:
            return {
                'logits': logits,
                'example_embeddings': example_embeddings,
                'task_embedding': task_embedding
            }

        return logits


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    for examples, targets, _ in dataloader:
        examples = examples.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(examples)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device) -> Tuple[float, float, Dict[str, Dict]]:
    """
    Evaluate model and return per-rule metrics.

    Returns:
        avg_loss: Average loss
        avg_accuracy: Overall primitive prediction accuracy
        rule_metrics: Per-rule accuracy and predictions
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    all_predictions = []
    all_targets = []
    rule_predictions = {}  # rule_id -> list of predictions

    with torch.no_grad():
        for examples, targets, rule_ids in dataloader:
            examples = examples.to(device)
            targets = targets.to(device)

            logits = model(examples)
            loss = criterion(logits, targets)

            total_loss += loss.item()
            num_batches += 1

            # Get predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            all_predictions.append(preds.cpu())
            all_targets.append(targets.cpu())

            # Store per-rule
            for i, rule_id in enumerate(rule_ids):
                if rule_id not in rule_predictions:
                    rule_predictions[rule_id] = {
                        'predictions': [],
                        'targets': [],
                        'probs': []
                    }
                rule_predictions[rule_id]['predictions'].append(preds[i].cpu())
                rule_predictions[rule_id]['targets'].append(targets[i].cpu())
                rule_predictions[rule_id]['probs'].append(probs[i].cpu())

    avg_loss = total_loss / num_batches

    # Calculate overall accuracy
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    accuracy = (all_predictions == all_targets).float().mean().item()

    # Calculate per-rule metrics
    rule_metrics = {}
    for rule_id, data in rule_predictions.items():
        preds = torch.stack(data['predictions'])
        tgts = torch.stack(data['targets'])
        probs = torch.stack(data['probs'])

        # Per-primitive accuracy for this rule
        rule_acc = (preds == tgts).float().mean().item()

        # Get average prediction probabilities
        avg_probs = probs.mean(dim=0).numpy()

        rule_metrics[rule_id] = {
            'accuracy': rule_acc,
            'avg_probs': avg_probs,
            'num_samples': len(data['predictions'])
        }

    return avg_loss, accuracy, rule_metrics


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_recognition_network(
    num_tasks_per_rule: int = 100,
    num_epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    device: str = None
) -> Tuple[RecognitionNetwork, Dict]:
    """
    Train the recognition network on all rules.

    Returns:
        model: Trained model
        results: Training history and per-rule metrics
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("TRAINING DREAMCODER RECOGNITION NETWORK")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Rules: {len(ALL_RULES)}")
    print(f"  Primitives: {NUM_PRIMITIVES}")
    print(f"  Tasks per rule: {num_tasks_per_rule}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Device: {device}")
    print()

    # Generate training data
    print("Generating training tasks...")
    all_tasks = []
    for rule in ALL_RULES:
        for _ in range(num_tasks_per_rule):
            task = generate_task(rule, num_examples=8)
            all_tasks.append(task)

    print(f"  Total tasks: {len(all_tasks)}")

    # Create dataset and split
    dataset = CardRuleDataset(all_tasks)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    print()

    # Initialize model
    model = RecognitionNetwork(
        example_feature_dim=EXAMPLE_FEATURE_DIM,
        embedding_dim=128,
        num_primitives=NUM_PRIMITIVES
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print()

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    # Training loop
    print("Training...")
    train_losses = []
    val_losses = []
    val_accuracies = []
    best_accuracy = 0

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        scheduler.step(val_loss)

        if val_acc > best_accuracy:
            best_accuracy = val_acc

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{num_epochs}: "
                  f"Train Loss={train_loss:.4f}, "
                  f"Val Loss={val_loss:.4f}, "
                  f"Val Acc={val_acc:.4f}")

    print()
    print(f"Training complete! Best validation accuracy: {best_accuracy:.4f}")

    # Final evaluation with per-rule metrics
    print("\nComputing per-rule metrics...")
    _, final_acc, rule_metrics = evaluate(model, val_loader, criterion, device)

    results = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'final_accuracy': final_acc,
        'best_accuracy': best_accuracy,
        'rule_metrics': rule_metrics,
        'num_primitives': NUM_PRIMITIVES,
        'num_rules': len(ALL_RULES),
        'primitive_list': PRIMITIVE_LIST,
    }

    return model, results


# ============================================================================
# SAVE/LOAD FUNCTIONS
# ============================================================================

def save_model(model: RecognitionNetwork, results: Dict, path: Path):
    """Save model and results."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'results': results,
        'primitive_list': PRIMITIVE_LIST,
        'primitive_to_idx': PRIMITIVE_TO_IDX,
    }, path)
    print(f"Model saved to: {path}")


def load_model(path: Path, device: str = 'cpu') -> Tuple[RecognitionNetwork, Dict]:
    """Load model and results."""
    checkpoint = torch.load(path, map_location=device)

    model = RecognitionNetwork(
        example_feature_dim=EXAMPLE_FEATURE_DIM,
        embedding_dim=128,
        num_primitives=len(checkpoint['primitive_list'])
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])

    return model, checkpoint['results']


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_primitives(
    model: RecognitionNetwork,
    rule: Rule,
    num_examples: int = 8,
    device: str = 'cpu'
) -> Dict:
    """
    Predict primitives for a rule based on generated examples.

    Returns:
        predictions: Dict with probabilities and predicted primitives
    """
    model.eval()

    # Generate task
    task = generate_task(rule, num_examples)
    examples_tensor, _ = task.to_tensor()
    examples_tensor = examples_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(examples_tensor, return_intermediates=True)
        logits = outputs['logits'][0]
        probs = torch.sigmoid(logits).cpu().numpy()
        task_embedding = outputs['task_embedding'][0].cpu().numpy()

    # Get predicted primitives (threshold 0.5)
    predicted_indices = np.where(probs > 0.5)[0]
    predicted_primitives = [PRIMITIVE_LIST[i] for i in predicted_indices]

    # Get top-k predictions
    top_k = 10
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_primitives = [(PRIMITIVE_LIST[i], float(probs[i])) for i in top_indices]

    return {
        'rule_id': rule.id,
        'probabilities': probs,
        'predicted_primitives': predicted_primitives,
        'ground_truth_primitives': rule.primitives_used,
        'top_predictions': top_primitives,
        'task_embedding': task_embedding,
        'accuracy': len(set(predicted_primitives) & set(rule.primitives_used)) / len(rule.primitives_used)
            if rule.primitives_used else 0.0
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Train the model
    model, results = train_recognition_network(
        num_tasks_per_rule=100,
        num_epochs=30,
        batch_size=32,
        learning_rate=0.001
    )

    # Save model
    model_path = output_dir / "recognition_model.pt"
    save_model(model, results, model_path)

    # Save results as JSON (without numpy arrays)
    json_results = {
        'final_accuracy': results['final_accuracy'],
        'best_accuracy': results['best_accuracy'],
        'train_losses': results['train_losses'],
        'val_losses': results['val_losses'],
        'val_accuracies': results['val_accuracies'],
        'num_primitives': results['num_primitives'],
        'num_rules': results['num_rules'],
        'primitive_list': results['primitive_list'],
        'rule_metrics': {
            rule_id: {
                'accuracy': float(metrics['accuracy']),
                'num_samples': metrics['num_samples']
            }
            for rule_id, metrics in results['rule_metrics'].items()
        }
    }

    with open(output_dir / "recognition_results.json", 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"Results saved to: {output_dir / 'recognition_results.json'}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(results['train_losses'], label='Train Loss', linewidth=2)
    axes[0].plot(results['val_losses'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (BCE)', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(results['val_accuracies'], label='Val Accuracy', color='green', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].axhline(results['best_accuracy'], color='red', linestyle='--',
                    label=f'Best: {results["best_accuracy"]:.4f}')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'recognition_training_curves.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: recognition_training_curves.png")
    plt.close()

    # 2. Per-rule accuracy distribution
    rule_accuracies = [m['accuracy'] for m in results['rule_metrics'].values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(rule_accuracies, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(rule_accuracies), color='red', linestyle='--',
               label=f'Mean: {np.mean(rule_accuracies):.3f}')
    ax.set_xlabel('Per-Rule Accuracy', fontsize=12)
    ax.set_ylabel('Number of Rules', fontsize=12)
    ax.set_title('Distribution of Per-Rule Prediction Accuracy', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'recognition_rule_accuracy_dist.png', dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: recognition_rule_accuracy_dist.png")
    plt.close()

    print("\n" + "=" * 70)
    print("RECOGNITION NETWORK TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nFinal Results:")
    print(f"  Overall Accuracy: {results['final_accuracy']:.4f}")
    print(f"  Best Accuracy: {results['best_accuracy']:.4f}")
    print(f"  Mean Per-Rule Accuracy: {np.mean(rule_accuracies):.4f}")
    print(f"\nFiles saved to: {output_dir}/")
