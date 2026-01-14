#!/usr/bin/env python3
"""
Diagnostic script to understand why PrimitiveEmbeddingHead fails.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.recognition_variants import (
    RecognitionModelVariant,
    PrimitiveEmbeddingHead
)
from dreamcoder_core.primitives import build_lean_grammar
from rules.catalogue import create_all_rules
from rules.cards import sample_hand

def diagnose():
    grammar = build_lean_grammar()
    rules = create_all_rules()

    # Create a model with embedding head
    model = RecognitionModelVariant(
        grammar=grammar,
        card_encoder_type='standard',
        hand_encoder_type='mean',
        task_encoder_type='standard',
        prediction_head_type='embedding',
        loss_type='bce'
    )

    # Get a sample rule
    rule = rules[0]

    # Sample some hands
    pos_hands = []
    neg_hands = []
    for _ in range(100):
        hand = sample_hand(5)
        try:
            if rule.predicate(hand):
                if len(pos_hands) < 10:
                    pos_hands.append(hand)
            else:
                if len(neg_hands) < 10:
                    neg_hands.append(hand)
        except:
            pass
        if len(pos_hands) >= 10 and len(neg_hands) >= 10:
            break

    print(f"Rule: {rule.id}")
    print(f"Positive hands: {len(pos_hands)}, Negative hands: {len(neg_hands)}")

    # Encode the task
    class DummyTask:
        def __init__(self, examples):
            self.examples = examples

    task = DummyTask([(h, True) for h in pos_hands] + [(h, False) for h in neg_hands])

    model.eval()
    with torch.no_grad():
        τ = model.encode_task_batched(task)
        print(f"\nTask embedding τ:")
        print(f"  Shape: {τ.shape}")
        print(f"  Mean: {τ.mean().item():.4f}")
        print(f"  Std: {τ.std().item():.4f}")
        print(f"  Min: {τ.min().item():.4f}")
        print(f"  Max: {τ.max().item():.4f}")
        print(f"  L2 norm: {torch.norm(τ).item():.4f}")

        # Look inside the embedding head
        head = model.primitive_head
        if isinstance(head, PrimitiveEmbeddingHead):
            print("\n--- Inside PrimitiveEmbeddingHead ---")

            # Task projection
            task_proj = head.task_proj(τ.unsqueeze(0))
            print(f"\nAfter task_proj (before layer norm):")
            print(f"  Mean: {task_proj.mean().item():.4f}")
            print(f"  Std: {task_proj.std().item():.4f}")
            print(f"  L2 norm: {torch.norm(task_proj).item():.4f}")

            # After layer norm
            task_proj_normed = head.layer_norm(task_proj)
            print(f"\nAfter layer norm:")
            print(f"  Mean: {task_proj_normed.mean().item():.4f}")
            print(f"  Std: {task_proj_normed.std().item():.4f}")
            print(f"  L2 norm: {torch.norm(task_proj_normed).item():.4f}")

            # Primitive embeddings
            print(f"\nPrimitive embeddings:")
            print(f"  Shape: {head.prim_embeddings.shape}")
            print(f"  Mean: {head.prim_embeddings.mean().item():.4f}")
            print(f"  Std: {head.prim_embeddings.std().item():.4f}")

            prim_normed = F.normalize(head.prim_embeddings, p=2, dim=-1)
            print(f"\nNormalized primitive embeddings:")
            print(f"  Mean: {prim_normed.mean().item():.4f}")
            print(f"  Std: {prim_normed.std().item():.4f}")
            print(f"  L2 norms: {torch.norm(prim_normed, dim=-1).mean().item():.4f} (should be 1.0)")

            # Raw scores before sigmoid
            raw_scores = torch.matmul(task_proj_normed, prim_normed.t()) / head.temperature
            print(f"\nRaw scores (before sigmoid):")
            print(f"  Mean: {raw_scores.mean().item():.4f}")
            print(f"  Std: {raw_scores.std().item():.4f}")
            print(f"  Min: {raw_scores.min().item():.4f}")
            print(f"  Max: {raw_scores.max().item():.4f}")

            # Final predictions
            preds = head(τ.unsqueeze(0)).squeeze(0)
            print(f"\nFinal predictions (after sigmoid):")
            print(f"  Mean: {preds.mean().item():.4f}")
            print(f"  Std: {preds.std().item():.4f}")
            print(f"  Min: {preds.min().item():.4f}")
            print(f"  Max: {preds.max().item():.4f}")
            print(f"  Unique values (rounded to 2 decimal): {len(torch.unique(torch.round(preds * 100)))}")

            # Check if predictions are collapsing
            if preds.std().item() < 0.01:
                print("\n⚠️ WARNING: Predictions are collapsed (very low std)")
                print("  This means all primitives get similar scores")

            # Cosine similarities between task projection and primitive embeddings
            cos_sim = F.cosine_similarity(
                task_proj_normed.expand(prim_normed.shape[0], -1),
                prim_normed,
                dim=-1
            )
            print(f"\nCosine similarities (task_proj vs primitives):")
            print(f"  Mean: {cos_sim.mean().item():.4f}")
            print(f"  Std: {cos_sim.std().item():.4f}")
            print(f"  Min: {cos_sim.min().item():.4f}")
            print(f"  Max: {cos_sim.max().item():.4f}")

if __name__ == "__main__":
    diagnose()
