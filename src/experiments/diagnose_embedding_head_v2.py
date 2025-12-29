#!/usr/bin/env python3
"""
Diagnostic script V2 - trace the full encoding pipeline to find where magnitude collapses.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.recognition_variants import (
    RecognitionModelVariant,
    hand_to_tensors
)
from dreamcoder_core.lean_primitives import build_lean_grammar
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
    for _ in range(500):
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

    model.eval()
    with torch.no_grad():
        print("\n" + "="*70)
        print("STEP 1: Individual Hand Encodings")
        print("="*70)

        # Encode positive hands
        pos_embeddings = []
        for i, hand in enumerate(pos_hands[:5]):
            suits, ranks, positions = hand_to_tensors(hand)
            suits = suits.unsqueeze(0).to(model.device)
            ranks = ranks.unsqueeze(0).to(model.device)
            positions = positions.unsqueeze(0).to(model.device)

            # Card embeddings
            card_emb = model.card_encoder(suits, ranks, positions)
            if i == 0:
                print(f"\nCard embeddings (first positive hand):")
                print(f"  Shape: {card_emb.shape}")
                print(f"  Mean: {card_emb.mean().item():.4f}")
                print(f"  Std: {card_emb.std().item():.4f}")
                print(f"  L2 norm per card: {torch.norm(card_emb, dim=-1).mean().item():.4f}")

            # Card MLP
            card_features = model.card_mlp(card_emb)
            if i == 0:
                print(f"\nAfter card_mlp:")
                print(f"  Shape: {card_features.shape}")
                print(f"  Mean: {card_features.mean().item():.4f}")
                print(f"  Std: {card_features.std().item():.4f}")
                print(f"  L2 norm per card: {torch.norm(card_features, dim=-1).mean().item():.4f}")

            # Mean pooling
            hand_emb = card_features.mean(dim=1).squeeze(0)
            if i == 0:
                print(f"\nAfter mean pooling:")
                print(f"  Shape: {hand_emb.shape}")
                print(f"  L2 norm: {torch.norm(hand_emb).item():.4f}")

            pos_embeddings.append(hand_emb)

        pos_tensor = torch.stack(pos_embeddings)
        print(f"\n--- All positive embeddings ---")
        print(f"  Shape: {pos_tensor.shape}")
        print(f"  Mean L2 norm: {torch.norm(pos_tensor, dim=-1).mean().item():.4f}")

        # Same for negative hands
        neg_embeddings = []
        for hand in neg_hands[:5]:
            suits, ranks, positions = hand_to_tensors(hand)
            suits = suits.unsqueeze(0).to(model.device)
            ranks = ranks.unsqueeze(0).to(model.device)
            positions = positions.unsqueeze(0).to(model.device)
            card_emb = model.card_encoder(suits, ranks, positions)
            card_features = model.card_mlp(card_emb)
            hand_emb = card_features.mean(dim=1).squeeze(0)
            neg_embeddings.append(hand_emb)

        neg_tensor = torch.stack(neg_embeddings)
        print(f"\n--- All negative embeddings ---")
        print(f"  Shape: {neg_tensor.shape}")
        print(f"  Mean L2 norm: {torch.norm(neg_tensor, dim=-1).mean().item():.4f}")

        print("\n" + "="*70)
        print("STEP 2: Cosine Similarities")
        print("="*70)

        # Pairwise cosine similarity within positives
        pos_normed = F.normalize(pos_tensor, p=2, dim=-1)
        neg_normed = F.normalize(neg_tensor, p=2, dim=-1)

        pos_pos_sim = torch.mm(pos_normed, pos_normed.t())
        neg_neg_sim = torch.mm(neg_normed, neg_normed.t())
        pos_neg_sim = torch.mm(pos_normed, neg_normed.t())

        # Extract off-diagonal for within-class
        pos_pos_off_diag = pos_pos_sim[~torch.eye(5, dtype=bool)]
        neg_neg_off_diag = neg_neg_sim[~torch.eye(5, dtype=bool)]

        print(f"\nPositive-Positive cosine similarity (off-diagonal):")
        print(f"  Mean: {pos_pos_off_diag.mean().item():.4f}")
        print(f"  Std: {pos_pos_off_diag.std().item():.4f}")

        print(f"\nNegative-Negative cosine similarity (off-diagonal):")
        print(f"  Mean: {neg_neg_off_diag.mean().item():.4f}")
        print(f"  Std: {neg_neg_off_diag.std().item():.4f}")

        print(f"\nPositive-Negative cosine similarity:")
        print(f"  Mean: {pos_neg_sim.mean().item():.4f}")
        print(f"  Std: {pos_neg_sim.std().item():.4f}")

        print("\n" + "="*70)
        print("STEP 3: Contrastive Encoding")
        print("="*70)

        pos_mean = pos_tensor.mean(dim=0)
        neg_mean = neg_tensor.mean(dim=0)

        print(f"\nMean of positive embeddings:")
        print(f"  L2 norm: {torch.norm(pos_mean).item():.4f}")

        print(f"\nMean of negative embeddings:")
        print(f"  L2 norm: {torch.norm(neg_mean).item():.4f}")

        print(f"\nCosine similarity between pos_mean and neg_mean:")
        print(f"  {F.cosine_similarity(pos_mean.unsqueeze(0), neg_mean.unsqueeze(0)).item():.4f}")

        τ_raw = pos_mean - neg_mean
        print(f"\nτ = pos_mean - neg_mean (BEFORE normalization):")
        print(f"  L2 norm: {torch.norm(τ_raw).item():.4f}")
        print(f"  Mean: {τ_raw.mean().item():.4f}")
        print(f"  Std: {τ_raw.std().item():.4f}")

        # The problem: if pos_mean ≈ neg_mean, then τ ≈ 0

        # Check what's happening with the actual normalization
        if model.normalize_embeddings:
            print("\n--- Applying normalization ---")
            τ_unsqueezed = τ_raw.unsqueeze(0)
            τ_normed = model.embedding_norm(τ_unsqueezed)
            print(f"After LayerNorm:")
            print(f"  L2 norm: {torch.norm(τ_normed).item():.4f}")
            print(f"  Mean: {τ_normed.mean().item():.4f}")
            print(f"  Std: {τ_normed.std().item():.4f}")

            τ_scaled = τ_normed * model.embedding_scale_param
            print(f"\nAfter scaling (scale={model.embedding_scale_param.item():.2f}):")
            print(f"  L2 norm: {torch.norm(τ_scaled).item():.4f}")

        print("\n" + "="*70)
        print("DIAGNOSIS")
        print("="*70)

        cosine_pos_neg_mean = F.cosine_similarity(pos_mean.unsqueeze(0), neg_mean.unsqueeze(0)).item()
        if cosine_pos_neg_mean > 0.95:
            print("\n⚠️ PROBLEM IDENTIFIED: pos_mean ≈ neg_mean (cosine similarity > 0.95)")
            print("   This causes τ = pos_mean - neg_mean to be near-zero.")
            print("   The model cannot distinguish positive from negative hands!")
            print("\n   SOLUTION: The card encoder is not capturing features that")
            print("   differentiate positive vs negative hands for this rule.")
        else:
            print(f"\n✓ pos_mean and neg_mean are different (cosine = {cosine_pos_neg_mean:.4f})")
            print("   τ should have meaningful signal.")

if __name__ == "__main__":
    diagnose()
