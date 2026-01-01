#!/usr/bin/env python3
"""
Diagnostic script for baseline model - check if it has the same embedding collapse.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.contrastive_recognition_v1_baseline import (
    ContrastiveRecognitionModel
)
from dreamcoder_core.lean_primitives import build_lean_grammar
from rules.catalogue import create_all_rules
from rules.cards import sample_hand

def diagnose():
    grammar = build_lean_grammar()
    rules = create_all_rules()

    # Create baseline model
    model = ContrastiveRecognitionModel(grammar=grammar)

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
        # Encode hands using baseline model's method
        pos_embeddings = []
        for hand in pos_hands[:5]:
            suits, ranks, positions, mask = model._encode_hand_raw(hand)
            emb = model.hand_encoder(
                suits.unsqueeze(0).to(model.device),
                ranks.unsqueeze(0).to(model.device),
                positions.unsqueeze(0).to(model.device),
                mask.unsqueeze(0).to(model.device)
            ).squeeze(0)
            pos_embeddings.append(emb)

        neg_embeddings = []
        for hand in neg_hands[:5]:
            suits, ranks, positions, mask = model._encode_hand_raw(hand)
            emb = model.hand_encoder(
                suits.unsqueeze(0).to(model.device),
                ranks.unsqueeze(0).to(model.device),
                positions.unsqueeze(0).to(model.device),
                mask.unsqueeze(0).to(model.device)
            ).squeeze(0)
            neg_embeddings.append(emb)

        pos_tensor = torch.stack(pos_embeddings)
        neg_tensor = torch.stack(neg_embeddings)

        print(f"\n--- Positive embeddings ---")
        print(f"  Mean L2 norm: {torch.norm(pos_tensor, dim=-1).mean().item():.4f}")

        print(f"\n--- Negative embeddings ---")
        print(f"  Mean L2 norm: {torch.norm(neg_tensor, dim=-1).mean().item():.4f}")

        # Cosine similarities
        pos_normed = F.normalize(pos_tensor, p=2, dim=-1)
        neg_normed = F.normalize(neg_tensor, p=2, dim=-1)

        pos_pos_sim = torch.mm(pos_normed, pos_normed.t())
        neg_neg_sim = torch.mm(neg_normed, neg_normed.t())
        pos_neg_sim = torch.mm(pos_normed, neg_normed.t())

        pos_pos_off_diag = pos_pos_sim[~torch.eye(5, dtype=bool)]
        neg_neg_off_diag = neg_neg_sim[~torch.eye(5, dtype=bool)]

        print(f"\nPositive-Positive cosine similarity (off-diagonal):")
        print(f"  Mean: {pos_pos_off_diag.mean().item():.4f}")

        print(f"\nNegative-Negative cosine similarity (off-diagonal):")
        print(f"  Mean: {neg_neg_off_diag.mean().item():.4f}")

        print(f"\nPositive-Negative cosine similarity:")
        print(f"  Mean: {pos_neg_sim.mean().item():.4f}")

        pos_mean = pos_tensor.mean(dim=0)
        neg_mean = neg_tensor.mean(dim=0)

        print(f"\nCosine similarity between pos_mean and neg_mean:")
        cos_sim = F.cosine_similarity(pos_mean.unsqueeze(0), neg_mean.unsqueeze(0)).item()
        print(f"  {cos_sim:.4f}")

        τ_raw = pos_mean - neg_mean
        print(f"\nτ = pos_mean - neg_mean:")
        print(f"  L2 norm: {torch.norm(τ_raw).item():.4f}")

        if cos_sim > 0.95:
            print("\n⚠️ BASELINE also has embedding collapse!")
        else:
            print(f"\n✓ BASELINE embeddings are diverse (cosine < 0.95)")

if __name__ == "__main__":
    diagnose()
