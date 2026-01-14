#!/usr/bin/env python3
"""
Deep diagnostic - check each layer to find where collapse happens.
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
from dreamcoder_core.primitives import build_lean_grammar
from rules.catalogue import create_all_rules
from rules.cards import sample_hand

def diagnose():
    grammar = build_lean_grammar()

    # Create model
    model = RecognitionModelVariant(
        grammar=grammar,
        card_encoder_type='standard',
        hand_encoder_type='mean',
        task_encoder_type='standard',
        prediction_head_type='embedding',
        loss_type='bce'
    )

    model.eval()
    with torch.no_grad():
        # Sample 3 different hands
        hands = []
        for seed in [42, 123, 456]:
            torch.manual_seed(seed)
            np.random.seed(seed)
            hand = sample_hand(5)
            hands.append(hand)

        print("="*70)
        print("CHECKING 3 DIFFERENT HANDS")
        print("="*70)

        for i, hand in enumerate(hands):
            print(f"\n--- Hand {i+1}: {[(c.rank, c.suit) for c in hand]} ---")

            suits, ranks, positions = hand_to_tensors(hand)
            print(f"Suits tensor: {suits.tolist()}")
            print(f"Ranks tensor: {ranks.tolist()}")

        print("\n" + "="*70)
        print("LAYER-BY-LAYER ANALYSIS")
        print("="*70)

        all_card_embs = []
        all_hand_embs = []

        for i, hand in enumerate(hands):
            suits, ranks, positions = hand_to_tensors(hand)
            suits = suits.unsqueeze(0).to(model.device)
            ranks = ranks.unsqueeze(0).to(model.device)
            positions = positions.unsqueeze(0).to(model.device)

            # Step 1: Card encoder (embedding lookup)
            suit_emb = model.card_encoder.suit_embed(suits)
            rank_emb = model.card_encoder.rank_embed(ranks)
            pos_emb = model.card_encoder.pos_embed(positions)
            card_emb = torch.cat([suit_emb, rank_emb, pos_emb], dim=-1)

            print(f"\nHand {i+1} - Card embeddings:")
            print(f"  Shape: {card_emb.shape}")
            print(f"  First 5 values of card 0: {card_emb[0, 0, :5].tolist()}")
            print(f"  L2 norm per card: {torch.norm(card_emb, dim=-1).squeeze().tolist()[:5]}")

            all_card_embs.append(card_emb)

            # Step 2: Through card_mlp layer by layer
            x = card_emb
            for j, layer in enumerate(model.card_mlp):
                x = layer(x)
                if isinstance(layer, torch.nn.Linear):
                    print(f"  After Linear {j}: mean={x.mean().item():.4f}, std={x.std().item():.4f}")
                elif isinstance(layer, torch.nn.ReLU):
                    print(f"  After ReLU {j}: mean={x.mean().item():.4f}, std={x.std().item():.4f}, zeros%={((x==0).sum()/x.numel()*100).item():.1f}%")

            # Step 3: Mean pooling
            hand_emb = x.mean(dim=1).squeeze(0)
            print(f"  Final hand embedding first 5 values: {hand_emb[:5].tolist()}")

            all_hand_embs.append(hand_emb)

        print("\n" + "="*70)
        print("COMPARING HAND EMBEDDINGS")
        print("="*70)

        hand_tensor = torch.stack(all_hand_embs)
        hand_normed = F.normalize(hand_tensor, p=2, dim=-1)

        print(f"\nPairwise cosine similarities:")
        for i in range(3):
            for j in range(i+1, 3):
                sim = F.cosine_similarity(hand_normed[i:i+1], hand_normed[j:j+1]).item()
                print(f"  Hand {i+1} vs Hand {j+1}: {sim:.6f}")

        # Check if embeddings are EXACTLY the same (bit-for-bit)
        print(f"\nAre embeddings EXACTLY equal?")
        for i in range(3):
            for j in range(i+1, 3):
                equal = torch.allclose(all_hand_embs[i], all_hand_embs[j], atol=1e-7)
                print(f"  Hand {i+1} vs Hand {j+1}: {equal}")

        # Check card embeddings
        print("\n" + "="*70)
        print("COMPARING RAW CARD EMBEDDINGS")
        print("="*70)

        print(f"\nPairwise cosine similarities of first card from each hand:")
        for i in range(3):
            for j in range(i+1, 3):
                card_i = F.normalize(all_card_embs[i][0, 0:1], p=2, dim=-1)
                card_j = F.normalize(all_card_embs[j][0, 0:1], p=2, dim=-1)
                sim = F.cosine_similarity(card_i, card_j).item()
                print(f"  Card 0 of Hand {i+1} vs Card 0 of Hand {j+1}: {sim:.6f}")

if __name__ == "__main__":
    diagnose()
