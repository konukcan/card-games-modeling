#!/usr/bin/env python3
"""
Check if embeddings are diverse BEFORE vs AFTER training.
Determines if the collapse happens at:
1. Embedding level (training destroys diversity)
2. Prediction head (embeddings still diverse, but head collapses)
"""

import sys
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.primitives import build_lean_grammar
from dreamcoder_core.dreamcoder_original import create_tasks_from_rules
from rules.catalogue import create_all_rules


def check_embeddings(model, tasks, model_name):
    """Check embedding diversity."""
    model.eval()

    print(f"\n{'='*70}")
    print(f"{model_name}")
    print(f"{'='*70}")

    embeddings = []
    with torch.no_grad():
        for task in tasks[:10]:
            emb = model.encode_task(task)
            embeddings.append(emb.cpu().numpy())

    embeddings = np.array(embeddings)

    # Compute similarity
    sim_matrix = cosine_similarity(embeddings)
    upper_tri = sim_matrix[np.triu_indices(10, k=1)]

    print(f"\nEmbedding similarity (first 10 tasks):")
    print(f"  Mean: {np.mean(upper_tri):.4f}")
    print(f"  Min:  {np.min(upper_tri):.4f}")
    print(f"  Max:  {np.max(upper_tri):.4f}")
    print(f"  Std:  {np.std(upper_tri):.4f}")

    # Show first few embeddings (just first 5 dimensions)
    print(f"\nSample embeddings (first 5 dims):")
    for i, task in enumerate(tasks[:5]):
        print(f"  {task.name[:20]}: {embeddings[i][:5]}")

    if np.mean(upper_tri) > 0.99:
        print(f"\n❌ EMBEDDINGS ARE COLLAPSED (mean sim > 0.99)")
    elif np.mean(upper_tri) > 0.8:
        print(f"\n⚠️ EMBEDDINGS ARE SOMEWHAT SIMILAR")
    else:
        print(f"\n✓ EMBEDDINGS ARE DIVERSE")


if __name__ == "__main__":
    print("Loading grammar and tasks...")
    grammar = build_lean_grammar()
    rules = create_all_rules()
    tasks = create_tasks_from_rules(rules, n_examples=20, seed=42)

    # UNTRAINED model
    print("\n" + "="*70)
    print("UNTRAINED ContrastiveRecognitionModel")
    print("="*70)
    model_untrained = ContrastiveRecognitionModel(
        grammar=grammar,
        card_hidden=128,
        card_out=64,
        pred_hidden=128
    )
    check_embeddings(model_untrained, tasks, "UNTRAINED")

    # TRAINED model
    print("\n" + "="*70)
    print("TRAINED ContrastiveRecognitionModel")
    print("="*70)
    contrastive_path = Path("results/warmstart_experiment/contrastive_WARM_20251225_122102/pretrained_recognition.pt")
    model_trained = ContrastiveRecognitionModel(
        grammar=grammar,
        card_hidden=128,
        card_out=64,
        pred_hidden=128
    )
    checkpoint = torch.load(contrastive_path, map_location='cpu')
    model_trained.load_state_dict(checkpoint['model_state_dict'])
    check_embeddings(model_trained, tasks, "TRAINED")

    # Compare predictions directly
    print("\n" + "="*70)
    print("PREDICTION COMPARISON")
    print("="*70)

    print("\nRaw logits from prediction head (first 3 tasks):")
    with torch.no_grad():
        for i, task in enumerate(tasks[:3]):
            # Get embeddings
            emb_untrained = model_untrained.encode_task(task)
            emb_trained = model_trained.encode_task(task)

            # Get predictions
            pred_untrained = model_untrained.predict_primitives(task)
            pred_trained = model_trained.predict_primitives(task)

            print(f"\n{task.name}:")
            print(f"  Untrained embedding norm: {torch.norm(emb_untrained):.4f}")
            print(f"  Trained embedding norm:   {torch.norm(emb_trained):.4f}")
            print(f"  Untrained pred mean/std:  {pred_untrained.mean():.4f} / {pred_untrained.std():.4f}")
            print(f"  Trained pred mean/std:    {pred_trained.mean():.4f} / {pred_trained.std():.4f}")
