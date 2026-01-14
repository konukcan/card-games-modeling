#!/usr/bin/env python3
"""
Check predictions from TRAINED recognition models.
Shows detailed predictions for first 10 tasks to verify discrimination.
"""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.neural_recognition import NeuralRecognitionModel
from dreamcoder_core.primitives import build_lean_grammar
from dreamcoder_core.dreamcoder_original import create_tasks_from_rules
from rules.catalogue import create_all_rules


def show_predictions(model, tasks, model_name, primitives):
    """Show detailed predictions for first 10 tasks."""
    model.eval()

    print(f"\n{'='*70}")
    print(f"{model_name} - PREDICTIONS FOR FIRST 10 TASKS")
    print(f"{'='*70}")

    all_predictions = []

    with torch.no_grad():
        for i, task in enumerate(tasks[:10]):
            # Get predictions
            if hasattr(model, 'predict_primitives'):
                pred = model.predict_primitives(task)
            else:
                pred = model.predict_primitive_logits(task)

            pred_np = pred.cpu().numpy()
            all_predictions.append(pred_np)

            # Get top-5 with probabilities
            top5_idx = np.argsort(pred_np)[::-1][:5]

            print(f"\n{i+1}. {task.name}")
            print(f"   Top-5 primitives:")
            for rank, idx in enumerate(top5_idx, 1):
                prim_name = primitives[idx]
                prob = pred_np[idx]
                print(f"      {rank}. {prim_name}: {prob:.4f}")

    # Check prediction similarity
    print(f"\n{'-'*70}")
    print("PREDICTION SIMILARITY ANALYSIS")
    print(f"{'-'*70}")

    all_predictions = np.array(all_predictions)

    # Cosine similarity between predictions
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(all_predictions)
    upper_tri = sim_matrix[np.triu_indices(10, k=1)]

    print(f"Mean prediction similarity: {np.mean(upper_tri):.4f}")
    print(f"Min prediction similarity:  {np.min(upper_tri):.4f}")
    print(f"Max prediction similarity:  {np.max(upper_tri):.4f}")

    # Check if top-5 are the same
    top5_sets = []
    for pred in all_predictions:
        top5_idx = set(np.argsort(pred)[::-1][:5])
        top5_sets.append(frozenset(top5_idx))

    unique_top5 = len(set(top5_sets))
    print(f"Unique top-5 sets: {unique_top5} out of 10")

    if unique_top5 == 1:
        print("❌ ALL 10 tasks have IDENTICAL top-5 predictions!")
    elif unique_top5 < 5:
        print("⚠️ Limited diversity in predictions")
    else:
        print("✓ Good diversity in predictions")


if __name__ == "__main__":
    print("Loading grammar...")
    grammar = build_lean_grammar()
    primitives = [str(p) for p in grammar.primitives()]

    print("Loading tasks...")
    rules = create_all_rules()
    tasks = create_tasks_from_rules(rules, n_examples=20, seed=42)
    print(f"Loaded {len(tasks)} tasks")

    # Check for trained contrastive model
    contrastive_path = Path("/Users/cankonuk/Documents/self-explanations-project/card-games-modelling/src/results/warmstart_experiment/contrastive_WARM_20251225_122102/pretrained_recognition.pt")

    if contrastive_path.exists():
        print(f"\nLoading trained CONTRASTIVE model from: {contrastive_path}")
        # Model was saved with card_hidden=128, card_out=64, pred_hidden=128
        model = ContrastiveRecognitionModel(
            grammar=grammar,
            card_hidden=128,
            card_out=64,
            pred_hidden=128
        )
        checkpoint = torch.load(contrastive_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        show_predictions(model, tasks, "TRAINED ContrastiveRecognitionModel", model.primitive_names)
    else:
        print(f"\n⚠️ Trained contrastive model not found at {contrastive_path}")
        print("Running with UNTRAINED model for comparison...")
        model = ContrastiveRecognitionModel(grammar=grammar)
        show_predictions(model, tasks, "UNTRAINED ContrastiveRecognitionModel", model.primitive_names)

    # Also check neural model for comparison
    neural_path = Path("/Users/cankonuk/Documents/self-explanations-project/card-games-modelling/src/results/warmstart_experiment/neural_COLD_20251228_183521/recognition_model_COLD.pt")

    if neural_path.exists():
        print(f"\nLoading trained NEURAL model from: {neural_path}")
        # Model was saved with hidden_dim=128
        model = NeuralRecognitionModel(grammar=grammar, hidden_dim=128)
        checkpoint = torch.load(neural_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        show_predictions(model, tasks, "TRAINED NeuralRecognitionModel", primitives)
