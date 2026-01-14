#!/usr/bin/env python3
"""
Diagnostic script to investigate why all models predict the same primitives.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from dreamcoder_core.neural_recognition import NeuralRecognitionModel
from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.primitives import build_lean_grammar
from dreamcoder_core.dreamcoder_original import create_tasks_from_rules
from rules.pretraining_rules import get_all_pretraining_rules
from rules.catalogue import create_all_rules as get_catalogue_rules

RESULTS_DIR = Path("results/warmstart_experiment")

def main():
    # Build grammar
    grammar = build_lean_grammar()
    print(f"Grammar has {len(grammar.productions)} primitives")

    # Create a few test tasks
    catalogue_rules = get_catalogue_rules()[:5]  # Just first 5
    tasks = create_tasks_from_rules(catalogue_rules, n_examples=100, n_holdout=20, hand_size=6)

    # Load models
    print("\n" + "="*60)
    print("Loading models...")
    print("="*60)

    # Neural model
    neural_path = RESULTS_DIR / "neural_BOTH_20251225_145356" / "pretrained_recognition.pt"
    neural = NeuralRecognitionModel(grammar)
    neural.load(str(neural_path))
    print(f"Loaded neural model from {neural_path}")

    # Contrastive sigmoid
    sigmoid_path = RESULTS_DIR / "contrastive_BOTH_20251225_145818" / "pretrained_recognition.pt"
    sigmoid = ContrastiveRecognitionModel(grammar, card_hidden=128, card_out=64, pred_hidden=128, output_mode='sigmoid')
    sigmoid.load(str(sigmoid_path))
    print(f"Loaded sigmoid model from {sigmoid_path}")

    # Contrastive softmax
    softmax_path = RESULTS_DIR / "contrastive_softmax_WARM_20251226_174158" / "pretrained_recognition.pt"
    softmax = ContrastiveRecognitionModel(grammar, card_hidden=64, card_out=32, pred_hidden=64, output_mode='softmax')
    softmax.load(str(softmax_path))
    print(f"Loaded softmax model from {softmax_path}")

    # Check primitive name mapping
    print("\n" + "="*60)
    print("Checking primitive name mappings...")
    print("="*60)

    print(f"\nNeural primitive_to_idx has {len(neural.primitive_to_idx)} entries")
    print(f"First 5: {list(neural.primitive_to_idx.items())[:5]}")

    print(f"\nSigmoid primitive_names has {len(sigmoid.primitive_names)} entries")
    print(f"First 5: {sigmoid.primitive_names[:5]}")

    print(f"\nSoftmax primitive_names has {len(softmax.primitive_names)} entries")
    print(f"First 5: {softmax.primitive_names[:5]}")

    # Now check predictions for each task
    print("\n" + "="*60)
    print("Checking predictions for each task...")
    print("="*60)

    for task in tasks:
        print(f"\n--- Task: {task.name} ---")

        # Neural predictions
        neural_log_probs = neural.predict_primitive_probs(task)
        print(f"\nNeural model output shape: {neural_log_probs.shape}")
        print(f"Neural model output range: [{neural_log_probs.min():.4f}, {neural_log_probs.max():.4f}]")
        print(f"Neural model output mean: {neural_log_probs.mean():.4f}")

        # Top 5 for neural
        neural_prim_names = [str(p.program) for p in neural.grammar.productions]
        values, indices = torch.topk(neural_log_probs, 5)
        print("Neural top 5:")
        for v, i in zip(values.cpu(), indices.cpu()):
            print(f"  {neural_prim_names[i]}: {float(v):.4f}")

        # Sigmoid predictions
        sigmoid_probs = sigmoid.predict_primitives(task)
        print(f"\nSigmoid model output shape: {sigmoid_probs.shape}")
        print(f"Sigmoid model output range: [{sigmoid_probs.min():.4f}, {sigmoid_probs.max():.4f}]")
        print(f"Sigmoid model output mean: {sigmoid_probs.mean():.4f}")
        print(f"Sigmoid model output sum: {sigmoid_probs.sum():.4f}")  # Should NOT be 1 for sigmoid

        values, indices = torch.topk(sigmoid_probs, 5)
        print("Sigmoid top 5:")
        for v, i in zip(values.cpu(), indices.cpu()):
            print(f"  {sigmoid.primitive_names[int(i)]}: {float(v):.4f}")

        # Softmax predictions
        softmax_probs = softmax.predict_primitives(task)
        print(f"\nSoftmax model output shape: {softmax_probs.shape}")
        print(f"Softmax model output range: [{softmax_probs.min():.4f}, {softmax_probs.max():.4f}]")
        print(f"Softmax model output mean: {softmax_probs.mean():.4f}")
        print(f"Softmax model output sum: {softmax_probs.sum():.4f}")  # Should be ~1 for softmax

        values, indices = torch.topk(softmax_probs, 5)
        print("Softmax top 5:")
        for v, i in zip(values.cpu(), indices.cpu()):
            print(f"  {softmax.primitive_names[int(i)]}: {float(v):.4f}")

        print()

if __name__ == '__main__':
    main()
