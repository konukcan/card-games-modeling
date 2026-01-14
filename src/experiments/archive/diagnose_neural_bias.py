#!/usr/bin/env python3
"""
Diagnostic: Does the neural model also have bias-dominated predictions?

The contrastive model has near-zero task encodings (τ ≈ 0) because
τ = pos_mean - neg_mean collapses when examples aren't differentiated enough.

But the neural model uses a completely different architecture:
- CardEncoder (GRU over card features)
- ExampleEncoder (combines input/output)
- TaskEncoder (attention pooling over examples)

If the neural model ALSO predicts the same primitives for all tasks,
we need to understand why. Possible explanations:
1. Task encodings are also near-zero (same symptom, different cause)
2. Task encodings have reasonable magnitude but the network learned
   to ignore them and rely on biases
3. The training data only reinforced a few primitives
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from dreamcoder_core.neural_recognition import NeuralRecognitionModel
from dreamcoder_core.primitives import build_lean_grammar
from dreamcoder_core.dreamcoder_original import create_tasks_from_rules
from rules.catalogue import create_all_rules as get_catalogue_rules

RESULTS_DIR = Path("results/warmstart_experiment")


def main():
    grammar = build_lean_grammar()

    # Load neural model
    neural_path = RESULTS_DIR / "neural_BOTH_20251225_145356" / "pretrained_recognition.pt"
    neural = NeuralRecognitionModel(grammar)
    neural.load(str(neural_path))
    neural.eval()

    print("=" * 60)
    print("NEURAL MODEL ARCHITECTURE ANALYSIS")
    print("=" * 60)

    # Check the primitive predictor structure
    print("\nPrimitive Predictor Structure:")
    for name, module in neural.primitive_predictor.predictor.named_modules():
        if len(name) > 0:
            print(f"  {name}: {type(module).__name__}")
            if isinstance(module, nn.Linear):
                print(f"    in_features: {module.in_features}, out_features: {module.out_features}")
                if module.bias is not None:
                    print(f"    has bias: True")

    print("\n" + "=" * 60)
    print("TASK ENCODING ANALYSIS")
    print("=" * 60)

    # Create diverse test tasks
    all_rules = get_catalogue_rules()
    tasks = create_tasks_from_rules(all_rules[:10], n_examples=100, n_holdout=20, hand_size=6)

    task_encodings = {}
    with torch.no_grad():
        for task in tasks:
            enc = neural.encode_task(task)
            task_encodings[task.name] = enc
            print(f"\n{task.name}:")
            print(f"  Encoding shape: {enc.shape}")
            print(f"  Encoding norm: {enc.norm():.4f}")
            print(f"  Encoding range: [{enc.min():.4f}, {enc.max():.4f}]")
            print(f"  Encoding mean: {enc.mean():.4f}, std: {enc.std():.4f}")

    # Compute average norm
    norms = [enc.norm().item() for enc in task_encodings.values()]
    print(f"\n\nAverage task encoding norm: {np.mean(norms):.4f}")
    print(f"Min norm: {np.min(norms):.4f}, Max norm: {np.max(norms):.4f}")

    print("\n" + "=" * 60)
    print("PREDICTION WITH ZERO INPUT vs ACTUAL TASKS")
    print("=" * 60)

    hidden_dim = neural.hidden_dim
    print(f"Hidden dim: {hidden_dim}")

    with torch.no_grad():
        # Zero input prediction
        zero_input = torch.zeros(1, hidden_dim)
        zero_logits = neural.primitive_predictor(zero_input, return_logits=True).squeeze(0)
        zero_probs = torch.exp(torch.log_softmax(zero_logits, dim=0))

        print(f"\nWith ZERO input:")
        values, indices = torch.topk(zero_probs, 10)
        for v, i in zip(values.cpu(), indices.cpu()):
            prim_name = neural.primitive_names[int(i)]
            print(f"  {prim_name}: {float(v):.4f}")

        # Compare with actual tasks
        print("\n--- Actual Task Predictions ---")
        for task in tasks[:5]:
            task_enc = neural.encode_task(task).unsqueeze(0)
            task_probs = torch.exp(neural.primitive_predictor(task_enc)).squeeze(0)

            print(f"\n{task.name} (encoding norm: {task_enc.norm():.4f}):")
            values, indices = torch.topk(task_probs, 5)
            for v, i in zip(values.cpu(), indices.cpu()):
                prim_name = neural.primitive_names[int(i)]
                print(f"  {prim_name}: {float(v):.4f}")

    print("\n" + "=" * 60)
    print("BIAS ANALYSIS - FINAL LINEAR LAYER")
    print("=" * 60)

    # Get the final linear layer in the primitive predictor
    final_linear = neural.primitive_predictor.predictor[-1]
    print(f"\nFinal layer: Linear({final_linear.in_features} -> {final_linear.out_features})")

    if final_linear.bias is not None:
        bias = final_linear.bias.data
        print(f"\nBias statistics:")
        print(f"  Shape: {bias.shape}")
        print(f"  Range: [{bias.min():.4f}, {bias.max():.4f}]")
        print(f"  Mean: {bias.mean():.4f}")
        print(f"  Std: {bias.std():.4f}")

        print(f"\nTop 10 biases:")
        values, indices = torch.topk(bias, 10)
        for v, i in zip(values.cpu(), indices.cpu()):
            prim_name = neural.primitive_names[int(i)]
            print(f"  {prim_name}: {float(v):.4f}")

        print(f"\nBottom 5 biases:")
        values, indices = torch.topk(bias, 5, largest=False)
        for v, i in zip(values.cpu(), indices.cpu()):
            prim_name = neural.primitive_names[int(i)]
            print(f"  {prim_name}: {float(v):.4f}")

    print("\n" + "=" * 60)
    print("SENSITIVITY ANALYSIS: How much do inputs matter?")
    print("=" * 60)

    # Test: How much does the output change when we perturb the input?
    with torch.no_grad():
        base_enc = neural.encode_task(tasks[0]).unsqueeze(0)
        base_output = neural.primitive_predictor(base_enc, return_logits=True).squeeze(0)

        # Add noise to encoding
        noise_scales = [0.01, 0.1, 1.0, 10.0]
        for scale in noise_scales:
            noisy_enc = base_enc + torch.randn_like(base_enc) * scale
            noisy_output = neural.primitive_predictor(noisy_enc, return_logits=True).squeeze(0)

            # Measure output change
            output_diff = (noisy_output - base_output).abs().mean().item()
            ranking_diff = (noisy_output.argsort() != base_output.argsort()).sum().item()

            print(f"\nNoise scale {scale}:")
            print(f"  Input perturbation: {(torch.randn_like(base_enc) * scale).abs().mean():.4f}")
            print(f"  Output change (mean abs diff): {output_diff:.4f}")
            print(f"  Ranking changes: {ranking_diff} / {len(base_output)}")

    print("\n" + "=" * 60)
    print("WEIGHT MAGNITUDE ANALYSIS")
    print("=" * 60)

    # Check if weights are small relative to biases
    weight = final_linear.weight.data
    bias = final_linear.bias.data

    print(f"\nWeight matrix:")
    print(f"  Shape: {weight.shape}")
    print(f"  Frobenius norm: {weight.norm():.4f}")
    print(f"  Mean abs: {weight.abs().mean():.4f}")

    print(f"\nBias vector:")
    print(f"  Norm: {bias.norm():.4f}")
    print(f"  Mean abs: {bias.abs().mean():.4f}")

    # For a typical input, what's the contribution from weights vs bias?
    print(f"\nContribution analysis:")
    typical_input_norm = np.mean(norms)
    max_weight_contribution = weight.abs().sum(dim=1).max().item() * typical_input_norm
    bias_contribution = bias.abs().max().item()
    print(f"  Typical input norm: {typical_input_norm:.4f}")
    print(f"  Max weight contribution (|Wx|): ~{max_weight_contribution:.4f}")
    print(f"  Max bias contribution (|b|): {bias_contribution:.4f}")
    print(f"  Ratio (weight/bias): {max_weight_contribution/bias_contribution:.2f}")


if __name__ == '__main__':
    main()
