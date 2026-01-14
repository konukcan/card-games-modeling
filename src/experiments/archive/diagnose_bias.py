#!/usr/bin/env python3
"""
Check if primitive head bias dominates predictions when τ is near-zero.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.primitives import build_lean_grammar

RESULTS_DIR = Path("results/warmstart_experiment")

def main():
    grammar = build_lean_grammar()

    # Load contrastive softmax model
    softmax_path = RESULTS_DIR / "contrastive_softmax_WARM_20251226_174158" / "pretrained_recognition.pt"
    softmax = ContrastiveRecognitionModel(grammar, card_hidden=64, card_out=32, pred_hidden=64, output_mode='softmax')
    softmax.load(str(softmax_path))
    softmax.eval()

    print("="*60)
    print("ANALYZING PRIMITIVE HEAD STRUCTURE")
    print("="*60)

    # Get the primitive head
    prim_head = softmax.primitive_head

    print(f"\nPrimitive head type: {type(prim_head)}")

    # Check if it's a sequential model
    if hasattr(prim_head, 'named_modules'):
        print("\nModules in primitive head:")
        for name, module in prim_head.named_modules():
            if len(name) > 0:  # Skip the root
                print(f"  {name}: {type(module).__name__}")

    # Find the final linear layer and check its bias
    print("\n" + "="*60)
    print("CHECKING FINAL LAYER BIAS")
    print("="*60)

    # The primitive head should end with a Linear layer
    # Let's find all Linear layers and check their biases
    for name, module in prim_head.named_modules():
        if isinstance(module, nn.Linear):
            print(f"\n{name}: Linear({module.in_features} -> {module.out_features})")
            if module.bias is not None:
                bias = module.bias.data
                print(f"  Bias shape: {bias.shape}")
                print(f"  Bias range: [{bias.min():.4f}, {bias.max():.4f}]")
                print(f"  Bias mean: {bias.mean():.4f}")

                # What are the top 5 biases?
                values, indices = torch.topk(bias, 5)
                print(f"  Top 5 biases:")
                for v, i in zip(values.cpu(), indices.cpu()):
                    prim_name = softmax.primitive_names[int(i)]
                    print(f"    {prim_name}: {float(v):.4f}")

    # Now test: what happens when we pass ZERO input?
    print("\n" + "="*60)
    print("PREDICTION WITH ZERO INPUT")
    print("="*60)

    with torch.no_grad():
        # Create zero input with correct shape
        # The primitive head expects shape (batch, input_dim)
        # Get input_dim from the first layer of primitive head
        input_dim = prim_head.mlp[0].in_features
        print(f"Primitive head input dim: {input_dim}")
        zero_input = torch.zeros(1, input_dim)
        output = prim_head(zero_input)
        probs = torch.exp(output).squeeze(0)  # Convert log-probs to probs

        print(f"\nWith zero input (τ = 0):")
        print(f"  Output shape: {output.shape}")
        print(f"  Output sum (should be ~1): {probs.sum():.4f}")

        values, indices = torch.topk(probs, 10)
        print(f"  Top 10 predictions:")
        for v, i in zip(values.cpu(), indices.cpu()):
            prim_name = softmax.primitive_names[int(i)]
            print(f"    {prim_name}: {float(v):.4f}")

    # Compare with actual task predictions
    print("\n" + "="*60)
    print("COMPARISON: ZERO INPUT vs ACTUAL TASK")
    print("="*60)

    from dreamcoder_core.dreamcoder_original import create_tasks_from_rules
    from rules.catalogue import create_all_rules
    tasks = create_tasks_from_rules(create_all_rules()[:3], n_examples=100, n_holdout=20, hand_size=6)

    with torch.no_grad():
        for task in tasks:
            τ = softmax.encode_task_batched(task).unsqueeze(0)
            output = prim_head(τ)
            probs = torch.exp(output).squeeze(0)

            values, indices = torch.topk(probs, 5)
            print(f"\n{task.name} (τ norm = {τ.norm():.4f}):")
            for v, i in zip(values.cpu(), indices.cpu()):
                prim_name = softmax.primitive_names[int(i)]
                print(f"  {prim_name}: {float(v):.4f}")

if __name__ == '__main__':
    main()
