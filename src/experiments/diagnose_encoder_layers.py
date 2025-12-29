#!/usr/bin/env python3
"""
Deep diagnostic for task encoder - trace where the collapse happens.

This script checks each layer of the encoding pipeline to find
where all tasks collapse to the same representation.
"""

import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.neural_recognition import (
    NeuralRecognitionModel, encode_hand, encode_output
)
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.dreamcoder_original import create_tasks_from_rules
from rules.catalogue import create_all_rules


def trace_encoding_pipeline(model, tasks):
    """Trace through each layer to find where collapse happens."""
    model.eval()

    print("=" * 70)
    print("LAYER-BY-LAYER ENCODING TRACE")
    print("=" * 70)

    # Sample 5 diverse tasks
    sample_tasks = [tasks[0], tasks[10], tasks[20], tasks[30], tasks[40]]
    task_names = [t.name for t in sample_tasks]

    print(f"\nAnalyzing tasks: {task_names}")

    with torch.no_grad():
        # ========================================
        # LAYER 1: Raw input features
        # ========================================
        print("\n" + "-" * 50)
        print("LAYER 1: RAW INPUT FEATURES")
        print("-" * 50)

        all_input_features = []
        for task in sample_tasks:
            task_features = []
            for inp, out in task.examples[:5]:  # First 5 examples
                hand_features = encode_hand(inp, 8)
                task_features.append(hand_features.numpy())
            all_input_features.append(np.array(task_features))

        # Check if raw features are different
        print(f"  Shape per task: {all_input_features[0].shape}")

        # Compare first example of each task
        first_examples = [f[0].flatten() for f in all_input_features]
        for i in range(len(first_examples)):
            for j in range(i+1, len(first_examples)):
                sim = np.dot(first_examples[i], first_examples[j]) / (
                    np.linalg.norm(first_examples[i]) * np.linalg.norm(first_examples[j]) + 1e-8
                )
                if sim < 0.99:
                    print(f"  ✓ {task_names[i][:15]} vs {task_names[j][:15]}: {sim:.4f} (different)")
                else:
                    print(f"  ⚠️ {task_names[i][:15]} vs {task_names[j][:15]}: {sim:.4f} (very similar)")

        # ========================================
        # LAYER 2: Card encoder output
        # ========================================
        print("\n" + "-" * 50)
        print("LAYER 2: CARD ENCODER OUTPUT")
        print("-" * 50)

        all_card_encodings = []
        for task in sample_tasks:
            task_card_encs = []
            for inp, out in task.examples[:5]:
                hand_features = encode_hand(inp, 8).unsqueeze(0)
                # Pass through card encoder (part of example encoder)
                card_enc = model.example_encoder.input_encoder(hand_features)
                task_card_encs.append(card_enc.squeeze(0).numpy())
            all_card_encodings.append(np.array(task_card_encs))

        print(f"  Shape per task: {all_card_encodings[0].shape}")

        # Compare
        first_card_encs = [f[0].flatten() for f in all_card_encodings]
        for i in range(len(first_card_encs)):
            for j in range(i+1, len(first_card_encs)):
                sim = np.dot(first_card_encs[i], first_card_encs[j]) / (
                    np.linalg.norm(first_card_encs[i]) * np.linalg.norm(first_card_encs[j]) + 1e-8
                )
                print(f"  {task_names[i][:15]} vs {task_names[j][:15]}: {sim:.4f}")

        # ========================================
        # LAYER 3: Example encoder output
        # ========================================
        print("\n" + "-" * 50)
        print("LAYER 3: EXAMPLE ENCODER OUTPUT")
        print("-" * 50)

        all_example_encodings = []
        for task in sample_tasks:
            task_example_encs = []
            for inp, out in task.examples[:5]:
                hand_features = encode_hand(inp, 8).unsqueeze(0)
                out_features = encode_output(out).unsqueeze(0)
                example_enc = model.example_encoder(hand_features, out_features)
                task_example_encs.append(example_enc.squeeze(0).numpy())
            all_example_encodings.append(np.array(task_example_encs))

        print(f"  Shape per task: {all_example_encodings[0].shape}")

        # Compare first example of each task
        first_example_encs = [f[0] for f in all_example_encodings]
        for i in range(len(first_example_encs)):
            for j in range(i+1, len(first_example_encs)):
                sim = np.dot(first_example_encs[i], first_example_encs[j]) / (
                    np.linalg.norm(first_example_encs[i]) * np.linalg.norm(first_example_encs[j]) + 1e-8
                )
                print(f"  {task_names[i][:15]} vs {task_names[j][:15]}: {sim:.4f}")

        # Also check: are all examples within a task similar?
        print("\n  Within-task example similarity:")
        for t_idx, task_encs in enumerate(all_example_encodings):
            sims = []
            for i in range(len(task_encs)):
                for j in range(i+1, len(task_encs)):
                    sim = np.dot(task_encs[i], task_encs[j]) / (
                        np.linalg.norm(task_encs[i]) * np.linalg.norm(task_encs[j]) + 1e-8
                    )
                    sims.append(sim)
            print(f"    {task_names[t_idx][:15]}: mean={np.mean(sims):.4f}, std={np.std(sims):.4f}")

        # ========================================
        # LAYER 4: Task encoder (attention-pooled)
        # ========================================
        print("\n" + "-" * 50)
        print("LAYER 4: TASK ENCODER OUTPUT (FINAL)")
        print("-" * 50)

        all_task_encodings = []
        for task in sample_tasks:
            task_enc = model.encode_task(task)
            all_task_encodings.append(task_enc.numpy())

        print(f"  Shape: {all_task_encodings[0].shape}")

        # Compare
        for i in range(len(all_task_encodings)):
            for j in range(i+1, len(all_task_encodings)):
                sim = np.dot(all_task_encodings[i], all_task_encodings[j]) / (
                    np.linalg.norm(all_task_encodings[i]) * np.linalg.norm(all_task_encodings[j]) + 1e-8
                )
                print(f"  {task_names[i][:15]} vs {task_names[j][:15]}: {sim:.4f}")

        # ========================================
        # Check attention weights
        # ========================================
        print("\n" + "-" * 50)
        print("ATTENTION WEIGHTS ANALYSIS")
        print("-" * 50)

        for task in sample_tasks[:3]:
            # Get example encodings
            example_encs = []
            for inp, out in task.examples[:model.max_examples]:
                hand_features = encode_hand(inp, 8).unsqueeze(0)
                out_features = encode_output(out).unsqueeze(0)
                enc = model.example_encoder(hand_features, out_features)
                example_encs.append(enc)

            if example_encs:
                stacked = torch.stack([e.squeeze(0) for e in example_encs], dim=0)
                stacked = stacked.unsqueeze(0)  # (1, num_examples, hidden)

                # Get attention scores
                attn_scores = model.task_encoder.attention(stacked).squeeze(-1)
                attn_weights = torch.softmax(attn_scores, dim=-1)

                weights = attn_weights.squeeze(0).numpy()
                print(f"\n  {task.name[:20]}:")
                print(f"    Weights: {weights[:5]}...")
                print(f"    Entropy: {-np.sum(weights * np.log(weights + 1e-8)):.4f}")
                print(f"    Max weight: {np.max(weights):.4f}")
                print(f"    Uniform would be: {1/len(weights):.4f}")

        # ========================================
        # SUMMARY
        # ========================================
        print("\n" + "=" * 70)
        print("DIAGNOSIS SUMMARY")
        print("=" * 70)

        # Determine where collapse happens
        card_enc_sim = np.mean([
            np.dot(first_card_encs[i], first_card_encs[j]) /
            (np.linalg.norm(first_card_encs[i]) * np.linalg.norm(first_card_encs[j]) + 1e-8)
            for i in range(len(first_card_encs))
            for j in range(i+1, len(first_card_encs))
        ])

        example_enc_sim = np.mean([
            np.dot(first_example_encs[i], first_example_encs[j]) /
            (np.linalg.norm(first_example_encs[i]) * np.linalg.norm(first_example_encs[j]) + 1e-8)
            for i in range(len(first_example_encs))
            for j in range(i+1, len(first_example_encs))
        ])

        task_enc_sim = np.mean([
            np.dot(all_task_encodings[i], all_task_encodings[j]) /
            (np.linalg.norm(all_task_encodings[i]) * np.linalg.norm(all_task_encodings[j]) + 1e-8)
            for i in range(len(all_task_encodings))
            for j in range(i+1, len(all_task_encodings))
        ])

        print(f"\nSimilarity at each layer:")
        print(f"  Card encoder:    {card_enc_sim:.4f}")
        print(f"  Example encoder: {example_enc_sim:.4f}")
        print(f"  Task encoder:    {task_enc_sim:.4f}")

        if task_enc_sim > 0.99:
            if example_enc_sim > 0.99:
                if card_enc_sim > 0.99:
                    print("\n⚠️  COLLAPSE HAPPENS AT CARD ENCODING")
                    print("   The card encoder treats all hands the same.")
                else:
                    print("\n⚠️  COLLAPSE HAPPENS AT EXAMPLE ENCODING")
                    print("   The example encoder combines card/output in a way that loses information.")
            else:
                print("\n⚠️  COLLAPSE HAPPENS AT TASK AGGREGATION (ATTENTION)")
                print("   The attention mechanism is producing identical pooled outputs.")
        else:
            print("\n✓ No collapse detected - task embeddings are diverse")


if __name__ == "__main__":
    grammar = build_lean_grammar()
    model = NeuralRecognitionModel(grammar=grammar)

    rules = create_all_rules()
    tasks = create_tasks_from_rules(rules, n_examples=20, seed=42)

    trace_encoding_pipeline(model, tasks)
