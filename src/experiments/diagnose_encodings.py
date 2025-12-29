#!/usr/bin/env python3
"""
Diagnostic script to check if task encodings are differentiated.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from dreamcoder_core.neural_recognition import NeuralRecognitionModel
from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.dreamcoder_original import create_tasks_from_rules
from rules.catalogue import create_all_rules as get_catalogue_rules

RESULTS_DIR = Path("results/warmstart_experiment")

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    v1_np = v1.cpu().numpy() if hasattr(v1, 'cpu') else v1
    v2_np = v2.cpu().numpy() if hasattr(v2, 'cpu') else v2
    return np.dot(v1_np, v2_np) / (np.linalg.norm(v1_np) * np.linalg.norm(v2_np))

def main():
    grammar = build_lean_grammar()

    # Create diverse test tasks
    all_rules = get_catalogue_rules()
    # Pick diverse rules that should have different primitive needs
    task_names = [
        "Sorted_by_rank",      # Needs: rank comparison, ordering
        "Uniform_color",       # Needs: all_same_color
        "Has_pair_ranks",      # Needs: n_unique_ranks, le
        "Suits_palindrome",    # Needs: reverse, map_suit, eq
        "AP_len3_anywhere_anyk"  # Needs: arithmetic, has_ap
    ]

    selected_rules = [r for r in all_rules if r.name in task_names]
    tasks = create_tasks_from_rules(selected_rules, n_examples=100, n_holdout=20, hand_size=6)

    print(f"Selected {len(tasks)} diverse tasks: {[t.name for t in tasks]}")

    # Load contrastive softmax model
    softmax_path = RESULTS_DIR / "contrastive_softmax_WARM_20251226_174158" / "pretrained_recognition.pt"
    softmax = ContrastiveRecognitionModel(grammar, card_hidden=64, card_out=32, pred_hidden=64, output_mode='softmax')
    softmax.load(str(softmax_path))
    softmax.eval()

    print("\n" + "="*60)
    print("CONTRASTIVE MODEL ANALYSIS")
    print("="*60)

    # Get task encodings (τ vectors)
    task_encodings = {}
    with torch.no_grad():
        for task in tasks:
            τ = softmax.encode_task_batched(task)
            task_encodings[task.name] = τ
            print(f"\n{task.name}:")
            print(f"  τ shape: {τ.shape}")
            print(f"  τ range: [{τ.min():.4f}, {τ.max():.4f}]")
            print(f"  τ mean: {τ.mean():.4f}, std: {τ.std():.4f}")
            print(f"  τ norm: {τ.norm():.4f}")

    # Compute pairwise similarities
    print("\n" + "="*60)
    print("PAIRWISE COSINE SIMILARITIES")
    print("="*60)

    task_list = list(task_encodings.keys())
    for i, t1 in enumerate(task_list):
        for t2 in task_list[i+1:]:
            sim = cosine_similarity(task_encodings[t1], task_encodings[t2])
            print(f"{t1} vs {t2}: {sim:.4f}")

    # Check what the primitive head receives
    print("\n" + "="*60)
    print("PRIMITIVE HEAD INPUT ANALYSIS")
    print("="*60)

    with torch.no_grad():
        for task in tasks:
            τ = softmax.encode_task_batched(task).unsqueeze(0)

            # Get the primitive head's input
            print(f"\n{task.name}:")
            print(f"  Input to primitive_head: shape={τ.shape}")

            # Get output
            output = softmax.primitive_head(τ)
            print(f"  Output from primitive_head: shape={output.shape}")
            print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

    # Check if the problem is in the contrastive encoding (τ = pos - neg)
    print("\n" + "="*60)
    print("RAW CARD EMBEDDINGS ANALYSIS")
    print("="*60)

    with torch.no_grad():
        for task in tasks[:2]:  # Just first 2 tasks
            print(f"\n{task.name}:")

            # Get positive and negative examples
            pos_examples = task.examples[:5]  # First 5 positive
            neg_examples = task.examples[5:10] if len(task.examples) > 5 else []

            print(f"  Positive examples: {len(pos_examples)}")
            print(f"  Negative examples: {len(neg_examples)}")

            # Encode positive examples
            pos_encs = []
            for hand, _ in pos_examples:
                enc = softmax.encode_hand(list(hand))
                pos_encs.append(enc)

            if pos_encs:
                pos_stack = torch.stack(pos_encs)
                pos_mean = pos_stack.mean(dim=0)
                print(f"  Positive embeddings mean: shape={pos_mean.shape}, norm={pos_mean.norm():.4f}")

            # If we have negative examples, compare
            if neg_examples:
                neg_encs = []
                for hand, _ in neg_examples:
                    enc = softmax.encode_hand(list(hand))
                    neg_encs.append(enc)

                neg_stack = torch.stack(neg_encs)
                neg_mean = neg_stack.mean(dim=0)
                print(f"  Negative embeddings mean: shape={neg_mean.shape}, norm={neg_mean.norm():.4f}")

                # Compute τ = pos - neg
                τ = pos_mean - neg_mean
                print(f"  τ = pos - neg: norm={τ.norm():.4f}")

    # Final check: what primitives were actually used in solved tasks?
    print("\n" + "="*60)
    print("CHECKING TRAINING DATA - WHAT PRIMITIVES WERE IN SOLUTIONS?")
    print("="*60)

    # Load results to see what primitives appeared in solved tasks
    import json
    softmax_results_path = RESULTS_DIR / "contrastive_softmax_WARM_20251226_174158" / "results_WARM.json"
    if softmax_results_path.exists():
        with open(softmax_results_path) as f:
            results = json.load(f)

        # Check pretraining solved tasks
        pretraining = results.get('pretraining', {})
        solved = pretraining.get('solved_tasks', [])
        print(f"\nPretraining solved {len(solved)} tasks")

        # Check main training
        main = results.get('main_training', {})
        task_metrics = main.get('task_metrics', {})
        primitives_seen = {}
        for task_name, metrics in task_metrics.items():
            if metrics.get('solved'):
                prims = metrics.get('primitives_used', [])
                print(f"  {task_name}: {prims}")
                for p in prims:
                    primitives_seen[p] = primitives_seen.get(p, 0) + 1

        print(f"\nPrimitive frequency in solutions:")
        for p, count in sorted(primitives_seen.items(), key=lambda x: -x[1]):
            print(f"  {p}: {count}")

if __name__ == '__main__':
    main()
