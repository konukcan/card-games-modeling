#!/usr/bin/env python3
"""
Diagnostic script to check if task encodings are differentiated.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import json
import numpy as np
from dreamcoder_core.neural_recognition import NeuralRecognitionModel
from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.dreamcoder_original import create_tasks_from_rules
from rules.catalogue import create_all_rules as get_catalogue_rules
from rules.pretraining_rules import get_all_pretraining_rules

RESULTS_DIR = Path("results/warmstart_experiment")

def cosine_similarity(v1, v2):
    """Compute cosine similarity between two vectors."""
    v1_np = v1.cpu().numpy() if hasattr(v1, 'cpu') else v1
    v2_np = v2.cpu().numpy() if hasattr(v2, 'cpu') else v2
    return np.dot(v1_np, v2_np) / (np.linalg.norm(v1_np) * np.linalg.norm(v2_np))

def main():
    grammar = build_lean_grammar()

    # Check what was solved and what primitives were used
    print("="*60)
    print("CHECKING TRAINING DATA - WHAT PRIMITIVES WERE IN SOLUTIONS?")
    print("="*60)

    # Check pretraining results from neural model (which solved the most)
    neural_results_path = RESULTS_DIR / "neural_BOTH_20251225_145356" / "results_WARM.json"
    with open(neural_results_path) as f:
        neural_results = json.load(f)

    # Check what primitives appear in solved tasks across both phases
    all_prims = {}

    # Pretraining phase
    pretraining_log = RESULTS_DIR / "neural_BOTH_20251225_145356" / "pretraining_log.json"
    if pretraining_log.exists():
        with open(pretraining_log) as f:
            pretrain_log = json.load(f)
        print("\nPRETRAINING PHASE - Neural Model:")
        for task_name, data in pretrain_log.items():
            if data.get('solved'):
                prims = data.get('primitives_used', [])
                print(f"  {task_name}: {prims}")
                for p in prims:
                    all_prims[p] = all_prims.get(p, 0) + 1

    # Main training phase
    main_metrics = neural_results.get('main_training', {}).get('task_metrics', {})
    print("\nMAIN TRAINING PHASE - Neural Model:")
    for task_name, metrics in main_metrics.items():
        if metrics.get('solved'):
            prims = metrics.get('primitives_used', [])
            print(f"  {task_name}: {prims}")
            for p in prims:
                all_prims[p] = all_prims.get(p, 0) + 1

    print(f"\n\nPrimitive frequency in ALL neural solutions:")
    for p, count in sorted(all_prims.items(), key=lambda x: -x[1]):
        print(f"  {p}: {count}")

    # Now check what solutions look like from pretraining
    print("\n" + "="*60)
    print("CHECKING PRETRAINING SOLUTIONS IN DETAIL")
    print("="*60)

    # The pretraining logs may have solution info
    pretrain_tasks = get_all_pretraining_rules()
    pretrain_solved = neural_results.get('pretraining', {}).get('solved_tasks', [])
    print(f"\nNeural model solved {len(pretrain_solved)} / {len(pretrain_tasks)} pretraining tasks")
    print(f"Solved tasks: {pretrain_solved}")

    # Now let's check task encoding similarity
    print("\n" + "="*60)
    print("TASK ENCODING SIMILARITY ANALYSIS")
    print("="*60)

    # Load models
    softmax_path = RESULTS_DIR / "contrastive_softmax_WARM_20251226_174158" / "pretrained_recognition.pt"
    softmax = ContrastiveRecognitionModel(grammar, card_hidden=64, card_out=32, pred_hidden=64, output_mode='softmax')
    softmax.load(str(softmax_path))
    softmax.eval()

    # Create a subset of catalogue tasks
    catalogue_rules = get_catalogue_rules()
    tasks = create_tasks_from_rules(catalogue_rules[:10], n_examples=100, n_holdout=20, hand_size=6)

    # Get task encodings
    task_encodings = {}
    with torch.no_grad():
        for task in tasks:
            τ = softmax.encode_task_batched(task)
            task_encodings[task.name] = τ

    # Compute pairwise similarities
    print("\nPairwise cosine similarities between task encodings:")
    task_list = list(task_encodings.keys())
    similarity_matrix = []
    for t1 in task_list:
        row = []
        for t2 in task_list:
            sim = cosine_similarity(task_encodings[t1], task_encodings[t2])
            row.append(sim)
        similarity_matrix.append(row)
        print(f"{t1}: {[f'{s:.3f}' for s in row]}")

    avg_off_diagonal = []
    for i, row in enumerate(similarity_matrix):
        for j, val in enumerate(row):
            if i != j:
                avg_off_diagonal.append(val)

    print(f"\nAverage off-diagonal similarity: {np.mean(avg_off_diagonal):.4f}")
    print(f"(1.0 = identical encodings, 0.0 = orthogonal)")

    # Also check encoding norms - are some tasks getting near-zero encodings?
    print("\nTask encoding norms:")
    for name, enc in task_encodings.items():
        print(f"  {name}: {enc.norm():.4f}")

if __name__ == '__main__':
    main()
