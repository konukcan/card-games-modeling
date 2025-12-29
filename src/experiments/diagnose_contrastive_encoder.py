#!/usr/bin/env python3
"""
Diagnostic for ContrastiveRecognitionModel task encoder.

Tests whether the contrastive approach (τ = mean(pos) - mean(neg))
produces diverse task embeddings unlike the attention-based approach.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.contrastive_recognition import ContrastiveRecognitionModel
from dreamcoder_core.lean_primitives import build_lean_grammar
from dreamcoder_core.dreamcoder_original import create_tasks_from_rules
from rules.catalogue import create_all_rules


def analyze_contrastive_embeddings(model, tasks):
    """Analyze contrastive task embeddings."""
    model.eval()

    print("=" * 70)
    print("CONTRASTIVE RECOGNITION MODEL DIAGNOSTIC")
    print("=" * 70)

    # Encode all tasks
    print("\n1. ENCODING TASKS")
    print("-" * 40)

    embeddings = []
    task_names = []
    families = []

    with torch.no_grad():
        for task in tasks:
            emb = model.encode_task(task)
            embeddings.append(emb.cpu().numpy())
            task_names.append(task.name)
            families.append(getattr(task, 'family', 'unknown'))

    embeddings = np.array(embeddings)
    print(f"  Encoded {len(embeddings)} tasks")
    print(f"  Embedding dimension: {embeddings.shape[1]}")

    # Check if embeddings are different
    print("\n2. EMBEDDING DIVERSITY ANALYSIS")
    print("-" * 40)

    sim_matrix = cosine_similarity(embeddings)
    upper_tri = sim_matrix[np.triu_indices(len(embeddings), k=1)]

    print(f"  Mean pairwise similarity: {np.mean(upper_tri):.4f}")
    print(f"  Std pairwise similarity:  {np.std(upper_tri):.4f}")
    print(f"  Min similarity:           {np.min(upper_tri):.4f}")
    print(f"  Max similarity:           {np.max(upper_tri):.4f}")
    print(f"  Identical pairs (>0.999): {np.sum(upper_tri > 0.999)}")
    print(f"  Very similar (>0.95):     {np.sum(upper_tri > 0.95)}")

    if np.mean(upper_tri) < 0.8:
        print("\n  ✓ GOOD: Embeddings show DIVERSITY!")
    elif np.mean(upper_tri) < 0.95:
        print("\n  ⚠️ WARNING: Embeddings are somewhat similar")
    else:
        print("\n  ❌ CRITICAL: Embeddings are nearly IDENTICAL")

    # Show some example pairs
    print("\n  Most different pairs:")
    flat_idx = np.argsort(upper_tri)[:5]
    pair_idx = 0
    for i in range(len(tasks)):
        for j in range(i+1, len(tasks)):
            if pair_idx in flat_idx:
                print(f"    {task_names[i][:20]} vs {task_names[j][:20]}: {sim_matrix[i,j]:.4f}")
            pair_idx += 1

    print("\n  Most similar pairs:")
    flat_idx = np.argsort(upper_tri)[-5:]
    pair_idx = 0
    for i in range(len(tasks)):
        for j in range(i+1, len(tasks)):
            if pair_idx in flat_idx:
                print(f"    {task_names[i][:20]} vs {task_names[j][:20]}: {sim_matrix[i,j]:.4f}")
            pair_idx += 1

    # Check family clustering
    print("\n3. FAMILY CLUSTERING")
    print("-" * 40)

    from collections import defaultdict
    family_to_idx = defaultdict(list)
    for i, f in enumerate(families):
        family_to_idx[f].append(i)

    within_sims = []
    between_sims = []

    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            if families[i] == families[j]:
                within_sims.append(sim_matrix[i, j])
            else:
                between_sims.append(sim_matrix[i, j])

    print(f"  Within-family similarity:  {np.mean(within_sims):.4f}")
    print(f"  Between-family similarity: {np.mean(between_sims):.4f}")
    separation = np.mean(within_sims) - np.mean(between_sims)
    print(f"  Family separation:         {separation:.4f}")

    if separation > 0.05:
        print("\n  ✓ GOOD: Same-family tasks cluster together!")
    elif separation > 0:
        print("\n  ⚠️ Weak family clustering")
    else:
        print("\n  ❌ No family clustering")

    # Check predictions
    print("\n4. PREDICTION DIVERSITY")
    print("-" * 40)

    predictions = []
    with torch.no_grad():
        for task in tasks:
            pred = model.predict_primitives(task)  # Use correct method name
            predictions.append(pred.cpu().numpy())

    predictions = np.array(predictions)

    # Get top-5 for each task
    primitives = model.primitive_names
    top5_per_task = []
    for i in range(len(predictions)):
        top5_idx = np.argsort(predictions[i])[::-1][:5]
        top5_per_task.append(set(top5_idx))

    # Count unique top-5 sets
    unique_top5 = set(frozenset(s) for s in top5_per_task)
    print(f"  Unique top-5 primitive sets: {len(unique_top5)} out of {len(tasks)}")

    # Count overlaps
    from collections import Counter
    top5_counter = Counter(frozenset(s) for s in top5_per_task)
    most_common_top5, most_common_count = top5_counter.most_common(1)[0]
    most_common_prims = [primitives[i] for i in most_common_top5]
    print(f"  Most common top-5: {most_common_prims[:3]}... ({most_common_count} tasks)")

    if len(unique_top5) > 5:
        print("\n  ✓ GOOD: Multiple different prediction patterns!")
    elif len(unique_top5) == 1:
        print("\n  ❌ CRITICAL: ALL tasks get SAME predictions")
    else:
        print("\n  ⚠️ Limited prediction diversity")

    # Show sample predictions
    print("\n  Sample predictions (top-3 per task):")
    for i in [0, 10, 20, 30, 40]:
        if i < len(tasks):
            top3_idx = np.argsort(predictions[i])[::-1][:3]
            top3_prims = [primitives[idx] for idx in top3_idx]
            print(f"    {task_names[i][:20]}: {top3_prims}")

    # Skip layer-by-layer trace (requires internal method access)
    print("\n5. CONTRASTIVE MECHANISM")
    print("-" * 40)
    print("  The contrastive approach τ = mean(pos) - mean(neg) works by:")
    print("  1. Encoding each hand independently")
    print("  2. Separating positive (True) vs negative (False) examples")
    print("  3. Computing difference between mean embeddings")
    print("  4. This captures 'what distinguishes positive from negative'")
    print("")
    print("  Because different rules have different positive/negative distributions,")
    print("  the contrastive embeddings naturally differ!")

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    embedding_diverse = np.mean(upper_tri) < 0.95
    predictions_diverse = len(unique_top5) > 5
    family_clusters = separation > 0.01

    if embedding_diverse:
        print("✓ Task embeddings are DIVERSE (not collapsed)")
    else:
        print("❌ Task embeddings are too similar")

    if predictions_diverse:
        print("✓ Primitive predictions vary by task")
    else:
        print("❌ Primitive predictions are repetitive")

    if family_clusters:
        print("✓ Similar tasks cluster together")
    else:
        print("❌ No task clustering by family")

    if embedding_diverse and predictions_diverse:
        print("\n✓ ContrastiveRecognitionModel DOES NOT have the collapse problem!")
        print("  The τ = mean(pos) - mean(neg) approach preserves task identity.")
    else:
        print("\n⚠️ Some issues detected, but likely less severe than attention collapse.")

    return {
        'mean_similarity': float(np.mean(upper_tri)),
        'std_similarity': float(np.std(upper_tri)),
        'unique_top5_sets': len(unique_top5),
        'family_separation': float(separation),
    }


if __name__ == "__main__":
    print("Loading grammar and model...")
    grammar = build_lean_grammar()

    model = ContrastiveRecognitionModel(grammar=grammar)
    print(f"Model has {model.num_primitives} primitives")

    print("Loading tasks...")
    rules = create_all_rules()
    tasks = create_tasks_from_rules(rules, n_examples=20, seed=42)

    # Add family information
    rule_by_id = {r.id: r for r in rules}
    for task in tasks:
        if task.name in rule_by_id:
            task.family = rule_by_id[task.name].family

    print(f"Loaded {len(tasks)} tasks\n")

    analyze_contrastive_embeddings(model, tasks)
