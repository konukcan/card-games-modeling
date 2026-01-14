#!/usr/bin/env python3
"""
Test the ContrastiveRecognitionModel embedding normalization fix.

This script verifies that:
1. NEW models with normalize_embeddings=True produce diverse predictions
2. Different encoding modes work correctly
3. The fix preserves semantic structure (similar tasks → similar predictions)
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


def test_encoding_mode(model, tasks, mode_name):
    """Test a specific encoding mode."""
    print(f"\n{'='*60}")
    print(f"Testing: {mode_name}")
    print(f"{'='*60}")

    # Get embeddings and predictions for first 10 tasks
    embeddings = []
    predictions = []
    task_names = []

    with torch.no_grad():
        for task in tasks[:10]:
            emb = model.encode_task_batched(task)
            pred = model.predict_primitives(task)
            embeddings.append(emb.cpu().numpy())
            predictions.append(pred.cpu().numpy())
            task_names.append(task.name)

    embeddings_np = np.array(embeddings)
    predictions_np = np.array(predictions)

    # Embedding statistics
    norms = np.linalg.norm(embeddings_np, axis=1)
    emb_sim = cosine_similarity(embeddings_np)
    upper_tri = emb_sim[np.triu_indices(10, k=1)]

    print(f"\nEmbedding Statistics:")
    print(f"  Mean norm: {np.mean(norms):.4f} ± {np.std(norms):.4f}")
    print(f"  Norm range: [{np.min(norms):.4f}, {np.max(norms):.4f}]")
    print(f"  Mean cosine similarity: {np.mean(upper_tri):.4f}")

    # Prediction statistics
    pred_stds = np.std(predictions_np, axis=1)
    pred_spreads = np.max(predictions_np, axis=1) - np.min(predictions_np, axis=1)

    # Get unique top-5 predictions
    top5_sets = [frozenset(np.argsort(pred)[::-1][:5]) for pred in predictions_np]
    unique_top5 = len(set(top5_sets))

    print(f"\nPrediction Statistics:")
    print(f"  Mean prediction std: {np.mean(pred_stds):.4f}")
    print(f"  Mean prediction spread: {np.mean(pred_spreads):.4f}")
    print(f"  Unique top-5 sets: {unique_top5}/10")

    # Sample predictions
    print(f"\nSample Predictions (top 3 for each task):")
    for i in range(min(5, len(task_names))):
        top3_idx = np.argsort(predictions_np[i])[::-1][:3]
        top3 = [f"{model.primitive_names[idx]}:{predictions_np[i][idx]:.3f}" for idx in top3_idx]
        print(f"  {task_names[i][:30]}: {top3}")

    return {
        'mode': mode_name,
        'unique_top5': unique_top5,
        'mean_spread': np.mean(pred_spreads),
        'mean_emb_norm': np.mean(norms),
        'mean_emb_sim': np.mean(upper_tri)
    }


def main():
    print("=" * 70)
    print("CONTRASTIVE RECOGNITION MODEL - EMBEDDING FIX TEST")
    print("=" * 70)

    # Load grammar and tasks
    print("\nLoading grammar and tasks...")
    grammar = build_lean_grammar()
    rules = create_all_rules()
    tasks = create_tasks_from_rules(rules, n_examples=20, seed=42)
    print(f"  Grammar: {len(grammar.productions)} primitives")
    print(f"  Tasks: {len(tasks)}")

    results = []

    # ========================================================================
    # Test 1: Baseline (normalize_embeddings=False)
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: BASELINE (normalize_embeddings=False)")
    print("=" * 70)

    model_baseline = ContrastiveRecognitionModel(
        grammar=grammar,
        card_hidden=128,
        card_out=64,
        pred_hidden=128,
        normalize_embeddings=False,  # No fix
        encoding_mode='standard'
    )

    result = test_encoding_mode(model_baseline, tasks, "Baseline (no normalization)")
    results.append(result)

    # ========================================================================
    # Test 2: Standard mode with normalization
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: STANDARD MODE WITH NORMALIZATION")
    print("=" * 70)

    model_standard = ContrastiveRecognitionModel(
        grammar=grammar,
        card_hidden=128,
        card_out=64,
        pred_hidden=128,
        normalize_embeddings=True,  # THE FIX
        embedding_scale=20.0,
        encoding_mode='standard'
    )

    result = test_encoding_mode(model_standard, tasks, "Standard + LayerNorm (scale=20)")
    results.append(result)

    # ========================================================================
    # Test 3: Random contrast mode
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: RANDOM CONTRAST MODE")
    print("=" * 70)

    model_random = ContrastiveRecognitionModel(
        grammar=grammar,
        card_hidden=128,
        card_out=64,
        pred_hidden=128,
        normalize_embeddings=True,
        embedding_scale=20.0,
        encoding_mode='random_contrast',
        n_random_hands=10,
        lambda_random=0.5
    )

    result = test_encoding_mode(model_random, tasks, "RandomContrast + LayerNorm")
    results.append(result)

    # ========================================================================
    # Test 4: Triple contrast mode
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST 4: TRIPLE CONTRAST MODE")
    print("=" * 70)

    model_triple = ContrastiveRecognitionModel(
        grammar=grammar,
        card_hidden=128,
        card_out=64,
        pred_hidden=128,
        normalize_embeddings=True,
        embedding_scale=20.0,
        encoding_mode='triple_contrast',
        n_random_hands=10
    )

    result = test_encoding_mode(model_triple, tasks, "TripleContrast + LayerNorm")
    results.append(result)

    # ========================================================================
    # Test 5: Different scale factors
    # ========================================================================
    print("\n" + "=" * 70)
    print("TEST 5: SCALE FACTOR COMPARISON")
    print("=" * 70)

    for scale in [5.0, 10.0, 20.0, 50.0]:
        model = ContrastiveRecognitionModel(
            grammar=grammar,
            card_hidden=128,
            card_out=64,
            pred_hidden=128,
            normalize_embeddings=True,
            embedding_scale=scale,
            encoding_mode='standard'
        )

        # Quick test
        with torch.no_grad():
            predictions = []
            for task in tasks[:10]:
                pred = model.predict_primitives(task)
                predictions.append(pred.cpu().numpy())

        predictions_np = np.array(predictions)
        top5_sets = [frozenset(np.argsort(pred)[::-1][:5]) for pred in predictions_np]
        unique_top5 = len(set(top5_sets))
        pred_spreads = np.max(predictions_np, axis=1) - np.min(predictions_np, axis=1)

        print(f"  Scale={scale}: unique_top5={unique_top5}/10, mean_spread={np.mean(pred_spreads):.4f}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Mode':<35} {'Unique Top-5':<15} {'Mean Spread':<15} {'Emb Norm':<15}")
    print("-" * 80)
    for r in results:
        print(f"{r['mode']:<35} {r['unique_top5']}/10{'':<7} {r['mean_spread']:<15.4f} {r['mean_emb_norm']:<15.4f}")

    # Check if fix worked
    baseline_unique = results[0]['unique_top5']
    fixed_unique = results[1]['unique_top5']

    print("\n" + "=" * 70)
    if fixed_unique > baseline_unique and fixed_unique >= 8:
        print("✅ FIX SUCCESSFUL!")
        print(f"   Baseline: {baseline_unique}/10 unique predictions")
        print(f"   With fix: {fixed_unique}/10 unique predictions")
    else:
        print("⚠️ FIX MAY NEED ADJUSTMENT")
        print(f"   Baseline: {baseline_unique}/10 unique predictions")
        print(f"   With fix: {fixed_unique}/10 unique predictions")
    print("=" * 70)


if __name__ == "__main__":
    main()
