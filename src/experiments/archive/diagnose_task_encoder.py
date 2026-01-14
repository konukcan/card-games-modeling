#!/usr/bin/env python3
"""
Task Encoder Diagnostic Script

This script investigates whether the recognition model's task encoder can
differentiate between different tasks based on their examples.

Key questions:
1. Are task embeddings different for different tasks?
2. Do similar tasks (same family) cluster together?
3. Is the model learning task-specific features or just generic patterns?

Usage:
    python diagnose_task_encoder.py [--model-path PATH]
"""

import sys
import os
import json
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dreamcoder_core.neural_recognition import NeuralRecognitionModel
from dreamcoder_core.primitives import build_lean_grammar
from dreamcoder_core.task import Task
from dreamcoder_core.dreamcoder_original import create_tasks_from_rules
from rules.catalogue import create_all_rules


def load_or_create_model(model_path: str = None) -> NeuralRecognitionModel:
    """Load a trained model or create a fresh one for comparison."""
    grammar = build_lean_grammar()

    if model_path and os.path.exists(model_path):
        print(f"Loading model from: {model_path}")
        model = NeuralRecognitionModel(grammar=grammar)
        model.load(model_path)
        print(f"  Loaded model with {len(model.training_losses)} training steps")
        return model, "trained"
    else:
        print("Creating fresh (untrained) model for comparison")
        model = NeuralRecognitionModel(grammar=grammar)
        return model, "untrained"


def encode_all_tasks(model: NeuralRecognitionModel, tasks: list, primitives: list) -> dict:
    """Encode all tasks and return embeddings with metadata."""
    model.eval()

    results = {
        'embeddings': [],
        'task_names': [],
        'families': [],
        'predictions': [],
        'top_k_primitives': [],
        'primitives': primitives,
    }

    with torch.no_grad():
        for task in tasks:
            # Get embedding
            embedding = model.encode_task(task)
            results['embeddings'].append(embedding.cpu().numpy())
            results['task_names'].append(task.name)

            # Get family if available
            family = getattr(task, 'family', 'unknown')
            results['families'].append(family)

            # Get predictions
            log_probs = model.predict_primitive_probs(task)
            probs = torch.exp(log_probs)
            results['predictions'].append(probs.cpu().numpy())

            # Get top-5 primitives
            top_k = torch.topk(log_probs, k=5)
            top_prims = [primitives[i] for i in top_k.indices.cpu().numpy()]
            results['top_k_primitives'].append(top_prims)

    results['embeddings'] = np.array(results['embeddings'])
    results['predictions'] = np.array(results['predictions'])

    return results


def analyze_embedding_diversity(embeddings: np.ndarray, task_names: list) -> dict:
    """Analyze how diverse the embeddings are."""
    n_tasks = len(embeddings)

    # Compute pairwise cosine similarities
    sim_matrix = cosine_similarity(embeddings)

    # Statistics
    upper_tri = sim_matrix[np.triu_indices(n_tasks, k=1)]

    stats = {
        'mean_similarity': float(np.mean(upper_tri)),
        'std_similarity': float(np.std(upper_tri)),
        'min_similarity': float(np.min(upper_tri)),
        'max_similarity': float(np.max(upper_tri)),
        'n_identical_pairs': int(np.sum(upper_tri > 0.999)),
        'n_very_similar_pairs': int(np.sum(upper_tri > 0.95)),
        'similarity_matrix': sim_matrix,
    }

    # Find most similar pairs (excluding self)
    flat_idx = np.argsort(upper_tri)[::-1][:10]
    most_similar = []
    pair_idx = 0
    for i in range(n_tasks):
        for j in range(i+1, n_tasks):
            if pair_idx in flat_idx:
                most_similar.append({
                    'task1': task_names[i],
                    'task2': task_names[j],
                    'similarity': float(sim_matrix[i, j])
                })
            pair_idx += 1
    stats['most_similar_pairs'] = most_similar[:5]

    # Find most different pairs
    least_similar = []
    flat_idx_least = np.argsort(upper_tri)[:10]
    pair_idx = 0
    for i in range(n_tasks):
        for j in range(i+1, n_tasks):
            if pair_idx in flat_idx_least:
                least_similar.append({
                    'task1': task_names[i],
                    'task2': task_names[j],
                    'similarity': float(sim_matrix[i, j])
                })
            pair_idx += 1
    stats['least_similar_pairs'] = least_similar[:5]

    return stats


def analyze_prediction_diversity(predictions: np.ndarray, primitives: list, task_names: list) -> dict:
    """Analyze how diverse the primitive predictions are."""
    n_tasks = predictions.shape[0]

    # Get top-5 for each task
    top5_per_task = []
    for i in range(n_tasks):
        top5_idx = np.argsort(predictions[i])[::-1][:5]
        top5_per_task.append(set(top5_idx))

    # Count how many tasks share the SAME top-5
    overlap_counts = []
    for i in range(n_tasks):
        for j in range(i+1, n_tasks):
            overlap = len(top5_per_task[i] & top5_per_task[j])
            overlap_counts.append(overlap)

    # Count unique top-5 sets
    unique_top5 = set(frozenset(s) for s in top5_per_task)

    # Find the most common top-5 set
    from collections import Counter
    top5_counter = Counter(frozenset(s) for s in top5_per_task)
    most_common_top5, most_common_count = top5_counter.most_common(1)[0]
    most_common_prims = [primitives[i] for i in most_common_top5]

    # Compute prediction similarity matrix
    pred_sim = cosine_similarity(predictions)
    upper_tri = pred_sim[np.triu_indices(n_tasks, k=1)]

    stats = {
        'n_unique_top5_sets': len(unique_top5),
        'most_common_top5': most_common_prims,
        'most_common_top5_count': most_common_count,
        'mean_top5_overlap': float(np.mean(overlap_counts)),
        'all_same_top5': len(unique_top5) == 1,
        'mean_prediction_similarity': float(np.mean(upper_tri)),
        'n_identical_predictions': int(np.sum(upper_tri > 0.999)),
    }

    # Show top-5 for a sample of tasks
    sample_predictions = {}
    for i in range(min(10, n_tasks)):
        top5_idx = np.argsort(predictions[i])[::-1][:5]
        sample_predictions[task_names[i]] = [
            f"{primitives[idx]}: {predictions[i, idx]:.4f}"
            for idx in top5_idx
        ]
    stats['sample_predictions'] = sample_predictions

    return stats


def analyze_family_clustering(embeddings: np.ndarray, families: list) -> dict:
    """Check if tasks from the same family cluster together."""
    unique_families = list(set(families))
    family_to_idx = defaultdict(list)
    for i, f in enumerate(families):
        family_to_idx[f].append(i)

    sim_matrix = cosine_similarity(embeddings)

    # Compute within-family vs between-family similarity
    within_sims = []
    between_sims = []

    for i in range(len(embeddings)):
        for j in range(i+1, len(embeddings)):
            if families[i] == families[j]:
                within_sims.append(sim_matrix[i, j])
            else:
                between_sims.append(sim_matrix[i, j])

    stats = {
        'n_families': len(unique_families),
        'families': unique_families,
        'mean_within_family_sim': float(np.mean(within_sims)) if within_sims else None,
        'mean_between_family_sim': float(np.mean(between_sims)) if between_sims else None,
        'family_separation': None,
    }

    if within_sims and between_sims:
        # Good clustering = high within, low between
        stats['family_separation'] = stats['mean_within_family_sim'] - stats['mean_between_family_sim']

    return stats


def visualize_embeddings(embeddings: np.ndarray, task_names: list, families: list,
                         output_dir: str = None):
    """Create t-SNE and PCA visualizations."""
    # Use PCA if too few samples for t-SNE
    if len(embeddings) < 10:
        reducer = PCA(n_components=2)
        reduced = reducer.fit_transform(embeddings)
        method = "PCA"
    else:
        # t-SNE
        perplexity = min(30, len(embeddings) - 1)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced = reducer.fit_transform(embeddings)
        method = "t-SNE"

    # Create family color mapping
    unique_families = list(set(families))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_families)))
    family_colors = {f: colors[i] for i, f in enumerate(unique_families)}

    plt.figure(figsize=(12, 8))

    for family in unique_families:
        mask = [f == family for f in families]
        idxs = np.where(mask)[0]
        plt.scatter(reduced[idxs, 0], reduced[idxs, 1],
                   c=[family_colors[family]], label=family, s=100, alpha=0.7)

    # Add task name labels
    for i, name in enumerate(task_names):
        plt.annotate(name[:15], (reduced[i, 0], reduced[i, 1]),
                    fontsize=6, alpha=0.7)

    plt.title(f"Task Embeddings ({method})")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, 'task_embeddings_visualization.png'),
                   dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to {output_dir}/task_embeddings_visualization.png")
    plt.close()

    return reduced


def visualize_similarity_matrix(sim_matrix: np.ndarray, task_names: list,
                                output_dir: str = None):
    """Visualize the similarity matrix as a heatmap."""
    plt.figure(figsize=(16, 14))

    # Truncate long names
    short_names = [n[:12] for n in task_names]

    sns.heatmap(sim_matrix, xticklabels=short_names, yticklabels=short_names,
                cmap='RdYlBu_r', vmin=0, vmax=1, annot=False)
    plt.title("Task Embedding Cosine Similarity Matrix")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()

    if output_dir:
        plt.savefig(os.path.join(output_dir, 'similarity_matrix.png'),
                   dpi=150, bbox_inches='tight')
        print(f"  Saved similarity matrix to {output_dir}/similarity_matrix.png")
    plt.close()


def run_diagnostic(model_path: str = None, output_dir: str = None):
    """Run the full diagnostic."""
    print("=" * 70)
    print("TASK ENCODER DIAGNOSTIC")
    print("=" * 70)

    # Setup output directory
    if output_dir is None:
        output_dir = "results/diagnostics"
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    print("\n1. LOADING MODEL")
    print("-" * 40)
    model, model_status = load_or_create_model(model_path)
    grammar = build_lean_grammar()
    primitives = [str(p) for p in grammar.primitives()]

    # Load tasks
    print("\n2. LOADING TASKS")
    print("-" * 40)
    rules = create_all_rules()
    tasks = create_tasks_from_rules(rules, n_examples=20, seed=42)
    print(f"  Loaded {len(tasks)} tasks from catalogue")

    # Add family information to tasks
    rule_by_id = {r.id: r for r in rules}
    for task in tasks:
        if task.name in rule_by_id:
            task.family = rule_by_id[task.name].family
        else:
            task.family = "unknown"

    families = [task.family for task in tasks]
    print(f"  Families: {set(families)}")

    # Encode all tasks
    print("\n3. ENCODING TASKS")
    print("-" * 40)
    results = encode_all_tasks(model, tasks, primitives)
    print(f"  Encoded {len(results['embeddings'])} tasks")
    print(f"  Embedding dimension: {results['embeddings'].shape[1]}")

    # Analyze embedding diversity
    print("\n4. EMBEDDING DIVERSITY ANALYSIS")
    print("-" * 40)
    emb_stats = analyze_embedding_diversity(results['embeddings'], results['task_names'])

    print(f"  Mean pairwise similarity: {emb_stats['mean_similarity']:.4f}")
    print(f"  Std pairwise similarity:  {emb_stats['std_similarity']:.4f}")
    print(f"  Min similarity:           {emb_stats['min_similarity']:.4f}")
    print(f"  Max similarity:           {emb_stats['max_similarity']:.4f}")
    print(f"  Identical pairs (>0.999): {emb_stats['n_identical_pairs']}")
    print(f"  Very similar (>0.95):     {emb_stats['n_very_similar_pairs']}")

    if emb_stats['mean_similarity'] > 0.95:
        print("\n  ⚠️  WARNING: Embeddings are nearly IDENTICAL!")
        print("     The model cannot differentiate between tasks.")
    elif emb_stats['mean_similarity'] > 0.8:
        print("\n  ⚠️  WARNING: Embeddings are VERY SIMILAR")
        print("     Limited task differentiation.")
    else:
        print("\n  ✓ Embeddings show reasonable diversity")

    print("\n  Most similar pairs:")
    for pair in emb_stats['most_similar_pairs'][:3]:
        print(f"    {pair['task1']} <-> {pair['task2']}: {pair['similarity']:.4f}")

    print("\n  Least similar pairs:")
    for pair in emb_stats['least_similar_pairs'][:3]:
        print(f"    {pair['task1']} <-> {pair['task2']}: {pair['similarity']:.4f}")

    # Analyze prediction diversity
    print("\n5. PREDICTION DIVERSITY ANALYSIS")
    print("-" * 40)
    pred_stats = analyze_prediction_diversity(
        results['predictions'], primitives, results['task_names']
    )

    print(f"  Unique top-5 primitive sets: {pred_stats['n_unique_top5_sets']}")
    print(f"  All tasks have same top-5:   {pred_stats['all_same_top5']}")
    print(f"  Mean top-5 overlap:          {pred_stats['mean_top5_overlap']:.2f}/5")
    print(f"  Mean prediction similarity:  {pred_stats['mean_prediction_similarity']:.4f}")

    if pred_stats['all_same_top5']:
        print("\n  ⚠️  CRITICAL: ALL tasks receive IDENTICAL top-5 predictions!")
        print(f"     Most common top-5: {pred_stats['most_common_top5']}")
    elif pred_stats['n_unique_top5_sets'] < 5:
        print("\n  ⚠️  WARNING: Very few unique prediction patterns")
    else:
        print("\n  ✓ Predictions show some diversity")

    print("\n  Sample predictions (top-5 per task):")
    for task_name, preds in list(pred_stats['sample_predictions'].items())[:5]:
        print(f"    {task_name}:")
        for p in preds[:3]:
            print(f"      {p}")

    # Analyze family clustering
    print("\n6. FAMILY CLUSTERING ANALYSIS")
    print("-" * 40)
    fam_stats = analyze_family_clustering(results['embeddings'], results['families'])

    print(f"  Number of families: {fam_stats['n_families']}")
    if fam_stats['mean_within_family_sim']:
        print(f"  Within-family similarity:  {fam_stats['mean_within_family_sim']:.4f}")
        print(f"  Between-family similarity: {fam_stats['mean_between_family_sim']:.4f}")
        print(f"  Family separation:         {fam_stats['family_separation']:.4f}")

        if fam_stats['family_separation'] > 0.05:
            print("\n  ✓ Same-family tasks cluster together (good!)")
        elif fam_stats['family_separation'] < -0.01:
            print("\n  ⚠️  Same-family tasks are LESS similar than different families!")
        else:
            print("\n  ⚠️  No meaningful family clustering")

    # Create visualizations
    print("\n7. CREATING VISUALIZATIONS")
    print("-" * 40)
    visualize_embeddings(results['embeddings'], results['task_names'],
                        results['families'], output_dir)
    visualize_similarity_matrix(emb_stats['similarity_matrix'],
                               results['task_names'], output_dir)

    # Save detailed results
    print("\n8. SAVING RESULTS")
    print("-" * 40)

    # Prepare JSON-serializable results
    json_results = {
        'model_status': model_status,
        'n_tasks': len(tasks),
        'embedding_dim': int(results['embeddings'].shape[1]),
        'embedding_stats': {
            'mean_similarity': emb_stats['mean_similarity'],
            'std_similarity': emb_stats['std_similarity'],
            'min_similarity': emb_stats['min_similarity'],
            'max_similarity': emb_stats['max_similarity'],
            'n_identical_pairs': emb_stats['n_identical_pairs'],
            'n_very_similar_pairs': emb_stats['n_very_similar_pairs'],
            'most_similar_pairs': emb_stats['most_similar_pairs'],
            'least_similar_pairs': emb_stats['least_similar_pairs'],
        },
        'prediction_stats': {
            'n_unique_top5_sets': pred_stats['n_unique_top5_sets'],
            'all_same_top5': pred_stats['all_same_top5'],
            'most_common_top5': pred_stats['most_common_top5'],
            'most_common_top5_count': pred_stats['most_common_top5_count'],
            'mean_top5_overlap': pred_stats['mean_top5_overlap'],
            'mean_prediction_similarity': pred_stats['mean_prediction_similarity'],
            'sample_predictions': pred_stats['sample_predictions'],
        },
        'family_stats': fam_stats,
        'top_k_primitives_per_task': {
            name: prims for name, prims in
            zip(results['task_names'], results['top_k_primitives'])
        },
    }

    json_path = os.path.join(output_dir, 'diagnostic_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"  Saved results to: {json_path}")

    # Save embeddings for further analysis
    np.save(os.path.join(output_dir, 'task_embeddings.npy'), results['embeddings'])
    print(f"  Saved embeddings to: {output_dir}/task_embeddings.npy")

    # Summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    issues = []
    if emb_stats['mean_similarity'] > 0.95:
        issues.append("CRITICAL: Task embeddings are nearly identical")
    if pred_stats['all_same_top5']:
        issues.append("CRITICAL: All tasks get same primitive predictions")
    if fam_stats['family_separation'] and fam_stats['family_separation'] < 0.01:
        issues.append("WARNING: No family-based clustering")

    if not issues:
        print("✓ No major issues detected")
    else:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")

        print("\nPOSSIBLE ROOT CAUSES:")
        if "embeddings are nearly identical" in str(issues):
            print("  1. Example encoder may not be extracting meaningful features")
            print("  2. Task aggregation (attention) may be collapsing to a constant")
            print("  3. Input examples may lack distinguishing information")
        if "same primitive predictions" in str(issues):
            print("  4. Training signal may be teaching marginal P(prim) not P(prim|task)")
            print("  5. Too few solved tasks for conditional learning")

    print("\n" + "=" * 70)
    return json_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose task encoder behavior")
    parser.add_argument('--model-path', type=str, default=None,
                       help="Path to trained model (default: use fresh model)")
    parser.add_argument('--output-dir', type=str, default='results/diagnostics',
                       help="Output directory for results")

    args = parser.parse_args()

    # Try to find the most recent trained model
    if args.model_path is None:
        # Look for recent experiment results
        results_dir = Path(__file__).parent.parent / "results" / "warmstart_experiment"
        if results_dir.exists():
            subdirs = sorted(results_dir.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True)
            for subdir in subdirs[:5]:
                model_file = subdir / "recognition_model.pt"
                if model_file.exists():
                    args.model_path = str(model_file)
                    print(f"Found model: {args.model_path}")
                    break

    run_diagnostic(args.model_path, args.output_dir)
