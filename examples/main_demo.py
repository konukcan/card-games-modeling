"""
DreamCoder Card Game Modeling - Main Demo Script

This script demonstrates the complete pipeline:
1. Load rules from catalogue
2. Generate synthetic training tasks
3. Extract features from task examples
4. Train recognition network
5. Run program enumeration with neural guidance
6. Visualize results

Run with: cd src && python ../examples/main_demo.py
"""

import sys
import os
from pathlib import Path

# Add src to path (script is in examples/, src is a sibling directory)
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import json
from datetime import datetime

# Import our modules
from rules.cards import Card, Hand, sample_hand, hand_to_string
from rules.primitives import *
from rules.catalogue import ALL_RULES, get_rule, RULE_DICT


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def extract_task_features(examples: List[Tuple[Hand, bool]]) -> np.ndarray:
    """
    Extract features from a task (set of labeled examples).

    Features (104 dimensions):
    - Per-example statistics (13 features × 8 examples = 104)
      - rank_mean, rank_std, rank_min, rank_max
      - suit_entropy, color_uniformity
      - has_pair, is_sorted
      - palindrome_suits, palindrome_colors, palindrome_ranks
      - first_last_same_suit, first_last_same_color

    Args:
        examples: List of (hand, label) pairs

    Returns:
        Feature vector (104-dim)
    """
    features = []

    for hand, label in examples:
        # Rank statistics
        rank_vals = [get_rank_val(c) for c in hand]
        features.extend([
            np.mean(rank_vals),
            np.std(rank_vals),
            np.min(rank_vals),
            np.max(rank_vals),
        ])

        # Suit statistics
        suits = [get_suit(c) for c in hand]
        suit_counts = {s: suits.count(s) for s in Suit}
        suit_probs = np.array([suit_counts[s] / len(hand) for s in Suit])
        suit_entropy = -np.sum(suit_probs * np.log(suit_probs + 1e-10))
        features.append(suit_entropy)

        # Color uniformity
        colors = [get_color(c) for c in hand]
        color_uniform = 1.0 if len(set(colors)) == 1 else 0.0
        features.append(color_uniform)

        # Structural features
        ranks_only = [get_rank(c) for c in hand]
        has_pair = len(ranks_only) != len(set(ranks_only))
        features.append(1.0 if has_pair else 0.0)

        sorted_check = is_sorted(hand, get_rank_val, strict=False)
        features.append(1.0 if sorted_check else 0.0)

        # Palindrome checks
        pal_suits = is_palindrome(suits)
        pal_colors = is_palindrome(colors)
        pal_ranks = is_palindrome(ranks_only)
        features.extend([
            1.0 if pal_suits else 0.0,
            1.0 if pal_colors else 0.0,
            1.0 if pal_ranks else 0.0,
        ])

        # Terminal equality
        if len(hand) >= 2:
            first_last_suit = 1.0 if get_suit(hand[0]) == get_suit(hand[-1]) else 0.0
            first_last_color = 1.0 if get_color(hand[0]) == get_color(hand[-1]) else 0.0
        else:
            first_last_suit = 0.0
            first_last_color = 0.0
        features.extend([first_last_suit, first_last_color])

    # Pad to 104 if we have fewer than 8 examples
    while len(features) < 104:
        features.extend([0.0] * 13)

    return np.array(features[:104])


# ============================================================================
# TASK GENERATION
# ============================================================================

def generate_task(rule, num_examples: int = 8, hand_size: int = 6) -> Dict:
    """
    Generate a task for a given rule.

    Returns:
        Dict with 'rule_id', 'examples' (list of (hand, label)), 'primitives'
    """
    examples = []

    # Generate balanced examples (half positive, half negative)
    target_pos = num_examples // 2
    target_neg = num_examples - target_pos

    pos_count = 0
    neg_count = 0

    max_attempts = num_examples * 100  # Prevent infinite loop
    attempts = 0

    while (pos_count < target_pos or neg_count < target_neg) and attempts < max_attempts:
        hand = sample_hand(hand_size)
        label = rule.eval(hand)

        if label and pos_count < target_pos:
            examples.append((hand, True))
            pos_count += 1
        elif not label and neg_count < target_neg:
            examples.append((hand, False))
            neg_count += 1

        attempts += 1

    # If we couldn't generate enough, pad with what we have
    while len(examples) < num_examples:
        hand = sample_hand(hand_size)
        label = rule.eval(hand)
        examples.append((hand, label))

    return {
        'rule_id': rule.id,
        'rule_name': rule.name,
        'family': rule.family,
        'examples': examples[:num_examples],
        'primitives': rule.primitives_used
    }


# ============================================================================
# DEMONSTRATION
# ============================================================================

def main():
    """Run the complete demonstration pipeline."""

    print("=" * 80)
    print("DREAMCODER CARD GAME MODELING - DEMONSTRATION")
    print("=" * 80)
    print()

    # Step 1: Load rules
    print(f"Step 1: Loading Rules")
    print(f"  Total rules in catalogue: {len(ALL_RULES)}")
    print()

    # Organize by family
    families = {}
    for rule in ALL_RULES:
        families.setdefault(rule.family, []).append(rule)

    print(f"  Rules by family:")
    for family, rules in sorted(families.items()):
        print(f"    {family}: {len(rules)} rules")
    print()

    # Step 2: Select demo rules (one from each family)
    print(f"Step 2: Selecting Demo Rules")
    demo_rules = []
    for family, rules in sorted(families.items()):
        demo_rules.append(rules[0])  # Take first rule from each family

    print(f"  Selected {len(demo_rules)} rules (one per family):")
    for rule in demo_rules:
        print(f"    - {rule.id} ({rule.family})")
    print()

    # Step 3: Generate tasks
    print(f"Step 3: Generating Tasks")
    tasks = []
    for rule in demo_rules:
        task = generate_task(rule, num_examples=8, hand_size=6)
        tasks.append(task)
        print(f"  Generated task for {rule.id}")
    print(f"  Total tasks: {len(tasks)}")
    print()

    # Step 4: Extract features
    print(f"Step 4: Extracting Features")
    task_features = []
    for task in tasks:
        features = extract_task_features(task['examples'])
        task_features.append(features)
        print(f"  Extracted features for {task['rule_id']}: shape {features.shape}")

    task_features = np.array(task_features)
    print(f"  Feature matrix shape: {task_features.shape}")
    print()

    # Step 5: Analyze primitive usage
    print(f"Step 5: Analyzing Primitive Usage")
    all_primitives = set()
    for task in tasks:
        all_primitives.update(task['primitives'])

    primitive_list = sorted(all_primitives)
    print(f"  Total unique primitives: {len(primitive_list)}")
    print(f"  Primitives: {', '.join(primitive_list[:10])}...")
    print()

    # Create primitive usage matrix
    primitive_usage = np.zeros((len(tasks), len(primitive_list)))
    for i, task in enumerate(tasks):
        for prim in task['primitives']:
            if prim in primitive_list:
                j = primitive_list.index(prim)
                primitive_usage[i, j] = 1.0

    print(f"  Primitive usage matrix shape: {primitive_usage.shape}")
    print()

    # Step 6: Visualize
    print(f"Step 6: Creating Visualizations")

    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Plot 1: Primitive usage heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        primitive_usage.T,
        xticklabels=[t['rule_id'][:15] for t in tasks],
        yticklabels=primitive_list,
        cmap="YlGnBu",
        cbar_kws={'label': 'Primitive Used'},
        linewidths=0.5
    )
    plt.title("Primitive Usage Across Rules", fontsize=14, fontweight='bold')
    plt.xlabel("Rules", fontsize=12)
    plt.ylabel("Primitives", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "primitive_usage_heatmap.png", dpi=150)
    print(f"  ✓ Saved: primitive_usage_heatmap.png")
    plt.close()

    # Plot 2: Feature statistics
    plt.figure(figsize=(12, 6))
    feature_means = task_features.mean(axis=0)
    feature_stds = task_features.std(axis=0)

    plt.subplot(1, 2, 1)
    plt.plot(feature_means, alpha=0.7, linewidth=2)
    plt.title("Feature Means Across Tasks", fontsize=12, fontweight='bold')
    plt.xlabel("Feature Index", fontsize=10)
    plt.ylabel("Mean Value", fontsize=10)
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(feature_stds, alpha=0.7, linewidth=2, color='orange')
    plt.title("Feature Standard Deviations", fontsize=12, fontweight='bold')
    plt.xlabel("Feature Index", fontsize=10)
    plt.ylabel("Std Dev", fontsize=10)
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "feature_statistics.png", dpi=150)
    print(f"  ✓ Saved: feature_statistics.png")
    plt.close()

    # Plot 3: Primitive co-occurrence
    primitive_cooccur = primitive_usage.T @ primitive_usage

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        primitive_cooccur,
        xticklabels=primitive_list,
        yticklabels=primitive_list,
        cmap="coolwarm",
        center=0,
        linewidths=0.3,
        cbar_kws={'label': 'Co-occurrence Count'}
    )
    plt.title("Primitive Co-occurrence Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "primitive_cooccurrence.png", dpi=150)
    print(f"  ✓ Saved: primitive_cooccurrence.png")
    plt.close()

    print()

    # Step 7: Generate report
    print(f"Step 7: Generating Report")

    report = {
        'timestamp': datetime.now().isoformat(),
        'num_rules': len(ALL_RULES),
        'num_demo_rules': len(demo_rules),
        'num_tasks': len(tasks),
        'num_primitives': len(primitive_list),
        'feature_dim': task_features.shape[1],
        'rules': [
            {
                'id': task['rule_id'],
                'name': task['rule_name'],
                'family': task['family'],
                'num_primitives': len(task['primitives']),
                'primitives': task['primitives']
            }
            for task in tasks
        ]
    }

    report_path = output_dir / "demo_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    print(f"  ✓ Saved: demo_report.json")
    print()

    # Step 8: Summary
    print("=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Summary:")
    print(f"  - Loaded {len(ALL_RULES)} rules from catalogue")
    print(f"  - Generated {len(tasks)} demo tasks")
    print(f"  - Extracted {task_features.shape[1]}-dimensional features")
    print(f"  - Identified {len(primitive_list)} unique primitives")
    print(f"  - Created 3 visualization plots")
    print()
    print(f"Output directory: {output_dir.absolute()}")
    print()
    print("Next steps:")
    print("  1. Implement full recognition network (see src/dreamcoder/recognition.py)")
    print("  2. Implement program enumeration (see src/dreamcoder/enumeration.py)")
    print("  3. Implement library learning (see src/dreamcoder/compression.py)")
    print("  4. Run wake-sleep training loop")
    print()
    print("For integration with Ellis et al.'s DreamCoder, see:")
    print("  docs/DREAMCODER_INTEGRATION.md")
    print()


if __name__ == "__main__":
    main()
