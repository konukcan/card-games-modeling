#!/usr/bin/env python3
"""
Generate Comprehensive DreamCoder Report

This script generates a full HTML report with:
1. All rules with compositional decompositions
2. Discovered abstractions (patterns across rules)
3. Model performance metrics per rule
4. Embedded visualizations with legends and captions
5. Didactic explanations for each section

Usage:
    python src/generate_full_report.py
"""

import sys
import os
from pathlib import Path
import json
import base64
from datetime import datetime
from typing import Dict, List, Tuple
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from rules.cards import sample_hand, hand_to_string
from rules.catalogue import ALL_RULES, DISCOVERED_ABSTRACTIONS, get_all_families


def generate_tasks_and_evaluate(num_examples: int = 100) -> Dict:
    """
    Generate synthetic tasks and evaluate model performance on each rule.

    Returns metrics including:
    - Accuracy of rule evaluation
    - Base rate (how often rule is satisfied)
    - Difficulty estimate (based on compositional complexity)
    """
    results = {}

    for rule in ALL_RULES:
        # Generate random hands and evaluate
        true_count = 0
        total = num_examples

        for _ in range(num_examples):
            hand = sample_hand(6)
            try:
                if rule.eval(hand):
                    true_count += 1
            except:
                pass

        base_rate = true_count / total

        # Estimate difficulty based on:
        # 1. Compositional level (higher = harder)
        # 2. Number of primitives (more = harder)
        # 3. Base rate (extreme base rates = easier to guess)
        level_score = rule.level / 4.0
        primitive_score = min(len(rule.primitives_used) / 10.0, 1.0)
        base_rate_score = 1 - abs(base_rate - 0.5) * 2  # 0.5 is hardest

        difficulty = (level_score * 0.3 + primitive_score * 0.3 + base_rate_score * 0.4)

        # Simulated model accuracy (would come from actual recognition network)
        # Higher difficulty → lower accuracy, with some noise
        model_accuracy = max(0.5, min(0.99, 0.95 - difficulty * 0.4 + random.gauss(0, 0.05)))

        results[rule.id] = {
            'base_rate': base_rate,
            'difficulty': difficulty,
            'model_accuracy': model_accuracy,
            'level': rule.level,
            'num_primitives': len(rule.primitives_used),
            'family': rule.family
        }

    return results


def create_visualizations(results: Dict, output_dir: Path) -> Dict[str, str]:
    """
    Create all visualizations and return paths/base64 encodings.
    """
    images = {}

    # 1. Rules by Family - Bar Chart
    fig, ax = plt.subplots(figsize=(12, 6))
    families = {}
    for rule in ALL_RULES:
        families.setdefault(rule.family, []).append(rule)

    family_names = sorted(families.keys())
    counts = [len(families[f]) for f in family_names]
    colors = plt.cm.Set3(np.linspace(0, 1, len(family_names)))

    bars = ax.bar(family_names, counts, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Rule Family', fontsize=12)
    ax.set_ylabel('Number of Rules', fontsize=12)
    ax.set_title('Distribution of Rules by Family', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = output_dir / 'rules_by_family.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    images['rules_by_family'] = str(path)

    # 2. Compositional Level Distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    levels = [rule.level for rule in ALL_RULES]
    level_counts = [levels.count(i) for i in range(5)]

    colors = ['#e74c3c', '#f39c12', '#27ae60', '#3498db', '#9b59b6']
    bars = ax.bar(range(5), level_counts, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Compositional Level', fontsize=12)
    ax.set_ylabel('Number of Rules', fontsize=12)
    ax.set_title('Rules by Compositional Complexity Level', fontsize=14, fontweight='bold')
    ax.set_xticks(range(5))
    ax.set_xticklabels(['Level 0\n(Atomic)', 'Level 1\n(Combinators)',
                        'Level 2\n(Structural)', 'Level 3\n(Domain)', 'Level 4\n(Meta)'])

    for bar, count in zip(bars, level_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    path = output_dir / 'compositional_levels.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    images['compositional_levels'] = str(path)

    # 3. Model Performance Heatmap
    fig, ax = plt.subplots(figsize=(16, 10))

    # Organize by family for heatmap
    family_order = sorted(set(r['family'] for r in results.values()))
    rule_ids = []
    accuracies = []

    for family in family_order:
        family_rules = [r for r in ALL_RULES if r.family == family]
        for rule in family_rules:
            rule_ids.append(f"{rule.token}: {rule.id[:20]}")
            accuracies.append(results[rule.id]['model_accuracy'])

    # Create heatmap data
    n_rules = len(rule_ids)
    cols = 10
    rows = (n_rules + cols - 1) // cols

    # Pad to fill grid
    while len(accuracies) < rows * cols:
        accuracies.append(np.nan)
        rule_ids.append('')

    heatmap_data = np.array(accuracies).reshape(rows, cols)

    sns.heatmap(heatmap_data, annot=False, cmap='RdYlGn', vmin=0.5, vmax=1.0,
                cbar_kws={'label': 'Model Accuracy'}, ax=ax,
                linewidths=0.5, linecolor='white')

    ax.set_title('Model Accuracy by Rule (Simulated)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Rule Index (mod 10)', fontsize=12)
    ax.set_ylabel('Rule Group', fontsize=12)

    plt.tight_layout()
    path = output_dir / 'model_accuracy_heatmap.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    images['model_accuracy_heatmap'] = str(path)

    # 4. Difficulty vs Base Rate Scatter
    fig, ax = plt.subplots(figsize=(12, 8))

    for family in family_order:
        family_rules = [r for r in ALL_RULES if r.family == family]
        base_rates = [results[r.id]['base_rate'] for r in family_rules]
        difficulties = [results[r.id]['difficulty'] for r in family_rules]

        ax.scatter(base_rates, difficulties, label=family, s=80, alpha=0.7, edgecolors='black', linewidth=0.5)

    ax.set_xlabel('Base Rate (proportion satisfying rule)', fontsize=12)
    ax.set_ylabel('Estimated Difficulty', fontsize=12)
    ax.set_title('Rule Difficulty vs Base Rate by Family', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = output_dir / 'difficulty_vs_baserate.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    images['difficulty_vs_baserate'] = str(path)

    # 5. Abstraction Frequency
    fig, ax = plt.subplots(figsize=(12, 6))

    abstractions = list(DISCOVERED_ABSTRACTIONS.keys())
    frequencies = [DISCOVERED_ABSTRACTIONS[a]['frequency'] for a in abstractions]
    names = [DISCOVERED_ABSTRACTIONS[a]['name'] for a in abstractions]

    # Sort by frequency
    sorted_idx = np.argsort(frequencies)[::-1]
    names = [names[i] for i in sorted_idx]
    frequencies = [frequencies[i] for i in sorted_idx]

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(names)))
    bars = ax.barh(names, frequencies, color=colors, edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Number of Rules Using This Abstraction', fontsize=12)
    ax.set_title('Discovered Abstractions by Usage Frequency', fontsize=14, fontweight='bold')
    ax.invert_yaxis()

    for bar, freq in zip(bars, frequencies):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(freq), va='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = output_dir / 'abstraction_frequency.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    images['abstraction_frequency'] = str(path)

    # 6. Primitive Usage Network (simplified as heatmap)
    fig, ax = plt.subplots(figsize=(14, 10))

    # Get all primitives
    all_prims = set()
    for rule in ALL_RULES:
        all_prims.update(rule.primitives_used)
    prim_list = sorted(all_prims)

    # Create co-occurrence matrix
    co_occur = np.zeros((len(prim_list), len(prim_list)))
    for rule in ALL_RULES:
        prims = rule.primitives_used
        for i, p1 in enumerate(prim_list):
            if p1 in prims:
                for j, p2 in enumerate(prim_list):
                    if p2 in prims:
                        co_occur[i, j] += 1

    mask = np.triu(np.ones_like(co_occur, dtype=bool), k=1)
    sns.heatmap(co_occur, mask=mask, xticklabels=prim_list, yticklabels=prim_list,
                cmap='YlOrRd', ax=ax, linewidths=0.3,
                cbar_kws={'label': 'Co-occurrence Count'})

    ax.set_title('Primitive Co-occurrence Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)

    plt.tight_layout()
    path = output_dir / 'primitive_cooccurrence.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    images['primitive_cooccurrence'] = str(path)

    return images


def generate_html_report(results: Dict, images: Dict, output_path: Path):
    """Generate comprehensive HTML report with embedded images and explanations."""

    # Convert images to base64 for embedding
    def img_to_base64(path):
        with open(path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')

    images_b64 = {k: img_to_base64(v) for k, v in images.items()}

    # Group rules by family
    families = {}
    for rule in ALL_RULES:
        families.setdefault(rule.family, []).append(rule)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DreamCoder Card Game Modeling - Comprehensive Report</title>
    <style>
        :root {{
            --primary: #2c3e50;
            --secondary: #3498db;
            --accent: #e74c3c;
            --bg-light: #ecf0f1;
            --text: #2c3e50;
        }}

        * {{ box-sizing: border-box; }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: var(--text);
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f6fa;
        }}

        h1 {{
            color: var(--primary);
            border-bottom: 4px solid var(--secondary);
            padding-bottom: 15px;
            margin-bottom: 30px;
        }}

        h2 {{
            color: var(--primary);
            margin-top: 40px;
            padding: 10px 15px;
            background: linear-gradient(90deg, var(--secondary) 0%, transparent 100%);
            color: white;
            border-radius: 5px;
        }}

        h3 {{
            color: var(--secondary);
            margin-top: 25px;
        }}

        .info-box {{
            background: #fff;
            border-left: 4px solid var(--secondary);
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 0 8px 8px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}

        .info-box.warning {{
            border-left-color: #f39c12;
            background: #fef9e7;
        }}

        .info-box.success {{
            border-left-color: #27ae60;
            background: #eafaf1;
        }}

        .legend {{
            background: #fff;
            border: 1px solid #ddd;
            padding: 15px;
            margin: 15px 0;
            border-radius: 8px;
            font-size: 14px;
        }}

        .legend h4 {{
            margin: 0 0 10px 0;
            color: var(--primary);
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}

        .legend ul {{
            margin: 0;
            padding-left: 20px;
        }}

        .legend li {{
            margin: 5px 0;
        }}

        .figure-container {{
            background: white;
            padding: 20px;
            margin: 25px 0;
            border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }}

        .figure-container img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 0 auto;
            border-radius: 8px;
        }}

        .figure-caption {{
            text-align: center;
            font-style: italic;
            color: #666;
            margin-top: 15px;
            padding-top: 15px;
            border-top: 1px solid #eee;
        }}

        .figure-caption strong {{
            display: block;
            font-style: normal;
            color: var(--primary);
            margin-bottom: 5px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
            border-radius: 8px;
            overflow: hidden;
        }}

        th {{
            background: var(--primary);
            color: white;
            padding: 15px 12px;
            text-align: left;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        td {{
            padding: 12px;
            border-bottom: 1px solid #eee;
            font-size: 14px;
        }}

        tr:hover {{
            background: #f8f9fa;
        }}

        .rule-id {{
            font-family: 'Consolas', 'Monaco', monospace;
            font-weight: bold;
            color: var(--secondary);
        }}

        .composition {{
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 12px;
            background: #f8f9fa;
            padding: 8px;
            border-radius: 4px;
            white-space: pre-wrap;
            word-break: break-all;
        }}

        .lambda {{
            color: #9b59b6;
            font-weight: bold;
        }}

        .primitive-tag {{
            display: inline-block;
            background: #e8f4f8;
            color: #2980b9;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            margin: 2px;
            font-family: monospace;
        }}

        .family-badge {{
            display: inline-block;
            background: var(--secondary);
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }}

        .level-badge {{
            display: inline-block;
            padding: 3px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: bold;
        }}

        .level-0 {{ background: #e74c3c; color: white; }}
        .level-1 {{ background: #f39c12; color: white; }}
        .level-2 {{ background: #27ae60; color: white; }}
        .level-3 {{ background: #3498db; color: white; }}
        .level-4 {{ background: #9b59b6; color: white; }}

        .accuracy-bar {{
            height: 20px;
            border-radius: 4px;
            background: linear-gradient(90deg, #27ae60 var(--acc), #eee var(--acc));
        }}

        .metric {{
            text-align: center;
            font-weight: bold;
        }}

        .metric.good {{ color: #27ae60; }}
        .metric.medium {{ color: #f39c12; }}
        .metric.poor {{ color: #e74c3c; }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 25px 0;
        }}

        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        }}

        .stat-card .number {{
            font-size: 48px;
            font-weight: bold;
            color: var(--secondary);
        }}

        .stat-card .label {{
            color: #666;
            font-size: 14px;
            margin-top: 8px;
        }}

        .abstraction-card {{
            background: white;
            padding: 20px;
            margin: 15px 0;
            border-radius: 10px;
            border-left: 5px solid var(--secondary);
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}

        .abstraction-card h4 {{
            margin: 0 0 10px 0;
            color: var(--primary);
        }}

        .abstraction-card .composition {{
            margin: 10px 0;
        }}

        .toc {{
            background: white;
            padding: 20px 30px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }}

        .toc h3 {{
            margin-top: 0;
        }}

        .toc ul {{
            columns: 2;
            column-gap: 40px;
        }}

        .toc a {{
            color: var(--secondary);
            text-decoration: none;
        }}

        .toc a:hover {{
            text-decoration: underline;
        }}

        footer {{
            text-align: center;
            padding: 30px;
            margin-top: 50px;
            color: #666;
            border-top: 1px solid #ddd;
        }}

        @media (max-width: 768px) {{
            .toc ul {{ columns: 1; }}
            .stats-grid {{ grid-template-columns: 1fr 1fr; }}
        }}
    </style>
</head>
<body>
    <h1>🃏 DreamCoder Card Game Modeling - Comprehensive Report</h1>

    <p><strong>Generated:</strong> {datetime.now().strftime("%B %d, %Y at %H:%M")}</p>

    <div class="toc">
        <h3>📋 Table of Contents</h3>
        <ul>
            <li><a href="#overview">Overview & Summary Statistics</a></li>
            <li><a href="#visualizations">Visualizations & Analysis</a></li>
            <li><a href="#rules">Complete Rule Catalogue</a></li>
            <li><a href="#abstractions">Discovered Abstractions</a></li>
            <li><a href="#performance">Model Performance</a></li>
            <li><a href="#glossary">Glossary & Interpretation Guide</a></li>
        </ul>
    </div>

    <!-- SECTION: Overview -->
    <h2 id="overview">📊 Overview & Summary Statistics</h2>

    <div class="info-box">
        <strong>What is this report?</strong><br>
        This report documents the complete compositional analysis of {len(ALL_RULES)} card game rules.
        Each rule is decomposed into primitive functions following the DreamCoder framework
        (Ellis et al., 2023). The goal is to understand shared structure across rules that may
        explain transfer learning in human participants.
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="number">{len(ALL_RULES)}</div>
            <div class="label">Total Rules</div>
        </div>
        <div class="stat-card">
            <div class="number">{len(families)}</div>
            <div class="label">Rule Families</div>
        </div>
        <div class="stat-card">
            <div class="number">{len(DISCOVERED_ABSTRACTIONS)}</div>
            <div class="label">Discovered Abstractions</div>
        </div>
        <div class="stat-card">
            <div class="number">{len(set(p for r in ALL_RULES for p in r.primitives_used))}</div>
            <div class="label">Unique Primitives</div>
        </div>
    </div>

    <h3>Rules by Family</h3>
    <table>
        <tr>
            <th>Family</th>
            <th>Count</th>
            <th>Description</th>
            <th>Example Rule</th>
        </tr>
"""

    family_descriptions = {
        'LOCAL': 'Simple positional or ordering constraints',
        'COUNT': 'Counting and set cardinality rules',
        'POSITION': 'Specific position constraints',
        'TOKEN': 'Specific card presence',
        'AP': 'Arithmetic progression detection',
        'SCORE': 'Scoring formulas',
        'HIER': 'Hierarchical (halves share boolean property)',
        'LANG': 'Language/grammar rules (bracket matching)',
        'PAL': 'Palindrome patterns',
        'ALTCLR': 'Alternative color groupings',
        'COPY': 'Halves copy sequence patterns',
        'SHIFT': 'Positional rank differences',
        'MAP': 'Suit cycle mapping rules',
        'ADJ': 'Adjacent/skip constraints',
        'PARITY': 'Odd/even rank rules',
        'CENTER': 'Distance from center rules'
    }

    for family in sorted(families.keys()):
        rules_in_family = families[family]
        example = rules_in_family[0]
        desc = family_descriptions.get(family, 'Various rules')
        html += f"""        <tr>
            <td><span class="family-badge">{family}</span></td>
            <td style="text-align:center;font-weight:bold;">{len(rules_in_family)}</td>
            <td>{desc}</td>
            <td><span class="rule-id">{example.id}</span></td>
        </tr>
"""

    html += """    </table>

    <!-- SECTION: Visualizations -->
    <h2 id="visualizations">📈 Visualizations & Analysis</h2>

    <div class="info-box">
        <strong>How to read these visualizations:</strong><br>
        Each figure below reveals different aspects of the compositional structure of our rule set.
        Hover over elements or refer to the legends for detailed explanations.
    </div>
"""

    # Add each visualization with legend and caption
    viz_info = [
        ('rules_by_family', 'Figure 1: Distribution of Rules by Family',
         'This bar chart shows how many rules belong to each family. Families are conceptual groupings based on the type of constraint (e.g., counting, position, palindrome). Larger families indicate common computational patterns in our rule set.',
         ['Height of bar = number of rules in that family', 'Colors are for visual distinction only', 'Families with more rules may have more transfer potential']),

        ('compositional_levels', 'Figure 2: Rules by Compositional Complexity Level',
         'Rules are classified by their maximum compositional level (0-4). Higher levels indicate more complex compositions. Level 0 uses only atomic primitives; Level 4 uses meta-combinators like halves_equal or seq_palindrome.',
         ['Level 0 (Atomic): Simple property checks (e.g., get_suit, at)', 'Level 1 (Combinators): List operations (e.g., map, filter, count)', 'Level 2 (Structural): Hand decomposition (e.g., halves, shifted_pairs)', 'Level 3 (Domain): Specialized algorithms (e.g., hasAP, bracket_match)', 'Level 4 (Meta): Higher-order patterns (e.g., halves_equal, seq_palindrome)']),

        ('model_accuracy_heatmap', 'Figure 3: Model Accuracy Heatmap (Simulated)',
         'This heatmap shows the predicted accuracy of the recognition network on each rule. Green = high accuracy, Red = low accuracy. Currently simulated based on difficulty estimates; will be replaced with actual model predictions.',
         ['Color scale: 0.5 (red/chance) to 1.0 (green/perfect)', 'Each cell represents one rule', 'Harder rules (higher level, more primitives) tend to have lower accuracy']),

        ('difficulty_vs_baserate', 'Figure 4: Rule Difficulty vs Base Rate',
         'Scatter plot showing estimated difficulty against base rate (how often random hands satisfy the rule). Rules with extreme base rates (near 0 or 1) are easier because guessing works well. Rules near 0.5 base rate are hardest.',
         ['X-axis: Base rate (0=never satisfied, 1=always satisfied)', 'Y-axis: Estimated difficulty (0=easy, 1=hard)', 'Each point is one rule, colored by family', 'Dashed lines mark 0.5 thresholds']),

        ('abstraction_frequency', 'Figure 5: Discovered Abstractions by Frequency',
         'Bar chart showing how many rules use each discovered abstraction. These are compositional patterns that appear in multiple rules - they represent reusable "chunks" that may transfer between related rules.',
         ['Bar length = number of rules using this abstraction', 'Higher frequency = more potential for transfer learning', 'These abstractions are candidates for DreamCoder\'s library learning']),

        ('primitive_cooccurrence', 'Figure 6: Primitive Co-occurrence Matrix',
         'Heatmap showing how often pairs of primitives appear together in rules. Bright colors indicate frequent co-occurrence. This reveals which primitives form natural "bundles" that might be abstracted.',
         ['Diagonal = frequency of each primitive', 'Off-diagonal = co-occurrence count', 'High values suggest primitives that "go together"', 'Lower triangle shown (matrix is symmetric)'])
    ]

    for img_key, title, description, legend_items in viz_info:
        if img_key in images_b64:
            html += f"""
    <div class="figure-container">
        <img src="data:image/png;base64,{images_b64[img_key]}" alt="{title}">
        <div class="figure-caption">
            <strong>{title}</strong>
            {description}
        </div>
    </div>

    <div class="legend">
        <h4>📖 Legend for {title}</h4>
        <ul>
"""
            for item in legend_items:
                html += f"            <li>{item}</li>\n"
            html += """        </ul>
    </div>
"""

    # SECTION: Complete Rule Catalogue
    html += """
    <!-- SECTION: Complete Rule Catalogue -->
    <h2 id="rules">📚 Complete Rule Catalogue</h2>

    <div class="info-box">
        <strong>How to read this catalogue:</strong><br>
        Each rule shows its compositional decomposition using our primitive vocabulary.
        The <strong>Composition</strong> column shows the function composition tree.
        The <strong>Lambda</strong> notation shows the rule as a lambda expression.
        <strong>Primitives</strong> are the atomic building blocks used.
    </div>

    <div class="info-box warning">
        <strong>⚠️ Note on Model Accuracy:</strong><br>
        The accuracy values shown are currently <em>simulated</em> based on difficulty estimates.
        When the full recognition network is trained, these will be replaced with actual predictions.
    </div>
"""

    for family in sorted(families.keys()):
        family_rules = families[family]
        html += f"""
    <h3><span class="family-badge">{family}</span> {family_descriptions.get(family, '')}</h3>
    <table>
        <tr>
            <th style="width:8%">Token</th>
            <th style="width:15%">Rule ID</th>
            <th style="width:20%">Name</th>
            <th style="width:5%">Level</th>
            <th style="width:30%">Composition</th>
            <th style="width:12%">Primitives</th>
            <th style="width:10%">Accuracy</th>
        </tr>
"""
        for rule in family_rules:
            acc = results[rule.id]['model_accuracy']
            acc_class = 'good' if acc > 0.8 else ('medium' if acc > 0.65 else 'poor')
            acc_pct = acc * 100

            prims_html = ''.join(f'<span class="primitive-tag">{p}</span>' for p in rule.primitives_used[:5])
            if len(rule.primitives_used) > 5:
                prims_html += f'<span class="primitive-tag">+{len(rule.primitives_used)-5} more</span>'

            html += f"""        <tr>
            <td><code>{rule.token}</code></td>
            <td><span class="rule-id">{rule.id}</span></td>
            <td>{rule.name}</td>
            <td><span class="level-badge level-{rule.level}">L{rule.level}</span></td>
            <td><div class="composition">{rule.composition_str()}</div></td>
            <td>{prims_html}</td>
            <td>
                <div class="accuracy-bar" style="--acc: {acc_pct}%"></div>
                <div class="metric {acc_class}">{acc_pct:.1f}%</div>
            </td>
        </tr>
"""
        html += "    </table>\n"

    # SECTION: Discovered Abstractions
    html += """
    <!-- SECTION: Discovered Abstractions -->
    <h2 id="abstractions">🔍 Discovered Abstractions</h2>

    <div class="info-box success">
        <strong>What are abstractions?</strong><br>
        Abstractions are compositional patterns that appear in multiple rules. In DreamCoder,
        these would be "learned" by the library learning component and added to the primitive
        vocabulary. Rules sharing abstractions are predicted to show transfer learning.
    </div>
"""

    for key, abstraction in sorted(DISCOVERED_ABSTRACTIONS.items(), key=lambda x: -x[1]['frequency']):
        html += f"""
    <div class="abstraction-card">
        <h4>🧩 {abstraction['name']}</h4>
        <p><strong>Description:</strong> {abstraction['description']}</p>
        <div class="composition"><span class="lambda">{abstraction['composition']}</span></div>
        <p><strong>Level:</strong> <span class="level-badge level-{abstraction['level']}">Level {abstraction['level']}</span>
           <strong style="margin-left: 20px;">Frequency:</strong> {abstraction['frequency']} rules</p>
        <p><strong>Used by:</strong> {', '.join(abstraction['used_by'][:6])}{'...' if len(abstraction['used_by']) > 6 else ''}</p>
    </div>
"""

    # SECTION: Model Performance Summary
    html += """
    <!-- SECTION: Model Performance -->
    <h2 id="performance">🎯 Model Performance Summary</h2>

    <div class="info-box">
        <strong>Performance metrics explained:</strong><br>
        <ul>
            <li><strong>Base Rate:</strong> How often random hands satisfy the rule (0-1)</li>
            <li><strong>Difficulty:</strong> Estimated learning difficulty (0=easy, 1=hard)</li>
            <li><strong>Model Accuracy:</strong> Recognition network's predicted accuracy (simulated)</li>
        </ul>
    </div>

    <table>
        <tr>
            <th>Rule ID</th>
            <th>Family</th>
            <th>Level</th>
            <th>Base Rate</th>
            <th>Difficulty</th>
            <th>Model Accuracy</th>
        </tr>
"""

    # Sort by difficulty
    sorted_rules = sorted(ALL_RULES, key=lambda r: results[r.id]['difficulty'], reverse=True)

    for rule in sorted_rules[:20]:  # Top 20 hardest
        r = results[rule.id]
        acc_class = 'good' if r['model_accuracy'] > 0.8 else ('medium' if r['model_accuracy'] > 0.65 else 'poor')

        html += f"""        <tr>
            <td><span class="rule-id">{rule.id}</span></td>
            <td><span class="family-badge">{rule.family}</span></td>
            <td><span class="level-badge level-{rule.level}">L{rule.level}</span></td>
            <td>{r['base_rate']:.2%}</td>
            <td>{r['difficulty']:.2f}</td>
            <td class="metric {acc_class}">{r['model_accuracy']:.1%}</td>
        </tr>
"""

    html += """        <tr><td colspan="6" style="text-align:center;font-style:italic;">... showing top 20 hardest rules ...</td></tr>
    </table>
"""

    # SECTION: Glossary
    html += """
    <!-- SECTION: Glossary -->
    <h2 id="glossary">📖 Glossary & Interpretation Guide</h2>

    <div class="info-box">
        <h4>Key Terms</h4>
        <dl>
            <dt><strong>Primitive</strong></dt>
            <dd>A basic function that cannot be decomposed further (e.g., <code>get_suit</code>, <code>first</code>)</dd>

            <dt><strong>Combinator</strong></dt>
            <dd>A function that combines other functions (e.g., <code>map</code>, <code>filter</code>, <code>all</code>)</dd>

            <dt><strong>Composition</strong></dt>
            <dd>The tree structure showing how a rule is built from primitives</dd>

            <dt><strong>Abstraction</strong></dt>
            <dd>A reusable pattern discovered across multiple rules</dd>

            <dt><strong>Level</strong></dt>
            <dd>The compositional complexity (0=atomic, 4=meta-combinator)</dd>

            <dt><strong>Base Rate</strong></dt>
            <dd>The probability that a random hand satisfies the rule</dd>

            <dt><strong>Recognition Network</strong></dt>
            <dd>Neural network that predicts which primitives are needed for a task</dd>
        </dl>
    </div>

    <div class="info-box">
        <h4>How to Interpret Results</h4>
        <ol>
            <li><strong>Transfer Prediction:</strong> Rules sharing abstractions (see Section 4) should show transfer. For example, learning <code>Halves_copy_suits</code> should help with <code>Halves_copy_colors</code> because both use <code>halves_equal</code>.</li>
            <li><strong>Difficulty Prediction:</strong> Higher-level rules with more primitives should be harder to learn. Check the performance table (Section 5) to see if this holds.</li>
            <li><strong>Curriculum Design:</strong> Start with Level 0-1 rules, then progress to higher levels. Within a family, order by base rate (extreme base rates first).</li>
        </ol>
    </div>

    <footer>
        <p>Generated by DreamCoder Card Game Modeling System</p>
        <p>Based on Ellis et al. (2023) - DreamCoder: Growing Generalizable, Interpretable Knowledge with Wake-Sleep Bayesian Program Learning</p>
        <p><a href="https://github.com/konukcan/card-games-modeling">GitHub Repository</a></p>
    </footer>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


def main():
    """Generate the full comprehensive report."""
    print("=" * 70)
    print("GENERATING COMPREHENSIVE DREAMCODER REPORT")
    print("=" * 70)
    print()

    # Setup output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Step 1: Generate tasks and evaluate
    print("Step 1: Generating synthetic tasks and computing metrics...")
    results = generate_tasks_and_evaluate(num_examples=200)
    print(f"  ✓ Computed metrics for {len(results)} rules")
    print()

    # Step 2: Create visualizations
    print("Step 2: Creating visualizations...")
    images = create_visualizations(results, output_dir)
    print(f"  ✓ Generated {len(images)} figures")
    for name, path in images.items():
        print(f"    - {name}: {path}")
    print()

    # Step 3: Generate HTML report
    print("Step 3: Generating HTML report...")
    report_path = output_dir / "comprehensive_report.html"
    generate_html_report(results, images, report_path)
    print(f"  ✓ Saved: {report_path}")
    print()

    # Step 4: Save JSON data
    print("Step 4: Saving JSON data...")
    json_data = {
        'timestamp': datetime.now().isoformat(),
        'num_rules': len(ALL_RULES),
        'num_families': len(set(r.family for r in ALL_RULES)),
        'num_abstractions': len(DISCOVERED_ABSTRACTIONS),
        'rules': [
            {
                'id': rule.id,
                'token': rule.token,
                'name': rule.name,
                'family': rule.family,
                'level': rule.level,
                'composition': rule.composition_str(),
                'lambda': rule.lambda_str(),
                'primitives': rule.primitives_used,
                'description': rule.description,
                'metrics': results[rule.id]
            }
            for rule in ALL_RULES
        ],
        'abstractions': DISCOVERED_ABSTRACTIONS
    }

    json_path = output_dir / "full_report_data.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"  ✓ Saved: {json_path}")
    print()

    # Summary
    print("=" * 70)
    print("REPORT GENERATION COMPLETE")
    print("=" * 70)
    print()
    print(f"  Total rules: {len(ALL_RULES)}")
    print(f"  Rule families: {len(set(r.family for r in ALL_RULES))}")
    print(f"  Discovered abstractions: {len(DISCOVERED_ABSTRACTIONS)}")
    print(f"  Unique primitives: {len(set(p for r in ALL_RULES for p in r.primitives_used))}")
    print()
    print(f"  Output files:")
    print(f"    - HTML Report: {report_path}")
    print(f"    - JSON Data: {json_path}")
    for name, path in images.items():
        print(f"    - {name}: {path}")
    print()
    print(f"  Open the report: open {report_path}")
    print()


if __name__ == "__main__":
    main()
