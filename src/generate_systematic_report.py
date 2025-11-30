#!/usr/bin/env python3
"""
Systematic Report Generator for DreamCoder Overnight Runs

This script generates comprehensive HTML reports from overnight run data, including:
1. Executive Summary - solve rates, grammar growth, training progress
2. Solve Timeline - when each task was solved
3. Grammar Evolution - primitive additions over iterations
4. Recognition Model Analysis - 4 snapshots showing model's learned representations
5. Abstraction Dependency Trees - how knowledge flows between tasks
6. Solution Gallery - programs discovered for each task (with paraphrases)

Usage:
    python generate_systematic_report.py --run-dir results/overnight_v3/run_v3_YYYYMMDD_HHMMSS

    # Merge original and resume runs:
    python generate_systematic_report.py --run-dir results/overnight_v3/run_v3_YYYYMMDD_HHMMSS \\
        --resume-dir results/overnight_v3/resume_v3_YYYYMMDD_HHMMSS
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import re

import torch
import numpy as np

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class IterationData:
    """Data from a single iteration checkpoint."""
    iteration: int
    timestamp: str
    tasks_solved: int
    tasks_total: int
    programs_enumerated: int
    recognition_loss: float
    grammar_size: int
    new_abstractions: List[str]
    model_path: str
    task_embeddings: Dict[str, List[float]]
    primitive_predictions: Dict[str, Dict]  # May be empty in legacy runs


@dataclass
class FrontierData:
    """Data about solved tasks and their programs."""
    task_name: str
    solved: bool
    best_program: Optional[str]
    n_entries: int


@dataclass
class AbstractionInfo:
    """Information about a learned abstraction."""
    name: str
    iteration_invented: int
    source_tasks: List[str]  # Tasks whose solutions contained this pattern
    used_in_tasks: List[str]  # Tasks that used this abstraction in solution


@dataclass
class RunData:
    """Complete data from an overnight run."""
    run_dir: Path
    iterations: List[IterationData]
    frontiers: Dict[str, FrontierData]
    grammar_primitives: List[str]
    base_primitives: List[str]
    learned_abstractions: List[AbstractionInfo]
    run_config: Dict


# ============================================================================
# PROGRAM PARAPHRASING
# ============================================================================

def paraphrase_program(program: str) -> str:
    """
    Convert a lambda expression to human-readable English.

    Examples:
        (λ all_same_suit $0) → "All cards have the same suit"
        (λ eq 14 (max_rank $0)) → "The highest rank is 14 (Ace)"
    """
    if not program:
        return ""

    # Common patterns and their paraphrases
    paraphrases = {
        # Suit patterns
        r'\(λ all_same_suit \$0\)': "All cards have the same suit (flush)",
        r'\(λ all_same_color \$0\)': "All cards have the same color",
        r'\(λ not \(all_same_suit \$0\)\)': "Cards have at least two different suits",
        r'\(λ has_suit \$0 SPADES\)': "Hand contains at least one spade",
        r'\(λ has_suit \$0 HEARTS\)': "Hand contains at least one heart",
        r'\(λ has_suit \$0 DIAMONDS\)': "Hand contains at least one diamond",
        r'\(λ has_suit \$0 CLUBS\)': "Hand contains at least one club",

        # Rank patterns
        r'\(λ eq 14 \(max_rank \$0\)\)': "The highest card is an Ace",
        r'\(λ lt 10 \(max_rank \$0\)\)': "The highest card is a face card (J/Q/K/A)",
        r'\(λ lt 10 \(min_rank \$0\)\)': "All cards are face cards",
        r'\(λ le \(n_unique_ranks \$0\) 5\)': "At most 5 unique ranks (has at least one pair)",
        r'\(λ lt 5 \(n_unique_ranks \$0\)\)': "All cards have different ranks",
        r'\(λ eq 3 \(n_unique_ranks \$0\)\)': "Exactly 3 different ranks",
        r'\(λ lt 3 \(n_unique_ranks \$0\)\)': "More than 3 different ranks",

        # Sum patterns
        r'\(λ lt \(sum_ranks \$0\) 21\)': "Sum of ranks is less than 21",
        r'\(λ lt \(sum_ranks \$0\) \(\+ 2 21\)\)': "Sum of ranks is less than 23",
        r'\(λ eq 0 \(mod \(sum_ranks \$0\) 2\)\)': "Sum of ranks is even",

        # Color patterns
        r'\(λ eq RED \(get_color \(head \$0\)\)\)': "First card is red",
        r'\(λ eq BLACK \(get_color \(last \$0\)\)\)': "Last card is black",
        r'\(λ eq \(get_color \(head \$0\)\) \(get_color \(last \$0\)\)\)': "First and last cards have the same color",

        # Count patterns
        r'\(λ eq 2 \(count_color \$0 RED\)\)': "Exactly 2 red cards",
        r'\(λ lt 3 \(count_color \$0 RED\)\)': "More than 3 red cards",
        r'\(λ neq 3 \(count_color \$0 RED\)\)': "Not exactly 3 red cards (majority color)",
        r'\(λ eq 2 \(n_unique_suits \$0\)\)': "Exactly 2 different suits",

        # Position patterns
        r'\(λ lt 10 \(rank_val \(at \$0 3\)\)\)': "The 4th card is a face card",
    }

    # Try exact matches first
    for pattern, paraphrase in paraphrases.items():
        if re.match(pattern, program):
            return paraphrase

    # Fall back to structured parsing
    return parse_program_structure(program)


def parse_program_structure(program: str) -> str:
    """Parse program structure for less common patterns."""

    # Extract key components
    if 'all_same_suit' in program:
        return "All cards share the same suit"
    if 'all_same_color' in program:
        return "All cards share the same color"
    if 'n_unique_ranks' in program:
        if 'le' in program or 'lt' in program:
            return "Constraint on number of unique ranks"
        if 'eq' in program:
            return "Specific number of unique ranks required"
    if 'sum_ranks' in program:
        return "Constraint on sum of card ranks"
    if 'count_color' in program:
        return "Constraint on count of cards by color"
    if 'max_rank' in program:
        return "Constraint on highest card rank"
    if 'min_rank' in program:
        return "Constraint on lowest card rank"
    if 'get_color' in program and 'head' in program:
        return "Constraint on first card's color"
    if 'get_color' in program and 'last' in program:
        return "Constraint on last card's color"
    if 'has_suit' in program:
        return "Requires specific suit to be present"

    return "Complex rule (hover for formula)"


def paraphrase_abstraction(abstraction: str) -> str:
    """Convert a learned abstraction to human-readable description."""

    # Remove the #(( prefix and )) suffix
    core = abstraction.strip('#()')

    if 'n_unique_ranks' in core:
        return "Count unique ranks in hand"
    if 'max_rank' in core:
        return "Get highest rank"
    if 'min_rank' in core:
        return "Get lowest rank"
    if 'count_color' in core:
        if 'RED' in core:
            return "Count red cards"
        if 'BLACK' in core:
            return "Count black cards"
        return "Count cards by color"
    if 'sum_ranks' in core:
        return "Sum all ranks"
    if 'all_same_suit' in core:
        return "Check if all same suit"
    if 'all_same_color' in core:
        return "Check if all same color"
    if 'reverse' in core:
        return "Reverse card order"
    if 'unique' in core:
        return "Get unique elements"
    if 'get_color' in core and 'head' in core:
        return "Get first card's color"
    if 'get_color' in core and 'last' in core:
        return "Get last card's color"
    if 'n_unique_suits' in core:
        return "Count unique suits"
    if 'has_suit' in core:
        return "Check for suit presence"
    if 'mod' in core:
        return "Modulo operation"
    if 'lt' in core or 'le' in core:
        return "Less-than comparison"
    if 'eq' in core:
        return "Equality check"
    if 'neq' in core:
        return "Inequality check"

    return "Learned pattern"


# ============================================================================
# DATA LOADING
# ============================================================================

def load_run_data(run_dir: Path, resume_dir: Optional[Path] = None) -> RunData:
    """Load all data from an overnight run directory, optionally merging with resume."""

    # Load iteration checkpoints from main run
    iterations = []
    checkpoint_dir = run_dir / "iteration_checkpoints"

    if checkpoint_dir.exists():
        json_files = sorted(checkpoint_dir.glob("iteration_*.json"))
        for json_file in json_files:
            if "_model" not in json_file.name:
                with open(json_file) as f:
                    data = json.load(f)
                    iterations.append(IterationData(
                        iteration=data['iteration'],
                        timestamp=data['timestamp'],
                        tasks_solved=data['metrics']['tasks_solved'],
                        tasks_total=data['metrics']['tasks_total'],
                        programs_enumerated=data['metrics']['programs_enumerated'],
                        recognition_loss=data['metrics']['recognition_loss'],
                        grammar_size=data['metrics']['grammar_size'],
                        new_abstractions=data['metrics'].get('new_abstractions', []),
                        model_path=data.get('model_path', ''),
                        task_embeddings=data.get('task_embeddings', {}),
                        primitive_predictions=data.get('primitive_predictions', {})
                    ))

    # Merge resume run iterations if provided
    if resume_dir and resume_dir.exists():
        resume_checkpoint_dir = resume_dir / "iteration_checkpoints"
        if resume_checkpoint_dir.exists():
            max_iter = max(it.iteration for it in iterations) if iterations else 0
            resume_files = sorted(resume_checkpoint_dir.glob("iteration_*.json"))
            for json_file in resume_files:
                if "_model" not in json_file.name:
                    with open(json_file) as f:
                        data = json.load(f)
                        # Renumber iterations to continue from main run
                        new_iter = max_iter + data['iteration']
                        iterations.append(IterationData(
                            iteration=new_iter,
                            timestamp=data['timestamp'],
                            tasks_solved=data['metrics']['tasks_solved'],
                            tasks_total=data['metrics']['tasks_total'],
                            programs_enumerated=data['metrics']['programs_enumerated'],
                            recognition_loss=data['metrics']['recognition_loss'],
                            grammar_size=data['metrics']['grammar_size'],
                            new_abstractions=data['metrics'].get('new_abstractions', []),
                            model_path=data.get('model_path', ''),
                            task_embeddings=data.get('task_embeddings', {}),
                            primitive_predictions=data.get('primitive_predictions', {})
                        ))

    # Load frontiers - prefer resume if available
    frontiers = {}
    frontier_source = resume_dir if resume_dir and resume_dir.exists() else run_dir
    frontier_files = sorted(frontier_source.glob("frontiers_phase*.json"), reverse=True)
    if frontier_files:
        with open(frontier_files[0]) as f:
            frontier_data = json.load(f)
            for task_name, task_data in frontier_data.items():
                frontiers[task_name] = FrontierData(
                    task_name=task_name,
                    solved=task_data['solved'],
                    best_program=task_data.get('best_program'),
                    n_entries=task_data.get('n_entries', 0)
                )

    # Load grammar - prefer resume if available
    grammar_primitives = []
    base_primitives = []
    grammar_source = resume_dir if resume_dir and resume_dir.exists() else run_dir
    grammar_files = sorted(grammar_source.glob("grammar_phase*.json"), reverse=True)
    if grammar_files:
        with open(grammar_files[0]) as f:
            grammar_data = json.load(f)
            grammar_primitives = grammar_data.get('primitives', [])
            for p in grammar_primitives:
                if not p.startswith('#('):
                    base_primitives.append(p)

    # Build abstraction info
    learned_abstractions = infer_abstraction_dependencies(iterations, frontiers, grammar_primitives)

    # Load run config
    run_config = {}
    config_file = run_dir / "run_config.json"
    if config_file.exists():
        with open(config_file) as f:
            run_config = json.load(f)

    return RunData(
        run_dir=run_dir,
        iterations=iterations,
        frontiers=frontiers,
        grammar_primitives=grammar_primitives,
        base_primitives=base_primitives,
        learned_abstractions=learned_abstractions,
        run_config=run_config
    )


def infer_abstraction_dependencies(
    iterations: List[IterationData],
    frontiers: Dict[str, FrontierData],
    grammar_primitives: List[str]
) -> List[AbstractionInfo]:
    """
    Infer abstraction dependencies from iteration data and solved programs.
    """
    abstractions = []

    # Track when each abstraction was invented
    abstraction_invention_iter = {}
    for it in iterations:
        for abstr in it.new_abstractions:
            if abstr not in abstraction_invention_iter:
                abstraction_invention_iter[abstr] = it.iteration

    def extract_core_primitives(abstr: str) -> Set[str]:
        """Extract base primitive names from an abstraction."""
        core = abstr.strip('#()')
        words = re.findall(r'\b([a-z_]+)\b', core)
        return set(words)

    learned = [p for p in grammar_primitives if p.startswith('#(')]

    for abstr in learned:
        invention_iter = abstraction_invention_iter.get(abstr, 0)
        core_prims = extract_core_primitives(abstr)

        used_in = []
        for task_name, frontier in frontiers.items():
            if frontier.solved and frontier.best_program:
                program_prims = set(re.findall(r'\b([a-z_]+)\b', frontier.best_program))
                if core_prims & program_prims:
                    used_in.append(task_name)

        source_tasks = []
        for it in iterations:
            if it.iteration < invention_iter:
                for task_name, frontier in frontiers.items():
                    if frontier.solved and frontier.best_program:
                        program_prims = set(re.findall(r'\b([a-z_]+)\b', frontier.best_program))
                        if core_prims & program_prims:
                            if task_name not in source_tasks:
                                source_tasks.append(task_name)

        abstractions.append(AbstractionInfo(
            name=abstr,
            iteration_invented=invention_iter,
            source_tasks=source_tasks,
            used_in_tasks=used_in
        ))

    return abstractions


# ============================================================================
# RECOGNITION MODEL ANALYSIS
# ============================================================================

def select_snapshot_iterations(iterations: List[IterationData], n_snapshots: int = 4) -> List[int]:
    """Select evenly-spaced iteration indices for deep analysis."""
    total = len(iterations)
    if total <= n_snapshots:
        return list(range(total))

    indices = [0]
    step = (total - 1) / (n_snapshots - 1)
    for i in range(1, n_snapshots - 1):
        indices.append(int(i * step))
    indices.append(total - 1)

    return indices


def cluster_embeddings(embeddings: Dict[str, List[float]], n_clusters: int = 5) -> Dict[str, int]:
    """Cluster task embeddings using k-means."""
    if not embeddings:
        return {}

    task_names = list(embeddings.keys())
    X = np.array([embeddings[name] for name in task_names])

    try:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=min(n_clusters, len(task_names)), random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        return {name: int(label) for name, label in zip(task_names, labels)}
    except ImportError:
        return {name: 0 for name in task_names}


def reduce_dimensions(embeddings: Dict[str, List[float]], method: str = 'pca') -> Dict[str, Tuple[float, float]]:
    """Reduce embedding dimensions to 2D for visualization."""
    if not embeddings:
        return {}

    task_names = list(embeddings.keys())
    X = np.array([embeddings[name] for name in task_names])

    if method == 'pca':
        try:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(X)
            return {name: (float(coords[i, 0]), float(coords[i, 1]))
                    for i, name in enumerate(task_names)}
        except ImportError:
            pass

    return {name: (float(X[i, 0] if X.shape[1] > 0 else 0),
                   float(X[i, 1] if X.shape[1] > 1 else 0))
            for i, name in enumerate(task_names)}


# ============================================================================
# HTML GENERATION
# ============================================================================

def html_escape(text: str) -> str:
    """Escape HTML special characters."""
    return (str(text)
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;'))


def generate_tooltip_js() -> str:
    """Generate JavaScript for custom tooltips."""
    return """
    <script>
        // Custom tooltip implementation
        document.addEventListener('DOMContentLoaded', function() {
            // Create tooltip element
            const tooltip = document.createElement('div');
            tooltip.id = 'custom-tooltip';
            tooltip.style.cssText = `
                position: fixed;
                background: #1a1a2e;
                border: 1px solid #4ecca3;
                border-radius: 6px;
                padding: 8px 12px;
                color: #e8e8e8;
                font-size: 13px;
                max-width: 400px;
                z-index: 10000;
                pointer-events: none;
                display: none;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            `;
            document.body.appendChild(tooltip);

            // Add event listeners to all elements with data-tooltip
            document.querySelectorAll('[data-tooltip]').forEach(el => {
                el.addEventListener('mouseenter', function(e) {
                    tooltip.innerHTML = this.dataset.tooltip;
                    tooltip.style.display = 'block';
                });

                el.addEventListener('mousemove', function(e) {
                    tooltip.style.left = (e.clientX + 15) + 'px';
                    tooltip.style.top = (e.clientY + 15) + 'px';
                });

                el.addEventListener('mouseleave', function() {
                    tooltip.style.display = 'none';
                });
            });

            // Also handle SVG elements
            document.querySelectorAll('circle[data-tooltip], rect[data-tooltip]').forEach(el => {
                el.style.cursor = 'pointer';
                el.addEventListener('mouseenter', function(e) {
                    tooltip.innerHTML = this.dataset.tooltip;
                    tooltip.style.display = 'block';
                });

                el.addEventListener('mousemove', function(e) {
                    tooltip.style.left = (e.clientX + 15) + 'px';
                    tooltip.style.top = (e.clientY + 15) + 'px';
                });

                el.addEventListener('mouseleave', function() {
                    tooltip.style.display = 'none';
                });
            });
        });
    </script>
    """


def generate_html_report(run_data: RunData) -> str:
    """Generate the complete HTML report."""

    snapshot_indices = select_snapshot_iterations(run_data.iterations, n_snapshots=4)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DreamCoder Run Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #0f3460;
            --text-primary: #e8e8e8;
            --text-secondary: #a0a0a0;
            --accent: #e94560;
            --success: #4ecca3;
            --warning: #ffc107;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 20px;
        }}

        .container {{ max-width: 1400px; margin: 0 auto; }}

        h1 {{ color: var(--accent); margin-bottom: 10px; font-size: 2.5rem; }}
        h2 {{ color: var(--success); margin: 30px 0 15px; padding-bottom: 10px; border-bottom: 2px solid var(--bg-card); }}
        h3 {{ color: var(--text-primary); margin: 20px 0 10px; }}
        h4 {{ color: var(--text-secondary); margin: 15px 0 8px; }}

        .subtitle {{ color: var(--text-secondary); font-size: 1.1rem; margin-bottom: 30px; }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}

        .stat-card {{
            background: var(--bg-card);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }}

        .stat-value {{ font-size: 2.5rem; font-weight: bold; color: var(--accent); }}
        .stat-label {{ color: var(--text-secondary); font-size: 0.9rem; margin-top: 5px; }}

        .chart-container {{
            background: var(--bg-secondary);
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-secondary);
            border-radius: 12px;
            overflow: hidden;
            margin: 20px 0;
        }}

        th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid var(--bg-card); }}
        th {{ background: var(--bg-card); color: var(--success); font-weight: 600; }}
        tr:hover {{ background: var(--bg-card); }}

        .solved {{ color: var(--success); }}
        .unsolved {{ color: var(--accent); }}

        .program {{
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.85rem;
            background: var(--bg-primary);
            padding: 5px 10px;
            border-radius: 4px;
            word-break: break-all;
            cursor: help;
        }}

        .paraphrase {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            font-style: italic;
            margin-top: 5px;
        }}

        .snapshot-section {{
            background: var(--bg-secondary);
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
        }}

        .cluster-group {{
            background: var(--bg-card);
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
        }}

        .cluster-tasks {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }}

        .task-tag {{
            background: var(--bg-primary);
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 0.85rem;
            cursor: help;
        }}

        .task-tag.solved {{ border-left: 3px solid var(--success); }}
        .task-tag.unsolved {{ border-left: 3px solid var(--accent); }}

        .abstraction-tree {{
            background: var(--bg-secondary);
            padding: 20px;
            border-radius: 12px;
            margin: 20px 0;
        }}

        .abstraction-item {{
            background: var(--bg-card);
            padding: 15px;
            border-radius: 8px;
            margin: 10px 0;
            border-left: 4px solid var(--warning);
        }}

        .abstraction-name {{
            font-family: monospace;
            font-size: 0.9rem;
            color: var(--warning);
            cursor: help;
        }}

        .solution-card {{
            background: var(--bg-card);
            padding: 15px;
            border-radius: 12px;
            text-align: left;
        }}

        .solution-task {{ color: var(--success); font-weight: bold; margin-bottom: 8px; }}

        svg {{ max-width: 100%; height: auto; }}

        .legend {{ display: flex; gap: 20px; margin: 10px 0; flex-wrap: wrap; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; }}
        .legend-color {{ width: 20px; height: 20px; border-radius: 4px; }}

        .dag-container {{
            background: var(--bg-card);
            padding: 20px;
            border-radius: 12px;
            overflow-x: auto;
        }}

        /* Collapsible sections */
        .collapsible-section {{
            background: var(--bg-secondary);
            border-radius: 12px;
            margin: 20px 0;
            overflow: hidden;
        }}

        .collapsible-header {{
            background: var(--bg-card);
            padding: 15px 20px;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 10px;
            user-select: none;
        }}

        .collapsible-header:hover {{
            background: var(--bg-primary);
        }}

        .collapsible-icon {{
            font-size: 0.8rem;
            transition: transform 0.2s;
        }}

        .collapsible-section.collapsed .collapsible-icon {{
            transform: rotate(-90deg);
        }}

        .collapsible-content {{
            padding: 20px;
            max-height: 800px;
            overflow-y: auto;
        }}

        .collapsible-section.collapsed .collapsible-content {{
            display: none;
        }}

        /* Library refresher grid */
        .primitives-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
            gap: 15px;
        }}

        .primitive-category {{
            background: var(--bg-card);
            padding: 15px;
            border-radius: 8px;
        }}

        .category-title {{
            color: var(--warning);
            font-weight: bold;
            margin-bottom: 10px;
            font-size: 0.9rem;
        }}

        .primitive-item {{
            font-family: monospace;
            font-size: 0.8rem;
            padding: 3px 8px;
            margin: 3px 0;
            background: var(--bg-primary);
            border-radius: 4px;
            display: inline-block;
            margin-right: 5px;
        }}

        /* Composition tree */
        .composition-tree {{
            font-family: monospace;
        }}

        .tree-node {{
            margin: 5px 0;
            padding: 10px 15px;
            background: var(--bg-card);
            border-radius: 8px;
            border-left: 3px solid var(--text-secondary);
        }}

        .tree-node.base {{
            border-left-color: var(--success);
        }}

        .tree-node.abstraction {{
            border-left-color: var(--warning);
        }}

        .tree-node-header {{
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .tree-node.collapsed .tree-children {{
            display: none;
        }}

        .tree-node-icon {{
            font-size: 0.7rem;
            width: 20px;
            text-align: center;
        }}

        .tree-node-name {{
            font-weight: bold;
            color: var(--warning);
        }}

        .tree-node-meta {{
            font-size: 0.8rem;
            color: var(--text-secondary);
            margin-left: auto;
        }}

        .tree-children {{
            margin-left: 25px;
            margin-top: 10px;
            border-left: 1px dashed var(--bg-card);
            padding-left: 15px;
        }}

        .tree-details {{
            font-size: 0.85rem;
            margin-top: 8px;
            padding: 10px;
            background: var(--bg-primary);
            border-radius: 4px;
        }}

        .tree-details .paraphrase {{
            color: var(--text-secondary);
            font-style: italic;
        }}

        .tree-details .depends-on {{
            margin-top: 5px;
        }}

        .tree-details .dep {{
            display: inline-block;
            background: var(--bg-card);
            padding: 2px 6px;
            border-radius: 3px;
            margin: 2px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>DreamCoder Overnight Run Report</h1>
        <p class="subtitle">Run: {run_data.run_dir.name}</p>

        {generate_executive_summary_section(run_data)}

        {generate_library_refresher_section(run_data)}

        {generate_training_progress_section(run_data)}

        {generate_recognition_analysis_section(run_data, snapshot_indices)}

        {generate_composition_dag_section(run_data)}

        {generate_abstraction_tree_section(run_data)}

        {generate_solve_timeline_section(run_data)}

        {generate_solution_gallery_section(run_data)}
    </div>

    {generate_tooltip_js()}
</body>
</html>
"""
    return html


def generate_executive_summary_section(run_data: RunData) -> str:
    """Generate the executive summary section."""

    n_solved = sum(1 for f in run_data.frontiers.values() if f.solved)
    n_total = len(run_data.frontiers)
    solve_rate = (n_solved / n_total * 100) if n_total > 0 else 0

    n_iterations = len(run_data.iterations)
    final_grammar = run_data.iterations[-1].grammar_size if run_data.iterations else 0
    initial_grammar = run_data.iterations[0].grammar_size if run_data.iterations else 0

    final_loss = run_data.iterations[-1].recognition_loss if run_data.iterations else 0
    initial_loss = run_data.iterations[0].recognition_loss if run_data.iterations else 0

    n_abstractions = final_grammar - len(run_data.base_primitives)

    return f"""
        <h2>Executive Summary</h2>
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{n_solved}/{n_total}</div>
                <div class="stat-label">Tasks Solved ({solve_rate:.1f}%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{n_iterations}</div>
                <div class="stat-label">Iterations Completed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{initial_grammar} → {final_grammar}</div>
                <div class="stat-label">Grammar Size</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{n_abstractions}</div>
                <div class="stat-label">Learned Abstractions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{initial_loss:.2f} → {final_loss:.2f}</div>
                <div class="stat-label">Recognition Loss</div>
            </div>
        </div>
    """


def generate_library_refresher_section(run_data: RunData) -> str:
    """Generate the initial library refresher section (collapsible)."""

    # Categorize base primitives
    categories = {
        'Suit Constants': ['CLUBS', 'DIAMONDS', 'HEARTS', 'SPADES'],
        'Color Constants': ['RED', 'BLACK'],
        'Number Constants': ['0', '1', '2', '3', '4', '5', '10', '11', '12', '13', '14', '17', '21'],
        'Boolean Constants': ['true', 'false'],
        'Card Accessors': ['get_suit', 'get_rank', 'rank_val', 'get_color'],
        'Position Operations': ['head', 'last', 'at', 'length', 'reverse'],
        'Direct Queries': ['has_suit', 'has_color', 'count_suit', 'count_color',
                          'all_same_suit', 'all_same_color', 'n_unique_suits',
                          'n_unique_ranks', 'n_unique_colors'],
        'Aggregates': ['sum_ranks', 'max_rank', 'min_rank'],
        'Comparisons': ['eq', 'neq', 'lt', 'le', 'gt', 'ge'],
        'Boolean Operators': ['and', 'or', 'not', 'if'],
        'Higher-Order': ['map', 'filter', 'all', 'any', 'unique'],
        'Arithmetic': ['+', '-', 'mod']
    }

    # Count how many base primitives we have
    n_base = len(run_data.base_primitives)

    categories_html = ""
    for cat_name, prims in categories.items():
        # Check which primitives from this category are in the actual base primitives
        present = [p for p in prims if p in run_data.base_primitives]
        if present:
            prims_html = "".join(f'<span class="primitive-item">{p}</span>' for p in present)
            categories_html += f'''
                <div class="primitive-category">
                    <div class="category-title">{cat_name} ({len(present)})</div>
                    {prims_html}
                </div>
            '''

    return f"""
        <h2>Initial Library Reference</h2>
        <div class="collapsible-section collapsed">
            <div class="collapsible-header" onclick="toggleCollapsible(this)">
                <span class="collapsible-icon">▼</span>
                <span>Base Primitives ({n_base} total) - Click to expand</span>
            </div>
            <div class="collapsible-content">
                <p style="color: var(--text-secondary); margin-bottom: 15px;">
                    These are the cognitive primitives available at the start of training.
                    The system can learn new abstractions by composing these primitives.
                </p>
                <div class="primitives-grid">
                    {categories_html}
                </div>
            </div>
        </div>

        <script>
        function toggleCollapsible(header) {{
            const section = header.parentElement;
            section.classList.toggle('collapsed');
            const icon = header.querySelector('.collapsible-icon');
            icon.textContent = section.classList.contains('collapsed') ? '▼' : '▲';
        }}
        </script>
    """


def generate_composition_dag_section(run_data: RunData) -> str:
    """Generate interactive abstraction composition DAG section."""

    abstractions = run_data.learned_abstractions
    if not abstractions:
        return ""

    # Parse abstractions to find dependencies
    def parse_dependencies(name: str, all_prims: List[str]) -> List[str]:
        """Find which primitives/abstractions this abstraction uses."""
        deps = []
        # Extract the body from abstractions like "#((λ eq 14 (max_rank $0)))"
        body = name
        if body.startswith('#('):
            body = body[2:-1]  # Remove #( and )

        for prim in all_prims:
            # Don't include self or the exact same abstraction
            if prim == name:
                continue
            # Check if this primitive appears in the body
            prim_clean = prim.replace('#(', '').replace(')', '')
            if prim_clean in body or prim in body:
                deps.append(prim)
        return deps

    # Build composition tree data
    all_prims = run_data.base_primitives + [a.name for a in abstractions]

    tree_nodes = []
    for abs_info in abstractions:
        deps = parse_dependencies(abs_info.name, run_data.base_primitives)
        abs_deps = parse_dependencies(abs_info.name, [a.name for a in abstractions])

        paraphrase = paraphrase_program(abs_info.name)

        tree_nodes.append({
            'name': abs_info.name,
            'iteration': abs_info.iteration_invented,
            'deps': deps + abs_deps,
            'paraphrase': paraphrase,
            'used_in': abs_info.used_in_tasks,
            'depth': 1 + max([0] + [d.count('(') for d in abs_deps])  # Rough depth estimate
        })

    # Sort by depth for display
    tree_nodes.sort(key=lambda x: (x['depth'], x['iteration']))

    # Generate HTML for each node
    nodes_html = ""
    for node in tree_nodes:
        deps_html = ""
        if node['deps']:
            deps_html = '<div class="depends-on">Built from: ' + \
                ''.join(f'<span class="dep">{d}</span>' for d in node['deps'][:5]) + \
                ('...' if len(node['deps']) > 5 else '') + '</div>'

        used_html = ""
        if node['used_in']:
            used_html = f'<div style="margin-top: 5px;">Used in: {", ".join(node["used_in"][:3])}{"..." if len(node["used_in"]) > 3 else ""}</div>'

        node_class = "abstraction"
        nodes_html += f'''
            <div class="tree-node {node_class} collapsed">
                <div class="tree-node-header" onclick="toggleTreeNode(this)">
                    <span class="tree-node-icon">▶</span>
                    <span class="tree-node-name">{node['name'][:50]}{'...' if len(node['name']) > 50 else ''}</span>
                    <span class="tree-node-meta">iter {node['iteration']}</span>
                </div>
                <div class="tree-children">
                    <div class="tree-details">
                        <div class="paraphrase">{node['paraphrase'] or 'No paraphrase available'}</div>
                        {deps_html}
                        {used_html}
                    </div>
                </div>
            </div>
        '''

    return f"""
        <h2>Abstraction Composition Tree</h2>
        <div class="collapsible-section">
            <div class="collapsible-header" onclick="toggleCollapsible(this)">
                <span class="collapsible-icon">▼</span>
                <span>Learned Abstractions ({len(abstractions)} total) - Click nodes to expand</span>
            </div>
            <div class="collapsible-content">
                <p style="color: var(--text-secondary); margin-bottom: 15px;">
                    Each abstraction is a reusable pattern learned during training.
                    Click on an abstraction to see its composition and where it's used.
                </p>
                <div class="composition-tree">
                    {nodes_html}
                </div>
            </div>
        </div>

        <script>
        function toggleTreeNode(header) {{
            const node = header.parentElement;
            node.classList.toggle('collapsed');
            const icon = header.querySelector('.tree-node-icon');
            icon.textContent = node.classList.contains('collapsed') ? '▶' : '▼';
        }}
        </script>
    """


def generate_training_progress_section(run_data: RunData) -> str:
    """Generate the training progress charts section."""

    iterations = [it.iteration for it in run_data.iterations]
    solved_counts = [it.tasks_solved for it in run_data.iterations]
    losses = [it.recognition_loss for it in run_data.iterations]
    grammar_sizes = [it.grammar_size for it in run_data.iterations]

    return f"""
        <h2>Training Progress</h2>

        <div class="chart-container">
            <h3>Tasks Solved Over Iterations</h3>
            <canvas id="solvedChart"></canvas>
        </div>

        <div class="chart-container">
            <h3>Recognition Model Loss</h3>
            <canvas id="lossChart"></canvas>
        </div>

        <div class="chart-container">
            <h3>Grammar Size Growth</h3>
            <canvas id="grammarChart"></canvas>
        </div>

        <script>
            new Chart(document.getElementById('solvedChart'), {{
                type: 'line',
                data: {{
                    labels: {json.dumps(iterations)},
                    datasets: [{{
                        label: 'Tasks Solved',
                        data: {json.dumps(solved_counts)},
                        borderColor: '#4ecca3',
                        backgroundColor: 'rgba(78, 204, 163, 0.1)',
                        fill: true,
                        tension: 0.3
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{ beginAtZero: true, grid: {{ color: '#333' }} }},
                        x: {{ grid: {{ color: '#333' }} }}
                    }},
                    plugins: {{ legend: {{ labels: {{ color: '#e8e8e8' }} }} }}
                }}
            }});

            new Chart(document.getElementById('lossChart'), {{
                type: 'line',
                data: {{
                    labels: {json.dumps(iterations)},
                    datasets: [{{
                        label: 'Recognition Loss',
                        data: {json.dumps(losses)},
                        borderColor: '#e94560',
                        backgroundColor: 'rgba(233, 69, 96, 0.1)',
                        fill: true,
                        tension: 0.3
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{ grid: {{ color: '#333' }} }},
                        x: {{ grid: {{ color: '#333' }} }}
                    }},
                    plugins: {{ legend: {{ labels: {{ color: '#e8e8e8' }} }} }}
                }}
            }});

            new Chart(document.getElementById('grammarChart'), {{
                type: 'line',
                data: {{
                    labels: {json.dumps(iterations)},
                    datasets: [{{
                        label: 'Grammar Primitives',
                        data: {json.dumps(grammar_sizes)},
                        borderColor: '#ffc107',
                        backgroundColor: 'rgba(255, 193, 7, 0.1)',
                        fill: true,
                        tension: 0.3
                    }}]
                }},
                options: {{
                    responsive: true,
                    scales: {{
                        y: {{ beginAtZero: true, grid: {{ color: '#333' }} }},
                        x: {{ grid: {{ color: '#333' }} }}
                    }},
                    plugins: {{ legend: {{ labels: {{ color: '#e8e8e8' }} }} }}
                }}
            }});
        </script>
    """


def generate_recognition_analysis_section(run_data: RunData, snapshot_indices: List[int]) -> str:
    """Generate the recognition model analysis section with snapshots."""

    if not run_data.iterations:
        return "<h2>Recognition Model Analysis</h2><p>No iteration data available.</p>"

    sections = ["<h2>Recognition Model Analysis</h2>"]
    sections.append("<p>Analysis of the recognition model's learned representations at 4 key points during training. <strong>Hover over points</strong> to see task names and details.</p>")

    for i, idx in enumerate(snapshot_indices):
        if idx >= len(run_data.iterations):
            continue

        iteration = run_data.iterations[idx]
        snapshot_html = generate_snapshot_analysis(iteration, run_data.frontiers, i + 1)
        sections.append(snapshot_html)

    return "\n".join(sections)


def generate_snapshot_analysis(
    iteration: IterationData,
    frontiers: Dict[str, FrontierData],
    snapshot_num: int
) -> str:
    """Generate analysis for a single iteration snapshot."""

    clusters = cluster_embeddings(iteration.task_embeddings, n_clusters=5)
    coords_2d = reduce_dimensions(iteration.task_embeddings, method='pca')

    cluster_groups: Dict[int, List[str]] = defaultdict(list)
    for task_name, cluster_id in clusters.items():
        cluster_groups[cluster_id].append(task_name)

    cluster_html = []
    for cluster_id in sorted(cluster_groups.keys()):
        tasks = cluster_groups[cluster_id]
        task_tags = []
        for task in sorted(tasks):
            frontier = frontiers.get(task, FrontierData(task, False, None, 0))
            solved = frontier.solved
            css_class = "solved" if solved else "unsolved"
            program = frontier.best_program if frontier.best_program else "Not solved"
            paraphrase = paraphrase_program(frontier.best_program) if frontier.best_program else "No solution yet"
            tooltip = f"<strong>{task}</strong><br>{paraphrase}"
            task_tags.append(f'<span class="task-tag {css_class}" data-tooltip="{html_escape(tooltip)}">{task}</span>')

        cluster_html.append(f"""
            <div class="cluster-group">
                <strong>Cluster {cluster_id + 1}</strong> ({len(tasks)} tasks)
                <div class="cluster-tasks">
                    {''.join(task_tags)}
                </div>
            </div>
        """)

    scatter_svg = generate_embedding_scatter_svg(coords_2d, frontiers)

    return f"""
        <div class="snapshot-section">
            <h3>Snapshot {snapshot_num}: Iteration {iteration.iteration}</h3>
            <p>
                <strong>Tasks Solved:</strong> {iteration.tasks_solved}/{iteration.tasks_total} |
                <strong>Recognition Loss:</strong> {iteration.recognition_loss:.3f} |
                <strong>Grammar Size:</strong> {iteration.grammar_size}
            </p>

            <h4>Task Embedding Space (PCA)</h4>
            <p>How the model organizes tasks in its internal representation. <strong>Hover over points</strong> to see task names.</p>
            {scatter_svg}

            <h4>Emergent Clustering</h4>
            <p>Tasks spontaneously grouped by the model's learned representations:</p>
            {''.join(cluster_html)}
        </div>
    """


def generate_embedding_scatter_svg(
    coords_2d: Dict[str, Tuple[float, float]],
    frontiers: Dict[str, FrontierData],
    width: int = 700,
    height: int = 450
) -> str:
    """Generate an SVG scatter plot of task embeddings with working tooltips."""

    if not coords_2d:
        return "<p>No embedding data available.</p>"

    xs = [c[0] for c in coords_2d.values()]
    ys = [c[1] for c in coords_2d.values()]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_range = x_max - x_min if x_max != x_min else 1
    y_range = y_max - y_min if y_max != y_min else 1

    margin = 60
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin

    points = []
    for task_name, (x, y) in coords_2d.items():
        px = margin + (x - x_min) / x_range * plot_width
        py = margin + (1 - (y - y_min) / y_range) * plot_height

        frontier = frontiers.get(task_name, FrontierData(task_name, False, None, 0))
        solved = frontier.solved
        color = "#4ecca3" if solved else "#e94560"

        paraphrase = paraphrase_program(frontier.best_program) if frontier.best_program else "Not solved yet"
        tooltip = f"<strong>{task_name}</strong><br>Status: {'Solved' if solved else 'Unsolved'}<br>{paraphrase}"

        points.append(f"""
            <circle cx="{px:.1f}" cy="{py:.1f}" r="8" fill="{color}" opacity="0.85"
                    data-tooltip="{html_escape(tooltip)}" style="cursor: pointer;"/>
        """)

    return f"""
        <svg width="{width}" height="{height}" style="background: var(--bg-card); border-radius: 8px;">
            <line x1="{margin}" y1="{height - margin}" x2="{width - margin}" y2="{height - margin}" stroke="#555" stroke-width="1"/>
            <line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height - margin}" stroke="#555" stroke-width="1"/>

            <text x="{width/2}" y="{height - 15}" fill="#888" font-size="12" text-anchor="middle">PC1</text>
            <text x="15" y="{height/2}" fill="#888" font-size="12" text-anchor="middle" transform="rotate(-90, 15, {height/2})">PC2</text>

            {''.join(points)}

            <rect x="{width - 120}" y="20" width="12" height="12" rx="2" fill="#4ecca3"/>
            <text x="{width - 102}" y="30" fill="#e8e8e8" font-size="11">Solved</text>
            <rect x="{width - 120}" y="40" width="12" height="12" rx="2" fill="#e94560"/>
            <text x="{width - 102}" y="50" fill="#e8e8e8" font-size="11">Unsolved</text>
        </svg>
    """


def generate_abstraction_tree_section(run_data: RunData) -> str:
    """Generate the abstraction dependency tree section."""

    if not run_data.learned_abstractions:
        return """
            <h2>Abstraction Dependency Tree</h2>
            <p>No learned abstractions to display.</p>
        """

    sorted_abstractions = sorted(
        run_data.learned_abstractions,
        key=lambda a: a.iteration_invented
    )

    # Build a proper DAG
    dag_svg = generate_abstraction_dag_svg_improved(run_data)

    # Timeline items with hover
    timeline_items = []
    for abstr in sorted_abstractions[:25]:
        used_in = abstr.used_in_tasks[:5]
        more = len(abstr.used_in_tasks) - 5 if len(abstr.used_in_tasks) > 5 else 0

        used_text = ", ".join(used_in)
        if more > 0:
            used_text += f" (+{more} more)"

        paraphrase = paraphrase_abstraction(abstr.name)
        tooltip = f"<strong>{html_escape(abstr.name[:60])}</strong><br>Meaning: {paraphrase}"

        timeline_items.append(f"""
            <div class="abstraction-item">
                <div><strong>Iteration {abstr.iteration_invented}</strong></div>
                <div class="abstraction-name" data-tooltip="{tooltip}">{html_escape(abstr.name[:80])}</div>
                <div class="paraphrase">{paraphrase}</div>
                <div style="margin-top: 8px; font-size: 0.85rem; color: var(--text-secondary);">
                    Used in: {used_text if used_text else "No direct usage detected"}
                </div>
            </div>
        """)

    return f"""
        <h2>Abstraction Dependency Tree</h2>
        <p>How learned abstractions flow from solved tasks to future solutions. <strong>Hover over elements</strong> for details.</p>

        <h3>Knowledge Flow Diagram</h3>
        <div class="dag-container">
            {dag_svg}
        </div>

        <h3>Abstraction Timeline</h3>
        <p>Abstractions in order of invention (showing first 25):</p>
        <div class="abstraction-tree">
            {''.join(timeline_items)}
        </div>
    """


def generate_abstraction_dag_svg_improved(run_data: RunData, width: int = 900, height: int = 600) -> str:
    """Generate an improved SVG visualization of the abstraction dependency DAG."""

    solved_tasks = [(name, f) for name, f in run_data.frontiers.items() if f.solved]
    abstractions = [a for a in run_data.learned_abstractions if a.used_in_tasks][:15]

    if not abstractions:
        return "<p>No significant abstraction dependencies detected.</p>"

    margin = 80

    # Layout: 3 columns - source tasks, abstractions, benefiting tasks
    col1_x = margin
    col2_x = width / 2
    col3_x = width - margin - 100

    # Position source tasks (left)
    source_tasks = set()
    for abstr in abstractions:
        source_tasks.update(abstr.source_tasks[:3])
    source_tasks = list(source_tasks)[:12]

    task_positions = {}
    for i, task in enumerate(source_tasks):
        y = margin + 30 + (i / max(len(source_tasks) - 1, 1)) * (height - 2 * margin - 60)
        task_positions[f"src_{task}"] = (col1_x, y)

    # Position abstractions (middle)
    abstr_positions = {}
    for i, abstr in enumerate(abstractions):
        y = margin + 30 + (i / max(len(abstractions) - 1, 1)) * (height - 2 * margin - 60)
        abstr_positions[abstr.name] = (col2_x, y)

    # Position benefiting tasks (right)
    benefiting_tasks = set()
    for abstr in abstractions:
        benefiting_tasks.update(abstr.used_in_tasks[:3])
    benefiting_tasks = list(benefiting_tasks)[:12]

    for i, task in enumerate(benefiting_tasks):
        y = margin + 30 + (i / max(len(benefiting_tasks) - 1, 1)) * (height - 2 * margin - 60)
        task_positions[f"dst_{task}"] = (col3_x, y)

    # Generate SVG elements
    edges = []
    nodes = []

    # Draw edges from source tasks to abstractions
    for abstr in abstractions:
        if abstr.name in abstr_positions:
            ax, ay = abstr_positions[abstr.name]
            for task in abstr.source_tasks[:3]:
                key = f"src_{task}"
                if key in task_positions:
                    tx, ty = task_positions[key]
                    edges.append(f"""
                        <path d="M {tx + 80} {ty} C {tx + 150} {ty}, {ax - 100} {ay}, {ax - 50} {ay}"
                              stroke="#666" stroke-width="1.5" fill="none" opacity="0.4"
                              marker-end="url(#arrowhead)"/>
                    """)

    # Draw edges from abstractions to benefiting tasks
    for abstr in abstractions:
        if abstr.name in abstr_positions:
            ax, ay = abstr_positions[abstr.name]
            for task in abstr.used_in_tasks[:3]:
                key = f"dst_{task}"
                if key in task_positions:
                    tx, ty = task_positions[key]
                    edges.append(f"""
                        <path d="M {ax + 50} {ay} C {ax + 100} {ay}, {tx - 100} {ty}, {tx} {ty}"
                              stroke="#4ecca3" stroke-width="1.5" fill="none" opacity="0.5"
                              marker-end="url(#arrowhead-green)"/>
                    """)

    # Draw source task nodes
    for key, (x, y) in task_positions.items():
        if key.startswith("src_"):
            task = key[4:]
            frontier = run_data.frontiers.get(task)
            paraphrase = paraphrase_program(frontier.best_program) if frontier and frontier.best_program else ""
            tooltip = f"<strong>{task}</strong><br>{paraphrase}"
            nodes.append(f"""
                <rect x="{x}" y="{y - 12}" width="80" height="24" rx="4" fill="#4ecca3" opacity="0.9"
                      data-tooltip="{html_escape(tooltip)}" style="cursor: pointer;"/>
                <text x="{x + 40}" y="{y + 4}" fill="#1a1a2e" font-size="9" text-anchor="middle"
                      font-weight="bold" pointer-events="none">{task[:10]}</text>
            """)
        elif key.startswith("dst_"):
            task = key[4:]
            frontier = run_data.frontiers.get(task)
            paraphrase = paraphrase_program(frontier.best_program) if frontier and frontier.best_program else ""
            tooltip = f"<strong>{task}</strong><br>{paraphrase}"
            nodes.append(f"""
                <rect x="{x}" y="{y - 12}" width="100" height="24" rx="4" fill="#e94560" opacity="0.9"
                      data-tooltip="{html_escape(tooltip)}" style="cursor: pointer;"/>
                <text x="{x + 50}" y="{y + 4}" fill="#fff" font-size="9" text-anchor="middle"
                      font-weight="bold" pointer-events="none">{task[:12]}</text>
            """)

    # Draw abstraction nodes
    for abstr_name, (x, y) in abstr_positions.items():
        paraphrase = paraphrase_abstraction(abstr_name)
        short_name = abstr_name[2:30] + "..." if len(abstr_name) > 32 else abstr_name[2:]
        tooltip = f"<strong>Abstraction</strong><br>{html_escape(abstr_name[:80])}<br><em>{paraphrase}</em>"
        nodes.append(f"""
            <rect x="{x - 45}" y="{y - 14}" width="90" height="28" rx="5" fill="#ffc107" opacity="0.95"
                  data-tooltip="{html_escape(tooltip)}" style="cursor: pointer;"/>
            <text x="{x}" y="{y + 4}" fill="#1a1a2e" font-size="8" text-anchor="middle"
                  font-weight="bold" pointer-events="none">{html_escape(short_name[:15])}</text>
        """)

    return f"""
        <svg width="{width}" height="{height}" style="background: var(--bg-secondary); border-radius: 8px;">
            <defs>
                <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#666"/>
                </marker>
                <marker id="arrowhead-green" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                    <polygon points="0 0, 10 3.5, 0 7" fill="#4ecca3"/>
                </marker>
            </defs>

            <!-- Column labels -->
            <text x="{col1_x + 40}" y="25" fill="#888" font-size="13" text-anchor="middle" font-weight="bold">Source Tasks</text>
            <text x="{col2_x}" y="25" fill="#888" font-size="13" text-anchor="middle" font-weight="bold">Learned Abstractions</text>
            <text x="{col3_x + 50}" y="25" fill="#888" font-size="13" text-anchor="middle" font-weight="bold">Benefiting Tasks</text>

            <!-- Edges -->
            {''.join(edges)}

            <!-- Nodes -->
            {''.join(nodes)}

            <!-- Legend -->
            <rect x="{margin}" y="{height - 30}" width="12" height="12" rx="2" fill="#4ecca3"/>
            <text x="{margin + 18}" y="{height - 20}" fill="#e8e8e8" font-size="10">Tasks providing patterns</text>

            <rect x="{margin + 180}" y="{height - 30}" width="12" height="12" rx="2" fill="#ffc107"/>
            <text x="{margin + 198}" y="{height - 20}" fill="#e8e8e8" font-size="10">Learned abstractions</text>

            <rect x="{margin + 360}" y="{height - 30}" width="12" height="12" rx="2" fill="#e94560"/>
            <text x="{margin + 378}" y="{height - 20}" fill="#e8e8e8" font-size="10">Tasks using abstractions</text>
        </svg>
    """


def generate_solve_timeline_section(run_data: RunData) -> str:
    """Generate the solve timeline section."""

    solved_tasks = [(name, f.best_program) for name, f in run_data.frontiers.items() if f.solved]
    unsolved_tasks = [name for name, f in run_data.frontiers.items() if not f.solved]

    solved_rows = []
    for task_name, program in sorted(solved_tasks):
        paraphrase = paraphrase_program(program)
        program_display = html_escape(program) if program else "-"
        tooltip = f"<strong>{task_name}</strong><br>{paraphrase}"
        solved_rows.append(f"""
            <tr>
                <td class="solved">{task_name}</td>
                <td>
                    <span class="program" data-tooltip="{html_escape(tooltip)}">{program_display}</span>
                    <div class="paraphrase">{paraphrase}</div>
                </td>
            </tr>
        """)

    unsolved_rows = []
    for task_name in sorted(unsolved_tasks):
        unsolved_rows.append(f"""
            <tr>
                <td class="unsolved">{task_name}</td>
                <td>-</td>
            </tr>
        """)

    return f"""
        <h2>Solve Timeline</h2>

        <h3>Solved Tasks ({len(solved_tasks)})</h3>
        <table>
            <thead>
                <tr><th>Task</th><th>Program (hover for explanation)</th></tr>
            </thead>
            <tbody>
                {''.join(solved_rows)}
            </tbody>
        </table>

        <h3>Unsolved Tasks ({len(unsolved_tasks)})</h3>
        <table>
            <thead>
                <tr><th>Task</th><th>Status</th></tr>
            </thead>
            <tbody>
                {''.join(unsolved_rows)}
            </tbody>
        </table>
    """


def generate_solution_gallery_section(run_data: RunData) -> str:
    """Generate the solution gallery section."""

    solutions = []
    for task_name, frontier in sorted(run_data.frontiers.items()):
        if frontier.solved and frontier.best_program:
            solutions.append((task_name, frontier.best_program))

    if not solutions:
        return "<h2>Solution Gallery</h2><p>No solutions to display.</p>"

    gallery_items = []
    for task_name, program in solutions:
        paraphrase = paraphrase_program(program)
        tooltip = f"<strong>{task_name}</strong><br>Formula: {html_escape(program)}"
        gallery_items.append(f"""
            <div class="solution-card">
                <div class="solution-task">{task_name}</div>
                <div class="program" data-tooltip="{html_escape(tooltip)}">{html_escape(program)}</div>
                <div class="paraphrase">{paraphrase}</div>
            </div>
        """)

    return f"""
        <h2>Solution Gallery</h2>
        <p>Programs discovered for each solved task. <strong>Hover over programs</strong> for the full formula.</p>
        <div class="stats-grid" style="grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));">
            {''.join(gallery_items)}
        </div>
    """


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Generate systematic report from overnight run")
    parser.add_argument("--run-dir", type=Path, required=True, help="Path to run directory")
    parser.add_argument("--resume-dir", type=Path, help="Path to resume run directory (optional, for merging)")
    parser.add_argument("--output", type=Path, help="Output HTML file (default: run_dir/report.html)")

    args = parser.parse_args()

    if not args.run_dir.exists():
        print(f"Error: Run directory not found: {args.run_dir}")
        sys.exit(1)

    if args.resume_dir and not args.resume_dir.exists():
        print(f"Warning: Resume directory not found: {args.resume_dir}")
        args.resume_dir = None

    print(f"Loading data from {args.run_dir}...")
    if args.resume_dir:
        print(f"Merging with resume run: {args.resume_dir}")

    run_data = load_run_data(args.run_dir, args.resume_dir)

    print(f"Found {len(run_data.iterations)} iterations")
    print(f"Found {len(run_data.frontiers)} tasks ({sum(1 for f in run_data.frontiers.values() if f.solved)} solved)")
    print(f"Grammar: {len(run_data.grammar_primitives)} primitives")

    print("Generating report...")
    html = generate_html_report(run_data)

    output_path = args.output or (args.run_dir / "report.html")
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
