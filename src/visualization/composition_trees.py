#!/usr/bin/env python3
"""
Composition Tree Visualization

This module generates visual representations of rule compositions as trees,
showing how rules are built from primitives.

Features:
1. ASCII tree representation
2. SVG tree diagrams (embeddable in HTML)
3. Graphviz DOT format
"""

import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rules.catalogue import ALL_RULES, CompositionNode, Rule


# ============================================================================
# ASCII TREE REPRESENTATION
# ============================================================================

def composition_to_ascii_tree(node: CompositionNode, prefix: str = "", is_last: bool = True) -> str:
    """
    Generate ASCII tree representation of a composition.

    Example output:
    is_sorted
    └── map
        ├── get_rank_val
        └── h

    Args:
        node: CompositionNode to visualize
        prefix: Current line prefix (for indentation)
        is_last: Whether this is the last child

    Returns:
        ASCII tree string
    """
    connector = "└── " if is_last else "├── "
    result = prefix + connector + node.primitive

    # Add parameters if present
    if node.params:
        param_str = ", ".join(f"{k}={v}" for k, v in node.params.items())
        result += f" ({param_str})"

    result += "\n"

    # Process children
    child_prefix = prefix + ("    " if is_last else "│   ")

    for i, child in enumerate(node.args):
        is_last_child = (i == len(node.args) - 1)
        result += composition_to_ascii_tree(child, child_prefix, is_last_child)

    return result


def rule_to_ascii_tree(rule: Rule) -> str:
    """
    Generate ASCII tree for a rule.

    Returns header + tree.
    """
    header = f"Rule: {rule.id} ({rule.token})\n"
    header += f"Lambda: λh. ...\n"
    header += "─" * 40 + "\n"

    # Start with the root (no prefix, treat as "last" item)
    tree = composition_to_ascii_tree(rule.composition, "", True)

    # Remove the leading connector from root
    tree = tree.replace("└── ", "", 1)

    return header + tree


# ============================================================================
# SVG TREE REPRESENTATION
# ============================================================================

@dataclass
class TreeNode:
    """Node in the visual tree layout."""
    label: str
    x: float = 0
    y: float = 0
    children: List['TreeNode'] = None
    width: float = 0

    def __post_init__(self):
        if self.children is None:
            self.children = []


def composition_to_tree_nodes(node: CompositionNode) -> TreeNode:
    """Convert CompositionNode to TreeNode for layout."""
    label = node.primitive
    if node.params:
        # Shorten parameter display
        params = []
        for k, v in node.params.items():
            if isinstance(v, str):
                params.append(f"{v}")
            else:
                params.append(f"{k}={v}")
        if params:
            label += f"\n({', '.join(params[:2])})"

    children = [composition_to_tree_nodes(child) for child in node.args]

    return TreeNode(label=label, children=children)


def layout_tree(node: TreeNode, x: float = 0, y: float = 0, level_height: float = 60) -> Tuple[float, float]:
    """
    Lay out tree nodes with positions.

    Returns (total_width, total_height).
    """
    node.y = y

    if not node.children:
        # Leaf node
        node.width = 80  # Fixed width for leaves
        node.x = x + node.width / 2
        return node.width, level_height

    # Layout children first
    child_x = x
    max_child_height = 0
    total_child_width = 0

    for child in node.children:
        w, h = layout_tree(child, child_x, y + level_height, level_height)
        child_x += w + 20  # spacing
        total_child_width += w + 20
        max_child_height = max(max_child_height, h)

    total_child_width -= 20  # Remove last spacing

    # Center parent above children
    first_child_x = node.children[0].x
    last_child_x = node.children[-1].x
    node.x = (first_child_x + last_child_x) / 2
    node.width = total_child_width

    return total_child_width, level_height + max_child_height


def tree_to_svg(node: TreeNode, width: float, height: float) -> str:
    """
    Generate SVG representation of the tree.

    Args:
        node: Root TreeNode with layout
        width: SVG width
        height: SVG height

    Returns:
        SVG string
    """
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
    ]

    # Style definitions
    svg_parts.append('''
    <defs>
        <style>
            .node-box { fill: #e3f2fd; stroke: #1976d2; stroke-width: 2; rx: 8; }
            .node-text { font-family: 'Consolas', monospace; font-size: 11px; fill: #1a237e; text-anchor: middle; }
            .edge { stroke: #90a4ae; stroke-width: 2; fill: none; }
            .lambda { fill: #7b1fa2; font-weight: bold; }
        </style>
    </defs>
    ''')

    def render_node(n: TreeNode):
        parts = []

        # Draw edges to children first (so they're behind boxes)
        for child in n.children:
            parts.append(
                f'<path class="edge" d="M{n.x},{n.y + 18} Q{n.x},{n.y + 35} {child.x},{child.y - 15}"/>'
            )

        # Draw node box
        box_width = max(60, len(n.label.split('\n')[0]) * 8 + 16)
        box_height = 36 if '\n' in n.label else 28

        parts.append(
            f'<rect class="node-box" x="{n.x - box_width/2}" y="{n.y - 14}" '
            f'width="{box_width}" height="{box_height}"/>'
        )

        # Draw text
        lines = n.label.split('\n')
        for i, line in enumerate(lines):
            text_y = n.y + (i * 12) - (6 if len(lines) > 1 else 0)
            css_class = "node-text lambda" if line.startswith('λ') else "node-text"
            parts.append(
                f'<text class="{css_class}" x="{n.x}" y="{text_y + 4}">{line}</text>'
            )

        # Render children
        for child in n.children:
            parts.extend(render_node(child))

        return parts

    svg_parts.extend(render_node(node))
    svg_parts.append('</svg>')

    return '\n'.join(svg_parts)


def rule_to_svg(rule: Rule, max_width: float = 600, max_height: float = 400) -> str:
    """
    Generate SVG tree diagram for a rule.

    Args:
        rule: Rule to visualize
        max_width: Maximum SVG width
        max_height: Maximum SVG height

    Returns:
        SVG string
    """
    # Convert to layout tree
    root = composition_to_tree_nodes(rule.composition)

    # Layout
    tree_width, tree_height = layout_tree(root, x=20, y=30, level_height=55)

    # Adjust dimensions
    width = min(max_width, tree_width + 40)
    height = min(max_height, tree_height + 40)

    return tree_to_svg(root, width, height)


# ============================================================================
# HTML GENERATION
# ============================================================================

def generate_tree_html(rules: List[Rule] = None, include_svg: bool = True) -> str:
    """
    Generate HTML page with composition trees for all rules.

    Args:
        rules: List of rules (defaults to ALL_RULES)
        include_svg: Whether to include SVG diagrams (larger output)

    Returns:
        HTML string
    """
    if rules is None:
        rules = ALL_RULES

    html = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Rule Composition Trees</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #1a237e; }
        h2 { color: #1976d2; margin-top: 40px; }
        .rule-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .rule-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #e3f2fd;
            padding-bottom: 10px;
            margin-bottom: 15px;
        }
        .rule-id {
            font-size: 1.2em;
            font-weight: bold;
            color: #1565c0;
        }
        .rule-token {
            background: #e3f2fd;
            padding: 4px 12px;
            border-radius: 20px;
            font-family: monospace;
        }
        .rule-family {
            background: #1976d2;
            color: white;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
        }
        .lambda-notation {
            background: #fce4ec;
            padding: 12px;
            border-radius: 8px;
            font-family: 'Consolas', monospace;
            font-size: 14px;
            margin: 10px 0;
            overflow-x: auto;
        }
        .lambda-notation .lambda {
            color: #7b1fa2;
            font-weight: bold;
        }
        .ascii-tree {
            background: #263238;
            color: #80cbc4;
            padding: 15px;
            border-radius: 8px;
            font-family: 'Consolas', monospace;
            font-size: 13px;
            overflow-x: auto;
            white-space: pre;
        }
        .tree-svg {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 10px;
            margin-top: 10px;
            overflow-x: auto;
        }
        .primitives {
            margin-top: 10px;
        }
        .primitive-tag {
            display: inline-block;
            background: #e8f5e9;
            color: #2e7d32;
            padding: 3px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-family: monospace;
            margin: 2px;
        }
        .level-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: bold;
            margin-left: 10px;
        }
        .level-0 { background: #c8e6c9; color: #1b5e20; }
        .level-1 { background: #bbdefb; color: #0d47a1; }
        .level-2 { background: #e1bee7; color: #6a1b9a; }
        .level-3 { background: #ffccbc; color: #bf360c; }
        .level-4 { background: #ffcdd2; color: #b71c1c; }
        .toc {
            background: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 30px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .toc ul {
            columns: 3;
            list-style: none;
            padding: 0;
        }
        .toc li {
            margin: 5px 0;
        }
        .toc a {
            color: #1976d2;
            text-decoration: none;
        }
        .toc a:hover {
            text-decoration: underline;
        }
    </style>
</head>
<body>
    <h1>🌳 Rule Composition Trees</h1>
    <p>This page shows how each of the 57 card game rules is composed from primitive functions.
    The tree structure reveals the compositional hierarchy.</p>

    <div class="toc">
        <h3>Quick Navigation</h3>
        <ul>
'''

    # Table of contents
    for rule in rules:
        html += f'            <li><a href="#{rule.id}">{rule.token}: {rule.id}</a></li>\n'

    html += '''        </ul>
    </div>

'''

    # Rule cards
    for rule in rules:
        # Lambda notation with syntax highlighting
        lambda_str = rule.lambda_str().replace('λ', '<span class="lambda">λ</span>')

        html += f'''    <div class="rule-card" id="{rule.id}">
        <div class="rule-header">
            <span class="rule-id">{rule.id}</span>
            <span>
                <span class="rule-token">{rule.token}</span>
                <span class="rule-family">{rule.family}</span>
                <span class="level-badge level-{rule.level}">Level {rule.level}</span>
            </span>
        </div>

        <p><strong>Description:</strong> {rule.description}</p>

        <div class="lambda-notation">
            <strong>Lambda notation:</strong> {lambda_str}
        </div>

        <div class="ascii-tree">{rule_to_ascii_tree(rule)}</div>
'''

        if include_svg:
            svg = rule_to_svg(rule)
            html += f'''        <div class="tree-svg">
            {svg}
        </div>
'''

        html += f'''        <div class="primitives">
            <strong>Primitives used:</strong>
'''
        for p in rule.primitives_used:
            html += f'            <span class="primitive-tag">{p}</span>\n'

        html += '''        </div>
    </div>

'''

    html += '''</body>
</html>
'''

    return html


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    print("Generating composition tree visualizations...")

    # Generate HTML with all trees
    html = generate_tree_html(ALL_RULES, include_svg=True)

    output_path = output_dir / "composition_trees.html"
    with open(output_path, 'w') as f:
        f.write(html)

    print(f"✓ Saved: {output_path}")

    # Print sample ASCII trees
    print("\nSample ASCII trees:")
    for rule in ALL_RULES[:3]:
        print(rule_to_ascii_tree(rule))
        print()

    print(f"\nGenerated trees for {len(ALL_RULES)} rules")
