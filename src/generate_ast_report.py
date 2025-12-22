#!/usr/bin/env python3
"""
=============================================================================
AST-ENRICHED REPORT GENERATOR
=============================================================================

Generates a comprehensive HTML report using all the new AST utilities:
- pretty_print() - Visualize solution structure
- compact_str() - Readable format
- collect_primitives() - Which primitives used
- count_primitive_uses() - Frequency of primitives
- find_shared_subexpressions() - Common patterns across solutions
- alpha_equivalent() - Group equivalent solutions
- program_to_tree_dict() - JSON tree for visualization
- lambda_depth() - Nesting analysis
- count_applications() - Complexity metric
- uses_variable() - Variable usage patterns

Usage:
    python generate_ast_report.py --results <path_to_results.json>
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from dreamcoder_core.program import (
    Program, Primitive, Application, Abstraction, Index, Invented,
    pretty_print, compact_str, collect_primitives, collect_primitive_names,
    count_primitive_uses, find_shared_subexpressions, alpha_equivalent,
    program_to_tree_dict, lambda_depth, count_applications, uses_variable
)
from dreamcoder_core.lean_primitives import build_lean_grammar


# =============================================================================
# PROGRAM PARSER (simple parser for solution strings)
# =============================================================================

def parse_program(s: str, primitives: Dict[str, Primitive]) -> Program:
    """
    Parse a program string back into a Program AST.

    This is a simple recursive descent parser for our program syntax.
    Handles both (λ. body) and (λ body) formats.
    """
    tokens = _tokenize(s)
    pos = [0]  # Use list for mutation in nested function

    def parse_expr() -> Program:
        if pos[0] >= len(tokens):
            raise ValueError("Unexpected end of input")

        tok = tokens[pos[0]]

        if tok == '(':
            pos[0] += 1

            if pos[0] >= len(tokens):
                raise ValueError("Unexpected end after '('")

            # Check for lambda: (λ body) or (λ. body)
            if tokens[pos[0]] == 'λ':
                pos[0] += 1
                # Skip optional '.' after λ
                if pos[0] < len(tokens) and tokens[pos[0]] == '.':
                    pos[0] += 1

                # Parse the body - it might be multiple applications
                # e.g., (λ le (n_unique_ranks $0) 5) means λ.((le (n_unique_ranks $0)) 5)
                parts = []
                while pos[0] < len(tokens) and tokens[pos[0]] != ')':
                    parts.append(parse_expr())

                if pos[0] < len(tokens) and tokens[pos[0]] == ')':
                    pos[0] += 1

                # Build the body from parts
                if len(parts) == 0:
                    raise ValueError("Empty lambda body")
                elif len(parts) == 1:
                    body = parts[0]
                else:
                    # Multiple parts = application chain
                    body = parts[0]
                    for arg in parts[1:]:
                        body = Application(body, arg)

                return Abstraction(body)
            else:
                # Application: (f x1 x2 ...)
                func = parse_expr()
                args = []
                while pos[0] < len(tokens) and tokens[pos[0]] != ')':
                    args.append(parse_expr())
                if pos[0] < len(tokens):
                    pos[0] += 1  # consume ')'

                result = func
                for arg in args:
                    result = Application(result, arg)
                return result

        elif tok.startswith('$'):
            pos[0] += 1
            return Index(int(tok[1:]))

        elif tok.startswith('#'):
            pos[0] += 1
            # Invented - for now just return a placeholder
            name = tok[1:] if len(tok) > 1 else "invented"
            return Invented(Index(0), name=name)

        else:
            pos[0] += 1
            # Primitive or constant
            if tok in primitives:
                return primitives[tok]
            # Try as numeric constant
            try:
                val = int(tok)
                return Primitive(tok, None, val)
            except ValueError:
                pass
            # Return as unknown primitive
            return Primitive(tok, None, tok)

    result = parse_expr()
    return result


def _tokenize(s: str) -> List[str]:
    """Tokenize a program string."""
    tokens = []
    i = 0
    while i < len(s):
        c = s[i]
        if c.isspace():
            i += 1
        elif c in '()':
            tokens.append(c)
            i += 1
        elif c == 'λ':
            tokens.append('λ')
            i += 1
        elif c == '$':
            j = i + 1
            while j < len(s) and s[j].isdigit():
                j += 1
            tokens.append(s[i:j])
            i = j
        elif c == '#':
            j = i + 1
            while j < len(s) and (s[j].isalnum() or s[j] in '_-'):
                j += 1
            tokens.append(s[i:j])
            i = j
        else:
            # Token: alphanumeric, operators, etc.
            j = i
            while j < len(s) and not s[j].isspace() and s[j] not in '()':
                j += 1
            tokens.append(s[i:j])
            i = j
    return tokens


# =============================================================================
# AST ANALYSIS FUNCTIONS
# =============================================================================

@dataclass
class ProgramAnalysis:
    """Complete analysis of a program using AST utilities."""
    raw_string: str
    program: Program
    compact: str
    pretty: str
    tree_dict: Dict[str, Any]
    primitives_used: Set[str]
    primitive_counts: Dict[str, int]
    size: int
    depth: int
    lambda_depth: int
    application_count: int
    uses_arg: bool  # Does it use $0 (the hand argument)?


def analyze_program(prog_str: str, primitives: Dict[str, Primitive]) -> ProgramAnalysis:
    """Run full AST analysis on a program."""
    try:
        program = parse_program(prog_str, primitives)
    except Exception as e:
        # Create a dummy analysis for unparseable programs
        return ProgramAnalysis(
            raw_string=prog_str,
            program=None,
            compact=prog_str,
            pretty=f"[Parse error: {e}]",
            tree_dict={'type': 'Error', 'value': str(e)},
            primitives_used=set(),
            primitive_counts={},
            size=0,
            depth=0,
            lambda_depth=0,
            application_count=0,
            uses_arg=False
        )

    return ProgramAnalysis(
        raw_string=prog_str,
        program=program,
        compact=compact_str(program),
        pretty=pretty_print(program),
        tree_dict=program_to_tree_dict(program),
        primitives_used=collect_primitive_names(program),
        primitive_counts=count_primitive_uses(program),
        size=program.size(),
        depth=program.depth(),
        lambda_depth=lambda_depth(program),
        application_count=count_applications(program),
        uses_arg=uses_variable(program, 0)
    )


def find_cross_solution_patterns(analyses: List[ProgramAnalysis]) -> Dict[str, Any]:
    """Find patterns shared across multiple solutions."""
    # Aggregate primitive usage
    primitive_usage = defaultdict(int)  # primitive -> number of solutions using it
    primitive_total = defaultdict(int)  # primitive -> total uses across all solutions

    for analysis in analyses:
        for prim in analysis.primitives_used:
            primitive_usage[prim] += 1
        for prim, count in analysis.primitive_counts.items():
            primitive_total[prim] += count

    # Find common subexpressions across solutions
    # (We collect all subexpressions and count them)
    subexpr_counts = defaultdict(int)
    for analysis in analyses:
        if analysis.program:
            shared = find_shared_subexpressions(analysis.program)
            for subprog, count in shared:
                subexpr_counts[str(subprog)] += count

    # Group structurally similar solutions
    # (solutions with same primitive set and structure)
    structure_groups = defaultdict(list)
    for analysis in analyses:
        # Key: frozenset of primitives + depth + lambda_depth
        key = (frozenset(analysis.primitives_used), analysis.depth, analysis.lambda_depth)
        structure_groups[key].append(analysis.raw_string)

    return {
        'primitive_usage': dict(primitive_usage),
        'primitive_total': dict(primitive_total),
        'subexpr_counts': dict(subexpr_counts),
        'structure_groups': {str(k): v for k, v in structure_groups.items() if len(v) > 1}
    }


# =============================================================================
# HTML REPORT GENERATOR
# =============================================================================

def generate_html_report(results: Dict, analyses: Dict[str, ProgramAnalysis],
                         patterns: Dict[str, Any]) -> str:
    """Generate an HTML report with AST-enriched analysis."""

    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AST-Enriched DreamCoder Report</title>
    <style>
        :root {
            --bg: #1a1a2e;
            --surface: #16213e;
            --primary: #0f3460;
            --accent: #e94560;
            --text: #eaeaea;
            --muted: #888;
            --success: #4ade80;
            --warning: #fbbf24;
        }
        body {
            font-family: 'Fira Code', 'Consolas', monospace;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }
        h1, h2, h3 { color: var(--accent); margin-top: 2rem; }
        h1 { border-bottom: 2px solid var(--accent); padding-bottom: 10px; }
        .card {
            background: var(--surface);
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }
        .metric {
            display: inline-block;
            background: var(--primary);
            padding: 8px 16px;
            border-radius: 4px;
            margin: 5px;
        }
        .metric .value { font-size: 1.5em; color: var(--accent); }
        .metric .label { font-size: 0.8em; color: var(--muted); }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--primary);
        }
        th { background: var(--primary); color: var(--accent); }
        tr:hover { background: var(--primary); }
        .solved { color: var(--success); }
        .unsolved { color: var(--muted); }
        pre {
            background: #0d1117;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            font-size: 0.9em;
        }
        .pretty-tree {
            font-family: 'Fira Code', monospace;
            white-space: pre;
            background: #0d1117;
            padding: 15px;
            border-radius: 4px;
            overflow-x: auto;
            line-height: 1.4;
        }
        .prim-chip {
            display: inline-block;
            background: var(--primary);
            color: var(--text);
            padding: 2px 8px;
            border-radius: 12px;
            margin: 2px;
            font-size: 0.85em;
        }
        .bar {
            height: 20px;
            background: linear-gradient(90deg, var(--accent), var(--primary));
            border-radius: 4px;
            margin: 2px 0;
        }
        .stat-row { display: flex; align-items: center; margin: 5px 0; }
        .stat-label { width: 150px; }
        .stat-bar { flex: 1; background: var(--surface); border-radius: 4px; }
        .collapsible {
            cursor: pointer;
            padding: 10px;
            background: var(--primary);
            border-radius: 4px;
            margin: 5px 0;
            user-select: none;
        }
        .collapsible:hover { background: #1a4a7e; }
        .collapsible::before {
            content: '▶ ';
            display: inline-block;
            transition: transform 0.2s;
        }
        .collapsible.active::before {
            transform: rotate(90deg);
        }
        .collapsible-content {
            display: none;
            padding: 15px;
            background: rgba(0,0,0,0.2);
            border-radius: 0 0 4px 4px;
            margin-top: -5px;
            margin-bottom: 10px;
        }
        .collapsible.active + .collapsible-content { display: block; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .json-tree {
            font-size: 0.8em;
            max-height: 300px;
            overflow-y: auto;
        }
        .highlight { background: var(--accent); color: white; padding: 2px 4px; border-radius: 2px; }
    </style>
</head>
<body>
"""

    # Header
    summary = results.get('summary', {})
    config = results.get('config', {})

    html += f"""
    <h1>🔬 AST-Enriched DreamCoder Report</h1>
    <p style="color: var(--muted);">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

    <div class="card">
        <h2>📊 Summary</h2>
        <div>
            <div class="metric">
                <div class="value">{summary.get('tasks_solved', 0)}/{summary.get('tasks_total', 0)}</div>
                <div class="label">Tasks Solved</div>
            </div>
            <div class="metric">
                <div class="value">{summary.get('iterations_run', 0)}</div>
                <div class="label">Iterations</div>
            </div>
            <div class="metric">
                <div class="value">{summary.get('final_grammar_size', 0)}</div>
                <div class="label">Grammar Size</div>
            </div>
            <div class="metric">
                <div class="value">{summary.get('total_abstractions', 0)}</div>
                <div class="label">New Abstractions</div>
            </div>
            <div class="metric">
                <div class="value">{int(summary.get('total_time', 0))}s</div>
                <div class="label">Total Time</div>
            </div>
        </div>
    </div>
"""

    # Primitive Usage Analysis
    html += """
    <div class="card">
        <h2>🧬 Primitive Usage Analysis</h2>
        <p>How primitives are used across all solved tasks:</p>
"""

    # Sort by usage count
    prim_usage = patterns.get('primitive_usage', {})
    prim_total = patterns.get('primitive_total', {})
    sorted_prims = sorted(prim_usage.items(), key=lambda x: -x[1])

    if sorted_prims:
        max_usage = max(v for _, v in sorted_prims) if sorted_prims else 1
        html += "<div style='max-height: 400px; overflow-y: auto;'>"
        for prim, count in sorted_prims[:30]:  # Top 30
            total = prim_total.get(prim, count)
            pct = (count / len(analyses)) * 100
            bar_width = (count / max_usage) * 100
            html += f"""
            <div class="stat-row">
                <div class="stat-label"><code>{prim}</code></div>
                <div class="stat-bar">
                    <div class="bar" style="width: {bar_width}%"></div>
                </div>
                <div style="width: 150px; text-align: right;">
                    {count} tasks ({pct:.0f}%) · {total} uses
                </div>
            </div>
"""
        html += "</div>"
    html += "</div>"

    # Program Structure Analysis
    html += """
    <div class="card">
        <h2>🏗️ Program Structure Analysis</h2>
        <div class="grid">
"""

    # Compute aggregate stats
    solved_analyses = [a for a in analyses.values() if a.program is not None]
    if solved_analyses:
        avg_size = sum(a.size for a in solved_analyses) / len(solved_analyses)
        avg_depth = sum(a.depth for a in solved_analyses) / len(solved_analyses)
        avg_lambda = sum(a.lambda_depth for a in solved_analyses) / len(solved_analyses)
        avg_apps = sum(a.application_count for a in solved_analyses) / len(solved_analyses)
        uses_arg_pct = sum(1 for a in solved_analyses if a.uses_arg) / len(solved_analyses) * 100

        html += f"""
            <div class="card">
                <h3>Size Distribution</h3>
                <div class="metric">
                    <div class="value">{avg_size:.1f}</div>
                    <div class="label">Average AST Size</div>
                </div>
                <div class="metric">
                    <div class="value">{min(a.size for a in solved_analyses)}-{max(a.size for a in solved_analyses)}</div>
                    <div class="label">Range</div>
                </div>
            </div>
            <div class="card">
                <h3>Depth Analysis</h3>
                <div class="metric">
                    <div class="value">{avg_depth:.1f}</div>
                    <div class="label">Avg AST Depth</div>
                </div>
                <div class="metric">
                    <div class="value">{avg_lambda:.1f}</div>
                    <div class="label">Avg Lambda Depth</div>
                </div>
            </div>
            <div class="card">
                <h3>Complexity</h3>
                <div class="metric">
                    <div class="value">{avg_apps:.1f}</div>
                    <div class="label">Avg Applications</div>
                </div>
                <div class="metric">
                    <div class="value">{uses_arg_pct:.0f}%</div>
                    <div class="label">Use $0 (arg)</div>
                </div>
            </div>
"""

    html += "</div></div>"

    # Individual Solution Analysis
    html += """
    <div class="card">
        <h2>🔍 Individual Solution Analysis</h2>
        <p>Click on each task to expand detailed AST analysis:</p>
"""

    task_metrics = results.get('task_metrics', {})
    sorted_tasks = sorted(
        task_metrics.items(),
        key=lambda x: (not x[1].get('solved', False), x[1].get('programs_to_solve', float('inf')))
    )

    for task_name, metrics in sorted_tasks:
        solved = metrics.get('solved', False)
        status_class = 'solved' if solved else 'unsolved'
        status_icon = '✅' if solved else '❌'

        html += f"""
        <div class="collapsible">
            <span class="{status_class}">{status_icon}</span> <strong>{task_name}</strong>
            <span style="float: right; color: var(--muted);">
                {metrics.get('task_family', '')} ·
                {metrics.get('programs_to_solve', 'N/A')} programs ·
                {metrics.get('description_length', 0):.1f} bits
            </span>
        </div>
        <div class="collapsible-content">
"""

        if solved and task_name in analyses:
            analysis = analyses[task_name]

            # Compact representation
            html += f"""
            <h4>Compact Form</h4>
            <pre>{analysis.compact}</pre>

            <h4>AST Pretty Print</h4>
            <div class="pretty-tree">{analysis.pretty}</div>

            <h4>Metrics</h4>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>AST Size</td><td>{analysis.size}</td></tr>
                <tr><td>AST Depth</td><td>{analysis.depth}</td></tr>
                <tr><td>Lambda Depth</td><td>{analysis.lambda_depth}</td></tr>
                <tr><td>Application Count</td><td>{analysis.application_count}</td></tr>
                <tr><td>Uses Argument ($0)</td><td>{'Yes' if analysis.uses_arg else 'No'}</td></tr>
            </table>

            <h4>Primitives Used ({len(analysis.primitives_used)})</h4>
            <div>
"""
            for prim in sorted(analysis.primitives_used):
                count = analysis.primitive_counts.get(prim, 1)
                html += f'<span class="prim-chip">{prim} ×{count}</span>'

            html += f"""
            </div>

            <h4>Tree Structure (JSON)</h4>
            <div class="json-tree">
                <pre>{json.dumps(analysis.tree_dict, indent=2)}</pre>
            </div>
"""
        elif not solved:
            html += "<p style='color: var(--muted);'>Task not solved - no program to analyze.</p>"

        html += "</div>"

    html += "</div>"

    # Shared Subexpressions
    subexpr_counts = patterns.get('subexpr_counts', {})
    if subexpr_counts:
        html += """
    <div class="card">
        <h2>🔗 Shared Subexpressions</h2>
        <p>Subexpressions that appear multiple times (candidates for abstraction):</p>
        <table>
            <tr><th>Subexpression</th><th>Occurrences</th></tr>
"""
        sorted_subexpr = sorted(subexpr_counts.items(), key=lambda x: -x[1])[:20]
        for subexpr, count in sorted_subexpr:
            if count > 1:
                html += f"<tr><td><code>{subexpr[:80]}{'...' if len(subexpr) > 80 else ''}</code></td><td>{count}</td></tr>"
        html += "</table></div>"

    # Structure Groups (similar solutions)
    structure_groups = patterns.get('structure_groups', {})
    if structure_groups:
        html += """
    <div class="card">
        <h2>🧩 Structurally Similar Solutions</h2>
        <p>Tasks with similar program structure (same primitives, depth, lambda nesting):</p>
"""
        for key, tasks in structure_groups.items():
            if len(tasks) > 1:
                html += f"""
        <div class="collapsible">
            Group of {len(tasks)} similar solutions
        </div>
        <div class="collapsible-content">
            <ul>
"""
                for task in tasks:
                    html += f"<li><code>{task[:60]}...</code></li>"
                html += "</ul></div>"
        html += "</div>"

    # Learning Curve with Enumeration Stats
    html += """
    <div class="card">
        <h2>📈 Learning Curve</h2>
        <table>
            <tr><th>Iteration</th><th>Solved</th><th>Complete Progs</th><th>Partial Progs</th><th>Ratio</th><th>Loss</th></tr>
"""
    for lc in results.get('learning_curve', []):
        complete = lc.get('programs_enumerated', 0)
        partial = lc.get('partial_programs_explored', 0)
        ratio = partial / max(1, complete)
        html += f"""
            <tr>
                <td>{lc.get('iteration', 0) + 1}</td>
                <td>{lc.get('tasks_solved', 0)}/{lc.get('tasks_total', 0)}</td>
                <td>{complete:,}</td>
                <td>{partial:,}</td>
                <td>{ratio:.1f}:1</td>
                <td>{lc.get('recognition_loss', 0):.4f}</td>
            </tr>
"""
    html += "</table></div>"

    # Library Evolution
    library_evo = results.get('library_evolution', [])
    if library_evo:
        html += """
    <div class="card">
        <h2>📚 Library Evolution</h2>
        <p>Abstractions learned at each iteration:</p>
"""
        for i, abstractions in enumerate(library_evo):
            if abstractions:
                html += f"<h4>Iteration {i+1}</h4><ul>"
                for abstr in abstractions:
                    html += f"<li><code>{abstr}</code></li>"
                html += "</ul>"
        html += "</div>"

    html += """
    <script>
    // Collapsible functionality using event delegation
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('collapsible')) {
            e.target.classList.toggle('active');
        }
    });
    </script>
</body>
</html>
"""

    return html


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate AST-enriched report')
    parser.add_argument('--results', '-r', type=str, required=True,
                        help='Path to results JSON file')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output HTML file (default: same dir as results)')
    args = parser.parse_args()

    # Load results
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)

    print(f"Loading results from: {results_path}")
    with open(results_path) as f:
        results = json.load(f)

    # Build primitive lookup
    print("Building primitive lookup...")
    grammar = build_lean_grammar()
    primitives = {}
    for prod in grammar.productions:
        if isinstance(prod.program, Primitive):
            primitives[prod.program.name] = prod.program
    print(f"  Found {len(primitives)} primitives")

    # Analyze each solved program
    print("Analyzing solutions...")
    analyses = {}
    task_metrics = results.get('task_metrics', {})

    for task_name, metrics in task_metrics.items():
        if metrics.get('solved') and metrics.get('best_program'):
            prog_str = metrics['best_program']
            analysis = analyze_program(prog_str, primitives)
            analyses[task_name] = analysis
            print(f"  {task_name}: size={analysis.size}, depth={analysis.depth}, "
                  f"prims={len(analysis.primitives_used)}")

    # Find cross-solution patterns
    print("Finding cross-solution patterns...")
    patterns = find_cross_solution_patterns(list(analyses.values()))
    print(f"  Found {len(patterns.get('primitive_usage', {}))} unique primitives")
    print(f"  Found {len(patterns.get('structure_groups', {}))} structure groups")

    # Generate report
    print("Generating HTML report...")
    html = generate_html_report(results, analyses, patterns)

    # Write output
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = results_path.parent / f"ast_report_{results_path.stem}.html"

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"\nReport saved to: {output_path}")
    print("Open in browser to view the AST-enriched analysis!")


if __name__ == "__main__":
    main()
