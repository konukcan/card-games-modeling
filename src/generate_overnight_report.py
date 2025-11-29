#!/usr/bin/env python3
"""
Generate HTML Report for DreamCoder Overnight Runs

Parses the log file from run_overnight_cython.py and generates a comprehensive
interactive HTML report with charts, metrics, and analysis.

Usage:
    python src/generate_overnight_report.py results/overnight_cython/run_cognitive_20251128_150300.log
    python src/generate_overnight_report.py  # Uses most recent log file
"""

import re
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class IterationMetrics:
    """Metrics for a single iteration."""
    iteration: int
    phase: int
    phase_name: str
    phase_iter: int
    timestamp: str
    tasks_solved: int
    tasks_total: int
    programs_enumerated: int
    grammar_size: int
    recognition_loss: float
    dreams_generated: int
    abstractions_found: int
    cumulative_solved: int
    new_solutions: List[Tuple[str, int]] = field(default_factory=list)  # (task_name, programs)


@dataclass
class RunSummary:
    """Summary of the entire run."""
    start_time: str
    end_time: str
    total_time: str
    total_iterations: int
    total_tasks: int
    tasks_solved: int
    final_grammar_size: int
    total_abstractions: int
    total_dreams: int
    total_programs: int
    solved_tasks: List[Tuple[str, int]]  # (task_name, iteration_solved)
    iterations: List[IterationMetrics] = field(default_factory=list)
    rejected_spurious: Dict[str, List[str]] = field(default_factory=dict)  # task -> programs
    solved_programs: Dict[str, str] = field(default_factory=dict)  # task_name -> program string
    abstractions_by_iteration: Dict[int, List[str]] = field(default_factory=dict)  # iter -> abstractions


def _render_program_tree(program_str: str, task_name: str) -> str:
    """Render a program string as a simple tree structure in HTML.

    For example: (all_same_suit $0) becomes a tree showing the function and argument.
    """
    if not program_str or program_str == "Unknown":
        return '<span class="tree-unknown">Program not recorded</span>'

    # Parse the program into a tree structure
    def parse_sexpr(s: str, depth: int = 0) -> str:
        """Parse S-expression and render as HTML tree."""
        s = s.strip()

        if not s:
            return ''

        # Variable reference
        if s.startswith('$'):
            return f'<span class="tree-var">{s}</span>'

        # Lambda abstraction
        if s.startswith('(λ') or s.startswith('(\\') or s.startswith('(lambda'):
            # Extract body
            inner = s[2:-1].strip() if s.startswith('(λ') else s[s.find(' ')+1:-1].strip()
            return f'<span class="tree-lambda">λ</span> {parse_sexpr(inner, depth+1)}'

        # Application (function call)
        if s.startswith('('):
            # Remove outer parens
            inner = s[1:-1].strip() if s.endswith(')') else s[1:].strip()

            # Find function name and arguments
            parts = []
            paren_depth = 0
            current = []

            for char in inner:
                if char == '(':
                    paren_depth += 1
                    current.append(char)
                elif char == ')':
                    paren_depth -= 1
                    current.append(char)
                elif char == ' ' and paren_depth == 0:
                    if current:
                        parts.append(''.join(current))
                        current = []
                else:
                    current.append(char)

            if current:
                parts.append(''.join(current))

            if len(parts) == 0:
                return s

            func_name = parts[0]
            args = parts[1:]

            # Build tree HTML
            result = f'<span class="tree-func">{func_name}</span>'

            if args:
                result += '<span class="tree-args">'
                for arg in args:
                    result += f'<span class="tree-arg">{parse_sexpr(arg, depth+1)}</span>'
                result += '</span>'

            return result

        # Primitive or constant
        return f'<span class="tree-prim">{s}</span>'

    try:
        return f'<div class="tree-content">{parse_sexpr(program_str)}</div>'
    except Exception:
        # Fallback to showing the raw program
        return f'<code class="tree-raw">{program_str}</code>'


def parse_log_file(log_path: Path) -> Tuple[RunSummary, List[IterationMetrics]]:
    """Parse the overnight run log file."""

    with open(log_path, 'r') as f:
        content = f.read()

    lines = content.split('\n')

    # Initialize
    iterations = []
    current_iter = None
    solved_tasks = []
    rejected_spurious = {}
    solved_programs = {}  # task_name -> program string
    abstractions_by_iteration = {}  # iter_num -> list of abstractions

    # Parse run summary from end of file
    start_time = ""
    end_time = ""
    total_time = ""

    for line in lines:
        # Start/end times
        if 'Start time:' in line and not start_time:
            match = re.search(r'Start time: (.+)', line)
            if match:
                start_time = match.group(1)

        if 'End time:' in line:
            match = re.search(r'End time: (.+)', line)
            if match:
                end_time = match.group(1)

        if 'Total time:' in line:
            match = re.search(r'Total time: (.+)', line)
            if match:
                total_time = match.group(1)

        # Iteration header
        iter_match = re.match(r'\[(\d{2}:\d{2}:\d{2})\] ITERATION (\d+) \(Phase (\d+), iter (\d+)/(\d+)\)', line)
        if iter_match:
            timestamp = iter_match.group(1)
            iter_num = int(iter_match.group(2))
            phase = int(iter_match.group(3))
            phase_iter = int(iter_match.group(4))

            current_iter = IterationMetrics(
                iteration=iter_num,
                phase=phase,
                phase_name=f"Phase {phase}",
                phase_iter=phase_iter,
                timestamp=timestamp,
                tasks_solved=0,
                tasks_total=0,
                programs_enumerated=0,
                grammar_size=0,
                recognition_loss=0.0,
                dreams_generated=0,
                abstractions_found=0,
                cumulative_solved=0,
                new_solutions=[]
            )
            continue

        # SOLVED tasks - also capture the program if on the same or next line
        solved_match = re.search(r'SOLVED \(verified\): (\S+) \(([\d,]+) programs\)', line)
        if solved_match and current_iter:
            task_name = solved_match.group(1)
            programs = int(solved_match.group(2).replace(',', ''))
            current_iter.new_solutions.append((task_name, programs))
            solved_tasks.append((task_name, current_iter.iteration))

        # Capture program for solved task (format: "    Program: (λ ...)")
        prog_match = re.search(r'Program: (.+)', line)
        if prog_match and current_iter and current_iter.new_solutions:
            last_task = current_iter.new_solutions[-1][0]
            if last_task not in solved_programs:
                solved_programs[last_task] = prog_match.group(1).strip()

        # Rejected spurious
        spurious_match = re.search(r'Rejected spurious: (\S+) - (.+)', line)
        if spurious_match:
            task_name = spurious_match.group(1)
            program = spurious_match.group(2)
            if task_name not in rejected_spurious:
                rejected_spurious[task_name] = []
            if len(rejected_spurious[task_name]) < 10:  # Limit storage
                rejected_spurious[task_name].append(program)

        # Iteration summary metrics
        if current_iter:
            # Solved count
            solved_match = re.search(r'Solved: (\d+)/(\d+)', line)
            if solved_match:
                current_iter.tasks_solved = int(solved_match.group(1))
                current_iter.tasks_total = int(solved_match.group(2))

            # Programs
            prog_match = re.search(r'Programs: ([\d,]+)', line)
            if prog_match:
                current_iter.programs_enumerated = int(prog_match.group(1).replace(',', ''))

            # Grammar size
            grammar_match = re.search(r'Grammar size: (\d+)', line)
            if grammar_match:
                current_iter.grammar_size = int(grammar_match.group(1))

            # Recognition loss
            loss_match = re.search(r'Recognition loss: ([\d.]+)', line)
            if loss_match:
                current_iter.recognition_loss = float(loss_match.group(1))

            # Dreams
            dreams_match = re.search(r'Dreams generated: (\d+)', line)
            if dreams_match:
                current_iter.dreams_generated = int(dreams_match.group(1))

            # Abstractions count
            abstr_match = re.search(r'Found (\d+) abstraction', line)
            if abstr_match:
                current_iter.abstractions_found = int(abstr_match.group(1))

            # Abstraction details (format: "  - abstraction_name" or "    Invented: abstraction_name")
            invented_match = re.search(r'Invented: (\S+)', line)
            if invented_match:
                abstr_name = invented_match.group(1)
                if current_iter.iteration not in abstractions_by_iteration:
                    abstractions_by_iteration[current_iter.iteration] = []
                abstractions_by_iteration[current_iter.iteration].append(abstr_name)

            # Cumulative solved
            cum_match = re.search(r'Cumulative solved \(all tasks\): (\d+)/(\d+)', line)
            if cum_match:
                current_iter.cumulative_solved = int(cum_match.group(1))
                # This marks end of iteration summary, save it
                iterations.append(current_iter)
                current_iter = None

    # Parse final summary
    tasks_solved_final = 0
    total_tasks = 43
    final_grammar = 60
    total_abstractions = 0
    total_dreams = 0

    for line in lines[-50:]:  # Check last 50 lines for summary
        if 'Tasks solved:' in line:
            match = re.search(r'Tasks solved: (\d+)/(\d+)', line)
            if match:
                tasks_solved_final = int(match.group(1))
                total_tasks = int(match.group(2))
        if 'Final grammar:' in line:
            match = re.search(r'Final grammar: (\d+)', line)
            if match:
                final_grammar = int(match.group(1))
        if 'Total abstractions:' in line:
            match = re.search(r'Total abstractions: (\d+)', line)
            if match:
                total_abstractions = int(match.group(1))
        if 'Total dreams:' in line:
            match = re.search(r'Total dreams: (\d+)', line)
            if match:
                total_dreams = int(match.group(1))

    # Calculate total programs
    total_programs = sum(it.programs_enumerated for it in iterations)

    summary = RunSummary(
        start_time=start_time,
        end_time=end_time,
        total_time=total_time,
        total_iterations=len(iterations),
        total_tasks=total_tasks,
        tasks_solved=tasks_solved_final,
        final_grammar_size=final_grammar,
        total_abstractions=total_abstractions,
        total_dreams=total_dreams,
        total_programs=total_programs,
        solved_tasks=solved_tasks,
        iterations=iterations,
        rejected_spurious=rejected_spurious,
        solved_programs=solved_programs,
        abstractions_by_iteration=abstractions_by_iteration
    )

    return summary, iterations


def generate_html_report(summary: RunSummary, output_path: Path, log_name: str) -> str:
    """Generate comprehensive HTML report."""

    iterations = summary.iterations

    # Prepare chart data
    iter_labels = [f"Iter {it.iteration}" for it in iterations]
    cumulative_solved = [it.cumulative_solved for it in iterations]
    grammar_sizes = [it.grammar_size for it in iterations]
    recognition_losses = [it.recognition_loss for it in iterations]
    programs_per_iter = [it.programs_enumerated for it in iterations]
    dreams_per_iter = [it.dreams_generated for it in iterations]

    # Solve rate as percentage
    solve_rates = [it.cumulative_solved / it.tasks_total * 100 if it.tasks_total > 0 else 0
                   for it in iterations]

    # Phase boundaries
    phase_boundaries = []
    current_phase = 0
    for i, it in enumerate(iterations):
        if it.phase != current_phase:
            phase_boundaries.append((i, it.phase))
            current_phase = it.phase

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DreamCoder Overnight Run Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --primary: #6366f1;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg-dark: #0f172a;
            --bg-card: #1e293b;
            --bg-light: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --border: #475569;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}

        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-dark);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 2rem;
        }}

        .container {{ max-width: 1400px; margin: 0 auto; }}

        header {{
            text-align: center;
            padding: 2rem;
            background: linear-gradient(135deg, var(--bg-card) 0%, var(--bg-light) 100%);
            border-radius: 1rem;
            margin-bottom: 2rem;
            border: 1px solid var(--border);
        }}

        h1 {{
            font-size: 2.5rem;
            background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}

        .subtitle {{ color: var(--text-secondary); font-size: 1.1rem; }}
        .run-info {{ color: var(--text-secondary); font-size: 0.9rem; margin-top: 1rem; }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .stat-card {{
            background: var(--bg-card);
            padding: 1.5rem;
            border-radius: 1rem;
            border: 1px solid var(--border);
            text-align: center;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .stat-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0,0,0,0.3);
        }}

        .stat-value {{
            font-size: 2.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--primary), #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}

        .stat-value.success {{ background: linear-gradient(135deg, var(--success), #4ade80); -webkit-background-clip: text; }}
        .stat-value.warning {{ background: linear-gradient(135deg, var(--warning), #fbbf24); -webkit-background-clip: text; }}

        .stat-label {{ color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.5rem; }}

        .section {{
            background: var(--bg-card);
            border-radius: 1rem;
            border: 1px solid var(--border);
            margin-bottom: 2rem;
            overflow: hidden;
        }}

        .section-header {{
            padding: 1.25rem 1.5rem;
            background: linear-gradient(90deg, rgba(99, 102, 241, 0.15), transparent);
            border-bottom: 1px solid var(--border);
            display: flex;
            justify-content: space-between;
            align-items: center;
            cursor: pointer;
        }}

        .section-header:hover {{ background: rgba(99, 102, 241, 0.2); }}
        .section-header h2 {{ font-size: 1.25rem; display: flex; align-items: center; gap: 0.5rem; }}
        .section-content {{ padding: 1.5rem; }}

        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 1.5rem;
        }}

        .chart-card {{
            background: var(--bg-light);
            padding: 1.5rem;
            border-radius: 0.75rem;
            border: 1px solid var(--border);
        }}

        .chart-card h3 {{
            color: var(--text-secondary);
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 1rem;
        }}

        .chart-container {{ position: relative; height: 280px; }}

        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 0.75rem 1rem; text-align: left; border-bottom: 1px solid var(--border); }}
        th {{ background: rgba(99, 102, 241, 0.1); font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }}
        tr:hover {{ background: rgba(255,255,255,0.02); }}

        .status-solved {{ color: var(--success); font-weight: 600; }}
        .status-unsolved {{ color: var(--text-secondary); }}

        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
        }}
        .badge-success {{ background: rgba(34, 197, 94, 0.2); color: var(--success); }}
        .badge-warning {{ background: rgba(245, 158, 11, 0.2); color: var(--warning); }}
        .badge-danger {{ background: rgba(239, 68, 68, 0.2); color: var(--danger); }}
        .badge-info {{ background: rgba(99, 102, 241, 0.2); color: #818cf8; }}

        .phase-tag {{
            display: inline-block;
            padding: 0.2rem 0.6rem;
            border-radius: 0.25rem;
            font-size: 0.7rem;
            font-weight: 600;
            background: var(--primary);
            color: white;
        }}

        .code {{
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 0.85rem;
            background: var(--bg-dark);
            padding: 0.5rem 0.75rem;
            border-radius: 0.5rem;
            overflow-x: auto;
            white-space: nowrap;
        }}

        /* Composition tree styles */
        .composition-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 1rem;
        }}

        .tree-card {{
            background: var(--bg-dark);
            border: 1px solid var(--border);
            border-radius: 0.75rem;
            padding: 1rem;
            transition: transform 0.2s, box-shadow 0.2s;
        }}

        .tree-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }}

        .tree-task {{
            font-weight: 600;
            color: var(--success);
            font-size: 0.95rem;
            margin-bottom: 0.25rem;
        }}

        .tree-iter {{
            font-size: 0.75rem;
            color: var(--text-secondary);
            margin-bottom: 0.75rem;
        }}

        .tree-program {{
            font-family: 'JetBrains Mono', 'Fira Code', monospace;
            font-size: 0.85rem;
            line-height: 1.6;
        }}

        .tree-content {{
            padding: 0.5rem;
            background: rgba(0,0,0,0.2);
            border-radius: 0.5rem;
        }}

        .tree-func {{
            color: #60a5fa;
            font-weight: 600;
        }}

        .tree-var {{
            color: #f472b6;
        }}

        .tree-lambda {{
            color: #a78bfa;
            font-weight: 700;
        }}

        .tree-prim {{
            color: #4ade80;
        }}

        .tree-args {{
            display: inline;
            margin-left: 0.25rem;
        }}

        .tree-arg {{
            margin-left: 0.5rem;
        }}

        .tree-arg::before {{
            content: '(';
            color: var(--text-secondary);
        }}

        .tree-arg::after {{
            content: ')';
            color: var(--text-secondary);
        }}

        .tree-raw {{
            word-break: break-all;
            white-space: pre-wrap;
        }}

        .tree-unknown {{
            color: var(--text-secondary);
            font-style: italic;
        }}

        .warning-box {{
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid rgba(245, 158, 11, 0.3);
            border-radius: 0.75rem;
            padding: 1rem 1.5rem;
            margin-bottom: 1.5rem;
        }}

        .warning-box h4 {{ color: var(--warning); margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.5rem; }}

        .timeline {{
            position: relative;
            padding-left: 2rem;
        }}

        .timeline::before {{
            content: '';
            position: absolute;
            left: 0.5rem;
            top: 0;
            bottom: 0;
            width: 2px;
            background: var(--border);
        }}

        .timeline-item {{
            position: relative;
            padding-bottom: 1.5rem;
        }}

        .timeline-item::before {{
            content: '';
            position: absolute;
            left: -1.65rem;
            top: 0.25rem;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--primary);
            border: 2px solid var(--bg-card);
        }}

        .timeline-item.solved::before {{ background: var(--success); }}

        .collapsible {{ max-height: 0; overflow: hidden; transition: max-height 0.3s ease; }}
        .collapsible.open {{ max-height: 5000px; }}

        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
            font-size: 0.9rem;
        }}

        @media (max-width: 768px) {{
            .chart-grid {{ grid-template-columns: 1fr; }}
            .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>DreamCoder Overnight Run Report</h1>
            <p class="subtitle">Cognitive Primitives Library v2 (60 primitives)</p>
            <p class="run-info">
                {summary.start_time} - {summary.end_time} | Duration: {summary.total_time}
            </p>
        </header>

        <!-- Executive Summary -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value success">{summary.tasks_solved}/{summary.total_tasks}</div>
                <div class="stat-label">Tasks Solved ({summary.tasks_solved/summary.total_tasks*100:.1f}%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary.total_iterations}</div>
                <div class="stat-label">Iterations</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary.final_grammar_size}</div>
                <div class="stat-label">Final Grammar Size</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary.total_abstractions}</div>
                <div class="stat-label">Abstractions Learned</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary.total_dreams:,}</div>
                <div class="stat-label">Dreams Generated</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{summary.total_programs:,}</div>
                <div class="stat-label">Programs Enumerated</div>
            </div>
        </div>

        <!-- Bug Warning -->
        <div class="warning-box">
            <h4>⚠️ Known Issue: Task-Result Scrambling Bug</h4>
            <p>This run was affected by a bug where parallel worker results were mismatched with tasks due to
            <code>as_completed()</code> returning results in completion order instead of submission order.
            This caused {len(summary.rejected_spurious)} tasks to receive and reject programs meant for other tasks.
            Some valid solutions may have been lost. <strong>The bug has been fixed for future runs.</strong></p>
        </div>

        <!-- Learning Curves -->
        <div class="section">
            <div class="section-header" onclick="toggleSection('learning-curves')">
                <h2>📈 Learning Curves</h2>
                <span id="learning-curves-icon">▼</span>
            </div>
            <div class="section-content collapsible open" id="learning-curves">
                <div class="chart-grid">
                    <div class="chart-card">
                        <h3>Cumulative Solve Rate</h3>
                        <div class="chart-container">
                            <canvas id="solveRateChart"></canvas>
                        </div>
                    </div>
                    <div class="chart-card">
                        <h3>Grammar Growth</h3>
                        <div class="chart-container">
                            <canvas id="grammarChart"></canvas>
                        </div>
                    </div>
                    <div class="chart-card">
                        <h3>Recognition Loss</h3>
                        <div class="chart-container">
                            <canvas id="lossChart"></canvas>
                        </div>
                    </div>
                    <div class="chart-card">
                        <h3>Programs Enumerated per Iteration</h3>
                        <div class="chart-container">
                            <canvas id="programsChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Solved Tasks -->
        <div class="section">
            <div class="section-header" onclick="toggleSection('solved-tasks')">
                <h2>✅ Solved Tasks ({len(summary.solved_tasks)})</h2>
                <span id="solved-tasks-icon">▼</span>
            </div>
            <div class="section-content collapsible open" id="solved-tasks">
                <table>
                    <thead>
                        <tr>
                            <th>Task</th>
                            <th>Iteration</th>
                            <th>Phase</th>
                            <th>Programs</th>
                            <th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    # Add solved tasks table rows
    for task_name, iter_solved in summary.solved_tasks:
        # Find the iteration details
        iter_data = next((it for it in iterations if it.iteration == iter_solved), None)
        phase_str = f"Phase {iter_data.phase}" if iter_data else "-"

        # Find programs count
        programs = "-"
        if iter_data:
            for sol_task, sol_prog in iter_data.new_solutions:
                if sol_task == task_name:
                    programs = f"{sol_prog:,}"
                    break

        html += f"""                        <tr>
                            <td><code>{task_name}</code></td>
                            <td><span class="badge badge-info">Iter {iter_solved}</span></td>
                            <td><span class="phase-tag">{phase_str}</span></td>
                            <td>{programs}</td>
                            <td><span class="status-solved">✓ Solved</span></td>
                        </tr>
"""

    html += """                    </tbody>
                </table>
            </div>
        </div>

        <!-- Program Composition Trees -->
        <div class="section">
            <div class="section-header" onclick="toggleSection('composition')">
                <h2>🌳 Program Composition Trees</h2>
                <span id="composition-icon">▼</span>
            </div>
            <div class="section-content collapsible open" id="composition">
                <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                    Visual decomposition of discovered solutions showing how primitives combine.
                </p>
                <div class="composition-grid">
"""

    # Add composition trees for solved tasks
    for task_name, iter_solved in summary.solved_tasks[:12]:  # Limit to 12 for readability
        # Find the solution program
        solution = summary.solved_programs.get(task_name, "Unknown")
        tree_html = _render_program_tree(solution, task_name)
        html += f"""                    <div class="tree-card">
                        <div class="tree-task">{task_name.replace('_', ' ').title()}</div>
                        <div class="tree-iter">Solved at iteration {iter_solved}</div>
                        <div class="tree-program">{tree_html}</div>
                    </div>
"""

    html += """                </div>
            </div>
        </div>

        <!-- Grammar Evolution -->
        <div class="section">
            <div class="section-header" onclick="toggleSection('grammar-evolution')">
                <h2>📚 Grammar Evolution</h2>
                <span id="grammar-evolution-icon">▼</span>
            </div>
            <div class="section-content collapsible" id="grammar-evolution">
                <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                    How the primitive library evolved through learning. Abstractions are new patterns
                    discovered through compression of successful programs.
                </p>
"""

    # Grammar evolution timeline
    if summary.abstractions_by_iteration:
        html += """                <div class="timeline">
"""
        for iter_num, abstractions in summary.abstractions_by_iteration.items():
            if abstractions:
                html += f"""                    <div class="timeline-item solved">
                        <strong>Iteration {iter_num}</strong>
                        <div style="margin-top: 0.5rem;">
"""
                for abstraction in abstractions:
                    html += f'                            <span class="badge badge-info" style="margin: 0.2rem;">{abstraction}</span>\n'
                html += """                        </div>
                    </div>
"""
        html += """                </div>
"""
    else:
        html += """                <p style="color: var(--text-secondary);">No abstractions were learned during this run.</p>
"""

    html += """            </div>
        </div>

        <!-- Iteration Details -->
        <div class="section">
            <div class="section-header" onclick="toggleSection('iterations')">
                <h2>📊 Iteration Details</h2>
                <span id="iterations-icon">▼</span>
            </div>
            <div class="section-content collapsible" id="iterations">
                <table>
                    <thead>
                        <tr>
                            <th>Iter</th>
                            <th>Phase</th>
                            <th>Solved</th>
                            <th>Programs</th>
                            <th>Grammar</th>
                            <th>Loss</th>
                            <th>Dreams</th>
                            <th>Abstractions</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    for it in iterations:
        html += f"""                        <tr>
                            <td><strong>{it.iteration}</strong></td>
                            <td><span class="phase-tag">P{it.phase} i{it.phase_iter}</span></td>
                            <td>{it.cumulative_solved}/{it.tasks_total}</td>
                            <td>{it.programs_enumerated:,}</td>
                            <td>{it.grammar_size}</td>
                            <td>{it.recognition_loss:.2f}</td>
                            <td>{it.dreams_generated}</td>
                            <td>{it.abstractions_found}</td>
                        </tr>
"""

    html += """                    </tbody>
                </table>
            </div>
        </div>

        <!-- Rejected Spurious Analysis -->
        <div class="section">
            <div class="section-header" onclick="toggleSection('spurious')">
                <h2>🔍 Rejected Spurious Analysis (Bug Evidence)</h2>
                <span id="spurious-icon">▼</span>
            </div>
            <div class="section-content collapsible" id="spurious">
                <p style="color: var(--text-secondary); margin-bottom: 1rem;">
                    These are programs that were found by workers for other tasks but got misattributed
                    due to the ordering bug. Some of these may be valid solutions for their intended tasks.
                </p>
                <table>
                    <thead>
                        <tr>
                            <th>Task (Received Wrong Programs)</th>
                            <th>Count</th>
                            <th>Sample Programs</th>
                        </tr>
                    </thead>
                    <tbody>
"""

    # Sort by count
    spurious_sorted = sorted(summary.rejected_spurious.items(), key=lambda x: -len(x[1]))
    for task_name, programs in spurious_sorted[:20]:
        sample = programs[0][:60] + "..." if len(programs[0]) > 60 else programs[0]
        html += f"""                        <tr>
                            <td><code>{task_name}</code></td>
                            <td><span class="badge badge-warning">{len(programs)}+</span></td>
                            <td class="code">{sample}</td>
                        </tr>
"""

    html += f"""                    </tbody>
                </table>
            </div>
        </div>

        <footer>
            <p>Generated by DreamCoder Overnight Run Analysis</p>
            <p>Log file: {log_name} | Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </footer>
    </div>

    <script>
        // Chart data
        const labels = {json.dumps(iter_labels)};
        const solveRates = {json.dumps(solve_rates)};
        const grammarSizes = {json.dumps(grammar_sizes)};
        const losses = {json.dumps(recognition_losses)};
        const programs = {json.dumps(programs_per_iter)};

        Chart.defaults.color = '#94a3b8';
        Chart.defaults.borderColor = '#475569';

        // Solve Rate Chart
        new Chart(document.getElementById('solveRateChart'), {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [{{
                    label: 'Solve Rate (%)',
                    data: solveRates,
                    borderColor: '#22c55e',
                    backgroundColor: 'rgba(34, 197, 94, 0.1)',
                    fill: true,
                    tension: 0.3
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{ beginAtZero: true, max: 100 }}
                }}
            }}
        }});

        // Grammar Chart
        new Chart(document.getElementById('grammarChart'), {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [{{
                    label: 'Grammar Size',
                    data: grammarSizes,
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    fill: true,
                    tension: 0.3
                }}]
            }},
            options: {{ responsive: true, maintainAspectRatio: false }}
        }});

        // Loss Chart
        new Chart(document.getElementById('lossChart'), {{
            type: 'line',
            data: {{
                labels: labels,
                datasets: [{{
                    label: 'Recognition Loss',
                    data: losses,
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    fill: true,
                    tension: 0.3
                }}]
            }},
            options: {{ responsive: true, maintainAspectRatio: false }}
        }});

        // Programs Chart
        new Chart(document.getElementById('programsChart'), {{
            type: 'bar',
            data: {{
                labels: labels,
                datasets: [{{
                    label: 'Programs Enumerated',
                    data: programs,
                    backgroundColor: 'rgba(99, 102, 241, 0.6)',
                    borderColor: '#6366f1',
                    borderWidth: 1
                }}]
            }},
            options: {{ responsive: true, maintainAspectRatio: false }}
        }});

        // Toggle sections
        function toggleSection(id) {{
            const content = document.getElementById(id);
            const icon = document.getElementById(id + '-icon');
            content.classList.toggle('open');
            icon.textContent = content.classList.contains('open') ? '▼' : '▶';
        }}
    </script>
</body>
</html>
"""

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(html)

    return str(output_path)


def main():
    """Main entry point."""
    # Find log file
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        # Find most recent log
        log_dir = Path("results/overnight_cython")
        if log_dir.exists():
            logs = list(log_dir.glob("run_*.log"))
            if logs:
                log_path = max(logs, key=lambda p: p.stat().st_mtime)
            else:
                print("No log files found in results/overnight_cython/")
                sys.exit(1)
        else:
            print("Results directory not found")
            sys.exit(1)

    if not log_path.exists():
        print(f"Log file not found: {log_path}")
        sys.exit(1)

    print(f"Parsing log file: {log_path}")

    # Parse log
    summary, iterations = parse_log_file(log_path)

    print(f"  Found {len(iterations)} iterations")
    print(f"  Tasks solved: {summary.tasks_solved}/{summary.total_tasks}")

    # Generate report
    output_path = log_path.parent / f"{log_path.stem}_report.html"
    report_path = generate_html_report(summary, output_path, log_path.name)

    print(f"\nReport generated: {report_path}")
    print(f"Open with: open {report_path}")


if __name__ == "__main__":
    main()
