#!/usr/bin/env python3
"""
HTML Report Generator for DreamCoder Experiments

Creates beautiful, interactive HTML reports with:
- Interactive charts (using Chart.js)
- Collapsible sections
- Color-coded status indicators
- Responsive design
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


def generate_html_report(
    experiment_name: str,
    batches_data: List[Dict],
    cumulative_data: Dict,
    output_path: str
) -> str:
    """
    Generate a comprehensive HTML report for curriculum learning.

    Args:
        experiment_name: Name of the experiment
        batches_data: List of per-batch metrics
        cumulative_data: Cumulative metrics across all batches
        output_path: Where to save the HTML file
    """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DreamCoder Report: {experiment_name}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --primary: #4f46e5;
            --success: #22c55e;
            --warning: #f59e0b;
            --danger: #ef4444;
            --bg-dark: #1e293b;
            --bg-card: #334155;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --border: #475569;
        }}

        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 2rem;
        }}

        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}

        header {{
            text-align: center;
            margin-bottom: 3rem;
            padding: 2rem;
            background: var(--bg-card);
            border-radius: 1rem;
            border: 1px solid var(--border);
        }}

        h1 {{
            font-size: 2.5rem;
            background: linear-gradient(90deg, #818cf8, #c084fc);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}

        .subtitle {{
            color: var(--text-secondary);
            font-size: 1.1rem;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}

        .stat-card {{
            background: var(--bg-card);
            padding: 1.5rem;
            border-radius: 0.75rem;
            border: 1px solid var(--border);
            text-align: center;
        }}

        .stat-value {{
            font-size: 2rem;
            font-weight: bold;
            color: var(--primary);
        }}

        .stat-label {{
            color: var(--text-secondary);
            font-size: 0.9rem;
            margin-top: 0.25rem;
        }}

        .section {{
            background: var(--bg-card);
            border-radius: 1rem;
            border: 1px solid var(--border);
            margin-bottom: 2rem;
            overflow: hidden;
        }}

        .section-header {{
            padding: 1rem 1.5rem;
            background: rgba(79, 70, 229, 0.1);
            border-bottom: 1px solid var(--border);
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .section-header:hover {{
            background: rgba(79, 70, 229, 0.2);
        }}

        .section-header h2 {{
            font-size: 1.25rem;
            color: var(--text-primary);
        }}

        .section-content {{
            padding: 1.5rem;
        }}

        .chart-container {{
            position: relative;
            height: 300px;
            margin-bottom: 1rem;
        }}

        .chart-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 1.5rem;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}

        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}

        th {{
            background: rgba(79, 70, 229, 0.1);
            font-weight: 600;
            color: var(--text-primary);
        }}

        tr:hover {{
            background: rgba(255, 255, 255, 0.05);
        }}

        .status-solved {{
            color: var(--success);
            font-weight: bold;
        }}

        .status-unsolved {{
            color: var(--danger);
        }}

        .badge {{
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 600;
        }}

        .badge-success {{
            background: rgba(34, 197, 94, 0.2);
            color: var(--success);
        }}

        .badge-warning {{
            background: rgba(245, 158, 11, 0.2);
            color: var(--warning);
        }}

        .badge-danger {{
            background: rgba(239, 68, 68, 0.2);
            color: var(--danger);
        }}

        .abstraction-item {{
            background: rgba(79, 70, 229, 0.1);
            padding: 0.75rem 1rem;
            border-radius: 0.5rem;
            margin-bottom: 0.5rem;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.9rem;
            border-left: 3px solid var(--primary);
        }}

        .batch-card {{
            background: rgba(0, 0, 0, 0.2);
            border-radius: 0.75rem;
            padding: 1rem;
            margin-bottom: 1rem;
        }}

        .batch-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.75rem;
        }}

        .batch-title {{
            font-weight: 600;
            color: var(--primary);
        }}

        .progress-bar {{
            height: 8px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 4px;
            overflow: hidden;
        }}

        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, var(--primary), #c084fc);
            transition: width 0.3s ease;
        }}

        .code-block {{
            background: #0f172a;
            padding: 1rem;
            border-radius: 0.5rem;
            font-family: 'Consolas', 'Monaco', monospace;
            font-size: 0.85rem;
            overflow-x: auto;
            margin: 0.5rem 0;
        }}

        .metric-row {{
            display: flex;
            justify-content: space-between;
            padding: 0.5rem 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }}

        .metric-label {{
            color: var(--text-secondary);
        }}

        .metric-value {{
            font-weight: 600;
        }}

        .collapsible {{
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
        }}

        .collapsible.open {{
            max-height: 5000px;
        }}

        .toggle-icon {{
            transition: transform 0.3s ease;
        }}

        .toggle-icon.rotated {{
            transform: rotate(180deg);
        }}

        footer {{
            text-align: center;
            padding: 2rem;
            color: var(--text-secondary);
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🧠 DreamCoder Curriculum Learning Report</h1>
            <p class="subtitle">{experiment_name} • Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </header>

        <!-- Executive Summary -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{cumulative_data.get('total_batches', 0)}</div>
                <div class="stat-label">Batches Processed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{cumulative_data.get('total_tasks_solved', 0)}/{cumulative_data.get('total_tasks', 0)}</div>
                <div class="stat-label">Tasks Solved</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{cumulative_data.get('total_abstractions', 0)}</div>
                <div class="stat-label">Abstractions Learned</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{cumulative_data.get('final_grammar_size', 0)}</div>
                <div class="stat-label">Final Grammar Size</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{cumulative_data.get('total_time', 0):.1f}s</div>
                <div class="stat-label">Total Time</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{cumulative_data.get('total_programs', 0):,}</div>
                <div class="stat-label">Programs Enumerated</div>
            </div>
        </div>

        <!-- Learning Curves Section -->
        <div class="section">
            <div class="section-header" onclick="toggleSection('learning-curves')">
                <h2>📈 Learning Curves</h2>
                <span class="toggle-icon" id="learning-curves-icon">▼</span>
            </div>
            <div class="section-content collapsible open" id="learning-curves">
                <div class="chart-row">
                    <div>
                        <h3 style="margin-bottom: 1rem; color: var(--text-secondary);">Cumulative Solve Rate</h3>
                        <div class="chart-container">
                            <canvas id="solveRateChart"></canvas>
                        </div>
                    </div>
                    <div>
                        <h3 style="margin-bottom: 1rem; color: var(--text-secondary);">Library Growth</h3>
                        <div class="chart-container">
                            <canvas id="libraryChart"></canvas>
                        </div>
                    </div>
                </div>
                <div class="chart-row">
                    <div>
                        <h3 style="margin-bottom: 1rem; color: var(--text-secondary);">Search Efficiency</h3>
                        <div class="chart-container">
                            <canvas id="efficiencyChart"></canvas>
                        </div>
                    </div>
                    <div>
                        <h3 style="margin-bottom: 1rem; color: var(--text-secondary);">Recognition Model Evolution</h3>
                        <div class="chart-container">
                            <canvas id="recognitionChart"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Batch Details Section -->
        <div class="section">
            <div class="section-header" onclick="toggleSection('batch-details')">
                <h2>📦 Batch-by-Batch Progress</h2>
                <span class="toggle-icon" id="batch-details-icon">▼</span>
            </div>
            <div class="section-content collapsible open" id="batch-details">
                {generate_batch_cards(batches_data)}
            </div>
        </div>

        <!-- Library Evolution Section -->
        <div class="section">
            <div class="section-header" onclick="toggleSection('library-evolution')">
                <h2>📚 Library Evolution</h2>
                <span class="toggle-icon" id="library-evolution-icon">▼</span>
            </div>
            <div class="section-content collapsible open" id="library-evolution">
                <h3 style="margin-bottom: 1rem;">Abstractions Timeline</h3>
                {generate_abstractions_timeline(cumulative_data.get('all_abstractions', []))}
            </div>
        </div>

        <!-- Task Results Section -->
        <div class="section">
            <div class="section-header" onclick="toggleSection('task-results')">
                <h2>✅ Task Results</h2>
                <span class="toggle-icon" id="task-results-icon">▼</span>
            </div>
            <div class="section-content collapsible open" id="task-results">
                {generate_task_table(cumulative_data.get('task_results', {}))}
            </div>
        </div>

        <!-- Grammar Analysis Section -->
        <div class="section">
            <div class="section-header" onclick="toggleSection('grammar-analysis')">
                <h2>⚖️ Grammar Distribution Analysis</h2>
                <span class="toggle-icon" id="grammar-analysis-icon">▼</span>
            </div>
            <div class="section-content collapsible open" id="grammar-analysis">
                <div class="chart-container" style="height: 400px;">
                    <canvas id="grammarChart"></canvas>
                </div>
                {generate_grammar_changes(cumulative_data.get('grammar_changes', []))}
            </div>
        </div>

        <footer>
            <p>Generated by DreamCoder Curriculum Learning System</p>
            <p style="margin-top: 0.5rem; font-size: 0.9rem;">
                {cumulative_data.get('total_batches', 0)} batches •
                {cumulative_data.get('total_tasks', 0)} tasks •
                {cumulative_data.get('total_time', 0):.1f} seconds
            </p>
        </footer>
    </div>

    <script>
        // Chart data from Python
        const batchLabels = {json.dumps([f"Batch {i+1}" for i in range(len(batches_data))])};
        const solveRates = {json.dumps([b.get('cumulative_solve_rate', 0) for b in batches_data])};
        const librarySizes = {json.dumps([b.get('grammar_size', 61) for b in batches_data])};
        const avgPrograms = {json.dumps([b.get('avg_programs_per_task', 0) for b in batches_data])};
        const recognitionDivergence = {json.dumps([b.get('recognition_divergence', 0) for b in batches_data])};
        const grammarLabels = {json.dumps(cumulative_data.get('top_primitives', []))};
        const grammarInitial = {json.dumps(cumulative_data.get('initial_probs', []))};
        const grammarFinal = {json.dumps(cumulative_data.get('final_probs', []))};

        // Chart.js configuration
        Chart.defaults.color = '#94a3b8';
        Chart.defaults.borderColor = '#475569';

        // Solve Rate Chart
        new Chart(document.getElementById('solveRateChart'), {{
            type: 'line',
            data: {{
                labels: batchLabels,
                datasets: [{{
                    label: 'Cumulative Solve Rate',
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
                    y: {{
                        beginAtZero: true,
                        max: 1,
                        ticks: {{
                            callback: function(value) {{ return (value * 100) + '%'; }}
                        }}
                    }}
                }}
            }}
        }});

        // Library Chart
        new Chart(document.getElementById('libraryChart'), {{
            type: 'line',
            data: {{
                labels: batchLabels,
                datasets: [{{
                    label: 'Grammar Size',
                    data: librarySizes,
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    fill: true,
                    tension: 0.3
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false
            }}
        }});

        // Efficiency Chart
        new Chart(document.getElementById('efficiencyChart'), {{
            type: 'bar',
            data: {{
                labels: batchLabels,
                datasets: [{{
                    label: 'Avg Programs per Task',
                    data: avgPrograms,
                    backgroundColor: 'rgba(79, 70, 229, 0.6)',
                    borderColor: '#4f46e5',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false
            }}
        }});

        // Recognition Chart
        new Chart(document.getElementById('recognitionChart'), {{
            type: 'line',
            data: {{
                labels: batchLabels,
                datasets: [{{
                    label: 'Recognition Divergence',
                    data: recognitionDivergence,
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    fill: true,
                    tension: 0.3
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false
            }}
        }});

        // Grammar Distribution Chart
        new Chart(document.getElementById('grammarChart'), {{
            type: 'bar',
            data: {{
                labels: grammarLabels,
                datasets: [
                    {{
                        label: 'Initial',
                        data: grammarInitial,
                        backgroundColor: 'rgba(148, 163, 184, 0.6)',
                        borderColor: '#94a3b8',
                        borderWidth: 1
                    }},
                    {{
                        label: 'Final',
                        data: grammarFinal,
                        backgroundColor: 'rgba(79, 70, 229, 0.6)',
                        borderColor: '#4f46e5',
                        borderWidth: 1
                    }}
                ]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                scales: {{
                    x: {{
                        title: {{
                            display: true,
                            text: 'Log Probability'
                        }}
                    }}
                }}
            }}
        }});

        // Toggle sections
        function toggleSection(id) {{
            const content = document.getElementById(id);
            const icon = document.getElementById(id + '-icon');
            content.classList.toggle('open');
            icon.classList.toggle('rotated');
        }}
    </script>
</body>
</html>
"""

    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(html)

    return str(output_path)


def generate_batch_cards(batches_data: List[Dict]) -> str:
    """Generate HTML for batch progress cards."""
    cards = []

    for i, batch in enumerate(batches_data):
        solve_rate = batch.get('solve_rate', 0)
        tasks = batch.get('tasks', [])
        solved = batch.get('tasks_solved', 0)
        total = batch.get('tasks_total', 0)
        programs = batch.get('programs_enumerated', 0)
        abstractions = batch.get('new_abstractions', [])
        time_taken = batch.get('time', 0)

        task_badges = ""
        for task in tasks[:6]:
            status = "success" if task.get('solved', False) else "danger"
            task_badges += f'<span class="badge badge-{status}">{task.get("name", "?")[:15]}</span> '

        abstraction_items = ""
        for a in abstractions[:3]:
            a_display = a[:60] + "..." if len(a) > 60 else a
            abstraction_items += f'<div class="abstraction-item">{a_display}</div>'

        cards.append(f"""
        <div class="batch-card">
            <div class="batch-header">
                <span class="batch-title">Batch {i+1}</span>
                <span class="badge badge-{'success' if solve_rate > 0.5 else 'warning'}">{solved}/{total} solved</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: {solve_rate * 100}%"></div>
            </div>
            <div style="margin-top: 0.75rem; font-size: 0.9rem;">
                {task_badges}
            </div>
            <div class="metric-row">
                <span class="metric-label">Programs enumerated</span>
                <span class="metric-value">{programs:,}</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">Time</span>
                <span class="metric-value">{time_taken:.1f}s</span>
            </div>
            <div class="metric-row">
                <span class="metric-label">New abstractions</span>
                <span class="metric-value">{len(abstractions)}</span>
            </div>
            {f'<div style="margin-top: 0.75rem;">{abstraction_items}</div>' if abstraction_items else ''}
        </div>
        """)

    return "\n".join(cards)


def generate_abstractions_timeline(abstractions: List[tuple]) -> str:
    """Generate HTML for abstractions timeline."""
    if not abstractions:
        return "<p style='color: var(--text-secondary);'>No abstractions learned yet.</p>"

    items = []
    for batch_num, abstr in abstractions:
        a_display = abstr[:70] + "..." if len(abstr) > 70 else abstr
        items.append(f"""
        <div class="abstraction-item">
            <span style="color: var(--primary); font-weight: bold;">Batch {batch_num + 1}:</span>
            {a_display}
        </div>
        """)

    return "\n".join(items)


def generate_task_table(task_results: Dict) -> str:
    """Generate HTML table for task results."""
    if not task_results:
        return "<p style='color: var(--text-secondary);'>No task results available.</p>"

    rows = []
    for name, data in sorted(task_results.items(), key=lambda x: (not x[1].get('solved', False), x[0])):
        solved = data.get('solved', False)
        batch = data.get('batch_solved', '-')
        programs = data.get('programs_to_solve', '-')
        best_prog = data.get('best_program', '-')
        dl = data.get('description_length', '-')

        status_class = 'status-solved' if solved else 'status-unsolved'
        status_text = '✓ Solved' if solved else '✗ Unsolved'

        prog_display = best_prog[:50] + "..." if len(str(best_prog)) > 50 else best_prog

        rows.append(f"""
        <tr>
            <td>{name}</td>
            <td class="{status_class}">{status_text}</td>
            <td>{batch}</td>
            <td>{programs:,}</td>
            <td>{dl if dl != '-' else '-'}</td>
            <td class="code-block" style="font-size: 0.8rem;">{prog_display}</td>
        </tr>
        """)

    return f"""
    <table>
        <thead>
            <tr>
                <th>Task</th>
                <th>Status</th>
                <th>Batch Solved</th>
                <th>Programs</th>
                <th>Desc. Length</th>
                <th>Best Solution</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
    """


def generate_grammar_changes(changes: List[tuple]) -> str:
    """Generate HTML for grammar probability changes."""
    if not changes:
        return ""

    rows = []
    for prim, init, final, change in changes[:15]:
        change_class = 'status-solved' if change > 0 else 'status-unsolved' if change < 0 else ''
        change_str = f"+{change:.3f}" if change > 0 else f"{change:.3f}"

        rows.append(f"""
        <tr>
            <td>{prim[:30]}</td>
            <td>{init:.3f}</td>
            <td>{final:.3f}</td>
            <td class="{change_class}">{change_str}</td>
        </tr>
        """)

    return f"""
    <h3 style="margin-top: 1.5rem; margin-bottom: 1rem;">Top Probability Changes</h3>
    <table>
        <thead>
            <tr>
                <th>Primitive</th>
                <th>Initial</th>
                <th>Final</th>
                <th>Change</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
    """
