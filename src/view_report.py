#!/usr/bin/env python3
"""
View Demo Report - Human-readable formatting of demo_report.json

Displays the DreamCoder demo results in nicely formatted tables.

Usage:
    python src/view_report.py
    python src/view_report.py results/demo_report.json
    python src/view_report.py --html  # Generate HTML report
"""

import json
import sys
from pathlib import Path
from datetime import datetime


def load_report(path: str = "results/demo_report.json") -> dict:
    """Load the JSON report."""
    with open(path, 'r') as f:
        return json.load(f)


def print_header(title: str, char: str = "="):
    """Print a formatted header."""
    width = 70
    print()
    print(char * width)
    print(f" {title}")
    print(char * width)


def print_summary_table(report: dict):
    """Print the summary statistics as a table."""
    print_header("SUMMARY STATISTICS")

    # Parse timestamp
    ts = report.get('timestamp', 'Unknown')
    if ts != 'Unknown':
        try:
            dt = datetime.fromisoformat(ts)
            ts = dt.strftime("%B %d, %Y at %H:%M:%S")
        except:
            pass

    rows = [
        ("Report Generated", ts),
        ("Total Rules in Catalogue", report.get('num_rules', 'N/A')),
        ("Demo Rules Used", report.get('num_demo_rules', 'N/A')),
        ("Tasks Generated", report.get('num_tasks', 'N/A')),
        ("Unique Primitives", report.get('num_primitives', 'N/A')),
        ("Feature Dimensions", report.get('feature_dim', 'N/A')),
    ]

    print()
    print(f"  {'Metric':<30} {'Value':>15}")
    print(f"  {'-'*30} {'-'*15}")
    for label, value in rows:
        print(f"  {label:<30} {str(value):>15}")


def print_rules_by_family(report: dict):
    """Print rules organized by family."""
    print_header("RULES BY FAMILY")

    rules = report.get('rules', [])

    # Group by family
    families = {}
    for rule in rules:
        family = rule.get('family', 'Unknown')
        families.setdefault(family, []).append(rule)

    for family in sorted(families.keys()):
        family_rules = families[family]
        print(f"\n  [{family}] - {len(family_rules)} rule(s)")
        print(f"  {'-' * 50}")

        for rule in family_rules:
            rule_id = rule.get('id', 'Unknown')
            rule_name = rule.get('name', 'No name')
            num_prims = rule.get('num_primitives', 0)

            # Truncate name if too long
            if len(rule_name) > 35:
                rule_name = rule_name[:32] + "..."

            print(f"    • {rule_id}")
            print(f"      Name: {rule_name}")
            print(f"      Primitives used: {num_prims}")


def print_primitive_details(report: dict):
    """Print detailed primitive usage per rule."""
    print_header("PRIMITIVE USAGE DETAILS")

    rules = report.get('rules', [])

    # Collect all primitives
    all_primitives = set()
    for rule in rules:
        all_primitives.update(rule.get('primitives', []))

    print(f"\n  All primitives found ({len(all_primitives)}):")
    print(f"  {'-' * 50}")

    # Print in columns
    sorted_prims = sorted(all_primitives)
    cols = 3
    for i in range(0, len(sorted_prims), cols):
        row = sorted_prims[i:i+cols]
        print("    " + "  ".join(f"{p:<20}" for p in row))

    print(f"\n  Primitive breakdown by rule:")
    print(f"  {'-' * 50}")

    for rule in rules:
        rule_id = rule.get('id', 'Unknown')
        prims = rule.get('primitives', [])
        print(f"\n    {rule_id}:")
        if prims:
            print(f"      → {', '.join(sorted(prims))}")
        else:
            print(f"      → (no primitives recorded)")


def print_coverage_analysis(report: dict):
    """Print analysis of primitive coverage."""
    print_header("COVERAGE ANALYSIS")

    rules = report.get('rules', [])

    # Count primitive usage
    prim_counts = {}
    for rule in rules:
        for prim in rule.get('primitives', []):
            prim_counts[prim] = prim_counts.get(prim, 0) + 1

    # Sort by frequency
    sorted_prims = sorted(prim_counts.items(), key=lambda x: -x[1])

    print(f"\n  Most frequently used primitives:")
    print(f"  {'Primitive':<25} {'Count':>8} {'Frequency':>12}")
    print(f"  {'-'*25} {'-'*8} {'-'*12}")

    total_rules = len(rules)
    for prim, count in sorted_prims[:15]:
        freq = count / total_rules * 100
        bar = "█" * int(freq / 10)
        print(f"  {prim:<25} {count:>8} {freq:>10.1f}% {bar}")

    if len(sorted_prims) > 15:
        print(f"\n  ... and {len(sorted_prims) - 15} more primitives")


def generate_html_report(report: dict, output_path: str = "results/report.html"):
    """Generate an HTML version of the report."""

    rules = report.get('rules', [])

    # Group by family
    families = {}
    for rule in rules:
        family = rule.get('family', 'Unknown')
        families.setdefault(family, []).append(rule)

    # Collect primitives
    prim_counts = {}
    for rule in rules:
        for prim in rule.get('primitives', []):
            prim_counts[prim] = prim_counts.get(prim, 0) + 1

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>DreamCoder Demo Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{
            border-collapse: collapse;
            width: 100%;
            background: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{ background: #f9f9f9; }}
        tr:hover {{ background: #f0f0f0; }}
        .stat-box {{
            display: inline-block;
            background: white;
            padding: 20px 30px;
            margin: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .stat-box .number {{ font-size: 36px; color: #3498db; font-weight: bold; }}
        .stat-box .label {{ color: #7f8c8d; margin-top: 5px; }}
        .family-tag {{
            display: inline-block;
            padding: 3px 10px;
            background: #3498db;
            color: white;
            border-radius: 4px;
            font-size: 12px;
        }}
        .prim-tag {{
            display: inline-block;
            padding: 2px 8px;
            background: #ecf0f1;
            border-radius: 3px;
            margin: 2px;
            font-family: monospace;
            font-size: 12px;
        }}
        .bar {{
            background: #3498db;
            height: 20px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <h1>🃏 DreamCoder Card Game Modeling Report</h1>

    <p>Generated: {report.get('timestamp', 'Unknown')}</p>

    <h2>📊 Summary Statistics</h2>
    <div>
        <div class="stat-box">
            <div class="number">{report.get('num_rules', 'N/A')}</div>
            <div class="label">Total Rules</div>
        </div>
        <div class="stat-box">
            <div class="number">{report.get('num_tasks', 'N/A')}</div>
            <div class="label">Demo Tasks</div>
        </div>
        <div class="stat-box">
            <div class="number">{report.get('num_primitives', 'N/A')}</div>
            <div class="label">Unique Primitives</div>
        </div>
        <div class="stat-box">
            <div class="number">{report.get('feature_dim', 'N/A')}</div>
            <div class="label">Feature Dimensions</div>
        </div>
        <div class="stat-box">
            <div class="number">{len(families)}</div>
            <div class="label">Rule Families</div>
        </div>
    </div>

    <h2>📋 Rules by Family</h2>
    <table>
        <tr>
            <th>Family</th>
            <th>Rule ID</th>
            <th>Name</th>
            <th>Primitives</th>
        </tr>
"""

    for family in sorted(families.keys()):
        for i, rule in enumerate(families[family]):
            family_cell = f'<span class="family-tag">{family}</span>' if i == 0 else ''
            prims_html = ''.join(f'<span class="prim-tag">{p}</span>' for p in rule.get('primitives', []))
            html += f"""        <tr>
            <td>{family_cell}</td>
            <td><code>{rule.get('id', '')}</code></td>
            <td>{rule.get('name', '')}</td>
            <td>{prims_html}</td>
        </tr>
"""

    html += """    </table>

    <h2>📈 Primitive Usage Frequency</h2>
    <table>
        <tr>
            <th>Primitive</th>
            <th>Count</th>
            <th>Frequency</th>
        </tr>
"""

    sorted_prims = sorted(prim_counts.items(), key=lambda x: -x[1])
    total_rules = len(rules)

    for prim, count in sorted_prims:
        freq = count / total_rules * 100
        bar_width = int(freq * 3)
        html += f"""        <tr>
            <td><code>{prim}</code></td>
            <td>{count}</td>
            <td>
                <div class="bar" style="width: {bar_width}px;"></div>
                {freq:.1f}%
            </td>
        </tr>
"""

    html += """    </table>

    <h2>🔗 Links</h2>
    <ul>
        <li><a href="primitive_usage_heatmap.png">Primitive Usage Heatmap</a></li>
        <li><a href="feature_statistics.png">Feature Statistics</a></li>
        <li><a href="primitive_cooccurrence.png">Primitive Co-occurrence Matrix</a></li>
    </ul>

    <hr>
    <p style="color: #7f8c8d; font-size: 12px;">
        Generated by DreamCoder Card Game Modeling •
        <a href="https://github.com/konukcan/card-games-modeling">GitHub Repository</a>
    </p>
</body>
</html>
"""

    with open(output_path, 'w') as f:
        f.write(html)

    return output_path


def main():
    """Main entry point."""
    # Parse arguments
    report_path = "results/demo_report.json"
    generate_html = False

    for arg in sys.argv[1:]:
        if arg == "--html":
            generate_html = True
        elif not arg.startswith("-"):
            report_path = arg

    # Check if report exists
    if not Path(report_path).exists():
        print(f"Error: Report file not found: {report_path}")
        print(f"\nRun the demo first:")
        print(f"  python src/main_demo.py")
        sys.exit(1)

    # Load report
    report = load_report(report_path)

    # Print to console
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " DREAMCODER DEMO REPORT ".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    print_summary_table(report)
    print_rules_by_family(report)
    print_primitive_details(report)
    print_coverage_analysis(report)

    # Generate HTML if requested
    if generate_html:
        html_path = generate_html_report(report)
        print_header("HTML REPORT GENERATED")
        print(f"\n  Saved to: {html_path}")
        print(f"  Open in browser: open {html_path}")

    print()
    print("=" * 70)
    print(" Report viewing complete!")
    print("=" * 70)
    print()
    print("  Tips:")
    print("    • Run with --html to generate an interactive HTML report")
    print("    • Visualizations are in results/*.png")
    print()


if __name__ == "__main__":
    main()
